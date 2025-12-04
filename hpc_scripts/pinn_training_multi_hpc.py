import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import time
import os
import json
import signal
import sys
from datetime import datetime

# Global flag for graceful termination
_TERMINATION_REQUESTED = False
_PINN_INSTANCE = None  # Will hold reference to PINN for signal handler

def signal_handler(signum, frame):
    """Handle termination signals (SIGTERM, SIGINT) gracefully"""
    global _TERMINATION_REQUESTED, _PINN_INSTANCE
    signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    print(f"\n{'='*80}")
    print(f"RECEIVED {signal_name} SIGNAL - Initiating graceful shutdown...")
    print(f"{'='*80}")
    _TERMINATION_REQUESTED = True
    
    # If we have a PINN instance, save immediately
    if _PINN_INSTANCE is not None:
        print("Saving checkpoint before termination...")
        try:
            _PINN_INSTANCE.save_checkpoint_on_interrupt()
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    # Don't exit immediately - let the training loop handle it gracefully
    # If called twice, force exit
    if hasattr(signal_handler, 'called_before') and signal_handler.called_before:
        print("Forced exit due to repeated signal")
        sys.exit(1)
    signal_handler.called_before = True

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)  # SLURM sends this before killing job
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C

# Brusselator RHS
def brusselator_rhs(x, y, A, B):
    dxdt = A + x * x * y - (B + 1.0) * x
    dydt = B * x - x * x * y
    return dxdt, dydt


# RK4 Integrator using scipy for stability
def generate_rk4_data(A, B, x0, y0, t_min=0.0, t_max=20.0, n_points=100):
    """Generate reference data using scipy's odeint (more stable than manual RK4)"""
    from scipy.integrate import odeint
    
    def system(state, t):
        x, y = state
        dxdt, dydt = brusselator_rhs(x, y, A, B)
        return [dxdt, dydt]
    
    t = np.linspace(t_min, t_max, n_points)
    solution = odeint(system, [x0, y0], t, rtol=1e-8, atol=1e-8)
    
    return t, solution[:, 0], solution[:, 1]


# Generate random parameter sets for training/validation
def generate_parameter_sets(n_sets, A_range=(0.5, 2.0), B_range=(2.0, 4.0), 
                           x0_range=(0.5, 2.0), y0_range=(0.5, 2.0), seed=None):
    """
    Generate random parameter sets for Brusselator system
    
    Args:
        n_sets: Number of parameter sets to generate
        A_range: Tuple of (min, max) for parameter A
        B_range: Tuple of (min, max) for parameter B
        x0_range: Tuple of (min, max) for initial condition x0
        y0_range: Tuple of (min, max) for initial condition y0
        seed: Random seed for reproducibility
    
    Returns:
        List of parameter dictionaries
    """
    if seed is not None:
        np.random.seed(seed)
    
    param_sets = []
    for _ in range(n_sets):
        params = {
            'A': np.random.uniform(*A_range),
            'B': np.random.uniform(*B_range),
            'x0': np.random.uniform(*x0_range),
            'y0': np.random.uniform(*y0_range)
        }
        param_sets.append(params)
    
    return param_sets


# Expanded PINN Class with parameter inputs (including initial conditions)
class PINN(nn.Module):   
    def __init__(self, hidden_layers=[128, 128, 128, 128, 128, 128, 128, 128], activation=nn.Tanh()):
        super(PINN, self).__init__()
        
        # Input layer: time t, A, B, x0, y0 (5 inputs total)
        # x0, y0 are initial conditions - needed to distinguish different trajectories!
        layers = [nn.Linear(5, hidden_layers[0]), activation]
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.extend([
                nn.Linear(hidden_layers[i], hidden_layers[i+1]),
                activation
            ])
        
        # Output layer: [x, y]
        layers.append(nn.Linear(hidden_layers[-1], 2))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights for stability
        self.init_weights()
    
    def init_weights(self):
        # Initialize network weights using Xavier initialization
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, t, A, B, x0, y0):
        # Forward pass with time, parameters, and initial conditions as input
        # Concatenate inputs: [t, A, B, x0, y0]
        inputs = torch.cat([t, A, B, x0, y0], dim=1)
        out = self.network(inputs)
        return out


# Multi-Parameter BrusselatorPINN Class with Validation
class MultiParamBrusselatorPINN:
    def __init__(self, param_sets_train, param_sets_val, t_min=0.0, t_max=20.0, 
                 device='cpu', output_dir='outputs'):
        """
        Initialize the PINN solver for multiple parameter sets with validation
        
        param_sets_train: list of dicts with keys ['A', 'B', 'x0', 'y0'] for training
        param_sets_val: list of dicts with keys ['A', 'B', 'x0', 'y0'] for validation
        """
        self.param_sets_train = param_sets_train
        self.param_sets_val = param_sets_val
        self.t_min = t_min
        self.t_max = t_max
        self.device = torch.device(device)
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        # Create model with expanded architecture
        self.model = PINN(hidden_layers=[128, 128, 128, 128, 128, 128, 128, 128]).to(self.device)
        
        # Ensure model parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Loss history
        self.loss_history = {
            'train_total': [],
            'train_physics': [],
            'train_ic': [],
            'train_data': [],
            'val_total': [],
            'val_physics': [],
            'val_ic': [],
            'val_data': []
        }
        
        # Early stopping tracking (on validation loss)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        self.best_epoch = 0
    
    def physics_loss(self, t_collocation, A, B, x0, y0):
        """Compute physics loss (PDE residuals) for given A, B, x0, y0 parameters"""
        # Clone and enable gradients for this computation
        # Must create a leaf variable that requires gradients
        t = t_collocation.clone().requires_grad_(True)
        
        # Create tensors for A, B, x0, y0 parameters  
        A_tensor = torch.full_like(t, A)
        B_tensor = torch.full_like(t, B)
        x0_tensor = torch.full_like(t, x0)
        y0_tensor = torch.full_like(t, y0)
        
        # Forward pass - ensure model is in correct state
        xy = self.model(t, A_tensor, B_tensor, x0_tensor, y0_tensor)
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        
        # Compute derivatives using automatic differentiation
        dx_dt = torch.autograd.grad(
            outputs=x,
            inputs=t,
            grad_outputs=torch.ones_like(x),
            create_graph=True,
            retain_graph=True,
            allow_unused=False
        )[0]
        
        dy_dt = torch.autograd.grad(
            outputs=y,
            inputs=t,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            allow_unused=False
        )[0]
        
        # Brusselator equations residuals
        f_x = dx_dt - (A + x * x * y - (B + 1.0) * x)
        f_y = dy_dt - (B * x - x * x * y)
        
        # Mean squared error
        loss_physics = torch.mean(f_x ** 2 + f_y ** 2)
        
        return loss_physics
    
    def initial_condition_loss(self, param_set):
        """Compute initial condition loss for a given parameter set"""
        t0 = torch.tensor([[self.t_min]], dtype=torch.float32, device=self.device)
        A_tensor = torch.tensor([[param_set['A']]], dtype=torch.float32, device=self.device)
        B_tensor = torch.tensor([[param_set['B']]], dtype=torch.float32, device=self.device)
        x0_tensor = torch.tensor([[param_set['x0']]], dtype=torch.float32, device=self.device)
        y0_tensor = torch.tensor([[param_set['y0']]], dtype=torch.float32, device=self.device)
        
        xy0_pred = self.model(t0, A_tensor, B_tensor, x0_tensor, y0_tensor)
        
        x0_pred = xy0_pred[:, 0]
        y0_pred = xy0_pred[:, 1]
        
        x0 = param_set['x0']
        y0 = param_set['y0']
        
        loss_ic = (x0_pred - x0) ** 2 + (y0_pred - y0) ** 2
        
        return loss_ic
    
    def data_loss(self, t_data, x_data, y_data, A_data, B_data, x0_data, y0_data, batch_size=5000):
        """
        Compute data loss with batching to prevent GPU OOM
        
        batch_size: number of data points to process at once
        Data is assumed to be on CPU and will be transferred to GPU in batches
        """
        if t_data is None or len(t_data) == 0:
            return torch.tensor(0.0, device=self.device)
        
        n_data = len(t_data)
        
        # If data is small enough, process all at once
        if n_data <= batch_size:
            # Transfer to GPU
            t_gpu = t_data.to(self.device)
            x_gpu = x_data.to(self.device)
            y_gpu = y_data.to(self.device)
            A_gpu = A_data.to(self.device)
            B_gpu = B_data.to(self.device)
            x0_gpu = x0_data.to(self.device)
            y0_gpu = y0_data.to(self.device)
            
            xy_pred = self.model(t_gpu, A_gpu, B_gpu, x0_gpu, y0_gpu)
            x_pred = xy_pred[:, 0:1]
            y_pred = xy_pred[:, 1:2]
            loss_data = torch.mean((x_pred - x_gpu) ** 2 + (y_pred - y_gpu) ** 2)
            return loss_data
        
        # Otherwise, batch the data loss computation
        loss_list = []
        n_batches = (n_data + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_data)
            batch_size_actual = end_idx - start_idx
            
            # Batch data (on CPU)
            t_batch = t_data[start_idx:end_idx]
            x_batch = x_data[start_idx:end_idx]
            y_batch = y_data[start_idx:end_idx]
            A_batch = A_data[start_idx:end_idx]
            B_batch = B_data[start_idx:end_idx]
            x0_batch = x0_data[start_idx:end_idx]
            y0_batch = y0_data[start_idx:end_idx]
            
            # Transfer batch to GPU
            t_gpu = t_batch.to(self.device)
            x_gpu = x_batch.to(self.device)
            y_gpu = y_batch.to(self.device)
            A_gpu = A_batch.to(self.device)
            B_gpu = B_batch.to(self.device)
            x0_gpu = x0_batch.to(self.device)
            y0_gpu = y0_batch.to(self.device)
            
            # Forward pass on batch
            xy_pred = self.model(t_gpu, A_gpu, B_gpu, x0_gpu, y0_gpu)
            x_pred = xy_pred[:, 0:1]
            y_pred = xy_pred[:, 1:2]
            
            # Compute batch loss (mean for this batch, weighted by batch size)
            batch_mse = torch.mean((x_pred - x_gpu) ** 2 + (y_pred - y_gpu) ** 2)
            # Weight by actual batch size
            loss_list.append(batch_mse * batch_size_actual)
        
        # Average over all data points (weighted sum / total points)
        loss_data = torch.stack(loss_list).sum() / n_data
        
        return loss_data
    
    def compute_loss_for_param_sets(self, param_sets, t_collocation, t_data, x_data, 
                                     y_data, A_data, B_data, x0_data, y0_data, 
                                     lambda_physics, lambda_ic, lambda_data, 
                                     batch_size=50, data_batch_size=5000):
        """
        Compute total loss for a set of parameters (train or val) using batching
        
        batch_size: number of parameter sets to process at once (prevents OOM)
        data_batch_size: number of data points to process at once in data loss
        
        Note: We use torch.stack + mean instead of accumulating with += to properly
              maintain gradient graphs and avoid "does not require grad" errors
        """
        # Compute physics loss for all parameter sets in batches
        loss_phys_list = []
        n_batches = (len(param_sets) + batch_size - 1) // batch_size
        
        # Ensure gradients are enabled for physics loss computation
        with torch.set_grad_enabled(True):
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(param_sets))
                batch_params = param_sets[start_idx:end_idx]
                
                # Accumulate losses for this batch
                for params in batch_params:
                    loss_phys_list.append(self.physics_loss(t_collocation, params['A'], params['B'], params['x0'], params['y0']))
        
        # Average all losses (stack and mean to maintain gradient)
        loss_phys = torch.stack(loss_phys_list).mean()
        
        # Compute initial condition loss for all parameter sets in batches
        loss_ic_list = []
        with torch.set_grad_enabled(True):
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(param_sets))
                batch_params = param_sets[start_idx:end_idx]
                
                # Accumulate losses for this batch
                for params in batch_params:
                    loss_ic_list.append(self.initial_condition_loss(params))
        
        # Average all losses (stack and mean to maintain gradient)
        loss_ic = torch.stack(loss_ic_list).mean()
        
        # Compute data loss with batching (always with gradients enabled for training)
        with torch.set_grad_enabled(True):
            loss_dat = self.data_loss(t_data, x_data, y_data, A_data, B_data, x0_data, y0_data, data_batch_size)
        
        # Total loss
        loss_total = lambda_physics * loss_phys + lambda_ic * loss_ic + lambda_data * loss_dat
        
        return loss_total, loss_phys, loss_ic, loss_dat
    
    def prepare_data(self, param_sets, n_data_points):
        """Generate training/validation data for parameter sets"""
        all_t_data = []
        all_x_data = []
        all_y_data = []
        all_A_data = []
        all_B_data = []
        all_x0_data = []
        all_y0_data = []
        
        n_total = len(param_sets)
        print(f"  Generating RK4 solutions for {n_total} parameter sets...")
        
        for idx, params in enumerate(param_sets):
            t, x, y = generate_rk4_data(
                params['A'], params['B'], params['x0'], params['y0'],
                self.t_min, self.t_max, n_data_points
            )
            all_t_data.append(t)
            all_x_data.append(x)
            all_y_data.append(y)
            all_A_data.append(np.full_like(t, params['A']))
            all_B_data.append(np.full_like(t, params['B']))
            all_x0_data.append(np.full_like(t, params['x0']))
            all_y0_data.append(np.full_like(t, params['y0']))
            
            # Progress updates every 10% or every 100 sets
            if (idx + 1) % max(1, n_total // 10) == 0 or (idx + 1) % 100 == 0:
                percent = (idx + 1) / n_total * 100
                print(f"    Progress: {idx+1}/{n_total} ({percent:.1f}%)")
        
        print(f"    Completed! Generated {n_total} RK4 solutions.")
        
        # Concatenate all data
        print(f"  Concatenating data...")
        t_data_np = np.concatenate(all_t_data)
        x_data_np = np.concatenate(all_x_data)
        y_data_np = np.concatenate(all_y_data)
        A_data_np = np.concatenate(all_A_data)
        B_data_np = np.concatenate(all_B_data)
        x0_data_np = np.concatenate(all_x0_data)
        y0_data_np = np.concatenate(all_y0_data)
        
        # Convert to tensors but KEEP ON CPU to save GPU memory
        # Data will be transferred to GPU in batches during training
        t_data = torch.tensor(t_data_np, dtype=torch.float32, device='cpu').view(-1, 1)
        x_data = torch.tensor(x_data_np, dtype=torch.float32, device='cpu').view(-1, 1)
        y_data = torch.tensor(y_data_np, dtype=torch.float32, device='cpu').view(-1, 1)
        A_data = torch.tensor(A_data_np, dtype=torch.float32, device='cpu').view(-1, 1)
        B_data = torch.tensor(B_data_np, dtype=torch.float32, device='cpu').view(-1, 1)
        x0_data = torch.tensor(x0_data_np, dtype=torch.float32, device='cpu').view(-1, 1)
        y0_data = torch.tensor(y0_data_np, dtype=torch.float32, device='cpu').view(-1, 1)
        
        return t_data, x_data, y_data, A_data, B_data, x0_data, y0_data
    
    def train(self, n_collocation=3000, n_epochs=50000, 
              learning_rate=5e-4, lambda_physics=1.0, lambda_ic=100.0,
              lambda_data=10.0, n_data_points=100,
              print_every=100, patience=500, batch_size=50, data_batch_size=5000,
              n_params_per_epoch=100, scheduler_type='cosine_warm_restarts',
              warmup_epochs=500, T_0=1000, T_mult=2, checkpoint_every=1000):
        """
        Train the model on multiple parameter sets with validation
        
        patience: number of epochs to wait for validation improvement before early stopping
        batch_size: number of parameter sets to process at once (prevents GPU OOM)
        data_batch_size: number of data points to process at once in data loss (prevents GPU OOM)
        n_params_per_epoch: number of parameter sets to randomly sample per training epoch
                           (smaller = less memory, more epochs needed for convergence)
        scheduler_type: 'cosine_warm_restarts' (recommended) or 'reduce_on_plateau'
        warmup_epochs: number of epochs for learning rate warmup (linear increase)
        T_0: initial restart period for cosine annealing (epochs between restarts)
        T_mult: multiplier for restart period after each restart (2 = doubling)
        checkpoint_every: save checkpoint (model, plots) every N epochs for crash recovery
        """
        print("=" * 80)
        print("MULTI-PARAMETER BRUSSELATOR PINN TRAINING")
        print("=" * 80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"\nDataset sizes:")
        print(f"  Training parameter sets: {len(self.param_sets_train)}")
        print(f"  Validation parameter sets: {len(self.param_sets_val)}")
        print(f"  Data points per parameter set: {n_data_points}")
        print(f"\nModel architecture:")
        print(f"  Hidden layers: 8 layers x 128 neurons")
        print(f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"\nTraining configuration:")
        print(f"  Collocation points: {n_collocation}")
        print(f"  Max epochs: {n_epochs}")
        print(f"  Learning rate: {learning_rate:.2e}")
        print(f"  Lambda physics: {lambda_physics}")
        print(f"  Lambda IC: {lambda_ic}")
        print(f"  Lambda data: {lambda_data}")
        print(f"  Param sets per epoch: {n_params_per_epoch}/{len(self.param_sets_train)} (sampled)")
        print(f"  Batch size (param sets): {batch_size}")
        print(f"  Batch size (data points): {data_batch_size}")
        print(f"  Early stopping patience: {patience} epochs")
        print(f"\nLearning Rate Schedule:")
        print(f"  Scheduler type: {scheduler_type}")
        print(f"  Warmup epochs: {warmup_epochs}")
        if scheduler_type == 'cosine_warm_restarts':
            print(f"  Cosine T_0 (first restart): {T_0} epochs")
            print(f"  Cosine T_mult (period multiplier): {T_mult}")
            # Calculate restart epochs
            restarts = []
            current = T_0
            total = T_0
            for i in range(5):  # Show first 5 restarts
                restarts.append(total)
                current *= T_mult
                total += current
            print(f"  Planned restarts at epochs: {restarts}")
        print("=" * 80)
        
        # Generate training data
        print("\nGenerating training data using RK4...")
        t_data_train, x_data_train, y_data_train, A_data_train, B_data_train, x0_data_train, y0_data_train = \
            self.prepare_data(self.param_sets_train, n_data_points)
        print(f"Total training data points: {len(t_data_train)}")
        print(f"Training data stored on: {t_data_train.device} (CPU to save GPU memory)")
        
        # Generate validation data
        print("Generating validation data using RK4...")
        t_data_val, x_data_val, y_data_val, A_data_val, B_data_val, x0_data_val, y0_data_val = \
            self.prepare_data(self.param_sets_val, n_data_points)
        print(f"Total validation data points: {len(t_data_val)}")
        print(f"Validation data stored on: {t_data_val.device} (CPU to save GPU memory)")
        
        # Generate collocation points
        t_collocation = torch.linspace(
            self.t_min, self.t_max, n_collocation, 
            device=self.device
        ).view(-1, 1)
        
        # Optimizer with weight decay for regularization
        # Using AdamW for better weight decay handling
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5, betas=(0.9, 0.999))
        
        # Learning rate scheduler selection
        if scheduler_type == 'cosine_warm_restarts':
            # Cosine Annealing with Warm Restarts - helps escape local minima
            # T_0: first restart period, T_mult: multiplier for subsequent periods
            # Example: T_0=1000, T_mult=2 → restarts at epoch 1000, 3000, 7000, 15000...
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-6)
            print(f"  Scheduler: Cosine Annealing with Warm Restarts (T_0={T_0}, T_mult={T_mult})")
            print(f"  Restart schedule: {T_0}, {T_0*(1+T_mult)}, {T_0*(1+T_mult+T_mult**2)}...")
        else:
            # Fallback to ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, 
                                          patience=200, min_lr=1e-6)
            print(f"  Scheduler: ReduceLROnPlateau (factor=0.7, patience=200)")
        
        # Warmup settings
        if warmup_epochs > 0:
            print(f"  Learning rate warmup: {warmup_epochs} epochs")
        
        # Check initial GPU memory if CUDA is available
        if torch.cuda.is_available() and self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"\nInitial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        # Verify model setup
        n_params_with_grad = sum(p.requires_grad for p in self.model.parameters())
        n_params_total = len(list(self.model.parameters()))
        print(f"\nModel parameters requiring gradients: {n_params_with_grad}/{n_params_total}")
        if n_params_with_grad == 0:
            print("WARNING: No model parameters require gradients!")
        
        # Training loop
        print("\nStarting training...")
        print("=" * 80)
        
        # Register this instance globally for signal handler
        global _PINN_INSTANCE
        _PINN_INSTANCE = self
        
        # Checkpoint settings - save periodically in case of termination
        last_checkpoint_epoch = 0
        print(f"  Periodic checkpoints: every {checkpoint_every} epochs")
        
        start_time = time.time()
        base_lr = learning_rate  # Store for warmup calculation
        
        for epoch in range(n_epochs):
            # Check for termination signal
            global _TERMINATION_REQUESTED
            if _TERMINATION_REQUESTED:
                print(f"\nTermination requested at epoch {epoch+1}. Stopping training...")
                break
            # Learning rate warmup (linear increase from 0.1*lr to lr)
            if warmup_epochs > 0 and epoch < warmup_epochs:
                warmup_factor = 0.1 + 0.9 * (epoch / warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = base_lr * warmup_factor
            
            # Training mode
            self.model.train()
            optimizer.zero_grad()
            
            # Randomly sample parameter sets for this epoch (mini-batch training)
            # This prevents GPU OOM by not processing all 5000 sets at once
            n_train_total = len(self.param_sets_train)
            if n_params_per_epoch < n_train_total:
                train_indices = np.random.choice(n_train_total, n_params_per_epoch, replace=False)
                param_sets_epoch = [self.param_sets_train[i] for i in train_indices]
                
                # Filter data to only include sampled parameter sets
                # Each param set has n_data_points, so we need to get the right slices
                data_indices = []
                for idx in train_indices:
                    start = idx * n_data_points
                    end = (idx + 1) * n_data_points
                    data_indices.extend(range(start, end))
                
                t_data_epoch = t_data_train[data_indices]
                x_data_epoch = x_data_train[data_indices]
                y_data_epoch = y_data_train[data_indices]
                A_data_epoch = A_data_train[data_indices]
                B_data_epoch = B_data_train[data_indices]
                x0_data_epoch = x0_data_train[data_indices]
                y0_data_epoch = y0_data_train[data_indices]
            else:
                # Use all training data
                param_sets_epoch = self.param_sets_train
                t_data_epoch = t_data_train
                x_data_epoch = x_data_train
                y_data_epoch = y_data_train
                A_data_epoch = A_data_train
                x0_data_epoch = x0_data_train
                y0_data_epoch = y0_data_train
                B_data_epoch = B_data_train
            
            # Compute training loss with batching
            loss_train, loss_phys_train, loss_ic_train, loss_dat_train = \
                self.compute_loss_for_param_sets(
                    param_sets_epoch, t_collocation, 
                    t_data_epoch, x_data_epoch, y_data_epoch, A_data_epoch, B_data_epoch,
                    x0_data_epoch, y0_data_epoch,
                    lambda_physics, lambda_ic, lambda_data, batch_size, data_batch_size
                )
            
            # Check for NaN in training
            if torch.isnan(loss_train):
                print(f"\nERROR: NaN detected in training loss at epoch {epoch+1}")
                print(f"  Physics Loss: {loss_phys_train.item()}")
                print(f"  IC Loss: {loss_ic_train.item()}")
                print(f"  Data Loss: {loss_dat_train.item()}")
                print("Stopping training.")
                break
            
            # Backward pass
            loss_train.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Validation (no gradient computation) with batching
            # Also sample validation sets to reduce memory (no gradients but still uses memory)
            self.model.eval()
            with torch.no_grad():
                n_val_total = len(self.param_sets_val)
                n_val_per_epoch = min(n_params_per_epoch, n_val_total)  # Same or fewer than training
                
                if n_val_per_epoch < n_val_total:
                    val_indices = np.random.choice(n_val_total, n_val_per_epoch, replace=False)
                    param_sets_val_epoch = [self.param_sets_val[i] for i in val_indices]
                    
                    # Filter validation data
                    data_indices_val = []
                    for idx in val_indices:
                        start = idx * n_data_points
                        end = (idx + 1) * n_data_points
                        data_indices_val.extend(range(start, end))
                    
                    t_data_val_epoch = t_data_val[data_indices_val]
                    x_data_val_epoch = x_data_val[data_indices_val]
                    y_data_val_epoch = y_data_val[data_indices_val]
                    A_data_val_epoch = A_data_val[data_indices_val]
                    B_data_val_epoch = B_data_val[data_indices_val]
                    x0_data_val_epoch = x0_data_val[data_indices_val]
                    y0_data_val_epoch = y0_data_val[data_indices_val]
                else:
                    param_sets_val_epoch = self.param_sets_val
                    t_data_val_epoch = t_data_val
                    x_data_val_epoch = x_data_val
                    y_data_val_epoch = y_data_val
                    A_data_val_epoch = A_data_val
                    B_data_val_epoch = B_data_val
                    x0_data_val_epoch = x0_data_val
                    y0_data_val_epoch = y0_data_val
                
                loss_val, loss_phys_val, loss_ic_val, loss_dat_val = \
                    self.compute_loss_for_param_sets(
                        param_sets_val_epoch, t_collocation,
                        t_data_val_epoch, x_data_val_epoch, y_data_val_epoch, A_data_val_epoch, B_data_val_epoch,
                        x0_data_val_epoch, y0_data_val_epoch,
                        lambda_physics, lambda_ic, lambda_data, batch_size, data_batch_size
                    )
            
            # Update learning rate scheduler (only after warmup)
            if epoch >= warmup_epochs:
                if scheduler_type == 'cosine_warm_restarts':
                    scheduler.step()  # Epoch-based stepping for cosine annealing
                else:
                    scheduler.step(loss_val)  # Loss-based stepping for ReduceLROnPlateau
            
            # Store history
            self.loss_history['train_total'].append(loss_train.item())
            self.loss_history['train_physics'].append(loss_phys_train.item())
            self.loss_history['train_ic'].append(loss_ic_train.item())
            self.loss_history['train_data'].append(loss_dat_train.item())
            self.loss_history['val_total'].append(loss_val.item())
            self.loss_history['val_physics'].append(loss_phys_val.item())
            self.loss_history['val_ic'].append(loss_ic_val.item())
            self.loss_history['val_data'].append(loss_dat_val.item())
            
            # Early stopping check (based on VALIDATION loss to prevent overfitting)
            # We track validation loss because training on validation data would defeat
            # the purpose. Model is saved when it performs best on unseen validation data.
            if loss_val.item() < self.best_val_loss:
                self.best_val_loss = loss_val.item()
                self.patience_counter = 0
                self.best_epoch = epoch + 1
                # Save best model state
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                self.patience_counter += 1
            
            # Check if we should stop
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"No improvement in validation loss for {patience} epochs")
                print(f"Best validation loss: {self.best_val_loss:.6e} at epoch {self.best_epoch}")
                # Restore best model
                if self.best_model_state is not None:
                    self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
                    print("Restored best model weights")
                break
            
            # Print progress
            if (epoch + 1) % print_every == 0 or epoch == 0:
                elapsed = time.time() - start_time
                current_lr = optimizer.param_groups[0]['lr']
                epoch_per_sec = (epoch + 1) / elapsed
                eta = (n_epochs - epoch - 1) / epoch_per_sec if epoch_per_sec > 0 else 0
                
                # Memory info for first few epochs
                mem_str = ""
                if epoch < 3 and torch.cuda.is_available() and self.device.type == 'cuda':
                    mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
                    mem_str = f" | GPU: {mem_allocated:.2f}GB"
                
                print(f"Epoch {epoch+1:6d}/{n_epochs} | "
                      f"Train: {loss_train.item():.4e} | "
                      f"Val: {loss_val.item():.4e} | "
                      f"T_Phys: {loss_phys_train.item():.4e} | "
                      f"V_Phys: {loss_phys_val.item():.4e} | "
                      f"LR: {current_lr:.2e} | "
                      f"Pat: {self.patience_counter:3d}/{patience} | "
                      f"Best: {self.best_val_loss:.4e} | "
                      f"ETA: {eta/60:.1f}min{mem_str}")
            
            # Periodic checkpoint - save best model, loss history, and examples every N epochs
            # This ensures we don't lose progress if the job is terminated
            if (epoch + 1) % checkpoint_every == 0 and (epoch + 1) > last_checkpoint_epoch:
                last_checkpoint_epoch = epoch + 1
                print(f"\n--- Periodic Checkpoint at epoch {epoch+1} ---")
                try:
                    # Save best model
                    if self.best_model_state is not None:
                        self.save_best_model('brusselator_pinn_best_model.pth')
                    # Save loss history plot
                    self.plot_loss_history()
                    # Save training/validation examples
                    self.plot_final_results()
                    print(f"--- Checkpoint complete ---\n")
                except Exception as e:
                    print(f"Warning: Checkpoint failed: {e}")
        
        total_time = time.time() - start_time
        print("=" * 80)
        
        # Determine how training ended
        if _TERMINATION_REQUESTED:
            print("TRAINING INTERRUPTED - Terminated by signal")
        elif self.patience_counter >= patience:
            print("TRAINING COMPLETED - Early stopping triggered")
        else:
            print("TRAINING COMPLETED - Reached max epochs")
        
        print(f"Training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"Total epochs trained: {epoch+1}")
        print(f"Best validation loss: {self.best_val_loss:.6e} at epoch {self.best_epoch}")
        if len(self.loss_history['train_total']) > 0:
            print(f"Final training loss: {self.loss_history['train_total'][-1]:.6e}")
            print(f"Final validation loss: {self.loss_history['val_total'][-1]:.6e}")
        print("=" * 80)
        
        # Always save final checkpoint after training ends (regardless of how)
        print("\nSaving final outputs...")
        try:
            if self.best_model_state is not None:
                self.save_best_model('brusselator_pinn_best_model.pth')
            self.save_model('brusselator_pinn_last_model.pth')
            self.plot_loss_history()
            self.plot_final_results()
            self.save_training_summary('training_summary.json')
            print("All outputs saved successfully!")
        except Exception as e:
            print(f"Warning: Error saving final outputs: {e}")
    
    def predict(self, t, A, B, x0, y0):
        """Extract predictions from PINN for given A, B, x0, y0 parameters"""
        self.model.eval()
        with torch.no_grad():
            t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device).view(-1, 1)
            A_tensor = torch.full_like(t_tensor, A)
            B_tensor = torch.full_like(t_tensor, B)
            x0_tensor = torch.full_like(t_tensor, x0)
            y0_tensor = torch.full_like(t_tensor, y0)
            xy_pred = self.model(t_tensor, A_tensor, B_tensor, x0_tensor, y0_tensor)
            x_pred = xy_pred[:, 0].cpu().numpy()
            y_pred = xy_pred[:, 1].cpu().numpy()
        return x_pred, y_pred
    
    def plot_final_results(self):
        """Plot comprehensive final results (called once after training)"""
        print("\nGenerating final plots...")
        
        # Select subset of parameter sets to plot (max 6 for readability)
        n_train_plot = min(3, len(self.param_sets_train))
        n_val_plot = min(3, len(self.param_sets_val))
        
        train_indices = np.random.choice(len(self.param_sets_train), n_train_plot, replace=False)
        val_indices = np.random.choice(len(self.param_sets_val), n_val_plot, replace=False)
        
        # Plot training examples
        fig_train = plt.figure(figsize=(16, 4 * n_train_plot))
        
        for plot_idx, idx in enumerate(train_indices):
            params = self.param_sets_train[idx]
            
            # Generate prediction points
            t_plot = np.linspace(self.t_min, self.t_max, 1000)
            x_pred, y_pred = self.predict(t_plot, params['A'], params['B'], params['x0'], params['y0'])
            
            # Generate reference data
            t_ref, x_ref, y_ref = generate_rk4_data(
                params['A'], params['B'], params['x0'], params['y0'],
                self.t_min, self.t_max, 200
            )
            
            # Time series for x
            ax1 = plt.subplot(n_train_plot, 3, plot_idx * 3 + 1)
            ax1.plot(t_ref, x_ref, 'b-', linewidth=2, alpha=0.7, label='RK4 x(t)')
            ax1.plot(t_plot, x_pred, 'r--', linewidth=2, label='PINN x(t)')
            ax1.set_xlabel('Time', fontsize=10)
            ax1.set_ylabel('x(t)', fontsize=10)
            ax1.set_title(f'Train Set {idx}: A={params["A"]:.2f}, B={params["B"]:.2f}', fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=8)
            
            # Time series for y
            ax2 = plt.subplot(n_train_plot, 3, plot_idx * 3 + 2)
            ax2.plot(t_ref, y_ref, 'b-', linewidth=2, alpha=0.7, label='RK4 y(t)')
            ax2.plot(t_plot, y_pred, 'r--', linewidth=2, label='PINN y(t)')
            ax2.set_xlabel('Time', fontsize=10)
            ax2.set_ylabel('y(t)', fontsize=10)
            ax2.set_title(f'Train Set {idx}: y vs Time', fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8)
            
            # Phase portrait
            ax3 = plt.subplot(n_train_plot, 3, plot_idx * 3 + 3)
            ax3.plot(x_ref, y_ref, 'b-', linewidth=2, alpha=0.7, label='RK4')
            ax3.plot(x_pred, y_pred, 'r--', linewidth=2, label='PINN')
            ax3.plot(x_pred[0], y_pred[0], 'ko', markersize=8, label='IC')
            ax3.set_xlabel('x', fontsize=10)
            ax3.set_ylabel('y', fontsize=10)
            ax3.set_title(f'Train Set {idx}: Phase Portrait', fontsize=11)
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=8)
        
        fig_train.suptitle('Training Set Examples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        train_path = os.path.join(self.output_dir, 'plots', 'training_examples.png')
        plt.savefig(train_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {train_path}")
        plt.close()
        
        # Plot validation examples
        fig_val = plt.figure(figsize=(16, 4 * n_val_plot))
        
        for plot_idx, idx in enumerate(val_indices):
            params = self.param_sets_val[idx]
            
            # Generate prediction points
            t_plot = np.linspace(self.t_min, self.t_max, 1000)
            x_pred, y_pred = self.predict(t_plot, params['A'], params['B'], params['x0'], params['y0'])
            
            # Generate reference data
            t_ref, x_ref, y_ref = generate_rk4_data(
                params['A'], params['B'], params['x0'], params['y0'],
                self.t_min, self.t_max, 200
            )
            
            # Time series for x
            ax1 = plt.subplot(n_val_plot, 3, plot_idx * 3 + 1)
            ax1.plot(t_ref, x_ref, 'b-', linewidth=2, alpha=0.7, label='RK4 x(t)')
            ax1.plot(t_plot, x_pred, 'r--', linewidth=2, label='PINN x(t)')
            ax1.set_xlabel('Time', fontsize=10)
            ax1.set_ylabel('x(t)', fontsize=10)
            ax1.set_title(f'Val Set {idx}: A={params["A"]:.2f}, B={params["B"]:.2f}', fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=8)
            
            # Time series for y
            ax2 = plt.subplot(n_val_plot, 3, plot_idx * 3 + 2)
            ax2.plot(t_ref, y_ref, 'b-', linewidth=2, alpha=0.7, label='RK4 y(t)')
            ax2.plot(t_plot, y_pred, 'r--', linewidth=2, label='PINN y(t)')
            ax2.set_xlabel('Time', fontsize=10)
            ax2.set_ylabel('y(t)', fontsize=10)
            ax2.set_title(f'Val Set {idx}: y vs Time', fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8)
            
            # Phase portrait
            ax3 = plt.subplot(n_val_plot, 3, plot_idx * 3 + 3)
            ax3.plot(x_ref, y_ref, 'b-', linewidth=2, alpha=0.7, label='RK4')
            ax3.plot(x_pred, y_pred, 'r--', linewidth=2, label='PINN')
            ax3.plot(x_pred[0], y_pred[0], 'ko', markersize=8, label='IC')
            ax3.set_xlabel('x', fontsize=10)
            ax3.set_ylabel('y', fontsize=10)
            ax3.set_title(f'Val Set {idx}: Phase Portrait', fontsize=11)
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=8)
        
        fig_val.suptitle('Validation Set Examples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        val_path = os.path.join(self.output_dir, 'plots', 'validation_examples.png')
        plt.savefig(val_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {val_path}")
        plt.close()
        
        # Plot loss history
        fig_loss, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        if len(self.loss_history['train_total']) > 0:
            # Total loss
            axes[0, 0].semilogy(self.loss_history['train_total'], 'b-', linewidth=2, label='Train')
            axes[0, 0].semilogy(self.loss_history['val_total'], 'r-', linewidth=2, label='Val')
            axes[0, 0].axvline(self.best_epoch-1, color='g', linestyle='--', linewidth=1, label=f'Best (epoch {self.best_epoch})')
            axes[0, 0].set_xlabel('Epoch', fontsize=12)
            axes[0, 0].set_ylabel('Total Loss', fontsize=12)
            axes[0, 0].set_title('Total Loss', fontsize=14)
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Physics loss
            axes[0, 1].semilogy(self.loss_history['train_physics'], 'b-', linewidth=2, label='Train')
            axes[0, 1].semilogy(self.loss_history['val_physics'], 'r-', linewidth=2, label='Val')
            axes[0, 1].set_xlabel('Epoch', fontsize=12)
            axes[0, 1].set_ylabel('Physics Loss', fontsize=12)
            axes[0, 1].set_title('Physics Loss', fontsize=14)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            
            # IC loss
            axes[1, 0].semilogy(self.loss_history['train_ic'], 'b-', linewidth=2, label='Train')
            axes[1, 0].semilogy(self.loss_history['val_ic'], 'r-', linewidth=2, label='Val')
            axes[1, 0].set_xlabel('Epoch', fontsize=12)
            axes[1, 0].set_ylabel('IC Loss', fontsize=12)
            axes[1, 0].set_title('Initial Condition Loss', fontsize=14)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            
            # Data loss
            axes[1, 1].semilogy(self.loss_history['train_data'], 'b-', linewidth=2, label='Train')
            axes[1, 1].semilogy(self.loss_history['val_data'], 'r-', linewidth=2, label='Val')
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Data Loss', fontsize=12)
            axes[1, 1].set_title('Data Loss', fontsize=14)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        loss_path = os.path.join(self.output_dir, 'plots', 'loss_history.png')
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {loss_path}")
        plt.close()
        
        print("All plots saved successfully!")
    
    def evaluate(self, param_sets, set_name="Test"):
        """Evaluate model on a set of parameters"""
        print(f"\n{'=' * 80}")
        print(f"{set_name.upper()} SET EVALUATION")
        print(f"{'=' * 80}")
        
        all_errors_x = []
        all_errors_y = []
        all_rmse_x = []
        all_rmse_y = []
        
        for idx, params in enumerate(param_sets):
            # Generate reference data
            t_ref, x_ref, y_ref = generate_rk4_data(
                params['A'], params['B'], params['x0'], params['y0'],
                self.t_min, self.t_max, n_points=200
            )
            
            # Get predictions
            x_pred, y_pred = self.predict(t_ref, params['A'], params['B'], params['x0'], params['y0'])
            
            # Compute errors
            error_x = np.mean(np.abs(x_pred - x_ref))
            error_y = np.mean(np.abs(y_pred - y_ref))
            rmse_x = np.sqrt(np.mean((x_pred - x_ref) ** 2))
            rmse_y = np.sqrt(np.mean((y_pred - y_ref) ** 2))
            
            all_errors_x.append(error_x)
            all_errors_y.append(error_y)
            all_rmse_x.append(rmse_x)
            all_rmse_y.append(rmse_y)
        
        # Print statistics
        print(f"Number of parameter sets: {len(param_sets)}")
        print(f"\nMean Absolute Error (MAE):")
        print(f"  x: {np.mean(all_errors_x):.6e} ± {np.std(all_errors_x):.6e}")
        print(f"  y: {np.mean(all_errors_y):.6e} ± {np.std(all_errors_y):.6e}")
        print(f"\nRoot Mean Square Error (RMSE):")
        print(f"  x: {np.mean(all_rmse_x):.6e} ± {np.std(all_rmse_x):.6e}")
        print(f"  y: {np.mean(all_rmse_y):.6e} ± {np.std(all_rmse_y):.6e}")
        print(f"{'=' * 80}")
        
        return {
            'mae_x': all_errors_x,
            'mae_y': all_errors_y,
            'rmse_x': all_rmse_x,
            'rmse_y': all_rmse_y
        }
    
    def save_model(self, filename='brusselator_pinn_model.pth'):
        """Save the trained model (current state)"""
        path = os.path.join(self.output_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'param_sets_train': self.param_sets_train,
            'param_sets_val': self.param_sets_val,
            't_min': self.t_min,
            't_max': self.t_max,
            'loss_history': self.loss_history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }, path)
        print(f"Model saved to {path}")
    
    def save_best_model(self, filename='brusselator_pinn_best_model.pth'):
        """Save the best model (lowest validation loss)"""
        if self.best_model_state is None:
            print("Warning: No best model state available")
            return
        
        path = os.path.join(self.output_dir, filename)
        torch.save({
            'model_state_dict': self.best_model_state,
            'param_sets_train': self.param_sets_train,
            'param_sets_val': self.param_sets_val,
            't_min': self.t_min,
            't_max': self.t_max,
            'loss_history': self.loss_history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }, path)
        print(f"Best model saved to {path} (epoch {self.best_epoch}, val_loss={self.best_val_loss:.6e})")
    
    def save_training_summary(self, filename='training_summary.json'):
        """Save training summary to JSON"""
        path = os.path.join(self.output_dir, filename)
        summary = {
            'n_train_params': len(self.param_sets_train),
            'n_val_params': len(self.param_sets_val),
            'best_val_loss': float(self.best_val_loss),
            'best_epoch': int(self.best_epoch),
            'total_epochs': len(self.loss_history['train_total']),
            'final_train_loss': float(self.loss_history['train_total'][-1]) if self.loss_history['train_total'] else None,
            'final_val_loss': float(self.loss_history['val_total'][-1]) if self.loss_history['val_total'] else None,
            't_min': float(self.t_min),
            't_max': float(self.t_max)
        }
        
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Training summary saved to {path}")
    
    def save_checkpoint_on_interrupt(self):
        """Emergency save when job is being terminated - saves everything important"""
        print("\n" + "="*80)
        print("EMERGENCY CHECKPOINT - Saving all progress before termination")
        print("="*80)
        
        try:
            # Save best model
            if self.best_model_state is not None:
                self.save_best_model('brusselator_pinn_best_model.pth')
            
            # Save last model state
            self.save_model('brusselator_pinn_last_model.pth')
            
            # Save loss history plot
            self.plot_loss_history()
            
            # Save training examples (quick version)
            self.plot_final_results()
            
            # Save training summary
            self.save_training_summary('training_summary.json')
            
            print("="*80)
            print("CHECKPOINT SAVED SUCCESSFULLY")
            print(f"  Best model: outputs/brusselator_pinn_best_model.pth (epoch {self.best_epoch})")
            print(f"  Last model: outputs/brusselator_pinn_last_model.pth")
            print(f"  Loss plots: outputs/plots/loss_history.png")
            print(f"  Examples: outputs/plots/training_examples.png, validation_examples.png")
            print("="*80)
        except Exception as e:
            print(f"ERROR during checkpoint save: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_loss_history(self):
        """Plot and save just the loss history (can be called during training)"""
        if len(self.loss_history['train_total']) == 0:
            print("No loss history to plot")
            return
        
        fig_loss, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Total loss
        axes[0, 0].semilogy(self.loss_history['train_total'], 'b-', linewidth=1, alpha=0.7, label='Train')
        axes[0, 0].semilogy(self.loss_history['val_total'], 'r-', linewidth=1, alpha=0.7, label='Val')
        if self.best_epoch > 0:
            axes[0, 0].axvline(self.best_epoch-1, color='g', linestyle='--', linewidth=1, label=f'Best (epoch {self.best_epoch})')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Total Loss', fontsize=12)
        axes[0, 0].set_title('Total Loss', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Physics loss
        axes[0, 1].semilogy(self.loss_history['train_physics'], 'b-', linewidth=1, alpha=0.7, label='Train')
        axes[0, 1].semilogy(self.loss_history['val_physics'], 'r-', linewidth=1, alpha=0.7, label='Val')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Physics Loss', fontsize=12)
        axes[0, 1].set_title('Physics Loss', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # IC loss
        axes[1, 0].semilogy(self.loss_history['train_ic'], 'b-', linewidth=1, alpha=0.7, label='Train')
        axes[1, 0].semilogy(self.loss_history['val_ic'], 'r-', linewidth=1, alpha=0.7, label='Val')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('IC Loss', fontsize=12)
        axes[1, 0].set_title('Initial Condition Loss', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Data loss
        axes[1, 1].semilogy(self.loss_history['train_data'], 'b-', linewidth=1, alpha=0.7, label='Train')
        axes[1, 1].semilogy(self.loss_history['val_data'], 'r-', linewidth=1, alpha=0.7, label='Val')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Data Loss', fontsize=12)
        axes[1, 1].set_title('Data Loss', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        # Add overall title with current status
        n_epochs = len(self.loss_history['train_total'])
        fig_loss.suptitle(f'Training Progress - {n_epochs} epochs (Best: epoch {self.best_epoch}, val_loss={self.best_val_loss:.4e})', 
                         fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        loss_path = os.path.join(self.output_dir, 'plots', 'loss_history.png')
        plt.savefig(loss_path, dpi=150, bbox_inches='tight')  # Lower DPI for faster saves
        print(f"  Loss history saved: {loss_path}")
        plt.close()


def main():
    """Main training script for multi-parameter PINN with large dataset"""
    
    # Try to import configuration, fallback to defaults if not found
    try:
        import config
        # Extract all uppercase variables from config module
        config_vars = {k: v for k, v in vars(config).items() if k.isupper()}
        print("Loaded configuration from config.py")
        print(f"Loaded {len(config_vars)} configuration variables")
        
        # Set variables from config
        N_TRAIN_SETS = config.N_TRAIN_SETS
        N_VAL_SETS = config.N_VAL_SETS
        A_RANGE = config.A_RANGE
        B_RANGE = config.B_RANGE
        X0_RANGE = config.X0_RANGE
        Y0_RANGE = config.Y0_RANGE
        T_MIN = config.T_MIN
        T_MAX = config.T_MAX
        TRAIN_SEED = config.TRAIN_SEED
        VAL_SEED = config.VAL_SEED
        OUTPUT_DIR = config.OUTPUT_DIR
        DEVICE = config.DEVICE
        N_COLLOCATION = config.N_COLLOCATION
        N_EPOCHS = config.N_EPOCHS
        LEARNING_RATE = config.LEARNING_RATE
        LAMBDA_PHYSICS = config.LAMBDA_PHYSICS
        LAMBDA_IC = config.LAMBDA_IC
        LAMBDA_DATA = config.LAMBDA_DATA
        N_DATA_POINTS = config.N_DATA_POINTS
        PRINT_EVERY = config.PRINT_EVERY
        PATIENCE = config.PATIENCE
        BATCH_SIZE = config.BATCH_SIZE
        DATA_BATCH_SIZE = config.DATA_BATCH_SIZE
        N_PARAMS_PER_EPOCH = config.N_PARAMS_PER_EPOCH
        N_EVAL_TRAIN = config.N_EVAL_TRAIN
        N_EVAL_VAL = config.N_EVAL_VAL
        # Scheduler parameters
        SCHEDULER_TYPE = getattr(config, 'SCHEDULER_TYPE', 'cosine_warm_restarts')
        WARMUP_EPOCHS = getattr(config, 'WARMUP_EPOCHS', 500)
        COSINE_T_0 = getattr(config, 'COSINE_T_0', 1000)
        COSINE_T_MULT = getattr(config, 'COSINE_T_MULT', 2)
        # Checkpoint settings
        CHECKPOINT_EVERY = getattr(config, 'CHECKPOINT_EVERY', 1000)
    except ImportError:
        print("config.py not found, using default configuration")
        # Default configuration
        N_TRAIN_SETS = 5000
        N_VAL_SETS = 1000
        A_RANGE = (0.5, 2.0)
        B_RANGE = (2.0, 4.0)
        X0_RANGE = (0.5, 2.0)
        Y0_RANGE = (0.5, 2.0)
        T_MIN = 0.0
        T_MAX = 20.0
        TRAIN_SEED = 42
        VAL_SEED = 123
        OUTPUT_DIR = 'outputs'
        DEVICE = 'auto'
        N_COLLOCATION = 3000
        N_EPOCHS = 50000
        LEARNING_RATE = 5e-4
        LAMBDA_PHYSICS = 1.0
        LAMBDA_IC = 100.0
        LAMBDA_DATA = 10.0
        N_DATA_POINTS = 50
        PRINT_EVERY = 10
        PATIENCE = 500
        BATCH_SIZE = 50
        DATA_BATCH_SIZE = 5000
        N_PARAMS_PER_EPOCH = 100
        N_EVAL_TRAIN = 100
        N_EVAL_VAL = 100
        # Scheduler parameters
        SCHEDULER_TYPE = 'cosine_warm_restarts'
        WARMUP_EPOCHS = 500
        COSINE_T_0 = 1000
        COSINE_T_MULT = 2
        # Checkpoint settings
        CHECKPOINT_EVERY = 1000
    
    # Configuration
    n_train_sets = N_TRAIN_SETS
    n_val_sets = N_VAL_SETS
    A_range = A_RANGE
    B_range = B_RANGE
    x0_range = X0_RANGE
    y0_range = Y0_RANGE
    t_min = T_MIN
    t_max = T_MAX
    output_dir = OUTPUT_DIR
    
    # Check if CUDA is available
    if DEVICE == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = DEVICE
    
    print("=" * 80)
    print("BRUSSELATOR PINN - MULTI-PARAMETER TRAINING")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"\nDataset Configuration:")
    print(f"  Training sets: {n_train_sets}")
    print(f"  Validation sets: {n_val_sets}")
    print(f"  Total parameter sets: {n_train_sets + n_val_sets}")
    print(f"\nParameter Ranges:")
    print(f"  A: {A_range}")
    print(f"  B: {B_range}")
    print(f"  x0: {x0_range}")
    print(f"  y0: {y0_range}")
    print(f"\nTime range: [{t_min}, {t_max}]")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Generate parameter sets
    print("\nGenerating random parameter combinations...")
    param_sets_train = generate_parameter_sets(
        n_train_sets, A_range, B_range, x0_range, y0_range, seed=TRAIN_SEED
    )
    param_sets_val = generate_parameter_sets(
        n_val_sets, A_range, B_range, x0_range, y0_range, seed=VAL_SEED
    )
    print(f"  Generated {len(param_sets_train)} training sets")
    print(f"  Generated {len(param_sets_val)} validation sets")
    print(f"  This will now generate {n_train_sets + n_val_sets} RK4 solutions (may take 20-60 mins)...")
    
    # Create PINN solver
    pinn = MultiParamBrusselatorPINN(
        param_sets_train=param_sets_train,
        param_sets_val=param_sets_val,
        t_min=t_min,
        t_max=t_max,
        device=device,
        output_dir=output_dir
    )
    
    # Train the model
    pinn.train(
        n_collocation=N_COLLOCATION,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        lambda_physics=LAMBDA_PHYSICS,
        lambda_ic=LAMBDA_IC,
        lambda_data=LAMBDA_DATA,
        n_data_points=N_DATA_POINTS,
        print_every=PRINT_EVERY,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        data_batch_size=DATA_BATCH_SIZE,
        n_params_per_epoch=N_PARAMS_PER_EPOCH,
        scheduler_type=SCHEDULER_TYPE,
        warmup_epochs=WARMUP_EPOCHS,
        T_0=COSINE_T_0,
        T_mult=COSINE_T_MULT,
        checkpoint_every=CHECKPOINT_EVERY
    )
    
    # Evaluate on training and validation sets (sample for speed)
    # Skip evaluation if terminated early to save time
    if not _TERMINATION_REQUESTED:
        n_eval_train = min(N_EVAL_TRAIN, len(param_sets_train))
        n_eval_val = min(N_EVAL_VAL, len(param_sets_val))
        
        train_metrics = pinn.evaluate(param_sets_train[:n_eval_train], set_name="Training Sample")
        val_metrics = pinn.evaluate(param_sets_val[:n_eval_val], set_name="Validation Sample")
    else:
        print("\nSkipping evaluation due to early termination")
    
    # Note: Models and plots are now saved automatically in train() function
    
    print("\n" + "=" * 80)
    print("JOB FINISHED")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All outputs saved to: {output_dir}")
    print(f"  - Best model: {output_dir}/brusselator_pinn_best_model.pth")
    print(f"  - Last model: {output_dir}/brusselator_pinn_last_model.pth")
    print(f"  - Loss plots: {output_dir}/plots/loss_history.png")
    print(f"  - Examples: {output_dir}/plots/training_examples.png, validation_examples.png")
    print("=" * 80)


if __name__ == "__main__":
    main()

