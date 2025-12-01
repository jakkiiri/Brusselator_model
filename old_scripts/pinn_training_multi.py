import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

# Brusselator RHS
def brusselator_rhs(x, y, A, B):
    dxdt = A + x * x * y - (B + 1.0) * x
    dydt = B * x - x * x * y
    return dxdt, dydt


# RK4 Integrator with adaptive stepping for stability
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


# Expanded PINN Class with much larger architecture and parameter inputs
class PINN(nn.Module):   
    def __init__(self, hidden_layers=[128, 128, 128, 128, 128, 128, 128, 128], activation=nn.Tanh()):
        super(PINN, self).__init__()
        
        # Input layer: time t, A, B (3 inputs total)
        layers = [nn.Linear(3, hidden_layers[0]), activation]
        
        # Hidden layers with residual-like connections
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
        for i, m in enumerate(self.network.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                # Initialize output layer bias to typical Brusselator values
                if i == len(list(self.network.modules())) - 1:
                    nn.init.constant_(m.bias, 1.0)
                else:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, t, A, B):
        # Forward pass with time and parameters as input
        # Concatenate inputs: [t, A, B]
        inputs = torch.cat([t, A, B], dim=1)
        out = self.network(inputs)
        # Return raw output (network will learn proper scaling)
        return out


# Multi-Parameter BrusselatorPINN Class
class MultiParamBrusselatorPINN:
    def __init__(self, param_sets, t_min=0.0, t_max=20.0, device='cpu'):
        """
        Initialize the PINN solver for multiple parameter sets
        
        param_sets: list of dicts with keys ['A', 'B', 'x0', 'y0']
        """
        self.param_sets = param_sets
        self.t_min = t_min
        self.t_max = t_max
        self.device = torch.device(device)
        
        # Create model with expanded architecture (8 layers x 128 neurons)
        self.model = PINN(hidden_layers=[128, 128, 128, 128, 128, 128, 128, 128]).to(self.device)
        
        # Loss history
        self.loss_history = {
            'total': [],
            'physics': [],
            'ic': [],
            'data': []
        }
        
        # Early stopping tracking
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
    
    def physics_loss(self, t_collocation, A, B):
        """Compute physics loss (PDE residuals) for given A, B parameters"""
        # Clone and enable gradients for this computation
        t = t_collocation.clone().detach().requires_grad_(True)
        
        # Create tensors for A and B parameters
        A_tensor = torch.full_like(t, A)
        B_tensor = torch.full_like(t, B)
        
        # Forward pass
        xy = self.model(t, A_tensor, B_tensor)
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        
        # Compute derivatives using automatic differentiation
        dx_dt = torch.autograd.grad(
            outputs=x,
            inputs=t,
            grad_outputs=torch.ones_like(x),
            create_graph=True,
            retain_graph=True
        )[0]
        
        dy_dt = torch.autograd.grad(
            outputs=y,
            inputs=t,
            grad_outputs=torch.ones_like(y),
            create_graph=True
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
        
        xy0_pred = self.model(t0, A_tensor, B_tensor)
        
        x0_pred = xy0_pred[:, 0]
        y0_pred = xy0_pred[:, 1]
        
        x0 = param_set['x0']
        y0 = param_set['y0']
        
        loss_ic = (x0_pred - x0) ** 2 + (y0_pred - y0) ** 2
        
        return loss_ic
    
    def data_loss(self, t_data, x_data, y_data, A_data, B_data):
        """Compute data loss"""
        if t_data is None or len(t_data) == 0:
            return torch.tensor(0.0, device=self.device)
        
        xy_pred = self.model(t_data, A_data, B_data)
        x_pred = xy_pred[:, 0:1]
        y_pred = xy_pred[:, 1:2]
        
        loss_data = torch.mean((x_pred - x_data) ** 2 + (y_pred - y_data) ** 2)
        
        return loss_data
    
    def train(self, n_collocation=1000, n_epochs=20000, 
              learning_rate=1e-3, lambda_physics=1.0, lambda_ic=10.0,
              lambda_data=10.0, n_data_points=50,
              print_every=500, plot_every=5000, patience=100):
        """
        Train the model on multiple parameter sets
        
        patience: number of epochs to wait for improvement before early stopping
        """
        # Generate training data for all parameter sets using RK4
        print("Generating training data using RK4...")
        all_t_data = []
        all_x_data = []
        all_y_data = []
        all_A_data = []
        all_B_data = []
        
        for params in self.param_sets:
            t, x, y = generate_rk4_data(
                params['A'], params['B'], params['x0'], params['y0'],
                self.t_min, self.t_max, n_data_points
            )
            all_t_data.append(t)
            all_x_data.append(x)
            all_y_data.append(y)
            # Store corresponding A, B values for each data point
            all_A_data.append(np.full_like(t, params['A']))
            all_B_data.append(np.full_like(t, params['B']))
            print(f"  Generated data for A={params['A']}, B={params['B']}, "
                  f"x0={params['x0']}, y0={params['y0']}")
        
        # Concatenate all data
        t_data_np = np.concatenate(all_t_data)
        x_data_np = np.concatenate(all_x_data)
        y_data_np = np.concatenate(all_y_data)
        A_data_np = np.concatenate(all_A_data)
        B_data_np = np.concatenate(all_B_data)
        
        # Convert to tensors
        t_data = torch.tensor(t_data_np, dtype=torch.float32, device=self.device).view(-1, 1)
        x_data = torch.tensor(x_data_np, dtype=torch.float32, device=self.device).view(-1, 1)
        y_data = torch.tensor(y_data_np, dtype=torch.float32, device=self.device).view(-1, 1)
        A_data = torch.tensor(A_data_np, dtype=torch.float32, device=self.device).view(-1, 1)
        B_data = torch.tensor(B_data_np, dtype=torch.float32, device=self.device).view(-1, 1)
        
        print(f"Total training data points: {len(t_data)}")
        
        # Generate collocation points
        t_collocation = torch.linspace(
            self.t_min, self.t_max, n_collocation, 
            device=self.device
        ).view(-1, 1)
        
        # Optimizer with weight decay for regularization
        optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        # Aggressive learning rate scheduling
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, 
                                      patience=200, min_lr=1e-6)
        
        # Training loop
        print("\nStarting training...")
        print(f"Device: {self.device}")
        print(f"Number of parameter sets: {len(self.param_sets)}")
        print(f"Model architecture: 8 hidden layers with 128 neurons each")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Learning rate: {learning_rate:.2e}")
        print(f"Early stopping patience: {patience} epochs")
        print("-" * 80)
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Compute physics loss for all parameter sets
            loss_phys_total = 0.0
            for params in self.param_sets:
                loss_phys_total += self.physics_loss(
                    t_collocation, params['A'], params['B']
                )
            loss_phys = loss_phys_total / len(self.param_sets)
            
            # Compute initial condition loss for all parameter sets
            loss_ic_total = 0.0
            for params in self.param_sets:
                loss_ic_total += self.initial_condition_loss(params)
            loss_ic = loss_ic_total / len(self.param_sets)
            
            # Compute data loss
            loss_dat = self.data_loss(t_data, x_data, y_data, A_data, B_data)
            
            # Check for NaN with detailed debugging
            if torch.isnan(loss_phys) or torch.isnan(loss_ic) or torch.isnan(loss_dat):
                print(f"Warning: NaN detected at epoch {epoch+1}")
                print(f"  Physics Loss: {loss_phys.item()}")
                print(f"  IC Loss: {loss_ic.item()}")
                print(f"  Data Loss: {loss_dat.item()}")
                if epoch == 0:
                    # Check model outputs on first failure
                    with torch.no_grad():
                        A_test = torch.tensor([[self.param_sets[0]['A']]] * 5, dtype=torch.float32, device=self.device)
                        B_test = torch.tensor([[self.param_sets[0]['B']]] * 5, dtype=torch.float32, device=self.device)
                        test_out = self.model(t_collocation[:5], A_test, B_test)
                        print(f"  Sample model outputs: {test_out}")
                print("Stopping training.")
                break
            
            # Total loss
            loss = lambda_physics * loss_phys + lambda_ic * loss_ic + lambda_data * loss_dat
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(loss)
            
            # Store history
            self.loss_history['total'].append(loss.item())
            self.loss_history['physics'].append(loss_phys.item())
            self.loss_history['ic'].append(loss_ic.item())
            self.loss_history['data'].append(loss_dat.item())
            
            # Early stopping check
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.patience_counter = 0
                # Save best model state
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                self.patience_counter += 1
            
            # Check if we should stop
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"No improvement for {patience} epochs")
                print(f"Best loss: {self.best_loss:.6e}")
                # Restore best model
                if self.best_model_state is not None:
                    self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
                    print("Restored best model weights")
                break
            
            # Print progress
            if (epoch + 1) % print_every == 0:
                elapsed = time.time() - start_time
                current_lr = optimizer.param_groups[0]['lr']
                epoch_per_sec = (epoch + 1) / elapsed
                eta = (n_epochs - epoch - 1) / epoch_per_sec if epoch_per_sec > 0 else 0
                print(f"Epoch {epoch+1:6d}/{n_epochs} | "
                      f"Total: {loss.item():.4e} | "
                      f"Phys: {loss_phys.item():.4e} | "
                      f"IC: {loss_ic.item():.4e} | "
                      f"Data: {loss_dat.item():.4e} | "
                      f"LR: {current_lr:.2e} | "
                      f"Pat: {self.patience_counter:3d}/{patience} | "
                      f"Best: {self.best_loss:.4e} | "
                      f"Time: {elapsed:.1f}s | "
                      f"ETA: {eta/60:.1f}min")
            
            # Plot intermediate results
            if (epoch + 1) % plot_every == 0:
                self.plot_results(epoch+1)
        
        total_time = time.time() - start_time
        print("-" * 80)
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"Total epochs trained: {epoch+1}")
        print(f"Final best loss: {self.best_loss:.6e}")
        if len(self.loss_history['total']) > 0:
            print(f"Final training loss: {self.loss_history['total'][-1]:.6e}")
            print(f"Loss reduction: {(self.loss_history['total'][0] - self.best_loss) / self.loss_history['total'][0] * 100:.2f}%")
    
    def predict(self, t, A, B):
        """Extract predictions from PINN for given A, B parameters"""
        self.model.eval()
        with torch.no_grad():
            t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device).view(-1, 1)
            A_tensor = torch.full_like(t_tensor, A)
            B_tensor = torch.full_like(t_tensor, B)
            xy_pred = self.model(t_tensor, A_tensor, B_tensor)
            x_pred = xy_pred[:, 0].cpu().numpy()
            y_pred = xy_pred[:, 1].cpu().numpy()
        return x_pred, y_pred
    
    def plot_results(self, epoch=None, save_path=None):
        """Plot the results for all parameter sets"""
        n_params = len(self.param_sets)
        
        # Create figure with subplots for each parameter set
        fig = plt.figure(figsize=(16, 4 * n_params))
        
        for idx, params in enumerate(self.param_sets):
            # Generate prediction points
            t_plot = np.linspace(self.t_min, self.t_max, 1000)
            x_pred, y_pred = self.predict(t_plot, params['A'], params['B'])
            
            # Generate reference data
            t_ref, x_ref, y_ref = generate_rk4_data(
                params['A'], params['B'], params['x0'], params['y0'],
                self.t_min, self.t_max, 200
            )
            
            # Time series for x
            ax1 = plt.subplot(n_params, 3, idx * 3 + 1)
            ax1.plot(t_ref, x_ref, 'b-', linewidth=2, alpha=0.7, label='RK4 x(t)')
            ax1.plot(t_plot, x_pred, 'r--', linewidth=2, label='PINN x(t)')
            ax1.set_xlabel('Time', fontsize=10)
            ax1.set_ylabel('x(t)', fontsize=10)
            ax1.set_title(f'A={params["A"]}, B={params["B"]}: x vs Time', fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=8)
            
            # Time series for y
            ax2 = plt.subplot(n_params, 3, idx * 3 + 2)
            ax2.plot(t_ref, y_ref, 'b-', linewidth=2, alpha=0.7, label='RK4 y(t)')
            ax2.plot(t_plot, y_pred, 'r--', linewidth=2, label='PINN y(t)')
            ax2.set_xlabel('Time', fontsize=10)
            ax2.set_ylabel('y(t)', fontsize=10)
            ax2.set_title(f'A={params["A"]}, B={params["B"]}: y vs Time', fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8)
            
            # Phase portrait
            ax3 = plt.subplot(n_params, 3, idx * 3 + 3)
            ax3.plot(x_ref, y_ref, 'b-', linewidth=2, alpha=0.7, label='RK4 trajectory')
            ax3.plot(x_pred, y_pred, 'r--', linewidth=2, label='PINN trajectory')
            ax3.plot(x_pred[0], y_pred[0], 'ko', markersize=8, label='Initial condition')
            ax3.set_xlabel('x', fontsize=10)
            ax3.set_ylabel('y', fontsize=10)
            ax3.set_title(f'A={params["A"]}, B={params["B"]}: Phase Portrait', fontsize=11)
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=8)
        
        if epoch is not None:
            fig.suptitle(f'Multi-Parameter Brusselator PINN Results (Epoch {epoch})', 
                        fontsize=16, fontweight='bold')
        else:
            fig.suptitle('Multi-Parameter Brusselator PINN Results', 
                        fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Separate plot for loss history
        fig_loss, ax = plt.subplots(1, 1, figsize=(10, 6))
        if len(self.loss_history['total']) > 0:
            ax.semilogy(self.loss_history['total'], 'k-', linewidth=2, label='Total Loss')
            ax.semilogy(self.loss_history['physics'], 'b--', linewidth=1.5, label='Physics Loss')
            ax.semilogy(self.loss_history['ic'], 'r--', linewidth=1.5, label='IC Loss')
            if max(self.loss_history['data']) > 0:
                ax.semilogy(self.loss_history['data'], 'g--', linewidth=1.5, label='Data Loss')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Training Loss History', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        if save_path:
            loss_path = save_path.replace('.png', '_loss.png')
            plt.savefig(loss_path, dpi=300, bbox_inches='tight')
            print(f"Loss plot saved to {loss_path}")
        else:
            plt.show()
        plt.close()
    
    def save_model(self, path='brusselator_pinn_multi.pth'):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'param_sets': self.param_sets,
            't_min': self.t_min,
            't_max': self.t_max,
            'loss_history': self.loss_history,
            'best_loss': self.best_loss
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='brusselator_pinn_multi.pth'):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.param_sets = checkpoint['param_sets']
        self.loss_history = checkpoint['loss_history']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Model loaded from {path}")


def main():
    """Main training script for multi-parameter PINN"""
    
    # Define multiple parameter sets to train on
    # Using stable parameter combinations
    param_sets = [
        {'A': 1.0, 'B': 3.0, 'x0': 1.0, 'y0': 1.0},
        {'A': 1.0, 'B': 2.5, 'x0': 1.2, 'y0': 0.8},
        {'A': 1.2, 'B': 3.0, 'x0': 0.9, 'y0': 1.1},
        {'A': 0.9, 'B': 2.8, 'x0': 1.1, 'y0': 0.9},
    ]
    
    t_max = 20.0
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Training on {len(param_sets)} parameter sets:\n")
    for i, params in enumerate(param_sets):
        print(f"  Set {i+1}: A={params['A']}, B={params['B']}, "
              f"x0={params['x0']}, y0={params['y0']}")
    print()
    
    # Create PINN solver
    pinn = MultiParamBrusselatorPINN(
        param_sets=param_sets,
        t_min=0.0,
        t_max=t_max,
        device=device
    )
    
    # Train the model
    pinn.train(
        n_collocation=3000,  # More collocation points for better coverage
        n_epochs=50000,  # More epochs for convergence
        learning_rate=5e-4,  # Moderate learning rate for stability
        lambda_physics=1.0,
        lambda_ic=100.0,  # Higher IC weight for better initial condition matching
        lambda_data=10.0,  # Higher data weight to match training data
        n_data_points=100,  # More data points per parameter set
        print_every=100,  # Print more frequently
        plot_every=5000,
        patience=500  # More patience for larger model
    )
    
    # Final plot (save to main folder)
    pinn.plot_results(save_path='brusselator_pinn_multi_results.png')
    
    # Save the model (save to main folder)
    pinn.save_model('brusselator_pinn_multi_model.pth')
    
    # Compare with RK4 solution for each parameter set
    print("\n" + "=" * 80)
    print("COMPARISON WITH RK4 NUMERICAL SOLUTION")
    print("=" * 80)
    
    all_errors_x = []
    all_errors_y = []
    
    for idx, params in enumerate(param_sets):
        print(f"\nParameter Set {idx+1}: A={params['A']}, B={params['B']}, "
              f"x0={params['x0']}, y0={params['y0']}")
        print("-" * 80)
        
        # Generate reference data
        t_ref, x_ref, y_ref = generate_rk4_data(
            params['A'], params['B'], params['x0'], params['y0'],
            t_max=t_max, n_points=200
        )
        
        # Get predictions
        x_pred, y_pred = pinn.predict(t_ref, params['A'], params['B'])
        
        # Compute errors
        error_x = np.mean(np.abs(x_pred - x_ref))
        error_y = np.mean(np.abs(y_pred - y_ref))
        max_error_x = np.max(np.abs(x_pred - x_ref))
        max_error_y = np.max(np.abs(y_pred - y_ref))
        rmse_x = np.sqrt(np.mean((x_pred - x_ref) ** 2))
        rmse_y = np.sqrt(np.mean((y_pred - y_ref) ** 2))
        
        all_errors_x.append(error_x)
        all_errors_y.append(error_y)
        
        print(f"  Mean Absolute Error - x: {error_x:.6e}")
        print(f"  Mean Absolute Error - y: {error_y:.6e}")
        print(f"  Max Absolute Error - x:  {max_error_x:.6e}")
        print(f"  Max Absolute Error - y:  {max_error_y:.6e}")
        print(f"  RMSE - x:                {rmse_x:.6e}")
        print(f"  RMSE - y:                {rmse_y:.6e}")
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Average MAE across all parameter sets:")
    print(f"  x: {np.mean(all_errors_x):.6e} ± {np.std(all_errors_x):.6e}")
    print(f"  y: {np.mean(all_errors_y):.6e} ± {np.std(all_errors_y):.6e}")
    if len(pinn.loss_history['total']) > 0:
        print(f"\nFinal Training Loss: {pinn.loss_history['total'][-1]:.6e}")
    print(f"Best Training Loss: {pinn.best_loss:.6e}")
    
    # Plot comprehensive comparison
    fig, axes = plt.subplots(len(param_sets), 2, figsize=(14, 4 * len(param_sets)))
    if len(param_sets) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, params in enumerate(param_sets):
        # Generate reference data
        t_ref, x_ref, y_ref = generate_rk4_data(
            params['A'], params['B'], params['x0'], params['y0'],
            t_max=t_max, n_points=200
        )
        x_pred, y_pred = pinn.predict(t_ref, params['A'], params['B'])
        
        # Plot x comparison
        axes[idx, 0].plot(t_ref, x_ref, 'b-', linewidth=2, label='RK4 (x)')
        axes[idx, 0].plot(t_ref, x_pred, 'r--', linewidth=2, label='PINN (x)')
        axes[idx, 0].set_xlabel('Time', fontsize=12)
        axes[idx, 0].set_ylabel('x(t)', fontsize=12)
        axes[idx, 0].set_title(f'Set {idx+1} (A={params["A"]}, B={params["B"]}): x(t)', fontsize=12)
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Plot y comparison
        axes[idx, 1].plot(t_ref, y_ref, 'b-', linewidth=2, label='RK4 (y)')
        axes[idx, 1].plot(t_ref, y_pred, 'r--', linewidth=2, label='PINN (y)')
        axes[idx, 1].set_xlabel('Time', fontsize=12)
        axes[idx, 1].set_ylabel('y(t)', fontsize=12)
        axes[idx, 1].set_title(f'Set {idx+1} (A={params["A"]}, B={params["B"]}): y(t)', fontsize=12)
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('brusselator_multi_comparison.png', dpi=300, bbox_inches='tight')
    print("\n" + "=" * 80)
    print("Comparison plot saved to brusselator_multi_comparison.png")
    print("=" * 80)
    plt.close()


if __name__ == "__main__":
    main()

