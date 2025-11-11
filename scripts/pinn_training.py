import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time


def brusselator_rhs(x, y, A, B):
    """
    Brusselator model equations
    """
    dxdt = A + x * x * y - (B + 1.0) * x
    dydt = B * x - x * x * y
    return dxdt, dydt


# PINN Class
class PINN(nn.Module):   
    def __init__(self, hidden_layers=[64, 64, 64], activation=nn.Tanh()):
        super(PINN, self).__init__()
        
        # Input layer: time t
        layers = [nn.Linear(1, hidden_layers[0]), activation]
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.extend([
                nn.Linear(hidden_layers[i], hidden_layers[i+1]),
                activation
            ])
        
        # Output layer: [x, y]
        layers.append(nn.Linear(hidden_layers[-1], 2))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self.init_weights()
    
    def init_weights(self):
        # Initialize network weights using Xavier initialization
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, t):
        # Forward pass
        return self.network(t)


# BrusselatorPINN Class
class BrusselatorPINN:
    def __init__(self, A=1.0, B=3.0, x0=1.0, y0=1.0, 
                 t_min=0.0, t_max=20.0, device='cpu'):
        # Initialize the PINN solver
        self.A = A
        self.B = B
        self.x0 = x0
        self.y0 = y0
        self.t_min = t_min
        self.t_max = t_max
        self.device = torch.device(device)
        
        # Create model
        self.model = PINN(hidden_layers=[64, 64, 64, 64]).to(self.device)
        
        # Loss history
        self.loss_history = {
            'total': [],
            'physics': [],
            'ic': [],
            'data': []
        }
    
    def physics_loss(self, t_collocation):
        # Compute physics loss (PDE residuals)
        t_collocation.requires_grad_(True)
        
        # Forward pass
        xy = self.model(t_collocation)
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        
        # Compute derivatives using automatic differentiation
        dxy_dt = torch.autograd.grad(
            outputs=xy,
            inputs=t_collocation,
            grad_outputs=torch.ones_like(xy),
            create_graph=True,
            retain_graph=True
        )[0]
        
        dx_dt = dxy_dt[:, 0:1]
        dy_dt = dxy_dt[:, 1:2]
        
        # Brusselator equations
        f_x = dx_dt - (self.A + x * x * y - (self.B + 1.0) * x)
        f_y = dy_dt - (self.B * x - x * x * y)
        
        # Mean squared error
        loss_physics = torch.mean(f_x ** 2 + f_y ** 2)
        
        return loss_physics
    
    def initial_condition_loss(self):
        # Compute initial condition loss
        t0 = torch.tensor([[self.t_min]], dtype=torch.float32, device=self.device)
        xy0_pred = self.model(t0)
        
        x0_pred = xy0_pred[:, 0]
        y0_pred = xy0_pred[:, 1]
        
        loss_ic = (x0_pred - self.x0) ** 2 + (y0_pred - self.y0) ** 2
        
        return loss_ic
    
    def data_loss(self, t_data, x_data, y_data):
        # Compute data loss (if training data is available)
        if t_data is None:
            return torch.tensor(0.0, device=self.device)
        
        xy_pred = self.model(t_data)
        x_pred = xy_pred[:, 0:1]
        y_pred = xy_pred[:, 1:2]
        
        loss_data = torch.mean((x_pred - x_data) ** 2 + (y_pred - y_data) ** 2)
        
        return loss_data
    
    # Training Code
    def train(self, n_collocation=1000, n_epochs=10000, 
              learning_rate=1e-3, lambda_physics=1.0, lambda_ic=100.0,
              lambda_data=1.0, t_data=None, x_data=None, y_data=None,
              print_every=500, plot_every=2000):
        # Generate collocation points
        t_collocation = torch.linspace(
            self.t_min, self.t_max, n_collocation, 
            device=self.device
        ).view(-1, 1).requires_grad_(True)
        
        # Convert data to tensors if provided
        if t_data is not None:
            t_data = torch.tensor(t_data, dtype=torch.float32, device=self.device).view(-1, 1)
            x_data = torch.tensor(x_data, dtype=torch.float32, device=self.device).view(-1, 1)
            y_data = torch.tensor(y_data, dtype=torch.float32, device=self.device).view(-1, 1)
        
        # Optimizer
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                      patience=500, verbose=True)
        
        # Training loop
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Parameters: A={self.A}, B={self.B}")
        print(f"Initial conditions: x0={self.x0}, y0={self.y0}")
        print("-" * 80)
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Compute losses
            loss_phys = self.physics_loss(t_collocation)
            loss_ic = self.initial_condition_loss()
            loss_dat = self.data_loss(t_data, x_data, y_data)
            
            # Total loss
            loss = lambda_physics * loss_phys + lambda_ic * loss_ic + lambda_data * loss_dat
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            # Store history
            self.loss_history['total'].append(loss.item())
            self.loss_history['physics'].append(loss_phys.item())
            self.loss_history['ic'].append(loss_ic.item())
            self.loss_history['data'].append(loss_dat.item())
            
            # Print progress
            if (epoch + 1) % print_every == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{n_epochs} | "
                      f"Loss: {loss.item():.6e} | "
                      f"Physics: {loss_phys.item():.6e} | "
                      f"IC: {loss_ic.item():.6e} | "
                      f"Data: {loss_dat.item():.6e} | "
                      f"Time: {elapsed:.2f}s")
            
            # Plot intermediate results
            if (epoch + 1) % plot_every == 0:
                self.plot_results(epoch+1)
        
        print("-" * 80)
        print(f"Training completed in {time.time() - start_time:.2f}s")
    
    # Extract Predictions from PINN
    def predict(self, t):
        self.model.eval()
        with torch.no_grad():
            t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device).view(-1, 1)
            xy_pred = self.model(t_tensor)
            x_pred = xy_pred[:, 0].cpu().numpy()
            y_pred = xy_pred[:, 1].cpu().numpy()
        return x_pred, y_pred
    
    def plot_results(self, epoch=None, save_path=None):
        # Plot the results
        # Generate prediction points
        t_plot = np.linspace(self.t_min, self.t_max, 1000)
        x_pred, y_pred = self.predict(t_plot)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Time series for x
        axes[0, 0].plot(t_plot, x_pred, 'b-', linewidth=2, label='PINN x(t)')
        axes[0, 0].set_xlabel('Time', fontsize=12)
        axes[0, 0].set_ylabel('x(t)', fontsize=12)
        axes[0, 0].set_title('Species x vs Time', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Time series for y
        axes[0, 1].plot(t_plot, y_pred, 'r-', linewidth=2, label='PINN y(t)')
        axes[0, 1].set_xlabel('Time', fontsize=12)
        axes[0, 1].set_ylabel('y(t)', fontsize=12)
        axes[0, 1].set_title('Species y vs Time', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Phase portrait
        axes[1, 0].plot(x_pred, y_pred, 'g-', linewidth=2, label='PINN trajectory')
        axes[1, 0].plot(x_pred[0], y_pred[0], 'ko', markersize=10, label='Initial condition')
        axes[1, 0].set_xlabel('x', fontsize=12)
        axes[1, 0].set_ylabel('y', fontsize=12)
        axes[1, 0].set_title('Phase Portrait', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Loss history
        if len(self.loss_history['total']) > 0:
            axes[1, 1].semilogy(self.loss_history['total'], 'k-', linewidth=2, label='Total Loss')
            axes[1, 1].semilogy(self.loss_history['physics'], 'b--', linewidth=1.5, label='Physics Loss')
            axes[1, 1].semilogy(self.loss_history['ic'], 'r--', linewidth=1.5, label='IC Loss')
            if max(self.loss_history['data']) > 0:
                axes[1, 1].semilogy(self.loss_history['data'], 'g--', linewidth=1.5, label='Data Loss')
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('Loss', fontsize=12)
            axes[1, 1].set_title('Training Loss History', fontsize=14)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        if epoch is not None:
            fig.suptitle(f'Brusselator PINN Results (Epoch {epoch})', fontsize=16, fontweight='bold')
        else:
            fig.suptitle('Brusselator PINN Results', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_model(self, path='brusselator_pinn.pth'):
        # Save the trained model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'parameters': {
                'A': self.A,
                'B': self.B,
                'x0': self.x0,
                'y0': self.y0,
                't_min': self.t_min,
                't_max': self.t_max
            },
            'loss_history': self.loss_history
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='brusselator_pinn.pth'):
        # Load a trained model
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.loss_history = checkpoint['loss_history']
        print(f"Model loaded from {path}")


def generate_reference_data(A, B, x0, y0, t_max=20.0, n_points=100):
    # Generate reference data using numerical integration (optional, for comparison)
    from scipy.integrate import odeint
    
    def system(state, t):
        x, y = state
        dxdt, dydt = brusselator_rhs(x, y, A, B)
        return [dxdt, dydt]
    
    t = np.linspace(0, t_max, n_points)
    solution = odeint(system, [x0, y0], t)
    
    return t, solution[:, 0], solution[:, 1]


def main():
    # Main training script
    # Parameters
    A = 1.0
    B = 3.0
    x0 = 1.0
    y0 = 1.0
    t_max = 20.0
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create PINN solver
    pinn = BrusselatorPINN(
        A=A, B=B, x0=x0, y0=y0,
        t_min=0.0, t_max=t_max,
        device=device
    )
    
    # Optional: Generate some reference data for training
    # Uncomment if you want to include data loss
    # t_data, x_data, y_data = generate_reference_data(A, B, x0, y0, t_max=t_max, n_points=50)
    
    # Train the model
    pinn.train(
        n_collocation=2000,
        n_epochs=15000,
        learning_rate=1e-3,
        lambda_physics=1.0,
        lambda_ic=100.0,
        lambda_data=0.0,  # Set to 1.0 if using data
        t_data=None,  # Pass t_data if using
        x_data=None,  # Pass x_data if using
        y_data=None,  # Pass y_data if using
        print_every=500,
        plot_every=5000
    )
    
    # Final plot
    pinn.plot_results(save_path='brusselator_pinn_results.png')
    
    # Save the model
    pinn.save_model('brusselator_pinn_model.pth')
    
    # Optional: Compare with numerical solution
    print("\nComparing with numerical solution...")
    t_ref, x_ref, y_ref = generate_reference_data(A, B, x0, y0, t_max=t_max, n_points=200)
    x_pred, y_pred = pinn.predict(t_ref)
    
    error_x = np.mean(np.abs(x_pred - x_ref))
    error_y = np.mean(np.abs(y_pred - y_ref))
    
    print(f"Mean Absolute Error - x: {error_x:.6e}")
    print(f"Mean Absolute Error - y: {error_y:.6e}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(t_ref, x_ref, 'b-', linewidth=2, label='Numerical (x)')
    axes[0].plot(t_ref, x_pred, 'r--', linewidth=2, label='PINN (x)')
    axes[0].set_xlabel('Time', fontsize=12)
    axes[0].set_ylabel('x(t)', fontsize=12)
    axes[0].set_title('Comparison: x(t)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t_ref, y_ref, 'b-', linewidth=2, label='Numerical (y)')
    axes[1].plot(t_ref, y_pred, 'r--', linewidth=2, label='PINN (y)')
    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].set_ylabel('y(t)', fontsize=12)
    axes[1].set_title('Comparison: y(t)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('brusselator_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved to brusselator_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()

