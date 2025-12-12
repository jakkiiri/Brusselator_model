"""
PINN Model for Brusselator - Inference Only
Loads the trained model and provides prediction interface
"""

import torch
import torch.nn as nn
import numpy as np
import time


class PINN(nn.Module):
    """Physics-Informed Neural Network for Brusselator system"""
    
    def __init__(self, hidden_layers=[128, 128, 128, 128, 128, 128, 128, 128], activation=nn.Tanh()):
        super(PINN, self).__init__()
        
        # Input layer: time t, A, B, x0, y0 (5 inputs total)
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
    
    def forward(self, t, A, B, x0, y0):
        """Forward pass with time, parameters, and initial conditions as input"""
        inputs = torch.cat([t, A, B, x0, y0], dim=1)
        out = self.network(inputs)
        return out


class BrusselatorPINNSolver:
    """Wrapper class for loading and using the trained PINN model"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize the PINN solver by loading a trained model
        
        Args:
            model_path: Path to the .pth file containing the trained model
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.model = PINN(hidden_layers=[128, 128, 128, 128, 128, 128, 128, 128]).to(self.device)
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Store metadata from checkpoint
        self.t_min = checkpoint.get('t_min', 0.0)
        self.t_max = checkpoint.get('t_max', 20.0)
        self.best_val_loss = checkpoint.get('best_val_loss', None)
        self.best_epoch = checkpoint.get('best_epoch', None)
    
    def predict(self, t: np.ndarray, A: float, B: float, x0: float, y0: float):
        """
        Generate predictions for given parameters
        
        Args:
            t: Time array (numpy array)
            A: Parameter A
            B: Parameter B
            x0: Initial condition for x
            y0: Initial condition for y
            
        Returns:
            x_pred: Predicted x values (numpy array)
            y_pred: Predicted y values (numpy array)
        """
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
    
    def solve(self, A: float, B: float, x0: float, y0: float, T: float = 20.0, n_points: int = 1000):
        """
        Solve the Brusselator system using the PINN
        
        Args:
            A: Parameter A
            B: Parameter B
            x0: Initial condition for x
            y0: Initial condition for y
            T: Final time (default 20.0)
            n_points: Number of time points to generate
            
        Returns:
            t: Time array
            x: Solution x(t)
            y: Solution y(t)
            elapsed_time: Time taken for inference
        """
        t = np.linspace(0.0, T, n_points)
        
        start_time = time.perf_counter()
        x_pred, y_pred = self.predict(t, A, B, x0, y0)
        elapsed_time = time.perf_counter() - start_time
        
        return t, x_pred, y_pred, elapsed_time

