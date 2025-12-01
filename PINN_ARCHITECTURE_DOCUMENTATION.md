# PINN Training Script Documentation
## `pinn_training_multi_hpc.py`

---

## Overview

This script trains a Physics-Informed Neural Network (PINN) to solve the Brusselator system of ODEs:

```
dx/dt = A + x²y - (B + 1)x
dy/dt = Bx - x²y
```

**Key Innovation:** Instead of solving for one parameter set at a time, the network learns a continuous function across the entire parameter space (A, B, x₀, y₀).

**Training Data:**
- 2,000 training parameter combinations
- 400 validation parameter combinations  
- Parameter ranges: A ∈ [0.5, 2.5], B ∈ [1.0, 6.0], x₀ ∈ [0.0, 3.0], y₀ ∈ [0.0, 3.0]
- Time domain: t ∈ [0, 20]

---

## Neural Network Architecture

**Class:** `PINN` (lines 71-106)

```
Input: [t, A, B, x0, y0]  (5 values)
    ↓
6 Hidden Layers × 256 neurons each (GELU activation)
    ↓
Output: [x, y]  (2 values)
```

**Key specifications:**
- Total parameters: ~400,000
- Input features: time `t`, Brusselator parameters `A` and `B`, initial conditions `x0` and `y0`
- Activation function: `GELU` (smooth, often better than tanh for PINNs)
- Weight initialization: Xavier normal (prevents gradient issues)

**Forward pass:** Concatenates inputs `[t, A, B, x0, y0]`, passes through hidden layers with GELU activation, outputs predictions `[x(t), y(t)]`.

**Why x0, y0 are inputs:** Different initial conditions produce different trajectories even for the same A, B values. By including x0, y0 as network inputs, the PINN can learn the entire family of solutions across the parameter space.

---

## Loss Functions

The network is trained using three loss components (weighted sum):

### 1. Physics Loss (λ = 1.0)

**Purpose:** Enforce that predictions satisfy the Brusselator ODEs.

**Implementation:** (lines 154-194)
```python
def physics_loss(self, t_collocation, A, B, x0, y0):
    # Get network predictions (x0, y0 are passed to identify the trajectory)
    x_pred, y_pred = self.model(t, A_tensor, B_tensor, x0_tensor, y0_tensor)
    
    # Compute time derivatives using automatic differentiation
    dx_dt = torch.autograd.grad(x_pred, t, create_graph=True)
    dy_dt = torch.autograd.grad(y_pred, t, create_graph=True)
    
    # Compute residuals (how much equations are violated)
    residual_x = dx_dt - (A + x_pred² * y_pred - (B+1)*x_pred)
    residual_y = dy_dt - (B*x_pred - x_pred² * y_pred)
    
    # Return mean squared residual
    return mean(residual_x² + residual_y²)
```

**Evaluated at:** 3,000 random time points (collocation points) per parameter set.

**Why it matters:** This is what makes it "physics-informed" – the network learns to satisfy the differential equations, not just fit data.

---

### 2. Initial Condition Loss (λ = 100.0)

**Purpose:** Enforce that predictions match initial conditions at t=0.

**Implementation:** (lines 196-212)
```python
def initial_condition_loss(self, param_set):
    # Predict at t=0
    x_pred, y_pred = self.model(t=0, A, B)
    
    # Compare to true initial conditions
    loss = (x_pred - x₀)² + (y_pred - y₀)²
    return loss
```

**Why weighted 100x:** Initial errors propagate through time, so getting the starting point exactly right is critical.

---

### 3. Data Loss (λ = 50.0)

**Purpose:** Match reference solutions from traditional ODE solver (RK4).

**Implementation:** (lines 214-277)
```python
def data_loss(self, t_data, x_data, y_data, A_data, B_data, x0_data, y0_data):
    # Get network predictions at data points (x0, y0 identify the trajectory)
    x_pred, y_pred = self.model(t_data, A_data, B_data, x0_data, y0_data)
    
    # Compute MSE against reference solutions
    loss = mean((x_pred - x_reference)² + (y_pred - y_reference)²)
    return loss
```

**Evaluated at:** 50 time points per parameter set (from RK4 solver).

**Why it matters:** Provides strong supervision to guide the network toward physically realistic solutions.

---

### Total Loss

```python
Total Loss = 1.0 × Physics Loss + 100.0 × IC Loss + 50.0 × Data Loss
```

**Computed in:** `compute_loss_for_param_sets()` (lines 279-331)

All three losses are computed for sampled parameter sets, then combined with their weights.

---

## Training Process

**Main training method:** `train()` (lines 369-617)

### Training Loop (Simplified)

```python
for epoch in range(max_epochs):
    # 1. Sample random parameter sets
    param_sample = random.choice(train_params, 100)  # 100 out of 5,000
    
    # 2. Compute total loss
    loss = compute_loss_for_param_sets(param_sample)
    
    # 3. Backpropagation
    loss.backward()
    clip_gradients(max_norm=1.0)  # Prevent exploding gradients
    optimizer.step()
    
    # 4. Validation
    val_loss = evaluate_on_validation(val_params)
    
    # 5. Early stopping check
    if val_loss < best_val_loss:
        save_model()
    elif no_improvement_for_500_epochs:
        stop_training()
```

### Key Training Details

**Optimizer:** Adam
- Learning rate: 0.001 (initial)
- Weight decay: 0.000001 (L2 regularization)
- Learning rate scheduling: Reduce by 50% when validation plateaus (patience=1000 epochs)
- Minimum learning rate: 1e-5

**Gradient Clipping:**
- Max norm: 1.0
- Prevents instability from physics loss (involves second derivatives)

**Early Stopping:**
- Monitors validation loss
- Stops if no improvement for 3000 epochs
- Restores best model automatically

**Mini-batch Training:**
- Samples 200 parameter sets per epoch (out of 2,000 total)
- Different random sample each epoch
- Reduces GPU memory usage dramatically

## Key Classes and Functions

### `MultiParamBrusselatorPINN` (lines 109-899)

Main class that handles training and evaluation.

**Initialization:**
- Creates neural network model
- Sets up loss tracking
- Initializes early stopping variables

**Methods:**
- `physics_loss(t, A, B, x0, y0)`: Computes PDE residual loss
- `initial_condition_loss(param_set)`: Computes IC loss
- `data_loss(t, x, y, A, B, x0, y0)`: Computes data matching loss  
- `compute_loss_for_param_sets()`: Combines all losses
- `prepare_data()`: Generates RK4 reference solutions
- `train()`: Main training loop
- `predict(t, A, B, x0, y0)`: Inference on new parameters
- `evaluate()`: Compute MAE/RMSE metrics
- `plot_final_results()`: Generate visualization plots
- `save_model()` / `save_best_model()`: Save checkpoints

### Helper Functions

- `generate_rk4_data()` (lines 22-34): Generate reference solutions using scipy's odeint
- `generate_parameter_sets()` (lines 38-67): Sample random parameter combinations
- `brusselator_rhs()` (lines 15-18): Define the ODE system

---

## Configuration

**Configurable via `config.py`:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| N_TRAIN_SETS | 2000 | Training parameter combinations |
| N_VAL_SETS | 400 | Validation parameter combinations |
| HIDDEN_LAYERS | [256]×6 | 6 layers × 256 neurons |
| ACTIVATION | gelu | Activation function |
| N_COLLOCATION | 1000 | Collocation points for physics loss |
| N_DATA_POINTS | 50 | Data points per param set |
| N_EPOCHS | 50000 | Maximum training epochs |
| LEARNING_RATE | 1e-3 | Initial learning rate |
| LAMBDA_PHYSICS | 1.0 | Physics loss weight |
| LAMBDA_IC | 100.0 | IC loss weight |
| LAMBDA_DATA | 50.0 | Data loss weight |
| BATCH_SIZE | 50 | Parameter sets per batch |
| DATA_BATCH_SIZE | 5000 | Data points per batch |
| N_PARAMS_PER_EPOCH | 200 | Parameter sets sampled per epoch |
| PATIENCE | 3000 | Early stopping patience |
| LR_MIN | 1e-5 | Minimum learning rate |

---

## Outputs

**Saved files** (in `outputs/` directory):

1. **Models:**
   - `brusselator_pinn_best_model.pth` - Model with lowest validation loss
   - `brusselator_pinn_last_model.pth` - Final model state

2. **Metrics:**
   - `training_summary.json` - Training statistics (losses, epochs, etc.)

3. **Visualizations** (in `outputs/plots/`):
   - `training_examples.png` - Predictions vs RK4 for training sets
   - `validation_examples.png` - Predictions vs RK4 for validation sets  
   - `loss_history.png` - Loss curves (train/val for all components)

## Important Implementation Notes

1. **Automatic differentiation** (lines 170-185): Uses PyTorch's `autograd.grad()` to compute time derivatives needed for physics loss.

2. **Gradient preservation** (lines 307, 322): Uses `torch.stack().mean()` instead of `+=` to maintain proper gradient graphs during batching.

3. **Validation-based early stopping** (lines 566-584): Tracks validation loss (not training loss) to prevent overfitting.

4. **CPU-GPU memory management** (lines 359-367, 250-262): Keeps large datasets on CPU, transfers only small batches to GPU during training.

5. **Three `torch.set_grad_enabled(True)` contexts** (lines 296, 311, 325): Ensures gradients are computed correctly for all loss components during batching.


