"""
Configuration file for Brusselator PINN training
Modify these parameters without changing the main training script
"""

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Number of parameter sets
N_TRAIN_SETS = 2000   # Training parameter combinations (start smaller to verify learning)
N_VAL_SETS = 400      # Validation parameter combinations (~20% of training)

# Parameter ranges for random sampling
A_RANGE = (0.5, 2.5)    # Brusselator parameter A
B_RANGE = (1.0, 6.0)    # Brusselator parameter B
X0_RANGE = (0.0, 3.0)   # Initial condition for x
Y0_RANGE = (0.0, 3.0)   # Initial condition for y

# Time domain
T_MIN = 0.0
T_MAX = 20.0

# Random seeds for reproducibility
TRAIN_SEED = 42
VAL_SEED = 123


# ============================================================================
# NETWORK ARCHITECTURE
# ============================================================================

# Hidden layer sizes (list of integers)
HIDDEN_LAYERS = [256, 256, 256, 256, 256, 256]  # 6 layers x 256 neurons (was 128 - too small!)

# Activation function: 'tanh', 'relu', 'silu', 'gelu'
ACTIVATION = 'gelu'  # GELU often works better than tanh for PINNs


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Physics constraints
N_COLLOCATION = 1000    # Number of collocation points for physics loss

# Training data
N_DATA_POINTS = 50      # Number of time points per parameter set for data loss

# Batching (IMPORTANT for memory management)
BATCH_SIZE = 50         # Number of parameter sets to process at once (prevents GPU OOM)
                        # This controls how many param sets are processed together
                        # With N_PARAMS_PER_EPOCH=200, this gives 4 batches per epoch
                        # DO NOT increase above 50 - causes memory issues!

DATA_BATCH_SIZE = 5000  # Number of data points to process at once in data loss
                        # 5000 sets × 50 points = 250,000 total, so batches of 5000 = 50 batches
                        # Decrease to 2000-3000 if still getting OOM errors

N_PARAMS_PER_EPOCH = 200   # Number of parameter sets to use per training epoch
                           # CRITICAL: This limits computation graph accumulation
                           # Even with batching, all losses are kept in memory before backward()
                           # 200 sets = good balance between memory and coverage
                           # Each set seen ~10 times over 50k epochs
                           # DO NOT increase beyond 500 or you'll get GPU OOM!

# Optimization
N_EPOCHS = 50000        # Maximum number of training epochs
LEARNING_RATE = 1e-3    # Initial learning rate (was 5e-4 - increased for faster learning)
WEIGHT_DECAY = 1e-6     # L2 regularization (was 1e-5 - reduced to allow more flexibility)

# Loss weights
LAMBDA_PHYSICS = 1.0    # Weight for physics loss (PDE residuals)
LAMBDA_IC = 100.0       # Weight for initial condition loss 
LAMBDA_DATA = 50.0      # Weight for data matching loss 

# Learning rate scheduler
LR_SCHEDULER_FACTOR = 0.5      # Multiply LR by this when loss plateaus (was 0.7)
LR_SCHEDULER_PATIENCE = 1000   # Epochs to wait before reducing LR (was 200 - too aggressive!)
LR_MIN = 1e-5                  # Minimum learning rate (was 1e-6 - too low!)

# Gradient clipping
GRAD_CLIP_NORM = 1.0    # Maximum gradient norm (prevents exploding gradients)


# ============================================================================
# EARLY STOPPING
# ============================================================================

PATIENCE = 3000         # Epochs to wait for validation improvement before stopping


# ============================================================================
# LOGGING AND OUTPUT
# ============================================================================

# Console output
PRINT_EVERY = 10        # Print detailed training progress every N epochs (was 100)

# Output directories
OUTPUT_DIR = 'outputs'
PLOTS_DIR = 'outputs/plots'
LOGS_DIR = 'logs'

# Model checkpoint
MODEL_FILENAME = 'brusselator_pinn_model.pth'
SUMMARY_FILENAME = 'training_summary.json'


# ============================================================================
# EVALUATION
# ============================================================================

# Number of parameter sets to evaluate in detail (subset of train/val for speed)
N_EVAL_TRAIN = 100
N_EVAL_VAL = 100

# Number of examples to plot (max 6 for readability)
N_PLOT_TRAIN = 3
N_PLOT_VAL = 3

# Number of points for plotting trajectories
N_PLOT_POINTS = 1000

# Number of points for evaluation
N_EVAL_POINTS = 200


# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================

# Device selection
# Options: 'auto' (automatically selects CUDA if available), 'cuda', 'cpu'
DEVICE = 'auto'

# Number of data loading workers (for future extensions)
NUM_WORKERS = 4


# ============================================================================
# ADVANCED OPTIONS
# ============================================================================

# Use mixed precision training (faster on modern GPUs, requires torch >= 1.6)
USE_AMP = False  # Set to True for A100/H100 GPUs

# Save intermediate checkpoints
SAVE_INTERMEDIATE = False
INTERMEDIATE_SAVE_EVERY = 5000  # Save every N epochs

# Verbose output
VERBOSE = True

# DPI for saved plots
PLOT_DPI = 300


# ============================================================================
# NOTES
# ============================================================================

"""
Recommended configurations for different use cases:

1. QUICK TEST (Fast training to verify setup):
   N_TRAIN_SETS = 100
   N_VAL_SETS = 20
   N_EPOCHS = 5000
   PATIENCE = 100

2. SMALL SCALE (Good for debugging):
   N_TRAIN_SETS = 500
   N_VAL_SETS = 100
   N_EPOCHS = 20000
   PATIENCE = 300

3. STANDARD (Default, balanced performance):
   N_TRAIN_SETS = 5000
   N_VAL_SETS = 1000
   N_EPOCHS = 50000
   PATIENCE = 500

4. LARGE SCALE (Maximum generalization):
   N_TRAIN_SETS = 10000
   N_VAL_SETS = 2000
   N_EPOCHS = 100000
   PATIENCE = 1000
   BATCH_SIZE = 100
   DATA_BATCH_SIZE = 10000
   N_PARAMS_PER_EPOCH = 200
   N_COLLOCATION = 5000
   N_DATA_POINTS = 100

5. LOW MEMORY (For limited GPU memory):
   N_COLLOCATION = 1000
   N_DATA_POINTS = 30
   BATCH_SIZE = 20
   DATA_BATCH_SIZE = 2000
   N_PARAMS_PER_EPOCH = 50
   HIDDEN_LAYERS = [64, 64, 64, 64]

6. HIGH ACCURACY (More capacity, longer training):
   HIDDEN_LAYERS = [256, 256, 256, 256, 256, 256]
   LAMBDA_IC = 200.0
   LAMBDA_DATA = 20.0
   LEARNING_RATE = 1e-4
"""

