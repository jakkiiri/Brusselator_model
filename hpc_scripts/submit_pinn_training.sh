#!/bin/bash

#SBATCH --job-name=brusselator_pinn
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pinn_training_%j.out
#SBATCH --error=logs/pinn_training_%j.err

# Note: Both stdout and stderr are written to the same .err file for easy monitoring
# To monitor training progress in real-time:
#   tail -f logs/pinn_training_JOBID.err
# Where JOBID is your job number from 'squeue -u $USER'

# Print job information
echo "=========================================="
echo "Brusselator PINN Training Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Started at: $(date)"
echo "Working directory: $(pwd)"
echo "Log file: logs/pinn_training_${SLURM_JOB_ID}.err"
echo "=========================================="

# Create necessary directories
mkdir -p logs
mkdir -p outputs
mkdir -p outputs/plots

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# CRITICAL: Disable Python output buffering for real-time logging
export PYTHONUNBUFFERED=1

# Disable interactive matplotlib backend
export MPLBACKEND=Agg

# Load required modules (uncomment and adjust for your HPC system)
# module load python/3.9
# module load cuda/11.8
# module load scipy

# Activate Python virtual environment
# Adjust this path to your actual virtual environment location
# Make sure you've installed requirements: pip install -r hpc_scripts/requirements.txt
source $HOME/projects/aip-yuweilai/jackyli/Brusselator_PINN/brusselator_venv/bin/activate || echo "WARNING: Could not activate virtual environment"

# Verify environment
echo ""
echo "Environment Information:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo "  PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '0')"
echo ""

# Navigate to script directory
cd $SLURM_SUBMIT_DIR

# Verify script exists
if [ ! -f "pinn_training_multi_hpc.py" ]; then
    echo "ERROR: Training script not found!"
    exit 1
fi

echo "=========================================="
echo "Starting PINN Training"
echo "=========================================="
echo ""
echo "To monitor training progress from another terminal:"
echo "  tail -f logs/pinn_training_${SLURM_JOB_ID}.err"
echo ""

# Run the training script (with -u for unbuffered output)
python -u pinn_training_multi_hpc.py

# Check if training completed successfully
EXIT_CODE=$?

echo ""
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Display output information
    echo ""
    echo "Output Information:"
    
    if [ -f "outputs/brusselator_pinn_model.pth" ]; then
        echo "  Model saved: $(ls -lh outputs/brusselator_pinn_model.pth | awk '{print $5}')"
    fi
    
    if [ -f "outputs/training_summary.json" ]; then
        echo "  Training summary:"
        cat outputs/training_summary.json
    fi
    
    if [ -d "outputs/plots" ]; then
        echo "  Generated plots: $(ls -1 outputs/plots/*.png 2>/dev/null | wc -l) files"
        ls -lh outputs/plots/
    fi
    
    echo ""
    echo "All outputs saved to: outputs/"
    
else
    echo "Training failed with exit code $EXIT_CODE"
    echo "Check log file for details: logs/pinn_training_${SLURM_JOB_ID}.err"
    exit $EXIT_CODE
fi

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="

