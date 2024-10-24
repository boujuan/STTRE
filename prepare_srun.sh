#!/bin/bash

# Load required modules with error checking
echo "Loading modules..."
module load hpc-env/13.1 || { echo "Failed to load hpc-env module"; exit 1; }
module load Mamba/24.3.0-0 || { echo "Failed to load Mamba module"; exit 1; }
module load CUDA/12.4.0 || { echo "Failed to load CUDA module"; exit 1; }
module load JupyterNotebook/7.0.4-GCCcore-13.1.0 || { echo "Failed to load JupyterNotebook module"; exit 1; }
module load jupyter-server/2.7.2-GCCcore-13.1.0 || { echo "Failed to load jupyter-server module"; exit 1; }
module load PyTorch/2.1.2-foss-2023a-CUDA-12.4.0 || { echo "Failed to load PyTorch module"; exit 1; }
# module list

# Activate conda environment
echo "Activating conda environment..."
mamba activate wind_forecasting_cuda || { echo "Failed to activate conda environment"; }

# Verify environment
which python
python --version
echo "CONDA_PREFIX: $CONDA_PREFIX"

# Check NVIDIA GPU
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found"
fi

# Print PyTorch info
echo "Checking PyTorch setup..."
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('Number of GPUs:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
" || { echo "Failed to check PyTorch setup"; exit 1; }

echo $SLURM_JOB_ID
