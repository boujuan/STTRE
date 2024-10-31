#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=test_gpu_%j.out
#SBATCH --error=test_gpu_%j.err
#SBATCH --partition=all_gpu.p
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

# Print basic info
hostname
date
pwd

# Debug system name
echo "LMOD_SYSTEM_NAME before: $LMOD_SYSTEM_NAME"
SYSTEMNAME=${LMOD_SYSTEM_NAME:-unknown}
echo "SYSTEMNAME: $SYSTEMNAME"

# Source global definitions first
if [ -f /etc/bashrc ]; then
    echo "Sourcing /etc/bashrc"
    . /etc/bashrc
fi

# Initialize basic PATH
PATH="$HOME/.local/bin:$HOME/bin:$PATH"
export PATH

# Check for and source system specific files
for file in ~/.${SYSTEMNAME}_systemrc ~/.bash_aliases ~/.bash_aliases.${SYSTEMNAME}; do
    if [ -f "$file" ]; then
        echo "Sourcing $file"
        . "$file"
    else
        echo "$file not found"
    fi
done

# Initialize conda directly
MAMBA_ROOT="/cm/shared/uniol/sw/SYSTEM/Mamba/24.3.0-0"
if [ -f "${MAMBA_ROOT}/etc/profile.d/conda.sh" ]; then
    echo "Sourcing conda.sh"
    . "${MAMBA_ROOT}/etc/profile.d/conda.sh"
fi

if [ -f "${MAMBA_ROOT}/etc/profile.d/mamba.sh" ]; then
    echo "Sourcing mamba.sh"
    . "${MAMBA_ROOT}/etc/profile.d/mamba.sh"
fi

# Load CUDA
echo "Loading CUDA module"
module purge
module load CUDA/12.4.0

# Activate conda environment
echo "Activating conda environment"
conda activate wind_forecasting_cuda

# Print environment info
echo "=== Environment Info ==="
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "LMOD_SYSTEM_NAME: $LMOD_SYSTEM_NAME"

# Basic tests
echo "=== Python and CUDA Tests ==="
python --version
nvidia-smi

# Simple PyTorch test
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'CUDA device name: {torch.cuda.get_device_name(0)}')
"

echo "Job complete"
