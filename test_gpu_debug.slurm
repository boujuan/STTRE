#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=test_gpu_%j.out
#SBATCH --error=test_gpu_%j.err
#SBATCH --partition=all_gpu.p
#SBATCH --nodelist=mpcg004
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:H100:1

# Enable verbose mode and error tracing
set -x
set -e

# Print basic info
echo "=== System Information ==="
hostname
date
pwd

# Load required modules
echo "=== Loading Modules ==="
module --force purge
module load GCCcore/13.1.0
module load CUDA/12.4.0

# Explicitly set CUDA paths
export PATH="/cm/shared/uniol/sw/SYSTEM/CUDA/12.4.0/bin:$PATH"
export LD_LIBRARY_PATH="/cm/shared/uniol/sw/SYSTEM/CUDA/12.4.0/lib64:/cm/shared/uniol/sw/SYSTEM/CUDA/12.4.0/lib:$LD_LIBRARY_PATH"

# Print environment
echo "=== Environment ==="
env | grep -i cuda
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Check CUDA installation
echo "=== CUDA Installation ==="
ls -l $CUDA_HOME/bin/
ls -l $CUDA_HOME/lib64/ || ls -l $CUDA_HOME/lib/

# Try multiple ways to find nvidia-smi
echo "=== Searching for nvidia-smi ==="
which nvidia-smi || echo "which nvidia-smi failed"
find $CUDA_HOME -name nvidia-smi 2>/dev/null || echo "nvidia-smi not in CUDA_HOME"
find /usr/bin /usr/local/bin -name nvidia-smi 2>/dev/null || echo "nvidia-smi not in standard paths"

# Try to run CUDA commands
echo "=== Testing CUDA ==="
if [ -x "$(command -v nvidia-smi)" ]; then
    nvidia-smi
else
    echo "nvidia-smi not found"
fi

if [ -x "$(command -v nvcc)" ]; then
    nvcc --version
else
    echo "nvcc not found"
fi

echo "Job complete"
