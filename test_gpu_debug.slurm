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

# Load CUDA first
echo "=== Loading CUDA ==="
module --force purge
module load CUDA/12.4.0

# Set explicit GPU visibility
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Print loaded modules
module list

# Basic GPU test
if [ -x "$(command -v nvidia-smi)" ]; then
    echo "=== GPU Information ==="
    nvidia-smi
else
    echo "nvidia-smi not found in PATH"
    echo "PATH=$PATH"
fi

echo "Job complete"
