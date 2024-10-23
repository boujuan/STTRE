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

# Enable error tracing
set -x

# Print basic info
echo "Starting job at $(date)"
echo "Running on host: $(hostname)"

# Load only CUDA module
module purge
module load CUDA/12.2.0

# Try basic nvidia-smi
nvidia-smi

echo "Job finished at $(date)"