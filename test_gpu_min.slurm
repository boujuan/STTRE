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
#SBATCH --gres=gpu:1

# Enable verbose mode
set -x

# Print basic info
hostname
date
pwd

# Try loading newer CUDA version
module --force purge
module load CUDA/12.4.0

# Check GPU visibility
nvidia-smi

echo "Job complete"