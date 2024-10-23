#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=test_gpu_%j.out
#SBATCH --error=test_gpu_%j.err
#SBATCH --partition=all_gpu.p
#SBATCH --nodelist=mpcg005    # Try a different node
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1         # Simplified GRES

# Basic info only
hostname
date

# Minimal module loading
module purge
module load CUDA

# Just try to print PATH
echo "PATH=$PATH"

echo "Job complete"