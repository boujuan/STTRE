#!/bin/bash
#SBATCH --job-name=test_module
#SBATCH --output=test_module_%j.out
#SBATCH --error=test_module_%j.err
#SBATCH --partition=cfdg.p
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

# Enable verbose mode
set -x

# Print basic info
pwd
ls -la

# Try loading newer CUDA version based on available modules
echo "Loading CUDA module..."
module --force purge
module load CUDA/12.4.0
echo "Module load exit status: $?"

# Print environment
env | sort

echo "Job complete"
