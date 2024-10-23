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

# Load environment
module purge
module load CUDA/12.4.0

# Activate conda environment
source ~/.bashrc
mamba activate wind_forecasting_cuda

# Print versions
python --version
nvcc --version
nvidia-smi

# Try a simple Python CUDA test
cat << EOF > test_cuda.py
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
EOF

python test_cuda.py

echo "Job complete"