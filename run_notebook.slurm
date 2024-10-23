#!/bin/bash

# To run the script: sbatch run_notebook.slurm
# squeue -u $USER
# sacct -j <jobid> --format=JobID,JobName,MaxRSS,Elapsed
# Check output logs
# tail -f logs/STTRE_GPU_*.out
# Check error logs
# tail -f logs/STTRE_GPU_*.err

#SBATCH --job-name=STTRE_GPU
#SBATCH --output=logs/STTRE_GPU_%j.out
#SBATCH --error=logs/STTRE_GPU_%j.err
#SBATCH --partition=all_gpu.p
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Create necessary directories
mkdir -p logs
mkdir -p DATA
mkdir -p plots
mkdir -p models

# Load required modules
module purge
module load Mamba/24.3.0-0
module load CUDA/12.2.0

# Activate conda environment
source activate wind_forecasting_cuda

# Print system info
nvidia-smi
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('Number of GPUs:', torch.cuda.device_count())
print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')
"

# Download data first
echo "Downloading Uber stock data..."
python -c "
import yfinance as yf
import os

try:
    uber = yf.download('UBER', start='2019-05-10', end='2024-01-01', progress=False)
    uber.to_csv('DATA/uber_stock.csv')
    print('Data downloaded successfully')
    print(f'File size: {os.path.getsize("DATA/uber_stock.csv")/1024:.2f} KB')
except Exception as e:
    print(f'Error downloading data: {e}')
    exit(1)
"

# Run the main script with error handling
echo "Starting main training script..."
python -O STTRE.py
if [ $? -ne 0 ]; then
    echo "Error in main script execution"
    exit 1
fi

# Compress results if successful
# echo "Compressing results..."
# tar -czf results_${SLURM_JOB_ID}.tar.gz plots/ models/ logs/

echo "Job completed successfully!"
