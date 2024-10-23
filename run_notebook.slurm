#!/bin/bash

# To run the script: sbatch run_notebook.slurm
# squeue -u $USER
# sacct -j <jobid> --format=JobID,JobName,MaxRSS,Elapsed
# Check output logs
# tail -f logs/STTRE_GPU_*.out
# Check error logs
# tail -f logs/STTRE_GPU_*.err

# Check partition limits
# sinfo -o "%20P %5D %6t %8z %10m %N"

# Check current usage
# squeue -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
# squeue -u $USER

# srun --partition=all_gpu.p --nodelist=mpcg004 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G --gres=gpu:H100:1 --time=02:00:00 --pty bash -i

# ssh mpcg004 nvidia-smi
# sstat --format=AveCPU,AveRSS,AveVMSize --jobs=<jobid>
# scontrol show nodes mpcg004 | grep "Gres"
# srun --jobid=<jobid> --pty top

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

# Exit on error and print commands
set -e
set -x

# Create logs directory first (since output/error files depend on it)
mkdir -p logs || { echo "Failed to create logs directory"; exit 1; }

# Echo start time and node info
echo "Job started at: $(date)"
echo "Running on node: $(hostname) at $(date) with SLURM_JOB_ID: ${SLURM_JOB_ID}"

# Create other necessary directories
for dir in DATA plots models; do
    mkdir -p $dir || { echo "Failed to create $dir directory"; exit 1; }
done

# Load required modules with error checking
echo "Loading modules..."
module purge || { echo "Failed to purge modules"; exit 1; }
module load hpc-env/13.1 || { echo "Failed to load hpc-env module"; exit 1; }
module load Mamba/24.3.0-0 || { echo "Failed to load Mamba module"; exit 1; }
module load CUDA/12.4.0 || { echo "Failed to load CUDA module"; exit 1; }
module list

# Activate conda environment
echo "Activating conda environment..."
mamba activate wind_forecasting_cuda || { echo "Failed to activate conda environment"; exit 1; }

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

# Download data
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
    sys.exit(1)
" || { echo "Failed to download data"; exit 1; }

# Run main script
echo "Starting main training script..."
python -O STTRE.py || { echo "Failed to run main script"; exit 1; }

# Compress results
echo "Compressing results..."
tar -czf results_${SLURM_JOB_ID}.tar.gz plots/ models/ logs/ || { echo "Failed to compress results"; exit 1; }

echo "Job completed at: $(date)"
