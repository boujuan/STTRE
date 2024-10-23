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

# Print job info
echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Working Directory: $PWD"

# Print SLURM environment variables
echo -e "\n=== SLURM Environment ==="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"

# Print system info
echo -e "\n=== System Info ==="
echo "CPU Info:"
lscpu | grep "Model name"
echo "Memory Info:"
free -h

# Module environment
echo -e "\n=== Module Environment ==="
module purge
echo "Loading CUDA module..."
module load CUDA/12.2.0
module list

# CUDA environment
echo -e "\n=== CUDA Environment ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# GPU Info
echo -e "\n=== GPU Information ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found!"
fi

# Test CUDA compilation
echo -e "\n=== Testing CUDA Compilation ==="
nvcc --version || echo "nvcc not found!"

# Print end time
echo -e "\n=== Job Complete ==="
echo "End Time: $(date)"
