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
#SBATCH --gres=gpu:H100:1
#SBATCH --constraint=H100

# Enable debugging
set -x

# Print initial environment
env | sort > initial_env.txt

# Create logs directory
mkdir -p logs

# Print job info
echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Working Directory: $PWD"

# Print SLURM GPU-related environment variables
echo "=== SLURM GPU Environment ==="
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GPU_DEVICE_ORDINAL: $GPU_DEVICE_ORDINAL"

# Load modules with error checking
echo "=== Loading Modules ==="
module purge
module load CUDA/12.2.0 || echo "Failed to load CUDA module"
module list

# Print updated environment
env | sort > after_module_env.txt
diff initial_env.txt after_module_env.txt || true

# Check CUDA installation
echo "=== CUDA Installation ==="
ls -l /usr/local/cuda* || echo "No CUDA in /usr/local"
echo "which nvidia-smi: $(which nvidia-smi || echo 'not found')"
echo "which nvcc: $(which nvcc || echo 'not found')"

# Try to get GPU info multiple ways
echo "=== GPU Information ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found"
    # Try to find it in common locations
    for p in /usr/bin /usr/local/bin /usr/local/cuda/bin; do
        if [ -x "$p/nvidia-smi" ]; then
            echo "Found nvidia-smi at $p/nvidia-smi"
            $p/nvidia-smi
            break
        fi
    done
fi

# Print final status
echo "=== Job Complete ==="
echo "End Time: $(date)"

# Cleanup
rm -f initial_env.txt after_module_env.txt