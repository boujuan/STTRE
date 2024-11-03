#!/bin/bash
source ~/.bashrc
mamba activate sttre

# Disable Python output buffering
export PYTHONUNBUFFERED=1

# Set the number of threads per process (8 CPU cores per GPU (32/4=8))
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN # Enable NCCL debugging

# Run with 4 GPUs
torchrun --nproc_per_node=4 --master_port=29500 ~/STTRE/STTRE_PL.py