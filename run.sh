#!/bin/bash
source ~/.bashrc
mamba activate sttre

# Set the number of threads per process (8 CPU cores per GPU (32/4=8))
export OMP_NUM_THREADS=8

# Run with 4 GPUs
torchrun --nproc_per_node=4 ~/STTRE/STTRE_PL.py