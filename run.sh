#!/bin/bash
source ~/.bashrc
mamba activate sttre

# Disable Python output buffering
export PYTHONUNBUFFERED=1

# Set the number of threads per process (8 CPU cores per GPU (32/4=8))
export OMP_NUM_THREADS=8
# Enable NCCL debugging, choose INFO for more details
export NCCL_DEBUG=WARN 

export NCCL_IB_DISABLE=0  # Enable InfiniBand
export NCCL_IB_HCA=mlx5  # Specify IB adapter
export NCCL_NET_GDR_LEVEL=5  # Enable GPUDirect RDMA
export NCCL_P2P_LEVEL=5  # Enable P2P between GPUs

# Get SLURM variables
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
NNODES=$SLURM_NNODES # Number of nodes
NODE_RANK=$SLURM_NODEID # Node rank
GPUS_PER_NODE=2 # Number of GPUs per node (default is 4)

# Calculate world size (total number of GPUs)
WORLD_SIZE=$(($NNODES * $GPUS_PER_NODE))

# Export variables for PyTorch Lightning
export SLURM_NNODES=$NNODES
export SLURM_GPUS_PER_NODE=$GPUS_PER_NODE
export WORLD_SIZE=$WORLD_SIZE

# Run distributed training
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --node_rank=$NODE_RANK \
    ~/STTRE/STTRE_PL.py