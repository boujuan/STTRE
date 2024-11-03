#!/bin/bash
#SBATCH --job-name=sttre_training
#SBATCH --output=sttre_%j.log
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=32G

# module load mamba
# mamba activate sttre

# srun -p all_gpu.p -w mpcg004 -N1 -n1 -c8 --mem=16G --gres=gpu:H100:1 -t 2:0:0 --pty bash -i
# srun -p all_gpu.p -N1 -n1 -c8 --mem=16G --gres=gpu:3 -t 2:0:0 --pty bash -i
# srun -p all_gpu.p -N1 -n1 -c32 --mem=32G --gres=gpu:4 -t 2:0:0 --pty bash -i
srun -p all_gpu.p -N1 -n1 -c32 --mem=32G --gres=gpu:H100:4 -t 2:0:0 ~/STTRE/run.sh