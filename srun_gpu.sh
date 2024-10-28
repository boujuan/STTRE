#!/bin/bash
# srun -p all_gpu.p -w mpcg004 -N1 -n1 -c8 --mem=16G --gres=gpu:H100:1 -t 2:0:0 --pty bash -i
# srun -p all_gpu.p -N1 -n1 -c8 --mem=16G --gres=gpu:3 -t 2:0:0 --pty bash -i
srun --partition=all_gpu.p --nodelist=mpcg004 --nodes=1 --ntasks=1 --cpus-per-task=128 --mem=256G --gres=gpu:H100:4 --time=02:00:00 --pty bash -i