#!/bin/bash
srun --partition=all_gpu.p --nodelist=mpcg004 --nodes=1 --ntasks=1 \
     --cpus-per-task=8 --mem=16G --gres=gpu:H100:1 --time=02:00:00 \
     --pty bash -i