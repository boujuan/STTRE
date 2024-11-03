#!/bin/bash
source ~/.bashrc
mamba activate sttre
torchrun --nproc_per_node=4 ~/STTRE/_Old/STTRE_PL.py