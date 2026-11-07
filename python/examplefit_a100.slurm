#!/usr/bin/env bash
#SBATCH --job-name fit
#SBATCH --constraint gpu
#SBATCH --qos regular
#SBATCH --nodes 2
#SBATCH --gpus-per-node 4
#SBATCH --time 2:00:00
#SBATCH --account XXXXX

source /path/to/venv/grace/bin/activate
# export TRITON_CACHE_DIR=$SCRATCH/.triton TORCHINDUCTOR_CACHE_DIR=$SCRATCH/.inductor

srun -N2 -n8 -c32 /path/to/peacemaker.py -r input.yaml
