#!/bin/bash
#SBATCH -A joecamp
#SBATCH --gpus-per-node=1
#SBATCH --constraint=A100-40gb
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=vla-student
#SBATCH --output=%x-%j.out
# Student baseline run on the full 10-task target.
# Usage: sbatch sbatch_student.sh <transfer> <seed> [timesteps]
set -u
WORK=$CLUSTER_SCRATCH/vla
source activate "$WORK/env"
export MUJOCO_GL=egl OMP_NUM_THREADS=4
cd "$WORK/code"
python train_vla_student.py --transfer "${1:-scratch}" --seed "${2:-0}" \
    --timesteps "${3:-500000}" --tasks 0 1 2 3 4 5 6 7 8 9 \
    --n-envs 8 --device cuda
