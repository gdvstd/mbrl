#!/bin/bash -l
#SBATCH -A joecamp
#SBATCH -p a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=steer1
#SBATCH --output=/scratch/gilbreth/nam120/vla/%x-%j.out
# Phase-1 pilot: success DISCOVERY on the unseen tasks (baseline ~0%).
# steer0b found up/down@a3 (n=4) produced 1/3 successes on task 5 from a
# 0% baseline. Scale that test: all unseen tasks, 8 paired episodes,
# gentle regime, concept set covering the pick-place skill sequence.
set -e
WORK=/scratch/gilbreth/nam120/vla
module load anaconda 2>/dev/null || module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK/env"
export HF_HOME=$WORK/hf_cache MUJOCO_GL=egl
cd "$WORK/code"
LORA=$WORK/runs_vla/openvla_seen5_v3/lora_ep4

python -u steer/steer_probe.py --lora $LORA \
    --library $WORK/runs_vla/steer/lens_lora.json \
    --tasks 5 6 7 8 9 --episodes 8 --n-neurons 4 --alphas 3 \
    --concepts up down grasp lift place push fast \
    --out $WORK/runs_vla/steer/steer1_unseen.json
echo STEER1 ALL DONE
