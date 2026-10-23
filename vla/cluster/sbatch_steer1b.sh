#!/bin/bash -l
#SBATCH -A joecamp
#SBATCH -p a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=steer1b
#SBATCH --output=/scratch/gilbreth/nam120/vla/%x-%j.out
# steer1 (gentle, n=4 a=3) was null on unseen tasks: 0% discovery, the
# earlier up/down 1/3 was luck. Missing matrix cell: STRONG steering on
# the weak-policy region -- unseen tasks have no competence to protect
# (baseline 0%), and steer0 showed fast@a10 opens +0.26m vertical range
# on t5. Test n=20, alpha {5,10} on all unseen tasks.
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
    --tasks 5 6 7 8 9 --episodes 6 --n-neurons 20 --alphas 5 10 \
    --concepts up down grasp place push fast \
    --out $WORK/runs_vla/steer/steer1b_unseen_strong.json
echo STEER1B ALL DONE
