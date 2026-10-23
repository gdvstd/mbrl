#!/bin/bash -l
#SBATCH -A joecamp
#SBATCH -p a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=steer0b
#SBATCH --output=/scratch/gilbreth/nam120/vla/%x-%j.out
# Calibration sweep: steer0 showed the LoRA policy is disrupted (success
# 44%->0%) at n=20/alpha>=5 while base-model steering is direction-correct
# and huge. Find the gentle regime where the LoRA policy is steered but
# stays competent: fewer neurons, smaller alpha.
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
    --tasks 0 2 5 --episodes 3 --n-neurons 8 --alphas 2 4 \
    --concepts fast slow up down stop push \
    --out $WORK/runs_vla/steer/steer_probe_lora_n8.json
python -u steer/steer_probe.py --lora $LORA \
    --library $WORK/runs_vla/steer/lens_lora.json \
    --tasks 0 2 5 --episodes 3 --n-neurons 4 --alphas 3 6 \
    --concepts fast slow up down stop push \
    --out $WORK/runs_vla/steer/steer_probe_lora_n4.json
echo STEER0B ALL DONE
