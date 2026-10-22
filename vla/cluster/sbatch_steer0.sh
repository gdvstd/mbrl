#!/bin/bash -l
#SBATCH -A joecamp
#SBATCH -p a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=steer0
#SBATCH --output=/scratch/gilbreth/nam120/vla/%x-%j.out
set -e
WORK=/scratch/gilbreth/nam120/vla
module load anaconda 2>/dev/null || module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK/env"
export HF_HOME=$WORK/hf_cache MUJOCO_GL=egl
cd "$WORK/code"
LORA=$WORK/runs_vla/openvla_seen5_v3/lora_ep4
NORM=$WORK/runs_vla/openvla_seen5_v3/norm_stats.json
mkdir -p $WORK/runs_vla/steer

python -u steer/logit_lens.py --out $WORK/runs_vla/steer/lens_base.json
python -u steer/logit_lens.py --lora $LORA \
    --out $WORK/runs_vla/steer/lens_lora.json
python -u steer/steer_probe.py --lora $LORA \
    --library $WORK/runs_vla/steer/lens_lora.json \
    --tasks 0 2 5 --episodes 3 \
    --out $WORK/runs_vla/steer/steer_probe_lora.json
python -u steer/steer_probe.py --norm-stats $NORM \
    --library $WORK/runs_vla/steer/lens_base.json \
    --tasks 0 2 5 --episodes 3 --concepts fast slow up down \
    --out $WORK/runs_vla/steer/steer_probe_base.json
echo STEER0 ALL DONE
