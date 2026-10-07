#!/bin/bash
#SBATCH -A joecamp
#SBATCH -p a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=openvla-seen5
#SBATCH --output=/scratch/gilbreth/nam120/vla/%x-%j.out
set -e
WORK=/scratch/gilbreth/nam120/vla
module load anaconda 2>/dev/null || module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK/env"
export HF_HOME=$WORK/hf_cache MUJOCO_GL=egl
cd "$WORK/code"
python -u finetune_openvla.py --epochs 3 --batch-size 8 --grad-accum 2 \
    --out "$WORK/runs_vla/openvla_seen5"
