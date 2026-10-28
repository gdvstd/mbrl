#!/bin/bash -l
#SBATCH -A joecamp
#SBATCH -p a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=ksrostoch
#SBATCH --output=/scratch/gilbreth/nam120/vla/%x-%j.out
# EBTL-recipe anchor: STOCHASTIC successful teacher rollouts (recoverable
# basin, not the modal path) -> rollout-anchored KS probe rerun.
set -e
WORK=/scratch/gilbreth/nam120/vla
module load anaconda 2>/dev/null || module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK/env"
export HF_HOME=$WORK/hf_cache MUJOCO_GL=egl
cd "$WORK/code"
LORA=$WORK/runs_vla/openvla_seen5_v3/lora_ep4

python -u collect_stoch_rollouts.py --lora $LORA --eps-per-task 20 \
    --temperature 0.6 --out $WORK/runs_vla/teacher_rollouts_stoch.npz
python -u vla_ks_probe_ro.py --lora $LORA --layer 8 \
    --rollouts $WORK/runs_vla/teacher_rollouts_stoch.npz \
    --out $WORK/runs_vla/ks_probe_vla_ro_stoch_L8.npz
echo KSROSTOCH ALL DONE
