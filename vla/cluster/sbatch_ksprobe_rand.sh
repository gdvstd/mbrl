#!/bin/bash -l
#SBATCH -A joecamp
#SBATCH -p a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=ksprand
#SBATCH --output=/scratch/gilbreth/nam120/vla/%x-%j.out
# Early-student starvation measurement: F_T from RANDOM-action rollouts
# (start-of-training proxy), scored at the teacher-rollout-calibrated
# tau (ref) and at a freshly computed KS tau. L8.
set -e
WORK=/scratch/gilbreth/nam120/vla
module load anaconda 2>/dev/null || module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK/env"
export HF_HOME=$WORK/hf_cache MUJOCO_GL=egl
cd "$WORK/code"
LORA=$WORK/runs_vla/openvla_seen5_v3/lora_ep4

python -u vla_ks_probe.py --lora $LORA --layer 8 --policy random \
    --ref-npz $WORK/runs_vla/ks_probe_vla_L8.npz \
    --out $WORK/runs_vla/ks_probe_vla_L8_random.npz
echo KSPRAND ALL DONE
