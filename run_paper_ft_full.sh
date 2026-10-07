#!/bin/bash
# Comparison variant: UNfrozen (full) fine-tuning, both scenarios x 3 seeds.
# Runs sequentially (1 at a time) so it can coexist with the main batch.
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
LOGDIR=runs/logs
export OMP_NUM_THREADS=2
for sc in altgoal locked; do
  for seed in 0 1 2; do
    echo "[start $(date +%H:%M:%S)] ft_full $sc seed$seed"
    $PY train_paper_student.py --scenario $sc --transfer finetune \
        --no-freeze-cnn --seed $seed \
        --outdir runs/paper_${sc}_finetune_full_seed${seed} \
        > $LOGDIR/ps_${sc}_finetune_full_s${seed}.log 2>&1
  done
done
echo "FT-FULL RUNS DONE $(date +%H:%M:%S)"
