#!/bin/bash
# Full-history KS gate ablation (buffer = random-rollout seed + ALL training
# recon scores, no window/EMA), 3 seeds x both scenarios.
# Chained after the saeksonce batch to avoid overloading the machine.
cd "$(dirname "$0")/.."
PY=./venv/bin/python
mkdir -p runs/logs
until grep -q V4_SAEKSONCE_DONE runs/logs/v4saeksonce_all.log 2>/dev/null; do
  sleep 300
done
for s in 0 1 2; do
  for sc in altgoal locked; do
    $PY fourroom/train_paper_student.py --scenario $sc --transfer saekfull \
        --seed $s --teacher runs/paperv4_${sc}_teacher/final_model.zip \
        --outdir runs/paperv4_${sc}_saekfull_seed$s \
        > runs/logs/v4_saekfull_${sc}_s$s.log 2>&1 &
  done
done
wait
echo V4_SAEKFULL_DONE
