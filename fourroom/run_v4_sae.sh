#!/bin/bash
# v4 SAE-recon gate (gap-rule threshold) x 3 seeds x both scenarios,
# chained after the original-param JSRL/KSRL batch.
cd "$(dirname "$0")/.."
PY=./venv/bin/python
until grep -q V4_JK_ORIG_DONE runs/logs/v4jk_all.log 2>/dev/null; do sleep 300; done
for sc in altgoal locked; do
  for s in 0 1 2; do
    $PY fourroom/train_paper_student.py --scenario $sc --transfer sae \
        --seed $s --teacher runs/paperv4_${sc}_teacher/final_model.zip \
        --outdir runs/paperv4_${sc}_sae_seed$s \
        > runs/logs/v4_sae_${sc}_s$s.log 2>&1 &
  done
  wait
done
echo V4_SAE_DONE
