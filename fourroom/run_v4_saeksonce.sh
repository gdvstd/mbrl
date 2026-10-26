#!/bin/bash
# KS-once: single pre-training two-sample calibration (source cal frames vs
# equal-size masked-random target frames), tau frozen, zero online knobs.
# 3 seeds x both scenarios -- if ~= saeks, online adaptation is unnecessary.
cd "$(dirname "$0")/.."
PY=./venv/bin/python
mkdir -p runs/logs
for s in 0 1 2; do
  for sc in altgoal locked; do
    $PY fourroom/train_paper_student.py --scenario $sc --transfer saeksonce \
        --seed $s --teacher runs/paperv4_${sc}_teacher/final_model.zip \
        --outdir runs/paperv4_${sc}_saeksonce_seed$s \
        > runs/logs/v4_saeksonce_${sc}_s$s.log 2>&1 &
  done
done
wait
echo V4_SAEKSONCE_DONE
