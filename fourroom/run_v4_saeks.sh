#!/bin/bash
# KS/Youden-optimal online-calibrated SAE gate (PU mixture; tau = argmax of
# the source-vs-target recon CDF gap, D<0.1 turns the gate off), 3 seeds x
# both scenarios. Heuristic-free rival to saeclip.
cd "$(dirname "$0")/.."
PY=./venv/bin/python
mkdir -p runs/logs
for s in 0 1 2; do
  for sc in altgoal locked; do
    $PY fourroom/train_paper_student.py --scenario $sc --transfer saeks \
        --seed $s --teacher runs/paperv4_${sc}_teacher/final_model.zip \
        --outdir runs/paperv4_${sc}_saeks_seed$s \
        > runs/logs/v4_saeks_${sc}_s$s.log 2>&1 &
  done
done
wait
echo V4_SAEKS_DONE
