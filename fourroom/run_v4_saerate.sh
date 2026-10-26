#!/bin/bash
# Rate-targeted SAE gate (tau = running 5% quantile of recent target recon
# errors; coin unchanged) x 3 seeds x both scenarios. Compare against the
# fixed-threshold sae runs: fixes the per-seed starvation/blowout brittleness?
cd "$(dirname "$0")/.."
PY=./venv/bin/python
mkdir -p runs/logs
for s in 0 1 2; do
  for sc in altgoal locked; do
    $PY fourroom/train_paper_student.py --scenario $sc --transfer saerate \
        --seed $s --teacher runs/paperv4_${sc}_teacher/final_model.zip \
        --outdir runs/paperv4_${sc}_saerate_seed$s \
        > runs/logs/v4_saerate_${sc}_s$s.log 2>&1 &
  done
done
wait
echo V4_SAERATE_DONE
