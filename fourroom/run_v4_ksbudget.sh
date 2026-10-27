#!/bin/bash
# budgeted-KS: tau_eff = max(tau_KS-once, Q_recent(b)) -- the proposed
# paper mechanism, NEVER yet run as such (saeclip used the gap-rule
# anchor, 11x looser in locked). b=2% primary + b=1%/5% sweep on seed 0.
cd "$(dirname "$0")/.."
PY=./venv/bin/python
mkdir -p runs/logs
for s in 0 1 2; do
  for sc in altgoal locked; do
    $PY fourroom/train_paper_student.py --scenario $sc --transfer ksbudget \
        --sae-rate-lo 0.02 --seed $s \
        --teacher runs/paperv4_${sc}_teacher/final_model.zip \
        --outdir runs/paperv4_${sc}_ksbudget_seed$s \
        > runs/logs/v4_ksbudget_${sc}_s$s.log 2>&1 &
  done
done
wait
echo V4_KSBUDGET_MAIN_DONE
for b in 0.01 0.05; do
  for sc in altgoal locked; do
    $PY fourroom/train_paper_student.py --scenario $sc --transfer ksbudget \
        --sae-rate-lo $b --seed 0 \
        --teacher runs/paperv4_${sc}_teacher/final_model.zip \
        --outdir runs/paperv4_${sc}_ksbudget_b${b#0.}_seed0 \
        > runs/logs/v4_ksbudget_b${b#0.}_${sc}_s0.log 2>&1 &
  done
done
wait
echo V4_KSBUDGET_DONE
