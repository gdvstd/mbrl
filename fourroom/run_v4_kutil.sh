#!/bin/bash
# Utility-anchored gate (advantage-calibrated depth): tau_eff =
# max(tube tau0, Q_recent(q_t)), q_t tuned by the randomized
# advised-vs-holdout GAE contrast. 3 seeds x both scenarios, chained
# after the ksgates NEED/coin ablation.
cd "$(dirname "$0")/.."
PY=./venv/bin/python
mkdir -p runs/logs
until grep -q V4_KSGATES_DONE runs/logs/v4ksgates_all.log 2>/dev/null; do
  sleep 300
done
for s in 0 1 2; do
  for sc in altgoal locked; do
    $PY fourroom/train_paper_student.py --scenario $sc --transfer kutil \
        --seed $s --teacher runs/paperv4_${sc}_teacher/final_model.zip \
        --outdir runs/paperv4_${sc}_kutil_seed$s \
        > runs/logs/v4_kutil_${sc}_s$s.log 2>&1 &
  done
done
wait
echo V4_KUTIL_DONE
