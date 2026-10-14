#!/bin/bash
# SAE-recon-gated advising vs EBTL/gap: 3 seeds x {altgoal, locked}, full-obs.
cd "$(dirname "$0")/.."
PY=./venv/bin/python
mkdir -p runs/logs
for s in 0 1 2; do
  $PY fourroom/train_paper_student.py --scenario altgoal --transfer sae \
      --seed $s > runs/logs/sae_altgoal_s$s.log 2>&1 &
done
wait
for s in 0 1 2; do
  $PY fourroom/train_paper_student.py --scenario locked --transfer sae \
      --seed $s > runs/logs/sae_locked_s$s.log 2>&1 &
done
wait
echo SAE_RUNS_DONE
