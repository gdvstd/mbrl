#!/bin/bash
# Gap-gate (teacher-familiar AND student-uncertain) vs EBTL comparison.
# 3 seeds x {altgoal, locked}, full-obs, same budgets as the paper runs.
cd "$(dirname "$0")/.."
PY=./venv/bin/python
mkdir -p runs/logs
for s in 0 1 2; do
  $PY fourroom/train_paper_student.py --scenario altgoal --transfer gap \
      --seed $s > runs/logs/gap_altgoal_s$s.log 2>&1 &
done
wait
for s in 0 1 2; do
  $PY fourroom/train_paper_student.py --scenario locked --transfer gap \
      --seed $s > runs/logs/gap_locked_s$s.log 2>&1 &
done
wait
echo GAP_RUNS_DONE
