#!/bin/bash
# Oracle-optimal teacher upper-bound controls: aa/jsrl/ksrl x 3 seeds x
# {altgoal, locked}, full-obs, same schedules/budgets as the baselines.
cd "$(dirname "$0")/.."
PY=./venv/bin/python
mkdir -p runs/logs
for sc in altgoal locked; do
  for tr in aa jsrl ksrl; do
    for s in 0 1 2; do
      $PY fourroom/train_paper_student.py --scenario $sc --transfer $tr \
          --oracle-teacher --seed $s \
          --outdir runs/paper_${sc}_o${tr}_seed$s \
          > runs/logs/o${tr}_${sc}_s$s.log 2>&1 &
    done
    wait
  done
done
echo ORACLE_RUNS_DONE
