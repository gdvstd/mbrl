#!/bin/bash
# Gap gate WITHOUT the delta(t) coin (purely state-driven, self-annealing
# via the student-entropy condition). Waits for the SAE batch to finish.
cd "$(dirname "$0")/.."
PY=./venv/bin/python
until grep -q SAE_RUNS_DONE runs/logs/sae_all.log 2>/dev/null; do sleep 120; done
for s in 0 1 2; do
  $PY fourroom/train_paper_student.py --scenario altgoal --transfer gap \
      --gap-no-coin --seed $s \
      --outdir runs/paper_altgoal_gapnc_seed$s \
      > runs/logs/gapnc_altgoal_s$s.log 2>&1 &
done
wait
for s in 0 1 2; do
  $PY fourroom/train_paper_student.py --scenario locked --transfer gap \
      --gap-no-coin --seed $s \
      --outdir runs/paper_locked_gapnc_seed$s \
      > runs/logs/gapnc_locked_s$s.log 2>&1 &
done
wait
echo GAPNC_RUNS_DONE
