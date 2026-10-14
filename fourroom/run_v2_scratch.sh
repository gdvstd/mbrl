#!/bin/bash
# No-transfer calibration under the author-confirmed config (max_steps 500,
# time-decayed reward, new locked source): 3 seeds x {altgoal, locked} x
# {full-obs, ego}. v2 outdirs -- old results are kept untouched.
cd "$(dirname "$0")/.."
PY=./venv/bin/python
mkdir -p runs/logs
for sc in altgoal locked; do
  for ego in "" "--ego"; do
    sfx=$([ -n "$ego" ] && echo "_ego" || echo "")
    for s in 0 1 2; do
      $PY fourroom/train_paper_student.py --scenario $sc --transfer scratch \
          $ego --seed $s \
          --outdir runs/paperv2_${sc}${sfx}_scratch_seed$s \
          > runs/logs/v2scratch_${sc}${sfx}_s$s.log 2>&1 &
    done
    wait
  done
done
echo V2_SCRATCH_DONE
