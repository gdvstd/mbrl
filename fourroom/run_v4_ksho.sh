#!/bin/bash
# ksho-v2: reachability-handover gate (AA until the student's episode-level
# ID occupancy exceeds the random baseline by Welch z>=2.5 on 2 consecutive
# windows, then strict tau=max(cal) gate). 4 sentinel probe envs outside
# the PPO buffer. Chained after ksclip.
cd "$(dirname "$0")/.."
PY=./venv/bin/python
mkdir -p runs/logs
until grep -q V4_KSCLIP_DONE runs/logs/v4ksclip_all.log 2>/dev/null; do
  sleep 300
done
for s in 0 1 2; do
  for sc in altgoal locked; do
    $PY fourroom/train_paper_student.py --scenario $sc --transfer ksho \
        --seed $s --teacher runs/paperv4_${sc}_teacher/final_model.zip \
        --outdir runs/paperv4_${sc}_ksho_seed$s \
        > runs/logs/v4_ksho_${sc}_s$s.log 2>&1 &
  done
done
wait
echo V4_KSHO_DONE
