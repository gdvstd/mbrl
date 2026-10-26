#!/bin/bash
# Two SAE-gate fixes vs the fixed-tau brittleness, 3 seeds x 2 scenarios:
#  saeclip: absolute tau with rate guardrails [1%, 10%]
#  saetau4: plain fixed tau but x4 gap-rule multiplier (more permissive)
cd "$(dirname "$0")/.."
PY=./venv/bin/python
mkdir -p runs/logs

for s in 0 1 2; do
  for sc in altgoal locked; do
    $PY fourroom/train_paper_student.py --scenario $sc --transfer saeclip \
        --seed $s --teacher runs/paperv4_${sc}_teacher/final_model.zip \
        --outdir runs/paperv4_${sc}_saeclip_seed$s \
        > runs/logs/v4_saeclip_${sc}_s$s.log 2>&1 &
  done
done
wait
echo V4_SAECLIP_DONE

for s in 0 1 2; do
  for sc in altgoal locked; do
    $PY fourroom/train_paper_student.py --scenario $sc --transfer sae \
        --sae-tau-mult 4.0 --seed $s \
        --teacher runs/paperv4_${sc}_teacher/final_model.zip \
        --outdir runs/paperv4_${sc}_saetau4_seed$s \
        > runs/logs/v4_saetau4_${sc}_s$s.log 2>&1 &
  done
done
wait
echo V4_SAETAU4_DONE
echo V4_SAEVARIANTS_DONE
