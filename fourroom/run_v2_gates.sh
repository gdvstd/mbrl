#!/bin/bash
# v2 gate-score variants (after the baseline pipeline): SAID-act gate and
# SAE-recon L1/L3 gates, with the v2 teachers. 3 seeds each.
cd "$(dirname "$0")/.."
PY=./venv/bin/python
until grep -q V2_BASELINES_DONE runs/logs/v2_all.log 2>/dev/null; do sleep 300; done

run3 () {  # scenario transfer extra outtag
  for s in 0 1 2; do
    $PY fourroom/train_paper_student.py --scenario $1 --transfer $2 $3 \
        --seed $s --teacher runs/paperv2_$1_teacher/final_model.zip \
        --outdir runs/paperv2_$1_$4_seed$s \
        > runs/logs/v2_$4_$1_s$s.log 2>&1 &
  done
  wait
}
run3 altgoal sae ""                 sae
run3 altgoal sae "--sae-layer L3"   saeL3
run3 altgoal said ""                said
run3 locked  sae ""                 sae
echo V2_GATES_DONE
