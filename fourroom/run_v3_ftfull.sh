#!/bin/bash
# v3 FULL fine-tuning (conv unfrozen) x 3 seeds x both scenarios,
# chained after the baseline phase.
cd "$(dirname "$0")/.."
PY=./venv/bin/python
until grep -q V3_BASELINES_DONE runs/logs/v3_all.log 2>/dev/null; do sleep 300; done
for sc in altgoal locked; do
  for s in 0 1 2; do
    $PY fourroom/train_paper_student.py --scenario $sc --transfer finetune \
        --no-freeze-cnn --seed $s \
        --teacher runs/paperv3_${sc}_teacher/final_model.zip \
        --outdir runs/paperv3_${sc}_ftfull_seed$s \
        > runs/logs/v3_ftfull_${sc}_s$s.log 2>&1 &
  done
  wait
done
echo V3_FTFULL_DONE
