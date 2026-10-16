#!/bin/bash
# v2 EGO phase: waits for the full-obs pipeline, then ego teachers +
# all ego baselines (plain teachers; 3 seeds).
cd "$(dirname "$0")/.."
PY=./venv/bin/python
until grep -q V2_BASELINES_DONE runs/logs/v2_all.log 2>/dev/null; do sleep 300; done

$PY fourroom/train_paper_teacher.py --scenario altgoal --ego \
    --outdir runs/paperv2_altgoal_teacher_ego \
    > runs/logs/v2teacher_altgoal_ego.log 2>&1 &
$PY fourroom/train_paper_teacher.py --scenario locked --ego \
    --outdir runs/paperv2_locked_teacher_ego \
    > runs/logs/v2teacher_locked_ego.log 2>&1 &
wait
echo V2_EGO_TEACHERS_DONE

for sc in altgoal locked; do
  for tr in finetune aa jsrl ksrl ebtl; do
    for s in 0 1 2; do
      $PY fourroom/train_paper_student.py --scenario $sc --transfer $tr \
          --ego --seed $s \
          --teacher runs/paperv2_${sc}_teacher_ego/final_model.zip \
          --outdir runs/paperv2_${sc}_ego_${tr}_seed$s \
          > runs/logs/v2ego_${tr}_${sc}_s$s.log 2>&1 &
    done
    wait
  done
done
echo V2_EGO_DONE
