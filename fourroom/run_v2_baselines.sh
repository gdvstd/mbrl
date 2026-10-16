#!/bin/bash
# v2 (author-feedback config) full pipeline: teachers then all baselines.
# Phase 1: 4 teachers in parallel (plain + ereg lambda=1e-4, full-obs).
# Phase 2: {finetune, aa, jsrl, ksrl, ebtl} x {altgoal, locked} x 3 seeds.
cd "$(dirname "$0")/.."
PY=./venv/bin/python
mkdir -p runs/logs

$PY fourroom/train_paper_teacher.py --scenario altgoal \
    --outdir runs/paperv2_altgoal_teacher \
    > runs/logs/v2teacher_altgoal.log 2>&1 &
$PY fourroom/train_paper_teacher.py --scenario altgoal --energy-reg \
    --outdir runs/paperv2_altgoal_teacher_ereg \
    > runs/logs/v2teacher_altgoal_ereg.log 2>&1 &
$PY fourroom/train_paper_teacher.py --scenario locked \
    --outdir runs/paperv2_locked_teacher \
    > runs/logs/v2teacher_locked.log 2>&1 &
$PY fourroom/train_paper_teacher.py --scenario locked --energy-reg \
    --outdir runs/paperv2_locked_teacher_ereg \
    > runs/logs/v2teacher_locked_ereg.log 2>&1 &
wait
echo V2_TEACHERS_DONE

for sc in altgoal locked; do
  for tr in finetune aa jsrl ksrl ebtl; do
    for s in 0 1 2; do
      $PY fourroom/train_paper_student.py --scenario $sc --transfer $tr \
          --seed $s --teacher runs/paperv2_${sc}_teacher/final_model.zip \
          --outdir runs/paperv2_${sc}_${tr}_seed$s \
          > runs/logs/v2_${tr}_${sc}_s$s.log 2>&1 &
    done
    wait
  done
done
echo V2_BASELINES_DONE
