#!/bin/bash
# v3 config (author feedback round 2): altgoal agent starts in Room 2
# (both tasks); forward masked only at walls. Full-obs pipeline:
# scratch calibration -> teachers (plain + ereg) -> 5 baselines x 3 seeds.
cd "$(dirname "$0")/.."
PY=./venv/bin/python
mkdir -p runs/logs

for sc in altgoal locked; do
  for s in 0 1 2; do
    $PY fourroom/train_paper_student.py --scenario $sc --transfer scratch \
        --seed $s --outdir runs/paperv3_${sc}_scratch_seed$s \
        > runs/logs/v3_scratch_${sc}_s$s.log 2>&1 &
  done
done
wait
echo V3_SCRATCH_DONE

$PY fourroom/train_paper_teacher.py --scenario altgoal \
    --outdir runs/paperv3_altgoal_teacher > runs/logs/v3teacher_altgoal.log 2>&1 &
$PY fourroom/train_paper_teacher.py --scenario altgoal --energy-reg \
    --outdir runs/paperv3_altgoal_teacher_ereg > runs/logs/v3teacher_altgoal_ereg.log 2>&1 &
$PY fourroom/train_paper_teacher.py --scenario locked \
    --outdir runs/paperv3_locked_teacher > runs/logs/v3teacher_locked.log 2>&1 &
$PY fourroom/train_paper_teacher.py --scenario locked --energy-reg \
    --outdir runs/paperv3_locked_teacher_ereg > runs/logs/v3teacher_locked_ereg.log 2>&1 &
wait
echo V3_TEACHERS_DONE

for sc in altgoal locked; do
  for tr in finetune aa jsrl ksrl ebtl; do
    for s in 0 1 2; do
      $PY fourroom/train_paper_student.py --scenario $sc --transfer $tr \
          --seed $s --teacher runs/paperv3_${sc}_teacher/final_model.zip \
          --outdir runs/paperv3_${sc}_${tr}_seed$s \
          > runs/logs/v3_${tr}_${sc}_s$s.log 2>&1 &
    done
    wait
  done
done
echo V3_BASELINES_DONE
