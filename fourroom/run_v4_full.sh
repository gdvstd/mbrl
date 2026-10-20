#!/bin/bash
# v4 config (author feedback round 3): 9x9 grid + Floor doorways,
# padding=1 first conv, TorchRL-parity PPO params, 100-episode evals,
# successful-episode tau calibration, jsrl-h0 100, ksrl-lambda 1.0.
# scratch calibration -> teachers (plain+ereg) -> 5 baselines -> full-FT.
cd "$(dirname "$0")/.."
PY=./venv/bin/python
mkdir -p runs/logs

for sc in altgoal locked; do
  for s in 0 1 2; do
    $PY fourroom/train_paper_student.py --scenario $sc --transfer scratch \
        --seed $s --outdir runs/paperv4_${sc}_scratch_seed$s \
        > runs/logs/v4_scratch_${sc}_s$s.log 2>&1 &
  done
done
wait
echo V4_SCRATCH_DONE

$PY fourroom/train_paper_teacher.py --scenario altgoal \
    --outdir runs/paperv4_altgoal_teacher > runs/logs/v4teacher_altgoal.log 2>&1 &
$PY fourroom/train_paper_teacher.py --scenario altgoal --energy-reg \
    --outdir runs/paperv4_altgoal_teacher_ereg > runs/logs/v4teacher_altgoal_ereg.log 2>&1 &
$PY fourroom/train_paper_teacher.py --scenario locked \
    --outdir runs/paperv4_locked_teacher > runs/logs/v4teacher_locked.log 2>&1 &
$PY fourroom/train_paper_teacher.py --scenario locked --energy-reg \
    --outdir runs/paperv4_locked_teacher_ereg > runs/logs/v4teacher_locked_ereg.log 2>&1 &
wait
echo V4_TEACHERS_DONE

for sc in altgoal locked; do
  for tr in finetune aa jsrl ksrl ebtl; do
    for s in 0 1 2; do
      $PY fourroom/train_paper_student.py --scenario $sc --transfer $tr \
          --seed $s --teacher runs/paperv4_${sc}_teacher/final_model.zip \
          --outdir runs/paperv4_${sc}_${tr}_seed$s \
          > runs/logs/v4_${tr}_${sc}_s$s.log 2>&1 &
    done
    wait
  done
done
echo V4_BASELINES_DONE

for sc in altgoal locked; do
  for s in 0 1 2; do
    $PY fourroom/train_paper_student.py --scenario $sc --transfer finetune \
        --no-freeze-cnn --seed $s \
        --teacher runs/paperv4_${sc}_teacher/final_model.zip \
        --outdir runs/paperv4_${sc}_ftfull_seed$s \
        > runs/logs/v4_ftfull_${sc}_s$s.log 2>&1 &
  done
  wait
done
echo V4_FTFULL_DONE
