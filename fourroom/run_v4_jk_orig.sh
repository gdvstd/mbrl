#!/bin/bash
# v4: JSRL/KSRL with our ORIGINAL params (h0=50, lambda0=0.5) alongside
# the author-suggested ones (h0=100, lambda0=1.0) -- the better of each
# pair becomes the reproduction baseline (fair comparison).
cd "$(dirname "$0")/.."
PY=./venv/bin/python
until grep -q V4_FTFULL_DONE runs/logs/v4_all.log 2>/dev/null; do sleep 300; done
for sc in altgoal locked; do
  for s in 0 1 2; do
    $PY fourroom/train_paper_student.py --scenario $sc --transfer jsrl \
        --jsrl-h0 50 --seed $s \
        --teacher runs/paperv4_${sc}_teacher/final_model.zip \
        --outdir runs/paperv4_${sc}_jsrl50_seed$s \
        > runs/logs/v4_jsrl50_${sc}_s$s.log 2>&1 &
  done
  wait
  for s in 0 1 2; do
    $PY fourroom/train_paper_student.py --scenario $sc --transfer ksrl \
        --ksrl-lambda0 0.5 --seed $s \
        --teacher runs/paperv4_${sc}_teacher/final_model.zip \
        --outdir runs/paperv4_${sc}_ksrl05_seed$s \
        > runs/logs/v4_ksrl05_${sc}_s$s.log 2>&1 &
  done
  wait
done
echo V4_JK_ORIG_DONE
