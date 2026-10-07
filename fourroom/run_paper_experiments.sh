#!/bin/bash
# Batch driver for the PAPER-FAITHFUL four-room track (11x11, MaskablePPO).
#
# Phase 1 (teachers, must precede students):
#   {altgoal, locked} x {plain, energy-reg}   (200K / 800K budgets)
# Phase 2 (students): per scenario x seed {0,1,2}:
#   scratch finetune aa jsrl ksrl ebtl ebtl_ereg   (200K / 1M budgets)
# 3 concurrent. Launch detached (survives terminal / harness):
#   nohup caffeinate -is ./run_paper_experiments.sh > runs/logs/paper_driver.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=./venv/bin/python
LOGDIR=runs/logs
mkdir -p "$LOGDIR"
export OMP_NUM_THREADS=2

run_batch() {  # runs commands from "$@" with <=3 concurrent jobs
    for cmd in "$@"; do
        while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge 3 ]; do sleep 10; done
        echo "[start $(date +%H:%M:%S)] $cmd"
        bash -c "$cmd" &
    done
    wait
}

T=()
SCENARIOS="${@:-altgoal locked}"
for sc in $SCENARIOS; do
    T+=("$PY fourroom/train_paper_teacher.py --scenario $sc > $LOGDIR/pt_${sc}.log 2>&1")
    T+=("$PY fourroom/train_paper_teacher.py --scenario $sc --energy-reg > $LOGDIR/pt_${sc}_ereg.log 2>&1")
done
echo "=== phase 1: teachers ==="
run_batch "${T[@]}"

S=()
for sc in $SCENARIOS; do
    for seed in 0 1 2; do
        for m in scratch finetune aa jsrl ksrl ebtl; do
            S+=("$PY fourroom/train_paper_student.py --scenario $sc --transfer $m --seed $seed > $LOGDIR/ps_${sc}_${m}_s${seed}.log 2>&1")
        done
        S+=("$PY fourroom/train_paper_student.py --scenario $sc --transfer ebtl --seed $seed \
            --teacher runs/paper_${sc}_teacher_ereg/final_model.zip \
            --outdir runs/paper_${sc}_ebtl_ereg_seed${seed} > $LOGDIR/ps_${sc}_ebtl_ereg_s${seed}.log 2>&1")
        S+=("$PY fourroom/train_paper_student.py --scenario $sc --transfer finetune --no-freeze-cnn --seed $seed \
            --outdir runs/paper_${sc}_finetune_full_seed${seed} > $LOGDIR/ps_${sc}_finetune_full_s${seed}.log 2>&1")
    done
done
echo "=== phase 2: students (${#S[@]} runs) ==="
run_batch "${S[@]}"
echo "ALL PAPER EXPERIMENTS DONE $(date +%H:%M:%S)"
