#!/bin/bash
# Ego-observation variant of the paper suite (11x11 egocentric, occluded).
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
LOGDIR=runs/logs
mkdir -p "$LOGDIR"
export OMP_NUM_THREADS=2
run_batch() {
    for cmd in "$@"; do
        while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge 3 ]; do sleep 10; done
        echo "[start $(date +%H:%M:%S)] $cmd"
        bash -c "$cmd" &
    done
    wait
}
T=()
for sc in altgoal locked; do
    T+=("$PY train_paper_teacher.py --scenario $sc --ego > $LOGDIR/pt_${sc}_ego.log 2>&1")
    T+=("$PY train_paper_teacher.py --scenario $sc --energy-reg --ego > $LOGDIR/pt_${sc}_ereg_ego.log 2>&1")
done
echo "=== ego phase 1: teachers ==="
run_batch "${T[@]}"
S=()
for sc in altgoal locked; do
    for seed in 0 1 2; do
        for m in scratch finetune aa jsrl ksrl ebtl; do
            S+=("$PY train_paper_student.py --scenario $sc --transfer $m --ego --seed $seed > $LOGDIR/ps_${sc}_ego_${m}_s${seed}.log 2>&1")
        done
        S+=("$PY train_paper_student.py --scenario $sc --transfer ebtl --ego --seed $seed \
            --teacher runs/paper_${sc}_teacher_ereg_ego/final_model.zip \
            --outdir runs/paper_${sc}_ego_ebtl_ereg_seed${seed} > $LOGDIR/ps_${sc}_ego_ebtl_ereg_s${seed}.log 2>&1")
        S+=("$PY train_paper_student.py --scenario $sc --transfer finetune --no-freeze-cnn --ego --seed $seed \
            --outdir runs/paper_${sc}_ego_finetune_full_seed${seed} > $LOGDIR/ps_${sc}_ego_finetune_full_s${seed}.log 2>&1")
    done
done
echo "=== ego phase 2: students (${#S[@]} runs) ==="
run_batch "${S[@]}"
echo "ALL EGO EXPERIMENTS DONE $(date +%H:%M:%S)"
