#!/bin/bash
# Batch driver for the MiniGrid EBTL-track comparison.
#
# Runs every remaining (method, seed) cell at a fixed 1M-step budget:
#   seed 0:    aa jsrl ksrl ebtl(q=.5) ebtl(q=.1)   (scratch/finetune done)
#   seeds 1,2: scratch finetune aa jsrl ksrl ebtl(q=.5) ebtl(q=.1)
# 3 runs at a time (M1 Pro, 8 cores; each run uses 8 DummyVecEnvs).
# Teacher: single seed-0 teacher reused across student seeds (documented
# simplification). Launch detached so it survives the terminal:
#   nohup caffeinate -is ./run_grid_experiments.sh > runs/logs/driver.log 2>&1 &
set -u
cd "$(dirname "$0")"
PY=./venv/bin/python
LOGDIR=runs/logs
mkdir -p "$LOGDIR"
export OMP_NUM_THREADS=2   # keep 3 concurrent torch procs from thrashing

CMDS=()
add() { CMDS+=("$1"); }

student() {  # method, seed, extra args..., logname
    local m="$1" s="$2" log="$3"; shift 3
    add "$PY train_grid_student.py --size 6 --transfer $m --seed $s $* \
        > $LOGDIR/${log}_s${s}.log 2>&1"
}

for s in 0 1 2; do
    if [ "$s" != "0" ]; then
        student scratch  "$s" scratch
        student finetune "$s" finetune
    fi
    student aa   "$s" aa
    student jsrl "$s" jsrl
    student ksrl "$s" ksrl
    student ebtl "$s" ebtl
    student ebtl "$s" ebtlq01 --ebtl-q 0.1 \
        --outdir "runs/grid_student6_ebtlq01_seed$s"
done

# bounded parallelism via bash job control (BSD xargs -I truncates long cmds)
MAX_JOBS=3
for cmd in "${CMDS[@]}"; do
    while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAX_JOBS" ]; do
        sleep 10
    done
    echo "[start $(date +%H:%M:%S)] $cmd"
    bash -c "$cmd" &
done
wait
echo "ALL EXPERIMENTS DONE $(date +%H:%M:%S)"
