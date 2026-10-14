"""Collect SUCCESSFUL OpenVLA-teacher rollouts as a distillation dataset.

Rolls the LoRA teacher deterministically over each seen task's fixed init
states and keeps only successful episodes, recording the STUDENT's
preprocessed observation (6ch 128px image + proprio) and the executed
7-dim action at every step. This is distillation from the teacher's own
behavior, not from demos -- the transfer premise stays "knowledge flows
from the teacher".

Output npz: images (N,6,128,128) uint8, proprios (N,P) f32,
actions (N,7) f32, ep_id (N,) i64, task_id (N,) i64.

Usage:
    python collect_teacher_rollouts.py --lora <dir> \
        --eps-per-task 30 --out runs_vla/teacher_rollouts.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vla_common import (  # noqa: E402
    MAX_STEPS, SEEN_TASKS, get_init_states, make_env, preprocess_obs,
)


def task_language(tid: int) -> str:
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()["libero_object"]().get_task(
        tid).language


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--norm-stats", default=None)
    ap.add_argument("--tasks", type=int, nargs="*", default=list(SEEN_TASKS))
    ap.add_argument("--eps-per-task", type=int, default=30)
    ap.add_argument("--out", default="runs_vla/teacher_rollouts.npz")
    a = ap.parse_args()

    from eval_openvla import act, load_teacher
    norm = a.norm_stats or os.path.join(
        os.path.dirname(a.lora.rstrip("/")), "norm_stats.json")
    model, processor = load_teacher(a.lora, norm)

    imgs, props, acts, ep_ids, task_ids = [], [], [], [], []
    ep_counter = 0
    for tid in a.tasks:
        env = make_env(tid)
        inits = get_init_states(tid)
        lang = task_language(tid)
        n_succ = 0
        for ep in range(min(a.eps_per_task, len(inits))):
            env.reset()
            obs = env.set_init_state(inits[ep])
            traj, r = [], 0.0
            for _ in range(MAX_STEPS):
                action = act(model, processor, obs, lang)
                img, prop = preprocess_obs(obs, tid)
                traj.append(((img * 255).astype(np.uint8),
                             prop.astype(np.float32),
                             action.astype(np.float32)))
                obs, r, done, _ = env.step(action)
                if done:
                    break
            if r > 0:
                n_succ += 1
                for im, pr, ac in traj:
                    imgs.append(im)
                    props.append(pr)
                    acts.append(ac)
                    ep_ids.append(ep_counter)
                    task_ids.append(tid)
                ep_counter += 1
            print(f"[collect] task {tid} ep {ep}: "
                  f"{'SUCCESS' if r > 0 else 'fail'} ({len(traj)} steps)",
                  flush=True)
        env.close()
        print(f"[collect] task {tid}: {n_succ} successes kept", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(
        a.out, images=np.stack(imgs), proprios=np.stack(props),
        actions=np.stack(acts), ep_id=np.array(ep_ids),
        task_id=np.array(task_ids))
    print(f"[collect] saved {len(imgs)} frames / {ep_counter} episodes "
          f"-> {a.out}")
    print("COLLECT DONE")


if __name__ == "__main__":
    main()
