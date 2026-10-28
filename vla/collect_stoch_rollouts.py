"""Stochastic successful teacher rollouts for SAE anchoring (the EBTL
author-confirmed recipe: stochastic episodes, keep successful only).

Unlike the greedy BC-collection rollouts, stochastic sampling + success
filtering defines ID as the teacher's RECOVERABLE BASIN -- states it can
wander into and still solve from -- not just the modal path. Stores raw
agentview frames in the same key layout vla_ks_probe_ro expects
(images (N,3,H,W) uint8, ep_id, task_id).

Usage (GPU node):
    python collect_stoch_rollouts.py --lora <dir> --eps-per-task 25 \
        --temperature 0.6 --out runs_vla/teacher_rollouts_stoch.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vla_common import MAX_STEPS, SEEN_TASKS, get_init_states, make_env  # noqa: E402
from eval_openvla import UNNORM_KEY, load_teacher  # noqa: E402
from vla_energy import task_language  # noqa: E402


@torch.no_grad()
def act_stoch(model, processor, obs, lang, temperature, device="cuda"):
    img = Image.fromarray(obs["agentview_image"]).resize((224, 224))
    prompt = (f"In: What action should the robot take to "
              f"{lang.lower()}?\nOut:")
    inputs = processor(prompt, img).to(device, dtype=torch.bfloat16)
    action = model.predict_action(**inputs, unnorm_key=UNNORM_KEY,
                                  do_sample=True, temperature=temperature)
    return np.asarray(action, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--norm-stats", default=None)
    ap.add_argument("--eps-per-task", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--out", default="runs_vla/teacher_rollouts_stoch.npz")
    a = ap.parse_args()
    norm = a.norm_stats or os.path.join(
        os.path.dirname(a.lora.rstrip("/")), "norm_stats.json")
    model, processor = load_teacher(a.lora, norm)

    imgs, ep_ids, task_ids = [], [], []
    ep_counter = 0
    for tid in SEEN_TASKS:
        env = make_env(tid)
        inits = get_init_states(tid)
        lang = task_language(tid)
        n_succ = 0
        for ep in range(a.eps_per_task):
            env.reset()
            obs = env.set_init_state(inits[ep % len(inits)])
            traj, r = [], 0.0
            for _ in range(MAX_STEPS):
                traj.append(np.transpose(obs["agentview_image"],
                                         (2, 0, 1)).astype(np.uint8))
                obs, r, done, _ = env.step(
                    act_stoch(model, processor, obs, lang, a.temperature))
                if done:
                    break
            if r > 0:
                n_succ += 1
                for im in traj:
                    imgs.append(im)
                    ep_ids.append(ep_counter)
                    task_ids.append(tid)
                ep_counter += 1
            print(f"[stoch] task {tid} ep {ep}: "
                  f"{'SUCCESS' if r > 0 else 'fail'} ({len(traj)} steps)",
                  flush=True)
        env.close()
        print(f"[stoch] task {tid}: {n_succ}/{a.eps_per_task} successes",
              flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, images=np.stack(imgs),
                        ep_id=np.array(ep_ids), task_id=np.array(task_ids))
    print(f"[stoch] saved {len(imgs)} frames / {ep_counter} successful "
          f"episodes (T={a.temperature}) -> {a.out}")
    print("STOCH COLLECT DONE")


if __name__ == "__main__":
    main()
