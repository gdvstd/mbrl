"""Roll a trained policy on libero_object tasks and report success rates.

The core sanity check for the transfer premise: the BC teacher should
succeed on SEEN tasks (0-4) and fail on UNSEEN ones (5-9).

Usage:
    python eval_policy.py --ckpt runs_vla/teacher_bc.pt --episodes 10
    python eval_policy.py --ckpt ... --tasks 0 5      # subset
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from vla_common import (
    MAX_STEPS,
    SEEN_TASKS,
    UNSEEN_TASKS,
    GaussianBCPolicy,
    make_env,
    preprocess_obs,
)


def rollout(pol, env, episodes: int, device: str) -> float:
    from vla_common import get_init_states
    inits = get_init_states(env.task_id)
    succ = 0
    for ep in range(episodes):
        env.reset()
        obs = env.set_init_state(inits[ep % len(inits)])
        if hasattr(pol, "reset_plan"):
            pol.reset_plan()
        for _ in range(MAX_STEPS):
            img, prop = preprocess_obs(obs, env.task_id)
            action = pol.act(img, prop, deterministic=True, device=device)
            obs, r, done, info = env.step(action)
            if done:
                break
        succ += int(done and r > 0)
    return succ / episodes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--tasks", type=int, nargs="*", default=None)
    p.add_argument("--device", type=str,
                   default="mps" if torch.backends.mps.is_available() else "cpu")
    a = p.parse_args()

    pol = GaussianBCPolicy().to(a.device)
    pol.load_state_dict(torch.load(a.ckpt, map_location=a.device, weights_only=False))
    pol.eval()

    tasks = a.tasks if a.tasks is not None else list(SEEN_TASKS + UNSEEN_TASKS)
    results = {}
    for tid in tasks:
        env = make_env(tid)
        sr = rollout(pol, env, a.episodes, a.device)
        env.close()
        tag = "seen" if tid in SEEN_TASKS else "UNSEEN"
        results[tid] = sr
        print(f"[eval] task {tid} ({tag}): success {sr:.0%}  | {env.language}")

    seen = [results[t] for t in tasks if t in SEEN_TASKS]
    uns = [results[t] for t in tasks if t in UNSEEN_TASKS]
    if seen:
        print(f"[eval] SEEN mean:   {np.mean(seen):.0%}")
    if uns:
        print(f"[eval] UNSEEN mean: {np.mean(uns):.0%}")


if __name__ == "__main__":
    main()
