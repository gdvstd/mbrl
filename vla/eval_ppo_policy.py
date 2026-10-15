"""Evaluate a saved PPO student zip on the 10 LIBERO tasks.

Usage: python eval_ppo_policy.py --model <zip> [--stochastic] [--episodes 10]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vla_common import (  # noqa: E402
    MAX_STEPS, SEEN_TASKS, UNSEEN_TASKS, get_init_states, make_env,
    preprocess_obs,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--set-std", type=float, default=None,
                    help="override policy std before eval")
    ap.add_argument("--init-mode", choices=["first", "random"],
                    default="first",
                    help="first: inits[ep %% n] (distill-eval convention); "
                         "random: uniform over the full init set "
                         "(TaskMixEnv collection convention)")
    a = ap.parse_args()
    from stable_baselines3 import PPO
    model = PPO.load(a.model, device="cuda")
    if a.set_std is not None:
        import torch as th
        with th.no_grad():
            model.policy.log_std.fill_(float(np.log(a.set_std)))
    log_std = model.policy.log_std.detach().cpu().numpy()
    print(f"[eval] {a.model}  deterministic={not a.stochastic}")
    print(f"[eval] policy log_std: {np.round(log_std, 2)} "
          f"(std {np.round(np.exp(log_std), 2)})")
    tot = {"seen": [], "unseen": []}
    for tid in list(SEEN_TASKS) + list(UNSEEN_TASKS):
        env = make_env(tid)
        inits = get_init_states(tid)
        rng = np.random.default_rng(tid)
        succ = 0
        for ep in range(a.episodes):
            env.reset()
            k = (ep % len(inits) if a.init_mode == "first"
                 else int(rng.integers(len(inits))))
            raw = env.set_init_state(inits[k])
            r = 0.0
            for _ in range(MAX_STEPS):
                img, prop = preprocess_obs(raw, tid)
                obs = {"image": (img * 255).astype(np.uint8),
                       "proprio": prop.astype(np.float32)}
                act, _ = model.predict(obs, deterministic=not a.stochastic)
                raw, r, done, _ = env.step(np.asarray(act, np.float64))
                if done:
                    break
            succ += r > 0
        tag = "seen" if tid in SEEN_TASKS else "unseen"
        tot[tag].append(succ / a.episodes)
        print(f"[eval] task {tid} ({tag}): {succ}/{a.episodes}", flush=True)
        env.close()
    print(f"[eval] SEEN mean {np.mean(tot['seen']):.2f}  "
          f"UNSEEN mean {np.mean(tot['unseen']):.2f}")
    print("PPO EVAL DONE")


if __name__ == "__main__":
    main()
