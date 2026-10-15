"""Bisect the collection-vs-eval gap: the SAME BC policy scores ~70% on
task 2 via make_env eval but ~0% inside PPO collection. Which link breaks?

  A: TaskMixEnv([2]) raw gym loop, deterministic predict
  B: make_vla_vec_env([2], n_envs=4) loop, deterministic predict

A low + B low  -> TaskMixEnv obs/reset path differs from make_env eval
A ok  + B low  -> VecEnv/collection layer
A ok  + B ok   -> the gap is in PPO's sampling/learning after all

Usage: python probe_pathway.py --model <zip> [--task 2] [--episodes 10]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vla_ppo_common import TaskMixEnv, make_vla_vec_env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=10)
    a = ap.parse_args()
    from stable_baselines3 import PPO
    model = PPO.load(a.model, device="cuda")

    # A: raw TaskMixEnv
    env = TaskMixEnv([a.task], seed=0)
    succ = 0
    for ep in range(a.episodes):
        obs, _ = env.reset()
        r, done = 0.0, False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, te, tr, _ = env.step(act)
            done = te or tr
        succ += r > 0
    env.close()
    print(f"A TaskMixEnv raw : {succ}/{a.episodes}", flush=True)

    # B: the vec pathway PPO collection uses
    venv = make_vla_vec_env([a.task], seed=0, n_envs=4)
    obs = venv.reset()
    done_eps, succ = 0, 0
    while done_eps < 4 * a.episodes:
        act, _ = model.predict(obs, deterministic=True)
        obs, rews, dones, infos = venv.step(act)
        for i, d in enumerate(dones):
            if d:
                done_eps += 1
                succ += infos[i].get("episode", {}).get("r", rews[i]) > 0
    venv.close()
    print(f"B vec pathway    : {succ}/{done_eps}")
    print("PATHWAY PROBE DONE")


if __name__ == "__main__":
    main()
