"""Watch the LATEST snapshot of a training run while it is still training.

Non-intrusive: this opens a separate render window on the most recent saved
model (best_model.zip or the newest checkpoint), so you can eyeball how the
policy is doing right now without slowing down the background training.

Usage (inside the venv, in a separate terminal):
    python watch_latest.py runs/student_none_seed0 --env hardcore
    python watch_latest.py runs/student_none_seed0 --env hardcore --loop
    python watch_latest.py runs/teacher_sac --env normal

--loop: after each set of episodes, re-scan for a newer snapshot and play it,
so as training writes new checkpoints you keep seeing the latest policy.
Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import time

import gymnasium as gym
from stable_baselines3 import SAC

from sac_common import ENV_IDS


def _ckpt_steps(path: str) -> int:
    m = re.search(r"sac_(\d+)_steps", os.path.basename(path))
    return int(m.group(1)) if m else -1


def latest_snapshot(rundir: str) -> str | None:
    """Newest model in a run dir: prefer highest-step checkpoint, else best_model."""
    ckpts = glob.glob(os.path.join(rundir, "checkpoints", "sac_*_steps.zip"))
    if ckpts:
        return max(ckpts, key=_ckpt_steps)
    best = os.path.join(rundir, "best_model.zip")
    return best if os.path.exists(best) else None


def play(model_path: str, env_id: str, episodes: int, seed: int) -> None:
    env = gym.make(env_id, render_mode="human")
    model = SAC.load(model_path, device="cpu")
    label = os.path.basename(model_path)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        done = trunc = False
        steps = 0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = env.step(action)
            total += r
            steps += 1
        print(f"[{label}] episode {ep + 1}/{episodes} steps={steps} return={total:.1f}")
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rundir", type=str)
    parser.add_argument("--env", choices=ENV_IDS, default="hardcore")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--loop",
        action="store_true",
        help="keep replaying the newest snapshot as training produces new ones",
    )
    parser.add_argument(
        "--poll", type=int, default=15,
        help="with --loop, seconds to wait before checking for a newer snapshot",
    )
    args = parser.parse_args()

    env_id = ENV_IDS[args.env]
    last_played = None
    while True:
        snap = latest_snapshot(args.rundir)
        if snap is None:
            print(f"[watch] no snapshot yet in {args.rundir} "
                  f"(waiting for first checkpoint/best_model)...")
        elif snap != last_played or not args.loop:
            print(f"[watch] latest snapshot -> {snap}")
            play(snap, env_id, args.episodes, args.seed)
            last_played = snap
        else:
            print(f"[watch] no newer snapshot yet ({os.path.basename(snap)}); "
                  f"waiting {args.poll}s...")
        if not args.loop:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
