"""Load a trained SAC policy and watch it with render_mode="human".

Usage (inside the venv):
    python watch_policy.py runs/teacher_sac/best_model.zip
    python watch_policy.py runs/teacher_sac/best_model.zip --env normal --episodes 5
    python watch_policy.py runs/teacher_sac/best_model.zip --env hardcore
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse

import gymnasium as gym
from stable_baselines3 import SAC

ENV_IDS = {
    "normal": "BipedalWalker-v3",
    "hardcore": "BipedalWalkerHardcore-v3",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=str)
    parser.add_argument("--env", choices=ENV_IDS, default="normal")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample actions instead of using the deterministic mean",
    )
    args = parser.parse_args()

    env = gym.make(ENV_IDS[args.env], render_mode="human")
    model = SAC.load(args.model_path, device="cpu")
    print(f"[watch] {args.model_path} on {ENV_IDS[args.env]}")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
        print(f"[episode {ep + 1}/{args.episodes}] steps={steps} return={total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    main()
