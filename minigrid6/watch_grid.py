"""Render a trained MiniGrid policy as a GIF (or live pygame window).

Rolls DETERMINISTIC episodes of the checkpoint in its env and writes an
animated GIF, so each method's final behavior can be eyeballed and compared.

Usage (inside the venv):
    python watch_grid.py --run runs/grid_student6_ebtl_seed1            # GIF
    python watch_grid.py --run runs/grid_teacher_ppo_6x6 --env empty
    python watch_grid.py --run runs/grid_student6_aa_seed0 --human      # live

The env defaults to the student task (doorkey). GIFs land in runs/videos/
named after the run directory. Failed policies just wander, so episodes are
cut at --max-steps frames to keep GIFs short.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import os

import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from PIL import Image
from stable_baselines3 import PPO

from minigrid6.grid_common import grid_env_id


def rollout_frames(model, env, episodes: int, max_steps: int,
                   hold: int = 0) -> tuple[list, list]:
    """`hold` repeats each episode's final frame, pausing the GIF there."""
    frames, returns = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        ep_ret, steps = 0.0, 0
        frames.append(env.render())
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            steps += 1
            frames.append(env.render())
            if terminated or truncated or steps >= max_steps:
                break
        frames.extend([frames[-1]] * hold)
        returns.append(ep_ret)
    return frames, returns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True,
                        help="run dir containing best_model.zip (or a .zip path)")
    parser.add_argument("--env", choices=["doorkey", "empty"], default="doorkey")
    parser.add_argument("--size", type=int, default=6, choices=[5, 6, 8, 16])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=120,
                        help="cut each episode's GIF after this many steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=32,
                        help="pixels per grid cell (64+ for presentations)")
    parser.add_argument("--hold", type=int, default=0,
                        help="freeze each episode's last frame for N frames")
    parser.add_argument("--out", type=str, default=None,
                        help="output gif path (default runs/videos/<run>.gif)")
    parser.add_argument("--human", action="store_true",
                        help="open a live pygame window instead of writing a GIF")
    args = parser.parse_args()

    ckpt = args.run if args.run.endswith(".zip") else os.path.join(
        args.run, "best_model.zip")
    model = PPO.load(ckpt, device="cpu")

    render_mode = "human" if args.human else "rgb_array"
    env = ImgObsWrapper(
        gym.make(grid_env_id(args.env, args.size), render_mode=render_mode,
                 tile_size=args.tile_size))
    env.reset(seed=args.seed)

    if args.human:
        while True:  # Ctrl-C to quit
            rollout_frames(model, env, episodes=1, max_steps=args.max_steps)
    frames, returns = rollout_frames(model, env, args.episodes, args.max_steps,
                                     hold=args.hold)
    env.close()

    out = args.out or os.path.join(
        "runs", "videos",
        os.path.basename(os.path.normpath(args.run)) + ".gif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / args.fps), loop=0)
    print(f"[watch-grid] {ckpt}: episode returns {[f'{r:.2f}' for r in returns]}")
    print(f"[watch-grid] wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
