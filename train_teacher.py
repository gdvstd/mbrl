"""Train the teacher policy with SAC on BipedalWalker-v3 (Task A, flat terrain).

The learned policy is later transferred to BipedalWalkerHardcore-v3 (Task B).
Policy network is a simple 3-layer MLP (24 -> 256 -> 256 -> 4). Hyperparameters
are shared with the student via sac_common so every run uses an identical setup.

Usage (inside the venv):
    python train_teacher.py                         # 500k steps, seed 0
    python train_teacher.py --timesteps 300000 --seed 1
    python train_teacher.py --outdir runs/teacher_sac_seed0

Outputs (under --outdir): best_model.zip / final_model.zip / checkpoints/ /
eval/ / tb/ / train_monitor.monitor.csv
"""
from __future__ import annotations

import argparse
import os

from sac_common import ENV_IDS, SOLVED_REWARD, build_sac, make_callbacks, make_env

TEACHER_ENV = ENV_IDS["normal"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--outdir", type=str, default="runs/teacher_sac")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="train the full budget instead of stopping when solved (>=300)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tb_dir = os.path.join(args.outdir, "tb")

    train_env = make_env(
        TEACHER_ENV, args.seed, os.path.join(args.outdir, "train_monitor")
    )
    eval_env = make_env(
        TEACHER_ENV, args.seed + 1000,
        os.path.join(args.outdir, "eval", "eval_monitor"),
    )

    model = build_sac(train_env, args.seed, args.device, tb_dir)

    callbacks = make_callbacks(
        eval_env,
        args.outdir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        early_stop_threshold=None if args.no_early_stop else SOLVED_REWARD,
    )

    print(f"[teacher] env={TEACHER_ENV} timesteps={args.timesteps} "
          f"device={args.device}")
    print(f"[teacher] outputs -> {os.path.abspath(args.outdir)}")
    model.learn(
        total_timesteps=args.timesteps, callback=callbacks, progress_bar=True
    )

    final_path = os.path.join(args.outdir, "final_model.zip")
    model.save(final_path)
    print(f"[teacher] saved final model -> {final_path}")
    print(f"[teacher] best model (by eval) -> "
          f"{os.path.join(args.outdir, 'best_model.zip')}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
