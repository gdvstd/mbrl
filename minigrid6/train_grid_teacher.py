"""Train the MiniGrid teacher with PPO on MiniGrid-Empty-8x8-v0 (Task A).

The learned policy is later transferred to MiniGrid-DoorKey-8x8-v0 (Task B),
where the pre-key phase is out-of-distribution for this teacher -- the setup
EBTL targets. Config is shared with the student via grid_common.

Usage (inside the venv):
    python train_grid_teacher.py                      # 200k steps, seed 0
    python train_grid_teacher.py --timesteps 100000 --seed 1

Outputs (under --outdir): best_model.zip / final_model.zip / checkpoints/ /
eval/ / tb/ / monitor csvs
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import os

from minigrid6.grid_common import (
    GRID_SOLVED_REWARD,
    N_ENVS,
    build_ppo,
    grid_env_id,
    make_grid_vec_env,
)
from shared.callbacks import make_callbacks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size", type=int, default=6, choices=[5, 6, 8, 16],
        help="grid side length (6 recommended; see grid_common.grid_env_id)",
    )
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-freq", type=int, default=20_000,
        help="evaluate every N total env steps (across all vec workers)",
    )
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="train the full budget instead of stopping when solved (>=0.90)",
    )
    # -- EBTL teacher energy regularization (paper Sec. 4.2) --
    parser.add_argument(
        "--energy-reg", action="store_true",
        help="add the margin-based energy loss (OOD states from random "
             "rollouts in the same-size DoorKey target env)",
    )
    parser.add_argument("--energy-lambda", type=float, default=0.1)
    parser.add_argument("--m-in", type=float, default=7.0)
    parser.add_argument("--m-out", type=float, default=2.0)
    args = parser.parse_args()

    teacher_env = grid_env_id("empty", args.size)
    default_dir = f"runs/grid_teacher_ppo_{args.size}x{args.size}" + (
        "_ereg" if args.energy_reg else "")
    outdir = args.outdir or default_dir
    os.makedirs(outdir, exist_ok=True)
    tb_dir = os.path.join(outdir, "tb")

    train_env = make_grid_vec_env(
        teacher_env, args.seed, monitor_dir=os.path.join(outdir, "train_monitor")
    )
    eval_env = make_grid_vec_env(
        teacher_env, args.seed + 1000, n_envs=1,
        monitor_dir=os.path.join(outdir, "eval", "eval_monitor"),
    )

    if args.energy_reg:
        from minigrid6.grid_energy_reg import EnergyRegPPO, collect_random_states

        ood_env = grid_env_id("doorkey", args.size)
        print(f"[grid-teacher] collecting OOD states from random {ood_env} rollouts")
        ood = collect_random_states(ood_env, n_frames=8000, seed=args.seed + 3000)
        model = build_ppo(
            train_env, args.seed, args.device, tb_dir,
            algo_cls=EnergyRegPPO, ood_states=ood,
            energy_lambda=args.energy_lambda, m_in=args.m_in, m_out=args.m_out,
        )
        print(f"[grid-teacher] energy reg on: lambda={args.energy_lambda} "
              f"m_in={args.m_in} m_out={args.m_out} ood_n={len(ood)}")
    else:
        model = build_ppo(train_env, args.seed, args.device, tb_dir)

    callbacks = make_callbacks(
        eval_env,
        outdir,
        # EvalCallback counts per-vec-step, each of which is N_ENVS frames.
        eval_freq=max(args.eval_freq // N_ENVS, 1),
        n_eval_episodes=args.n_eval_episodes,
        early_stop_threshold=None if args.no_early_stop else GRID_SOLVED_REWARD,
        name_prefix="ppo",
    )

    print(f"[grid-teacher] env={teacher_env} timesteps={args.timesteps} "
          f"device={args.device}")
    print(f"[grid-teacher] outputs -> {os.path.abspath(outdir)}")
    model.learn(
        total_timesteps=args.timesteps, callback=callbacks, progress_bar=True
    )

    final_path = os.path.join(outdir, "final_model.zip")
    model.save(final_path)
    print(f"[grid-teacher] saved final model -> {final_path}")
    print(f"[grid-teacher] best model (by eval) -> "
          f"{os.path.join(outdir, 'best_model.zip')}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
