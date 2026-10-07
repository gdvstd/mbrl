"""Train the student with SAC on BipedalWalkerHardcore-v3 (Task B).

Defaults to `--transfer none` = from-scratch baseline (no teacher). Transfer
methods plug in via transfer.py without touching this script; run the same
command with a different `--transfer` to compare against the baseline under an
identical algorithm setup and step budget.

Usage (inside the venv):
    # from-scratch baseline
    python train_student.py --seed 0 --timesteps 1000000

    # transfer variants (all need a teacher)
    python train_student.py --transfer weight_init    --teacher runs/teacher_sac/best_model.zip
    python train_student.py --transfer jsrl           --teacher runs/teacher_sac/best_model.zip
    python train_student.py --transfer reward_shaping --teacher runs/teacher_sac/best_model.zip

Outputs mirror train_teacher.py (best/final model, checkpoints, eval/, tb/).
For a fair comparison, keep --timesteps, --seed, --eval-freq and
--n-eval-episodes identical across runs.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import os

from bipedal.sac_common import ENV_IDS, SAC_HYPERPARAMS, make_callbacks, make_env
from bipedal.transfer import TRANSFER_METHODS, get_transfer

STUDENT_ENV = ENV_IDS["hardcore"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--transfer", choices=TRANSFER_METHODS, default="none",
        help="transfer strategy (default: none = from-scratch baseline)",
    )
    parser.add_argument(
        "--teacher", type=str, default=None,
        help="path to teacher model, required by teacher-based transfer methods",
    )
    parser.add_argument(
        "--early-stop", action="store_true",
        help="stop when eval reward >= 300 (off by default so baseline and "
        "transfer runs cover the same fixed budget)",
    )
    # method-specific knobs
    parser.add_argument("--jsrl-max-horizon", type=int, default=100)
    parser.add_argument("--jsrl-n-stages", type=int, default=10)
    parser.add_argument("--jsrl-tolerance", type=float, default=0.05)
    parser.add_argument("--shaping-beta", type=float, default=1.0)
    parser.add_argument("--ksrl-lambda0", type=float, default=1.0)
    parser.add_argument(
        "--ksrl-decay-frac", type=float, default=0.5,
        help="anneal lambda_k to 0 over this fraction of total timesteps",
    )
    args = parser.parse_args()

    outdir = args.outdir or f"runs/student_{args.transfer}_seed{args.seed}"
    os.makedirs(outdir, exist_ok=True)
    tb_dir = os.path.join(outdir, "tb")

    strategy = get_transfer(
        args.transfer,
        args.teacher,
        jsrl_max_horizon=args.jsrl_max_horizon,
        jsrl_n_stages=args.jsrl_n_stages,
        jsrl_tolerance=args.jsrl_tolerance,
        gamma=SAC_HYPERPARAMS["gamma"],
        shaping_beta=args.shaping_beta,
        ksrl_lambda0=args.ksrl_lambda0,
        ksrl_decay_steps=int(args.ksrl_decay_frac * args.timesteps),
    )

    # training env may be wrapped by the strategy; eval env stays pure so eval
    # always measures the standalone student (comparable across methods).
    train_env = make_env(
        STUDENT_ENV, args.seed, os.path.join(outdir, "train_monitor"),
        base_wrapper=strategy.wrap_base_env,
    )
    eval_env = make_env(
        STUDENT_ENV, args.seed + 1000,
        os.path.join(outdir, "eval", "eval_monitor"),
    )

    model = strategy.build_model(train_env, args.seed, args.device, tb_dir)
    model = strategy.modify_model(model)  # one-shot transfer (no-op for "none")

    callbacks = make_callbacks(
        eval_env,
        outdir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        early_stop_threshold=300.0 if args.early_stop else None,
        after_eval_callback=strategy.after_eval_callback(),
    )

    print(f"[student] env={STUDENT_ENV} transfer={args.transfer} "
          f"timesteps={args.timesteps} seed={args.seed} device={args.device}")
    print(f"[student] outputs -> {os.path.abspath(outdir)}")

    model.learn(
        total_timesteps=args.timesteps, callback=callbacks, progress_bar=True
    )

    final_path = os.path.join(outdir, "final_model.zip")
    model.save(final_path)
    print(f"[student] saved final model -> {final_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
