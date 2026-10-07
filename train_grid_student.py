"""Train the MiniGrid student on MiniGrid-DoorKey-8x8-v0 (Task B).

Six modes, compared under an identical step budget (EBTL paper's lineup):
  * scratch   -- PPO from random init (the "No Transfer" baseline).
  * finetune  -- PPO initialized from the Empty teacher's weights.
                 --freeze-cnn additionally freezes the conv feature extractor,
                 matching the Fine-Tuning baseline in the EBTL paper.
  * aa        -- Action Advising: teacher acts with decaying probability.
  * jsrl      -- JumpStart RL: teacher acts for the first h(t) episode steps.
  * ksrl      -- Kickstarting: decaying cross-entropy distillation loss.
  * ebtl      -- EBTL: teacher acts only in states it deems in-distribution
                 (energy score >= tau) with decaying probability.

DoorKey's reward is sparse (only on reaching the goal behind the locked
door), so the scratch baseline needs substantial exploration -- expect slow
initial progress; that is the point of the comparison.

Usage (inside the venv):
    python train_grid_student.py --transfer scratch
    python train_grid_student.py --transfer finetune \
        --teacher runs/grid_teacher_ppo/best_model.zip

Outputs (under --outdir): best_model.zip / final_model.zip / checkpoints/ /
eval/ / tb/ / monitor csvs
"""
from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO

from grid_common import (
    N_ENVS,
    build_ppo,
    grid_env_id,
    make_grid_vec_env,
)
from sac_common import make_callbacks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size", type=int, default=6, choices=[5, 6, 8, 16],
        help="grid side length (6 recommended; 8x8 DoorKey is beyond vanilla "
             "PPO within 1M steps -- see grid_common.grid_env_id)",
    )
    parser.add_argument(
        "--transfer",
        choices=["scratch", "finetune", "aa", "jsrl", "ksrl", "ebtl"],
        default="scratch",
    )
    parser.add_argument(
        "--teacher", type=str, default=None,
        help="teacher checkpoint (finetune mode only); defaults to the "
             "same-size teacher's best_model.zip",
    )
    parser.add_argument(
        "--freeze-cnn", action="store_true",
        help="freeze the conv feature extractor (paper's Fine-Tuning variant)",
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-freq", type=int, default=20_000,
        help="evaluate every N total env steps (across all vec workers)",
    )
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--ent-coef", type=float, default=None,
        help="override the shared entropy coefficient (scratch mode only)",
    )
    # -- guidance methods (aa / jsrl / ebtl) --
    parser.add_argument(
        "--delta0", type=float, default=1.0,
        help="initial guidance probability (aa/ebtl)",
    )
    parser.add_argument(
        "--end-frac", type=float, default=0.5,
        help="fraction of the budget by which guidance decays to 0 "
             "(aa/ebtl prob, jsrl horizon, ksrl lambda)",
    )
    parser.add_argument(
        "--jsrl-h0", type=int, default=50,
        help="initial JSRL guide horizon (env steps per episode)",
    )
    parser.add_argument(
        "--ksrl-lambda0", type=float, default=0.5,
        help="initial KSRL distillation weight",
    )
    parser.add_argument(
        "--ebtl-q", type=float, default=0.5,
        help="EBTL energy-threshold quantile over teacher source states",
    )
    args = parser.parse_args()

    student_env = grid_env_id("doorkey", args.size)
    teacher_path = args.teacher or (
        f"runs/grid_teacher_ppo_{args.size}x{args.size}/best_model.zip"
    )
    outdir = args.outdir or (
        f"runs/grid_student{args.size}_{args.transfer}_seed{args.seed}"
    )
    os.makedirs(outdir, exist_ok=True)
    tb_dir = os.path.join(outdir, "tb")

    train_env = make_grid_vec_env(
        student_env, args.seed, monitor_dir=os.path.join(outdir, "train_monitor")
    )
    eval_env = make_grid_vec_env(
        student_env, args.seed + 1000, n_envs=1,
        monitor_dir=os.path.join(outdir, "eval", "eval_monitor"),
    )

    overrides = {} if args.ent_coef is None else {"ent_coef": args.ent_coef}

    if args.transfer == "finetune":
        # Same obs/action spaces on both tasks (7x7 egocentric crop), so the
        # teacher's weights load directly; only the env (and seed) change.
        model = PPO.load(
            teacher_path, env=train_env, device=args.device,
            tensorboard_log=tb_dir, seed=args.seed,
        )
        if args.freeze_cnn:
            for p in model.policy.features_extractor.parameters():
                p.requires_grad = False
            print("[grid-student] froze CNN feature extractor")
        print(f"[grid-student] initialized from teacher: {teacher_path}")
    elif args.transfer in ("aa", "jsrl", "ebtl"):
        from grid_mixed_ppo import MixedPolicyPPO
        from grid_strategies import AAStrategy, EBTLStrategy, JSRLStrategy
        from grid_teacher import FrozenTeacher, calibrate_energy_threshold

        teacher = FrozenTeacher(teacher_path, device=args.device)
        if args.transfer == "aa":
            strategy = AAStrategy(args.delta0, args.end_frac)
        elif args.transfer == "jsrl":
            strategy = JSRLStrategy(args.jsrl_h0, args.end_frac)
        else:
            tau = calibrate_energy_threshold(
                teacher, grid_env_id("empty", args.size),
                seed=args.seed + 2000, quantile=args.ebtl_q,
                device=args.device,
            )
            print(f"[grid-student] EBTL tau={tau:.3f} (q={args.ebtl_q})")
            strategy = EBTLStrategy(tau, args.delta0, args.end_frac)
        model = build_ppo(
            train_env, args.seed, args.device, tb_dir,
            algo_cls=MixedPolicyPPO, teacher=teacher, strategy=strategy,
            **overrides,
        )
        print(f"[grid-student] guidance teacher: {teacher_path}")
    elif args.transfer == "ksrl":
        from grid_ksrl import KickstartPPO
        from grid_teacher import FrozenTeacher

        teacher = FrozenTeacher(teacher_path, device=args.device)
        model = build_ppo(
            train_env, args.seed, args.device, tb_dir,
            algo_cls=KickstartPPO, teacher=teacher,
            kick_lambda0=args.ksrl_lambda0, kick_decay_frac=args.end_frac,
            **overrides,
        )
        print(f"[grid-student] distillation teacher: {teacher_path}")
    else:
        model = build_ppo(train_env, args.seed, args.device, tb_dir, **overrides)

    callbacks = make_callbacks(
        eval_env,
        outdir,
        # EvalCallback counts per-vec-step, each of which is N_ENVS frames.
        eval_freq=max(args.eval_freq // N_ENVS, 1),
        n_eval_episodes=args.n_eval_episodes,
        early_stop_threshold=None,  # fixed budget: baseline-vs-transfer curves
        name_prefix="ppo",
    )

    print(f"[grid-student] env={student_env} transfer={args.transfer} "
          f"timesteps={args.timesteps} device={args.device}")
    print(f"[grid-student] outputs -> {os.path.abspath(outdir)}")
    model.learn(
        total_timesteps=args.timesteps, callback=callbacks, progress_bar=True
    )

    final_path = os.path.join(outdir, "final_model.zip")
    model.save(final_path)
    print(f"[grid-student] saved final model -> {final_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
