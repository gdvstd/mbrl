"""Train a paper-track student on the four-room TARGET env (MaskablePPO).

Transfer modes (the paper's lineup):
    scratch | finetune (conv frozen, as in the paper; --no-freeze-cnn to
    unfreeze) | aa | jsrl | ksrl | ebtl

Budgets follow the paper: altgoal 200K, locked 1M. EBTL threshold quantile
defaults also follow the paper's figures: altgoal q=0.5, locked q=0.7.
The teacher defaults to the scenario's plain teacher; pass --teacher to use
the energy-regularized one for EBTL (paper Fig. 5 compares both).

Usage:
    python train_paper_student.py --scenario altgoal --transfer scratch
    python train_paper_student.py --scenario locked --transfer ebtl \
        --teacher runs/paper_locked_teacher_ereg/final_model.zip
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import os

from sb3_contrib import MaskablePPO

from fourroom.fourroom_env import SCENARIOS
from fourroom.paper_common import (
    N_ENVS,
    FrozenMaskableTeacher,
    build_mppo,
    calibrate_energy_threshold,
    make_paper_callbacks,
    make_paper_vec_env,
)

DEFAULT_TIMESTEPS = {"altgoal": 200_000, "locked": 1_000_000}
DEFAULT_EBTL_Q = {"altgoal": 0.5, "locked": 0.7}
DEFAULT_EVAL_FREQ = {"altgoal": 5_000, "locked": 20_000}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["altgoal", "locked"],
                        required=True)
    parser.add_argument(
        "--transfer",
        choices=["scratch", "finetune", "aa", "jsrl", "ksrl", "ebtl"],
        default="scratch")
    parser.add_argument("--teacher", type=str, default=None)
    parser.add_argument("--no-freeze-cnn", action="store_true",
                        help="finetune: do NOT freeze the conv towers")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--delta0", type=float, default=1.0)
    parser.add_argument("--end-frac", type=float, default=0.5)
    parser.add_argument("--jsrl-h0", type=int, default=50)
    parser.add_argument("--ksrl-lambda0", type=float, default=0.5)
    parser.add_argument("--ebtl-q", type=float, default=None)
    parser.add_argument("--ego", action="store_true",
                        help="11x11 egocentric partial obs (occluded)")
    args = parser.parse_args()

    src_env, tgt_env = SCENARIOS[args.scenario]
    timesteps = args.timesteps or DEFAULT_TIMESTEPS[args.scenario]
    eval_freq = args.eval_freq or DEFAULT_EVAL_FREQ[args.scenario]
    ebtl_q = args.ebtl_q if args.ebtl_q is not None else DEFAULT_EBTL_Q[args.scenario]
    ego_sfx = "_ego" if args.ego else ""
    teacher_path = args.teacher or (
        f"runs/paper_{args.scenario}_teacher{ego_sfx}/final_model.zip")
    outdir = args.outdir or (
        f"runs/paper_{args.scenario}{ego_sfx}_{args.transfer}_seed{args.seed}")
    os.makedirs(outdir, exist_ok=True)
    tb_dir = os.path.join(outdir, "tb")

    train_env = make_paper_vec_env(
        tgt_env, args.seed, monitor_dir=os.path.join(outdir, "train_monitor"),
        ego=args.ego)
    eval_env = make_paper_vec_env(
        tgt_env, args.seed + 1000, n_envs=1,
        monitor_dir=os.path.join(outdir, "eval", "eval_monitor"), ego=args.ego)

    if args.transfer == "finetune":
        model = MaskablePPO.load(
            teacher_path, env=train_env, device=args.device,
            tensorboard_log=tb_dir, seed=args.seed)
        if not args.no_freeze_cnn:
            frozen = 0
            for name in ("pi_features_extractor", "vf_features_extractor",
                         "features_extractor"):
                fe = getattr(model.policy, name, None)
                if fe is not None:
                    for p in fe.parameters():
                        p.requires_grad = False
                        frozen += 1
            print(f"[paper-student] froze conv towers ({frozen} tensors)")
        print(f"[paper-student] initialized from teacher: {teacher_path}")
    elif args.transfer in ("aa", "jsrl", "ebtl"):
        from shared.strategies import AAStrategy, EBTLStrategy, JSRLStrategy
        from fourroom.paper_mixed_ppo import MixedPolicyMaskablePPO

        teacher = FrozenMaskableTeacher(teacher_path, device=args.device)
        if args.transfer == "aa":
            strategy = AAStrategy(args.delta0, args.end_frac)
        elif args.transfer == "jsrl":
            strategy = JSRLStrategy(args.jsrl_h0, args.end_frac)
        else:
            tau = calibrate_energy_threshold(
                teacher, src_env, seed=args.seed + 2000, quantile=ebtl_q,
                device=args.device, ego=args.ego)
            print(f"[paper-student] EBTL tau={tau:.3f} (q={ebtl_q})")
            strategy = EBTLStrategy(tau, args.delta0, args.end_frac)
        model = build_mppo(
            train_env, args.seed, args.device, tb_dir,
            algo_cls=MixedPolicyMaskablePPO, teacher=teacher,
            strategy=strategy,
            advlog_path=os.path.join(outdir, "advlog.npz"))
        print(f"[paper-student] guidance teacher: {teacher_path}")
    elif args.transfer == "ksrl":
        from fourroom.paper_ksrl import KickstartMaskablePPO

        teacher = FrozenMaskableTeacher(teacher_path, device=args.device)
        model = build_mppo(
            train_env, args.seed, args.device, tb_dir,
            algo_cls=KickstartMaskablePPO, teacher=teacher,
            kick_lambda0=args.ksrl_lambda0, kick_decay_frac=args.end_frac)
        print(f"[paper-student] distillation teacher: {teacher_path}")
    else:
        model = build_mppo(train_env, args.seed, args.device, tb_dir)

    callbacks = make_paper_callbacks(
        eval_env, outdir,
        eval_freq=max(eval_freq // N_ENVS, 1),
        n_eval_episodes=args.n_eval_episodes,
        early_stop_threshold=None,  # fixed budget for curve comparison
    )

    print(f"[paper-student] scenario={args.scenario} env={tgt_env} "
          f"transfer={args.transfer} timesteps={timesteps}")
    print(f"[paper-student] outputs -> {os.path.abspath(outdir)}")
    model.learn(total_timesteps=timesteps, callback=callbacks,
                progress_bar=True)

    final_path = os.path.join(outdir, "final_model.zip")
    model.save(final_path)
    print(f"[paper-student] saved final model -> {final_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
