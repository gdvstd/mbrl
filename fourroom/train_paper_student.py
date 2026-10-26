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
        choices=["scratch", "finetune", "aa", "jsrl", "ksrl", "ebtl", "gap",
                 "sae", "saerate", "saeclip", "said"],
        default="scratch")
    parser.add_argument("--teacher", type=str, default=None)
    parser.add_argument("--no-freeze-cnn", action="store_true",
                        help="finetune: do NOT freeze the conv towers")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--n-eval-episodes", type=int, default=100)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--delta0", type=float, default=1.0)
    parser.add_argument("--end-frac", type=float, default=0.5)
    parser.add_argument("--jsrl-h0", type=int, default=100)
    parser.add_argument("--ksrl-lambda0", type=float, default=1.0)
    parser.add_argument("--ebtl-q", type=float, default=None)
    parser.add_argument("--gap-eta", type=float, default=0.5,
                        help="gap: min normalized student entropy to advise")
    parser.add_argument("--sae-layer", choices=["L1", "L2", "L3"],
                        default="L1", help="sae: which activation layer")
    parser.add_argument("--sae-rate", type=float, default=0.05,
                        help="saerate: target gate-open fraction (running "
                             "quantile of recent target recon errors)")
    parser.add_argument("--sae-tau-mult", type=float, default=2.0,
                        help="sae/saeclip: gap-rule multiplier on the "
                             "p99.9 calibration recon error")
    parser.add_argument("--sae-rate-lo", type=float, default=0.01,
                        help="saeclip: min gate-open fraction guardrail")
    parser.add_argument("--sae-rate-hi", type=float, default=0.10,
                        help="saeclip: max gate-open fraction guardrail")
    parser.add_argument("--gap-no-coin", action="store_true",
                        help="gap: drop the delta(t) coin (the entropy "
                             "condition self-anneals)")
    parser.add_argument("--oracle-teacher", action="store_true",
                        help="aa/jsrl/ksrl: replace the trained teacher "
                             "with the BFS-optimal oracle policy "
                             "(upper-bound control; full-obs only)")
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
    elif args.transfer in ("aa", "jsrl", "ebtl", "gap", "sae", "saerate",
                           "saeclip", "said"):
        from shared.strategies import (
            AAStrategy, ClipGateStrategy, EBTLStrategy, GapStrategy,
            JSRLStrategy, RateGateStrategy, ScoreGateStrategy)
        from fourroom.paper_mixed_ppo import MixedPolicyMaskablePPO

        if args.oracle_teacher:
            assert args.transfer in ("aa", "jsrl") and not args.ego, \
                "--oracle-teacher: aa/jsrl/ksrl, full-obs only"
            from fourroom.oracle_teacher import OracleTeacher
            teacher = OracleTeacher(args.scenario)
            print("[paper-student] using ORACLE teacher (BFS-optimal)")
        else:
            teacher = FrozenMaskableTeacher(teacher_path, device=args.device)
        if args.transfer == "aa":
            strategy = AAStrategy(args.delta0, args.end_frac)
        elif args.transfer == "jsrl":
            strategy = JSRLStrategy(args.jsrl_h0, args.end_frac)
        elif args.transfer == "sae":
            from fourroom.sae_gate import build_sae_gate
            score_fn, tau_r = build_sae_gate(
                teacher, src_env, ebtl_q, args.ego, seed=args.seed + 3000,
                layer=args.sae_layer, tau_mult=args.sae_tau_mult)
            print(f"[paper-student] sae gate [{args.sae_layer}] "
                  f"tau_r={tau_r:.4f} (q={ebtl_q})")
            strategy = ScoreGateStrategy(score_fn, tau_r, higher_is_id=False,
                                         delta0=args.delta0,
                                         end_frac=args.end_frac)
        elif args.transfer == "saeclip":
            from fourroom.sae_gate import build_sae_gate
            score_fn, tau_r = build_sae_gate(
                teacher, src_env, ebtl_q, args.ego, seed=args.seed + 3000,
                layer=args.sae_layer, tau_mult=args.sae_tau_mult)
            print(f"[paper-student] CLIPPED sae gate [{args.sae_layer}] "
                  f"tau_r={tau_r:.4f} guardrails "
                  f"[{args.sae_rate_lo}, {args.sae_rate_hi}]")
            strategy = ClipGateStrategy(score_fn, init_tau=tau_r,
                                        rate_lo=args.sae_rate_lo,
                                        rate_hi=args.sae_rate_hi,
                                        delta0=args.delta0,
                                        end_frac=args.end_frac)
        elif args.transfer == "saerate":
            from fourroom.sae_gate import build_sae_gate
            score_fn, tau_r = build_sae_gate(
                teacher, src_env, ebtl_q, args.ego, seed=args.seed + 3000,
                layer=args.sae_layer)
            print(f"[paper-student] RATE-targeted sae gate "
                  f"[{args.sae_layer}] rate={args.sae_rate} "
                  f"init_tau={tau_r:.4f}")
            strategy = RateGateStrategy(score_fn, args.sae_rate,
                                        init_tau=tau_r, higher_is_id=False,
                                        delta0=args.delta0,
                                        end_frac=args.end_frac)
        elif args.transfer == "said":
            from fourroom.sae_gate import build_said_gate
            score_fn, tau_c = build_said_gate(
                teacher, src_env, ebtl_q, args.ego, seed=args.seed + 3000)
            print(f"[paper-student] said gate tau_c={tau_c:.4f} (q={ebtl_q})")
            strategy = ScoreGateStrategy(score_fn, tau_c, higher_is_id=True,
                                         delta0=args.delta0,
                                         end_frac=args.end_frac)
        else:
            tau = calibrate_energy_threshold(
                teacher, src_env, seed=args.seed + 2000, quantile=ebtl_q,
                device=args.device, ego=args.ego)
            print(f"[paper-student] {args.transfer} tau={tau:.3f} "
                  f"(q={ebtl_q})")
            if args.transfer == "gap":
                strategy = GapStrategy(tau, args.gap_eta, args.delta0,
                                       args.end_frac,
                                       use_coin=not args.gap_no_coin)
            else:
                strategy = EBTLStrategy(tau, args.delta0, args.end_frac)
        model = build_mppo(
            train_env, args.seed, args.device, tb_dir,
            algo_cls=MixedPolicyMaskablePPO, teacher=teacher,
            strategy=strategy,
            advlog_path=os.path.join(outdir, "advlog.npz"),
            advlog_scenario=args.scenario)
        if args.transfer == "gap":
            strategy.student_policy = model.policy
        print(f"[paper-student] guidance teacher: {teacher_path}")
    elif args.transfer == "ksrl":
        from fourroom.paper_ksrl import KickstartMaskablePPO

        if args.oracle_teacher:
            assert not args.ego, "--oracle-teacher: full-obs only"
            from fourroom.oracle_teacher import OracleTeacher
            teacher = OracleTeacher(args.scenario)
            print("[paper-student] using ORACLE teacher (BFS-optimal)")
        else:
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
