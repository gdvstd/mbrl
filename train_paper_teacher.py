"""Train the paper-track teacher (MaskablePPO on the four-room SOURCE env).

Scenarios (see fourroom_env): altgoal (200K budget, the paper's checkpoint)
and locked (800K). Trains the full budget -- the paper selects fixed-step
checkpoints rather than early-stopping. --energy-reg adds the paper's margin
energy loss (Appendix A.1: margins (10, 15) over phi; ID = recent 3000
frames; OOD = 100 masked-random episodes in the TARGET env, any-room start).

Usage:
    python train_paper_teacher.py --scenario altgoal
    python train_paper_teacher.py --scenario locked --energy-reg
"""
from __future__ import annotations

import argparse
import os

from fourroom_env import SCENARIOS
from paper_common import (
    N_ENVS,
    build_mppo,
    make_paper_callbacks,
    make_paper_vec_env,
)

DEFAULT_TIMESTEPS = {"altgoal": 200_000, "locked": 800_000}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["altgoal", "locked"],
                        required=True)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--energy-reg", action="store_true")
    parser.add_argument("--ego", action="store_true",
                        help="11x11 egocentric partial obs (occluded) instead of full grid")
    parser.add_argument("--energy-lambda", type=float, default=0.1)
    parser.add_argument("--m-in", type=float, default=10.0)
    parser.add_argument("--m-out", type=float, default=15.0)
    args = parser.parse_args()

    src_env, tgt_env = SCENARIOS[args.scenario]
    timesteps = args.timesteps or DEFAULT_TIMESTEPS[args.scenario]
    outdir = args.outdir or (
        f"runs/paper_{args.scenario}_teacher"
        + ("_ereg" if args.energy_reg else "") + ("_ego" if args.ego else ""))
    os.makedirs(outdir, exist_ok=True)
    tb_dir = os.path.join(outdir, "tb")

    train_env = make_paper_vec_env(
        src_env, args.seed, monitor_dir=os.path.join(outdir, "train_monitor"),
        ego=args.ego)
    eval_env = make_paper_vec_env(
        src_env, args.seed + 1000, n_envs=1,
        monitor_dir=os.path.join(outdir, "eval", "eval_monitor"), ego=args.ego)

    if args.energy_reg:
        from paper_energy_reg import (
            EnergyRegMaskablePPO,
            collect_random_states_masked,
        )

        kw = {"agent_rooms": (1, 2, 3, 4)} if args.scenario == "locked" else {}
        print(f"[paper-teacher] collecting OOD states: 100 random eps in {tgt_env}")
        ood = collect_random_states_masked(
            tgt_env, n_episodes=100, seed=args.seed + 3000, env_kwargs=kw,
            ego=args.ego)
        model = build_mppo(
            train_env, args.seed, args.device, tb_dir,
            algo_cls=EnergyRegMaskablePPO, ood_states=ood,
            energy_lambda=args.energy_lambda, m_in=args.m_in, m_out=args.m_out)
        print(f"[paper-teacher] energy reg: lambda={args.energy_lambda} "
              f"margins=({args.m_in},{args.m_out}) ood_n={len(ood)}")
    else:
        model = build_mppo(train_env, args.seed, args.device, tb_dir)

    callbacks = make_paper_callbacks(
        eval_env, outdir,
        eval_freq=max(args.eval_freq // N_ENVS, 1),
        n_eval_episodes=args.n_eval_episodes,
        early_stop_threshold=None,  # paper uses fixed-step checkpoints
    )

    print(f"[paper-teacher] scenario={args.scenario} env={src_env} "
          f"timesteps={timesteps}")
    print(f"[paper-teacher] outputs -> {os.path.abspath(outdir)}")
    model.learn(total_timesteps=timesteps, callback=callbacks,
                progress_bar=True)

    final_path = os.path.join(outdir, "final_model.zip")
    model.save(final_path)
    print(f"[paper-teacher] saved final model -> {final_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
