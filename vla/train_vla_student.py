"""Train a LIBERO student on the task-mix target (sparse success reward).

Transfer modes: scratch | finetune | aa | jsrl | ksrl  (EBTL comes later,
with a token-based policy). Teacher = runs_vla/teacher_bc.pt by default.

Local runs are SMOKE-scale only (rendering-bound, ~15-30 fps); full runs
belong on the lab server. --tasks defaults to a 2-task mini target
(4=ketchup seen, 7=milk unseen); use 0..9 for the full target.

Usage:
    python train_vla_student.py --transfer scratch --timesteps 2000
    python train_vla_student.py --transfer aa --tasks 0 1 2 3 4 5 6 7 8 9
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vla_ppo_common import (
    FrozenBCTeacher,
    build_vla_ppo,
    load_teacher_into,
    make_vla_vec_env,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transfer",
                   choices=["scratch", "finetune", "aa", "jsrl", "ksrl",
                            "bcrl"],
                   default="scratch")
    p.add_argument("--init-from", type=str, default=None,
                   help="bcrl: PPO zip to initialize from (BC-distilled)")
    p.add_argument("--teacher", type=str, default="runs_vla/teacher_bc.pt")
    p.add_argument("--openvla-lora", type=str, default=None,
                   help="aa/jsrl: use the OpenVLA-LoRA teacher at this dir "
                        "instead of the local BC teacher")
    p.add_argument("--norm-stats", type=str, default=None,
                   help="norm_stats.json for the OpenVLA teacher "
                        "(default: <lora>/../norm_stats.json)")
    p.add_argument("--tasks", type=int, nargs="*", default=[4, 7])
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--delta0", type=float, default=1.0)
    p.add_argument("--end-frac", type=float, default=0.5)
    p.add_argument("--jsrl-h0", type=int, default=80)
    p.add_argument("--ksrl-lambda0", type=float, default=0.5)
    p.add_argument("--outdir", type=str, default=None)
    a = p.parse_args()

    outdir = a.outdir or f"runs_vla/student_{a.transfer}_seed{a.seed}"
    os.makedirs(outdir, exist_ok=True)
    env = make_vla_vec_env(a.tasks, a.seed, n_envs=a.n_envs,
                           monitor_dir=os.path.join(outdir, "monitor"))

    if a.transfer == "bcrl":
        import numpy as np
        import torch as th
        from stable_baselines3 import PPO
        assert a.init_from, "--transfer bcrl requires --init-from"
        # diagnosis (evalprobe 11953035): BC leaves std~0.26, which kills
        # manipulation precision under stochastic collection (16% -> 2%),
        # and zero-reward PPO updates then destroy the BC mean. Start the
        # finetune tight and conservative instead.
        model = PPO.load(a.init_from, env=env, device=a.device,
                         tensorboard_log=os.path.join(outdir, "tb"),
                         seed=a.seed, ent_coef=0.0, learning_rate=1e-4)
        with th.no_grad():
            model.policy.log_std.fill_(float(np.log(0.1)))
        print(f"[vla-student] RL finetune from BC-distilled: {a.init_from} "
              f"(std reset to 0.1, ent_coef 0, lr 1e-4)")
    elif a.transfer == "finetune":
        model = build_vla_ppo(env, a.seed, a.device,
                              tb_dir=os.path.join(outdir, "tb"))
        load_teacher_into(model, a.teacher)
        print(f"[vla-student] initialized from BC teacher: {a.teacher}")
    elif a.transfer in ("aa", "jsrl"):
        from strategies import AAStrategy, JSRLStrategy
        from vla_mixed_ppo import MixedPolicyVLAPPO

        if a.openvla_lora:
            from openvla_teacher import OpenVLATeacher
            norm = a.norm_stats or os.path.join(
                os.path.dirname(a.openvla_lora.rstrip("/")),
                "norm_stats.json")
            teacher = OpenVLATeacher(a.openvla_lora, norm, device="cuda")
            print(f"[vla-student] OpenVLA teacher: {a.openvla_lora}")
        else:
            teacher = FrozenBCTeacher(a.teacher, device=a.device)
            print(f"[vla-student] guidance teacher: {a.teacher}")
        strategy = (AAStrategy(a.delta0, a.end_frac) if a.transfer == "aa"
                    else JSRLStrategy(a.jsrl_h0, a.end_frac))
        model = build_vla_ppo(env, a.seed, a.device,
                              tb_dir=os.path.join(outdir, "tb"),
                              algo_cls=MixedPolicyVLAPPO,
                              teacher=teacher, strategy=strategy)
        if a.openvla_lora:
            teacher.attach(env)
    elif a.transfer == "ksrl":
        from vla_ksrl import KickstartVLAPPO

        teacher = FrozenBCTeacher(a.teacher, device=a.device)
        model = build_vla_ppo(env, a.seed, a.device,
                              tb_dir=os.path.join(outdir, "tb"),
                              algo_cls=KickstartVLAPPO, teacher=teacher,
                              kick_lambda0=a.ksrl_lambda0,
                              kick_decay_frac=a.end_frac)
        print(f"[vla-student] distillation teacher: {a.teacher}")
    else:
        model = build_vla_ppo(env, a.seed, a.device,
                              tb_dir=os.path.join(outdir, "tb"))

    print(f"[vla-student] transfer={a.transfer} tasks={a.tasks} "
          f"timesteps={a.timesteps}")
    model.learn(total_timesteps=a.timesteps, progress_bar=True)
    model.save(os.path.join(outdir, "final_model.zip"))
    print(f"[vla-student] saved -> {outdir}/final_model.zip")
    env.close()


if __name__ == "__main__":
    main()
