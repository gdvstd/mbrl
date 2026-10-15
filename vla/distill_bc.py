"""Distill the OpenVLA teacher into the small PPO student by BC.

Trains the SB3 PPO policy's actor (BCBackboneExtractor + Gaussian head) on
teacher-rollout (obs, action) pairs by maximizing log-prob, then saves a
standard PPO zip usable as RL-finetune init (train_vla_student --transfer
bcrl --init-from <zip>). Also evaluates the distilled policy on all 10
tasks (deterministic, fixed init states) -- that eval IS the distillation
result: expect decent seen / ~0 unseen.

Usage:
    python distill_bc.py --data runs_vla/teacher_rollouts.npz \
        --outdir runs_vla/bc_distill
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vla_common import (  # noqa: E402
    MAX_STEPS, SEEN_TASKS, UNSEEN_TASKS, get_init_states, make_env,
    preprocess_obs,
)
from vla_ppo_common import build_vla_ppo, make_vla_vec_env  # noqa: E402


def bc_train(model, data, epochs: int, bs: int, device: str) -> None:
    pol = model.policy
    pol.set_training_mode(True)
    opt = th.optim.Adam(pol.parameters(), lr=3e-4)
    n = len(data["actions"])
    imgs = th.as_tensor(data["images"])
    props = th.as_tensor(data["proprios"])
    acts = th.as_tensor(data["actions"])
    for ep in range(epochs):
        perm = th.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            obs = {"image": imgs[idx].to(device),
                   "proprio": props[idx].to(device)}
            dist = pol.get_distribution(obs)
            loss = -dist.log_prob(acts[idx].to(device)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        print(f"[bc] epoch {ep + 1}/{epochs} nll {tot / nb:.3f}", flush=True)
    pol.set_training_mode(False)


@th.no_grad()
def evaluate(model, episodes: int = 10) -> None:
    for tid in list(SEEN_TASKS) + list(UNSEEN_TASKS):
        env = make_env(tid)
        inits = get_init_states(tid)
        succ = 0
        for ep in range(episodes):
            env.reset()
            raw = env.set_init_state(inits[ep % len(inits)])
            r = 0.0
            for _ in range(MAX_STEPS):
                img, prop = preprocess_obs(raw, tid)
                obs = {"image": (img * 255).astype(np.uint8),
                       "proprio": prop.astype(np.float32)}
                action, _ = model.predict(obs, deterministic=True)
                raw, r, done, _ = env.step(np.asarray(action, np.float64))
                if done:
                    break
            succ += r > 0
        tag = "seen" if tid in SEEN_TASKS else "UNSEEN"
        print(f"[bc-eval] task {tid} ({tag}): {succ}/{episodes}", flush=True)
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="runs_vla/bc_distill")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-episodes", type=int, default=10)
    a = ap.parse_args()

    data = np.load(a.data)
    print(f"[bc] {len(data['actions'])} frames, "
          f"{len(np.unique(data['ep_id']))} episodes")
    env = make_vla_vec_env([SEEN_TASKS[0]], seed=0, n_envs=1)
    # SEPARATE pi/vf extractors: with the default shared trunk, the
    # untrained value head's gradients destroy the BC actor within the
    # first PPO update (bcrl/bcrl2 post-mortem, evalprobe 11955533).
    from vla_ppo_common import PPO_HYPERPARAMS
    pk = dict(PPO_HYPERPARAMS["policy_kwargs"],
              share_features_extractor=False)
    model = build_vla_ppo(env, seed=0, device=a.device, policy_kwargs=pk)
    bc_train(model, data, a.epochs, a.batch_size, a.device)
    # give the value branch the trained features as a starting point --
    # from here on its gradients stay on its own copy.
    model.policy.vf_features_extractor.load_state_dict(
        model.policy.pi_features_extractor.state_dict())

    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "bc_model.zip")
    model.save(out)
    print(f"[bc] saved -> {out}")
    env.close()
    evaluate(model, a.eval_episodes)
    print("DISTILL DONE")


if __name__ == "__main__":
    main()
