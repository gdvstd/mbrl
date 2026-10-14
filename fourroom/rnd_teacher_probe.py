"""Causal test of the altgoal phi inversion: does a goal-READING teacher
fix the energy gate?

Hypothesis (from the sae_probe depth result): the plain altgoal teacher
shortcuts on position (goal always in Room 1 -> the goal channel is
ignored by the policy head), so phi stays high on goal-in-Room-3 states
and the gate inverts. A teacher trained with the goal randomized over
rooms 1/2/4 (room 3 stays OOD) must read the goal channel; its phi should
then drop on Room-3 episodes and the gate should stop inverting.

For each teacher (plain altgoal vs altgoal_rnd), on the SAME target
rollouts (EBTL student seed0):
  - source sanity: teacher success on its own source env
  - oracle advice labels (helpful = argmax advice optimal)
  - AUROC(phi; helpful), AUROC(phi; goal-room condition),
    P(helpful | open/closed) at tau = Quantile_0.5(source phi) + z-test

Usage: python fourroom/rnd_teacher_probe.py
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import torch as th

from fourroom.fourroom_env import SCENARIOS, action_mask
from fourroom.oracle_advice import _step, collect_oracle, teacher_logits
from fourroom.paper_common import FrozenMaskableTeacher, _wrap
from fourroom.sae_probe import N_FRAMES, auroc, student_actor, teacher_actor

TGT = "FourRoomAltGoalTgt-v0"
Q = 0.5


def source_success(teacher, env_id: str, n_ep: int = 20) -> float:
    env = _wrap(gym.make(env_id), ego=False)
    act = teacher_actor(teacher)
    succ = 0
    for ep in range(n_ep):
        obs, _ = env.reset(seed=500 + ep)
        for _ in range(100):
            obs, r, te, tr, _ = env.step(act(obs, action_mask(env)))
            if te or tr:
                succ += r > 0
                break
    env.close()
    return succ / n_ep


def source_phi(teacher, env_id: str) -> np.ndarray:
    from fourroom.sae_probe import collect
    src_obs, _ = collect(env_id, teacher_actor(teacher), N_FRAMES, 11,
                         "goal", False)
    return th.logsumexp(teacher_logits(teacher, src_obs), -1).numpy()


def probe(teacher_path: str, src_env_id: str, label: str,
          tgt_data) -> None:
    tgt_obs, tgt_masks, states, lay_keys, layouts = tgt_data
    teacher = FrozenMaskableTeacher(teacher_path)
    print(f"\n== {label} ==")
    print(f"source success: {source_success(teacher, src_env_id):.2f}")

    logits = teacher_logits(teacher, tgt_obs)
    masked = logits.masked_fill(~th.as_tensor(tgt_masks), -1e9)
    adv_a = masked.argmax(-1).numpy()
    phi = th.logsumexp(logits, -1).numpy()

    helpful = np.zeros(len(states), dtype=bool)
    keep = np.zeros(len(states), dtype=bool)
    room3 = np.zeros(len(states), dtype=bool)
    for i, (s, lk) in enumerate(zip(states, lay_keys)):
        lay, D = layouts[lk]
        d0, d1 = D.get(s), D.get(_step(lay, s, int(adv_a[i])))
        if d0 is None or d0 == 0 or d1 is None:
            continue
        keep[i] = True
        helpful[i] = (d0 - d1) == 1
        room3[i] = lay.goal[0] >= 5 and lay.goal[1] >= 5
    phi, helpful, room3 = phi[keep], helpful[keep], room3[keep]

    tau = float(np.quantile(source_phi(teacher, src_env_id), Q))
    open_ = phi >= tau
    a_help = auroc(phi[helpful], phi[~helpful])
    a_cond = auroc(phi[~room3], phi[room3])  # >0.5: room1(ID) scores higher
    p1 = helpful[open_].mean() if open_.any() else float("nan")
    p2 = helpful[~open_].mean() if (~open_).any() else float("nan")
    n1, n2 = open_.sum(), (~open_).sum()
    p = helpful.mean()
    se = np.sqrt(p * (1 - p) * (1 / max(n1, 1) + 1 / max(n2, 1)))
    z = (p1 - p2) / se if se > 0 else float("nan")
    print(f"AUROC(phi; helpful):        {a_help:.3f}")
    print(f"AUROC(phi; room1 vs room3): {a_cond:.3f}")
    print(f"P(helpful|open) {p1:.3f} vs closed {p2:.3f}  "
          f"diff {p1 - p2:+.3f}  z {z:+.1f}  (open {open_.mean():.0%})")


def main() -> None:
    from sb3_contrib import MaskablePPO
    student = MaskablePPO.load("runs/paper_altgoal_ebtl_seed0/best_model.zip",
                               device="cpu")
    tgt_data = collect_oracle(TGT, student_actor(student), N_FRAMES, 22,
                              False)
    probe("runs/paper_altgoal_teacher/final_model.zip",
          SCENARIOS["altgoal"][0], "plain teacher (goal fixed to Room 1)",
          tgt_data)
    probe("runs/paper_altgoal_rnd_teacher/final_model.zip",
          SCENARIOS["altgoal_rnd"][0], "rnd teacher (goal in rooms 1/2/4)",
          tgt_data)
    print("\nRND PROBE DONE")


if __name__ == "__main__":
    main()
