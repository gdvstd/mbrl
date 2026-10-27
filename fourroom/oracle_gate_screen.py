"""Oracle pre-screen for advice-gate designs (no RL, minutes on CPU).

Per frame of target-task rollouts, compute:
  gain(s) = E_{a~pi_S}[regret(a)] - regret(a_T)   (steps the teacher's
            action saves vs what the student would have done; >0 helpful)
using the BFS oracle, plus the gate signals: teacher-SAE recon error,
phi, student masked entropy, student prob of the teacher action.

Candidate gates are compared at MATCHED advice budgets (top-b fraction by
each gate's ranking), on two visitation distributions:
  random  -- masked-uniform rollouts (start-of-training student)
  student -- trained saeks student rollouts (late training)
NEED signals always come from the trained student (no mid checkpoints).

Usage: python fourroom/oracle_gate_screen.py
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import torch as th

from fourroom.fourroom_env import SCENARIOS
from fourroom.oracle_advice import collect_oracle, regrets
from fourroom.paper_common import FrozenMaskableTeacher
from fourroom.sae_gate import build_sae_gate

N_FRAMES = 6000
BUDGETS = (0.02, 0.05, 0.10)
BATCH = 512


def masked_probs(policy, obs, masks):
    out_p, out_h = [], []
    for i in range(0, len(obs), BATCH):
        ot, _ = policy.obs_to_tensor(obs[i:i + BATCH])
        with th.no_grad():
            dist = policy.get_distribution(
                ot, action_masks=masks[i:i + BATCH])
            out_p.append(dist.distribution.probs.cpu().numpy())
            out_h.append(dist.entropy().cpu().numpy())
    return np.concatenate(out_p), np.concatenate(out_h)


def teacher_probs(teacher, obs, masks):
    out = []
    for i in range(0, len(obs), BATCH):
        ot, _ = teacher.policy.obs_to_tensor(obs[i:i + BATCH])
        with th.no_grad():
            d = teacher.distribution(ot, action_masks=masks[i:i + BATCH])
            out.append(d.distribution.probs.cpu().numpy())
    return np.concatenate(out)


def recon_scores(score_fn, policy, obs):
    out = []
    for i in range(0, len(obs), BATCH):
        ot, _ = policy.obs_to_tensor(obs[i:i + BATCH])
        out.append(score_fn(ot).cpu().numpy())
    return np.concatenate(out)


def phi_scores(teacher, obs):
    out = []
    for i in range(0, len(obs), BATCH):
        ot, _ = teacher.policy.obs_to_tensor(obs[i:i + BATCH])
        with th.no_grad():
            out.append(teacher.energy_score(ot).cpu().numpy())
    return np.concatenate(out)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx*ry).sum()/den) if den > 0 else float("nan")


def main() -> None:
    from sb3_contrib import MaskablePPO
    rng = np.random.default_rng(0)
    results = {}
    for sc in ("altgoal", "locked"):
        src_env, tgt_env = SCENARIOS[sc]
        teacher = FrozenMaskableTeacher(
            f"runs/paperv4_{sc}_teacher/final_model.zip")
        student = MaskablePPO.load(
            f"runs/paperv4_{sc}_saeks_seed0/best_model.zip", device="cpu")
        score_fn, tau_ks, _, _ = build_sae_gate(
            teacher, src_env, 0.5, False, seed=7000, ks_once_env=tgt_env)

        def rand_actor(obs, mask):
            return int(rng.choice(np.flatnonzero(np.asarray(mask))))

        def stu_actor(obs, mask):
            a, _ = student.predict(obs, action_masks=mask,
                                   deterministic=False)
            return int(a)

        for stage, actor in (("random", rand_actor), ("student", stu_actor)):
            obs, masks, states, lay_keys, layouts = collect_oracle(
                tgt_env, actor, N_FRAMES, 123, False)
            p_T = teacher_probs(teacher, obs, masks)
            p_S, H = masked_probs(student.policy, obs, masks)
            rec = recon_scores(score_fn, teacher.policy, obs)
            phi = phi_scores(teacher, obs)
            n_valid = masks.sum(-1).clip(min=2)
            Hn = H / np.log(n_valid)

            gains, keep = [], []
            a_T = p_T.argmax(-1)
            p_agree = p_S[np.arange(len(obs)), a_T]
            for i in range(len(obs)):
                lay, D = layouts[lay_keys[i]]
                reg = regrets(lay, D, states[i])
                valid = masks[i] & ~np.isnan(reg)
                if not valid.any() or np.isnan(reg[a_T[i]]):
                    keep.append(False); gains.append(0.0); continue
                ps = p_S[i] * valid
                ps = ps / max(ps.sum(), 1e-9)
                gains.append(float((ps * np.nan_to_num(reg)).sum()
                                   - reg[a_T[i]]))
                keep.append(True)
            keep = np.array(keep)
            g = np.array(gains)[keep]
            rec_k, phi_k = rec[keep], phi[keep]
            Hn_k, pa_k = Hn[keep], p_agree[keep]
            ks_open = rec_k <= tau_ks
            n = len(g)

            orders = {
                "random-pick": rng.permutation(n),
                "phi (EBTL-style)": np.argsort(-phi_k),
                "KS recon": np.argsort(rec_k),
                "KS & entropy": np.lexsort((-Hn_k, ~ks_open)),
                "KS & disagree": np.lexsort((pa_k, ~ks_open)),
            }
            print(f"\n== {sc} / {stage} visitation  (n={n:,}, "
                  f"KS-open {ks_open.mean():.1%}, base mean gain "
                  f"{g.mean():+.3f}) ==")
            print(f"{'gate':18s}" + "".join(
                f"  b={b:.0%}: gain, P(+), P(-)      " for b in BUDGETS))
            rows = {}
            for name, order in orders.items():
                line = f"{name:18s}"
                rows[name] = {}
                for b in BUDGETS:
                    top = order[:max(int(n*b), 20)]
                    gg = g[top]
                    line += (f"  {gg.mean():+7.3f} {np.mean(gg > 0):.2f} "
                             f"{np.mean(gg < 0):.2f}   ")
                    rows[name][b] = (gg.mean(), np.mean(gg > 0))
                print(line)
            ink = ks_open
            print(f"within KS-open: spearman(entropy, gain) "
                  f"{spearman(Hn_k[ink], g[ink]):+.2f}, "
                  f"spearman(-p_agree, gain) "
                  f"{spearman(-pa_k[ink], g[ink]):+.2f}")
            for b in BUDGETS:
                top = orders["KS & disagree"][:max(int(n*b), 20)]
                print(f"  kappa at b={b:.0%}: p_agree <= "
                      f"{pa_k[top].max():.3f}")
            results[f"{sc}_{stage}"] = dict(
                gain=g, rec=rec_k, phi=phi_k, Hn=Hn_k, p_agree=pa_k,
                tau_ks=tau_ks)
    np.savez("runs/oracle_gate_screen.npz",
             **{f"{k}__{kk}": vv for k, d in results.items()
                for kk, vv in d.items()})
    print("\nsaved -> runs/oracle_gate_screen.npz")
    print("SCREEN DONE")


if __name__ == "__main__":
    main()
