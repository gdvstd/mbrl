"""Oracle evaluation of teacher advice value in the four-room targets.

The env is small and deterministic, so for every target layout we compute
D(s) = exact minimum steps-to-goal over the full state space
(x, y, dir, carrying, door_state) by backward BFS (equivalent to value
iteration with sparse reward 1 at the goal and gamma<1: V* = gamma^(D-1)).
The per-state value of a teacher advice action a is then its REGRET

    delta(s, a) = D(step(s, a)) - (D(s) - 1)   (0 = optimal, k = k wasted steps)

For each of the 4 plain teachers we roll the EBTL student in the target,
and at every visited state ask: had the teacher advised here, how good
would the advice have been?  We report, against this oracle label:
  - AUROC of phi and of SAE-L1 recon error for helpful (argmax-regret == 0)
  - Spearman correlation of each score with expected regret under the
    teacher's masked action distribution (what Algorithm 1 samples)
  - gate-conditioned advice quality: P(helpful | gate open/closed) and
    mean expected regret, for the phi gate (tau = q-quantile of source phi,
    q = 0.5 altgoal / 0.7 locked) and a recon gate matched to the same
    source-side open rate.

Figures: runs/videos/keynote/oracle_advice.png (helpful/harmful histograms)
and oracle_gate.png (gate-conditioned advice quality bars).
Scores cached to runs/oracle_scores_<cfg>.npz.

Usage: python fourroom/oracle_advice.py
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from collections import deque

import gymnasium as gym
import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fourroom.fourroom_env import SCENARIOS, action_mask
from fourroom.paper_common import FrozenMaskableTeacher, _wrap
from fourroom.sae_probe import (
    N_FRAMES, TAPS, activations, auroc, recon_err, student_actor,
    teacher_actor, train_sae, collect,
)

DIR_VEC = ((1, 0), (0, 1), (-1, 0), (0, -1))  # minigrid: E, S, W, N
LOCKED, OPEN, CLOSED = 0, 1, 2  # door states (CLOSED = unlocked but shut)
EBTL_Q = {"altgoal": 0.5, "locked": 0.7}
CFGS = [("altgoal", False), ("locked", False),
        ("altgoal", True), ("locked", True)]
LAYER = "L1_pool400"

INK, MUTED, GRID = "#1f2430", "#5c6370", "#e6e8ec"
C_GOOD, C_BAD = "#2a78d6", "#eb6834"
KEYNOTE = "runs/videos/keynote"


# ---------------------------------------------------------------- oracle
class Layout:
    """One target episode's geometry, read off the actual reset grid."""

    def __init__(self, env_u) -> None:
        g = env_u.grid
        self.walls, self.key, self.goal, self.door = set(), None, None, None
        for x in range(g.width):
            for y in range(g.height):
                c = g.get(x, y)
                if c is None:
                    continue
                if c.type == "wall":
                    self.walls.add((x, y))
                elif c.type == "key":
                    self.key = (x, y)
                elif c.type == "goal":
                    self.goal = (x, y)
                elif c.type == "door":
                    self.door = (x, y)
        self.width, self.height = g.width, g.height
        # key may already be carried at observation time
        if self.key is None and env_u.carrying is not None:
            self.key = tuple(env_u.carrying.init_pos)

    def id(self):
        return (self.goal, self.key, self.door)


def _passable(lay: Layout, pos, carry: int, door: int) -> bool:
    if pos in lay.walls:
        return False
    if lay.key is not None and pos == lay.key and not carry:
        return False  # key still on the grid blocks the cell
    if lay.door is not None and pos == lay.door and door != OPEN:
        return False
    return True


def _step(lay: Layout, s, a):
    """Deterministic transition (drop/done are no-ops; masked anyway)."""
    x, y, d, carry, door = s
    if a == 0:
        return (x, y, (d - 1) % 4, carry, door)
    if a == 1:
        return (x, y, (d + 1) % 4, carry, door)
    fx, fy = x + DIR_VEC[d][0], y + DIR_VEC[d][1]
    if a == 2:
        if _passable(lay, (fx, fy), carry, door):
            return (fx, fy, d, carry, door)
        return s
    if a == 3:
        if lay.key is not None and (fx, fy) == lay.key and not carry:
            return (x, y, d, 1, door)
        return s
    if a == 5 and lay.door is not None and (fx, fy) == lay.door:
        if door == LOCKED and carry:
            return (x, y, d, carry, OPEN)
        if door == OPEN:
            return (x, y, d, carry, CLOSED)
        if door == CLOSED:
            return (x, y, d, carry, OPEN)
    return s


def dist_table(lay: Layout) -> dict:
    """D(s) = min steps to reach the goal cell, by backward BFS."""
    cells = [(x, y) for x in range(lay.width) for y in range(lay.height)
             if (x, y) not in lay.walls]
    doors = (LOCKED, OPEN, CLOSED) if lay.door is not None else (LOCKED,)
    states = [(x, y, d, c, dr) for (x, y) in cells for d in range(4)
              for c in (0, 1) for dr in doors]
    preds: dict = {s: [] for s in states}
    for s in states:
        if (s[0], s[1]) == lay.goal:
            continue  # terminal; no outgoing moves
        for a in (0, 1, 2, 3, 5):
            s2 = _step(lay, s, a)
            if s2 != s:
                preds[s2].append(s)
    D = {}
    q = deque()
    for s in states:
        if (s[0], s[1]) == lay.goal:
            D[s] = 0
            q.append(s)
    while q:
        s = q.popleft()
        for p in preds[s]:
            if p not in D:
                D[p] = D[s] + 1
                q.append(p)
    return D


def regrets(lay: Layout, D: dict, s) -> np.ndarray:
    """Per-action regret vector (wasted steps vs the optimal action)."""
    out = np.full(7, np.nan)
    ds = D.get(s)
    if ds is None or ds == 0:
        return out
    for a in (0, 1, 2, 3, 5):
        s2 = _step(lay, s, a)
        d2 = D.get(s2)
        if d2 is not None:
            out[a] = d2 - (ds - 1)
    return out


# ---------------------------------------------------------------- collection
def collect_oracle(env_id: str, actor, n_frames: int, seed: int, ego: bool):
    """Roll `actor`; per frame record obs, mask, oracle state and layout."""
    kw = {"agent_view_size": 11} if ego else {}
    env = _wrap(gym.make(env_id, **kw), ego=ego)
    u = env.unwrapped

    layouts: dict = {}

    def snap():
        lay = Layout(u)
        lay_key = lay.id()
        if lay_key not in layouts:
            layouts[lay_key] = (lay, dist_table(lay))
        door = LOCKED
        if lay.door is not None:
            dobj = u.grid.get(*lay.door)
            door = OPEN if dobj.is_open else (LOCKED if dobj.is_locked
                                              else CLOSED)
        s = (int(u.agent_pos[0]), int(u.agent_pos[1]), int(u.agent_dir),
             int(u.carrying is not None), door)
        return lay_key, s

    obs, _ = env.reset(seed=seed)
    obs_buf, masks, states, lay_keys = [], [], [], []
    for _ in range(n_frames):
        m = action_mask(env)
        lk, s = snap()
        obs_buf.append(obs.copy())
        masks.append(np.asarray(m, dtype=bool).copy())
        states.append(s)
        lay_keys.append(lk)
        obs, _, te, tr, _ = env.step(actor(obs, m))
        if te or tr:
            obs, _ = env.reset()
    env.close()
    return np.stack(obs_buf), np.stack(masks), states, lay_keys, layouts


@th.no_grad()
def teacher_logits(teacher, obs_batch) -> th.Tensor:
    pol = teacher.policy
    ot, _ = pol.obs_to_tensor(obs_batch)
    feats = pol.extract_features(ot)
    if not pol.share_features_extractor:
        feats = feats[0]
    return pol.action_net(pol.mlp_extractor.forward_actor(feats))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(v):
        order = v.argsort(kind="mergesort")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        # average ranks over ties
        for val in np.unique(v):
            sel = v == val
            r[sel] = r[sel].mean()
        return r
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


# ---------------------------------------------------------------- per config
def probe_config(scenario: str, ego: bool) -> dict[str, np.ndarray]:
    from sb3_contrib import MaskablePPO
    sfx = "_ego" if ego else ""
    cache = f"runs/oracle_scores_{scenario}{sfx}.npz"
    if _os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}

    src_env, tgt_env = SCENARIOS[scenario]
    teacher = FrozenMaskableTeacher(
        f"runs/paper_{scenario}_teacher{sfx}/final_model.zip")
    student = MaskablePPO.load(
        f"runs/paper_{scenario}{sfx}_ebtl_seed0/best_model.zip", device="cpu")

    # target rollouts under the student + oracle state per frame
    tgt_obs, tgt_masks, states, lay_keys, layouts = collect_oracle(
        tgt_env, student_actor(student), N_FRAMES, 22, ego)

    # oracle labels for the teacher's advice at every visited state
    logits = teacher_logits(teacher, tgt_obs)
    masked = logits.masked_fill(~th.as_tensor(tgt_masks), -1e9)
    probs = th.softmax(masked, dim=-1).numpy()
    argmax_a = masked.argmax(-1).numpy()
    phi = th.logsumexp(logits, dim=-1).numpy()  # raw logits, as in EBTL

    reg_arg = np.full(len(states), np.nan)
    reg_exp = np.full(len(states), np.nan)
    for i, (s, lk) in enumerate(zip(states, lay_keys)):
        lay, D = layouts[lk]
        r = regrets(lay, D, s)
        if np.isnan(r[argmax_a[i]]):
            continue
        reg_arg[i] = r[argmax_a[i]]
        valid = ~np.isnan(r)
        p = probs[i] * valid
        if p.sum() > 0:
            reg_exp[i] = (p / p.sum() * np.where(valid, r, 0.0)).sum()
    keep = ~np.isnan(reg_arg) & ~np.isnan(reg_exp)

    # scores: phi (above) + SAE-L1 recon error trained on source rollouts
    tag_mode = "goal" if scenario == "altgoal" else "key"
    src_obs, _ = collect(src_env, teacher_actor(teacher), N_FRAMES, 11,
                         tag_mode, ego)
    A_src = activations(teacher, src_obs)
    A_tgt = activations(teacher, tgt_obs)
    n_tr = int(len(src_obs) * 0.8)
    sae = train_sae(A_src[LAYER][:n_tr])
    recon_src = recon_err(sae, A_src[LAYER][:n_tr])
    recon_tgt = recon_err(sae, A_tgt[LAYER])
    phi_src = A_src["phi"][:n_tr]

    out = dict(
        phi=phi[keep], recon=recon_tgt[keep],
        reg_arg=reg_arg[keep], reg_exp=reg_exp[keep],
        phi_src=phi_src, recon_src=recon_src,
        q=np.array(EBTL_Q[scenario]))
    np.savez(cache, **out)
    print(f"[oracle] {scenario}{sfx}: {keep.sum()}/{len(states)} frames, "
          f"{len(layouts)} layouts", flush=True)
    return out


def report(name: str, s: dict) -> dict:
    helpful = s["reg_arg"] == 0
    q = float(s["q"])
    tau_phi = np.quantile(s["phi_src"], q)
    tau_rec = np.quantile(s["recon_src"], 1 - q)  # same source-side open rate
    open_phi = s["phi"] >= tau_phi
    open_rec = s["recon"] <= tau_rec

    r = {
        "p_helpful": helpful.mean(),
        "auroc_phi": auroc(s["phi"][helpful], s["phi"][~helpful]),
        "auroc_recon": auroc(-s["recon"][helpful], -s["recon"][~helpful]),
        "rho_phi": spearman(s["phi"], -s["reg_exp"]),
        "rho_recon": spearman(-s["recon"], -s["reg_exp"]),
    }
    for gate, sel in (("phi", open_phi), ("recon", open_rec)):
        r[f"{gate}_open_rate"] = sel.mean()
        r[f"{gate}_help_open"] = helpful[sel].mean() if sel.any() else np.nan
        r[f"{gate}_help_closed"] = (helpful[~sel].mean() if (~sel).any()
                                    else np.nan)
        r[f"{gate}_reg_open"] = (s["reg_exp"][sel].mean() if sel.any()
                                 else np.nan)
        r[f"{gate}_reg_closed"] = (s["reg_exp"][~sel].mean() if (~sel).any()
                                   else np.nan)

    print(f"\n== {name} (q={q}) ==")
    print(f"P(advice helpful) overall:        {r['p_helpful']:.3f}")
    print(f"AUROC helpful  phi: {r['auroc_phi']:.3f}   "
          f"recon: {r['auroc_recon']:.3f}")
    print(f"Spearman(score, -E[regret])  phi: {r['rho_phi']:.3f}   "
          f"recon: {r['rho_recon']:.3f}")
    for gate in ("phi", "recon"):
        print(f"{gate:5s} gate  open {r[f'{gate}_open_rate']:.2f} | "
              f"P(helpful) open {r[f'{gate}_help_open']:.3f} vs closed "
              f"{r[f'{gate}_help_closed']:.3f} | E[regret] open "
              f"{r[f'{gate}_reg_open']:.2f} vs closed "
              f"{r[f'{gate}_reg_closed']:.2f}")
    return r


# ---------------------------------------------------------------- figures
def hist_panel(ax, good, bad, title):
    lo = min(np.percentile(good, 0.5), np.percentile(bad, 0.5))
    hi = max(np.percentile(good, 99.5), np.percentile(bad, 99.5))
    bins = np.linspace(lo, hi, 40)
    ax.hist(good, bins=bins, density=True, alpha=0.55, color=C_GOOD,
            label="helpful advice")
    ax.hist(bad, bins=bins, density=True, alpha=0.55, color=C_BAD,
            label="harmful advice")
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7)


def figures(all_s: dict, all_r: dict) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(9, 10), dpi=200)
    fig.patch.set_facecolor("white")
    for i, (cfg, s) in enumerate(all_s.items()):
        h = s["reg_arg"] == 0
        r = all_r[cfg]
        hist_panel(axes[i, 0], s["phi"][h], s["phi"][~h],
                   f"{cfg} — energy φ   AUROC {r['auroc_phi']:.2f}")
        hist_panel(axes[i, 1], -s["recon"][h], -s["recon"][~h],
                   f"{cfg} — −(SAE recon error)   "
                   f"AUROC {r['auroc_recon']:.2f}")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Oracle advice value: score distributions for "
                 "helpful vs harmful teacher advice", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{KEYNOTE}/oracle_advice.png", facecolor="white")
    print(f"wrote {KEYNOTE}/oracle_advice.png")

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), dpi=200, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, (cfg, r) in zip(axes, all_r.items()):
        vals = [r["p_helpful"], r["phi_help_open"], r["phi_help_closed"],
                r["recon_help_open"], r["recon_help_closed"]]
        labels = ["overall", "φ gate\nopen", "φ gate\nclosed",
                  "recon gate\nopen", "recon gate\nclosed"]
        colors = ["#9aa1ad", C_GOOD, C_BAD, C_GOOD, C_BAD]
        x = np.arange(len(vals))
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, color=INK)
        ax.set_title(cfg, fontsize=10, color=INK)
        ax.set_ylim(0, 1.0)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5,
                        color=INK)
    axes[0].set_ylabel("P(advice helpful)", fontsize=9, color=INK)
    fig.suptitle("Advice quality conditioned on the gate "
                 "(thresholds matched to the same source-side open rate)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{KEYNOTE}/oracle_gate.png", facecolor="white")
    print(f"wrote {KEYNOTE}/oracle_gate.png")


def main() -> None:
    _os.makedirs(KEYNOTE, exist_ok=True)
    all_s, all_r = {}, {}
    for scenario, ego in CFGS:
        cfg = f"{scenario}{'_ego' if ego else ''}"
        all_s[cfg] = probe_config(scenario, ego)
        all_r[cfg] = report(cfg, all_s[cfg])
    figures(all_s, all_r)
    print("ORACLE ADVICE ALL DONE")


if __name__ == "__main__":
    main()
