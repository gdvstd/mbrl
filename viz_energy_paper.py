"""EBTL energy-gate visualization for the paper-faithful four-room track.

Produces, per scenario:
  1. Fig-5b-style POSITION HEATMAPS: mean energy quantile (w.r.t. the
     teacher's own source-state distribution) of states visited at each
     cell. Rows = plain / energy-reg teacher; columns = source states and
     target states split by condition (goal room for altgoal, pre/post-key
     for locked). Diverging colors anchored at the EBTL threshold quantile:
     blue = teacher would advise, orange = withhold.
  2. A STATE MONTAGE: rendered target states binned by phi around tau
     (well below / just below / just above / well above) -- the qualitative
     check that the boundary is semantically right.

Usage: python viz_energy_paper.py --scenario altgoal
"""
from __future__ import annotations

import argparse
import os

import gymnasium as gym
import matplotlib
import numpy as np
import torch as th

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sb3_contrib import MaskablePPO

from fourroom_env import GAP_RIGHT, SCENARIOS, SIZE, action_mask
from paper_common import FrozenMaskableTeacher, _wrap

INK, MUTED, WALL = "#1f2430", "#5c6370", "#454a54"
DIVERGING = LinearSegmentedColormap.from_list(
    "advise", ["#eb6834", "#f2b28f", "#e8e8e8", "#8fb8e8", "#2a78d6"])
EBTL_Q = {"altgoal": 0.5, "locked": 0.7}
N_FRAMES = 6000


def goal_room(u) -> int:
    for x in range(SIZE):
        for y in range(SIZE):
            c = u.grid.get(x, y)
            if c is not None and c.type == "goal":
                return 1 if x < 5 else 3
    return 0


def roll(env_id: str, actor, teachers: dict, n_frames: int, seed: int,
         render: bool = False, env_kwargs: dict | None = None,
         tag_mode: str = "goal", ego: bool = False):
    """Roll `actor` (callable obs,mask->action); record pos/tag/phi per
    teacher (+frame if render)."""
    env_kwargs = dict(env_kwargs or {})
    if ego:
        env_kwargs.setdefault("agent_view_size", 11)
    env = _wrap(gym.make(env_id, render_mode="rgb_array" if render else None,
                         **env_kwargs), ego=ego)
    u = env.unwrapped
    obs, _ = env.reset(seed=seed)
    u.highlight = False  # render decoration only; input is the full grid
    groom = goal_room(u) if tag_mode == "goal" else 0
    recs = {k: [] for k in ("x", "y", "tag", "frame")}
    phis = {name: [] for name in teachers}
    for _ in range(n_frames):
        for name, t in teachers.items():
            ot, _ = t.policy.obs_to_tensor(obs)
            phis[name].append(float(t.energy_score(ot)[0]))
        recs["x"].append(int(u.agent_pos[0]))
        recs["y"].append(int(u.agent_pos[1]))
        recs["tag"].append(groom if tag_mode == "goal"
                           else (1 if u.carrying else 0))
        recs["frame"].append(env.render() if render else None)
        mask = action_mask(env)
        a = actor(obs, mask)
        obs, _, te, tr, _ = env.step(a)
        if te or tr:
            obs, _ = env.reset()
            groom = goal_room(u) if tag_mode == "goal" else 0
    env.close()
    return recs, {k: np.array(v) for k, v in phis.items()}


def teacher_actor(t):
    def act(obs, mask):
        ot, _ = t.policy.obs_to_tensor(obs)
        d = t.distribution(ot, action_masks=mask[None])
        return int(d.get_actions(deterministic=False)[0])
    return act


def student_actor(model):
    def act(obs, mask):
        a, _ = model.predict(obs, action_masks=mask, deterministic=False)
        return int(a)
    return act


def heat(recs, advised, sel) -> np.ndarray:
    """Per-cell advice rate: fraction of visits with phi >= tau."""
    grid = np.full((SIZE, SIZE), np.nan)
    cnt = np.zeros((SIZE, SIZE))
    acc = np.zeros((SIZE, SIZE))
    for i in np.flatnonzero(sel):
        acc[recs["y"][i], recs["x"][i]] += advised[i]
        cnt[recs["y"][i], recs["x"][i]] += 1
    m = cnt >= 3
    grid[m] = acc[m] / cnt[m]
    return grid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", choices=["altgoal", "locked"], required=True)
    ap.add_argument("--student", default=None,
                    help="target-rolling policy (default: ebtl seed0 best)")
    ap.add_argument("--ego", action="store_true")
    a = ap.parse_args()
    sc = a.scenario
    sfx = "_ego" if a.ego else ""
    src_env, tgt_env = SCENARIOS[sc]
    tau_q = EBTL_Q[sc]

    teachers = {
        "plain": FrozenMaskableTeacher(
            f"runs/paper_{sc}_teacher{sfx}/final_model.zip"),
        "ereg": FrozenMaskableTeacher(
            f"runs/paper_{sc}_teacher_ereg{sfx}/final_model.zip"),
    }
    student = MaskablePPO.load(
        a.student or f"runs/paper_{sc}{sfx}_ebtl_seed0/best_model.zip",
        device="cpu")

    # source states: each teacher rolls its own env; ECDF reference per teacher
    src = {name: roll(src_env, teacher_actor(t), {name: t}, 4000, seed=11,
                      ego=a.ego)
           for name, t in teachers.items()}
    ecdf = {name: np.sort(src[name][1][name]) for name in teachers}

    def q_of(name, phi):
        return np.searchsorted(ecdf[name], phi, side="right") / len(ecdf[name])

    # target states: ONE student roll, phi under BOTH teachers
    tgt_recs, tgt_phis = roll(tgt_env, student_actor(student), teachers,
                              N_FRAMES, seed=22, render=True,
                              tag_mode="goal" if sc == "altgoal" else "key",
                              ego=a.ego)

    if sc == "altgoal":
        cols = [("target, goal in Room 1", np.array(tgt_recs["tag"]) == 1),
                ("target, goal in Room 3", np.array(tgt_recs["tag"]) == 3)]
    else:
        cols = [("target, pre-key", np.array(tgt_recs["tag"]) == 0),
                ("target, post-key", np.array(tgt_recs["tag"]) == 1)]

    # ---------- heatmap figure ----------
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6), dpi=200)
    fig.patch.set_facecolor("white")
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    tau = {name: float(np.quantile(ecdf[name], tau_q)) for name in teachers}
    for r, name in enumerate(["plain", "ereg"]):
        panels = [("source (teacher rollouts)", *src[name])] + [
            (lbl, tgt_recs, {name: tgt_phis[name]}, sel) for lbl, sel in cols]
        for c in range(3):
            ax = axes[r, c]
            if c == 0:
                recs, phis = src[name][0], src[name][1][name]
                sel = np.ones(len(phis), bool)
                lbl = "source (teacher rollouts)"
            else:
                lbl, sel = cols[c - 1]
                recs, phis = tgt_recs, tgt_phis[name]
            g = heat(recs, (phis >= tau[name]).astype(float), sel)
            im = ax.imshow(g, cmap=DIVERGING, norm=norm)
            ax.set_facecolor("#d9dce1")
            if sc == "locked":
                ax.plot(*GAP_RIGHT, marker="s", ms=6, mfc="none",
                        mec=INK, mew=1.2)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(lbl, fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(f"{name} teacher", fontsize=10, color=INK)
    cb = fig.colorbar(im, ax=axes, shrink=0.75, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["0%", "50%", "100%"], fontsize=9)
    cb.set_label("advice rate  P(φ ≥ τ)", fontsize=9, color=INK)
    fig.suptitle(f"Where the energy gate opens — {sc}{sfx} (τ at q={tau_q})",
                 fontsize=12, color=INK, x=0.44)
    out = f"runs/videos/keynote/energy_map_{sc}{sfx}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print("wrote", out)

    # ---------- montage: target states binned by plain-teacher phi ----------
    q = q_of("plain", tgt_phis["plain"])
    bins = [("well below τ", q < max(tau_q - 0.3, 0.05)),
            ("just below τ", (q >= tau_q - 0.12) & (q < tau_q)),
            ("just above τ", (q >= tau_q) & (q < min(tau_q + 0.12, 0.999))),
            ("well above τ", q >= min(tau_q + 0.3, 0.999))]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9.6), dpi=200)
    fig.patch.set_facecolor("white")
    for r, (lbl, sel) in enumerate(bins):
        idx = np.flatnonzero(sel)
        pick = rng.choice(idx, size=min(4, len(idx)), replace=False) if len(idx) else []
        for c in range(4):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if c < len(pick):
                i = pick[c]
                ax.imshow(tgt_recs["frame"][i])
                ax.set_xlabel(f"φ={tgt_phis['plain'][i]:.2f} (q={q[i]:.2f})",
                              fontsize=8, color=MUTED)
            else:
                ax.axis("off")
        axes[r, 0].set_ylabel(lbl, fontsize=10, color=INK)
    fig.suptitle(f"Target states by teacher familiarity — {sc}{sfx} "
                 f"(plain teacher, τ at q={tau_q})", fontsize=12, color=INK)
    out = f"runs/videos/keynote/energy_states_{sc}{sfx}.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
