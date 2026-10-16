"""Oracle advantage under the v2 (author-confirmed) reward:
r = 1 - 0.9*(steps/500) on reaching the goal.

    V*(s) = gamma^(D-1) * (1 - 0.9*(t+D)/500)
    Q(s,a) = gamma^D'   * (1 - 0.9*(t+1+D')/500)
    A = Q - V*   (0 for optimal advice; evaluated at t=0 -- the caches
                  store D(s), D(s') but not the elapsed step, and t only
                  adds a small linear tilt shared by both terms)

Re-renders the gate histograms from the cached runs/oracle_val_*.npz
(old-config rollouts; the full v2 redo needs the v2 teachers/students):
  oracle_advantage_hist_v2r.png (4x3) and adv_hist_4panel_v2r.png.

Usage: python fourroom/oracle_adv_v2reward.py
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fourroom.oracle_hist import CFGS, GATES, value_panel, INK

GAMMA, MAXS = 0.9, 500
KEYNOTE = "runs/videos/keynote"


def adv_v2(ds: np.ndarray, ds2: np.ndarray) -> np.ndarray:
    v = GAMMA ** (ds - 1) * (1 - 0.9 * ds / MAXS)
    q = GAMMA ** ds2 * (1 - 0.9 * (1 + ds2) / MAXS)
    return q - v


def panels(fig_axes, picks):
    for ax, (cfg, key, hi_id, tkey, title) in zip(fig_axes, picks):
        d = np.load(f"runs/oracle_val_{cfg}.npz")
        dA = adv_v2(d["ds"], d["ds2"])
        score, tau = d[key], float(d[tkey])
        open_ = (score >= tau) if hi_id else (score <= tau)
        vo = dA[open_].mean() if open_.any() else float("nan")
        vc = dA[~open_].mean()
        value_panel(ax, dA[open_], dA[~open_],
                    f"{title}   (open {open_.mean():.0%},  "
                    f"Ā {vo:+.3f} vs {vc:+.3f})")


def main() -> None:
    # full 4x3 grid
    fig, axes = plt.subplots(4, 3, figsize=(11.5, 11), dpi=200)
    fig.patch.set_facecolor("white")
    picks = []
    for cfg_sc, ego in CFGS:
        cfg = f"{cfg_sc}{'_ego' if ego else ''}"
        for gname, key, hi_id, tkey in (
                ("EBTL φ", "phi", True, "tau_phi"),
                ("SAE recon", "recon", False, "tau_recon"),
                ("SAID cosine", "said", True, "tau_said")):
            picks.append((cfg, key, hi_id, tkey, f"{cfg} — {gname}"))
    panels(axes.flat, picks)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Oracle ADVANTAGE under the v2 reward "
                 "r = 1 − 0.9·(steps/500):  A = Q − V*, 0 = optimal advice",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(f"{KEYNOTE}/oracle_advantage_hist_v2r.png",
                facecolor="white")
    print(f"wrote {KEYNOTE}/oracle_advantage_hist_v2r.png")

    # 4-panel lab-meeting crop (same picks as adv_hist_4panel)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), dpi=200)
    fig.patch.set_facecolor("white")
    crop = (("altgoal", "phi", True, "tau_phi", "altgoal — EBTL φ gate"),
            ("altgoal", "recon", False, "tau_recon",
             "altgoal — SAE recon gate"),
            ("locked", "phi", True, "tau_phi", "locked — EBTL φ gate"),
            ("locked_ego", "phi", True, "tau_phi",
             "locked_ego — EBTL φ gate"))
    panels(axes.flat, crop)
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Oracle advantage (v2 reward) at gate-open vs gate-closed "
                 "states  (0 = optimal advice)", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{KEYNOTE}/adv_hist_4panel_v2r.png", facecolor="white")
    print(f"wrote {KEYNOTE}/adv_hist_4panel_v2r.png")


if __name__ == "__main__":
    main()
