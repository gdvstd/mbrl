"""Robustness check for the GAE inverted-U: drop the pre-first-success
phase (where sparse reward makes GAE advantages pure bootstrap noise) and
re-draw the phi-decile curve from the cached advlogs.

First-success time is read from the AA runs' train monitors (per-env
episode lengths -> approximate global timestep of the first reward).

-> runs/videos/keynote/gae_advice_postsuccess.png
Usage: python fourroom/advlog_postsuccess.py
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fourroom.advlog_viz import decile_curve, spearman, zscore_windows

INK, MUTED, GRID = "#1f2430", "#5c6370", "#e6e8ec"
C_ADV, C_STU = "#2a78d6", "#eb6834"
N_ENVS = 8
N_DECILES = 10


def first_success_step(run_dir: str) -> int:
    """Earliest global timestep with episode reward > 0 (approx: per-env
    cumulative episode lengths x N_ENVS)."""
    best = None
    for f in glob.glob(os.path.join(run_dir, "train_monitor", "*.csv")):
        steps = 0
        with open(f) as fh:
            for i, row in enumerate(csv.reader(fh)):
                if i < 2 or not row:
                    continue
                r, length = float(row[0]), int(float(row[1]))
                steps += length
                if r > 0:
                    g = steps * N_ENVS
                    best = g if best is None else min(best, g)
                    break
    assert best is not None, f"no success found in {run_dir}"
    return best


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=200)
    fig.patch.set_facecolor("white")
    for ax, sc in zip(axes, ("altgoal", "locked")):
        run = f"runs/paper_{sc}_aa_advlog_seed0"
        d = np.load(os.path.join(run, "advlog.npz"))
        t0 = first_success_step(run)
        # under AA (delta=1 at start) rewards flow from the very first
        # episodes (t0 ~ 0), so the real concern is critic MATURITY, not
        # missing reward: keep the 25-50% window (critic settled, the
        # advising coin still active).
        T = d["t"].max()
        keep = (d["t"] >= 0.25 * T) & (d["t"] <= 0.5 * T)
        phi, advised = d["phi"][keep], d["advised"][keep]
        z = zscore_windows(d["adv"][keep])
        rho = spearman(phi[advised], z[advised])
        print(f"[{sc}] first success ~{t0/1000:.0f}K; window 25-50% keeps "
              f"{keep.mean():.0%}; Spearman(advised) {rho:+.3f}")

        edges = np.quantile(phi, np.linspace(0, 1, N_DECILES + 1))
        edges[-1] += 1e-6
        mids = 0.5 * (edges[:-1] + edges[1:])
        for sel, color, label in ((advised, C_ADV, "advised steps"),
                                  (~advised, C_STU, "student steps")):
            m, se = decile_curve(phi[sel], z[sel], edges)
            ax.errorbar(mids, m, yerr=se, color=color, lw=1.6, capsize=2,
                        label=label)
        tau = float(np.load(f"runs/oracle_val_{sc}.npz")["tau_phi"])
        ax.axvline(tau, color=MUTED, lw=1.0, ls="--")
        ax.text(tau, ax.get_ylim()[1], " τ", fontsize=8, color=MUTED,
                va="top")
        ax.axhline(0, color=GRID, lw=0.8)
        ax.set_xlabel("teacher energy φ", fontsize=9, color=INK)
        ax.set_title(f"{sc} — mature-critic window (25–50% of training; "
                     f"ρ_advised {rho:+.2f})", fontsize=10, color=INK)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel("z-scored GAE advantage", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Robustness check: the advised-vs-student contrast in the "
                 "mature-critic window (rewards flow from ~0K under AA)",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = "runs/videos/keynote/gae_advice_postsuccess.png"
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
