"""Learning-curve comparison: gap gate vs EBTL / AA / fine-tune / scratch.

Mean +/- std over 3 seeds of the periodic eval reward, per scenario
(full-obs). -> runs/videos/keynote/curves_gap.png

Usage: python fourroom/plot_gap_curves.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#1f2430", "#5c6370", "#e6e8ec"
COLORS = {"gap": "#2a78d6", "gapnc": "#17bebb", "sae": "#d81159",
          "ebtl": "#7a4fd3", "aa": "#eb6834", "finetune": "#7fb069",
          "scratch": "#9aa1ad"}
LABELS = {"gap": "Gap gate", "gapnc": "Gap gate (no coin)",
          "sae": "SAE gate", "ebtl": "EBTL", "aa": "AA",
          "finetune": "Fine-tune (frozen conv)", "scratch": "No transfer"}


def load(sc: str, tr: str, seed: int):
    for p in (f"runs/paper_{sc}_{tr}_seed{seed}/evaluations.npz",
              f"runs/paper_{sc}_{tr}_seed{seed}/eval/evaluations.npz"):
        if os.path.exists(p):
            d = np.load(p)
            return d["timesteps"], d["results"].mean(axis=1)
    return None


def oracle_figure() -> None:
    """Solid = oracle-optimal teacher, dashed = trained teacher."""
    pairs = (("aa", "oaa", "#eb6834"), ("jsrl", "ojsrl", "#7fb069"),
             ("ksrl", "oksrl", "#2a78d6"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=200)
    fig.patch.set_facecolor("white")
    for ax, sc in zip(axes, ("altgoal", "locked")):
        for real, orac, color in pairs:
            for tr, ls, lab in ((orac, "-", f"{real.upper()} (oracle)"),
                                (real, "--", f"{real.upper()} (teacher)")):
                runs = [load(sc, tr, s) for s in range(3)]
                runs = [r for r in runs if r is not None]
                if not runs:
                    continue
                n = min(len(r[1]) for r in runs)
                t = runs[0][0][:n] / 1000
                ys = np.stack([r[1][:n] for r in runs])
                ax.plot(t, ys.mean(0), color=color, lw=1.8, ls=ls,
                        label=lab)
                ax.fill_between(t, ys.mean(0) - ys.std(0),
                                ys.mean(0) + ys.std(0), color=color,
                                alpha=0.10)
        ax.set_title(sc, fontsize=11, color=INK)
        ax.set_xlabel("training steps (K)", fontsize=9, color=INK)
        ax.set_ylim(-0.02, 1.02)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel("eval reward", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("How much does advice QUALITY matter? Oracle-optimal vs "
                 "trained teacher, same schedules (3 seeds)",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = "runs/videos/keynote/curves_oracle.png"
    fig.savefig(out, facecolor="white")
    print("wrote", out)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=200)
    fig.patch.set_facecolor("white")
    for ax, sc in zip(axes, ("altgoal", "locked")):
        for tr in ("scratch", "finetune", "aa", "ebtl", "gap", "gapnc",
                   "sae"):
            runs = [load(sc, tr, s) for s in range(3)]
            runs = [r for r in runs if r is not None]
            if not runs:
                continue
            n = min(len(r[1]) for r in runs)
            t = runs[0][0][:n] / 1000
            ys = np.stack([r[1][:n] for r in runs])
            m, sd = ys.mean(0), ys.std(0)
            ax.plot(t, m, color=COLORS[tr], lw=1.8, label=LABELS[tr])
            ax.fill_between(t, m - sd, m + sd, color=COLORS[tr], alpha=0.15)
        ax.set_title(sc, fontsize=11, color=INK)
        ax.set_xlabel("training steps (K)", fontsize=9, color=INK)
        ax.set_ylim(-0.02, 1.02)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel("eval reward", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Gap gate (teacher-familiar AND student-uncertain) vs "
                 "baselines, 3 seeds", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = "runs/videos/keynote/curves_gap.png"
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
    oracle_figure()
