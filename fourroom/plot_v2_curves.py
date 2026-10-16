"""No-transfer learning curves under the author-confirmed v2 config
(max_steps 500, time-decayed reward, new locked source). Shaded vertical
bands mark where the paper's No-Transfer solves (altgoal ~150-175K,
locked takeoff ~600-800K) -- the calibration reference.

-> runs/videos/keynote/curves_v2_scratch.png
Usage: python fourroom/plot_v2_curves.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#1f2430", "#5c6370", "#e6e8ec"
C_FULL, C_EGO = "#2a78d6", "#eb6834"
PAPER_BAND = {"altgoal": (150, 175), "locked": (600, 800)}


def load(sc, sfx, seed):
    for p in (f"runs/paperv2_{sc}{sfx}_scratch_seed{seed}/eval/evaluations.npz",
              f"runs/paperv2_{sc}{sfx}_scratch_seed{seed}/evaluations.npz"):
        if os.path.exists(p):
            d = np.load(p)
            return d["timesteps"], d["results"].mean(axis=1)
    return None


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=200)
    fig.patch.set_facecolor("white")
    for ax, sc in zip(axes, ("altgoal", "locked")):
        for sfx, color, label in (("", C_FULL, "full-obs"),
                                  ("_ego", C_EGO, "egocentric")):
            runs = [load(sc, sfx, s) for s in range(3)]
            runs = [r for r in runs if r is not None]
            n = min(len(r[1]) for r in runs)
            t = runs[0][0][:n] / 1000
            ys = np.stack([r[1][:n] for r in runs])
            ax.plot(t, ys.mean(0), color=color, lw=1.8, label=label)
            ax.fill_between(t, ys.mean(0) - ys.std(0),
                            ys.mean(0) + ys.std(0), color=color, alpha=0.15)
        lo, hi = PAPER_BAND[sc]
        ax.axvspan(lo, hi, color="#9aa1ad", alpha=0.15)
        ax.text((lo + hi) / 2, 1.0, "paper\nNo-Transfer", ha="center",
                va="top", fontsize=7.5, color=MUTED)
        ax.axhline(0.8, color=GRID, lw=0.8, ls=":")
        ax.set_title(f"{sc} — No Transfer (v2 config, 3 seeds)",
                     fontsize=11, color=INK)
        ax.set_xlabel("training steps (K)", fontsize=9, color=INK)
        ax.set_ylim(-0.02, 1.02)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel("eval reward (time-decayed)", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    fig.suptitle("Author-feedback config: No-Transfer calibration curves",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = "runs/videos/keynote/curves_v2_scratch.png"
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
