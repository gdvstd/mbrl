"""GAE-advantage view of advice quality, from instrumented AA runs.

AA issues advice by a state-INDEPENDENT coin flip (delta(t) schedule), so
correlating the teacher's energy phi(s_t) with the realized GAE advantage
A_t of advised steps is free of gate-selection bias. Student-acted steps
serve as the control group.

Per scenario (full-obs AA runs with advlog.npz):
  left  : mean z-scored advantage per phi-decile, advised vs student steps
          (z-scored within 2048-step windows to remove the advantage-scale
          drift as the value function improves); EBTL tau marked.
  right : raw mean advantage of advised vs student steps over training.

-> runs/videos/keynote/gae_advice.png

Usage: python fourroom/advlog_viz.py
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Fast rank correlation for continuous data (no tie averaging)."""
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")

INK, MUTED, GRID = "#1f2430", "#5c6370", "#e6e8ec"
C_ADV, C_STU = "#2a78d6", "#eb6834"
KEYNOTE = "runs/videos/keynote"
SCN = ("altgoal", "locked")
WIN = 2048  # z-normalization window (~1 rollout of 256 steps x 8 envs)
N_DECILES = 10
N_TBINS = 20


def zscore_windows(adv: np.ndarray) -> np.ndarray:
    out = np.empty_like(adv)
    for i in range(0, len(adv), WIN):
        w = adv[i:i + WIN]
        out[i:i + WIN] = (w - w.mean()) / (w.std() + 1e-8)
    return out


def decile_curve(phi, vals, edges):
    mean, se = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (phi >= lo) & (phi < hi)
        v = vals[sel]
        mean.append(v.mean() if len(v) else np.nan)
        se.append(v.std() / max(np.sqrt(len(v)), 1) if len(v) else np.nan)
    return np.array(mean), np.array(se)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=200)
    fig.patch.set_facecolor("white")
    for r, sc in enumerate(SCN):
        d = np.load(f"runs/paper_{sc}_aa_advlog_seed0/advlog.npz")
        phi, advised = d["phi"], d["advised"]
        adv, t = d["adv"], d["t"]
        tau = float(np.load(f"runs/oracle_val_{sc}.npz")["tau_phi"])
        z = zscore_windows(adv)

        rho_a = spearman(phi[advised], z[advised])
        rho_s = spearman(phi[~advised], z[~advised])
        print(f"[{sc}] Spearman(phi, z-adv): advised {rho_a:+.3f}  "
              f"student {rho_s:+.3f}  | raw mean adv: advised "
              f"{adv[advised].mean():+.4f}  student {adv[~advised].mean():+.4f}")

        edges = np.quantile(phi, np.linspace(0, 1, N_DECILES + 1))
        edges[-1] += 1e-6
        mids = 0.5 * (edges[:-1] + edges[1:])
        ax = axes[r, 0]
        for sel, color, label in ((advised, C_ADV, "advised steps"),
                                  (~advised, C_STU, "student steps")):
            m, se = decile_curve(phi[sel], z[sel], edges)
            ax.errorbar(mids, m, yerr=se, color=color, lw=1.6, capsize=2,
                        label=label)
        ax.axvline(tau, color=MUTED, lw=1.0, ls="--")
        ax.text(tau, ax.get_ylim()[1], " τ", fontsize=8, color=MUTED,
                va="top")
        ax.axhline(0, color=GRID, lw=0.8)
        ax.set_xlabel("teacher energy φ", fontsize=9, color=INK)
        ax.set_ylabel("z-scored GAE advantage", fontsize=9, color=INK)
        ax.set_title(f"{sc} — advantage vs φ (deciles), "
                     f"ρ_advised {rho_a:+.2f}", fontsize=10, color=INK)

        ax = axes[r, 1]
        tb = np.linspace(t.min(), t.max() + 1, N_TBINS + 1)
        tm = 0.5 * (tb[:-1] + tb[1:]) / 1000
        for sel, color, label in ((advised, C_ADV, "advised steps"),
                                  (~advised, C_STU, "student steps")):
            m = [adv[sel & (t >= lo) & (t < hi)].mean()
                 if (sel & (t >= lo) & (t < hi)).any() else np.nan
                 for lo, hi in zip(tb[:-1], tb[1:])]
            ax.plot(tm, m, color=color, lw=1.6, label=label)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.set_xlabel("training steps (K)", fontsize=9, color=INK)
        ax.set_ylabel("mean GAE advantage", fontsize=9, color=INK)
        ax.set_title(f"{sc} — advantage over training", fontsize=10,
                     color=INK)
        for a2 in axes[r]:
            for sp in ("top", "right"):
                a2.spines[sp].set_visible(False)
            a2.tick_params(colors=MUTED, labelsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Empirical advice value under AA (state-independent "
                 "advising): GAE advantage vs teacher energy φ",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{KEYNOTE}/gae_advice.png", facecolor="white")
    print(f"wrote {KEYNOTE}/gae_advice.png")


if __name__ == "__main__":
    main()
