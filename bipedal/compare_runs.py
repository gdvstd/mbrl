"""Overlay eval learning curves of several runs and report data-efficiency metrics.

Compares the from-scratch baseline against transfer variants (and, when present,
averages across multiple seeds of the same method). Reads each run's
eval/evaluations.npz written by EvalCallback.

Usage (inside the venv):
    # explicit runs
    python compare_runs.py runs/student_none_seed0 runs/student_weight_init_seed0

    # or auto-discover all student runs
    python compare_runs.py --glob "runs/student_*"

    # custom threshold for "steps-to-threshold" and output path
    python compare_runs.py --glob "runs/student_*" --threshold 0 --out compare.png

Runs are grouped by method (dir name with the `student_` prefix and `_seedN`
suffix stripped), so multiple seeds of one method are averaged with a shaded
std band. A single-seed group is shaded with its across-eval-episode std.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")  # headless: save PNG, no display needed
import matplotlib.pyplot as plt
import numpy as np

SOLVED_REWARD = 300.0


def method_label(rundir: str) -> str:
    base = os.path.basename(os.path.normpath(rundir))
    base = re.sub(r"^student_", "", base)
    base = re.sub(r"_seed\d+$", "", base)
    return base or os.path.basename(rundir)


def load_run(rundir: str):
    """Return (timesteps (T,), per_episode_results (T, n_eval))."""
    path = os.path.join(rundir, "eval", "evaluations.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["timesteps"], d["results"]


def steps_to_threshold(timesteps, curve, threshold):
    hits = np.where(curve >= threshold)[0]
    return int(timesteps[hits[0]]) if len(hits) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rundirs", nargs="*", help="run directories to compare")
    parser.add_argument("--glob", type=str, default=None,
                        help="glob pattern for run dirs (alternative to listing)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="reward level for the steps-to-threshold metric")
    parser.add_argument("--smooth", type=int, default=1,
                        help="moving-average window (in eval points) for the plotted line")
    parser.add_argument("--out", type=str, default="runs/compare.png")
    args = parser.parse_args()

    rundirs = list(args.rundirs)
    if args.glob:
        rundirs += sorted(glob.glob(args.glob))
    rundirs = [r for r in dict.fromkeys(rundirs) if os.path.isdir(r)]
    if not rundirs:
        parser.error("no run directories found")

    # group runs by method label
    groups: dict[str, list[str]] = {}
    for r in rundirs:
        groups.setdefault(method_label(r), []).append(r)

    plt.figure(figsize=(9, 5.5))
    print(f"{'method':<16}{'best':>8}{'final':>8}{'AUC':>9}"
          f"{'steps>=%g' % args.threshold:>14}{'seeds':>7}")
    print("-" * 62)

    for label in sorted(groups):
        curves, timesteps_ref = [], None
        per_ep_std = None
        for r in groups[label]:
            loaded = load_run(r)
            if loaded is None:
                print(f"[warn] no evaluations.npz in {r}, skipping")
                continue
            timesteps, results = loaded
            timesteps_ref = timesteps
            curves.append(results.mean(axis=1))       # mean over eval episodes
            per_ep_std = results.std(axis=1)           # (kept for single-seed band)
        if not curves:
            continue

        curves = np.vstack(curves)                     # (n_seeds, T)
        mean_curve = curves.mean(axis=0)
        # band: across seeds if >1, else across eval episodes of the single run
        band = curves.std(axis=0) if curves.shape[0] > 1 else per_ep_std

        # optional smoothing of the plotted line
        line = mean_curve
        if args.smooth > 1:
            k = args.smooth
            line = np.convolve(mean_curve, np.ones(k) / k, mode="same")

        (h,) = plt.plot(timesteps_ref, line, label=f"{label} (n={curves.shape[0]})")
        plt.fill_between(timesteps_ref, mean_curve - band, mean_curve + band,
                         color=h.get_color(), alpha=0.15)

        best = mean_curve.max()
        final = mean_curve[-10:].mean() if len(mean_curve) >= 10 else mean_curve[-1]
        auc = mean_curve.mean()  # avg eval performance over training = data-efficiency proxy
        s2t = steps_to_threshold(timesteps_ref, mean_curve, args.threshold)
        print(f"{label:<16}{best:>8.1f}{final:>8.1f}{auc:>9.1f}"
              f"{('n/a' if s2t is None else f'{s2t:,}'):>14}{curves.shape[0]:>7}")

    plt.axhline(SOLVED_REWARD, color="gray", ls="--", lw=0.8, label="solved (300)")
    plt.axhline(0, color="black", lw=0.5, alpha=0.3)
    plt.xlabel("training env steps")
    plt.ylabel("eval mean return (10 episodes)")
    plt.title("BipedalWalkerHardcore: baseline vs transfer")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print(f"\n[plot] saved -> {os.path.abspath(args.out)}")
    print("AUC = mean eval return over the whole run (higher = more data-efficient)")


if __name__ == "__main__":
    main()
