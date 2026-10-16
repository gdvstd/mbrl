"""Architecture diagram: the paper teacher's actor tower (Fig. 8a spec)
with the three SAE tap locations (L1-L3 activations) and the energy phi.

-> runs/videos/keynote/arch_taps.png
Usage: python fourroom/plot_arch_taps.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, FAINT = "#242a35", "#6b7280", "#aab0ba"
SHADOW = "#e3e6ea"
C_L1, C_L2, C_L3, C_PHI = "#2a78d6", "#7a4fd3", "#d81159", "#e8730c"

BOXES = [  # (title, sub, out shape, fill tint)
    ("Input", "full-grid obs", "3 × 11 × 11", "#f7f8fa"),
    ("Conv 2×2, 16", "ReLU", "16 × 10 × 10", "#eef4fc"),
    ("MaxPool 2×2", "", "16 × 5 × 5", "#eef4fc"),
    ("Conv 2×2, 32", "ReLU", "32 × 4 × 4", "#f1effb"),
    ("Conv 2×2, 64", "ReLU", "64 × 3 × 3", "#fdeef3"),
    ("Flatten", "", "576", "#fdeef3"),
    ("Linear 576 → 7", "action logits", "7", "#fdf3e7"),
]
TAPS = {2: ("L1 activation", "400-d", C_L1),
        3: ("L2 activation", "512-d", C_L2),
        5: ("L3 activation", "576-d", C_L3)}

W, H, GAP = 1.62, 1.00, 0.34
Y0 = 2.05
X0 = 0.45


def rounded(ax, x, y, w, h, fc, ec, lw=1.2, shadow=True):
    if shadow:
        ax.add_patch(FancyBboxPatch((x + 0.035, y - 0.045), w, h,
                                    boxstyle="round,pad=0.035,rounding_size=0.09",
                                    fc=SHADOW, ec="none", zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.035,rounding_size=0.09",
                                fc=fc, ec=ec, lw=lw, zorder=2))


def main() -> None:
    fig, ax = plt.subplots(figsize=(14.7, 4.9), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 14.7)
    ax.set_ylim(0, 4.9)
    ax.axis("off")

    xs = []
    for i, (title, sub, shape, tint) in enumerate(BOXES):
        x = X0 + i * (W + GAP)
        xs.append(x)
        rounded(ax, x, Y0, W, H, tint, "#d4d8de")
        cy = Y0 + H
        ax.text(x + W / 2, cy - 0.26, title, ha="center", va="center",
                fontsize=10, color=INK, weight="bold", zorder=3)
        if sub:
            ax.text(x + W / 2, cy - 0.52, sub, ha="center", va="center",
                    fontsize=8, color=MUTED, zorder=3)
        ax.text(x + W / 2, Y0 + 0.17, shape, ha="center", va="center",
                fontsize=8.5, color=FAINT, style="italic", zorder=3)
        if i:
            ax.add_patch(FancyArrowPatch(
                (x - GAP + 0.03, Y0 + H / 2), (x - 0.04, Y0 + H / 2),
                arrowstyle="-|>", mutation_scale=11, color=FAINT, lw=1.4,
                zorder=2))

    # taps dropped below the tower
    for i, (name, dim, c) in TAPS.items():
        cx = xs[i] + W / 2
        ax.plot([cx, cx], [Y0 - 0.04, Y0 - 0.52], color=c, lw=1.6,
                solid_capstyle="round", zorder=2)
        ax.plot(cx, Y0 - 0.56, marker="v", ms=5, color=c, zorder=2)
        rounded(ax, cx - 0.86, Y0 - 1.30, 1.72, 0.62, "white", c, lw=1.5)
        ax.text(cx, Y0 - 0.88, name, ha="center", va="center",
                fontsize=9.5, color=c, weight="bold", zorder=3)
        ax.text(cx, Y0 - 1.13, f"{dim}  →  TopK SAE", ha="center",
                va="center", fontsize=8, color=MUTED, zorder=3)

    # phi raised above the policy head
    cx = xs[-1] + W / 2
    ax.plot([cx, cx], [Y0 + H + 0.04, Y0 + H + 0.46], color=C_PHI, lw=1.6,
            solid_capstyle="round", zorder=2)
    ax.plot(cx, Y0 + H + 0.50, marker="^", ms=5, color=C_PHI, zorder=2)
    rounded(ax, cx - 1.22, Y0 + H + 0.58, 2.44, 0.66, "white", C_PHI,
            lw=1.5)
    ax.text(cx, Y0 + H + 1.02, "energy  φ(s)", ha="center", va="center",
            fontsize=10, color=C_PHI, weight="bold", zorder=3)
    ax.text(cx, Y0 + H + 0.76, "logsumexp of the raw logits", ha="center",
            va="center", fontsize=8, color=MUTED, zorder=3)

    # input-side <-> policy-side axis under the taps
    y_ax = 0.42
    ax.add_patch(FancyArrowPatch((X0 + 0.2, y_ax), (xs[-1] + W - 0.2, y_ax),
                                 arrowstyle="<|-|>", mutation_scale=10,
                                 color=FAINT, lw=1.1))
    ax.text(X0 + 0.25, y_ax + 0.17, "closer to the input", fontsize=8,
            color=MUTED, ha="left")
    ax.text(xs[-1] + W - 0.25, y_ax + 0.17, "closer to the policy",
            fontsize=8, color=MUTED, ha="right")
    ax.text((X0 + xs[-1] + W) / 2, y_ax - 0.24,
            "ID/OOD separation decays along this axis "
            "(L1 ≥ L2 ≥ L3 > φ)", fontsize=8.5, color=MUTED, ha="center",
            style="italic")

    ax.text(X0, 4.72,
            "Teacher actor tower (25.7K params, paper Fig. 8a) — probe "
            "locations", fontsize=12.5, color=INK, weight="bold",
            va="center")
    ax.text(X0, 4.42,
            "SAEs are trained on source-rollout activations at each tap; "
            "the critic is an identical, separate tower ending in "
            "Linear 576 → 1.", fontsize=8.5, color=MUTED, va="center")

    out = "runs/videos/keynote/arch_taps.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    print("wrote", out)


if __name__ == "__main__":
    main()
