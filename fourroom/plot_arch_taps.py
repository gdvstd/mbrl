"""Architecture diagram: the paper teacher's actor tower (Fig. 8a spec)
with the three SAE tap locations (L1/L2/L3) and the energy score phi.

-> runs/videos/keynote/arch_taps.png
Usage: python fourroom/plot_arch_taps.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED = "#1f2430", "#5c6370"
BOX_FC, BOX_EC = "#f2f4f7", "#c9ced6"
C_TAP = {"L1": "#2a78d6", "L2": "#7a4fd3", "L3": "#d81159"}
C_PHI = "#eb6834"

BOXES = [  # (title, subtitle, output shape)
    ("input", "full grid obs", "3×11×11"),
    ("Conv 2×2 → 16", "+ ReLU", "16×10×10"),
    ("MaxPool 2×2", "", "16×5×5"),
    ("Conv 2×2 → 32", "+ ReLU", "32×4×4"),
    ("Conv 2×2 → 64", "+ ReLU", "64×3×3"),
    ("Flatten", "", "576"),
    ("Linear 576 → 7", "action logits", "7"),
]
TAPS = {2: ("L1", "400-d"), 3: ("L2", "512-d"), 5: ("L3", "576-d")}


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 3.8), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 3.8)
    ax.axis("off")

    w, h, y0, gap = 1.55, 1.05, 1.85, 0.32
    xs = []
    for i, (title, sub, shape) in enumerate(BOXES):
        x = 0.25 + i * (w + gap)
        xs.append(x)
        ax.add_patch(FancyBboxPatch(
            (x, y0), w, h, boxstyle="round,pad=0.03",
            fc=BOX_FC, ec=BOX_EC, lw=1.2))
        ax.text(x + w / 2, y0 + h * 0.70, title, ha="center", va="center",
                fontsize=9.5, color=INK, weight="bold")
        if sub:
            ax.text(x + w / 2, y0 + h * 0.42, sub, ha="center", va="center",
                    fontsize=8, color=MUTED)
        ax.text(x + w / 2, y0 + h * 0.15, shape, ha="center", va="center",
                fontsize=8.5, color=MUTED, style="italic")
        if i:
            ax.add_patch(FancyArrowPatch(
                (x - gap + 0.02, y0 + h / 2), (x - 0.02, y0 + h / 2),
                arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.3))

    # SAE taps (dropped below the tower)
    for i, (name, dim) in TAPS.items():
        cx = xs[i] + w / 2
        c = C_TAP[name]
        ax.add_patch(FancyArrowPatch((cx, y0 - 0.02), (cx, y0 - 0.55),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=c, lw=1.6))
        ax.add_patch(FancyBboxPatch((cx - 0.72, y0 - 1.12), 1.44, 0.52,
                                    boxstyle="round,pad=0.03",
                                    fc="white", ec=c, lw=1.6))
        ax.text(cx, y0 - 0.78, f"{name} tap", ha="center", va="center",
                fontsize=9.5, color=c, weight="bold")
        ax.text(cx, y0 - 1.00, f"{dim} → TopK SAE", ha="center",
                va="center", fontsize=8, color=MUTED)

    # phi from the raw logits (raised above the head)
    cx = xs[-1] + w / 2
    ax.add_patch(FancyArrowPatch((cx, y0 + h + 0.02), (cx, y0 + h + 0.45),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=C_PHI, lw=1.6))
    ax.add_patch(FancyBboxPatch((cx - 1.15, y0 + h + 0.48), 2.3, 0.52,
                                boxstyle="round,pad=0.03",
                                fc="white", ec=C_PHI, lw=1.6))
    ax.text(cx, y0 + h + 0.82, "energy φ(s)", ha="center", va="center",
            fontsize=9.5, color=C_PHI, weight="bold")
    ax.text(cx, y0 + h + 0.58, "= logsumexp(raw logits)", ha="center",
            va="center", fontsize=8, color=MUTED)

    ax.text(0.25, 0.22,
            "Actor tower shown (25.7K params, paper Fig. 8a); the critic is "
            "an identical separate tower → Linear 576→1 (value). "
            "SAEs are trained on SOURCE-rollout activations at each tap; "
            "π(a|s) = masked softmax of the logits.",
            fontsize=8, color=MUTED, ha="left")
    ax.set_title("Teacher network and probe locations: "
                 "SAE taps L1/L2/L3 and the energy score φ",
                 fontsize=12, color=INK, pad=14)
    fig.tight_layout()
    out = "runs/videos/keynote/arch_taps.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
