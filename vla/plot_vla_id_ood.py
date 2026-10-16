"""VLA ID/OOD overview: the task split criterion + ALL scores x ALL layers.

Left: per-task mean teacher phi over rollout frames (seen vs unseen),
with the teacher's gate success rates. Right: every measured separation
score -- phi, SAE recon (L8-L32), SAID task-cosine (L8-L32), SAID global
(L8-L32) -- as AUROC on demo frames.

-> runs/videos/keynote/vla_id_ood_overview.png
Usage: python vla/plot_vla_id_ood.py
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED = "#242a35", "#6b7280"
C_ID, C_OOD = "#2a78d6", "#eb6834"

TASKS = ["alphabet soup", "cream cheese", "salad dressing", "bbq sauce",
         "ketchup", "tomato sauce", "butter", "milk", "choc. pudding",
         "orange juice"]
PHI_MEAN = [22.29, 19.64, 20.38, 20.09, 19.69,
            19.53, 17.94, 18.33, 18.66, 18.31]
SUCC = [60, 30, 100, 70, 60, 0, 0, 0, 0, 0]
SEEN = [True] * 5 + [False] * 5

LAYERS = ["L8", "L16", "L24", "L32"]
GROUPS = [  # (name, color, aurocs per layer or single)
    ("energy φ", "#e8730c", [0.844]),
    ("SAE recon", "#d81159", [0.923, 0.828, 0.707, 0.665]),
    ("SAID task-cos", "#7a4fd3", [0.793, 0.679, 0.559, 0.445]),
    ("SAID global-cos", "#4c9f70", [0.443, 0.282, 0.328, 0.310]),
]


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 4.8), dpi=200,
                                   gridspec_kw={"width_ratios": [1, 1.35]})
    fig.patch.set_facecolor("white")

    y = np.arange(10)[::-1]
    colors = [C_ID if s else C_OOD for s in SEEN]
    ax1.barh(y, PHI_MEAN, color=colors, height=0.62)
    for yi, name, pm, sc, s in zip(y, TASKS, PHI_MEAN, SUCC, SEEN):
        ax1.text(pm + 0.15, yi, f"{pm:.1f}", va="center", fontsize=7.5,
                 color=INK)
        ax1.text(13.2, yi, f"{name}  ({sc}%)", va="center", ha="left",
                 fontsize=8, color=INK if s else MUTED)
    ax1.set_xlim(13, 24)
    ax1.set_yticks([])
    ax1.set_xlabel("mean teacher energy φ over rollout frames", fontsize=9,
                   color=INK)
    ax1.set_title("LIBERO-Object: teacher finetuned on 5 'seen' objects "
                  "(ID);\n5 held-out objects = OOD (same scene, only the "
                  "object differs)", fontsize=9.5, color=INK)
    for sp in ("top", "right", "left"):
        ax1.spines[sp].set_visible(False)
    ax1.tick_params(colors=MUTED, labelsize=8)
    ax1.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=C_ID),
                        plt.Rectangle((0, 0), 1, 1, fc=C_OOD)],
               labels=["seen (ID) — (teacher success %)", "unseen (OOD)"],
               frameon=False, fontsize=8, loc="lower right")

    xpos, ticks, ticklabs = 0.0, [], []
    for name, color, vals in GROUPS:
        for i, v in enumerate(vals):
            alpha = 1.0 - 0.22 * i
            ax2.bar(xpos, v, width=0.8, color=color, alpha=alpha)
            ax2.text(xpos, v + 0.012, f"{v:.2f}", ha="center",
                     fontsize=7.5, color=INK, rotation=0)
            ticks.append(xpos)
            ticklabs.append(LAYERS[i] if len(vals) > 1 else "")
            xpos += 1.0
        mid = xpos - (len(vals) + 1) / 2.0
        ax2.text(mid, -0.035, name, ha="center", va="top", fontsize=8.5,
                 color=color, weight="bold",
                 transform=ax2.get_xaxis_transform())
        xpos += 0.7
    ax2.axhline(0.5, color=MUTED, lw=0.9, ls=":")
    ax2.text(xpos - 0.8, 0.513, "0.5 = no separation", fontsize=7.5,
             color=MUTED, ha="right")
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(ticklabs, fontsize=7.5, color=MUTED)
    ax2.tick_params(axis="x", length=0, pad=2)
    ax2.set_ylim(0.2, 1.0)
    ax2.set_ylabel("AUROC (ID vs OOD demo frames)", fontsize=9, color=INK)
    ax2.set_title("All scores × all layers: separation of ID vs OOD",
                  fontsize=10, color=INK)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.tick_params(axis="y", colors=MUTED, labelsize=8)

    fig.suptitle("OpenVLA-7B / LIBERO-Object: the ID/OOD split and how "
                 "each score separates it", fontsize=12.5, color=INK,
                 y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out = "runs/videos/keynote/vla_id_ood_overview.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
