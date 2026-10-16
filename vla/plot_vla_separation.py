"""Act-1 slide figure: ID/OOD separation of the three scores on OpenVLA.

Reads runs/sae_probe_vla.npz (fetched from the cluster vla_sae_probe run)
and renders phi | SAE-recon(L8) | SAID task-cosine(L8) histograms.
-> runs/videos/keynote/vla_separation.png
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#1f2430", "#5c6370", "#e6e8ec"
C_ID, C_OOD = "#2a78d6", "#eb6834"


def auroc(pos, neg):
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (
        len(pos) * len(neg))


def main() -> None:
    d = np.load("runs/sae_probe_vla.npz")
    panels = [
        ("energy φ", d["phi_id"], d["phi_ood"], False),
        ("SAE recon error (L8)", d["L8_recon_id"], d["L8_recon_ood"], True),
        ("SAE-OOD: SAID task-cosine (L8)", d["L8_cmax_id"],
         d["L8_cmax_ood"], False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), dpi=200)
    fig.patch.set_facecolor("white")
    for ax, (title, vid, vood, hi_ood) in zip(axes, panels):
        a = auroc(vood, vid) if hi_ood else auroc(-vood, -vid)
        lo = min(np.percentile(vid, .5), np.percentile(vood, .5))
        hi = max(np.percentile(vid, 99.5), np.percentile(vood, 99.5))
        bins = np.linspace(lo, hi, 45)
        ax.hist(vid, bins=bins, density=True, alpha=0.55, color=C_ID,
                label="seen tasks (ID)")
        ax.hist(vood, bins=bins, density=True, alpha=0.55, color=C_OOD,
                label="unseen tasks (OOD)")
        ax.set_title(f"{title}   AUROC {a:.2f}", fontsize=10, color=INK)
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("OpenVLA-7B / LIBERO: ID vs OOD separation of the three "
                 "scores (demo frames)", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig("runs/videos/keynote/vla_separation.png", facecolor="white")
    print("wrote runs/videos/keynote/vla_separation.png")


if __name__ == "__main__":
    main()
