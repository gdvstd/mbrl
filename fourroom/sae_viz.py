"""Separation figure: phi vs SAE-L1 reconstruction error, per teacher.

For each of the four plain teachers, overlaid score histograms on the
HARD split (target ID-condition vs OOD-condition), left = energy score
phi, right = SAE(L1) recon error. The visual companion to sae_probe's
AUROC table. Scores are also cached to runs/sae_scores_<cfg>.npz.

Usage: python fourroom/sae_viz.py
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fourroom.sae_probe import (
    N_FRAMES, activations, auroc, collect, recon_err, student_actor,
    teacher_actor, train_sae,
)
from fourroom.fourroom_env import SCENARIOS
from fourroom.paper_common import FrozenMaskableTeacher

INK, MUTED, GRID = "#1f2430", "#5c6370", "#e6e8ec"
C_ID, C_OOD = "#2a78d6", "#eb6834"
LAYER = "L1_pool400"


def scores_for(scenario: str, ego: bool):
    from sb3_contrib import MaskablePPO
    sfx = "_ego" if ego else ""
    cache = f"runs/sae_scores_{scenario}{sfx}.npz"
    if _os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    src_env, tgt_env = SCENARIOS[scenario]
    teacher = FrozenMaskableTeacher(
        f"runs/paper_{scenario}_teacher{sfx}/final_model.zip")
    student = MaskablePPO.load(
        f"runs/paper_{scenario}{sfx}_ebtl_seed0/best_model.zip", device="cpu")
    tag_mode = "goal" if scenario == "altgoal" else "key"
    src_obs, _ = collect(src_env, teacher_actor(teacher), N_FRAMES, 11,
                         tag_mode, ego)
    tgt_obs, tags = collect(tgt_env, student_actor(student), N_FRAMES, 22,
                            tag_mode, ego)
    id_sel = tags == 1
    ood_sel = tags == (3 if scenario == "altgoal" else 0)
    A_src, A_tgt = activations(teacher, src_obs), activations(teacher, tgt_obs)
    n_tr = int(len(src_obs) * 0.8)
    sae = train_sae(A_src[LAYER][:n_tr])
    out = dict(
        phi_id=A_tgt["phi"][id_sel], phi_ood=A_tgt["phi"][ood_sel],
        sae_id=recon_err(sae, A_tgt[LAYER][id_sel]),
        sae_ood=recon_err(sae, A_tgt[LAYER][ood_sel]))
    np.savez(cache, **out)
    return out


def panel(ax, id_s, ood_s, title, higher_is_ood: bool):
    lo = min(id_s.min(), ood_s.min())
    hi = max(np.percentile(id_s, 99.5), np.percentile(ood_s, 99.5))
    bins = np.linspace(lo, hi, 40)
    ax.hist(id_s, bins=bins, density=True, alpha=0.55, color=C_ID,
            label="ID-cond")
    ax.hist(ood_s, bins=bins, density=True, alpha=0.55, color=C_OOD,
            label="OOD-cond")
    a = auroc(ood_s, id_s) if higher_is_ood else auroc(-ood_s, -id_s)
    ax.set_title(f"{title}   AUROC {a:.2f}", fontsize=10, color=INK)
    ax.set_yticks([])
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def main() -> None:
    cfgs = [("altgoal", False), ("locked", False),
            ("altgoal", True), ("locked", True)]
    fig, axes = plt.subplots(4, 2, figsize=(9, 10), dpi=200)
    fig.patch.set_facecolor("white")
    for r, (sc, ego) in enumerate(cfgs):
        s = scores_for(sc, ego)
        name = f"{sc}{'_ego' if ego else ''}"
        panel(axes[r, 0], s["phi_id"], s["phi_ood"],
              f"{name} — energy φ", higher_is_ood=False)
        panel(axes[r, 1], s["sae_id"], s["sae_ood"],
              f"{name} — SAE L1 recon error", higher_is_ood=True)
        print(f"[viz] {name} done", flush=True)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("ID/OOD separation on the hard split: energy score vs "
                 "SAE reconstruction error", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = "runs/videos/keynote/sae_vs_phi.png"
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
