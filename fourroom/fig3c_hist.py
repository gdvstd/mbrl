"""Fig-3c/4c style energy histograms (author feedback): label frames by
SUB-TASK condition, not by which task they came from.

Three distributions per scenario:
  blue   -- SOURCE task frames (the teacher's training distribution,
            re-rolled with the trained teacher)
  orange -- TARGET frames in the ID condition (altgoal: episodes whose
            goal is in Room 1 -- the shared sub-task; locked: post-key
            frames -- navigation the teacher knows)
  purple -- TARGET frames in the OOD condition (goal in Room 3 /
            pre-key frames)
The paper's claim to reproduce: the target distribution is BIMODAL, with
the ID-condition mode overlapping the source. Second row: the same
three-way labeling for the SAE-L1 reconstruction error.

Uses the v4 teachers; target frames are collected with a trained v4
student (ebtl for altgoal, aa for locked). tau line = Quantile_q over the
successful-episode calibration protocol.

-> runs/videos/keynote/fig3c_hist.png
Usage: python fourroom/fig3c_hist.py
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fourroom.fourroom_env import SCENARIOS
from fourroom.paper_common import (FrozenMaskableTeacher,
                                   calibrate_energy_threshold)
from fourroom.sae_probe import (
    N_FRAMES, activations, collect, recon_err, student_actor, teacher_actor,
    train_sae,
)

INK, MUTED, GRID = "#1f2430", "#5c6370", "#e6e8ec"
C_SRC, C_ID, C_OOD = "#4c86c6", "#f0a05a", "#9a6bd1"  # paper-ish palette
Q = {"altgoal": 0.5, "locked": 0.7}
STUDENT = {"altgoal": "runs/paperv4_altgoal_ebtl_seed0/best_model.zip",
           "locked": "runs/paperv4_locked_aa_seed0/best_model.zip"}
LABELS = {
    "altgoal": ("target, goal in Room 1 (shared sub-task)",
                "target, goal in Room 3 (non-shared)"),
    "locked": ("target, key obtained (known navigation)",
               "target, before key (novel sub-task)"),
}


def panel(ax, src, tid, tood, tau, title, labels, xlabel):
    lo = min(np.percentile(v, 0.5) for v in (src, tid, tood))
    hi = max(np.percentile(v, 99.5) for v in (src, tid, tood))
    bins = np.linspace(lo, hi, 48)
    ax.hist(src, bins=bins, density=True, alpha=0.5, color=C_SRC,
            label="source task (teacher distribution)")
    ax.hist(tid, bins=bins, density=True, alpha=0.55, color=C_ID,
            label=labels[0])
    ax.hist(tood, bins=bins, density=True, alpha=0.55, color=C_OOD,
            label=labels[1])
    if tau is not None:
        ax.axvline(tau, color=INK, lw=1.0, ls="--")
        ax.text(tau, ax.get_ylim()[1] * 0.97, " τ", fontsize=9, color=INK,
                va="top")
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def main() -> None:
    from sb3_contrib import MaskablePPO
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=200)
    fig.patch.set_facecolor("white")
    for c, sc in enumerate(("altgoal", "locked")):
        src_env, tgt_env = SCENARIOS[sc]
        teacher = FrozenMaskableTeacher(
            f"runs/paperv4_{sc}_teacher/final_model.zip")
        student = MaskablePPO.load(STUDENT[sc], device="cpu")
        tag_mode = "goal" if sc == "altgoal" else "key"

        src_obs, _ = collect(src_env, teacher_actor(teacher), N_FRAMES, 11,
                             tag_mode, False)
        tgt_obs, tags = collect(tgt_env, student_actor(student), N_FRAMES,
                                22, tag_mode, False)
        id_sel = tags == 1
        ood_sel = tags == (3 if sc == "altgoal" else 0)

        A_src = activations(teacher, src_obs)
        A_tgt = activations(teacher, tgt_obs)
        tau = calibrate_energy_threshold(teacher, src_env, seed=777,
                                         quantile=Q[sc])
        panel(axes[0, c], A_src["phi"], A_tgt["phi"][id_sel],
              A_tgt["phi"][ood_sel], tau,
              f"{sc} — energy score φ (q={Q[sc]})", LABELS[sc],
              "energy score")

        n_tr = int(len(src_obs) * 0.8)
        sae = train_sae(A_src["L1_pool400"][:n_tr])
        panel(axes[1, c], recon_err(sae, A_src["L1_pool400"][n_tr:]),
              recon_err(sae, A_tgt["L1_pool400"][id_sel]),
              recon_err(sae, A_tgt["L1_pool400"][ood_sel]), None,
              f"{sc} — SAE L1 reconstruction error", LABELS[sc],
              "reconstruction error")
        print(f"[fig3c] {sc} done", flush=True)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Energy / SAE-recon distributions with sub-task condition "
                 "labels (paper Fig. 3c/4c style, v4 teachers)",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = "runs/videos/keynote/fig3c_hist.png"
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
