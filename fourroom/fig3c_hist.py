"""Fig-3c/4c style histograms with sub-task condition labels, v4 teachers.

Three distributions per scenario (source / target-ID-cond / target-OOD
cond) for: energy phi, SAE recon error at EACH layer (L1/L2/L3), and the
SAID action-cosine at EACH layer plus the 3-layer average. Every strip is annotated with how many
frames from how many episodes it contains.

Outputs:
  fig3c_hist.png        overlay view (5 scores x 2 scenarios)
  fig3c_hist_split.png  one strip per distribution (readable overlap)
Scores cached in runs/fig3c_scores.npz (delete to recompute).

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
from fourroom.said_probe import (
    build_prototypes, cos_sim, encode_np, masked_argmax_actions,
)

INK, MUTED, GRID = "#1f2430", "#5c6370", "#e6e8ec"
C_SRC, C_ID, C_OOD = "#4c86c6", "#f0a05a", "#9a6bd1"
Q = {"altgoal": 0.5, "locked": 0.7}
STUDENT = {"altgoal": "runs/paperv4_altgoal_ebtl_seed0/best_model.zip",
           "locked": "runs/paperv4_locked_aa_seed0/best_model.zip"}
LABELS = {
    "altgoal": ("target, goal in Room 1 (shared sub-task)",
                "target, goal in Room 3 (non-shared)"),
    "locked": ("target, key obtained (known navigation)",
               "target, before key (novel sub-task)"),
}
SCORES = (("phi", "energy score φ", "energy score"),
          ("recL1", "SAE recon error — L1", "reconstruction error"),
          ("recL2", "SAE recon error — L2", "reconstruction error"),
          ("recL3", "SAE recon error — L3", "reconstruction error"),
          ("cosL1", "SAID action-cosine — L1", "cosine to action prototype"),
          ("cosL2", "SAID action-cosine — L2", "cosine to action prototype"),
          ("cosL3", "SAID action-cosine — L3", "cosine to action prototype"),
          ("cos", "SAID action-cosine — 3-layer avg",
           "cosine to action prototype"))
LKEY = {"recL1": "L1_pool400", "recL2": "L2_conv512", "recL3": "L3_flat576"}


def compute_scores() -> dict:
    cache = "runs/fig3c_scores.npz"
    if _os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    from sb3_contrib import MaskablePPO
    out = {}
    for sc in ("altgoal", "locked"):
        src_env, tgt_env = SCENARIOS[sc]
        teacher = FrozenMaskableTeacher(
            f"runs/paperv4_{sc}_teacher/final_model.zip")
        student = MaskablePPO.load(STUDENT[sc], device="cpu")
        tag_mode = "goal" if sc == "altgoal" else "key"

        src_obs, _, src_masks, src_eps = collect(
            src_env, teacher_actor(teacher), N_FRAMES, 11, tag_mode, False,
            return_masks=True, return_eps=True)
        tgt_obs, tags, tgt_masks, tgt_eps = collect(
            tgt_env, student_actor(student), N_FRAMES, 22, tag_mode, False,
            return_masks=True, return_eps=True)
        id_sel = tags == 1
        ood_sel = tags == (3 if sc == "altgoal" else 0)

        n_tr = int(len(src_obs) * 0.8)
        out[f"{sc}_counts"] = np.array([
            len(src_obs) - n_tr, len(np.unique(src_eps[n_tr:])),
            int(id_sel.sum()), len(np.unique(tgt_eps[id_sel])),
            int(ood_sel.sum()), len(np.unique(tgt_eps[ood_sel]))])

        A_src = activations(teacher, src_obs)
        A_tgt = activations(teacher, tgt_obs)
        out[f"{sc}_tau"] = np.array(
            calibrate_energy_threshold(teacher, src_env, seed=777,
                                       quantile=Q[sc]))
        out[f"{sc}_phi_src"] = A_src["phi"][n_tr:]
        out[f"{sc}_phi_id"] = A_tgt["phi"][id_sel]
        out[f"{sc}_phi_ood"] = A_tgt["phi"][ood_sel]

        act_src = masked_argmax_actions(teacher, src_obs, src_masks)
        act_tgt = masked_argmax_actions(teacher, tgt_obs, tgt_masks)
        cos = {"src": np.zeros(len(src_obs) - n_tr),
               "id": np.zeros(int(id_sel.sum())),
               "ood": np.zeros(int(ood_sel.sum()))}
        for skey, lname in LKEY.items():
            sae = train_sae(A_src[lname][:n_tr])
            out[f"{sc}_{skey}_src"] = recon_err(sae, A_src[lname][n_tr:])
            out[f"{sc}_{skey}_id"] = recon_err(sae, A_tgt[lname][id_sel])
            out[f"{sc}_{skey}_ood"] = recon_err(sae, A_tgt[lname][ood_sel])
            z_tr = encode_np(sae, A_src[lname][:n_tr])
            protos, _ = build_prototypes(z_tr, act_src[:n_tr])
            ckey = "cos" + skey[3:]  # recL1 -> cosL1
            cl = {"src": cos_sim(encode_np(sae, A_src[lname][n_tr:]),
                                 protos[act_src[n_tr:]]),
                  "id": cos_sim(encode_np(sae, A_tgt[lname][id_sel]),
                                protos[act_tgt[id_sel]]),
                  "ood": cos_sim(encode_np(sae, A_tgt[lname][ood_sel]),
                                 protos[act_tgt[ood_sel]])}
            for grp in ("src", "id", "ood"):
                out[f"{sc}_{ckey}_{grp}"] = cl[grp]
                cos[grp] += cl[grp] / 3
        for grp in ("src", "id", "ood"):
            out[f"{sc}_cos_{grp}"] = cos[grp]
        print(f"[fig3c] {sc} computed", flush=True)
    np.savez(cache, **out)
    return out


def groups(d, sc, key):
    vals = [d[f"{sc}_{key}_src"], d[f"{sc}_{key}_id"], d[f"{sc}_{key}_ood"]]
    cnt = d[f"{sc}_counts"]
    names = [
        f"source task  (n={cnt[0]:,} / {cnt[1]} eps, held-out)",
        f"{LABELS[sc][0]}  (n={cnt[2]:,} / {cnt[3]} eps)",
        f"{LABELS[sc][1]}  (n={cnt[4]:,} / {cnt[5]} eps)"]
    return vals, names


def main() -> None:
    d = compute_scores()

    # overlay view
    fig, axes = plt.subplots(len(SCORES), 2, figsize=(12, 22), dpi=200)
    fig.patch.set_facecolor("white")
    for c, sc in enumerate(("altgoal", "locked")):
        for r, (key, name, xlabel) in enumerate(SCORES):
            vals, names = groups(d, sc, key)
            ax = axes[r, c]
            lo = min(np.percentile(v, 0.5) for v in vals)
            hi = max(np.percentile(v, 99.5) for v in vals)
            bins = np.linspace(lo, hi, 48)
            for v, nm, col in zip(vals, names, (C_SRC, C_ID, C_OOD)):
                ax.hist(v, bins=bins, density=True, alpha=0.55, color=col,
                        label=nm)
            if key == "phi":
                ax.axvline(float(d[f"{sc}_tau"]), color=INK, lw=1.0,
                           ls="--")
            ax.set_title(f"{sc} — {name}", fontsize=10, color=INK)
            ax.set_xlabel(xlabel, fontsize=8.5, color=INK)
            ax.set_yticks([])
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=7.5)
            if r == 0:
                ax.legend(frameon=False, fontsize=6.5)
    fig.suptitle("Sub-task condition labels (paper Fig. 3c/4c style, "
                 "v4 teachers)", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig("runs/videos/keynote/fig3c_hist.png", facecolor="white")
    print("wrote runs/videos/keynote/fig3c_hist.png")

    # split view: one strip per distribution
    fig = plt.figure(figsize=(13.5, 26), dpi=200)
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(len(SCORES), 2, hspace=0.5, wspace=0.1,
                             top=0.955, bottom=0.03, left=0.04, right=0.99)
    for c, sc in enumerate(("altgoal", "locked")):
        for r, (key, name, xlabel) in enumerate(SCORES):
            vals, names = groups(d, sc, key)
            lo = min(np.percentile(v, 0.5) for v in vals)
            hi = max(np.percentile(v, 99.5) for v in vals)
            bins = np.linspace(lo, hi, 50)
            inner = outer[r, c].subgridspec(3, 1, hspace=0.15)
            tau = float(d[f"{sc}_tau"]) if key == "phi" else None
            for i, (v, nm, col) in enumerate(
                    zip(vals, names, (C_SRC, C_ID, C_OOD))):
                ax = fig.add_subplot(inner[i])
                ax.hist(v, bins=bins, density=True, color=col, alpha=0.85)
                if tau is not None:
                    ax.axvline(tau, color=INK, lw=0.9, ls="--")
                ax.text(0.995, 0.75, nm, transform=ax.transAxes, ha="right",
                        fontsize=7, color=col, weight="bold")
                ax.set_yticks([])
                ax.set_xlim(lo, hi)
                for sp in ("top", "right", "left"):
                    ax.spines[sp].set_visible(False)
                ax.spines["bottom"].set_color(GRID)
                if i < 2:
                    ax.set_xticks([])
                    ax.spines["bottom"].set_visible(False)
                else:
                    ax.tick_params(colors=MUTED, labelsize=7)
                    ax.set_xlabel(xlabel, fontsize=8, color=INK)
                if i == 0:
                    ttl = f"{sc} — {name}"
                    if tau is not None:
                        ttl += f"   (τ dashed, q={Q[sc]})"
                    ax.set_title(ttl, fontsize=10, color=INK)
    fig.suptitle("Split view: one strip per distribution, with frame / "
                 "episode counts", fontsize=12.5, color=INK)
    fig.savefig("runs/videos/keynote/fig3c_hist_split.png",
                facecolor="white")
    print("wrote runs/videos/keynote/fig3c_hist_split.png")


if __name__ == "__main__":
    main()
