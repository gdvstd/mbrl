"""SAID-style OOD scoring vs energy phi and SAE reconstruction error.

Adapts SAID (Karmacharya et al., "Sparse Autoencoders for Interpretable
Out-of-Distribution Detection") to the four-room teachers. SAID scores a
test input by the cosine similarity between its SAE latent code and the
mean ID latent of the *predicted class*; here the policy's masked argmax
ACTION plays the role of the class ("act" variant). We also run a
class-agnostic variant against the single global ID mean ("glob"), and
report per-layer scores plus the paper's uniform layer average.

For each of the 4 plain teachers (altgoal, locked) x (full-obs, ego):
  - collect source rollouts (80/20 train/held-out) + target rollouts
  - per tap layer (L1/L2/L3): train TopK SAE on source-train activations,
    build per-action prototypes (fallback to global mean when an action
    has < MIN_PROTO_COUNT source frames) and the global prototype
  - AUROC (src-vs-ood and hard id-vs-ood) for: phi, recon error,
    SAID-act {L1,L2,L3,avg}, SAID-glob {L1,L2,L3,avg}

Scores are cached to runs/said_scores_<cfg>.npz; figures go to
runs/videos/keynote/said_comparison.png (AUROC bars) and
runs/videos/keynote/said_hist.png (score histograms, hard split).

Usage: python fourroom/said_probe.py
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fourroom.sae_probe import (
    N_FRAMES, TAPS, activations, auroc, collect, recon_err, student_actor,
    teacher_actor, train_sae,
)
from fourroom.fourroom_env import SCENARIOS
from fourroom.paper_common import FrozenMaskableTeacher

MIN_PROTO_COUNT = 25
N_ACTIONS = 7
CFGS = [("altgoal", False), ("locked", False),
        ("altgoal", True), ("locked", True)]
LAYERS = list(TAPS)  # ["L1_pool400", "L2_conv512", "L3_flat576"]

INK, MUTED, GRID = "#1f2430", "#5c6370", "#e6e8ec"
C_ID, C_OOD = "#2a78d6", "#eb6834"
KEYNOTE = "runs/videos/keynote"


# ---------------------------------------------------------------- scoring
@th.no_grad()
def masked_argmax_actions(teacher, obs_batch, masks) -> np.ndarray:
    """Predicted 'class' per frame = argmax of the masked raw logits."""
    pol = teacher.policy
    ot, _ = pol.obs_to_tensor(obs_batch)
    feats = pol.extract_features(ot)
    if not pol.share_features_extractor:
        feats = feats[0]
    logits = pol.action_net(pol.mlp_extractor.forward_actor(feats))
    logits = logits.masked_fill(~th.as_tensor(masks, dtype=th.bool), -1e9)
    return logits.argmax(-1).cpu().numpy()


@th.no_grad()
def encode_np(sae, acts: np.ndarray) -> np.ndarray:
    return sae.encode(th.as_tensor(acts, dtype=th.float32)).numpy()


def build_prototypes(z_train, actions_train):
    """Per-action mean latents; rare actions fall back to the global mean."""
    glob = z_train.mean(0)
    protos = np.tile(glob, (N_ACTIONS, 1))
    for a in range(N_ACTIONS):
        sel = actions_train == a
        if sel.sum() >= MIN_PROTO_COUNT:
            protos[a] = z_train[sel].mean(0)
    return protos, glob


def cos_sim(z, mu):
    num = (z * mu).sum(-1)
    den = (np.linalg.norm(z, axis=-1).clip(1e-8)
           * np.linalg.norm(mu, axis=-1).clip(1e-8))
    return num / den


# ---------------------------------------------------------------- per config
def probe_config(scenario: str, ego: bool) -> dict[str, np.ndarray]:
    from sb3_contrib import MaskablePPO
    sfx = "_ego" if ego else ""
    cache = f"runs/said_scores_{scenario}{sfx}.npz"
    if _os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}

    src_env, tgt_env = SCENARIOS[scenario]
    teacher = FrozenMaskableTeacher(
        f"runs/paper_{scenario}_teacher{sfx}/final_model.zip")
    student = MaskablePPO.load(
        f"runs/paper_{scenario}{sfx}_ebtl_seed0/best_model.zip", device="cpu")
    tag_mode = "goal" if scenario == "altgoal" else "key"

    src_obs, _, src_masks = collect(src_env, teacher_actor(teacher), N_FRAMES,
                                    11, tag_mode, ego, return_masks=True)
    tgt_obs, tags, tgt_masks = collect(tgt_env, student_actor(student),
                                       N_FRAMES, 22, tag_mode, ego,
                                       return_masks=True)
    id_sel = tags == 1
    ood_sel = tags == (3 if scenario == "altgoal" else 0)

    A_src = activations(teacher, src_obs)
    A_tgt = activations(teacher, tgt_obs)
    act_src = masked_argmax_actions(teacher, src_obs, src_masks)
    act_tgt = masked_argmax_actions(teacher, tgt_obs, tgt_masks)
    n_tr = int(len(src_obs) * 0.8)

    out = {
        "phi_src": A_src["phi"][n_tr:],
        "phi_id": A_tgt["phi"][id_sel],
        "phi_ood": A_tgt["phi"][ood_sel],
    }
    for name in LAYERS:
        sae = train_sae(A_src[name][:n_tr])
        out[f"recon_{name}_src"] = recon_err(sae, A_src[name][n_tr:])
        out[f"recon_{name}_id"] = recon_err(sae, A_tgt[name][id_sel])
        out[f"recon_{name}_ood"] = recon_err(sae, A_tgt[name][ood_sel])

        z_tr = encode_np(sae, A_src[name][:n_tr])
        protos, glob = build_prototypes(z_tr, act_src[:n_tr])
        for split, z, acts in (
                ("src", encode_np(sae, A_src[name][n_tr:]), act_src[n_tr:]),
                ("id", encode_np(sae, A_tgt[name][id_sel]), act_tgt[id_sel]),
                ("ood", encode_np(sae, A_tgt[name][ood_sel]), act_tgt[ood_sel])):
            out[f"act_{name}_{split}"] = cos_sim(z, protos[acts])
            out[f"glob_{name}_{split}"] = cos_sim(z, glob[None])
        print(f"[said] {scenario}{sfx} {name} done", flush=True)

    # SAID's uniform layer average (same frames per split, so plain mean)
    for kind in ("act", "glob"):
        for split in ("src", "id", "ood"):
            out[f"{kind}_avg_{split}"] = np.mean(
                [out[f"{kind}_{n}_{split}"] for n in LAYERS], axis=0)
    np.savez(cache, **out)
    return out


def report(name: str, s: dict[str, np.ndarray]) -> dict[str, float]:
    """Print the AUROC table; return {method: id-vs-ood AUROC} for plotting.

    OOD-score direction: phi and cosine are LOWER on OOD (negate);
    recon error is HIGHER on OOD.
    """
    rows: dict[str, float] = {}
    print(f"\n== {name}: AUROC (src-vs-ood | id-vs-ood) ==")

    def line(label, ood, src, idd, flip):
        sgn = -1.0 if flip else 1.0
        a1 = auroc(sgn * ood, sgn * src)
        a2 = auroc(sgn * ood, sgn * idd)
        print(f"{label:22s}  {a1:.3f} | {a2:.3f}")
        rows[label] = a2

    line("phi", s["phi_ood"], s["phi_src"], s["phi_id"], flip=True)
    for n in LAYERS:
        line(f"recon {n[:2]}", s[f"recon_{n}_ood"], s[f"recon_{n}_src"],
             s[f"recon_{n}_id"], flip=False)
    for kind, lab in (("act", "SAID-act"), ("glob", "SAID-glob")):
        for n in LAYERS + ["avg"]:
            short = "avg" if n == "avg" else n[:2]
            line(f"{lab} {short}", s[f"{kind}_{n}_ood"],
                 s[f"{kind}_{n}_src"], s[f"{kind}_{n}_id"], flip=True)
    return rows


# ---------------------------------------------------------------- figures
def bar_figure(all_rows: dict[str, dict[str, float]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=200)
    fig.patch.set_facecolor("white")
    for ax, (cfg, rows) in zip(axes.flat, all_rows.items()):
        labels = list(rows)
        vals = [rows[k] for k in labels]
        colors = []
        for k in labels:
            if k == "phi":
                colors.append("#9aa1ad")
            elif k.startswith("recon"):
                colors.append("#7fb069")
            elif k.startswith("SAID-act"):
                colors.append(C_ID)
            else:
                colors.append(C_OOD)
        x = np.arange(len(labels))
        ax.bar(x, vals, color=colors)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7,
                           color=INK)
        ax.set_title(cfg, fontsize=11, color=INK)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=6,
                    color=INK)
    fig.suptitle("Hard-split ID/OOD AUROC: phi vs SAE recon vs SAID cosine",
                 fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{KEYNOTE}/said_comparison.png", facecolor="white")
    print(f"wrote {KEYNOTE}/said_comparison.png")


def hist_panel(ax, id_s, ood_s, title):
    lo = min(np.percentile(id_s, 0.5), np.percentile(ood_s, 0.5))
    hi = max(np.percentile(id_s, 99.5), np.percentile(ood_s, 99.5))
    bins = np.linspace(lo, hi, 40)
    ax.hist(id_s, bins=bins, density=True, alpha=0.55, color=C_ID,
            label="ID-cond")
    ax.hist(ood_s, bins=bins, density=True, alpha=0.55, color=C_OOD,
            label="OOD-cond")
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7)


def hist_figure(all_scores: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(11, 10), dpi=200)
    fig.patch.set_facecolor("white")
    for r, (cfg, s) in enumerate(all_scores.items()):
        a_phi = auroc(-s["phi_ood"], -s["phi_id"])
        a_act = auroc(-s["act_avg_ood"], -s["act_avg_id"])
        a_glb = auroc(-s["glob_avg_ood"], -s["glob_avg_id"])
        hist_panel(axes[r, 0], s["phi_id"], s["phi_ood"],
                   f"{cfg} — energy φ   AUROC {a_phi:.2f}")
        hist_panel(axes[r, 1], s["act_avg_id"], s["act_avg_ood"],
                   f"{cfg} — SAID-act avg   AUROC {a_act:.2f}")
        hist_panel(axes[r, 2], s["glob_avg_id"], s["glob_avg_ood"],
                   f"{cfg} — SAID-glob avg   AUROC {a_glb:.2f}")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Hard-split separation: energy φ vs SAID cosine "
                 "(action-conditional / global prototype)",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{KEYNOTE}/said_hist.png", facecolor="white")
    print(f"wrote {KEYNOTE}/said_hist.png")


def main() -> None:
    _os.makedirs(KEYNOTE, exist_ok=True)
    all_scores, all_rows = {}, {}
    for scenario, ego in CFGS:
        cfg = f"{scenario}{'_ego' if ego else ''}"
        s = probe_config(scenario, ego)
        all_scores[cfg] = s
        all_rows[cfg] = report(cfg, s)
    bar_figure(all_rows)
    hist_figure(all_scores)
    print("SAID PROBE ALL DONE")


if __name__ == "__main__":
    main()
