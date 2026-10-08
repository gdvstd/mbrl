"""Oracle value-gain histograms: advised vs non-advised steps, per gate.

For every target state visited by the student, the teacher's advice
(masked argmax action) has an oracle value gain

    dD = D(s) - D(step(s, a_T))   (+1 optimal, 0 wasted step, <0 regression)

Three gating rules decide where advice would be issued, all matched to the
same source-side open rate (tau from the source-train split, q = 0.5
altgoal / 0.7 locked):
    EBTL phi    : open iff phi >= Quantile_q(phi_src)
    SAE recon   : open iff recon <= Quantile_{1-q}(recon_src)
    SAID cosine : open iff cos >= Quantile_q(cos_src)   (act-cond, layer avg)

Figure: 4 configs x 3 gates, grouped bars of the dD distribution for
gate-open (advised) vs gate-closed (not advised) states
-> runs/videos/keynote/oracle_gate_hist.png
Cache: runs/oracle_hist_<cfg>.npz

Usage: python fourroom/oracle_hist.py
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fourroom.fourroom_env import SCENARIOS
from fourroom.paper_common import FrozenMaskableTeacher
from fourroom.oracle_advice import (
    EBTL_Q, collect_oracle, regrets, teacher_logits,
)
from fourroom.sae_probe import (
    N_FRAMES, TAPS, activations, collect, recon_err, student_actor,
    teacher_actor, train_sae,
)
from fourroom.said_probe import (
    build_prototypes, cos_sim, encode_np, masked_argmax_actions,
)

CFGS = [("altgoal", False), ("locked", False),
        ("altgoal", True), ("locked", True)]
LAYERS = list(TAPS)
GATES = ("EBTL φ", "SAE recon", "SAID cosine")

INK, MUTED, GRID = "#1f2430", "#5c6370", "#e6e8ec"
C_OPEN, C_CLOSED = "#2a78d6", "#eb6834"
KEYNOTE = "runs/videos/keynote"
BUCKETS = (1, 0, -1, -2, -3)  # last bucket = "<= -3"


def probe_config(scenario: str, ego: bool) -> dict[str, np.ndarray]:
    from sb3_contrib import MaskablePPO
    sfx = "_ego" if ego else ""
    cache = f"runs/oracle_hist_{scenario}{sfx}.npz"
    if _os.path.exists(cache):
        d = np.load(cache)
        return {k: d[k] for k in d.files}

    src_env, tgt_env = SCENARIOS[scenario]
    teacher = FrozenMaskableTeacher(
        f"runs/paper_{scenario}_teacher{sfx}/final_model.zip")
    student = MaskablePPO.load(
        f"runs/paper_{scenario}{sfx}_ebtl_seed0/best_model.zip", device="cpu")
    q = EBTL_Q[scenario]

    # target rollouts + oracle value gain of the teacher's advice
    tgt_obs, tgt_masks, states, lay_keys, layouts = collect_oracle(
        tgt_env, student_actor(student), N_FRAMES, 22, ego)
    logits = teacher_logits(teacher, tgt_obs)
    masked = logits.masked_fill(~th.as_tensor(tgt_masks), -1e9)
    adv_a = masked.argmax(-1).numpy()
    phi = th.logsumexp(logits, dim=-1).numpy()
    dD = np.full(len(states), np.nan)
    for i, (s, lk) in enumerate(zip(states, lay_keys)):
        lay, D = layouts[lk]
        r = regrets(lay, D, s)[adv_a[i]]
        if not np.isnan(r):
            dD[i] = 1.0 - r

    # source rollouts: SAE training + thresholds
    tag_mode = "goal" if scenario == "altgoal" else "key"
    src_obs, _, src_masks = collect(src_env, teacher_actor(teacher), N_FRAMES,
                                    11, tag_mode, ego, return_masks=True)
    A_src, A_tgt = activations(teacher, src_obs), activations(teacher, tgt_obs)
    act_src = masked_argmax_actions(teacher, src_obs, src_masks)
    act_tgt = masked_argmax_actions(teacher, tgt_obs, tgt_masks)
    n_tr = int(len(src_obs) * 0.8)

    recon_src = recon_tgt = None
    cos_src = np.zeros(n_tr)
    cos_tgt = np.zeros(len(tgt_obs))
    for name in LAYERS:
        sae = train_sae(A_src[name][:n_tr])
        if name == "L1_pool400":
            recon_src = recon_err(sae, A_src[name][:n_tr])
            recon_tgt = recon_err(sae, A_tgt[name])
        z_tr = encode_np(sae, A_src[name][:n_tr])
        protos, _ = build_prototypes(z_tr, act_src[:n_tr])
        cos_src += cos_sim(z_tr, protos[act_src[:n_tr]]) / len(LAYERS)
        cos_tgt += cos_sim(encode_np(sae, A_tgt[name]),
                           protos[act_tgt]) / len(LAYERS)
        print(f"[hist] {scenario}{sfx} {name} done", flush=True)

    keep = ~np.isnan(dD)
    out = dict(
        dD=dD[keep], phi=phi[keep], recon=recon_tgt[keep], said=cos_tgt[keep],
        tau_phi=np.array(np.quantile(A_src["phi"][:n_tr], q)),
        tau_recon=np.array(np.quantile(recon_src, 1 - q)),
        tau_said=np.array(np.quantile(cos_src, q)))
    np.savez(cache, **out)
    return out


def bucketize(dd: np.ndarray) -> np.ndarray:
    """Fraction of steps per dD bucket (last bucket catches <= -3)."""
    if len(dd) == 0:
        return np.zeros(len(BUCKETS))
    counts = np.array([
        (dd >= b).sum() if b == BUCKETS[0] else
        ((dd <= b).sum() if b == BUCKETS[-1] else (dd == b).sum())
        for b in BUCKETS], dtype=float)
    return counts / len(dd)


def panel(ax, dd_open, dd_closed, title):
    x = np.arange(len(BUCKETS))
    w = 0.38
    ax.bar(x - w / 2, bucketize(dd_open), w, color=C_OPEN,
           label="advised (gate open)")
    ax.bar(x + w / 2, bucketize(dd_closed), w, color=C_CLOSED,
           label="not advised (gate closed)")
    ax.set_xticks(x)
    ax.set_xticklabels(["+1", "0", "−1", "−2", "≤−3"], fontsize=8, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=9, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7)


def main() -> None:
    _os.makedirs(KEYNOTE, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(11.5, 11), dpi=200, sharey=True)
    fig.patch.set_facecolor("white")
    for r, (scenario, ego) in enumerate(CFGS):
        cfg = f"{scenario}{'_ego' if ego else ''}"
        s = probe_config(scenario, ego)
        gates = (
            s["phi"] >= s["tau_phi"],
            s["recon"] <= s["tau_recon"],
            s["said"] >= s["tau_said"],
        )
        for c, (gname, sel) in enumerate(zip(GATES, gates)):
            mean_o = s["dD"][sel].mean() if sel.any() else float("nan")
            mean_c = s["dD"][~sel].mean() if (~sel).any() else float("nan")
            panel(axes[r, c], s["dD"][sel], s["dD"][~sel],
                  f"{cfg} — {gname}  (open {sel.mean():.0%}, "
                  f"ΔD̄ {mean_o:.2f} vs {mean_c:.2f})")
        print(f"[hist] {cfg} plotted", flush=True)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].set_ylabel("fraction of steps", fontsize=9, color=INK)
    fig.suptitle("Oracle value gain ΔD of teacher advice: gate-open vs "
                 "gate-closed states (+1 = optimal step)",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(f"{KEYNOTE}/oracle_gate_hist.png", facecolor="white")
    print(f"wrote {KEYNOTE}/oracle_gate_hist.png")
    print("ORACLE HIST ALL DONE")


if __name__ == "__main__":
    main()
