"""Lab-meeting assets: cropped oracle-advantage panels + result tables.

Outputs (runs/videos/keynote/):
  adv_hist_4panel.png   2x2 crop of the oracle-advantage histograms
                        (altgoal-EBTL, altgoal-recon, locked-EBTL,
                        locked_ego-EBTL), log-y, from the oracle_val caches
  table_gate_stats.png  AUROC +/- CI and gate-conditioned P(helpful) + z
  table_methods.png     7-method final/AUC, 3 seeds, both scenarios
  table_oracle.png      trained vs oracle teacher (aa/jsrl/ksrl)

Usage: python fourroom/keynote_assets.py
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#1f2430", "#5c6370", "#e6e8ec"
C_OPEN, C_CLOSED = "#2a78d6", "#eb6834"
KEYNOTE = "runs/videos/keynote"
GAMMA = 0.9


def auroc_ci(pos, neg):
    s = np.concatenate([pos, neg])
    r = s.argsort(kind="mergesort").argsort().astype(float)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1, n2 = len(pos), len(neg)
    a = (r[:n1].sum() + n1 - n1 * (n1 + 1) / 2) / (n1 * n2)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a)
                  + (n2 - 1) * (q2 - a * a)) / (n1 * n2))
    return a, 1.96 * se


def gate_rows():
    rows = []
    for cfg in ("altgoal", "locked", "altgoal_ego", "locked_ego"):
        d = np.load(f"runs/oracle_val_{cfg}.npz")
        helpful = (d["ds"] - d["ds2"]) == 1
        for gate, score, hi_id, tau in (
                ("EBTL φ", d["phi"], True, float(d["tau_phi"])),
                ("SAE recon", d["recon"], False, float(d["tau_recon"])),
                ("SAID cos", d["said"], True, float(d["tau_said"]))):
            sc = score if hi_id else -score
            a, ci = auroc_ci(sc[helpful], sc[~helpful])
            open_ = (score >= tau) if hi_id else (score <= tau)
            if open_.sum() == 0:
                rows.append((cfg, gate, f"{a:.2f}±{ci:.2f}", "—",
                             f"{helpful[~open_].mean():.2f}", "—", "—"))
                continue
            p1, p2 = helpful[open_].mean(), helpful[~open_].mean()
            p = helpful.mean()
            se = np.sqrt(p * (1 - p) * (1 / open_.sum() + 1 / (~open_).sum()))
            z = (p1 - p2) / se
            rows.append((cfg, gate, f"{a:.2f}±{ci:.2f}", f"{p1:.2f}",
                         f"{p2:.2f}", f"{p1 - p2:+.2f}", f"{z:+.0f}"))
    return rows


def render_table(path, header, rows, col_w, highlight=None, title=""):
    n_r, n_c = len(rows) + 1, len(header)
    fig_h = 0.42 * n_r + (0.5 if title else 0.15)
    fig, ax = plt.subplots(figsize=(sum(col_w), fig_h), dpi=200)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=header, loc="center",
                   cellLoc="center",
                   colWidths=[w / sum(col_w) for w in col_w])
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.55)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.get_text().set_color(INK)
        if r == 0:
            cell.set_facecolor("#f2f4f7")
            cell.get_text().set_weight("bold")
        elif highlight and highlight(r - 1, rows[r - 1]):
            cell.set_facecolor("#eaf2fc")
    if title:
        ax.set_title(title, fontsize=11, color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def panel(ax, dv_open, dv_closed, title):
    both = np.concatenate([dv_open, dv_closed]) if len(dv_open) else dv_closed
    lo, hi = np.percentile(both, 0.5), np.percentile(both, 99.5)
    pad = 0.05 * (hi - lo + 1e-9)
    bins = np.linspace(lo - pad, hi + pad, 45)
    if len(dv_open):
        ax.hist(dv_open, bins=bins, density=True, alpha=0.6, color=C_OPEN,
                label="advised (gate open)", log=True)
    ax.hist(dv_closed, bins=bins, density=True, alpha=0.6, color=C_CLOSED,
            label="not advised (gate closed)", log=True)
    ax.axvline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_title(title, fontsize=11, color=INK)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def adv_4panel():
    picks = (("altgoal", "phi", True, "tau_phi", "altgoal — EBTL φ gate"),
             ("altgoal", "recon", False, "tau_recon",
              "altgoal — SAE recon gate"),
             ("locked", "phi", True, "tau_phi", "locked — EBTL φ gate"),
             ("locked_ego", "phi", True, "tau_phi",
              "locked_ego — EBTL φ gate"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), dpi=200)
    fig.patch.set_facecolor("white")
    for ax, (cfg, key, hi_id, tkey, title) in zip(axes.flat, picks):
        d = np.load(f"runs/oracle_val_{cfg}.npz")
        dV = GAMMA ** d["ds2"] - GAMMA ** (d["ds"] - 1)
        score, tau = d[key], float(d[tkey])
        open_ = (score >= tau) if hi_id else (score <= tau)
        vo = dV[open_].mean() if open_.any() else float("nan")
        vc = dV[~open_].mean()
        panel(ax, dV[open_], dV[~open_],
              f"{title}   (open {open_.mean():.0%},  Ā {vo:+.3f} vs "
              f"{vc:+.3f})")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Oracle advantage of teacher advice at gate-open vs "
                 "gate-closed states  (0 = optimal advice)",
                 fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{KEYNOTE}/adv_hist_4panel.png", facecolor="white")
    plt.close(fig)
    print(f"wrote {KEYNOTE}/adv_hist_4panel.png")


def eval_stats(sc, tr):
    fins, aucs = [], []
    for s in range(3):
        for p in (f"runs/paper_{sc}_{tr}_seed{s}/evaluations.npz",
                  f"runs/paper_{sc}_{tr}_seed{s}/eval/evaluations.npz"):
            if _os.path.exists(p):
                r = np.load(p)["results"].mean(axis=1)
                fins.append(r[-3:].mean())
                aucs.append(r.mean())
                break
    if not fins:
        return None
    return (f"{np.mean(fins):.2f}±{np.std(fins):.2f}", f"{np.mean(aucs):.2f}")


def main() -> None:
    _os.makedirs(KEYNOTE, exist_ok=True)
    adv_4panel()

    render_table(
        f"{KEYNOTE}/table_gate_stats.png",
        ["config", "gate", "AUROC(helpful)", "P(h|open)", "P(h|closed)",
         "diff", "z"],
        gate_rows(), [1.5, 1.3, 1.5, 1.1, 1.2, 0.9, 0.7],
        highlight=lambda i, r: r[5] != "—" and r[5].startswith("+"),
        title="Does the gate select helpful advice?  (oracle labels, "
              "n=6000; blue rows: significantly positive)")

    methods = ("scratch", "finetune", "aa", "ebtl", "gap", "gapnc", "sae")
    names = {"scratch": "No transfer", "finetune": "Fine-tune",
             "aa": "AA", "ebtl": "EBTL", "gap": "Gap gate",
             "gapnc": "Gap (no coin)", "sae": "SAE gate"}
    rows = []
    for tr in methods:
        a, l = eval_stats("altgoal", tr), eval_stats("locked", tr)
        rows.append((names[tr], a[0], a[1], l[0], l[1]))
    render_table(
        f"{KEYNOTE}/table_methods.png",
        ["method", "altgoal final", "altgoal AUC", "locked final",
         "locked AUC"],
        rows, [1.7, 1.4, 1.2, 1.4, 1.2],
        highlight=lambda i, r: r[0] in ("EBTL", "Gap gate", "SAE gate"),
        title="Gating methods, 3 seeds (full-obs)")

    rows = []
    for real, orac in (("aa", "oaa"), ("jsrl", "ojsrl"), ("ksrl", "oksrl")):
        ra, oa = eval_stats("altgoal", real), eval_stats("altgoal", orac)
        rl, ol = eval_stats("locked", real), eval_stats("locked", orac)
        rows.append((real.upper(), ra[1], oa[1], rl[1], ol[1]))
    render_table(
        f"{KEYNOTE}/table_oracle.png",
        ["method", "altgoal AUC\n(teacher)", "altgoal AUC\n(oracle)",
         "locked AUC\n(teacher)", "locked AUC\n(oracle)"],
        rows, [1.2, 1.5, 1.5, 1.5, 1.5],
        title="Same schedule, perfect advice: trained vs oracle teacher "
              "(3 seeds)")
    print("ASSETS DONE")


if __name__ == "__main__":
    main()
