"""KS-threshold probe on OpenVLA-LIBERO: does the two-sample KS tau give a
useful ID/OOD separation in the VLA domain?

Mirrors the gridworld budgeted-KS calibration:
  F_S  = SAE recon error on HELD-OUT seen-task demo frames (the source
         calibration sample; SAE trained on the other 80% of seen demos)
  F_T  = recon on frames from TEACHER ROLLOUTS over all 10 tasks (the
         target-mixture visitation at advising time; ID = seen tasks,
         OOD = unseen -- ground-truth labels available!)
  tau_KS = argmax[F_S - F_T],  D = max gap

Reported:
  * per-task gate-open rates at tau_KS (should be high seen / ~0 unseen)
  * frame AUROC (recon, rollout seen vs unseen) and the same for phi
  * the P_ID ~ P_S check: KS distance between F_S (demo) and the
    seen-task ROLLOUT recon distribution -- if large, the demo-anchored
    tau misses teacher-competent rollout states (the gridworld tube
    phenomenon, now measurable at 7B scale)
  * phi baseline: same KS procedure on -phi

Usage (GPU node):
    python vla_ks_probe.py --lora <dir> [--layer 8] [--episodes 2]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vla_common import MAX_STEPS, SEEN_TASKS, UNSEEN_TASKS, get_init_states, make_env  # noqa: E402
from eval_openvla import act, load_teacher  # noqa: E402
from vla_energy import EMPTY_TOKEN, N_BINS, task_language  # noqa: E402
from vla_sae_probe import auroc, batched, collect_acts, train_sae  # noqa: E402


@torch.no_grad()
def frame_act_phi(model, processor, obs, lang, layer, device="cuda"):
    """One forward: last-prompt-position hidden state at `layer` + phi_1."""
    prompt = (f"In: What action should the robot take to "
              f"{lang.lower()}?\nOut:")
    pil = Image.fromarray(obs["agentview_image"]).resize((224, 224))
    inputs = processor(prompt, pil).to(device, dtype=torch.bfloat16)
    ids = inputs["input_ids"]
    if ids[0, -1] != EMPTY_TOKEN:
        pad = torch.full((1, 1), EMPTY_TOKEN, device=ids.device,
                         dtype=ids.dtype)
        inputs["input_ids"] = torch.cat([ids, pad], dim=1)
        am = inputs["attention_mask"]
        inputs["attention_mask"] = torch.cat([am, torch.ones_like(pad)],
                                             dim=1)
    out = model(**inputs, output_hidden_states=True)
    vocab = processor.tokenizer.vocab_size
    bin_ids = torch.arange(vocab - (N_BINS - 1), vocab, device=device)
    phi = torch.logsumexp(out.logits[0, -1, bin_ids].float(), -1).item()
    return out.hidden_states[layer][0, -1].float().cpu(), phi


def ks_tau(src: np.ndarray, tgt: np.ndarray):
    s, t = np.sort(src), np.sort(tgt)
    cand = np.sort(np.concatenate([s, t]))
    gap = (np.searchsorted(s, cand, side="right") / len(s)
           - np.searchsorted(t, cand, side="right") / len(t))
    i = int(np.argmax(gap))
    return float(cand[i]), float(gap[i])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--norm-stats", default=None)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--frames-per-task", type=int, default=600)
    ap.add_argument("--stride", type=int, default=3,
                    help="record every k-th rollout step")
    ap.add_argument("--out", default="runs_vla/ks_probe_vla.npz")
    a = ap.parse_args()
    device = "cuda"
    norm = a.norm_stats or os.path.join(
        os.path.dirname(a.lora.rstrip("/")), "norm_stats.json")
    model, processor = load_teacher(a.lora, norm)
    L = a.layer

    # ---- source side: SAE on seen demos, F_S on held-out episodes
    acts_tr, acts_cal = [], []
    for tid in SEEN_TASKS:
        A, _, eps = collect_acts(model, processor, tid, a.frames_per_task,
                                 device)
        cut = np.quantile(np.unique(eps), 0.8)
        acts_tr.append(A[L][eps <= cut])
        acts_cal.append(A[L][eps > cut])
        print(f"[demo] task {tid}: {len(eps)} frames", flush=True)
    sae = train_sae(torch.cat(acts_tr), device=device)

    def recon(x):
        return ((batched(sae, x) - x) ** 2).mean(-1).numpy()
    rec_S = recon(torch.cat(acts_cal))
    print(f"[src] F_S: {len(rec_S)} held-out seen demo frames, "
          f"median {np.median(rec_S):.4f}", flush=True)

    # ---- target side: teacher rollouts over ALL tasks
    rec_T, phi_T, tid_T = [], [], []
    for tid in list(SEEN_TASKS) + list(UNSEEN_TASKS):
        env = make_env(tid)
        inits = get_init_states(tid)
        lang = task_language(tid)
        for ep in range(a.episodes):
            env.reset()
            obs = env.set_init_state(inits[ep % len(inits)])
            done = False
            for t in range(MAX_STEPS):
                if t % a.stride == 0:
                    h, phi = frame_act_phi(model, processor, obs, lang, L,
                                           device)
                    rec_T.append(recon(h[None])[0])
                    phi_T.append(phi)
                    tid_T.append(tid)
                obs, r, done, _ = env.step(act(model, processor, obs, lang))
                if done:
                    break
        env.close()
        sel = np.array(tid_T) == tid
        tag = "seen" if tid in SEEN_TASKS else "UNSEEN"
        print(f"[roll] task {tid} ({tag}): {sel.sum()} frames, recon med "
              f"{np.median(np.array(rec_T)[sel]):.4f}", flush=True)
    rec_T = np.array(rec_T)
    phi_T = np.array(phi_T)
    tid_T = np.array(tid_T)
    seen = np.isin(tid_T, SEEN_TASKS)

    # ---- KS threshold + separation quality
    tau, D = ks_tau(rec_S, rec_T)
    open_ = rec_T <= tau
    print(f"\n== KS gate (recon, layer {L}) ==")
    print(f"tau_KS={tau:.4f}  D={D:.3f}")
    print(f"open | seen-task rollout frames : {open_[seen].mean():.1%}")
    print(f"open | unseen-task frames       : {open_[~seen].mean():.1%}")
    print(f"frame AUROC recon (unseen vs seen): "
          f"{auroc(rec_T[~seen], rec_T[seen]):.3f}")
    for tid in list(SEEN_TASKS) + list(UNSEEN_TASKS):
        sel = tid_T == tid
        tag = "seen" if tid in SEEN_TASKS else "UNSN"
        print(f"  task {tid} ({tag}): open {open_[sel].mean():5.1%}")

    # ---- P_ID ~ P_S check: demo-calibration vs seen-task ROLLOUT recon
    tau_sr, D_sr = ks_tau(rec_S, rec_T[seen])
    print(f"\n== P_ID ~ P_S check ==")
    print(f"KS distance(F_S demo, F_seen-rollout) = {D_sr:.3f} "
          f"(0 = identical; large = demo tube misses rollout states)")
    print(f"median recon: demo cal {np.median(rec_S):.4f} | seen-rollout "
          f"{np.median(rec_T[seen]):.4f} | unseen-rollout "
          f"{np.median(rec_T[~seen]):.4f}")

    # ---- phi baseline (higher = ID -> negate for the same KS machinery)
    # phi source sample: demo frames' phi -- reuse rollout seen as proxy is
    # circular; use the teacher-rollout seen frames as the phi reference.
    tau_p, D_p = ks_tau(-phi_T[seen], -phi_T)
    open_p = -phi_T <= tau_p
    print(f"\n== phi baseline (KS on -phi, seen-rollout as reference) ==")
    print(f"D={D_p:.3f}  open|seen {open_p[seen].mean():.1%}  "
          f"open|unseen {open_p[~seen].mean():.1%}  AUROC "
          f"{auroc(-phi_T[~seen], -phi_T[seen]):.3f}")

    np.savez(a.out, rec_S=rec_S, rec_T=rec_T, phi_T=phi_T, tid_T=tid_T,
             tau=tau, D=D, tau_sr=tau_sr, D_sr=D_sr, layer=L)
    print(f"saved -> {a.out}")
    print("VLA KS PROBE DONE")


if __name__ == "__main__":
    main()
