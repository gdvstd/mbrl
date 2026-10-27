"""Rollout-anchored KS probe: SAE trained on the teacher's own SUCCESSFUL
rollouts instead of demos (the gridworld recipe, ported to VLA).

Anchor data = teacher_rollouts.npz (successful seen-task teacher episodes,
collected for BC distillation; images[:, :3] = raw agentview RGB).
Episode-level 80/20 split -> SAE train / F_S calibration. Evaluation is
identical to vla_ks_probe: F_T from teacher rollouts over all 10 tasks
(init-state offset to stay disjoint from the anchor episodes) and from
random-action rollouts (starvation check). Hypothesis: the demo->rollout
shift (KS dist 0.216) disappears, raising seen-task open rates.

Usage (GPU node):
    python vla_ks_probe_ro.py --lora <dir> [--layer 8]
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
from vla_ks_probe import ks_tau  # noqa: E402
from vla_sae_probe import auroc, batched, train_sae  # noqa: E402


@torch.no_grad()
def img_hidden_phi(model, processor, img, lang, layer, device="cuda"):
    prompt = (f"In: What action should the robot take to "
              f"{lang.lower()}?\nOut:")
    pil = Image.fromarray(img).resize((224, 224))
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--norm-stats", default=None)
    ap.add_argument("--rollouts", default="runs_vla/teacher_rollouts.npz")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--anchor-stride", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--init-offset", type=int, default=40,
                    help="init-state offset for F_T episodes (disjoint "
                         "from the anchor rollouts)")
    ap.add_argument("--out", default="runs_vla/ks_probe_vla_ro.npz")
    a = ap.parse_args()
    device = "cuda"
    norm = a.norm_stats or os.path.join(
        os.path.dirname(a.lora.rstrip("/")), "norm_stats.json")
    model, processor = load_teacher(a.lora, norm)
    L = a.layer

    # ---- anchor: successful teacher rollouts (stored agentview frames)
    d = np.load(a.rollouts)
    imgs = d["images"][:, :3].transpose(0, 2, 3, 1)  # (N,128,128,3) uint8
    eps, tids = d["ep_id"], d["task_id"]
    sel = np.zeros(len(imgs), dtype=bool)
    sel[::a.anchor_stride] = True
    print(f"[anchor] {sel.sum()}/{len(imgs)} frames from "
          f"{len(np.unique(eps))} successful episodes", flush=True)
    H = []
    for i in np.flatnonzero(sel):
        h, _ = img_hidden_phi(model, processor, imgs[i],
                              task_language(int(tids[i])), L, device)
        H.append(h)
    H = torch.stack(H)
    eps_s = eps[sel]
    cut = np.quantile(np.unique(eps_s), 0.8)
    sae = train_sae(H[eps_s <= cut], device=device)

    def recon(x):
        return ((batched(sae, x) - x) ** 2).mean(-1).numpy()
    rec_S = recon(H[eps_s > cut])
    print(f"[src] F_S: {len(rec_S)} held-out ROLLOUT frames, median "
          f"{np.median(rec_S):.4f}", flush=True)

    # ---- F_T passes: teacher (all tasks) and random
    results = {}
    rng = np.random.default_rng(0)
    for pol in ("teacher", "random"):
        rec_T, phi_T, tid_T = [], [], []
        for tid in list(SEEN_TASKS) + list(UNSEEN_TASKS):
            env = make_env(tid)
            inits = get_init_states(tid)
            lang = task_language(tid)
            for ep in range(a.episodes):
                env.reset()
                obs = env.set_init_state(
                    inits[(a.init_offset + ep) % len(inits)])
                done = False
                for t in range(MAX_STEPS):
                    if t % a.stride == 0:
                        h, phi = img_hidden_phi(
                            model, processor, obs["agentview_image"], lang,
                            L, device)
                        rec_T.append(recon(h[None])[0])
                        phi_T.append(phi)
                        tid_T.append(tid)
                    action = (rng.uniform(-1, 1, 7) if pol == "random"
                              else act(model, processor, obs, lang))
                    obs, r, done, _ = env.step(action)
                    if done:
                        break
            env.close()
        rec_T = np.array(rec_T)
        tid_T = np.array(tid_T)
        seen = np.isin(tid_T, SEEN_TASKS)
        tau, D = ks_tau(rec_S, rec_T)
        o = rec_T <= tau
        print(f"\n== {pol} visitation (rollout-anchored SAE, L{L}) ==")
        print(f"tau_KS={tau:.4f}  D={D:.3f}")
        print(f"open | seen  : {o[seen].mean():.1%}")
        print(f"open | unseen: {o[~seen].mean():.1%}")
        print(f"AUROC recon (unseen vs seen): "
              f"{auroc(rec_T[~seen], rec_T[seen]):.3f}")
        if pol == "teacher":
            tau_sr, D_sr = ks_tau(rec_S, rec_T[seen])
            print(f"P_ID~P_S check: KS dist(F_S, seen-rollout) = {D_sr:.3f}"
                  f" (demo-anchored was 0.216)")
        results[pol] = dict(rec_T=rec_T, tid_T=tid_T,
                            phi_T=np.array(phi_T), tau=tau, D=D)
    np.savez(a.out, rec_S=rec_S,
             **{f"{p}_{k}": v for p, r in results.items()
                for k, v in r.items()})
    print(f"saved -> {a.out}")
    print("VLA KS-RO PROBE DONE")


if __name__ == "__main__":
    main()
