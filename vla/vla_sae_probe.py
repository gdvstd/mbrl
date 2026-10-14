"""Layer-wise TopK-SAE ID/OOD probe on OpenVLA activations (VLA analog of
fourroom/sae_probe.py).

Activations: the LLM hidden state at the LAST prompt position (the token
whose logits produce the first action token -- the fused image+language
state summary), tapped at several Llama layers. SAEs are trained on
SEEN-task demo frames only (episode-level 80/20 split); scores on held-out
seen (ID) vs unseen-task frames (OOD):

    SAE-recon : reconstruction error (higher = OOD)
    SAE-OOD   : SAID-style cosine to the nearest seen-TASK prototype
                (max over 5 task-mean latents; lower = OOD); also the
                class-free global-mean variant
    baseline  : phi_1 = logsumexp over action-bin logits at that same
                position (one forward gives acts + phi together)

Usage:
    python vla_sae_probe.py --lora <dir> [--frames-per-task 1200]
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vla_common import SEEN_TASKS, UNSEEN_TASKS  # noqa: E402
from eval_openvla import load_teacher  # noqa: E402
from finetune_openvla import demo_file, task_language  # noqa: E402
from vla_energy import EMPTY_TOKEN, N_BINS  # noqa: E402

TAPS = (8, 16, 24, 32)  # Llama hidden_states indices (32 = final layer)


# ------------------------------------------------------------- TopK SAE
# (same recipe as fourroom/sae_probe.py, re-vendored because vla/ stays
# self-contained for cluster deploys)
class TopKSAE(torch.nn.Module):
    def __init__(self, d_in: int, expansion: int = 4, k: int = 64):
        super().__init__()
        d_dict = d_in * expansion
        self.k = k
        self.W_enc = torch.nn.Parameter(torch.randn(d_in, d_dict) * 0.02)
        self.b_enc = torch.nn.Parameter(torch.zeros(d_dict))
        self.W_dec = torch.nn.Parameter(torch.randn(d_dict, d_in) * 0.02)
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))

    def encode(self, x):
        h = (x - self.b_dec) @ self.W_enc + self.b_enc
        topv, topi = torch.topk(h, self.k, dim=-1)
        return torch.zeros_like(h).scatter_(-1, topi, torch.relu(topv))

    def forward(self, x):
        return self.encode(x) @ self.W_dec + self.b_dec

    @torch.no_grad()
    def renorm(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=-1, keepdim=True).clamp(
            1e-8)


def train_sae(acts: torch.Tensor, steps: int = 2000, bs: int = 256,
              device: str = "cuda", seed: int = 0) -> TopKSAE:
    torch.manual_seed(seed)
    sae = TopKSAE(acts.shape[1]).to(device)
    with torch.no_grad():
        sae.b_dec.copy_(acts.mean(0).to(device))
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    n = len(acts)
    for _ in range(steps):
        xb = acts[torch.randint(0, n, (bs,))].to(device)
        loss = torch.nn.functional.mse_loss(sae(xb), xb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.renorm()
    return sae


@torch.no_grad()
def batched(sae_fn, acts, bs=1024, device="cuda"):
    outs = []
    for i in range(0, len(acts), bs):
        outs.append(sae_fn(acts[i:i + bs].to(device)).cpu())
    return torch.cat(outs)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


# ------------------------------------------------------------ activations
@torch.no_grad()
def collect_acts(model, processor, tid: int, n_frames: int, device: str,
                 batch: int = 16):
    """Subsampled demo frames -> per-tap last-position hidden states,
    phi_1, and episode ids."""
    lang = task_language(tid)
    prompt = (f"In: What action should the robot take to "
              f"{lang.lower()}?\nOut:")
    vocab = processor.tokenizer.vocab_size
    bin_ids = torch.arange(vocab - (N_BINS - 1), vocab, device=device)

    imgs, eps = [], []
    with h5py.File(demo_file(tid), "r") as f:
        demos = sorted(f["data"].keys(),
                       key=lambda k: int(k.split("_")[-1]))
        per_ep = max(1, n_frames // len(demos))
        for di, d in enumerate(demos):
            rgb = f["data"][d]["obs"]["agentview_rgb"][:]
            idx = np.linspace(0, len(rgb) - 1,
                              min(per_ep, len(rgb))).astype(int)
            for t in idx:
                imgs.append(rgb[t])
                eps.append(di)
            if len(imgs) >= n_frames:
                break
    acts = {L: [] for L in TAPS}
    phis = []
    for im in imgs:  # per-frame: the OpenVLA processor is single-sample
        pil = Image.fromarray(im).resize((224, 224))
        inputs = processor(prompt, pil).to(device, dtype=torch.bfloat16)
        ids = inputs["input_ids"]
        if ids[0, -1] != EMPTY_TOKEN:
            pad = torch.full((1, 1), EMPTY_TOKEN, device=ids.device,
                             dtype=ids.dtype)
            inputs["input_ids"] = torch.cat([ids, pad], dim=1)
            am = inputs["attention_mask"]
            inputs["attention_mask"] = torch.cat(
                [am, torch.ones_like(pad)], dim=1)
        out = model(**inputs, output_hidden_states=True)
        for L in TAPS:
            acts[L].append(out.hidden_states[L][0, -1].float().cpu())
        phis.append(torch.logsumexp(
            out.logits[0, -1, bin_ids].float(), -1).item())
    return ({L: torch.stack(v) for L, v in acts.items()},
            np.array(phis), np.array(eps))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--norm-stats", default=None)
    ap.add_argument("--frames-per-task", type=int, default=1200)
    ap.add_argument("--out", default="runs_vla/sae_probe_vla.npz")
    a = ap.parse_args()
    device = "cuda"
    norm = a.norm_stats or os.path.join(
        os.path.dirname(a.lora.rstrip("/")), "norm_stats.json")
    model, processor = load_teacher(a.lora, norm)

    data = {}
    for tid in list(SEEN_TASKS) + list(UNSEEN_TASKS):
        data[tid] = collect_acts(model, processor, tid, a.frames_per_task,
                                 device)
        print(f"[acts] task {tid}: {len(data[tid][1])} frames", flush=True)

    # episode-level 80/20 split of the seen frames
    tr_sel, te_sel = {}, {}
    for tid in SEEN_TASKS:
        eps = data[tid][2]
        cut = np.quantile(np.unique(eps), 0.8)
        tr_sel[tid], te_sel[tid] = eps <= cut, eps > cut

    phi_id = np.concatenate([data[t][1][te_sel[t]] for t in SEEN_TASKS])
    phi_ood = np.concatenate([data[t][1] for t in UNSEEN_TASKS])
    print(f"\nphi_1 baseline  AUROC(ID vs OOD): "
          f"{auroc(phi_id, phi_ood):.3f}", flush=True)

    results = {}
    for L in TAPS:
        A_tr = torch.cat([data[t][0][L][tr_sel[t]] for t in SEEN_TASKS])
        A_id = torch.cat([data[t][0][L][te_sel[t]] for t in SEEN_TASKS])
        A_ood = torch.cat([data[t][0][L] for t in UNSEEN_TASKS])
        sae = train_sae(A_tr, device=device)

        def recon_err(x):
            return ((batched(sae, x) - x) ** 2).mean(-1).numpy()
        e_id, e_ood = recon_err(A_id), recon_err(A_ood)
        a_recon = auroc(e_ood, e_id)

        z_tr = batched(sae.encode, A_tr)
        protos = torch.stack([
            batched(sae.encode, data[t][0][L][tr_sel[t]]).mean(0)
            for t in SEEN_TASKS])
        protos = protos / protos.norm(dim=-1, keepdim=True).clamp(1e-8)
        glob = z_tr.mean(0)
        glob = glob / glob.norm().clamp(1e-8)

        def cos_scores(x):
            z = batched(sae.encode, x)
            z = z / z.norm(dim=-1, keepdim=True).clamp(1e-8)
            return ((z @ protos.T).max(-1).values.numpy(),
                    (z @ glob).numpy())
        cmax_id, cglob_id = cos_scores(A_id)
        cmax_ood, cglob_ood = cos_scores(A_ood)
        a_cmax = auroc(-cmax_ood, -cmax_id)
        a_cglob = auroc(-cglob_ood, -cglob_id)
        print(f"layer {L:2d}:  recon {a_recon:.3f}   "
              f"SAID-task {a_cmax:.3f}   SAID-glob {a_cglob:.3f}",
              flush=True)
        results[L] = dict(recon_id=e_id, recon_ood=e_ood, cmax_id=cmax_id,
                          cmax_ood=cmax_ood)

    np.savez(a.out, phi_id=phi_id, phi_ood=phi_ood,
             **{f"L{L}_{k}": v for L, r in results.items()
                for k, v in r.items()})
    print(f"saved -> {a.out}")
    print("VLA SAE PROBE DONE")


if __name__ == "__main__":
    main()
