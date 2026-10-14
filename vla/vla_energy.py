"""Energy score phi for OpenVLA: logsumexp over the action-token logits.

The gridworld EBTL gate uses phi(s) = logsumexp(action logits). OpenVLA's
"action logits" live at the 7 autoregressive action-token positions, each a
distribution over the last 255 vocab ids (token = vocab_size - disc,
disc in 1..255 -- same convention as finetune_openvla). We define

    phi_d(s) = logsumexp_{t in action-bin tokens} logits_d[t]
    phi(s)   = mean_d phi_d(s)          (7 positions)

computed from a single greedy generate(output_scores=True) that ALSO yields
the action, so the gate costs nothing beyond the teacher forward the
advising step already pays. Decoding is validated against predict_action.

Probe (main): roll the teacher over all 10 tasks (2 episodes each, fixed
init states), record per-step phi, and report per-task means + the
seen-vs-unseen frame AUROC -- the VLA analog of the gridworld src-vs-ood
check. Saves runs_vla/phi_probe.npz.

Usage:
    python vla_energy.py --lora <dir> [--episodes 2]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vla_common import (  # noqa: E402
    MAX_STEPS, SEEN_TASKS, UNSEEN_TASKS, get_init_states, make_env,
)
from eval_openvla import UNNORM_KEY, act, load_teacher  # noqa: E402

N_BINS = 256
EMPTY_TOKEN = 29871  # predict_action appends this before generating


def _prepare_inputs(processor, obs, language: str, device: str):
    img = Image.fromarray(obs["agentview_image"]).resize((224, 224))
    prompt = (f"In: What action should the robot take to "
              f"{language.lower()}?\nOut:")
    inputs = processor(prompt, img).to(device, dtype=torch.bfloat16)
    ids = inputs["input_ids"]
    if ids[0, -1] != EMPTY_TOKEN:
        pad = torch.tensor([[EMPTY_TOKEN]], device=ids.device,
                           dtype=ids.dtype)
        inputs["input_ids"] = torch.cat([ids, pad], dim=1)
        am = inputs["attention_mask"]
        inputs["attention_mask"] = torch.cat(
            [am, torch.ones_like(pad)], dim=1)
    return inputs


@torch.no_grad()
def phi_and_action(model, processor, obs, language: str,
                   device: str = "cuda"):
    """One generate pass -> (phi, 7-dim unnormalized action)."""
    inputs = _prepare_inputs(processor, obs, language, device)
    vocab = processor.tokenizer.vocab_size
    out = model.generate(**inputs, max_new_tokens=7, do_sample=False,
                         output_scores=True, return_dict_in_generate=True)
    bin_ids = torch.arange(vocab - (N_BINS - 1), vocab, device=device)
    phis = [torch.logsumexp(s[0, bin_ids].float(), -1).item()
            for s in out.scores]
    tokens = out.sequences[0, -7:].cpu().numpy()
    disc = vocab - tokens  # 1..255
    centers = (np.linspace(-1, 1, N_BINS)[:-1]
               + np.linspace(-1, 1, N_BINS)[1:]) / 2
    norm = centers[np.clip(disc, 1, N_BINS - 1) - 1]
    st = model.norm_stats[UNNORM_KEY]["action"]
    action = np.where(st["mask"],
                      0.5 * (norm + 1) * (st["q99"] - st["q01"]) + st["q01"],
                      norm)
    return float(np.mean(phis)), action.astype(np.float64)


def task_language(tid: int) -> str:
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()["libero_object"]().get_task(
        tid).language


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--norm-stats", default=None)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--out", default="runs_vla/phi_probe.npz")
    a = ap.parse_args()
    norm = a.norm_stats or os.path.join(
        os.path.dirname(a.lora.rstrip("/")), "norm_stats.json")
    model, processor = load_teacher(a.lora, norm)

    # decode validation: our generate must reproduce predict_action exactly
    env = make_env(SEEN_TASKS[0])
    env.reset()
    obs = env.set_init_state(get_init_states(SEEN_TASKS[0])[0])
    lang = task_language(SEEN_TASKS[0])
    for i in range(3):
        ref = act(model, processor, obs, lang)
        phi, ours = phi_and_action(model, processor, obs, lang)
        err = np.abs(ref - ours).max()
        print(f"[validate] frame {i}: max|diff| {err:.2e}  phi {phi:.2f}")
        assert err < 1e-4, "decode mismatch vs predict_action"
        obs, *_ = env.step(ref)
    env.close()

    phis, tids = [], []
    for tid in list(SEEN_TASKS) + list(UNSEEN_TASKS):
        env = make_env(tid)
        inits = get_init_states(tid)
        lang = task_language(tid)
        for ep in range(a.episodes):
            env.reset()
            obs = env.set_init_state(inits[ep % len(inits)])
            for _ in range(MAX_STEPS):
                phi, action = phi_and_action(model, processor, obs, lang)
                phis.append(phi)
                tids.append(tid)
                obs, r, done, _ = env.step(action)
                if done:
                    break
        env.close()
        sel = np.array(tids) == tid
        tag = "seen" if tid in SEEN_TASKS else "UNSEEN"
        print(f"[phi] task {tid} ({tag}): mean {np.mean(np.array(phis)[sel]):.2f} "
              f"+/- {np.std(np.array(phis)[sel]):.2f} "
              f"({sel.sum()} frames)", flush=True)

    phis, tids = np.array(phis), np.array(tids)
    seen = np.isin(tids, SEEN_TASKS)
    print(f"\n[phi] seen frames  : {phis[seen].mean():.2f} +/- "
          f"{phis[seen].std():.2f}")
    print(f"[phi] unseen frames: {phis[~seen].mean():.2f} +/- "
          f"{phis[~seen].std():.2f}")
    print(f"[phi] AUROC(seen vs unseen frames): "
          f"{auroc(phis[seen], phis[~seen]):.3f}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, phi=phis, task_id=tids)
    print(f"saved -> {a.out}")
    print("PHI PROBE DONE")


if __name__ == "__main__":
    main()
