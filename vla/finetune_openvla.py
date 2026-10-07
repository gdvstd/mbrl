"""LoRA-finetune OpenVLA-7B on the SEEN half of libero_object (hdf5 direct).

Bypasses OpenVLA's RLDS pipeline: reads our LIBERO demo hdf5 files, builds
the model's own prompt format, discretizes actions with the model's
256-bin scheme (last 256 vocab tokens), and trains next-token loss on the
action tokens only, with a PEFT LoRA adapter. Single A100-40GB, bf16.

Normalization: computes q01/q99 per action dim over the training demos and
registers them as norm_stats["libero_object_seen"] so inference can
un-normalize via predict_action(..., unnorm_key="libero_object_seen").

Usage (on a GPU node):
    python finetune_openvla.py --epochs 3 --batch-size 8 --grad-accum 2
"""
from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np
import torch
from PIL import Image

SEEN_TASKS = (0, 1, 2, 3, 4)
MODEL_ID = "openvla/openvla-7b"
ACTION_DIM = 7
N_BINS = 256


def demo_file(task_id: int) -> str:
    from libero.libero import benchmark, get_libero_path
    bm = benchmark.get_benchmark_dict()["libero_object"]()
    return os.path.join(get_libero_path("datasets"),
                        bm.get_task_demonstration(task_id))


def task_language(task_id: int) -> str:
    from libero.libero import benchmark
    bm = benchmark.get_benchmark_dict()["libero_object"]()
    return bm.get_task(task_id).language


def load_dataset(max_demos=None):
    frames = []  # (task_id, rgb128, action7)
    for tid in SEEN_TASKS:
        lang = task_language(tid)
        with h5py.File(demo_file(tid), "r") as f:
            demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            if max_demos:
                demos = demos[:max_demos]
            for d in demos:
                g = f["data"][d]
                rgb = g["obs"]["agentview_rgb"][:]
                act = g["actions"][:]
                for t in range(len(act)):
                    frames.append((lang, rgb[t], act[t].astype(np.float32)))
        print(f"[ft] task {tid}: cumulative {len(frames)} frames", flush=True)
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--llm-only", action="store_true",
                    help="LoRA on the language model only (leave the vision "
                         "tower/projector frozen-clean)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap optimizer steps (for short A/B probes)")
    ap.add_argument("--max-demos", type=int, default=None)
    ap.add_argument("--out", type=str, default="runs_vla/openvla_seen5")
    a = ap.parse_args()

    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import LoraConfig, get_peft_model

    device = "cuda"
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to(device)

    targets = (["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
               if a.llm_only else "all-linear")
    lora = LoraConfig(r=a.lora_r, lora_alpha=min(a.lora_r * 2, 64),
                      lora_dropout=0.0, target_modules=targets,
                      init_lora_weights="gaussian")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    data = load_dataset(a.max_demos)

    # --- action normalization stats (q01/q99, OpenVLA convention) ---------
    acts = np.stack([f[2] for f in data])
    q01, q99 = np.quantile(acts, 0.01, axis=0), np.quantile(acts, 0.99, axis=0)
    os.makedirs(a.out, exist_ok=True)
    norm = {"libero_object_seen": {"action": {
        "q01": q01.tolist(), "q99": q99.tolist(),
        "mask": [True] * 6 + [False]}}}  # gripper dim not normalized
    with open(os.path.join(a.out, "norm_stats.json"), "w") as f:
        json.dump(norm, f)

    tok = processor.tokenizer
    vocab = tok.vocab_size

    # OpenVLA's EXACT pretrained convention (prismatic ActionTokenizer):
    # bins = linspace(-1,1,256); disc = clip(digitize(a, bins), 1, 255);
    # token_id = vocab_size - disc.  predict_action() inverts precisely
    # this, so training MUST match it or decoded actions are garbage.
    BINS = np.linspace(-1, 1, N_BINS)

    def action_to_tokens(action: np.ndarray) -> list[int]:
        na = action.copy()
        for i in range(6):
            na[i] = 2 * (action[i] - q01[i]) / max(q99[i] - q01[i], 1e-8) - 1
        na = np.clip(na, -1, 1)
        disc = np.clip(np.digitize(na, BINS), 1, N_BINS - 1)
        return (vocab - disc).tolist()

    def collate(batch):
        imgs, input_ids, labels = [], [], []
        for lang, rgb, act in batch:
            img = Image.fromarray(rgb).resize((224, 224))
            prompt = f"In: What action should the robot take to {lang.lower()}?\nOut: "
            ids = tok(prompt, add_special_tokens=True).input_ids
            a_ids = action_to_tokens(act) + [tok.eos_token_id]
            input_ids.append(torch.tensor(ids + a_ids))
            labels.append(torch.tensor([-100] * len(ids) + a_ids))
            imgs.append(img)
        pix = processor.image_processor(imgs, return_tensors="pt")["pixel_values"]
        maxlen = max(len(x) for x in input_ids)
        pad = tok.pad_token_id or 0
        ii = torch.full((len(batch), maxlen), pad, dtype=torch.long)
        ll = torch.full((len(batch), maxlen), -100, dtype=torch.long)
        am = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for j, (x, y) in enumerate(zip(input_ids, labels)):
            ii[j, :len(x)] = x; ll[j, :len(y)] = y; am[j, :len(x)] = 1
        return pix, ii, am, ll

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr)
    n = len(data)
    steps_per_epoch = n // a.batch_size
    print(f"[ft] {n} frames, {steps_per_epoch} steps/epoch", flush=True)

    model.train()
    for ep in range(a.epochs):
        perm = np.random.permutation(n)
        tot, cnt = 0.0, 0
        done_steps = 0
        for s in range(steps_per_epoch):
            if a.max_steps and ep * steps_per_epoch + s >= a.max_steps:
                break
            idx = perm[s * a.batch_size:(s + 1) * a.batch_size]
            pix, ii, am, ll = collate([data[i] for i in idx])
            out = model(pixel_values=pix.to(device, torch.bfloat16),
                        input_ids=ii.to(device),
                        attention_mask=am.to(device),
                        labels=ll.to(device))
            (out.loss / a.grad_accum).backward()
            if (s + 1) % a.grad_accum == 0:
                opt.step(); opt.zero_grad()
            tot += out.loss.item(); cnt += 1
            if s % 200 == 0:
                print(f"[ft] ep{ep+1} step {s}/{steps_per_epoch} "
                      f"loss {tot/max(cnt,1):.3f}", flush=True)
                tot, cnt = 0.0, 0
        model.save_pretrained(os.path.join(a.out, f"lora_ep{ep+1}"))
        print(f"[ft] saved epoch {ep+1}", flush=True)
    print("[ft] DONE", flush=True)


if __name__ == "__main__":
    main()
