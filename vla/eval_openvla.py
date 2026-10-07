"""Gate evaluation for the OpenVLA-LoRA teacher: seen vs unseen success.

Loads openvla-7b + the LoRA adapter, registers our norm stats under
"libero_object_seen", and rolls deterministic (greedy) episodes from
LIBERO's fixed init states. The transfer premise requires seen >> unseen.

Run on a GPU node:
    python eval_openvla.py --lora runs_vla/openvla_seen5/lora_ep3 --episodes 10
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from vla_common import MAX_STEPS, SEEN_TASKS, UNSEEN_TASKS, get_init_states, make_env

MODEL_ID = "openvla/openvla-7b"
UNNORM_KEY = "libero_object_seen"


def load_teacher(lora_dir: str | None, norm_path: str, device: str = "cuda"):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa", low_cpu_mem_usage=True).to(device)
    if lora_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir).merge_and_unload()
    with open(norm_path) as f:
        stats = json.load(f)[UNNORM_KEY]["action"]
    # register for predict_action's unnormalization
    model.norm_stats[UNNORM_KEY] = {"action": {
        "q01": np.array(stats["q01"]), "q99": np.array(stats["q99"]),
        "mask": np.array(stats["mask"])}}
    model.eval()
    return model, processor


@torch.no_grad()
def act(model, processor, obs, language: str, device: str = "cuda"):
    img = Image.fromarray(obs["agentview_image"]).resize((224, 224))
    prompt = (f"In: What action should the robot take to "
              f"{language.lower()}?\nOut:")
    inputs = processor(prompt, img).to(device, dtype=torch.bfloat16)
    action = model.predict_action(**inputs, unnorm_key=UNNORM_KEY,
                                  do_sample=False)
    return np.asarray(action, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lora", default=None,
                    help="LoRA dir; omit for base-model control")
    ap.add_argument("--norm-stats", default=None,
                    help="norm_stats.json (default: <lora>/../norm_stats.json)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    a = ap.parse_args()

    norm_path = a.norm_stats or os.path.join(
        os.path.dirname(a.lora.rstrip("/")), "norm_stats.json")
    model, processor = load_teacher(a.lora, norm_path)
    tasks = a.tasks if a.tasks is not None else list(SEEN_TASKS + UNSEEN_TASKS)
    results = {}
    for tid in tasks:
        env = make_env(tid)
        inits = get_init_states(tid)
        succ = 0
        for ep in range(a.episodes):
            env.reset()
            obs = env.set_init_state(inits[ep % len(inits)])
            done, r = False, 0.0
            for _ in range(MAX_STEPS):
                obs, r, done, info = env.step(act(model, processor, obs,
                                                  env.language))
                if done:
                    break
            succ += int(done and r > 0)
        env.close()
        results[tid] = succ / a.episodes
        tag = "seen" if tid in SEEN_TASKS else "UNSEEN"
        print(f"[eval] task {tid} ({tag}): {results[tid]:.0%} | {env.language}",
              flush=True)
    seen = [results[t] for t in tasks if t in SEEN_TASKS]
    uns = [results[t] for t in tasks if t in UNSEEN_TASKS]
    if seen:
        print(f"[eval] SEEN mean:   {np.mean(seen):.0%}")
    if uns:
        print(f"[eval] UNSEEN mean: {np.mean(uns):.0%}")


if __name__ == "__main__":
    main()
