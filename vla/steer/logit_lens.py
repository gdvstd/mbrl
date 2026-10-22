"""Logit-lens concept library for OpenVLA FFN value vectors.

Following arXiv 2509.00328: each column of a Llama layer's
`mlp.down_proj.weight` (d_model x d_ff) is a "value vector" whose meaning
is read by projecting it through the LM head onto the vocabulary. A neuron
is aligned with a concept keyword iff a token spelling that keyword ranks
inside the neuron's top-K projected tokens.

Output: a concept library JSON
    {concept: [{"layer": l, "neuron": j, "token": str, "rank": r,
                "logit": v}, ...]}   (sorted by rank, best first)
plus, for the best few neurons per concept, their full top-30 token list
for manual inspection.

Usage (GPU node):
    python steer/logit_lens.py --out runs_vla/steer/lens_base.json
    python steer/logit_lens.py --lora runs_vla/openvla_seen5_v3/lora_ep4 \
        --out runs_vla/steer/lens_lora.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONCEPTS = ["fast", "slow", "up", "down", "left", "right", "push", "pull",
            "open", "close", "grasp", "grab", "lift", "place", "stop",
            "forward", "back"]
TOP_K = 30        # a neuron "means" a concept if the token is in its top-30
PER_CONCEPT = 40  # keep at most this many neurons per concept
DUMP_TOP = 5      # neurons per concept to dump full top-30 tokens for


def concept_token_ids(tokenizer, word: str) -> list[int]:
    """All vocab ids whose decoded string is exactly the word (case/space
    variants), e.g. 'fast', ' fast', 'Fast'."""
    ids = []
    vocab = tokenizer.get_vocab()
    for tok, i in vocab.items():
        t = tok.replace("▁", "").strip().lower()
        if t == word.lower():
            ids.append(i)
    return ids


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--norm-stats", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concepts", nargs="*", default=CONCEPTS)
    a = ap.parse_args()

    if a.lora:
        norm = a.norm_stats or os.path.join(
            os.path.dirname(a.lora.rstrip("/")), "norm_stats.json")
        from eval_openvla import load_teacher
        model, processor = load_teacher(a.lora, norm)
        tokenizer = processor.tokenizer
    else:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        processor = AutoProcessor.from_pretrained(
            "openvla/openvla-7b", trust_remote_code=True)
        tokenizer = processor.tokenizer
        model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b", torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa",
            low_cpu_mem_usage=True).to("cuda").eval()

    lm = getattr(model, "language_model", model)
    W_lm = lm.lm_head.weight.float()            # (V, d_model)
    layers = lm.model.layers
    cids = {c: concept_token_ids(tokenizer, c) for c in a.concepts}
    for c, ids in cids.items():
        print(f"[lens] concept '{c}': {len(ids)} token variants {ids[:6]}")

    library: dict[str, list[dict]] = {c: [] for c in a.concepts}
    dumps: dict[str, list[dict]] = {}
    for li, layer in enumerate(layers):
        W_down = layer.mlp.down_proj.weight.float()      # (d_model, d_ff)
        logits = W_lm @ W_down                           # (V, d_ff)
        # per-neuron top-K threshold over the vocab axis
        thresh = torch.topk(logits, TOP_K, dim=0).values[-1]   # (d_ff,)
        for c, ids in cids.items():
            if not ids:
                continue
            cl = logits[ids]                             # (n_var, d_ff)
            best_logit, best_var = cl.max(dim=0)
            hit = best_logit >= thresh                   # token in top-K
            for j in torch.nonzero(hit).flatten().tolist():
                rank = int((logits[:, j] > best_logit[j]).sum().item()) + 1
                library[c].append({
                    "layer": li, "neuron": j,
                    "token": tokenizer.convert_ids_to_tokens(
                        ids[int(best_var[j].item())]),
                    "rank": rank, "logit": float(best_logit[j])})
        del logits, W_down
        torch.cuda.empty_cache()
        print(f"[lens] layer {li}: "
              + ", ".join(f"{c}:{len(library[c])}" for c in a.concepts),
              flush=True)

    for c in a.concepts:
        library[c].sort(key=lambda e: (e["rank"], -e["logit"]))
        library[c] = library[c][:PER_CONCEPT]
        dumps[c] = []
        for e in library[c][:DUMP_TOP]:
            w = layers[e["layer"]].mlp.down_proj.weight[:, e["neuron"]]
            top = torch.topk(W_lm @ w.float(), TOP_K).indices.tolist()
            dumps[c].append({**e, "top_tokens":
                             tokenizer.convert_ids_to_tokens(top)})
        print(f"[lens] '{c}': kept {len(library[c])} neurons, best "
              + str([(e['layer'], e['neuron'], e['rank'])
                     for e in library[c][:3]]))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"library": library, "inspect": dumps,
                   "top_k": TOP_K, "lora": a.lora}, f, indent=1)
    print(f"saved -> {a.out}")
    print("LENS DONE")


if __name__ == "__main__":
    main()
