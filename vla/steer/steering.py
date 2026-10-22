"""Activation steering hooks for OpenVLA (arXiv 2509.00328 style).

Steering overrides selected FFN neuron activations with a fixed scalar
alpha at every forward pass: a forward_pre_hook on each involved layer's
`mlp.down_proj` sets input[..., neuron_ids] = alpha. Applies to all token
positions of every decode step while attached (the paper's full-rollout
intervention).
"""
from __future__ import annotations

import random

import torch


class SteeringHooks:
    """Attach/detach steering on a set of (layer, neuron) targets."""

    def __init__(self, model, entries: list[dict], alpha: float):
        self.alpha = alpha
        lm = getattr(model, "language_model", model)
        self._layers = lm.model.layers
        self._by_layer: dict[int, list[int]] = {}
        for e in entries:
            self._by_layer.setdefault(e["layer"], []).append(e["neuron"])
        self._handles = []

    def _make_hook(self, idx: list[int]):
        ids = torch.tensor(idx)

        def hook(module, args):
            x = args[0]
            x = x.clone()
            x[..., ids.to(x.device)] = self.alpha
            return (x,) + tuple(args[1:])
        return hook

    def __enter__(self):
        for li, idx in self._by_layer.items():
            self._handles.append(
                self._layers[li].mlp.down_proj.register_forward_pre_hook(
                    self._make_hook(idx)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    @property
    def n_neurons(self) -> int:
        return sum(len(v) for v in self._by_layer.values())


def random_entries(model, n: int, seed: int = 0) -> list[dict]:
    """Random (layer, neuron) control matched in size to a concept set."""
    lm = getattr(model, "language_model", model)
    n_layers = len(lm.model.layers)
    d_ff = lm.model.layers[0].mlp.down_proj.in_features
    rng = random.Random(seed)
    return [{"layer": rng.randrange(n_layers), "neuron": rng.randrange(d_ff)}
            for _ in range(n)]
