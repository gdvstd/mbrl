"""SAE-recon-error gate for action advising (the sae_probe finding, used
online): a TopK SAE is trained on the teacher's SOURCE-rollout L1
activations at student-training start; during rollouts the teacher advises
only where the current observation's L1 activation reconstructs well
(recon error <= tau_r). tau_r is the (1-q)-quantile of the source-train
recon errors, matching EBTL's source-side open rate for fair comparison.

Oracle prediction (oracle_advantage_hist): a high-precision / low-recall
gate in altgoal full-obs; in locked full-obs the gate barely opens, so the
run should degrade toward No-Transfer -- worth running as evidence.
"""
from __future__ import annotations

import numpy as np
import torch as th

from fourroom.sae_probe import (
    N_FRAMES, activations, collect, recon_err, teacher_actor, train_sae,
)

L1_IDX = 2  # cnn Sequential index of the conv16+pool tap (see sae_probe)


def build_sae_gate(teacher, src_env_id: str, q: float, ego: bool,
                   seed: int):
    """Returns (score_fn, tau_r). score_fn maps a rollout obs tensor
    (channel-first, same orientation as policy.obs_to_tensor output) to
    the per-env L1 reconstruction error."""
    src_obs, _ = collect(src_env_id, teacher_actor(teacher), N_FRAMES,
                         seed, "goal", ego)
    A_src = activations(teacher, src_obs)
    n_tr = int(len(src_obs) * 0.8)
    sae = train_sae(A_src["L1_pool400"][:n_tr])
    tau_r = float(np.quantile(recon_err(sae, A_src["L1_pool400"][:n_tr]),
                              1.0 - q))
    cnn = teacher.policy.pi_features_extractor.cnn

    @th.no_grad()
    def score_fn(obs_tensor: th.Tensor) -> th.Tensor:
        h = obs_tensor.float()
        for i, layer in enumerate(cnn):
            h = layer(h)
            if i == L1_IDX:
                break
        a = h.flatten(1)
        return ((sae(a) - a) ** 2).mean(-1)

    return score_fn, tau_r
