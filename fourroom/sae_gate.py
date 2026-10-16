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

LAYER_IDX = {"L1": 2, "L2": 4, "L3": 7}  # cnn Sequential indices
LAYER_KEY = {"L1": "L1_pool400", "L2": "L2_conv512", "L3": "L3_flat576"}


def build_sae_gate(teacher, src_env_id: str, q: float, ego: bool,
                   seed: int, layer: str = "L1"):
    """Returns (score_fn, tau_r). score_fn maps a rollout obs tensor
    (channel-first, same orientation as policy.obs_to_tensor output) to
    the per-env L1 reconstruction error."""
    key, idx = LAYER_KEY[layer], LAYER_IDX[layer]
    src_obs, _ = collect(src_env_id, teacher_actor(teacher), N_FRAMES,
                         seed, "goal", ego)
    A_src = activations(teacher, src_obs)
    n_tr = int(len(src_obs) * 0.8)
    sae = train_sae(A_src[key][:n_tr])
    tau_r = float(np.quantile(recon_err(sae, A_src[key][:n_tr]), 1.0 - q))
    cnn = teacher.policy.pi_features_extractor.cnn

    @th.no_grad()
    def score_fn(obs_tensor: th.Tensor) -> th.Tensor:
        h = obs_tensor.float()
        for i, mod in enumerate(cnn):
            h = mod(h)
            if i == idx:
                break
        a = h.flatten(1)
        return ((sae(a) - a) ** 2).mean(-1)

    return score_fn, tau_r


def build_said_gate(teacher, src_env_id: str, q: float, ego: bool,
                    seed: int):
    """SAID-act gate: mean over the three layers of the cosine between the
    state's sparse code and the prototype of the teacher's argmax action
    (fourroom/said_probe conventions). Gate opens on cos >= tau (higher
    is ID). Oracle analysis predicts ANTI-selection in full-obs -- this
    online run is the predicted-negative control."""
    from fourroom.said_probe import (
        build_prototypes, cos_sim, encode_np, masked_argmax_actions)
    src_obs, _, src_masks = collect(src_env_id, teacher_actor(teacher),
                                    N_FRAMES, seed, "goal", ego,
                                    return_masks=True)
    A_src = activations(teacher, src_obs)
    act_src = masked_argmax_actions(teacher, src_obs, src_masks)
    n_tr = int(len(src_obs) * 0.8)
    saes, protos_l, cos_tr = {}, {}, []
    for lname, key in LAYER_KEY.items():
        sae = train_sae(A_src[key][:n_tr])
        z_tr = encode_np(sae, A_src[key][:n_tr])
        protos, _ = build_prototypes(z_tr, act_src[:n_tr])
        saes[lname], protos_l[lname] = sae, protos
        cos_tr.append(cos_sim(z_tr, protos[act_src[:n_tr]]))
    tau_c = float(np.quantile(np.mean(cos_tr, axis=0), q))
    pol = teacher.policy
    cnn = pol.pi_features_extractor.cnn

    @th.no_grad()
    def score_fn(obs_tensor: th.Tensor) -> th.Tensor:
        feats = pol.extract_features(obs_tensor)
        if not pol.share_features_extractor:
            feats = feats[0]
        logits = pol.action_net(pol.mlp_extractor.forward_actor(feats))
        acts = logits.argmax(-1).cpu().numpy()
        h = obs_tensor.float()
        taps = {}
        for i, mod in enumerate(cnn):
            h = mod(h)
            for lname, idx in LAYER_IDX.items():
                if i == idx:
                    taps[lname] = h.flatten(1).cpu().numpy()
        scores = []
        for lname in LAYER_KEY:
            z = encode_np(saes[lname], taps[lname])
            scores.append(cos_sim(z, protos_l[lname][acts]))
        return th.as_tensor(np.mean(scores, axis=0))

    return score_fn, tau_c
