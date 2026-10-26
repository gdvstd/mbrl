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
                   seed: int, layer: str = "L1", tau_mult: float = 2.0,
                   ks_once_env: str | None = None):
    """Returns (score_fn, tau_r). The SAE is trained on source-rollout
    activations; tau_r mirrors the author-confirmed EBTL calibration:
    the (1-q)-quantile of recon error over the frames of SUCCESSFUL
    episodes from 100 fresh stochastic teacher rollouts (disjoint from
    the SAE training data, so no overfit-tight threshold)."""
    import gymnasium as gym
    from fourroom.fourroom_env import action_mask
    from fourroom.paper_common import _wrap

    key, idx = LAYER_KEY[layer], LAYER_IDX[layer]
    src_obs, _ = collect(src_env_id, teacher_actor(teacher), N_FRAMES,
                         seed, "goal", ego)
    A_src = activations(teacher, src_obs)
    sae = train_sae(A_src[key])

    # calibration: 100 stochastic episodes, successful frames only
    kw = {"agent_view_size": 9} if ego else {}
    env = _wrap(gym.make(src_env_id, **kw), ego=ego)
    act = teacher_actor(teacher)
    cal_obs, n_succ = [], 0
    for ep in range(100):
        obs, _ = env.reset(seed=seed + 5000 + ep)
        ep_obs, r = [], 0.0
        while True:
            ep_obs.append(obs.copy())
            obs, r, te, tr, _ = env.step(act(obs, action_mask(env)))
            if te or tr:
                break
        if r > 0:
            cal_obs.extend(ep_obs)
            n_succ += 1
    env.close()
    A_cal = activations(teacher, np.stack(cal_obs))
    rec_cal = recon_err(sae, A_cal[key])
    # The in-distribution recon errors are DEGENERATE (the SAE covers the
    # tiny source state space almost exactly), so a within-source quantile
    # collapses to float noise. Use a gap rule instead: open wherever the
    # reconstruction is within 2x of the worst quality ever observed on
    # successful source frames (the ID/OOD recon gap is >5x in practice).
    tau_r = max(float(np.quantile(rec_cal, 0.999)) * tau_mult, 1e-6)
    print(f"[sae-gate] calibration: {n_succ}/100 successful episodes, "
          f"{len(cal_obs)} frames, cal p99.9={np.quantile(rec_cal, 0.999):.6f}, "
          f"tau_r={tau_r:.6f} (gap rule x{tau_mult:g}; q unused)")

    if ks_once_env is not None:
        # KS-once: one pre-training two-sample calibration. Target sample =
        # masked-uniform random rollouts (the paper's ereg OOD-set recipe),
        # SAME size as the source calibration sample; tau = argmax of the
        # ECDF gap F_S - F_T (Youden-optimal under the PU mixture,
        # pi-independent). No online knobs.
        rng = np.random.default_rng(seed + 9000)

        def rand_act(obs, mask):
            return int(rng.choice(np.flatnonzero(np.asarray(mask))))

        tenv = _wrap(gym.make(ks_once_env, **kw), ego=ego)
        tgt_obs = []
        ep = 0
        while len(tgt_obs) < len(rec_cal):
            obs, _ = tenv.reset(seed=seed + 9000 + ep)
            while len(tgt_obs) < len(rec_cal):
                tgt_obs.append(obs.copy())
                obs, r, te, tr, _ = tenv.step(
                    rand_act(obs, action_mask(tenv)))
                if te or tr:
                    break
            ep += 1
        tenv.close()
        rec_tgt = recon_err(sae, activations(teacher, np.stack(tgt_obs))[key])
        src_s = np.sort(rec_cal)
        tgt_s = np.sort(rec_tgt)
        cand = np.sort(np.concatenate([src_s, tgt_s]))
        gap = (np.searchsorted(src_s, cand, side="right") / len(src_s)
               - np.searchsorted(tgt_s, cand, side="right") / len(tgt_s))
        i = int(np.argmax(gap))
        tau_r = float(cand[i])
        print(f"[sae-gate] KS-once: {len(rec_cal)} src vs {len(tgt_obs)} tgt"
              f" frames ({ep} random episodes), tau_ks={tau_r:.6f} "
              f"D={gap[i]:.3f}")
    else:
        rec_tgt = None
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

    if ks_once_env is not None:
        return score_fn, tau_r, rec_cal, rec_tgt
    return score_fn, tau_r, rec_cal


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
