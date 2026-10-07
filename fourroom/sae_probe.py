"""Layer-wise TopK-SAE probe: does a sparse dictionary separate ID/OOD
states better than the raw energy score phi?

For one plain (non-ereg) teacher: tap three actor-tower activations
(L1 conv16+pool 400d, L2 conv32 512d, L3 flatten 576d), train a TopK
sparse autoencoder (OpenAI Gao et al. 2024 style; dict 8x, no L1 tuning)
on SOURCE-rollout activations only, then score held-out states by
reconstruction error. Report AUROC for
  (a) source vs target-OOD-condition   (easy separation)
  (b) target-ID-cond vs target-OOD-cond (the split the gate actually needs)
against the phi = logsumexp(logits) baseline.

ID/OOD condition labels: altgoal -> goal in Room 1 vs Room 3;
locked -> post-key vs pre-key.

Usage:  python fourroom/sae_probe.py --scenario altgoal [--ego]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse

import gymnasium as gym
import numpy as np
import torch as th

from fourroom.fourroom_env import SCENARIOS, SIZE, action_mask
from fourroom.paper_common import FrozenMaskableTeacher, _wrap

TAPS = {"L1_pool400": 2, "L2_conv512": 4, "L3_flat576": 7}  # cnn Sequential idx
N_FRAMES = 6000


# ---------------------------------------------------------------- collection
def goal_room(u) -> int:
    for x in range(SIZE):
        for y in range(SIZE):
            c = u.grid.get(x, y)
            if c is not None and c.type == "goal":
                return 1 if x < 5 else 3
    return 0


def collect(env_id, actor, n_frames, seed, tag_mode, ego):
    kw = {"agent_view_size": 11} if ego else {}
    env = _wrap(gym.make(env_id, **kw), ego=ego)
    u = env.unwrapped
    obs, _ = env.reset(seed=seed)
    groom = goal_room(u) if tag_mode == "goal" else 0
    obs_buf, tags = [], []
    for _ in range(n_frames):
        obs_buf.append(obs.copy())
        tags.append(groom if tag_mode == "goal" else (1 if u.carrying else 0))
        a = actor(obs, action_mask(env))
        obs, _, te, tr, _ = env.step(a)
        if te or tr:
            obs, _ = env.reset()
            groom = goal_room(u) if tag_mode == "goal" else 0
    env.close()
    return np.stack(obs_buf), np.array(tags)


def teacher_actor(t):
    def act(obs, mask):
        ot, _ = t.policy.obs_to_tensor(obs)
        return int(t.distribution(ot, action_masks=mask[None])
                   .get_actions(deterministic=False)[0])
    return act


def student_actor(model):
    def act(obs, mask):
        a, _ = model.predict(obs, action_masks=mask, deterministic=False)
        return int(a)
    return act


@th.no_grad()
def activations(teacher, obs_batch) -> dict[str, np.ndarray]:
    """Run obs through the actor tower, tapping TAPS; also return phi."""
    pol = teacher.policy
    ot, _ = pol.obs_to_tensor(obs_batch)
    feats = pol.extract_features(ot)
    if not pol.share_features_extractor:
        feats = feats[0]  # unused; taps re-run the pi extractor below
    cnn = pol.pi_features_extractor.cnn
    x = pol.obs_to_tensor(obs_batch)[0].float()
    outs, taps = {}, dict(TAPS)
    h = x
    for i, layer in enumerate(cnn):
        h = layer(h)
        for name, idx in taps.items():
            if i == idx:
                outs[name] = h.flatten(1).cpu().numpy()
    outs["phi"] = teacher.energy_score(ot).cpu().numpy()
    return outs


# ---------------------------------------------------------------- TopK SAE
class TopKSAE(th.nn.Module):
    def __init__(self, d_in: int, expansion: int = 8, k: int = 32):
        super().__init__()
        d_dict = d_in * expansion
        self.k = k
        self.W_enc = th.nn.Parameter(th.randn(d_in, d_dict) * 0.02)
        self.b_enc = th.nn.Parameter(th.zeros(d_dict))
        self.W_dec = th.nn.Parameter(th.randn(d_dict, d_in) * 0.02)
        self.b_dec = th.nn.Parameter(th.zeros(d_in))

    def forward(self, x):
        h = (x - self.b_dec) @ self.W_enc + self.b_enc
        topv, topi = th.topk(h, self.k, dim=-1)
        hs = th.zeros_like(h).scatter_(-1, topi, th.relu(topv))
        return hs @ self.W_dec + self.b_dec

    @th.no_grad()
    def renorm(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=-1, keepdim=True).clamp(1e-8)


def train_sae(acts: np.ndarray, steps: int = 2000, bs: int = 256,
              seed: int = 0) -> TopKSAE:
    th.manual_seed(seed)
    x = th.as_tensor(acts, dtype=th.float32)
    sae = TopKSAE(x.shape[1])
    with th.no_grad():
        sae.b_dec.copy_(x.mean(0))
    opt = th.optim.Adam(sae.parameters(), lr=1e-3)
    n = len(x)
    for s in range(steps):
        idx = th.randint(0, n, (bs,))
        xb = x[idx]
        loss = th.nn.functional.mse_loss(sae(xb), xb)
        opt.zero_grad(); loss.backward(); opt.step(); sae.renorm()
    return sae


@th.no_grad()
def recon_err(sae: TopKSAE, acts: np.ndarray) -> np.ndarray:
    x = th.as_tensor(acts, dtype=th.float32)
    return ((sae(x) - x) ** 2).mean(-1).numpy()


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score(pos) > score(neg)); rank-based, no sklearn."""
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", choices=["altgoal", "locked"], required=True)
    ap.add_argument("--ego", action="store_true")
    a = ap.parse_args()
    sfx = "_ego" if a.ego else ""
    src_env, tgt_env = SCENARIOS[a.scenario]

    from sb3_contrib import MaskablePPO
    teacher = FrozenMaskableTeacher(
        f"runs/paper_{a.scenario}_teacher{sfx}/final_model.zip")
    student = MaskablePPO.load(
        f"runs/paper_{a.scenario}{sfx}_ebtl_seed0/best_model.zip", device="cpu")

    tag_mode = "goal" if a.scenario == "altgoal" else "key"
    src_obs, _ = collect(src_env, teacher_actor(teacher), N_FRAMES, 11,
                         tag_mode, a.ego)
    tgt_obs, tgt_tags = collect(tgt_env, student_actor(student), N_FRAMES, 22,
                                tag_mode, a.ego)
    if a.scenario == "altgoal":
        id_sel, ood_sel = tgt_tags == 1, tgt_tags == 3
    else:
        id_sel, ood_sel = tgt_tags == 1, tgt_tags == 0

    A_src = activations(teacher, src_obs)
    A_tgt = activations(teacher, tgt_obs)

    # phi baseline: OOD-score = -phi (lower energy = more OOD)
    print(f"\n== {a.scenario}{sfx}: AUROC (OOD-score higher on OOD) ==")
    b1 = auroc(-A_tgt["phi"][ood_sel], -A_src["phi"])
    b2 = auroc(-A_tgt["phi"][ood_sel], -A_tgt["phi"][id_sel])
    print(f"{'phi baseline':14s}  src-vs-ood: {b1:.3f}   id-vs-ood: {b2:.3f}")

    n_train = int(len(src_obs) * 0.8)
    for name in TAPS:
        sae = train_sae(A_src[name][:n_train])
        e_src = recon_err(sae, A_src[name][n_train:])  # held-out source
        e_id = recon_err(sae, A_tgt[name][id_sel])
        e_ood = recon_err(sae, A_tgt[name][ood_sel])
        print(f"{name:14s}  src-vs-ood: {auroc(e_ood, e_src):.3f}   "
              f"id-vs-ood: {auroc(e_ood, e_id):.3f}")


if __name__ == "__main__":
    main()
