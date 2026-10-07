"""Frozen teacher wrapper + EBTL energy-threshold calibration.

The teacher is the PPO policy trained on Empty-NxN. All transfer methods share
this wrapper; EBTL additionally needs an energy threshold tau computed from
the teacher's own (source-task) state distribution.

Energy score (EBTL, arXiv 2506.16590, Sec. 3-4):
    E(s) = -T * logsumexp(logits(s) / T),  phi(s) = -E(s)   (T = 1 here)
States with phi(s) >= tau are treated as in-distribution for the teacher.
"""
from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecTransposeImage


class FrozenTeacher:
    """A trained PPO policy, frozen, exposed for guidance/distillation.

    All methods take the SAME channel-first uint8 obs tensors the student's
    collect/train loops already use (both policies were trained through the
    identical VecTransposeImage pipeline), so tensors pass straight through.
    """

    def __init__(self, path: str, device: str = "cpu") -> None:
        model = PPO.load(path, device=device)
        self.policy = model.policy
        self.policy.set_training_mode(False)
        for p in self.policy.parameters():
            p.requires_grad_(False)

    @th.no_grad()
    def distribution(self, obs_tensor: th.Tensor):
        """SB3 Distribution over actions (Categorical for MiniGrid)."""
        return self.policy.get_distribution(obs_tensor)

    @th.no_grad()
    def energy_score(self, obs_tensor: th.Tensor) -> th.Tensor:
        """phi(s) = logsumexp of the teacher's RAW action logits (higher = more ID).

        Must use action_net's raw output: torch's Categorical normalizes its
        stored `.logits` to log-probs, whose logsumexp is identically 0 --
        that would make the energy score constant and useless.
        """
        features = self.policy.extract_features(obs_tensor)
        if not self.policy.share_features_extractor:
            features = features[0]  # (pi_features, vf_features) tuple
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        logits = self.policy.action_net(latent_pi)
        return th.logsumexp(logits, dim=-1)


@th.no_grad()
def calibrate_energy_threshold(
    teacher: FrozenTeacher,
    source_env_id: str,
    seed: int,
    quantile: float,
    n_frames: int = 16_000,
    n_envs: int = 8,
    device: str = "cpu",
) -> float:
    """tau = Quantile_q of phi(s) over teacher states in the SOURCE env.

    The paper computes tau over states from the teacher's training experience
    (random subsampling). We approximate that set by re-rolling the trained
    teacher STOCHASTICALLY in the source env, which covers the same visitation
    distribution without needing training-time state dumps.
    """
    from stable_baselines3.common.env_util import make_vec_env
    from minigrid.wrappers import ImgObsWrapper

    env = VecTransposeImage(
        make_vec_env(source_env_id, n_envs=n_envs, seed=seed,
                     wrapper_class=ImgObsWrapper)
    )
    scores: list[np.ndarray] = []
    obs = env.reset()
    for _ in range(max(n_frames // n_envs, 1)):
        obs_t = th.as_tensor(obs, device=device)
        scores.append(teacher.energy_score(obs_t).cpu().numpy())
        dist = teacher.distribution(obs_t)
        actions = dist.get_actions(deterministic=False).cpu().numpy()
        obs, _, _, _ = env.step(actions)
    env.close()

    tau = float(np.quantile(np.concatenate(scores), quantile))
    return tau
