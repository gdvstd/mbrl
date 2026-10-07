"""Teacher energy regularization on MaskablePPO (paper Appendix A.1/D).

Port of grid_energy_reg.EnergyRegPPO with the paper's exact recipe:
  * margins over phi: (m_in, m_out) = (10, 15) -- note m_in < m_out, i.e. an
    OVERLAP BAND is allowed, much softer than a hard separation
  * ID set: the most RECENT 3000 frames of the teacher's own trajectory
    (rolling buffer refilled from each rollout)
  * OOD set: fixed, sampled from 100 masked-random-policy episodes in the
    target env (for Locked, the agent starts in ANY room for coverage)
  * phi computed from RAW (unmasked) logits, with gradients.

train() is a copy of sb3-contrib 2.9.0 MaskablePPO.train with the energy
term added (marked).
"""
from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance

ID_BUFFER_FRAMES = 3000  # paper: "most recent 3,000 frames"


def collect_random_states_masked(env_id: str, n_episodes: int = 100,
                                 seed: int = 7, env_kwargs: dict | None = None,
                                 ego: bool = False) -> np.ndarray:
    """OOD set: masked-random rollouts in the target env (paper: 100 eps)."""
    import gymnasium as gym
    from stable_baselines3.common.vec_env import (
        DummyVecEnv, VecTransposeImage)
    from fourroom.paper_common import _wrap
    from fourroom.fourroom_env import action_mask

    env_kwargs = dict(env_kwargs or {})
    if ego:
        env_kwargs.setdefault("agent_view_size", 11)
    env = VecTransposeImage(DummyVecEnv(
        [lambda: _wrap(gym.make(env_id, **env_kwargs), ego=ego)]))
    inner = env.venv.envs[0]
    rng = np.random.default_rng(seed)
    out = []
    env.seed(seed)
    obs = env.reset()
    done_eps = 0
    while done_eps < n_episodes:
        out.append(obs.copy())
        valid = np.flatnonzero(action_mask(inner))
        obs, _, dones, _ = env.step(np.array([rng.choice(valid)]))
        if dones[0]:
            done_eps += 1
    env.close()
    return np.concatenate(out)


class EnergyRegMaskablePPO(MaskablePPO):
    def __init__(self, *args, ood_states: np.ndarray | None = None,
                 energy_lambda: float = 0.1, m_in: float = 10.0,
                 m_out: float = 15.0, **kwargs):
        self.energy_lambda = energy_lambda
        self.m_in = m_in
        self.m_out = m_out
        self._ood_states = ood_states
        self._id_buffer: np.ndarray | None = None  # rolling recent frames
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_ood_states", "_id_buffer"]

    def _phi(self, obs: th.Tensor) -> th.Tensor:
        features = self.policy.extract_features(obs)
        if not self.policy.share_features_extractor:
            features = features[0]
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        return th.logsumexp(self.policy.action_net(latent_pi), dim=-1)

    def _update_id_buffer(self) -> None:
        obs = self.rollout_buffer.observations  # (n_steps, n_envs, C, H, W)
        flat = obs.reshape((-1,) + obs.shape[2:])
        if self._id_buffer is None:
            self._id_buffer = flat.copy()
        else:
            self._id_buffer = np.concatenate([self._id_buffer, flat])
        if len(self._id_buffer) > ID_BUFFER_FRAMES:
            self._id_buffer = self._id_buffer[-ID_BUFFER_FRAMES:]

    def _sample(self, arr: np.ndarray, n: int) -> th.Tensor:
        idx = np.random.randint(0, len(arr), size=n)
        return th.as_tensor(arr[idx], device=self.device)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        self._update_id_buffer()

        entropy_losses = []
        pg_losses, value_losses, energy_losses = [], [], []
        clip_fractions = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions,
                    action_masks=rollout_data.action_masks)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf, clip_range_vf)
                value_loss = th.nn.functional.mse_loss(
                    rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # === energy regularization (ID = recent frames, OOD fixed) ==
                n = len(rollout_data.observations)
                phi_in = self._phi(self._sample(self._id_buffer, n))
                phi_out = self._phi(self._sample(self._ood_states, n))
                energy_loss = (
                    th.clamp(self.m_in - phi_in, min=0).pow(2).mean()
                    + th.clamp(phi_out - self.m_out, min=0).pow(2).mean())
                energy_losses.append(energy_loss.item())
                # =============================================================

                loss = (policy_loss + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                        + self.energy_lambda * energy_loss)

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching"
                              f" max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/energy_loss", np.mean(energy_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates,
                           exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
