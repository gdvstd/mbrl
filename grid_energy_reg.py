"""Teacher energy regularization (EBTL paper, Sec. 4.2 "Energy Regularization").

Adds a margin-based energy loss to the TEACHER's PPO training so that its
energy score phi(s) = logsumexp(raw logits) separates ID from OOD states:

    L_energy = E_{s_in} [ max(0, m_in - phi(s_in))^2 ]     (ID pushed above m_in)
             + E_{s_out}[ max(0, phi(s_out) - m_out)^2 ]   (OOD pushed below m_out)
    L_total  = L_RL + lambda * L_energy

Following the paper: ID samples are the teacher's own training experience
(here: the current rollout minibatch -- exactly those states), and OOD samples
are a FIXED set of states collected by a random policy in the TARGET env
before training starts.

Margins are in phi units. With the unregularized teacher we measured source
phi ~ 5.9-6.2 and target phi ~ 3.5-6.2, so defaults m_in=7, m_out=2 ask for a
wide, unambiguous gap. train() is a copy of SB3 2.9.0 PPO.train with the
energy term added (marked).
"""
from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import VecTransposeImage


def collect_random_states(env_id: str, n_frames: int = 8000, seed: int = 7,
                          n_envs: int = 8) -> np.ndarray:
    """OOD sample set: states from RANDOM rollouts in the target env,
    channel-first uint8, matching the training pipeline's obs format."""
    env = VecTransposeImage(
        make_vec_env(env_id, n_envs=n_envs, seed=seed,
                     wrapper_class=ImgObsWrapper))
    obs = env.reset()
    out = []
    rng = np.random.default_rng(seed)
    for _ in range(max(n_frames // n_envs, 1)):
        out.append(obs.copy())
        actions = rng.integers(0, env.action_space.n, size=n_envs)
        obs, _, _, _ = env.step(actions)
    env.close()
    return np.concatenate(out)


class EnergyRegPPO(PPO):
    def __init__(self, *args, ood_states: np.ndarray | None = None,
                 energy_lambda: float = 0.1, m_in: float = 7.0,
                 m_out: float = 2.0, **kwargs):
        self.energy_lambda = energy_lambda
        self.m_in = m_in
        self.m_out = m_out
        self._ood_states = ood_states
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_ood_states"]

    def _phi(self, obs: th.Tensor) -> th.Tensor:
        """logsumexp of RAW action logits, with grad (cf. FrozenTeacher)."""
        features = self.policy.extract_features(obs)
        if not self.policy.share_features_extractor:
            features = features[0]
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        return th.logsumexp(self.policy.action_net(latent_pi), dim=-1)

    def _sample_ood(self, n: int) -> th.Tensor:
        idx = np.random.randint(0, len(self._ood_states), size=n)
        return th.as_tensor(self._ood_states[idx], device=self.device)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

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
                    rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
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

                # === energy regularization (ID = this minibatch's states, ====
                # === OOD = fixed random-rollout states from the target env) ==
                phi_in = self._phi(rollout_data.observations)
                phi_out = self._phi(self._sample_ood(len(phi_in)))
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
