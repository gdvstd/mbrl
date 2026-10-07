"""Kickstarting (KSRL) on MaskablePPO for the paper-faithful track.

Port of grid_ksrl.KickstartPPO. The distillation cross-entropy is computed
between the MASKED teacher and student distributions on the rollout states
(both see the same per-state action mask, stored in the rollout buffer);
invalid actions carry zero teacher probability and are excluded from the sum
to avoid 0 * -inf. lambda_k linearly decays to 0 over kick_decay_frac of the
budget. train() is a copy of sb3-contrib 2.9.0 MaskablePPO.train with the
kick term added (marked).
"""
from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance


class KickstartMaskablePPO(MaskablePPO):
    def __init__(self, *args, teacher=None, kick_lambda0: float = 0.5,
                 kick_decay_frac: float = 0.5, **kwargs):
        self.teacher = teacher
        self.kick_lambda0 = kick_lambda0
        self.kick_decay_frac = kick_decay_frac
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["teacher"]

    def _kick_lambda(self) -> float:
        progress = 1.0 - self._current_progress_remaining
        if self.kick_decay_frac <= 0:
            return 0.0
        return self.kick_lambda0 * max(0.0, 1.0 - progress / self.kick_decay_frac)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        lambda_k = self._kick_lambda()

        entropy_losses = []
        pg_losses, value_losses, kick_losses = [], [], []
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

                # === KSRL: exact masked cross-entropy distillation ===========
                if lambda_k > 0:
                    with th.no_grad():
                        t_probs = self.teacher.distribution(
                            rollout_data.observations,
                            action_masks=rollout_data.action_masks,
                        ).distribution.probs
                    s_dist = self.policy.get_distribution(
                        rollout_data.observations,
                        action_masks=rollout_data.action_masks)
                    s_logp = s_dist.distribution.logits  # masked log-probs
                    term = th.where(t_probs > 0, t_probs * s_logp,
                                    th.zeros_like(s_logp))
                    kick_loss = -term.sum(dim=-1).mean()
                else:
                    kick_loss = th.zeros((), device=values.device)
                kick_losses.append(kick_loss.item())
                # =============================================================

                loss = (policy_loss + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss + lambda_k * kick_loss)

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
        self.logger.record("train/kick_loss", np.mean(kick_losses))
        self.logger.record("train/kick_lambda", lambda_k)
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates,
                           exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
