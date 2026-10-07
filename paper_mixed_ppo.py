"""MaskablePPO with mixed teacher/student rollouts + EBTL Eq. 3 correction.

Port of grid_mixed_ppo.MixedPolicyPPO onto sb3-contrib MaskablePPO for the
paper-faithful four-room track. Same design:
  * a strategy decides per env-step whether the FROZEN teacher acts
    (both policies sample under the current action mask);
  * the BEHAVIOR log-prob is stored, so MaskablePPO's ratio equals Eq. 3;
  * train() weights the value loss by the (detached, clamped) ratio.

collect_rollouts/train are copies of sb3-contrib 2.9.0 MaskablePPO with the
marked modifications.
"""
from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import (
    MaskableDictRolloutBuffer,
    MaskableRolloutBuffer,
)
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common.utils import explained_variance, obs_as_tensor


class MixedPolicyMaskablePPO(MaskablePPO):
    def __init__(self, *args, teacher=None, strategy=None, **kwargs):
        self.teacher = teacher
        self.strategy = strategy
        self._episode_steps: np.ndarray | None = None
        self._guided = 0
        self._acted = 0
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["teacher", "strategy"]

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps,
                         use_masking: bool = True) -> bool:
        assert isinstance(
            rollout_buffer, (MaskableRolloutBuffer, MaskableDictRolloutBuffer)
        ), "RolloutBuffer doesn't support action masking"
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        if self._episode_steps is None:
            self._episode_steps = np.zeros(env.num_envs, dtype=np.int64)

        n_steps = 0
        action_masks = None
        rollout_buffer.reset()
        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking.")
        callback.on_rollout_start()
        progress = 1.0 - self._current_progress_remaining

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                if use_masking:
                    action_masks = get_action_masks(env)
                actions, values, log_probs = self.policy(
                    obs_tensor, action_masks=action_masks)
                # === strategy-masked teacher acting (behavior log-prob) =====
                mask = self.strategy.guidance_mask(
                    self.teacher, obs_tensor, self._episode_steps, progress
                ).to(actions.device)
                if mask.any():
                    t_dist = self.teacher.distribution(
                        obs_tensor, action_masks=action_masks)
                    t_actions = t_dist.get_actions(deterministic=False)
                    t_log_probs = t_dist.log_prob(t_actions)
                    actions = th.where(mask, t_actions, actions)
                    log_probs = th.where(mask, t_log_probs, log_probs)
                self._guided += int(mask.sum().item())
                self._acted += env.num_envs
                # =============================================================
            actions = actions.cpu().numpy()

            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            self._episode_steps += 1
            self._episode_steps[dones] = 0

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs, actions, rewards,
                self._last_episode_starts, values, log_probs,
                action_masks=action_masks,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        self.logger.record("guidance/rate", self._guided / max(self._acted, 1))
        self.logger.record("guidance/progress", progress)
        self._guided = self._acted = 0
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
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
                # === EBTL Eq. 3: importance-weighted value loss ==============
                vf_ratio = th.clamp(ratio.detach(), max=10.0)
                value_loss = (vf_ratio * (rollout_data.returns - values_pred) ** 2).mean()
                # =============================================================
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = (policy_loss + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss)

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
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates,
                           exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
