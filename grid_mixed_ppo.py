"""PPO with mixed teacher/student rollouts + off-policy correction (EBTL Eq. 3).

Used by AA, JSRL and EBTL: a `strategy` decides per env-step whether the
FROZEN teacher or the student acts. Faithfulness to the paper's Section 4:

  * Actor correction: we store the BEHAVIOR policy's log-prob in the rollout
    buffer (teacher's log pi_T(a|s) when the teacher acted, student's
    log pi_S(a|s) otherwise). SB3's PPO ratio exp(log pi_new - old_log_prob)
    then equals Eq. 3's r_t exactly, clipping included -- no train() change
    needed for the actor term.
  * Value correction: Eq. 3 trains V with the same importance ratio,
    L^V = E[r_t (V(s) - V^target)^2]. SB3's plain MSE lacks r_t, so train()
    is overridden below (a copy of SB3 2.9.0 PPO.train with the value loss
    replaced). The ratio is detached and clamped to [0, 10] for stability
    (the clamp is ours, not the paper's).

The teacher/strategy are excluded from checkpoints, so saved models load as
plain PPO (same trick as JSRLSAC in jsrl.py).
"""
from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance, obs_as_tensor


class MixedPolicyPPO(PPO):
    def __init__(self, *args, teacher=None, strategy=None, **kwargs):
        self.teacher = teacher
        self.strategy = strategy
        self._episode_steps: np.ndarray | None = None
        self._guided = 0
        self._acted = 0
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["teacher", "strategy"]

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """Copy of SB3 2.9.0 OnPolicyAlgorithm.collect_rollouts with the
        action-selection block replaced by strategy-masked mixed acting."""
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        if self._episode_steps is None:
            self._episode_steps = np.zeros(env.num_envs, dtype=np.int64)

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()
        progress = 1.0 - self._current_progress_remaining

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                mask = self.strategy.guidance_mask(
                    self.teacher, obs_tensor, self._episode_steps, progress
                ).to(actions.device)
                if mask.any():
                    t_dist = self.teacher.distribution(obs_tensor)
                    t_actions = t_dist.get_actions(deterministic=False)
                    t_log_probs = t_dist.log_prob(t_actions)
                    actions = th.where(mask, t_actions, actions)
                    log_probs = th.where(mask, t_log_probs, log_probs)
                self._guided += int(mask.sum().item())
                self._acted += env.num_envs
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

            # Timeout bootstrap (SB3 GH issue #633)
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
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        self.logger.record("guidance/rate", self._guided / max(self._acted, 1))
        self.logger.record("guidance/progress", progress)
        self._guided = self._acted = 0

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        """Copy of SB3 2.9.0 PPO.train; ONLY change: importance-weighted value
        loss per EBTL Eq. 3 (marked below)."""
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
                # === EBTL Eq. 3: L^V = E[r_t (V(s) - V^target)^2] ============
                # (SB3 default is an unweighted MSE.) Detached + clamped ratio.
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
