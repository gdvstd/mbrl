"""PPO with mixed teacher/student rollouts for the LIBERO track (continuous).

Port of the four-room MixedPolicyMaskablePPO to plain PPO with dict obs and
a Gaussian teacher. The behavior policy's log-prob is stored in the buffer,
so PPO's importance ratio implements the off-policy correction for
teacher-taken actions (actor term of EBTL Eq. 3). The value-loss weighting
is deliberately NOT ported here (baselines-first phase; the four-room
diagnostics showed it is not load-bearing).

Strategies reused from grid_strategies (AAStrategy / JSRLStrategy);
EBTLStrategy is NOT applicable to this continuous teacher.
collect_rollouts is a copy of SB3 2.9.0 OnPolicyAlgorithm.collect_rollouts
with the marked block added.
"""
from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor


class MixedPolicyVLAPPO(PPO):
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
        assert self._last_obs is not None
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
                # === strategy-masked teacher acting ========================
                mask = self.strategy.guidance_mask(
                    self.teacher, obs_tensor["proprio"],
                    self._episode_steps, progress).to(self.device)
                if mask.any():
                    t_dist = self.teacher.distribution(obs_tensor)
                    t_actions = t_dist.sample().clamp(-1, 1)
                    t_log_probs = t_dist.log_prob(t_actions).sum(-1)
                    m = mask.unsqueeze(-1)
                    actions = th.where(m, t_actions, actions)
                    log_probs = th.where(mask, t_log_probs, log_probs)
                self._guided += int(mask.sum().item())
                self._acted += env.num_envs
                # ===========================================================
            actions = actions.cpu().numpy()

            clipped = np.clip(actions, self.action_space.low,
                              self.action_space.high)
            new_obs, rewards, dones, infos = env.step(clipped)

            self.num_timesteps += env.num_envs
            self._episode_steps += 1
            self._episode_steps[dones] = 0

            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1

            for idx, done in enumerate(dones):
                if (done and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(self._last_obs, actions, rewards,
                               self._last_episode_starts, values, log_probs)
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
