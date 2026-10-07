"""Reward-shaping-based policy transfer, inspired by Brys et al. (AAMAS 2015).

Potential-based reward shaping (Ng et al. 1999), which is policy-invariant (the
optimal policy is unchanged) while biasing exploration during training:

    r'(s, a, s') = r + gamma * Phi(s') - Phi(s),   Phi(terminal) = 0

DEVIATION FROM THE ORIGINAL PAPER -- read before citing:
  * Brys et al. (2015) build the potential from the source POLICY: a
    state-action "advice" potential (via Harutyunyan et al.'s dynamic/look-ahead
    advice) that rewards taking actions the source policy recommends -- i.e. it
    encourages policy imitation.
  * We instead use a simpler VALUE-based potential:
        Phi(s) = beta * V_teacher(s),
        V_teacher(s) = min(Q1, Q2)(s, mu_teacher(s))   (teacher actor + critic).
    This is NOT the paper's method; it is a value-based instantiation we chose,
    justified by the paper's remark that static PBRS is equivalent to Q-value
    initialization. To reproduce the paper faithfully, replace Phi with a
    source-policy action-advice potential.

The teacher never acts in the env here -- s' is the STUDENT's real next state;
the teacher only evaluates it. Applied to the TRAINING env only (inside
Monitor); eval uses the true reward.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch


class PotentialShapingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, teacher, gamma: float = 0.98, beta: float = 1.0):
        super().__init__(env)
        self.teacher = teacher
        self.gamma = gamma
        self.beta = beta
        self._phi_s = 0.0

    def _potential(self, obs: np.ndarray) -> float:
        policy = self.teacher.policy
        with torch.no_grad():
            obs_t = policy.obs_to_tensor(obs)[0]
            action = policy.actor(obs_t, deterministic=True)
            q_values = torch.cat(policy.critic(obs_t, action), dim=1)
            value = q_values.min(dim=1, keepdim=True).values.item()
        return self.beta * value

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._phi_s = self._potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi_next = 0.0 if terminated else self._potential(obs)
        shaped = reward + self.gamma * phi_next - self._phi_s
        self._phi_s = phi_next
        info["reward_original"] = reward
        info["reward_shaped"] = shaped
        return obs, shaped, terminated, truncated, info
