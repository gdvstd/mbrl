"""SB3 PPO plumbing for the LIBERO student (continuous, dict obs).

The student's policy reuses the EXACT GaussianBCPolicy backbone (cnn+trunk)
as its SB3 features extractor, so fine-tuning transfer is a 1:1 state-dict
copy: bc.cnn/trunk -> features_extractor, bc.mu -> action_net,
bc.log_std -> policy.log_std.

TaskMixEnv: a gymnasium env that owns one LIBERO sim per task in its task
list and samples a task each episode -- the "target domain" where some
episodes involve objects the teacher never saw. Sparse reward (success=1).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from vla_common import (
    ACTION_DIM,
    IN_CHANNELS,
    IMG_SIZE,
    MAX_STEPS,
    PROPRIO_DIM,
    GaussianBCPolicy,
    make_env,
    preprocess_obs,
)


class TaskMixEnv(gym.Env):
    """Each episode: sample a task from `task_ids`, reset that LIBERO sim."""

    def __init__(self, task_ids, seed: int = 0):
        self.task_ids = list(task_ids)
        self.rng = np.random.default_rng(seed)
        self._envs = {}  # lazily created, kept alive per task
        self._cur = None
        self._t = 0
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (IN_CHANNELS, IMG_SIZE, IMG_SIZE), np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf, (PROPRIO_DIM,), np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)

    def _get(self, tid):
        if tid not in self._envs:
            self._envs[tid] = make_env(tid)
        return self._envs[tid]

    def _obs(self, raw):
        img, prop = preprocess_obs(raw, self._tid)
        return {"image": (img * 255).astype(np.uint8), "proprio": prop}

    def reset(self, *, seed=None, options=None):
        from vla_common import get_init_states
        self._tid = int(self.rng.choice(self.task_ids))
        self._cur = self._get(self._tid)
        inits = get_init_states(self._tid)
        self._cur.reset()
        raw = self._cur.set_init_state(
            inits[int(self.rng.integers(len(inits)))])
        self._t = 0
        return self._obs(raw), {"task_id": self._tid}

    def step(self, action):
        raw, r, done, info = self._cur.step(np.asarray(action, np.float64))
        self._t += 1
        terminated = bool(done and r > 0)
        truncated = self._t >= MAX_STEPS or (done and not terminated)
        return self._obs(raw), float(r), terminated, truncated, info

    def close(self):
        for e in self._envs.values():
            e.close()


class BCBackboneExtractor(BaseFeaturesExtractor):
    """Same cnn+trunk as GaussianBCPolicy, consuming the dict obs."""

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        ref = GaussianBCPolicy()
        self.cnn = ref.cnn
        self.trunk = ref.trunk

    def forward(self, obs) -> torch.Tensor:
        img = obs["image"].float() / 255.0
        return self.trunk(torch.cat([self.cnn(img), obs["proprio"]], dim=-1))


PPO_HYPERPARAMS = dict(
    learning_rate=3e-4,
    n_steps=256,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    policy_kwargs=dict(
        features_extractor_class=BCBackboneExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[], vf=[]),
        share_features_extractor=True,
        normalize_images=False,
        log_std_init=-1.0,
    ),
)


def make_vla_vec_env(task_ids, seed: int, n_envs: int = 2,
                     monitor_dir=None):
    import os
    def thunk(rank):
        def _f():
            env = TaskMixEnv(task_ids, seed=seed + rank)
            fn = None
            if monitor_dir:
                os.makedirs(monitor_dir, exist_ok=True)
                fn = os.path.join(monitor_dir, str(rank))
            return Monitor(env, filename=fn)
        return _f
    return DummyVecEnv([thunk(r) for r in range(n_envs)])


def build_vla_ppo(env, seed: int, device: str = "cpu", tb_dir=None,
                  algo_cls: type[PPO] = PPO, **overrides) -> PPO:
    params = {**PPO_HYPERPARAMS, **overrides}
    return algo_cls("MultiInputPolicy", env, seed=seed, device=device,
                    tensorboard_log=tb_dir, verbose=1, **params)


def load_teacher_into(model: PPO, bc_ckpt: str) -> None:
    """Fine-tuning transfer: copy BC teacher weights into the SB3 policy."""
    bc = GaussianBCPolicy()
    bc.load_state_dict(torch.load(bc_ckpt, map_location="cpu", weights_only=False))
    fe = model.policy.features_extractor
    fe.cnn.load_state_dict(bc.cnn.state_dict())
    fe.trunk.load_state_dict(bc.trunk.state_dict())
    if not model.policy.share_features_extractor:
        model.policy.vf_features_extractor.cnn.load_state_dict(bc.cnn.state_dict())
        model.policy.vf_features_extractor.trunk.load_state_dict(bc.trunk.state_dict())
    # bc.mu predicts a CHUNK of actions; the per-step student takes the
    # first action's slice of the head.
    with torch.no_grad():
        model.policy.action_net.weight.copy_(bc.mu.weight[:ACTION_DIM])
        model.policy.action_net.bias.copy_(bc.mu.bias[:ACTION_DIM])
        model.policy.log_std.copy_(bc.log_std.data[:ACTION_DIM].to(model.device))


class FrozenBCTeacher:
    """Frozen Gaussian teacher for advising/distillation on dict obs."""

    def __init__(self, ckpt: str, device: str = "cpu"):
        self.device = device
        self.pol = GaussianBCPolicy().to(device)
        self.pol.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        self.pol.eval()
        for p in self.pol.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def distribution(self, obs_tensors: dict):
        """Per-step Gaussian = first action of the teacher's chunk."""
        img = obs_tensors["image"].float() / 255.0
        d = self.pol.distribution(img, obs_tensors["proprio"])
        return torch.distributions.Normal(d.mean[..., :ACTION_DIM],
                                          d.stddev[..., :ACTION_DIM])
