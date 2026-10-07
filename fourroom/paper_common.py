"""Shared config for the paper-faithful EBTL track (11x11 four-room GridWorld).

Everything follows the paper's Appendix A/B:
  * observation: FULL 11x11x3 grid (FullyObsWrapper, mission dropped)
  * action masking via sb3-contrib MaskablePPO + ActionMasker
  * PPO hyperparameters exactly as Table 3
  * architecture exactly as Fig. 8a: per-head conv towers
    (Conv16(2x2)+Pool(2x2) -> Conv32(2x2) -> Conv64(2x2) -> flatten 576),
    then a single Linear 576->7 / 576->1 (no hidden MLP), actor and critic
    NOT sharing the extractor.

As with grid_common: one config source for teacher/students of every method.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv

import fourroom_env  # noqa: F401  (registers the envs on import)
from fourroom.fourroom_env import SCENARIOS, action_mask

N_ENVS = 8  # paper Table 3: number of parallel environments
PAPER_SOLVED_REWARD = 0.95  # reward is exactly 1 on success; eval mean >=0.95


class PaperFeaturesExtractor(BaseFeaturesExtractor):
    """Fig. 8a conv tower: 11x11x3 -> 576 (64*3*3). No trailing linear."""

    def __init__(self, observation_space, features_dim: int = 576,
                 normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_in = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_in, 16, (2, 2)), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)), nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            assert self.cnn(sample).shape[1] == features_dim, "expect 576"

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations)


# Paper Table 3, verbatim. Train batch 256 with 8 envs -> n_steps 32.
PAPER_PPO_HYPERPARAMS = dict(
    learning_rate=5e-4,
    n_steps=32,
    batch_size=128,       # SGD minibatch size
    n_epochs=4,           # number of SGD iterations
    gamma=0.9,
    gae_lambda=0.8,
    clip_range=0.2,
    clip_range_vf=10.0,
    vf_coef=0.5,
    ent_coef=0.01,
    normalize_advantage=False,
    policy_kwargs=dict(
        features_extractor_class=PaperFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=576),
        net_arch=dict(pi=[], vf=[]),
        share_features_extractor=False,
        normalize_images=False,
    ),
)


def _wrap(env: gym.Env, ego: bool = False) -> gym.Env:
    """Default: full 11x11 grid obs. ego=True: the env's own PARTIAL
    egocentric view (rotated to the agent frame, occluded behind walls via
    MiniGrid's default see_through_walls=False) -- pair with
    agent_view_size=11 so the network input shape is unchanged."""
    env = ImgObsWrapper(env if ego else FullyObsWrapper(env))
    return ActionMasker(env, action_mask)


def make_paper_vec_env(env_id: str, seed: int, n_envs: int = N_ENVS,
                       monitor_dir: str | None = None,
                       env_kwargs: dict | None = None,
                       ego: bool = False) -> VecEnv:
    env_kwargs = dict(env_kwargs or {})
    if ego:
        env_kwargs.setdefault("agent_view_size", 11)
    return make_vec_env(env_id, n_envs=n_envs, seed=seed,
                        wrapper_class=lambda e: _wrap(e, ego=ego),
                        monitor_dir=monitor_dir, env_kwargs=env_kwargs)


def build_mppo(env: VecEnv, seed: int, device: str = "cpu",
               tb_dir: str | None = None,
               algo_cls: type[MaskablePPO] = MaskablePPO,
               **overrides) -> MaskablePPO:
    params = {**PAPER_PPO_HYPERPARAMS, **overrides}
    return algo_cls("CnnPolicy", env, seed=seed, device=device,
                    tensorboard_log=tb_dir, verbose=1, **params)


def make_paper_callbacks(eval_env: VecEnv, outdir: str, eval_freq: int,
                         n_eval_episodes: int,
                         early_stop_threshold: float | None = None,
                         checkpoint_freq: int = 100_000) -> list:
    import os
    from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold

    eval_dir = os.path.join(outdir, "eval")
    ckpt_dir = os.path.join(outdir, "checkpoints")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    on_new_best = (None if early_stop_threshold is None else
                   StopTrainingOnRewardThreshold(
                       reward_threshold=early_stop_threshold, verbose=1))
    eval_cb = MaskableEvalCallback(
        eval_env, best_model_save_path=outdir, log_path=eval_dir,
        eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
        deterministic=True, callback_on_new_best=on_new_best, verbose=1)
    ckpt_cb = CheckpointCallback(save_freq=checkpoint_freq,
                                 save_path=ckpt_dir, name_prefix="mppo")
    return [eval_cb, ckpt_cb]


class FrozenMaskableTeacher:
    """Frozen MaskablePPO teacher: masked action sampling for guidance,
    RAW (unmasked) logits for the energy score."""

    def __init__(self, path: str, device: str = "cpu") -> None:
        model = MaskablePPO.load(path, device=device)
        self.policy = model.policy
        self.policy.set_training_mode(False)
        for p in self.policy.parameters():
            p.requires_grad_(False)

    @th.no_grad()
    def distribution(self, obs_tensor: th.Tensor,
                     action_masks: np.ndarray | None = None):
        return self.policy.get_distribution(obs_tensor,
                                            action_masks=action_masks)

    @th.no_grad()
    def energy_score(self, obs_tensor: th.Tensor) -> th.Tensor:
        """phi(s) = logsumexp of RAW action logits (pre-masking; the mask
        would put -inf on invalid actions and corrupt the free energy)."""
        features = self.policy.extract_features(obs_tensor)
        if not self.policy.share_features_extractor:
            features = features[0]
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        logits = self.policy.action_net(latent_pi)
        return th.logsumexp(logits, dim=-1)


@th.no_grad()
def calibrate_energy_threshold(teacher: FrozenMaskableTeacher,
                               source_env_id: str, seed: int,
                               quantile: float, n_frames: int = 16_000,
                               device: str = "cpu", ego: bool = False) -> float:
    """tau = Quantile_q of phi over teacher states re-rolled (stochastically,
    with masking) in the source env."""
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.vec_env import VecTransposeImage

    env = VecTransposeImage(make_paper_vec_env(source_env_id, seed=seed,
                                               ego=ego))
    obs = env.reset()
    scores = []
    for _ in range(max(n_frames // env.num_envs, 1)):
        obs_t = th.as_tensor(obs, device=device)
        scores.append(teacher.energy_score(obs_t).cpu().numpy())
        masks = get_action_masks(env)
        dist = teacher.distribution(obs_t, action_masks=masks)
        actions = dist.get_actions(deterministic=False).cpu().numpy()
        obs, _, _, _ = env.step(actions)
    env.close()
    return float(np.quantile(np.concatenate(scores), quantile))
