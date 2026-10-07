"""Shared PPO config and helpers for the MiniGrid transfer-RL study (EBTL track).

Task pair (both off-the-shelf MiniGrid envs, identical obs/action spaces):
  * Teacher: MiniGrid-Empty-NxN-v0   -- plain navigation to the goal square.
  * Student: MiniGrid-DoorKey-NxN-v0 -- pick up a key, open the locked door,
    then reach the goal. The teacher has never seen a key or a door, so the
    pre-key phase of an episode is out-of-distribution for it -- the structure
    EBTL (arXiv 2506.16590) exploits with energy-based OOD detection.

Both envs emit the same 7x7x3 egocentric image observation and the same 7-way
discrete action set, so teacher weights / logits transfer directly. We follow
the paper in using discrete-action PPO (they used TorchRL; we use SB3 to reuse
this repo's infrastructure).

As with sac_common: everything algorithm-related lives HERE so the teacher,
the from-scratch baseline and every transfer variant share one identical
config, and curve differences are attributable to the transfer method only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv

def grid_env_id(task: str, size: int) -> str:
    """Env id for a task/size pair.

    task: 'empty' (Task A, teacher) or 'doorkey' (Task B, student).
    size: grid side length. NOTE: DoorKey-8x8's sparse reward is out of reach
    for vanilla PPO within ~1M steps (both scratch and finetune flatlined at 0
    in our runs); 6x6 is the largest size that stays a tractable toy task.
    The observation is a 7x7 egocentric crop either way, so policies transfer
    across sizes unchanged.
    """
    name = {"empty": "Empty", "doorkey": "DoorKey"}[task]
    return f"MiniGrid-{name}-{size}x{size}-v0"
# MiniGrid reward is 1 - 0.9 * steps/max_steps on success, 0 otherwise;
# a near-optimal policy scores ~0.96 on both envs.
GRID_SOLVED_REWARD = 0.90

N_ENVS = 8  # vectorized rollout workers (cheap pure-python envs)


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    """Small CNN for the 7x7x3 MiniGrid image (SB3-docs recipe).

    SB3's default NatureCNN expects large frames; this 3-conv net is the
    standard extractor for MiniGrid-sized inputs. It is shared by the actor
    and critic heads (SB3 on-policy default).
    """

    def __init__(self, observation_space, features_dim: int = 128,
                 normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# Follows the torch-ac / rl-starter-files reference config for MiniGrid
# (lr 1e-3, 128-step rollouts, 4 epochs, ent 0.01), which reliably solves
# DoorKey. Two gotchas found the hard way (both runs flatlined without them):
#   * normalize_images=False -- SB3's CnnPolicy divides by 255 by default,
#     but MiniGrid pixels are small integers (0..~10), so the default squashes
#     inputs to ~0 and the CNN barely learns.
#   * lr 2.5e-4 (Atari-style) is too slow for these tiny nets; 1e-3 is right.
PPO_HYPERPARAMS = dict(
    learning_rate=1e-3,
    n_steps=128,          # x N_ENVS = 1024-frame rollouts
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    # 0.05, not the usual 0.01: with 0.01 DoorKey-6x6 never took off within
    # 1M steps (policy stayed near-uniform; rare successes never amplified).
    # 0.05 solves it from scratch in ~220K steps -- the exploration pressure
    # is what cracks the sparse pickup-key -> open-door -> goal chain.
    ent_coef=0.05,
    policy_kwargs=dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[64], vf=[64]),
        normalize_images=False,
    ),
)


def make_grid_vec_env(
    env_id: str,
    seed: int,
    n_envs: int = N_ENVS,
    monitor_dir: str | None = None,
) -> VecEnv:
    """Vectorized MiniGrid env with the mission string dropped (image only).

    ImgObsWrapper keeps just the 7x7x3 egocentric view, which is what makes
    Empty and DoorKey share an identical observation space for transfer.
    """
    return make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        wrapper_class=ImgObsWrapper,
        monitor_dir=monitor_dir,
    )


def build_ppo(
    env: VecEnv,
    seed: int,
    device: str = "cpu",
    tb_dir: str | None = None,
    algo_cls: type[PPO] = PPO,
    **overrides,
) -> PPO:
    """Construct a PPO model with the shared hyperparameters.

    Mirrors sac_common.build_sac: `algo_cls` lets a transfer variant subclass
    PPO, `overrides` tweaks single hyperparameters without duplicating config.
    """
    params = {**PPO_HYPERPARAMS, **overrides}
    return algo_cls(
        "CnnPolicy",
        env,
        seed=seed,
        device=device,
        tensorboard_log=tb_dir,
        verbose=1,
        **params,
    )
