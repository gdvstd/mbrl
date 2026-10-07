"""Shared config for the LIBERO transfer track (VLA-shaped, Pair A).

Task pair (the altgoal analog, see project notes):
  * Teacher: BC on the SEEN half of libero_object (tasks 0-4)
  * Student: target = all 10 objects, episodes sampled uniformly ->
    ~50% of episodes involve an object the teacher never saw.

Obs: agentview+wrist RGB (6x128x128) + 9-D proprio; CHUNK-step
action chunks (ACT-style) for the BC teacher
(joint_states 7 + gripper_states 2). Action: 7-D continuous
(delta-EEF pose 6 + gripper 1), robosuite OSC convention, in [-1, 1].

The teacher is a Gaussian policy (mean head + state-independent log-std)
so that action advising / KSRL / Eq.3-style corrections have densities
and log-probs to work with. Energy-based gating (EBTL) is intentionally
NOT defined for this continuous head -- it arrives with the token-based
policy later.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

SEEN_TASKS = (0, 1, 2, 3, 4)      # alphabet soup, cream cheese, salad
                                   # dressing, bbq sauce, ketchup
UNSEEN_TASKS = (5, 6, 7, 8, 9)     # tomato sauce, butter, milk,
                                   # chocolate pudding, orange juice
IMG_SIZE = 128
IN_CHANNELS = 6   # agentview + eye_in_hand, stacked
CHUNK = 8         # action chunk length (ACT-style)
N_TASKS = 10
PROPRIO_DIM = 9 + N_TASKS  # joints+gripper + task one-hot
ACTION_DIM = 7
MAX_STEPS = 300  # libero_object episodes are ~150 steps in demos


def make_env(task_id: int, camera_hw: int = 128):
    """LIBERO OffScreenRenderEnv for one libero_object task."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bm = benchmark.get_benchmark_dict()["libero_object"]()
    task = bm.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder,
                        task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=camera_hw,
                             camera_widths=camera_hw)
    env.task_id = task_id
    env.language = task.language
    return env


def task_onehot(task_id: int) -> np.ndarray:
    v = np.zeros(N_TASKS, dtype=np.float32)
    v[task_id] = 1.0
    return v


def preprocess_obs(obs: dict, task_id: int) -> tuple[np.ndarray, np.ndarray]:
    """LIBERO obs dict -> (6x128x128 float[0,1], proprio 9 + task one-hot)."""
    img = np.concatenate([obs["agentview_image"],
                          obs["robot0_eye_in_hand_image"]], axis=-1)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
    prop = np.concatenate([obs["robot0_joint_pos"],
                           obs["robot0_gripper_qpos"],
                           task_onehot(task_id)]).astype(np.float32)
    return img, prop


def get_init_states(task_id: int):
    """LIBERO's fixed eval/demo-matched init state set for a task.

    Loads the .init file directly with weights_only=False: LIBERO's own
    loader trips over torch>=2.6's safe-unpickling default."""
    from libero.libero import benchmark, get_libero_path
    bm = benchmark.get_benchmark_dict()["libero_object"]()
    task = bm.get_task(task_id)
    path = os.path.join(get_libero_path("init_states"), task.problem_folder,
                        task.init_states_file)
    return torch.load(path, weights_only=False)


class GaussianBCPolicy(nn.Module):
    """Small CNN+MLP Gaussian policy: pi(a|image, proprio).

    Used as the teacher (BC on seen-task demos) and as the student's
    architecture for weight-init transfer. ~1M params, M1-friendly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE)).shape[1]
        self.trunk = nn.Sequential(
            nn.Linear(n_flat + PROPRIO_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mu = nn.Linear(256, ACTION_DIM * CHUNK)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM * CHUNK,), -1.0))
        self._queue: list = []

    def forward(self, img: torch.Tensor, prop: torch.Tensor):
        z = self.trunk(torch.cat([self.cnn(img), prop], dim=-1))
        return self.mu(z), self.log_std.expand_as(self.mu(z))

    def distribution(self, img, prop):
        mu, log_std = self.forward(img, prop)
        return torch.distributions.Normal(mu, log_std.exp())

    @torch.no_grad()
    def act(self, img: np.ndarray, prop: np.ndarray,
            deterministic: bool = True, device: str = "cpu") -> np.ndarray:
        """Chunked open-loop execution: re-plan every CHUNK steps."""
        if not self._queue:
            im = torch.as_tensor(img, device=device)[None]
            pr = torch.as_tensor(prop, device=device)[None]
            dist = self.distribution(im, pr)
            a = dist.mean if deterministic else dist.sample()
            chunk = a.clamp(-1, 1).cpu().numpy()[0].reshape(CHUNK, ACTION_DIM)
            self._queue = list(chunk)
        return self._queue.pop(0)

    def reset_plan(self) -> None:
        self._queue = []
