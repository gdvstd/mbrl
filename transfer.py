"""Pluggable transfer strategies for student training.

The student script is agnostic to *how* teacher knowledge is used. Each strategy
gets three hooks so different method families fit the same seam:

    wrap_base_env(env)       -- wrap the TRAINING env before Monitor
                                (e.g. JSRL guide-rollin, reward shaping)
    modify_model(model)      -- one-shot changes before learning
                                (e.g. weight initialization from the teacher)
    after_eval_callback()    -- a callback run after each evaluation
                                (e.g. JSRL curriculum horizon advancement)

Implemented: none, weight_init, jsrl, reward_shaping.
To add a method, subclass TransferStrategy, register it in `get_transfer`, and
add its name to `TRANSFER_METHODS`. The student script needs no changes.
"""
from __future__ import annotations

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from jsrl import JSRLCurriculumCallback, JSRLSAC, build_horizons
from kickstarting import KickstartSAC
from reward_shaping import PotentialShapingWrapper
from sac_common import build_sac

TRANSFER_METHODS = ["none", "weight_init", "jsrl", "reward_shaping", "ksrl"]

# Planned methods (not yet implemented) -- explicit roadmap:
#   "action_advising" -- teacher advises an action every step, decaying prob
_PLANNED = ["action_advising"]


class TransferStrategy:
    """Base strategy: no transfer (from-scratch)."""

    name = "none"

    def __init__(self, teacher_path: str | None = None):
        self.teacher_path = teacher_path

    def build_model(self, env, seed, device, tb_dir) -> SAC:
        return build_sac(env, seed, device, tb_dir)

    def wrap_base_env(self, env: gym.Env) -> gym.Env:
        return env

    def modify_model(self, model: SAC) -> SAC:
        return model

    def after_eval_callback(self) -> BaseCallback | None:
        return None

    # shared helper
    def _load_teacher(self, device: str = "cpu") -> SAC:
        if not self.teacher_path:
            raise ValueError(
                f"transfer '{self.name}' requires a teacher model: "
                f"pass --teacher <path/to/best_model.zip>"
            )
        return SAC.load(self.teacher_path, device=device)


class NoTransfer(TransferStrategy):
    name = "none"


class WeightInitTransfer(TransferStrategy):
    """One-shot transfer: copy the teacher's network weights into the student.

    Teacher and student share identical obs/action spaces and net_arch, so the
    full SAC policy state_dict (actor + twin critics + critic targets) is copied
    verbatim. The replay buffer still starts empty and `learning_starts` random
    exploration still runs, so only the network initialization differs.
    """

    name = "weight_init"

    def modify_model(self, model: SAC) -> SAC:
        teacher = self._load_teacher(device=model.device)
        if (teacher.observation_space != model.observation_space
                or teacher.action_space != model.action_space):
            raise ValueError(
                "teacher/student spaces differ -- weight_init needs identical "
                f"obs/action spaces.\n  teacher: {teacher.observation_space}, "
                f"{teacher.action_space}\n  student: {model.observation_space}, "
                f"{model.action_space}"
            )
        model.policy.load_state_dict(teacher.policy.state_dict())
        print(f"[transfer] weight_init: copied teacher policy weights from "
              f"{self.teacher_path}")
        del teacher
        return model


class JSRLTransfer(TransferStrategy):
    """Jump-Start RL: guide controls the first `horizon` steps of each episode,
    the student the rest, and ALL transitions train the student. A curriculum
    lowers `horizon` toward 0 as the student improves (Algorithm 1, faithful)."""

    name = "jsrl"

    def __init__(
        self,
        teacher_path: str | None = None,
        max_horizon: int = 100,
        n_stages: int = 10,
        tolerance: float = 0.05,
        window: int = 1,
    ):
        super().__init__(teacher_path)
        self.horizons = build_horizons(max_horizon, n_stages)
        self.tolerance = tolerance
        self.window = window
        self._horizon_box: list[int] | None = None

    def build_model(self, env, seed, device, tb_dir) -> SAC:
        model = build_sac(env, seed, device, tb_dir, algo_cls=JSRLSAC)
        guide = self._load_teacher(device=device)
        model.set_jsrl(guide, self.horizons[0])
        self._horizon_box = model.jsrl_horizon
        print(f"[transfer] jsrl: guide loaded, horizons={self.horizons}")
        return model

    def after_eval_callback(self) -> BaseCallback | None:
        if self._horizon_box is None:
            return None
        return JSRLCurriculumCallback(
            self._horizon_box, self.horizons,
            tolerance=self.tolerance, window=self.window, verbose=1,
        )


class RewardShapingTransfer(TransferStrategy):
    """Potential-based reward shaping from the teacher's value (Brys et al.)."""

    name = "reward_shaping"

    def __init__(
        self,
        teacher_path: str | None = None,
        gamma: float = 0.98,
        beta: float = 1.0,
    ):
        super().__init__(teacher_path)
        self.gamma = gamma
        self.beta = beta

    def wrap_base_env(self, env: gym.Env) -> gym.Env:
        teacher = self._load_teacher()
        print(f"[transfer] reward_shaping: teacher loaded, "
              f"gamma={self.gamma} beta={self.beta}")
        return PotentialShapingWrapper(env, teacher, gamma=self.gamma, beta=self.beta)


class KickstartTransfer(TransferStrategy):
    """Kickstarting RL: add lambda_k * H(pi_teacher || pi_student) to the actor
    loss (evaluated on student states), lambda_k linearly annealed to 0."""

    name = "ksrl"

    def __init__(
        self,
        teacher_path: str | None = None,
        lambda0: float = 1.0,
        decay_steps: int = 500_000,
    ):
        super().__init__(teacher_path)
        self.lambda0 = lambda0
        self.decay_steps = decay_steps

    def build_model(self, env, seed, device, tb_dir) -> SAC:
        model = build_sac(env, seed, device, tb_dir, algo_cls=KickstartSAC)
        teacher = self._load_teacher(device=device)
        model.set_kickstart(teacher, self.lambda0, self.decay_steps)
        print(f"[transfer] ksrl: teacher loaded, lambda0={self.lambda0} "
              f"decay_steps={self.decay_steps}")
        return model


def get_transfer(
    method: str, teacher_path: str | None = None, **opts
) -> TransferStrategy:
    if method == "none":
        return NoTransfer(teacher_path)
    if method == "weight_init":
        return WeightInitTransfer(teacher_path)
    if method == "ksrl":
        return KickstartTransfer(
            teacher_path,
            lambda0=opts.get("ksrl_lambda0", 1.0),
            decay_steps=opts.get("ksrl_decay_steps", 500_000),
        )
    if method == "jsrl":
        return JSRLTransfer(
            teacher_path,
            max_horizon=opts.get("jsrl_max_horizon", 100),
            n_stages=opts.get("jsrl_n_stages", 10),
            tolerance=opts.get("jsrl_tolerance", 0.05),
        )
    if method == "reward_shaping":
        return RewardShapingTransfer(
            teacher_path,
            gamma=opts.get("gamma", 0.98),
            beta=opts.get("shaping_beta", 1.0),
        )
    if method in _PLANNED:
        raise NotImplementedError(
            f"transfer method '{method}' is planned but not implemented yet."
        )
    raise ValueError(
        f"unknown transfer method '{method}'. Available: {TRANSFER_METHODS}"
    )
