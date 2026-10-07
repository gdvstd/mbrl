"""Jump-Start RL (Uchendu et al., ICML 2023) for our SAC + BipedalWalker setup.

Faithful to Algorithm 1: at each episode the combined policy runs the guide
(teacher) for the first `horizon` steps and the student thereafter, and the
WHOLE rolled-out trajectory -- guide-controlled AND student-controlled steps --
is appended to the training data (here: the replay buffer). A curriculum lowers
`horizon` from max to 0 as the student's eval reward plateaus, so the student
gradually takes over the entire episode.

Design (matches the third-party SB3 impl github.com/steventango/jumpstart-rl,
MIT, re-expressed for our seam):
  * JSRLSAC overrides `predict`. During rollout collection (deterministic=False)
    it switches guide/student per episode step, so guide transitions enter the
    replay buffer. During evaluation (deterministic=True) it is the pure student,
    so eval stays comparable to the baseline.
  * JSRLCurriculumCallback advances the horizon via a shared mutable box.
The guide and horizon state are excluded from model.save (checkpoints stay small
and load back as a plain student).
"""
from __future__ import annotations

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback


def build_horizons(max_horizon: int, n_stages: int) -> list[int]:
    """Curriculum of guide-rollin lengths from `max_horizon` down to 0."""
    if n_stages < 1:
        raise ValueError("n_stages must be >= 1")
    if n_stages == 1:
        return [0]
    return [int(round(h)) for h in np.linspace(max_horizon, 0, n_stages)]


class JSRLSAC(SAC):
    """SAC whose rollout action selection is the JSRL combined policy.

    Call `set_jsrl(guide, horizon)` after construction. `horizon` lives in a
    one-element list so the curriculum callback can mutate it in place.
    """

    def set_jsrl(self, guide: SAC, horizon: int) -> None:
        self._jsrl_guide = guide
        self._jsrl_horizon = [int(horizon)]
        self._jsrl_timesteps = np.zeros(self.env.num_envs, dtype=np.int64)

    @property
    def jsrl_horizon(self) -> list[int]:
        return self._jsrl_horizon

    def _excluded_save_params(self) -> list[str]:
        # keep the guide / horizon / counters out of saved checkpoints
        return super()._excluded_save_params() + [
            "_jsrl_guide", "_jsrl_horizon", "_jsrl_timesteps"
        ]

    def predict(self, observation, state=None, episode_start=None,
                deterministic=False):
        # Evaluation (or before set_jsrl): pure student.
        if deterministic or not hasattr(self, "_jsrl_guide"):
            return super().predict(observation, state, episode_start, deterministic)

        # Reset per-env episode step counter when the previous step ended.
        dones = getattr(self.env, "buf_dones", None)
        if dones is not None:
            self._jsrl_timesteps[np.asarray(dones, dtype=bool)] = 0

        horizon = self._jsrl_horizon[0]
        use_guide = self._jsrl_timesteps < horizon  # first `horizon` steps -> guide
        guide_action, _ = self._jsrl_guide.predict(observation, deterministic=True)
        student_action, _ = super().predict(
            observation, state, episode_start, deterministic=False
        )
        action = np.where(use_guide[:, None], guide_action, student_action)
        self._jsrl_timesteps += 1
        return action, state


class JSRLCurriculumCallback(BaseCallback):
    """Advance the horizon downward as the student's eval reward plateaus.

    Attached as EvalCallback's `callback_after_eval`, so `self.parent` is the
    EvalCallback and `self.parent.last_mean_reward` is the latest (pure-student)
    eval score. Advance one stage whenever the windowed eval reward stays within
    `tolerance` of the best seen so far.
    """

    def __init__(
        self,
        horizon_box: list[int],
        horizons: list[int],
        tolerance: float = 0.05,
        window: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.horizon_box = horizon_box
        self.horizons = list(horizons)
        self.tolerance = tolerance
        self.window = max(1, window)
        self.stage = 0
        self.recent: list[float] = []
        self.best = -np.inf
        self.tolerated = -np.inf
        self.horizon_box[0] = self.horizons[0]

    def _on_step(self) -> bool:
        mean_r = float(self.parent.last_mean_reward)
        self.recent.append(mean_r)
        self.recent = self.recent[-self.window:]
        moving = float(np.mean(self.recent))

        self.logger.record("jsrl/horizon", self.horizon_box[0])
        self.logger.record("jsrl/moving_mean_reward", moving)

        at_last_stage = self.stage >= len(self.horizons) - 1
        if not at_last_stage:
            if self.best == -np.inf:
                self.best = moving
            elif moving >= self.tolerated:
                self.stage += 1
                self.horizon_box[0] = self.horizons[self.stage]
                if self.verbose:
                    print(f"[jsrl] horizon -> {self.horizon_box[0]} "
                          f"(stage {self.stage}/{len(self.horizons) - 1})")
        if moving >= self.best:
            self.best = moving
            self.tolerated = moving - self.tolerance * abs(moving)
        return True
