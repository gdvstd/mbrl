"""Standard eval/checkpoint callback wiring, shared across tracks."""
from __future__ import annotations

import os

from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)


def make_callbacks(
    eval_env: gym.Env,
    outdir: str,
    eval_freq: int,
    n_eval_episodes: int,
    early_stop_threshold: float | None = None,
    checkpoint_freq: int = 50_000,
    after_eval_callback=None,
    name_prefix: str = "sac",
) -> list:
    """Standard eval + checkpoint callbacks.

    Pass `early_stop_threshold=None` to always run the full step budget -- the
    right choice when comparing baseline vs transfer under a FIXED budget.
    `after_eval_callback` runs after every evaluation (used by JSRL to advance
    its curriculum horizon based on the student's eval reward).
    """
    eval_dir = os.path.join(outdir, "eval")
    ckpt_dir = os.path.join(outdir, "checkpoints")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    on_new_best = (
        None
        if early_stop_threshold is None
        else StopTrainingOnRewardThreshold(
            reward_threshold=early_stop_threshold, verbose=1
        )
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=outdir,
        log_path=eval_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        callback_on_new_best=on_new_best,
        callback_after_eval=after_eval_callback,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=checkpoint_freq, save_path=ckpt_dir, name_prefix=name_prefix
    )
    return [eval_cb, ckpt_cb]
