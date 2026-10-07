"""Shared SAC config and helpers for the BipedalWalker transfer-RL study.

Keeping the env makers, hyperparameters and callback wiring in ONE place
guarantees the teacher, the from-scratch student baseline and every future
transfer variant use an identical algorithm setup -- so any difference in the
learning curve is attributable to the transfer method, not to config drift.
"""
from __future__ import annotations

import os

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor

ENV_IDS = {
    "normal": "BipedalWalker-v3",          # Task A (teacher)
    "hardcore": "BipedalWalkerHardcore-v3",  # Task B (student)
}
SOLVED_REWARD = 300.0

# RL-Baselines3-Zoo tuned config for BipedalWalker-v3 (SAC),
# net_arch fixed to [256, 256] (a simple 3-layer MLP: 24->256->256->4).
SAC_HYPERPARAMS = dict(
    learning_rate=7.3e-4,
    buffer_size=300_000,
    batch_size=256,
    ent_coef="auto",
    gamma=0.98,
    tau=0.02,
    train_freq=64,
    gradient_steps=64,
    learning_starts=10_000,
    use_sde=True,
    policy_kwargs=dict(log_std_init=-3, net_arch=[256, 256]),
)


def make_env(
    env_id: str,
    seed: int,
    monitor_path: str | None = None,
    base_wrapper=None,
) -> gym.Env:
    """Create a Monitored env.

    `base_wrapper` (a callable env -> env) is applied to the RAW env BEFORE the
    Monitor, so transfer strategies can inject e.g. a JSRL guide-rollin wrapper
    or a reward-shaping wrapper on the training env only (eval envs pass none).
    """
    env = gym.make(env_id)
    if base_wrapper is not None:
        env = base_wrapper(env)
    env = Monitor(env, filename=monitor_path)
    env.reset(seed=seed)
    return env


def build_sac(
    env: gym.Env,
    seed: int,
    device: str = "cpu",
    tb_dir: str | None = None,
    algo_cls: type[SAC] = SAC,
    **overrides,
) -> SAC:
    """Construct a SAC model with the shared hyperparameters.

    `algo_cls` lets a transfer variant substitute a SAC subclass (e.g. JSRL's
    action-switching algorithm) while keeping identical hyperparameters.
    `overrides` lets a caller tweak individual hyperparameters (e.g. a warm-start
    transfer variant that wants a smaller `learning_starts`) without duplicating
    the whole config.
    """
    params = {**SAC_HYPERPARAMS, **overrides}
    return algo_cls(
        "MlpPolicy",
        env,
        seed=seed,
        device=device,
        tensorboard_log=tb_dir,
        verbose=1,
        **params,
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
