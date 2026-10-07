"""Manually run and watch BipedalWalker episodes with render_mode="human".

Transfer-RL toy setup:
    Task A (teacher):  BipedalWalker-v3          -- flat terrain
    Task B (student):  BipedalWalkerHardcore-v3  -- pits, stumps, stairs

Both tasks share the SAME 24-D observation and SAME 4-D action space, so a
teacher policy trained on the flat task can be transferred directly (e.g. as a
weight initialization) to bootstrap the hardcore task. Flat walking is a
sub-skill of hardcore walking, so positive transfer is expected.

Observation (24-D):
    [0]   hull angle            [1]   hull angular velocity
    [2]   x velocity            [3]   y velocity
    [4,5] hip-1 angle, speed    [6,7] knee-1 angle, speed
    [8]   leg-1 ground contact
    [9,10] hip-2 angle, speed   [11,12] knee-2 angle, speed
    [13]  leg-2 ground contact
    [14..23] 10 lidar rangefinder readings
Action (4-D): motor torques for [hip-1, knee-1, hip-2, knee-2] in [-1, 1].

Usage (inside the venv):
    python explore_bipedal.py --env normal
    python explore_bipedal.py --env hardcore --episodes 3
    python explore_bipedal.py --env normal --policy cpg   # open-loop gait

A pygame window opens; close it or press Ctrl+C to stop. Runs with plain
`python` on macOS (Box2D + pygame, no MuJoCo).
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import math

import numpy as np
import gymnasium as gym

ENV_IDS = {
    "normal": "BipedalWalker-v3",         # task A  (teacher)
    "hardcore": "BipedalWalkerHardcore-v3",  # task B  (student)
}


def cpg_action(t: int) -> np.ndarray:
    """Open-loop central-pattern-generator gait (NOT a trained policy).

    Two legs oscillate out of phase so the walker shuffles instead of just
    flailing -- purely to make the demo look purposeful. Do not expect it to
    walk well; the trained teacher will replace this later.
    """
    phase = t * 0.15
    hip1 = 0.8 * math.sin(phase)
    knee1 = 0.8 * math.sin(phase + math.pi / 2)
    hip2 = 0.8 * math.sin(phase + math.pi)          # opposite leg
    knee2 = 0.8 * math.sin(phase + math.pi + math.pi / 2)
    return np.array([hip1, knee1, hip2, knee2], dtype=np.float32)


def run(env_id: str, episodes: int, policy: str, seed: int) -> None:
    env = gym.make(env_id, render_mode="human")
    print(f"[env] {env_id}")
    print(f"[env] observation_space = {env.observation_space.shape}")
    print(f"[env] action_space      = {env.action_space}")

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            if policy == "cpg":
                action = cpg_action(steps)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        # Reward convention: forward progress rewarded, -100 on a fall.
        fell = total_reward < -50
        print(
            f"[episode {ep + 1}/{episodes}] steps={steps} "
            f"return={total_reward:.1f} fell={fell}"
        )

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENV_IDS, default="normal")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--policy", choices=["random", "cpg"], default="cpg")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run(ENV_IDS[args.env], args.episodes, args.policy, args.seed)


if __name__ == "__main__":
    main()
