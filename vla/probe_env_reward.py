"""Decisive probe: does TaskMixEnv's reward wiring work at all?

Replays recorded demo actions for one seen task through the SAME env the
PPO students train in (TaskMixEnv pinned to that task + the demo's init
state) and checks that the sparse success reward fires. If replay cannot
earn reward here, every student result is void; if it can, scratch=0% is
a genuine exploration floor.

Usage: python probe_env_reward.py [--task 4] [--n 3]
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vla_ppo_common import TaskMixEnv  # noqa: E402


def demo_file(task_id: int) -> str:
    from libero.libero import benchmark, get_libero_path

    bm = benchmark.get_benchmark_dict()["libero_object"]()
    return os.path.join(get_libero_path("datasets"),
                        bm.get_task_demonstration(task_id))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", type=int, default=4)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()

    env = TaskMixEnv([a.task], seed=0)
    path = demo_file(a.task)
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for di in range(a.n):
            d = f["data"][demos[di]]
            actions = np.asarray(d["actions"])
            init = np.asarray(d["states"][0])
            env.reset()
            env._cur.reset()
            raw = env._cur.set_init_state(init)
            env._t = 0
            total, done_at = 0.0, None
            for t, act in enumerate(actions):
                _, r, te, tr, _ = env.step(act)
                total += r
                if te or tr:
                    done_at = (t, "terminated" if te else "truncated")
                    break
            print(f"demo {di}: total reward {total:.1f}, end {done_at}, "
                  f"{len(actions)} demo steps")
    env.close()
    print("ENV REWARD PROBE DONE")


if __name__ == "__main__":
    main()
