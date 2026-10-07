"""Calibration probe: scratch MaskablePPO on a target env variant.

Goal: find the paper-unspecified env parameters (max_steps, start rooms)
under which No-Transfer's curve matches the paper's (altgoal solved <200K,
locked solved <1M). Only scratch is run; methods are compared later.
"""
import argparse, os
from paper_common import (N_ENVS, build_mppo, make_paper_callbacks,
                          make_paper_vec_env)
from fourroom_env import SCENARIOS

p = argparse.ArgumentParser()
p.add_argument("--scenario", required=True)
p.add_argument("--max-steps", type=int, required=True)
p.add_argument("--timesteps", type=int, required=True)
p.add_argument("--tag", required=True)
p.add_argument("--agent-start", type=int, nargs=2, default=None)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

tgt = SCENARIOS[a.scenario][1]
kw = {"max_steps": a.max_steps}
if a.agent_start: kw["agent_start"] = tuple(a.agent_start)
outdir = f"runs/calib_{a.scenario}_{a.tag}"
os.makedirs(outdir, exist_ok=True)
train = make_paper_vec_env(tgt, a.seed, env_kwargs=kw,
                           monitor_dir=os.path.join(outdir, "train_monitor"))
ev = make_paper_vec_env(tgt, a.seed + 1000, n_envs=1, env_kwargs=kw)
m = build_mppo(train, a.seed)
cbs = make_paper_callbacks(ev, outdir, eval_freq=max(10_000 // N_ENVS, 1),
                           n_eval_episodes=20)
m.learn(total_timesteps=a.timesteps, callback=cbs, progress_bar=False)
