"""Phase-0 behavioral probe: does concept steering move OpenVLA behavior?

For each config (baseline / concept x alpha / random control) roll fixed
init-state episodes on libero_object tasks and record end-effector metrics:
  step_disp  mean per-step |delta EE-pos|      ("fast" up / "slow" down)
  max_z      max EE height                     ("up" up / "down" down)
  net_disp   |EE(final) - EE(0)|
  success    task success
Episodes are PAIRED with baseline via identical init states; we report the
paired mean difference and a paired t statistic per metric.

Usage (GPU node):
    python steer/steer_probe.py --lora <dir> --library runs_vla/steer/lens_lora.json \
        --tasks 0 2 5 --episodes 3 [--concepts fast slow up down stop push]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla_common import get_init_states, make_env  # noqa: E402
from vla_energy import phi_and_action, task_language  # noqa: E402
from steer.steering import SteeringHooks, random_entries  # noqa: E402

DEFAULT_CONCEPTS = ["fast", "slow", "up", "down", "stop", "push"]


def ee_pos(env, obs) -> np.ndarray:
    if isinstance(obs, dict) and "robot0_eef_pos" in obs:
        return np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    return np.asarray(env.env.sim.data.site_xpos[
        env.env.sim.model.site_name2id("gripper0_grip_site")],
        dtype=np.float64)


def rollout(model, processor, task_id: int, init, lang: str,
            max_steps: int) -> dict:
    env = make_env(task_id)
    env.reset()
    obs = env.set_init_state(init)
    pos = [ee_pos(env, obs)]
    done, r = False, 0.0
    for _ in range(max_steps):
        _, action = phi_and_action(model, processor, obs, lang)
        obs, r, done, _ = env.step(action)
        pos.append(ee_pos(env, obs))
        if done:
            break
    env.close()
    pos = np.stack(pos)
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return {"step_disp": float(steps.mean()), "max_z": float(pos[:, 2].max()),
            "net_disp": float(np.linalg.norm(pos[-1] - pos[0])),
            "success": float(done and r > 0), "n_steps": len(steps)}


def paired_t(diff: np.ndarray) -> float:
    sd = diff.std(ddof=1)
    return float(diff.mean() / (sd / np.sqrt(len(diff)) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--norm-stats", default=None)
    ap.add_argument("--library", required=True)
    ap.add_argument("--tasks", type=int, nargs="*", default=[0, 2, 5])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--concepts", nargs="*", default=DEFAULT_CONCEPTS)
    ap.add_argument("--alphas", type=float, nargs="*", default=[5.0, 10.0])
    ap.add_argument("--n-neurons", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from eval_openvla import load_teacher
    norm = (a.norm_stats
            or os.path.join(os.path.dirname((a.lora or ".").rstrip("/")),
                            "norm_stats.json"))
    if a.lora is None and a.norm_stats is None:
        # base model: norm stats must be given explicitly
        raise SystemExit("--norm-stats required when --lora is omitted")
    model, processor = load_teacher(a.lora, norm)
    lib = json.load(open(a.library))["library"]

    episodes = []
    for tid in a.tasks:
        inits = get_init_states(tid)
        lang = task_language(tid)
        for ep in range(a.episodes):
            episodes.append((tid, inits[ep % len(inits)], lang))

    configs: list[tuple[str, list[dict] | None, float]] = [
        ("baseline", None, 0.0)]
    for c in a.concepts:
        if not lib.get(c):
            print(f"[probe] concept '{c}' empty in library, skipped")
            continue
        for al in a.alphas:
            configs.append((f"{c}@a{al:g}", lib[c][:a.n_neurons], al))
    configs.append(("random@a10", random_entries(model, a.n_neurons), 10.0))

    metrics = ["step_disp", "max_z", "net_disp", "success"]
    results: dict[str, list[dict]] = {}
    for name, entries, al in configs:
        rows = []
        for tid, init, lang in episodes:
            if entries is None:
                rows.append(rollout(model, processor, tid, init, lang,
                                    a.max_steps))
            else:
                with SteeringHooks(model, entries, al):
                    rows.append(rollout(model, processor, tid, init, lang,
                                        a.max_steps))
        results[name] = rows
        m = {k: np.mean([r[k] for r in rows]) for k in metrics}
        print(f"[probe] {name:14s} " + "  ".join(
            f"{k}={m[k]:.4f}" for k in metrics), flush=True)

    base = {k: np.array([r[k] for r in results["baseline"]])
            for k in metrics}
    print("\n== paired vs baseline (mean diff, t) ==")
    for name, _, _ in configs[1:]:
        vals = {k: np.array([r[k] for r in results[name]]) for k in metrics}
        line = f"{name:14s} "
        for k in metrics:
            d = vals[k] - base[k]
            line += f"{k}: {d.mean():+.4f} (t={paired_t(d):+.1f})   "
        print(line)

    out = a.out or os.path.join(os.path.dirname(a.library),
                                "steer_probe.json")
    with open(out, "w") as f:
        json.dump({"configs": [c[0] for c in configs], "results": results,
                   "tasks": a.tasks, "episodes": a.episodes,
                   "n_neurons": a.n_neurons, "lora": a.lora}, f, indent=1)
    print(f"saved -> {out}")
    print("STEER PROBE DONE")


if __name__ == "__main__":
    main()
