"""Behavior-clone the teacher on the SEEN half of libero_object.

Trains GaussianBCPolicy by negative log-likelihood on the LIBERO demo
hdf5 files of tasks 0-4 (50 demos each). This is "teacher v0": the local,
VLA-shaped stand-in; swapping in a token-based VLA later changes only the
policy class, not the transfer machinery.

NOTE (image convention): BC is only as good as train/eval consistency.
After training, eval_policy.py rolls the SAME preprocess_obs() pipeline;
check_image_convention() below compares a demo frame to an env frame so a
vertical-flip mismatch (a known LIBERO/robosuite gotcha) is caught early.

Usage:
    python train_teacher_bc.py                 # 30 epochs -> teacher_bc.pt
    python train_teacher_bc.py --epochs 5 --out smoke.pt --max-demos 2
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch

from vla_common import (
    CHUNK,
    task_onehot,
    IMG_SIZE,
    SEEN_TASKS,
    GaussianBCPolicy,
    make_env,
    preprocess_obs,
)


def demo_file(task_id: int) -> str:
    from libero.libero import benchmark, get_libero_path

    bm = benchmark.get_benchmark_dict()["libero_object"]()
    return os.path.join(get_libero_path("datasets"),
                        bm.get_task_demonstration(task_id))


def load_frames(task_ids, max_demos: int | None = None):
    import cv2

    imgs, props, acts = [], [], []
    for tid in task_ids:
        with h5py.File(demo_file(tid), "r") as f:
            demos = sorted(f["data"].keys(),
                           key=lambda k: int(k.split("_")[-1]))
            if max_demos:
                demos = demos[:max_demos]
            for d in demos:
                g = f["data"][d]
                rgb = g["obs"]["agentview_rgb"][:]          # (T,128,128,3)
                wrist = g["obs"]["eye_in_hand_rgb"][:]
                joint = g["obs"]["joint_states"][:]
                grip = g["obs"]["gripper_states"][:]
                act = g["actions"][:]
                T = len(act)
                for t in range(T):
                    im = np.concatenate([rgb[t], wrist[t]], axis=-1)
                    imgs.append(np.transpose(im, (2, 0, 1)))
                    props.append(np.concatenate([joint[t], grip[t], task_onehot(tid)]))
                    idx = np.minimum(np.arange(t, t + CHUNK), T - 1)
                    acts.append(act[idx].reshape(-1))
        print(f"[bc] task {tid}: cumulative frames={len(acts)}")
    return (np.array(imgs, dtype=np.uint8),
            np.array(props, dtype=np.float32),
            np.array(acts, dtype=np.float32))


def check_image_convention() -> None:
    """Compare a demo frame to a live env frame; warn if one looks
    vertically flipped relative to the other (row-brightness profile)."""
    import cv2

    with h5py.File(demo_file(SEEN_TASKS[0]), "r") as f:
        d0 = sorted(f["data"].keys())[0]
        demo_img = f["data"][d0]["obs"]["agentview_rgb"][0]
    env = make_env(SEEN_TASKS[0])
    obs = env.reset()
    env_img = obs["agentview_image"]
    env.close()
    prof = lambda im: im.mean(axis=(1, 2))
    a, b = prof(demo_img), prof(env_img)
    same = np.corrcoef(a, b)[0, 1]
    flip = np.corrcoef(a, b[::-1])[0, 1]
    print(f"[bc] image convention corr: as-is={same:.2f} flipped={flip:.2f}")
    if flip > same + 0.2:
        print("[bc] WARNING: env frames look vertically FLIPPED vs demos -- "
              "fix preprocess before trusting eval!")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-demos", type=int, default=None)
    p.add_argument("--out", type=str, default="runs_vla/teacher_bc.pt")
    p.add_argument("--device", type=str,
                   default="mps" if torch.backends.mps.is_available() else "cpu")
    a = p.parse_args()

    check_image_convention()
    imgs, props, acts = load_frames(SEEN_TASKS, a.max_demos)
    n = len(acts)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    pol = GaussianBCPolicy().to(a.device)
    opt = torch.optim.Adam(pol.parameters(), lr=a.lr)
    props_t = torch.as_tensor(props, device=a.device)
    acts_t = torch.as_tensor(acts, device=a.device)

    for ep in range(a.epochs):
        perm = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, a.batch_size):
            idx = perm[i:i + a.batch_size]
            im = torch.as_tensor(imgs[idx], device=a.device).float() / 255.0
            dist = pol.distribution(im, props_t[idx])
            loss = -dist.log_prob(acts_t[idx]).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        print(f"[bc] epoch {ep+1}/{a.epochs}  nll={tot/n:.3f}")
        torch.save(pol.state_dict(), a.out)
    print(f"[bc] saved -> {a.out}")


if __name__ == "__main__":
    main()
