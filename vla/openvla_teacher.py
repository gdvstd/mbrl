"""OpenVLA-LoRA teacher adapter for the mixed-policy PPO student.

Wraps the finetuned OpenVLA-7B (eval_openvla.load_teacher, same pinned
loading path that passed the gate at 64% seen / 0% unseen) behind the
FrozenBCTeacher interface MixedPolicyVLAPPO expects:

    .distribution(obs_tensor_dict) -> Normal over the 7-dim action

OpenVLA is deterministic per frame, so the "distribution" is a narrow
Gaussian N(a_T, sigma^2) around its predicted action -- sigma only gives
the behavior log-prob a density for the Eq.3-style importance ratio (the
sampled action is clamped to [-1, 1] like everything else).

The student's obs is preprocessed (128px, 6ch), which OpenVLA cannot
consume; the adapter instead pulls each env's RAW agentview frame and the
task's language instruction via vec_env.get_attr (TaskMixEnv stores both
every reset/step). attach(vec_env) must be called after the env is built.
"""
from __future__ import annotations

import numpy as np
import torch


class OpenVLATeacher:
    def __init__(self, lora_dir: str, norm_path: str, device: str = "cuda",
                 sigma: float = 0.05):
        from eval_openvla import load_teacher
        self.model, self.processor = load_teacher(lora_dir, norm_path,
                                                  device)
        self.device = device
        self.sigma = sigma
        self.vec_env = None

    def attach(self, vec_env) -> None:
        self.vec_env = vec_env

    @torch.no_grad()
    def distribution(self, obs_tensors: dict) -> torch.distributions.Normal:
        from eval_openvla import act
        assert self.vec_env is not None, "call attach(vec_env) first"
        imgs = self.vec_env.get_attr("last_raw_image")
        langs = self.vec_env.get_attr("task_language")
        mus = [act(self.model, self.processor, {"agentview_image": im},
                   lang, self.device) for im, lang in zip(imgs, langs)]
        out_dev = obs_tensors["proprio"].device
        mu = torch.as_tensor(np.stack(mus), dtype=torch.float32,
                             device=out_dev).clamp(-1.0, 1.0)
        return torch.distributions.Normal(
            mu, torch.full_like(mu, self.sigma))
