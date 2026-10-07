"""(Vendored copy of shared/strategies.py so vla/ stays self-contained
for cluster deploys.) Guidance strategies: WHO acts at each env step (teacher vs student).

Each strategy returns a boolean mask over the vectorized envs; True = the
teacher acts in that env this step. MixedPolicyPPO consumes the mask during
rollout collection. All three follow the EBTL paper's baselines/method:

  * AAStrategy   -- Action Advising: teacher acts with probability delta(t),
    linearly decayed to 0 by `end_frac` of the training budget.
  * JSRLStrategy -- JumpStart RL (time-based variant used in the EBTL paper):
    teacher acts for the first h(t) steps of each episode, h linearly decayed.
  * EBTLStrategy -- teacher acts only when the state is in-distribution for
    the teacher (energy score phi(s) >= tau) AND the decaying probability
    fires (Algorithm 1: a_t ~ pi_T if -E(s_t) >= tau and p < delta(t)).

delta(t) = delta0 * max(0, 1 - progress/end_frac), progress in [0, 1].
"""
from __future__ import annotations

import torch as th


def _linear_decay(delta0: float, end_frac: float, progress: float) -> float:
    if end_frac <= 0:
        return 0.0
    return delta0 * max(0.0, 1.0 - progress / end_frac)


class AAStrategy:
    def __init__(self, delta0: float = 1.0, end_frac: float = 0.5) -> None:
        self.delta0 = delta0
        self.end_frac = end_frac

    def guidance_mask(self, teacher, obs_tensor, episode_steps, progress):
        p = _linear_decay(self.delta0, self.end_frac, progress)
        return th.rand(obs_tensor.shape[0]) < p


class JSRLStrategy:
    def __init__(self, h0: int = 50, end_frac: float = 0.5) -> None:
        self.h0 = h0
        self.end_frac = end_frac

    def guidance_mask(self, teacher, obs_tensor, episode_steps, progress):
        h = _linear_decay(float(self.h0), self.end_frac, progress)
        return th.as_tensor(episode_steps < h)


class EBTLStrategy:
    def __init__(self, tau: float, delta0: float = 1.0,
                 end_frac: float = 0.5) -> None:
        self.tau = tau
        self.delta0 = delta0
        self.end_frac = end_frac

    def guidance_mask(self, teacher, obs_tensor, episode_steps, progress):
        p = _linear_decay(self.delta0, self.end_frac, progress)
        in_dist = teacher.energy_score(obs_tensor) >= self.tau
        return in_dist.cpu() & (th.rand(obs_tensor.shape[0]) < p)
