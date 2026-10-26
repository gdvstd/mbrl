"""Guidance strategies: WHO acts at each env step (teacher vs student).

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


class ScoreGateStrategy:
    """Generic EBTL-shaped gate for an arbitrary OOD score: advise iff
    score(obs) is on the ID side of tau AND the decaying coin fires.
    `higher_is_id=False` suits reconstruction-error-style scores."""

    def __init__(self, score_fn, tau: float, higher_is_id: bool = True,
                 delta0: float = 1.0, end_frac: float = 0.5) -> None:
        self.score_fn = score_fn
        self.tau = tau
        self.higher_is_id = higher_is_id
        self.delta0 = delta0
        self.end_frac = end_frac

    def guidance_mask(self, teacher, obs_tensor, episode_steps, progress):
        p = _linear_decay(self.delta0, self.end_frac, progress)
        s = self.score_fn(obs_tensor).cpu()
        in_dist = (s >= self.tau) if self.higher_is_id else (s <= self.tau)
        return in_dist & (th.rand(obs_tensor.shape[0]) < p)


class RateGateStrategy:
    """Rate-targeted score gate: instead of a fixed absolute threshold,
    tau tracks the `rate`-quantile of a rolling buffer of recent TARGET
    scores, so the gate opens on ~`rate` of coin-fired steps regardless of
    the score's absolute scale.

    Motivation (10-seed advlog analysis): the fixed gap-rule tau is
    brittle across seeds -- the same multiplier lands on either side of
    the target recon distribution, producing advice starvation (altgoal
    stragglers, 3-4%) or blowout (locked seeds at ~75% ~= AA). Rank-based
    gating keeps the selection (most-ID states first) while pinning the
    volume. Coin delta(t) is unchanged; until `warmup` scores accumulate
    the fixed `init_tau` is used.
    """

    def __init__(self, score_fn, rate: float, init_tau: float,
                 higher_is_id: bool = True, delta0: float = 1.0,
                 end_frac: float = 0.5, buf_size: int = 20000,
                 warmup: int = 2048) -> None:
        self.score_fn = score_fn
        self.rate = rate
        self.init_tau = init_tau
        self.higher_is_id = higher_is_id
        self.delta0 = delta0
        self.end_frac = end_frac
        self.buf_size = buf_size
        self.warmup = warmup
        self._buf = th.empty(0)

    def _tau(self) -> float:
        if len(self._buf) < self.warmup:
            return self.init_tau
        # open on the `rate` most-ID fraction of recent target states
        q = 1.0 - self.rate if self.higher_is_id else self.rate
        return float(th.quantile(self._buf, q))

    def guidance_mask(self, teacher, obs_tensor, episode_steps, progress):
        p = _linear_decay(self.delta0, self.end_frac, progress)
        s = self.score_fn(obs_tensor).cpu().float()
        self._buf = th.cat([self._buf, s])[-self.buf_size:]
        tau = self._tau()
        in_dist = (s >= tau) if self.higher_is_id else (s <= tau)
        return in_dist & (th.rand(obs_tensor.shape[0]) < p)


class GapStrategy:
    """Competence-GAP gate: advise only where the teacher is familiar
    (phi >= tau, as in EBTL) AND the student is still uncertain (masked
    policy entropy >= eta * max entropy over the valid actions).

    Motivation (oracle + GAE analyses): the empirical value of advice
    follows the teacher-student competence gap, not teacher familiarity
    alone -- at the highest-phi states the student is already competent
    and advice yields no advantage. The second condition targets exactly
    those states the student has not yet mastered.

    Requires `student_policy` to be set after the model is built (the
    strategy is constructed before the model exists). Falls back to plain
    EBTL gating until it is set.
    """

    def __init__(self, tau: float, eta: float = 0.5, delta0: float = 1.0,
                 end_frac: float = 0.5, use_coin: bool = True) -> None:
        self.tau = tau
        self.eta = eta
        self.delta0 = delta0
        self.end_frac = end_frac
        # the entropy condition is self-annealing (H_S drops as the student
        # masters states), so the hand-tuned decay coin is redundant --
        # use_coin=False makes the gate purely state-driven.
        self.use_coin = use_coin
        self.student_policy = None

    def guidance_mask(self, teacher, obs_tensor, episode_steps, progress,
                      action_masks=None):
        gate = (teacher.energy_score(obs_tensor) >= self.tau).cpu()
        if self.use_coin:
            p = _linear_decay(self.delta0, self.end_frac, progress)
            gate &= th.rand(obs_tensor.shape[0]) < p
        if self.student_policy is not None and gate.any():
            dist = self.student_policy.get_distribution(
                obs_tensor, action_masks=action_masks)
            ent = dist.entropy().cpu()
            if action_masks is not None:
                n_valid = th.as_tensor(action_masks).sum(-1).clamp(min=2)
            else:
                n_valid = th.full_like(ent, 7)
            uncertain = ent / th.log(n_valid.float()) >= self.eta
            gate &= uncertain
        return gate
