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


class ClipGateStrategy:
    """Absolute threshold with rate GUARDRAILS: tau = the fixed init_tau
    clipped into [quantile(rate_lo), quantile(rate_hi)] of a rolling buffer
    of recent target scores.

    Motivation (saerate 3-seed result): pure rate-targeting pins the
    volume but forces advice onto relatively-ID-yet-actually-OOD states
    when nothing truly ID is around -- the fixed gate's fast seeds owed
    their speed to ABSOLUTE ID-ness, not volume. Here the absolute tau
    stays in charge except when it would starve (<rate_lo open) or blow
    out (>rate_hi open). lower-is-ID scores only (recon error)."""

    def __init__(self, score_fn, init_tau: float, rate_lo: float = 0.01,
                 rate_hi: float = 0.10, delta0: float = 1.0,
                 end_frac: float = 0.5, buf_size: int = 20000,
                 warmup: int = 2048) -> None:
        self.score_fn = score_fn
        self.init_tau = init_tau
        self.rate_lo = rate_lo
        self.rate_hi = rate_hi
        self.delta0 = delta0
        self.end_frac = end_frac
        self.buf_size = buf_size
        self.warmup = warmup
        self._buf = th.empty(0)

    def _tau(self) -> float:
        if len(self._buf) < self.warmup:
            return self.init_tau
        lo = float(th.quantile(self._buf, self.rate_lo))
        hi = float(th.quantile(self._buf, self.rate_hi))
        return min(max(self.init_tau, lo), hi)

    def guidance_mask(self, teacher, obs_tensor, episode_steps, progress):
        p = _linear_decay(self.delta0, self.end_frac, progress)
        s = self.score_fn(obs_tensor).cpu().float()
        self._buf = th.cat([self._buf, s])[-self.buf_size:]
        return (s <= self._tau()) & (th.rand(obs_tensor.shape[0]) < p)


class KSGateStrategy:
    """KS/Youden-optimal gate under a PU mixture model (online calibration).

    Model: target visitation recon errors are a mixture
    F_T = pi * F_S + (1 - pi) * F_OOD with the ID component matching the
    source calibration distribution F_S. Then for any tau,
        F_S(tau) - F_T(tau) = (1 - pi) * [TPR(tau) - FPR(tau)],
    so argmax_tau of the OBSERVABLE CDF gap equals the Youden-optimal
    threshold of the LATENT ID-vs-OOD classifier, with no knowledge of pi.

    Online procedure: keep the fixed calibration sample (sorted); maintain
    a rolling buffer of recent target scores; every `recompute_every` new
    scores set tau* = argmax_c [F_S(c) - F_T(c)] over the pooled candidate
    points and EMA-smooth. D = max gap = (1-pi)*separability is a gate
    health signal: if D < d_min there is no reliably ID region in current
    visitation and the gate stays closed. Before warmup, init_tau is used.
    lower-is-ID scores (recon error) only; coin delta(t) unchanged."""

    def __init__(self, score_fn, cal_scores, init_tau: float,
                 d_min: float = 0.1, delta0: float = 1.0,
                 end_frac: float = 0.5, buf_size: int = 20000,
                 warmup: int = 2048, recompute_every: int = 512,
                 ema: float = 0.5, need: str = "none", eta: float = 0.5,
                 kappa: float = 0.5, coin: str = "clock") -> None:
        self.score_fn = score_fn
        self.cal = th.as_tensor(cal_scores, dtype=th.float32).sort().values
        self.init_tau = init_tau
        # NEED term ("entropy": student masked entropy >= eta*max;
        # "disagree": pi_S(teacher argmax action) <= kappa) and coin mode
        # ("clock" = linear decay; "success" = delta0*(1 - recent train
        # success rate), per-seed adaptive). student_policy / model are
        # attached after the model is built; until then NEED is inactive
        # and the clock coin is used.
        self.need = need
        self.eta = eta
        self.kappa = kappa
        self.coin = coin
        self.student_policy = None
        self.model = None
        self.d_min = d_min
        self.delta0 = delta0
        self.end_frac = end_frac
        self.buf_size = buf_size
        self.warmup = warmup
        self.recompute_every = recompute_every
        self.ema = ema
        self._buf = th.empty(0)
        self._since = 0
        self._tau = init_tau
        self._D = 1.0  # optimistic until first estimate
        self._n_recomputes = 0

    def _recalibrate(self) -> None:
        buf = self._buf.sort().values
        # candidate thresholds: pooled sample (calibration + buffer)
        cand = th.cat([self.cal, buf]).sort().values
        F_S = th.searchsorted(self.cal, cand, right=True).float() / len(self.cal)
        F_T = th.searchsorted(buf, cand, right=True).float() / len(buf)
        gap = F_S - F_T
        i = int(th.argmax(gap))
        self._D = float(gap[i])
        tau_star = float(cand[i])
        self._tau = (1 - self.ema) * self._tau + self.ema * tau_star
        self._n_recomputes += 1
        if self._n_recomputes % 50 == 1:
            print(f"[ks-gate] recal#{self._n_recomputes}: tau*={tau_star:.6f}"
                  f" tau={self._tau:.6f} D={self._D:.3f}"
                  f"{' (gate OFF)' if self._D < self.d_min else ''}",
                  flush=True)

    def _coin_p(self, progress: float) -> float:
        if self.coin == "success" and self.model is not None:
            buf = self.model.ep_info_buffer
            if buf is not None and len(buf) >= 10:
                succ = sum(e["r"] > 0 for e in buf) / len(buf)
                return self.delta0 * (1.0 - succ)
            return self.delta0
        return _linear_decay(self.delta0, self.end_frac, progress)

    def guidance_mask(self, teacher, obs_tensor, episode_steps, progress,
                      action_masks=None):
        p = self._coin_p(progress)
        s = self.score_fn(obs_tensor).cpu().float()
        self._buf = th.cat([self._buf, s])[-self.buf_size:]
        self._since += len(s)
        if len(self._buf) >= self.warmup and self._since >= self.recompute_every:
            self._recalibrate()
            self._since = 0
        if len(self._buf) >= self.warmup and self._D < self.d_min:
            return th.zeros(obs_tensor.shape[0], dtype=th.bool)
        tau = self._tau if len(self._buf) >= self.warmup else self.init_tau
        gate = (s <= tau) & (th.rand(obs_tensor.shape[0]) < p)
        if self.need != "none" and self.student_policy is not None \
                and gate.any():
            with th.no_grad():
                s_dist = self.student_policy.get_distribution(
                    obs_tensor, action_masks=action_masks)
                if self.need == "entropy":
                    ent = s_dist.entropy().cpu()
                    if action_masks is not None:
                        n_valid = th.as_tensor(action_masks).sum(-1).clamp(min=2)
                    else:
                        n_valid = th.full_like(ent, 7)
                    gate &= ent / th.log(n_valid.float()) >= self.eta
                else:  # disagree
                    t_dist = teacher.distribution(
                        obs_tensor, action_masks=action_masks)
                    a_T = t_dist.distribution.probs.argmax(-1)
                    p_agree = s_dist.distribution.probs.gather(
                        -1, a_T.unsqueeze(-1)).squeeze(-1).cpu()
                    gate &= p_agree <= self.kappa
        return gate


class FullKSGateStrategy:
    """Full-history KS gate (ablation arm): same two-sample argmax as
    KSGateStrategy, but the target buffer GROWS WITHOUT BOUND and is
    seeded with the pre-training masked-random target sample.

    Rationale: recon scores are values of a FROZEN teacher+SAE, so
    samples never go stale -- only their visitation weighting evolves,
    which the pi-invariance of the argmax absorbs. The seed sample
    removes the cold-start bias; unbounded accumulation keeps deep-OOD
    coverage even after the student's visitation concentrates on its own
    solution paths. No window, no EMA, no d_min -- growing n
    self-stabilizes tau."""

    def __init__(self, score_fn, cal_scores, seed_scores, init_tau: float,
                 delta0: float = 1.0, end_frac: float = 0.5,
                 recompute_every: int = 2048) -> None:
        self.score_fn = score_fn
        self.cal = th.as_tensor(cal_scores, dtype=th.float32).sort().values
        self._chunks = [th.as_tensor(seed_scores, dtype=th.float32)]
        self._tau = init_tau  # = the KS-once value of the seed sample
        self.delta0 = delta0
        self.end_frac = end_frac
        self.recompute_every = recompute_every
        self._since = 0
        self._n_recomputes = 0

    def _recalibrate(self) -> None:
        buf = th.cat(self._chunks).sort().values
        self._chunks = [buf]  # keep stored sorted/compact
        cand = th.cat([self.cal, buf]).sort().values
        F_S = th.searchsorted(self.cal, cand, right=True).float() / len(self.cal)
        F_T = th.searchsorted(buf, cand, right=True).float() / len(buf)
        gap = F_S - F_T
        i = int(th.argmax(gap))
        self._tau = float(cand[i])
        self._n_recomputes += 1
        if self._n_recomputes % 50 == 1:
            print(f"[ksfull-gate] recal#{self._n_recomputes}: "
                  f"tau={self._tau:.6f} D={float(gap[i]):.3f} "
                  f"n_buf={len(buf)}", flush=True)

    def guidance_mask(self, teacher, obs_tensor, episode_steps, progress):
        p = _linear_decay(self.delta0, self.end_frac, progress)
        s = self.score_fn(obs_tensor).cpu().float()
        self._chunks.append(s)
        self._since += len(s)
        if self._since >= self.recompute_every:
            self._recalibrate()
            self._since = 0
        return (s <= self._tau) & (th.rand(obs_tensor.shape[0]) < p)


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
