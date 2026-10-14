"""Oracle-optimal teacher: an upper-bound control for advising baselines.

Decodes the TRUE env state from the full observation (type plane: wall=2,
door=4, key=5, goal=8, agent=10; state plane: open=0/closed=1/locked=2 and
the agent's dir), builds the per-layout BFS distance table D(s) from
oracle_advice, and advises uniformly over the OPTIMAL action set
A*(s) = {a : D(step(s,a)) = D(s)-1}.

Used with --oracle-teacher for aa/jsrl/ksrl: same advising schedules, but
advice quality is perfect by construction. The gap between oracle-X and X
isolates how much each schedule loses to imperfect teacher actions (EBTL
is excluded: an oracle has no energy score). Full-obs only -- an
egocentric view does not determine the state.

Carrying inference: drop is disabled, so in the locked target "no key
visible" <=> the agent carries it; the key cell is then irrelevant to all
reachable transitions, so layouts cache by (goal, key-or-None, door).
If the agent stands ON the door cell, the door object is hidden by the
agent in the encoding and the door is necessarily open.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch as th

from fourroom.fourroom_env import GAP_RIGHT
from fourroom.oracle_advice import CLOSED, LOCKED, OPEN, _step, dist_table

WALL, DOOR, KEY, GOAL, AGENT = 2, 4, 5, 8, 10
MG_OPEN, MG_CLOSED, MG_LOCKED = 0, 1, 2  # minigrid STATE_TO_IDX
ACTIONS = (0, 1, 2, 3, 5)


class _Dist:
    """Minimal stand-in for sb3's distribution wrapper."""

    def __init__(self, cat: th.distributions.Categorical) -> None:
        self.distribution = cat

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        if deterministic:
            return self.distribution.probs.argmax(-1)
        return self.distribution.sample()

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(actions)


class OracleTeacher:
    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self._tables: dict = {}  # layout id -> (layout, D)

    # ------------------------------------------------------------- decoding
    def _decode(self, tp, st):
        """One env's (type, state) planes -> (layout, oracle state)."""
        ax, ay = [int(v) for v in (tp == AGENT).nonzero()[0]]
        d = int(st[ax, ay])
        gpos = tuple(int(v) for v in (tp == GOAL).nonzero()[0])
        kidx = (tp == KEY).nonzero()
        key = tuple(int(v) for v in kidx[0]) if len(kidx) else None
        door_pos, door = None, LOCKED
        if self.scenario == "locked":
            door_pos = GAP_RIGHT
            didx = (tp == DOOR).nonzero()
            if len(didx):
                ms = int(st[didx[0][0], didx[0][1]])
                door = {MG_OPEN: OPEN, MG_CLOSED: CLOSED,
                        MG_LOCKED: LOCKED}[ms]
            else:  # agent stands on the door cell -> it must be open
                door = OPEN
        carry = 1 if (self.scenario == "locked" and key is None) else 0
        walls = frozenset((int(x), int(y))
                          for x, y in (tp == WALL).nonzero().tolist())
        lay_key = (gpos, key, door_pos, walls)
        if lay_key not in self._tables:
            lay = SimpleNamespace(walls=set(walls), key=key, goal=gpos,
                                  door=door_pos, width=tp.shape[0],
                                  height=tp.shape[1])
            self._tables[lay_key] = (lay, dist_table(lay))
        lay, D = self._tables[lay_key]
        return lay, D, (ax, ay, d, carry, door)

    # ------------------------------------------------------------ interface
    @th.no_grad()
    def distribution(self, obs_tensor: th.Tensor, action_masks=None) -> _Dist:
        obs = obs_tensor.round().long().cpu()
        n = obs.shape[0]
        probs = th.zeros(n, 7)
        for i in range(n):
            tp, st = obs[i, 0], obs[i, 2]
            lay, D, s = self._decode(tp, st)
            ds = D.get(s)
            best = []
            if ds is not None and ds > 0:
                best = [a for a in ACTIONS
                        if D.get(_step(lay, s, a)) == ds - 1]
            if best:
                probs[i, best] = 1.0 / len(best)
            elif action_masks is not None:  # unreachable fallback: uniform
                m = th.as_tensor(action_masks[i], dtype=th.bool)
                probs[i, m] = 1.0 / int(m.sum())
            else:
                probs[i, :3] = 1.0 / 3
        return _Dist(th.distributions.Categorical(probs=probs))

    @th.no_grad()
    def energy_score(self, obs_tensor: th.Tensor) -> th.Tensor:
        """No meaningful energy for an oracle; zeros keep advlog happy."""
        return th.zeros(obs_tensor.shape[0])
