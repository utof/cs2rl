"""Random-walker opponent: a scripted target that moves but never fights back (#152).

WHAT
----
Each walker row holds one move direction for a random number of ticks, then draws a
fresh one:

  * the direction is a move-head bin drawn uniformly from 1..8 (``MOVE_BINS``). Bin 0
    (no move) is never drawn for a walker without a mix, so such a row always presses a
    direction; a mix family can also stand still (``p_stop``) or press only some ticks
    (``duty``);
  * the hold is drawn uniformly from ``hold_min..hold_max`` ticks, both ends included
    (default ``HOLD_MIN..HOLD_MAX`` = 4..16);
  * every other discrete head is 0 and the aim is (0, 0): it never shoots, reloads,
    switches weapon, uses, crouches, jumps or turns.

A ``WalkerParams`` mix (``TRAIN_MIX``) makes the walker a DISTRIBUTION: at each
``reset(rows)`` a row draws one ``WalkerParams`` for its episode, plus a per-episode
``p_stop`` and ``duty`` inside the ranges that family gives. Training on one walker is
documented to overfit; ``HELD_OUT`` is a family kept out of ``TRAIN_MIX`` for evaluation.

``RandomWalker`` is the vectorised core over any number of rows (one numpy
``Generator``, no per-row Python loop), for a caller that owns its own action array.
``WalkerActor`` adapts it to the ``act(obs, st, vis_prev, env)`` shape of the
cs2rl.eval.baselines actors, for the scripted-bot harnesses
(cs2rl.experiment.oracle_tracker).

WHY
---
It is the moving target of #152 layer L1 (the oracle against something that moves),
built so a trainer can also use it as a scripted opponent: vectorised over any number of
rows, and import-light. The hold is load-bearing: measured on 2026-10-09 (the #152 L1
investigation, with the investigator's sketch walker, pre-#157), the Aug-31 Rung 1a
checkpoint killed a walker that redraws its direction every tick in 99/100 rounds, about
as often as a statue (100/100), and a hold-4..16 walker in 81/100. A per-tick redraw
jitters in place.

PITFALLS
--------
* Move bins are LOCAL to the agent's facing (cs2_movement.h's _LOCAL_MOVE_X/Y). The
  walker never turns, so its directions stay relative to its spawn facing.
* It is blind: it reads neither obs nor state. It walks into walls and never dodges.
  It also ignores the action mask, which is harmless for the move head: the mask
  restricts move only for dead agents, and process_movement skips dead agents.
* The random stream depends on the row count. Each step draws for the expiring rows
  only, first all their bins, then all their holds. Two walkers built from one seed
  agree only if they have the same number of rows and take the same number of steps.
* Without a ``mix`` the walker draws exactly what it drew before the mix existed
  (tests/eval/test_walker.py pins the stream), so lane-L1 numbers do not move. With a mix
  the stream also depends on the mix, and ``reset(rows)`` draws.
* ``duty`` is the fraction of ticks a row presses its direction. The move head has no
  slower bin, so it is a deterministic credit accumulator (no RNG draw): a duty of 0.5
  presses every other tick.
* Module scope imports numpy and cs2rl.spec.action only, so a light module (one the
  `python -m cs2rl.train --dump-config` path imports) may import this one.
  tests/eval/test_walker.py checks that in a fresh interpreter against
  tests/train/test_w1_modules.py's HEAVY list. Do not import eval.baselines here: it
  imports torch and env.c.cs2_env at module scope.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cs2rl.spec.action import ACTION_DIM, ACTION_HEAD_NAMES, ACTION_HEAD_SIZES, AIM_DIM

H_MOVE = ACTION_HEAD_NAMES.index("move")
# Bins 1..8 are the eight directions; 0 is "no move" and is never drawn.
MOVE_BINS = np.arange(1, ACTION_HEAD_SIZES[H_MOVE], dtype=np.int32)
HOLD_MIN = 4
HOLD_MAX = 16


@dataclass(frozen=True)
class WalkerParams:
    """One walker family: ``hold`` ticks per direction (inclusive range, drawn per hold),
    ``p_stop`` = probability that a new hold presses nothing (bin 0), ``duty`` = fraction
    of ticks the direction is pressed. ``p_stop`` and ``duty`` are (lo, hi) ranges drawn
    once per episode per row; a degenerate range (lo == hi) draws nothing."""
    hold: tuple[int, int] = (HOLD_MIN, HOLD_MAX)
    p_stop: tuple[float, float] = (0.0, 0.0)
    duty: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self):
        if not 1 <= self.hold[0] <= self.hold[1]:
            raise ValueError(f"need 1 <= hold lo <= hi, got {self.hold}")
        if not 0.0 <= self.p_stop[0] <= self.p_stop[1] <= 1.0:
            raise ValueError(f"need 0 <= p_stop lo <= hi <= 1, got {self.p_stop}")
        if not 0.0 < self.duty[0] <= self.duty[1] <= 1.0:
            raise ValueError(f"need 0 < duty lo <= hi <= 1, got {self.duty}")


# The Rung 1b training distribution: a statue, a fast walker, a stop-and-go walker.
# HELD_OUT (long straight runs) is absent from it, for evaluation.
STATUE = WalkerParams(hold=(1, 1), p_stop=(1.0, 1.0))
TRAIN_MIX = (STATUE, WalkerParams(hold=(4, 16)), WalkerParams(hold=(2, 8), p_stop=(0.25, 0.25)))
TRAIN_WEIGHTS = (0.2, 0.4, 0.4)
HELD_OUT = WalkerParams(hold=(24, 48))


class RandomWalker:
    """Vectorised hold-and-redraw move bins for ``n_rows`` rows.

    ``step()`` returns one int32 move bin per row. A row keeps its bin for its drawn
    hold, then draws a new bin and a new hold on the step after it expires. A row's
    first step after construction or ``reset()`` always draws.

    With ``mix`` (a sequence of ``WalkerParams``; ``weights`` default uniform) every
    row's family, ``p_stop`` and ``duty`` are drawn at construction and again at each
    ``reset(rows)``; without it ``hold_min``/``hold_max`` apply to every row for ever.
    """

    def __init__(self,
                 n_rows: int,
                 rng: np.random.Generator,
                 hold_min: int = HOLD_MIN,
                 hold_max: int = HOLD_MAX,
                 mix=None,
                 weights=None):
        if n_rows < 0:
            raise ValueError(f"n_rows must be >= 0, got {n_rows}")
        if not 1 <= hold_min <= hold_max:
            raise ValueError(f"need 1 <= hold_min <= hold_max, got {hold_min}..{hold_max}")
        self.rng = rng
        self.n_rows = n_rows
        self.hold_min, self.hold_max = int(hold_min), int(hold_max)
        # _left[i] = steps row i still returns its current bin before redrawing.
        self._left = np.zeros(n_rows, dtype=np.int64)
        self._move = np.zeros(n_rows, dtype=np.int32)
        self.mix = tuple(mix) if mix else None
        if self.mix is None:
            return
        self.weights = np.full(len(self.mix), 1.0 / len(self.mix)) if weights is None else \
            np.asarray(weights, dtype=np.float64)
        if self.weights.shape != (len(self.mix), ) or abs(self.weights.sum() - 1.0) > 1e-9:
            raise ValueError(f"weights must be {len(self.mix)} numbers summing to 1, "
                             f"got {self.weights}")
        # Columns: hold lo, hi; p_stop lo, hi; duty lo, hi. reset() overwrites a row's
        # p_stop and duty "lo" columns with the drawn value.
        self._fam = np.array([[*f.hold, *f.p_stop, *f.duty] for f in self.mix], dtype=np.float64)
        self._row = np.zeros((n_rows, 6))              # the same six numbers, per row, this episode
        self._credit = np.zeros(n_rows)
        self.reset()

    def reset(self, rows=None) -> None:
        """Expire ``rows``' holds (default every row), so their next ``step()`` redraws.

        With a mix, ``rows`` (an index array, bool mask or slice) also draw a new family,
        ``p_stop`` and ``duty`` for the new episode.
        """
        rows = slice(None) if rows is None else rows
        self._left[rows] = 0
        if self.mix is None:
            return
        idx = np.arange(len(self._left))[rows]
        if not len(idx):
            return
        fam = self._fam[self.rng.choice(len(self.mix), size=len(idx), p=self.weights)]
        for col in (2, 4):             # p_stop, then duty: draw only a non-degenerate range
            lo, hi = fam[:, col], fam[:, col + 1]
            if (lo != hi).any():
                fam[:, col] = lo + (hi - lo) * self.rng.random(len(idx))
        self._row[idx] = fam
        self._credit[idx] = 0.0

    def step(self, obs=None) -> np.ndarray:
        """One tick of move bins, shape ``(n_rows,)``, int32. Returns a copy.

        ``obs`` is unused: the hook for a state-aware family later."""
        expired = self._left <= 0
        n = int(expired.sum())
        if n:
            self._move[expired] = self.rng.choice(MOVE_BINS, size=n)
            if self.mix is None:
                self._left[expired] = self.rng.integers(self.hold_min, self.hold_max + 1, size=n)
            else:
                r = self._row[expired]
                self._left[expired] = self.rng.integers(r[:, 0].astype(np.int64),
                                                        r[:, 1].astype(np.int64) + 1)
                if (r[:, 2] > 0).any():
                    self._move[expired] *= ~(self.rng.random(n) < r[:, 2])
        self._left -= 1
        move = self._move.copy()
        if self.mix is not None:
            self._credit += self._row[:, 4]
            press = self._credit >= 1.0
            self._credit[press] -= 1.0
            move[~press] = 0
        return move


class WalkerActor:
    """``RandomWalker`` over the given env rows, in the eval.baselines actor shape.

    ``act`` returns ``(act, cont)`` sized from ``obs``: int32 ``(len(obs), ACTION_DIM)``
    with only the move column of ``rows`` set, and float32 zeros ``(len(obs), AIM_DIM)``
    (no turn, level pitch). Every other row is all zero,
    so the hero's rows are the caller's to overwrite, as with IdleActor.
    """

    name = "walker"

    def __init__(self,
                 rng: np.random.Generator,
                 rows,
                 hold_min: int = HOLD_MIN,
                 hold_max: int = HOLD_MAX):
        self.rows = np.asarray(list(rows), dtype=np.intp)
        self.core = RandomWalker(len(self.rows), rng, hold_min, hold_max)

    def reset(self):
        self.core.reset()

    def act(self, obs, st, vis_prev, env):
        n = len(obs)
        act = np.zeros((n, ACTION_DIM), dtype=np.int32)
        act[self.rows, H_MOVE] = self.core.step()
        return act, np.zeros((n, AIM_DIM), dtype=np.float32)
