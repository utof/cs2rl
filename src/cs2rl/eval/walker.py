"""Random-walker opponent: a scripted target that moves but never fights back (#152).

WHAT
----
Each walker row holds one move direction for a random number of ticks, then draws a
fresh one:

  * the direction is a move-head bin drawn uniformly from 1..8 (``MOVE_BINS``). Bin 0
    (no move) is never drawn, so a live walker row is always pressing a direction;
  * the hold is drawn uniformly from ``hold_min..hold_max`` ticks, both ends included
    (default ``HOLD_MIN..HOLD_MAX`` = 4..16);
  * every other discrete head is 0 and the aim is (0, 0): it never shoots, reloads,
    switches weapon, uses, crouches, jumps or turns.

``RandomWalker`` is the vectorised core over any number of rows (one numpy
``Generator``, no per-row Python loop), for a caller that owns its own action array.
``WalkerActor`` adapts it to the ``act(obs, st, vis_prev, env)`` shape of the
cs2rl.eval.baselines actors, for the scripted-bot harnesses
(cs2rl.experiment.oracle_tracker).

WHY
---
It is the moving target of #152 layer L1 (the oracle against something that moves),
built so a trainer can also use it as a scripted opponent: vectorised over any number of
rows, and import-light. The hold is load-bearing: measured on
2026-10-09 (the #152 L1 investigation), the Aug-31 Rung 1a checkpoint killed a walker
that redraws its direction every tick in 99/100 rounds, about as often as a statue
(100/100), and a hold-4..16 walker in 81/100. A per-tick redraw jitters in place.

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
* Module scope imports numpy and cs2rl.spec.action only, so a light module (one the
  `python -m cs2rl.train --dump-config` path imports) may import this one.
  tests/eval/test_walker.py checks that in a fresh interpreter against
  tests/train/test_w1_modules.py's HEAVY list. Do not import eval.baselines here: it
  imports torch and env.c.cs2_env at module scope.
"""
from __future__ import annotations

import numpy as np

from cs2rl.spec.action import ACTION_DIM, ACTION_HEAD_NAMES, ACTION_HEAD_SIZES, AIM_DIM

H_MOVE = ACTION_HEAD_NAMES.index("move")
# Bins 1..8 are the eight directions; 0 is "no move" and is never drawn.
MOVE_BINS = np.arange(1, ACTION_HEAD_SIZES[H_MOVE], dtype=np.int32)
HOLD_MIN = 4
HOLD_MAX = 16


class RandomWalker:
    """Vectorised hold-and-redraw move bins for ``n_rows`` rows.

    ``step()`` returns one int32 move bin per row. A row keeps its bin for its drawn
    hold, then draws a new bin and a new hold on the step after it expires. A row's
    first step after construction or ``reset()`` always draws.
    """

    def __init__(self,
                 n_rows: int,
                 rng: np.random.Generator,
                 hold_min: int = HOLD_MIN,
                 hold_max: int = HOLD_MAX):
        if n_rows < 0:
            raise ValueError(f"n_rows must be >= 0, got {n_rows}")
        if not 1 <= hold_min <= hold_max:
            raise ValueError(f"need 1 <= hold_min <= hold_max, got {hold_min}..{hold_max}")
        self.rng = rng
        self.hold_min, self.hold_max = int(hold_min), int(hold_max)
        # _left[i] = steps row i still returns its current bin before redrawing.
        self._left = np.zeros(n_rows, dtype=np.int64)
        self._move = np.zeros(n_rows, dtype=np.int32)

    def reset(self) -> None:
        """Expire every row's hold, so the next ``step()`` redraws every row."""
        self._left[:] = 0

    def step(self) -> np.ndarray:
        """One tick of move bins, shape ``(n_rows,)``, int32. Returns a copy."""
        expired = self._left <= 0
        n = int(expired.sum())
        if n:
            self._move[expired] = self.rng.choice(MOVE_BINS, size=n)
            self._left[expired] = self.rng.integers(self.hold_min, self.hold_max + 1, size=n)
        self._left -= 1
        return self._move.copy()


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
