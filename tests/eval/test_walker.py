"""cs2rl.eval.walker: the random-walker opponent (#152 L1).

WHAT IS PINNED
* Determinism: one seed and one row count give one stream; another seed gives another.
* The hold. With a fixed hold k, a row's bin changes only on steps that are multiples
  of k, and some row changes on every one of them. With the default 4..16, every hold
  drawn is in 4..16 and both ends are drawn (an exclusive upper bound would never draw
  16), and no complete run of an unchanged bin is shorter than 4 ticks.
* Bins: only 1..8 are drawn, and all eight are.
* Uniformity: over about 3200 redraws, each hold 4..16 and each bin 1..8 occurs within
  +-30% of its mean count.
* reset() makes the next step redraw every row.
* WalkerActor sets only the move column of its own rows; every other head, every other
  row and the aim are zero, and the shapes follow obs.
* In the sim, on the arena duel, the walker row moves on most ticks, never turns and
  fires no shot.
* Importing the module in a fresh interpreter pulls none of the HEAVY modules of
  tests/train/test_w1_modules.py (the trainer's light modules may import it).
"""
import math
from typing import cast

import numpy as np
import pytest

from cs2rl.eval.walker import H_MOVE, HOLD_MAX, HOLD_MIN, MOVE_BINS, RandomWalker, WalkerActor
from cs2rl.spec.action import ACTION_DIM, AIM_DIM


def _stream(seed, n_rows=16, steps=300, **kw):
    w = RandomWalker(n_rows, np.random.default_rng(seed), **kw)
    return np.stack([w.step() for _ in range(steps)])


def test_one_seed_gives_one_stream_and_another_seed_another():
    a, b = _stream(7), _stream(7)
    assert a.dtype == np.int32 and a.shape == (300, 16)
    np.testing.assert_array_equal(a, b)
    assert (a != _stream(8)).any()


def test_a_fixed_hold_changes_bins_only_on_its_multiples():
    k = 5
    m = _stream(0, n_rows=64, steps=60, hold_min=k, hold_max=k)
    changed_at = {t for t in range(1, len(m)) if (m[t] != m[t - 1]).any()}
    assert changed_at == set(range(k, len(m), k))


class _RecordingRng:
    """A Generator stand-in that records every hold drawn (the walker's integers() calls)."""

    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)
        self.holds = []

    def choice(self, *args, **kwargs):
        return self._rng.choice(*args, **kwargs)

    def integers(self, *args, **kwargs):
        out = self._rng.integers(*args, **kwargs)
        self.holds.extend(np.atleast_1d(out).tolist())
        return out


def test_default_holds_span_4_to_16_inclusive_and_no_run_is_shorter():
    assert (HOLD_MIN, HOLD_MAX) == (4, 16)
    rng = _RecordingRng(3)
    w = RandomWalker(16, cast(np.random.Generator, rng))
    m = np.stack([w.step() for _ in range(2000)])
    assert set(rng.holds) == set(range(HOLD_MIN, HOLD_MAX + 1))
    for r in range(m.shape[1]):
        change = np.flatnonzero(np.diff(m[:, r]) != 0)
        assert len(change) > 50
        assert np.diff(change).min() >= HOLD_MIN


def test_holds_and_bins_are_drawn_uniformly():
    """Each realised hold and bin occurs within +-30% of its mean count.

    Read from the walker's own state right after each redraw (``_left + 1`` is the
    hold just drawn, ``_move`` the bin), not from the Generator's calls, so a change
    in how the draws are combined (e.g. the max of two holds) shows too. About 3200
    redraws (16 rows x 2000 steps / mean hold 10): +-30% is about 5 sd for a hold
    count and 6 sd for a bin count, and the stream is seeded.
    """
    w = RandomWalker(16, np.random.default_rng(3))
    holds, bins = [], []
    for _ in range(2000):
        expiring = w._left <= 0
        w.step()
        holds.extend((w._left[expiring] + 1).tolist())
        bins.extend(w._move[expiring].tolist())
    assert len(holds) > 3000, len(holds)
    for counts in (np.bincount(holds, minlength=HOLD_MAX + 1)[HOLD_MIN:HOLD_MAX + 1],
                   np.bincount(bins, minlength=9)[1:9]):
        assert np.abs(counts / counts.mean() - 1.0).max() <= 0.30, counts


def test_only_bins_1_to_8_are_drawn_and_all_of_them_are():
    assert MOVE_BINS.tolist() == list(range(1, 9))
    assert set(np.unique(_stream(1)).tolist()) == set(range(1, 9))


def test_reset_makes_the_next_step_redraw_every_row():
    w1 = RandomWalker(32, np.random.default_rng(5), hold_min=50, hold_max=50)
    w2 = RandomWalker(32, np.random.default_rng(5), hold_min=50, hold_max=50)
    first = w1.step()
    np.testing.assert_array_equal(w2.step(), first)
    w1.reset()
    np.testing.assert_array_equal(w2.step(), first)    # mid-hold: same bins
    assert (w1.step() != first).any()                  # after reset: fresh draws


@pytest.mark.parametrize("kw", [dict(n_rows=-1), dict(hold_min=0), dict(hold_min=5, hold_max=4)])
def test_bad_arguments_raise(kw):
    args = dict(n_rows=1, hold_min=HOLD_MIN, hold_max=HOLD_MAX) | kw
    with pytest.raises(ValueError):
        RandomWalker(args.pop("n_rows"), np.random.default_rng(0), **args)


def test_the_actor_sets_only_the_move_column_of_its_rows():
    rows = [5, 6, 7, 8, 9]
    actor = WalkerActor(np.random.default_rng(2), rows)
    actor.reset()
    obs = np.zeros((10, 3), dtype=np.float32)
    for _ in range(50):
        act, cont = actor.act(obs, None, None, None)
        assert act.shape == (10, ACTION_DIM) and act.dtype == np.int32
        assert cont.shape == (10, AIM_DIM) and cont.dtype == np.float32
        assert not cont.any()
        assert not act[:5].any()
        others = np.delete(act[rows], H_MOVE, axis=1)
        assert not others.any()
        assert np.isin(act[rows, H_MOVE], MOVE_BINS).all()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_in_the_arena_the_walker_row_moves_never_turns_and_never_fires(seed):
    from cs2rl.experiment.oracle_statue import HERO, STATUE, build_env
    env = build_env(seed)
    try:
        obs, _ = env.reset()
        walker = WalkerActor(np.random.default_rng(seed), [STATUE])
        walker.reset()
        ag = env._c_env.game.agents
        facing0 = ag[STATUE].facing
        moving = 0
        for _ in range(40):
            act, cont = walker.act(obs, None, None, env)
            obs, *_ = env.step(act, cont)
            moving += math.hypot(ag[STATUE].vx, ag[STATUE].vy) > 1.0
        # The floor is half the ticks, not all: the walker is blind, so a hold can
        # press it into geometry and stall it.
        assert moving >= 20, moving
        assert ag[STATUE].facing == facing0
        assert env._c_env.episode_stats.shots_fired == 0
        assert ag[HERO].alive and ag[STATUE].alive
    finally:
        env.close()


def test_the_module_imports_without_anything_heavy():
    from tests.train.test_w1_modules import HEAVY, _run_child
    r = _run_child(f"""
import cs2rl.eval.walker
heavy = [m for m in {HEAVY!r} if m in sys.modules]
assert not heavy, f"cs2rl.eval.walker's module scope pulled {{heavy}}"
""")
    assert r.returncode == 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
