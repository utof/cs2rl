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

from cs2rl.eval.walker import (
    H_MOVE,
    HELD_OUT,
    HOLD_MAX,
    HOLD_MIN,
    MOVE_BINS,
    TRAIN_MIX,
    TRAIN_WEIGHTS,
    RandomWalker,
    WalkerActor,
    WalkerParams,
)
from cs2rl.eval.walker import (
    STATUE as STATUE_FAMILY, )
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


def _moves(w, steps):
    """The (n_rows, steps) move bins of ``steps`` consecutive ticks."""
    return np.array([w.step() for _ in range(steps)]).T


def test_without_a_mix_the_stream_is_the_one_from_before_the_mix_existed():
    """Literals drawn from the pre-mix RandomWalker (seed 7, 3 rows, 24 ticks): the
    default path must make exactly the same draws, so the L1 numbers do not move."""
    default = [[8] * 15 + [3] * 9, [6] * 11 + [7] * 6 + [8] * 4 + [4] * 3,
               [6] * 14 + [1] * 7 + [7] * 3]
    assert _moves(RandomWalker(3, np.random.default_rng(7)), 24).tolist() == default
    short = [[8, 8, 8, 8, 8, 1, 1, 1, 4, 4, 1, 1, 1, 1, 1, 8, 8, 8, 5, 5, 5, 5, 5, 3],
             [6, 6, 6, 6, 7, 7, 8, 8, 7, 7, 7, 7, 7, 3, 3, 3, 3, 5, 5, 5, 5, 6, 6, 6],
             [6, 6, 6, 6, 6, 3, 3, 3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4, 4, 7, 7, 7, 7]]
    assert _moves(RandomWalker(3, np.random.default_rng(7), 2, 5), 24).tolist() == short


def test_params_refuse_nonsense():
    for bad in (lambda: WalkerParams(hold=(0, 4)), lambda: WalkerParams(hold=(5, 4)),
                lambda: WalkerParams(p_stop=(0.5, 0.2)), lambda: WalkerParams(p_stop=(0.0, 1.5)),
                lambda: WalkerParams(duty=(0.0, 1.0)), lambda: WalkerParams(duty=(0.5, 1.2))):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError):
        RandomWalker(2, np.random.default_rng(0), mix=TRAIN_MIX, weights=(0.5, 0.5))
    with pytest.raises(ValueError):    # numpy's Generator.choice(p=...) refuses these
        RandomWalker(2, np.random.default_rng(0), mix=TRAIN_MIX, weights=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        RandomWalker(2, np.random.default_rng(0), mix=TRAIN_MIX, weights=(1.5, -0.5, 0.0))


def test_a_statue_family_never_moves_and_a_held_out_family_always_does():
    for fam, moves in ((STATUE_FAMILY, False), (HELD_OUT, True)):
        bins = _moves(RandomWalker(8, np.random.default_rng(1), mix=[fam]), 200)
        assert bool((bins != 0).all()) is moves and bool((bins == 0).all()) is not moves
    # HELD_OUT holds are 24..48 ticks: no run is shorter, and runs of 24+ exist.
    bins = _moves(RandomWalker(1, np.random.default_rng(2), mix=[HELD_OUT]), 600)[0]
    runs = np.diff(np.flatnonzero(np.diff(bins) != 0))
    assert runs.size and runs.min() >= 24 and runs.max() <= 48 + 48, runs


def test_a_mix_draws_each_family_with_its_weight_and_a_row_keeps_it_for_the_episode():
    n = 4000
    w = RandomWalker(n, np.random.default_rng(3), mix=TRAIN_MIX, weights=TRAIN_WEIGHTS)
    fam_hold_hi = w._row[:, 1]
    for hi, weight in zip((1, 16, 8), TRAIN_WEIGHTS, strict=True):
        assert abs((fam_hold_hi == hi).mean() - weight) < 0.03, (hi, weight)
    before = w._row.copy()
    _moves(w, 50)
    assert np.array_equal(w._row, before), "the episode's parameters change only at reset"


def test_reset_rows_redraws_only_those_rows_and_none_when_empty():
    w = RandomWalker(6, np.random.default_rng(4), mix=TRAIN_MIX, weights=TRAIN_WEIGHTS)
    _moves(w, 3)
    rows, left = w._row.copy(), w._left.copy()
    w.reset(np.zeros(6, dtype=bool))
    assert np.array_equal(w._left, left), "an empty mask must reset nothing"
    w.reset(np.array([1, 4]))
    kept = [0, 2, 3, 5]
    assert np.array_equal(w._left[kept], left[kept]) and np.array_equal(w._row[kept], rows[kept])
    assert (w._left[[1, 4]] == 0).all()
    w.reset(np.arange(6) < 2)          # a bool mask names rows too
    assert (w._left[:2] == 0).all() and np.array_equal(w._left[[2, 3, 5]], left[[2, 3, 5]])


def test_reset_rows_draws_a_new_family_for_exactly_those_rows():
    """The per-episode re-draw: a fresh draw is another family with probability
    1 - sum(w^2) = 0.64 under TRAIN_MIX; kept rows never change."""
    n = 4000
    w = RandomWalker(n, np.random.default_rng(11), mix=TRAIN_MIX, weights=TRAIN_WEIGHTS)
    before = w._row.copy()
    w.reset(np.arange(n) < n // 2)
    changed = (w._row != before).any(axis=1)
    assert not changed[n // 2:].any()
    assert abs(changed[:n // 2].mean() - (1 - sum(x * x for x in TRAIN_WEIGHTS))) < 0.05


def test_reset_rows_restarts_the_duty_credit_of_those_rows_only():
    w = RandomWalker(3, np.random.default_rng(0), mix=[WalkerParams(duty=(0.5, 0.5))])
    w.step()
    assert w._credit.tolist() == [0.5, 0.5, 0.5]
    w.reset(np.array([0]))
    assert w._credit.tolist() == [0.0, 0.5, 0.5]


def test_a_mix_is_deterministic_per_seed():

    def run(seed):
        w = RandomWalker(5, np.random.default_rng(seed), mix=TRAIN_MIX, weights=TRAIN_WEIGHTS)
        out = []
        for t in range(60):
            if t % 20 == 0:
                w.reset(np.array([t % 5]))
            out.append(w.step())
        return np.array(out)

    assert np.array_equal(run(5), run(5)) and not np.array_equal(run(5), run(6))


def test_duty_gates_the_press_without_one_rng_draw():
    """Duty 0.5 presses every other tick, and the RNG ends where the draws of a plain
    (no mix) walker, after the mix's one family draw, leave it."""
    half = RandomWalker(4, np.random.default_rng(9), mix=[WalkerParams(duty=(0.5, 0.5))])
    plain_rng = np.random.default_rng(9)
    plain_rng.choice(1, size=4, p=[1.0])               # the mix's per-reset family draw
    plain = RandomWalker(4, plain_rng)
    a, b = _moves(plain, 40), _moves(half, 40)
    assert (b == 0).sum(axis=1).tolist() == [20] * 4
    assert np.array_equal(b[:, 1::2], a[:, 1::2]) and not b[:, 0::2].any()
    assert plain.rng.bit_generator.state == half.rng.bit_generator.state


def test_p_stop_gives_stand_still_holds_at_about_that_rate():
    fam = WalkerParams(hold=(1, 1), p_stop=(0.25, 0.25))
    w = RandomWalker(2000, np.random.default_rng(10), mix=[fam])
    assert abs((_moves(w, 1) == 0).mean() - 0.25) < 0.03


def test_the_module_imports_without_anything_heavy():
    from tests.train.test_w1_modules import HEAVY, _run_child
    r = _run_child(f"""
import cs2rl.eval.walker
heavy = [m for m in {HEAVY!r} if m in sys.modules]
assert not heavy, f"cs2rl.eval.walker's module scope pulled {{heavy}}"
""")
    assert r.returncode == 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
