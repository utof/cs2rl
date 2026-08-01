"""Zero-sum reward symmetrization (spec 2026-08-01 §4.3).

Pinned semantics, per tick, for agent i on team A against team B:

    r_i' = 0.5 * ( r_i - mean_{j in B}(r_j) )

mean (not sum) of the opposing five keeps per-agent scale comparable; the
0.5x factor stops r_A - r_B from ~doubling reward scale, which would interact
with the return-norm patch and the BC-warm-started critic. Applied on EVERY
tick including terminal ones (win bonuses land on the terminal tick under
auto_reset=True — pinned below).
"""
import numpy as np
import pytest

from _action_spec import (             # definition site — nav/train only re-export, and importing train drags torch into a numpy-only test
    ACTION_DIM, ACTION_HEAD_SIZES,
)
from c_env.cs2_env import make_env, symmetrize_rewards
from nav import TEAM_SIZE


def _expected(raw):
    """Reference implementation of the §4.3 formula (deliberately naive)."""
    out = np.empty_like(raw)
    mean_t = raw[:TEAM_SIZE].mean()
    mean_ct = raw[TEAM_SIZE:].mean()
    out[:TEAM_SIZE] = 0.5 * (raw[:TEAM_SIZE] - mean_ct)
    out[TEAM_SIZE:] = 0.5 * (raw[TEAM_SIZE:] - mean_t)
    return out


def test_pure_transform_uniform_teams():
    r = np.array([1.0] * TEAM_SIZE + [0.0] * TEAM_SIZE, dtype=np.float32)
    symmetrize_rewards(r)
    assert r[:TEAM_SIZE] == pytest.approx(0.5)
    assert r[TEAM_SIZE:] == pytest.approx(-0.5)
    assert r.sum() == pytest.approx(0.0)


def test_pure_transform_matches_reference_on_asymmetric_vector():
    raw = np.array([2.0, -1.0, 0.5, 0.0, 0.25, -0.75, 1.5, 0.0, 0.125, -2.0], dtype=np.float32)
    r = raw.copy()
    symmetrize_rewards(r)
    assert r == pytest.approx(_expected(raw), abs=1e-6)


def test_both_team_means_are_read_before_either_is_written():
    """Regression guard for the ordering bug: computing mean_t AFTER writing
    the T slice makes the CT half depend on transformed values."""
    raw = np.array([2.0, 0.0, 0.0, 0.0, 0.0] + [1.0] * TEAM_SIZE, dtype=np.float32)
    r = raw.copy()
    symmetrize_rewards(r)
    # mean_ct = 1.0, mean_t = 0.4  ->  T = [0.5, -0.5, -0.5, -0.5, -0.5], CT = 0.3 each
    assert r[0] == pytest.approx(0.5)
    assert r[1] == pytest.approx(-0.5)
    assert r[TEAM_SIZE:] == pytest.approx(0.3)


def test_transform_is_in_place_and_preserves_dtype():
    r = np.zeros(2 * TEAM_SIZE, dtype=np.float32)
    r[0] = 1.0
    assert symmetrize_rewards(r) is r, (
        "must mutate the caller's buffer: under the vecenv path this array IS "
        "the PufferLib shared reward buffer the trainer reads")
    assert r.dtype == np.float32


def test_env_default_is_off():
    env = make_env()
    try:
        assert env._reward_symmetrize is False
    finally:
        env.close()


def _twin_episode_check(n_steps=900, seed=1234, team_spirit=0.0):
    """Drive a symmetrized and an unsymmetrized env with identical seeds and
    identical actions, and assert the per-tick formula holds. Rewards do not
    feed back into the sim, so the two trajectories stay in lockstep — the
    terminals assertion pins that.

    team_spirit is parametrized (review finding 2): training runs at 0.3, and
    the C mixing loop blends only ALIVE agents while symmetrize_rewards divides
    by a fixed TEAM_SIZE — the transform must hold on post-mixing values in
    both regimes, especially terminal ticks with partially-eliminated teams."""
    plain = make_env(seed=seed, team_spirit=team_spirit)
    sym = make_env(seed=seed, reward_symmetrize=True, team_spirit=team_spirit)
    rng = np.random.default_rng(7)
    saw_terminal_with_win_bonus = False
    try:
        plain.reset(seed=seed)
        sym.reset(seed=seed)
        for step_n in range(n_steps):
            actions = rng.integers(ACTION_HEAD_SIZES, size=(10, ACTION_DIM)).astype(np.int32)
            _, r_plain, term_plain, _, info_plain = plain.step(actions)
            _, r_sym, term_sym, _, info_sym = sym.step(actions.copy())
            raw = np.array(r_plain, dtype=np.float32)
            got = np.array(r_sym, dtype=np.float32)
            assert np.array_equal(
                term_plain, term_sym), (f"twins diverged at step {step_n} — seeds are not lockstep")
            assert got == pytest.approx(_expected(raw), abs=1e-5), f"step {step_n}"
            if info_plain and np.abs(raw).max() >= 1.0:
                # terminal tick: win/loss magnitudes (>=3.0 pre-mixing) dwarf
                # the per-tick shaping terms, so this is the §6.4 terminal case
                saw_terminal_with_win_bonus = True
                assert got.sum() == pytest.approx(0.0, abs=1e-4)
    finally:
        plain.close()
        sym.close()
    return saw_terminal_with_win_bonus


@pytest.mark.parametrize("team_spirit", [0.0, 0.3])
def test_symmetrization_holds_every_tick_including_terminals(team_spirit):
    # 0.0 = make_env default; 0.3 = the value training actually uses
    # (src/train.py team_spirit config) — review finding 2.
    assert _twin_episode_check(team_spirit=team_spirit), (
        "no terminal tick with a win bonus was observed in 900 steps — the "
        "terminal case (spec §6.4) went unexercised; raise n_steps")


def test_c_side_reward_channels_are_pre_transform():
    """Spec §4.3 caveat, pinned so analysis cannot be misled: the C
    environment/reward_* stat channels accumulate BEFORE the Python transform,
    so they are identical with and without --reward-symmetrize."""
    seed = 99
    plain = make_env(seed=seed)
    sym = make_env(seed=seed, reward_symmetrize=True)
    rng = np.random.default_rng(3)
    try:
        plain.reset(seed=seed)
        sym.reset(seed=seed)
        for _ in range(900):
            actions = rng.integers(ACTION_HEAD_SIZES, size=(10, ACTION_DIM)).astype(np.int32)
            _, _, _, _, ip = plain.step(actions)
            _, _, _, _, is_ = sym.step(actions.copy())
            if ip and is_:
                for key in ("reward_kills", "reward_pbrs", "reward_survival", "reward_win"):
                    assert ip[0][key] == pytest.approx(is_[0][key], abs=1e-4), key
                return
    finally:
        plain.close()
        sym.close()
    pytest.fail("no round ended within 900 steps")


def test_make_puffer_env_threads_the_flag():
    import train
    env = train.make_puffer_env(seed=5, reward_symmetrize=True)
    try:
        assert env._reward_symmetrize is True
    finally:
        env.close()


def test_env_factory_threads_the_flag():
    import multiprocessing as mp

    import train
    factory = train.build_env_factory(shared_ts=mp.Value("f", 0.3),
                                      map_data=None,
                                      reward_symmetrize=True)
    env = factory(seed=0)
    try:
        assert env._reward_symmetrize is True
    finally:
        env.close()
