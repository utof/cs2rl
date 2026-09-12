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
from env_config import EnvConfig
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
    sym = make_env(seed=seed, team_spirit=team_spirit, config=EnvConfig(reward_symmetrize=True))
    rng = np.random.default_rng(7)
    saw_terminal_with_win_bonus = False
    try:
        plain.reset(seed=seed)
        sym.reset(seed=seed)
        for step_n in range(n_steps):
            actions = rng.integers(ACTION_HEAD_SIZES, size=(10, ACTION_DIM)).astype(np.int32)
            obs_plain, r_plain, term_plain, _, info_plain = plain.step(actions)
            obs_sym, r_sym, term_sym, _, info_sym = sym.step(actions.copy())
            raw = np.array(r_plain, dtype=np.float32)
            got = np.array(r_sym, dtype=np.float32)
            # Lockstep guard covers observations too, not just terminals: the
            # transform must not perturb the sim, and obs is the widest
            # per-tick surface that would show it if it did.
            assert np.array_equal(
                term_plain, term_sym), (f"twins diverged at step {step_n} — seeds are not lockstep")
            assert np.array_equal(
                obs_plain,
                obs_sym), (f"observations diverged at step {step_n} — symmetrization must "
                           "not feed back into the sim")
            assert got == pytest.approx(_expected(raw), abs=1e-5), f"step {step_n}"
            # Zero-sum is the whole point, so assert it on EVERY tick, not only
            # the terminal ones.
            assert got.sum() == pytest.approx(0.0, abs=1e-4), f"step {step_n} not zero-sum"
            if info_plain and np.abs(raw).max() >= 1.0:
                # terminal tick: win/loss magnitudes (>=3.0 pre-mixing) dwarf
                # the per-tick shaping terms, so this is the §6.4 terminal case
                saw_terminal_with_win_bonus = True
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
    sym = make_env(seed=seed, config=EnvConfig(reward_symmetrize=True))
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


def test_symmetrization_holds_on_the_external_buffer_vecenv_path():
    """The path training actually runs on (review finding 1).

    Every other test here builds a bare env with buf=None, where self.rewards
    IS the zero-copy C view; under pufferlib.vector.make the env gets EXTERNAL
    buffers and _sync_outputs np.copyto's the C rewards into a separate array.
    Symmetrizing the wrong one of those two arrays would leave the whole suite
    green while training silently learned on raw rewards, so this drives the
    real factory -> vector.make stack and reads the rewards the trainer reads.
    Serial (not Multiprocessing): same external-buffer code path, no worker
    processes to make the assertion failures unreadable.
    """
    import multiprocessing as mp

    import pufferlib.vector

    import train

    def _make(symmetrize):
        factory = train.build_env_factory(shared_ts=mp.Value("f", 0.3),
                                          map_data=None,
                                          config=EnvConfig(reward_symmetrize=symmetrize))
        return pufferlib.vector.make([factory],
                                     env_args=[[]],
                                     env_kwargs=[{}],
                                     num_envs=1,
                                     backend=pufferlib.vector.Serial,
                                     batch_size=1,
                                     zero_copy=True)

    plain = _make(False)
    sym = _make(True)
    rng = np.random.default_rng(21)
    try:
        # Pin the premise: if this ever goes False the test has quietly
        # regressed into re-testing the buf=None path the others cover.
        assert plain.driver_env._uses_external_buffers, (
            "vecenv did not hand the env external buffers — this test no "
            "longer covers the training path it exists for")
        plain.async_reset(seed=4242)
        sym.async_reset(seed=4242)
        plain.recv()
        sym.recv()
        for step_n in range(300):
            actions = rng.integers(ACTION_HEAD_SIZES, size=(10, ACTION_DIM)).astype(np.int32)
            plain.send(actions)
            sym.send(actions.copy())
            _o_p, r_plain, term_plain, _t_p, _i_p, _id_p, _m_p = plain.recv()
            _o_s, r_sym, term_sym, _t_s, _i_s, _id_s, _m_s = sym.recv()
            raw = np.asarray(r_plain, dtype=np.float32)
            got = np.asarray(r_sym, dtype=np.float32)
            assert np.array_equal(
                term_plain, term_sym), (f"twins diverged at step {step_n} — seeds are not lockstep")
            assert got == pytest.approx(_expected(raw), abs=1e-5), f"step {step_n}"
            assert got.sum() == pytest.approx(0.0, abs=1e-4), f"step {step_n} not zero-sum"
    finally:
        plain.close()
        sym.close()


def test_make_env_threads_the_flag():
    from c_env.cs2_env import make_env
    from env_config import EnvConfig
    env = make_env(seed=5, config=EnvConfig(reward_symmetrize=True))
    try:
        assert env._reward_symmetrize is True
    finally:
        env.close()


def test_env_factory_threads_the_flag():
    import multiprocessing as mp

    import train
    factory = train.build_env_factory(shared_ts=mp.Value("f", 0.3),
                                      map_data=None,
                                      config=EnvConfig(reward_symmetrize=True))
    env = factory(seed=0)
    try:
        assert env._reward_symmetrize is True
    finally:
        env.close()
