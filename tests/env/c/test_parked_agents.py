"""Parked agents (spec 2026-08-29 §2.1): n_active_per_team < TEAM_SIZE leaves
the remaining slots participating=0 / alive=0 / area_idx=INVALID and every
reward/obs consumer degrades to "dead", never "standing at the origin".

WHY these four tests and not one: the failure modes of a "reduced team size"
knob are independent of each other. A parked slot can (a) still spawn, (b) get
counted in an obs normaliser, (c) collect team-wide reward, or (d) leak into the
full-team default. Each test pins exactly one of those.

PITFALL: parked rows are *not* just alive=0. The GameState memset before spawn
leaves them team=0 and area_idx=0, and area 0 is a REAL area — so "dead at
area 0 on team T" is a perfectly consistent-looking lie. The asserts below check
area_idx/enemy_mem_idx == -1 and team, not merely `alive`.
"""
import numpy as np
import pytest

from cs2rl.env.c.cs2_env import make_env, symmetrize_rewards
from cs2rl.env.config import EnvConfig, RewardWeights
from cs2rl.spec.obs import OBS_BLOCKS

N_AGENTS, TEAM_SIZE, AIM_DIM = 10, 5, 2
HEAD_SIZES = (9, 2, 2, 3, 2, 2, 2)
GB = OBS_BLOCKS["global"][0]


def _random_actions(rng):
    """One uniformly random legal action per agent, discrete heads + aim.

    Deliberately ignores the action masks: parked slots must survive ANY action
    buffer the trainer can hand the C env, including nonsense for rows the
    policy would normally mask out. Returns (int32[N_AGENTS, 7], float32
    [N_AGENTS, 2]) in the exact dtypes env.step requires.
    """
    act = np.stack([rng.integers(0, n, size=N_AGENTS) for n in HEAD_SIZES], axis=1).astype(np.int32)
    cont = rng.uniform(-0.5, 0.5, size=(N_AGENTS, AIM_DIM)).astype(np.float32)
    return act, cont


def test_one_active_per_team_spawns_slots_0_and_5(simple_map):
    env = make_env(map_data=simple_map, config=EnvConfig(n_active_per_team=1), seed=3)
    try:
        env.reset()
        ag = env._c_env.game.agents
        for i in range(N_AGENTS):
            if i in (0, 5):
                assert ag[i].participating == 1 and ag[i].alive == 1 and ag[i].area_idx >= 0
            else:
                assert ag[i].participating == 0
                assert ag[i].alive == 0
                assert ag[i].area_idx == -1
                # INVALID_AREA_IDX, not the memset zero (area 0 is a real area)
                assert all(ag[i].enemy_mem_idx[k] == -1 for k in range(TEAM_SIZE)), i
            assert ag[i].team == (0 if i < TEAM_SIZE else 1), "parked CT slots must be team 1"
        assert env._c_env.game.bomb_carrier_id == 0
        assert ag[0].has_bomb == 1
    finally:
        env.close()


def test_alive_count_obs_normalised_by_n_active(simple_map):
    env = make_env(map_data=simple_map, config=EnvConfig(n_active_per_team=1), seed=3)
    try:
        env.reset()
        rng = np.random.default_rng(0)
        obs, *_ = env.step(*_random_actions(rng))
        assert obs[0, GB + 11] == pytest.approx(1.0)   # t_alive / n_active
        assert obs[0, GB + 12] == pytest.approx(1.0)   # ct_alive / n_active
    finally:
        env.close()


def test_parked_agents_get_zero_reward_every_tick(simple_map):
    env = make_env(map_data=simple_map,
                   seed=5,
                   config=EnvConfig(
                       n_active_per_team=2,
                       rewards=RewardWeights(
                           pbrs_alive_weight=0.3,
                           pbrs_hp_weight=0.002,
                           reward_inaction=0.0005,
                       ),
                   ))
    try:
        env.reset()
        rng = np.random.default_rng(1)
        parked = [i for i in range(N_AGENTS) if i % TEAM_SIZE >= 2]
        for _ in range(300):
            _, rew, term, _, info = env.step(*_random_actions(rng))
            assert np.all(rew[parked] == 0.0), rew
    finally:
        env.close()


def test_full_team_is_default_and_all_participate(simple_map):
    env = make_env(map_data=simple_map, seed=3)
    try:
        env.reset()
        ag = env._c_env.game.agents
        assert all(ag[i].participating == 1 and ag[i].alive == 1 for i in range(N_AGENTS))
    finally:
        env.close()


# ── symmetrize_rewards × parked slots (review fix, 2026-08-29) ────────────────
#
# `symmetrize_rewards` is applied AFTER the C step in Cs2Env.step, so the C-side
# `participating` guards do not protect it: it is pure Python arithmetic over the
# full 10-row vector. Its first Rung 0 version divided both team means by the
# constant TEAM_SIZE and wrote all 10 rows, which broke two invariants at once —
# parked rows left at -0.5*mean_opponent instead of 0.0, and ACTIVE rows had the
# opponent-mean subtraction attenuated by n/TEAM_SIZE (5x too weak at n=1). The
# tests below pin both halves; see PITFALL 6 on the function for the proof that
# restricting the means AND the writes to the active slots keeps zero-sum exact.


def _pre_rung0_symmetrize(rewards):
    """The exact pre-Rung-0 body of symmetrize_rewards, kept verbatim.

    Exists only as the identity oracle for the n_active_per_team == TEAM_SIZE
    path: Rung 0 must not perturb a single float of the 5v5 sim, and the
    sim_fingerprint script cannot prove that because it never enables
    reward_symmetrize.
    """
    mean_t = rewards[:TEAM_SIZE].mean()
    mean_ct = rewards[TEAM_SIZE:].mean()
    rewards[:TEAM_SIZE] = 0.5 * (rewards[:TEAM_SIZE] - mean_ct)
    rewards[TEAM_SIZE:] = 0.5 * (rewards[TEAM_SIZE:] - mean_t)
    return rewards


def _active_slots(n):
    """Row indices of the n active slots per team, T first then CT."""
    return list(range(n)) + list(range(TEAM_SIZE, TEAM_SIZE + n))


def test_symmetrize_default_matches_pre_rung0_bitwise():
    """n_active_per_team=TEAM_SIZE must be BIT-identical, not merely close.

    Threading a parameter through arithmetic is exactly the kind of change that
    silently reassociates a float sum (e.g. by masking instead of slicing).
    np.array_equal, not approx, is the point of this test.
    """
    rng = np.random.default_rng(20260829)
    for _ in range(200):
        raw = (rng.standard_normal(N_AGENTS) * rng.choice([1e-3, 1.0, 5.0])).astype(np.float32)
        assert np.array_equal(symmetrize_rewards(raw.copy()), _pre_rung0_symmetrize(raw.copy()))
        # Explicitly passing the default must be identical too.
        assert np.array_equal(symmetrize_rewards(raw.copy(), TEAM_SIZE),
                              _pre_rung0_symmetrize(raw.copy()))


@pytest.mark.parametrize("n", [1, 2])
def test_symmetrize_leaves_parked_rows_at_exactly_zero(n):
    """Unit-level: fabricated vector, parked rows must come out bitwise 0.0.

    `== 0.0` (not approx): with the old TEAM_SIZE divisor these rows would read
    -0.5*mean_opponent, e.g. -0.1 at n=2 below — an approx check with a loose
    tolerance would have passed the bug.
    """
    raw = np.zeros(N_AGENTS, dtype=np.float32)
    raw[:n] = [2.0, -1.0][:n]
    raw[TEAM_SIZE:TEAM_SIZE + n] = [0.5, 1.5][:n]
    out = symmetrize_rewards(raw.copy(), n)
    parked = [i for i in range(N_AGENTS) if i % TEAM_SIZE >= n]
    assert np.all(out[parked] == 0.0), out


def test_symmetrize_active_rows_match_the_hand_computed_formula():
    """n=2, known raw vector, arithmetic done by hand off the §4.3 definition.

    raw T = [2.0, -1.0] -> mean_t = 0.5;  raw CT = [0.5, 1.5] -> mean_ct = 1.0
        T:  0.5*(2.0 - 1.0) =  0.5      0.5*(-1.0 - 1.0) = -1.0
        CT: 0.5*(0.5 - 0.5) =  0.0      0.5*( 1.5 - 0.5) =  0.5
        sum = 0.5 - 1.0 + 0.0 + 0.5 = 0.0
    Under the buggy TEAM_SIZE divisor (mean_ct = 2.0/5 = 0.4, mean_t = 0.2) the
    T rows would instead be 0.8 / -0.7 — this test is what separates the two.
    """
    raw = np.zeros(N_AGENTS, dtype=np.float32)
    raw[0], raw[1] = 2.0, -1.0
    raw[TEAM_SIZE], raw[TEAM_SIZE + 1] = 0.5, 1.5
    out = symmetrize_rewards(raw.copy(), 2)
    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(-1.0)
    assert out[TEAM_SIZE] == pytest.approx(0.0)
    assert out[TEAM_SIZE + 1] == pytest.approx(0.5)
    assert out.sum() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("n", [1, 2])
def test_symmetrize_is_exactly_zero_sum_at_reduced_team_size(n):
    """Zero-sum is the whole reason this transform exists; it must survive the
    active-slot restriction on arbitrary vectors, not just the tidy one above.

    NOTE this test does NOT discriminate against the bug the block above fixes:
    the old TEAM_SIZE-divisor form was zero-sum too (over all 10 rows). Zero-sum
    is necessary, not sufficient — it is the property that must be PRESERVED,
    and the parked/hand-computed tests are what catch the regression.
    """
    rng = np.random.default_rng(7 + n)
    for _ in range(200):
        raw = np.zeros(N_AGENTS, dtype=np.float32)
        act = _active_slots(n)
        raw[act] = (rng.standard_normal(len(act)) * 3.0).astype(np.float32)
        out = symmetrize_rewards(raw.copy(), n)
        assert out.sum() == pytest.approx(0.0, abs=1e-5), (raw, out)


@pytest.mark.parametrize("n", [1, 2])
def test_symmetrized_env_parked_rows_are_zero_and_ticks_are_zero_sum(simple_map, n):
    """End-to-end on the path the trainer runs: reward_symmetrize=True.

    The pre-existing test_parked_agents_get_zero_reward_every_tick runs with the
    default reward_symmetrize=False and therefore never touched this code.
    """
    env = make_env(map_data=simple_map,
                   seed=5,
                   config=EnvConfig(
                       n_active_per_team=n,
                       reward_symmetrize=True,
                       rewards=RewardWeights(
                           pbrs_alive_weight=0.3,
                           pbrs_hp_weight=0.002,
                           reward_ct_survival=0.001,
                           reward_inaction=0.0005,
                       ),
                   ))
    try:
        env.reset()
        rng = np.random.default_rng(1)
        parked = [i for i in range(N_AGENTS) if i % TEAM_SIZE >= n]
        for step_n in range(300):
            _, rew, _, _, _ = env.step(*_random_actions(rng))
            assert np.all(rew[parked] == 0.0), (step_n, rew)
            assert rew.sum() == pytest.approx(0.0, abs=1e-4), (step_n, rew)
    finally:
        env.close()


def test_round_rollover_reparks_the_same_slots(simple_map):
    """Reviewer minor #4: env_reset parks slots, but auto_reset calls it again
    mid-rollout. If the parking loop ever moved into __init__ or ran only on the
    first reset, everything above would still pass and the second round would
    quietly play 5v5. Drive past a real terminal and re-check the invariants on
    the FRESH state.
    """
    n = 2
    env = make_env(map_data=simple_map,
                   seed=5,
                   config=EnvConfig(n_active_per_team=n, reward_symmetrize=True))
    try:
        env.reset()
        rng = np.random.default_rng(11)
        parked = [i for i in range(N_AGENTS) if i % TEAM_SIZE >= n]
        # A timeout terminal is guaranteed within ROUND_TIME ticks even if the
        # random actions never produce an elimination.
        for step_n in range(env.round_time + 5):
            _, rew, term, _, _ = env.step(*_random_actions(rng))
            assert np.all(rew[parked] == 0.0), (step_n, rew)
            if np.any(term):
                break
        else:
            pytest.fail(f"no terminal within {env.round_time + 5} steps — test is not "
                        "exercising the rollover it exists for")
        # auto_reset=True: the observation returned above is already the fresh
        # round's, so game.agents is the re-spawned state.
        ag = env._c_env.game.agents
        for i in parked:
            assert ag[i].participating == 0, i
            assert ag[i].alive == 0, i
            assert ag[i].area_idx == -1, i
            assert ag[i].team == (0 if i < TEAM_SIZE else 1), i
        for i in _active_slots(n):
            assert ag[i].participating == 1 and ag[i].alive == 1, i
        # And the invariant still holds for a further stretch of the new round.
        for step_n in range(50):
            _, rew, _, _, _ = env.step(*_random_actions(rng))
            assert np.all(rew[parked] == 0.0), (step_n, rew)
    finally:
        env.close()


def test_oracle_episode_kills_and_credits_rewards():
    """Spec §8: vendored-oracle 1v1 through env.step() ends in a kill; reward_kill on the
    shooter's row; elimination win on the terminal tick; parked rows zero throughout."""
    from cs2rl.env.map import make_arena_duel_map
    from cs2rl.eval.baselines import BaselineEvaluator, OracleActor, StateReader, vis_from_obs
    env = make_env(map_data=make_arena_duel_map(),
                   auto_reset=False,
                   seed=2,
                   config=EnvConfig(
                       n_active_per_team=1,
                       pin_pitch=1,
                       crouch_enabled=0,
                       round_time=320,
                       rewards=RewardWeights(reward_kill=0.3),
                   ))
    try:
        ev = BaselineEvaluator(env, episodes=2,
                               seed=0)                 # even count is required; only constants are read here
        oracle = OracleActor(np.random.default_rng(0), ev.max_turn_speed, ev.laser_range, ev.nav)
        obs, _ = env.reset()
        oracle.reset()
        reader = StateReader(env)
        st = reader.read().snapshot()
        vis_prev = None
        parked = [i for i in range(10) if i not in (0, 5)]
        for _ in range(env.round_time):
            act, cont = oracle.act(obs, st, vis_prev, env)
            obs, rew, term, trunc, info = env.step(act, cont)
            assert (rew[parked] == 0).all()
            st = reader.read().snapshot()
            vis_prev, _ = vis_from_obs(obs, st,
                                       ev.map_diag)    # oracle's memory/peek logic needs last-tick vis
            if term.any() or trunc.any():
                break
        assert term.any(), "no elimination within the round"
        es = env._c_env.episode_stats
        assert es.shots_hit >= 1 and (es.kills_t + es.kills_ct) == 1
                                                       # Credit lands on the shooter's row: episode_stats.reward_kills is the
                                                       # accumulated reward_kill term (cs2_rewards.h); rew[winner] on the
                                                       # terminal tick also carries the win share + PBRS, so assert the stat.
        assert es.reward_kills == pytest.approx(0.3, abs=1e-6)
        winner = 0 if es.kills_t else 5
        assert rew[winner] > 0.0
    finally:
        env.close()
