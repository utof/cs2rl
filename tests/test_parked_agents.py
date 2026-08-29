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

from _obs_spec import OBS_BLOCKS
from c_env.cs2_env import make_env

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
    env = make_env(map_data=simple_map, n_active_per_team=1, seed=3)
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
    env = make_env(map_data=simple_map, n_active_per_team=1, seed=3)
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
                   n_active_per_team=2,
                   seed=5,
                   pbrs_alive_weight=0.3,
                   pbrs_hp_weight=0.002,
                   reward_inaction=0.0005)
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
