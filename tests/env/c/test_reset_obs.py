"""#157: the obs env.reset() returns describes the fresh spawn state.

WHAT IS PINNED
* On the Rung 1a arena (n_active=1, permanent 2D LoS) the hero's reset obs
  decodes to the geometry of the live C state: self position and facing, the
  enemy's visibility, facing-relative bearing, rotated rel-pos and distance,
  and the alive counts. Every expected value is computed here from the C
  AgentState, never read back from the obs.
* On a full 5v5 simple_map, the reset obs equals the obs of one zero-action
  step in every slot but the round clock. Nothing moves, fires or is heard on
  that step, so compute_observations must write the same vector both times;
  any slot env_reset gets wrong (or leaves zero) differs.
* On the arena, env.reset() leaves both agents' enemy memory at init_agent's
  values (no area, stale tick), although each sees the other. That memory is
  what update_enemy_memory would write; the C comment forbids calling it at
  reset because env_step reads that memory.

WHY: env_reset used to zero the obs buffer and never call
compute_observations, so every actor's first action of a round was blind
(gh #157). The fix must change ONLY that obs. The per-tick parent-vs-HEAD stream
compare that showed it lives in the #157 prototype report. The memory pin holds
in a test the state write that compare's knock-out caught (update_enemy_memory
added at reset).
"""
import math

import numpy as np
import pytest

from cs2rl.env import nav
from cs2rl.env.c.cs2_env import make_env
from cs2rl.env.config import EnvConfig
from cs2rl.spec.obs import OBS_BLOCKS, OBS_ENEMY_STRIDE
from tests._helpers.scenario import zero_actions

ENEMY_BASE = OBS_BLOCKS["enemy"][0]
GLOBAL_BASE = OBS_BLOCKS["global"][0]
HERO, ENEMY = 0, 5                     # n_active=1 spawns rows 0 (T) and TEAM_SIZE (CT)


def _wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _arena_env(seed):
    """The Rung 1a arena duel (n_active=1, permanent 2D LoS), no auto-reset."""
    from cs2rl.env.map import make_arena_duel_map
    return make_env(config=EnvConfig(n_active_per_team=1,
                                     pin_pitch=1,
                                     crouch_enabled=0,
                                     round_time=160),
                    map_data=make_arena_duel_map(),
                    auto_reset=False,
                    seed=seed)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_arena_reset_obs_decodes_to_the_c_state_geometry(seed):
    env = _arena_env(seed)
    try:
        obs, _ = env.reset()
        sd = env._c_env.sd.contents
        ag = env._c_env.game.agents
        a, e = ag[HERO], ag[ENEMY]
        map_diag = math.sqrt((1.0 / sd.inv_x_range)**2 + (1.0 / sd.inv_y_range)**2)
        o = obs[HERO]
        # Self block: position, facing, hp, alive, team.
        assert o[0] == pytest.approx(a.hp / 100.0)
        assert o[3] == pytest.approx(a.x * sd.inv_x_range - sd.x_offset, abs=1e-6)
        assert o[4] == pytest.approx(a.y * sd.inv_y_range - sd.y_offset, abs=1e-6)
        assert o[9] == pytest.approx(math.sin(a.facing), abs=1e-6)
        assert o[10] == pytest.approx(math.cos(a.facing), abs=1e-6)
        assert o[23] == 1.0 and o[24] == 1.0
        # Enemy slot 0 is the one enemy: alive, visible, facing-relative geometry.
        b = ENEMY_BASE
        dx, dy = e.x - a.x, e.y - a.y
        rel = _wrap_pi(math.atan2(dy, dx) - a.facing)
        dist = math.hypot(dx, dy)
        assert o[b + 3] == 1.0 and o[b + 4] == 1.0, o[b:b + OBS_ENEMY_STRIDE]
        assert o[b + 5] == pytest.approx(math.sin(rel), abs=1e-5)
        assert o[b + 6] == pytest.approx(math.cos(rel), abs=1e-5)
        assert o[b + 7] == pytest.approx(dist / map_diag, abs=1e-6)
        assert o[b + 0] == pytest.approx(dist * math.cos(rel) / map_diag, abs=1e-5)
        assert o[b + 1] == pytest.approx(dist * math.sin(rel) / map_diag, abs=1e-5)
        assert o[b + 2] == pytest.approx((e.z - a.z) / 128.0, abs=1e-6)
        # Global block: full round clock, one of one alive per side.
        assert o[GLOBAL_BASE + 0] == 1.0
        assert o[GLOBAL_BASE + 11] == 1.0 and o[GLOBAL_BASE + 12] == 1.0
        # A parked row is dead in its own obs.
        assert obs[1][23] == 0.0
    finally:
        env.close()


def test_reset_obs_equals_the_obs_after_a_zero_action_step(simple_map):
    env = make_env(map_data=simple_map, auto_reset=False, seed=3)
    try:
        reset_obs = np.array(env.reset()[0], copy=True)
        assert all(reset_obs[i].any() for i in range(len(reset_obs))), "a reset obs row is all zero"
        disc, cont = zero_actions(len(reset_obs))
        step_obs, *_ = env.step(disc, cont)
        clock = GLOBAL_BASE + 0
        differs = np.argwhere(reset_obs != step_obs)
        assert set(differs[:, 1].tolist()) == {clock}, differs
        assert (reset_obs[:, clock] == 1.0).all()
    finally:
        env.close()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_reset_leaves_enemy_memory_at_its_init_values(seed):
    """Both agents see each other at reset, so an update_enemy_memory call in
    env_reset would record the enemy's area and the reset tick here. The
    visibility check keeps the pin from going vacuous: an enemy nobody sees
    would leave the memory untouched either way.
    """
    env = _arena_env(seed)
    try:
        obs, _ = env.reset()
        ag = env._c_env.game.agents
        for i in (HERO, ENEMY):
            assert list(ag[i].enemy_mem_idx) == [-1] * nav.TEAM_SIZE, (i, list(ag[i].enemy_mem_idx))
            assert list(ag[i].enemy_mem_tick) == [int(nav.STALE_MEMORY_TICK)] * nav.TEAM_SIZE, i
        assert obs[HERO][ENEMY_BASE + 3] == 1.0, "the hero must see the enemy at reset"
    finally:
        env.close()
