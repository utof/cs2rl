"""R0-E.1 (#130): enemy bearing / rel-pos are in the observer's facing frame."""
import math

import numpy as np
import pytest

from cs2rl.env.c.cs2_env import make_env
from cs2rl.env.config import EnvConfig
from cs2rl.spec.obs import OBS_BLOCKS

ENEMY_BASE = OBS_BLOCKS["enemy"][0]    # 56; agent 0's enemy slot 0 = agent 5
TM_BASE = OBS_BLOCKS["teammate"][0]    # 28
N_AGENTS, ACTION_DIM, AIM_DIM = 10, 7, 2


def _place_fixed(env, facing, dx=40.0, dy=0.0):
    """Enemy at a FIXED world offset (+40u in x, same room as Task 5's duel); only
    the observer's facing rotates — that is the invariant under test."""
    env.reset()
    ag = env._c_env.game.agents
    a0, a5 = ag[0], ag[5]
    a0.facing = facing
    a5.x, a5.y, a5.z, a5.area_idx = a0.x + dx, a0.y + dy, a0.z, a0.area_idx
    obs, *_ = env.step(np.zeros((N_AGENTS, ACTION_DIM), np.int32),
                       np.zeros((N_AGENTS, AIM_DIM), np.float32))
    assert obs[0][ENEMY_BASE + 3] == 1.0, "no LoS at +40u — placement geometry, not bearing"
    return obs[0]


@pytest.mark.parametrize("facing", [0.0, 1.0, math.pi / 2, -2.5, 3.0])
def test_enemy_bearing_is_facing_relative(simple_map, facing):
    env = make_env(map_data=simple_map, config=EnvConfig(n_active_per_team=1), seed=1)
    try:
        o = _place_fixed(env, facing)
        rel = math.atan2(0.0, 40.0) - facing           # world bearing 0 minus facing
        assert o[ENEMY_BASE + 5] == pytest.approx(math.sin(rel), abs=1e-4)
        assert o[ENEMY_BASE + 6] == pytest.approx(math.cos(rel), abs=1e-4)
        d_norm = o[ENEMY_BASE + 7]                     # dist / map_diag
        assert o[ENEMY_BASE + 0] == pytest.approx(d_norm * math.cos(rel), abs=1e-4)
        assert o[ENEMY_BASE + 1] == pytest.approx(d_norm * math.sin(rel), abs=1e-4)
        if facing == 0.0:                              # dead ahead ⇒ (0,1), (+d, 0)
            assert o[ENEMY_BASE + 5] == pytest.approx(0.0, abs=1e-5)
            assert o[ENEMY_BASE + 6] == pytest.approx(1.0, abs=1e-5)
    finally:
        env.close()


def test_memory_fallback_is_rotated(simple_map):
    """n_active=2 so killing ONE CT does not end the round (at n=1 the round would
    reset and the obs row would be all-zeros — a vacuous pass). Expected values are
    computed independently from centroid_xy and inv_x/y_range, never read back."""
    env = make_env(map_data=simple_map, config=EnvConfig(n_active_per_team=2), seed=1)
    try:
        env.reset()
        ag = env._c_env.game.agents
        a0, a5 = ag[0], ag[5]
        a0.facing = 1.2
        a5.alive = 0                                                                               # agent 5 invisible ⇒ memory path
        sd = env._c_env.sd.contents
                                                                                                   # spawn_team puts a0 EXACTLY on its area centroid, so the memory area
                                                                                                   # must be a different one — pick the first centroid > 1u away.
        mem_area = next(k for k in range(len(simple_map.centroids))
                        if abs(sd.centroid_xy[k * 2] - a0.x) +
                        abs(sd.centroid_xy[k * 2 + 1] - a0.y) > 1.0)
        a0.enemy_mem_idx[0] = mem_area                                                             # indexed by team-relative enemy idx (agent 5)
        cx, cy = sd.centroid_xy[mem_area * 2], sd.centroid_xy[mem_area * 2 + 1]
        map_diag = math.sqrt((1.0 / sd.inv_x_range)**2 + (1.0 / sd.inv_y_range)**2)
        obs, *_ = env.step(np.zeros((N_AGENTS, ACTION_DIM), np.int32),
                           np.zeros((N_AGENTS, AIM_DIM), np.float32))
        o = obs[0]
                                                                                                   # Enemy slots are sorted by KNOWN distance, so agent 5 is not necessarily
                                                                                                   # slot 0 (agent 6 is alive) — find the slot that is dead + not visible +
                                                                                                   # has a nonzero memory position.
        slot = next(s for s in range(5)
                    if o[ENEMY_BASE + s * 8 + 4] == 0.0 and o[ENEMY_BASE + s * 8 + 3] == 0.0 and (
                        o[ENEMY_BASE + s * 8 + 0] != 0.0 or o[ENEMY_BASE + s * 8 + 1] != 0.0))
        b = ENEMY_BASE + slot * 8
        mx, my = cx - a0.x, cy - a0.y
        cf, sf = math.cos(a0.facing), math.sin(a0.facing)
        rx, ry = mx * cf + my * sf, -mx * sf + my * cf
        assert o[b + 0] == pytest.approx(rx / map_diag, abs=1e-5)
        assert o[b + 1] == pytest.approx(ry / map_diag, abs=1e-5)
    finally:
        env.close()


def test_teammate_slots_stay_absolute(simple_map):
    env = make_env(map_data=simple_map, seed=1)                        # full 5v5
    try:
        env.reset()
        ag = env._c_env.game.agents
        ag[0].facing = 2.0
        obs, *_ = env.step(np.zeros((N_AGENTS, ACTION_DIM), np.int32),
                           np.zeros((N_AGENTS, AIM_DIM), np.float32))
        o = obs[0]
        dx, dy = ag[1].x - ag[0].x, ag[1].y - ag[0].y
        ang = math.atan2(dy, dx)                                       # absolute — unchanged
        assert o[TM_BASE + 5] == pytest.approx(math.sin(ang), abs=1e-4)
        assert o[TM_BASE + 6] == pytest.approx(math.cos(ang), abs=1e-4)
    finally:
        env.close()
