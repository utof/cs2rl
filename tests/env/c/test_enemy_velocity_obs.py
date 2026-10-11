"""sim-v4: enemy slots +8/+9 are the enemy's velocity relative to the observer, facing frame, /250.

What each test catches (cs2_observations.h, the enemy slot loop):
  * the rotation: the slot rotated by +facing instead of -facing, or not rotated;
  * the scale: any divisor other than 250 (self obs[6]/[7]'s);
  * relative, not own: the observer's velocity not subtracted;
  * the slot order: a slot carrying another enemy's velocity than the one its +0/+1 locate
    (the distance sort reorders the slots; one active enemy cannot show it);
  * the gating: velocity written for an unseen enemy, alive (behind a wall, with and without a
    last-known position) or dead.
Every expected value is computed here from the post-step C state (positions, velocities and
facing after the step that wrote the obs), never read back from the obs.
"""
import math

import numpy as np
import pytest

from cs2rl.env.c.cs2_env import make_env
from cs2rl.env.config import EnvConfig
from cs2rl.spec.obs import OBS_BLOCKS, OBS_ENEMY_STRIDE
from tests._helpers.scenario import kill_agent, place_agent, place_in_area

ENEMY_BASE = OBS_BLOCKS["enemy"][0]
SEE, ALIVE, VX, VY = 3, 4, 8, 9        # field offsets inside one enemy slot
VEL_SCALE = 250.0                      # cs2_observations.h, same as self obs[6]/[7]
N_AGENTS, ACTION_DIM, AIM_DIM = 10, 7, 2


def _step(env):
    obs, *_ = env.step(np.zeros((N_AGENTS, ACTION_DIM), np.int32),
                       np.zeros((N_AGENTS, AIM_DIM), np.float32))
    return obs


def _expected(observer, enemy):
    """(+8, +9) from the C state: (v_enemy - v_observer) rotated by -facing, / 250."""
    rvx, rvy = enemy.vx - observer.vx, enemy.vy - observer.vy
    cf, sf = math.cos(observer.facing), math.sin(observer.facing)
    return (rvx * cf + rvy * sf) / VEL_SCALE, (-rvx * sf + rvy * cf) / VEL_SCALE


@pytest.mark.parametrize("facing", [0.0, 1.0, math.pi / 2, -2.5, 3.0])
def test_visible_enemy_slot_carries_relative_velocity_in_the_facing_frame(simple_map, facing):
    env = make_env(map_data=simple_map, config=EnvConfig(n_active_per_team=1), seed=1)
    try:
        env.reset()
        g = env._c_env.game
        a0 = g.agents[0]
        a0.facing = facing
        place_agent(env, 5, a0.x + 40.0, a0.y)
        a0.vx, a0.vy = 30.0, -50.0
        g.agents[5].vx, g.agents[5].vy = 120.0, 80.0
        o = _step(env)[0]
        en = g.agents[5]
        assert o[ENEMY_BASE + SEE] == 1.0, "no LoS at +40u: placement geometry, not velocity"
        # Not vacuous: friction leaves both moving and the relative velocity well away from 0.
        assert math.hypot(en.vx - a0.vx, en.vy - a0.vy) > 50.0, (en.vx, en.vy, a0.vx, a0.vy)
        ex, ey = _expected(a0, en)
        assert o[ENEMY_BASE + VX] == pytest.approx(ex, abs=1e-5)
        assert o[ENEMY_BASE + VY] == pytest.approx(ey, abs=1e-5)
        # Own velocity would differ by the observer's rotated velocity (> 0.1 here).
        cf, sf = math.cos(a0.facing), math.sin(a0.facing)
        own = ((en.vx * cf + en.vy * sf) / VEL_SCALE, (-en.vx * sf + en.vy * cf) / VEL_SCALE)
        assert abs(own[0] - ex) + abs(own[1] - ey) > 0.1
    finally:
        env.close()


def _enemy_located_by(px, py, observer, game, candidates):
    """The candidate whose direction from the observer, rotated by -facing, is the slot's
    rel-pos (+0, +1) direction (the 1 / map_diag scale cancels). Exactly one must match."""
    cf, sf = math.cos(observer.facing), math.sin(observer.facing)
    hits = []
    for j in candidates:
        dx, dy = game.agents[j].x - observer.x, game.agents[j].y - observer.y
        rx, ry = dx * cf + dy * sf, -dx * sf + dy * cf
        if (rx * px + ry * py) / (math.hypot(rx, ry) * math.hypot(px, py)) > 0.9999:
            hits.append(j)
    assert len(hits) == 1, (px, py, hits)
    return hits[0]


def test_each_slot_carries_the_velocity_of_the_enemy_it_locates(simple_map):
    """Two visible enemies, the higher index (6) nearer: the distance sort puts 6 in slot 0 and
    5 in slot 1, so a writer that took the velocity from another agent than the slot's sorted
    enemy (say the slot index's, en_start + slot) fails here while every one-enemy case passes."""
    env = make_env(map_data=simple_map, config=EnvConfig(n_active_per_team=2), seed=1)
    try:
        env.reset()
        g = env._c_env.game
        a0 = g.agents[0]
        a0.facing = 1.0
        place_agent(env, 5, a0.x + 50.0, a0.y + 50.0)  # farther, lower index
        place_agent(env, 6, a0.x + 40.0, a0.y)         # nearer, higher index
        a0.vx, a0.vy = 30.0, -50.0
        g.agents[5].vx, g.agents[5].vy = 120.0, 80.0
        g.agents[6].vx, g.agents[6].vy = -90.0, 40.0
        o = _step(env)[0]

        # Not vacuous: the two enemies' expected slot velocities are far apart.
        e5, e6 = _expected(a0, g.agents[5]), _expected(a0, g.agents[6])
        assert abs(e5[0] - e6[0]) + abs(e5[1] - e6[1]) > 0.1, (e5, e6)
        carried = []
        for s in range(2):
            b = ENEMY_BASE + s * OBS_ENEMY_STRIDE
            assert o[b + SEE] == 1.0, f"slot {s} not visible: placement geometry, not velocity"
            j = _enemy_located_by(o[b], o[b + 1], a0, g, (5, 6))
            carried.append(j)
            ex, ey = _expected(a0, g.agents[j])
            assert o[b + VX] == pytest.approx(ex, abs=1e-5), (s, j)
            assert o[b + VY] == pytest.approx(ey, abs=1e-5), (s, j)
        assert carried == [6, 5], f"slots in index order, the sort was not exercised: {carried}"
    finally:
        env.close()


def test_dead_enemy_slot_has_no_velocity(simple_map):
    """n_active 2, so one death does not end the round (the obs would be the next round's)."""
    env = make_env(map_data=simple_map, config=EnvConfig(n_active_per_team=2), seed=1)
    try:
        env.reset()
        g = env._c_env.game
        a0 = g.agents[0]
        place_agent(env, 5, a0.x + 40.0, a0.y)
        kill_agent(env, 5)
        g.agents[5].vx, g.agents[5].vy = 150.0, -90.0
        o = _step(env)[0]
        assert (g.agents[5].vx, g.agents[5].vy) != (0.0, 0.0), "the corpse kept no velocity"
        dead = [
            s for s in range(5) if o[ENEMY_BASE + s * OBS_ENEMY_STRIDE + ALIVE] == 0.0 and s < 2
        ]
        assert dead, "no dead enemy slot among the two active"
        for s in dead:
            b = ENEMY_BASE + s * OBS_ENEMY_STRIDE
            assert o[b + SEE] == 0.0
            assert (o[b + VX], o[b + VY]) == (0.0, 0.0), f"dead slot {s} carries a velocity"
    finally:
        env.close()


def test_unseen_alive_enemy_slots_have_no_velocity():
    """dust2 geometry from test_enemy_slot_sort_does_not_leak_invisible_rank: enemy 6 stands
    behind a wall ~36 u from the observer at area 0's centroid; enemy 7 stays at the CT spawn.
    Both move and neither is visible. Enemy 6 is heard (its footsteps), so its slot takes the
    last-known-position branch (+0/+1 from memory); enemy 7 is out of earshot and the
    observer's memory is wiped, so its slot is the no-memory sentinel (+0/+1 zero). +8/+9 stay
    0 in both: the memory keeps an area index, no velocity."""
    env = make_env(seed=0, auto_reset=False)
    try:
        g = env._c_env.game
        env.reset()
        observer = g.agents[0]
        for m in range(5):
            observer.enemy_mem_idx[m] = -1             # INVALID_AREA_IDX
        place_in_area(env, 0, 0)
        place_agent(env, 6, 162.5, -10.875)
        g.agents[6].vx, g.agents[6].vy = 60.0, 0.0     # +y runs into the wall: speed 0
        g.agents[7].vx, g.agents[7].vy = 90.0, 0.0
        o = _step(env)[0]
        for j in (6, 7):
            assert math.hypot(g.agents[j].vx, g.agents[j].vy) > 10.0, f"enemy {j} stopped"
        slots = [ENEMY_BASE + s * OBS_ENEMY_STRIDE for s in range(5)]
        assert all(o[b + SEE] == 0.0 for b in slots), "an enemy is visible: wall geometry moved"
        live = [b for b in slots if o[b + ALIVE] == 1.0]
        assert any(o[b] != 0.0 or o[b + 1] != 0.0 for b in live), "no slot took the memory branch"
        assert any(o[b] == 0.0 and o[b + 1] == 0.0 for b in live), "no slot is the sentinel"
        for b in slots:
            assert (o[b + VX], o[b + VY]) == (0.0, 0.0), f"unseen slot at {b} carries a velocity"
    finally:
        env.close()
