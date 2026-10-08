"""Pin the scenario helpers (tests/_helpers/scenario.py) and the two binding queries under them.

binding.ground_at and binding.weapon_defs export the sim's own rules (#170): which area holds a
point and where its floor is (cs2_movement.h), and the weapon table (cs2_weapons.h). These tests
pin that each export says what the sim does, by comparing it with what a step or a reset
writes, and that the helpers refuse the impossible states the old field-by-field setups built.
"""
import numpy as np
import pytest

from cs2rl.env.c import binding
from cs2rl.env.c.cs2_env import make_env
from cs2rl.env.config import EnvConfig
from cs2rl.env.map import make_arena_duel_map, make_simple_map
from cs2rl.spec.action import ACTION_HEAD_SIZES
from tests._helpers import scenario
from tests._helpers.scenario import (
    ground_at,
    kill_agent,
    place_agent,
    place_in_area,
    ready_to_fire,
    set_clip,
    state_violations,
    zero_actions,
)


def _masked_random(rng, masks):
    """One legal random action per head, read from the env's own masks."""
    acts, _ = zero_actions()
    off = 0
    for h, size in enumerate(ACTION_HEAD_SIZES):
        for i in range(acts.shape[0]):
            ok = np.flatnonzero(masks[i, off:off + size])
            acts[i, h] = rng.choice(ok) if ok.size else 0
        off += size
    return acts


@pytest.mark.parametrize("map_name", ["simple", "dust2"])
def test_sim_states_satisfy_ground_at_and_the_invariants(map_name):
    """Every state a seeded random rollout reaches agrees with ground_at and has no violation.

    After a step, movement has written each live agent's area_idx (`_area_at`) and, when it
    is grounded, its z (`_surface_z` on that area); ground_at must give the same answer, or
    the helpers would place agents where the sim never puts them. state_violations must find
    nothing in the sim's own states, the reset (dust2's off-area spawn centroid) included,
    or assert_state_consistent would reject reachable scenarios.
    """
    env = make_env(seed=3,
                   auto_reset=False,
                   map_data=make_simple_map() if map_name == "simple" else None)
    try:
        rng = np.random.default_rng(3)
        env.reset()
        assert state_violations(env) == []
        grounded = 0
        for _ in range(120):
            env.step(_masked_random(rng, env._masks_view))
            assert state_violations(env) == []
            for i in range(10):
                a = env._c_env.game.agents[i]
                if not a.alive:
                    continue
                area, z = ground_at(env, a.x, a.y)
                assert area == a.area_idx, (i, a.x, a.y)
                if not a.is_airborne:
                    grounded += 1
                    assert z == pytest.approx(a.z, abs=1e-4), (i, a.x, a.y)
            if env._c_env.game.round_over:
                env.reset()
        assert grounded > 0
    finally:
        env.close()


@pytest.mark.parametrize(
    "x,y,unsettled_z,area,floor",
    [
        (785.0, 304.0, 64.0, 13, 32.0),                # T-ramp midpoint: x-slope 0 -> 64
        (1235.0, 136.0, 128.0, 16, 64.0),              # stairs midpoint: y-slope 128 -> 0
        (950.0, 300.0, 0.0, 6, 64.0),                  # flat elevated bombsite
                                                       # The raster cell here is labelled 13 (a later room owns the portal column),
                                                       # but the point is inside room 5's quad: _area_at's room fallback answers 5.
        (740.0, 300.0, 0.0, 5, 0.0),
    ])
def test_ground_at_is_what_the_ground_snap_writes(x, y, unsettled_z, area, floor):
    """A grounded agent placed with a wrong z settles, on one zero step, on ground_at's floor."""
    env = make_env(seed=1, auto_reset=False, map_data=make_simple_map())
    try:
        env.reset()
        assert ground_at(env, x, y) == (area, floor)
        a = place_agent(env, 0, x, y, z=unsettled_z)
        assert (a.area_idx, a.z) == (area, unsettled_z)
        env.step(*zero_actions())
        assert (a.x, a.y, a.area_idx, a.is_airborne) == (x, y, area, 0)
        assert a.z == pytest.approx(floor, abs=1e-4)
    finally:
        env.close()


def test_ground_at_off_the_mesh_and_place_agent_refusals():
    """No area holds a point outside the grid: ground_at says so and place_agent refuses it."""
    env = make_env(seed=1,
                   auto_reset=False,
                   map_data=make_simple_map(),
                   config=EnvConfig(n_active_per_team=2))
    try:
        env.reset()
        assert ground_at(env, -5000.0, 0.0) == (scenario.INVALID_AREA_IDX, None)
        with pytest.raises(ValueError, match="off the mesh"):
            place_agent(env, 0, -5000.0, 0.0)
        with pytest.raises(ValueError, match="dead or parked"):
            place_agent(env, 4, 950.0, 300.0)          # slot 4 is parked at n_active=2
        with pytest.raises(ValueError, match="above the floor"):
            place_agent(env, 0, 950.0, 300.0, z=64.0, airborne=True)
        kill_agent(env, 1)
        with pytest.raises(ValueError, match="dead or parked"):
            place_agent(env, 1, 950.0, 300.0)
        with pytest.raises(ValueError, match="dead or parked"):
            kill_agent(env, 1)
    finally:
        env.close()


def test_place_agent_derives_the_area_and_the_state_stays_consistent():
    """The placed agent is where the sim would hold it: consistent before and after a step."""
    env = make_env(seed=1, auto_reset=False, map_data=make_simple_map())
    try:
        env.reset()
        a = place_agent(env, 0, 950.0, 300.0)
        assert (a.area_idx, a.z, a.is_airborne) == (6, 64.0, 0)
        b = place_agent(env, 5, 950.0, 300.0, z=200.0, airborne=True)
        assert (b.area_idx, b.vz, b.is_airborne) == (6, 0.0, 1)
        assert state_violations(env) == []
        env.step(*zero_actions())
        assert state_violations(env) == []
        assert b.z < 200.0 and b.is_airborne == 1      # falling: the step integrated gravity
    finally:
        env.close()


def test_place_in_area_uses_the_centroid_and_refuses_one_outside_its_area():
    """Simple-map centroids sit in their own area; dust2 has centroids that do not."""
    env = make_env(seed=1, auto_reset=False, map_data=make_simple_map())
    try:
        env.reset()
        a = place_in_area(env, 0, 6)
        assert (a.x, a.y, a.area_idx) == (*scenario.area_centroid(env, 6), 6)
    finally:
        env.close()
    env = make_env(seed=1, auto_reset=False)
    try:
        env.reset()
        x, y = scenario.area_centroid(env, 1645)       # a CT spawn area, measured off-area
        assert ground_at(env, x, y)[0] != 1645
        with pytest.raises(ValueError, match="resolves to area"):
            place_in_area(env, 0, 1645)
    finally:
        env.close()


def test_weapon_defs_are_the_spawn_loadout():
    """init_agent_ammo fills each gun's clip with mag_size and its reserve with reserve_mags."""
    defs = binding.weapon_defs()
    assert [d["type"] for d in defs] == [0, 1, 2]
    assert (defs[2]["mag_size"], defs[2]["reserve_mags"]) == (-1, -1)  # the knife
    env = make_env(seed=1, auto_reset=False, map_data=make_simple_map())
    try:
        env.reset()
        for i in range(10):
            a = env._c_env.game.agents[i]
            for s, d in enumerate(defs):
                assert (a.ammo_clip[s], a.ammo_reserve[s]) == (d["mag_size"], d["reserve_mags"])
    finally:
        env.close()


def test_weapon_defs_cycle_ticks_is_the_fire_cooldown():
    """A rifle shot leaves fire_cd = cycle_ticks, as process_combat writes it."""
    env = make_env(seed=1,
                   auto_reset=False,
                   map_data=make_arena_duel_map(),
                   config=EnvConfig(n_active_per_team=1))
    try:
        env.reset()
        act, cont = zero_actions()
        act[0, 1] = 1
        env.step(act, cont)
        assert env._c_env.game.agents[0].fire_cd == binding.weapon_defs()[0]["cycle_ticks"]
    finally:
        env.close()


def test_clip_and_readiness_refuse_impossible_weapon_states():
    """A 30-round rifle clip (WEAPON_DEFS says 25), an empty clip and a mid-switch are refused."""
    env = make_env(seed=1, auto_reset=False, map_data=make_simple_map())
    try:
        env.reset()
        mag = binding.weapon_defs()[0]["mag_size"]
        with pytest.raises(ValueError, match="do not fit"):
            set_clip(env, 0, mag + 1)
        with pytest.raises(ValueError, match="no clip"):
            set_clip(env, 0, 1, slot=2)
        a = set_clip(env, 0, 0)
        with pytest.raises(ValueError, match="empty clip"):
            ready_to_fire(env, 0)
        set_clip(env, 0, mag)
        a.fire_cd, a.reload_ticks = 2, 5
        ready_to_fire(env, 0)
        assert (a.fire_cd, a.reload_ticks, a.switch_ticks) == (0, 0, 0)
        a.weapon_slot_target = 1
        with pytest.raises(ValueError, match="switching"):
            ready_to_fire(env, 0)
    finally:
        env.close()


def _break(env, case):
    """Write one impossible state into a fresh simple-map 5v5 env (n_active 4 for 'parked')."""
    g = env._c_env.game
    a = g.agents[0]
    if case == "alive_without_hp":
        a.hp = 0
    elif case == "dead_with_hp":
        a.alive = 0
    elif case == "area_not_at_position":
        a.x += 400.0                   # leaves area_idx behind
    elif case == "area_none_while_alive":
        a.area_idx = scenario.INVALID_AREA_IDX
    elif case == "clip_over_mag":
        a.ammo_clip[0] = 30
    elif case == "reserve_over_mags":
        a.ammo_reserve[1] = 9
    elif case == "bad_weapon_slot":
        a.weapon_slot = 3
    elif case == "wrong_team":
        a.team = 1
    elif case == "parked_alive":
        g.agents[4].alive = 1
    elif case == "participating_parked_slot":
        g.agents[4].participating = 1
    else:
        raise AssertionError(case)


@pytest.mark.parametrize("case", [
    "alive_without_hp", "dead_with_hp", "area_not_at_position", "area_none_while_alive",
    "clip_over_mag", "reserve_over_mags", "bad_weapon_slot", "wrong_team", "parked_alive",
    "participating_parked_slot"
])
def test_assert_state_consistent_rejects_each_impossible_state(case):
    """Each invariant has its own impossible state, and assert_state_consistent names it."""
    env = make_env(seed=1,
                   auto_reset=False,
                   map_data=make_simple_map(),
                   config=EnvConfig(n_active_per_team=4))
    try:
        env.reset()
        scenario.assert_state_consistent(env)          # the sim's state passes
        _break(env, case)
        with pytest.raises(AssertionError, match="breaks the sim's invariants"):
            scenario.assert_state_consistent(env)
    finally:
        env.close()


def test_kill_agent_writes_the_combat_kill_and_the_sim_drops_the_bomb():
    """hp and alive go to 0 together; the next step runs the sim's own drop for a carrier."""
    env = make_env(seed=1, auto_reset=False, map_data=make_simple_map())
    try:
        env.reset()
        g = env._c_env.game
        carrier = g.bomb_carrier
        a = kill_agent(env, carrier)
        assert (a.alive, a.hp) == (0, 0)
        assert state_violations(env) == []
        env.step(*zero_actions())
        assert g.bomb_carrier != carrier
    finally:
        env.close()


def test_kill_all_but_kills_only_the_live_others():
    """Parked slots are left alone (they are not alive) and the kept agents stay alive."""
    env = make_env(seed=1,
                   auto_reset=False,
                   map_data=make_simple_map(),
                   config=EnvConfig(n_active_per_team=2))
    try:
        env.reset()
        assert scenario.kill_all_but(env, 0, 5) == [1, 6]
        assert [env._c_env.game.agents[i].alive
                for i in range(10)] == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        assert state_violations(env) == []
    finally:
        env.close()


def test_visible_area_pair_is_a_sim_sighting_and_face_aims_a_kill():
    """The pair is in sight by the sim's own raycast, and face turns the shooter onto it.

    visible_area_pair judges sight with MapData.line_of_sight_2d, the Python mirror of the
    raycast build_vis_matrix runs on positions; the step's mutual_vis_pair_ticks counts the
    sim's verdict for the one live T/CT pair. A 1-hp target shot at pitch 0 on dust2's flat
    floor dies only if face pointed the shot at it.
    """
    env = make_env(auto_reset=False)
    try:
        env.reset()
        area_t, area_ct = scenario.visible_area_pair(env, min_dist=50, max_dist=1500)
        scenario.kill_all_but(env, 0, 5)
        place_in_area(env, 0, area_t)
        ct = place_in_area(env, 5, area_ct)
        ct.hp, ct.armor = 1, 0
        scenario.face(env, 0, 5)
        assert state_violations(env) == []
        act, cont = zero_actions()
        act[0, 1] = 1
        env.step(act, cont)
        assert env._c_env.step_stats.mutual_vis_pair_ticks == 1
        assert (ct.alive, ct.hp) == (0, 0)
        with pytest.raises(AssertionError, match="no area pair"):
            scenario.visible_area_pair(env, min_dist=1e9, max_dist=2e9)
    finally:
        env.close()


def test_zero_actions_shapes():
    """One row per agent, ACTION_DIM int32 discrete heads and AIM_DIM float32 continuous."""
    act, cont = zero_actions()
    assert act.shape == (10, len(ACTION_HEAD_SIZES)) and act.dtype == np.int32 and not act.any()
    assert cont.shape[0] == 10 and cont.dtype == np.float32 and not cont.any()
    assert zero_actions(3)[0].shape[0] == 3
