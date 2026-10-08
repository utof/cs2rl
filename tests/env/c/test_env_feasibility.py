# tests/env/c/test_env_feasibility.py

import numpy as np

from cs2rl.env.c.cs2_env import make_env
from cs2rl.env.nav import BOMB_PLANT_TIME, LASER_RANGE, TEAM_SIZE

# Batch 6 Task 2: the pathing + driving helpers were extracted to
# src/cs2rl/eval/scripted_expert.py (spec D-3) so the BC demo generator shares one
# canonical implementation with these tests. drive_agent_through_area_path
# is the facing-poke TEST driver; the action-interface expert used for demo
# recording is scripted_expert.ScriptedBomber.
from cs2rl.eval.scripted_expert import (
    bfs_area_path,
    bombsite_areas,
    drive_agent_through_area_path,
    setup_bomb_carrier,
)
from cs2rl.spec.action import ACTION_DIM
from tests._helpers.scenario import (
    assert_state_consistent,
    face,
    kill_all_but,
    place_in_area,
    visible_area_pair,
)


def test_spawn_areas_are_distinct_and_site_reachable():
    env = make_env()
    env.reset()
    bombsites = bombsite_areas(env.map_data)
    t_spawn_areas = env.map_data.t_spawn_areas
    ct_spawn_areas = env.map_data.ct_spawn_areas

    assert len(t_spawn_areas) == TEAM_SIZE
    assert len(ct_spawn_areas) == TEAM_SIZE
    assert len(set(t_spawn_areas)) == TEAM_SIZE
    assert len(set(ct_spawn_areas)) == TEAM_SIZE

    for spawn_area in t_spawn_areas:
        path = bfs_area_path(env.map_data, spawn_area, bombsites)
        assert path, f"T spawn area {spawn_area} cannot reach any bombsite"

    env.close()


def test_scripted_bomber_can_reach_site_and_plant():
    # auto_reset=False so a long walk across the map can't silently be
    # interrupted by a round reset if round_ticks_left hits zero while
    # we're still driving. We also lift the round timer below.
    env = make_env(auto_reset=False)
    env.reset()
    bombsites = bombsite_areas(env.map_data)

    # Lift the round timer for this test. The area-level BFS path on real
    # de_dust2 can be 30+ hops long, and the face-centroid + accel driver
    # spends more ticks per hop than the old world-compass one-move-per-
    # tick planner did. The production round budget (640 ticks) is tuned
    # for playable matches, not scripted test traversal.
    env._c_env.game.round_ticks_left = 100000

    bomber_idx = 4
    # The bomb and the knife (highest wishspeed, 250 u/s: the bomber rolls up
    # to max speed in ~3 accel ticks and spends less budget per area hop).
    setup_bomb_carrier(env, bomber_idx)

    bomber = env._c_env.game.agents[bomber_idx]
    start_area_id = int(env.map_data.area_ids[bomber.area_idx])
    area_path = bfs_area_path(env.map_data, start_area_id, bombsites)
    assert area_path, "No area-level path from bomber spawn to any bombsite"
    final_site_area = area_path[-1]

    assert drive_agent_through_area_path(
        env, bomber_idx, area_path), (f"Bomber failed to walk area path {area_path}")

    assert int(env.map_data.area_ids[bomber.area_idx]) == final_site_area
    assert int(env.map_data.area_ids[bomber.area_idx]) in bombsites

    for _ in range(BOMB_PLANT_TIME):
        actions = np.zeros((10, ACTION_DIM), dtype=np.int64)
        # Batch 3: head order changed — USE is now index 4 (was 5).
        # Order is now: move=0, shoot=1, reload=2, weapon=3, use=4, crouch=5, jump=6.
        actions[bomber_idx, 4] = 1     # use action plants the bomb
        env.step(actions)
        if env._c_env.game.bomb_planted:
            break

    assert env._c_env.game.bomb_planted, "Bomber failed to plant after reaching bombsite"
    env.close()


def test_controlled_visible_agents_can_kill():
    env = make_env(auto_reset=False)
    env.reset()
    # Two visible areas within half the laser range, by runtime position-LoS
    # (gh #36 follow-up), not the centroid-baked vis_matrix: see visible_area_pair.
    area_t, area_ct = visible_area_pair(env, min_dist=50, max_dist=LASER_RANGE * 0.5)
    kill_all_but(env, 0, 5)
    place_in_area(env, 0, area_t)
    ct_agent = place_in_area(env, 5, area_ct)
    ct_agent.hp = 1                    # low HP so any hit kills
    ct_agent.armor = 0

    # Batch 3: aim is a continuous head; facing is set directly on the agent
    # and the discrete actions buffer carries no aim bin. Pitch is absolute
    # per step and the zero continuous buffer sets 0, which hits a same-floor
    # target (dust2 is flat). SHOOT moved from index 2 to index 1 in the new
    # enum (move=0, shoot=1, reload=2, weapon=3, use=4, crouch=5, jump=6).
    face(env, 0, 5)
    assert_state_consistent(env)

    actions = np.zeros((10, ACTION_DIM), dtype=np.int64)
    # Batch 3: SHOOT moved from head 2 → 1 after HEAD_AIM removal.
    actions[0, 1] = 1                  # t0 shoots (shoot is head 1)
    _, rewards, _, _, _ = env.step(actions)

    assert not bool(env._c_env.game.agents[5].alive)
    assert rewards[0] > 0

    env.close()


def test_fixed_seed_agents_can_leave_spawn():
    env = make_env()
    env.reset()
    bombsites = bombsite_areas(env.map_data)
    t_spawn_areas = env.map_data.t_spawn_areas

    for agent_idx in range(10):
        ca = env._c_env.game.agents[agent_idx]
        area_id = int(env.map_data.area_ids[ca.area_idx])
        start_x, start_y = float(ca.x), float(ca.y)

        if agent_idx < TEAM_SIZE:
            goals = bombsites
        else:
            goals = set(t_spawn_areas)

        path = bfs_area_path(env.map_data, area_id, goals)
        assert len(path) >= 2, f"No route out of spawn for agent {agent_idx}"

        # Drive only to the first area beyond spawn — this test just wants
        # to confirm the agent can leave its starting area under the
        # current movement model, not traverse the full route.
        solo_env = make_env()
        solo_env.reset()
        drive_agent_through_area_path(solo_env, agent_idx, path[:2], max_ticks_per_hop=64)

        moved = solo_env._c_env.game.agents[agent_idx]
        moved_area_id = int(solo_env.map_data.area_ids[moved.area_idx])
        assert moved_area_id != area_id or not np.allclose(
            [moved.x, moved.y], [start_x, start_y]), f"agent {agent_idx} failed to leave spawn"
        solo_env.close()

    env.close()
