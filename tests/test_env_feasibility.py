# tests/test_env_feasibility.py
import math
from collections import deque

import numpy as np

from c_env.cs2_env import make_env
from nav import ACTION_DIM, BOMB_PLANT_TIME, LASER_RANGE, TEAM_SIZE


def _bombsite_areas(env):
    return {
        int(aid)
        for i, aid in enumerate(env.map_data.area_ids) if env.map_data.bombsite_by_idx[i]
    }


def _bfs_area_path(nav_graph, adjacency, start_area: int, goal_areas) -> list[int]:
    """Area-level BFS over the nav adjacency graph. Returns a list of area
    ids [start, ..., goal] or [] if unreachable. Used only to pick *which*
    areas the agent should traverse; the actual driving is done by
    _drive_agent_through_area_path below."""
    goal_areas = set(goal_areas)
    id_to_idx = nav_graph._id_to_idx
    q = deque([start_area])
    prev = {start_area: None}
    found = None

    while q:
        cur = q.popleft()
        if cur in goal_areas:
            found = cur
            break
        cur_idx = id_to_idx[cur]
        for nbr_idx in np.flatnonzero(adjacency[cur_idx]):
            nbr_area = nav_graph.area_ids[int(nbr_idx)]
            if nbr_area == cur or nbr_area in prev:
                continue
            prev[nbr_area] = cur
            q.append(nbr_area)

    if found is None:
        return []

    path = []
    cur = found
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _drive_agent_through_area_path(env, agent_idx, area_path, max_ticks_per_hop=256) -> bool:
    """Walk an agent from its current position through `area_path` by
    repeatedly (face next-area centroid, action = move forward) each tick.

    Why this shape: movement is now facing-local with Source-style accel +
    friction, so world-compass step planning no longer maps 1:1 to actions.
    We directly poke `a->facing` before each env.step instead of going
    through the aim action head, which avoids the aim-applied-after-
    movement single-tick lag. The agent rolls up to wishspeed over ~3
    ticks and naturally curves toward each centroid.

    Corner/wall unsticking: collision blocks velocity, so if the bot's
    straight line to the next centroid clips a wall it will stall at the
    wall. We detect a static position and jitter facing ±45°/±90° to find
    a clear direction. This mirrors how a trained policy would learn to
    wiggle around corners; it's good enough for test driving without
    requiring a full sub-cell planner.

    Returns True iff the agent ends its journey inside `area_path[-1]`."""
    jitter_seq = [
        0.0,
        math.pi / 8,
        -math.pi / 8,
        math.pi / 4,
        -math.pi / 4,
        math.pi / 2,
        -math.pi / 2,
    ]
    for target_area in area_path[1:]:
        last_pos = None
        stuck = 0
        reached = False
        for _ in range(max_ticks_per_hop):
            ca = env._c_env.game.agents[agent_idx]
            cur_area_id = int(env.map_data.area_ids[ca.area_idx])
            if cur_area_id == target_area:
                reached = True
                break

            pos = (round(float(ca.x), 1), round(float(ca.y), 1))
            if pos == last_pos:
                stuck += 1
            else:
                stuck = 0
                last_pos = pos

            cx, cy = env.nav_graph.centroids[target_area]
            base = math.atan2(float(cy) - float(ca.y), float(cx) - float(ca.x))
            ca.facing = base + jitter_seq[min(stuck // 3, len(jitter_seq) - 1)]

            actions = np.zeros((10, ACTION_DIM), dtype=np.int64)
            actions[agent_idx, 0] = 1  # facing-local "move forward"
            env.step(actions)
        if not reached:
            return False
    return True


def test_spawn_areas_are_distinct_and_site_reachable():
    env = make_env()
    env.reset()
    nav_graph = env.nav_graph
    adjacency = env.map_data.adjacency
    bombsites = _bombsite_areas(env)
    t_spawn_areas = env.map_data.t_spawn_areas
    ct_spawn_areas = env.map_data.ct_spawn_areas

    assert len(t_spawn_areas) == TEAM_SIZE
    assert len(ct_spawn_areas) == TEAM_SIZE
    assert len(set(t_spawn_areas)) == TEAM_SIZE
    assert len(set(ct_spawn_areas)) == TEAM_SIZE

    for spawn_area in t_spawn_areas:
        path = _bfs_area_path(nav_graph, adjacency, spawn_area, bombsites)
        assert path, f"T spawn area {spawn_area} cannot reach any bombsite"

    env.close()


def test_scripted_bomber_can_reach_site_and_plant():
    # auto_reset=False so a long walk across the map can't silently be
    # interrupted by a round reset if round_ticks_left hits zero while
    # we're still driving. We also lift the round timer below.
    env = make_env(auto_reset=False)
    env.reset()
    bombsites = _bombsite_areas(env)

    # Lift the round timer for this test. The area-level BFS path on real
    # de_dust2 can be 30+ hops long, and the face-centroid + accel driver
    # spends more ticks per hop than the old world-compass one-move-per-
    # tick planner did. The production round budget (640 ticks) is tuned
    # for playable matches, not scripted test traversal.
    env._c_env.game.round_ticks_left = 100000

    bomber_idx = 4
    for i in range(10):
        env._c_env.game.agents[i].has_bomb = 0
    env._c_env.game.agents[bomber_idx].has_bomb = 1
    env._c_env.game.bomb_carrier_id = bomber_idx

    # Knife has the highest wishspeed (250 u/s) so the bomber rolls up to
    # max speed in ~3 accel ticks and spends less budget per area hop.
    env._c_env.game.agents[bomber_idx].weapon_slot = 2
    env._c_env.game.agents[bomber_idx].weapon_slot_target = 2
    env._c_env.game.agents[bomber_idx].switch_ticks = 0

    bomber = env._c_env.game.agents[bomber_idx]
    start_area_id = int(env.map_data.area_ids[bomber.area_idx])
    area_path = _bfs_area_path(env.nav_graph, env.map_data.adjacency, start_area_id, bombsites)
    assert area_path, "No area-level path from bomber spawn to any bombsite"
    final_site_area = area_path[-1]

    assert _drive_agent_through_area_path(
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
    nav_graph = env.nav_graph
    id2idx = {int(aid): i for i, aid in enumerate(env.map_data.area_ids)}

    pair = None
    area_ids = nav_graph.area_ids
    for i, area_i in enumerate(area_ids[:400]):
        for area_j in area_ids[i + 1:i + 200]:
            if not env.map_data.vis_matrix[id2idx[area_i], id2idx[area_j]]:
                continue
            dx = nav_graph.centroids[area_j][0] - nav_graph.centroids[area_i][0]
            dy = nav_graph.centroids[area_j][1] - nav_graph.centroids[area_i][1]
            dist = float((dx * dx + dy * dy)**0.5)
            if 50 < dist < LASER_RANGE * 0.5:
                pair = (area_i, area_j)
                break
        if pair is not None:
            break

    assert pair is not None, "Failed to find a visible test pair"
    area_t, area_ct = pair

    for i in range(10):
        ca = env._c_env.game.agents[i]
        ca.alive = 0
        ca.hp = 0

    t_centroid = nav_graph.centroids[area_t]
    ct_centroid = nav_graph.centroids[area_ct]

    t_agent = env._c_env.game.agents[0]
    ct_agent = env._c_env.game.agents[5]

    t_agent.alive = 1
    t_agent.hp = 100
    t_agent.area_idx = id2idx[area_t]
    t_agent.x = float(t_centroid[0])
    t_agent.y = float(t_centroid[1])
    t_agent.z = 0.0

    ct_agent.alive = 1
    ct_agent.hp = 1                    # low HP so any hit kills
    ct_agent.armor = 0
    ct_agent.area_idx = id2idx[area_ct]
    ct_agent.x = float(ct_centroid[0])
    ct_agent.y = float(ct_centroid[1])
    ct_agent.z = 0.0

    # Batch 3: aim is now a continuous head — facing is set directly on the
    # agent (above) and the discrete actions buffer no longer carries an
    # aim bin. SHOOT moved from index 2 to index 1 in the new enum
    # (move=0, shoot=1, reload=2, weapon=3, use=4, crouch=5, jump=6).
    t_facing = math.atan2(ct_agent.y - t_agent.y, ct_agent.x - t_agent.x)
    t_agent.facing = t_facing
    # Batch 3.5: 3D combat hit-test requires correct pitch in addition to yaw.
    # eye_z = t_agent.z + EYE_HEIGHT_STAND (64); torso_z = ct_agent.z + TORSO_OFFSET_STAND (32).
    # Both agents at z=0 → rz = 32 - 64 = -32; set pitch so the aim ray hits
    # the target torso exactly (perp=0). Without this, pitch=0 (horizontal) would
    # miss because the torso is 32u below eye height — expected 3D behavior change.
    rx_3d = ct_agent.x - t_agent.x
    ry_3d = ct_agent.y - t_agent.y
    eye_z_t = t_agent.z + 64.0         # EYE_HEIGHT_STAND
    torso_z_ct = ct_agent.z + 32.0     # TORSO_OFFSET_STAND
    rz_3d = torso_z_ct - eye_z_t
    dist_2d_3d = math.sqrt(rx_3d * rx_3d + ry_3d * ry_3d)
    t_agent.pitch = math.atan2(rz_3d, dist_2d_3d)

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
    nav_graph = env.nav_graph
    adjacency = env.map_data.adjacency
    bombsites = _bombsite_areas(env)
    t_spawn_areas = env.map_data.t_spawn_areas

    for agent_idx in range(10):
        ca = env._c_env.game.agents[agent_idx]
        area_id = int(env.map_data.area_ids[ca.area_idx])
        start_x, start_y = float(ca.x), float(ca.y)

        if agent_idx < TEAM_SIZE:
            goals = bombsites
        else:
            goals = set(t_spawn_areas)

        path = _bfs_area_path(nav_graph, adjacency, area_id, goals)
        assert len(path) >= 2, f"No route out of spawn for agent {agent_idx}"

        # Drive only to the first area beyond spawn — this test just wants
        # to confirm the agent can leave its starting area under the
        # current movement model, not traverse the full route.
        solo_env = make_env()
        solo_env.reset()
        _drive_agent_through_area_path(solo_env, agent_idx, path[:2], max_ticks_per_hop=64)

        moved = solo_env._c_env.game.agents[agent_idx]
        moved_area_id = int(solo_env.map_data.area_ids[moved.area_idx])
        assert moved_area_id != area_id or not np.allclose(
            [moved.x, moved.y], [start_x, start_y]), f"agent {agent_idx} failed to leave spawn"
        solo_env.close()

    env.close()
