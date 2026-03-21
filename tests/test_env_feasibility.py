# tests/test_env_feasibility.py
from collections import deque

import numpy as np

from c_env.wrapper import make_env
from nav import _DELTA_VECTORS, BOMB_PLANT_TIME, LASER_RANGE, TEAM_SIZE


def _bombsite_areas(env):
    return {
        int(aid) for i, aid in enumerate(env.map_data.area_ids) if env.map_data.bombsite_by_idx[i]
    }


def _bfs_area_path(nav_graph, adjacency, start_area: int, goal_areas) -> list[int]:
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


def _step_state(nav_graph, adjacency, state, move: int):
    area_id, x, y = state
    delta = _DELTA_VECTORS[move]
    tx = x + float(delta[0])
    ty = y + float(delta[1])
    target_area, on_mesh = nav_graph.get_area_if_on_mesh((tx, ty))
    if not on_mesh:
        return None
    i = nav_graph._id_to_idx[area_id]
    j = nav_graph._id_to_idx[target_area]
    if not adjacency[i, j]:
        return None
    return target_area, round(tx, 3), round(ty, 3)


def _plan_transition(nav_graph, adjacency, start_state, target_area: int, max_depth: int = 32):
    q = deque([(start_state, [])])
    seen = {start_state}
    while q:
        state, path = q.popleft()
        if state[0] == target_area:
            return path, state
        if len(path) >= max_depth:
            continue
        for move in range(1, 9):
            nxt = _step_state(nav_graph, adjacency, state, move)
            if nxt is None or nxt in seen:
                continue
            seen.add(nxt)
            q.append((nxt, path + [move]))
    return None, None


def _plan_route_to_bombsite(env, bomber_idx: int):
    nav_graph = env.nav_graph
    adjacency = env.map_data.adjacency
    bombsites = _bombsite_areas(env)
    ca = env._c_env.game.agents[bomber_idx]
    area_id = int(env.map_data.area_ids[ca.area_idx])
    start_state = (area_id, round(float(ca.x), 3), round(float(ca.y), 3))

    area_path = _bfs_area_path(nav_graph, adjacency, area_id, bombsites)
    assert area_path, f"No executable area path from spawn area {area_id}"

    cur_state = start_state
    all_moves = []
    for target_area in area_path[1:]:
        moves, cur_state = _plan_transition(nav_graph, adjacency, cur_state, target_area)
        assert moves is not None, f"Failed local transition {cur_state[0]} -> {target_area}"
        all_moves.extend(moves)

    return all_moves, area_path[-1]


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
    env = make_env()
    env.reset()
    bombsites = _bombsite_areas(env)

    bomber_idx = 4
    for i in range(10):
        env._c_env.game.agents[i].has_bomb = 0
    env._c_env.game.agents[bomber_idx].has_bomb = 1
    env._c_env.game.bomb_carrier_id = bomber_idx

    moves, final_site_area = _plan_route_to_bombsite(env, bomber_idx)
    assert len(moves) < env._c_env.game.round_ticks_left, "Route exceeds round budget"

    for move in moves:
        actions = np.zeros((10, 4), dtype=np.int64)
        actions[bomber_idx, 0] = move
        env.step(actions)

    bomber = env._c_env.game.agents[bomber_idx]
    assert int(env.map_data.area_ids[bomber.area_idx]) == final_site_area
    assert int(env.map_data.area_ids[bomber.area_idx]) in bombsites

    for _ in range(BOMB_PLANT_TIME):
        actions = np.zeros((10, 4), dtype=np.int64)
        actions[bomber_idx, 2] = 1
        env.step(actions)
        if env._c_env.game.bomb_planted:
            break

    assert env._c_env.game.bomb_planted, "Bomber failed to plant after reaching bombsite"
    env.close()


def test_controlled_visible_agents_can_kill():
    import math

    env = make_env(auto_reset=False)
    env.reset()
    nav_graph = env.nav_graph
    id2idx = {int(aid): i for i, aid in enumerate(env.map_data.area_ids)}

    pair = None
    area_ids = nav_graph.area_ids
    for i, area_i in enumerate(area_ids[:400]):
        for area_j in area_ids[i + 1 : i + 200]:
            if not env.map_data.vis_matrix[id2idx[area_i], id2idx[area_j]]:
                continue
            dx = nav_graph.centroids[area_j][0] - nav_graph.centroids[area_i][0]
            dy = nav_graph.centroids[area_j][1] - nav_graph.centroids[area_i][1]
            dist = float((dx * dx + dy * dy) ** 0.5)
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
    ct_agent.hp = 100
    ct_agent.area_idx = id2idx[area_ct]
    ct_agent.x = float(ct_centroid[0])
    ct_agent.y = float(ct_centroid[1])
    ct_agent.z = 0.0

    t_agent.facing = math.atan2(ct_agent.y - t_agent.y, ct_agent.x - t_agent.x)
    ct_agent.facing = math.atan2(t_agent.y - ct_agent.y, t_agent.x - ct_agent.x)

    actions = np.zeros((10, 4), dtype=np.int64)
    actions[0, 1] = 1  # t0 shoots
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

        start_state = (area_id, round(start_x, 3), round(start_y, 3))
        moves, _ = _plan_transition(nav_graph, adjacency, start_state, path[1], max_depth=16)
        assert moves, f"No local exit plan from spawn for agent {agent_idx}"

        # Both make_env() calls default to seed=0, so agents start at identical positions.
        solo_env = make_env()
        solo_env.reset()
        for move in moves:
            actions = np.zeros((10, 4), dtype=np.int64)
            actions[agent_idx, 0] = move
            solo_env.step(actions)

        moved = solo_env._c_env.game.agents[agent_idx]
        moved_area_id = int(solo_env.map_data.area_ids[moved.area_idx])
        assert moved_area_id != area_id or not np.allclose(
            [moved.x, moved.y], [start_x, start_y]
        ), f"agent {agent_idx} failed to leave spawn"
        solo_env.close()

    env.close()
