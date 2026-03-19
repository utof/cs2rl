from collections import deque
import math

import numpy as np

from sim import BOMB_PLANT_TIME, Dust2Env, LASER_RANGE, TEAM_SIZE, _DELTA_VECTORS


def _bfs_area_path(env: Dust2Env, start_area: int, goal_areas) -> list[int]:
    goal_areas = set(goal_areas)
    id_to_idx = env.nav_graph._id_to_idx
    q = deque([start_area])
    prev = {start_area: None}
    found = None

    while q:
        cur = q.popleft()
        if cur in goal_areas:
            found = cur
            break

        cur_idx = id_to_idx[cur]
        for nbr_idx in np.flatnonzero(env._area_adjacency[cur_idx]):
            nbr_area = env.nav_graph.area_ids[int(nbr_idx)]
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


def _step_state(env: Dust2Env, state, move: int):
    area_id, x, y = state
    delta = _DELTA_VECTORS[move]
    tx = x + float(delta[0])
    ty = y + float(delta[1])
    target_area, on_mesh = env.nav_graph.get_area_if_on_mesh((tx, ty))
    if not on_mesh:
        return None

    i = env.nav_graph._id_to_idx[area_id]
    j = env.nav_graph._id_to_idx[target_area]
    if not env._area_adjacency[i, j]:
        return None

    return target_area, round(tx, 3), round(ty, 3)


def _plan_transition(env: Dust2Env, start_state, target_area: int, max_depth: int = 32):
    q = deque([(start_state, [])])
    seen = {start_state}

    while q:
        state, path = q.popleft()
        if state[0] == target_area:
            return path, state
        if len(path) >= max_depth:
            continue

        for move in range(1, 9):
            nxt = _step_state(env, state, move)
            if nxt is None or nxt in seen:
                continue
            seen.add(nxt)
            q.append((nxt, path + [move]))

    return None, None


def _plan_route_to_bombsite(env: Dust2Env, bomber_idx: int):
    bomber = env.state.agents[bomber_idx]
    start_state = (
        bomber.area_id,
        round(float(bomber.pos[0]), 3),
        round(float(bomber.pos[1]), 3),
    )
    area_path = _bfs_area_path(env, bomber.area_id, env.bombsite_areas)
    assert area_path, f"No executable area path from spawn area {bomber.area_id}"

    cur_state = start_state
    all_moves = []
    for target_area in area_path[1:]:
        moves, cur_state = _plan_transition(env, cur_state, target_area)
        assert moves is not None, f"Failed local transition {cur_state[0]} -> {target_area}"
        all_moves.extend(moves)

    return all_moves, area_path[-1]


def test_spawn_areas_are_distinct_and_site_reachable():
    env = Dust2Env()
    env.reset(seed=42)

    assert len(env.t_spawn_areas) == TEAM_SIZE
    assert len(env.ct_spawn_areas) == TEAM_SIZE
    assert len(set(env.t_spawn_areas)) == TEAM_SIZE
    assert len(set(env.ct_spawn_areas)) == TEAM_SIZE

    for spawn_area in env.t_spawn_areas:
        path = _bfs_area_path(env, spawn_area, env.bombsite_areas)
        assert path, f"T spawn area {spawn_area} cannot reach any bombsite"


def test_scripted_bomber_can_reach_site_and_plant():
    env = Dust2Env()
    env.reset(seed=42)

    for agent in env.state.agents:
        agent.has_bomb = False

    bomber_idx = 4
    env.state.bomb_carrier_id = bomber_idx
    env.state.agents[bomber_idx].has_bomb = True

    moves, final_site_area = _plan_route_to_bombsite(env, bomber_idx)
    assert len(moves) < env.state.round_ticks_left, "Route exceeds round budget"

    for move in moves:
        actions = {aid: np.array([0, 0, 0, 0], dtype=np.int64) for aid in env.agents}
        actions[f"t{bomber_idx}"][0] = move
        env.step(actions)

    assert env.state.agents[bomber_idx].area_id == final_site_area
    assert env.state.agents[bomber_idx].area_id in env.bombsite_areas

    for _ in range(BOMB_PLANT_TIME):
        actions = {aid: np.array([0, 0, 0, 0], dtype=np.int64) for aid in env.agents}
        actions[f"t{bomber_idx}"][2] = 1
        _, _, _, _, infos = env.step(actions)
        if env.state.bomb_planted:
            break

    assert env.state.bomb_planted, "Bomber failed to plant after reaching bombsite"
    assert infos[f"t{bomber_idx}"]["bomb_planted"] == 1


def test_controlled_visible_agents_can_kill():
    env = Dust2Env()
    env.reset(seed=1)

    pair = None
    for i, area_i in enumerate(env.nav_graph.area_ids[:400]):
        for area_j in env.nav_graph.area_ids[i + 1 : i + 200]:
            if not env.nav_graph.can_see(area_i, area_j):
                continue
            dx = env.nav_graph.centroids[area_j][0] - env.nav_graph.centroids[area_i][0]
            dy = env.nav_graph.centroids[area_j][1] - env.nav_graph.centroids[area_i][1]
            dist = float((dx * dx + dy * dy) ** 0.5)
            if 50 < dist < LASER_RANGE * 0.5:
                pair = (area_i, area_j)
                break
        if pair is not None:
            break

    assert pair is not None, "Failed to find a visible test pair"
    area_t, area_ct = pair

    for agent in env.state.agents:
        agent.alive = False
        agent.hp = 0

    t_agent = env.state.agents[0]
    ct_agent = env.state.agents[5]
    t_centroid = env.nav_graph.areas[area_t].centroid
    ct_centroid = env.nav_graph.areas[area_ct].centroid

    t_agent.alive = True
    t_agent.hp = 100
    t_agent.area_id = area_t
    t_agent.pos[:] = (t_centroid.x, t_centroid.y, t_centroid.z)
    ct_agent.alive = True
    ct_agent.hp = 100
    ct_agent.area_id = area_ct
    ct_agent.pos[:] = (ct_centroid.x, ct_centroid.y, ct_centroid.z)

    t_agent.facing = math.atan2(ct_agent.pos[1] - t_agent.pos[1], ct_agent.pos[0] - t_agent.pos[0])
    ct_agent.facing = math.atan2(t_agent.pos[1] - ct_agent.pos[1], t_agent.pos[0] - ct_agent.pos[0])

    actions = {
        "t0": np.array([0, 1, 0, 0], dtype=np.int64),
        "ct0": np.array([0, 0, 0, 0], dtype=np.int64),
    }
    _, rewards, _, _, infos = env.step(actions)

    assert not env.state.agents[5].alive
    assert infos["t0"]["kills_t"] == 1
    assert rewards["t0"] > 0


def test_fixed_seed_agents_can_leave_spawn():
    env = Dust2Env()
    env.reset(seed=42)

    for agent_idx, agent in enumerate(env.state.agents):
        if agent.team == 0:
            goals = env.bombsite_areas
            aid = f"t{agent_idx}"
        else:
            goals = env.t_spawn_areas
            aid = f"ct{agent_idx - TEAM_SIZE}"

        path = _bfs_area_path(env, agent.area_id, goals)
        assert len(path) >= 2, f"No route out of spawn for {aid}"

        start_area = agent.area_id
        start_pos = agent.pos.copy()
        state = (start_area, round(float(start_pos[0]), 3), round(float(start_pos[1]), 3))
        moves, _ = _plan_transition(env, state, path[1], max_depth=16)
        assert moves, f"No local exit plan from spawn for {aid}"

        solo_env = Dust2Env()
        solo_env.reset(seed=42)
        for move in moves:
            actions = {name: np.array([0, 0, 0, 0], dtype=np.int64) for name in solo_env.agents}
            actions[aid][0] = move
            solo_env.step(actions)

        moved_agent = solo_env.state.agents[agent_idx]
        assert moved_agent.area_id != start_area or not np.allclose(
            moved_agent.pos[:2], start_pos[:2]
        ), f"{aid} failed to leave spawn"
