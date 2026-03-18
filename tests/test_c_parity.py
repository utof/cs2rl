"""Parity test: C env must produce same rewards and terminals as Python env."""
import numpy as np
import pytest
from sim import Dust2Env, NAV_PATH, CACHE_PATH, INVALID_AREA_ID, STALE_MEMORY_TICK, TEAM_SIZE


def _make_c_env():
    from c_env.wrapper import make_env
    return make_env(seed=42)


def _copy_py_state_to_c(py_env, c_env):
    """Set C env game state to match Python env state exactly.

    sim.py key facts:
    - s.agents[0..4] are T agents (team=0, agent_id=0..4)
    - s.agents[5..9] are CT agents (team=1, agent_id=5..9)
    - EnemyMemoryStore has .area[slot] and .tick[slot] arrays
      - For T (team=0): slot = enemy_id - TEAM_SIZE  (enemies are CTs, ids 5-9)
      - For CT (team=1): slot = enemy_id             (enemies are Ts, ids 0-4)
    - GameStateC.bomb_area_idx is an area index (not raw area_id)
    - GameState.bomb_area_id is a raw area_id (-1 if not planted)
    """
    nav = py_env.nav_graph
    id2idx = nav._id_to_idx
    g = c_env._c_env.game

    s = py_env.state
    g.tick             = s.tick
    g.round_ticks_left = s.round_ticks_left
    g.bomb_planted     = int(s.bomb_planted)
    g.round_over       = int(s.round_over)
    g.winner           = s.winner
    g.bomb_carrier_id  = s.bomb_carrier_id
    # bomb_area_id in Python is a raw area_id; C env stores bomb_area_idx (index)
    g.bomb_area_idx    = id2idx[s.bomb_area_id] if s.bomb_area_id != INVALID_AREA_ID else -1
    g.bomb_x           = float(s.bomb_pos[0])
    g.bomb_y           = float(s.bomb_pos[1])
    g.bomb_z           = float(s.bomb_pos[2])
    g.bomb_ticks_left          = s.bomb_ticks_left
    g.bomb_being_planted_by    = s.bomb_being_planted_by
    g.bomb_plant_ticks         = s.bomb_plant_ticks
    g.bomb_being_defused_by    = s.bomb_being_defused_by
    g.bomb_defuse_ticks        = s.bomb_defuse_ticks

    for i, agent in enumerate(s.agents):
        ca = g.agents[i]
        ca.x        = float(agent.pos[0])
        ca.y        = float(agent.pos[1])
        ca.z        = float(agent.pos[2])
        ca.area_idx = id2idx.get(agent.area_id, -1)
        ca.facing   = float(agent.facing)
        ca.hp       = agent.hp
        ca.shoot_cd = agent.shoot_cd
        ca.alive    = int(agent.alive)
        ca.has_bomb = int(agent.has_bomb)
        ca.has_kit  = int(agent.has_kit)
        ca.team     = agent.team

        # EnemyMemoryStore: for T (team=0) enemies have ids 5..9, slot = enemy_id - TEAM_SIZE
        #                   for CT (team=1) enemies have ids 0..4, slot = enemy_id
        mem = agent.enemy_memory  # EnemyMemoryStore instance
        for slot in range(TEAM_SIZE):
            if agent.team == 0:
                enemy_id = TEAM_SIZE + slot   # CT enemy agent ids: 5,6,7,8,9
            else:
                enemy_id = slot               # T enemy agent ids: 0,1,2,3,4

            result = mem.get(enemy_id)
            if result is None:
                ca.enemy_mem_idx[slot]  = -1
                ca.enemy_mem_tick[slot] = STALE_MEMORY_TICK
            else:
                area_id_mem, tick_mem = result
                ca.enemy_mem_idx[slot]  = id2idx.get(area_id_mem, -1)
                ca.enemy_mem_tick[slot] = tick_mem


FIXED_ACTIONS = np.array([
    [1, 0, 0, 0],  # t0 move dir 1
    [3, 0, 0, 0],  # t1 move dir 3
    [0, 0, 0, 0],  # t2 noop
    [5, 0, 0, 0],  # t3 move dir 5
    [0, 0, 0, 0],  # t4 noop
    [7, 0, 0, 0],  # ct0 move dir 7
    [0, 0, 0, 0],  # ct1 noop
    [1, 0, 0, 0],  # ct2 move dir 1
    [0, 0, 0, 0],  # ct3 noop
    [0, 0, 0, 0],  # ct4 noop
], dtype=np.int32)


def test_rewards_match():
    py_env = Dust2Env()
    py_env.reset(seed=42)
    c_env = _make_c_env()
    c_env.reset()
    _copy_py_state_to_c(py_env, c_env)

    for step in range(1000):
        py_actions = {
            aid: FIXED_ACTIONS[i]
            for i, aid in enumerate(py_env.possible_agents)
            if aid in py_env.agents
        }
        c_actions = FIXED_ACTIONS.copy()

        _, py_rew, py_term, _, _ = py_env.step(py_actions)
        _, c_rew, c_term, _, _  = c_env.step(c_actions)

        # Build per-agent arrays ordered by possible_agents
        py_rew_arr = np.array(
            [py_rew.get(aid, 0.0) for aid in py_env.possible_agents],
            dtype=np.float32
        )
        assert np.allclose(py_rew_arr, c_rew, atol=1e-4), (
            f"Step {step}: rewards diverged\nPython: {py_rew_arr}\nC:      {np.array(c_rew)}"
        )

        py_term_arr = [py_term.get(aid, True) for aid in py_env.possible_agents]
        c_term_bool = [bool(t) for t in c_term]
        assert c_term_bool == [bool(t) for t in py_term_arr], (
            f"Step {step}: terminals diverged\nPython: {py_term_arr}\nC:      {c_term_bool}"
        )

        if all(py_term_arr):
            py_env.reset()
            c_env.reset()
            _copy_py_state_to_c(py_env, c_env)
