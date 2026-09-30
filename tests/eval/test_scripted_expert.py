# tests/test_scripted_expert.py
"""Regression tests for the extracted scripted bomber expert (Batch 6 Task 2).

Contract under test (spec D-3/D-5 + Gate 0 measurement,
`git show 9b9bf2f:scripts/measure_budget.py`):
on the SIMPLE map the expert must reach the bombsite and plant well inside the
real round timer (ROUND_TIME = 640 ticks; Gate 0 measured 54-93 ticks across
all 5 T-spawns) using ONLY the real action interface — no facing pokes.

These tests also pin the Task-3 (demo generation) contract: every tick the
expert yields the exact (discrete, continuous) arrays it will execute, with
the shapes/dtypes env.step consumes, so a recorder can capture
(obs_t, discrete_t, continuous_t) triples verbatim.
"""
import numpy as np

from cs2rl.env.c.cs2_env import make_env
from cs2rl.env.map import make_simple_map
from cs2rl.env.nav import N_AGENTS, ROUND_TIME
from cs2rl.eval.scripted_expert import (
    ScriptedBomber,
    bfs_area_path,
    bombsite_areas,
    setup_bomb_carrier,
)
from cs2rl.spec.action import ACTION_DIM, AIM_DIM


def test_scripted_bomber_plants_within_round_time_on_simple_map():
    """Representative spawn (seed 0, agent 0): walk to the site and plant
    within the REAL round budget, via env.step(discrete, continuous) only."""
    env = make_env(seed=0, map_data=make_simple_map(), auto_reset=False)
    env.reset()
    setup_bomb_carrier(env, bomber_idx=0)

    bomber = ScriptedBomber(env, bomber_idx=0)
    assert bomber.path, "BFS found no spawn->bombsite path on the simple map"

    recorded = []
    for disc, cont in bomber.run():
        # Task-3 contract: arrays are exactly what env.step will execute.
        assert disc.shape == (N_AGENTS, ACTION_DIM)
        assert disc.dtype == np.int32
        assert cont.shape == (N_AGENTS, AIM_DIM)
        assert cont.dtype == np.float32
        # D-5: pitch label is absolute 0 (level) — never steered by the expert.
        assert float(cont[0, 1]) == 0.0
        recorded.append((disc.copy(), cont.copy()))

    assert bomber.reached_site, "expert never entered the bombsite area"
    assert bomber.planted, "expert reached the site but failed to plant"
    assert bomber.ticks <= ROUND_TIME
    assert len(recorded) == bomber.ticks
    assert bool(env._c_env.game.bomb_planted)

    # Recorded per-tick arrays stack into BC-dataset shape (T, N, dim).
    disc_stack = np.stack([d for d, _ in recorded])
    cont_stack = np.stack([c for _, c in recorded])
    assert disc_stack.shape == (bomber.ticks, N_AGENTS, ACTION_DIM)
    assert cont_stack.shape == (bomber.ticks, N_AGENTS, AIM_DIM)
    env.close()


def test_bfs_area_path_reaches_bombsite_on_simple_map():
    """Pathing helpers work straight off MapData (nav_graph is None here)."""
    env = make_env(seed=0, map_data=make_simple_map(), auto_reset=False)
    env.reset()
    sites = bombsite_areas(env.map_data)
    assert sites

    agent = env._c_env.game.agents[0]
    start_area = int(env.map_data.area_ids[agent.area_idx])
    path = bfs_area_path(env.map_data, start_area, sites)
    assert path
    assert path[0] == start_area
    assert path[-1] in sites
    env.close()
