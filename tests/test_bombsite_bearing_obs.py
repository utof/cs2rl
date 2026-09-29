"""tests/test_bombsite_bearing_obs.py — Batch 6 Task 2.5 (spec R9/D4, task 016).

Goal-direction observation: obs[OBS_SELF_BASE+25..27] =
    [sin(rel_bearing), cos(rel_bearing), xy_dist / map_diag]
to the Euclidean-NEAREST bombsite area centroid, where
    rel_bearing = wrap_pi(atan2(site_y - y, site_x - x) - facing).

Sign convention (load-bearing for BC "turn toward the site"):
    rel_bearing > 0  ⇒ site is counter-clockwise (to the LEFT) of facing
    ⇒ sin slot > 0 means "positive Δyaw turns toward the site".

Encoding rationale: every other angle in the obs (yaw, pitch, teammate/enemy
bearings) is a sin/cos pair, so the goal bearing follows the same idiom
(3 slots, not a single normalized angle) — continuous at ±π, no wraparound
cliff when the site is directly behind.

Pitfalls covered here:
  * bearing is straight-line XY (may point through walls — BFS routing is the
    expert's job, the obs only supplies the goal direction);
  * distance normalizer is map_diag = hypot(x_range/2, y_range/2), the same
    half-diagonal used by the teammate/enemy dx/dy/dist slots;
  * nearest-site selection must be per-agent Euclidean (dust2 has 2 sites).
"""
import math

import numpy as np
import pytest

from cs2rl.spec.obs import OBS_BLOCKS

# Slot indices derived from the generated spec — never hardcoded, so this test
# keeps working if an earlier self-block slot is ever inserted.
_SELF_START, _SELF_STOP = OBS_BLOCKS["self"]
BEARING_SIN = _SELF_STOP - 3
BEARING_COS = _SELF_STOP - 2
SITE_DIST = _SELF_STOP - 1


def _zero_actions():
    """Zero discrete+continuous actions: dyaw=0 preserves poked facing,
    move=0 preserves poked position (velocity is 0 after reset)."""
    from cs2rl.env.nav import N_AGENTS
    from cs2rl.spec.action import ACTION_DIM, AIM_DIM
    return (np.zeros((N_AGENTS, ACTION_DIM),
                     dtype=np.int32), np.zeros((N_AGENTS, AIM_DIM), dtype=np.float32))


def _map_diag(md) -> float:
    """Python mirror of the C normalizer: xr = 1/inv_x_range = (x_max-x_min)/2."""
    return math.hypot((md.x_max - md.x_min) / 2.0, (md.y_max - md.y_min) / 2.0)


def _nearest_site_centroid(md, x: float, y: float):
    """Expected value oracle: Euclidean-nearest bombsite-area centroid."""
    site_idxs = np.flatnonzero(np.asarray(md.bombsite_by_idx))
    cents = np.asarray(md.centroids, dtype=np.float64)[site_idxs]
    d2 = (cents[:, 0] - x)**2 + (cents[:, 1] - y)**2
    return cents[int(np.argmin(d2))]


def _obs_after_pose(env, x: float, y: float, facing: float):
    """Poke agent 0 to (x, y, facing), step with zero actions, return its obs row."""
    a = env._c_env.game.agents[0]
    a.x, a.y = x, y
    a.facing = facing
    env.step(*_zero_actions())
    return env.observations[0]


def test_bearing_facing_directly_at_site():
    """Agent due west of the (single) simple-map site, facing +X straight at it:
    rel_bearing = 0 → sin=0, cos=1; distance slot = 400/map_diag."""
    from cs2rl.c_env.cs2_env import Cs2Env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map
    md = make_simple_map()
    env = Cs2Env(config=EnvConfig(), map_data=md)
    try:
        env.reset(seed=42)
        sx, sy = _nearest_site_centroid(md, 0.0, 0.0)  # only one site on this map
        obs = _obs_after_pose(env, sx - 400.0, sy, 0.0)
        assert obs[BEARING_SIN] == pytest.approx(0.0, abs=1e-5)
        assert obs[BEARING_COS] == pytest.approx(1.0, abs=1e-5)
        assert obs[SITE_DIST] == pytest.approx(400.0 / _map_diag(md), abs=1e-5)
    finally:
        env.close()


def test_bearing_facing_directly_away_from_site():
    """Same pose but facing -X (away): rel = ±π → sin=0, cos=-1."""
    from cs2rl.c_env.cs2_env import Cs2Env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map
    md = make_simple_map()
    env = Cs2Env(config=EnvConfig(), map_data=md)
    try:
        env.reset(seed=42)
        sx, sy = _nearest_site_centroid(md, 0.0, 0.0)
        obs = _obs_after_pose(env, sx - 400.0, sy, math.pi)
        assert obs[BEARING_SIN] == pytest.approx(0.0, abs=1e-5)
        assert obs[BEARING_COS] == pytest.approx(-1.0, abs=1e-5)
    finally:
        env.close()


def test_bearing_sign_convention_site_to_the_left():
    """Site due east (+X), agent facing -Y (south, facing=-π/2): the site is
    90° counter-clockwise → rel = +π/2 → sin=+1. A policy that turns with
    positive Δyaw when sin>0 turns TOWARD the site — the BC-critical sign."""
    from cs2rl.c_env.cs2_env import Cs2Env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map
    md = make_simple_map()
    env = Cs2Env(config=EnvConfig(), map_data=md)
    try:
        env.reset(seed=42)
        sx, sy = _nearest_site_centroid(md, 0.0, 0.0)
        obs = _obs_after_pose(env, sx - 400.0, sy, -math.pi / 2)
        assert obs[BEARING_SIN] == pytest.approx(1.0, abs=1e-5)
        assert obs[BEARING_COS] == pytest.approx(0.0, abs=1e-5)
    finally:
        env.close()


def test_bearing_diagonal_offset_and_distance():
    """Agent offset both in x and y: full atan2 path (not axis-aligned) and
    the distance slot must equal hypot/map_diag exactly."""
    from cs2rl.c_env.cs2_env import Cs2Env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map
    md = make_simple_map()
    env = Cs2Env(config=EnvConfig(), map_data=md)
    try:
        env.reset(seed=42)
        sx, sy = _nearest_site_centroid(md, 0.0, 0.0)
        ax, ay, facing = sx - 300.0, sy - 100.0, 0.3
        obs = _obs_after_pose(env, ax, ay, facing)
        rel = math.atan2(sy - ay, sx - ax) - facing
        assert obs[BEARING_SIN] == pytest.approx(math.sin(rel), abs=1e-5)
        assert obs[BEARING_COS] == pytest.approx(math.cos(rel), abs=1e-5)
        expected_d = math.hypot(sx - ax, sy - ay) / _map_diag(md)
        assert obs[SITE_DIST] == pytest.approx(expected_d, abs=1e-5)
    finally:
        env.close()


def test_bearing_nearest_site_selection_dust2():
    """de_dust2 has TWO bombsites: every agent's slots must reflect the
    Euclidean-nearest one (per-agent selection, not a global site pick).
    Checked for all 10 agents at their natural spawn poses."""
    from cs2rl.c_env.cs2_env import make_env
    env = make_env(seed=7)             # bare make_env → real de_dust2
    try:
        env.reset(seed=7)
        env.step(*_zero_actions())
        md = env.map_data
        diag = _map_diag(md)
        for i in range(10):
            a = env._c_env.game.agents[i]
            sx, sy = _nearest_site_centroid(md, float(a.x), float(a.y))
            rel = math.atan2(sy - float(a.y), sx - float(a.x)) - float(a.facing)
            obs = env.observations[i]
            assert obs[BEARING_SIN] == pytest.approx(math.sin(rel), abs=1e-4), f"agent {i}"
            assert obs[BEARING_COS] == pytest.approx(math.cos(rel), abs=1e-4), f"agent {i}"
            expected_d = math.hypot(sx - float(a.x), sy - float(a.y)) / diag
            assert obs[SITE_DIST] == pytest.approx(expected_d, abs=1e-4), f"agent {i}"
    finally:
        env.close()
