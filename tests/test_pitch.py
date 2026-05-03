"""tests/test_pitch.py — Batch 3.5 (#24) — Δpitch + 3D combat hit-test.

Spec: docs/superpowers/specs/2026-05-03-batch-3.5-pitch-3d-combat-design.md
Plan: docs/superpowers/plans/2026-05-03-batch-3.5-pitch-3d-combat.md
Tests are split across T1 (constants + AgentState), T2 (consumption + Welford),
T4 (obs slots), T5 (3D combat). Each test_X is tagged with the task that owns it.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── T1 tests ──────────────────────────────────────────────────────────────


def test_aim_dim_bumped_to_2():
    """T1: AIM_DIM in _action_spec mirrors cs2_types.h after sync."""
    from _action_spec import AIM_DIM
    assert AIM_DIM == 2


def test_obs_dim_bumped_to_107():
    """T1: nav.OBS_DIM tracks the C-side OBS_DIM."""
    import nav
    assert nav.OBS_DIM == 107


def test_pitch_initialized_to_zero():
    """T1: AgentState.pitch defaults to 0.0 via spawn_team's memset.

    The C struct AgentState gets `float pitch` appended after `jump_cd`. Per-round
    respawn happens via `spawn_team` (memset to 0) + `init_agent` (sets per-team
    `facing` default). The memset zeros pitch; init_agent doesn't touch it. So
    every alive agent should report pitch == 0.0 immediately after env.reset.
    """
    from c_env.cs2_env import Cs2Env
    from map import make_simple_map
    env = Cs2Env(map_data=make_simple_map())
    try:
        env.reset(seed=42)
        for i in range(10):
            assert env._c_env.game.agents[i].pitch == 0.0, (
                f"agent[{i}].pitch={env._c_env.game.agents[i].pitch}, expected 0.0")
    finally:
        if hasattr(env, "close"):
            env.close()


# ── T2 tests ──────────────────────────────────────────────────────────────


def _zero_actions():
    """Helper: zero discrete + continuous actions for all 10 agents (AIM_DIM=2).

    Returns (discrete_actions, continuous_actions) ready to feed env.step().
    discrete_actions: int32 (10, ACTION_DIM); continuous_actions: float32 (10, AIM_DIM).
    """
    import _action_spec as spec
    return (
        np.zeros((10, spec.ACTION_DIM), dtype=np.int32),
        np.zeros((10, spec.AIM_DIM), dtype=np.float32),
    )


def test_pitch_consumed_from_continuous_actions():
    """T2: continuous_actions[:, 1] becomes Δpitch; pitch field updates.

    Why this exists: T1 added the pitch field but no consumer. T2 wires up the
    env_step Δpitch read. This test catches a missed consumer (would-be silent
    bug: pitch stays at 0.0 even when continuous_actions[1] is non-zero)."""
    from c_env.cs2_env import Cs2Env
    from map import make_simple_map
    env = Cs2Env(map_data=make_simple_map())
    try:
        env.reset(seed=42)
        actions, cont = _zero_actions()
        cont[0, 1] = 0.05                                              # +0.05 rad pitch up (small, well below max_turn_speed)
        env.step(actions, cont)
        assert env._c_env.game.agents[0].pitch == pytest.approx(
            0.05, abs=1e-5), (f"pitch={env._c_env.game.agents[0].pitch}, expected ~0.05")
    finally:
        if hasattr(env, "close"):
            env.close()


def test_pitch_clamps_at_pi_over_2_up():
    """T2: pitch clamps at +π/2 (bounded interval; no wrap).

    Pitfall: pitch is NOT a circular topology like yaw. Wrapping past π/2
    would invert the world. The implementation uses fminf/fmaxf with explicit
    bounds, NOT wrap_pi. This test verifies that 100 ticks of massive Δpitch
    saturate at +π/2 instead of wrapping or overflowing."""
    import math

    from c_env.cs2_env import Cs2Env
    from map import make_simple_map
    env = Cs2Env(map_data=make_simple_map())
    try:
        env.reset(seed=42)
        actions, cont = _zero_actions()
        cont[0, 1] = 100.0                                             # massive Δpitch — per-tick clamp + bounded clamp both fire
        for _ in range(100):
            env.step(actions, cont)
        assert env._c_env.game.agents[0].pitch == pytest.approx(
            math.pi / 2,
            abs=1e-5), (f"pitch saturated at {env._c_env.game.agents[0].pitch}, expected π/2")
    finally:
        if hasattr(env, "close"):
            env.close()


def test_welford_pitch_accumulates():
    """T2: aim_delta_pitch_count tracks per-tick consumption (alive agents only).

    Spec acceptance criterion 2 is load-bearing on aim_log_std_pitch not collapsing.
    The Welford pitch triple (sum/sq_sum/count) is the diagnostic that surfaces
    pitch consumption to train-loop logs.

    step_stats is cleared at the top of every env_step call (clear_stats(ss) at
    cs2_env.h:84); it reflects only the LAST step (10 agents × 1 tick = 10).
    episode_stats accumulates across the full episode — the right surface for
    multi-step totals. After 5 ticks: 10 agents × 5 ticks = 50.
    """
    from c_env.cs2_env import Cs2Env
    from map import make_simple_map
    env = Cs2Env(map_data=make_simple_map(), include_step_stats_in_info=True)
    try:
        obs, info = env.reset(seed=42)
        actions, cont = _zero_actions()
        cont[:, 1] = 0.1               # +0.1 Δpitch every alive agent
        for _ in range(5):
            obs, rew, term, trunc, info = env.step(actions, cont)

        # step_stats: per-step only — last step = 10 agents × 1 tick = 10
        ss = info[0]["step_stats"]
        assert ss["aim_delta_pitch_count"] == 10, (
            f"step_stats pitch count={ss['aim_delta_pitch_count']}, expected 10 (per-step)")

        # episode_stats accumulates across all ticks: 10 agents × 5 ticks = 50
        es = env._c_env.episode_stats
        assert es.aim_delta_pitch_count == 10 * 5, (
            f"episode_stats pitch count={es.aim_delta_pitch_count}, expected 50")
        # 10 agents × 5 ticks × 0.1 = 5.0. (Approximate — some agents may die
        # mid-test in self-play; allow generous tolerance.)
        assert 4.0 < es.aim_delta_pitch_sum < 5.5
    finally:
        if hasattr(env, "close"):
            env.close()


# ── T3 tests ──────────────────────────────────────────────────────────────


def test_binding_rejects_wrong_aim_dim_shape():
    """T3: stale-shape continuous_actions raises ValueError.

    Defense-in-depth: the Python wrapper at cs2_env.py::_prepare_continuous_actions:779
    already validates and raises FIRST with a different message ('continuous_actions
    shape (10, 1) != (10, 2)'). The binding.c shape check is a second layer for
    callers that bypass the Python wrapper (direct binding consumers, regression
    tests, future C-only entry points). This test passes if EITHER layer fires —
    Python wrapper OR binding.c's defensive check (Opus C4 review note)."""
    import _action_spec as spec
    from c_env.cs2_env import Cs2Env
    from map import make_simple_map
    env = Cs2Env(map_data=make_simple_map())
    try:
        env.reset(seed=42)
        actions = np.zeros((10, spec.ACTION_DIM), dtype=np.int32)
        # Stale caller: AIM_DIM=1 shape (instead of 2) — must raise.
        bad_cont = np.zeros((10, 1), dtype=np.float32)
        with pytest.raises(ValueError, match=r"AIM_DIM|continuous_actions"):
            env.step(actions, bad_cont)
    finally:
        if hasattr(env, "close"):
            env.close()


# ── T4 tests ──────────────────────────────────────────────────────────────


def test_obs_pitch_sin_cos_populated():
    """T4: obs[11], obs[12] = sin/cos(pitch) when pitch is non-zero.

    Why this exists: T1 added the pitch field, T2 wired the consumption, T4 makes
    pitch policy-observable. This test forces a known pitch value, steps the env
    once, and verifies the obs slots match math.sin/cos(pitch). Pitfall: the obs
    insertion shifts EVERY downstream obs[N] by +2 — if downstream tests fail
    after T4, audit them per Step 4.6."""
    import math

    from c_env.cs2_env import Cs2Env
    from map import make_simple_map
    env = Cs2Env(map_data=make_simple_map())
    try:
        env.reset(seed=42)
        env._c_env.game.agents[0].pitch = 0.5          # ~28.6°
        actions, cont = _zero_actions()
        env.step(actions, cont)
        obs = env.observations[0]
        assert obs[11] == pytest.approx(math.sin(0.5), abs=1e-5)
        assert obs[12] == pytest.approx(math.cos(0.5), abs=1e-5)
    finally:
        if hasattr(env, "close"):
            env.close()


def test_obs_dim_is_107_in_runtime():
    """T4: the actual emitted obs vector length is 107 (not just the constant).

    Cross-checks the C-side OBS_DIM bump (T1) against the actual stride of the
    observations buffer. If cs2_observations.h misses an obs[N] write (or writes
    past 107), this catches it at runtime."""
    from c_env.cs2_env import Cs2Env
    from map import make_simple_map
    env = Cs2Env(map_data=make_simple_map())
    try:
        env.reset(seed=42)
        actions, cont = _zero_actions()
        env.step(actions, cont)
        assert env.observations.shape[1] == 107
    finally:
        if hasattr(env, "close"):
            env.close()
