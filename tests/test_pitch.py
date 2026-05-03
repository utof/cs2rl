"""tests/test_pitch.py — Batch 3.5 (#24) — Δpitch + 3D combat hit-test.

Spec: docs/superpowers/specs/2026-05-03-batch-3.5-pitch-3d-combat-design.md
Plan: docs/superpowers/plans/2026-05-03-batch-3.5-pitch-3d-combat.md
Tests are split across T1 (constants + AgentState), T2 (consumption + Welford),
T4 (obs slots), T5 (3D combat). Each test_X is tagged with the task that owns it.
"""
import sys
from pathlib import Path

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
