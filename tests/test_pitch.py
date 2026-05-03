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


# ── T5 tests ──────────────────────────────────────────────────────────────


def _setup_3d_hit_scenario(env, sx, sy, sz, shooter_area_idx, tx, ty, tz, target_area_idx,
                           shooter_pitch):
    """Helper: place agent[0] (T) and agent[5] (CT) at given positions with
    correct yaw + given pitch, ready to shoot.

    Args:
        env: Cs2Env instance (already reset).
        sx, sy, sz: shooter position in world units.
        shooter_area_idx: nav area_idx for shooter position. MUST match (sx, sy, sz)
            so that the vis_matrix check (build_vis_matrix uses area_idx, NOT x/y)
            gates correctly. Pitfall: env.reset() places agents at spawn positions;
            if we only set x/y/z without updating area_idx, the combat vis check
            will use the OLD area_idx from the spawn position. Always keep in sync.
        tx, ty, tz: target position in world units.
        target_area_idx: nav area_idx for target position (same sync requirement).
        shooter_pitch: aim pitch in radians.
    Resets fire_cd / reload_ticks / switch_ticks / is_crouching / is_airborne
    on both agents so a single shot can fire immediately. Sets target HP=100.
    """
    import math
    g = env._c_env.game
    g.agents[0].x, g.agents[0].y, g.agents[0].z = sx, sy, sz
    g.agents[0].area_idx = shooter_area_idx            # must match position for vis check
    g.agents[0].facing = math.atan2(ty - sy, tx - sx)
    g.agents[0].pitch = shooter_pitch
    g.agents[0].is_crouching = 0
    g.agents[0].is_airborne = 0
    g.agents[0].fire_cd = 0
    g.agents[0].reload_ticks = 0
    g.agents[0].switch_ticks = 0
    g.agents[5].x, g.agents[5].y, g.agents[5].z = tx, ty, tz
    g.agents[5].area_idx = target_area_idx             # must match position for vis check
    g.agents[5].is_crouching = 0
    g.agents[5].is_airborne = 0
    g.agents[5].hp = 100                               # need full hp so single shot is below kill threshold


def test_3d_hit_at_correct_pitch_elevated_target():
    """T5: shooter at z=0, target at z=64 (elevated), correct pitch → hit (HP drops).

    Map geometry (make_simple_map): area 5 (x=575, y=352, z=0) and area 6
    (x=960, y=304, z=64) are mutually visible. Using area centroids ensures
    vis_matrix[5][6] = True so the combat visibility gate passes.
    eye_z = 0 + 64 = 64; target torso_z = 64 + 32 = 96; Δz = 32;
    dist_2d ≈ 388 → required pitch = atan2(32, dist_2d) ≈ 0.082 rad ≈ 4.7°.
    Why this exists: validates that the 3D hit-test gates ON correct pitch alignment
    when there's a vertical offset. Without 3D geometry, this would either always
    hit (2D logic ignoring z) or always miss (broken implementation).
    Pitfall: area_idx MUST be set to match the new x/y/z — the combat code uses
    area_idx for vis_matrix lookup, not raw coordinates."""
    import math

    import _action_spec as spec
    from c_env.cs2_env import Cs2Env
    from map import make_simple_map
    env = Cs2Env(map_data=make_simple_map())
    try:
        env.reset(seed=42)
        # Area 5 (T-side, z=0) → area 6 (elevated z=64): vis[5][6]=True
        sx, sy, sz = 575.0, 352.0, 0.0                 # area 5 centroid
        tx, ty, tz = 960.0, 304.0, 64.0                # area 6 centroid
        rx, ry = tx - sx, ty - sy
        eye_z = sz + 64.0                              # EYE_HEIGHT_STAND
        torso_z = tz + 32.0                            # TORSO_OFFSET_STAND
        rz = torso_z - eye_z
        dist_2d = math.sqrt(rx * rx + ry * ry)
        pitch = math.atan2(rz, dist_2d)                # ~0.082 rad: correct 3D pitch
        _setup_3d_hit_scenario(env,
                               sx=sx,
                               sy=sy,
                               sz=sz,
                               shooter_area_idx=5,
                               tx=tx,
                               ty=ty,
                               tz=tz,
                               target_area_idx=6,
                               shooter_pitch=pitch)
        hp_before = env._c_env.game.agents[5].hp
        actions = np.zeros((10, spec.ACTION_DIM), dtype=np.int32)
        actions[0, 1] = 1                              # HEAD_SHOOT
        cont = np.zeros((10, spec.AIM_DIM), dtype=np.float32)
        env.step(actions, cont)
        hp_after = env._c_env.game.agents[5].hp
        assert hp_after < hp_before, f"shot didn't connect; hp {hp_before}→{hp_after}"
    finally:
        if hasattr(env, "close"):
            env.close()


def test_3d_miss_at_zero_pitch_elevated_target():
    """T5: same geometry as hit-test, BUT shooter pitch = 0 → miss (HP unchanged).

    Same area 5→6 geometry. When pitch=0 the aim ray is horizontal but the
    target torso is 32 units ABOVE eye level, so the perpendicular offset
    from the horizontal ray to the torso is 32u >> HIT_HALF_WIDTH=16 → miss.
    Pitfall: if the 3D hit-test reduced back to 2D-equivalent (e.g., dz term
    dropped, or perp computed without rz), this test would FAIL — the shot
    would land. This catches an implementation that "compiled but ignored z"."""
    import _action_spec as spec
    from c_env.cs2_env import Cs2Env
    from map import make_simple_map
    env = Cs2Env(map_data=make_simple_map())
    try:
        env.reset(seed=42)
        # Same area 5→6 geometry, but pitch=0 (wrong — horizontal, not upward)
        sx, sy, sz = 575.0, 352.0, 0.0
        tx, ty, tz = 960.0, 304.0, 64.0
        _setup_3d_hit_scenario(env,
                               sx=sx,
                               sy=sy,
                               sz=sz,
                               shooter_area_idx=5,
                               tx=tx,
                               ty=ty,
                               tz=tz,
                               target_area_idx=6,
                               shooter_pitch=0.0)      # WRONG pitch — horizontal
        hp_before = env._c_env.game.agents[5].hp
        actions = np.zeros((10, spec.ACTION_DIM), dtype=np.int32)
        actions[0, 1] = 1
        cont = np.zeros((10, spec.AIM_DIM), dtype=np.float32)
        env.step(actions, cont)
        hp_after = env._c_env.game.agents[5].hp
        assert hp_after == hp_before, (f"shot connected at wrong pitch; hp {hp_before}→{hp_after}")
    finally:
        if hasattr(env, "close"):
            env.close()


def test_3d_hit_pitch_down_from_catwalk():
    """T5: shooter on catwalk z=128 shooting DOWN at z=0 target — correct negative pitch.

    Map geometry: area 15 (x=995, y=136, z=128 — catwalk) is visible to area 5
    (x=575, y=352, z=0 — ground): vis[15][5]=True.
    shooter eye_z = 128 + 64 = 192; target torso_z = 0 + 32 = 32; Δz = -160.
    dist_2d ≈ 470 → pitch = atan2(-160, dist_2d) ≈ -0.327 rad ≈ -18.7°.
    Why this exists: validates the SYMMETRIC down-pitch case (the up-pitch case
    proved positive Δz works; this proves negative). Catches a sign-flip bug
    in the 3D direction vector."""
    import math

    import _action_spec as spec
    from c_env.cs2_env import Cs2Env
    from map import make_simple_map
    env = Cs2Env(map_data=make_simple_map())
    try:
        env.reset(seed=42)
        # Area 15 (catwalk z=128) → area 5 (ground z=0): vis[15][5]=True
        sx, sy, sz = 995.0, 136.0, 128.0               # area 15 centroid
        tx, ty, tz = 575.0, 352.0, 0.0                 # area 5 centroid
        rx, ry = tx - sx, ty - sy
        eye_z = sz + 64.0                              # EYE_HEIGHT_STAND
        torso_z = tz + 32.0                            # TORSO_OFFSET_STAND
        rz = torso_z - eye_z                           # -160
        dist_2d = math.sqrt(rx * rx + ry * ry)
        pitch = math.atan2(rz, dist_2d)                # ~-0.327 rad: downward pitch
        _setup_3d_hit_scenario(env,
                               sx=sx,
                               sy=sy,
                               sz=sz,
                               shooter_area_idx=15,
                               tx=tx,
                               ty=ty,
                               tz=tz,
                               target_area_idx=5,
                               shooter_pitch=pitch)
        hp_before = env._c_env.game.agents[5].hp
        actions = np.zeros((10, spec.ACTION_DIM), dtype=np.int32)
        actions[0, 1] = 1
        cont = np.zeros((10, spec.AIM_DIM), dtype=np.float32)
        env.step(actions, cont)
        hp_after = env._c_env.game.agents[5].hp
        assert hp_after < hp_before, (f"down-pitch shot didn't connect; hp {hp_before}→{hp_after}")
    finally:
        if hasattr(env, "close"):
            env.close()


def test_3d_perp_perfectly_aligned_no_nan():
    """T5: shooter aimed PERFECTLY at target (forward ≈ |r|) — no NaN in obs/perp.

    Why this exists: validates the cross-product magnitude form's numerical stability
    per spec L4 + Opus I1. The naive sqrt(|r|² - forward²) form has catastrophic
    cancellation when forward ≈ |r| (perfectly aligned shot) and would produce NaN
    or 0/0. The cross-product form |r × d| stays finite at perfect alignment.
    Pitfall: if NaN propagated, downstream obs computations would also NaN, so we
    assert no NaN in env.observations after the step.
    Uses area 5→6 (vis=True) with the exact 3D pitch so forward = dist_3d exactly."""
    import math

    import _action_spec as spec
    from c_env.cs2_env import Cs2Env
    from map import make_simple_map
    env = Cs2Env(map_data=make_simple_map())
    try:
        env.reset(seed=42)
        # Area 5→6: same geometry as the hit test — perfectly aimed
        sx, sy, sz = 575.0, 352.0, 0.0
        tx, ty, tz = 960.0, 304.0, 64.0
        rx, ry = tx - sx, ty - sy
        eye_z = sz + 64.0
        torso_z = tz + 32.0
        rz = torso_z - eye_z
        dist_2d = math.sqrt(rx * rx + ry * ry)
        pitch = math.atan2(rz, dist_2d)                # perfect 3D alignment → perp = 0
        _setup_3d_hit_scenario(env,
                               sx=sx,
                               sy=sy,
                               sz=sz,
                               shooter_area_idx=5,
                               tx=tx,
                               ty=ty,
                               tz=tz,
                               target_area_idx=6,
                               shooter_pitch=pitch)
        actions = np.zeros((10, spec.ACTION_DIM), dtype=np.int32)
        actions[0, 1] = 1
        cont = np.zeros((10, spec.AIM_DIM), dtype=np.float32)
        env.step(actions, cont)
                                                       # Validate no NaN propagated to obs vector.
        assert not np.any(np.isnan(env.observations)), "NaN in observations"
    finally:
        if hasattr(env, "close"):
            env.close()


def test_pitch_clamps_at_pi_over_2_down():
    """T2 reviewer follow-up (deferred to T5): pitch clamps at -π/2 (symmetric to up).

    The original test_pitch_clamps_at_pi_over_2_up (T2) covers +π/2 saturation.
    Symmetric coverage is needed — if the implementation accidentally used
    `-(float)M_PI/2` in BOTH bounds (typo), the upward test would still pass
    while the downward bound would silently be wrong. This test fires Δpitch=-100
    for 100 ticks and asserts saturation at -π/2 (looking straight down)."""
    import math

    from c_env.cs2_env import Cs2Env
    from map import make_simple_map
    env = Cs2Env(map_data=make_simple_map())
    try:
        env.reset(seed=42)
        actions, cont = _zero_actions()
        cont[0, 1] = -100.0                                            # massive negative Δpitch
        for _ in range(100):
            env.step(actions, cont)
        assert env._c_env.game.agents[0].pitch == pytest.approx(
            -math.pi / 2,
            abs=1e-5), (f"pitch saturated at {env._c_env.game.agents[0].pitch}, expected -π/2")
    finally:
        if hasattr(env, "close"):
            env.close()
