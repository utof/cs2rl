"""tests/env/c/test_pitch.py — Batch 3.5 (#24) — Δpitch + 3D combat hit-test.

Historical design: Batch 3.5, gh#24; current contract:
docs/formats.md#observation-and-action
Tests are split across T1 (constants + AgentState), T2 (consumption + Welford),
T4 (obs slots), T5 (3D combat). Each test_X is tagged with the task that owns it.
"""

import numpy as np
import pytest

# ── T1 tests ──────────────────────────────────────────────────────────────


def test_aim_dim_bumped_to_2():
    """T1: AIM_DIM in spec.action mirrors cs2_types.h after sync."""
    from cs2rl.spec.action import AIM_DIM
    assert AIM_DIM == 2


def test_obs_dim_bumped_to_110():
    """T1 (re-pinned Batch 6 Task 2.5): nav.OBS_DIM tracks the C-side OBS_DIM."""
    from cs2rl.env import nav
    assert nav.OBS_DIM == 110


def test_pitch_initialized_to_zero():
    """T1: AgentState.pitch defaults to 0.0 via spawn_team's memset.

    The C struct AgentState gets `float pitch` appended after `jump_cd`. Per-round
    respawn happens via `spawn_team` (memset to 0) + `init_agent` (sets per-team
    `facing` default). The memset zeros pitch; init_agent doesn't touch it. So
    every alive agent should report pitch == 0.0 immediately after env.reset.
    """
    from cs2rl.env.c.cs2_env import Cs2Env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map
    env = Cs2Env(config=EnvConfig(), map_data=make_simple_map())
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
    from cs2rl.spec import action as spec
    return (
        np.zeros((10, spec.ACTION_DIM), dtype=np.int32),
        np.zeros((10, spec.AIM_DIM), dtype=np.float32),
    )


def test_pitch_consumed_from_continuous_actions():
    """T2 (v1c, gh #36 fix B-3): continuous_actions[:, 1] becomes ABSOLUTE pitch.

    v1a/v1b read this slot as a Δpitch (delta accumulator). v1c reads it as the
    absolute target — `a->pitch = clamp(value, ±π/2)` each tick. This test still
    passes under both semantics because we set 0.05 from a zero start and read
    0.05 back, but the meaning differs: under v1c, writing 0.05 puts pitch
    AT 0.05; under v1b it added 0.05 to whatever was there. The other T2 tests
    (clamp/Welford) cover the semantic difference.

    Why this exists: T1 added the pitch field but no consumer. T2 wires up the
    env_step pitch read. This test catches a missed consumer (would-be silent
    bug: pitch stays at 0.0 even when continuous_actions[1] is non-zero)."""
    from cs2rl.env.c.cs2_env import Cs2Env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map
    env = Cs2Env(config=EnvConfig(), map_data=make_simple_map())
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
    """T2 (v1c): pitch clamps at +π/2 (bounded interval; no wrap).

    Pitfall: pitch is NOT a circular topology like yaw. Wrapping past π/2
    would invert the world. The implementation uses fminf/fmaxf with explicit
    bounds, NOT wrap_pi. v1c: writing absolute=100.0 saturates instantly to
    +π/2 (single fminf hit). The 100-tick loop is preserved from v1b for
    parity but the saturation now happens on tick 1, not progressively."""
    import math

    from cs2rl.env.c.cs2_env import Cs2Env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map
    env = Cs2Env(config=EnvConfig(), map_data=make_simple_map())
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
    """T2 (v1c): aim_delta_pitch_count tracks per-tick consumption (alive agents only).

    Spec acceptance criterion 2 is load-bearing on aim_log_std_pitch not collapsing.
    The Welford pitch triple (sum/sq_sum/count) is the diagnostic that surfaces
    pitch consumption to train-loop logs.

    v1c semantic shift: the Welford fields keep the name `aim_delta_pitch_*`
    (ctypes mirror compatibility) but now record the APPLIED ABSOLUTE pitch,
    not the delta. mean = avg look-direction, σ = engagement-angle spread.
    For this test the numerical assertion is unchanged because writing a
    constant 0.1 absolute every tick produces the same per-tick contribution
    (0.1) as writing a delta of 0.1 once (also 0.1) under v1b.

    step_stats is cleared at the top of every env_step call (clear_stats(ss) at
    cs2_env.h:84); it reflects only the LAST step (10 agents × 1 tick = 10).
    episode_stats accumulates across the full episode — the right surface for
    multi-step totals. After 5 ticks: 10 agents × 5 ticks = 50.
    """
    from cs2rl.env.c.cs2_env import Cs2Env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map
    env = Cs2Env(config=EnvConfig(), map_data=make_simple_map(), include_step_stats_in_info=True)
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
    from cs2rl.env.c.cs2_env import Cs2Env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map
    from cs2rl.spec import action as spec
    env = Cs2Env(config=EnvConfig(), map_data=make_simple_map())
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
    """T4 (v1c): obs[11], obs[12] = sin/cos(pitch) when pitch is non-zero.

    Why this exists: T1 added the pitch field, T2 wired the consumption, T4 makes
    pitch policy-observable. This test forces a known pitch value, steps the env
    once, and verifies the obs slots match math.sin/cos(pitch). Pitfall: the obs
    insertion shifts EVERY downstream obs[N] by +2 — if downstream tests fail
    after T4, audit them per Step 4.6.

    v1c (gh #36 fix B-3): pitch is now ABSOLUTE — env_step does
    `a->pitch = clamp(continuous_actions[i*AIM_DIM+1], ±π/2)` BEFORE
    compute_observations, so pre-setting `agent.pitch` via ctypes WRITE is
    overwritten on every step. Set the desired pitch via cont[0, 1] instead."""
    import math

    from cs2rl.env.c.cs2_env import Cs2Env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map
    env = Cs2Env(config=EnvConfig(), map_data=make_simple_map())
    try:
        env.reset(seed=42)
        actions, cont = _zero_actions()
        cont[0, 1] = 0.5               # ~28.6° absolute pitch (v1c semantics)
        env.step(actions, cont)
        obs = env.observations[0]
        assert obs[11] == pytest.approx(math.sin(0.5), abs=1e-5)
        assert obs[12] == pytest.approx(math.cos(0.5), abs=1e-5)
    finally:
        if hasattr(env, "close"):
            env.close()


def test_obs_dim_is_110_in_runtime():
    """T4 (re-pinned Batch 6 Task 2.5): the actual emitted obs vector length is
    110 (not just the constant).

    Cross-checks the C-side OBS_DIM bump (T1) against the actual stride of the
    observations buffer. If cs2_observations.h misses an obs[N] write (or writes
    past 110), this catches it at runtime."""
    from cs2rl.env.c.cs2_env import Cs2Env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map
    env = Cs2Env(config=EnvConfig(), map_data=make_simple_map())
    try:
        env.reset(seed=42)
        actions, cont = _zero_actions()
        env.step(actions, cont)
        assert env.observations.shape[1] == 110
    finally:
        if hasattr(env, "close"):
            env.close()


# ── T5 tests ──────────────────────────────────────────────────────────────


def _setup_3d_hit_scenario(env, sx, sy, sz, shooter_area_idx, tx, ty, tz, target_area_idx):
    """Helper: place agent[0] (T) and agent[5] (CT) at given positions with
    correct yaw, ready to shoot.

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

    RAMP PITFALL: sz/tz are only honoured on FLAT areas. On a ramp area
    (is_ramp=1, e.g. 13/14/16 in make_simple_map) the ground-snap in
    cs2_movement.h::process_movement overwrites z with the interpolated surface
    z of the ramp quad on the very next env.step, before combat resolves. Never
    compute a pitch from an sz you passed here for a ramp — step once without
    firing and read the settled `agents[i].z` back instead. See
    test_3d_hit_pitch_down_from_ramp.

    Resets fire_cd / reload_ticks / switch_ticks / is_crouching / is_airborne
    on both agents so a single shot can fire immediately. Sets target HP=100.

    PITCH NOTE (v1c, gh #36 fix B-3): pitch is ABSOLUTE per env_step. This helper
    no longer accepts `shooter_pitch` because writing `g.agents[0].pitch = X`
    here would be overwritten by `a->pitch = clamp(cont[i*AIM_DIM+1], ±π/2)` on
    the very next env.step. Callers must set pitch via the continuous_actions
    buffer they pass to env.step (e.g., `cont[0, 1] = pitch`).
    """
    import math
    g = env._c_env.game
    g.agents[0].x, g.agents[0].y, g.agents[0].z = sx, sy, sz
    g.agents[0].area_idx = shooter_area_idx            # must match position for vis check
    g.agents[0].facing = math.atan2(ty - sy, tx - sx)
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
    v1b geometry (gh #36 fix A): eye_z = 0 + 48 = 48; torso_z = 64 + 48 = 112;
    Δz = 64; dist_2d ≈ 388 → required pitch = atan2(64, 388) ≈ 0.164 rad ≈ 9.4°.
    (Pre-v1b had eye=64/torso=32 giving Δz=32 → 0.082 rad, but that broke
    flat-ground combat — see EYE_HEIGHT comment in cs2_combat.h.)
    Why this exists: validates that the 3D hit-test gates ON correct pitch alignment
    when there's a vertical offset. Without 3D geometry, this would either always
    hit (2D logic ignoring z) or always miss (broken implementation).
    Pitfall: area_idx MUST be set to match the new x/y/z — the combat code uses
    area_idx for vis_matrix lookup, not raw coordinates."""
    import math

    from cs2rl.env.c.cs2_env import Cs2Env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map
    from cs2rl.spec import action as spec
    env = Cs2Env(config=EnvConfig(), map_data=make_simple_map())
    try:
        env.reset(seed=42)
        # Area 5 (T-side, z=0) → area 6 (elevated z=64): vis[5][6]=True
        sx, sy, sz = 575.0, 352.0, 0.0                 # area 5 centroid
        tx, ty, tz = 960.0, 304.0, 64.0                # area 6 centroid
        rx, ry = tx - sx, ty - sy
        eye_z = sz + 48.0                              # EYE_HEIGHT_STAND (v1b: gh #36 fix A — center-to-center)
        torso_z = tz + 48.0                            # TORSO_OFFSET_STAND (v1b: equal to EYE_HEIGHT_STAND)
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
                               target_area_idx=6)
        hp_before = env._c_env.game.agents[5].hp
        actions = np.zeros((10, spec.ACTION_DIM), dtype=np.int32)
        actions[0, 1] = 1                              # HEAD_SHOOT
        cont = np.zeros((10, spec.AIM_DIM), dtype=np.float32)
        cont[0, 1] = pitch                             # v1c: pitch is absolute, set via cont
        env.step(actions, cont)
        hp_after = env._c_env.game.agents[5].hp
        assert hp_after < hp_before, f"shot didn't connect; hp {hp_before}→{hp_after}"
    finally:
        if hasattr(env, "close"):
            env.close()


def test_3d_miss_at_zero_pitch_elevated_target():
    """T5: same geometry as hit-test, BUT shooter pitch = 0 → miss (HP unchanged).

    Same area 5→6 geometry. When pitch=0 the aim ray is horizontal but the
    target torso is rz = 64 units ABOVE eye level (target ground z=64, and
    since v1b EYE == TORSO == 48 at the same stance, so rz is the ground
    delta). The vertical offset is measured against the v1c 36u vertical
    semi-axis, not HIT_HALF_WIDTH: (64/36)² = 3.2 > 1 → miss.
    Pitfall: if the 3D hit-test reduced back to 2D-equivalent (e.g., dz term
    dropped, or perp computed without rz), this test would FAIL — the shot
    would land. This catches an implementation that "compiled but ignored z"."""
    from cs2rl.env.c.cs2_env import Cs2Env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map
    from cs2rl.spec import action as spec
    env = Cs2Env(config=EnvConfig(), map_data=make_simple_map())
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
                               target_area_idx=6)
        hp_before = env._c_env.game.agents[5].hp
        actions = np.zeros((10, spec.ACTION_DIM), dtype=np.int32)
        actions[0, 1] = 1
        cont = np.zeros((10, spec.AIM_DIM), dtype=np.float32)
        # cont[0, 1] left at 0.0 — WRONG pitch (horizontal); v1c absolute semantics
        env.step(actions, cont)
        hp_after = env._c_env.game.agents[5].hp
        assert hp_after == hp_before, (f"shot connected at wrong pitch; hp {hp_before}→{hp_after}")
    finally:
        if hasattr(env, "close"):
            env.close()


def test_3d_hit_pitch_down_from_ramp():
    """T5: down-pitch shot from an elevated ramp at a floor target (z=0).

    Why this exists: validates the SYMMETRIC down-pitch case (the up-pitch
    test proved positive Δz works; this proves negative Δz). Catches a
    sign-flip bug in the 3D direction vector.

    Geometry: T-ramp (area 13) → T-corridor (area 5, flat z=0). adj[13][5]=1
    so the position-raycast in build_vis_matrix passes (single adjacent
    transition).

    RAMP-SURFACE PITFALL — do NOT hard-code a ramp's z. This test used to place
    the shooter at a literal z=64 ("area 13 centroid z"), which was right only
    while every area was flat at its centroids_z. Area 13 is a RAMP: since
    `feat(sim): interpolate z on ramps`, the ground-snap in
    cs2_movement.h::process_movement replaces a->z with the interpolated
    surface z of the ramp quad, and it runs BEFORE combat (cs2_env.h calls
    process_movement, then build_vis_matrix + process_combat). Area 13 spans
    x=750..820 rising 0→64 west→east, so at the centroid x=785 the real surface
    is z=32 — centroids_z holds the ramp's TOP, which the cliff-guard wants but
    a standing agent does not. Aiming from a hard-coded z=64 over-aims by ~8°:
    perpendicular offset ≈30u > HIT_HALF_WIDTH=16 → clean miss, even though
    visibility, walls and the hit-cone math are all fine.

    So: step once WITHOUT firing, let the sim snap the shooter onto the ramp,
    read the settled z back, and derive the pitch from that. A zero movement
    action leaves x/y untouched, so the firing step snaps to the same z. This
    stays correct if the ramp interpolation or the map geometry changes again.

    History: this used to be "catwalk z=128 → T-corridor z=0", but the new
    position-raycast `vis_matrix` (gh #36 follow-up) correctly blocks that
    line. Why: catwalk and bombsite are 2D-adjacent at y=192 with non-zero
    cliff dz=64 → adj[15][6]=0 → wall between them at z=0..WALL_H=150 (per
    viz/render.py). The geometric line from catwalk eye z=176 down to T-corridor
    target z=48 crosses that wall at z≈143 < 150, so the wall blocks it.
    With our current full-height-wall model in simple_map, catwalk-to-floor
    shots are physically impossible. Real CS has lower parapets (low cover
    you shoot OVER); modeling that needs either per-edge wall heights or a
    3D over-wall exemption — deferred. T-ramp→T-corridor exercises the
    SAME down-pitch math without hitting the wall-height limitation."""
    import math

    from cs2rl.env.c.cs2_env import Cs2Env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map
    from cs2rl.spec import action as spec
    env = Cs2Env(config=EnvConfig(), map_data=make_simple_map())
    try:
        env.reset(seed=42)
        # T-ramp → T-corridor (flat z=0); adj[13][5]=1 (ramp connection).
        sx, sy = 785.0, 304.0                          # area 13 centroid x/y; z comes from the sim
        tx, ty, tz = 575.0, 352.0, 0.0                 # inside area 5 (its centroid is (575, 304))
        actions = np.zeros((10, spec.ACTION_DIM), dtype=np.int32)
        cont = np.zeros((10, spec.AIM_DIM), dtype=np.float32)

        # Settle step: put the shooter on the ramp and step with NO shoot action
        # so the ground-snap resolves the true interpolated surface z for us.
        # sz here is only a seed value — process_movement overwrites it.
        _setup_3d_hit_scenario(env,
                               sx=sx,
                               sy=sy,
                               sz=64.0,
                               shooter_area_idx=13,
                               tx=tx,
                               ty=ty,
                               tz=tz,
                               target_area_idx=5)
        env.step(actions, cont)
        ramp_z = env._c_env.game.agents[0].z
        assert env._c_env.game.agents[0].area_idx == 13, "shooter left the ramp"
        # Guard the test's premise: this must stay a genuine DOWN-pitch shot. If
        # the ramp ever interpolated to the floor, the shot below would become a
        # flat-ground shot and still "pass" without exercising negative Δz.
        assert ramp_z > tz + 16.0, f"shooter not elevated above target: {ramp_z} vs {tz}"

        eye_z = ramp_z + 48.0                          # EYE_HEIGHT_STAND
        torso_z = tz + 48.0                            # TORSO_OFFSET_STAND
        rz = torso_z - eye_z                           # negative: aiming down
        dist_2d = math.hypot(tx - sx, ty - sy)
        pitch = math.atan2(rz, dist_2d)                # ~-0.15 rad at the ramp midpoint
        assert pitch < 0.0, f"expected a downward pitch, got {pitch}"

        # Fire step: re-arm (fire_cd=0, target hp=100) at the settled ramp z.
        _setup_3d_hit_scenario(env,
                               sx=sx,
                               sy=sy,
                               sz=ramp_z,
                               shooter_area_idx=13,
                               tx=tx,
                               ty=ty,
                               tz=tz,
                               target_area_idx=5)
        hp_before = env._c_env.game.agents[5].hp
        actions[0, 1] = 1
        cont[0, 1] = pitch             # v1c: pitch is absolute, set via cont
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

    from cs2rl.env.c.cs2_env import Cs2Env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map
    from cs2rl.spec import action as spec
    env = Cs2Env(config=EnvConfig(), map_data=make_simple_map())
    try:
        env.reset(seed=42)
        # Area 5→6: same geometry as the hit test — perfectly aimed
        sx, sy, sz = 575.0, 352.0, 0.0
        tx, ty, tz = 960.0, 304.0, 64.0
        rx, ry = tx - sx, ty - sy
        eye_z = sz + 48.0                              # EYE_HEIGHT_STAND (v1b: gh #36 fix A)
        torso_z = tz + 48.0                            # TORSO_OFFSET_STAND (v1b: equal to EYE_HEIGHT_STAND)
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
                               target_area_idx=6)
        actions = np.zeros((10, spec.ACTION_DIM), dtype=np.int32)
        actions[0, 1] = 1
        cont = np.zeros((10, spec.AIM_DIM), dtype=np.float32)
        cont[0, 1] = pitch                             # v1c: pitch is absolute, set via cont
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
    while the downward bound would silently be wrong. v1c: writing absolute=-100
    saturates instantly to -π/2 (single fmaxf hit). The 100-tick loop is preserved
    from v1b for parity but saturation now happens on tick 1, not progressively."""
    import math

    from cs2rl.env.c.cs2_env import Cs2Env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map
    env = Cs2Env(config=EnvConfig(), map_data=make_simple_map())
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


# ── R0-E.2 (#131) pin cases ───────────────────────────────────────────────


def test_pinned_pitch_stays_zero_across_ticks_and_default_moves():
    """pin_pitch=1: cont[:,1] is ignored on EVERY tick (not just the first) and
    the Welford pitch counters never increment. The same actions on the
    default env move pitch — so the gate is the StaticData flag, not a
    coincidence of zero inputs."""
    from cs2rl.env.c.cs2_env import Cs2Env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_simple_map
    rng = np.random.default_rng(0)
    for pin in (1, 0):
        env = Cs2Env(config=EnvConfig(pin_pitch=pin), map_data=make_simple_map())
        try:
            env.reset(seed=42)
            for _ in range(5):
                act, cont = _zero_actions()
                cont[:, 1] = rng.uniform(-0.7, 0.7, size=10).astype(np.float32)
                env.step(act, cont)
            pitches = [env._c_env.game.agents[i].pitch for i in range(10)]
            cnt = env._c_env.episode_stats.aim_delta_pitch_count
            if pin:
                assert pitches == [0.0] * 10, pitches
                assert cnt == 0
            else:
                assert any(p != 0.0 for p in pitches), pitches
                assert cnt == 50
        finally:
            env.close()
