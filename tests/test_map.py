"""tests/test_map.py — MapData verticality (Batch 5 prerequisite for Δpitch).

Verifies the simple map's verticality fields (centroids_z, is_ramp) are coherent,
the elevated bombsite is reachable on foot, vis matrix sees catwalk-bombsite, and
the L9 adjacency post-prune removes cliff edges from the nav graph.

T1 tests (steps 1.8-1.10): 5 pure-Python MapData tests, no C env required.
T2 tests (step 2.7): 1 binding smoke test — env construction with centroids_z/is_ramp plumbed.
T3 tests (step 3.7): 5 movement/behaviour tests requiring the C env — added later.
T4 tests (step 4.5): 1 obs z-delta test — added in T4.
"""
import math
import sys
from pathlib import Path

import numpy as np

# Ensure src/ is importable when running pytest from the repo root.
# conftest.py does this too via SRC_DIR insertion, but this file is
# self-contained so subagents can run it in isolation.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── T1: MapData verticality field tests ──────────────────────────────────────


def test_simple_map_has_verticality(simple_map):
    """At least one adjacent pair with non-zero Δz exists.

    Acceptance criterion: centroids_z is not flat-0 across all areas, and
    at least one adjacency edge spans a non-zero elevation difference.
    Pitfall: adjacency diagonal (self↔self) is always True but Δz=0; skip those.
    """
    seen = False
    for i in range(simple_map.N):
        for j in range(simple_map.N):
            if i == j:
                continue
            if (simple_map.adjacency[i, j]
                    and simple_map.centroids_z[i] != simple_map.centroids_z[j]):
                seen = True
                break
        if seen:
            break
    assert seen, "no adjacent pair with non-zero Δz found — centroids_z may not have been populated"


def test_simple_map_bombsite_elevated(simple_map):
    """Bombsite (area_idx=6) sits at z >= 64.

    The bombsite is the load-bearing elevated platform: T attackers must walk up
    a ramp (z=0→64 snap) and CT defenders may fire down from the catwalk (z=128).
    area_idx=6 maps to area_id=6 for simple maps (area_id == area_idx invariant).
    """
    assert simple_map.centroids_z[6] >= 64.0, (
        f"bombsite centroids_z[6]={simple_map.centroids_z[6]}, expected >= 64.0")


def test_simple_map_catwalk_overlooks_bombsite(simple_map):
    """Catwalk (area_idx=15) has 2D LOS to bombsite (6) AND is at least 64u above it.

    Spec §6 acceptance criterion 4: vis_matrix[catwalk, bombsite] must be True
    (2D Bresenham LOS approximation; z-occlusion is out-of-scope per spec L2).
    Δz >= 64 confirms the catwalk is the elevated position that drives Δpitch signal.
    """
    catwalk_idx = 15
    bombsite_idx = 6
    assert simple_map.vis_matrix[catwalk_idx, bombsite_idx], (
        "catwalk must see bombsite in 2D LOS (Bresenham); check area geometry")
    dz = simple_map.centroids_z[catwalk_idx] - simple_map.centroids_z[bombsite_idx]
    assert dz >= 64.0, f"catwalk-bombsite dz={dz} < 64u; catwalk must overlook by >=64"


def test_simple_map_cliff_adjacency_pruned(simple_map):
    """Catwalk↔Bombsite (Δz=64, both non-ramp) is xy-adjacent in raster but pruned by L9.

    The catwalk shares the y=192 boundary with the bombsite in 8-neighbour raster terms,
    but both are non-ramp and |Δz|=64 >> MAX_STEP_HEIGHT=18. L9 must prune this edge so
    nav-distance shaping does not award a shortcut bonus for a movement the C cliff guard
    will refuse at runtime.
    """
    catwalk_idx = 15
    bombsite_idx = 6
    assert not simple_map.adjacency[catwalk_idx, bombsite_idx], (
        "catwalk-bombsite cliff edge must be pruned by L9 to keep nav-shaping consistent")
    assert not simple_map.adjacency[bombsite_idx,
                                    catwalk_idx], ("adjacency must be symmetric after pruning")


def test_simple_map_ramps_kept_in_adjacency(simple_map):
    """T-corridor↔T-ramp (Δz=64, target is_ramp=True) survives L9 pruning.

    The L9 rule exempts edges where at least one endpoint has is_ramp=True,
    so ramps remain navigable in the nav graph. The T-corridor (area 5, z=0)
    connects to T-ramp (area 13, z=64, is_ramp=True) — this is the primary
    T-side ascent path to the bombsite.
    """
    t_corridor_idx = 5
    t_ramp_idx = 13
    assert simple_map.adjacency[t_corridor_idx, t_ramp_idx], (
        "T-corridor → T-ramp must remain adjacent (ramp exemption)")
    assert simple_map.adjacency[t_ramp_idx, t_corridor_idx], (
        "adjacency must be symmetric (T-ramp → T-corridor)")


# ── T2: binding smoke test ────────────────────────────────────────────────────


def test_centroids_z_plumbed_through_binding():
    """Round-trip: MapData.centroids_z[bombsite] == 64 and is_ramp[13] == True survive
    Python→C→Python via env init (T2 plumbing smoke).

    The C side stores raw pointers to numpy buffers; we can't read back through
    ctypes without a debug accessor.  This smoke test verifies:
      1. binding.init() accepts the two new array args without raising.
      2. The Python-side MapData is intact after construction (centroids_z[6]==64.0,
         is_ramp[13]==True) — ruling out accidental mutation during the _arr / astype calls.
      3. env.close() cleans up without segfault.

    Pitfall: if is_ramp_int8 were collected by GC before/during env init, the C pointer
    would dangle and env.step() would segfault.  This test can't catch that (GC is
    non-deterministic), but if is_ramp_int8 is NOT in self._refs, a gc.collect() call
    inside this test would expose it.  The actual GC-lifetime check is done by reading
    back through the ctypes pointer in T3's full movement tests.
    """
    import gc
    import sys

    # Extend path in case test is run in isolation (conftest.py also does this)
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from c_env.cs2_env import Cs2Env
    from env_config import EnvConfig
    from map import make_simple_map

    map_data = make_simple_map()

    # Verify MapData fields are in place before constructing env
    assert map_data.centroids_z[6] == 64.0, (
        f"bombsite area_idx=6 should have centroids_z=64.0, got {map_data.centroids_z[6]}")
    assert bool(map_data.is_ramp[13]), "is_ramp[13] should be True (T-ramp area)"

    # Construct env — if the packed layout or the pointer-argument set is wrong, this raises
    env = Cs2Env(config=EnvConfig(), map_data=map_data)

    # Force GC to try to collect any would-be dangling is_ramp_int8 array.
    # If it's NOT in self._refs, this can expose a use-after-free on the next step().
    gc.collect()

    # Verify MapData fields still intact after env construction (no accidental mutation)
    assert map_data.centroids_z[6] == 64.0, ("centroids_z[6] mutated during env construction")
    assert bool(map_data.is_ramp[13]), ("is_ramp[13] mutated during env construction")

    if hasattr(env, "close"):
        env.close()


# ── T3: movement behaviour tests (ground-snap, landing, cliff guard, ε) ──────
#
# These tests mutate AgentState fields directly via the ctypes overlay
# (env._c_env.game) to set up specific scenarios, then step the env and
# assert post-tick invariants.  All tests use the simple map (make_simple_map)
# so that centroids_z[6]=64 (bombsite) and is_ramp[13]=True (T-ramp) are live.
#
# Movement direction conventions (post-F9 right-handed basis, 2026-07-06):
#   The env uses facing-LOCAL movement bins (cs2_movement.h _LOCAL_MOVE_X/Y),
#   rotated into world via world = fy*forward + fx*right with
#   forward = (cos f, sin f) and right = (sin f, -cos f) (CW perpendicular;
#   yaw is CCW in the x-east/y-north frame).
#   Bin 1 ("W" = forward): _LOCAL_MOVE_Y[1]=+1, _LOCAL_MOVE_X[1]=0.
#   With a->facing = -π/2 (agent facing -Y in world):
#       wx = fy*cos(-π/2) + fx*sin(-π/2) = -fx
#       wy = fy*sin(-π/2) - fx*cos(-π/2) = -fy
#   So bin 1 → world (wx=0, wy=-1): moves in -Y direction (toward lower y, i.e.
#   toward the catwalk at y=80 from the bombsite at y=192-416).
#   Bin 2 ("WD" = forward+right): _LOCAL_MOVE_X[2]=+0.707, _LOCAL_MOVE_Y[2]=+0.707.
#       wx = -0.707  (moves -X, toward west — facing south, right IS west)
#       wy = -0.707  (moves -Y, toward catwalk)
#   So bin 2 with facing=-π/2 drives SW in screen terms (decreasing x and y).
#   (Pre-F9 the strafe axis was mirrored and bin 8 "WA" produced this vector.)
#
# Assertion: bin 1 drives decreasing y and bin 2 drives decreasing x+y, verified
# by the pre-step asserts in test_cliff_guard_blocks_walkup below.


def _make_simple_env(seed=42):
    """Create a PufferEnv seeded from the simple map with verticality active.

    Using the default make_puffer_env() would give the dust2 map (centroids_z
    all zeros), so verticality tests MUST explicitly pass map_data=simple_map.
    """
    from c_env.cs2_env import make_env
    from map import make_simple_map
    return make_env(seed=seed, map_data=make_simple_map())


def _zero_actions(n_agents=10):
    """Return (actions, continuous_actions) zero buffers for n_agents."""
    import _action_spec as spec
    return (
        np.zeros((n_agents, spec.ACTION_DIM), dtype=np.int32),
        np.zeros((n_agents, spec.AIM_DIM), dtype=np.float32),
    )


def _room_x_lerp(x0, x1, z_w, z_e, x):
    """X-slope room lerp: z = (1-u)*z_w + u*z_e, u=(x-x0)/(x1-x0)."""
    return (1.0 - (x - x0) / (x1 - x0)) * z_w + ((x - x0) / (x1 - x0)) * z_e


def _room_y_lerp(y0, y1, z_s, z_n, y):
    """Y-slope room lerp: z = (1-v)*z_s + v*z_n, v=(y-y0)/(y1-y0)."""
    return (1.0 - (y - y0) / (y1 - y0)) * z_s + ((y - y0) / (y1 - y0)) * z_n


def test_simple_map_area_bounds_match_rooms(simple_map):
    """make_simple_map publishes the room tuples, not a raster AABB."""
    from map import SIMPLE_ROOMS
    assert simple_map.area_bounds is not None
    assert simple_map.area_bounds.shape == (simple_map.N, 4)
    assert simple_map.area_bounds.dtype == np.float32
    for idx, x0, y0, x1, y1, *_ in SIMPLE_ROOMS:
        assert list(simple_map.area_bounds[idx]) == [x0, y0, x1, y1]


def test_agent_on_t_ramp_does_not_snap_to_top():
    """T-ramp interpolation: grounded agent at (755, 300) is the room lerp.

    Area 13 is T-ramp (750–820, 192–416). X-slope 0→64 on the SIMPLE_ROOMS
    quad: z ≈ (755-750)/(820-750)*64. Must not snap to top 64.
    """
    env = _make_simple_env(seed=42)
    try:
        env.reset(seed=42)
        # Live overlay must be the room quad (750,192,820,416), not a raster AABB.
        b = env._c_env.sd.contents.area_bounds
        assert [b[13 * 4 + i] for i in range(4)] == [750.0, 192.0, 820.0, 416.0]
        g = env._c_env.game
        g.agents[0].x = 755.0
        g.agents[0].y = 300.0
        g.agents[0].area_idx = 13
        g.agents[0].z = 0.0
        g.agents[0].vz = 0.0
        g.agents[0].is_airborne = 0
        actions, cont = _zero_actions()
        env.step(actions, cont)
        z_want = _room_x_lerp(750.0, 820.0, 0.0, 64.0, 755.0)
        assert abs(g.agents[0].z - z_want) < 0.5, (
            f"T-ramp room lerp at (755,300) want {z_want:.3f} ±0.5, got z={g.agents[0].z}")
        assert g.agents[0].is_airborne == 0, (
            "agent should remain grounded on the interpolated ramp surface")
        assert g.agents[0].area_idx == 13, (
            f"agent left T-ramp (area 13) for area_idx={g.agents[0].area_idx}")
    finally:
        env.close()


def test_agent_on_ct_ramp_follows_x_slope():
    """CT-ramp (14: 1100–1170, 192–416) is X-slope 64→0 on the room quad."""
    env = _make_simple_env(seed=42)
    try:
        env.reset(seed=42)
        g = env._c_env.game
        g.agents[0].x = 1135.0
        g.agents[0].y = 300.0
        g.agents[0].area_idx = 14
        g.agents[0].z = 64.0
        g.agents[0].vz = 0.0
        g.agents[0].is_airborne = 0
        actions, cont = _zero_actions()
        env.step(actions, cont)
        z_want = _room_x_lerp(1100.0, 1170.0, 64.0, 0.0, 1135.0)
        assert abs(g.agents[0].z - z_want) < 0.5, (
            f"CT-ramp room X-slope at (1135,300) want {z_want:.3f} ±0.5, got z={g.agents[0].z}")
        assert g.agents[0].is_airborne == 0
        assert g.agents[0].area_idx == 14, (
            f"agent left CT-ramp (area 14) for area_idx={g.agents[0].area_idx}")
    finally:
        env.close()


def test_agent_on_stairs_follows_y_slope():
    """Stairs (16: 1170–1300, 80–192) is Y-slope 128→0 on the room quad."""
    env = _make_simple_env(seed=42)
    try:
        env.reset(seed=42)
        g = env._c_env.game
        g.agents[0].x = 1235.0
        g.agents[0].y = 136.0
        g.agents[0].area_idx = 16
        g.agents[0].z = 128.0
        g.agents[0].vz = 0.0
        g.agents[0].is_airborne = 0
        actions, cont = _zero_actions()
        env.step(actions, cont)
        z_want = _room_y_lerp(80.0, 192.0, 128.0, 0.0, 136.0)
        assert abs(g.agents[0].z - z_want) < 0.5, (
            f"stairs room Y-slope at (1235,136) want {z_want:.3f} ±0.5, got z={g.agents[0].z}")
        assert g.agents[0].is_airborne == 0
        assert g.agents[0].area_idx == 16, (
            f"agent left stairs (area 16) for area_idx={g.agents[0].area_idx}")
    finally:
        env.close()


def test_agent_walk_to_bombsite_reaches_elevation():
    """T3 load-bearing: agent teleported to bombsite (area_idx=6, z=0) snaps to z=64
    on the next ground-snap tick.

    Spec §6 acceptance criterion 3: "a grounded agent whose area_idx is set to the
    bombsite has z = centroids_z[bombsite] = 64 after one step."
    This directly validates the ground-snap rule (plan step 3.4).
    """
    env = _make_simple_env(seed=42)
    try:
        env.reset(seed=42)
        g = env._c_env.game
        # Teleport agent 0 to bombsite centroid (xy in area 6) at z=0.
        # The ground-snap rule (T3 step 3.4) must set z=centroids_z[6]=64 on
        # the next tick, even though we placed the agent at z=0.
        g.agents[0].x = 950.0
        g.agents[0].y = 300.0
        g.agents[0].area_idx = 6
        g.agents[0].z = 0.0
        g.agents[0].vz = 0.0
        g.agents[0].is_airborne = 0
        actions, cont = _zero_actions()
        env.step(actions, cont)
        assert g.agents[0].z >= 64.0, (
            f"ground-snap failed: agent at bombsite (area 6) should have z=64, "
            f"got z={g.agents[0].z}")
        assert g.agents[0].is_airborne == 0, (
            "agent should remain grounded after z-snap to terrain surface")
    finally:
        env.close()


def test_landing_on_elevated_area():
    """T3 load-bearing: airborne agent above bombsite lands at z=64 (not z=0).

    Spec §6 acceptance criterion 5: "an airborne agent whose xy is in the bombsite
    and who falls from z=200 lands at z=64 (terrain_z of that area)."
    This validates the landing rule rewrite (plan step 3.3).
    """
    env = _make_simple_env(seed=42)
    try:
        env.reset(seed=42)
        g = env._c_env.game
        # Place agent airborne above bombsite at z=200, falling with vz=0.
        # Gravity (SV_GRAVITY_CS=800 u/s²) will pull it down; it should land
        # at terrain_z=64, not 0.
        g.agents[0].x = 950.0
        g.agents[0].y = 300.0
        g.agents[0].area_idx = 6
        g.agents[0].z = 200.0
        g.agents[0].vz = 0.0
        g.agents[0].is_airborne = 1
        actions, cont = _zero_actions()
        # Step until landed (max 40 ticks; at 16 Hz falling 136u under gravity
        # takes ~0.58s ≈ 9–10 ticks, plus ε).
        for _ in range(40):
            env.step(actions, cont)
            if g.agents[0].is_airborne == 0:
                break
        assert g.agents[0].is_airborne == 0, (
            "agent never landed within 40 ticks — landing rule may be broken")
        assert abs(g.agents[0].z - 64.0) < 0.1, (
            f"landed at z={g.agents[0].z}, expected ~64.0 (bombsite terrain_z)")
        assert g.agents[0].vz == 0.0, (f"vz should be cleared on landing, got {g.agents[0].vz}")
    finally:
        env.close()


def test_cliff_guard_blocks_walkup():
    """T3 load-bearing: grounded agent at bombsite (z=64) cannot walk directly
    into the catwalk area (z=128) since Δz=64 >> SV_MAX_STEP_HEIGHT_CS=18 and
    catwalk is non-ramp.

    Spec §6 acceptance criterion 6: "the cliff guard rejects the transition."
    Validates plan step 3.2 (cliff guard in _resolve_xy_collision).
    The cliff boundary is the y=192 edge: bombsite y=192-416, catwalk y=80-192.
    Agent placed at y=195 (just inside bombsite) and driven north (decreasing y)
    toward the catwalk for 20 ticks — must stay in area_idx=6 throughout.

    Movement: facing=-π/2, bin 1 → world wy=-1 (decreasing y).
    Pre-step assertion verifies the direction before trusting the main assert.
    """
    env = _make_simple_env(seed=42)
    try:
        env.reset(seed=42)
        g = env._c_env.game
        # Place on the bombsite, just south of the y=192 cliff to catwalk.
        g.agents[0].x = 900.0
        g.agents[0].y = 195.0
        g.agents[0].area_idx = 6
        g.agents[0].z = 64.0
        g.agents[0].vz = 0.0
        g.agents[0].is_airborne = 0
        # facing = -π/2 so bin 1 (forward) drives wy < 0 (decreasing y = toward catwalk).
        g.agents[0].facing = float(-math.pi / 2)
        y0 = g.agents[0].y
        actions, cont = _zero_actions()
        # Bin 1 = "W" (forward). With facing=-π/2: wy = -1 → drives decreasing y.
        # HEAD_MOVE = 0; sanity pre-check below confirms the agent moved in -y.
        actions[0, 0] = 1
        env.step(actions, cont)
        assert g.agents[0].y <= y0, (f"facing=-π/2 + bin 1 should drive decreasing y; "
                                     f"y0={y0} but y={g.agents[0].y}. Check facing/bin convention.")
        # Cliff guard must keep agent in area 6 as it approaches y=192 boundary.
        for _ in range(19):
            env.step(actions, cont)
            assert g.agents[0].area_idx == 6, (
                f"cliff guard failed: agent escaped to area_idx={g.agents[0].area_idx} "
                f"(catwalk is area 15); z={g.agents[0].z}")
        assert g.agents[0].z == 64.0, (f"agent z drifted off bombsite terrain ({g.agents[0].z})")
    finally:
        env.close()


def test_cliff_guard_diagonal_slides():
    """T3 load-bearing: NW move at the catwalk cliff lets the agent slide west
    (x decreasing) while the north component is rejected by the cliff guard.

    Validates the interaction between the cliff guard and the existing axis-split
    wall-slide logic (plan step 3.2 + the pre-existing diagonal retry in
    process_movement). If sliding is broken the agent would be fully stuck.

    Movement: facing=-π/2, bin 2 ("WD") → world (-0.707, -0.707) — post-F9
    right-handed basis: facing south, geometric right IS west.
    The -y component hits the cliff (area 15 target, Δz=64 > 18, non-ramp) → rejected.
    The -x component tries to move west within area 6 → allowed if area 6 covers it.
    The agent must move west (x decrease) but stay in area 6.
    """
    env = _make_simple_env(seed=42)
    try:
        env.reset(seed=42)
        g = env._c_env.game
        # Starting position: bombsite, near the north cliff edge, with room to go west.
        g.agents[0].x = 900.0
        g.agents[0].y = 195.0
        g.agents[0].area_idx = 6
        g.agents[0].z = 64.0
        g.agents[0].vz = 0.0
        g.agents[0].is_airborne = 0
        g.agents[0].facing = float(-math.pi / 2)
        x0 = g.agents[0].x
        actions, cont = _zero_actions()
        # Bin 2 = "WD" (forward+right), post-F9 basis. With facing=-π/2:
        #   wx = fy*cos(-π/2) + fx*sin(-π/2) = 0 + 0.707*(-1) = -0.707  (west)
        #   wy = fy*sin(-π/2) - fx*cos(-π/2) = -0.707 - 0     = -0.707  (toward
        #        catwalk, blocked by the cliff guard)
        # (Pre-F9 the mirrored basis produced this vector from bin 8 "WA".)
        # HEAD_MOVE = 0; x should decrease (west slide allowed) post-step.
        actions[0, 0] = 2
        env.step(actions, cont)
        env.step(actions, cont)
        assert g.agents[0].x < x0, (
            f"axis-split slide failed: x did not decrease ({x0:.1f} → {g.agents[0].x:.1f}). "
            f"Either cliff guard is blocking both axes or bin/facing convention is wrong.")
        # y-component (north → catwalk) must be rejected by cliff guard;
        # y stays approximately at the start (within float epsilon of 195.0).
        # Proves axis-split is genuine (only x moved, not "magically nudged south").
        assert g.agents[0].y >= 195.0 - 0.5, (
            f"y drifted during slide ({g.agents[0].y:.2f}); guard should reject y")
        assert g.agents[0].area_idx == 6, (
            f"agent escaped to area_idx={g.agents[0].area_idx} during diagonal slide")
        assert g.agents[0].z == 64.0, (f"z drifted during diagonal slide: {g.agents[0].z}")
    finally:
        env.close()


def test_is_airborne_no_flicker_on_elevated_terrain():
    """T3 load-bearing: standing still at bombsite terrain (z=64) for 10 ticks must
    not flicker is_airborne to 1 (the ε guard must tolerate exact terrain_z).

    Validates plan step 3.5 (ε=1.0u guard). Without the ε guard, floating-point
    noise from the integration could push z fractionally above terrain_z and trigger
    is_airborne=1 every tick, causing the agent to fall through the floor on the next
    tick and oscillate. The ε=1.0u absorbs reasonable integration noise.
    """
    env = _make_simple_env(seed=42)
    try:
        env.reset(seed=42)
        g = env._c_env.game
        # Place grounded on bombsite at exactly terrain_z=64.
        g.agents[0].x = 950.0
        g.agents[0].y = 300.0
        g.agents[0].area_idx = 6
        g.agents[0].z = 64.0
        g.agents[0].vz = 0.0
        g.agents[0].is_airborne = 0
        actions, cont = _zero_actions()
        for tick in range(10):
            env.step(actions, cont)
            assert g.agents[0].is_airborne == 0, (
                f"is_airborne flickered to 1 at tick {tick + 1} "
                f"(z={g.agents[0].z:.4f}, vz={g.agents[0].vz:.6f}) — "
                f"ε guard may be missing or terrain_z mismatch")
        assert abs(g.agents[0].z -
                   64.0) < 0.01, (f"z drifted from 64 after 10 idle ticks: {g.agents[0].z}")
    finally:
        env.close()


# ── T4: obs z-delta slot population ──────────────────────────────────────────


def test_obs_z_delta_populated_for_elevated_teammate():
    """T4 load-bearing: obs[30] reflects (teammate.z - self.z)/128 = 0.5 when
    self is at z=0 and teammate 1 is at z=64.

    Slot derivation (plan §4.5, re-pinned for Batch 6 Task 2.5): teammate slots
    start at obs[28] (OBS_TEAMMATE_BASE; was 25 before the bombsite-bearing
    slots grew the self block to 28), base = 28 + tm_count * 7.  Teammate
    iteration skips self (j == i), so for agent 0 viewing agent 1, tm_count=0
    → base=28 → z_delta at obs[base+2] = obs[30].

    Includes a sign-flip sub-case (self at z=64, teammate at z=0 → obs[30] = -0.5)
    to catch a subtle direction bug that wouldn't surface with just the positive case.

    Pitfall: obs population only happens in compute_observations() which runs
    during env.step(), NOT during env.reset().  The test must call env.step()
    after placing agents, then read env.observations[0].
    """
    env = _make_simple_env(seed=42)
    try:
        env.reset(seed=42)
        g = env._c_env.game
        # Self at T-spawn (z=0), teammate at bombsite (z=64).
        # Both on team T (agents 0-4 are T-side in the simple map layout).
        g.agents[0].x = 100.0
        g.agents[0].y = 500.0
        g.agents[0].z = 0.0
        g.agents[0].area_idx = 0                                                        # T-spawn-A
        g.agents[0].is_airborne = 0
        g.agents[1].x = 950.0
        g.agents[1].y = 300.0
        g.agents[1].z = 64.0
        g.agents[1].area_idx = 6                                                        # bombsite (elevated)
        g.agents[1].is_airborne = 0
        actions, cont = _zero_actions()
        env.step(actions, cont)
                                                                                        # Verify ground-snap didn't mutate the teammate's z (T3 invariant).
        assert g.agents[1].z == 64.0, f"teammate z drifted ({g.agents[1].z})"
        assert g.agents[1].is_airborne == 0
                                                                                        # env.observations is a flat buffer; [0] gives the OBS_DIM slice for agent 0.
        obs = env.observations[0]
                                                                                        # Teammate slot: tm_count=0, base=28, z_delta at obs[base+2]=obs[30].
        z_delta = obs[30]
        expected = (64.0 - 0.0) / 128.0                                                 # = 0.5
        assert abs(z_delta - expected) < 1e-5, (
            f"teammate z-delta slot obs[30] = {z_delta:.6f}, expected {expected:.6f}. "
            f"Check (tm->z - a->z)/128.0f; agent 1 alive={g.agents[1].alive}.")

        # Sign-flip sub-case: swap z values; expect obs[30] = -0.5.
        g.agents[0].z = 64.0
        g.agents[0].area_idx = 6
        g.agents[1].z = 0.0
        g.agents[1].area_idx = 0
        env.step(actions, cont)
        obs = env.observations[0]
        expected_neg = (0.0 - 64.0) / 128.0                                                       # = -0.5
        assert abs(obs[30] - expected_neg) < 1e-5, (
            f"sign-flip: teammate z-delta obs[30] = {obs[30]:.6f}, expected {expected_neg:.6f}. "
            f"Direction bug? Formula must be (tm->z - a->z), NOT (a->z - tm->z).")
    finally:
        env.close()


# DrawCylinder radius in cs2_render.h — collision hull must keep the visible
# body on the walkable side of an exterior wall. Named here so a radius
# change fails this file, not a silent viz/sim drift.
_AGENT_VIZ_RADIUS = 12.0


def test_exterior_wall_keeps_body_inside_room():
    """Point collision lets the 12u body sit inside an 8u exterior wall.

    T-spawn-A west face is x=0, 16u-aligned, no raster overshoot. The visual
    wall is WALL_DEPTH=8 pushed into x<0, so it occupies [-8, 0]. A center
    at x≈0 puts the cylinder in [-12, 12] — fully through the wall, hugging
    the outer face. After the hull, the center must stay at x >= 12.
    """
    env = _make_simple_env(seed=42)
    try:
        env.reset(seed=42)
        g = env._c_env.game
        g.agents[0].x = 40.0
        g.agents[0].y = 500.0
        g.agents[0].z = 0.0
        g.agents[0].vx = 0.0
        g.agents[0].vy = 0.0
        g.agents[0].area_idx = 0
        g.agents[0].is_airborne = 0
        g.agents[0].facing = float(math.pi)                              # bin 1 = west (−x)
        actions, cont = _zero_actions()
        actions[0, 0] = 1
        env.step(actions, cont)
        assert g.agents[0].x < 40.0, (
            f"facing=π + bin 1 should drive −x; still at x={g.agents[0].x}")
        for _ in range(19):
            env.step(actions, cont)
        assert g.agents[0].area_idx == 0, (
            f"walked off T-spawn-A into area_idx={g.agents[0].area_idx}")
        assert abs(g.agents[0].x - _AGENT_VIZ_RADIUS) < 2.0, (
            f"center x={g.agents[0].x:.2f} should stop at the 12u hull "
            f"on T-spawn-A west (x=0), not inside the [-8, 0] cube")
    finally:
        env.close()


def test_raster_overshoot_cannot_enter_exterior_wall():
    """Simple-map raster marks whole cells; catwalk west AABB is x=820.

    col 51 covers [816, 832). Point collision walks to x≈816, which is
    inside the exterior wall cube centered at 816. The room quad starts
    at 820 — the center must stay in-bounds by the viz radius.
    """
    env = _make_simple_env(seed=42)
    try:
        env.reset(seed=42)
        g = env._c_env.game
        g.agents[0].x = 860.0
        g.agents[0].y = 130.0
        g.agents[0].z = 128.0
        g.agents[0].vx = 0.0
        g.agents[0].vy = 0.0
        g.agents[0].area_idx = 15
        g.agents[0].is_airborne = 0
        g.agents[0].facing = float(math.pi)
        actions, cont = _zero_actions()
        actions[0, 0] = 1
        for _ in range(20):
            env.step(actions, cont)
        assert g.agents[0].area_idx == 15, (f"left catwalk for area_idx={g.agents[0].area_idx}")
        min_x = 820.0 + _AGENT_VIZ_RADIUS
        assert abs(g.agents[0].x -
                   min_x) < 2.0, (f"center x={g.agents[0].x:.2f} should stop at catwalk west "
                                  f"hull ({min_x}), not in the [816, 820) overshoot")
    finally:
        env.close()


def test_t_ramp_portal_is_walkable():
    """T-corridor → T-ramp → bombsite must stay walkable after the hull.

    Later rooms own the 16u column that straddles x=750 and x=820. Testing
    the raster label's AABB on that column rejects the earlier room and
    severs the ramp. Drive east from (700, 300); area_idx must become 13
    then 6.
    """
    env = _make_simple_env(seed=42)
    try:
        env.reset(seed=42)
        g = env._c_env.game
        g.agents[0].x = 700.0
        g.agents[0].y = 300.0
        g.agents[0].z = 0.0
        g.agents[0].vx = 0.0
        g.agents[0].vy = 0.0
        g.agents[0].area_idx = 5
        g.agents[0].is_airborne = 0
        g.agents[0].facing = 0.0                                                       # bin 1 = east (+x)
        actions, cont = _zero_actions()
        actions[0, 0] = 1
        seen = {5}
        for _ in range(40):
            env.step(actions, cont)
            seen.add(int(g.agents[0].area_idx))
        assert 13 in seen, (
            f"never entered T-ramp (13); areas={sorted(seen)} x={g.agents[0].x:.1f} "
            f"— raster AABB at the 750 portal is sealing the doorway")
        assert 6 in seen, (
            f"never entered bombsite (6); areas={sorted(seen)} x={g.agents[0].x:.1f} "
            f"— raster AABB at the 820 portal is sealing the ramp top")
        assert g.agents[0].x > 820.0, (
            f"ended at x={g.agents[0].x:.1f}, expected past bombsite west 820")
    finally:
        env.close()


def test_ct_ramp_portal_is_walkable():
    """CT-corridor → CT-ramp → bombsite, mirror of the T doorway."""
    env = _make_simple_env(seed=42)
    try:
        env.reset(seed=42)
        g = env._c_env.game
        g.agents[0].x = 1220.0
        g.agents[0].y = 300.0
        g.agents[0].z = 0.0
        g.agents[0].vx = 0.0
        g.agents[0].vy = 0.0
        g.agents[0].area_idx = 7
        g.agents[0].is_airborne = 0
        g.agents[0].facing = float(math.pi)
        actions, cont = _zero_actions()
        actions[0, 0] = 1
        seen = {7}
        for _ in range(40):
            env.step(actions, cont)
            seen.add(int(g.agents[0].area_idx))
        assert 14 in seen, (
            f"never entered CT-ramp (14); areas={sorted(seen)} x={g.agents[0].x:.1f}")
        assert 6 in seen, (
            f"never entered bombsite (6); areas={sorted(seen)} x={g.agents[0].x:.1f}")
        assert g.agents[0].x < 1100.0, (
            f"ended at x={g.agents[0].x:.1f}, expected past bombsite east 1100")
    finally:
        env.close()


# Note: enemy z-delta slot at obs[base+2] (where base=51 for the closest enemy slot2=0)
# is intentionally NOT covered here. The slot uses a distance-sorted indirection
# (`order[]` array in cs2_observations.h:100-110) that needs setup-coordination across
# multiple agents to test reliably. The formula matches the teammate write site verbatim,
# and visibility-gating is documented at the write site (cs2_observations.h:122-134).
# A live deploy-side test in T6 (or a dedicated test_obs_enemy_z_delta after T4 lands)
# is the better venue. Tracking gap as a follow-up.


def test_corridors_do_not_enter_spawn():
    from map import SIMPLE_ROOMS
    t = next(r for r in SIMPLE_ROOMS if r[0] == 5)
    ct = next(r for r in SIMPLE_ROOMS if r[0] == 7)
    assert t[4] == 416.0 or t[4] == 416, t
    assert ct[4] == 416.0 or ct[4] == 416, ct
    assert t[4] <= 416
    assert ct[4] <= 416
