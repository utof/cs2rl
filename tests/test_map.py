"""tests/test_map.py — MapData verticality (Batch 5 prerequisite for Δpitch).

Verifies the simple map's verticality fields (centroids_z, is_ramp) are coherent,
the elevated bombsite is reachable on foot, vis matrix sees catwalk-bombsite, and
the L9 adjacency post-prune removes cliff edges from the nav graph.

T1 tests (steps 1.8-1.10): 5 pure-Python MapData tests, no C env required.
T3 tests (step 3.7): 5 movement/behaviour tests requiring the C env — added later.
T4 tests (step 4.5): 1 obs z-delta test — added in T4.
"""
import sys
from pathlib import Path

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
        "catwalk↔bombsite cliff edge must be pruned by L9; "
        "if this fails, check the post-prune loop in make_simple_map")


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
        "T-corridor → T-ramp must remain adjacent after L9 pruning (ramp exemption); "
        "check is_ramp[13] == True in SIMPLE_ROOMS")
