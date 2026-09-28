"""Smoke-test for export_mapdata: verifies the 5 normalization constants are
mathematically consistent and sane for de_dust2.

This test exercises the full make_cs2_map() → export_mapdata() pipeline, so
it requires the awpy nav data to be present on disk (same as all other integration
tests in this suite).  The constants here must be identical to what cs2_env.py
embeds at runtime — any drift breaks observation normalization in the C# plugin.
"""

import math
from pathlib import Path


def test_out_dir_is_the_deploy_mapdata_dir_the_plugin_reads():
    """OUT_DIR is <repo>/deploy/mapdata, the gitignored dir the sidecar has always gone to.

    WHY (#204, review C1): the output dir is computed from the module's own location,
    and moving it into src/cs2rl/deploy/ turned the old `Path(__file__).parent /
    "mapdata"` into src/cs2rl/deploy/mapdata: not gitignored, and not where the
    plugin's remedy message and the deploy docs send the user for the file.
    """
    from cs2rl.deploy.export_mapdata import OUT_DIR

    repo = Path(__file__).resolve().parents[1]
    assert OUT_DIR == repo / "deploy" / "mapdata", (
        f"export_mapdata would write to {OUT_DIR}, not {repo / 'deploy' / 'mapdata'}")


def test_export_mapdata_dust2():
    """Verify that export_mapdata produces correct, internally-consistent constants
    for de_dust2.

    Checks:
      1. obs_version and map name fields are present and correct.
      2. All float constants are positive and finite (required for safe division/normalization).
      3. map_diag == sqrt((1/inv_x)^2 + (1/inv_y)^2)  — internal consistency.
      4. x_offset / y_offset are within a plausible range for dust2 world coords.
      5. map_diag is within a plausible range for dust2 (~3 000–5 000 units).
    """
    from cs2rl.deploy.export_mapdata import export_mapdata

    data = export_mapdata("de_dust2")

    # --- Identity fields ---
    # Batch 3: bumped 104dim → 105dim (carrier-bit added at obs[104]).
    # Batch 5 (map-verticality T5): bumped v1→v2 to signal centroids_z presence.
    # Must stay in sync with cs2rl/deploy/export_mapdata.py:OBS_VERSION,
    # cs2rl/deploy/export_policy.py sidecar `obs_version`, and the C# plugin
    # ObservationBuilder.SupportedVersion.
    assert data["obs_version"] == "v2-105dim", (f"obs_version mismatch: {data['obs_version']!r}")
    assert data["map"] == "de_dust2", f"map field mismatch: {data['map']!r}"

    inv_x = data["inv_x_range"]
    inv_y = data["inv_y_range"]
    x_off = data["x_offset"]
    y_off = data["y_offset"]
    map_diag = data["map_diag"]

    # --- All float values must be finite ---
    # Non-finite constants (NaN/inf) would corrupt policy inputs in the C# plugin.
    for key, val in data.items():
        if isinstance(val, float):
            assert math.isfinite(val), f"{key} must be finite, got {val}"

    # inv_x, inv_y, and map_diag must be strictly positive (used as scale factors).
    # x_offset / y_offset CAN be negative (they depend on which half-space the map center is in).
    assert inv_x > 0, f"inv_x_range={inv_x} must be positive"
    assert inv_y > 0, f"inv_y_range={inv_y} must be positive"
    assert map_diag > 0, f"map_diag={map_diag} must be positive"

    # --- Internal consistency: map_diag must match formula ---
    # map_diag = sqrt((1/inv_x)^2 + (1/inv_y)^2)
    # 1/inv_x = (x_max - x_min) / 2.0  (the half-width in world units)
    expected_diag = math.sqrt((1.0 / inv_x)**2 + (1.0 / inv_y)**2)
    assert abs(map_diag -
               expected_diag) < 1e-3, (f"map_diag={map_diag:.4f} but expected {expected_diag:.4f} "
                                       f"(inv_x={inv_x:.6f}, inv_y={inv_y:.6f})")

    # --- Sanity bounds for de_dust2 world coordinates ---
    # de_dust2 centroid X runs roughly -2476 to 2000, Y roughly -1050 to 3420.
    # offsets = (max + min) / (max - min); for dust2 these should be moderate.
    # If abs(offset) >= 5 something is badly wrong with the nav data or formula.
    assert -5.0 < x_off < 5.0, (
        f"x_offset={x_off:.3f} is outside [-5, 5] — nav bounds or formula wrong")
    assert -5.0 < y_off < 5.0, (
        f"y_offset={y_off:.3f} is outside [-5, 5] — nav bounds or formula wrong")

    # --- map_diag sanity ---
    # Half-diagonal of dust2's bounding rectangle should be ~3 000–5 000 world units.
    assert 500 < map_diag < 10000, (
        f"map_diag={map_diag:.1f} is outside [500, 10000] — suspicious nav bounds")


def test_export_mapdata_includes_centroids_z():
    """Verify that the v2 sidecar contains a correctly-shaped centroids_z list.

    Batch 5 (map-verticality T5) added centroids_z to the sidecar so the C#
    plugin (T6) can look up per-area terrain elevation at deploy time.

    Checks:
      1. "centroids_z" key is present in the returned dict.
      2. Length matches the reported nav-area count N.
      3. All entries are numeric (int or float) — the C# plugin expects JSON
         number arrays; non-numeric values would silently corrupt elevation lookups.
      4. For de_dust2 all values are 0.0 — make_cs2_map zero-fills centroids_z
         because dust2 full verticality is deferred (spec §2 L1 / out-of-scope).
         This confirms plumbing is wired, not that real elevations are present.

    NOTE: is_ramp is intentionally absent from the sidecar; movement enforcement
    is handled by the CS2 server, not the deploy plugin.
    """
    from cs2rl.deploy.export_mapdata import export_mapdata

    data = export_mapdata("de_dust2")

    # --- centroids_z key must be present ---
    assert "centroids_z" in data, "sidecar dict is missing 'centroids_z' key"

    centroids_z = data["centroids_z"]

    # --- N must be present and match centroids_z length ---
    assert "N" in data, "sidecar dict is missing 'N' key"
    assert len(centroids_z) == data["N"], (
        f"centroids_z length {len(centroids_z)} != N={data['N']}")

    # --- Must be non-empty (dust2 has hundreds of nav areas) ---
    assert len(centroids_z) > 0, "centroids_z is empty — nav mesh not loaded?"

    # --- All entries must be numeric ---
    # Pitfall: if the astype(float).tolist() conversion ever produces strings or
    # None values (e.g., on a future custom map with bad data), the C# plugin would
    # silently read 0.0 and agents would clip through geometry.
    for i, z in enumerate(centroids_z):
        assert isinstance(z,
                          (int, float)), (f"centroids_z[{i}]={z!r} is not numeric; expected float")

    # --- de_dust2 stub: all zeros until full verticality data is added ---
    # make_cs2_map zero-fills centroids_z (spec §2 L1 / out-of-scope deferred).
    # If this assertion ever fails it means real elevation data was added and
    # the test should be updated to check actual values instead.
    assert all(z == 0.0
               for z in centroids_z), ("Expected all-zero centroids_z for de_dust2 (stub); "
                                       "if real verticality data was added, update this assertion")
