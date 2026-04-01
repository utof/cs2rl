"""Smoke-test for export_mapdata: verifies the 5 normalization constants are
mathematically consistent and sane for de_dust2.

This test exercises the full make_cs2_map() → export_mapdata() pipeline, so
it requires the awpy nav data to be present on disk (same as all other integration
tests in this suite).  The constants here must be identical to what cs2_env.py
embeds at runtime — any drift breaks observation normalization in the C# plugin.
"""

import math
import sys
from pathlib import Path

# Ensure src/ is importable (nav.py, map.py live there)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


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
    # Import after sys.path is set so deploy/ modules resolve correctly
    sys.path.insert(0, str(Path(__file__).parent.parent / "deploy"))
    from export_mapdata import export_mapdata

    data = export_mapdata("de_dust2")

    # --- Identity fields ---
    assert data["obs_version"] == "v1-104dim", (
        f"obs_version mismatch: {data['obs_version']!r}"
    )
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
    expected_diag = math.sqrt((1.0 / inv_x) ** 2 + (1.0 / inv_y) ** 2)
    assert abs(map_diag - expected_diag) < 1e-3, (
        f"map_diag={map_diag:.4f} but expected {expected_diag:.4f} "
        f"(inv_x={inv_x:.6f}, inv_y={inv_y:.6f})"
    )

    # --- Sanity bounds for de_dust2 world coordinates ---
    # de_dust2 centroid X runs roughly -2476 to 2000, Y roughly -1050 to 3420.
    # offsets = (max + min) / (max - min); for dust2 these should be moderate.
    # If abs(offset) >= 5 something is badly wrong with the nav data or formula.
    assert -5.0 < x_off < 5.0, (
        f"x_offset={x_off:.3f} is outside [-5, 5] — nav bounds or formula wrong"
    )
    assert -5.0 < y_off < 5.0, (
        f"y_offset={y_off:.3f} is outside [-5, 5] — nav bounds or formula wrong"
    )

    # --- map_diag sanity ---
    # Half-diagonal of dust2's bounding rectangle should be ~3 000–5 000 world units.
    assert 500 < map_diag < 10000, (
        f"map_diag={map_diag:.1f} is outside [500, 10000] — suspicious nav bounds"
    )
