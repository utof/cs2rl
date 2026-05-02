#!/usr/bin/env python
"""Export map normalization constants from the CS2RL sim to a JSON sidecar.

Usage (run from repo root):
    python deploy/export_mapdata.py --map de_dust2

Reads MapData from the real dust2 nav mesh via make_cs2_map().
Writes deploy/mapdata/<map>.json with the 5 constants needed by ObservationBuilder.

Formulas (from src/c_env/cs2_env.py:312-315):
    inv_x = 2.0 / (x_max - x_min)
    inv_y = 2.0 / (y_max - y_min)
    x_off = (x_max + x_min) / (x_max - x_min)
    y_off = (y_max + y_min) / (y_max - y_min)
    map_diag = sqrt((1/inv_x)^2 + (1/inv_y)^2)  # from src/c_env/cs2_observations.h:15-17

IMPORTANT — normalization convention:
    The formula applied per-coordinate is:  norm = x * inv_range - offset
    NOT:  norm = (x - offset) * inv_range
    These are NOT equivalent. The C# plugin must match this convention.

The C# ObservationBuilder loads this JSON at startup and must validate
obs_version matches the baked-in constant before proceeding.
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Add src/ to sys.path so map.py and nav.py are importable without package install.
# We insert at position 0 so local src/ takes precedence over any installed packages.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import nav
from map import make_cs2_map

# This version tag must stay in sync with:
#   - deploy/export_policy.py  (obs_version field)
#   - C# plugin  ObservationBuilder.cs  (OBS_VERSION constant)
OBS_VERSION = "v1-105dim"


def export_mapdata(map_name: str) -> dict:
    """Compute normalization constants for the named map and return them as a dict.

    The returned dict has keys:
        map          — map name (e.g. "de_dust2")
        obs_version  — OBS_VERSION string for cross-validation with the ONNX sidecar
        inv_x_range  — 2 / (x_max - x_min); multiply raw X to get [-1, 1]
        inv_y_range  — 2 / (y_max - y_min); multiply raw Y to get [-1, 1]
        x_offset     — (x_max + x_min) / (x_max - x_min); subtract after multiplying
        y_offset     — (y_max + y_min) / (y_max - y_min); subtract after multiplying
        map_diag     — sqrt((1/inv_x)^2 + (1/inv_y)^2); used to normalize distances

    These are derived from the centroids of the loaded nav mesh — the same
    source make_cs2_map() uses in cs2_env.py:312-315. Any change to the nav mesh
    will update the bounds and therefore these constants.

    Args:
        map_name: Currently only "de_dust2" is supported.

    Returns:
        dict suitable for json.dump.

    Raises:
        ValueError: if map_name is not supported.
    """
    if map_name != "de_dust2":
        raise ValueError(f"Only de_dust2 is supported; got {map_name!r}")

    # make_cs2_map builds NavGraph, loads the vis matrix cache (or rebuilds it),
    # and returns a MapData with x_min/x_max/y_min/y_max from nav centroid bounds.
    # This is the authoritative source — identical to what cs2_env.py:82 uses.
    md = make_cs2_map(nav.NAV_PATH, nav.CACHE_PATH)

    # --- Normalization constants (cs2_env.py:312-315) ---
    # inv_x/inv_y: multiply raw world coordinate to get value in [-1, 1]
    inv_x = 2.0 / (md.x_max - md.x_min)
    inv_y = 2.0 / (md.y_max - md.y_min)

    # x_off/y_off: subtract after multiplication so midpoint maps to 0.0
    # norm = coord * inv_range - offset  (NOT (coord - offset) * inv_range)
    x_off = (md.x_max + md.x_min) / (md.x_max - md.x_min)
    y_off = (md.y_max + md.y_min) / (md.y_max - md.y_min)

    # half-widths of the map in world units; used to compute diagonal
    xr = 1.0 / inv_x                   # = (x_max - x_min) / 2.0
    yr = 1.0 / inv_y                   # = (y_max - y_min) / 2.0

    # map_diag: Euclidean half-diagonal in world units, used to normalize
    # distances (e.g. entity-to-entity range) into a [0, 1]-ish range.
    # Formula: cs2_observations.h:15-17
    map_diag = math.sqrt(xr * xr + yr * yr)

    print(f"[export_mapdata] Map bounds: X=[{md.x_min:.1f},{md.x_max:.1f}] "
          f"Y=[{md.y_min:.1f},{md.y_max:.1f}]")
    print(f"[export_mapdata] inv_x={inv_x:.6f} inv_y={inv_y:.6f}")
    print(f"[export_mapdata] x_off={x_off:.6f} y_off={y_off:.6f}")
    print(f"[export_mapdata] map_diag={map_diag:.2f}")

    return {
        "map": map_name,
        "obs_version": OBS_VERSION,
        "inv_x_range": float(inv_x),
        "inv_y_range": float(inv_y),
        "x_offset": float(x_off),
        "y_offset": float(y_off),
        "map_diag": float(map_diag),
    }


def main():
    """CLI entry point — parse --map, compute constants, write JSON sidecar."""
    parser = argparse.ArgumentParser(
        description="Export map normalization constants for the CS2RL plugin.")
    parser.add_argument(
        "--map",
        default="de_dust2",
        help="Map name (default: de_dust2).  Only de_dust2 is currently supported.",
    )
    args = parser.parse_args()

    data = export_mapdata(args.map)

    # Write to deploy/mapdata/<map>.json (deploy/mapdata/ is gitignored — runtime artifact)
    out_dir = Path(__file__).parent / "mapdata"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.map}.json"

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[export_mapdata] Written to {out_path}")


if __name__ == "__main__":
    main()
