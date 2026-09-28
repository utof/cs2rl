#!/usr/bin/env python
"""Export map normalization constants from the CS2RL sim to a JSON sidecar.

⚠ DEPLOY SUSPENDED 2026-05-03 ⚠ — active development paused after Batch 3.5
(sim-only training take-priority). Last-known-good OBS_VERSION=v2-105dim.
Do NOT bump OBS_VERSION or extend the sidecar schema as sim obs evolves; sim
should grow its own internal versioning independent of deploy. Tests stay
green to prevent silent bit-rot. See gh #(filed) for resume criteria.

Usage (run from repo root; from a worktree, prefix `env PYTHONPATH=<checkout>/src`):
    python -m cs2rl.deploy.export_mapdata --map de_dust2

Reads MapData from the real dust2 nav mesh via make_cs2_map().
Writes deploy/mapdata/<map>.json with the 5 constants needed by ObservationBuilder.

Formulas (from src/cs2rl/c_env/cs2_env.py:312-315):
    inv_x = 2.0 / (x_max - x_min)
    inv_y = 2.0 / (y_max - y_min)
    x_off = (x_max + x_min) / (x_max - x_min)
    y_off = (y_max + y_min) / (y_max - y_min)
    map_diag = sqrt((1/inv_x)^2 + (1/inv_y)^2)  # from src/cs2rl/c_env/cs2_observations.h:15-17

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
from pathlib import Path

from cs2rl import nav
from cs2rl.map import make_cs2_map

# This version tag must stay in sync with:
#   - cs2rl/deploy/export_policy.py  (obs_version field)
#   - C# plugin  ObservationBuilder.cs  (OBS_VERSION constant)
# Batch 5 (map-verticality T5): bumped v1→v2 to signal centroids_z is now
# present in the sidecar.  The C# plugin T6 will reject v1 exports that lack
# this field (spec §2 L4 / OBS_VERSION discovery row).
OBS_VERSION = "v2-105dim"

# Where main() writes <map>.json: <repo>/deploy/mapdata/, gitignored, where it has
# always been written and copied from to the server's mapdata/ dir that
# deploy/CS2RLBot/CS2RLBot.cs loads. This file is
# <repo>/src/cs2rl/deploy/export_mapdata.py, so parents[3] is the checkout root;
# `Path(__file__).parent` would now be src/cs2rl/deploy/, which is not gitignored
# (pinned by tests/test_export_mapdata.py).
OUT_DIR = Path(__file__).resolve().parents[3] / "deploy" / "mapdata"


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
        centroids_z  — list[float] of length N; per-area terrain elevation in world Z
                        units (0.0 for flat areas; populated by map-verticality work).
                        NOTE: is_ramp is NOT exported — movement enforcement is done by
                        the CS2 server, not the deploy plugin.

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

    # centroids_z: per-area terrain elevation (float32[N] → list[float]).
    # is_ramp is intentionally NOT exported — the CS2 server enforces movement
    # constraints at deploy time; the plugin has no use for the ramp flag.
    # Key must be "centroids_z" (snake_case) — T6 C# plugin looks for this exact key.
    centroids_z = md.centroids_z.astype(float).tolist()

    # N: number of nav areas.  Included so consumers can sanity-check that
    # centroids_z has the expected length without having to reload the nav mesh.
    return {
        "map": map_name,
        "obs_version": OBS_VERSION,
        "N": md.N,
        "inv_x_range": float(inv_x),
        "inv_y_range": float(inv_y),
        "x_offset": float(x_off),
        "y_offset": float(y_off),
        "map_diag": float(map_diag),
        "centroids_z": centroids_z,
    }


def main():
    """CLI entry point — parse --map, compute constants, write JSON sidecar."""
    parser = argparse.ArgumentParser(
        prog="python -m cs2rl.deploy.export_mapdata",
        description="Export map normalization constants for the CS2RL plugin.")
    parser.add_argument(
        "--map",
        default="de_dust2",
        help="Map name (default: de_dust2).  Only de_dust2 is currently supported.",
    )
    args = parser.parse_args()

    data = export_mapdata(args.map)

    # Write to deploy/mapdata/<map>.json (deploy/mapdata/ is gitignored — runtime artifact)
    out_dir = OUT_DIR
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.map}.json"

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[export_mapdata] Written to {out_path}")


if __name__ == "__main__":
    main()
