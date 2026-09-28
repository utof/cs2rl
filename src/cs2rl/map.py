"""map.py — MapData abstraction for the CS2RL C environment.

Provides:
  MapData           — typed container for all geometry/game-constants the C env needs
  make_cs2_map()    — builds MapData from the real dust2 nav mesh (requires awpy/CS2 data)
  make_simple_map() — builds MapData from a list of rectangular rooms (no awpy needed)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from cs2rl.nav import (
    _A_SITE,
    _B_SITE,
    _CT_SPAWN_SLOTS,
    _T_SPAWN_SLOTS,
    TEAM_SIZE,
    NavGraph,
    _areas_near,
    _build_area_adjacency,
    _compute_area_distance_to_targets,
    _select_distinct_spawn_areas,
)


@dataclass
class MapData:
    # Core geometry (area_idx-indexed, 0..N-1)
    N: int
    area_ids: np.ndarray               # int32[N] — for simple maps, just np.arange(N)
    centroids: np.ndarray              # float32[N, 2] — XY
    vis_matrix: np.ndarray             # bool[N, N]
    adjacency: np.ndarray              # bool[N, N]

    # Verticality — per-area terrain elevation and ramp flag (spec L1, L8).
    # centroids_z: terrain z-height for each area (0.0 for flat/ground areas).
    #   - Ramps store their TOP elevation (e.g., a ramp from z=0 to z=64 has
    #     centroids_z=64.0). Cliff-guard Δz still uses that top. Grounded z
    #     interpolates along the room quad when MapData.area_bounds is set
    #     (simple map). NULL bounds (dust2 / make_cs2_map) still snap to top.
    #   - make_cs2_map zero-fills this; dust2 verticality is a separate future task.
    # is_ramp: True means the C-env cliff guard exempts this area from the
    #   Δz > SV_MAX_STEP_HEIGHT check, allowing grounded agents to step up into it.
    #   Pitfall: is_ramp=True on BOTH endpoint areas would let agents climb non-ramp
    #   cliffs — only mark the ramp/stairs area, not the destination platform.
    centroids_z: np.ndarray            # float32[N] — per-area terrain elevation (z), 0.0 for flat
    is_ramp: np.ndarray                # bool[N]   — area is a ramp/stairs (cliff-guard exemption)

    # Raster grid
    grid: np.ndarray                   # int32[H, W] — cell → area_idx (-1 = off-mesh)
    grid_x_min: float
    grid_y_min: float
    grid_cell_size: float              # grid_inv_cell = 1.0 / grid_cell_size

    # Bombsite data (area_id-indexed; for simple maps area_id == area_idx)
    bombsite_mask: np.ndarray          # bool[max_area_id+1]
    bombsite_by_idx: np.ndarray        # int8[N]
    bombsite_dist: np.ndarray          # float32[max_area_id+1]
    bombsite_dist_scale: float

    # Spawns as area_ids (not indices)
    t_spawn_areas: list                # len 1–15
    ct_spawn_areas: list               # len 1–5

    # Map bounds (used to compute normalization constants in Cs2Env.__init__)
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    # Optional: reference to the underlying NavGraph (needed for viz/snapshot).
    # None for simple maps.
    nav_graph: object = field(default=None, repr=False)
    # Room AABB float32[N,4] x0,y0,x1,y1 for ramp interpolation.
    # make_simple_map fills this from the room tuples. make_cs2_map leaves
    # None so demo_terrain_z stays on centroids_z.
    area_bounds: np.ndarray | None = field(default=None)

    def line_of_sight_2d(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Pure-Python mirror of cs2_combat.h::line_of_sight_2d.

        Walks raster grid cells from (x1,y1) to (x2,y2) via Amanatides-Woo
        DDA. Returns True iff every cell along the line has raster_grid >= 0
        AND every area transition along the path has adjacency[prev,curr]=1.

        Used by tests + diagnostic tooling. Runtime combat / obs / memory go
        through the C function; this Python copy must stay in numerical
        lock-step with the C version. If you change the C algorithm, update
        this too — there's no shared source of truth (the only way to share
        across language boundaries would be to expose the C function via the
        binding, which adds maintenance for marginal benefit since the cost
        is dominated by the per-cell lookup not the function-call overhead).

        Why tests need this: the centroid-baked self.vis_matrix gives a
        coarse "are these areas roughly visible to each other" signal that
        doesn't necessarily match what the C raycast says about specific
        positions inside those areas. Tests that need "agent at position A
        can shoot agent at position B" must check this function, not
        vis_matrix, after the build_vis_matrix C-side change in cs2_combat.h.

        Termination: explicit step cap (|Δgx| + |Δgy|) avoids the float
        bookkeeping bug where t_max overshoot makes a near-axis-aligned
        line never reach exact (gx_end, gy_end) equality.
        """
        cell = self.grid_cell_size
        H, W = self.grid.shape
        fx0 = (x1 - self.grid_x_min) / cell
        fy0 = (y1 - self.grid_y_min) / cell
        fx1 = (x2 - self.grid_x_min) / cell
        fy1 = (y2 - self.grid_y_min) / cell
        gx, gy = int(fx0), int(fy0)
        gxe, gye = int(fx1), int(fy1)
        if not (0 <= gx < W and 0 <= gy < H and 0 <= gxe < W and 0 <= gye < H):
            return False
        prev = int(self.grid[gy, gx])
        if prev < 0:
            return False
        if gx == gxe and gy == gye:
            return True
        dx, dy = fx1 - fx0, fy1 - fy0
        sx = 1 if dx > 0 else (-1 if dx < 0 else 0)
        sy = 1 if dy > 0 else (-1 if dy < 0 else 0)
        tdx = abs(1.0 / dx) if sx else 1e30
        tdy = abs(1.0 / dy) if sy else 1e30
        tmx = ((gx + 1) - fx0) * tdx if sx > 0 else ((fx0 - gx) * tdx if sx < 0 else 1e30)
        tmy = ((gy + 1) - fy0) * tdy if sy > 0 else ((fy0 - gy) * tdy if sy < 0 else 1e30)
        max_steps = abs(gxe - gx) + abs(gye - gy)
        for _ in range(max_steps):
            if gx == gxe and gy == gye:
                break
            if tmx < tmy:
                tmx += tdx
                gx += sx
            else:
                tmy += tdy
                gy += sy
            if not (0 <= gx < W and 0 <= gy < H):
                return False
            cur = int(self.grid[gy, gx])
            if cur < 0:
                return False
            if cur != prev:
                if not self.adjacency[prev, cur]:
                    return False
                prev = cur
        return True


_CS2_MAP_CACHE: dict = {}


def make_cs2_map(nav_path: str, cache_path: str, *, build_vis: bool = True) -> MapData:
    """Build MapData from the real dust2 nav mesh.

    ``build_vis=False`` (gh#251) skips NavGraph.build_vis_matrix and returns an
    UNCACHED MapData whose ``vis_matrix`` is None. It exists for
    `train.py --dump-config`, which only reads geometry (centroids_z → pin_pitch)
    and must never fork: on a cold `src/vis_cache.npy` (every fresh worktree —
    it is gitignored) build_vis_matrix spawns a cpu_count()-worker
    ProcessPoolExecutor for minutes, and a killed dump orphaned all 12 workers
    to PID 1 at ~900 MB each.

    PITFALL: never hand a build_vis=False MapData to an env (vis_matrix None) and
    never cache it — a later build_vis=True caller for the same key would get
    the vis-less object. A warm cached full MapData IS returned for either flag.
    """
    key = (nav_path, cache_path)
    cached = _CS2_MAP_CACHE.get(key)
    if cached is not None:
        return cached

    nav_graph = NavGraph(nav_path, cache_path)
    if build_vis:
        nav_graph.build_vis_matrix()

    xs = [c[0] for c in nav_graph.centroids.values()]
    ys = [c[1] for c in nav_graph.centroids.values()]
    map_bounds = (min(xs), max(xs), min(ys), max(ys))
    x_min, x_max, y_min, y_max = map_bounds

    a_site_areas = _areas_near(nav_graph, _A_SITE, 600)
    b_site_areas = _areas_near(nav_graph, _B_SITE, 600)
    bombsite_areas = set(a_site_areas + b_site_areas)
    bombsite_targets = tuple(bombsite_areas)

    area_adjacency = _build_area_adjacency(nav_graph)
    bombsite_area_distance = _compute_area_distance_to_targets(
        nav_graph,
        area_adjacency,
        bombsite_targets,
    )

    t_spawn_areas = _select_distinct_spawn_areas(
        nav_graph,
        _T_SPAWN_SLOTS,
        TEAM_SIZE,
        required_targets=bombsite_targets,
        area_adjacency=area_adjacency,
    )
    ct_spawn_areas = _select_distinct_spawn_areas(
        nav_graph,
        _CT_SPAWN_SLOTS,
        TEAM_SIZE,
        required_targets=bombsite_targets,
        area_adjacency=area_adjacency,
    )

    max_area_id = max(nav_graph.area_ids)
    bombsite_mask = np.zeros(max_area_id + 1, dtype=bool)
    bombsite_mask[np.asarray(list(bombsite_areas), dtype=np.int32)] = True

    bombsite_dist = np.full(max_area_id + 1, np.inf, dtype=np.float32)
    for area_id in nav_graph.area_ids:
        bombsite_dist[area_id] = bombsite_area_distance[nav_graph._id_to_idx[area_id]]

    finite_dist = bombsite_dist[np.isfinite(bombsite_dist)]
    bombsite_dist_scale = 0.0
    if finite_dist.size:
        max_dist = float(finite_dist.max())
        bombsite_dist_scale = 1.0 / max_dist if max_dist > 0 else 0.0
    # R0-F (#136): scale is computed from the FINITE entries above; only now
    # replace non-finite hops (area-id gaps + unreachable areas) with 4×max so
    # closeness = 1 − 4 < 0 → clamps to 0 in C exactly as the old isfinite()
    # skip did (isfinite folds to true under -ffast-math and leaked inf).
    # PITFALL: never fill before computing the scale — the sentinel would
    # shrink it 4× and silently rescale every nav reward. No finite entry
    # (bombsites=[]) ⇒ leave the array all-inf and scale 0.0; the C guard on
    # scale > 0 handles it.
    if finite_dist.size:
        bombsite_dist = np.where(np.isfinite(bombsite_dist), bombsite_dist,
                                 4.0 * max_dist).astype(np.float32)

    bm_int8 = bombsite_mask.astype(np.int8)
    bombsite_by_idx = np.array(
        [int(bm_int8[aid]) if 0 <= aid < len(bm_int8) else 0 for aid in nav_graph.area_ids],
        dtype=np.int8,
    )

    # Verticality deferred for real dust2 (spec §3 out-of-scope: "Real CS2 dust2
    # verticality" is a separate large-scope task tied to nav-mesh-Z parsing from awpy).
    # Zero-fill keeps the MapData contract uniform and lets the C env init succeed.
    centroids_z_cs2 = np.zeros(nav_graph.N, dtype=np.float32)
    is_ramp_cs2 = np.zeros(nav_graph.N, dtype=bool)

    map_data = MapData(
        N=nav_graph.N,
        area_ids=np.array(nav_graph.area_ids, dtype=np.int32),
        centroids=nav_graph._centroid_matrix.astype(np.float32),
        vis_matrix=nav_graph.vis_matrix,
        adjacency=area_adjacency,
        centroids_z=centroids_z_cs2,
        is_ramp=is_ramp_cs2,
        grid=nav_graph._pos_grid,
        grid_x_min=float(nav_graph._grid_x_min),
        grid_y_min=float(nav_graph._grid_y_min),
        grid_cell_size=1.0 / float(nav_graph._grid_inv_cell),
        bombsite_mask=bombsite_mask,
        bombsite_by_idx=bombsite_by_idx,
        bombsite_dist=bombsite_dist,
        bombsite_dist_scale=bombsite_dist_scale,
        t_spawn_areas=list(t_spawn_areas),
        ct_spawn_areas=list(ct_spawn_areas),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        nav_graph=nav_graph,
    )

    if build_vis:                      # vis-less MapData is never cached (see docstring)
        _CS2_MAP_CACHE[key] = map_data

    print(f"[MapData] Map bounds: X=[{x_min:.0f},{x_max:.0f}] Y=[{y_min:.0f},{y_max:.0f}]")
    print(f"[MapData] T-spawn: {len(t_spawn_areas)} areas  "
          f"CT-spawn: {len(ct_spawn_areas)} areas  "
          f"A-site: {len(a_site_areas)} areas  "
          f"B-site: {len(b_site_areas)} areas")
    return map_data


# ── Simple map definition ────────────────────────────────────────────────────
# Schema (spec L8): (area_idx, x0, y0, x1, y1, z, is_ramp) — world-space floats.
# z      — terrain elevation; ramps store their TOP elevation (spec L1).
# is_ramp — True exempts the area from the C-env cliff-guard (agents may step up
#            into it even if Δz > SV_MAX_STEP_HEIGHT_CS).
#
# Geometry per spec §4 Option A:
#   y_min=80 (new northern strip houses catwalk+stairs), y_max=1184 (unchanged).
#   Bombsite elevated to z=64. Dual ramps (T and CT side). Catwalk at z=128 (y=80-192).
#   Stairs (is_ramp=True) connect CT-corridor to catwalk.
#   T-corridor and CT-corridor x-extents shrunk to make room for ramps.
#
# BACKWARD-COMPAT NOTE: The 5-tuple schema is not preserved — this is an atomic
# schema bump; all callers that iterate over SIMPLE_ROOMS must use 7-tuple unpacking
# (or `for idx, x0, y0, x1, y1, *_ in rooms` for forward-compat ignoring z/is_ramp).
SIMPLE_ROOMS = [
                                                       # T-spawn cluster (areas 0-4) — flat, z=0
    (0, 0, 416, 256, 672, 0.0, False),                 # T-spawn-A
    (1, 256, 416, 512, 672, 0.0, False),               # T-spawn-B
    (2, 0, 672, 256, 928, 0.0, False),                 # T-spawn-C
    (3, 256, 672, 512, 928, 0.0, False),               # T-spawn-D
    (4, 0, 928, 512, 1184, 0.0, False),                # T-spawn-E
    (5, 400, 192, 750, 416, 0.0, False),               # T-corridor — stops at T-spawn south
                                                       # Bombsite — ELEVATED to z=64; shrunk on x edges (was 800-1100; now 820-1100)
    (6, 820, 192, 1100, 416, 64.0, False),             # Bombsite (elevated)
    (7, 1170, 192, 1600, 416, 0.0, False),             # CT-corridor — stops at CT-spawn south

                                                       # CT-spawn cluster (areas 8-12) — flat, z=0
    (8, 1500, 416, 1756, 672, 0.0, False),
    (9, 1756, 416, 2012, 672, 0.0, False),
    (10, 1500, 672, 1756, 928, 0.0, False),
    (11, 1756, 672, 2012, 928, 0.0, False),
    (12, 1500, 928, 2012, 1184, 0.0, False),
                                                       # T-ramp — ramp from T-corridor (z=0) up to bombsite (z=64); top elevation per spec L1
    (13, 750, 192, 820, 416, 64.0, True),              # T-ramp
                                                       # CT-ramp — ramp from CT-corridor (z=0) up to bombsite (z=64); top elevation
    (14, 1100, 192, 1170, 416, 64.0, True),            # CT-ramp
                                                       # Catwalk — z=128, NEW northern strip (y=80..192) overlooking bombsite from above
    (15, 820, 80, 1170, 192, 128.0, False),            # Catwalk
                                                       # Stairs — z=128, ramp connecting CT-corridor (z=0) up to catwalk (z=128);
                                                       #   is_ramp=True so cliff guard lets agents walk up from CT-corridor to catwalk
    (16, 1170, 80, 1300, 192, 128.0, True),            # Stairs
]
SIMPLE_T_SPAWNS = [0, 1, 2, 3, 4]
SIMPLE_CT_SPAWNS = [8, 9, 10, 11, 12]
SIMPLE_BOMBSITES = [6]

# ── R0-H: Rung 1 duel arena (spec 2026-08-29 §3 R0-H) ─────────────────────
# WHAT: one flat 600×400u rectangle gridded 6×4 into 100u areas
# (idx = row*6 + col, centroids x ∈ {50..550}, y ∈ {50..350}, z ≡ 0, no ramps).
# T spawns = the x=150 column (idx 1,7,13,19); CT spawns = x=350 (3,9,15,21).
# Four rows per side ⇒ 16 pairings, gaps 200–360.6u, opening bearings one of
# 7 signed values (span 112.6°) that T/CT must READ from the enemy-bearing
# obs (R0-E.1). No bombsites ⇒ bombsite_dist_scale == 0.0 and the R0-F C
# guards skip every bombsite potential.
# WHY the row randomisation is LOAD-BEARING: with one fixed spawn per side the
# opening obs is bit-identical every round and a constant Δyaw — reachable
# through the aim head's bias alone — would pass every §5 gate without ever
# reading the obs. tests/test_arena_duel.py pins this with a bias-only search.
# PITFALLS: cell_size=20 divides 100 exactly; the default 16 does not and
# make_simple_map's floor/ceil raster would overlap adjacent areas by a cell.
# spawn_team's `n_spawns < TEAM_SIZE` branch draws `xorshift32 % n_spawns` per
# agent, so 4 spawns add no RNG draw versus one (n=5 bit-exactness intact);
# ≥ TEAM_SIZE spawns per side would switch to the shuffle path (Rung 2).
# StaticData caps: t_spawns[15] / ct_spawns[5] (cs2_types.h) — asymmetric.
# yapf: disable
ARENA_DUEL_V1 = {
    "rooms": [(r * 6 + c, c * 100.0, r * 100.0, (c + 1) * 100.0, (r + 1) * 100.0, 0.0, False)
              for r in range(4) for c in range(6)],
    "t_spawns": [1, 7, 13, 19],
    "ct_spawns": [3, 9, 15, 21],
    "bombsites": [],
    "cell_size": 20.0,
}
# yapf: enable

# Maximum grounded up-step (Source `sv_stepsize` default = 18u). MUST stay numerically
# in lock-step with `SV_MAX_STEP_HEIGHT_CS` in src/cs2rl/c_env/cs2_movement.h (added in T3).
# Drift between the two breaks the env: the cliff guard would refuse a movement that
# nav-shaping treats as a shortcut, or vice versa.
MAX_STEP_HEIGHT = 18.0


def make_simple_map(
    rooms=SIMPLE_ROOMS,
    t_spawns=SIMPLE_T_SPAWNS,
    ct_spawns=SIMPLE_CT_SPAWNS,
    bombsites=SIMPLE_BOMBSITES,
    cell_size: float = 16.0,
) -> MapData:
    """Build a MapData from rectangular rooms with no awpy/CS2 dependency.

    Parameters
    ----------
    rooms      : list of (area_idx, x0, y0, x1, y1, z, is_ramp) — 7-tuple schema (spec L8).
                 z is terrain elevation (float); is_ramp is bool (cliff-guard exemption).
    t_spawns   : list of area_idx values that are T-spawn areas
    ct_spawns  : list of area_idx values that are CT-spawn areas
    bombsites  : list of area_idx values that are bombsite areas
    cell_size  : world units per raster cell

    Note: the 5-tuple schema (area_idx, x0, y0, x1, y1) used before Batch 5 is NOT
    backward-compatible — callers passing custom rooms must use the 7-tuple form.
    """
    N = len(rooms)
    area_ids = np.arange(N, dtype=np.int32)            # area_id == area_idx for simple maps

    # 1. Derive map bounds from rooms — rooms are (idx, x0, y0, x1, y1, z, is_ramp).
    # Use positional index to avoid requiring full 7-tuple if z/is_ramp are absent.
    x_min = float(min(r[1] for r in rooms))
    y_min = float(min(r[2] for r in rooms))
    x_max = float(max(r[3] for r in rooms))
    y_max = float(max(r[4] for r in rooms))

    # 2. Centroids (rect centres) and verticality arrays.
    # z, is_ramp unpacked from tuple positions 5 and 6.
    centroids = np.zeros((N, 2), dtype=np.float32)
    centroids_z = np.zeros(N, dtype=np.float32)
    is_ramp = np.zeros(N, dtype=bool)
    for idx, x0, y0, x1, y1, z, ramp in rooms:
        centroids[idx, 0] = (x0 + x1) * 0.5
        centroids[idx, 1] = (y0 + y1) * 0.5
        centroids_z[idx] = z
        is_ramp[idx] = ramp

    # 3. Raster grid: world coord → area_idx (-1 = off-mesh)
    grid_w = int(np.ceil((x_max - x_min) / cell_size))
    grid_h = int(np.ceil((y_max - y_min) / cell_size))
    grid = np.full((grid_h, grid_w), -1, dtype=np.int32)
    for idx, x0, y0, x1, y1, *_ in rooms:              # *_ ignores z, is_ramp
        col0 = int((x0 - x_min) / cell_size)
        col1 = int(np.ceil((x1 - x_min) / cell_size))
        row0 = int((y0 - y_min) / cell_size)
        row1 = int(np.ceil((y1 - y_min) / cell_size))
        grid[row0:row1, col0:col1] = idx

    # 4. Adjacency: two areas are adjacent if any of their raster cells are 8-neighbors.
    # Diagonal must be True: movement within the same area is always valid.
    adjacency = np.zeros((N, N), dtype=bool)
    np.fill_diagonal(adjacency, True)
    rows, cols = np.where(grid >= 0)
    for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
        a = grid[r, c]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_h and 0 <= nc < grid_w:
                    b = grid[nr, nc]
                    if b >= 0 and b != a:
                        adjacency[a, b] = True
                        adjacency[b, a] = True

    # 4b. L9 adjacency post-prune (spec §2 L9): remove edges where BOTH endpoints
    # are non-ramp AND |Δz| > MAX_STEP_HEIGHT. This mirrors the C-env cliff guard
    # (cs2_movement.h SV_MAX_STEP_HEIGHT_CS) so that nav-distance shaping does not
    # assign shortcut bonuses for movement edges that the C env will physically refuse.
    # Pitfall: only prune non-ramp↔non-ramp cliff edges — ramp targets are always
    # allowed (is_ramp=True is the explicit walk-up affordance, spec L11).
    # MAX_STEP_HEIGHT (module constant) MUST numerically match SV_MAX_STEP_HEIGHT_CS in
    # cs2_movement.h. See module-level constant for full rationale.
    for i in range(N):
        for j in range(N):
            if i == j or not adjacency[i, j]:
                continue
            if is_ramp[i] or is_ramp[j]:
                continue               # ramp endpoints exempt from cliff pruning
            if abs(centroids_z[i] - centroids_z[j]) > MAX_STEP_HEIGHT:
                adjacency[i, j] = False

    # 5. Visibility: Bresenham LOS centroid-to-centroid; blocked by cells == -1
    def _bresenham_visible(c0, c1) -> bool:
        x0, y0 = c0
        x1, y1 = c1
        col0 = int((x0 - x_min) / cell_size)
        row0 = int((y0 - y_min) / cell_size)
        col1 = int((x1 - x_min) / cell_size)
        row1 = int((y1 - y_min) / cell_size)
        dx, dy = abs(col1 - col0), abs(row1 - row0)
        sc = 1 if col1 > col0 else -1
        sr = 1 if row1 > row0 else -1
        err = dx - dy
        cc, cr = col0, row0
        while True:
            if not (0 <= cr < grid_h and 0 <= cc < grid_w):
                break
            if grid[cr, cc] < 0:
                return False
            if cc == col1 and cr == row1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cc += sc
            if e2 < dx:
                err += dx
                cr += sr
        return True

    vis_matrix = np.zeros((N, N), dtype=bool)
    for i in range(N):
        vis_matrix[i, i] = True
        for j in range(i + 1, N):
            v = _bresenham_visible(centroids[i], centroids[j])
            vis_matrix[i, j] = v
            vis_matrix[j, i] = v

    # 6. Bombsite data
    bombsite_set = set(bombsites)
    bombsite_mask = np.zeros(N, dtype=bool)            # area_id == area_idx
    bombsite_mask[np.array(list(bombsite_set), dtype=np.int32)] = True
    bombsite_by_idx = bombsite_mask.astype(np.int8)

    # BFS from bombsite cells to compute hop-distance per area
    dist_hops = np.full(N, np.inf, dtype=np.float32)
    for b in bombsite_set:
        dist_hops[b] = 0.0
    queue: deque = deque(b for b in bombsite_set)
    while queue:
        cur = queue.popleft()
        for nxt in range(N):
            if adjacency[cur, nxt] and dist_hops[nxt] == np.inf:
                dist_hops[nxt] = dist_hops[cur] + 1.0
                queue.append(nxt)

    bombsite_dist = dist_hops                                               # float32[N] (area_id == area_idx)
    finite = bombsite_dist[np.isfinite(bombsite_dist)]
    bombsite_dist_scale = 0.0
    if finite.size:
        mx = float(finite.max())
        bombsite_dist_scale = 1.0 / mx if mx > 0 else 0.0
                                                                            # R0-F (#136): same sentinel fill as the dust2 path — see comment there.
                                                                            # Scale first (from finite entries), then inf → 4×max (finite, clamps to 0).
    if finite.size:
        bombsite_dist = np.where(np.isfinite(bombsite_dist), bombsite_dist,
                                 4.0 * mx).astype(np.float32)

    area_bounds = np.zeros((N, 4), dtype=np.float32)
    for idx, x0, y0, x1, y1, *_ in rooms:
        area_bounds[idx] = (x0, y0, x1, y1)

    return MapData(
        N=N,
        area_ids=area_ids,
        centroids=centroids,
        vis_matrix=vis_matrix,
        adjacency=adjacency,
        centroids_z=centroids_z,
        is_ramp=is_ramp,
        grid=grid,
        grid_x_min=x_min,
        grid_y_min=y_min,
        grid_cell_size=cell_size,
        bombsite_mask=bombsite_mask,
        bombsite_by_idx=bombsite_by_idx,
        bombsite_dist=bombsite_dist,
        bombsite_dist_scale=bombsite_dist_scale,
        t_spawn_areas=list(t_spawns),
        ct_spawn_areas=list(ct_spawns),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        nav_graph=None,
        area_bounds=area_bounds,
    )


def make_arena_duel_map() -> MapData:
    """R0-H: build ARENA_DUEL_V1 (see the preset comment for what/why/pitfalls).

    Returns a MapData exactly as make_simple_map produces it — flat (centroids_z
    ≡ 0 ⇒ train.pin_pitch_for_map == 1), all 24 areas mutually visible, no
    bombsite. Raises RuntimeError (not assert: python -O strips asserts) if the
    preset ever drifts from the 6×4 / 4+4-spawn contract the sim and tests pin.
    """
    p = ARENA_DUEL_V1
    md = make_simple_map(rooms=p["rooms"],
                         t_spawns=p["t_spawns"],
                         ct_spawns=p["ct_spawns"],
                         bombsites=p["bombsites"],
                         cell_size=p["cell_size"])
    if md.N != 24 or len(md.t_spawn_areas) != 4 or len(md.ct_spawn_areas) != 4:
        raise RuntimeError(f"ARENA_DUEL_V1 drifted: N={md.N} t_spawns={len(md.t_spawn_areas)} "
                           f"ct_spawns={len(md.ct_spawn_areas)} (expected 24 / 4 / 4)")
    return md
