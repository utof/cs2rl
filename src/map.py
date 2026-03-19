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

from sim import (
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
    area_ids: np.ndarray  # int32[N] — for simple maps, just np.arange(N)
    centroids: np.ndarray  # float32[N, 2] — XY
    vis_matrix: np.ndarray  # bool[N, N]
    adjacency: np.ndarray  # bool[N, N]

    # Raster grid
    grid: np.ndarray  # int32[H, W] — cell → area_idx (-1 = off-mesh)
    grid_x_min: float
    grid_y_min: float
    grid_cell_size: float  # grid_inv_cell = 1.0 / grid_cell_size

    # Bombsite data (area_id-indexed; for simple maps area_id == area_idx)
    bombsite_mask: np.ndarray  # bool[max_area_id+1]
    bombsite_by_idx: np.ndarray  # int8[N]
    bombsite_dist: np.ndarray  # float32[max_area_id+1]
    bombsite_dist_scale: float

    # Spawns as area_ids (not indices)
    t_spawn_areas: list  # len 1–15
    ct_spawn_areas: list  # len 1–5

    # Map bounds (used to compute normalization constants in wrapper.py)
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    # Optional: reference to the underlying NavGraph (needed for viz/snapshot).
    # None for simple maps.
    nav_graph: object = field(default=None, repr=False)


_CS2_MAP_CACHE: dict = {}


def make_cs2_map(nav_path: str, cache_path: str) -> MapData:
    """Build MapData from the real dust2 nav mesh."""
    key = (nav_path, cache_path)
    cached = _CS2_MAP_CACHE.get(key)
    if cached is not None:
        return cached

    nav_graph = NavGraph(nav_path, cache_path)
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

    bm_int8 = bombsite_mask.astype(np.int8)
    bombsite_by_idx = np.array(
        [int(bm_int8[aid]) if 0 <= aid < len(bm_int8) else 0 for aid in nav_graph.area_ids],
        dtype=np.int8,
    )

    map_data = MapData(
        N=nav_graph.N,
        area_ids=np.array(nav_graph.area_ids, dtype=np.int32),
        centroids=nav_graph._centroid_matrix.astype(np.float32),
        vis_matrix=nav_graph.vis_matrix,
        adjacency=area_adjacency,
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

    _CS2_MAP_CACHE[key] = map_data

    print(f"[MapData] Map bounds: X=[{x_min:.0f},{x_max:.0f}] Y=[{y_min:.0f},{y_max:.0f}]")
    print(
        f"[MapData] T-spawn: {len(t_spawn_areas)} areas  "
        f"CT-spawn: {len(ct_spawn_areas)} areas  "
        f"A-site: {len(a_site_areas)} areas  "
        f"B-site: {len(b_site_areas)} areas"
    )
    return map_data


# ── Simple map definition ────────────────────────────────────────────────────
# Each room: (area_idx, x0, y0, x1, y1)  — coordinates are world-space floats.
SIMPLE_ROOMS = [
    # T-spawn cluster (areas 0–4) — placed ABOVE the approach corridor (y > 384)
    (0, 0, 416, 256, 672),  # T-spawn-A
    (1, 256, 416, 512, 672),  # T-spawn-B
    (2, 0, 672, 256, 928),  # T-spawn-C
    (3, 256, 672, 512, 928),  # T-spawn-D
    (4, 0, 928, 512, 1184),  # T-spawn-E
    # T-side approach corridor — narrow horizontal band
    (5, 400, 192, 800, 416),  # T-corridor
    # Bombsite — narrow horizontal band matching corridor height
    (6, 800, 192, 1100, 416),  # Bombsite
    # CT-side approach corridor — narrow horizontal band
    (7, 1100, 192, 1500, 416),  # CT-corridor
    # CT-spawn cluster (areas 8–12) — placed ABOVE the approach corridor (y > 416)
    (8, 1500, 416, 1756, 672),  # CT-spawn-A
    (9, 1756, 416, 2012, 672),  # CT-spawn-B
    (10, 1500, 672, 1756, 928),  # CT-spawn-C
    (11, 1756, 672, 2012, 928),  # CT-spawn-D
    (12, 1500, 928, 2012, 1184),  # CT-spawn-E
]
SIMPLE_T_SPAWNS = [0, 1, 2, 3, 4]
SIMPLE_CT_SPAWNS = [8, 9, 10, 11, 12]
SIMPLE_BOMBSITES = [6]


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
    rooms      : list of (area_idx, x0, y0, x1, y1)
    t_spawns   : list of area_idx values that are T-spawn areas
    ct_spawns  : list of area_idx values that are CT-spawn areas
    bombsites  : list of area_idx values that are bombsite areas
    cell_size  : world units per raster cell
    """
    N = len(rooms)
    area_ids = np.arange(N, dtype=np.int32)  # area_id == area_idx for simple maps

    # 1. Derive map bounds from rooms — rooms are (idx, x0, y0, x1, y1)
    x_min = float(min(r[1] for r in rooms))
    y_min = float(min(r[2] for r in rooms))
    x_max = float(max(r[3] for r in rooms))
    y_max = float(max(r[4] for r in rooms))

    # 2. Centroids (rect centres)
    centroids = np.zeros((N, 2), dtype=np.float32)
    for idx, x0, y0, x1, y1 in rooms:
        centroids[idx, 0] = (x0 + x1) * 0.5
        centroids[idx, 1] = (y0 + y1) * 0.5

    # 3. Raster grid: world coord → area_idx (-1 = off-mesh)
    grid_w = int(np.ceil((x_max - x_min) / cell_size))
    grid_h = int(np.ceil((y_max - y_min) / cell_size))
    grid = np.full((grid_h, grid_w), -1, dtype=np.int32)
    for idx, x0, y0, x1, y1 in rooms:
        col0 = int((x0 - x_min) / cell_size)
        col1 = int(np.ceil((x1 - x_min) / cell_size))
        row0 = int((y0 - y_min) / cell_size)
        row1 = int(np.ceil((y1 - y_min) / cell_size))
        grid[row0:row1, col0:col1] = idx

    # 4. Adjacency: two areas are adjacent if any of their raster cells are 8-neighbors
    adjacency = np.zeros((N, N), dtype=bool)
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
    bombsite_mask = np.zeros(N, dtype=bool)  # area_id == area_idx
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

    bombsite_dist = dist_hops  # float32[N] (area_id == area_idx)
    finite = bombsite_dist[np.isfinite(bombsite_dist)]
    bombsite_dist_scale = 0.0
    if finite.size:
        mx = float(finite.max())
        bombsite_dist_scale = 1.0 / mx if mx > 0 else 0.0

    return MapData(
        N=N,
        area_ids=area_ids,
        centroids=centroids,
        vis_matrix=vis_matrix,
        adjacency=adjacency,
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
    )
