# ── SECTION: NavGraph ──────────────────────────────────────────────────────

import math
import os
import pathlib
from collections import deque

import networkx as nx
import numpy as np
from awpy import Nav
from shapely.geometry import Point
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.strtree import STRtree

# ── Multiprocessing workers for vis matrix (must be module-level to be picklable) ──

_vc = None  # per-worker VisibilityChecker instance


def _vis_worker_init(tri_path_str: str):
    """Runs once per worker process — loads VisibilityChecker into a module global."""
    global _vc
    from pathlib import Path

    from awpy.visibility import VisibilityChecker

    _vc = VisibilityChecker(path=Path(tri_path_str))


def _vis_compute_rows(row_indices: list, pts: list) -> dict:
    """Computes upper-triangle visibility for each row in row_indices.

    Returns {i: [bool for j in range(i+1, N)]} using the worker-local _vc.
    """
    N = len(pts)
    result = {}
    for i in row_indices:
        upper = []
        for j in range(i + 1, N):
            upper.append(_vc.is_visible(pts[i], pts[j]))
        result[i] = upper
    return result


class NavGraph:
    """Navigation graph built from an awpy v2 JSON nav mesh file.

    Attributes:
        nav          -- the loaded awpy Nav object
        areas        -- dict: area_id (int) -> NavArea
        area_ids     -- list of all area_ids
        N            -- number of areas
        _id_to_idx   -- dict: area_id -> int index (0-based)
        graph        -- networkx Graph (nodes = area_ids, edges = connections)
        centroids    -- dict: area_id -> np.array([x, y])
        wall_segments-- list of ((x1,y1),(x2,y2)) boundary edge tuples
        _wall_lines  -- list of shapely LineStrings for wall segments
        _wall_strtree-- STRtree for wall lines
        _area_polys  -- list of shapely Polygons (one per area, index = _id_to_idx)
        _area_strtree-- STRtree for area bboxes
        vis_matrix   -- None (built in Task 2)
        _nav_path    -- stored nav file path
        _cache_path  -- stored vis cache path
    """

    def __init__(self, nav_path: str, cache_path: str | None = None):
        if cache_path is None:
            cache_path = str(pathlib.Path(__file__).with_name("vis_cache.npy"))
        self._nav_path = nav_path
        self._cache_path = cache_path

        # ── Load nav data ──────────────────────────────────────────────────
        self.nav = Nav.from_json(nav_path)
        self.areas: dict[int, object] = self.nav.areas  # dict[int, NavArea]
        self.area_ids: list[int] = sorted(
            self.areas.keys()
        )  # sorted for stable _id_to_idx indices across runs
        self.N: int = len(self.area_ids)
        self._id_to_idx: dict[int, int] = {aid: i for i, aid in enumerate(self.area_ids)}

        # ── Compute centroids ──────────────────────────────────────────────
        self.centroids: dict[int, np.ndarray] = {}
        for aid, area in self.areas.items():
            c = area.centroid
            self.centroids[aid] = np.array([c.x, c.y], dtype=np.float32)

        # Pre-built (N, 2) matrix for vectorised nearest-centroid lookups
        self._centroid_matrix: np.ndarray = np.array(
            [self.centroids[aid] for aid in self.area_ids], dtype=np.float32
        )  # shape (N, 2)

        # Pre-built (N, 3) matrix for 3D snapping (spawn slots need z to avoid floor mismatches)
        self._centroid_matrix_3d: np.ndarray = np.array(
            [
                [
                    self.areas[aid].centroid.x,
                    self.areas[aid].centroid.y,
                    self.areas[aid].centroid.z,
                ]
                for aid in self.area_ids
            ],
            dtype=np.float32,
        )  # shape (N, 3)

        # ── Build networkx graph ───────────────────────────────────────────
        # nx.Graph (undirected): a small fraction of CS2 nav connections are
        # one-way (~5/20 in a sample), but the nav mesh is overwhelmingly
        # symmetric and pathfinding works correctly with an undirected graph.
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.area_ids)
        for aid, area in self.areas.items():
            for neighbor_id in area.connections:
                if neighbor_id in self.areas:
                    self.graph.add_edge(aid, neighbor_id)

        # ── Extract wall segments (kept for --test-navgraph) ──────────────
        self.wall_segments = self._extract_wall_segments()

        # ── Build area spatial index ───────────────────────────────────────
        self._area_polys, self._area_strtree = self._build_area_index()

        # ── Rasterized grid for O(1) position lookup ───────────────────────
        # Replaces per-step Shapely Point+STRtree+contains in get_area_if_on_mesh.
        self._grid_cache_path = cache_path.replace(".npy", "_grid.npy") if cache_path else None
        self._build_pos_grid()

        # ── Visibility matrix (built in Task 2) ───────────────────────────
        self.vis_matrix = None

    # ── Wall segment extraction ────────────────────────────────────────────

    def _extract_wall_segments(
        self,
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Extract boundary edges — edges shared by exactly one area polygon."""
        edge_count: dict[tuple, int] = {}

        for area in self.areas.values():
            corners = area.corners
            n = len(corners)
            for i in range(n):
                p1 = (round(corners[i].x, 4), round(corners[i].y, 4))
                p2 = (
                    round(corners[(i + 1) % n].x, 4),
                    round(corners[(i + 1) % n].y, 4),
                )
                # Canonical form: smaller point first
                edge = (min(p1, p2), max(p1, p2))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        # Boundary edges appear exactly once
        wall_segments = [edge for edge, count in edge_count.items() if count == 1]
        return wall_segments

    # ── Area spatial index ────────────────────────────────────────────────

    def _build_area_index(self):
        """Build shapely Polygons and STRtree for all areas."""
        polys = []
        for aid in self.area_ids:
            area = self.areas[aid]
            corners_xy = [(c.x, c.y) for c in area.corners]
            if len(corners_xy) >= 3:
                poly = ShapelyPolygon(corners_xy)
            else:
                # Degenerate: create a tiny buffer around centroid
                cx, cy = self.centroids[aid]
                poly = Point(cx, cy).buffer(0.01)
            polys.append(poly)

        strtree = STRtree(polys)
        return polys, strtree

    def _build_pos_grid(self, cell_size: float = 4.0):
        """Build a rasterized 2D grid mapping (gx, gy) → area_idx for O(1) lookups.

        Replaces per-step Shapely Point + STRtree + contains chain in
        get_area_if_on_mesh.  Cached to disk alongside the vis matrix.

        Grid cell (gx, gy) covers the square [x_min + gx*cell, y_min + gy*cell].
        Value is the index into self.area_ids, or -1 for off-mesh.
        """
        import os

        import shapely as sh

        nav_mtime = os.path.getmtime(self._nav_path) if hasattr(self, "_nav_path") else 0
        cache = self._grid_cache_path

        if cache and os.path.exists(cache):
            if os.path.getmtime(cache) > nav_mtime:
                data = np.load(cache, allow_pickle=True).item()
                self._grid_cell_size = float(data["cell"])
                self._grid_inv_cell = 1.0 / self._grid_cell_size
                self._grid_x_min = float(data["x_min"])
                self._grid_y_min = float(data["y_min"])
                self._grid_w = int(data["w"])
                self._grid_h = int(data["h"])
                self._pos_grid = data["grid"]
                print(f"[NavGraph] Loaded position grid {self._grid_w}×{self._grid_h} from cache")
                return

        all_bounds = np.array([p.bounds for p in self._area_polys])  # (N, 4)
        x_min = float(all_bounds[:, 0].min()) - cell_size
        y_min = float(all_bounds[:, 1].min()) - cell_size
        x_max = float(all_bounds[:, 2].max()) + cell_size
        y_max = float(all_bounds[:, 3].max()) + cell_size

        W = int(np.ceil((x_max - x_min) / cell_size)) + 1
        H = int(np.ceil((y_max - y_min) / cell_size)) + 1
        inv = 1.0 / cell_size

        # Cell centres
        xs = x_min + (np.arange(W) + 0.5) * cell_size  # (W,)
        ys = y_min + (np.arange(H) + 0.5) * cell_size  # (H,)

        grid = np.full((H, W), -1, dtype=np.int32)
        for i, poly in enumerate(self._area_polys):
            minx, miny, maxx, maxy = poly.bounds
            gx0 = max(0, int((minx - x_min) * inv))
            gx1 = min(W - 1, int((maxx - x_min) * inv) + 1)
            gy0 = max(0, int((miny - y_min) * inv))
            gy1 = min(H - 1, int((maxy - y_min) * inv) + 1)
            if gx1 < gx0 or gy1 < gy0:
                continue
            cxs = xs[gx0 : gx1 + 1]
            cys = ys[gy0 : gy1 + 1]
            xx, yy = np.meshgrid(cxs, cys)
            inside = sh.contains_xy(poly, xx.ravel(), yy.ravel()).reshape(xx.shape)
            grid[gy0 : gy1 + 1, gx0 : gx1 + 1][inside] = i

        self._grid_cell_size = cell_size
        self._grid_inv_cell = inv
        self._grid_x_min = x_min
        self._grid_y_min = y_min
        self._grid_w = W
        self._grid_h = H
        self._pos_grid = grid

        coverage = (grid >= 0).mean()
        print(
            f"[NavGraph] Built position grid {W}×{H} (cell={cell_size}u, {coverage:.1%} coverage)"
        )

        if cache:
            np.save(
                cache,
                {
                    "cell": cell_size,
                    "x_min": x_min,
                    "y_min": y_min,
                    "w": W,
                    "h": H,
                    "grid": grid,
                },
            )

    # ── Public API ────────────────────────────────────────────────────────

    def get_area(self, pos_xy: np.ndarray) -> int:
        """Return the area_id that contains pos_xy.

        First checks which area polygon contains the point via an STRtree
        spatial index.  If the point falls outside every polygon (e.g. it
        was snapped to a slightly off-mesh coordinate), always falls back to
        the nearest centroid so that callers always receive a valid area_id.
        Agents always spawn on the map, so a None return is never appropriate.

        Args:
            pos_xy: np.array([x, y])

        Returns:
            area_id (int) — always the nearest valid area, never None
        """
        pt = Point(pos_xy[0], pos_xy[1])
        # Query candidates from STRtree
        candidate_indices = self._area_strtree.query(pt)
        for idx in candidate_indices:
            if self._area_polys[idx].contains(pt):
                return self.area_ids[idx]
        # Fallback: nearest centroid — vectorised over all N areas
        diff = self._centroid_matrix - pos_xy  # (N, 2)
        idx = int(np.argmin((diff * diff).sum(axis=1)))
        return self.area_ids[idx]

    def is_on_mesh(self, pos_xy: np.ndarray) -> bool:
        """Return True only if pos_xy falls inside a known nav polygon."""
        pt = Point(pos_xy[0], pos_xy[1])
        for idx in self._area_strtree.query(pt):
            if self._area_polys[idx].contains(pt):
                return True
        return False

    def get_area_if_on_mesh(self, pos_xy: np.ndarray):
        """Combined get_area + is_on_mesh — O(1) rasterized grid lookup.

        Returns (area_id, True) if pos_xy is inside a nav polygon.
        Returns (None, False) if the point is outside the mesh.

        Uses a precomputed raster grid built at init (no Shapely overhead per call).
        """
        gx = int((pos_xy[0] - self._grid_x_min) * self._grid_inv_cell)
        gy = int((pos_xy[1] - self._grid_y_min) * self._grid_inv_cell)
        if 0 <= gx < self._grid_w and 0 <= gy < self._grid_h:
            idx = self._pos_grid[gy, gx]
            if idx >= 0:
                return self.area_ids[idx], True
        return None, False

    def can_see(self, area_i: int, area_j: int) -> bool:
        """Return True if area_i can see area_j (requires vis_matrix from Task 2).

        Falls back to graph connectivity if vis_matrix not yet built.
        Returns False for any unknown area_id rather than raising KeyError.
        """
        if area_i == area_j:
            return True
        if area_i not in self._id_to_idx or area_j not in self._id_to_idx:
            return False
        if self.vis_matrix is not None:
            i = self._id_to_idx[area_i]
            j = self._id_to_idx[area_j]
            return bool(self.vis_matrix[i, j])
        # Fallback: connected in graph
        return self.graph.has_edge(area_i, area_j) or area_i == area_j

    def path(self, area_i: int, area_j: int) -> list[int]:
        """Return shortest path of area_ids from area_i to area_j.

        Returns empty list if no path exists.
        """
        try:
            return nx.shortest_path(self.graph, area_i, area_j)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def build_vis_matrix(self):
        """Build and cache the N×N visibility matrix.

        Uses awpy VisibilityChecker with real .tri geometry.

        Parallelised: one worker process per CPU core, each loading its own VisibilityChecker
        instance once (via ProcessPoolExecutor initializer), then processing a share of rows.
        """
        import os
        import time
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from awpy.data import TRIS_DIR

        nav_mtime = os.path.getmtime(self._nav_path) if hasattr(self, "_nav_path") else 0

        if os.path.exists(self._cache_path):
            cache_mtime = os.path.getmtime(self._cache_path)
            if cache_mtime > nav_mtime:
                self.vis_matrix = np.load(self._cache_path, mmap_mode="r")
                print(f"[NavGraph] Loaded visibility matrix from cache ({self.N}×{self.N})")
                return

        tri_path = TRIS_DIR / "de_dust2.tri"
        if not tri_path.exists():
            raise FileNotFoundError(
                f".tri file not found at {tri_path}\nDownload it with:  awpy get tris"
            )

        pts = [
            (
                float(self.areas[aid].centroid.x),
                float(self.areas[aid].centroid.y),
                float(self.areas[aid].centroid.z),
            )
            for aid in self.area_ids
        ]

        n_workers = os.cpu_count() or 4
        # ~8 tasks per worker for good load-balancing without excessive IPC overhead
        chunk_size = max(1, self.N // (n_workers * 8))
        chunks = [list(range(i, min(i + chunk_size, self.N))) for i in range(0, self.N, chunk_size)]

        print(
            f"[NavGraph] Building {self.N}×{self.N} vis matrix "
            f"({n_workers} workers, {len(chunks)} chunks)..."
        )
        print("[NavGraph] Workers initialising VisibilityChecker in parallel (~40s)...")
        t0 = time.time()

        partial_rows: dict = {}
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_vis_worker_init,
            initargs=(str(tri_path),),
        ) as executor:
            futures = [executor.submit(_vis_compute_rows, chunk, pts) for chunk in chunks]
            for fut in as_completed(futures):
                partial_rows.update(fut.result())
                done = len(partial_rows)
                elapsed = time.time() - t0
                eta = (elapsed / done) * (self.N - done) if done else 0
                print(f"  [{done}/{self.N} rows] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

        vis = np.zeros((self.N, self.N), dtype=bool)
        for i in range(self.N):
            vis[i][i] = True
            for j_off, val in enumerate(partial_rows.get(i, [])):
                j = i + j_off + 1
                if val:
                    vis[i][j] = vis[j][i] = True

        self.vis_matrix = vis
        np.save(self._cache_path, vis)
        elapsed = time.time() - t0
        print(f"[NavGraph] Visibility matrix built in {elapsed:.0f}s, cached to {self._cache_path}")


# ── SECTION: Constants ─────────────────────────────────────────────────────

MOVE_SPEED = 250
TICK_RATE = 16
DT = 1.0 / TICK_RATE
LASER_DAMAGE = 100
LASER_RANGE = 3000
SHOOT_COOLDOWN = 10
BOMB_PLANT_TIME = int(1.2 * TICK_RATE)
BOMB_DEFUSE_TIME = 10 * TICK_RATE
BOMB_DEFUSE_KIT = 5 * TICK_RATE
BOMB_TIMER = int(40 * TICK_RATE)
ROUND_TIME = int(40 * TICK_RATE)
FOOTSTEP_RADIUS = 800
GUNSHOT_RADIUS = 2000
BOMB_BEEP_RADIUS = 1500
ENEMY_MEMORY_TICKS = 32

MAP_X_MIN, MAP_X_MAX = -2476.0, 2000.0
MAP_Y_MIN, MAP_Y_MAX = -1050.0, 3420.0

# Precomputed reciprocals for _norm_xy — avoids repeated division inside step()
_INV_MAP_X_RANGE = 2.0 / (MAP_X_MAX - MAP_X_MIN)
_INV_MAP_Y_RANGE = 2.0 / (MAP_Y_MAX - MAP_Y_MIN)
_MAP_X_OFFSET = (MAP_X_MAX + MAP_X_MIN) / (MAP_X_MAX - MAP_X_MIN)
_MAP_Y_OFFSET = (MAP_Y_MAX + MAP_Y_MIN) / (MAP_Y_MAX - MAP_Y_MIN)

# Direction vectors for movement actions (built once at import time)
_DIR_VECTORS = {
    0: np.array([0.0, 0.0]),
    1: np.array([0.0, 1.0]),  # N
    2: np.array([0.7071067811865476, 0.7071067811865476]),  # NE (pre-normalised)
    3: np.array([1.0, 0.0]),  # E
    4: np.array([0.7071067811865476, -0.7071067811865476]),  # SE
    5: np.array([0.0, -1.0]),  # S
    6: np.array([-0.7071067811865476, -0.7071067811865476]),  # SW
    7: np.array([-1.0, 0.0]),  # W
    8: np.array([-0.7071067811865476, 0.7071067811865476]),  # NW
}
# Pre-scaled delta per direction — avoids multiplying every step
_DELTA_VECTORS = {k: v * (MOVE_SPEED * DT) for k, v in _DIR_VECTORS.items()}
# Pre-computed facing angle per direction — avoids np.arctan2 every step
_DIR_FACING = {k: math.atan2(float(v[1]), float(v[0])) for k, v in _DIR_VECTORS.items()}
MAX_TURN_SPEED_RAD = math.pi / 4  # 45 degrees per tick — max facing rotation rate

# Team Spirit: controls individual↔team reward blending (annealed 0→1 during training).
# WARNING: uses module-level global — only process-safe when num_cpus=1 in train.py.
# If num_cpus > 1 is ever needed, replace with multiprocessing.Value.
_TEAM_SPIRIT: float = 0.0
N_AGENTS = 10
TEAM_SIZE = 5
OBS_DIM = 71
ACTION_DIM = 4
INVALID_AREA_ID = -1
STALE_MEMORY_TICK = -9999
_NOOP_ACTION = np.zeros(ACTION_DIM, dtype=np.int64)
_POSSIBLE_AGENTS = tuple([f"t{i}" for i in range(TEAM_SIZE)] + [f"ct{i}" for i in range(TEAM_SIZE)])

# CS2 setpos_exact spawn slots (from Valve competitive map data, with z)
_T_SPAWN_SLOTS = (
    (-881, -754, 120),
    (-841, -808, 117),
    (-776, -843, 117),
    (-715, -808, 116),
    (-680, -754, 120),
    (-557, -738, 122),
    (-522, -795, 117),
    (-460, -836, 117),
    (-396, -806, 117),
    (-357, -755, 120),
    (-233, -754, 114),
    (-193, -808, 109),
    (-128, -843, 95),
    (-67, -808, 84),
    (-32, -754, 79),
)
_CT_SPAWN_SLOTS = (
    (160, 2370, -120),
    (182, 2439, -121),
    (258, 2481, -121),
    (334, 2434, -120),
    (351, 2353, -120),
)
_A_SITE = (1200.0, 2400.0, 100.0)
_B_SITE = (-1530.0, 2600.0, 5.0)
_DUST2_STATIC_CACHE = {}


def _resolve_nav_path(map_name: str = "de_dust2") -> str:
    override = os.environ.get("CS2RL_NAV_PATH")
    if override:
        return override

    candidates = [
        pathlib.Path.home() / ".awpy" / "navs" / f"{map_name}.json",
        pathlib.Path(os.path.expanduser("~")) / ".awpy" / "navs" / f"{map_name}.json",
    ]

    if os.name != "nt":
        candidates.extend(pathlib.Path("/mnt/c/Users").glob(f"*/.awpy/navs/{map_name}.json"))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0])


NAV_PATH = _resolve_nav_path()
CACHE_PATH = str(pathlib.Path(__file__).with_name("vis_cache.npy"))


def _snap_to_nav(nav_graph: NavGraph, xyz) -> int:
    if len(xyz) >= 3:
        pt = np.asarray(xyz[:3], dtype=np.float32)
        diff = nav_graph._centroid_matrix_3d - pt
    else:
        pt = np.asarray(xyz[:2], dtype=np.float32)
        diff = nav_graph._centroid_matrix - pt
    idx = int(np.argmin((diff * diff).sum(axis=1)))
    return nav_graph.area_ids[idx]


def _areas_near(nav_graph: NavGraph, xy, radius: float):
    pt = np.asarray(xy[:2], dtype=np.float32)
    diff = nav_graph._centroid_matrix - pt
    dists = np.sqrt((diff * diff).sum(axis=1))
    order = np.argsort(dists)
    return [nav_graph.area_ids[i] for i in order if dists[i] < radius]


def _nearest_area_candidates(nav_graph: NavGraph, xyz, limit: int = 64):
    if len(xyz) >= 3:
        pt = np.asarray(xyz[:3], dtype=np.float32)
        diff = nav_graph._centroid_matrix_3d - pt
    else:
        pt = np.asarray(xyz[:2], dtype=np.float32)
        diff = nav_graph._centroid_matrix - pt

    order = np.argsort((diff * diff).sum(axis=1))
    return [nav_graph.area_ids[i] for i in order[:limit]]


def _select_distinct_spawn_areas(
    nav_graph: NavGraph,
    slots,
    count: int,
    required_targets=None,
    area_adjacency=None,
    candidate_limit: int = 64,
):
    selected = []
    used = set()
    targets = tuple(required_targets or ())

    for slot in slots:
        for area_id in _nearest_area_candidates(nav_graph, slot, limit=candidate_limit):
            if area_id in used:
                continue
            if targets and area_adjacency is not None:
                if not _area_reaches_any_target(nav_graph, area_adjacency, area_id, targets):
                    continue
            elif targets and not any(nav_graph.path(area_id, target) for target in targets):
                continue

            selected.append(area_id)
            used.add(area_id)
            break

        if len(selected) >= count:
            return selected

    raise ValueError(f"Failed to select {count} distinct spawn areas from {len(slots)} slots")


def _build_area_adjacency(nav_graph: NavGraph) -> np.ndarray:
    """Build the executable XY adjacency used by the simplified movement model.

    Raw nav connections can include vertical/one-way links that are valid for the
    Source navmesh, but are not directly traversable in this sim because movement
    is a 2D point step over the rasterized walkable surface. Derive adjacency from
    neighboring on-mesh raster cells so pathfinding matches the areas agents can
    actually enter via get_area_if_on_mesh + fixed XY moves.
    """
    adj = np.zeros((nav_graph.N, nav_graph.N), dtype=bool)
    np.fill_diagonal(adj, True)

    grid = nav_graph._pos_grid
    height, width = grid.shape
    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    for dy, dx in offsets:
        src_y0 = max(0, -dy)
        src_y1 = min(height, height - dy) if dy >= 0 else height
        src_x0 = max(0, -dx)
        src_x1 = min(width, width - dx) if dx >= 0 else width

        dst_y0 = max(0, dy)
        dst_y1 = min(height, height + dy) if dy <= 0 else height
        dst_x0 = max(0, dx)
        dst_x1 = min(width, width + dx) if dx <= 0 else width

        src = grid[src_y0:src_y1, src_x0:src_x1]
        dst = grid[dst_y0:dst_y1, dst_x0:dst_x1]
        mask = (src >= 0) & (dst >= 0) & (src != dst)
        if not np.any(mask):
            continue

        src_idx = src[mask]
        dst_idx = dst[mask]
        adj[src_idx, dst_idx] = True
        adj[dst_idx, src_idx] = True

    return adj


def _area_reaches_any_target(
    nav_graph: NavGraph,
    area_adjacency: np.ndarray,
    start_area: int,
    targets,
) -> bool:
    target_ids = [target for target in targets if target in nav_graph._id_to_idx]
    if not target_ids:
        return False

    start_idx = nav_graph._id_to_idx.get(start_area)
    if start_idx is None:
        return False

    target_mask = np.zeros(nav_graph.N, dtype=bool)
    target_mask[[nav_graph._id_to_idx[target] for target in target_ids]] = True

    stack = [start_idx]
    seen = np.zeros(nav_graph.N, dtype=bool)
    seen[start_idx] = True

    while stack:
        idx = stack.pop()
        if target_mask[idx]:
            return True

        neighbors = np.flatnonzero(area_adjacency[idx] & ~seen)
        if neighbors.size == 0:
            continue
        seen[neighbors] = True
        stack.extend(int(nbr) for nbr in neighbors)

    return False


def _compute_area_distance_to_targets(
    nav_graph: NavGraph,
    area_adjacency: np.ndarray,
    targets,
):
    target_ids = [target for target in targets if target in nav_graph._id_to_idx]
    dist = np.full(nav_graph.N, np.inf, dtype=np.float32)
    if not target_ids:
        return dist

    q = deque()
    for target in target_ids:
        idx = nav_graph._id_to_idx[target]
        dist[idx] = 0.0
        q.append(idx)

    while q:
        idx = q.popleft()
        next_dist = dist[idx] + 1.0
        neighbors = np.flatnonzero(area_adjacency[idx])
        for nbr in neighbors:
            nbr = int(nbr)
            if next_dist >= dist[nbr]:
                continue
            dist[nbr] = next_dist
            q.append(nbr)

    return dist


def _load_dust2_static_data(nav_path: str, cache_path: str):
    key = (nav_path, cache_path)
    static = _DUST2_STATIC_CACHE.get(key)
    if static is not None:
        return static

    nav_graph = NavGraph(nav_path, cache_path)
    nav_graph.build_vis_matrix()

    xs = [c[0] for c in nav_graph.centroids.values()]
    ys = [c[1] for c in nav_graph.centroids.values()]
    map_bounds = (min(xs), max(xs), min(ys), max(ys))

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

    centroid_lookup = np.zeros((max_area_id + 1, 2), dtype=np.float32)
    for area_id, centroid in nav_graph.centroids.items():
        centroid_lookup[area_id] = centroid

    bombsite_distance_lookup = np.full(max_area_id + 1, np.inf, dtype=np.float32)
    for area_id in nav_graph.area_ids:
        bombsite_distance_lookup[area_id] = bombsite_area_distance[nav_graph._id_to_idx[area_id]]
    finite_dist = bombsite_distance_lookup[np.isfinite(bombsite_distance_lookup)]
    bombsite_distance_scale = 0.0
    if finite_dist.size:
        max_dist = float(finite_dist.max())
        bombsite_distance_scale = 1.0 / max_dist if max_dist > 0 else 0.0

    static = {
        "nav_graph": nav_graph,
        "map_bounds": map_bounds,
        "t_spawn_areas": t_spawn_areas,
        "ct_spawn_areas": ct_spawn_areas,
        "a_site_areas": a_site_areas,
        "b_site_areas": b_site_areas,
        "bombsite_areas": bombsite_areas,
        "bombsite_mask": bombsite_mask,
        "centroid_lookup": centroid_lookup,
        "bombsite_distance_lookup": bombsite_distance_lookup,
        "bombsite_distance_scale": bombsite_distance_scale,
        "area_adjacency": area_adjacency,
    }
    _DUST2_STATIC_CACHE[key] = static

    return static
