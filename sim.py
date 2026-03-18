# ── SECTION: NavGraph ──────────────────────────────────────────────────────

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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

    def __init__(self, nav_path: str, cache_path: str = "vis_cache.npy"):
        self._nav_path = nav_path
        self._cache_path = cache_path

        # ── Load nav data ──────────────────────────────────────────────────
        self.nav = Nav.from_json(nav_path)
        self.areas: Dict[int, object] = self.nav.areas  # dict[int, NavArea]
        self.area_ids: List[int] = sorted(
            self.areas.keys()
        )  # sorted for stable _id_to_idx indices across runs
        self.N: int = len(self.area_ids)
        self._id_to_idx: Dict[int, int] = {
            aid: i for i, aid in enumerate(self.area_ids)
        }

        # ── Compute centroids ──────────────────────────────────────────────
        self.centroids: Dict[int, np.ndarray] = {}
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
        self._grid_cache_path = (
            cache_path.replace(".npy", "_grid.npy") if cache_path else None
        )
        self._build_pos_grid()

        # ── Visibility matrix (built in Task 2) ───────────────────────────
        self.vis_matrix = None

    # ── Wall segment extraction ────────────────────────────────────────────

    def _extract_wall_segments(
        self,
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Extract boundary edges — edges shared by exactly one area polygon."""
        edge_count: Dict[Tuple, int] = {}

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

        nav_mtime = (
            os.path.getmtime(self._nav_path) if hasattr(self, "_nav_path") else 0
        )
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
                print(
                    f"[NavGraph] Loaded position grid {self._grid_w}×{self._grid_h} from cache"
                )
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

    def path(self, area_i: int, area_j: int) -> List[int]:
        """Return shortest path of area_ids from area_i to area_j.

        Returns empty list if no path exists.
        """
        try:
            return nx.shortest_path(self.graph, area_i, area_j)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def build_vis_matrix(self):
        """Build and cache the N×N visibility matrix using awpy VisibilityChecker + real .tri geometry.

        Parallelised: one worker process per CPU core, each loading its own VisibilityChecker
        instance once (via ProcessPoolExecutor initializer), then processing a share of rows.
        """
        import os
        import time
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from awpy.data import TRIS_DIR

        nav_mtime = (
            os.path.getmtime(self._nav_path) if hasattr(self, "_nav_path") else 0
        )

        if os.path.exists(self._cache_path):
            cache_mtime = os.path.getmtime(self._cache_path)
            if cache_mtime > nav_mtime:
                self.vis_matrix = np.load(self._cache_path)
                print(
                    f"[NavGraph] Loaded visibility matrix from cache ({self.N}×{self.N})"
                )
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
        chunks = [
            list(range(i, min(i + chunk_size, self.N)))
            for i in range(0, self.N, chunk_size)
        ]

        print(
            f"[NavGraph] Building {self.N}×{self.N} vis matrix "
            f"({n_workers} workers, {len(chunks)} chunks)..."
        )
        print(
            f"[NavGraph] Workers initialising VisibilityChecker in parallel (~40s)..."
        )
        t0 = time.time()

        partial_rows: dict = {}
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_vis_worker_init,
            initargs=(str(tri_path),),
        ) as executor:
            futures = [
                executor.submit(_vis_compute_rows, chunk, pts) for chunk in chunks
            ]
            for fut in as_completed(futures):
                partial_rows.update(fut.result())
                done = len(partial_rows)
                elapsed = time.time() - t0
                eta = (elapsed / done) * (self.N - done) if done else 0
                print(
                    f"  [{done}/{self.N} rows] {elapsed:.0f}s elapsed, ETA {eta:.0f}s"
                )

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
        print(
            f"[NavGraph] Visibility matrix built in {elapsed:.0f}s, cached to {self._cache_path}"
        )


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

# Team Spirit: controls individual↔team reward blending (annealed 0→1 during training).
# WARNING: uses module-level global — only process-safe when num_cpus=1 in train.py.
# If num_cpus > 1 is ever needed, replace with multiprocessing.Value.
_TEAM_SPIRIT: float = 0.0

# ── SECTION: Dataclasses ───────────────────────────────────────────────────


@dataclass
class AgentState:
    agent_id: int
    team: int  # 0 = T, 1 = CT
    pos: np.ndarray  # [x, y, z] HU
    area_id: int
    facing: float  # radians, 0 = +X
    hp: int
    alive: bool
    has_bomb: bool
    has_kit: bool
    shoot_cd: int
    is_moving: bool
    fired_this_tick: bool
    enemy_memory: dict = field(default_factory=dict)  # {enemy_id: (area_id, tick)}


@dataclass
class GameState:
    tick: int
    agents: list
    bomb_planted: bool
    bomb_carrier_id: int
    bomb_area_id: int
    bomb_pos: np.ndarray
    bomb_ticks_left: int
    bomb_being_planted_by: int
    bomb_plant_ticks: int
    bomb_being_defused_by: int
    bomb_defuse_ticks: int
    round_ticks_left: int
    round_over: bool
    winner: int  # 0=T, 1=CT, -1=ongoing


@dataclass
class SoundEvent:
    source_pos: np.ndarray
    source_id: int
    radius: float
    type: str


# ── SECTION: Dust2Env ──────────────────────────────────────────────────────

import pathlib

import gymnasium
from gymnasium import spaces
from pettingzoo import ParallelEnv

NAV_PATH = "C:/Users/vboxuser/.awpy/navs/de_dust2.json"
CACHE_PATH = "vis_cache.npy"


class Dust2Env(ParallelEnv):
    metadata = {"name": "dust2_v0", "render_modes": []}
    render_mode = None

    def __init__(
        self, nav_path=NAV_PATH, cache_path=CACHE_PATH, record_fn=None, team_spirit=None
    ):
        super().__init__()
        self.nav_graph = NavGraph(nav_path, cache_path)
        self.nav_graph.build_vis_matrix()

        self._calibrate_map_bounds()
        self._identify_special_areas()

        self.possible_agents = [f"t{i}" for i in range(5)] + [
            f"ct{i}" for i in range(5)
        ]
        self.agents = list(self.possible_agents)
        self._record_fn = record_fn
        self._team_spirit_shared = team_spirit  # multiprocessing.Value or None
        self.state: GameState = None

    def _calibrate_map_bounds(self):
        global MAP_X_MIN, MAP_X_MAX, MAP_Y_MIN, MAP_Y_MAX
        global _INV_MAP_X_RANGE, _INV_MAP_Y_RANGE, _MAP_X_OFFSET, _MAP_Y_OFFSET
        xs = [c[0] for c in self.nav_graph.centroids.values()]
        ys = [c[1] for c in self.nav_graph.centroids.values()]
        MAP_X_MIN, MAP_X_MAX = min(xs), max(xs)
        MAP_Y_MIN, MAP_Y_MAX = min(ys), max(ys)
        _INV_MAP_X_RANGE = 2.0 / (MAP_X_MAX - MAP_X_MIN)
        _INV_MAP_Y_RANGE = 2.0 / (MAP_Y_MAX - MAP_Y_MIN)
        _MAP_X_OFFSET = (MAP_X_MAX + MAP_X_MIN) / (MAP_X_MAX - MAP_X_MIN)
        _MAP_Y_OFFSET = (MAP_Y_MAX + MAP_Y_MIN) / (MAP_Y_MAX - MAP_Y_MIN)
        print(
            f"[Dust2Env] Map bounds: X=[{MAP_X_MIN:.0f},{MAP_X_MAX:.0f}] Y=[{MAP_Y_MIN:.0f},{MAP_Y_MAX:.0f}]"
        )

    def _snap_to_nav(self, xyz) -> int:
        """Return the area_id whose centroid is nearest to xyz.

        Uses 3D distance when z is provided (len >= 3) so spawn slots on
        different floor levels (e.g. CT spawn vs catwalk) don't cross-snap.
        """
        if len(xyz) >= 3:
            pt = np.array(xyz[:3], dtype=np.float32)
            diff = self.nav_graph._centroid_matrix_3d - pt  # (N, 3)
        else:
            pt = np.array(xyz[:2], dtype=np.float32)
            diff = self.nav_graph._centroid_matrix - pt  # (N, 2)
        idx = int(np.argmin((diff * diff).sum(axis=1)))
        return self.nav_graph.area_ids[idx]

    def _areas_near(self, xy, radius: float):
        """Return all area_ids within radius of 2D position xy, sorted by distance."""
        pt = np.array(xy[:2], dtype=np.float32)
        diff = self.nav_graph._centroid_matrix - pt  # (N, 2)
        dists = np.sqrt((diff * diff).sum(axis=1))
        order = np.argsort(dists)
        return [self.nav_graph.area_ids[i] for i in order if dists[i] < radius]

    def _identify_special_areas(self):
        # CS2 setpos_exact spawn slots (from Valve competitive map data, with z)
        T_SPAWN_SLOTS = [
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
        ]
        CT_SPAWN_SLOTS = [
            (160, 2370, -120),
            (182, 2439, -121),
            (258, 2481, -121),
            (334, 2434, -120),
            (351, 2353, -120),
        ]
        # Bombsite centers (CS2 in-game coords, verified 20-27u from nearest nav area)
        A_SITE = (1200.0, 2400.0, 100.0)
        B_SITE = (-1530.0, 2600.0, 5.0)

        # Spawn: snap each individual slot to its nearest nav area.
        # This guarantees agents land at actual spawn box positions, not
        # arbitrary areas that happen to share a low area_id.
        self.t_spawn_areas = [self._snap_to_nav(s) for s in T_SPAWN_SLOTS]
        self.ct_spawn_areas = [self._snap_to_nav(s) for s in CT_SPAWN_SLOTS]

        # Bombsites: all areas sorted by distance from site centre (closest first).
        self.a_site_areas = self._areas_near(A_SITE, 600)
        self.b_site_areas = self._areas_near(B_SITE, 600)
        self.bombsite_areas = set(self.a_site_areas + self.b_site_areas)

        print(
            f"[Dust2Env] T-spawn: {len(self.t_spawn_areas)} slots  "
            f"CT-spawn: {len(self.ct_spawn_areas)} slots  "
            f"A-site: {len(self.a_site_areas)} areas  "
            f"B-site: {len(self.b_site_areas)} areas"
        )

    def observation_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(71,), dtype=np.float32)

    def action_space(self, agent):
        return spaces.MultiDiscrete([9, 2, 2, 2])

    def _build_vis10(self, s) -> np.ndarray:
        """Build a (10, 10) bool matrix of inter-agent visibility for this tick.

        Called once after movement so area_ids are final.  Reused by the shoot
        loop, enemy-memory update, and obs computation — avoids ~150 individual
        can_see() dict lookups per step.
        """
        _vm = self.nav_graph.vis_matrix
        _iix = self.nav_graph._id_to_idx
        if _vm is not None:
            idxs = np.array([_iix.get(s.agents[k].area_id, 0) for k in range(10)])
            vis10 = _vm[np.ix_(idxs, idxs)]  # (10,10) bool slice — one C-level op
            np.fill_diagonal(vis10, True)
        else:
            vis10 = np.zeros((10, 10), dtype=bool)
            np.fill_diagonal(vis10, True)
        return vis10

    def _build_obs_context(self, s):
        """Compute shared values for _compute_obs — called once per tick.

        Returns (alive_t, alive_ct, vis10) where:
          alive_t  = fraction of T-side agents alive (float)
          alive_ct = fraction of CT-side agents alive (float)
          vis10    = (10, 10) bool numpy array: vis10[i, j] = can_see this tick
        """
        alive_t = sum(1 for a in s.agents if a.team == 0 and a.alive) / 5.0
        alive_ct = sum(1 for a in s.agents if a.team == 1 and a.alive) / 5.0
        vis10 = self._build_vis10(s)
        return alive_t, alive_ct, vis10

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.agents = list(self.possible_agents)
        self.state = self._make_initial_state()
        alive_t, alive_ct, vis10 = self._build_obs_context(self.state)
        obs = {
            aid: self._compute_obs(i, alive_t, alive_ct, vis10)
            for i, aid in enumerate(self.possible_agents)
        }
        infos = {aid: {} for aid in self.possible_agents}
        return obs, infos

    def _make_initial_state(self):
        agents = []
        bomb_carrier = np.random.randint(0, 5)

        for i in range(5):
            spawn_area = self.t_spawn_areas[i % len(self.t_spawn_areas)]
            c = self.nav_graph.areas[spawn_area].centroid
            agents.append(
                AgentState(
                    agent_id=i,
                    team=0,
                    pos=np.array([c.x, c.y, c.z], dtype=np.float32),
                    area_id=spawn_area,
                    facing=0.0,
                    hp=100,
                    alive=True,
                    has_bomb=(i == bomb_carrier),
                    has_kit=False,
                    shoot_cd=0,
                    is_moving=False,
                    fired_this_tick=False,
                )
            )

        for i in range(5):
            spawn_area = self.ct_spawn_areas[i % len(self.ct_spawn_areas)]
            c = self.nav_graph.areas[spawn_area].centroid
            agents.append(
                AgentState(
                    agent_id=5 + i,
                    team=1,
                    pos=np.array([c.x, c.y, c.z], dtype=np.float32),
                    area_id=spawn_area,
                    facing=np.pi,
                    hp=100,
                    alive=True,
                    has_bomb=False,
                    has_kit=(np.random.random() < 0.5),
                    shoot_cd=0,
                    is_moving=False,
                    fired_this_tick=False,
                )
            )

        return GameState(
            tick=0,
            agents=agents,
            bomb_planted=False,
            bomb_carrier_id=bomb_carrier,
            bomb_area_id=-1,
            bomb_pos=np.zeros(3),
            bomb_ticks_left=0,
            bomb_being_planted_by=-1,
            bomb_plant_ticks=0,
            bomb_being_defused_by=-1,
            bomb_defuse_ticks=0,
            round_ticks_left=ROUND_TIME,
            round_over=False,
            winner=-1,
        )

    def _norm_xy(self, pos):
        return np.array(
            [
                pos[0] * _INV_MAP_X_RANGE - _MAP_X_OFFSET,
                pos[1] * _INV_MAP_Y_RANGE - _MAP_Y_OFFSET,
            ],
            dtype=np.float32,
        )

    def _compute_obs(
        self, agent_idx: int, alive_t: float, alive_ct: float, vis10: np.ndarray
    ) -> np.ndarray:
        """Build the 71-float observation vector for agent at agent_idx.

        Parameters pre-computed by step() once per tick to avoid redundant work:
          alive_t  -- fraction of T-side agents alive (computed once, shared across all 10 obs)
          alive_ct -- fraction of CT-side agents alive
          vis10    -- (10, 10) bool array: vis10[i, j] = can agent i see agent j this tick
        """
        obs = np.zeros(71, dtype=np.float32)
        s = self.state
        agent = s.agents[agent_idx]
        team = agent.team

        # ── Self features ──────────────────────────────────────────────────
        obs[0] = float(team)
        # Inline _norm_xy — avoids creating a numpy array on each call
        obs[1] = agent.pos[0] * _INV_MAP_X_RANGE - _MAP_X_OFFSET
        obs[2] = agent.pos[1] * _INV_MAP_Y_RANGE - _MAP_Y_OFFSET
        obs[3] = math.sin(agent.facing)
        obs[4] = math.cos(agent.facing)
        obs[5] = agent.hp / 100.0
        obs[6] = float(agent.has_bomb if team == 0 else agent.has_kit)
        obs[7] = 1.0 if agent.shoot_cd == 0 else (1.0 - agent.shoot_cd / SHOOT_COOLDOWN)

        # ── Teammate features ──────────────────────────────────────────────
        # Agents are stored in agent_id order: team-0 = s.agents[0:5], team-1 = s.agents[5:10]
        tm_start = 0 if team == 0 else 5
        tm_count = 0
        for j in range(tm_start, tm_start + 5):
            if j == agent_idx:
                continue
            tm = s.agents[j]
            base = 8 + tm_count * 5
            obs[base] = tm.pos[0] * _INV_MAP_X_RANGE - _MAP_X_OFFSET
            obs[base + 1] = tm.pos[1] * _INV_MAP_Y_RANGE - _MAP_Y_OFFSET
            obs[base + 2] = math.sin(tm.facing)
            obs[base + 3] = math.cos(tm.facing)
            obs[base + 4] = tm.hp / 100.0 if tm.alive else 0.0
            tm_count += 1
            if tm_count == 4:
                break

        # ── Enemy features ─────────────────────────────────────────────────
        # Enemies are the opposite team slice — already in agent_id order, no sort needed.
        en_start = 5 if team == 0 else 0
        for i in range(5):
            en = s.agents[en_start + i]
            en_idx = en_start + i
            base = 28 + i * 7
            mem = agent.enemy_memory.get(en.agent_id)
            can_see = vis10[agent_idx, en_idx] if en.alive else False

            if mem is None and not can_see:
                continue

            if mem is not None:
                mem_area, last_tick = mem
                mem_centroid = self.nav_graph.centroids.get(mem_area, agent.pos[:2])
                obs[base] = mem_centroid[0] * _INV_MAP_X_RANGE - _MAP_X_OFFSET
                obs[base + 1] = mem_centroid[1] * _INV_MAP_Y_RANGE - _MAP_Y_OFFSET
                obs[base + 2] = math.sin(en.facing)
                obs[base + 3] = math.cos(en.facing)
                obs[base + 4] = en.hp / 100.0 if en.alive else 0.0
                obs[base + 5] = 1.0 if can_see else 0.0
                freshness = (
                    max(0, ENEMY_MEMORY_TICKS - (s.tick - last_tick))
                    / ENEMY_MEMORY_TICKS
                    if last_tick >= 0
                    else 0.0
                )
                obs[base + 6] = freshness

        # ── Global features ────────────────────────────────────────────────
        obs[63] = float(s.bomb_planted)
        if s.bomb_planted:
            obs[64] = s.bomb_pos[0] * _INV_MAP_X_RANGE - _MAP_X_OFFSET
            obs[65] = s.bomb_pos[1] * _INV_MAP_Y_RANGE - _MAP_Y_OFFSET
            obs[66] = s.bomb_ticks_left / BOMB_TIMER
        else:
            obs[64] = -1.0
            obs[65] = -1.0
            obs[66] = 0.0
        obs[67] = s.round_ticks_left / ROUND_TIME
        obs[68] = alive_t
        obs[69] = alive_ct
        obs[70] = float(agent.area_id in self.bombsite_areas)

        return obs

    def step(self, actions: dict):
        s = self.state
        # Capture pre-action potential before ANY state mutation
        phi_before = {0: self._potential(s, 0), 1: self._potential(s, 1)}
        s.tick += 1
        s.round_ticks_left -= 1

        # Pre-initialize accumulators (later tasks will populate these)
        kills_this_tick = []
        bomb_just_planted = False
        bomb_just_defused = False
        _bomb_planter_id = -1
        _bomb_defuser_id = -1

        # 1. Decrement cooldowns
        for agent in s.agents:
            if agent.shoot_cd > 0:
                agent.shoot_cd -= 1
            agent.fired_this_tick = False
            agent.is_moving = False

        # 2. Process movement for all alive agents simultaneously
        for i, aid in enumerate(self.possible_agents):
            agent = s.agents[i]
            if not agent.alive:
                continue
            action = actions.get(aid, np.array([0, 0, 0, 0]))
            move_dir = int(action[0])

            if move_dir == 0:
                continue

            delta = _DELTA_VECTORS[move_dir]  # pre-scaled at module level
            target_x = agent.pos[0] + delta[0]
            target_y = agent.pos[1] + delta[1]

            # Update facing from movement direction
            agent.facing = _DIR_FACING[move_dir]

            # Find target area — only accept moves that land inside the nav mesh
            target_area, on_mesh = self.nav_graph.get_area_if_on_mesh(
                (target_x, target_y)
            )
            can_move = on_mesh and (
                target_area == agent.area_id
                or self.nav_graph.graph.has_edge(agent.area_id, target_area)
            )

            if can_move:
                agent.pos[0] = target_x
                agent.pos[1] = target_y
                agent.area_id = target_area
                agent.is_moving = True

        # Build vis10 once after movement (area_ids are final for rest of this tick).
        # vis10[i, j] = True if agent i can see agent j; reused in shoot, memory, obs.
        _vis10 = self._build_vis10(s)

        # 4. Process shoot actions (simultaneous)
        _LASER_RANGE_SQ = LASER_RANGE * LASER_RANGE
        for i, aid in enumerate(self.possible_agents):
            agent = s.agents[i]
            if not agent.alive:
                continue
            action = actions.get(aid, np.array([0, 0, 0, 0]))
            if int(action[1]) == 0 or agent.shoot_cd > 0:
                continue

            # Fire
            agent.shoot_cd = SHOOT_COOLDOWN
            agent.fired_this_tick = True

            # Determine facing ray direction
            dx = math.cos(agent.facing)
            dy = math.sin(agent.facing)

            # Find enemies along ray — use pre-built vis10 (no per-enemy dict lookup)
            en_start = 5 if agent.team == 0 else 0
            best_enemy = None
            best_dist = LASER_RANGE

            for en_j in range(en_start, en_start + 5):
                enemy = s.agents[en_j]
                if not enemy.alive:
                    continue
                # Check LOS via vis10 (built after movement, no dict lookup)
                if not _vis10[i, en_j]:
                    continue

                rel = enemy.pos[:2] - agent.pos[:2]
                dist_sq = rel[0] * rel[0] + rel[1] * rel[1]
                if dist_sq > _LASER_RANGE_SQ or dist_sq == 0:
                    continue
                dist = dist_sq**0.5

                # Check angular alignment with facing direction (within ~45° cone)
                rel_norm = rel / dist
                dot = rel_norm[0] * dx + rel_norm[1] * dy
                if dot < 0.7:
                    continue

                if dist < best_dist:
                    best_dist = dist
                    best_enemy = enemy

            if best_enemy is not None:
                best_enemy.hp -= LASER_DAMAGE
                if best_enemy.hp <= 0:
                    best_enemy.alive = False
                    best_enemy.hp = 0
                    kills_this_tick.append((agent.agent_id, best_enemy.agent_id))

        # 5. Check round end conditions
        t_alive = [a for a in s.agents if a.team == 0 and a.alive]
        ct_alive = [a for a in s.agents if a.team == 1 and a.alive]

        if not t_alive and not s.round_over:
            s.round_over = True
            s.winner = 1
        elif not ct_alive and not s.round_over:
            s.round_over = True
            s.winner = 0
        elif s.round_ticks_left <= 0 and not s.round_over:
            s.round_over = True
            s.winner = 1  # CT wins on timeout

        # 6. Process plant/defuse actions
        # Clear defuse state if the defuser stopped or left
        if s.bomb_being_defused_by != -1:
            defuser = next(
                (a for a in s.agents if a.agent_id == s.bomb_being_defused_by), None
            )
            defuser_aid = f"ct{s.bomb_being_defused_by - 5}"
            defuser_action = actions.get(defuser_aid, np.array([0, 0, 0, 0]))
            if (
                defuser is None
                or not defuser.alive
                or defuser.area_id != s.bomb_area_id
                or int(defuser_action[2]) == 0
            ):
                s.bomb_being_defused_by = -1
                s.bomb_defuse_ticks = 0

        for i, aid in enumerate(self.possible_agents):
            agent = s.agents[i]
            if not agent.alive:
                continue
            action = actions.get(aid, np.array([0, 0, 0, 0]))
            if int(action[2]) == 0:
                continue

            # T planting
            if agent.team == 0 and agent.has_bomb and not s.bomb_planted:
                if agent.area_id in self.bombsite_areas:
                    if s.bomb_being_planted_by == -1:
                        s.bomb_being_planted_by = agent.agent_id
                        s.bomb_plant_ticks = 0
                    if s.bomb_being_planted_by == agent.agent_id:
                        s.bomb_plant_ticks += 1
                        if s.bomb_plant_ticks >= BOMB_PLANT_TIME:
                            s.bomb_planted = True
                            s.bomb_area_id = agent.area_id
                            s.bomb_pos = agent.pos.copy()
                            s.bomb_ticks_left = BOMB_TIMER
                            s.bomb_being_planted_by = -1
                            agent.has_bomb = False
                            bomb_just_planted = True
                            _bomb_planter_id = agent.agent_id
                else:
                    # Left site — cancel plant
                    if s.bomb_being_planted_by == agent.agent_id:
                        s.bomb_being_planted_by = -1
                        s.bomb_plant_ticks = 0

            # CT defusing
            elif agent.team == 1 and s.bomb_planted:
                if agent.area_id == s.bomb_area_id:
                    defuse_time = BOMB_DEFUSE_KIT if agent.has_kit else BOMB_DEFUSE_TIME
                    if s.bomb_being_defused_by == -1:
                        s.bomb_being_defused_by = agent.agent_id
                        s.bomb_defuse_ticks = 0
                    if s.bomb_being_defused_by == agent.agent_id:
                        s.bomb_defuse_ticks += 1
                        if s.bomb_defuse_ticks >= defuse_time:
                            s.round_over = True
                            s.winner = 1
                            bomb_just_defused = True
                            _bomb_defuser_id = agent.agent_id

        # 7. Bomb timer countdown
        if s.bomb_planted and not s.round_over:
            s.bomb_ticks_left -= 1
            if s.bomb_ticks_left <= 0:
                s.round_over = True
                s.winner = 0

        # 10. Update enemy memory per agent (sound + vision)
        sounds = self._compute_sounds(s)
        # Pre-group sounds by source_id — avoids iterating all sounds for each
        # agent-enemy pair; _update_enemy_memory does a O(1) dict lookup instead.
        sounds_by_src: dict = {}
        for snd in sounds:
            sounds_by_src.setdefault(snd.source_id, []).append(
                (snd.source_pos[0], snd.source_pos[1], snd.radius * snd.radius)
            )
        for i, agent in enumerate(s.agents):
            if not agent.alive:
                continue
            self._update_enemy_memory(agent, s, sounds_by_src, _vis10)

        # Compute outputs
        self.agents = [
            aid for i, aid in enumerate(self.possible_agents) if s.agents[i].alive
        ]

        # Build obs context (alive counts + vis10 already built above, reuse it)
        alive_t = sum(1 for a in s.agents if a.team == 0 and a.alive) / 5.0
        alive_ct = sum(1 for a in s.agents if a.team == 1 and a.alive) / 5.0
        vis10 = _vis10
        obs = {
            aid: self._compute_obs(i, alive_t, alive_ct, vis10)
            for i, aid in enumerate(self.possible_agents)
            if s.agents[i].alive
        }

        # 9. Compute rewards
        rewards = {aid: 0.0 for aid in self.possible_agents}

        if s.round_over:
            for i, aid in enumerate(self.possible_agents):
                agent = s.agents[i]
                if agent.alive:
                    rewards[aid] += 1.0 if s.winner == agent.team else -1.0

        for killer_id, victim_id in kills_this_tick:
            killer_aid = f"t{killer_id}" if killer_id < 5 else f"ct{killer_id - 5}"
            victim_aid = f"t{victim_id}" if victim_id < 5 else f"ct{victim_id - 5}"
            rewards[killer_aid] += 0.3
            rewards[victim_aid] -= 0.1

        if bomb_just_planted:
            rewards[f"t{_bomb_planter_id}"] += 0.2

        if bomb_just_defused:
            rewards[f"ct{_bomb_defuser_id - 5}"] += 0.2

        if not s.bomb_planted and s.round_ticks_left > ROUND_TIME * 0.5:
            for i, aid in enumerate(self.possible_agents):
                if s.agents[i].alive:
                    rewards[aid] += 0.0001

        # Potential-based reward shaping — Ng et al. ICML 1999
        # F(s,a,s') = γΦ(s') − Φ(s) preserves the optimal policy.
        # γ matches TRAINING_CONFIG["gamma"]=0.99; kept as a local literal to
        # avoid a circular import (sim.py cannot import train.py).
        _PBRS_GAMMA = 0.99
        phi_after = {0: self._potential(s, 0), 1: self._potential(s, 1)}
        for i, aid in enumerate(self.possible_agents):
            agent = s.agents[i]
            rewards[aid] += _PBRS_GAMMA * phi_after[agent.team] - phi_before[agent.team]

        # Team Spirit blending — OpenAI Five pattern (τ=0: individual, τ=1: team avg)
        # Averages over ALIVE agents only; dead agents receive 0.0 reward unchanged.
        # Use the process-safe shared value when available (set by train.py's daemon
        # thread via multiprocessing.Value so all SF worker processes see the same τ).
        _ts = (
            self._team_spirit_shared.value
            if self._team_spirit_shared is not None
            else _TEAM_SPIRIT
        )
        if _ts > 0.0:
            for team in (0, 1):
                alive_aids = [
                    aid
                    for i, aid in enumerate(self.possible_agents)
                    if s.agents[i].team == team and s.agents[i].alive
                ]
                if alive_aids:
                    team_avg = float(np.mean([rewards[aid] for aid in alive_aids]))
                    for aid in alive_aids:
                        rewards[aid] = (1.0 - _ts) * rewards[aid] + _ts * team_avg

        terms = {aid: s.round_over for aid in self.possible_agents}
        truncs = {aid: False for aid in self.possible_agents}
        infos = {aid: {} for aid in self.possible_agents}

        if self._record_fn:
            self._record_fn(s, s.tick, rewards)

        return obs, rewards, terms, truncs, infos

    def _compute_sounds(self, s: GameState) -> list:
        sounds = []
        for agent in s.agents:
            if not agent.alive:
                continue
            if agent.is_moving:
                sounds.append(
                    SoundEvent(
                        source_pos=agent.pos.copy(),
                        source_id=agent.agent_id,
                        radius=FOOTSTEP_RADIUS,
                        type="footstep",
                    )
                )
            if agent.fired_this_tick:
                sounds.append(
                    SoundEvent(
                        source_pos=agent.pos.copy(),
                        source_id=agent.agent_id,
                        radius=GUNSHOT_RADIUS,
                        type="shot",
                    )
                )
        return sounds

    def _potential(self, gs: "GameState", team: int) -> float:
        """Compute potential Φ(s, team) for potential-based reward shaping.

        Φ reflects game-state advantage via alive count, HP, and site control.
        Used as F(s,a,s') = γΦ(s') − Φ(s) per Ng et al. ICML 1999.
        Only counts alive agents — dead agents contribute 0 HP and 0 site presence.
        Single pass over 10 agents instead of 6 separate sum() calls.
        """
        alive_t = alive_o = hp_t = hp_o = site_t = site_o = 0
        _sites = self.bombsite_areas
        for a in gs.agents:
            if not a.alive:
                continue
            if a.team == team:
                alive_t += 1
                hp_t += a.hp
                if a.area_id in _sites:
                    site_t += 1
            else:
                alive_o += 1
                hp_o += a.hp
                if a.area_id in _sites:
                    site_o += 1
        return (
            (alive_t - alive_o) * 0.3 + (hp_t - hp_o) / 500.0 + (site_t - site_o) * 0.2
        )

    def _update_enemy_memory(
        self, agent: AgentState, gs: GameState, sounds_by_src: dict, vis10: np.ndarray
    ):
        # NOTE: parameter named `gs` to avoid collision with any local var `s`
        # sounds_by_src: dict[source_id -> list[(sx, sy, radius_sq)]]
        # vis10: pre-built (10, 10) bool inter-agent visibility for this tick
        agent_idx = agent.agent_id  # 0-9; matches axis in vis10
        enemy_team = 1 - agent.team
        en_start = 5 if agent.team == 0 else 0
        ax = agent.pos[0]
        ay = agent.pos[1]

        for en_j in range(en_start, en_start + 5):
            enemy = gs.agents[en_j]
            if not enemy.alive:
                if enemy.agent_id in agent.enemy_memory:
                    area, _ = agent.enemy_memory[enemy.agent_id]
                    agent.enemy_memory[enemy.agent_id] = (area, -9999)  # stale = dead
                continue

            can_see = vis10[agent_idx, en_j]
            can_hear = False
            if not can_see:
                enemy_sounds = sounds_by_src.get(enemy.agent_id)
                if enemy_sounds:
                    for sx, sy, radius_sq in enemy_sounds:
                        dx = ax - sx
                        dy = ay - sy
                        if dx * dx + dy * dy <= radius_sq:
                            can_hear = True
                            break

            if can_see or can_hear:
                agent.enemy_memory[enemy.agent_id] = (enemy.area_id, gs.tick)
            else:
                last_tick = agent.enemy_memory.get(enemy.agent_id, (None, -9999))[1]
                if last_tick >= 0 and gs.tick - last_tick >= ENEMY_MEMORY_TICKS:
                    agent.enemy_memory.pop(enemy.agent_id, None)

    def render(self):
        pass


# ── SECTION: Tests ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--test-navgraph" in sys.argv:
        nav = NavGraph("C:/Users/vboxuser/.awpy/navs/de_dust2.json")
        assert nav.graph is not None, "graph not built"
        assert len(nav.graph.nodes) > 100, (
            f"expected >100 nodes, got {len(nav.graph.nodes)}"
        )
        assert len(nav.wall_segments) > 0, "no wall segments extracted"
        print(
            f"NavGraph test PASSED — {len(nav.graph.nodes)} nodes, "
            f"{len(nav.wall_segments)} wall segments"
        )

    if "--test-vis" in sys.argv:
        nav = NavGraph("C:/Users/vboxuser/.awpy/navs/de_dust2.json")
        nav.build_vis_matrix()
        assert nav.vis_matrix is not None
        assert nav.vis_matrix.shape == (nav.N, nav.N)
        assert np.array_equal(nav.vis_matrix, nav.vis_matrix.T), (
            "vis matrix not symmetric"
        )
        assert nav.vis_matrix[0, 0] == True
        true_frac = nav.vis_matrix.sum() / nav.vis_matrix.size
        assert 0.005 < true_frac < 0.95, f"suspicious vis fraction: {true_frac:.2f}"
        print(f"Visibility test PASSED — {true_frac:.1%} of pairs are visible")

    if "--test-env-init" in sys.argv:
        env = Dust2Env()
        assert hasattr(env, "possible_agents")
        assert len(env.possible_agents) == 10
        obs, infos = env.reset(seed=42)
        assert len(obs) == 10, f"Expected 10 obs, got {len(obs)}"
        for agent_id, ob in obs.items():
            assert ob.shape == (71,), f"{agent_id}: shape {ob.shape} != (71,)"
            assert np.isfinite(ob).all(), f"{agent_id}: NaN in reset obs"
        print("Env init test PASSED")

    if "--test-movement" in sys.argv:
        env = Dust2Env()
        obs, _ = env.reset(seed=0)
        initial_pos = {
            aid: env.state.agents[i].pos.copy()
            for i, aid in enumerate(env.possible_agents)
        }

        # Action: move North (action[0]=1) for all agents
        actions = {aid: np.array([1, 0, 0, 0]) for aid in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)

        moved_pos = {
            aid: env.state.agents[i].pos for i, aid in enumerate(env.possible_agents)
        }

        any_moved = any(
            not np.allclose(initial_pos[aid], moved_pos[aid])
            for aid in env.possible_agents
        )
        assert any_moved, "No agents moved after movement action"

        for aid, ob in obs.items():
            assert np.isfinite(ob).all(), f"{aid}: NaN after movement"

        print("Movement test PASSED")

    if "--test-obs-masking" in sys.argv:
        env = Dust2Env()
        obs, _ = env.reset(seed=5)

        # Move all CT agents far from T agents
        for ct in env.state.agents[5:]:
            ct.area_id = env.ct_spawn_areas[0]
            c = env.nav_graph.areas[env.ct_spawn_areas[0]].centroid
            ct.pos = np.array([c.x, c.y, c.z], dtype=np.float32)

        # Clear all enemy memory for t0
        env.state.agents[0].enemy_memory = {}

        obs_t0 = env._compute_obs(0)

        # Enemy slots should be zeroed (no memory, no LOS)
        enemy_obs = obs_t0[28:63]  # 5 enemies × 7 = 35 floats
        visible_flags = [enemy_obs[i * 7 + 5] for i in range(5)]
        assert all(f == 0.0 for f in visible_flags), (
            f"Expected no visibility for distant enemies, got {visible_flags}"
        )

        print("Obs masking test PASSED")

    if "--test-bomb" in sys.argv:
        env = Dust2Env()
        obs, _ = env.reset(seed=2)

        # Force T bomb carrier onto bombsite A
        bomber = next(a for a in env.state.agents if a.has_bomb)
        site_area = env.a_site_areas[0]
        bomber.area_id = site_area
        bomber.pos = np.array([*env.nav_graph.centroids[site_area], 0.0])

        # Hold plant for required ticks
        planted = False
        for _ in range(BOMB_PLANT_TIME + 5):
            actions = {aid: np.array([0, 0, 0, 0]) for aid in env.agents}
            actions[f"t{bomber.agent_id}"] = np.array([0, 0, 1, 0])  # plant action
            obs, rewards, terms, truncs, infos = env.step(actions)
            if env.state.bomb_planted:
                planted = True
                break

        assert planted, "Bomb should have been planted after holding plant action"
        assert env.state.bomb_area_id == site_area
        print("Bomb plant test PASSED")

    if "--test-shoot" in sys.argv:
        env = Dust2Env()
        obs, _ = env.reset(seed=1)

        # Force t0 and ct0 into the same area
        t0_agent = env.state.agents[0]
        ct0_agent = env.state.agents[5]
        ct0_agent.area_id = t0_agent.area_id

        # Move ct0 slightly ahead in t0's facing direction so the shot connects
        dx_face = np.cos(t0_agent.facing)
        dy_face = np.sin(t0_agent.facing)
        ct0_agent.pos = t0_agent.pos + np.array([dx_face * 50, dy_face * 50, 0.0])

        # T agent 0 shoots, all others stand still
        actions = {aid: np.array([0, 0, 0, 0]) for aid in env.agents}
        actions["t0"] = np.array([0, 1, 0, 0])  # shoot

        initial_ct0_hp = ct0_agent.hp
        env.step(actions)

        assert (
            env.state.agents[5].hp < initial_ct0_hp or not env.state.agents[5].alive
        ), "Shooting in same area should deal damage"

        print("Shoot test PASSED")

    if "--test-sb3-wrap" in sys.argv:
        from supersuit import concat_vec_envs_v1, pettingzoo_env_to_vec_env_v1

        vec_env = concat_vec_envs_v1(
            lambda: pettingzoo_env_to_vec_env_v1(Dust2Env()),
            1,
            num_cpus=1,
            base_class="stable_baselines3",
        )

        obs = vec_env.reset()
        # obs might be (obs_arr, infos) tuple in newer gymnasium versions
        if isinstance(obs, tuple):
            obs = obs[0]
        assert obs.shape[1] == 71, f"Expected obs dim 71, got {obs.shape}"
        print(f"SB3 wrap test PASSED — obs shape: {obs.shape}")
