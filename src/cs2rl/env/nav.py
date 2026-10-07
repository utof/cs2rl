# ── SECTION: NavGraph ──────────────────────────────────────────────────────

import math
import os
import pathlib
from collections import deque

import networkx as nx
import numpy as np
from awpy import Nav
from awpy.nav import NavArea
from shapely.geometry import Point
from shapely.geometry import Polygon as ShapelyPolygon

import cs2rl
from cs2rl.spec.obs import (           # noqa: F401  generated; re-exported (see Constants)
    OBS_BLOCKS, OBS_DIM,
)

# ── Multiprocessing workers for vis matrix (must be module-level to be picklable) ──

_vc = None                             # per-worker VisibilityChecker instance


def _die_with_parent(parent_pid: int, poll_s: float = 0.5) -> None:
    """Make THIS process die when the process that forked it dies (gh#254).

    WHAT: two independent guards, installed once in a pool worker's initializer.
      1. Linux `prctl(PR_SET_PDEATHSIG, SIGKILL)`: the kernel SIGKILLs us the
         moment the parent THREAD that forked us exits. Kernel-level: no
         0.5 s polling latency, and immune to a worker stuck in GIL-holding
         native code, which (2) is not.
      2. A daemon watchdog thread that polls `os.getppid()` every `poll_s`
         and SIGKILLs this process when it stops being `parent_pid`. Portable
         (no prctl on macOS), and the fallback if (1) is unavailable.
    Both are followed by a re-check of `os.getppid()`: if the parent already
    died between the fork and the prctl call, PDEATHSIG never fires (it is
    only sent on a FUTURE parent exit), so we exit right here.

    WHY: `build_vis_matrix` forks `os.cpu_count()` workers that each grow to
    ~900 MB. `ProcessPoolExecutor.__exit__` is the only thing that stops
    them, so any parent death that skips it (pytest-timeout's `os._exit`, an
    outer `timeout`, SIGKILL, OOM) reparented all 12 to PID 1 and they kept
    computing: 24 orphans from two killed parents took the 16 GB box to
    15.3 GB used on 2026-09-25.

    PITFALLS:
    - PDEATHSIG is per forking THREAD, not per process: if the thread that
      called `Process.start()` exits while the parent lives, the worker is
      killed anyway. That cannot happen in `build_vis_matrix` — under the
      `fork` start method every worker is forked by the thread that made the
      first `submit()` (measured: `_launch_processes` runs there), and that
      thread is blocked in `as_completed` inside the `with` block until every
      future is done; replacement workers are forked by the executor's
      manager thread, which lives until shutdown. A caller that starts the
      build on a thread and lets that thread die mid-build would regress.
    - `parent_pid` must be captured in the PARENT (before the fork) and passed
      in; `os.getppid()` from inside a worker whose parent is already dead
      returns the reaper's pid, which is exactly the case we must detect.
    - Use SIGKILL on ourselves, not `sys.exit`: from the watchdog thread
      `sys.exit` would end only that thread, not the worker.
    - `fork` start method ONLY, pinned at the pool (`mp_context`). Under
      `forkserver` every worker's parent is the fork server, not the caller,
      so the post-prctl re-check SIGKILLs each worker in its initializer and
      the build dies with BrokenProcessPool (measured on a 2-worker pool:
      fork ok, spawn ok, forkserver broken). Python 3.14 makes forkserver
      the Linux default, so the pin is what keeps a cold build working there;
      the test's pre-fork stub injection depends on fork as well.
    - Never call this in the parent: it would arm the parent to die with ITS
      parent (the shell).
    """
    import os
    import signal
    import sys
    import threading

    def _suicide():
        os.kill(os.getpid(), signal.SIGKILL)

    if sys.platform == "linux":
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(1, int(signal.SIGKILL), 0, 0, 0)    # 1 == PR_SET_PDEATHSIG
    if os.getppid() != parent_pid:                     # parent died before the prctl landed
        _suicide()

    def _watch():
        import time
        while True:
            time.sleep(poll_s)
            if os.getppid() != parent_pid:
                _suicide()

    threading.Thread(target=_watch, name="cs2rl-parent-watchdog", daemon=True).start()


def _vis_worker_init(tri_path_str: str, parent_pid: int):
    """Runs once per worker process: arms the parent-death guard (gh#254), then loads
    VisibilityChecker into a module global. The guard goes FIRST so a worker whose
    parent dies during the ~40 s checker load is killed too."""
    global _vc
    from pathlib import Path

    _die_with_parent(parent_pid)
    from awpy.visibility import VisibilityChecker

    _vc = VisibilityChecker(path=Path(tri_path_str))


def _vis_compute_rows(row_indices: list, pts: list) -> dict:
    """Computes upper-triangle visibility for each row in row_indices.

    Returns {i: [bool for j in range(i+1, N)]} using the worker-local _vc.
    """
    assert _vc is not None, "_vis_worker_init (the pool initializer) runs before any task"
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
        _area_polys  -- list of shapely Polygons (one per area, index = _id_to_idx);
                        _build_pos_grid rasterises them
        vis_matrix   -- None (built in Task 2)
        _nav_path    -- stored nav file path
        _cache_path  -- stored vis cache path
    """

    def __init__(self, nav_path: str, cache_path: str):
        self._nav_path = nav_path
        self._cache_path = cache_path

        # ── Load nav data ──────────────────────────────────────────────────
        self.nav = Nav.from_json(nav_path)
        self.areas: dict[int, NavArea] = self.nav.areas
        # sorted for stable _id_to_idx indices across runs
        self.area_ids: list[int] = sorted(self.areas.keys())
        self.N: int = len(self.area_ids)
        self._id_to_idx: dict[int, int] = {aid: i for i, aid in enumerate(self.area_ids)}

        # ── Compute centroids ──────────────────────────────────────────────
        self.centroids: dict[int, np.ndarray] = {}
        for aid, area in self.areas.items():
            c = area.centroid
            self.centroids[aid] = np.array([c.x, c.y], dtype=np.float32)

        # Pre-built (N, 2) matrix for vectorised nearest-centroid lookups
        # shape (N, 2)
        self._centroid_matrix: np.ndarray = np.array(
            [self.centroids[aid] for aid in self.area_ids],
            dtype=np.float32,
        )

        # Pre-built (N, 3) matrix for 3D snapping (spawn slots need z to avoid floor mismatches)
        self._centroid_matrix_3d: np.ndarray = np.array(
            [[
                self.areas[aid].centroid.x,
                self.areas[aid].centroid.y,
                self.areas[aid].centroid.z,
            ] for aid in self.area_ids],
            dtype=np.float32,
        )                                                              # shape (N, 3)

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

        # ── Area polygons (the position grid below rasterises them) ─────────
        self._area_polys = self._build_area_polys()

        # ── Rasterized grid for O(1) position lookup ───────────────────────
        # env/map.py exports it to C, whose _raster_at (cs2_movement.h) does the per-step lookup.
        self._grid_cache_path = cache_path.replace(".npy", "_grid.npy")
        self._build_pos_grid()

        # ── Visibility matrix (built in Task 2) ───────────────────────────
        self.vis_matrix: np.ndarray | None = None

    # ── Wall segment extraction ────────────────────────────────────────────

    def _extract_wall_segments(self, ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
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

    # ── Area polygons ─────────────────────────────────────────────────────

    def _build_area_polys(self):
        """Build one shapely Polygon per area, in area_ids order (index = _id_to_idx).

        Only _build_pos_grid reads them. The STRtree that used to be built here fed
        the nearest-area queries #205 deleted, and nothing read it afterwards.
        """
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
        return polys

    def _build_pos_grid(self, cell_size: float = 4.0):
        """Build a rasterized 2D grid mapping (gx, gy) → area_idx for O(1) lookups.

        env/map.py exports it to C as MapData.grid, where _raster_at (cs2_movement.h)
        does the per-step position lookup.  Cached to disk alongside the vis matrix.

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

        all_bounds = np.array([p.bounds for p in self._area_polys])    # (N, 4)
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
            cxs = xs[gx0:gx1 + 1]
            cys = ys[gy0:gy1 + 1]
            xx, yy = np.meshgrid(cxs, cys)
            inside = sh.contains_xy(poly, xx.ravel(), yy.ravel()).reshape(xx.shape)
            grid[gy0:gy1 + 1, gx0:gx1 + 1][inside] = i

        self._grid_cell_size = cell_size
        self._grid_inv_cell = inv
        self._grid_x_min = x_min
        self._grid_y_min = y_min
        self._grid_w = W
        self._grid_h = H
        self._pos_grid = grid

        coverage = (grid >= 0).mean()
        print(
            f"[NavGraph] Built position grid {W}×{H} (cell={cell_size}u, {coverage:.1%} coverage)")

        if cache:
            # A dict saved as a 0-d object array (what np.save would wrap it in anyway;
            # same bytes); the cache branch above unwraps it with .item().
            np.save(
                cache,
                np.array(
                    {
                        "cell": cell_size,
                        "x_min": x_min,
                        "y_min": y_min,
                        "w": W,
                        "h": H,
                        "grid": grid,
                    },
                    dtype=object,
                ),
            )

    # ── Public API ────────────────────────────────────────────────────────

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
        import multiprocessing
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
                f".tri file not found at {tri_path}\nDownload it with:  awpy get tris")

        pts = [(
            float(self.areas[aid].centroid.x),
            float(self.areas[aid].centroid.y),
            float(self.areas[aid].centroid.z),
        ) for aid in self.area_ids]

        n_workers = os.cpu_count() or 4
        # ~8 tasks per worker for good load-balancing without excessive IPC overhead
        chunk_size = max(1, self.N // (n_workers * 8))
        chunks = [list(range(i, min(i + chunk_size, self.N))) for i in range(0, self.N, chunk_size)]

        print(f"[NavGraph] Building {self.N}×{self.N} vis matrix "
              f"({n_workers} workers, {len(chunks)} chunks)...")
        print("[NavGraph] Workers initialising VisibilityChecker in parallel (~40s)...")
        t0 = time.time()

        partial_rows: dict = {}
        # gh#254: workers die with THIS process (see _die_with_parent). os.getpid()
        # is captured here, in the parent, and handed to every worker; a worker
        # reading os.getppid() after its parent died would see the reaper instead.
        # The start method is pinned to fork: the guard compares against the pid
        # of the process that forked the worker, which under forkserver is the
        # fork server, so an unpinned pool would kill every worker at init once
        # forkserver becomes the Linux default (Python 3.14).
        with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_vis_worker_init,
                initargs=(str(tri_path), os.getpid()),
                mp_context=multiprocessing.get_context("fork"),
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
ENEMY_MEMORY_TICKS = 32

# Direction vectors for movement actions (built once at import time)
_DIR_VECTORS = {
    0: np.array([0.0, 0.0]),
    1: np.array([0.0, 1.0]),                                           # N
    2: np.array([0.7071067811865476, 0.7071067811865476]),             # NE (pre-normalised)
    3: np.array([1.0, 0.0]),                                           # E
    4: np.array([0.7071067811865476, -0.7071067811865476]),            # SE
    5: np.array([0.0, -1.0]),                                          # S
    6: np.array([-0.7071067811865476, -0.7071067811865476]),           # SW
    7: np.array([-1.0, 0.0]),                                          # W
    8: np.array([-0.7071067811865476, 0.7071067811865476]),            # NW
}

# Pre-scaled delta per direction
_DELTA_VECTORS = {k: v * (MOVE_SPEED * DT) for k, v in _DIR_VECTORS.items()}

# Pre-computed facing angle per direction
_DIR_FACING = {k: math.atan2(float(v[1]), float(v[0])) for k, v in _DIR_VECTORS.items()}

# 45 deg/tick max facing rotation rate
MAX_TURN_SPEED_RAD = math.pi / 4

N_AGENTS = 10
TEAM_SIZE = 5
# OBS_DIM + OBS_BLOCKS are imported at module top from spec.obs (generated from
# cs2_types.h by scripts/sync_action_spec.py) — the single source of truth for
# the obs layout. Do NOT reintroduce a literal here; a bump is a cs2_types.h edit
# followed by `uv run python scripts/sync_action_spec.py`.

STALE_MEMORY_TICK = -9999

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
# The vis cache sits at the PACKAGE root, src/cs2rl/vis_cache.npy, anchored on the cs2rl
# package and never on this file: when #205 moved nav into env/, a `__file__`-relative path
# silently moved the cache with it. A moved path is a cold cache, and a cold cache on dust2
# rebuilds the grid, then the vis matrix through a cpu_count() worker pool (~900 MB each,
# orphaned on a kill). *.npy is gitignored, so a stray copy anywhere is invisible to git
# status. tests/integration/test_path_constants_exist.py pins this location.
# PITFALL: not importlib.resources.files("cs2rl"). It promises only a Traversable, whose str()
# is a filesystem path for a regular on-disk package but not under zipimport or for a namespace
# package, and it is framed as read-only package data, not a writable cache.
CACHE_PATH = str(pathlib.Path(cs2rl.__file__).resolve().parent / "vis_cache.npy")


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
    actually enter via the raster lookup (C's _raster_at) + fixed XY moves. The cell
    rule is _raster_adjacency's.
    """
    return _raster_adjacency(nav_graph._pos_grid, nav_graph.N)


def _raster_adjacency(grid: np.ndarray, n: int) -> np.ndarray:
    """Area adjacency of a raster: two areas touch where any of their cells are 8-neighbours.

    `grid` is int[H, W], cell -> area index, -1 off-mesh; returns bool[n, n]. The diagonal is
    True (staying in an area is always a legal move) and the matrix is symmetric.
    make_cs2_map reaches this through _build_area_adjacency (the nav mesh's raster).
    """
    adj = np.zeros((n, n), dtype=bool)
    np.fill_diagonal(adj, True)

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
    """Hop distance from every area to the nearest of `targets` (area IDS), by area index.

    Target ids the nav graph does not know are dropped; the BFS itself is _hop_distances.
    """
    target_ids = [target for target in targets if target in nav_graph._id_to_idx]
    return _hop_distances(area_adjacency, [nav_graph._id_to_idx[target] for target in target_ids])


def _hop_distances(adjacency: np.ndarray, target_idxs) -> np.ndarray:
    """Multi-source BFS over a bool[N, N] adjacency: float32[N] hop counts to the nearest target.

    `target_idxs` are area INDICES (rows of `adjacency`), not area ids: make_cs2_map reaches
    this through _compute_area_distance_to_targets, which maps ids to indices. Targets are
    0.0; an area no target reaches stays np.inf, and with no targets every entry is np.inf.
    make_cs2_map replaces the inf entries with a finite sentinel only after computing its
    scale from the finite ones.
    """
    dist = np.full(adjacency.shape[0], np.inf, dtype=np.float32)
    q = deque()
    for idx in target_idxs:
        dist[idx] = 0.0
        q.append(idx)

    while q:
        idx = q.popleft()
        next_dist = dist[idx] + 1.0
        neighbors = np.flatnonzero(adjacency[idx])
        for nbr in neighbors:
            nbr = int(nbr)
            if next_dist >= dist[nbr]:
                continue
            dist[nbr] = next_dist
            q.append(nbr)

    return dist
