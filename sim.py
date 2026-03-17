# ── SECTION: NavGraph ──────────────────────────────────────────────────────

import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

from awpy import Nav
from shapely.geometry import LineString, Point, Polygon as ShapelyPolygon
from shapely.strtree import STRtree


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
        self.area_ids: List[int] = sorted(self.areas.keys())  # sorted for stable _id_to_idx indices across runs
        self.N: int = len(self.area_ids)
        self._id_to_idx: Dict[int, int] = {aid: i for i, aid in enumerate(self.area_ids)}

        # ── Compute centroids ──────────────────────────────────────────────
        self.centroids: Dict[int, np.ndarray] = {}
        for aid, area in self.areas.items():
            c = area.centroid
            self.centroids[aid] = np.array([c.x, c.y], dtype=np.float32)

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

        # ── Extract wall segments ──────────────────────────────────────────
        self.wall_segments = self._extract_wall_segments()

        # ── Build spatial indices ──────────────────────────────────────────
        self._wall_lines: List[LineString] = [
            LineString(seg) for seg in self.wall_segments
        ]
        self._wall_strtree = STRtree(self._wall_lines) if self._wall_lines else STRtree([])

        self._area_polys, self._area_strtree = self._build_area_index()

        # ── Visibility matrix (built in Task 2) ───────────────────────────
        self.vis_matrix = None

    # ── Wall segment extraction ────────────────────────────────────────────

    def _extract_wall_segments(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Extract boundary edges — edges shared by exactly one area polygon."""
        edge_count: Dict[Tuple, int] = {}

        for area in self.areas.values():
            corners = area.corners
            n = len(corners)
            for i in range(n):
                p1 = (round(corners[i].x, 4), round(corners[i].y, 4))
                p2 = (round(corners[(i + 1) % n].x, 4), round(corners[(i + 1) % n].y, 4))
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
        # Fallback: nearest centroid (always returns a valid area_id)
        dists = [np.linalg.norm(pos_xy - self.centroids[aid]) for aid in self.area_ids]
        return self.area_ids[int(np.argmin(dists))]

    def can_see(self, area_i: int, area_j: int) -> bool:
        """Return True if area_i can see area_j (requires vis_matrix from Task 2).

        Falls back to graph connectivity if vis_matrix not yet built.
        Returns False for any unknown area_id rather than raising KeyError.
        """
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
