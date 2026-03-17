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

        # Pre-built (N, 2) matrix for vectorised nearest-centroid lookups
        self._centroid_matrix: np.ndarray = np.array(
            [self.centroids[aid] for aid in self.area_ids], dtype=np.float32
        )  # shape (N, 2)

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
        # Fallback: nearest centroid — vectorised over all N areas
        diff = self._centroid_matrix - pos_xy  # (N, 2)
        idx = int(np.argmin((diff * diff).sum(axis=1)))
        return self.area_ids[idx]

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
        """Build and cache the N×N visibility matrix."""
        import os
        import time
        from shapely.geometry import LineString

        nav_mtime = os.path.getmtime(self._nav_path) if hasattr(self, '_nav_path') else 0

        if os.path.exists(self._cache_path):
            cache_mtime = os.path.getmtime(self._cache_path)
            if cache_mtime > nav_mtime:
                self.vis_matrix = np.load(self._cache_path)
                print(f"[NavGraph] Loaded visibility matrix from cache ({self.N}×{self.N})")
                return

        print(f"[NavGraph] Building visibility matrix ({self.N}×{self.N})... "
              f"(this takes 2-5 min, cached after)")
        t0 = time.time()

        vis = np.zeros((self.N, self.N), dtype=bool)
        centroids = [self.centroids[aid] for aid in self.area_ids]

        for i in range(self.N):
            if i % 100 == 0:
                elapsed = time.time() - t0
                eta = (elapsed / max(i, 1)) * (self.N - i)
                print(f"  [{i}/{self.N}] elapsed={elapsed:.0f}s ETA={eta:.0f}s")

            cx, cy = centroids[i]

            for j in range(i, self.N):
                if i == j:
                    vis[i][j] = True
                    continue

                dx, dy = centroids[j]
                ray = LineString([(cx, cy), (dx, dy)])

                candidates = self._wall_strtree.query(ray)

                blocked = False
                for k in candidates:
                    wall = self._wall_lines[k]
                    if ray.crosses(wall):
                        blocked = True
                        break

                vis[i][j] = vis[j][i] = not blocked

        self.vis_matrix = vis
        np.save(self._cache_path, vis)
        elapsed = time.time() - t0
        print(f"[NavGraph] Visibility matrix built in {elapsed:.0f}s, cached to {self._cache_path}")


# ── SECTION: Constants ─────────────────────────────────────────────────────

MOVE_SPEED          = 250
TICK_RATE           = 64
DT                  = 1.0 / TICK_RATE
LASER_DAMAGE        = 100
LASER_RANGE         = 3000
SHOOT_COOLDOWN      = 10
BOMB_PLANT_TIME     = int(3.2 * TICK_RATE)
BOMB_DEFUSE_TIME    = 10 * TICK_RATE
BOMB_DEFUSE_KIT     = 5 * TICK_RATE
BOMB_TIMER          = int(40 * TICK_RATE)
ROUND_TIME          = int(115 * TICK_RATE)
FOOTSTEP_RADIUS     = 800
GUNSHOT_RADIUS      = 2000
BOMB_BEEP_RADIUS    = 1500
ENEMY_MEMORY_TICKS  = 32

MAP_X_MIN, MAP_X_MAX = -2476.0, 2000.0
MAP_Y_MIN, MAP_Y_MAX = -1050.0, 3420.0

# Precomputed reciprocals for _norm_xy — avoids repeated division inside step()
_INV_MAP_X_RANGE = 2.0 / (MAP_X_MAX - MAP_X_MIN)
_INV_MAP_Y_RANGE = 2.0 / (MAP_Y_MAX - MAP_Y_MIN)
_MAP_X_OFFSET    = (MAP_X_MAX + MAP_X_MIN) / (MAP_X_MAX - MAP_X_MIN)
_MAP_Y_OFFSET    = (MAP_Y_MAX + MAP_Y_MIN) / (MAP_Y_MAX - MAP_Y_MIN)

# Direction vectors for movement actions (built once at import time)
_DIR_VECTORS = {
    0: np.array([0.0,  0.0]),
    1: np.array([0.0,  1.0]),    # N
    2: np.array([0.7071067811865476,  0.7071067811865476]),  # NE (pre-normalised)
    3: np.array([1.0,  0.0]),    # E
    4: np.array([0.7071067811865476, -0.7071067811865476]),  # SE
    5: np.array([0.0, -1.0]),    # S
    6: np.array([-0.7071067811865476, -0.7071067811865476]),  # SW
    7: np.array([-1.0, 0.0]),    # W
    8: np.array([-0.7071067811865476,  0.7071067811865476]),  # NW
}

# ── SECTION: Dataclasses ───────────────────────────────────────────────────

@dataclass
class AgentState:
    agent_id:    int
    team:        int          # 0 = T, 1 = CT
    pos:         np.ndarray   # [x, y, z] HU
    area_id:     int
    facing:      float        # radians, 0 = +X
    hp:          int
    alive:       bool
    has_bomb:    bool
    has_kit:     bool
    shoot_cd:    int
    is_moving:   bool
    fired_this_tick: bool
    enemy_memory: dict = field(default_factory=dict)  # {enemy_id: (area_id, tick)}

@dataclass
class GameState:
    tick:                  int
    agents:                list
    bomb_planted:          bool
    bomb_carrier_id:       int
    bomb_area_id:          int
    bomb_pos:              np.ndarray
    bomb_ticks_left:       int
    bomb_being_planted_by: int
    bomb_plant_ticks:      int
    bomb_being_defused_by: int
    bomb_defuse_ticks:     int
    round_ticks_left:      int
    round_over:            bool
    winner:                int   # 0=T, 1=CT, -1=ongoing

@dataclass
class SoundEvent:
    source_pos: np.ndarray
    source_id:  int
    radius:     float
    type:       str

# ── SECTION: Dust2Env ──────────────────────────────────────────────────────

from pettingzoo import ParallelEnv
import gymnasium
from gymnasium import spaces
import pathlib

NAV_PATH   = "C:/Users/vboxuser/.awpy/navs/de_dust2.json"
CACHE_PATH = "vis_cache.npy"

class Dust2Env(ParallelEnv):
    metadata = {"name": "dust2_v0", "render_modes": []}
    render_mode = None

    def __init__(self, nav_path=NAV_PATH, cache_path=CACHE_PATH, record_fn=None):
        super().__init__()
        self.nav_graph = NavGraph(nav_path, cache_path)
        self.nav_graph.build_vis_matrix()

        self._calibrate_map_bounds()
        self._identify_special_areas()

        self.possible_agents = [f"t{i}" for i in range(5)] + [f"ct{i}" for i in range(5)]
        self.agents = list(self.possible_agents)
        self._record_fn = record_fn
        self.state: GameState = None

    def _calibrate_map_bounds(self):
        global MAP_X_MIN, MAP_X_MAX, MAP_Y_MIN, MAP_Y_MAX
        xs = [c[0] for c in self.nav_graph.centroids.values()]
        ys = [c[1] for c in self.nav_graph.centroids.values()]
        MAP_X_MIN, MAP_X_MAX = min(xs), max(xs)
        MAP_Y_MIN, MAP_Y_MAX = min(ys), max(ys)
        print(f"[Dust2Env] Map bounds: X=[{MAP_X_MIN:.0f},{MAP_X_MAX:.0f}] Y=[{MAP_Y_MIN:.0f},{MAP_Y_MAX:.0f}]")

    def _identify_special_areas(self):
        A_SITE  = np.array([720.0, 2600.0])
        B_SITE  = np.array([-1278.0, 820.0])
        T_SPAWN = np.array([-500.0, -300.0])
        CT_SPAWN= np.array([800.0, 3100.0])

        def areas_near(target, radius=400):
            return [aid for aid, c in self.nav_graph.centroids.items()
                    if np.linalg.norm(c - target) < radius]

        self.a_site_areas  = areas_near(A_SITE,  400) or [self.nav_graph.area_ids[0]]
        self.b_site_areas  = areas_near(B_SITE,  400) or [self.nav_graph.area_ids[1]]
        self.t_spawn_areas = areas_near(T_SPAWN, 600) or [self.nav_graph.area_ids[2]]
        self.ct_spawn_areas= areas_near(CT_SPAWN,600) or [self.nav_graph.area_ids[3]]
        self.bombsite_areas= set(self.a_site_areas + self.b_site_areas)

        print(f"[Dust2Env] A-site: {len(self.a_site_areas)} areas, B-site: {len(self.b_site_areas)} areas")

    def observation_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(71,), dtype=np.float32)

    def action_space(self, agent):
        return spaces.MultiDiscrete([9, 2, 2, 2])

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.agents = list(self.possible_agents)
        self.state = self._make_initial_state()
        obs = {aid: self._compute_obs(i) for i, aid in enumerate(self.possible_agents)}
        infos = {aid: {} for aid in self.possible_agents}
        return obs, infos

    def _make_initial_state(self):
        agents = []
        bomb_carrier = np.random.randint(0, 5)

        for i in range(5):
            spawn_area = self.t_spawn_areas[i % len(self.t_spawn_areas)]
            centroid = self.nav_graph.centroids[spawn_area]
            agents.append(AgentState(
                agent_id=i, team=0,
                pos=np.array([centroid[0], centroid[1], 0.0]),
                area_id=spawn_area,
                facing=0.0, hp=100, alive=True,
                has_bomb=(i == bomb_carrier),
                has_kit=False, shoot_cd=0,
                is_moving=False, fired_this_tick=False,
            ))

        for i in range(5):
            spawn_area = self.ct_spawn_areas[i % len(self.ct_spawn_areas)]
            centroid = self.nav_graph.centroids[spawn_area]
            agents.append(AgentState(
                agent_id=5+i, team=1,
                pos=np.array([centroid[0], centroid[1], 0.0]),
                area_id=spawn_area,
                facing=np.pi, hp=100, alive=True,
                has_bomb=False,
                has_kit=(np.random.random() < 0.5),
                shoot_cd=0, is_moving=False, fired_this_tick=False,
            ))

        return GameState(
            tick=0, agents=agents,
            bomb_planted=False, bomb_carrier_id=bomb_carrier,
            bomb_area_id=-1, bomb_pos=np.zeros(3),
            bomb_ticks_left=0,
            bomb_being_planted_by=-1, bomb_plant_ticks=0,
            bomb_being_defused_by=-1, bomb_defuse_ticks=0,
            round_ticks_left=ROUND_TIME, round_over=False, winner=-1,
        )

    def _norm_xy(self, pos):
        return np.array([
            pos[0] * _INV_MAP_X_RANGE - _MAP_X_OFFSET,
            pos[1] * _INV_MAP_Y_RANGE - _MAP_Y_OFFSET,
        ], dtype=np.float32)

    def _compute_obs(self, agent_idx: int) -> np.ndarray:
        obs = np.zeros(71, dtype=np.float32)
        s = self.state
        agent = s.agents[agent_idx]
        team = agent.team

        obs[0] = float(team)
        obs[1:3] = self._norm_xy(agent.pos)
        obs[3] = np.sin(agent.facing)
        obs[4] = np.cos(agent.facing)
        obs[5] = agent.hp / 100.0
        obs[6] = float(agent.has_bomb if team == 0 else agent.has_kit)
        obs[7] = 1.0 if agent.shoot_cd == 0 else (1.0 - agent.shoot_cd / SHOOT_COOLDOWN)

        teammates = [a for a in s.agents if a.team == team and a.agent_id != agent.agent_id]
        for i, tm in enumerate(teammates[:4]):
            base = 8 + i * 5
            obs[base:base+2] = self._norm_xy(tm.pos)
            obs[base+2] = np.sin(tm.facing)
            obs[base+3] = np.cos(tm.facing)
            obs[base+4] = tm.hp / 100.0 if tm.alive else 0.0

        enemy_team = 1 - team
        enemies = sorted([a for a in s.agents if a.team == enemy_team], key=lambda a: a.agent_id)
        for i, en in enumerate(enemies[:5]):
            base = 28 + i * 7
            mem = agent.enemy_memory.get(en.agent_id)
            can_see = self.nav_graph.can_see(agent.area_id, en.area_id) if en.alive else False

            if mem is None and not can_see:
                continue

            if mem is not None:
                mem_area, last_tick = mem
                mem_centroid = self.nav_graph.centroids.get(mem_area, agent.pos[:2])
                obs[base:base+2] = self._norm_xy(mem_centroid)
                obs[base+2] = np.sin(en.facing)
                obs[base+3] = np.cos(en.facing)
                obs[base+4] = en.hp / 100.0 if en.alive else 0.0
                obs[base+5] = 1.0 if can_see else 0.0
                freshness = (max(0, ENEMY_MEMORY_TICKS - (s.tick - last_tick)) / ENEMY_MEMORY_TICKS
                             if last_tick >= 0 else 0.0)
                obs[base+6] = freshness

        obs[63] = float(s.bomb_planted)
        if s.bomb_planted:
            obs[64:66] = self._norm_xy(s.bomb_pos[:2])
            obs[66] = s.bomb_ticks_left / BOMB_TIMER
        else:
            obs[64:66] = -1.0
            obs[66] = 0.0
        obs[67] = s.round_ticks_left / ROUND_TIME
        obs[68] = sum(1 for a in s.agents if a.team == 0 and a.alive) / 5.0
        obs[69] = sum(1 for a in s.agents if a.team == 1 and a.alive) / 5.0
        obs[70] = float(agent.area_id in self.bombsite_areas)

        return np.clip(obs, -1.0, 1.0)

    def step(self, actions: dict):
        s = self.state
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

            direction = _DIR_VECTORS[move_dir]  # pre-normalised at module level

            delta = direction * MOVE_SPEED * DT
            target_pos = agent.pos[:2] + delta

            # Update facing from movement direction
            agent.facing = np.arctan2(direction[1], direction[0])

            # Find target area
            target_area = self.nav_graph.get_area(target_pos)

            # Move only if target area is connected to current area (or same area)
            if (target_area == agent.area_id or
                    self.nav_graph.graph.has_edge(agent.area_id, target_area)):
                agent.pos = np.array([target_pos[0], target_pos[1], agent.pos[2]])
                agent.area_id = target_area
                agent.is_moving = True

        # 4. Process shoot actions (simultaneous)
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
            dx = np.cos(agent.facing)
            dy = np.sin(agent.facing)

            # Find enemies along ray — use vis_matrix as proxy
            enemy_team = 1 - agent.team
            enemies_alive = [a for a in s.agents if a.team == enemy_team and a.alive]

            best_enemy = None
            best_dist = LASER_RANGE

            for enemy in enemies_alive:
                # Check LOS via vis_matrix (same area always visible)
                if not self.nav_graph.can_see(agent.area_id, enemy.area_id):
                    continue

                rel = enemy.pos[:2] - agent.pos[:2]
                dist = np.linalg.norm(rel)
                if dist > LASER_RANGE or dist == 0:
                    continue

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
            defuser = next((a for a in s.agents if a.agent_id == s.bomb_being_defused_by), None)
            defuser_aid = (f"ct{s.bomb_being_defused_by - 5}")
            defuser_action = actions.get(defuser_aid, np.array([0, 0, 0, 0]))
            if (defuser is None or not defuser.alive or
                    defuser.area_id != s.bomb_area_id or
                    int(defuser_action[2]) == 0):
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
        for i, agent in enumerate(s.agents):
            if not agent.alive:
                continue
            self._update_enemy_memory(agent, s, sounds)

        # Compute outputs
        self.agents = [aid for i, aid in enumerate(self.possible_agents)
                       if s.agents[i].alive]

        obs = {aid: self._compute_obs(i)
               for i, aid in enumerate(self.possible_agents) if s.agents[i].alive}

        # 9. Compute rewards
        rewards = {aid: 0.0 for aid in self.possible_agents}

        if s.round_over:
            for i, aid in enumerate(self.possible_agents):
                agent = s.agents[i]
                if agent.alive:
                    if s.winner == agent.team:
                        rewards[aid] += 1.0
                    else:
                        rewards[aid] -= 1.0

        for killer_id, victim_id in kills_this_tick:
            killer_aid = (f"t{killer_id}" if killer_id < 5 else f"ct{killer_id - 5}")
            victim_aid  = (f"t{victim_id}" if victim_id < 5 else f"ct{victim_id - 5}")
            rewards[killer_aid] += 0.3
            rewards[victim_aid] -= 0.1

        if bomb_just_planted:
            planter_aid = f"t{_bomb_planter_id}"
            rewards[planter_aid] += 0.2

        if bomb_just_defused:
            defuser_aid = f"ct{_bomb_defuser_id - 5}"
            rewards[defuser_aid] += 0.2

        # Survival bonus (first half of round, no bomb planted)
        if not s.bomb_planted and s.round_ticks_left > ROUND_TIME * 0.5:
            for i, aid in enumerate(self.possible_agents):
                if s.agents[i].alive:
                    rewards[aid] += 0.0001

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
                sounds.append(SoundEvent(
                    source_pos=agent.pos.copy(),
                    source_id=agent.agent_id,
                    radius=FOOTSTEP_RADIUS,
                    type="footstep",
                ))
            if agent.fired_this_tick:
                sounds.append(SoundEvent(
                    source_pos=agent.pos.copy(),
                    source_id=agent.agent_id,
                    radius=GUNSHOT_RADIUS,
                    type="shot",
                ))
        return sounds

    def _update_enemy_memory(self, agent: AgentState, gs: GameState, sounds: list):
        # NOTE: parameter named `gs` to avoid collision with any local var `s`
        enemy_team = 1 - agent.team
        enemies = [a for a in gs.agents if a.team == enemy_team]

        for enemy in enemies:
            if not enemy.alive:
                if enemy.agent_id in agent.enemy_memory:
                    area, _ = agent.enemy_memory[enemy.agent_id]
                    agent.enemy_memory[enemy.agent_id] = (area, -9999)  # stale = dead
                continue

            can_see = self.nav_graph.can_see(agent.area_id, enemy.area_id)
            can_hear = any(
                snd.source_id == enemy.agent_id and
                np.linalg.norm(agent.pos[:2] - snd.source_pos[:2]) <= snd.radius
                for snd in sounds
                if snd.source_id // 5 != agent.agent_id // 5  # enemy team only
            )

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
        assert np.array_equal(nav.vis_matrix, nav.vis_matrix.T), "vis matrix not symmetric"
        assert nav.vis_matrix[0, 0] == True
        true_frac = nav.vis_matrix.sum() / nav.vis_matrix.size
        assert 0.005 < true_frac < 0.95, f"suspicious vis fraction: {true_frac:.2f}"
        print(f"Visibility test PASSED — {true_frac:.1%} of pairs are visible")

    if "--test-env-init" in sys.argv:
        env = Dust2Env()
        assert hasattr(env, 'possible_agents')
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
        initial_pos = {aid: env.state.agents[i].pos.copy()
                       for i, aid in enumerate(env.possible_agents)}

        # Action: move North (action[0]=1) for all agents
        actions = {aid: np.array([1, 0, 0, 0]) for aid in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)

        moved_pos = {aid: env.state.agents[i].pos for i, aid in enumerate(env.possible_agents)}

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
            ct.pos = np.array([*env.nav_graph.centroids[env.ct_spawn_areas[0]], 0.0])

        # Clear all enemy memory for t0
        env.state.agents[0].enemy_memory = {}

        obs_t0 = env._compute_obs(0)

        # Enemy slots should be zeroed (no memory, no LOS)
        enemy_obs = obs_t0[28:63]   # 5 enemies × 7 = 35 floats
        visible_flags = [enemy_obs[i*7 + 5] for i in range(5)]
        assert all(f == 0.0 for f in visible_flags), \
            f"Expected no visibility for distant enemies, got {visible_flags}"

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

        assert env.state.agents[5].hp < initial_ct0_hp or not env.state.agents[5].alive, \
            "Shooting in same area should deal damage"

        print("Shoot test PASSED")

    if "--test-sb3-wrap" in sys.argv:
        from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

        env = Dust2Env()
        vec_env = pettingzoo_env_to_vec_env_v1(env)
        vec_env = concat_vec_envs_v1(vec_env, 1, num_cpus=1, base_class="stable_baselines3")

        obs = vec_env.reset()
        # obs might be (obs_arr, infos) tuple in newer gymnasium versions
        if isinstance(obs, tuple):
            obs = obs[0]
        assert obs.shape[1] == 71, f"Expected obs dim 71, got {obs.shape}"
        print(f"SB3 wrap test PASSED — obs shape: {obs.shape}")
