"""CS2 RL Sim — rerun.io visualization.

Imported ONLY when --record is passed to `cs2rl.train.__main__`, by
`cs2rl.train.record.record_episode`.
Zero rerun dependency during training.
"""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


def init_recording(save_path: str = None):
    rr.init("cs2rl", spawn=(save_path is None))
    if save_path:
        rr.save(save_path)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="/"),
            rrb.Vertical(
                rrb.TimeSeriesView(name="Rewards", origin="/rewards"),
                rrb.TimeSeriesView(name="HP", origin="/hp"),
            ),
            column_shares=[0.7, 0.3],
        ))
    rr.send_blueprint(blueprint)


def log_trimap():
    """Log real CS2 map geometry from .tri file (floor surfaces only).

    Filters to upward-facing triangles so ceiling/walls don't obscure agents.
    Called once at startup.
    """
    from awpy.data import TRIS_DIR
    from awpy.visibility import VisibilityChecker

    tri_path = TRIS_DIR / "de_dust2.tri"
    if not tri_path.exists():
        print("[viz] .tri file not found — skipping 3D map geometry. Run: awpy get tris")
        return

    print("[viz] Loading .tri geometry for 3D map render...")
    tris = VisibilityChecker.read_tri_file(tri_path)

    verts = []
    indices = []
    base = 0
    for tri in tris:
        p1 = np.array([tri.p1.x, tri.p1.y, tri.p1.z], dtype=np.float32)
        p2 = np.array([tri.p2.x, tri.p2.y, tri.p2.z], dtype=np.float32)
        p3 = np.array([tri.p3.x, tri.p3.y, tri.p3.z], dtype=np.float32)
        # Keep only floor-facing triangles (normal z-component > 0.3)
        normal = np.cross(p2 - p1, p3 - p1)
        nz = normal[2]
        norm_len = np.linalg.norm(normal)
        if norm_len > 0 and nz / norm_len > 0.99:
            verts.extend([p1, p2, p3])
            indices.append([base, base + 1, base + 2])
            base += 3

    if not verts:
        print("[viz] No floor triangles found in .tri file.")
        return

    vertex_arr = np.array(verts, dtype=np.float32)
    index_arr = np.array(indices, dtype=np.uint32)
    colors = np.full((len(verts), 3), [90, 80, 65], dtype=np.uint8)

    rr.log(
        "map/geometry",
        rr.Mesh3D(
            vertex_positions=vertex_arr,
            triangle_indices=index_arr,
            vertex_colors=colors,
        ),
    )
    print(f"[viz] Logged {len(indices):,} floor triangles to map/geometry")


def log_navmesh(nav_graph):
    """Log nav mesh polygons + bombsite markers. Called once at startup.

    In rerun, toggle map/navmesh vs map/geometry (tri) via the entity tree eye icons.
    """
    vertices = []
    triangles = []
    vtx_idx = 0

    for area in nav_graph.nav.areas.values():
        corners = area.corners
        if not corners:
            continue
        cx = np.mean([c.x for c in corners])
        cy = np.mean([c.y for c in corners])
        cz = np.mean([getattr(c, "z", 0.0) for c in corners])

        c_idx = vtx_idx
        vertices.append([cx, cy, cz])
        vtx_idx += 1

        for corner in corners:
            vertices.append([corner.x, corner.y, getattr(corner, "z", 0.0)])

        n = len(corners)
        for i in range(n):
            triangles.append([c_idx, c_idx + 1 + i, c_idx + 1 + (i + 1) % n])
        vtx_idx += n

    rr.log(
        "map/navmesh",
        rr.Mesh3D(
            vertex_positions=np.array(vertices, dtype=np.float32),
            triangle_indices=np.array(triangles, dtype=np.uint32),
            vertex_colors=np.full((len(vertices), 3), [60, 100, 80], dtype=np.uint8),
        ),
    )

    rr.log("map/sites/a", rr.Points3D([[1200, 2400, 100]], colors=[[255, 120, 0]], radii=[60]))
    rr.log("map/sites/b", rr.Points3D([[-1530, 2600, 5]], colors=[[255, 120, 0]], radii=[60]))


def log_simple_map(map_data):
    """Log simple-map room rectangles as flat floor quads + walls + site/spawn markers.

    Called once at recording startup when map_data is not the real dust2 map.
    Toggle map/rooms and map/walls in the Rerun entity tree.
    """
    from cs2rl.env.map import SIMPLE_ROOMS

    rooms = SIMPLE_ROOMS               # list of (idx, x0, y0, x1, y1)

    t_spawns = set(map_data.t_spawn_areas)
    ct_spawns = set(map_data.ct_spawn_areas)
    bombsites = set(int(i) for i, v in enumerate(map_data.bombsite_by_idx) if v)

    vertices = []
    triangles = []
    colors = []
    base = 0

    for room in rooms:
        # Verticality batch added z + is_ramp to SIMPLE_ROOMS tuples (now 7-element).
        # Old tuple was (idx, x0, y0, x1, y1); new is (idx, x0, y0, x1, y1, z, is_ramp).
        # Unpack flexibly so old/new shapes both work.
        if len(room) == 5:
            idx, x0, y0, x1, y1 = room
            z = 0.0
            is_ramp = False
        else:
            idx, x0, y0, x1, y1, z, is_ramp = room
        quad = [
            [x0, y0, z],
            [x1, y0, z],
            [x1, y1, z],
            [x0, y1, z],
        ]
        vertices.extend(quad)
        triangles.append([base, base + 1, base + 2])
        triangles.append([base, base + 2, base + 3])
        # Color priority: spawn > bombsite > ramp > floor. Ramp is distinct
        # cyan so the user can spot which areas are sloped (verticality batch).
        if idx in t_spawns:
            c = [180, 80, 80]          # red tint — T-side
        elif idx in ct_spawns:
            c = [80, 80, 180]          # blue tint — CT-side
        elif idx in bombsites:
            c = [200, 140, 40]         # orange — bombsite
        elif is_ramp:
            c = [80, 200, 200]         # cyan — ramp/stairs (verticality)
        else:
            c = [100, 110, 100]        # grey — corridor/mid
        colors.extend([c, c, c, c])
        base += 4

    rr.log(
        "map/rooms",
        rr.Mesh3D(
            vertex_positions=np.array(vertices, dtype=np.float32),
            triangle_indices=np.array(triangles, dtype=np.uint32),
            vertex_colors=np.array(colors, dtype=np.uint8),
        ),
    )

    # ── Walls: vertical quads for each room's 4 perimeter edges ────────────
    # Rooms connect via overlapping regions (not shared exact edges), so we
    # can't detect doorways from edge counts. Instead draw the full perimeter
    # of every room as vertical wall quads (z=0 → WALL_H). Overlapping wall
    # faces at connections are fine — they still look like a 3D maze from any
    # non-top-down angle.
    WALL_H = 150.0
    WALL_COLOR = [220, 210, 180]       # warm off-white

    w_verts = []
    w_tris = []
    w_base = 0
    for room in rooms:
        # Verticality batch added z + is_ramp; tuple is now 7-element. Extract
        # only the bbox here (walls don't currently care about z extrusion).
        if len(room) == 5:
            _, x0, y0, x1, y1 = room
        else:
            _, x0, y0, x1, y1, *_ = room
        # 4 edges of the rectangle: bottom, right, top, left
        edges = [
            ((x0, y0), (x1, y0)),
            ((x1, y0), (x1, y1)),
            ((x1, y1), (x0, y1)),
            ((x0, y1), (x0, y0)),
        ]
        for (ax, ay), (bx, by) in edges:
            # Vertical quad: 4 corners
            w_verts += [
                [ax, ay, 0.0],
                [bx, by, 0.0],
                [bx, by, WALL_H],
                [ax, ay, WALL_H],
            ]
            w_tris += [
                [w_base, w_base + 1, w_base + 2],
                [w_base, w_base + 2, w_base + 3],
            ]
            w_base += 4

    rr.log(
        "map/walls",
        rr.Mesh3D(
            vertex_positions=np.array(w_verts, dtype=np.float32),
            triangle_indices=np.array(w_tris, dtype=np.uint32),
            vertex_colors=np.full((len(w_verts), 3), WALL_COLOR, dtype=np.uint8),
        ),
    )

    # Bombsite markers
    for i, v in enumerate(map_data.bombsite_by_idx):
        if v:
            cx, cy = float(map_data.centroids[i, 0]), float(map_data.centroids[i, 1])
            rr.log(
                "map/sites/bombsite",
                rr.Points3D([[cx, cy, 10]], colors=[[255, 120, 0]], radii=[40]),
            )

    print(f"[viz] Logged {len(rooms)} simple-map rooms + {len(rooms) * 4} wall quads")


def log_tick(game_state, tick: int, rewards: dict):
    # rerun 0.30.2 API: rr.set_time(timeline, sequence=value)
    rr.set_time("tick", sequence=tick)

    for agent in game_state.agents:
        team_str = "T" if agent.team == 0 else "CT"
        color = [255, 80, 80] if agent.team == 0 else [80, 120, 255]
        entity = f"agents/{team_str}/{agent.agent_id}"

        if agent.alive:
            rr.log(
                entity,
                rr.Points3D(
                    positions=[agent.pos],
                    colors=[color],
                    radii=[16.0],
                    labels=[f"{'[B]' if agent.has_bomb else ''}{agent.hp}hp"],
                ),
            )
            # Batch 3.5: aim direction is 3D (yaw + pitch). Direction vector
            # d = (cos(p)cos(y), cos(p)sin(y), sin(p)) matches the 3D combat
            # hit-test in cs2_combat.h::process_combat. pitch=0 → horizontal
            # (z-component = 0); pitch=±π/2 → straight up/down. Length 120u
            # makes the arrow visible at agent scale (HIT_HALF_WIDTH=16).
            pitch = getattr(agent, "pitch", 0.0)                                        # backward-compat: pre-Batch-3.5 viz
            cos_p = np.cos(pitch)
            sin_p = np.sin(pitch)
            dx = cos_p * np.cos(agent.facing) * 120
            dy = cos_p * np.sin(agent.facing) * 120
            dz = sin_p * 120
            aim_color = [min(c + 80, 255) for c in color]                               # brighter than body
            rr.log(
                f"{entity}/aim",
                rr.LineStrips3D(
                    [[agent.pos.tolist(), (agent.pos + [dx, dy, dz]).tolist()]],
                    colors=[aim_color],
                    radii=[4.0],
                    labels=[f"aim p={np.degrees(pitch):+.1f}°"],
                ),
            )
        else:
            rr.log(
                entity,
                rr.Points3D(positions=[agent.pos], colors=[[80, 80, 80]], radii=[8.0]),
            )

        team = "T" if agent.team == 0 else "CT"
        rr.log(
            f"hp/{team}/{agent.agent_id}",
            rr.Scalars(float(agent.hp if agent.alive else 0)),
        )

    if game_state.bomb_planted:
        rr.log(
            "bomb",
            rr.Points3D(positions=[game_state.bomb_pos], colors=[[255, 200, 0]], radii=[24.0]),
        )

    for agent_id_str, rew in rewards.items():
        team = "T" if agent_id_str.startswith("t") else "CT"
        idx = agent_id_str[1:]
        rr.log(f"rewards/{team}/{idx}", rr.Scalars(float(rew)))
