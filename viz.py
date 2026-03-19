"""CS2 RL Sim — rerun.io visualization.

Imported ONLY when --record is passed to train.py.
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
        )
    )
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
            dx = np.cos(agent.facing) * 80
            dy = np.sin(agent.facing) * 80
            rr.log(
                f"{entity}/facing",
                rr.LineStrips3D(
                    [[agent.pos.tolist(), (agent.pos + [dx, dy, 0]).tolist()]],
                    colors=[color],
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


def log_shot(shooter_pos, hit_pos, tick: int, hit: bool):
    rr.set_time("tick", sequence=tick)
    color = [255, 255, 0] if hit else [255, 255, 200]
    shot_id = f"shots/s{tick}"
    rr.log(shot_id, rr.LineStrips3D([[shooter_pos, hit_pos]], colors=[color]))
    rr.set_time("tick", sequence=tick + 1)
    rr.log(shot_id, rr.Clear(recursive=False))
