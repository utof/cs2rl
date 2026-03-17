"""CS2 RL Sim — rerun.io visualization.

Imported ONLY when --record is passed to train.py.
Zero rerun dependency during training.
"""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from sim import NavGraph, GameState, BOMB_TIMER, ROUND_TIME


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


def log_navmesh(nav_graph: NavGraph):
    """Log Dust2 nav mesh as 3D geometry. Called once at startup."""
    vertices = []
    triangles = []
    vtx_idx = 0

    for area in nav_graph.nav.areas.values():
        corners = area.corners
        if not corners:
            continue

        xs = [c.x for c in corners]
        ys = [c.y for c in corners]
        zs = [getattr(c, 'z', 0) for c in corners]
        cx, cy, cz = np.mean(xs), np.mean(ys), np.mean(zs)

        c_idx = vtx_idx
        vertices.append([cx, cy, cz])
        vtx_idx += 1

        for corner in corners:
            vertices.append([corner.x, corner.y, getattr(corner, 'z', 0)])

        n = len(corners)
        for i in range(n):
            triangles.append([c_idx, c_idx + 1 + i, c_idx + 1 + (i + 1) % n])

        vtx_idx += n

    rr.log("map/dust2", rr.Mesh3D(
        vertex_positions=np.array(vertices, dtype=np.float32),
        triangle_indices=np.array(triangles, dtype=np.uint32),
        vertex_colors=np.full((len(vertices), 3), [80, 70, 58], dtype=np.uint8),
    ))

    rr.log("map/sites/a", rr.Points3D(
        [[720, 2600, 0]], colors=[[255, 120, 0]], radii=[60]))
    rr.log("map/sites/b", rr.Points3D(
        [[-1278, 820, 0]], colors=[[255, 120, 0]], radii=[60]))


def log_tick(game_state: GameState, tick: int, rewards: dict):
    # rerun 0.30.2 API: rr.set_time(timeline, sequence=value)
    rr.set_time("tick", sequence=tick)

    for agent in game_state.agents:
        team_str = "T" if agent.team == 0 else "CT"
        color = [255, 80, 80] if agent.team == 0 else [80, 120, 255]
        entity = f"agents/{team_str}/{agent.agent_id}"

        if agent.alive:
            rr.log(entity, rr.Points3D(
                positions=[agent.pos],
                colors=[color],
                radii=[16.0],
                labels=[f"{'[B]' if agent.has_bomb else ''}{agent.hp}hp"],
            ))
            dx = np.cos(agent.facing) * 80
            dy = np.sin(agent.facing) * 80
            rr.log(f"{entity}/facing", rr.LineStrips3D(
                [[agent.pos.tolist(), (agent.pos + [dx, dy, 0]).tolist()]],
                colors=[color],
            ))
        else:
            rr.log(entity, rr.Points3D(
                positions=[agent.pos], colors=[[80, 80, 80]], radii=[8.0]))

        team = "T" if agent.team == 0 else "CT"
        rr.log(f"hp/{team}/{agent.agent_id}",
               rr.Scalars(float(agent.hp if agent.alive else 0)))

    if game_state.bomb_planted:
        rr.log("bomb", rr.Points3D(
            positions=[game_state.bomb_pos],
            colors=[[255, 200, 0]], radii=[24.0]))

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
