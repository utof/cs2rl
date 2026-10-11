"""Rerun replays of the sim: rounds saved to one .rrd file, watched with ``rerun FILE``.

WHAT
----
``ReplayRecorder`` turns per-tick C-state snapshots (the dict
``cs2rl.eval.baselines.StateReader.snapshot`` returns) into Rerun recordings, ONE PER ROUND,
appended to one ``.rrd`` file. It is the one way the sim is recorded, with two callers:
  * ``python -m cs2rl.experiment.replay``: rounds of a rung's arena harness, as
    ``oracle_statue.run_check``'s ``on_step`` callback;
  * ``python -m cs2rl.train --record``: one episode on the run's ``--map``
    (``cs2rl.train.record``).

Per round the recording holds, on the ``tick`` timeline (tick 0 = the reset state):
  * ``map/...``: the map, static (dust2's floor triangles and nav mesh, or a room map's
    rooms from ``MapData.area_bounds``);
  * per agent row alive at the round's start, ``agents/<T|CT><row>``: a transform at its feet
    and a body capsule labelled with its hp; ``.../eye``, the eye's pose (sim eye height,
    facing, pitch) holding ``aim`` (a 120 u aim stick), ``shot`` (a tracer on the ticks it
    fired, red when an enemy lost hp that tick) and ``pov``, a pinhole camera;
  * ``hp/<name>`` scalars and an ``events`` text log (fires, hp losses, deaths);
    ``rewards/<name>`` scalars when the caller passes rewards (``--record`` does).
The round is named in the viewer by its deaths, e.g. "round 3: CT5 died t14".

VIEWING (rerun 0.30.2, the viewer the pinned rerun-sdk ships as the ``rerun`` command;
controls read from its source at tag 0.30.2: crates/viewer/re_ui/src/command.rs,
re_view_spatial/src/ui_3d.rs and eye.rs, re_view/src/controls.rs; on a Mac Ctrl is Cmd, and
Ctrl, not Alt, slows the camera)
  * tabs: a free camera (first-person eye kind: WASD / QE fly, left-drag turns, Shift
    10x faster, Alt 10x slower), "follow <hero>" (an orbit camera tracking the hero), and
    "<name> eyes" per side, whose camera takes the pose of that agent's ``pov`` pinhole.
    Alt + double-click any agent tracks it, Escape lets go;
  * Space plays / pauses, Ctrl + Left / Right step one tick, Left / Right move the time
    marker, Alt + Left restarts the round, Home / End jump, typing digits sets the speed
    (2 = 2x, 05 = 0.5x), dragging the time line scrubs;
  * the rounds are separate recordings under Sources: click one, or Ctrl + Alt + Down / Up
    for the next / previous round.

PITFALLS
--------
* A pinhole's children live in its 2D image space, so ``aim`` and ``shot`` are siblings of
  ``pov`` under ``eye``, never children of ``pov``.
* The eye frame is FLU (x forward, y left, z up; ``_eye_quat``), the pinhole is declared FLU
  too, so x is the aim direction in both. The combat ray's direction is
  (cos p cos y, cos p sin y, sin p) (cs2_combat.h); tests/viz/test_replay.py pins the
  quaternion to it.
* Snapshots are read after ``env.step``: the positions and facing ``process_combat`` used.
  A tracer is drawn from the post-step pose for that reason.
* The tracer colour reads "an enemy of the shooter lost hp this tick". With several
  shooters on a side (dust2) it cannot say whose bullet it was (KNOWN LIMIT).
* The tracer runs along facing / pitch, the aimed ray. The combat ray is also turned by the
  movement-inaccuracy cone (``EnvConfig.accuracy``, on by default) and, with
  ``EnvConfig.recoil`` on (off by default, no CLI flag), by a recoil punch. The snapshot
  carries neither offset, so a missed shot's tracer can run through the target (KNOWN LIMIT).
* Rerun stores positions and the eye quaternion as float32. The C positions are float32
  too, so they are exact; the aim direction is within float32 rounding.
* The file is opened at the first round's end, then written round by round: a crash
  mid-round loses only that round, and a run that fails before any round ends leaves an
  existing file at the path untouched.
* A round's name and blueprint are written when it ends, so while a long round is still
  loading the viewer shows it under a timestamp name (KNOWN LIMIT).
* Each round's blueprint send leaves 2 rerun threads alive until exit (KNOWN LIMIT): 5 rounds
  do not matter, a run of thousands would.
"""
from __future__ import annotations

import itertools
import math
import shlex
import sys
from pathlib import Path
from typing import IO

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import shapely

from cs2rl.env.nav import TICK_RATE
from cs2rl.eval.baselines import EYE_CROUCH, EYE_STAND

APP_ID = "cs2rl-replay"
TIMELINE = "tick"
AIM_LEN = 120.0                        # u, the aim stick
SHOT_LEN = 1000.0                      # u, a tracer with no live enemy to measure to
BODY_RADIUS = 16.0                     # u, a body to look at, not the hitbox (that is
BODY_HEIGHT_STAND = 72.0               # cs2_combat.h's ellipsoid, not drawn here)
BODY_HEIGHT_CROUCH = 54.0
POV_FOV_Y = math.radians(70.0)         # cs2_demo's camera.fovy
WALL_H = 150.0                         # u, drawn wall height; cs2_combat.h's 2D sight treats
                                       # every wall as full height and cites this name
TEAM_COLOR = {0: (255, 80, 80), 1: (80, 120, 255)}
DEAD_COLOR = (90, 90, 90)


def agent_name(row: int, team: int) -> str:
    """``T0`` / ``CT5``: the team and the agent row, which the harnesses index by."""
    return f"{'T' if team == 0 else 'CT'}{row}"


def view_command(path) -> str:
    """The shell command that opens ``path`` in the viewer, from any directory, with no venv.

    This interpreter's ``-m rerun`` and the absolute path: a bare ``rerun`` is not on PATH
    outside an activated venv, and a relative path names the saving checkout's file (an
    agent's worktree), not the owner's. ``--bind 127.0.0.1`` keeps the viewer's SDK port
    local; its default is 0.0.0.0.
    """
    return shlex.join(
        [sys.executable, "-m", "rerun", "--bind", "127.0.0.1",
         str(Path(path).resolve())])


def _eye_quat(yaw: float, pitch: float) -> list[float]:
    """xyzw quaternion turning the FLU eye frame's +x onto the aim direction.

    Yaw about +z, then pitch about the turned +y (negative angle: +pitch looks up), so
    +x lands on (cos p cos y, cos p sin y, sin p), the combat ray of cs2_combat.h.
    """
    sz, cz = math.sin(yaw / 2), math.cos(yaw / 2)
    sy, cy = math.sin(-pitch / 2), math.cos(-pitch / 2)
    return [-sz * sy, cz * sy, cy * sz, cz * cy]


def log_trimap(rec):
    """Log real CS2 map geometry from .tri file (floor surfaces only).

    Filters to upward-facing triangles so ceiling/walls don't obscure agents.
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
        # Keep only floor-facing triangles (normal z-component > 0.99)
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

    rec.log(
        "map/geometry",
        rr.Mesh3D(
            vertex_positions=vertex_arr,
            triangle_indices=index_arr,
            vertex_colors=colors,
        ),
        static=True,
    )
    print(f"[viz] Logged {len(indices):,} floor triangles to map/geometry")


def log_navmesh(rec, nav_graph):
    """Log nav mesh polygons + bombsite markers, static.

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

    rec.log(
        "map/navmesh",
        rr.Mesh3D(
            vertex_positions=np.array(vertices, dtype=np.float32),
            triangle_indices=np.array(triangles, dtype=np.uint32),
            vertex_colors=np.full((len(vertices), 3), [60, 100, 80], dtype=np.uint8),
        ),
        static=True,
    )

    rec.log("map/sites/a",
            rr.Points3D([[1200, 2400, 100]], colors=[[255, 120, 0]], radii=[60]),
            static=True)
    rec.log("map/sites/b",
            rr.Points3D([[-1530, 2600, 5]], colors=[[255, 120, 0]], radii=[60]),
            static=True)


def room_walls(map_data) -> list[tuple[float, float, float, float]]:
    """``(ax, ay, bx, by)`` wall segments of a room map: the room edges the sim treats as walls.

    The training sim's line of sight (cs2_combat.h line_of_sight_2d) is blocked by dead space
    (no room) and by a crossing between two rooms that are not adjacent, at any height. So
    the walls are the boundary of the union of the room boxes (shapely), plus the shared edge
    of every touching pair ``map_data.adjacency`` does not join, each drawn once. The arena's
    24 cells are all adjacent, so only its outline is wall.

    WHY not the faces cs2_solids.h bakes (``binding.bake_solids``, what the raylib demo
    draws): they need a live env, and they mark the catwalk face over the bombsite as a 64 u
    lip, where training sight is blocked at any height. Merged per line, the two layouts are
    equal on the arena and on simple_map (tests/viz/test_replay.py).

    KNOWN LIMIT: two rooms that overlap rather than touch merge in the union, so no wall is
    drawn between them even when they are not adjacent.
    """
    boxes = [shapely.box(*r) for r in np.asarray(map_data.area_bounds, dtype=float)]
    adj = np.asarray(map_data.adjacency, dtype=bool)
    lines = [shapely.union_all(boxes).boundary]
    for i, j in itertools.combinations(range(len(boxes)), 2):
        if not adj[i, j]:
            shared = boxes[i].intersection(boxes[j])
            if shared.geom_type in ("LineString", "MultiLineString"):  # not a corner point
                lines.append(shared)
    walls = []
    for line in lines:
        for part in getattr(line, "geoms", [line]):
            walls += [(ax, ay, bx, by) for (ax, ay), (bx, by) in itertools.pairwise(part.coords)]
    return walls


def log_rooms(rec, map_data):
    """A room map (make_simple_map, make_arena_duel_map) from its OWN rooms, static.

    Floor quads at each area's ``centroids_z`` coloured by role (T spawn red, CT spawn
    blue, bombsite orange, ramp cyan), and ``room_walls`` as ``WALL_H`` walls. Read from
    ``map_data.area_bounds``, so the arena draws as the arena (this used to draw the
    module's SIMPLE_ROOMS for every room map, with a wall on every room edge).
    """
    t_spawns = set(int(i) for i in map_data.t_spawn_areas)
    ct_spawns = set(int(i) for i in map_data.ct_spawn_areas)
    sites = set(int(i) for i, v in enumerate(map_data.bombsite_by_idx) if v)
    verts, tris, colors, w_verts, w_tris = [], [], [], [], []
    for i, (x0, y0, x1, y1) in enumerate(np.asarray(map_data.area_bounds, dtype=float)):
        z = float(map_data.centroids_z[i])
        k = len(verts)
        verts += [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]
        tris += [[k, k + 1, k + 2], [k, k + 2, k + 3]]
        c = ((180, 80, 80) if i in t_spawns else (80, 80, 180) if i in ct_spawns else
             (200, 140, 40) if i in sites else (80, 200, 200) if map_data.is_ramp[i] else
             (100, 110, 100))
        colors += [c] * 4
    for ax, ay, bx, by in room_walls(map_data):
        k = len(w_verts)
        w_verts += [[ax, ay, 0.0], [bx, by, 0.0], [bx, by, WALL_H], [ax, ay, WALL_H]]
        w_tris += [[k, k + 1, k + 2], [k, k + 2, k + 3]]
    rec.log("map/rooms",
            rr.Mesh3D(vertex_positions=np.array(verts, dtype=np.float32),
                      triangle_indices=np.array(tris, dtype=np.uint32),
                      vertex_colors=np.array(colors, dtype=np.uint8)),
            static=True)
    if w_verts:
        rec.log("map/walls",
                rr.Mesh3D(vertex_positions=np.array(w_verts, dtype=np.float32),
                          triangle_indices=np.array(w_tris, dtype=np.uint32),
                          vertex_colors=np.full((len(w_verts), 3), [220, 210, 180],
                                                dtype=np.uint8)),
                static=True)


def log_map(rec, map_data):
    """The map, static: a room map's rooms, or dust2's triangles and nav mesh."""
    if map_data.area_bounds is not None:
        log_rooms(rec, map_data)
    else:
        log_trimap(rec)
        log_navmesh(rec, map_data.nav_graph)


def blueprint(names: dict[int, str],
              hero: int | None,
              opp: int | None,
              map_data,
              rewards: bool = False) -> rrb.Blueprint:
    """The layout: camera tabs, the hp plot (and a rewards plot when ``rewards``) and the
    event log, a tick timeline played at ``TICK_RATE`` (1x is real time).

    ``names`` maps row -> entity name; ``hero`` / ``opp`` choose the follow and eyes tabs.
    A room map opens the free camera above its south edge looking at its centre; dust2
    lets rerun frame the scene.
    """
    free_eye = rrb.EyeControls3D(kind=rrb.Eye3DKind.FirstPerson)
    if map_data.area_bounds is not None:
        b = np.asarray(map_data.area_bounds, dtype=float)
        x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
        span = max(x1 - x0, y1 - y0)
        centre = ((x0 + x1) / 2, (y0 + y1) / 2, 0.0)
        free_eye = rrb.EyeControls3D(kind=rrb.Eye3DKind.FirstPerson,
                                     position=(centre[0], y0 - 0.3 * span, 0.5 * span),
                                     look_target=centre)
    tabs = [rrb.Spatial3DView(name="free camera", origin="/", eye_controls=free_eye)]
    if hero is not None:
        tabs.append(
            rrb.Spatial3DView(name=f"follow {names[hero]}",
                              origin="/",
                              eye_controls=rrb.EyeControls3D(
                                  kind=rrb.Eye3DKind.Orbital,
                                  tracking_entity=f"/agents/{names[hero]}")))
    for row in (hero, opp):
        if row is not None:
            tabs.append(
                rrb.Spatial3DView(name=f"{names[row]} eyes",
                                  origin="/",
                                  eye_controls=rrb.EyeControls3D(
                                      tracking_entity=f"/agents/{names[row]}/eye/pov")))
    side: list[rrb.View] = [rrb.TimeSeriesView(name="hp", origin="/hp")]
    if rewards:
        side.append(rrb.TimeSeriesView(name="rewards", origin="/rewards"))
    side.append(rrb.TextLogView(name="events", origin="/events"))
    return rrb.Blueprint(rrb.Horizontal(rrb.Tabs(*tabs), rrb.Vertical(*side), column_shares=[3, 1]),
                         rrb.TimePanel(state="expanded", timeline=TIMELINE, fps=TICK_RATE),
                         collapse_panels=False)


class ReplayRecorder:
    """Rounds of one env to one .rrd file; call it as ``(tick, st_prev, st)`` after each step.

    The call shape is ``oracle_statue.run_check``'s ``on_step``: ``tick`` counts from 1 in
    each round, ``st_prev`` is the snapshot the actors saw (the reset state on tick 1) and
    ``st`` the one after the step. Tick 1 starts a new round and records ``st_prev`` as tick
    0. ``close()`` writes the last round; the file is not even opened before a round ends.

    The rows drawn are those alive in the round's tick-0 state: the two duelists in the
    arena (``n_active_per_team=1`` parks the rest), all ten on simple_map and dust2.

    ``hero`` / ``opp`` choose the follow and eyes tabs; by default the first T and the first
    CT row alive at the first round's start. ``rewards``, optional per call, is the step's
    per-row reward array; once given it is plotted for the round.

    KNOWN LIMIT: rounds are keyed on the caller's tick 1, not on the env's round reset. That
    fits run_check and --record (auto_reset=False); a caller on auto_reset envs (a training
    auto-save) must restart its tick at 1 each episode. Such a caller also needs no thread
    growth over N rounds first (the blueprint threads, module PITFALLS).
    """

    def __init__(self, path, map_data, hero: int | None = None, opp: int | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.map_data = map_data
        # "[B]" marks the bomb carrier only where a bombsite exists: env_reset also hands the
        # arena's T the bomb, where it has no use.
        self._has_sites = bool(np.any(map_data.bombsite_by_idx))
        self.hero, self.opp = hero, opp
        self.rounds = 0
        self._fh: IO[bytes] | None = None              # opened at the first round's end
        self._closed = False
        self._rec: rr.RecordingStream | None = None    # the round being logged
        self._stream: rr.BinaryStream | None = None
        self._deaths: list[str] = []
        self._shot_drawn: set[int] = set()
        self._rewards = False

    def __call__(self, tick, st_prev, st, rewards=None):
        if tick == 1:
            self._end_round()
            self._begin_round(st_prev)
        rec = self._rec
        if rec is None:
            raise ValueError(f"tick {tick} before any tick 1: a round starts at tick 1")
        self._log_state(rec, tick, st_prev, st)
        if rewards is not None:
            self._rewards = True
            for r in self.rows:
                rec.log(f"rewards/{self.names[r]}", rr.Scalars(float(rewards[r])))

    def close(self):
        """Write the round in progress and close the file. Safe to call twice."""
        if self._closed:
            return
        self._end_round()
        if self._fh is not None:
            self._fh.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _begin_round(self, st0):
        self.rows = [int(i) for i in np.flatnonzero(st0["alive"])]
        self.names = {r: agent_name(r, int(st0["team"][r])) for r in self.rows}
        if self.hero is None:
            self.hero = next((r for r in self.rows if st0["team"][r] == 0), self.rows[0])
        if self.opp is None:
            self.opp = next((r for r in self.rows if st0["team"][r] == 1), None)
        rec = rr.RecordingStream(APP_ID, recording_id=f"{self.path.stem}-round-{self.rounds + 1}")
        self._rec, self._stream = rec, rr.binary_stream(recording=rec)
        self._deaths, self._shot_drawn, self._rewards = [], set(), False
        log_map(rec, self.map_data)
        for r in self.rows:
            rec.log(f"agents/{self.names[r]}/eye/pov",
                    rr.Pinhole(fov_y=POV_FOV_Y,
                               aspect_ratio=16 / 9,
                               camera_xyz=rr.ViewCoordinates.FLU,
                               image_plane_distance=1.0),
                    static=True)
        self._log_state(rec, 0, st0, st0)

    def _end_round(self):
        rec, stream = self._rec, self._stream
        if rec is None or stream is None:
            return
        self.rounds += 1
        died = ", ".join(self._deaths) if self._deaths else "no death"
        rr.send_recording_name(f"round {self.rounds}: {died}", recording=rec)
        rec.send_blueprint(
            blueprint(self.names, self.hero, self.opp, self.map_data, rewards=self._rewards))
        rec.flush()
        data = stream.read(flush=True)                 # None: nothing buffered
        if self._fh is None:
            self._fh = open(self.path, "wb")
        if data:
            self._fh.write(data)
        self._fh.flush()
        self._rec = self._stream = None

        # Free the dropped round stream's 2 threads, which neither gc nor disconnect() frees.
        # A private binding: rr.init calls it "to avoid memory leaks".
        del rec, stream
        rr.bindings.flush_and_cleanup_orphaned_recordings()

    def _log_state(self, rec, tick, st_prev, st):
        rec.set_time(TIMELINE, sequence=tick)
        for r in self.rows:
            name, team = self.names[r], int(st["team"][r])
            alive = bool(st["alive"][r])
            pos = [float(st["x"][r]), float(st["y"][r]), float(st["z"][r])]
            crouch = bool(st["is_crouching"][r])
            height = BODY_HEIGHT_CROUCH if crouch else BODY_HEIGHT_STAND
            # Indexed before the map check: a snapshot that lost the key fails on every map.
            carrier = int(st["bomb_carrier"]) == r and self._has_sites
            label = f"{name} {'[B] ' if carrier else ''}{int(st['hp'][r]) if alive else 'dead'}"
            rec.log(
                f"agents/{name}", rr.Transform3D(translation=pos),
                rr.Capsules3D(lengths=[height - 2 * BODY_RADIUS],
                              radii=[BODY_RADIUS],
                              translations=[[0.0, 0.0, BODY_RADIUS]],
                              colors=[TEAM_COLOR[team] if alive else DEAD_COLOR],
                              labels=[label]))
            rec.log(f"hp/{name}", rr.Scalars(float(st["hp"][r]) if alive else 0.0))
            eye = EYE_CROUCH if crouch else EYE_STAND
            rec.log(
                f"agents/{name}/eye",
                rr.Transform3D(translation=[0.0, 0.0, eye],
                               quaternion=rr.Quaternion(
                                   xyzw=_eye_quat(float(st["facing"][r]), float(st["pitch"][r])))))
            if alive:
                bright = [min(c + 80, 255) for c in TEAM_COLOR[team]]
                rec.log(
                    f"agents/{name}/eye/aim",
                    rr.LineStrips3D([[[0.0, 0.0, 0.0], [AIM_LEN, 0.0, 0.0]]],
                                    colors=[bright],
                                    radii=[2.0]))
            else:
                rec.log(f"agents/{name}/eye/aim", rr.Clear(recursive=False))
            self._log_shot(rec, tick, r, st_prev, st)
            if st_prev["alive"][r] and not alive:
                self._deaths.append(f"{name} died t{tick}")
                rec.log("events", rr.TextLog(f"{name} died", level="WARN"))
            elif st["hp"][r] < st_prev["hp"][r]:
                rec.log("events",
                        rr.TextLog(f"{name} hp {int(st_prev['hp'][r])} -> {int(st['hp'][r])}"))
        if st["bomb_planted"]:
            rec.log("bomb", rr.Points3D([list(st["bomb_pos"])],
                                        colors=[[255, 200, 0]],
                                        radii=[24.0]))

    def _log_shot(self, rec, tick, r, st_prev, st):
        """The tracer of row ``r`` on a tick it fired, cleared on the next tick."""
        name = self.names[r]
        if tick == 0 or not st["fired_this_tick"][r]:
            if r in self._shot_drawn:
                rec.log(f"agents/{name}/eye/shot", rr.Clear(recursive=False))
                self._shot_drawn.discard(r)
            return
        enemies = [e for e in self.rows if st["team"][e] != st["team"][r]]
        live = [e for e in enemies if st_prev["alive"][e]]
        length = min((math.hypot(st["x"][e] - st["x"][r], st["y"][e] - st["y"][r]) for e in live),
                     default=SHOT_LEN)
        hit = any(st["hp"][e] < st_prev["hp"][e] for e in enemies)
        rec.log(
            f"agents/{name}/eye/shot",
            rr.LineStrips3D([[[0.0, 0.0, 0.0], [float(length), 0.0, 0.0]]],
                            colors=[[255, 40, 40] if hit else [255, 230, 90]],
                            radii=[1.5]))
        rec.log("events", rr.TextLog(f"{name} fired{' (hit)' if hit else ''}"))
        self._shot_drawn.add(r)
