"""cs2rl.viz.render: the rerun replay recorder (``ReplayRecorder``) and what it draws.

WHAT IS PINNED HERE
-------------------
* The eye quaternion turns the FLU eye frame's +x onto cs2_combat.h's combat ray
  (cos p cos y, cos p sin y, sin p) and keeps +y level (no roll), so the aim stick, the
  tracer and the eyes camera all look where the sim shoots.
* ``room_walls``: the arena (24 adjacent cells) is walled by its outline only; on the
  arena and on simple_map its walls, merged per line, are the faces cs2_solids.h bakes from
  the same rooms (``binding.bake_solids``, what the raylib demo draws).
* ``ReplayRecorder``: one recording per round, ids ``<stem>-round-<k>``, named by its
  deaths; a round reaches the file when the next one starts (a crash loses one round);
  each agent row alive at the round's start gets its entity tree; rewards are drawn only
  for a round that passed them; a call before any tick 1 is refused.

The file is read back with ``rerun.recording.load_archive`` (ids and entity paths). The
per-tick VALUES need rerun's datafusion reader, absent from the project venv; they were
measured once against the harness (lane F report, F-Q2), not pinned here.
"""
import math

import numpy as np
import pytest

rr = pytest.importorskip("rerun")


def _rotate(q, v):
    """``v`` turned by the xyzw quaternion ``q``."""
    x, y, z, w = q
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=float)
    return v + 2 * w * np.cross(u, v) + 2 * np.cross(u, np.cross(u, v))


@pytest.mark.parametrize("yaw", [-3.0, -1.2, 0.0, 0.4, 1.5707963, 2.9])
@pytest.mark.parametrize("pitch", [-1.2, -0.3, 0.0, 0.25, 1.1])
def test_the_eye_quaternion_looks_along_the_combat_ray(yaw, pitch):
    from cs2rl.viz.render import _eye_quat
    q = _eye_quat(yaw, pitch)
    assert math.isclose(float(np.linalg.norm(q)), 1.0, abs_tol=1e-12)
    ray = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
    assert np.allclose(_rotate(q, [1, 0, 0]), ray, atol=1e-12)
    assert np.allclose(_rotate(q, [0, 1, 0]), [-math.sin(yaw), math.cos(yaw), 0.0], atol=1e-12)


def test_the_arena_is_walled_by_its_outline_only():
    from cs2rl.env.map import make_arena_duel_map
    from cs2rl.viz.render import room_walls
    md = make_arena_duel_map()
    b = np.asarray(md.area_bounds, dtype=float)
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    walls = room_walls(md)
    assert len(b) == 24 and walls
    for ax, ay, bx, by in walls:
        assert (ax == bx and ax in (x0, x1)) or (ay == by and ay in (y0, y1)), (ax, ay, bx, by)
    length = sum(math.hypot(bx - ax, by - ay) for ax, ay, bx, by in walls)
    assert length == 2 * ((x1 - x0) + (y1 - y0))       # the whole outline, once


def _merged(segs):
    """Collinear axis-aligned segments unioned per line: {(axis, coordinate): [(lo, hi), ...]}."""
    lines = {}
    for ax, ay, bx, by in segs:
        key, lo, hi = ((("y", ay), min(ax, bx), max(ax, bx)) if ay == by else
                       (("x", ax), min(ay, by), max(ay, by)))
        lines.setdefault(key, []).append((lo, hi))
    out = {}
    for key, ivs in lines.items():
        acc = []
        for lo, hi in sorted(ivs):
            if acc and lo <= acc[-1][1] + 1e-4:
                acc[-1][1] = max(acc[-1][1], hi)
            else:
                acc.append([lo, hi])
        out[key] = [tuple(a) for a in acc]
    return out


@pytest.mark.parametrize("which, n_faces", [("arena", 20), ("simple_map", 33)])
def test_room_walls_trace_the_faces_the_sim_bakes(which, n_faces):
    """The faces cs2_solids.h bakes from the same rooms (what the raylib demo draws) are an
    independent oracle: merged per line, the two layouts are equal."""
    from cs2rl.env.c import binding
    from cs2rl.env.c.cs2_env import make_env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_arena_duel_map, make_simple_map
    from cs2rl.viz.render import room_walls
    md = make_arena_duel_map() if which == "arena" else make_simple_map()
    env = make_env(
        map_data=md,
        config=EnvConfig(n_active_per_team=1, pin_pitch=1) if which == "arena" else EnvConfig(),
        seed=1,
        auto_reset=False)
    try:
        assert binding.bake_solids(env._capsule) == n_faces            # known count
        wl = env._c_env.sd.contents.wall_list
        baked = _merged([(w.x0, w.y0, w.x1, w.y1) for w in (wl.walls[i] for i in range(n_faces))])
    finally:
        env.close()
    assert _merged(room_walls(md)) == baked


def _snap(hp5=100, fired0=0, alive5=1, x5=300.0):
    """Ten rows, T0 and CT5 alive (the arena's duelists), the keys the recorder reads."""
    st = {
        "x": np.zeros(10, np.float32),
        "y": np.full(10, 200.0, np.float32),
        "z": np.zeros(10, np.float32),
        "facing": np.zeros(10, np.float32),
        "pitch": np.zeros(10, np.float32),
        "hp": np.zeros(10, np.int32),
        "alive": np.zeros(10, np.int32),
        "team": np.array([0] * 5 + [1] * 5),
        "is_crouching": np.zeros(10, np.int32),
        "fired_this_tick": np.zeros(10, np.int32),
        "bomb_planted": 0,
        "bomb_carrier": 0,
        "bomb_pos": (0.0, 0.0, 0.0),
    }
    st["x"][0], st["x"][5] = 100.0, x5
    st["hp"][0], st["hp"][5] = 100, hp5
    st["alive"][0], st["alive"][5] = 1, alive5
    st["fired_this_tick"][0] = fired0
    return st


def _round(rec, states, rewards=None):
    for tick in range(1, len(states)):
        rec(tick, states[tick - 1], states[tick], **({} if rewards is None else {
            "rewards": rewards
        }))


def test_each_round_is_one_recording_named_by_its_deaths(tmp_path, monkeypatch):
    from rerun.recording import load_archive

    from cs2rl.env.map import make_arena_duel_map
    from cs2rl.viz import render
    names = []
    send = render.rr.send_recording_name
    monkeypatch.setattr(render.rr, "send_recording_name", lambda name, recording:
                        (names.append(name), send(name, recording)))
    path = tmp_path / "duel.rrd"
    rec = render.ReplayRecorder(path, make_arena_duel_map())
    _round(rec, [_snap(), _snap(fired0=1, hp5=50), _snap(fired0=1, hp5=0, alive5=0)])
    assert rec.rounds == 0 and path.stat().st_size == 0                # nothing before a round ends
    _round(rec, [_snap(), _snap(x5=310.0)], rewards=np.arange(10, dtype=np.float32))
    assert rec.rounds == 1
    assert [r.recording_id() for r in load_archive(path).all_recordings()] == ["duel-round-1"]
    rec.close()
    rec.close()                                                        # safe twice
    recs = load_archive(path).all_recordings()
    assert [r.recording_id() for r in recs] == ["duel-round-1", "duel-round-2"]
    assert {r.application_id() for r in recs} == {render.APP_ID}
    assert names == ["round 1: CT5 died t2", "round 2: no death"]
    paths = [set(r.schema().entity_paths()) for r in recs]
    for agent in ("/agents/T0", "/agents/CT5"):
        for child in ("", "/eye", "/eye/aim", "/eye/pov"):
            assert all(agent + child in p for p in paths), agent + child
    assert "/agents/T0/eye/shot" in paths[0] and "/agents/T1" not in paths[0]
    assert "/map/rooms" in paths[0] and "/map/walls" in paths[0]
    assert "/rewards/T0" not in paths[0] and {"/rewards/T0", "/rewards/CT5"} <= paths[1]


def test_a_round_starts_at_tick_1(tmp_path):
    from cs2rl.env.map import make_arena_duel_map
    from cs2rl.viz.render import ReplayRecorder
    path = tmp_path / "late.rrd"
    rec = ReplayRecorder(path, make_arena_duel_map())
    with pytest.raises(ValueError, match="a round starts at tick 1"):
        rec(2, _snap(), _snap())
    rec.close()
    assert rec.rounds == 0 and path.stat().st_size == 0
