"""R0-H (spec 2026-08-29 §3, §8): ARENA_DUEL_V1 — 6×4 flat arena, four spawn
rows per side, mutually visible, bombsites=[].

What is pinned here and why:
- geometry (grid/centroid agreement, spawn columns, flatness, full visibility);
- the spawn-row randomisation is LOAD-BEARING: a bias-only opening turn
  (constant Δyaw on tick 0, then nothing) must NOT clear the §5 gate on the
  4-row arena, and the same search MUST clear it on a 1-spawn-per-side
  variant — the second half is what gives the first half teeth;
- the preset survives PR #127's solids bake (exterior faces only, spawn
  cells clear, every T→CT lane visible through the baked list);
- `--map` / config["env"] / pin_pitch through the REAL CLI (--dump-config).
"""
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

N_AGENTS, ACTION_DIM, AIM_DIM = 10, 7, 2
H_SHOOT = 1                            # cs2_types.h head order (same pin as test_pitch_pin)
AGENT_HULL_RADIUS = 12.0               # cs2_types.h:26
EYE_STAND = 48.0                       # cs2_combat.h standing eye height
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "src" / "train.py"


def _zero():
    return (np.zeros((N_AGENTS, ACTION_DIM), np.int32), np.zeros((N_AGENTS, AIM_DIM), np.float32))


def test_preset_geometry():
    from map import ARENA_DUEL_V1, make_arena_duel_map
    md = make_arena_duel_map()
    assert md.N == 24 and md.grid_cell_size == 20.0
    for idx in range(md.N):
        cx, cy = md.centroids[idx]
        r = int((cy - md.grid_y_min) / md.grid_cell_size)
        c = int((cx - md.grid_x_min) / md.grid_cell_size)
        assert md.grid[r, c] == idx
    assert (md.centroids_z == 0).all()
    assert list(md.t_spawn_areas) == [1, 7, 13, 19] and list(md.ct_spawn_areas) == [3, 9, 15, 21]
    assert all(md.centroids[i][0] == 150 for i in md.t_spawn_areas)
    assert all(md.centroids[i][0] == 350 for i in md.ct_spawn_areas)
    assert md.bombsite_dist_scale == 0.0 and not md.bombsite_mask.any()
    assert md.vis_matrix.all()                         # flat, no walls
    assert (md.adjacency.sum(1) >= 4).all()            # self + ≥3 neighbours (corner) — no isolated cell
    assert ARENA_DUEL_V1["cell_size"] == 20.0

    # R0-E.2 geometry resolver (Task 9) agrees with the spec: flat ⇒ pinned.
    from train import pin_pitch_for_map
    assert pin_pitch_for_map(md) == 1


def test_spawn_gaps_and_bearing_span():
    from map import make_arena_duel_map
    md = make_arena_duel_map()
    gaps, bear_t, bear_ct = [], [], []
    for t, ct in itertools.product(md.t_spawn_areas, md.ct_spawn_areas):
        (tx, ty), (cx, cy) = md.centroids[t], md.centroids[ct]
        gaps.append(math.hypot(cx - tx, cy - ty))
        # SIGNED opening turn (T faces +x; CT faces -x): 7 distinct values over dy ∈ {-300..300}
        bear_t.append(math.atan2(cy - ty, cx - tx))
        bear_ct.append(math.atan2(ty - cy, tx - cx) - math.pi)
    assert min(gaps) >= 200 and max(gaps) <= 361
    assert math.degrees(max(bear_t) - min(bear_t)) >= 40               # actually 112.6°
    assert len({round(b, 3) for b in bear_t}) == 7
    assert len({round(math.remainder(b, 2 * math.pi), 3) for b in bear_ct}) == 7


def test_dir_facing_3_is_plus_x_and_7_is_minus_x():
    """R12.4: spawn facing comes from sd->dir_facing[3] (T) / [7] (CT)
    (cs2_player.h:16); pin the nav.py direction table those indices read."""
    from nav import _DIR_FACING, _DIR_VECTORS
    assert _DIR_VECTORS[3].tolist() == [1.0, 0.0] and _DIR_FACING[3] == 0.0
    assert _DIR_VECTORS[7].tolist() == [-1.0, 0.0] and abs(_DIR_FACING[7]) == math.pi


def test_env_runs_and_spawns_in_columns():
    from c_env.cs2_env import make_env
    from map import make_arena_duel_map
    env = make_env(map_data=make_arena_duel_map(),
                   n_active_per_team=1,
                   pin_pitch=1,
                   crouch_enabled=0,
                   seed=3,
                   auto_reset=False)
    try:
        xs_t, xs_ct, ys_t, ys_ct = set(), set(), set(), set()
        for _ in range(20):
            env.reset()
            ag = env._c_env.game.agents
            xs_t.add(round(ag[0].x))
            xs_ct.add(round(ag[5].x))
            ys_t.add(round(ag[0].y))
            ys_ct.add(round(ag[5].y))
            assert ag[0].facing == pytest.approx(0.0) and abs(abs(ag[5].facing) - math.pi) < 1e-5
        assert xs_t == {150} and xs_ct == {350}
        # R12.3: the row draw is live (sidx = xorshift32 % 4, not always 0).
        assert len(ys_t) >= 3 and len(ys_ct) >= 3, (ys_t, ys_ct)
        assert ys_t <= {50, 150, 250, 350} and ys_ct <= {50, 150, 250, 350}
        a, c = _zero()
        for _ in range(50):
            _, rew, *_ = env.step(a, c)
            assert np.isfinite(rew).all()
        from _obs_spec import OBS_BLOCKS
        obs, *_ = env.step(a, c)
        assert obs[0][OBS_BLOCKS["enemy"][0] + 3] == 1.0               # mutually visible
    finally:
        env.close()


def _best_bias_only_score(md, seed, n_rounds=16, ticks=160):
    """Spec §3 R0-H threat model: an aim head whose bias alone decides the
    opening turn. Agent 0 (T) turns by a CONSTANT Δyaw on tick 0 only, then
    holds and shoots every tick; agent 5 (CT) is a statue. Returns the best
    over 13 constants of min(kills/ep ÷ 0.5, hit/facing ÷ 0.45) — both §5
    bullets normalised by their thresholds, so ≥ 1.0 means "passes both"."""
    from c_env.cs2_env import make_env
    env = make_env(map_data=md,
                   n_active_per_team=1,
                   pin_pitch=1,
                   crouch_enabled=0,
                   seed=seed,
                   auto_reset=False,
                   round_time=ticks)
    try:
        best = 0.0
        for dyaw in np.linspace(-0.6, 0.6, 13):
            kills = hit = facing = 0
            for _ in range(n_rounds):
                env.reset()
                for tick in range(ticks):
                    a, c = _zero()
                    a[0, H_SHOOT] = 1
                    c[0, 0] = dyaw if tick == 0 else 0.0
                    _, _, term, trunc, _ = env.step(a, c)
                    if term.any() or trunc.any():
                        break
                es = env._c_env.episode_stats
                kills += int(es.kills_t)
                hit += int(es.shots_hit)
                facing += int(es.shots_facing_enemy)
            ratio = hit / max(facing, 1)
            best = max(best, min((kills / n_rounds) / 0.5, ratio / 0.45))
        return best
    finally:
        env.close()


@pytest.mark.slow                                      # 2 × 13 × 16 × ≤160 ticks ≈ 66k env ticks
def test_constant_yaw_open_loop_fails_a_gate_bullet():
    """Best single constant tick-0 Δyaw over 13 × 16 rounds cannot clear BOTH
    shots_hit/shots_facing > 0.45 AND kills > 0.5 on the 4-row arena — the
    opening turn is one of 7 values the policy must read from the obs.
    Teeth: the same search on a 1-spawn-per-side variant (same row, dead
    ahead) DOES clear both, so a regression that made every round identical
    (e.g. sidx ≡ 0) would flip the first assert."""
    from map import ARENA_DUEL_V1, make_arena_duel_map, make_simple_map
    arena = make_arena_duel_map()
    assert _best_bias_only_score(arena, seed=5) < 1.0
    p = ARENA_DUEL_V1
    one_row = make_simple_map(rooms=p["rooms"],
                              t_spawns=[7],
                              ct_spawns=[9],
                              bombsites=[],
                              cell_size=p["cell_size"])
    assert _best_bias_only_score(one_row, seed=5) >= 1.0


def _walls(env):
    wl = env._c_env.sd.contents.wall_list
    return [wl.walls[i] for i in range(int(wl.count))]


def test_arena_survives_solids_bake():
    """Spec §8 / R12.5: bake PR #127's solid faces from the arena's room quads
    and check nothing the duel needs got walled off: only exterior faces on
    the 600×400 perimeter, every spawn centroid ≥ hull radius from any face,
    every T→CT lane clear at standing eye height. A ray that leaves the arena
    is blocked, proving the list is live (an empty bake would make every
    ray "clear")."""
    import binding

    from c_env.cs2_env import make_env
    from map import make_arena_duel_map
    md = make_arena_duel_map()
    env = make_env(map_data=md, n_active_per_team=1, pin_pitch=1, seed=1, auto_reset=False)
    try:
        assert int(env._c_env.sd.contents.wall_list.count) == 0                                      # training path does not bake
        count = binding.bake_solids(env._capsule)
        walls = _walls(env)
        assert count == len(walls) == 20                                                             # 6+6 horizontal, 4+4 vertical
        for w in walls:
            on_x = w.x0 == w.x1 and w.x0 in (0.0, 600.0)
            on_y = w.y0 == w.y1 and w.y0 in (0.0, 400.0)
            assert on_x or on_y, (w.x0, w.y0, w.x1, w.y1)                                            # no interior divider / lip
            assert w.z0 == 0.0 and w.height > 0.0
        for idx in (*md.t_spawn_areas, *md.ct_spawn_areas):
            cx, cy = (float(v) for v in md.centroids[idx])
            for w in walls:
                d = abs(cx - w.x0) if w.x0 == w.x1 else abs(cy - w.y0)
                assert d >= AGENT_HULL_RADIUS
        for t, ct in itertools.product(md.t_spawn_areas, md.ct_spawn_areas):
            (tx, ty), (cx, cy) = md.centroids[t], md.centroids[ct]
            assert binding.solid_ray_clear(env._capsule, float(tx), float(ty), EYE_STAND, float(cx),
                                           float(cy), EYE_STAND) == 1
        assert binding.solid_ray_clear(env._capsule, 150.0, 150.0, EYE_STAND, -50.0, 150.0,
                                       EYE_STAND) == 0
                                                                                                     # The sim keeps stepping (finite rewards) with the list attached; env.close frees it.
        env.reset()
        a, c = _zero()
        for _ in range(10):
            _, rew, *_ = env.step(a, c)
            assert np.isfinite(rew).all()
    finally:
        env.close()
    assert int(env._c_env.sd.contents.wall_list.count) == 0                                          # free_solids ran


def test_check_spawn_counts_raises_not_asserts():
    """R12.1: the train() spawn guards are RuntimeErrors on a fake driver env
    (python -O would strip a bare assert). Generic bounds mirror
    cs2_types.h t_spawns[15] / ct_spawns[5]; the arena pins 4/4."""
    from types import SimpleNamespace

    from train import check_spawn_counts

    def fake(nt, nct):
        sd = SimpleNamespace(n_t_spawns=nt, n_ct_spawns=nct)
        return SimpleNamespace(driver_env=SimpleNamespace(_c_env=SimpleNamespace(sd=SimpleNamespace(
            contents=sd))))

    assert check_spawn_counts(fake(4, 4), "arena-duel") == (4, 4)
    assert check_spawn_counts(fake(5, 5), "simple") == (5, 5)
    assert check_spawn_counts(fake(15, 5), "dust2") == (15, 5)
    for nt, nct, name in ((0, 4, "simple"), (16, 5, "dust2"), (5, 6, "dust2"), (3, 4, "arena-duel"),
                          (4, 5, "arena-duel")):
        with pytest.raises(RuntimeError):
            check_spawn_counts(fake(nt, nct), name)
    with pytest.raises(RuntimeError):
        check_spawn_counts(SimpleNamespace(driver_env=object()), "simple")


def test_build_map_data_names():
    from train import MAP_NAMES, build_map_data
    assert MAP_NAMES == ("simple", "dust2", "arena-duel")
    assert build_map_data("dust2") is None             # make_env(None) loads the nav map
    assert build_map_data("arena-duel").N == 24
    assert build_map_data("simple").N == 17
    with pytest.raises(ValueError):
        build_map_data("nope")


@pytest.mark.slow                                                                      # 7 train.py --dump-config subprocesses
def test_config_env_label(tmp_path):
    """Real CLI: --map sets config["env"] and pin_pitch is resolved from the
    built map ABOVE the --dump-config exit (the Modal runner fingerprints
    every launch from this dump). --dust2 stays an alias; --map wins."""
    cases = (
        (["--map", "simple"], "cs2-simple", 0),
        (["--map", "dust2"], "cs2-dust2", 1),
        (["--map", "arena-duel"], "cs2-arena-duel", 1),
        ([], "cs2-simple", 0),                                                         # default map
        (["--dust2"], "cs2-dust2", 1),                                                 # alias
        (["--dust2", "--map", "simple"], "cs2-simple", 0),                             # --map wins
    )
    for i, (flags, label, pin) in enumerate(cases):
        d = tmp_path / str(i)
        r = subprocess.run([
            sys.executable,
            str(TRAIN_SCRIPT), "--dump-config", *flags, "--checkpoint-dir",
            str(d)
        ],
                           capture_output=True,
                           text=True,
                           cwd=REPO_ROOT,
                           timeout=120)
        assert r.returncode == 0, r.stderr[-2000:]
        cfg = json.loads((d / "config.json").read_text())
        assert cfg["env"] == label and cfg["pin_pitch"] == pin, (flags, cfg["env"],
                                                                 cfg["pin_pitch"])
                                                                                       # An explicit pin that disagrees with the map is refused before the dump.
    r = subprocess.run([
        sys.executable,
        str(TRAIN_SCRIPT), "--dump-config", "--map", "arena-duel", "--pin-pitch", "0",
        "--checkpoint-dir",
        str(tmp_path / "bad")
    ],
                       capture_output=True,
                       text=True,
                       cwd=REPO_ROOT,
                       timeout=120)
    assert r.returncode != 0 and "pin_pitch=0" in r.stderr
