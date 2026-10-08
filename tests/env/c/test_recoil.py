"""Sim recoil v1 (#120) — punch on the hit ray when recoil_enabled."""
import inspect
import math

import numpy as np
import pytest

from cs2rl.env.c.cs2_env import make_env
from cs2rl.env.config import EnvConfig

ALPHA = math.exp(-1.0 / 16.0 / 0.08)


def _zero_actions():
    from cs2rl.spec import action as spec
    return (
        np.zeros((10, spec.ACTION_DIM), dtype=np.int32),
        np.zeros((10, spec.AIM_DIM), dtype=np.float32),
    )


def _prep_rifle(env, i=0):
    a = env._c_env.game.agents[i]
    a.alive, a.hp = 1, 100
    a.ammo_clip[0] = 30
    a.weapon_slot = 0
    a.fire_cd = 0
    a.reload_ticks = 0
    a.switch_ticks = 0
    return a


def test_flag_off_three_shots_punch_stays_zero(make_map):
    env = make_env(seed=0, auto_reset=False, map_data=make_map, config=EnvConfig(recoil=False))
    env.reset()
    _prep_rifle(env)
    acts, cont = _zero_actions()
    acts[0, 1] = 1
    for _ in range(3):
        env.step(acts, cont)
    a = env._c_env.game.agents[0]
    assert a.punch_pitch == 0.0 and a.punch_yaw == 0.0


def test_flag_on_rifle_three_steps_punch_table(make_map):
    env = make_env(seed=0, auto_reset=False, map_data=make_map, config=EnvConfig(recoil=True))
    env.reset()
    _prep_rifle(env)
    acts, cont = _zero_actions()
    acts[0, 1] = 1
    env.step(acts, cont)
    assert env._c_env.game.agents[0].punch_pitch == pytest.approx(0.045, abs=1e-6)
    acts[0, 1] = 0
    env.step(acts, cont)
    assert env._c_env.game.agents[0].punch_pitch == pytest.approx(0.045 * ALPHA, rel=1e-5)
    acts[0, 1] = 1
    env.step(acts, cont)
    assert env._c_env.game.agents[0].punch_pitch == pytest.approx(0.045 * ALPHA**2 + 0.045,
                                                                  rel=1e-5)


def test_increment_is_after_the_ray(make_map):
    env = make_env(seed=0, auto_reset=False, map_data=make_map, config=EnvConfig(recoil=True))
    env.reset()
    g = env._c_env.game
    g.agents[0].x, g.agents[0].y, g.agents[0].z = 8.0, 1056.0, 0.0
    g.agents[0].area_idx = 4
    g.agents[0].facing = 0.0
    g.agents[0].alive, g.agents[0].hp = 1, 100
    g.agents[0].ammo_clip[0] = 30
    g.agents[0].weapon_slot = 0
    g.agents[0].fire_cd = g.agents[0].reload_ticks = g.agents[0].switch_ticks = 0
    g.agents[0].is_crouching = g.agents[0].is_airborne = 0
    # Pre-loaded punch straddles the v1c 36u VERTICAL semi-axis (gh #150), which
    # is what makes this test discriminating again: at punch 0 the shot would
    # hit under BOTH orderings, so it must start close enough to the edge that
    # one extra 0.045 kick pushes it out. Decay runs before combat, so 0.131
    # becomes 0.131*α ≈ 0.060 (α = exp(-1/16/0.08) ≈ 0.4578) at ray time; |r| =
    # 496, so p_v = 248*sin(2p).
    #   correct order (ray at 0.060): p_v ≈ 29.7, ell ≈ 0.69 ≤ 1 → HIT
    #   buggy order  (ray at 0.105): p_v ≈ 51.7, ell ≈ 2.18 > 1  → MISS
    g.agents[0].punch_pitch = 0.131
    g.agents[0].punch_yaw = 0.0
    g.agents[5].x, g.agents[5].y, g.agents[5].z = 504.0, 1056.0, 0.0
    g.agents[5].area_idx = 4
    g.agents[5].alive, g.agents[5].hp = 1, 100
    g.agents[5].is_crouching = g.agents[5].is_airborne = 0
    acts, cont = _zero_actions()
    acts[0, 1] = 1
    env.step(acts, cont)
    assert g.agents[5].hp < 100, ("shot-1 must hit (ray at the pre-existing punch 0.060). If the "
                                  "0.045 increment ran before d, the ray is at 0.105, p_v≈52 > "
                                  "the 36u semi-axis and HP stays 100")


def test_ray_uses_existing_punch(make_map):
    env = make_env(seed=0, auto_reset=False, map_data=make_map, config=EnvConfig(recoil=True))
    env.reset()
    g = env._c_env.game
    g.agents[0].x, g.agents[0].y, g.agents[0].z = 8.0, 1056.0, 0.0
    g.agents[0].area_idx = 4
    g.agents[0].facing = 0.0
    g.agents[0].alive, g.agents[0].hp = 1, 100
    g.agents[0].ammo_clip[0] = 30
    g.agents[0].weapon_slot = 0
    g.agents[0].fire_cd = g.agents[0].reload_ticks = g.agents[0].switch_ticks = 0
    # Decay runs before combat: 0.2 * α ≈ 0.092, 496*0.092 ≈ 46 > 36 (the
    # v1c VERTICAL semi-axis, gh #150 — a pitch punch is a vertical offset, so
    # the old 0.12 → 27u case now HITS: 27 < 36). 0.05 * α ≈ 0.023 still hits
    # (perp≈11).
    g.agents[0].punch_pitch = 0.2
    g.agents[0].punch_yaw = 0.0
    g.agents[5].x, g.agents[5].y, g.agents[5].z = 504.0, 1056.0, 0.0
    g.agents[5].area_idx = 4
    g.agents[5].alive, g.agents[5].hp = 1, 100
    acts, cont = _zero_actions()
    acts[0, 1] = 1
    env.step(acts, cont)
    assert g.agents[5].hp == 100, ("ray must include punch (0.2 after decay still misses). "
                                   "HP drop means d ignored punch")


def test_dry_fire_does_not_kick(make_map):
    env = make_env(seed=0, auto_reset=False, map_data=make_map, config=EnvConfig(recoil=True))
    env.reset()
    a = _prep_rifle(env)
    a.ammo_clip[0] = 0
    acts, cont = _zero_actions()
    acts[0, 1] = 1
    env.step(acts, cont)
    assert a.punch_pitch == 0.0


def test_miss_still_increments(make_map):
    env = make_env(seed=0, auto_reset=False, map_data=make_map, config=EnvConfig(recoil=True))
    env.reset()
    _prep_rifle(env)
    for j in range(5, 10):
        env._c_env.game.agents[j].alive = 0
    acts, cont = _zero_actions()
    acts[0, 1] = 1
    env.step(acts, cont)
    assert env._c_env.game.agents[0].punch_pitch == pytest.approx(0.045, abs=1e-6)


def test_recoil_is_not_reachable_from_the_cli():
    """No --recoil flag exists, and the parse layer must not invent one.

    Re-pointed from a source scan of `train.make_puffer_env` (spec Phase B R3):
    once that function accepts legacy names through **legacy, "recoil" not in
    its source stays true while `make_puffer_env(recoil=True)` — a TypeError
    today — quietly builds a recoil env. The real property is that the args →
    EnvConfig path never reads it, so a namespace that happens to carry
    recoil=True builds a recoil-free env. The source pin stays as a cheap
    second check on the same function.
    """
    import ast
    import textwrap
    from argparse import Namespace

    from cs2rl.train import config as train_config
    assert train_config.env_config_from_args(Namespace(recoil=True)).recoil is False
    # AST, not a raw getsource scan: getsource INCLUDES the docstring, and that
    # docstring is where the rule is explained — a plain substring pin would
    # forbid the function from documenting itself. Strip the docstring and pin
    # the CODE, which is what the rule is about.
    fn = ast.parse(textwrap.dedent(inspect.getsource(train_config.env_config_from_args))).body[0]
    assert isinstance(fn, ast.FunctionDef), "getsource of a function parses to its def"
    if ast.get_docstring(fn) is not None:
        fn.body = fn.body[1:]
    assert "recoil" not in ast.unparse(fn)
