"""The movement-inaccuracy cone: EnvConfig.accuracy -> Dust2Env.accuracy_enabled, the radius
law (cs2_weapons.h accuracy_cone_radius) and the ray tilt (cs2_combat.h accuracy_cone_ray).

WHAT IS PINNED, and the knock-out each one catches:
  * default off reaches C, and an off shot draws nothing from either RNG stream (the
    default-off identity: every pre-cone pin and fingerprint depends on it);
  * an on shot draws exactly two values from env->accuracy_rng and none from env->rng (the
    shared stream also rolls hit locations and spawns);
  * the units: the rifle row is the averaged AK-47/M4A4 research row, and C's per-shot radius
    is 0.001 x the mrad sum (rad), standing and at full speed;
  * the 0.34 x move_speed threshold, from both sides, and the crouch term;
  * the tilt at a non-zero pitch: the exact rotation (the `up` axis carries the pitch terms),
    with the cone's next two draws set to a known (u, t);
  * the statistics: the full-speed hit rate at 360 u matches an independent numpy integral of
    the cone geometry (law 0, r = R u). Law 1 (r = R sqrt(u)) gives about 0.17 there against
    law 0's 0.39 (lane D, D-Q2), dozens of standard errors apart;
  * the plumbing below the CLI (the resolver, config.json's key, the Modal mirror, resume).
    The CLI flag itself is tests/train/test_train_cli.py::test_env_knob_cli_overrides_round_trip
    (its accuracy-off and accuracy-on rows) and test_env_knob_cli_defaults.

SETUP: the arena duel map, one agent a side (rows 0 and 5), rifle drawn. The shooter is placed
so that AFTER this tick's ground friction it moves along -x at the wanted speed and is `dist`
from the target, facing +x with pitch 0; the target's hp is reset to 1000 so it never dies.
PITFALL: _pre_speed inverts cs2_movement.h's ground friction (friction 5.2, stopspeed 80) by
hand; _shoot asserts the realised speed, so a physics change fails loudly here instead of
silently testing another speed.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from cs2rl.env.c.cs2_env import make_env
from cs2rl.env.config import EnvConfig
from cs2rl.env.map import make_arena_duel_map, make_simple_map
from tests._helpers.scenario import place_agent, ready_to_fire, set_clip, weapon_defs, zero_actions

DT = 1.0 / 16.0
TARGET_X, LANE_Y = 560.0, 200.0
HALF_W, HALF_H = 16.0, 36.0            # cs2_combat.h's standing hit ellipsoid semi-axes
RAMP_START = 0.34
                                       # research-1c Q1 (CS:GO items_game, mrad): spread, inaccuracy_stand, _crouch, _move.
AK47, M4A4 = (0.6, 6.41, 4.81, 175.06), (0.6, 4.9, 4.1, 137.88)
GLOCK, USPS = (2.0, 5.6, 4.2, 10.0), (2.5, 4.9, 3.68, 13.87)
CONE_FIELDS = ("spread_mrad", "inacc_stand_mrad", "inacc_crouch_mrad", "inacc_move_mrad")


def _xorshift32(x):
    """cs2_types.h xorshift32, one step."""
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    return x


def _env(accuracy):
    return make_env(config=EnvConfig(n_active_per_team=1,
                                     pin_pitch=1,
                                     crouch_enabled=1,
                                     jump_enabled=0,
                                     round_time=1_000_000,
                                     accuracy=accuracy),
                    map_data=make_arena_duel_map(),
                    auto_reset=False,
                    seed=0)


def _pre_speed(s):
    """The pre-step speed that this tick's ground friction turns into `s` (no move key held)."""
    if s == 0.0:
        return 0.0
    v = s / (1.0 - 5.2 * DT)
    return v if v >= 80.0 else s + 80.0 * 5.2 * DT


def _shoot(env, speed, dist=360.0, *, away=False, crouch=False):
    """Fire one rifle shot at `speed` u/s from `dist` u; returns the step's StepStats."""
    g = env._c_env.game
    a, t = g.agents[0], g.agents[5]
    place_agent(env, 5, TARGET_X, LANE_Y)
    t.hp = 1000
    place_agent(env, 0, TARGET_X - dist + speed * DT, LANE_Y)
    a.vx, a.vy = -_pre_speed(speed), 0.0
    a.facing = math.pi if away else 0.0
    set_clip(env, 0, weapon_defs()[0]["mag_size"])
    ready_to_fire(env, 0)
    acts, cont = zero_actions()
    acts[0, 1] = 1
    acts[0, 5] = 1 if crouch else 0    # HEAD_CROUCH, hold-to-crouch
    env.step(acts, cont)
    ss = env._c_env.step_stats
    assert ss.shots_fired == 1
    assert math.hypot(a.vx, a.vy) == pytest.approx(speed, abs=1e-3), "the shot speed moved"
    assert a.is_crouching == (1 if crouch else 0)
    return ss


def _rifle_R(speed, crouch=False):
    """accuracy_cone_radius for the rifle, in rad, from the binding's WEAPON_DEFS row."""
    d = weapon_defs()[0]
    s0 = RAMP_START * d["move_speed"]
    f = min(max((speed - s0) / (d["move_speed"] - s0), 0.0), 1.0)
    stance = d["inacc_crouch_mrad"] if crouch else d["inacc_stand_mrad"]
    return 0.001 * (d["spread_mrad"] + stance + f * d["inacc_move_mrad"])


def _hit_chance(R, D, n_u=2000, n_th=720):
    """P(hit) for a shot on a cone of max radius R (law 0) at a standing target D u away.

    Independent of the C code: the tilted ray d' = cos r d + sin r (cos t right + sin t up)
    around d = +x, and cs2_combat.h's gate (the ray's closest point to the target centre
    inside the 16 u x 36 u ellipsoid), integrated on a midpoint grid in (u, t).
    """
    r = R * (np.arange(n_u) + 0.5) / n_u
    th = (np.arange(n_th) + 0.5) / n_th * 2.0 * np.pi
    rr, tt = np.meshgrid(r, th, indexing="ij")
    dx, dy, dz = np.cos(rr), -np.sin(rr) * np.cos(tt), np.sin(rr) * np.sin(tt)
    fwd = D * dx
    px, py, pz = D - fwd * dx, -fwd * dy, -fwd * dz
    ell = (px * px + py * py) / HALF_W**2 + pz * pz / HALF_H**2
    return float(((fwd > 0) & (ell <= 1.0)).mean())


@pytest.fixture
def env_on():
    env = _env(True)
    env.reset()
    yield env
    env.close()


@pytest.fixture
def env_off():
    env = _env(False)
    env.reset()
    yield env
    env.close()


def test_the_switch_is_off_by_default_and_reaches_c():
    assert EnvConfig().accuracy is False
    # pyrefly: ignore[bad-argument-type]  an int flag is the point: it must normalise to bool
    assert EnvConfig(accuracy=1).accuracy is True
    for cfg, want in ((EnvConfig(), 0), (EnvConfig(accuracy=True), 1)):
        env = make_env(config=cfg, map_data=make_arena_duel_map(), auto_reset=False, seed=0)
        try:
            assert env._c_env.accuracy_enabled == want
        finally:
            env.close()


def test_an_off_shot_draws_nothing_and_still_counts_fast(env_off):
    c = env_off._c_env
    rng, acc_rng = c.rng, c.accuracy_rng
    ss = _shoot(env_off, 220.0, away=True)             # a guaranteed miss: no hit-location roll
    assert (c.rng, c.accuracy_rng) == (rng, acc_rng)
    assert ss.shots_fired_fast == 1 and ss.cone_radius_sum == 0.0
                                                       # Off, the full-speed ray is the aimed ray: 50 of 50 hit at 520 u (on, about 27%).
    assert sum(_shoot(env_off, 220.0, 520.0).shots_hit for _ in range(50)) == 50


def test_an_on_shot_draws_twice_from_its_own_stream_only(env_on):
    c = env_on._c_env
    rng, acc_rng = c.rng, c.accuracy_rng
    assert acc_rng != 0, "a zero stream stays zero and every cone draw would be r = 0"
    _shoot(env_on, 220.0, away=True)
    assert c.rng == rng, "the cone drew from env->rng (hit-location rolls and spawns)"
    assert c.accuracy_rng == _xorshift32(_xorshift32(acc_rng))


def test_the_cone_rows_are_the_averaged_research_rows():
    defs = weapon_defs()
    for slot, pair in ((0, (AK47, M4A4)), (1, (GLOCK, USPS))):
        row = tuple(defs[slot][k] for k in CONE_FIELDS)
        assert row == pytest.approx([(a + b) / 2 for a, b in zip(*pair, strict=True)],
                                    rel=1e-6), slot
    assert all(defs[2][k] == 0.0 for k in CONE_FIELDS), "the knife has no cone"
    # The ramp's speed scale: AK-47 215 and M4A4 225 u/s average to the rifle's 220.
    assert defs[0]["move_speed"] == 220.0


def test_the_radius_is_millirad_times_0_001_standing_and_running(env_on):
    stand = _shoot(env_on, 0.0).cone_radius_sum
    assert stand == pytest.approx(0.001 * (0.6 + 5.655), rel=1e-6)
    full = _shoot(env_on, 220.0).cone_radius_sum
    assert full == pytest.approx(0.001 * (0.6 + 5.655 + 156.47), rel=1e-6)
    # In degrees: the mean of the AK-47's and the M4A4's full-speed standing cones (spread +
    # stand + move; the crouch term is not in it), about 10.43 and 8.22 degrees.
    ak, m4 = (math.degrees(0.001 * (r[0] + r[1] + r[3])) for r in (AK47, M4A4))
    assert math.degrees(full) == pytest.approx((ak + m4) / 2, rel=1e-5)


@pytest.mark.parametrize("frac,fast", [(0.33, 0), (0.35, 1)])
def test_the_move_term_starts_at_034_of_max_speed(env_on, frac, fast):
    speed = frac * weapon_defs()[0]["move_speed"]
    ss = _shoot(env_on, speed)
    assert ss.shots_fired_fast == fast
    assert ss.cone_radius_sum == pytest.approx(_rifle_R(speed), rel=1e-5)
    assert (ss.cone_radius_sum > _rifle_R(0.0) * (1 + 1e-6)) == bool(fast)


def test_a_crouched_shot_uses_the_crouch_term(env_on):
    ss = _shoot(env_on, 0.0, crouch=True)
    assert ss.cone_radius_sum == pytest.approx(0.001 * (0.6 + 4.455), rel=1e-6)


def test_the_full_speed_hit_rate_matches_the_cone_integral(env_on):
    """4000 shots: SE ~ 0.008, so 4 SE ~ 0.031; law 1 or no speed term misses by > 0.2."""
    n, dist = 4000, 360.0
    R = _rifle_R(220.0)
    hits = sum(_shoot(env_on, 220.0, dist).shots_hit for _ in range(n))
    p = _hit_chance(R, dist)
    se = math.sqrt(p * (1 - p) / n)
    assert abs(hits / n - p) < 4 * se, (hits / n, p, se)


def test_the_resolver_config_json_and_modal_carry_the_switch():
    from cs2rl.train.config import env_config_from_args
    from scripts.modal_runner import request

    assert env_config_from_args(SimpleNamespace(accuracy=True)).accuracy is True
    assert env_config_from_args(SimpleNamespace()).accuracy is EnvConfig().accuracy
    assert EnvConfig(accuracy=True).to_config_dict()["accuracy"] is True
    assert request.LIVE_TRAIN_OPTION_ARITY["--accuracy"] == 1
    assert request.validate_train_args(["--timesteps", "1000", "--accuracy", "0"]) == 1000


def test_a_run_from_before_the_key_cannot_resume(tmp_path):
    """Not resume-allowlisted: a different cone is a different experiment, and a config.json
    written before the key existed reads as a mismatch (the jump_enabled precedent)."""
    import json

    from cs2rl.train.resume import RESUME_CONFIG_ALLOWLIST, check_resume_config

    assert "accuracy" not in RESUME_CONFIG_ALLOWLIST
    (tmp_path / "config.json").write_text(json.dumps({"seed": 1}))
    with pytest.raises(SystemExit, match="accuracy"):
        check_resume_config(tmp_path, {"seed": 1, "accuracy": False})


# The pitched-tilt scene, make_simple_map: the shooter ends the step at area 5's centroid
# (floor 0), the target stands on area 6 (floor 64), and the two see each other
# (test_pitch.py's elevated-target pair). Eye and torso are both 48 u above the floor
# (EYE_HEIGHT_STAND = TORSO_OFFSET_STAND, cs2_combat.h).
PITCH_SHOOTER_XY, PITCH_TARGET_XY, EYE_TORSO_Z = (575.0, 352.0), (960.0, 304.0), 48.0


def _cone_state(u_min, t_frac, tol):
    """A state for env->accuracy_rng whose next two draws (accuracy_unit, u first) give
    u >= u_min and t / (2 pi) within tol of t_frac, found by walking states 1, 2, ..."""
    for state in range(1, 10_000_000):
        x1 = _xorshift32(state)
        x2 = _xorshift32(x1)
        u, f = (x1 >> 8) / 2**24, (x2 >> 8) / 2**24
        if u >= u_min and abs(f - t_frac) < tol:
            return state, u, 2.0 * math.pi * f
    raise AssertionError("no state gives that (u, t)")


def _tilted(yaw, pitch, r, t, pitch_terms=True):
    """accuracy_cone_ray in float64. pitch_terms=False is the knock-out's up = (0, 0, 1)."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    d = np.array([cp * cy, cp * sy, sp])
    right = np.array([sy, -cy, 0.0])
    up = np.array([-sp * cy, -sp * sy, cp]) if pitch_terms else np.array([0.0, 0.0, 1.0])
    return math.cos(r) * d + math.sin(r) * (math.cos(t) * right + math.sin(t) * up)


def _gate(eye, torso, d):
    """cs2_combat.h's hit gate for the ray d (unit or not) at a standing target: <= 1 hits."""
    rv = torso - eye
    p = rv - (rv @ d) * d
    return (p[0]**2 + p[1]**2) / HALF_W**2 + p[2]**2 / HALF_H**2


def _pitched_shot(env, state, yaw, pitch, speed):
    """One full-speed rifle shot at (yaw, pitch) with env->accuracy_rng set to `state`; returns
    the StepStats and the eye and target-torso positions read after the step."""
    g = env._c_env.game
    a, tg = g.agents[0], g.agents[5]
    place_agent(env, 5, *PITCH_TARGET_XY)
    tg.hp = 1000
    place_agent(env, 0, PITCH_SHOOTER_XY[0] + speed * DT, PITCH_SHOOTER_XY[1])
    a.vx, a.vy = -_pre_speed(speed), 0.0
    a.facing = yaw
    set_clip(env, 0, weapon_defs()[0]["mag_size"])
    ready_to_fire(env, 0)
    env._c_env.accuracy_rng = state
    acts, cont = zero_actions()
    acts[0, 1] = 1
    cont[0, 1] = pitch                 # pitch is absolute per step (test_pitch.py, v1c)
    env.step(acts, cont)
    ss = env._c_env.step_stats
    assert ss.shots_fired == 1
    assert math.hypot(a.vx, a.vy) == pytest.approx(speed, abs=1e-3), "the shot speed moved"
    eye = np.array([a.x, a.y, a.z + EYE_TORSO_Z])
    return ss, eye, np.array([tg.x, tg.y, tg.z + EYE_TORSO_Z])


def test_a_pitched_shot_tilts_by_the_exact_rotation():
    """At pitch ~0.33 the cone's ray is the exact rotation, so its `up` axis carries the pitch.

    The cone's next two draws are set to u ~ 1 and t ~ 3 pi / 2 (a tilt straight down by
    r ~ R). Shot A aims so that the PREDICTED tilted ray passes through the elevated target's
    torso: it hits. Shot B aims straight at the torso with the same draws: the tilt takes it
    r below, a miss, so the cone is live here. A tilt built with up = (0, 0, 1) is the same
    ray at pitch 0, which every other test uses; here its ray misses shot A's target (its gate
    value is asserted below from cs2_combat.h's own gate).
    """
    speed = weapon_defs()[0]["move_speed"]
    R = _rifle_R(speed)
    state, u, t = _cone_state(0.99, 0.75, 0.005)
    r = R * u                                                          # law 0
    env = make_env(config=EnvConfig(n_active_per_team=1,
                                    pin_pitch=0,
                                    crouch_enabled=1,
                                    jump_enabled=0,
                                    round_time=1_000_000,
                                    accuracy=True),
                   map_data=make_simple_map(),
                   auto_reset=False,
                   seed=0)
    try:
        env.reset()
        eye0 = np.array([*PITCH_SHOOTER_XY, EYE_TORSO_Z])
        torso0 = np.array([*PITCH_TARGET_XY, 64.0 + EYE_TORSO_Z])
        aim = (torso0 - eye0) / np.linalg.norm(torso0 - eye0)
                                                                       # Shot A's (yaw, pitch): solve _tilted(yaw, pitch, r, t) = aim by fixed-point steps.
        yaw, pitch = math.atan2(aim[1], aim[0]), math.asin(aim[2])
        for _ in range(100):
            d = _tilted(yaw, pitch, r, t)
            yaw += math.atan2(aim[1], aim[0]) - math.atan2(d[1], d[0])
            pitch += math.asin(aim[2]) - math.asin(d[2])
        assert np.allclose(_tilted(yaw, pitch, r, t), aim, atol=1e-9)
        assert pitch > 0.3

        ss, eye, torso = _pitched_shot(env, state, yaw, pitch, speed)
        assert ss.cone_radius_sum == pytest.approx(R, rel=1e-6)
        assert np.allclose(eye, eye0, atol=1e-3) and np.allclose(torso, torso0, atol=1e-3)
        assert _gate(eye, torso, _tilted(yaw, pitch, r, t)) < 0.01
        ko = _gate(eye, torso, _tilted(yaw, pitch, r, t, pitch_terms=False))
        assert ko > 2.0, f"the no-pitch-terms ray must miss this target clearly, gate {ko}"
        assert ss.shots_hit == 1, "the predicted tilted ray missed: the tilt is not the rotation"

        yaw_b, pitch_b = math.atan2(aim[1], aim[0]), math.asin(aim[2])
        ss, eye, torso = _pitched_shot(env, state, yaw_b, pitch_b, speed)
        assert _gate(eye, torso, _tilted(yaw_b, pitch_b, r, t)) > 2.0
        assert ss.shots_hit == 0, "the aimed ray hit: the cone did not tilt the shot"
    finally:
        env.close()
