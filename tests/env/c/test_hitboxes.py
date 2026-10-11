"""The hit model (#391): capsule hitboxes (EnvConfig.hitboxes, the default) and the legacy ellipsoid.

WHAT IS PINNED (T1-T10 of the #391 brief), and the knock-out each one caught:
  * T1 the C ray test (binding.hitbox_first_region, the code process_combat runs) gives the
    numpy reference's region on 20000 seeded agent-local rays, away from a surface or a tie;
    plus the two degenerate branches (a ray parallel to a capsule's axis, the neck sphere).
    Caught: the end-cap branch dropped; the body's end-plane check dropped.
  * T2 pinned pitch, standing vs standing at 360 u, front-on: a fan of level shots across
    the head hits exactly where the reference says, and every hit counts as the head.
    Caught: the eye left at 48 under hitboxes; a head hit counted as the neck.
  * T3 pinned pitch, standing vs crouched: 0 hits and every shot stance-blocked; with the
    pitch aimed at the crouched head's midpoint the shot hits the head, at five target facings.
    Caught: the stand pose used for a crouched target; the stand pose's z band in the
    stance-blocked counter; the world-to-agent rotation reversed in C or in target_xyz (red at
    the four facings other than pi).
  * T4 damage: a rifle head through a helmet kills from 100 hp at 200 u and leaves 1 hp at
    360 u (at five target facings); a leg hit leaves armor untouched.
    Caught: legs armored; head multiplier 3.9; the ray direction's rotation reversed in C.
  * T5 the helmet rule: without a helmet the head takes unarmored damage (a kill at 360 u,
    armor untouched); with one, armor applies. Caught: the head armored without the helmet
    check; the helmet check inverted.
  * T6 a hit draws nothing from env->rng under hitboxes and one value on the legacy
    ellipsoid. Caught: the legacy roll's draw kept in hitbox mode; a draw on every shot.
  * T7 the neck rule: a neck-first ray that goes on into the head is the head, one that does
    not stays the neck. Caught: the rule dropped; the rule applied without the head test.
  * T8 sum(hits_*) == shots_hit in both models, every counter reached under hitboxes and the
    roll's four on the ellipsoid; under hitboxes each shot's counter is the reference region's
    group. Caught: a hit counted twice; the arm groups not counted; the world-to-agent
    rotation reversed in C.
  * T9 the table's invariants: 2 poses x 18 regions, r > 0, the CS hit groups, the stand
    and crouch tops of the CS:GO hull, and both CS eyes inside their pose's head.
    Caught: the crouch eye moved below the head; a region's group set to generic (0).
  * T10 legacy reproduction: with hitboxes=False and the cone off, scripts/sim_fingerprint.py
    reproduces its documented pre-#391 hashes (random and track rows here; the bomb rows, in
    the slow tier, are its SIMULTANEOUS-COMBAT rows, #152 L2). Caught: one legacy roll
    threshold moved; the legacy stomach multiplier moved.
  The knock-out runs are in the #391 phase 2 record (lane G evidence), not in this file.

SETUP: the arena duel map, one agent a side (rows 0 and 5), rifle drawn, the cone off (an
exact ray, D3), a round long enough never to end. The shooter stands at x = 100 and the
target on the same lane; its hp and armor are reset before each shot, so no shot kills
unless the test means it to.
PITFALL: the target's crouch is the crouch action held on the stepped tick (process_movement
writes is_crouching before process_combat), not a raw field write, which the step would undo.
"""
import functools
import importlib.util
import math

import numpy as np
import pytest

from cs2rl.env.c import binding
from cs2rl.env.c.cs2_env import make_env
from cs2rl.env.c.hit_model import hit_geometry
from cs2rl.env.config import EnvConfig
from cs2rl.env.map import make_arena_duel_map
from cs2rl.spec.action import ACTION_HEAD_NAMES
from tests._helpers import hitbox_ref
from tests._helpers.scenario import place_agent, ready_to_fire, set_clip, weapon_defs, zero_actions
from tests.conftest import REPO_ROOT

SHOOTER_X, LANE_Y = 100.0, 200.0
HEAD_SHOOT = ACTION_HEAD_NAMES.index("shoot")
HEAD_CROUCH = ACTION_HEAD_NAMES.index("crouch")
TABLE = binding.hitbox_table()
GROUP = dict(zip(TABLE["names"], TABLE["groups"], strict=True))
HITGROUP_LEGS = {GROUP["thigh_l"], GROUP["thigh_r"]}
HIT_COUNTERS = ("hits_head", "hits_neck", "hits_chest", "hits_stomach", "hits_arm", "hits_leg")
COUNTER_OF_GROUP = {
    GROUP["head"]: "hits_head",
    GROUP["neck"]: "hits_neck",
    GROUP["chest"]: "hits_chest",
    GROUP["stomach"]: "hits_stomach",
    GROUP["upper_arm_l"]: "hits_arm",
    GROUP["upper_arm_r"]: "hits_arm",
    GROUP["thigh_l"]: "hits_leg",
    GROUP["thigh_r"]: "hits_leg"
}
# Target facings of the aimed-shot tests. At facing pi (_place's default) sin(facing) is 0 to
# float32, so a world-to-agent rotation of the wrong sense is invisible there; T3's crouched-head
# aim shows it at the others.
TARGET_FACINGS = (math.pi, 0.5, 1.5, 2.5, 4.0)


def _env(hitboxes=True, pin_pitch=0):
    env = make_env(config=EnvConfig(n_active_per_team=1,
                                    pin_pitch=pin_pitch,
                                    jump_enabled=0,
                                    round_time=1_000_000,
                                    accuracy=False,
                                    hitboxes=hitboxes),
                   map_data=make_arena_duel_map(),
                   auto_reset=False,
                   seed=0)
    env.reset()
    return env


@pytest.fixture
def env():
    e = _env()
    yield e
    e.close()


def _place(env, d, target_facing=math.pi):
    """Shooter at (SHOOTER_X, LANE_Y), target d u along +x facing `target_facing`."""
    place_agent(env, 0, SHOOTER_X, LANE_Y)
    place_agent(env, 5, SHOOTER_X + d, LANE_Y).facing = target_facing


def _aim(env, point, crouched_shooter=False):
    """(yaw, pitch) from the shooter's eye under the env's hit model to a world point."""
    a = env._c_env.game.agents[0]
    eye = hit_geometry(env.config.hitboxes).eye_z(a.z, crouched_shooter)
    dx, dy, dz = point[0] - a.x, point[1] - a.y, point[2] - eye
    return math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))


def _shoot(env, yaw, pitch=0.0, *, crouch_target=False, hp=100, armor=100, helmet=1):
    """One rifle shot from agent 0 at (yaw, pitch); the target's hp, armor and helmet are set
    first. Returns the step's StepStats."""
    g = env._c_env.game
    a, t = g.agents[0], g.agents[5]
    t.hp, t.armor, t.has_helmet = hp, armor, helmet
    a.facing = yaw
    set_clip(env, 0, weapon_defs()[0]["mag_size"])
    ready_to_fire(env, 0)
    acts, cont = zero_actions()
    acts[0, HEAD_SHOOT] = 1
    acts[5, HEAD_CROUCH] = 1 if crouch_target else 0
    cont[0, 1] = pitch                 # absolute pitch (v1c); pinned to 0 under pin_pitch
    env.step(acts, cont)
    ss = env._c_env.step_stats
    assert ss.shots_fired == 1
    assert t.is_crouching == int(crouch_target)
    return ss


def _local_ray(env, yaw, pitch):
    """The shot ray in the target's (fwd, left, up) frame, as cs2_hitboxes.h hb_ray_vs_agent
    builds it: (origin, direction, pose)."""
    g = env._c_env.game
    a, t = g.agents[0], g.agents[5]
    eye = hit_geometry(env.config.hitboxes).eye_z(a.z, a.is_crouching)
    d = np.array(
        [math.cos(pitch) * math.cos(yaw),
         math.cos(pitch) * math.sin(yaw),
         math.sin(pitch)])
    c, s = math.cos(t.facing), math.sin(t.facing)
    rx, ry = a.x - t.x, a.y - t.y
    o = np.array([rx * c + ry * s, -rx * s + ry * c, eye - t.z])
    return o, np.array([d[0] * c + d[1] * s, -d[0] * s + d[1] * c, d[2]]), int(t.is_crouching)


# ── T1 ────────────────────────────────────────────────────────────────────────────────────────


def _seeded_rays(n, seed):
    """n agent-local rays as #391 phase 1 drew them: a target d in [200, 520] u away at a random
    facing, the eye at one of the four eye heights, aimed at a point uniform in lateral
    [-40, 40] u and height [-10, 90] u on the target's plane. Returns (pose, origin, dir)."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(200.0, 520.0, n)
    facing = rng.uniform(0.0, 2.0 * np.pi, n)
    pose = rng.integers(0, 2, n)
    eye = rng.choice(np.array([*TABLE["eye"], *TABLE["legacy_eye"]]), n)
    ro = np.stack([np.zeros(n), np.zeros(n), eye], 1)
    rd = np.stack([d, rng.uniform(-40.0, 40.0, n), rng.uniform(-10.0, 90.0, n)], 1) - ro
    rd /= np.linalg.norm(rd, axis=1, keepdims=True)
    c, s = np.cos(-facing), np.sin(-facing)
    rel = ro - np.stack([d, np.zeros(n), np.zeros(n)], 1)

    def rot(v):
        return np.stack([v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c, v[:, 2]], 1)

    return pose, rot(rel), rot(rd)


def _c_regions(pose, ro, rd):
    out = [
        binding.hitbox_first_region(int(p), *map(float, o), *map(float, d))
        for p, o, d in zip(pose, ro, rd, strict=True)
    ]
    return np.array([r for r, _ in out]), np.array([np.inf if t is None else t for _, t in out])


def _ref_regions(pose, ro, rd):
    region, t, margin = (np.empty(len(pose), int), np.empty(len(pose)), np.empty(len(pose)))
    for p in (0, 1):
        sel = pose == p
        region[sel], t[sel], margin[sel] = hitbox_ref.first_region(ro[sel], rd[sel],
                                                                   TABLE["caps"][p], TABLE["head"],
                                                                   TABLE["neck"])
    return region, t, margin


def test_t1_the_c_ray_test_matches_the_numpy_reference():
    pose, ro, rd = _seeded_rays(20000, 391)
    ref, ref_t, margin = _ref_regions(pose, ro, rd)
    got, got_t = _c_regions(pose, ro, rd)
    keep = margin >= 1e-2
    # Known counts first, so an emptied comparison cannot pass: phase 1 excluded under 1 %, and
    # about one ray in ten of this distribution hits.
    assert keep.sum() > 19500, keep.sum()
    assert (ref[keep] >= 0).sum() > 1500 and (ref[keep] < 0).sum() > 15000
    assert len(set(ref[keep][ref[keep] >= 0])) == len(TABLE["names"]), "a region never reached"
    bad = np.nonzero(keep & (got != ref))[0]
    assert len(
        bad) == 0, f"{len(bad)} mismatches, first {[(int(ref[i]), int(got[i])) for i in bad[:5]]}"
    hit = keep & (ref >= 0)
    # The entry distance in float32 under -ffast-math: measured within 0.082 u of the float64
    # reference at up to 520 u (#391 phase 2); it feeds the range check and the damage falloff.
    assert np.abs(got_t[hit] - ref_t[hit]).max() < 0.25


def test_t1_the_degenerate_branches():
    """Rays exactly parallel to each capsule's axis (the C `par` branch: only the caps can be
    entered), the neck row (A == B, a sphere), and vertical rays (hb_bound_pass's zero
    horizontal component): C equals the reference on every one."""
    caps = np.array(TABLE["caps"][0], dtype=float)
    neck = caps[TABLE["neck"]]
    assert (neck[0:3] == neck[3:6]).all(), "the neck row is a sphere (A == B)"
    ro, rd = [], []
    for c in caps:
        axis = c[3:6] - c[0:3]
        if not axis.any():
            continue
        u = axis / np.linalg.norm(axis)
        side = np.cross(u, [0.0, 0.0, 1.0] if abs(u[2]) < 0.9 else [1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)
        for k in (0.0, 0.5, 0.9):                                                          # on the axis, inside the radius, near its edge
            ro.append(c[0:3] - 40.0 * u + k * c[6] * side)
            rd.append(u)
    neck_rays = [(neck[0:3] + [-50.0, dy, dz], [1.0, 0.0, 0.0]) for dy in (-1.0, 0.0, 1.0)
                 for dz in (-1.0, 0.0, 1.0)]
    vertical = [([f, left, -30.0], [0.0, 0.0, 1.0]) for f in (-6.0, 0.0, 6.0)
                for left in (-9.0, 0.0, 9.0)]
    for o, d in neck_rays + vertical:
        ro.append(np.asarray(o, dtype=float))
        rd.append(np.asarray(d, dtype=float))
    ro, rd = np.array(ro), np.array(rd)
    ref, ref_t, margin = hitbox_ref.first_region(ro, rd, TABLE["caps"][0], TABLE["head"],
                                                 TABLE["neck"])
    got, got_t = _c_regions(np.zeros(len(ro), int), ro, rd)
    keep = margin >= 1e-2
    assert keep.sum() >= 40 and (ref[keep] >= 0).sum() >= 30, (keep.sum(), (ref[keep] >= 0).sum())
    assert (got[keep] == ref[keep]).all(), list(zip(ref[keep], got[keep], strict=True))
    hit = keep & (ref >= 0)
    assert np.abs(got_t[hit] - ref_t[hit]).max() < 0.25


# ── T2, T3 ────────────────────────────────────────────────────────────────────────────────────


def _level_fan(env, d, n=33, half_width=8.0, crouch_target=False):
    """n level shots across the target at lateral offsets in [-half_width, half_width] u; returns
    (C hits per counter, shots_hit, stance-blocked count, reference hit count)."""
    totals = dict.fromkeys(HIT_COUNTERS, 0)
    hits = blocked = ref_hits = 0
    for lat in np.linspace(-half_width, half_width, n):
        yaw = math.atan2(lat, d)
        ss = _shoot(env, yaw, 0.0, crouch_target=crouch_target)
        o, dvec, pose = _local_ray(env, yaw, 0.0)
        region, _, margin = hitbox_ref.first_region(o[None], dvec[None], TABLE["caps"][pose],
                                                    TABLE["head"], TABLE["neck"])
        assert margin[0] >= 1e-2, f"lateral {lat}: a ray on a surface; move the fan"
        ref_hits += int(region[0] >= 0)
        hits += ss.shots_hit
        blocked += ss.shots_stance_blocked
        for k in HIT_COUNTERS:
            totals[k] += getattr(ss, k)
    return totals, hits, blocked, ref_hits


def test_t2_a_pinned_level_shot_at_a_standing_head_hits_the_head():
    env = _env(pin_pitch=1)
    try:
        _place(env, 360.0)
        totals, hits, blocked, ref_hits = _level_fan(env, 360.0)
        assert hits == ref_hits and hits >= 10, (hits, ref_hits)
        assert totals["hits_head"] == hits, totals
        assert blocked == 0
    finally:
        env.close()


def test_t3_a_pinned_standing_shot_cannot_reach_a_crouched_target():
    env = _env(pin_pitch=1)
    try:
        _place(env, 360.0)
        totals, hits, blocked, ref_hits = _level_fan(env, 360.0, crouch_target=True)
        assert (hits, ref_hits) == (0, 0), totals
        assert blocked == 33, "the stance-blocked counter must see every one of these shots"
    finally:
        env.close()


@pytest.mark.parametrize("facing", TARGET_FACINGS)
def test_t3_a_shot_aimed_at_the_crouched_head_midpoint_hits_the_head(env, facing):
    _place(env, 360.0, facing)
    t = env._c_env.game.agents[5]
    point = hit_geometry(True).target_xyz(t.x, t.y, t.z, t.facing, True)
    ss = _shoot(env, *_aim(env, point), crouch_target=True)
    assert (ss.shots_hit, ss.hits_head, ss.shots_stance_blocked) == (1, 1, 1)


# ── T4, T5 ────────────────────────────────────────────────────────────────────────────────────


def _head_shot(env, d, facing=math.pi, **kw):
    _place(env, d, facing)
    t = env._c_env.game.agents[5]
    ss = _shoot(env, *_aim(env,
                           hit_geometry(True).target_xyz(t.x, t.y, t.z, t.facing, False)), **kw)
    assert (ss.shots_hit, ss.hits_head) == (1, 1)
    return t


@pytest.mark.parametrize("facing", TARGET_FACINGS)
def test_t4_a_rifle_head_through_a_helmet_kills_at_200_and_leaves_1_hp_at_360(env, facing):
    """#391 phase 1 5.4: 100.41 hp at 200 u and 99.60 at 360 u (34.5 x 4 x 0.975^(d/500)
    x 0.735 armor_pen). The entry distance is a few units under d, which raises both a little."""
    t = _head_shot(env, 360.0, facing)
    assert (t.hp, t.alive) == (1, 1) and t.armor < 100
    t = _head_shot(env, 200.0, facing)
    assert (t.hp, t.alive) == (0, 0)


def test_t4_a_leg_hit_leaves_armor_untouched(env):
    _place(env, 360.0)
    t = env._c_env.game.agents[5]
    ax, ay, az, bx, by, bz, _ = TABLE["caps"][0][TABLE["names"].index("thigh_l")]
    c, s = math.cos(t.facing), math.sin(t.facing)
    f, left, up = (ax + bx) / 2, (ay + by) / 2, (az + bz) / 2
    yaw, pitch = _aim(env, (t.x + f * c - left * s, t.y + f * s + left * c, t.z + up))
    o, dvec, pose = _local_ray(env, yaw, pitch)
    region, _, _ = hitbox_ref.first_region(o[None], dvec[None], TABLE["caps"][pose], TABLE["head"],
                                           TABLE["neck"])
    assert TABLE["groups"][region[0]] in HITGROUP_LEGS, "precondition: the aimed ray enters a leg"
    ss = _shoot(env, yaw, pitch)
    assert (ss.shots_hit, ss.hits_leg) == (1, 1)
    assert t.armor == 100 and 0 < t.hp < 100


def test_t5_the_head_is_armored_only_with_a_helmet(env):
    t = _head_shot(env, 360.0, helmet=0)
    assert (t.hp, t.alive, t.armor) == (0, 0, 100), "no helmet: unarmored head damage, a kill"
    env.reset()
    t = _head_shot(env, 360.0, helmet=1)
    assert (t.hp, t.alive) == (1, 1) and t.armor < 100, "a helmet: armor applies"


# ── T6 ────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("hitboxes", [True, False])
def test_t6_a_hit_draws_from_env_rng_only_on_the_legacy_ellipsoid(hitboxes):
    env = _env(hitboxes=hitboxes)
    try:
        _place(env, 360.0)
        t = env._c_env.game.agents[5]
        point = hit_geometry(hitboxes).target_xyz(t.x, t.y, t.z, t.facing, False)
        rng = env._c_env.rng
        ss = _shoot(env, *_aim(env, point))
        assert ss.shots_hit == 1
        assert (env._c_env.rng != rng) is (not hitboxes)
    finally:
        env.close()


# ── T7 ────────────────────────────────────────────────────────────────────────────────────────


def test_t7_a_neck_first_ray_counts_as_the_head_only_when_it_enters_the_head():
    """Side-on and front-on level rays through the neck band of the standing pose."""
    neck = TABLE["caps"][0][TABLE["neck"]]
    fwd, z = np.meshgrid(np.linspace(neck[0] - 8.0, neck[0] + 8.0, 81),
                         np.linspace(neck[2] - 8.0, neck[2] + 8.0, 81))
    fwd, z = fwd.ravel(), z.ravel()
    side_ro = np.stack([fwd, np.full_like(fwd, -300.0), z], 1)
    front_ro = np.stack([np.full_like(fwd, 300.0), fwd - neck[0] + neck[1], z], 1)
    ro = np.concatenate([side_ro, front_ro])
    rd = np.concatenate(
        [np.tile([0.0, 1.0, 0.0], (len(fwd), 1)),
         np.tile([-1.0, 0.0, 0.0], (len(fwd), 1))])
    caps = TABLE["caps"][0]
    t_all = np.stack([hitbox_ref.ray_capsule_t(ro, rd, c[0:3], c[3:6], c[6]) for c in caps], 1)
    first = np.where(np.isfinite(t_all.min(1)), t_all.argmin(1), -1)
    _, _, margin = hitbox_ref.first_region(ro, rd, caps, TABLE["head"], TABLE["neck"])
    keep = (first == TABLE["neck"]) & (margin >= 1e-2)
    into_head = keep & np.isfinite(t_all[:, TABLE["head"]])
    neck_only = keep & ~np.isfinite(t_all[:, TABLE["head"]])
    assert into_head.sum() >= 20 and neck_only.sum() >= 20, (into_head.sum(), neck_only.sum())
    got, _ = _c_regions(np.zeros(len(ro), int), ro, rd)
    assert set(got[into_head]) == {TABLE["head"]}
    assert set(got[neck_only]) == {TABLE["neck"]}


# ── T8 ────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("hitboxes", [True, False])
def test_t8_the_group_counters_sum_to_shots_hit(hitboxes):
    """300 seeded shots at random body points (lateral [-20, 20] u, height [0, 75] u) of a
    target 300 to 500 u away at a random facing, standing or crouched. From 300 u no single
    rifle hit kills a full-hp, helmeted target (a head through the helmet does 99.9 there).
    Under hitboxes each shot's counter is also checked against the numpy reference's region for
    the same ray (rays at least 1e-2 u from a surface or a tie): this pins the C world-to-agent
    rotation at the random facings, and the neck counter on every shot that enters the neck.
    KNOWN LIMIT: the reached-all-six check below is seed-dependent (deterministic, not flaky): a
    correct build hits the neck 0 times in 300 shots on 8 of seeds 0-19 (0 to 3 times on any),
    so a new seed or a refitted table can turn it red without a defect."""
    env = _env(hitboxes=hitboxes)
    rng = np.random.default_rng(8)
    totals = dict.fromkeys(HIT_COUNTERS, 0)
    hits = compared = 0
    try:
        for _ in range(300):
            _place(env, float(rng.uniform(300.0, 500.0)), float(rng.uniform(0.0, 2.0 * np.pi)))
            t = env._c_env.game.agents[5]
            yaw, pitch = _aim(env, (t.x, t.y + rng.uniform(-20.0, 20.0), rng.uniform(0.0, 75.0)))
            ss = _shoot(env, yaw, pitch, crouch_target=bool(rng.integers(0, 2)))
            hits += ss.shots_hit
            for k in HIT_COUNTERS:
                totals[k] += getattr(ss, k)
            if not hitboxes:
                continue
            o, dvec, pose = _local_ray(env, yaw, pitch)
            region, _, margin = hitbox_ref.first_region(o[None], dvec[None], TABLE["caps"][pose],
                                                        TABLE["head"], TABLE["neck"])
            if margin[0] < 1e-2:
                continue
            compared += 1
            want = [] if region[0] < 0 else [COUNTER_OF_GROUP[TABLE["groups"][region[0]]]]
            got = [k for k in HIT_COUNTERS if getattr(ss, k)]
            assert got == want, (got, want, t.facing, yaw, pitch)
    finally:
        env.close()
    assert not hitboxes or compared > 250, compared    # known count: an emptied check cannot pass
    assert sum(totals.values()) == hits >= 60, (totals, hits)
    reached = {k for k, v in totals.items() if v}
    if hitboxes:
        assert reached == set(HIT_COUNTERS), totals
    else:                                              # the roll's four outcomes; no neck, no arm
        assert reached == {"hits_head", "hits_chest", "hits_stomach", "hits_leg"}, totals


# ── T9 ────────────────────────────────────────────────────────────────────────────────────────


def test_t9_the_table_invariants():
    caps = np.array(TABLE["caps"], dtype=float)
    assert caps.shape == (2, 18, 7)
    assert (caps[:, :, 6] > 0).all()
    assert set(TABLE["groups"]) == set(range(1, 9)), "the CS hit groups head .. neck, no generic"
    tops = (np.maximum(caps[:, :, 2], caps[:, :, 5]) + caps[:, :, 6]).max(1)
    assert 71.0 <= tops[0] <= 73.0 and 54.0 <= tops[1] <= 57.0, tops   # the CS:GO hull 72 / 54
    for pose in (0, 1):
        head = caps[pose, TABLE["head"]]
        lo, hi = min(head[2], head[5]) - head[6], max(head[2], head[5]) + head[6]
        assert lo < TABLE["eye"][pose] < hi, (pose, lo, hi)


# ── T10 ───────────────────────────────────────────────────────────────────────────────────────

# scripts/sim_fingerprint.py's POST-#157 rows, pre-#391 and pre-cone (its docstring).
LEGACY_FINGERPRINTS = {
    ("--steps", "200"):
    "79f739495e1fb657688cb4274441c590386e319937d5c40aabb6c81f35e65f93",
    ("--steps", "200", "--aim-mode", "track"):
    "7a8f92545e083f8c9142561c5ea211ec87027c17bb7368c273e04d27fb4693ee",
    ("--steps", "700", "--aim-mode", "track"):
    "b7adbbe2a2c9e81ef86624e3699c38d832e7f66220b5980e7e2579bd26f8c8f5",
}
# The bomb rows as #152 L2 moved them: its SIMULTANEOUS-COMBAT rows, the second pair. Simultaneous
# combat moved them first (a dying CT fires); then the shot penalty charged a shooter dead at the
# end of its firing tick, which changed only those agents' rewards on those steps.
LEGACY_BOMB_FINGERPRINTS = {
    ("--steps", "5000", "--aim-mode", "bomb"):
    "1d00d994054e671d873df865562e8cc6052f29d6eadf4dff00f4a1c764544675",
    ("--steps", "5000", "--aim-mode", "bomb", "--n-active", "3"):
    "cbaa454b29ffeece96cb5c4f176f1b624c943ed6e53d27d8304d93adf27537d5",
}


def _fingerprint(monkeypatch, capsys, argv):
    """scripts/sim_fingerprint.py's hash for `argv` + --hitboxes 0, the cone held off: the
    script builds every env through its module-level EnvConfig name, rebound here (it has no
    --accuracy flag). --obs-layout sim-v3 hashes the 110 obs columns these pins were
    measured on: the sim-v4 enemy-velocity slots widen every obs."""
    path = REPO_ROOT / "scripts" / "sim_fingerprint.py"
    spec = importlib.util.spec_from_file_location("sim_fingerprint_t10", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "EnvConfig", functools.partial(EnvConfig, accuracy=False))
    monkeypatch.setattr("sys.argv",
                        ["sim_fingerprint.py", *argv, "--hitboxes", "0", "--obs-layout", "sim-v3"])
    capsys.readouterr()
    mod.main()
    out, err = capsys.readouterr()
    return out.strip().splitlines()[-1], err


@pytest.mark.parametrize("argv", list(LEGACY_FINGERPRINTS), ids="_".join)
def test_t10_the_legacy_model_reproduces_the_pre_hitbox_fingerprints(monkeypatch, capsys, argv):
    digest, err = _fingerprint(monkeypatch, capsys, argv)
    assert digest == LEGACY_FINGERPRINTS[argv], err
    if "track" in argv:
        assert "deaths=0 " not in err, "track mode must reach combat for this pin to cover it"


@pytest.mark.slow
@pytest.mark.parametrize("argv", list(LEGACY_BOMB_FINGERPRINTS), ids="_".join)
def test_t10_the_legacy_model_reproduces_the_bomb_fingerprints(monkeypatch, capsys, argv):
    digest, err = _fingerprint(monkeypatch, capsys, argv)
    assert digest == LEGACY_BOMB_FINGERPRINTS[argv], err
