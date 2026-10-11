/* cs2_hitboxes.h — the deterministic anatomical hit model (#391), Dust2Env.hit_model
 * HIT_MODEL_HITBOXES.
 *
 * WHAT: a shot ray is tested against each enemy's 18 capsules (cs2_hitbox_table.h, generated
 * by scripts/hitboxes/gen_table.py from a CC0 mannequin), and the capsule it ENTERS FIRST is
 * the region hit. The region's CS hit group then sets the damage multiplier and whether armor
 * applies. Nothing here draws a random number.
 * WHY: CS2 traces a bullet against capsule hitboxes that ride the skeleton (datamined: two
 * sources agree on 19 bone-attached capsules), so where a shot lands is decided by the aim.
 * The legacy model, HIT_MODEL_ELLIPSOID (cs2_combat.h: one ellipsoid per agent and a random
 * head/chest/stomach/legs roll), stays behind the switch to reproduce pre-#391 numbers.
 * Frame: the table is agent-local (fwd, left, up) with the feet at the origin and
 * fwd = (cos facing, sin facing); hb_ray_vs_agent rotates the world ray into it.
 * PITFALL: the capsule dimensions are fits to the mannequin, not Valve data (CS2's own are
 * not public). Edit the generator, never the table. */
#pragma once
#include <math.h>
#include "cs2_hitbox_table.h"
#include "cs2_types.h"

/* CS:GO's VEC_VIEW and VEC_DUCK_VIEW, the eye above the feet (research 1.4; CS2: no confirmed
 * change). Both lie inside the table's head capsule for their pose (the stand head spans
 * z 60.96 to 71.78, the crouched one 44.42 to 55.23), so a level shot between two agents of
 * the same stance on flat ground crosses the head band. cs2_combat.h agent_eye_height picks
 * these or the legacy model's 48 / 24. */
static const float HB_EYE_STAND  = 64.0f;
static const float HB_EYE_CROUCH = 46.0f;

/* hb_sphere_t — entry distance of the unit ray o + t d into the sphere (c, r), or -1.
 * WHAT: the smaller root of |o + t d - c|^2 = r^2, kept only when t >= 0.
 * PITFALL: d must be unit length (the quadratic's leading coefficient is taken as 1). An
 * origin inside the sphere has its smaller root behind it and returns -1: an agent standing
 * inside another one's head is not a hit. */
static inline float hb_sphere_t(float ox,
                                float oy,
                                float oz,
                                float dx,
                                float dy,
                                float dz,
                                float cx,
                                float cy,
                                float cz,
                                float r) {
    float qx = ox - cx, qy = oy - cy, qz = oz - cz;
    float b = dx * qx + dy * qy + dz * qz;
    float c = qx * qx + qy * qy + qz * qz - r * r;
    float h = b * b - c;
    if (h < 0.0f)
        return -1.0f;
    float t = -b - sqrtf(h);
    return t >= 0.0f ? t : -1.0f;
}

/* hb_capsule_t — entry distance of the unit ray o + t d into one table row
 * cp = {ax, ay, az, bx, by, bz, r}, or -1 for a miss.
 * WHAT: Inigo Quilez's capsule intersection: the infinite cylinder around A-B first; its entry
 * counts when it lands between the two end planes; otherwise the nearer of the two end
 * spheres.
 * WHY hand-written: no library available to the binding intersects a ray with a capsule
 * (raylib 5.5 has spheres, boxes and meshes only, and is not linked into the binding; #391
 * phase 1). Its numpy twin matched brute-force ray marching on 1000 rays, and this C form
 * matched the numpy one on 199712 rays away from a boundary; tests/env/c/test_hitboxes.py
 * keeps the second check (tests/_helpers/hitbox_ref.py is the numpy twin).
 * PITFALL: two degenerate rows need their own branch. A == B (the neck) is a sphere: the
 * cylinder terms divide by |B - A|^2. A ray parallel to the axis never enters the cylinder's
 * side (a ~ 0), only the caps; without the `par` branch it would divide by ~0.
 * KNOWN LIMIT: the float32 test is pinned to 520 u (T1). At 1000-3000 u it decides rays within
 * ~0.5 u of a surface differently from float64: 133 of 19960 rays at a 0.01 u margin, 0 from
 * 0.5 u, where the entry distance agrees within 1.92 u. The lane G review judged that far below
 * the cone's spread and the capsule fit error (its R2). */
static inline float
hb_capsule_t(const float* cp, float ox, float oy, float oz, float dx, float dy, float dz) {
    float ax = cp[0], ay = cp[1], az = cp[2], r = cp[6];
    float bax = cp[3] - ax, bay = cp[4] - ay, baz = cp[5] - az;
    float oax = ox - ax, oay = oy - ay, oaz = oz - az;
    float baba = bax * bax + bay * bay + baz * baz;
    if (baba < 1e-12f)
        return hb_sphere_t(ox, oy, oz, dx, dy, dz, ax, ay, az, r);
    float bard = bax * dx + bay * dy + baz * dz;
    float baoa = bax * oax + bay * oay + baz * oaz;
    float rdoa = dx * oax + dy * oay + dz * oaz;
    float oaoa = oax * oax + oay * oay + oaz * oaz;
    float a    = baba - bard * bard;
    float b    = baba * rdoa - baoa * bard;
    float c    = baba * oaoa - baoa * baoa - r * r * baba;
    float h    = b * b - a * c;
    int   par  = a < 1e-9f * baba;
    if (!par) {
        if (h < 0.0f)
            return -1.0f; /* misses the infinite cylinder, so the caps too */
        float t = (-b - sqrtf(h)) / a;
        float y = baoa + t * bard;
        if (y > 0.0f && y < baba && t >= 0.0f)
            return t;
    }
    float t0 = hb_sphere_t(ox, oy, oz, dx, dy, dz, ax, ay, az, r);
    float t1 = hb_sphere_t(ox, oy, oz, dx, dy, dz, cp[3], cp[4], cp[5], r);
    if (t0 < 0.0f)
        return t1;
    if (t1 < 0.0f)
        return t0;
    return t0 < t1 ? t0 : t1;
}

/* hb_bound_pass — 0 when the agent-local ray cannot reach any capsule of `pose`.
 * WHAT: the vertical cylinder of radius HB_BOUND_R[pose] around the agent's axis, z in
 * [HB_BOUND_Z0, HB_BOUND_Z1], both padded outward by the generator. A ray that never enters
 * that cylinder misses every capsule. Conservative the other way: 1 is "maybe".
 * WHY: it halves the per-test cost (phase 1, 287.82 to 155.89 ns), and its measured count of
 * rejected hits was 0 of 20569.
 * PITFALL: a vertical ray (no horizontal component) passes iff its origin is inside the
 * radius; its z direction is not checked, which only makes it more permissive. */
static inline int
hb_bound_pass(int pose, float ox, float oy, float oz, float dx, float dy, float dz) {
    float R = HB_BOUND_R[pose];
    float a = dx * dx + dy * dy, b = ox * dx + oy * dy, c = ox * ox + oy * oy - R * R;
    if (a < 1e-12f)
        return c <= 0.0f;
    float h = b * b - a * c;
    if (h < 0.0f)
        return 0;
    float sq = sqrtf(h), t0 = (-b - sq) / a, t1 = (-b + sq) / a;
    if (t1 < 0.0f)
        return 0;
    float z0 = oz + dz * t0, z1 = oz + dz * t1;
    float lo = z0 < z1 ? z0 : z1, hi = z0 < z1 ? z1 : z0;
    return !(hi < HB_BOUND_Z0[pose] || lo > HB_BOUND_Z1[pose]);
}

/* hb_local_region — the region an agent-local unit ray hits on `pose`, or -1; *t_out gets its
 * entry distance (unwritten on a miss).
 * WHAT: the bounding-cylinder early-out, then the capsule with the smallest entry t, then the
 * neck rule: a ray whose first capsule is the neck and which also enters the head counts as
 * the head.
 * WHY the neck rule: Valve's CS:GO release note of 2018-05-21, "Bullets that penetrate the neck
 * into the head will now count as headshots". That CS2 keeps it is INFERRED, not sourced. It
 * matters side-on, where about half the neck-first rays go on into the head (report 5.7).
 * PITFALL: ties (equal t, a ray through a joint where two capsules touch) go to the LOWER
 * index (the strict <), so the table's region order is the priority order, head first.
 * binding.hitbox_first_region calls this directly, so the test oracle sees the same code. */
static inline int hb_local_region(
    int pose, float ox, float oy, float oz, float dx, float dy, float dz, float* t_out) {
    if (!hb_bound_pass(pose, ox, oy, oz, dx, dy, dz))
        return -1;
    int   best = -1;
    float bt   = 1e30f;
    for (int i = 0; i < HB_N_REGIONS; i++) {
        float t = hb_capsule_t(HB_CAPS[pose][i], ox, oy, oz, dx, dy, dz);
        if (t >= 0.0f && t < bt) {
            bt   = t;
            best = i;
        }
    }
    if (best == HB_REGION_NECK &&
        hb_capsule_t(HB_CAPS[pose][HB_REGION_HEAD], ox, oy, oz, dx, dy, dz) >= 0.0f)
        best = HB_REGION_HEAD;
    if (best >= 0)
        *t_out = bt;
    return best;
}

/* hb_ray_vs_agent — the region the world ray o + t d (d unit) hits on agent `en`, or -1; *t_out
 * gets the entry distance from o, which process_combat uses as the shot's range.
 * WHAT: picks the pose from en->is_crouching, moves the origin to en's feet, rotates origin and
 * direction by -en->facing into the table's (fwd, left, up) frame, and calls hb_local_region.
 * PITFALL: the pose follows the stance only. A moving or airborne agent uses the standing
 * pose (run-cycle poses are deferred, #394). */
static inline int hb_ray_vs_agent(const AgentState* en,
                                  float             ox,
                                  float             oy,
                                  float             oz,
                                  float             dx,
                                  float             dy,
                                  float             dz,
                                  float*            t_out) {
    int   pose = en->is_crouching ? HB_POSE_CROUCH : HB_POSE_STAND;
    float c = cosf(en->facing), s = sinf(en->facing);
    float rx = ox - en->x, ry = oy - en->y;
    return hb_local_region(pose,
                           rx * c + ry * s,
                           -rx * s + ry * c,
                           oz - en->z,
                           dx * c + dy * s,
                           -dx * s + dy * c,
                           dz,
                           t_out);
}

/* hb_group_mult — the damage multiplier of a CS hit group.
 * WHAT: head 4.0, stomach 1.25, legs 0.75, everything else (neck, chest, arms) 1.0. The
 * stomach and leg factors are CS:GO's reverse-engineered ScaleDamage; the pelvis region is in
 * the stomach group (INFERRED, one community source).
 * PITFALL: CS takes the head factor from the weapon (m_flHeadshotMultiplier). 4.0 is the value
 * of every gun the sim's two averaged rows stand for (AK-47, M4A4, Glock-18, USP-S); a weapon
 * with another value needs a WeaponDef field (#395). The knife never reads this: its damage is
 * flat (cs2_combat.h). */
static inline float hb_group_mult(int group) {
    switch (group) {
        case HB_HITGROUP_HEAD:
            return 4.0f;
        case HB_HITGROUP_STOMACH:
            return 1.25f;
        case HB_HITGROUP_LEFTLEG:
        case HB_HITGROUP_RIGHTLEG:
            return 0.75f;
        default:
            return 1.0f;
    }
}

/* hb_group_armored — 1 when armor (if any) applies to a hit on `group`.
 * WHAT: the head only with a helmet, the legs never, every other group always (CS:GO's
 * IsArmored). The caller still requires armor > 0.
 * PITFALL: the legacy model armors the head on armor alone (cs2_combat.h hitbox_armored). A
 * round spawn gives a helmet (cs2_player.h), so the two agree at spawn defaults. */
static inline int hb_group_armored(int group, int has_helmet) {
    if (group == HB_HITGROUP_HEAD)
        return has_helmet != 0;
    if (group == HB_HITGROUP_LEFTLEG || group == HB_HITGROUP_RIGHTLEG)
        return 0;
    return 1;
}

/* hb_count_group — one hit on `group` into the per-group StepStats counters.
 * WHAT: hits_head, hits_neck, hits_chest, hits_stomach, hits_arm (either arm), hits_leg (either
 * leg). Called next to every shots_hit++, so the six sum to shots_hit in both hit models (the
 * legacy roll's four outcomes map to head, chest, stomach and leg). */
static inline void hb_count_group(StepStats* s, int group) {
    switch (group) {
        case HB_HITGROUP_HEAD:
            s->hits_head++;
            break;
        case HB_HITGROUP_NECK:
            s->hits_neck++;
            break;
        case HB_HITGROUP_CHEST:
            s->hits_chest++;
            break;
        case HB_HITGROUP_STOMACH:
            s->hits_stomach++;
            break;
        case HB_HITGROUP_LEFTARM:
        case HB_HITGROUP_RIGHTARM:
            s->hits_arm++;
            break;
        case HB_HITGROUP_LEFTLEG:
        case HB_HITGROUP_RIGHTLEG:
            s->hits_leg++;
            break;
        default:
            break; /* no such group: the sum then falls short of shots_hit, which the tests check */
    }
}
