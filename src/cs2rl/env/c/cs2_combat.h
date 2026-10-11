#pragma once
#include <math.h>
#include "cs2_hitboxes.h"
#include "cs2_types.h"
#include "cs2_weapons.h"

/* line_of_sight_2d — Amanatides-Woo grid raycast between two world (x,y).
 *
 * Returns 1 (clear) iff every grid cell along the line is in a valid area
 * AND every area transition along the path satisfies adjacency. Returns 0
 * otherwise.
 *
 * Why two checks (NOT just `raster_grid >= 0`):
 *   - raster_grid == -1 means "no room" (dead space outside any room
 *     rectangle). viz/render.py draws walls at room perimeters, so dead space
 *     between rooms IS a wall. The raycast must reject lines crossing it.
 *   - Adjacency catches walls between rooms that SHARE an edge but aren't
 *     connected — e.g. bombsite (z=64, area 6) and catwalk (z=128, area 15)
 *     share y=192 but the wall between them is full-height. Without the
 *     adjacency check, a line crossing 6→15 would falsely report "clear"
 *     because both cells have raster_grid >= 0.
 *
 * Replaces the prior centroid-based area→area `vis_matrix` lookup, which
 * gave false negatives like `vis[bombsite][T-spawn-C] = 0` when the
 * straight centroid-to-centroid line happened to exit the playable area
 * even though edge-of-room → edge-of-room lines stay inside valid rooms
 * the whole way. Demo bug: "step off bombsite, hits land". gh #36 follow-up.
 *
 * Cost: O(max(Δgx, Δgy)) cells per call. ~100-200 cells for typical
 * engagement distances on simple_map; cheap in C. 90 calls/tick × 256
 * envs × 16Hz fits easily inside a sub-1% slice of the step budget. No
 * pre-computed table required (pre-bake would be ~9 MB for simple_map,
 * grow O(cells²) for dust2 — not worth it given runtime cost).
 *
 * 2D-only by design: simple_map walls are full-height (z=0..150 per
 * viz/render.py WALL_H), so 2D LoS suffices for walls. Verticality (catwalk,
 * bombsite) is handled by adjacency: cross-z transitions like
 * bombsite↔catwalk are non-adjacent → blocked at the cell boundary.
 * If we ever model partial-height cover (low boxes, smokes), we'll need
 * a 3D voxel raycast — separate change.
 *
 * Pitfall: source/target positions must be inside valid cells. Agents
 * are pinned to their area_idx by process_movement; if you pass a
 * position from a non-agent caller (dropped weapon, projectile mid-air),
 * make sure it's in a valid cell or this returns blocked at step 0.
 */
static int line_of_sight_2d(StaticData* sd, float x1, float y1, float x2, float y2) {
    int   N        = sd->N;
    int   W        = sd->grid_w;
    int   H        = sd->grid_h;
    float inv_cell = sd->grid_inv_cell;
    float gx_min   = sd->grid_x_min;
    float gy_min   = sd->grid_y_min;

    /* World → grid coords (float). */
    float fx0 = (x1 - gx_min) * inv_cell;
    float fy0 = (y1 - gy_min) * inv_cell;
    float fx1 = (x2 - gx_min) * inv_cell;
    float fy1 = (y2 - gy_min) * inv_cell;

    int gx     = (int)fx0;
    int gy     = (int)fy0;
    int gx_end = (int)fx1;
    int gy_end = (int)fy1;

    /* Off-grid endpoints → treat as blocked (shouldn't happen for live
     * agents but defensive against pathological inputs). */
    if (gx < 0 || gx >= W || gy < 0 || gy >= H)
        return 0;
    if (gx_end < 0 || gx_end >= W || gy_end < 0 || gy_end >= H)
        return 0;

    /* Source cell must itself be valid; without this we'd skip its check
     * (the loop only inspects cells AFTER stepping). */
    int prev_area = sd->raster_grid[gy * W + gx];
    if (prev_area < 0)
        return 0;
    if (gx == gx_end && gy == gy_end)
        return 1; /* same cell — trivially clear */

    /* Amanatides-Woo setup: t at which the ray crosses the next x/y cell
     * boundary, in units of [0,1] over the segment. Whichever t is smaller
     * tells us whether the next cell crossed is east-west (gx step) or
     * north-south (gy step). */
    float dx     = fx1 - fx0;
    float dy     = fy1 - fy0;
    int   step_x = (dx > 0.0f) ? 1 : (dx < 0.0f ? -1 : 0);
    int   step_y = (dy > 0.0f) ? 1 : (dy < 0.0f ? -1 : 0);
    /* Δt to advance one cell in each axis. INFINITY when step==0 keeps the
     * `if (t_max_x < t_max_y)` branch from picking that axis. */
    float t_delta_x = (step_x != 0) ? fabsf(1.0f / dx) : 1e30f;
    float t_delta_y = (step_y != 0) ? fabsf(1.0f / dy) : 1e30f;
    /* Distance (in t) to the FIRST cell boundary in each axis. */
    float t_max_x = (step_x > 0) ? ((float)(gx + 1) - fx0) * t_delta_x
                                 : (step_x < 0 ? (fx0 - (float)gx) * t_delta_x : 1e30f);
    float t_max_y = (step_y > 0) ? ((float)(gy + 1) - fy0) * t_delta_y
                                 : (step_y < 0 ? (fy0 - (float)gy) * t_delta_y : 1e30f);

    /* Walk cells until we reach the target cell or hit a blocker.
     *
     * Termination via explicit step cap, NOT `gx == gx_end && gy == gy_end`.
     * Why: floating-point t_max bookkeeping can make a step take gx (or gy)
     * one cell PAST gx_end (gy_end) when the line is nearly axis-aligned —
     * a strict-equality termination then never fires and the loop walks off
     * across the entire grid (and into dead space, falsely reporting blocked).
     * abs(Δgx) + abs(Δgy) is the exact upper bound on steps in Amanatides-Woo
     * since each iteration advances exactly ONE of (gx, gy) by 1; +1 for the
     * already-checked source cell. */
    int max_steps =
        (gx_end > gx ? gx_end - gx : gx - gx_end) + (gy_end > gy ? gy_end - gy : gy - gy_end);
    for (int step = 0; step < max_steps; step++) {
        if (gx == gx_end && gy == gy_end)
            break; /* hit target cell exactly */
        if (t_max_x < t_max_y) {
            t_max_x += t_delta_x;
            gx      += step_x;
        } else {
            t_max_y += t_delta_y;
            gy      += step_y;
        }
        if (gx < 0 || gx >= W || gy < 0 || gy >= H)
            return 0; /* line walked off the grid */
        int curr_area = sd->raster_grid[gy * W + gx];
        if (curr_area < 0)
            return 0; /* dead space — wall by viz/render.py convention */
        if (curr_area != prev_area) {
            /* Crossed a room boundary. Connected? */
            if (!sd->adjacency[prev_area * N + curr_area])
                return 0; /* wall between two valid but disjoint rooms */
            prev_area = curr_area;
        }
    }
    return 1;
}

/* build_vis_matrix — per-tick agent-pair visibility (combat + obs + memory).
 *
 * Was: O(1) lookup into the pre-baked area→area sd->vis_matrix using each
 * agent's area_idx. Coarse and produced false negatives whenever the
 * centroid-to-centroid baking line exited the playable area even though
 * actual agent positions had clear LoS (the bombsite→spawn demo bug).
 *
 * Now: per-pair position raycast via line_of_sight_2d. Aligns combat,
 * fog-of-war observations, and enemy-memory with what the human sees in
 * raylib (and what awpy/CS gamestate would feed a deployed bot). Same
 * lookup interface for downstream consumers (vis10[i][j]).
 *
 * vis10[i][i] = 1 by convention (self-LoS); harmless — combat skips i==j
 * via team partitioning, obs skips via "enemy" indexing, memory likewise.
 *
 * area_idx < 0 short-circuits to 0 (off-mesh agent — shouldn't happen for
 * live agents but defensive against dead/unspawned slots).
 */
static void build_vis_matrix(GameState* g, StaticData* sd, int8_t vis10[N_AGENTS][N_AGENTS]) {
    for (int i = 0; i < N_AGENTS; i++) {
        AgentState* ai = &g->agents[i];
        for (int j = 0; j < N_AGENTS; j++) {
            if (i == j) {
                vis10[i][j] = 1;
                continue;
            }
            AgentState* aj = &g->agents[j];
            if (ai->area_idx < 0 || aj->area_idx < 0) {
                vis10[i][j] = 0;
                continue;
            }
            vis10[i][j] = (int8_t)line_of_sight_2d(sd, ai->x, ai->y, aj->x, aj->y);
        }
    }
}

/* accuracy_unit: one xorshift32 draw as a float in [0, 1), on 24 bits (exact in float). */
static inline float accuracy_unit(uint32_t* rng) {
    return (float)(xorshift32(rng) >> 8) * (1.0f / 16777216.0f);
}

/* accuracy_cone_ray — the movement-inaccuracy cone, applied to the unit ray d = (dx, dy, dz)
 * that process_combat built from (yaw, pitch).
 *
 * WHAT: tilts d by an angle r in [0, R) toward an azimuth t uniform in [0, 2 pi):
 *   d' = cos r d + sin r (cos t right + sin t up),
 *   right = (sin yaw, -cos yaw, 0), up = (-sin pitch cos yaw, -sin pitch sin yaw, cos pitch),
 * the two unit vectors perpendicular to d, so the angle between d' and d is exactly r and
 * |d'| = 1. r = R u (ACCURACY_RADIUS_LAW 0) or R sqrt(u) (1), u uniform in [0, 1).
 * Two draws from *rng per call, u first.
 * WHY a rotation, not yaw += r cos t and pitch += r sin t: the angle offsets tilt the ray by
 * less than r once pitch is non-zero; the rotation is the same cone at every pitch.
 * PITFALL: *rng must be env->accuracy_rng. A draw from env->rng would also move the next
 * round's spawns and, under the legacy hit model, every later hit-location roll. */
static inline void
accuracy_cone_ray(float* dx, float* dy, float* dz, float yaw, float pitch, float R, uint32_t* rng) {
    float u  = accuracy_unit(rng);
    float t  = 2.0f * (float)M_PI * accuracy_unit(rng);
    float r  = R * (ACCURACY_RADIUS_LAW ? sqrtf(u) : u);
    float cy = cosf(yaw), sy = sinf(yaw), cp = cosf(pitch), sp = sinf(pitch);
    float ct = cosf(t), st = sinf(t), cr = cosf(r), sr = sinf(r);
    *dx = cr * *dx + sr * (ct * sy - st * sp * cy);
    *dy = cr * *dy + sr * (-ct * cy - st * sp * sy);
    *dz = cr * *dz + sr * (st * cp);
}

/* ── LEGACY hit model (Dust2Env.hit_model HIT_MODEL_ELLIPSOID, EnvConfig.hitboxes False) ──
 * The six constants and every note in the block below describe the pre-#391 model: one
 * ellipsoid per agent and a random hit-location roll in process_combat. It is kept
 * bit-identical for reproducing pinned numbers. The default model is cs2_hitboxes.h's. */
/* Batch 3.5 (#24): eye-height + torso-offset for the 3D hit-test.
 * Shooter eye = a->z + EYE_HEIGHT_*; target torso = en->z + TORSO_OFFSET_*.
 *
 * v1a (T5 initial): EYE=64/32 (top-of-model), TORSO=32/16 (mid-body), per
 * CS view-angle convention. CAUSED COLD-START FAILURE in T7 30M smoke run
 * (gh #36): with eye_z=64 and torso_z=32, flat-ground (same-z) shots give
 * rz=-32 → perp_3d=32 > HIT_HALF_WIDTH=16 → MISS at pitch=0. A random-init
 * policy can't bootstrap because every flat-ground shot misses regardless
 * of yaw alignment, so there's never a kill signal to learn pitch from.
 * Verticality smoke (Batch 3, no pitch) achieved Plant=0.024 by epoch 180
 * on simple_map; pitch3d v1a achieved 0 across the full 30M run.
 *
 * v1b (this commit, fix A from gh #36 diagnosis): EYE = TORSO at the same
 * stance — both refer to the agent's vertical CENTER (~48u standing,
 * ~24u crouching, midway between feet and top-of-model). This restores
 * the spec L4 promise that "pitch=0, rz=0 reduces to original 2D logic"
 * for same-stance same-z combat: rz = (a->z + 48) − (a->z + 48) = 0.
 * Asymmetric-stance shots (stand-vs-crouch) get rz = ±24, which is just
 * over HIT_HALF_WIDTH=16 → policy must learn a small pitch correction.
 * Verticality engagements (catwalk z=128 vs bombsite z=64) unchanged in
 * spirit: rz = 64 dominates the eye/torso term either way.
 *
 * Tunable in one place; the existing per-shot stochastic hitbox roll
 * (head/chest/stomach/leg multipliers) is unchanged — these constants
 * only affect the 3D-perp gate of "did the shot connect to the target's
 * vertical centerline."
 *
 * v1c (2026-08-30, gh #150): ELLIPSOIDAL hitbox — the spec Risk 4 fix the
 * v1b note below predicted. Real CS player models are ~72u tall (54u
 * crouched) and ~32u wide; the v1b 16u SPHERE at the torso centre made any
 * target whose centre sat > 16u off the ray an unconditional miss. With
 * pitch pinned (R0-E.2) that meant a crouched target (|rz| = 24) and —
 * the Rung 1 sweep finding — ANY agent mid-jump (57u apex,
 * SV_JUMP_IMPULSE_CS): ~50% of shots were stance-blocked, hit/on_target
 * ≈ 0.5. Gate is now (perp_h/HIT_HALF_WIDTH)² + (perp_v/HIT_HALF_HEIGHT)²
 * ≤ 1 where perp_h/perp_v are the horizontal/vertical components of the
 * perpendicular from target centre to the ray and HIT_HALF_HEIGHT follows
 * the TARGET's stance (36 standing, 27 crouched = half the model height).
 * At pitch 0, same z AND SAME STANCE: perp_v = 0 and the gate reduces to
 * the v1b/2D perp ≤ 16 exactly — §5's yaw-window arithmetic (asin(16/d))
 * is intact. Mixed stance at pitch 0 is NOT that case: perp_v = rz = ±24,
 * which is precisely the shot v1c turns from a miss into a hit.
 * CAVEAT (the box is not centred on the model): the semi-heights are half
 * the model, but the centre is TORSO_OFFSET (48 / 24) — the v1b eye-parity
 * value, not the model's mid-point — so the standing box spans z+12 … z+84
 * (12u above the head, feet excluded) and the crouched box z−3 … z+51 (3u
 * below the ground). Harmless for parity, which needs only symmetry about
 * rz = 0, but it is not a faithful player bounding box. Do not "fix"
 * TORSO_OFFSET without redoing the pitch-0 reduction above.
 * CAVEAT (conservative): this tests the ray's CLOSEST POINT to the target
 * centre against the ellipsoid, not a true ray∩ellipsoid intersection
 * (which would scale d into the unit-sphere frame first). It is strictly
 * conservative — an oblique graze can read as a miss, never the reverse —
 * and it matches the spec formula verbatim.
 * RESIDUAL: a horizontal ray still cannot reach a jumper whose centre is
 * > 36u above the shooter's eye. For a 57u jump (v0 = 302, g = 800,
 * T = 0.755 s) the centre sits above the 36u band for t ∈ [0.148, 0.607] s
 * — ~61% of the airtime, since a projectile spends most of its airtime
 * near apex — and the horizontal tolerance 16·sqrt(1 − (perp_v/36)²)
 * collapses to 0 as the centre nears the band edge, so the usable window
 * is smaller still. That is a pitch-pin limitation, not geometry: v1c is a
 * PARTIAL mitigation of gh #150, not a close-out. */
static const float EYE_HEIGHT_STAND       = 48.0f;
static const float EYE_HEIGHT_CROUCH      = 24.0f;
static const float TORSO_OFFSET_STAND     = 48.0f;
static const float TORSO_OFFSET_CROUCH    = 24.0f;
static const float HIT_HALF_HEIGHT_STAND  = 36.0f; /* v1c: half of ~72u model */
static const float HIT_HALF_HEIGHT_CROUCH = 27.0f; /* v1c: half of ~54u crouched */

/* agent_eye_height — the shooter's eye above its feet under env's hit model.
 * WHAT: HB_EYE_STAND / HB_EYE_CROUCH (64 / 46, cs2_hitboxes.h) under HIT_MODEL_HITBOXES,
 * EYE_HEIGHT_STAND / EYE_HEIGHT_CROUCH (48 / 24, above) under HIT_MODEL_ELLIPSOID.
 * WHY one helper: a shooter that sees from one height and shoots from another is a defect
 * (#391 lead decision D1). It is the one eye for the shot ray's origin, the R0-A counters and
 * cs2_render.h's camera and sound positions; Python reads the same numbers through
 * binding.hitbox_table() (cs2rl.env.c.hit_model).
 * PITFALL: line_of_sight_2d is 2D and takes no eye height, so visibility does not follow the
 * switch; there is nothing there to follow. */
static inline float agent_eye_height(const Dust2Env* env, int crouched) {
    if (env->hit_model == HIT_MODEL_ELLIPSOID)
        return crouched ? EYE_HEIGHT_CROUCH : EYE_HEIGHT_STAND;
    return crouched ? HB_EYE_CROUCH : HB_EYE_STAND;
}

static void process_combat(Dust2Env*      env,
                           const int32_t* actions,
                           int8_t         vis10[N_AGENTS][N_AGENTS],
                           int            kills[N_AGENTS][2],
                           int*           n_kills,
                           StepStats*     ss,
                           StepStats*     es,
                           const int      nearest_vis_enemy[N_AGENTS]) {
    /* nearest_vis_enemy: R0-A pre-combat snapshot from env_step (cs2_env.h),
     * -1 = no visible participating enemy. Scoring target for the shooter
     * counters below; deliberately NOT the hit-ray's best_enemy, which
     * depends on where the ray goes. */
    StaticData* sd = env->sd;
    GameState*  g  = &env->game;
    /* The hit model (Dust2Env.hit_model): hitboxes = 1 under HIT_MODEL_HITBOXES (cs2_hitboxes.h's
     * capsules), 0 under HIT_MODEL_ELLIPSOID (the legacy ellipsoid and roll). Each branch below
     * that tests it leaves the legacy code as it was, so a legacy env reproduces the pre-#391
     * sim bit for bit (tests/env/c/test_hitboxes.py, the sim_fingerprint comparison). */
    int hitboxes = env->hit_model != HIT_MODEL_ELLIPSOID;

    /* LEGACY roll (hitboxes == 0): hitbox probabilities head, chest, stomach, legs */
    /* Standing:  10%, 40%, 25%, 25% */
    /* Crouching:  5%, 45%, 25%, 25% */
    static const float hitbox_mult[4]    = {4.0f, 1.0f, 1.25f, 0.75f};
    static const int   hitbox_armored[4] = {1, 1, 1, 0}; /* legs never armored */
    /* The roll's outcomes as CS hit groups, for the per-group counters. */
    static const int hitbox_group[4] = {
        HB_HITGROUP_HEAD, HB_HITGROUP_CHEST, HB_HITGROUP_STOMACH, HB_HITGROUP_LEFTLEG};

    /* HIT_HALF_WIDTH — perpendicular hit tolerance from the aim ray to the
     * enemy centre, in world units. 16.0f = rendered head-sphere radius
     * (widest horizontal extent of the agent silhouette; body cylinder is
     * 12). Applied uniformly to humans and bots — a real FPS needs precise
     * aim, so bot policies have to learn (or be retrained for) it too.
     * Bots still have a 16-bin discrete aim head (22.5° per bin), which
     * means at typical combat range their angular resolution is far wider
     * than this hit window and they will whiff most shots until either the
     * aim head is widened or policies are retrained. Expected.
     * Batch 3.5: now used as 3D spherical radius — same value, but applied
     * to 3D cross-product perp distance instead of 2D.
     * #391: the legacy ellipsoid's horizontal semi-axis, and in BOTH hit models the
     * yaw half-window of the shots_on_target counter. The hitbox model's own hit test
     * is cs2_hitboxes.h's capsules and does not read it. */
    static const float HIT_HALF_WIDTH = 16.0f;

    /* Simultaneous resolution (#152 L2): every shot of the tick is resolved against the agents
     * alive BEFORE combat. A shooter killed earlier in this loop still fires, and a later ray
     * can still hit an agent this tick has already killed (its body blocks the ray, the damage
     * lands, no second kill is recorded). WHY: with deaths applied inside the loop, the lower
     * index fired first, so when both duellists fired lethal shots on one tick the T (agents
     * 0..4) lived and the CT never fired (two identical scripted duel bots: the T won 646 of
     * 1000 rounds, 2026-10-11). A same-tick exchange is now a double kill, and
     * env_step scores a round where both teams die on one tick as a draw (WINNER_DRAW).
     * PITFALL: kill credit on a target hit by several shooters this tick still goes to the
     * lowest index whose shot took hp to 0. */
    int8_t alive_pre[N_AGENTS];
    for (int i = 0; i < N_AGENTS; i++)
        alive_pre[i] = g->agents[i].alive;

    for (int i = 0; i < N_AGENTS; i++) {
        AgentState* a = &g->agents[i];
        if (!alive_pre[i])
            continue;

        int shoot = actions[i * ACTION_DIM + HEAD_SHOOT];
        if (shoot == 0 || a->fire_cd > 0 || a->reload_ticks > 0 || a->switch_ticks > 0)
            continue;

        const WeaponDef* def = &WEAPON_DEFS[a->weapon_slot];
        /* Dry-fire: empty magazine + finite-ammo weapon. Skip the shot entirely
         * — no trigger cooldown, no muzzle flash, no round consumed. Mirrors CS
         * where pressing fire on an empty clip is a no-op until you reload. */
        if (def->mag_size > 0 && a->ammo_clip[a->weapon_slot] <= 0)
            continue;

        a->fire_cd         = def->cycle_ticks;
        a->fired_this_tick = 1;
        if (def->mag_size > 0)
            a->ammo_clip[a->weapon_slot]--; /* consume one round */

        /* The shooter's planar speed at the shot: process_movement has already run this
         * tick, so a counter-strafe pressed this tick has slowed it (env_step order). */
        float shot_speed = sqrtf(a->vx * a->vx + a->vy * a->vy);
        int   shot_fast  = shot_speed >= ACCURACY_RAMP_START_FRAC * def->move_speed;

        /* ── R0-A shooter-side counters (spec §3). Scored against the
         * pre-combat nearest visible participating enemy `tgt` (or none). ── */
        int tgt = nearest_vis_enemy[i];
        if (a->participating) {
            ss->shots_fired++;
            es->shots_fired++;
            ss->shots_fired_fast += shot_fast;
            es->shots_fired_fast += shot_fast;
            if (tgt >= 0) {
                AgentState* en = &g->agents[tgt];
                float       rx = en->x - a->x, ry = en->y - a->y;
                float       tgt_d    = sqrtf(rx * rx + ry * ry);
                float       tgt_derr = fabsf(wrap_pi(atan2f(ry, rx) - a->facing));
                /* The same eye as the hit ray below (agent_eye_height). */
                float eye_z0 = a->z + agent_eye_height(env, a->is_crouching);
                ss->shots_with_enemy_in_los++;
                es->shots_with_enemy_in_los++;
                if (tgt_derr < (float)M_PI / 4.0f) {
                    ss->shots_facing_enemy++;
                    es->shots_facing_enemy++;
                }
                /* MARGINAL tests: on_target is the yaw error alone against the
                 * half-window. On the legacy ellipsoid: stance_blocked is |dz|
                 * alone against the target's vertical semi-axis (v1c), and the hit
                 * ray's gate is the JOINT ellipsoid (perp_h/16)² + (perp_v/HH)² ≤ 1
                 * plus laser_range and forward > 0 (under hitboxes: the eye against
                 * the capsule z band, below, and the capsule test), so shots_hit /
                 * shots_on_target < 1 is EXPECTED near the boundary even at pitch 0.
                 * Do not "fix" either counter to match the ray — they answer
                 * different questions. Clamp keeps asinf NaN-free.
                 * Under the hitbox model on_target is the same yaw test against the
                 * same 16 u, kept so the counter means one thing in both models. It is
                 * a much looser marginal there than what a shot can hit (a level shot
                 * at a standing head, front-on at 360 u, has a 7.75 u wide window; #391
                 * phase 1): do not read on_target / fired as a hit-rate ceiling. */
                if (tgt_derr < asinf(fminf(HIT_HALF_WIDTH / fmaxf(tgt_d, 1e-6f), 1.0f))) {
                    ss->shots_on_target++;
                    es->shots_on_target++;
                }
                int blocked;
                if (hitboxes) {
                    /* The eye outside the target pose's capsule z band (cs2_hitboxes.h
                     * HB_BOUND_Z0/Z1): no level ray can reach any capsule. On flat
                     * ground that is exactly a standing shooter against a crouched
                     * target (the band tops out at 55.23 u, under the 64 u eye). */
                    int   pose = en->is_crouching ? HB_POSE_CROUCH : HB_POSE_STAND;
                    float lz   = eye_z0 - en->z;
                    blocked    = lz < HB_BOUND_Z0[pose] || lz > HB_BOUND_Z1[pose];
                } else {
                    /* Legacy: the eye-to-torso offset against the ellipsoid's vertical
                     * semi-axis (v1c). */
                    float torso_z0 =
                        en->z + (en->is_crouching ? TORSO_OFFSET_CROUCH : TORSO_OFFSET_STAND);
                    float tgt_rz = torso_z0 - eye_z0;
                    float tgt_hh =
                        en->is_crouching ? HIT_HALF_HEIGHT_CROUCH : HIT_HALF_HEIGHT_STAND;
                    blocked = fabsf(tgt_rz) > tgt_hh;
                }
                if (blocked) {
                    ss->shots_stance_blocked++;
                    es->shots_stance_blocked++;
                }
            }
        }

        /* Batch 3.5 (#24): 3D aim direction. Pitch tilts the (cos·yaw, sin·yaw)
         * 2D direction to a 3D unit vector. d = (cos·p·cos·y, cos·p·sin·y, sin·p).
         * |d| = 1 by construction.
         * Sim recoil v1 (#120): when recoil_enabled, add punch to locals only.
         * Do not write punch into facing / aim_rad / stored pitch. */
        float ray_yaw   = a->facing;
        float ray_pitch = a->pitch;
        if (env->recoil_enabled) {
            ray_yaw   += a->punch_yaw;
            ray_pitch += a->punch_pitch;
            if (ray_pitch > 1.5533f)
                ray_pitch = 1.5533f;
            if (ray_pitch < -1.5533f)
                ray_pitch = -1.5533f;
        }
        float cos_p = cosf(ray_pitch);
        float sin_p = sinf(ray_pitch);
        float dx    = cos_p * cosf(ray_yaw);
        float dy    = cos_p * sinf(ray_yaw);
        float dz    = sin_p;

        /* Movement-inaccuracy cone: the shot leaves on a cone around the aimed ray (punch
         * included) whose radius grows with the shooter's speed (cs2_weapons.h). Off: no
         * draw at all, so every RNG stream and every ray is the pre-cone sim's. The R0-A
         * counters above score the AIM (a->facing), not this ray, so under the cone
         * shots_hit < shots_on_target is expected. */
        if (env->accuracy_enabled) {
            float cone_r = accuracy_cone_radius(def, a->is_crouching, shot_speed);
            accuracy_cone_ray(&dx, &dy, &dz, ray_yaw, ray_pitch, cone_r, &env->accuracy_rng);
            if (a->participating) {
                ss->cone_radius_sum += cone_r;
                es->cone_radius_sum += cone_r;
            }
        }

        if (env->recoil_enabled) {
            /* After d is built, before the enemy loop — misses still kick. */
            unsigned u      = (unsigned)g->tick * 1664525u + 1013904223u;
            float    n      = ((float)((u >> 16) & 0xffff) / 32767.5f) - 1.0f;
            a->punch_pitch += 0.045f;
            a->punch_yaw   += n * 0.008f;
        }

        /* Shooter eye z (3D combat ray origin) — incorporates stand/crouch.
         * Without this, a crouched defender on an elevated catwalk and a
         * standing attacker at the bottom of the bombsite would have wrong
         * relative angles (off by ±32u). */
        float eye_z = a->z + agent_eye_height(env, a->is_crouching);

        int         en_start    = (a->team == 0) ? TEAM_SIZE : 0;
        float       best_dist   = sd->laser_range; /* reuse laser_range as max combat range */
        AgentState* best_enemy  = NULL;
        int         best_region = -1; /* hitbox model: the region the ray entered first */

        for (int ej = en_start; ej < en_start + TEAM_SIZE; ej++) {
            AgentState* en = &g->agents[ej];
            if (!alive_pre[ej] || !vis10[i][ej])
                continue;

            if (hitboxes) {
                /* The first capsule the ray enters on this enemy, and how far along the
                 * ray: that distance is the range check, the nearest-enemy tie-break and
                 * (below) the damage falloff distance. t >= 0, so an enemy behind the
                 * shooter is never hit. */
                float t;
                int   region = hb_ray_vs_agent(en, a->x, a->y, eye_z, dx, dy, dz, &t);
                if (region >= 0 && t < best_dist) {
                    best_dist   = t;
                    best_enemy  = en;
                    best_region = region;
                }
                continue;
            }

            float rx = en->x - a->x;
            float ry = en->y - a->y;
            /* LEGACY model from here to the end of the loop body.
             * Target torso z — mid-body for laser-style single-hitbox combat.
             * The existing per-shot stochastic head/chest/stomach/leg roll
             * (lines below) determines WHERE on the body the shot lands;
             * the 3D-perp gate here only decides IF a shot connects. */
            float torso_z = en->z + (en->is_crouching ? TORSO_OFFSET_CROUCH : TORSO_OFFSET_STAND);
            float rz      = torso_z - eye_z;
            float dist_sq = rx * rx + ry * ry + rz * rz;
            /* NB (Opus I4): best_dist is now 3D distance. Since 3D ≥ 2D, the
             * same sd->laser_range budget is STRICTER, not more permissive —
             * long-range engagements with mild Δz that the 2D version accepted
             * may now be filtered out. If T7's hit-rate criterion 3 dips, a
             * v1b tweak is bumping sd->laser_range to compensate.
             * Pitfall: the `dist_sq == 0.0f` skip below now requires FULL 3D
             * coincidence — two agents at the same (x, y) but different z
             * (e.g., directly above/below) used to be skipped under the 2D
             * formula and now proceed to the perp/forward checks. This is
             * arguably a 2D bug fix; flagged here so future readers know. */
            if (dist_sq > best_dist * best_dist || dist_sq == 0.0f)
                continue;

            float dist    = sqrtf(dist_sq);
            float forward = rx * dx + ry * dy + rz * dz; /* 3D signed projection */
            if (forward <= 0.0f)
                continue;                                /* enemy behind shooter */

            /* v1c (gh #150): ellipsoidal gate. Decompose the perpendicular
             * p = r − forward·d into its vertical (p_z) and horizontal
             * (|p_xy|) parts and test (|p_xy|/16)² + (p_z/HH)² ≤ 1, HH per the
             * TARGET's stance. See the constant block above for the geometry
             * caveats (box centre, conservatism, pitch-pin residual).
             * NUMERICS (spec L4 + Opus review I1): this componentwise vector
             * rejection is used INSTEAD of the sqrt(|r|² − forward²) form for
             * the same cancellation reason the old cross-product form was —
             * that form catastrophically cancels when forward ≈ |r| (perfectly
             * aligned shot), whereas each component here loses only ~eps·|r|.
             * Better still, the test is on squared quantities, so no sqrt is
             * needed at all; `dist` above is the only one, and it is for the
             * range check and the nearest-enemy tie-break, not the gate.
             * Reduction: at pitch 0 with rz = 0, p_z = 0 and
             * |p_xy| = |rx*dy − ry*dx| — the old 2D perpendicular — so the
             * gate is exactly the v1b perp ≤ 16. */
            float px = rx - forward * dx;
            float py = ry - forward * dy;
            float pz = rz - forward * dz;
            float hh = en->is_crouching ? HIT_HALF_HEIGHT_CROUCH : HIT_HALF_HEIGHT_STAND;
            float ell =
                (px * px + py * py) / (HIT_HALF_WIDTH * HIT_HALF_WIDTH) + (pz * pz) / (hh * hh);
            if (ell > 1.0f)
                continue;

            if (dist < best_dist) {
                best_dist  = dist;
                best_enemy = en;
            }
        }

        if (best_enemy == NULL)
            continue;

        int hitbox = 0; /* legacy: the roll's outcome */
        int group;      /* the CS hit group, both models */
        if (hitboxes) {
            /* No draw: the region the ray entered decides. env->rng is then drawn only at
             * round reset (the bomb carrier, spawns and defuse kits, cs2_env.h env_reset). */
            group = HB_GROUP[best_region];
        } else {
            /* LEGACY stochastic hitbox selection */
            uint32_t roll        = xorshift32(&env->rng) % 100;
            int      head_thresh = best_enemy->is_crouching ? 5 : 10;
            if ((int)roll < head_thresh)
                hitbox = 0; /* head   */
            else if ((int)roll < head_thresh + 40)
                hitbox = 1; /* chest  */
            else if ((int)roll < head_thresh + 65)
                hitbox = 2; /* stomach */
            else
                hitbox = 3; /* legs   */
            group = hitbox_group[hitbox];
        }

        /* Knife: flat damage, no range dropoff, ignore hitbox mult */
        float hp_damage;
        float armor_damage = 0.0f;
        if (def->type == 2) { /* knife */
            hp_damage = def->base_damage;
        } else {
            /* Hitbox model: best_dist is the capsule entry distance; the group sets the
             * multiplier and (with the helmet for the head) whether armor applies. */
            float mult = hitboxes ? hb_group_mult(group) : hitbox_mult[hitbox];
            int   armored =
                hitboxes ? hb_group_armored(group, best_enemy->has_helmet) : hitbox_armored[hitbox];
            float raw = def->base_damage * mult;
            if (def->range_modifier > 0.0f)
                raw *= powf(def->range_modifier, best_dist / 500.0f);
            if (best_enemy->armor > 0 && armored) {
                hp_damage    = raw * def->armor_pen;
                armor_damage = (raw - hp_damage) * 0.5f;
            } else {
                hp_damage = raw;
            }
        }

        /* Apply damage */
        best_enemy->armor -= (int32_t)armor_damage;
        if (best_enemy->armor < 0)
            best_enemy->armor = 0;
        best_enemy->hp -= (int32_t)hp_damage;
        /* R0-A: hit counters. Cast mirrors the line above so damage_dealt
         * equals the hp actually subtracted, not the float pre-truncation sum. */
        if (a->participating) {
            ss->shots_hit++;
            es->shots_hit++;
            hb_count_group(ss, group);
            hb_count_group(es, group);
            ss->damage_dealt += (float)(int32_t)hp_damage;
            es->damage_dealt += (float)(int32_t)hp_damage;
        }

        if (best_enemy->hp <= 0) {
            best_enemy->hp = 0;
            /* The tick's first lethal shot kills; a later one hits the body (alive_pre). */
            if (best_enemy->alive) {
                best_enemy->alive = 0;
                if (*n_kills < N_AGENTS) {
                    kills[*n_kills][0] = i;
                    kills[*n_kills][1] = (int)(best_enemy - g->agents);
                    (*n_kills)++;
                }
            }
        }
    }
}
