#pragma once
#include <math.h>
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
 *     rectangle). viz.py draws walls at room perimeters, so dead space
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
 * viz.py WALL_H), so 2D LoS suffices for walls. Verticality (catwalk,
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
            return 0; /* dead space — wall by viz.py convention */
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
     * depends on the aim roll. */
    StaticData* sd = env->sd;
    GameState*  g  = &env->game;

    /* Hitbox probabilities: head, chest, stomach, legs */
    /* Standing:  10%, 40%, 25%, 25% */
    /* Crouching:  5%, 45%, 25%, 25% */
    static const float hitbox_mult[4]    = {4.0f, 1.0f, 1.25f, 0.75f};
    static const int   hitbox_armored[4] = {1, 1, 1, 0}; /* legs never armored */

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
     * to 3D cross-product perp distance instead of 2D. */
    static const float HIT_HALF_WIDTH = 16.0f;

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
     * Pitfall: real CS player models are taller (~72u) than wide (~32u); the
     * spherical HIT_HALF_WIDTH=16 sphere under-counts vertical silhouette at
     * long range. If a future v1b smoke STILL fails hit-rate, the canonical
     * fix is the spec Risk 4 ellipsoidal hitbox (perp_h/16)² + (perp_v/36)² < 1. */
    static const float EYE_HEIGHT_STAND    = 48.0f;
    static const float EYE_HEIGHT_CROUCH   = 24.0f;
    static const float TORSO_OFFSET_STAND  = 48.0f;
    static const float TORSO_OFFSET_CROUCH = 24.0f;

    for (int i = 0; i < N_AGENTS; i++) {
        AgentState* a = &g->agents[i];
        if (!a->alive)
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

        /* ── R0-A shooter-side counters (spec §3). Scored against the
         * pre-combat nearest visible participating enemy `tgt` (or none). ── */
        int tgt = nearest_vis_enemy[i];
        if (a->participating) {
            ss->shots_fired++;
            es->shots_fired++;
            if (tgt >= 0) {
                AgentState* en = &g->agents[tgt];
                float       rx = en->x - a->x, ry = en->y - a->y;
                float       tgt_d    = sqrtf(rx * rx + ry * ry);
                float       tgt_derr = fabsf(wrap_pi(atan2f(ry, rx) - a->facing));
                /* Same eye/torso convention as the hit ray below (v1b). */
                float eye_z0 = a->z + (a->is_crouching ? EYE_HEIGHT_CROUCH : EYE_HEIGHT_STAND);
                float torso_z0 =
                    en->z + (en->is_crouching ? TORSO_OFFSET_CROUCH : TORSO_OFFSET_STAND);
                float tgt_rz = torso_z0 - eye_z0;
                ss->shots_with_enemy_in_los++;
                es->shots_with_enemy_in_los++;
                if (tgt_derr < (float)M_PI / 4.0f) {
                    ss->shots_facing_enemy++;
                    es->shots_facing_enemy++;
                }
                /* MARGINAL tests: on_target is the yaw error alone against the
                 * half-window, stance_blocked is |dz| alone. The hit ray's gate
                 * is the JOINT sqrt(yaw_perp^2 + rz^2) < HIT_HALF_WIDTH plus
                 * laser_range and forward > 0, so shots_hit / shots_on_target
                 * < 1 is EXPECTED near the boundary even at pitch 0. Do not
                 * "fix" either counter to match the ray — they answer
                 * different questions. Clamp keeps asinf NaN-free. */
                if (tgt_derr < asinf(fminf(HIT_HALF_WIDTH / fmaxf(tgt_d, 1e-6f), 1.0f))) {
                    ss->shots_on_target++;
                    es->shots_on_target++;
                }
                if (fabsf(tgt_rz) > HIT_HALF_WIDTH) {
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
        float eye_z = a->z + (a->is_crouching ? EYE_HEIGHT_CROUCH : EYE_HEIGHT_STAND);

        int         en_start   = (a->team == 0) ? TEAM_SIZE : 0;
        float       best_dist  = sd->laser_range; /* reuse laser_range as max combat range */
        AgentState* best_enemy = NULL;

        for (int ej = en_start; ej < en_start + TEAM_SIZE; ej++) {
            AgentState* en = &g->agents[ej];
            if (!en->alive || !vis10[i][ej])
                continue;

            float rx = en->x - a->x;
            float ry = en->y - a->y;
            /* Target torso z — mid-body for laser-style single-hitbox combat.
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

            /* Cross-product magnitude form for perpendicular distance.
             * Numerically stable: avoids the catastrophic cancellation of the
             * sqrt(|r|² - forward²) form when forward ≈ |r| (perfectly aligned
             * shot). Per spec L4 + Opus review I1.
             * Since |d| = 1, |r × d| = perpendicular distance from r to the
             * aim ray in world units. The 2D version (|rx*dy - ry*dx|) is the
             * pitch=0, rz=0 reduction (cz term only) — verifiable by setting
             * sin_p=0, cos_p=1: cx = ry*0 - 0*dy = 0; cy = 0*dx - rx*0 = 0;
             * cz = rx*dy - ry*dx. */
            float cx   = ry * dz - rz * dy;
            float cy   = rz * dx - rx * dz;
            float cz   = rx * dy - ry * dx;
            float perp = sqrtf(cx * cx + cy * cy + cz * cz);
            if (perp > HIT_HALF_WIDTH)
                continue;

            if (dist < best_dist) {
                best_dist  = dist;
                best_enemy = en;
            }
        }

        if (best_enemy == NULL)
            continue;

        /* Stochastic hitbox selection */
        uint32_t roll = xorshift32(&env->rng) % 100;
        int      hitbox;
        int      head_thresh = best_enemy->is_crouching ? 5 : 10;
        if ((int)roll < head_thresh)
            hitbox = 0; /* head   */
        else if ((int)roll < head_thresh + 40)
            hitbox = 1; /* chest  */
        else if ((int)roll < head_thresh + 65)
            hitbox = 2; /* stomach */
        else
            hitbox = 3; /* legs   */

        /* Knife: flat damage, no range dropoff, ignore hitbox mult */
        float hp_damage;
        float armor_damage = 0.0f;
        if (def->type == 2) { /* knife */
            hp_damage = def->base_damage;
        } else {
            float raw = def->base_damage * hitbox_mult[hitbox];
            if (def->range_modifier > 0.0f)
                raw *= powf(def->range_modifier, best_dist / 500.0f);
            if (best_enemy->armor > 0 && hitbox_armored[hitbox]) {
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
            ss->damage_dealt += (float)(int32_t)hp_damage;
            es->damage_dealt += (float)(int32_t)hp_damage;
        }

        if (best_enemy->hp <= 0) {
            best_enemy->hp    = 0;
            best_enemy->alive = 0;
            if (*n_kills < N_AGENTS) {
                kills[*n_kills][0] = i;
                kills[*n_kills][1] = (int)(best_enemy - g->agents);
                (*n_kills)++;
            }
        }
    }
}
