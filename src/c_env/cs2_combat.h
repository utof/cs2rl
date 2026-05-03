#pragma once
#include "cs2_types.h"
#include "cs2_weapons.h"

static void build_vis_matrix(GameState* g, StaticData* sd, int8_t vis10[N_AGENTS][N_AGENTS]) {
    for (int i = 0; i < N_AGENTS; i++) {
        for (int j = 0; j < N_AGENTS; j++) {
            int ai      = g->agents[i].area_idx;
            int aj      = g->agents[j].area_idx;
            vis10[i][j] = (ai >= 0 && aj >= 0) ? sd->vis_matrix[ai * sd->N + aj] : 0;
        }
    }
}

static void process_combat(Dust2Env*      env,
                           const int32_t* actions,
                           int8_t         vis10[N_AGENTS][N_AGENTS],
                           int            kills[N_AGENTS][2],
                           int*           n_kills,
                           StepStats*     ss,
                           StepStats*     es) {
    StaticData* sd = env->sd;
    GameState*  g  = &env->game;
    (void)ss;
    (void)es;

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
     * Numbers picked from CS view-angle convention proportionally scaled to our
     * world units (catwalk z=128, bombsite z=64). Tunable in one place; the
     * existing per-shot stochastic hitbox roll (head/chest/stomach/leg multipliers)
     * is unchanged — these constants only affect the 3D-perp gate of "did the shot
     * connect to the target's vertical centerline."
     * Pitfall: real CS player models are taller (~72u) than wide (~32u); the
     * spherical HIT_HALF_WIDTH=16 sphere under-counts vertical silhouette at long
     * range. If hit-rate criterion 3 fails (T7), v1b candidate is ellipsoidal
     * (perp_h/16)² + (perp_v/36)² < 1 — single-place change. */
    static const float EYE_HEIGHT_STAND    = 64.0f;
    static const float EYE_HEIGHT_CROUCH   = 32.0f;
    static const float TORSO_OFFSET_STAND  = 32.0f;
    static const float TORSO_OFFSET_CROUCH = 16.0f;

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

        /* Batch 3.5 (#24): 3D aim direction. Pitch tilts the (cos·yaw, sin·yaw)
         * 2D direction to a 3D unit vector. d = (cos·p·cos·y, cos·p·sin·y, sin·p).
         * |d| = 1 by construction. */
        float cos_p = cosf(a->pitch);
        float sin_p = sinf(a->pitch);
        float dx    = cos_p * cosf(a->facing);
        float dy    = cos_p * sinf(a->facing);
        float dz    = sin_p;

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
