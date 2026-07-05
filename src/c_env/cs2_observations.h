#pragma once
#include "cs2_types.h"
#include "cs2_weapons.h"
#include "cs2_movement.h" /* SV_JUMP_IMPULSE_CS for obs[8] normalisation */

static void
compute_observations(Dust2Env* env, int t_alive, int ct_alive, int8_t vis10[N_AGENTS][N_AGENTS]) {
    StaticData* sd = env->sd;
    GameState*  g  = &env->game;
    float       map_diag;
    {
        float xr = 1.0f / sd->inv_x_range; /* x range */
        float yr = 1.0f / sd->inv_y_range;
        map_diag = sqrtf(xr * xr + yr * yr);
    }

    for (int i = 0; i < N_AGENTS; i++) {
        float*      obs = &env->observations[i * OBS_DIM];
        AgentState* a   = &g->agents[i];

        memset(obs, 0, OBS_DIM * sizeof(float));

        /* ── Self state (0-24) ── */
        int              slot = a->weapon_slot;
        const WeaponDef* def  = &WEAPON_DEFS[slot];
        obs[0]                = a->hp / 100.0f;
        obs[1]                = a->armor / 100.0f;
        obs[2]                = (float)a->has_helmet;
        obs[3]                = a->x * sd->inv_x_range - sd->x_offset;
        obs[4]                = a->y * sd->inv_y_range - sd->y_offset;
        /* Self Z: jump apex is ~57 u, so normalise by 128 (WALL_HEIGHT-ish)
         * to keep values within ~[-1,1] for the foreseeable range.
         * Range note (Batch 5 verticality): with terrain z up to 128 (catwalk)
         * plus jump apex ≈57u, obs[5] can reach ≈1.45. Network ingests the
         * unclamped float same as velocity slots; no scaling change needed.
         * Teammate/enemy z-delta slots (were 0.0f placeholders) are now
         * (other->z - self->z)/128 — positive = other is above self. */
        obs[5] = a->z / 128.0f;
        obs[6] = (map_diag > 0.0f) ? a->vx / 250.0f : 0.0f;
        obs[7] = (map_diag > 0.0f) ? a->vy / 250.0f : 0.0f;
        /* Vz: jump impulse = 302 u/s, terminal gravity fall ~580 u/s before
         * damage threshold — normalise by the impulse so obs[8] ≈ 1.0 at
         * jump start and ≈ -1.0 at apex-return landing speed. */
        obs[8]  = a->vz / SV_JUMP_IMPULSE_CS;
        obs[9]  = sinf(a->facing);
        obs[10] = cosf(a->facing);
        /* Batch 3.5 (#24): pitch sin/cos appended after yaw cos. Mirrors the yaw
         * encoding pattern (sin/cos pair) so a linear model can recover pitch
         * directly. Range: pitch ∈ [-π/2, +π/2] → sin ∈ [-1,1], cos ∈ [0,1].
         * Pitfall: this insertion shifts EVERY downstream obs[N] for N≥13 by +2.
         * Audit obs index manifest comments AND test files for hardcoded indices. */
        obs[11] = sinf(a->pitch);
        obs[12] = cosf(a->pitch);
        obs[13] = (float)a->is_crouching;
        obs[14] = (slot == 0) ? 1.0f : 0.0f;
        obs[15] = (slot == 1) ? 1.0f : 0.0f;
        obs[16] = (slot == 2) ? 1.0f : 0.0f;
        obs[17] = (def->mag_size > 0) ? a->ammo_clip[slot] / (float)def->mag_size : 1.0f;
        obs[18] = (def->reserve_mags > 0) ? a->ammo_reserve[slot] / (float)def->reserve_mags : 1.0f;
        obs[19] = (a->reload_ticks > 0) ? 1.0f : 0.0f;
        obs[20] = (a->reload_ticks > 0 && def->reload_ticks > 0)
                      ? (def->reload_ticks - a->reload_ticks) / (float)def->reload_ticks
                      : 0.0f;
        obs[21] =
            (a->fire_cd > 0 && def->cycle_ticks > 0) ? a->fire_cd / (float)def->cycle_ticks : 0.0f;
        obs[22] = (float)(a->team == 0 && a->has_bomb);
        obs[23] = (float)a->alive;
        obs[24] = (float)(a->team == 0);

        /* ── Teammates (25-52): 4 × 7 ── */
        int tm_start = (a->team == 0) ? 0 : TEAM_SIZE;
        int tm_count = 0;
        for (int j = tm_start; j < tm_start + TEAM_SIZE && tm_count < 4; j++) {
            if (j == i)
                continue;
            AgentState* tm   = &g->agents[j];
            int         base = OBS_TEAMMATE_BASE + tm_count * OBS_TEAMMATE_STRIDE;
            if (tm->alive) {
                float dx = tm->x - a->x, dy = tm->y - a->y;
                obs[base + 0] = (map_diag > 0.0f) ? dx / map_diag : 0.0f;
                obs[base + 1] = (map_diag > 0.0f) ? dy / map_diag : 0.0f;
                /* Relative z-delta: positive = teammate above us.  Same /128
                 * scale as self obs[5].  Range: catwalk(128) - spawn(0) = +1.0;
                 * spawn - catwalk = -1.0.  Was constant-0 placeholder pre-T4. */
                obs[base + 2] = (tm->z - a->z) / 128.0f;
                obs[base + 3] = tm->hp / 100.0f;
                obs[base + 4] = 1.0f;
                float angle   = atan2f(dy, dx);
                obs[base + 5] = sinf(angle);
                obs[base + 6] = cosf(angle);
            }
            /* dead teammate: all zeros (already memset) */
            tm_count++;
        }

        /* ── Enemies (53-92): 5 × 8 ── */
        int en_start = (a->team == 0) ? TEAM_SIZE : 0;
        /* Sort by distance — simple insertion sort over 5 elements */
        int   order[TEAM_SIZE];
        float dists[TEAM_SIZE];
        for (int s = 0; s < TEAM_SIZE; s++) {
            order[s]       = en_start + s;
            AgentState* en = &g->agents[order[s]];
            float       dx = en->x - a->x, dy = en->y - a->y;
            dists[s] = dx * dx + dy * dy;
        }
        for (int s = 1; s < TEAM_SIZE; s++) {
            int   ko = order[s];
            float kd = dists[s];
            int   t  = s - 1;
            while (t >= 0 && dists[t] > kd) {
                order[t + 1] = order[t];
                dists[t + 1] = dists[t];
                t--;
            }
            order[t + 1] = ko;
            dists[t + 1] = kd;
        }
        for (int slot2 = 0; slot2 < TEAM_SIZE; slot2++) {
            int         ej      = order[slot2];
            AgentState* en      = &g->agents[ej];
            int         base    = OBS_ENEMY_BASE + slot2 * OBS_ENEMY_STRIDE;
            int         mem_s   = ej - en_start;
            int         can_see = en->alive ? vis10[i][ej] : 0;

            obs[base + 4] = (float)en->alive;
            obs[base + 3] = (float)can_see;

            if (can_see) {
                float dx = en->x - a->x, dy = en->y - a->y;
                float dist    = sqrtf(dx * dx + dy * dy);
                obs[base + 0] = (map_diag > 0.0f) ? dx / map_diag : 0.0f;
                obs[base + 1] = (map_diag > 0.0f) ? dy / map_diag : 0.0f;
                /* Relative z-delta: positive = enemy above us.  Same /128
                 * scale as self obs[5] and teammate slot.  Range mirrors
                 * teammate block.  Was constant-0 placeholder pre-T4.
                 * NB: visibility-gated — when !can_see (invisible / memory-only),
                 * obs[base+2] stays 0 from memset.  Asymmetry vs teammate slot
                 * (which writes z-delta whenever tm->alive).  Policies must
                 * disambiguate "0 = invisible enemy" from "0 = same height" via
                 * obs[base+3] (can_see flag at base+3). */
                obs[base + 2] = (en->z - a->z) / 128.0f;
                float angle   = atan2f(dy, dx);
                obs[base + 5] = sinf(angle);
                obs[base + 6] = cosf(angle);
                obs[base + 7] = (map_diag > 0.0f) ? dist / map_diag : 0.0f;
            } else if (a->enemy_mem_idx[mem_s] != INVALID_AREA_IDX) {
                /* Use last-known position from memory */
                float mx      = sd->centroid_xy[a->enemy_mem_idx[mem_s] * 2] - a->x;
                float my      = sd->centroid_xy[a->enemy_mem_idx[mem_s] * 2 + 1] - a->y;
                obs[base + 0] = (map_diag > 0.0f) ? mx / map_diag : 0.0f;
                obs[base + 1] = (map_diag > 0.0f) ? my / map_diag : 0.0f;
            }
        }

        /* ── Global / bomb block: obs[OBS_GLOBAL_BASE + 0 .. +13] ──
         * Written base-relative (not bare 93..106) so the whole block shifts
         * automatically if an upstream block (self/teammate/enemy) is resized.
         * Slot map: +0 round-time · +1..4 bomb one-hot · +5,6 bomb xy ·
         * +7 bomb-z placeholder · +8 bomb timer · +9 plant · +10 defuse ·
         * +11 t_alive · +12 ct_alive · +13 designated-carrier bit. */
        int gb      = OBS_GLOBAL_BASE;
        obs[gb + 0] = g->round_ticks_left / (float)sd->round_time;
        /* bomb status one-hot (+1..+4) */
        int carrier = g->bomb_carrier_id;
        if (!g->bomb_planted && !g->bomb_is_dropped) {
            if (carrier == i)
                obs[gb + 1] = 1.0f; /* carried by self */
            else if (carrier >= 0 && a->team == 0)
                obs[gb + 2] = 1.0f; /* carried by teammate */
        } else if (g->bomb_is_dropped) {
            obs[gb + 3] = 1.0f;
        } else if (g->bomb_planted) {
            obs[gb + 4] = 1.0f;
        }
        /* bomb position (+5,+6) */
        if (g->bomb_planted || g->bomb_is_dropped) {
            float bx = g->bomb_x - a->x, by = g->bomb_y - a->y;
            obs[gb + 5] = (map_diag > 0.0f) ? bx / map_diag : 0.0f;
            obs[gb + 6] = (map_diag > 0.0f) ? by / map_diag : 0.0f;
        } else if (carrier >= 0 && a->team == 0 && carrier != i) {
            /* Teammate carrying: show their position */
            float bx    = g->agents[carrier].x - a->x;
            float by    = g->agents[carrier].y - a->y;
            obs[gb + 5] = (map_diag > 0.0f) ? bx / map_diag : 0.0f;
            obs[gb + 6] = (map_diag > 0.0f) ? by / map_diag : 0.0f;
        }
        /* Bomb z deferred — out of scope for T4 (spec L4 covers teammate/enemy
         * z-delta only).  Bomb z would require obs version contract for plug-in.
         * See gh #(filed) for the bomb-z-aware obs follow-up. */
        obs[gb + 7] = 0.0f; /* bomb z placeholder (intentionally constant pre-followup) */
        obs[gb + 8] = g->bomb_planted ? g->bomb_ticks_left / (float)sd->bomb_timer : 0.0f;
        obs[gb + 9] = (g->bomb_being_planted_by >= 0 && sd->bomb_plant_time > 0)
                          ? g->bomb_plant_ticks / (float)sd->bomb_plant_time
                          : 0.0f;
        /* +10: defuse progress — extract to avoid GCC statement-expression */
        {
            float defuse_prog = 0.0f;
            if (g->bomb_being_defused_by >= 0) {
                AgentState* def2  = &g->agents[g->bomb_being_defused_by];
                int         dtime = def2->has_kit ? sd->bomb_defuse_kit : sd->bomb_defuse_time;
                if (dtime > 0)
                    defuse_prog = g->bomb_defuse_ticks / (float)dtime;
            }
            obs[gb + 10] = defuse_prog;
        }
        obs[gb + 11] = t_alive / (float)TEAM_SIZE;
        obs[gb + 12] = ct_alive / (float)TEAM_SIZE;

        /* Batch 2: round-fixed designated-carrier role bit (T-side semantic).
         * 1.0 only when this agent is the round's designated bomb carrier
         * (set in env_reset, never reassigned). Distinct from obs[22]
         * (transient self-has-bomb) — gives the policy a stable identity
         * signal that survives drop/pickup. CT agents always read 0.0. */
        obs[gb + 13] = (a->team == 0 && i == g->round_designated_carrier_id) ? 1.0f : 0.0f;

        /* Clip all obs to (-5, 5) */
        for (int k = 0; k < OBS_DIM; k++) {
            if (obs[k] > 5.0f)
                obs[k] = 5.0f;
            else if (obs[k] < -5.0f)
                obs[k] = -5.0f;
        }
    }
}
