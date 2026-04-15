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

        /* ── Self state (0-22) ── */
        int              slot = a->weapon_slot;
        const WeaponDef* def  = &WEAPON_DEFS[slot];
        obs[0]                = a->hp / 100.0f;
        obs[1]                = a->armor / 100.0f;
        obs[2]                = (float)a->has_helmet;
        obs[3]                = a->x * sd->inv_x_range - sd->x_offset;
        obs[4]                = a->y * sd->inv_y_range - sd->y_offset;
        /* Self Z: jump apex is ~57 u, so normalise by 128 (WALL_HEIGHT-ish)
         * to keep values within ~[-1,1] for the foreseeable range. */
        obs[5] = a->z / 128.0f;
        obs[6] = (map_diag > 0.0f) ? a->vx / 250.0f : 0.0f;
        obs[7] = (map_diag > 0.0f) ? a->vy / 250.0f : 0.0f;
        /* Vz: jump impulse = 302 u/s, terminal gravity fall ~580 u/s before
         * damage threshold — normalise by the impulse so obs[8] ≈ 1.0 at
         * jump start and ≈ -1.0 at apex-return landing speed. */
        obs[8]  = a->vz / SV_JUMP_IMPULSE_CS;
        obs[9]  = sinf(a->facing);
        obs[10] = cosf(a->facing);
        obs[11] = (float)a->is_crouching;
        obs[12] = (slot == 0) ? 1.0f : 0.0f;
        obs[13] = (slot == 1) ? 1.0f : 0.0f;
        obs[14] = (slot == 2) ? 1.0f : 0.0f;
        obs[15] = (def->mag_size > 0) ? a->ammo_clip[slot] / (float)def->mag_size : 1.0f;
        obs[16] = (def->reserve_mags > 0) ? a->ammo_reserve[slot] / (float)def->reserve_mags : 1.0f;
        obs[17] = (a->reload_ticks > 0) ? 1.0f : 0.0f;
        obs[18] = (a->reload_ticks > 0 && def->reload_ticks > 0)
                      ? (def->reload_ticks - a->reload_ticks) / (float)def->reload_ticks
                      : 0.0f;
        obs[19] =
            (a->fire_cd > 0 && def->cycle_ticks > 0) ? a->fire_cd / (float)def->cycle_ticks : 0.0f;
        obs[20] = (float)(a->team == 0 && a->has_bomb);
        obs[21] = (float)a->alive;
        obs[22] = (float)(a->team == 0);

        /* ── Teammates (23-50): 4 × 7 ── */
        int tm_start = (a->team == 0) ? 0 : TEAM_SIZE;
        int tm_count = 0;
        for (int j = tm_start; j < tm_start + TEAM_SIZE && tm_count < 4; j++) {
            if (j == i)
                continue;
            AgentState* tm   = &g->agents[j];
            int         base = 23 + tm_count * 7;
            if (tm->alive) {
                float dx = tm->x - a->x, dy = tm->y - a->y;
                obs[base + 0] = (map_diag > 0.0f) ? dx / map_diag : 0.0f;
                obs[base + 1] = (map_diag > 0.0f) ? dy / map_diag : 0.0f;
                obs[base + 2] = 0.0f; /* z placeholder */
                obs[base + 3] = tm->hp / 100.0f;
                obs[base + 4] = 1.0f;
                float angle   = atan2f(dy, dx);
                obs[base + 5] = sinf(angle);
                obs[base + 6] = cosf(angle);
            }
            /* dead teammate: all zeros (already memset) */
            tm_count++;
        }

        /* ── Enemies (51-90): 5 × 8 ── */
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
            int         base    = 51 + slot2 * 8;
            int         mem_s   = ej - en_start;
            int         can_see = en->alive ? vis10[i][ej] : 0;

            obs[base + 4] = (float)en->alive;
            obs[base + 3] = (float)can_see;

            if (can_see) {
                float dx = en->x - a->x, dy = en->y - a->y;
                float dist    = sqrtf(dx * dx + dy * dy);
                obs[base + 0] = (map_diag > 0.0f) ? dx / map_diag : 0.0f;
                obs[base + 1] = (map_diag > 0.0f) ? dy / map_diag : 0.0f;
                obs[base + 2] = 0.0f; /* z placeholder */
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

        /* ── Global / bomb (91-103) ── */
        obs[91] = g->round_ticks_left / (float)sd->round_time;
        /* bomb status one-hot (92-95) */
        int carrier = g->bomb_carrier_id;
        if (!g->bomb_planted && !g->bomb_is_dropped) {
            if (carrier == i)
                obs[92] = 1.0f; /* carried by self */
            else if (carrier >= 0 && a->team == 0)
                obs[93] = 1.0f; /* carried by teammate */
        } else if (g->bomb_is_dropped) {
            obs[94] = 1.0f;
        } else if (g->bomb_planted) {
            obs[95] = 1.0f;
        }
        /* bomb position (96-98) */
        if (g->bomb_planted || g->bomb_is_dropped) {
            float bx = g->bomb_x - a->x, by = g->bomb_y - a->y;
            obs[96] = (map_diag > 0.0f) ? bx / map_diag : 0.0f;
            obs[97] = (map_diag > 0.0f) ? by / map_diag : 0.0f;
        } else if (carrier >= 0 && a->team == 0 && carrier != i) {
            /* Teammate carrying: show their position */
            float bx = g->agents[carrier].x - a->x;
            float by = g->agents[carrier].y - a->y;
            obs[96]  = (map_diag > 0.0f) ? bx / map_diag : 0.0f;
            obs[97]  = (map_diag > 0.0f) ? by / map_diag : 0.0f;
        }
        obs[98]  = 0.0f; /* z placeholder */
        obs[99]  = g->bomb_planted ? g->bomb_ticks_left / (float)sd->bomb_timer : 0.0f;
        obs[100] = (g->bomb_being_planted_by >= 0 && sd->bomb_plant_time > 0)
                       ? g->bomb_plant_ticks / (float)sd->bomb_plant_time
                       : 0.0f;
        /* obs[101]: defuse progress — extract to avoid GCC statement-expression */
        {
            float defuse_prog = 0.0f;
            if (g->bomb_being_defused_by >= 0) {
                AgentState* def2  = &g->agents[g->bomb_being_defused_by];
                int         dtime = def2->has_kit ? sd->bomb_defuse_kit : sd->bomb_defuse_time;
                if (dtime > 0)
                    defuse_prog = g->bomb_defuse_ticks / (float)dtime;
            }
            obs[101] = defuse_prog;
        }
        obs[102] = t_alive / (float)TEAM_SIZE;
        obs[103] = ct_alive / (float)TEAM_SIZE;

        /* Clip all obs to (-5, 5) */
        for (int k = 0; k < OBS_DIM; k++) {
            if (obs[k] > 5.0f)
                obs[k] = 5.0f;
            else if (obs[k] < -5.0f)
                obs[k] = -5.0f;
        }
    }
}
