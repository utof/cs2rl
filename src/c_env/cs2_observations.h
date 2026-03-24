#pragma once
#include "cs2_types.h"

static void compute_observations(
    Dust2Env* env,
    int t_alive,
    int ct_alive,
    int8_t vis10[N_AGENTS][N_AGENTS])
{
    StaticData* sd            = env->sd;
    GameState*  g             = &env->game;
    float       alive_t_frac  = t_alive / (float)TEAM_SIZE;
    float       alive_ct_frac = ct_alive / (float)TEAM_SIZE;

    for (int i = 0; i < N_AGENTS; i++) {
        float*      obs = &env->observations[i * OBS_DIM];
        AgentState* a   = &g->agents[i];
        int         tm_start;
        int         tm_count = 0;
        int         en_start;

        memset(obs, 0, OBS_DIM * sizeof(float));

        obs[0] = (float)a->team;
        obs[1] = a->x * sd->inv_x_range - sd->x_offset;
        obs[2] = a->y * sd->inv_y_range - sd->y_offset;
        obs[3] = sinf(a->facing);
        obs[4] = cosf(a->facing);
        obs[5] = a->hp / 100.0f;
        obs[6] = (float)(a->team == 0 ? a->has_bomb : a->has_kit);
        obs[7] = (a->fire_cd == 0) ? 1.0f : 1.0f - a->fire_cd / 10.0f;

        tm_start = (a->team == 0) ? 0 : TEAM_SIZE;
        for (int j = tm_start; j < tm_start + TEAM_SIZE; j++) {
            AgentState* tm;
            int         base;
            if (j == i) {
                continue;
            }
            tm            = &g->agents[j];
            base          = 8 + tm_count * 5;
            obs[base + 0] = tm->x * sd->inv_x_range - sd->x_offset;
            obs[base + 1] = tm->y * sd->inv_y_range - sd->y_offset;
            obs[base + 2] = sinf(tm->facing);
            obs[base + 3] = cosf(tm->facing);
            obs[base + 4] = tm->alive ? tm->hp / 100.0f : 0.0f;
            tm_count++;
            if (tm_count == TEAM_SIZE - 1) {
                break;
            }
        }

        en_start = (a->team == 0) ? TEAM_SIZE : 0;
        for (int slot = 0; slot < TEAM_SIZE; slot++) {
            AgentState* en        = &g->agents[en_start + slot];
            int         base      = 28 + slot * 7;
            int         mem_idx   = a->enemy_mem_idx[slot];
            int         mem_tick  = a->enemy_mem_tick[slot];
            int         can_see   = en->alive ? vis10[i][en_start + slot] : 0;
            float       freshness = 0.0f;

            if (mem_idx == INVALID_AREA_IDX && !can_see) {
                continue;
            }

            if (mem_idx != INVALID_AREA_IDX) {
                obs[base + 0] = sd->centroid_xy[mem_idx * 2] * sd->inv_x_range - sd->x_offset;
                obs[base + 1] =
                    sd->centroid_xy[mem_idx * 2 + 1] * sd->inv_y_range - sd->y_offset;
            }
            obs[base + 2] = sinf(en->facing);
            obs[base + 3] = cosf(en->facing);
            obs[base + 4] = en->alive ? en->hp / 100.0f : 0.0f;
            obs[base + 5] = (float)can_see;
            if (mem_tick >= 0) {
                int age   = g->tick - mem_tick;
                freshness = (age < sd->enemy_memory_ticks)
                                ? (sd->enemy_memory_ticks - age) / (float)sd->enemy_memory_ticks
                                : 0.0f;
            }
            obs[base + 6] = freshness;
        }

        obs[63] = (float)g->bomb_planted;
        obs[64] = g->bomb_planted ? g->bomb_x * sd->inv_x_range - sd->x_offset : -1.0f;
        obs[65] = g->bomb_planted ? g->bomb_y * sd->inv_y_range - sd->y_offset : -1.0f;
        obs[66] = g->bomb_planted ? g->bomb_ticks_left / (float)sd->bomb_timer : 0.0f;
        obs[67] = g->round_ticks_left / (float)sd->round_time;
        obs[68] = alive_t_frac;
        obs[69] = alive_ct_frac;
        obs[70] = (a->area_idx >= 0) ? (float)sd->bombsite_by_idx[a->area_idx] : 0.0f;

        /* obs[71]: plant state for bomb carrier */
        {
            float plant_state = 0.0f;
            if (a->team == 0 && a->has_bomb && !g->bomb_planted) {
                int at_site = (a->area_idx >= 0) ? sd->bombsite_by_idx[a->area_idx] : 0;
                if (!at_site) {
                    plant_state = 0.33f; /* has bomb, not at site */
                } else if (g->bomb_being_planted_by == i && g->bomb_plant_ticks > 0) {
                    /* actively planting with progress */
                    plant_state = 0.67f + 0.33f * ((float)g->bomb_plant_ticks /
                                                   (float)sd->bomb_plant_time);
                } else {
                    plant_state = 0.67f; /* at site, not yet planting or progress=0 */
                }
            }
            obs[71] = plant_state;
        }
    }
}
