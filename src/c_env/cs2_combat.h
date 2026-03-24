#pragma once
#include "cs2_types.h"

static void build_vis_matrix(
    GameState* g, StaticData* sd,
    int8_t vis10[N_AGENTS][N_AGENTS])
{
    for (int i = 0; i < N_AGENTS; i++) {
        for (int j = 0; j < N_AGENTS; j++) {
            int ai      = g->agents[i].area_idx;
            int aj      = g->agents[j].area_idx;
            vis10[i][j] = (ai >= 0 && aj >= 0) ? sd->vis_matrix[ai * sd->N + aj] : 0;
        }
    }
}

static void process_combat(
    Dust2Env*  env,
    const int32_t* actions,
    int8_t     vis10[N_AGENTS][N_AGENTS],
    int        kills[N_AGENTS][2],
    int*       n_kills,
    StepStats* ss,
    StepStats* es)
{
    StaticData* sd = env->sd;
    GameState*  g  = &env->game;

    (void)ss;
    (void)es;

    for (int i = 0; i < N_AGENTS; i++) {
        AgentState* a = &g->agents[i];
        int         en_start;
        AgentState* best_enemy = NULL;
        float       dx;
        float       dy;
        float       best_dist;

        if (!a->alive) {
            continue;
        }
        if (actions[i * ACTION_DIM + 1] == 0 || a->fire_cd > 0) {
            continue;
        }

        a->fire_cd         = 10;
        a->fired_this_tick = 1;
        dx                 = cosf(a->facing);
        dy                 = sinf(a->facing);
        en_start           = (a->team == 0) ? TEAM_SIZE : 0;
        best_dist          = sd->laser_range;

        for (int ej = en_start; ej < en_start + TEAM_SIZE; ej++) {
            AgentState* en = &g->agents[ej];
            float       rx;
            float       ry;
            float       dist_sq;
            float       dist;
            float       dot;

            if (!en->alive || !vis10[i][ej]) {
                continue;
            }

            rx      = en->x - a->x;
            ry      = en->y - a->y;
            dist_sq = rx * rx + ry * ry;
            if (dist_sq > sd->laser_range_sq || dist_sq == 0.0f) {
                continue;
            }

            dist = sqrtf(dist_sq);
            dot  = (rx / dist) * dx + (ry / dist) * dy;
            if (dot < 0.7f) {
                continue;
            }

            if (dist < best_dist) {
                best_dist  = dist;
                best_enemy = en;
            }
        }

        if (best_enemy != NULL) {
            best_enemy->hp -= sd->laser_damage;
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
}
