#pragma once
#include "cs2_types.h"
#include "cs2_player.h"

static void clear_stats(StepStats* stats) {
    memset(stats, 0, sizeof(StepStats));
    stats->winner = -1;
}

static void spawn_team(GameState* g, StaticData* sd, uint32_t* rng, int team,
                       const int32_t* spawn_list, int n_spawns, int bomb_carrier) {
    int perm[TEAM_SIZE];
    for (int i = 0; i < TEAM_SIZE; i++)
        perm[i] = i;
    if (n_spawns >= TEAM_SIZE) {
        for (int i = TEAM_SIZE - 1; i > 0; i--) {
            int j   = (int)(xorshift32(rng) % (i + 1));
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }
    }

    for (int i = 0; i < TEAM_SIZE; i++) {
        int sidx;
        if (n_spawns >= TEAM_SIZE) {
            sidx = perm[i];
        } else {
            sidx = (int)(xorshift32(rng) % n_spawns);
        }
        int         area_idx = spawn_list[sidx];
        AgentState* a        = &g->agents[team == 0 ? i : TEAM_SIZE + i];

        memset(a, 0, sizeof(AgentState));
        a->x        = sd->centroid_xy[area_idx * 2];
        a->y        = sd->centroid_xy[area_idx * 2 + 1];
        a->area_idx = area_idx;

        init_agent(a, team, i, bomb_carrier, rng, sd);
    }
}

static void update_enemy_memory(Dust2Env* env, int8_t vis10[N_AGENTS][N_AGENTS]) {
    StaticData* sd = env->sd;
    GameState*  g  = &env->game;

    for (int i = 0; i < N_AGENTS; i++) {
        AgentState* a = &g->agents[i];
        int         en_start;

        if (!a->alive) {
            continue;
        }

        en_start = (a->team == 0) ? TEAM_SIZE : 0;
        for (int slot = 0; slot < TEAM_SIZE; slot++) {
            int         ej = en_start + slot;
            AgentState* en = &g->agents[ej];
            int         can_see;
            int         can_hear = 0;

            if (!en->alive) {
                if (a->enemy_mem_idx[slot] != INVALID_AREA_IDX) {
                    a->enemy_mem_tick[slot] = sd->stale_memory_tick;
                }
                continue;
            }

            can_see = vis10[i][ej];
            if (!can_see) {
                float rx      = a->x - en->x;
                float ry      = a->y - en->y;
                float dist_sq = rx * rx + ry * ry;
                if (en->is_moving && dist_sq <= sd->footstep_radius_sq) {
                    can_hear = 1;
                } else if (en->fired_this_tick && dist_sq <= sd->gunshot_radius_sq) {
                    can_hear = 1;
                }
            }

            if (can_see || can_hear) {
                a->enemy_mem_idx[slot]  = en->area_idx;
                a->enemy_mem_tick[slot] = g->tick;
            } else {
                int last_tick = a->enemy_mem_tick[slot];
                if (last_tick >= 0 && g->tick - last_tick >= sd->enemy_memory_ticks) {
                    a->enemy_mem_idx[slot]  = INVALID_AREA_IDX;
                    a->enemy_mem_tick[slot] = sd->stale_memory_tick;
                }
            }
        }
    }
}
