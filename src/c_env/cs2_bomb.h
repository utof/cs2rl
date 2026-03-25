#pragma once
#include "cs2_types.h"

static void process_bomb(Dust2Env*      env,
                         const int32_t* actions,
                         int8_t         bombsite_entry_bonus[TEAM_SIZE],
                         float          plant_progress_reward[TEAM_SIZE],
                         int8_t         plant_interrupted[TEAM_SIZE],
                         int*           bomb_just_planted,
                         int*           bomb_planter_id,
                         int*           bomb_just_defused,
                         int*           bomb_defuser_id,
                         StepStats*     ss,
                         StepStats*     es) {
    StaticData* sd = env->sd;
    GameState*  g  = &env->game;

    if (!g->round_over && g->bomb_being_defused_by != -1) {
        AgentState* def = &g->agents[g->bomb_being_defused_by];
        if (!def->alive || def->area_idx != g->bomb_area_idx ||
            actions[g->bomb_being_defused_by * ACTION_DIM + 5] == 0) {
            g->bomb_being_defused_by = -1;
            g->bomb_defuse_ticks     = 0;
        }
    }

    if (!g->round_over) {
        for (int i = 0; i < N_AGENTS; i++) {
            AgentState* a = &g->agents[i];
            if (!a->alive || actions[i * ACTION_DIM + 5] == 0) {
                continue;
            }

            if (a->team == 0 && a->has_bomb && !g->bomb_planted) {
                if (a->area_idx >= 0 && sd->bombsite_by_idx[a->area_idx]) {
                    /* One-time bombsite entry bonus */
                    if (!g->bombsite_entered[i]) {
                        g->bombsite_entered[i]  = 1;
                        bombsite_entry_bonus[i] = 1;
                    }
                    if (g->bomb_being_planted_by == -1) {
                        g->bomb_being_planted_by = i;
                        g->bomb_plant_ticks      = 0;
                    }
                    if (g->bomb_being_planted_by == i) {
                        g->bomb_plant_ticks++;
                        plant_progress_reward[i] =
                            sd->reward_plant_progress_scale; /* per-tick plant progress reward */
                        if (g->bomb_plant_ticks >= sd->bomb_plant_time) {
                            g->bomb_planted          = 1;
                            g->bomb_area_idx         = a->area_idx;
                            g->bomb_x                = a->x;
                            g->bomb_y                = a->y;
                            g->bomb_z                = a->z;
                            g->bomb_ticks_left       = sd->bomb_timer;
                            g->bomb_being_planted_by = -1;
                            a->has_bomb              = 0;
                            *bomb_just_planted       = 1;
                            *bomb_planter_id         = i;
                            ss->bomb_planted         = 1;
                            es->bomb_planted++;
                        }
                    }
                } else if (g->bomb_being_planted_by == i) {
                    if (g->bomb_plant_ticks > 0) {
                        plant_interrupted[i] =
                            1; /* interrupted plant penalty applied post-memset */
                    }
                    g->bomb_being_planted_by = -1;
                    g->bomb_plant_ticks      = 0;
                }
            } else if (a->team == 1 && g->bomb_planted && a->area_idx == g->bomb_area_idx) {
                int defuse_time = a->has_kit ? sd->bomb_defuse_kit : sd->bomb_defuse_time;
                if (g->bomb_being_defused_by == -1) {
                    g->bomb_being_defused_by = i;
                    g->bomb_defuse_ticks     = 0;
                }
                if (g->bomb_being_defused_by == i) {
                    g->bomb_defuse_ticks++;
                    if (g->bomb_defuse_ticks >= defuse_time) {
                        g->round_over      = 1;
                        g->winner          = 1;
                        *bomb_just_defused = 1;
                        *bomb_defuser_id   = i;
                        ss->bomb_defused   = 1;
                        es->bomb_defused++;
                    }
                }
            }
        }
    }

    /* Dropped-bomb pickup: alive T agents auto-pick up if within 32 units */
    if (!g->bomb_planted && g->bomb_is_dropped && !g->round_over) {
        float best_dist = 32.0f * 32.0f; /* compare dist_sq to radius_sq */
        int   best_t    = -1;
        for (int i = 0; i < TEAM_SIZE; i++) {
            AgentState* a = &g->agents[i];
            if (!a->alive)
                continue;
            float dx = a->x - g->bomb_x;
            float dy = a->y - g->bomb_y;
            float d  = dx * dx + dy * dy;
            if (d <= best_dist) {
                best_dist = d;
                best_t    = i;
            }
        }
        if (best_t >= 0) {
            g->agents[best_t].has_bomb = 1;
            g->bomb_carrier_id         = best_t;
            g->bomb_is_dropped         = 0;
        }
    }

    if (g->bomb_planted && !g->round_over) {
        g->bomb_ticks_left--;
        if (g->bomb_ticks_left <= 0) {
            g->round_over = 1;
            g->winner     = 0;
        }
    }

    if (g->round_ticks_left <= 0 && !g->round_over && !g->bomb_planted) {
        g->round_over = 1;
        g->winner     = -1;
        ss->timed_out = 1;
        es->timed_out = 1;
    }
}
