#include "cs2_env.h"
#include "cs2_combat.h"
#include "cs2_observations.h"
#include "cs2_movement.h"
#include "cs2_rewards.h"
#include <stdlib.h>

static uint32_t xorshift32(uint32_t* state) {
    uint32_t x  = *state;
    x          ^= x << 13;
    x          ^= x >> 17;
    x          ^= x << 5;
    return (*state = x);
}

static void clear_stats(StepStats* stats) {
    memset(stats, 0, sizeof(StepStats));
    stats->winner = -1;
}

void env_init(Dust2Env* env, StaticData* sd, uint32_t seed, float team_spirit) {
    memset(env, 0, sizeof(Dust2Env));
    env->sd          = sd;
    env->rng         = seed ? seed : 1;
    env->team_spirit = team_spirit;
    clear_stats(&env->step_stats);
    clear_stats(&env->episode_stats);
}

void env_reset(Dust2Env* env) {
    StaticData* sd = env->sd;
    GameState*  g  = &env->game;

    memset(g, 0, sizeof(GameState));
    memset(env->observations, 0, sizeof(env->observations));
    memset(env->rewards, 0, sizeof(env->rewards));
    memset(env->terminals, 0, sizeof(env->terminals));
    memset(env->truncations, 0, sizeof(env->truncations));
    clear_stats(&env->step_stats);
    clear_stats(&env->episode_stats);

    g->round_ticks_left      = sd->round_time;
    g->winner                = -1;
    g->bomb_area_idx         = INVALID_AREA_IDX;
    g->bomb_being_planted_by = -1;
    g->bomb_being_defused_by = -1;

    int bomb_carrier = (int)(xorshift32(&env->rng) % TEAM_SIZE);

    /* Build a shuffled index list for T-spawns so each agent gets a unique area
       when n_t_spawns >= TEAM_SIZE. */
    int t_perm[TEAM_SIZE];
    for (int i = 0; i < TEAM_SIZE; i++)
        t_perm[i] = i;
    if (sd->n_t_spawns >= TEAM_SIZE) {
        for (int i = TEAM_SIZE - 1; i > 0; i--) {
            int j     = (int)(xorshift32(&env->rng) % (i + 1));
            int tmp   = t_perm[i];
            t_perm[i] = t_perm[j];
            t_perm[j] = tmp;
        }
    }

    for (int i = 0; i < TEAM_SIZE; i++) {
        int sidx;
        if (sd->n_t_spawns >= TEAM_SIZE) {
            sidx = t_perm[i];
        } else {
            sidx = (int)(xorshift32(&env->rng) % sd->n_t_spawns);
        }
        int         area_idx = sd->t_spawns[sidx];
        AgentState* a        = &g->agents[i];

        memset(a, 0, sizeof(AgentState));
        a->x        = sd->centroid_xy[area_idx * 2];
        a->y        = sd->centroid_xy[area_idx * 2 + 1];
        a->area_idx = area_idx;
        a->facing   = sd->dir_facing[3];
        a->hp       = 100;
        a->alive    = 1;
        a->has_bomb = (i == bomb_carrier) ? 1 : 0;
        a->team     = 0;

        for (int s = 0; s < TEAM_SIZE; s++) {
            a->enemy_mem_idx[s]  = INVALID_AREA_IDX;
            a->enemy_mem_tick[s] = sd->stale_memory_tick;
        }
    }

    /* Build a shuffled index list for CT-spawns similarly. */
    int ct_perm[TEAM_SIZE];
    for (int i = 0; i < TEAM_SIZE; i++)
        ct_perm[i] = i;
    if (sd->n_ct_spawns >= TEAM_SIZE) {
        for (int i = TEAM_SIZE - 1; i > 0; i--) {
            int j      = (int)(xorshift32(&env->rng) % (i + 1));
            int tmp    = ct_perm[i];
            ct_perm[i] = ct_perm[j];
            ct_perm[j] = tmp;
        }
    }

    for (int i = 0; i < TEAM_SIZE; i++) {
        int sidx;
        if (sd->n_ct_spawns >= TEAM_SIZE) {
            sidx = ct_perm[i];
        } else {
            sidx = (int)(xorshift32(&env->rng) % sd->n_ct_spawns);
        }
        int         area_idx = sd->ct_spawns[sidx];
        AgentState* a        = &g->agents[TEAM_SIZE + i];

        memset(a, 0, sizeof(AgentState));
        a->x        = sd->centroid_xy[area_idx * 2];
        a->y        = sd->centroid_xy[area_idx * 2 + 1];
        a->area_idx = area_idx;
        a->facing   = sd->dir_facing[7];
        a->hp       = 100;
        a->alive    = 1;
        a->has_kit  = (xorshift32(&env->rng) & 1U) ? 1 : 0;
        a->team     = 1;

        for (int s = 0; s < TEAM_SIZE; s++) {
            a->enemy_mem_idx[s]  = INVALID_AREA_IDX;
            a->enemy_mem_tick[s] = sd->stale_memory_tick;
        }
    }

    g->bomb_carrier_id = bomb_carrier;
}

void env_step(Dust2Env* env, const int32_t* actions) {
    StaticData* sd = env->sd;
    GameState*  g  = &env->game;
    StepStats*  ss = &env->step_stats;
    StepStats*  es = &env->episode_stats;
    float       phi_before[2];
    int8_t      vis10[N_AGENTS][N_AGENTS];
    int         kills[N_AGENTS][2];
    int         n_kills                  = 0;
    int         bomb_just_planted        = 0;
    int         bomb_planter_id          = -1;
    int         bomb_just_defused        = 0;
    int         bomb_defuser_id          = -1;
    int         t_alive                  = 0;
    int         ct_alive                 = 0;
    int8_t      bombsite_entry_bonus[5]  = {0};    /* 1 if agent earned entry bonus this tick */
    float       plant_progress_reward[5] = {0.0f}; /* per-tick plant progress per T-agent */
    int8_t      plant_interrupted[5]     = {0};    /* 1 if plant was interrupted with progress */

    phi_before[0] = _potential(env, 0);
    phi_before[1] = _potential(env, 1);
    clear_stats(ss);

    g->tick++;
    g->round_ticks_left--;

    for (int i = 0; i < N_AGENTS; i++) {
        AgentState* a = &g->agents[i];
        if (a->shoot_cd > 0) {
            a->shoot_cd--;
        }
        a->is_moving       = 0;
        a->fired_this_tick = 0;
    }

    process_movement(env, actions, ss, es);

    build_vis_matrix(g, sd, vis10);
    process_combat(env, actions, vis10, kills, &n_kills, ss, es);

    for (int i = 0; i < N_AGENTS; i++) {
        if (!g->agents[i].alive) {
            continue;
        }
        if (g->agents[i].team == 0)
            t_alive++;
        else
            ct_alive++;
    }
    if (!t_alive && !g->round_over) {
        g->round_over = 1;
        g->winner     = 1;
    } else if (!ct_alive && !g->round_over) {
        g->round_over = 1;
        g->winner     = 0;
    }

    if (!g->round_over && g->bomb_being_defused_by != -1) {
        AgentState* def = &g->agents[g->bomb_being_defused_by];
        if (!def->alive || def->area_idx != g->bomb_area_idx ||
            actions[g->bomb_being_defused_by * ACTION_DIM + 2] == 0) {
            g->bomb_being_defused_by = -1;
            g->bomb_defuse_ticks     = 0;
        }
    }

    if (!g->round_over) {
        for (int i = 0; i < N_AGENTS; i++) {
            AgentState* a = &g->agents[i];
            if (!a->alive || actions[i * ACTION_DIM + 2] == 0) {
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
                        plant_progress_reward[i] = 0.05f; /* per-tick plant progress reward */
                        if (g->bomb_plant_ticks >= sd->bomb_plant_time) {
                            g->bomb_planted          = 1;
                            g->bomb_area_idx         = a->area_idx;
                            g->bomb_x                = a->x;
                            g->bomb_y                = a->y;
                            g->bomb_z                = a->z;
                            g->bomb_ticks_left       = sd->bomb_timer;
                            g->bomb_being_planted_by = -1;
                            a->has_bomb              = 0;
                            bomb_just_planted        = 1;
                            bomb_planter_id          = i;
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
                        g->round_over     = 1;
                        g->winner         = 1;
                        bomb_just_defused = 1;
                        bomb_defuser_id   = i;
                        ss->bomb_defused  = 1;
                        es->bomb_defused++;
                    }
                }
            }
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

    compute_observations(env, t_alive, ct_alive, vis10);

    memset(env->rewards, 0, N_AGENTS * sizeof(float));

    /* Inaction cost: discourage agents from standing still during active play */
    if (!g->round_over) {
        for (int i = 0; i < N_AGENTS; i++) {
            if (g->agents[i].alive && actions[i * ACTION_DIM + 0] == 0) {
                env->rewards[i] -= 0.0005f;
            }
        }
    }

    compute_rewards(env, t_alive, ct_alive, phi_before, kills, n_kills, bombsite_entry_bonus,
                    plant_progress_reward, plant_interrupted, bomb_just_planted, bomb_planter_id,
                    bomb_just_defused, bomb_defuser_id);
}

void env_close(Dust2Env* env) {
    (void)env;
}
