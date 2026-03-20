#include "dust2_env.h"
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

static inline void
count_action(int32_t* step_counts, int32_t* episode_counts, int value, int size) {
    if ((unsigned int)value < (unsigned int)size) {
        step_counts[value]++;
        episode_counts[value]++;
    }
}

static float _potential(Dust2Env* env, int team) {
    float       alive_t = 0.0f, alive_o = 0.0f;
    float       hp_t = 0.0f, hp_o = 0.0f;
    float       site_t = 0.0f, site_o = 0.0f;
    float       bomb_progress        = 0.0f;
    float       nav_approach         = 0.0f;
    int         bomb_carrier_area_id = INVALID_AREA_IDX;
    StaticData* sd                   = env->sd;

    for (int i = 0; i < N_AGENTS; i++) {
        AgentState* a       = &env->game.agents[i];
        int         area_id = INVALID_AREA_IDX;
        if (!a->alive) {
            continue;
        }

        if (a->area_idx >= 0 && a->area_idx < sd->N) {
            area_id = sd->area_ids[a->area_idx];
        }

        if (a->team == team) {
            alive_t += 1.0f;
            hp_t    += (float)a->hp;
            if (a->area_idx >= 0 && sd->bombsite_by_idx[a->area_idx]) {
                site_t += 1.0f;
            }
            /* Navigation shaping: reward every alive agent for being close to bombsite */
            if (area_id != INVALID_AREA_IDX && area_id <= sd->max_area_id) {
                float dist = sd->bombsite_dist[area_id];
                if (isfinite(dist)) {
                    float closeness = 1.0f - dist * sd->bombsite_dist_scale;
                    if (closeness < 0.0f)
                        closeness = 0.0f;
                    nav_approach += closeness;
                }
            }
        } else {
            alive_o += 1.0f;
            hp_o    += (float)a->hp;
            if (a->area_idx >= 0 && sd->bombsite_by_idx[a->area_idx]) {
                site_o += 1.0f;
            }
        }

        if (a->team == 0 && a->has_bomb) {
            bomb_carrier_area_id = area_id;
        }
    }

    if (!env->game.bomb_planted && bomb_carrier_area_id != INVALID_AREA_IDX &&
        bomb_carrier_area_id <= sd->max_area_id) {
        float dist = sd->bombsite_dist[bomb_carrier_area_id];
        if (isfinite(dist)) {
            float closeness = 1.0f - dist * sd->bombsite_dist_scale;
            if (closeness < 0.0f) {
                closeness = 0.0f;
            }
            bomb_progress = closeness * 0.3f;
            if (team == 1) {
                bomb_progress = -bomb_progress;
            }
        }
    }

    float nav_weight = (team == 1) ? 0.15f : 0.04f;
    return (alive_t - alive_o) * 0.3f + (hp_t - hp_o) / 500.0f + (site_t - site_o) * 0.2f +
           bomb_progress + nav_approach * nav_weight;
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

    for (int i = 0; i < N_AGENTS; i++) {
        AgentState* a = &g->agents[i];
        int         move_dir;
        int         shoot;
        int         use;
        int         last;
        int         gx;
        int         gy;
        int         target_idx;
        float       tx;
        float       ty;

        if (!a->alive) {
            continue;
        }

        move_dir = actions[i * ACTION_DIM + 0];
        shoot    = actions[i * ACTION_DIM + 1];
        use      = actions[i * ACTION_DIM + 2];
        last     = actions[i * ACTION_DIM + 3];

        count_action(ss->action_move, es->action_move, move_dir, 9);
        count_action(ss->action_shoot, es->action_shoot, shoot, 2);
        count_action(ss->action_use, es->action_use, use, 2);
        count_action(ss->action_last, es->action_last, last, 2);

        if (move_dir == 0) {
            continue;
        }

        if (move_dir < 0 || move_dir > 8 || a->area_idx < 0) {
            if (a->team == 0) {
                ss->blocked_moves_t++;
                es->blocked_moves_t++;
            } else {
                ss->blocked_moves_ct++;
                es->blocked_moves_ct++;
            }
            continue;
        }

        tx = a->x + sd->delta_x[move_dir];
        ty = a->y + sd->delta_y[move_dir];
        gx = (int)((tx - sd->grid_x_min) * sd->grid_inv_cell);
        gy = (int)((ty - sd->grid_y_min) * sd->grid_inv_cell);
        if (gx < 0 || gx >= sd->grid_w || gy < 0 || gy >= sd->grid_h) {
            if (a->team == 0) {
                ss->blocked_moves_t++;
                es->blocked_moves_t++;
            } else {
                ss->blocked_moves_ct++;
                es->blocked_moves_ct++;
            }
            continue;
        }

        target_idx = sd->raster_grid[gy * sd->grid_w + gx];
        if (target_idx < 0 || !sd->adjacency[a->area_idx * sd->N + target_idx]) {
            if (a->team == 0) {
                ss->blocked_moves_t++;
                es->blocked_moves_t++;
            } else {
                ss->blocked_moves_ct++;
                es->blocked_moves_ct++;
            }
            continue;
        }

        a->x        = tx;
        a->y        = ty;
        a->area_idx = target_idx;
        /* Smooth turn: clamp facing change to max_turn_speed radians/tick */
        {
            float target = sd->dir_facing[move_dir];
            float diff   = target - a->facing;
            /* Normalise diff to [-π, π] */
            while (diff > (float)M_PI)
                diff -= 2.0f * (float)M_PI;
            while (diff < -(float)M_PI)
                diff += 2.0f * (float)M_PI;
            if (fabsf(diff) <= sd->max_turn_speed) {
                a->facing = target;
            } else {
                a->facing += (diff > 0.0f ? 1.0f : -1.0f) * sd->max_turn_speed;
            }
        }
        a->is_moving = 1;
    }

    for (int i = 0; i < N_AGENTS; i++) {
        for (int j = 0; j < N_AGENTS; j++) {
            int ai      = g->agents[i].area_idx;
            int aj      = g->agents[j].area_idx;
            vis10[i][j] = (ai >= 0 && aj >= 0) ? sd->vis_matrix[ai * sd->N + aj] : 0;
        }
    }

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
        if (actions[i * ACTION_DIM + 1] == 0 || a->shoot_cd > 0) {
            continue;
        }

        a->shoot_cd        = sd->shoot_cooldown;
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
                if (n_kills < N_AGENTS) {
                    kills[n_kills][0] = i;
                    kills[n_kills][1] = (int)(best_enemy - g->agents);
                    n_kills++;
                }
            }
        }
    }

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

    {
        float alive_t_frac  = t_alive / (float)TEAM_SIZE;
        float alive_ct_frac = ct_alive / (float)TEAM_SIZE;

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
            obs[7] = (a->shoot_cd == 0) ? 1.0f : 1.0f - a->shoot_cd / (float)sd->shoot_cooldown;

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

    memset(env->rewards, 0, N_AGENTS * sizeof(float));

    /* Inaction cost: discourage agents from standing still during active play */
    if (!g->round_over) {
        for (int i = 0; i < N_AGENTS; i++) {
            if (g->agents[i].alive && actions[i * ACTION_DIM + 0] == 0) {
                env->rewards[i] -= 0.0005f;
            }
        }
    }

    if (g->round_over) {
        for (int i = 0; i < N_AGENTS; i++) {
            if (g->agents[i].alive) {
                env->rewards[i] += (g->winner == g->agents[i].team) ? 1.0f : -1.0f;
            }
        }
    }

    for (int k = 0; k < n_kills; k++) {
        env->rewards[kills[k][0]] += 0.3f;
        env->rewards[kills[k][1]] -= 0.1f;
        if (kills[k][0] < TEAM_SIZE) {
            ss->kills_t++;
            es->kills_t++;
        } else {
            ss->kills_ct++;
            es->kills_ct++;
        }
    }

    /* Bomb carrier subgoal rewards */
    for (int i = 0; i < TEAM_SIZE; i++) {
        if (bombsite_entry_bonus[i]) {
            env->rewards[i] += 0.3f; /* one-time bombsite entry bonus */
        }
        env->rewards[i] += plant_progress_reward[i]; /* per-tick plant progress */
        if (plant_interrupted[i]) {
            env->rewards[i] -= 0.1f; /* interrupted plant penalty */
        }
    }

    if (bomb_just_planted && bomb_planter_id >= 0) {
        env->rewards[bomb_planter_id] += 0.2f;
        env->rewards[bomb_planter_id] += 3.0f; /* plant completion bonus */
    }
    if (bomb_just_defused && bomb_defuser_id >= 0) {
        env->rewards[bomb_defuser_id] += 0.2f;
    }

    /* Small cost per shot fired — discourages infinite spray at walls */
    for (int i = 0; i < N_AGENTS; i++) {
        if (env->game.agents[i].alive && env->game.agents[i].fired_this_tick) {
            env->rewards[i] -= 0.005f;
        }
    }

    /* CT survival micro-reward: gives CTs incentive to stay alive long enough to learn */
    if (!g->round_over) {
        for (int i = TEAM_SIZE; i < N_AGENTS; i++) {
            if (g->agents[i].alive) {
                env->rewards[i] += 0.001f;
            }
        }
    }

    {
        float phi_after[2];
        phi_after[0] = _potential(env, 0);
        phi_after[1] = _potential(env, 1);
        for (int i = 0; i < N_AGENTS; i++) {
            env->rewards[i] +=
                sd->pbrs_gamma * phi_after[g->agents[i].team] - phi_before[g->agents[i].team];
        }
    }

    if (env->team_spirit > 0.0f) {
        for (int team = 0; team < 2; team++) {
            float sum = 0.0f;
            int   cnt = 0;
            for (int i = 0; i < N_AGENTS; i++) {
                if (g->agents[i].alive && g->agents[i].team == team) {
                    sum += env->rewards[i];
                    cnt++;
                }
            }
            if (cnt > 0) {
                float mean = sum / (float)cnt;
                float ts   = env->team_spirit;
                for (int i = 0; i < N_AGENTS; i++) {
                    if (g->agents[i].alive && g->agents[i].team == team) {
                        env->rewards[i] = (1.0f - ts) * env->rewards[i] + ts * mean;
                    }
                }
            }
        }
    }

    if (g->round_over) {
        ss->winner       = g->winner;
        ss->winner_t     = (g->winner == 0);
        ss->winner_ct    = (g->winner == 1);
        ss->alive_t_end  = t_alive;
        ss->alive_ct_end = ct_alive;
        ss->round_length = g->tick;

        es->winner       = g->winner;
        es->winner_t     = (g->winner == 0);
        es->winner_ct    = (g->winner == 1);
        es->alive_t_end  = t_alive;
        es->alive_ct_end = ct_alive;
        es->round_length = g->tick;
    }

    for (int i = 0; i < N_AGENTS; i++) {
        env->terminals[i]   = g->round_over;
        env->truncations[i] = 0;
    }
}

void env_close(Dust2Env* env) {
    (void)env;
}
