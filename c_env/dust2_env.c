#include "dust2_env.h"
#include <stdlib.h>
#include <stdio.h>

static uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return (*state = x);
}

static float _potential(Dust2Env* env, int team) {
    float alive_t=0, alive_o=0, hp_t=0, hp_o=0, site_t=0, site_o=0;
    StaticData* sd = env->sd;
    for (int i = 0; i < N_AGENTS; i++) {
        AgentState* a = &env->game.agents[i];
        if (!a->alive) continue;
        if (a->team == team) {
            alive_t += 1.0f;
            hp_t    += (float)a->hp;
            if (a->area_idx >= 0) site_t += (float)sd->bombsite_by_idx[a->area_idx];
        } else {
            alive_o += 1.0f;
            hp_o    += (float)a->hp;
            if (a->area_idx >= 0) site_o += (float)sd->bombsite_by_idx[a->area_idx];
        }
    }
    return (alive_t - alive_o) * 0.3f
         + (hp_t - hp_o)       / 500.0f
         + (site_t - site_o)   * 0.2f;
}

void env_init(Dust2Env* env, StaticData* sd, uint32_t seed, float team_spirit) {
    memset(env, 0, sizeof(Dust2Env));
    env->sd = sd;
    env->rng = seed ? seed : 1;
    env->team_spirit = team_spirit;
    /* Movement deltas — direction vectors scaled by MOVE_SPEED * DT */
    /* move_dir: 0=noop 1=N 2=NE 3=E 4=SE 5=S 6=SW 7=W 8=NW */
    float spd = MOVE_SPEED * DT;
    float d = spd * 0.7071067811865476f;
    env->delta_x[0]=0;   env->delta_y[0]=0;
    env->delta_x[1]=0;   env->delta_y[1]=spd;
    env->delta_x[2]=0;   env->delta_y[2]=0;   /* placeholder — overwritten below */
    env->delta_x[3]=spd; env->delta_y[3]=0;
    env->delta_x[4]=0;   env->delta_y[4]=0;   /* placeholder — overwritten below */
    env->delta_x[5]=0;   env->delta_y[5]=-spd;
    env->delta_x[6]=0;   env->delta_y[6]=0;   /* placeholder — overwritten below */
    env->delta_x[7]=-spd;env->delta_y[7]=0;
    env->delta_x[8]=0;   env->delta_y[8]=0;   /* placeholder — overwritten below */
    /* Diagonals at pre-normalised magnitude */
    env->delta_x[2]= d; env->delta_y[2]= d;
    env->delta_x[4]= d; env->delta_y[4]=-d;
    env->delta_x[6]=-d; env->delta_y[6]=-d;
    env->delta_x[8]=-d; env->delta_y[8]= d;
    /* Facing angles per move_dir (radians) */
    env->dir_facing[0]=0;
    env->dir_facing[1]=1.5707963f;   /* π/2  N  */
    env->dir_facing[2]=0.7853982f;   /* π/4  NE */
    env->dir_facing[3]=0.0f;         /* 0    E  */
    env->dir_facing[4]=-0.7853982f;  /* -π/4 SE */
    env->dir_facing[5]=-1.5707963f;  /* -π/2 S  */
    env->dir_facing[6]=-2.3561945f;  /* -3π/4 SW */
    env->dir_facing[7]=3.1415927f;   /* π    W  */
    env->dir_facing[8]=2.3561945f;   /* 3π/4 NW */
}

void env_reset(Dust2Env* env) {
    StaticData* sd = env->sd;
    GameState*  g  = &env->game;

    memset(g, 0, sizeof(GameState));
    g->round_ticks_left      = ROUND_TIME;
    g->winner                = -1;
    g->bomb_area_idx         = INVALID_AREA_IDX;
    g->bomb_being_planted_by = -1;
    g->bomb_being_defused_by = -1;

    int bomb_carrier = xorshift32(&env->rng) % TEAM_SIZE;

    /* Spawn T agents (indices 0-4) */
    for (int i = 0; i < TEAM_SIZE; i++) {
        int sidx     = xorshift32(&env->rng) % sd->n_t_spawns;
        int area_idx = sd->t_spawns[sidx];
        AgentState* a = &g->agents[i];
        memset(a, 0, sizeof(AgentState));
        a->x        = sd->centroid_xy[area_idx * 2];
        a->y        = sd->centroid_xy[area_idx * 2 + 1];
        a->area_idx = area_idx;
        a->facing   = 0.0f;
        a->hp       = 100;
        a->alive    = 1;
        a->has_bomb = (i == bomb_carrier) ? 1 : 0;
        a->team     = 0;
        for (int s = 0; s < TEAM_SIZE; s++) {
            a->enemy_mem_idx[s]  = INVALID_AREA_IDX;
            a->enemy_mem_tick[s] = STALE_MEMORY_TICK;
        }
    }

    /* Spawn CT agents (indices 5-9) */
    for (int i = 0; i < TEAM_SIZE; i++) {
        int sidx     = xorshift32(&env->rng) % sd->n_ct_spawns;
        int area_idx = sd->ct_spawns[sidx];
        AgentState* a = &g->agents[TEAM_SIZE + i];
        memset(a, 0, sizeof(AgentState));
        a->x        = sd->centroid_xy[area_idx * 2];
        a->y        = sd->centroid_xy[area_idx * 2 + 1];
        a->area_idx = area_idx;
        a->facing   = 3.14159265f;
        a->hp       = 100;
        a->alive    = 1;
        a->has_kit  = (xorshift32(&env->rng) & 1) ? 1 : 0;
        a->team     = 1;
        for (int s = 0; s < TEAM_SIZE; s++) {
            a->enemy_mem_idx[s]  = INVALID_AREA_IDX;
            a->enemy_mem_tick[s] = STALE_MEMORY_TICK;
        }
    }

    g->bomb_carrier_id = bomb_carrier;
    memset(env->observations, 0, sizeof(env->observations));
    memset(env->rewards,      0, sizeof(env->rewards));
    memset(env->terminals,    0, sizeof(env->terminals));
    memset(env->truncations,  0, sizeof(env->truncations));
}

void env_step(Dust2Env* env) {
    StaticData* sd = env->sd;
    GameState*  g  = &env->game;

    /* Step 1: capture phi_before */
    float phi_before[2];
    phi_before[0] = _potential(env, 0);
    phi_before[1] = _potential(env, 1);

    /* Step 2: advance tick */
    g->tick++;
    g->round_ticks_left--;

    /* Step 3: decrement cooldowns, clear per-tick flags */
    for (int i = 0; i < N_AGENTS; i++) {
        AgentState* a = &g->agents[i];
        if (a->shoot_cd > 0) a->shoot_cd--;
        a->is_moving       = 0;
        a->fired_this_tick = 0;
    }

    /* Step 4: movement */
    for (int i = 0; i < N_AGENTS; i++) {
        AgentState* a = &g->agents[i];
        if (!a->alive) continue;
        int move_dir = env->actions[i * 4 + 0];
        if (move_dir == 0) continue;
        /* bounds check — prevent UB on bad action values */
        if (move_dir < 0 || move_dir > 8) continue;
        /* area_idx guard — prevent negative-index UB in adjacency lookup */
        if (a->area_idx < 0) continue;

        float tx = a->x + env->delta_x[move_dir];
        float ty = a->y + env->delta_y[move_dir];

        int gx = (int)((tx - sd->grid_x_min) * sd->grid_inv_cell);
        int gy = (int)((ty - sd->grid_y_min) * sd->grid_inv_cell);
        if (gx < 0 || gx >= sd->grid_w || gy < 0 || gy >= sd->grid_h) continue;

        int target_idx = sd->raster_grid[gy * sd->grid_w + gx];
        if (target_idx < 0) continue;  /* off mesh */

        /* adjacency check */
        if (!sd->adjacency[a->area_idx * sd->N + target_idx]) continue;

        a->x        = tx;
        a->y        = ty;
        a->area_idx = target_idx;
        a->facing   = env->dir_facing[move_dir];
        a->is_moving = 1;
    }

    /* Step 5: build vis10 — local 10x10 bool matrix */
    int8_t vis10[N_AGENTS][N_AGENTS];
    for (int i = 0; i < N_AGENTS; i++) {
        for (int j = 0; j < N_AGENTS; j++) {
            int ai = g->agents[i].area_idx;
            int aj = g->agents[j].area_idx;
            vis10[i][j] = (ai >= 0 && aj >= 0)
                ? sd->vis_matrix[ai * sd->N + aj]
                : 0;
        }
    }

    /* Step 6: shooting */
    int kills[N_AGENTS][2];
    int n_kills = 0;

    for (int i = 0; i < N_AGENTS; i++) {
        AgentState* a = &g->agents[i];
        if (!a->alive) continue;
        if (env->actions[i * 4 + 1] == 0) continue;
        if (a->shoot_cd > 0) continue;

        a->shoot_cd        = SHOOT_COOLDOWN;
        a->fired_this_tick = 1;

        float dx = cosf(a->facing);
        float dy = sinf(a->facing);

        int en_start = (a->team == 0) ? TEAM_SIZE : 0;
        AgentState* best_enemy = NULL;
        float best_dist = (float)LASER_RANGE;

        for (int ej = en_start; ej < en_start + TEAM_SIZE; ej++) {
            AgentState* en = &g->agents[ej];
            if (!en->alive)    continue;
            if (!vis10[i][ej]) continue;

            float rx = en->x - a->x;
            float ry = en->y - a->y;
            float dist_sq = rx * rx + ry * ry;
            if (dist_sq > (float)LASER_RANGE_SQ || dist_sq == 0.0f) continue;

            float dist = sqrtf(dist_sq);
            float dot  = (rx / dist) * dx + (ry / dist) * dy;
            if (dot < 0.7f) continue;

            if (dist < best_dist) { best_dist = dist; best_enemy = en; }
        }

        if (best_enemy != NULL) {
            best_enemy->hp -= LASER_DAMAGE;
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

    /* Step 7: round-end detection */
    int t_alive = 0, ct_alive = 0;
    for (int i = 0; i < N_AGENTS; i++) {
        if (!g->agents[i].alive) continue;
        if (g->agents[i].team == 0) t_alive++; else ct_alive++;
    }
    if (!t_alive  && !g->round_over) { g->round_over = 1; g->winner = 1; }
    if (!ct_alive && !g->round_over) { g->round_over = 1; g->winner = 0; }
    if (g->round_ticks_left <= 0 && !g->round_over) { g->round_over = 1; g->winner = 1; }

    /* Step 8: bomb timer */
    int bomb_just_planted  = 0;
    int bomb_planter_id    = -1;
    int bomb_just_defused  = 0;
    int bomb_defuser_id    = -1;

    if (g->bomb_planted && !g->round_over) {
        g->bomb_ticks_left--;
        if (g->bomb_ticks_left <= 0) { g->round_over = 1; g->winner = 0; }
    }

    /* Step 9: bomb plant/defuse — skip entirely if round already decided */
    if (!g->round_over) {
        /* Clear defuse if defuser stopped/moved/died */
        if (g->bomb_being_defused_by != -1) {
            AgentState* def = &g->agents[g->bomb_being_defused_by];
            if (!def->alive
                || def->area_idx != g->bomb_area_idx
                || env->actions[g->bomb_being_defused_by * 4 + 2] == 0) {
                g->bomb_being_defused_by = -1;
                g->bomb_defuse_ticks     = 0;
            }
        }

        for (int i = 0; i < N_AGENTS; i++) {
            AgentState* a = &g->agents[i];
            if (!a->alive) continue;
            if (env->actions[i * 4 + 2] == 0) continue;

            if (a->team == 0 && a->has_bomb && !g->bomb_planted) {
                /* T planting */
                if (a->area_idx >= 0 && sd->bombsite_by_idx[a->area_idx]) {
                    if (g->bomb_being_planted_by == -1) {
                        g->bomb_being_planted_by = i;
                        g->bomb_plant_ticks      = 0;
                    }
                    if (g->bomb_being_planted_by == i) {
                        g->bomb_plant_ticks++;
                        if (g->bomb_plant_ticks >= BOMB_PLANT_TIME) {
                            g->bomb_planted          = 1;
                            g->bomb_area_idx         = a->area_idx;
                            g->bomb_x                = a->x;
                            g->bomb_y                = a->y;
                            g->bomb_z                = a->z;
                            g->bomb_ticks_left       = BOMB_TIMER;
                            g->bomb_being_planted_by = -1;
                            a->has_bomb              = 0;
                            bomb_just_planted        = 1;
                            bomb_planter_id          = i;
                        }
                    }
                } else {
                    if (g->bomb_being_planted_by == i) {
                        g->bomb_being_planted_by = -1;
                        g->bomb_plant_ticks      = 0;
                    }
                }
            } else if (a->team == 1 && g->bomb_planted) {
                /* CT defusing */
                if (a->area_idx == g->bomb_area_idx) {
                    int defuse_time = a->has_kit ? BOMB_DEFUSE_KIT : BOMB_DEFUSE_TIME;
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
                        }
                    }
                }
            }
        }
    } /* end !round_over guard for Step 9 */

    /* Step 10: enemy memory update (vision + sound) */
    for (int i = 0; i < N_AGENTS; i++) {
        AgentState* a = &g->agents[i];
        if (!a->alive) continue;

        int en_start = (a->team == 0) ? TEAM_SIZE : 0;
        for (int slot = 0; slot < TEAM_SIZE; slot++) {
            int ej = en_start + slot;
            AgentState* en = &g->agents[ej];

            if (!en->alive) {
                if (a->enemy_mem_idx[slot] != INVALID_AREA_IDX)
                    a->enemy_mem_tick[slot] = STALE_MEMORY_TICK;
                continue;
            }

            int can_see  = vis10[i][ej];
            int can_hear = 0;
            if (!can_see) {
                float rx = a->x - en->x;
                float ry = a->y - en->y;
                float dist_sq = rx * rx + ry * ry;
                if (en->is_moving && dist_sq <= (float)(FOOTSTEP_RADIUS * FOOTSTEP_RADIUS))
                    can_hear = 1;
                else if (en->fired_this_tick && dist_sq <= (float)(GUNSHOT_RADIUS * GUNSHOT_RADIUS))
                    can_hear = 1;
            }

            if (can_see || can_hear) {
                a->enemy_mem_idx[slot]  = en->area_idx;
                a->enemy_mem_tick[slot] = g->tick;
            } else {
                int last_tick = a->enemy_mem_tick[slot];
                if (last_tick >= 0 && g->tick - last_tick >= ENEMY_MEMORY_TICKS) {
                    a->enemy_mem_idx[slot]  = INVALID_AREA_IDX;
                    a->enemy_mem_tick[slot] = STALE_MEMORY_TICK;
                }
            }
        }
    }

    /* Step 11: compute observations for all 10 agents */
    int alive_t_count = 0, alive_ct_count = 0;
    for (int i = 0; i < N_AGENTS; i++) {
        if (!g->agents[i].alive) continue;
        if (g->agents[i].team == 0) alive_t_count++; else alive_ct_count++;
    }
    float alive_t_frac  = alive_t_count  / (float)TEAM_SIZE;
    float alive_ct_frac = alive_ct_count / (float)TEAM_SIZE;

    for (int i = 0; i < N_AGENTS; i++) {
        float* obs = &env->observations[i * OBS_DIM];
        memset(obs, 0, OBS_DIM * sizeof(float));

        AgentState* a = &g->agents[i];

        /* Self features [0-7] */
        obs[0] = (float)a->team;
        obs[1] = a->x * sd->inv_x_range - sd->x_offset;
        obs[2] = a->y * sd->inv_y_range - sd->y_offset;
        obs[3] = sinf(a->facing);
        obs[4] = cosf(a->facing);
        obs[5] = a->hp / 100.0f;
        obs[6] = (float)(a->team == 0 ? a->has_bomb : a->has_kit);
        obs[7] = (a->shoot_cd == 0) ? 1.0f : 1.0f - a->shoot_cd / (float)SHOOT_COOLDOWN;

        /* Teammate features [8-27] — 4 teammates × 5 floats */
        int tm_start = (a->team == 0) ? 0 : TEAM_SIZE;
        int tm_count = 0;
        for (int j = tm_start; j < tm_start + TEAM_SIZE; j++) {
            if (j == i) continue;
            AgentState* tm = &g->agents[j];
            int base = 8 + tm_count * 5;
            obs[base + 0] = tm->x * sd->inv_x_range - sd->x_offset;
            obs[base + 1] = tm->y * sd->inv_y_range - sd->y_offset;
            obs[base + 2] = sinf(tm->facing);
            obs[base + 3] = cosf(tm->facing);
            obs[base + 4] = tm->alive ? tm->hp / 100.0f : 0.0f;
            tm_count++;
            if (tm_count == TEAM_SIZE - 1) break;
        }

        /* Enemy features [28-62] — 5 enemies × 7 floats */
        int en_start2 = (a->team == 0) ? TEAM_SIZE : 0;
        for (int slot = 0; slot < TEAM_SIZE; slot++) {
            AgentState* en = &g->agents[en_start2 + slot];
            int base = 28 + slot * 7;
            int mem_idx  = a->enemy_mem_idx[slot];
            int mem_tick = a->enemy_mem_tick[slot];
            int can_see  = en->alive ? vis10[i][en_start2 + slot] : 0;

            if (mem_idx == INVALID_AREA_IDX && !can_see) continue;

            if (mem_idx != INVALID_AREA_IDX) {
                obs[base + 0] = sd->centroid_xy[mem_idx * 2]     * sd->inv_x_range - sd->x_offset;
                obs[base + 1] = sd->centroid_xy[mem_idx * 2 + 1] * sd->inv_y_range - sd->y_offset;
            }
            obs[base + 2] = sinf(en->facing);
            obs[base + 3] = cosf(en->facing);
            obs[base + 4] = en->alive ? en->hp / 100.0f : 0.0f;
            obs[base + 5] = (float)can_see;
            float freshness = 0.0f;
            if (mem_tick >= 0) {
                int age = g->tick - mem_tick;
                freshness = (age < ENEMY_MEMORY_TICKS)
                    ? (ENEMY_MEMORY_TICKS - age) / (float)ENEMY_MEMORY_TICKS
                    : 0.0f;
            }
            obs[base + 6] = freshness;
        }

        /* Global features [63-70] */
        obs[63] = (float)g->bomb_planted;
        obs[64] = g->bomb_planted ? g->bomb_x * sd->inv_x_range - sd->x_offset : -1.0f;
        obs[65] = g->bomb_planted ? g->bomb_y * sd->inv_y_range - sd->y_offset : -1.0f;
        obs[66] = g->bomb_planted ? g->bomb_ticks_left / (float)BOMB_TIMER : 0.0f;
        obs[67] = g->round_ticks_left / (float)ROUND_TIME;
        obs[68] = alive_t_frac;
        obs[69] = alive_ct_frac;
        obs[70] = (a->area_idx >= 0) ? (float)sd->bombsite_by_idx[a->area_idx] : 0.0f;
    }

    /* Step 12: rewards */
    memset(env->rewards, 0, N_AGENTS * sizeof(float));

    /* Round-end */
    if (g->round_over) {
        for (int i = 0; i < N_AGENTS; i++) {
            if (g->agents[i].alive)
                env->rewards[i] += (g->winner == g->agents[i].team) ? 1.0f : -1.0f;
        }
    }

    /* Kill/death */
    for (int k = 0; k < n_kills; k++) {
        env->rewards[kills[k][0]] += 0.3f;
        env->rewards[kills[k][1]] -= 0.1f;
    }

    /* Bomb events */
    if (bomb_just_planted && bomb_planter_id >= 0)
        env->rewards[bomb_planter_id] += 0.2f;
    if (bomb_just_defused && bomb_defuser_id >= 0)
        env->rewards[bomb_defuser_id] += 0.2f;

    /* Survival bonus */
    if (!g->bomb_planted && g->round_ticks_left > ROUND_TIME * 0.5f) {
        for (int i = 0; i < N_AGENTS; i++)
            if (g->agents[i].alive) env->rewards[i] += 0.0001f;
    }

    /* PBRS */
    float phi_after[2];
    phi_after[0] = _potential(env, 0);
    phi_after[1] = _potential(env, 1);
    for (int i = 0; i < N_AGENTS; i++)
        env->rewards[i] += PBRS_GAMMA * phi_after[g->agents[i].team]
                         - phi_before[g->agents[i].team];

    /* Team spirit blending */
    if (env->team_spirit > 0.0f) {
        for (int team = 0; team < 2; team++) {
            float sum = 0.0f; int cnt = 0;
            for (int i = 0; i < N_AGENTS; i++) {
                if (g->agents[i].alive && g->agents[i].team == team)
                    { sum += env->rewards[i]; cnt++; }
            }
            if (cnt > 0) {
                float mean = sum / (float)cnt;
                float ts   = env->team_spirit;
                for (int i = 0; i < N_AGENTS; i++) {
                    if (g->agents[i].alive && g->agents[i].team == team)
                        env->rewards[i] = (1.0f - ts) * env->rewards[i] + ts * mean;
                }
            }
        }
    }

    /* Step 13: terminals/truncations */
    for (int i = 0; i < N_AGENTS; i++) {
        env->terminals[i]   = g->round_over;
        env->truncations[i] = 0;
    }
}

void env_close(Dust2Env* env) { (void)env; /* no-op */ }
