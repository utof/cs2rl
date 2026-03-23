#pragma once

#include "cs2_types.h"

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

static void compute_rewards(
    Dust2Env*  env,
    const int32_t* actions,
    int        t_alive,
    int        ct_alive,
    float      phi_before[2],
    int        kills[N_AGENTS][2],
    int        n_kills,
    int8_t     bombsite_entry_bonus[TEAM_SIZE],
    float      plant_progress_reward[TEAM_SIZE],
    int8_t     plant_interrupted[TEAM_SIZE],
    int        bomb_just_planted,
    int        bomb_planter_id,
    int        bomb_just_defused,
    int        bomb_defuser_id)
{
    StaticData* sd = env->sd;
    GameState*  g  = &env->game;
    StepStats*  ss = &env->step_stats;
    StepStats*  es = &env->episode_stats;

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
