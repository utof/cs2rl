#include "cs2_env.h"
#include "cs2_observations.h"
#include "cs2_movement.h"
#include "cs2_rewards.h"
#include "cs2_combat.h"
#include "cs2_round.h"
#include "cs2_bomb.h"
#include <stdlib.h>

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
    spawn_team(g, sd, &env->rng, 0, sd->t_spawns, sd->n_t_spawns, bomb_carrier);
    spawn_team(g, sd, &env->rng, 1, sd->ct_spawns, sd->n_ct_spawns, bomb_carrier);

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

    process_bomb(env, actions, bombsite_entry_bonus, plant_progress_reward, plant_interrupted,
                 &bomb_just_planted, &bomb_planter_id, &bomb_just_defused, &bomb_defuser_id,
                 ss, es);

    update_enemy_memory(env, vis10);

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
