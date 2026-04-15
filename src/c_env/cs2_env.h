#pragma once
#include "cs2_types.h"
#include "cs2_navmesh.h"
#include "cs2_weapons.h"
#include "cs2_player.h"
#include "cs2_movement.h"
#include "cs2_combat.h"
#include "cs2_bomb.h"
#include "cs2_observations.h"
#include "cs2_rewards.h"
#include "cs2_round.h"

static void env_init(Dust2Env* env, StaticData* sd, uint32_t seed, float team_spirit) {
    memset(env, 0, sizeof(Dust2Env));
    env->sd          = sd;
    env->rng         = seed ? seed : 1;
    env->team_spirit = team_spirit;
    clear_stats(&env->step_stats);
    clear_stats(&env->episode_stats);
}

static void env_reset(Dust2Env* env) {
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

static void env_step(Dust2Env* env, const int32_t* actions) {
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
        /* Tick weapon state (fire_cd, reload, switch, crouch_cd) */
        if (a->alive)
            tick_weapon(a);

        /* Complete a weapon switch when switch_ticks just hit 0 */
        if (a->alive && a->switch_ticks == 0 && a->weapon_slot != a->weapon_slot_target) {
            a->weapon_slot = a->weapon_slot_target;
        }

        a->is_moving       = 0;
        a->fired_this_tick = 0;
    }

    process_movement(env, actions, ss, es);

    /* Parse non-movement action heads */
    for (int i = 0; i < N_AGENTS; i++) {
        AgentState* a = &g->agents[i];
        if (!a->alive)
            continue;

        int aim_act    = actions[i * ACTION_DIM + 1];
        int shoot_act  = actions[i * ACTION_DIM + 2];
        int reload_act = actions[i * ACTION_DIM + 3];
        int wswitch    = actions[i * ACTION_DIM + 4];
        /* use (5) and crouch (6) handled in cs2_bomb.h and cs2_movement.h */

        /* Aim: continuous for human, 16-bin for RL agents */
        if (a->human_controlled) {
            a->facing = a->aim_rad;
        } else if (aim_act >= 0 && aim_act < 16) {
            a->facing = (aim_act / 16.0f) * 2.0f * (float)M_PI;
        }
        count_action(ss->action_aim, es->action_aim, aim_act, 16);
        count_action(ss->action_shoot, es->action_shoot, shoot_act, 2);

        /* Reload */
        if (reload_act == 1)
            try_start_reload(a);
        count_action(ss->action_reload, es->action_reload, reload_act, 2);

        /* Weapon switch: 1=switch_to_primary(0), 2=switch_to_secondary(1) */
        if (wswitch == 1 && a->weapon_slot != 0) {
            a->weapon_slot_target = 0;
            try_weapon_switch(a, 0);
        } else if (wswitch == 2 && a->weapon_slot != 1) {
            a->weapon_slot_target = 1;
            try_weapon_switch(a, 1);
        }
        count_action(ss->action_weapon, es->action_weapon, wswitch, 3);
    }

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

    /* Drop bomb if carrier was killed */
    {
        AgentState* carrier = (g->bomb_carrier_id >= 0 && g->bomb_carrier_id < TEAM_SIZE)
                                  ? &g->agents[g->bomb_carrier_id]
                                  : NULL;
        if (carrier && !carrier->alive && !g->bomb_planted && !g->bomb_is_dropped) {
            g->bomb_x          = carrier->x;
            g->bomb_y          = carrier->y;
            g->bomb_z          = carrier->z;
            carrier->has_bomb  = 0;
            g->bomb_carrier_id = -1;
            g->bomb_is_dropped = 1;
        }
    }

    process_bomb(env,
                 actions,
                 bombsite_entry_bonus,
                 plant_progress_reward,
                 plant_interrupted,
                 &bomb_just_planted,
                 &bomb_planter_id,
                 &bomb_just_defused,
                 &bomb_defuser_id,
                 ss,
                 es);

    update_enemy_memory(env, vis10);

    compute_observations(env, t_alive, ct_alive, vis10);

    memset(env->rewards, 0, N_AGENTS * sizeof(float));

    /* Inaction cost: discourage agents from standing still during active play */
    if (!g->round_over) {
        for (int i = 0; i < N_AGENTS; i++) {
            if (g->agents[i].alive && actions[i * ACTION_DIM + 0] == 0) {
                env->rewards[i]                    -= env->sd->reward_inaction;
                env->step_stats.reward_inaction    -= env->sd->reward_inaction;
                env->episode_stats.reward_inaction -= env->sd->reward_inaction;
            }
        }
    }

    compute_rewards(env,
                    t_alive,
                    ct_alive,
                    phi_before,
                    kills,
                    n_kills,
                    bombsite_entry_bonus,
                    plant_progress_reward,
                    plant_interrupted,
                    bomb_just_planted,
                    bomb_planter_id,
                    bomb_just_defused,
                    bomb_defuser_id);

    /* Compute action masks for next step */
    memset(env->masks, 1, sizeof(env->masks)); /* default: all valid */
    for (int i = 0; i < N_AGENTS; i++) {
        int8_t*     m = &env->masks[i * ACTION_MASK_DIM];
        AgentState* a = &g->agents[i];
        /* Offsets in mask array: move=0..8, aim=9..24, shoot=25..26,
           reload=27..28, wswitch=29..31, use=32..33, crouch=34..35,
           jump=36..37 */
        if (!a->alive) {
            memset(m, 0, ACTION_MASK_DIM); /* dead: nothing valid */
            m[0] = 1;                      /* stop is always valid */
            continue;
        }
        /* Jump mask (offset 36+): no jump while airborne, on cooldown, or
         * crouching. "no-jump" (m[36]) is always valid. */
        if (a->is_airborne || a->jump_cd > 0 || a->is_crouching)
            m[36 + 1] = 0;
        int              slot = a->weapon_slot;
        const WeaponDef* def  = &WEAPON_DEFS[slot];
        /* Shoot mask (offset 25+) */
        int can_shoot = (a->fire_cd == 0 && a->reload_ticks == 0 && a->switch_ticks == 0);
        m[25 + 1]     = (int8_t)can_shoot; /* shoot=yes */
        /* Reload mask (offset 27+) */
        int can_reload =
            (def->mag_size > 0 && a->ammo_clip[slot] < def->mag_size && a->ammo_reserve[slot] > 0 &&
             a->reload_ticks == 0 && a->switch_ticks == 0);
        m[27 + 1] = (int8_t)can_reload;
        /* Weapon switch mask (offset 29+): mask already-held weapon option */
        if (slot == 0)
            m[29 + 1] = 0; /* already rifle: switch_to_primary masked */
        if (slot == 1)
            m[29 + 2] = 0; /* already pistol: switch_to_secondary masked */
        if (a->switch_ticks > 0) {
            m[29 + 1] = 0;
            m[29 + 2] = 0;
        } /* in-progress: no new switch */
        /* Use mask (offset 32+): T-use requires bomb zone; CT-use requires planted bomb nearby */
        if (a->team == 0 && a->has_bomb && !g->bomb_planted) {
            int at_site = (a->area_idx >= 0) ? sd->bombsite_by_idx[a->area_idx] : 0;
            if (!at_site)
                m[32 + 1] = 0;
        } else if (a->team == 1 && g->bomb_planted) {
            if (a->area_idx != g->bomb_area_idx)
                m[32 + 1] = 0;
        } else {
            m[32 + 1] = 0; /* no valid use in any other state */
        }
    }
}

static void env_close(Dust2Env* env) {
    (void)env;
}
