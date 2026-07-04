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
    /* Verify ACTION_HEAD_SIZES stays in sync with ACTION_DIM/ACTION_MASK_DIM */
    {
        int sum = 0;
        for (int h = 0; h < ACTION_DIM; h++)
            sum += ACTION_HEAD_SIZES[h];
        assert(sum == ACTION_MASK_DIM && "ACTION_MASK_DIM != sum(ACTION_HEAD_SIZES)");
    }
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
    /* Batch 2: round-fixed copy. NEVER reassigned mid-round (see cs2_types.h
     * field comment). compute_observations reads this for obs[106] (T4 shift). */
    g->round_designated_carrier_id = bomb_carrier;
}

/* Batch 3 / Batch 3.5: env_step now consumes TWO action buffers — separate, non-bit-cast.
 * `actions`             : (N_AGENTS, ACTION_DIM=7) int32 — discrete heads (move/shoot/etc).
 * `continuous_actions`  : (N_AGENTS, AIM_DIM=2)    float  — [Δyaw, Δpitch] radians per agent.
 * The discrete enum no longer contains HEAD_AIM; aim is applied via the float
 * buffer in the per-agent block below (Δyaw clamped to ±sd->max_turn_speed then
 * wrap_pi'd into [-π,+π]; Δpitch clamped to ±sd->max_turn_speed then a->pitch
 * accumulator clamped to ±π/2 — bounded interval, NOT wrapped, see the no-wrap_pi
 * pitfall comment near the consumption block). Owners: binding.c py_step plumbs
 * both AND validates the cont buffer shape (Batch 3.5 #24). */
static void env_step(Dust2Env* env, const int32_t* actions, const float* continuous_actions) {
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

        int shoot_act  = actions[i * ACTION_DIM + HEAD_SHOOT];
        int reload_act = actions[i * ACTION_DIM + HEAD_RELOAD];
        int wswitch    = actions[i * ACTION_DIM + HEAD_WEAPON];
        /* use and crouch handled in cs2_bomb.h and cs2_movement.h */

        /* Batch 3: continuous-aim Δyaw consumption.
         * Human-controlled agents still set facing directly via aim_rad
         * (mouse delta wired in by deploy/UI). RL agents: read Δyaw from
         * the continuous buffer, clamp to ±max_turn_speed (π/4 rad/tick
         * from sd), apply, then wrap into [-π, +π]. Stats (sum/sq_sum/
         * count) accumulate the CLAMPED value — this matches what the env
         * actually executed, so Welford recovery reflects effective policy
         * action, not raw network output. */
        if (a->human_controlled) {
            a->facing = a->aim_rad;
        } else {
            float delta_yaw    = continuous_actions[i * AIM_DIM + 0];
            float clamped      = fminf(fmaxf(delta_yaw, -sd->max_turn_speed), sd->max_turn_speed);
            a->facing          = wrap_pi(a->facing + clamped);
            ss->aim_delta_sum += clamped;
            ss->aim_delta_sq_sum += clamped * clamped;
            ss->aim_delta_count  += 1;
            es->aim_delta_sum    += clamped;
            es->aim_delta_sq_sum += clamped * clamped;
            es->aim_delta_count  += 1;
            /* Batch 3.5 v1c (gh #36 fix B-3): ABSOLUTE pitch (no accumulator).
             *
             * v1a/v1b had Δpitch (delta + accumulator + bounded clamp), mirroring
             * yaw's wrap-pi rotation. Failed cold-start training (T7 v1a, v1b):
             * random-init policies produce a small but non-zero MEAN Δpitch
             * (typically ±0.05 rad/tick from random aim_mu weights × obs).
             * Yaw tolerates this — wrap_pi keeps it ergodic. Pitch's bounded
             * ±π/2 clamp turns drift into hard saturation lock within ~16 ticks
             * (1 sec). Saturated pitch (looking straight up/down) → all shots
             * miss → no kill signal → no gradient → policy never recovers.
             * Chicken-and-egg, structurally unfixable under Δpitch semantics.
             *
             * Solution: pitch is now state-dependent ABSOLUTE (the policy picks
             * "where to look" each tick, not "how fast to turn"). No drift, no
             * saturation pathology. Policy output range (±max_turn_speed = ±π/4
             * after tanh*max_turn_speed) is preserved, which limits effective
             * pitch to ±π/4 (~45°). For our world geometry that covers all
             * realistic engagement angles (catwalk z=128 vs floor z=0 at 200u
             * → atan2(-128, 200) ≈ -32°, well within ±45°).
             *
             * Welford fields keep their NAMES (aim_delta_pitch_*) for ctypes
             * compatibility but now record the APPLIED ABSOLUTE pitch. mean =
             * "where the policy is on average looking", σ = "spread of look
             * directions". aim_log_std_pitch (T6 metric) is the primary
             * non-collapse signal; this Welford pair is a secondary surface
             * showing engagement-angle distribution.
             *
             * Pitfall: tests that pre-set agent.pitch via ctypes WRITE then
             * call env.step() will see pitch overwritten by the cont buffer.
             * Set the desired pitch via continuous_actions[i*AIM_DIM+1] instead. */
            float pitch_target = continuous_actions[i * AIM_DIM + 1];
            a->pitch           = fminf(fmaxf(pitch_target, -(float)M_PI / 2), (float)M_PI / 2);
            ss->aim_delta_pitch_sum    += a->pitch;
            ss->aim_delta_pitch_sq_sum += a->pitch * a->pitch;
            ss->aim_delta_pitch_count  += 1;
            es->aim_delta_pitch_sum    += a->pitch;
            es->aim_delta_pitch_sq_sum += a->pitch * a->pitch;
            es->aim_delta_pitch_count  += 1;
        }
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
            if (g->agents[i].alive && actions[i * ACTION_DIM + HEAD_MOVE] == 0) {
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

    /* Compute mask offsets from ACTION_HEAD_SIZES (derived, not hardcoded) */
    int moff[ACTION_DIM];
    moff[0] = 0;
    for (int h = 1; h < ACTION_DIM; h++)
        moff[h] = moff[h - 1] + ACTION_HEAD_SIZES[h - 1];

    /* Compute action masks for next step */
    memset(env->masks, 1, sizeof(env->masks)); /* default: all valid */
    for (int i = 0; i < N_AGENTS; i++) {
        int8_t*     m = &env->masks[i * ACTION_MASK_DIM];
        AgentState* a = &g->agents[i];
        if (!a->alive) {
            memset(m, 0, ACTION_MASK_DIM); /* dead: nothing valid */
            m[0] = 1;                      /* stop is always valid */
            continue;
        }
        /* Jump mask: no jump while airborne, on cooldown, or crouching */
        if (a->is_airborne || a->jump_cd > 0 || a->is_crouching)
            m[moff[HEAD_JUMP] + 1] = 0;
        int              slot = a->weapon_slot;
        const WeaponDef* def  = &WEAPON_DEFS[slot];
        /* Shoot mask: gate on cooldown, reload, switch, and ammo.
         * Knife (mag_size < 0) is always shootable. */
        int has_ammo = (def->mag_size < 0) || (a->ammo_clip[slot] > 0);
        int can_shoot =
            (a->fire_cd == 0 && a->reload_ticks == 0 && a->switch_ticks == 0 && has_ammo);
        m[moff[HEAD_SHOOT] + 1] = (int8_t)can_shoot;
        /* Reload mask */
        int can_reload =
            (def->mag_size > 0 && a->ammo_clip[slot] < def->mag_size && a->ammo_reserve[slot] > 0 &&
             a->reload_ticks == 0 && a->switch_ticks == 0);
        m[moff[HEAD_RELOAD] + 1] = (int8_t)can_reload;
        /* Weapon switch mask: mask already-held weapon option */
        if (slot == 0)
            m[moff[HEAD_WEAPON] + 1] = 0;
        if (slot == 1)
            m[moff[HEAD_WEAPON] + 2] = 0;
        if (a->switch_ticks > 0) {
            m[moff[HEAD_WEAPON] + 1] = 0;
            m[moff[HEAD_WEAPON] + 2] = 0;
        }
        /* Use mask: T needs bomb+bombsite; CT needs planted bomb nearby */
        if (a->team == 0 && a->has_bomb && !g->bomb_planted) {
            int at_site = (a->area_idx >= 0) ? sd->bombsite_by_idx[a->area_idx] : 0;
            if (!at_site)
                m[moff[HEAD_USE] + 1] = 0;
        } else if (a->team == 1 && g->bomb_planted) {
            if (a->area_idx != g->bomb_area_idx)
                m[moff[HEAD_USE] + 1] = 0;
        } else {
            m[moff[HEAD_USE] + 1] = 0;
        }
    }
}

static void env_close(Dust2Env* env) {
    (void)env;
}
