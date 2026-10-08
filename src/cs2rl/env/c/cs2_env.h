#pragma once
#include "cs2_types.h"
#include "cs2_solids.h" /* free_solids — env_close owns sd->wall_list */
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
    /* Verify the obs blocks (cs2_types.h OBS_* macros) tile OBS_DIM exactly.
     * Catches a block *_SIZE / *_STRIDE / OBS_DIM drift at startup — same
     * fail-fast contract as the ACTION check above. Since block bases derive
     * from the preceding block width, this reduces to: last block end == dim. */
    assert(OBS_GLOBAL_BASE + OBS_GLOBAL_SIZE == OBS_DIM &&
           "obs block sizes do not tile OBS_DIM (see cs2_types.h OBS_* macros)");
    memset(env, 0, sizeof(Dust2Env));
    env->sd = sd;
    /* Rung 0: a zeroed StaticData (cs2_demo.c load_nav_data forgot the field, or
     * a C caller that built its own StaticData) would make env_reset divide by
     * zero. Python callers never reach this — Cs2Env.__init__ raises ValueError
     * first — so this is the guard for the C-only callers (cs2_demo.c /
     * make_client). */
    assert(sd->n_active_per_team >= 1 && sd->n_active_per_team <= TEAM_SIZE &&
           "StaticData.n_active_per_team must be in 1..TEAM_SIZE");
    /* R0-D (#135): seeds 0 and 1 used to alias (`seed ? seed : 1`) — every
     * env pair (2k, 2k+1) handed adjacent seeds by pufferlib shared a stream.
     * Golden-ratio (0x9E3779B9) additive mix; uint32 wrap is intended. The
     * `: 1u` fallback fires ONLY at seed == 0x61C88647 (the one value whose
     * sum wraps to exactly 0 — xorshift32 would be stuck at 0 forever).
     * PITFALL: changing this moves every xorshift32 stream => sim fingerprints
     * (scripts/sim_fingerprint.py) legitimately change; record the new set. */
    env->rng         = (seed + 0x9E3779B9u) ? (seed + 0x9E3779B9u) : 1u;
    env->team_spirit = team_spirit;
    clear_stats(&env->step_stats);
    clear_stats(&env->episode_stats);
    /* area_bounds stays NULL unless a caller sets it (Cs2Env after init,
     * make_client). Rooms are the one source — do not derive from raster. */
}

/* compute_masks: refresh env->masks (N_AGENTS × ACTION_MASK_DIM int8, 1=valid)
 * from the CURRENT game state. Called at the tail of env_step (masks describe
 * the NEXT step's valid actions) and at the end of env_reset — without the
 * reset call the buffer would hold stale terminal-round masks (or all-zeros
 * on the very first reset), which matters now that the trainer actually
 * consumes masks (F8, 2026-07-06 adversarial review).
 *
 * Invariant the Python sampler relies on: EVERY head of EVERY agent keeps at
 * least one valid bin. For dead agents that is bin 0 (the no-op) of each head
 * — NOT just flat index 0 — because _hybrid_sample_logits masked-softmaxes
 * each head independently; an all-invalid head would be a NaN softmax.
 * Alive agents satisfy the invariant structurally (only non-zero bins are
 * ever gated below). */
static void compute_masks(Dust2Env* env) {
    StaticData* sd = env->sd;
    GameState*  g  = &env->game;

    /* Compute mask offsets from ACTION_HEAD_SIZES (derived, not hardcoded) */
    int moff[ACTION_DIM];
    moff[0] = 0;
    for (int h = 1; h < ACTION_DIM; h++)
        moff[h] = moff[h - 1] + ACTION_HEAD_SIZES[h - 1];

    memset(env->masks, 1, sizeof(env->masks)); /* default: all valid */
    for (int i = 0; i < N_AGENTS; i++) {
        int8_t*     m = &env->masks[i * ACTION_MASK_DIM];
        AgentState* a = &g->agents[i];
        if (!a->alive) {
            memset(m, 0, ACTION_MASK_DIM); /* dead: only per-head no-ops valid */
            for (int h = 0; h < ACTION_DIM; h++)
                m[moff[h]] = 1;
            continue;
        }
        /* Jump mask: no jump while airborne, on cooldown, or crouching.
         * Rung 1a (spec 2026-08-30 T2a): jump_enabled=0 masks the press for
         * the whole run — same shape and same rationale as the crouch gate
         * below (minimal action space for the aim rung; with pitch pinned an
         * airborne agent also spends most of its airtime outside the hit band,
         * gh #150). Defaults to 1 (cs2_demo.c forces 1 for the human player).
         * Bin 0 (no jump) stays valid, preserving the per-head no-op.
         * W5 (#156): this mask is NO LONGER the enforcement. process_movement
         * zeroes jump_act at the read when jump_enabled=0, so the flag holds on
         * every path, mask or not. Keeping the mask is a policy-side
         * optimisation — a masked policy should not spend probability mass on a
         * bin the sim is going to drop. */
        if (a->is_airborne || a->jump_cd > 0 || a->is_crouching || !sd->jump_enabled)
            m[moff[HEAD_JUMP] + 1] = 0;
        /* R0-E.2 (#131): stance parity. PRE-v1c rationale (no longer true):
         * with pitch pinned a stand-vs-crouch mismatch was |rz| = 24 >
         * HIT_HALF_WIDTH = 16 — an unconditional miss the policy cannot
         * observe (there is still no stance bit in the enemy block). Since
         * v1c (gh #150) the hitbox is an ellipsoid with a 27u crouched / 36u
         * standing vertical semi-axis, so 24 ≤ 27 CONNECTS and the mismatch is
         * a margin cost, not a wall (tests/train/test_pitch_pin.py asserts exactly
         * that). The mask STAYS anyway: Rung 1 deliberately keeps the action
         * space minimal, and stance still eats parity margin the policy cannot
         * see. 5v5 keeps crouch (crouch_enabled defaults to 1, cs2_demo.c
         * forces 1 for the human player). Bin 0 (stand) stays valid so the head
         * keeps its per-head no-op invariant (see header comment).
         * W5 (#156): as with the jump mask above, the mask is no longer what
         * makes crouch_enabled=0 stick — process_movement zeroes crouch_act at
         * the read. The mask is now the policy-side half of a sim-enforced
         * invariant. */
        if (!sd->crouch_enabled)
            m[moff[HEAD_CROUCH] + 1] = 0;
        /* R0-B (#129): this function runs at the TAIL of env_step, but the
         * masks it writes are consumed by the NEXT env_step, whose first act
         * is tick_weapon() — which decrements fire_cd/reload_ticks/switch_ticks
         * BEFORE process_combat / try_start_reload / try_weapon_switch read
         * them. Gating on the raw counter therefore reported "blocked" one
         * tick longer than the sim enforces (rifle cycle_ticks=2 became 1 shot
         * per 3 ticks; the first legal shot after a reload/switch was masked).
         * So the mask must describe the POST-decrement value, max(c-1, 0):
         * the counter is 0 next tick iff it is <= 1 now.
         * Two side effects ride on those decrements and are predicted too:
         *  - switch_ticks 1->0: env_step flips weapon_slot to weapon_slot_target
         *    right after tick_weapon, so `slot`/`def` below describe the weapon
         *    that will be HELD when actions are parsed (ammo + "already held").
         *  - reload_ticks 1->0: tick_weapon refills the clip to mag_size and
         *    takes one mag from reserve, so the reload head must close (a
         *    reload of a full clip is rejected by try_start_reload).
         * Pitfall: the SHOOT head's has_ammo deliberately reads the raw
         * (pre-refill) clip, NOT clip_next. On the refill tick process_combat
         * would actually accept a shot (refill precedes its dry-fire check),
         * but spec §3 R0-B pins empty-mag first shot at T+40 vs partial-mag
         * T+39 (tests/env/c/test_fire_mask.py) — the mask is one tick conservative
         * there by decision, not by equivalence; do not "fix" it to clip_next.
         * jump_cd is never set (cs2_movement.h); crouch_cd is not read here. */
        int              fire_cd_next = a->fire_cd > 0 ? a->fire_cd - 1 : 0;
        int              reload_next  = a->reload_ticks > 0 ? a->reload_ticks - 1 : 0;
        int              switch_next  = a->switch_ticks > 0 ? a->switch_ticks - 1 : 0;
        int              slot = (a->switch_ticks == 1) ? a->weapon_slot_target : a->weapon_slot;
        const WeaponDef* def  = &WEAPON_DEFS[slot];
        int              clip_next = (a->reload_ticks == 1) ? def->mag_size : a->ammo_clip[slot];
        int              reserve_next =
            (a->reload_ticks == 1) ? a->ammo_reserve[slot] - 1 : a->ammo_reserve[slot];
        /* Shoot mask: gate on cooldown, reload, switch, and ammo.
         * Knife (mag_size < 0) is always shootable. */
        int has_ammo  = (def->mag_size < 0) || (a->ammo_clip[slot] > 0);
        int can_shoot = (fire_cd_next == 0 && reload_next == 0 && switch_next == 0 && has_ammo);
        m[moff[HEAD_SHOOT] + 1] = (int8_t)can_shoot;
        /* Reload mask: predicted clip/reserve so the refill tick reads "full" */
        int can_reload = (def->mag_size > 0 && clip_next < def->mag_size && reserve_next > 0 &&
                          reload_next == 0 && switch_next == 0);
        m[moff[HEAD_RELOAD] + 1] = (int8_t)can_reload;
        /* Weapon switch mask: mask already-held weapon option */
        if (slot == 0)
            m[moff[HEAD_WEAPON] + 1] = 0;
        if (slot == 1)
            m[moff[HEAD_WEAPON] + 2] = 0;
        if (switch_next > 0) {
            m[moff[HEAD_WEAPON] + 1] = 0;
            m[moff[HEAD_WEAPON] + 2] = 0;
        }
        /* Eligibility is shared with bomb actions; masks still describe the
         * next step and do not gate on round_over or an in-progress lock. */
        m[moff[HEAD_USE] + 1] = (int8_t)(bomb_can_plant(sd, g, i) || bomb_can_defuse(g, a));
    }
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

    g->round_ticks_left = sd->round_time;
    g->winner           = -1;

    /* Rung 0: the carrier is drawn among ACTIVE T slots only (a parked
     * carrier would never drop/plant). Identity at n_active == TEAM_SIZE. */
    int n_active     = sd->n_active_per_team;
    int bomb_carrier = (int)(xorshift32(&env->rng) % n_active);
    spawn_team(g, sd, &env->rng, 0, sd->t_spawns, sd->n_t_spawns, n_active);
    spawn_team(g, sd, &env->rng, 1, sd->ct_spawns, sd->n_ct_spawns, n_active);
    /* Parked slots: the GameState memset above left them all-zero, i.e.
     * team=0 / alive=0 / area_idx=0 — area 0 is a REAL area and team=0 would
     * mis-attribute parked CT rows to T in every team-indexed loop. Make
     * them unambiguously "dead, nowhere, correct team". */
    for (int i = 0; i < N_AGENTS; i++) {
        if ((i % TEAM_SIZE) >= n_active) {
            AgentState* a    = &g->agents[i];
            a->participating = 0;
            a->alive         = 0;
            a->area_idx      = INVALID_AREA_IDX;
            a->team          = (int8_t)((i < TEAM_SIZE) ? 0 : 1);
            /* memset left enemy_mem_idx[*] = 0 = a REAL area; the struct
             * comment promises INVALID_AREA_IDX for parked rows. */
            for (int k = 0; k < TEAM_SIZE; k++)
                a->enemy_mem_idx[k] = INVALID_AREA_IDX;
        }
    }

    bomb_give(g, bomb_carrier);
    /* Batch 2: round-fixed copy. NEVER reassigned mid-round (see cs2_types.h
     * field comment). compute_observations reads this for obs[OBS_GLOBAL_BASE + 13]. */
    g->round_designated_carrier_id = bomb_carrier;

    /* F8: masks must describe the fresh spawn state, not the previous round's
     * terminal state (or all-zeros on first reset). Observations stay zeroed
     * at reset (pre-existing contract); masks can't, because the sampler
     * would divide by an all-invalid head. */
    compute_masks(env);
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
    int         n_kills = 0;
    int         nearest_vis_enemy[N_AGENTS]; /* R0-A: pre-combat target snapshot, -1 = none */
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
        int use_act    = actions[i * ACTION_DIM + HEAD_USE];
        /* use and crouch SEMANTICS handled in cs2_bomb.h and cs2_movement.h.
         * F13 (2026-07-06 adversarial review): the USE counter is wired HERE
         * because process_bomb never counted it — action_use was declared,
         * exported to W&B, and always 0 (misleading when diagnosing plant
         * behaviour). Like every other head counter this counts the INTENT
         * (raw action value from alive agents), not the effect. */
        count_action(ss->action_use, es->action_use, use_act, 2);

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
            if (!sd->pin_pitch) {
                float pitch_target = continuous_actions[i * AIM_DIM + 1];
                a->pitch           = fminf(fmaxf(pitch_target, -(float)M_PI / 2), (float)M_PI / 2);
                ss->aim_delta_pitch_sum    += a->pitch;
                ss->aim_delta_pitch_sq_sum += a->pitch * a->pitch;
                ss->aim_delta_pitch_count  += 1;
                es->aim_delta_pitch_sum    += a->pitch;
                es->aim_delta_pitch_sq_sum += a->pitch * a->pitch;
                es->aim_delta_pitch_count  += 1;
            } else {
                /* R0-E.2 (#131): flat maps — continuous_actions[i*AIM_DIM+1] is
                 * IGNORED and pitch is held at 0 so the 3D hit test reduces to
                 * yaw (rz = 0 for same-stance shots; see cs2_combat.h). The
                 * Welford pitch counters stay at 0 on purpose: a "pitch σ" of
                 * a dimension nobody consumes would read as a live signal in
                 * the metrics. Trainer side masks the pitch dim out of
                 * log_prob_c / entropy_c (policy.aim_dim_mask == [1, 0]) and
                 * assert_pin_pitch_agreement() refuses a mismatch at startup.
                 * PITFALL: Rung 1 still runs crouch_enabled=0. The PRE-v1c
                 * reason ("a crouched target is |rz| = 24 > HIT_HALF_WIDTH =
                 * 16, an unconditional miss") no longer holds — v1c (gh #150)
                 * gives a crouched target a 27u vertical semi-axis, so 24
                 * connects. The restriction survives to keep the Rung 1 action
                 * space minimal and because stance still costs parity margin
                 * the policy cannot observe. W5 (#156) went further: with
                 * crouch_enabled=0 the sim itself drops the press
                 * (process_movement), so under Rung 1 settings NO path can
                 * produce a crouched agent and the pinned-pitch shot geometry is
                 * stance-uniform by construction. See compute_masks in this
                 * file. */
                a->pitch = 0.0f;
            }
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

    if (env->recoil_enabled) {
        for (int i = 0; i < N_AGENTS; i++) {
            AgentState* a = &g->agents[i];
            if (!a->alive)
                continue;
            a->punch_pitch = recoil_decay_punch(a->punch_pitch, DT_SIM_MOVE);
            a->punch_yaw   = recoil_decay_punch(a->punch_yaw, DT_SIM_MOVE);
        }
    }

    build_vis_matrix(g, sd, vis10);

    /* ── R0-A pair counters + nearest-visible-enemy snapshot ──
     * MUST sit between build_vis_matrix and process_combat: process_combat
     * sets alive=0 on a kill, so anything after it would drop the kill tick
     * from the pair counters and make the shooter-side scoring depend on the
     * hitbox roll. One pre-combat liveness snapshot for everything below. */
    {
        int   seen_any[N_AGENTS] = {0};
        float nearest_d2[N_AGENTS];
        for (int i = 0; i < N_AGENTS; i++) {
            nearest_vis_enemy[i] = -1;
            nearest_d2[i]        = 1e30f;
        }
        for (int i = 0; i < N_AGENTS; i++) {
            AgentState* ai = &g->agents[i];
            if (!ai->participating || !ai->alive)
                continue;
            for (int j = 0; j < N_AGENTS; j++) {
                AgentState* aj = &g->agents[j];
                if (j == i || aj->team == ai->team || !aj->participating || !aj->alive)
                    continue;
                float dx = aj->x - ai->x, dy = aj->y - ai->y;
                float d2               = dx * dx + dy * dy;
                float d                = sqrtf(d2);
                ss->min_enemy_distance = fminf(ss->min_enemy_distance, d);
                es->min_enemy_distance = fminf(es->min_enemy_distance, d);
                if (vis10[i][j]) {
                    seen_any[i] = 1;
                    if (d2 < nearest_d2[i]) {
                        nearest_d2[i]        = d2;
                        nearest_vis_enemy[i] = j;
                    }
                    if (j > i && vis10[j][i]) { /* unordered pair, counted once */
                        ss->mutual_vis_pair_ticks++;
                        es->mutual_vis_pair_ticks++;
                    }
                }
            }
        }
        for (int i = 0; i < N_AGENTS; i++) {
            if (seen_any[i]) {
                ss->agent_ticks_with_visible_enemy++;
                es->agent_ticks_with_visible_enemy++;
            }
        }
    }

    process_combat(env, actions, vis10, kills, &n_kills, ss, es, nearest_vis_enemy);

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

    compute_masks(env);
}
/* env_close — release everything the C side malloc'd for this env.
 *
 * What: frees sd->wall_list (the baked solid faces). area_bounds is
 *       Python/nav-owned and is NOT freed here.
 * Why:  each BindingEnv carries its OWN StaticData (binding.c allocates
 *       Dust2Env + StaticData in one calloc), so the wall list is per-env,
 *       not shared. Today only the demo bakes, but as soon as env_step
 *       queries solids the training path bakes too and this would leak one
 *       list per env, per worker, for every run.
 * Pitfalls: free_solids is idempotent — c_close() also frees the list, and
 *           binding.py_close + capsule_destructor can both reach here. It
 *           NULLs the pointer and zeroes count/capacity, so a second call
 *           (in any order) is a no-op rather than a double free.
 */
static void env_close(Dust2Env* env) {
    if (env == NULL)
        return;
    free_solids(env->sd);
}
