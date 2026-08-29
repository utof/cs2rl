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
            bomb_progress = closeness * sd->pbrs_bomb_progress_weight;
            if (team == 1) {
                bomb_progress = -bomb_progress;
            }
        }
    }

    float nav_weight = (team == 1) ? sd->pbrs_nav_weight_ct : sd->pbrs_nav_weight_t;
    return (alive_t - alive_o) * sd->pbrs_alive_weight + (hp_t - hp_o) * sd->pbrs_hp_weight +
           (site_t - site_o) * sd->pbrs_site_weight + bomb_progress + nav_approach * nav_weight;
}

static void compute_rewards(Dust2Env* env,
                            int       t_alive,
                            int       ct_alive,
                            float     phi_before[2],
                            int       kills[N_AGENTS][2],
                            int       n_kills,
                            int8_t    bombsite_entry_bonus[TEAM_SIZE],
                            float     plant_progress_reward[TEAM_SIZE],
                            int8_t    plant_interrupted[TEAM_SIZE],
                            int       bomb_just_planted,
                            int       bomb_planter_id,
                            int       bomb_just_defused,
                            int       bomb_defuser_id) {
    StaticData* sd = env->sd;
    GameState*  g  = &env->game;
    StepStats*  ss = &env->step_stats;
    StepStats*  es = &env->episode_stats;

    if (g->round_over) {
        /* Batch 1 (RL overhaul): differential win rewards by outcome mechanism.
         *
         * Classification (mutually exclusive, evaluated in order):
         *   T win   (winner == 0):
         *     - detonation: bomb_planted && bomb_ticks_left <= 0 (bomb timer ran to zero)
         *     - elimination: otherwise (killed all CT, with or without planting attempt)
         *   CT win  (winner == 1):
         *     - defuse: bomb_just_defused set this tick (cs2_bomb.h defuse branch)
         *     - elimination: otherwise (killed all T, with bomb planted but not defused,
         *       or before any plant)
         *   Timeout (winner == -1, timed_out flag set):
         *     - CT gets the timeout magnitude as a reward for surviving without a plant.
         *     - T agents receive -timeout magnitude as a penalty.
         *     - Note: the C env uses winner=-1 for this case, not winner=1.
         *
         * We classify CT-defuse via the bomb_just_defused argument (set by
         * cs2_bomb.h only when the defuse branch actually completes). This is
         * the spec-compliant path: the live-play sequence where CT kills the
         * last T post-plant sets winner=1 in cs2_env.h before process_bomb
         * has a chance to fire, so bomb_just_defused stays 0 and we correctly
         * classify as elimination — NOT defuse. Tests that want to exercise
         * defuse MUST drive process_bomb naturally (see test_natural_defuse
         * in tests/test_reward.py).
         *
         * Sets StepStats.win_by_detonation / win_by_defuse for Python-side
         * channel routing: objective channel for detonation/defuse, combat
         * channel for elimination/timeout.
         */
        int t_won     = (g->winner == 0);
        int ct_won    = (g->winner == 1);
        int timed_out = (g->winner == -1);
        int detonated = t_won && g->bomb_planted && (g->bomb_ticks_left <= 0);
        int defused   = ct_won && bomb_just_defused;

        ss->win_by_detonation = detonated ? 1 : 0;
        ss->win_by_defuse     = defused ? 1 : 0;
        es->win_by_detonation = ss->win_by_detonation;
        es->win_by_defuse     = ss->win_by_defuse;

        float t_mag, ct_mag;
        if (t_won) {
            t_mag  = detonated ? sd->reward_win_t_detonation : sd->reward_win_t_elimination;
            ct_mag = -t_mag; /* losers receive equal-magnitude penalty */
        } else if (ct_won) {
            ct_mag = defused ? sd->reward_win_ct_defuse : sd->reward_win_ct_elimination;
            t_mag  = -ct_mag;
        } else if (timed_out) {
            /* Timeout: no team "won" but CTs achieved their objective (no plant).
             * We give CTs a positive reward and Ts the symmetric penalty. */
            ct_mag = sd->reward_win_ct_timeout;
            t_mag  = -ct_mag;
        } else {
            t_mag  = 0.0f; /* defensive: ongoing round, should not be reached */
            ct_mag = 0.0f;
        }

        /* Terminal win/loss applies to EVERY team member, dead or alive
         * (finding 3, docs/2026-07-06-adversarial-review-verification.md).
         * The block used to be gated on agents[i].alive, which made death an
         * escape hatch: a wiped losing team received 0 instead of -mag each
         * (death cost only reward_death = 0.1), and a winner who traded
         * itself for the round got nothing. Credit for the round outcome
         * belongs to the whole team.
         * PITFALL: ss/es->reward_win is now the truthful cross-team sum of
         * emitted terminal rewards — with symmetric magnitudes and equal
         * team sizes it nets to ~0 every round. Use winner_t / winner_ct /
         * timed_out (and per-agent rewards) for outcome metrics, not this
         * accumulator. */
        for (int i = 0; i < N_AGENTS; i++) {
            /* Rung 0: parked slots are not on the team — no payout. This loop
             * deliberately ignores `alive` (see PITFALL above), so it needs
             * its own participating guard. */
            if (!g->agents[i].participating)
                continue;
            float w          = (g->agents[i].team == 0) ? t_mag : ct_mag;
            env->rewards[i] += w;
            ss->reward_win  += w;
            es->reward_win  += w;
        }
    }

    for (int k = 0; k < n_kills; k++) {
        env->rewards[kills[k][0]] += sd->reward_kill;
        ss->reward_kills          += sd->reward_kill;
        es->reward_kills          += sd->reward_kill;
        env->rewards[kills[k][1]] -= sd->reward_death;
        ss->reward_deaths         -= sd->reward_death;
        es->reward_deaths         -= sd->reward_death;
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
            env->rewards[i] += sd->reward_bombsite_entry;
            ss->reward_bomb += sd->reward_bombsite_entry;
            es->reward_bomb += sd->reward_bombsite_entry;
        }
        env->rewards[i] += plant_progress_reward[i]; /* per-tick plant progress */
        ss->reward_bomb += plant_progress_reward[i];
        es->reward_bomb += plant_progress_reward[i];
        if (plant_interrupted[i]) {
            env->rewards[i] -= sd->reward_plant_interrupted;
            ss->reward_bomb -= sd->reward_plant_interrupted;
            es->reward_bomb -= sd->reward_plant_interrupted;
        }
    }

    if (bomb_just_planted && bomb_planter_id >= 0) {
        float pb                       = sd->reward_plant_base + sd->reward_plant_bonus;
        env->rewards[bomb_planter_id] += pb;
        ss->reward_bomb               += pb;
        es->reward_bomb               += pb;
    }
    if (bomb_just_defused && bomb_defuser_id >= 0) {
        env->rewards[bomb_defuser_id] += sd->reward_defuse;
        ss->reward_bomb               += sd->reward_defuse;
        es->reward_bomb               += sd->reward_defuse;
    }

    /* Small cost per shot fired — discourages infinite spray at walls */
    for (int i = 0; i < N_AGENTS; i++) {
        if (env->game.agents[i].alive && env->game.agents[i].fired_this_tick) {
            env->rewards[i]  -= sd->reward_shot_penalty;
            ss->reward_shots -= sd->reward_shot_penalty;
            es->reward_shots -= sd->reward_shot_penalty;
        }
    }

    /* CT survival micro-reward: gives CTs incentive to stay alive long enough to learn */
    if (!g->round_over) {
        for (int i = TEAM_SIZE; i < N_AGENTS; i++) {
            if (g->agents[i].alive) {
                env->rewards[i]     += sd->reward_ct_survival;
                ss->reward_survival += sd->reward_ct_survival;
                es->reward_survival += sd->reward_ct_survival;
            }
        }
    }

    {
        float phi_after[2];
        phi_after[0] = _potential(env, 0);
        phi_after[1] = _potential(env, 1);
        for (int i = 0; i < N_AGENTS; i++) {
            /* Rung 0: ungated by alive on purpose (dead agents still feel
             * team potential), but parked rows must not — they would inflate
             * reward_pbrs by (TEAM_SIZE / n_active)x. */
            if (!g->agents[i].participating)
                continue;
            float pbrs =
                sd->pbrs_gamma * phi_after[g->agents[i].team] - phi_before[g->agents[i].team];
            env->rewards[i] += pbrs;
            ss->reward_pbrs += pbrs;
            es->reward_pbrs += pbrs;
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
        /* F15 (2026-07-06 adversarial review): timeout COUNTS as a CT win in
         * winner_ct. Rewards already treated it that way (CTs get
         * reward_win_ct_timeout, Ts the symmetric penalty), but the stat used
         * to stay 0 — dashboards undercounted CT wins by exactly the timeout
         * rate, and self-play save/team-switch logic read the skewed rate.
         * The raw mechanism is still fully recoverable: `winner` stays -1 on
         * timeout and `timed_out` is its own flag, so
         * elimination/defuse-only CT wins = winner_ct - timed_out. */
        int ct_win_effective = (g->winner == 1) || (g->winner == -1);

        ss->winner       = g->winner;
        ss->winner_t     = (g->winner == 0);
        ss->winner_ct    = ct_win_effective;
        ss->alive_t_end  = t_alive;
        ss->alive_ct_end = ct_alive;
        ss->round_length = g->tick;

        es->winner       = g->winner;
        es->winner_t     = (g->winner == 0);
        es->winner_ct    = ct_win_effective;
        es->alive_t_end  = t_alive;
        es->alive_ct_end = ct_alive;
        es->round_length = g->tick;
    }

    for (int i = 0; i < N_AGENTS; i++) {
        env->terminals[i]   = g->round_over;
        env->truncations[i] = 0;
    }
}
