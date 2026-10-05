#pragma once
#include "cs2_types.h"

/* The exported fields remain the storage contract. In particular, planting
 * leaves bomb_carrier_id stale. This query suppresses it while planted/dropped,
 * but does not check alive/has_bomb: drop must still find a dead carrier and
 * observations have always reported the phase-qualified identity. */
static int bomb_current_carrier(const GameState* g) {
    return (g->bomb_planted || g->bomb_is_dropped) ? -1 : g->bomb_carrier_id;
}

/* Shared USE eligibility for a live agent. Callers own alive/USE/round checks;
 * masks intentionally ignore round_over and another agent's progress lock.
 * Plant eligibility follows has_bomb, not the possibly stale carrier id. */
static int bomb_can_plant(const StaticData* sd, const GameState* g, const AgentState* a) {
    return a->team == 0 && a->has_bomb && !g->bomb_planted && a->area_idx >= 0 &&
           sd->bombsite_by_idx[a->area_idx];
}

/* Defuse eligibility uses area equality, not the dropped-bomb pickup radius.
 * It follows planted even if a raw caller also set the dropped flag. */
static int bomb_can_defuse(const GameState* g, const AgentState* a) {
    return a->team == 1 && g->bomb_planted && a->area_idx == g->bomb_area_idx;
}

/* Bomb transitions run after combat/elimination. Drop is deliberately outside
 * the round-over guards; pickup, USE, progress invalidation and clocks are not. */
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

    /* Drop before progress invalidation/pickup, including after elimination.
     * Preserve the raw area/progress/designated-carrier fields on this path. */
    int         carrier_id = bomb_current_carrier(g);
    AgentState* carrier =
        (carrier_id >= 0 && carrier_id < TEAM_SIZE) ? &g->agents[carrier_id] : NULL;
    if (carrier && !carrier->alive) {
        g->bomb_x          = carrier->x;
        g->bomb_y          = carrier->y;
        g->bomb_z          = carrier->z;
        carrier->has_bomb  = 0;
        g->bomb_carrier_id = -1;
        g->bomb_is_dropped = 1;
    }

    if (!g->round_over && g->bomb_being_defused_by != -1) {
        AgentState* def = &g->agents[g->bomb_being_defused_by];
        if (!def->alive || def->area_idx != g->bomb_area_idx ||
            actions[g->bomb_being_defused_by * ACTION_DIM + HEAD_USE] == 0) {
            g->bomb_being_defused_by = -1;
            g->bomb_defuse_ticks     = 0;
        }
    }

    /* Planter invalidation on death (finding 7, 2026-07-06 adversarial
     * review). Without this, a planter dying mid-plant left
     * bomb_being_planted_by frozen on the dead index; the `== -1` / `== i`
     * guards below then rejected every subsequent carrier and planting was
     * bricked for the rest of the round. Mirrors the defuser block above,
     * with two DELIBERATE differences: (a) only death/bomb-loss invalidates
     * — USE release keeps the pause-not-reset semantics of the plant branch
     * (the defuser resets on release; the planter does not); (b) progress
     * resets to 0 so the next planter starts fresh instead of inheriting
     * ticks it didn't earn (keeps per-tick plant-progress reward accounting
     * coherent). !has_bomb is defensive: you cannot be planting a bomb you
     * no longer hold (death->drop is the only current path here). No
     * plant_interrupted penalty on death — dying is already penalized. */
    if (!g->round_over && g->bomb_being_planted_by != -1) {
        AgentState* pl = &g->agents[g->bomb_being_planted_by];
        if (!pl->alive || !pl->has_bomb) {
            g->bomb_being_planted_by = -1;
            g->bomb_plant_ticks      = 0;
        }
    }

    if (!g->round_over) {
        for (int i = 0; i < N_AGENTS; i++) {
            AgentState* a = &g->agents[i];
            if (!a->alive || actions[i * ACTION_DIM + HEAD_USE] == 0) {
                continue;
            }

            if (bomb_can_plant(sd, g, a)) {
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
                        /* Stamp plant completion tick. ss is cleared every
                         * env_step so it always gets this tick. es is
                         * episode-lifetime: first-write-wins is belt-and-
                         * braces (bomb_planted already blocks a second
                         * plant). g->tick was incremented at the top of
                         * env_step, so this is never 0. */
                        ss->plant_tick = g->tick;
                        if (es->plant_tick == 0)
                            es->plant_tick = g->tick;
                    }
                }
            } else if (a->team == 0 && a->has_bomb && !g->bomb_planted &&
                       g->bomb_being_planted_by == i) {
                /* A live holder using off-site interrupts. Releasing USE still
                 * pauses, and death/loss is handled above without a penalty. */
                if (g->bomb_plant_ticks > 0) {
                    plant_interrupted[i] = 1; /* interrupted plant penalty applied post-memset */
                }
                g->bomb_being_planted_by = -1;
                g->bomb_plant_ticks      = 0;
            } else if (bomb_can_defuse(g, a)) {
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
