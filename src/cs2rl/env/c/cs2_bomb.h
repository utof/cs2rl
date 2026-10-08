#pragma once
#include "cs2_types.h"

/* The bomb subsystem (#164). GameState.bomb (BombState, cs2_types.h) is the one
 * authoritative lifecycle state. This file owns:
 *   - the queries every consumer reads (masks, observations, rewards, render,
 *     the play host) instead of rebuilding the phase from raw fields;
 *   - the named transitions, the only code that changes the phase (apart from
 *     env_reset's memset, which bomb_give follows). Each one
 *     leaves every BombState field at its value in the cs2_types.h table for
 *     the new phase (a field it does not set already holds that value). The
 *     only other writes are process_bomb's progress++ and ticks_left--;
 *   - bomb_state_error(), the table as a check (binding py_step runs it
 *     before every step), and bomb_give_error(), the rule for who may be handed
 *     the bomb (binding give_bomb, env_reset and pickup all hand it over through
 *     bomb_give);
 *   - process_bomb, the per-tick driver.
 * Callers own alive/USE/round checks unless a query says otherwise. */

/* ── Queries ─────────────────────────────────────────────────────────────── */

/* The agent holding the bomb, or -1. A carrier is never cleared by death here:
 * the drop in process_bomb must still find a dead carrier. */
static inline int bomb_carrier(const GameState* g) {
    return (g->bomb.phase == BOMB_CARRIED || g->bomb.phase == BOMB_PLANTING) ? g->bomb.agent : -1;
}

/* Planted at any point this round, including after it was defused or blew up. */
static inline int bomb_planted(const GameState* g) {
    return g->bomb.phase >= BOMB_PLANTED;
}

/* The detonation countdown is running: planted and not yet resolved. */
static inline int bomb_ticking(const GameState* g) {
    return g->bomb.phase == BOMB_PLANTED || g->bomb.phase == BOMB_DEFUSING;
}

/* Lying at g->bomb.x/y/z: dropped, or planted. */
static inline int bomb_on_ground(const GameState* g) {
    return g->bomb.phase == BOMB_DROPPED || bomb_planted(g);
}

static inline int bomb_detonated(const GameState* g) {
    return g->bomb.phase == BOMB_DETONATED;
}

/* The agent defusing (or who finished defusing), or -1. */
static inline int bomb_defuser(const GameState* g) {
    return (g->bomb.phase == BOMB_DEFUSING || g->bomb.phase == BOMB_DEFUSED) ? g->bomb.agent : -1;
}

/* Shared USE eligibility for agent i (alive is the caller's check). Masks
 * intentionally ignore round_over and another agent's progress lock. */
static inline int bomb_can_plant(const StaticData* sd, const GameState* g, int i) {
    const AgentState* a = &g->agents[i];
    return a->team == 0 && bomb_carrier(g) == i && a->area_idx >= 0 &&
           sd->bombsite_by_idx[a->area_idx];
}

/* Defuse eligibility uses area equality, not the dropped-bomb pickup radius.
 * It stays true after the round resolved (DEFUSED/DETONATED), so the terminal
 * tick's masks match the planted state. */
static inline int bomb_can_defuse(const GameState* g, const AgentState* a) {
    return a->team == 1 && bomb_planted(g) && a->area_idx == g->bomb.area_idx;
}

/* ── Transitions: the only code that changes GameState.bomb.phase ─────────── */

static inline void bomb_set(BombState* b, int phase, int agent, int progress) {
    b->phase    = phase;
    b->agent    = agent;
    b->progress = progress;
}

/* -> CARRIED by agent i, from any unplanted phase: env_reset, pickup, and the
 * binding's give_bomb. Clears the ground position: the carrier's own x/y is
 * where a carried bomb is. */
static void bomb_give(GameState* g, int i) {
    BombState* b = &g->bomb;
    bomb_set(b, BOMB_CARRIED, i, 0);
    b->ticks_left = 0;
    b->area_idx   = INVALID_AREA_IDX;
    b->x = b->y = b->z = 0.0f;
}

/* NULL if agent i may be handed the bomb now (a live, active T and an
 * unplanted bomb), else why not. */
static const char* bomb_give_error(const GameState* g, int i) {
    if (i < 0 || i >= TEAM_SIZE)
        return "the bomb goes to a T agent index (0 .. TEAM_SIZE-1)";
    if (!g->agents[i].participating || !g->agents[i].alive)
        return "the bomb goes to a live, participating agent";
    if (bomb_planted(g))
        return "the bomb is already planted this round";
    return NULL;
}

/* CARRIED/PLANTING -> DROPPED at the carrier's position. Discards any plant
 * progress, including on a round-over tick. */
static void bomb_drop(GameState* g) {
    BombState*        b       = &g->bomb;
    const AgentState* carrier = &g->agents[b->agent];
    b->x                      = carrier->x;
    b->y                      = carrier->y;
    b->z                      = carrier->z;
    bomb_set(b, BOMB_DROPPED, -1, 0);
}

/* CARRIED -> PLANTING: the carrier starts planting from 0. */
static void bomb_begin_plant(GameState* g) {
    bomb_set(&g->bomb, BOMB_PLANTING, g->bomb.agent, 0);
}

/* PLANTING -> CARRIED: progress is lost. */
static void bomb_cancel_plant(GameState* g) {
    bomb_set(&g->bomb, BOMB_CARRIED, g->bomb.agent, 0);
}

/* PLANTING -> PLANTED where the planter stands; the countdown starts. */
static void bomb_finish_plant(GameState* g, const StaticData* sd, const AgentState* planter) {
    BombState* b  = &g->bomb;
    b->area_idx   = planter->area_idx;
    b->x          = planter->x;
    b->y          = planter->y;
    b->z          = planter->z;
    b->ticks_left = sd->bomb_timer;
    bomb_set(b, BOMB_PLANTED, -1, 0);
}

/* PLANTED -> DEFUSING by agent i from 0. */
static void bomb_begin_defuse(GameState* g, int i) {
    bomb_set(&g->bomb, BOMB_DEFUSING, i, 0);
}

/* DEFUSING -> PLANTED: progress is lost. */
static void bomb_cancel_defuse(GameState* g) {
    bomb_set(&g->bomb, BOMB_PLANTED, -1, 0);
}

/* DEFUSING -> DEFUSED, keeping who defused and their tick count. */
static void bomb_finish_defuse(GameState* g) {
    g->bomb.phase = BOMB_DEFUSED;
}

/* PLANTED/DEFUSING -> DETONATED. An interrupted defuse's progress is discarded. */
static void bomb_detonate(GameState* g) {
    bomb_set(&g->bomb, BOMB_DETONATED, -1, 0);
}

/* NULL if GameState.bomb matches the cs2_types.h table, else the first rule it
 * breaks. Reachable states always pass; this exists for states assembled by
 * hand through the Python overlay (tests, scripted setups). */
static const char* bomb_state_error(const GameState* g, const StaticData* sd) {
    const BombState* b     = &g->bomb;
    int              on_t  = b->agent >= 0 && b->agent < TEAM_SIZE;
    int              on_ct = b->agent >= TEAM_SIZE && b->agent < N_AGENTS;
    int site      = b->area_idx >= 0 && b->area_idx < sd->N && sd->bombsite_by_idx[b->area_idx];
    int no_pos    = b->x == 0.0f && b->y == 0.0f && b->z == 0.0f;
    int unplanted = b->area_idx == INVALID_AREA_IDX && b->ticks_left == 0;
    switch (b->phase) {
        case BOMB_CARRIED:
            if (!on_t || b->progress != 0 || !unplanted || !no_pos)
                return "CARRIED needs a T carrier, progress 0, no area/position/countdown";
            return NULL;
        case BOMB_PLANTING:
            if (!on_t || b->progress < 0 || !unplanted || !no_pos)
                return "PLANTING needs a T carrier, progress >= 0, no area/position/countdown";
            return NULL;
        case BOMB_DROPPED:
            if (b->agent != -1 || b->progress != 0 || !unplanted)
                return "DROPPED needs agent -1, progress 0, no area/countdown";
            return NULL;
        case BOMB_PLANTED:
            if (b->agent != -1 || b->progress != 0 || !site)
                return "PLANTED needs agent -1, progress 0 and a bombsite area";
            return NULL;
        case BOMB_DEFUSING:
            if (!on_ct || b->progress < 0 || !site)
                return "DEFUSING needs a CT defuser, progress >= 0 and a bombsite area";
            return NULL;
        case BOMB_DEFUSED:
            if (!on_ct || b->progress < 0 || !site || !g->round_over)
                return "DEFUSED needs a CT defuser, a bombsite area and round_over";
            return NULL;
        case BOMB_DETONATED:
            if (b->agent != -1 || b->progress != 0 || !site || b->ticks_left > 0 || !g->round_over)
                return "DETONATED needs agent -1, progress 0, a bombsite area, ticks_left <= 0 "
                       "and round_over";
            return NULL;
        default:
            return "bomb.phase is not a BombPhase value";
    }
}

/* ── Per-tick driver ─────────────────────────────────────────────────────── */

/* Bomb transitions run after combat/elimination. Drop is deliberately outside
 * the round-over guards; pickup, USE, defuse invalidation and clocks are not. */
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
    BombState*  b  = &g->bomb;

    /* A dead carrier drops before defuse invalidation and pickup, including
     * after elimination. The drop also ends a plant in progress, which is how
     * a planter's death frees the plant (finding 7, 2026-07-06 adversarial
     * review: a lock frozen on a dead planter bricked planting for the round).
     * No plant_interrupted penalty on death — dying is already penalized. */
    int carrier = bomb_carrier(g);
    if (carrier >= 0 && !g->agents[carrier].alive) {
        bomb_drop(g);
    }

    /* The defuser resets on death, leaving the bomb's area, or USE release
     * (unlike the planter, who only pauses on release). */
    if (!g->round_over && b->phase == BOMB_DEFUSING) {
        AgentState* def = &g->agents[b->agent];
        if (!def->alive || def->area_idx != b->area_idx ||
            actions[b->agent * ACTION_DIM + HEAD_USE] == 0) {
            bomb_cancel_defuse(g);
        }
    }

    if (!g->round_over) {
        for (int i = 0; i < N_AGENTS; i++) {
            AgentState* a = &g->agents[i];
            if (!a->alive || actions[i * ACTION_DIM + HEAD_USE] == 0) {
                continue;
            }

            if (bomb_can_plant(sd, g, i)) {
                /* One-time bombsite entry bonus */
                if (!g->bombsite_entered[i]) {
                    g->bombsite_entered[i]  = 1;
                    bombsite_entry_bonus[i] = 1;
                }
                /* i is the carrier, so the phase is CARRIED or PLANTING by i.
                 * Releasing USE pauses: PLANTING persists with its progress. */
                if (b->phase == BOMB_CARRIED) {
                    bomb_begin_plant(g);
                }
                b->progress++;
                plant_progress_reward[i] =
                    sd->reward_plant_progress_scale; /* per-tick plant progress reward */
                if (b->progress >= sd->bomb_plant_time) {
                    bomb_finish_plant(g, sd, a);
                    *bomb_just_planted = 1;
                    *bomb_planter_id   = i;
                    ss->bomb_planted   = 1;
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
            } else if (b->phase == BOMB_PLANTING && b->agent == i) {
                /* A live planter using off-site interrupts. Releasing USE still
                 * pauses, and death is the drop above, without a penalty. */
                if (b->progress > 0) {
                    plant_interrupted[i] = 1; /* interrupted plant penalty applied post-memset */
                }
                bomb_cancel_plant(g);
            } else if (bomb_can_defuse(g, a)) {
                int defuse_time = a->has_kit ? sd->bomb_defuse_kit : sd->bomb_defuse_time;
                if (b->phase == BOMB_PLANTED) {
                    bomb_begin_defuse(g, i);
                }
                if (b->phase == BOMB_DEFUSING && b->agent == i) {
                    b->progress++;
                    if (b->progress >= defuse_time) {
                        bomb_finish_defuse(g);
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

    /* Dropped-bomb pickup: alive T agents auto-pick up if within 32 units;
     * a tie goes to the later index. */
    if (b->phase == BOMB_DROPPED && !g->round_over) {
        float best_dist = 32.0f * 32.0f; /* compare dist_sq to radius_sq */
        int   best_t    = -1;
        for (int i = 0; i < TEAM_SIZE; i++) {
            AgentState* a = &g->agents[i];
            if (!a->alive)
                continue;
            float dx = a->x - b->x;
            float dy = a->y - b->y;
            float d  = dx * dx + dy * dy;
            if (d <= best_dist) {
                best_dist = d;
                best_t    = i;
            }
        }
        if (best_t >= 0) {
            bomb_give(g, best_t);
        }
    }

    if (bomb_ticking(g) && !g->round_over) {
        b->ticks_left--;
        if (b->ticks_left <= 0) {
            bomb_detonate(g);
            g->round_over = 1;
            g->winner     = 0;
        }
    }

    if (g->round_ticks_left <= 0 && !g->round_over && !bomb_planted(g)) {
        g->round_over = 1;
        g->winner     = -1;
        ss->timed_out = 1;
        es->timed_out = 1;
    }
}
