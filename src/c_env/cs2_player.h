#pragma once
#include "cs2_types.h"
#include "cs2_weapons.h"

/* Initialise a freshly spawned agent.
 * team: 0=T, 1=CT. bomb_carrier: index within team (0-4) that carries the bomb. */
static inline void init_agent(AgentState* a, int team, int agent_idx_in_team,
                               int bomb_carrier, uint32_t* rng,
                               const StaticData* sd) {
    /* a is already memset(0) by spawn_team */
    a->team      = (int8_t)team;
    a->hp        = 100;
    a->armor     = 100;
    a->has_helmet = 1;
    a->alive     = 1;
    a->facing    = sd->dir_facing[team == 0 ? 3 : 7];

    if (team == 0) {
        a->has_bomb = (agent_idx_in_team == bomb_carrier) ? 1 : 0;
    } else {
        /* Kit assignment: 50% chance (preserve existing logic) */
        a->has_kit = (xorshift32(rng) & 1U) ? 1 : 0;
    }

    init_agent_ammo(a);

    for (int s = 0; s < TEAM_SIZE; s++) {
        a->enemy_mem_idx[s]    = INVALID_AREA_IDX;
        a->enemy_mem_tick[s]   = sd->stale_memory_tick;
    }
}
