#pragma once
#include "cs2_types.h"
#include "cs2_weapons.h"

/* Initialise a freshly spawned agent.
 * team: 0=T, 1=CT. The bomb is not per-agent state: env_reset hands it to the
 * round's carrier with bomb_give (cs2_bomb.h) after both teams spawn. */
static inline void init_agent(AgentState* a, int team, uint32_t* rng, const StaticData* sd) {
    /* a is already memset(0) by spawn_team */
    a->team       = (int8_t)team;
    a->hp         = 100;
    a->armor      = 100;
    a->has_helmet = 1;
    a->alive      = 1;
    a->facing     = sd->dir_facing[team == 0 ? 3 : 7];

    if (team == 1) {
        /* Kit assignment: 50% chance (preserve existing logic) */
        a->has_kit = (xorshift32(rng) & 1U) ? 1 : 0;
    }

    init_agent_ammo(a);

    for (int s = 0; s < TEAM_SIZE; s++) {
        a->enemy_mem_idx[s]  = INVALID_AREA_IDX;
        a->enemy_mem_tick[s] = sd->stale_memory_tick;
    }
}
