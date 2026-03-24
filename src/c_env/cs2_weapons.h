#pragma once
#include "cs2_types.h"

/* Compile-time weapon table — indexed by weapon_slot (0=rifle, 1=pistol, 2=knife).
 * T-side rifle = AK-47 stats, CT-side = M4A4 stats, averaged for slot 0.
 * Tick counts at TICK_RATE=16 Hz. */
static const WeaponDef WEAPON_DEFS[3] = {
    /* rifle */  {0, 34.5f, 0.735f,  2, 25,  3, 39, 220.0f, 0.975f},
    /* pistol */ {1, 32.5f, 0.490f,  3, 16,  2, 35, 240.0f, 0.820f},
    /* knife */  {2, 34.0f, 0.850f,  6, -1, -1,  0, 250.0f, 0.000f},
};

/* Returns current move speed in units/second for the agent. */
static inline float get_move_speed(const AgentState* a) {
    float speed = WEAPON_DEFS[a->weapon_slot].move_speed;
    if (a->is_crouching) speed *= 0.34f;
    return speed;
}

/* Initialise ammo for a freshly spawned agent. */
static inline void init_agent_ammo(AgentState* a) {
    for (int s = 0; s < 3; s++) {
        const WeaponDef* def = &WEAPON_DEFS[s];
        a->ammo_clip[s]    = (def->mag_size    >= 0) ? def->mag_size    : -1;
        a->ammo_reserve[s] = (def->reserve_mags >= 0) ? def->reserve_mags : -1;
    }
    a->weapon_slot        = 0; /* start with rifle */
    a->weapon_slot_target = 0;
    a->reload_ticks       = 0;
    a->switch_ticks       = 0;
    a->fire_cd            = 0;
}

/* Decrement per-tick weapon state (call once per alive agent per tick). */
static inline void tick_weapon(AgentState* a) {
    if (a->fire_cd      > 0) a->fire_cd--;
    if (a->reload_ticks > 0) {
        a->reload_ticks--;
        if (a->reload_ticks == 0) {
            /* Complete reload: drop current mag, load fresh one from reserve */
            int slot = a->weapon_slot;
            a->ammo_reserve[slot]--;
            a->ammo_clip[slot] = WEAPON_DEFS[slot].mag_size;
        }
    }
    if (a->switch_ticks > 0) {
        a->switch_ticks--;
        /* switch completes when switch_ticks hits 0 — caller sets weapon_slot after */
    }
    if (a->crouch_cd > 0) a->crouch_cd--;
}

/* Try to start a reload. Returns 1 if started, 0 if invalid. */
static inline int try_start_reload(AgentState* a) {
    int slot = a->weapon_slot;
    const WeaponDef* def = &WEAPON_DEFS[slot];
    if (def->mag_size < 0) return 0;          /* knife: no reload */
    if (a->reload_ticks > 0) return 0;         /* already reloading */
    if (a->switch_ticks > 0) return 0;         /* switching */
    if (a->ammo_clip[slot] >= def->mag_size) return 0;  /* full */
    if (a->ammo_reserve[slot] <= 0) return 0;  /* no reserve */
    a->reload_ticks = def->reload_ticks;
    return 1;
}

/* Try to start a weapon switch to target_slot. Returns 1 if started, 0 if invalid. */
static inline int try_weapon_switch(AgentState* a, int target_slot) {
    if (target_slot < 0 || target_slot > 2) return 0;
    if (target_slot == a->weapon_slot) return 0;
    if (a->switch_ticks > 0) return 0;
    /* Switching cancels an in-progress reload */
    a->reload_ticks       = 0;
    a->switch_ticks       = WEAPON_SWITCH_TICKS;
    a->weapon_slot_target = (int8_t)target_slot;
    return 1;
}
