#pragma once
#include "cs2_types.h"

/* Compile-time weapon table — indexed by weapon_slot (0=rifle, 1=pistol, 2=knife).
 * T-side rifle = AK-47 stats, CT-side = M4A4 stats, averaged for slot 0.
 * Tick counts at TICK_RATE=16 Hz.
 * The last four columns are the cone's mrad attributes (WeaponDef), averaged the same way
 * from the CS:GO items_game rows (last CS:GO build, 2023-05-31; research-1c Q1):
 *   rifle  = AK-47 (0.6, 6.41, 4.81, 175.06) and M4A4 (0.6, 4.9, 4.1, 137.88);
 *   pistol = Glock-18 (2, 5.6, 4.2, 10) and USP-S (2.5, 4.9, 3.68, 13.87); knife: none.
 * AK-47 0.6 + 6.41 = 7.01 mrad times its "Accurate Range" is Valve's 15 cm plate radius,
 * which is what fixes x 0.001 as radians. tests/env/c/test_accuracy.py pins the rifle and
 * pistol rows. */
static const WeaponDef WEAPON_DEFS[3] = {
    /* rifle */ {0, 34.5f, 0.735f, 2, 25, 3, 39, 220.0f, 0.975f, 0.6f, 5.655f, 4.455f, 156.47f},
    /* pistol */ {1, 32.5f, 0.490f, 3, 16, 2, 35, 240.0f, 0.820f, 2.25f, 5.25f, 3.94f, 11.935f},
    /* knife */ {2, 34.0f, 0.850f, 6, -1, -1, 0, 250.0f, 0.000f, 0.0f, 0.0f, 0.0f, 0.0f},
};

/* Movement-inaccuracy ramp: the share f of inacc_move_mrad a shot carries rises linearly
 * from 0 at ACCURACY_RAMP_START_FRAC x move_speed to 1 at ACCURACY_RAMP_END_FRAC x
 * move_speed (the weapon's unscoped, uncrouched max speed).
 * START 0.34 is public: Leetify scores a rifle shot as counter-strafed below "34% of your
 * max speed with that weapon (34% is when inaccuracy kicks in)".
 * END is UNVERIFIED: no public source gives where the ramp ends or its curvature (research
 * D3; weapon_debug_spread_show in CS2 could measure it). 1.0 is the simplest monotone choice.
 * Setting END to START makes it a step (CS 1.6's model): every shot at or above 0.34 x
 * move_speed carries the whole inacc_move_mrad. Both were measured (lane D, D-Q2): at 0.35 x
 * move_speed the ramp's rifle cone is 8.6 mrad and every shot hit a standing target 200-520 u
 * away; the step's is the full-speed 162.7 mrad, which hits 0.39 of them at 360 u.
 * PITFALL: at 16 Hz one input tick from standstill reaches 75.6 u/s, just over the
 * rifle's 74.8; the ramp makes that shot nearly free, a step would not. */
#define ACCURACY_RAMP_START_FRAC 0.34f
#define ACCURACY_RAMP_END_FRAC   1.0f
/* Radius law inside the cone: 0 = r = R * u (u uniform in [0, 1): as many shots inside half
 * the radius as outside, centre-weighted); 1 = r = R * sqrt(u) (uniform over the disc).
 * UNVERIFIED which one CS uses (research D2); 0 is the chosen law. Both were measured
 * (lane D, D-Q2): a full-speed rifle shot at a standing target 360 u away hits 0.39 under 0
 * and 0.17 under 1. */
#define ACCURACY_RADIUS_LAW 0

/* accuracy_cone_radius: the cone's maximum radius in rad for a shot from `def` at planar
 * speed `speed` (u/s), crouched or not. spread + stance term + f x the move term, where f
 * is the ramp above. The crouch term is the CS crouch-still value; the speed term uses the
 * uncrouched move_speed for both stances, as the 34% threshold does.
 * KNOWN LIMIT: no air term (research A2 keeps jump masked until there is one, as the Rung 1
 * arena preset does), no
 * per-shot fire penalty, no landing term. Lane D computed CS's penalty (7.4 mrad a shot,
 * recovering over 0.35-0.49 s): a perfectly aimed standing spray still hits with its first
 * four shots out to the arena's 721 u diagonal, so it changes nothing there. */
static inline float accuracy_cone_radius(const WeaponDef* def, int crouched, float speed) {
    float s0 = ACCURACY_RAMP_START_FRAC * def->move_speed;
    float s1 = ACCURACY_RAMP_END_FRAC * def->move_speed;
    /* fmaxf keeps END == START (the step) finite: f jumps from 0 to 1 within 1e-3 u/s. */
    float f    = fminf(fmaxf((speed - s0) / fmaxf(s1 - s0, 1e-3f), 0.0f), 1.0f);
    float mrad = def->spread_mrad + (crouched ? def->inacc_crouch_mrad : def->inacc_stand_mrad) +
                 f * def->inacc_move_mrad;
    return 0.001f * mrad;
}

/* Returns current move speed in units/second for the agent. */
static inline float get_move_speed(const AgentState* a) {
    float speed = WEAPON_DEFS[a->weapon_slot].move_speed;
    if (a->is_crouching)
        speed *= 0.34f;
    return speed;
}

/* Initialise ammo for a freshly spawned agent. */
static inline void init_agent_ammo(AgentState* a) {
    for (int s = 0; s < 3; s++) {
        const WeaponDef* def = &WEAPON_DEFS[s];
        a->ammo_clip[s]      = (def->mag_size >= 0) ? def->mag_size : -1;
        a->ammo_reserve[s]   = (def->reserve_mags >= 0) ? def->reserve_mags : -1;
    }
    a->weapon_slot        = 0; /* start with rifle */
    a->weapon_slot_target = 0;
    a->reload_ticks       = 0;
    a->switch_ticks       = 0;
    a->fire_cd            = 0;
}

/* Decrement per-tick weapon state (call once per alive agent per tick). */
static inline void tick_weapon(AgentState* a) {
    if (a->fire_cd > 0)
        a->fire_cd--;
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
    if (a->crouch_cd > 0)
        a->crouch_cd--;
}

/* Try to start a reload. Returns 1 if started, 0 if invalid. */
static inline int try_start_reload(AgentState* a) {
    int              slot = a->weapon_slot;
    const WeaponDef* def  = &WEAPON_DEFS[slot];
    if (def->mag_size < 0)
        return 0; /* knife: no reload */
    if (a->reload_ticks > 0)
        return 0; /* already reloading */
    if (a->switch_ticks > 0)
        return 0; /* switching */
    if (a->ammo_clip[slot] >= def->mag_size)
        return 0; /* full */
    if (a->ammo_reserve[slot] <= 0)
        return 0; /* no reserve */
    a->reload_ticks = def->reload_ticks;
    return 1;
}

/* Try to start a weapon switch to target_slot. Returns 1 if started, 0 if invalid. */
static inline int try_weapon_switch(AgentState* a, int target_slot) {
    if (target_slot < 0 || target_slot > 2)
        return 0;
    if (target_slot == a->weapon_slot)
        return 0;
    if (a->switch_ticks > 0)
        return 0;
    /* Switching cancels an in-progress reload */
    a->reload_ticks       = 0;
    a->switch_ticks       = WEAPON_SWITCH_TICKS;
    a->weapon_slot_target = (int8_t)target_slot;
    return 1;
}
