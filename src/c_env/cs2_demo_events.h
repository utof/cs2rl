/* cs2_demo_events.h — raylib-free snapshot-diff for cs2_demo juice.
 *
 * Demo target + headless tests only. cs2_env.h must NOT include this file
 * (and this file must not include raylib.h / cs2_render.h) so binding.so
 * stays display/audio-free.
 *
 * Client.prev/curr AgentSnapshot poses cannot drive audio: they lack
 * fired_this_tick / is_airborne / bomb_ticks_left. cs2_demo.c owns a
 * DemoWorldTick pair and copies env.game into them around env_step.
 */
#pragma once
#include <math.h>
#include "cs2_types.h"

typedef struct {
    float x, y, z;
    int   alive, team, is_airborne;
    int   fired_this_tick; /* AgentState.fired_this_tick: set on a shot this env_step */
} DemoAgentTick;

typedef struct {
    DemoAgentTick agents[N_AGENTS]; /* N_AGENTS from cs2_types.h, not a hardcoded 10 */
    int           bomb_planted;
    int           bomb_ticks_left;
} DemoWorldTick;

typedef struct {
    unsigned shot_mask; /* bit i = agent i shot this tick */
    unsigned foot_mask;
    int      plant;
    int      beep;
} DemoEvents;

/* demo_detect_events — compare two 16 Hz world snapshots.
 *
 * What: emit shot/foot bitmasks plus plant/beep flags for the play wrapper.
 * Why:  detection is a pure POD diff so tests compile without Raylib, and
 *       so a later 3 Hz footstep drop / spatial PlaySound can live in the
 *       demo wrapper instead of env_step.
 *
 * Pitfalls:
 *   - Shots are curr.fired_this_tick != 0. Do NOT use "fire_cd just reached
 *     0" — that is cooldown *end*, not a shot. Knife and gun both set the bit.
 *   - Footstep: curr.alive && !curr.is_airborne && hypotf(dx, dy) > 1.0f.
 *     z is ignored. Rate-limit (3 Hz) is NOT here; a walking tick sets the
 *     foot bit every sim tick and the play wrapper drops extras.
 *   - Plant: prev.bomb_planted==0 && curr.bomb_planted==1.
 *   - Beep: curr.bomb_planted && (prev.ticks/p != curr.ticks/p) with integer
 *     division and p = (curr.ticks <= 80) ? 8 : 16. p is chosen from *curr*.
 *   - Plant-complete (0,0 → 1,639) must set BOTH plant and beep. The same
 *     tick assigns bomb_timer then decrements in process_bomb, so the first
 *     planted snapshot is 639, not 640. Do not use 100→99 as that fixture —
 *     100/16 == 99/16 == 6 is the silent in-bucket case.
 *   - Init/reset copies env.game into both prev and curr; that pair must be
 *     silent or spawn looks like a teleport (every agent footsteps).
 */
static inline DemoEvents demo_detect_events(const DemoWorldTick* prev, const DemoWorldTick* curr) {
    DemoEvents ev;
    ev.shot_mask = 0;
    ev.foot_mask = 0;
    ev.plant     = 0;
    ev.beep      = 0;

    int i;
    for (i = 0; i < N_AGENTS; i++) {
        const DemoAgentTick* a = &curr->agents[i];
        const DemoAgentTick* p = &prev->agents[i];
        if (a->fired_this_tick != 0)
            ev.shot_mask |= 1u << i;
        /* 3 Hz drop is the play wrapper's job — emit every hypot>1 walk. */
        if (a->alive && !a->is_airborne && hypotf(a->x - p->x, a->y - p->y) > 1.0f)
            ev.foot_mask |= 1u << i;
    }

    ev.plant = (prev->bomb_planted == 0 && curr->bomb_planted == 1) ? 1 : 0;

    if (curr->bomb_planted) {
        /* Bucket width flips at curr<=80. Integer /, truncates toward zero. */
        int pdiv = (curr->bomb_ticks_left <= 80) ? 8 : 16;
        if (prev->bomb_ticks_left / pdiv != curr->bomb_ticks_left / pdiv)
            ev.beep = 1;
    }
    return ev;
}
