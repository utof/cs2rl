/* cs2_input.h — Human keyboard/mouse → action buffer mapping. Phase 6. */
#pragma once
#include <math.h>
#include "raylib.h"
#include "cs2_types.h"
#include "cs2_render.h"

/* WASD → facing-local 8-bin movement encoding. No yaw dependency:
 * the actual world-space direction is reconstructed inside process_movement
 * from this bin + the agent's continuous aim_rad, so rotating the mouse
 * while a key is held smoothly rotates the movement direction instead of
 * snapping to 45° world bins.
 *
 * Bin layout (compass in facing-local frame, 0=none):
 *   1 W     2 WD    3 D    4 SD   5 S    6 SA   7 A    8 WA
 *   fwd     fwd-R   right  bk-R   back   bk-L   left   fwd-L
 *
 * Opposing keys (W+S, A+D) cancel to 0 on that axis, matching Source.
 */
static int _wasd_to_local_bin(int w, int a, int s, int d) {
    int fy = (w ? 1 : 0) - (s ? 1 : 0); /* +1=forward, -1=back */
    int fx = (d ? 1 : 0) - (a ? 1 : 0); /* +1=right,   -1=left */

    if (fx == 0 && fy == 0)
        return 0;
    if (fx == 0 && fy > 0)
        return 1; /* W  */
    if (fx > 0 && fy > 0)
        return 2; /* WD */
    if (fx > 0 && fy == 0)
        return 3; /* D  */
    if (fx > 0 && fy < 0)
        return 4; /* SD */
    if (fx == 0 && fy < 0)
        return 5; /* S  */
    if (fx < 0 && fy < 0)
        return 6; /* SA */
    if (fx < 0 && fy == 0)
        return 7; /* A  */
    return 8;     /* WA */
}

/* human_input — called once per sim tick (16 Hz).
 * Camera yaw/pitch are already updated by update_camera() in c_render().
 * This function writes the current yaw to aim_rad and encodes WASD into actions.
 */
void human_input(Client* cl, Dust2Env* env, int32_t* actions) {
    int idx = cl->human_agent_idx;
    if (idx < 0)
        return;

    AgentState* agent = &env->game.agents[idx];

    /* Continuous aim — write exact radians, set bypass flag */
    agent->aim_rad          = cl->yaw;
    agent->human_controlled = 1;

    int32_t* act = actions + idx * ACTION_DIM;

    /* Movement — WASD → facing-local 8-bin. Actual world direction is
     * computed in process_movement() from this bin + agent->aim_rad, which
     * keeps movement continuously aligned with the mouse. env->sd is no
     * longer consulted here (bin layout is fixed, not nav-derived). */
    act[HEAD_MOVE] =
        _wasd_to_local_bin(IsKeyDown(KEY_W), IsKeyDown(KEY_A), IsKeyDown(KEY_S), IsKeyDown(KEY_D));

    /* Batch 3: HEAD_AIM removed from action enum; human aim is set
     * directly via agent->aim_rad (line 55 above) and consumed by
     * cs2_env.h env_step's `if (a->human_controlled)` branch. The
     * continuous_actions float buffer is irrelevant for human agents —
     * the env_step path takes the human branch before reading it. */
    act[HEAD_SHOOT]  = IsMouseButtonDown(MOUSE_BUTTON_LEFT) ? 1 : 0;
    act[HEAD_RELOAD] = IsKeyDown(KEY_R) ? 1 : 0;
    act[HEAD_WEAPON] = IsKeyDown(KEY_ONE) ? 1 : IsKeyDown(KEY_TWO) ? 2 : 0;
    act[HEAD_USE]    = IsKeyDown(KEY_E) ? 1 : 0;
    act[HEAD_CROUCH] = IsKeyDown(KEY_LEFT_CONTROL) ? 1 : 0;
    /* Jump: consume the mousewheel-down press-edge latched in update_camera.
     * We can't poll GetMouseWheelMove directly here — this function only
     * runs at sim-tick rate (~16 Hz) while the wheel event lives on one
     * render frame out of ~4, so direct polling would silently drop 3/4
     * of jump inputs. */
    act[HEAD_JUMP]   = cl->jump_pending ? 1 : 0;
    cl->jump_pending = 0;
}
