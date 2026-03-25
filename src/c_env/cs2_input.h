/* cs2_input.h — Human keyboard/mouse → action buffer mapping. Phase 6. */
#pragma once
#include <math.h>
#include "raylib.h"
#include "cs2_types.h"
#include "cs2_render.h"

/* Camera-relative movement: find movement bin (1–8) that best matches
 * the desired direction given WASD keys and current camera yaw.
 * Returns 0 if no keys pressed (stop).
 */
static int _camera_relative_move_bin(int w, int a, int s, int d, float yaw, StaticData* sd) {
    if (!w && !a && !s && !d)
        return 0;

    /* Desired direction in camera space: right=+x, forward=+y */
    float dx = (float)(d - a);
    float dy = (float)(w - s);

    /* Rotate into world space by yaw */
    float world_dx = dx * cosf(yaw) - dy * sinf(yaw);
    float world_dy = dx * sinf(yaw) + dy * cosf(yaw);
    float desired  = atan2f(world_dy, world_dx);

    /* Find closest bin in sd->dir_facing[1..8] */
    int   best      = 1;
    float best_diff = 1e9f;
    for (int i = 1; i < 9; i++) {
        float diff = sd->dir_facing[i] - desired;
        /* Wrap to [-π, π] */
        while (diff > (float)M_PI)
            diff -= 2.0f * (float)M_PI;
        while (diff < -(float)M_PI)
            diff += 2.0f * (float)M_PI;
        float abs_diff = fabsf(diff);
        if (abs_diff < best_diff) {
            best_diff = abs_diff;
            best      = i;
        }
    }
    return best;
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

    /* Movement — camera-relative WASD */
    act[0] = _camera_relative_move_bin(
        IsKeyDown(KEY_W), IsKeyDown(KEY_A), IsKeyDown(KEY_S), IsKeyDown(KEY_D), cl->yaw, env->sd);

    act[1] = 0; /* aim bin unused — continuous aim via agent->aim_rad */
    act[2] = IsMouseButtonDown(MOUSE_BUTTON_LEFT) ? 1 : 0;        /* shoot  */
    act[3] = IsKeyDown(KEY_R) ? 1 : 0;                            /* reload */
    act[4] = IsKeyDown(KEY_ONE) ? 1 : IsKeyDown(KEY_TWO) ? 2 : 0; /* weapon */
    act[5] = IsKeyDown(KEY_E) ? 1 : 0;                            /* use (plant/defuse) */
    act[6] = IsKeyDown(KEY_LEFT_CONTROL) ? 1 : 0;                 /* crouch */
}
