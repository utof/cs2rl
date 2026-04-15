#pragma once
#include "cs2_types.h"
#include "cs2_weapons.h"

static inline void
count_action(int32_t* step_counts, int32_t* episode_counts, int value, int size) {
    if ((unsigned int)value < (unsigned int)size) {
        step_counts[value]++;
        episode_counts[value]++;
    }
}

/* Facing-local unit vectors for 8-bin movement input, used by the human
 * (keyboard) path: (right, forward) in the agent's own frame. Order matches
 * _wasd_to_local_bin in cs2_input.h: 1=W, 2=WD, 3=D, 4=SD, 5=S, 6=SA, 7=A, 8=WA.
 * Index 0 is a placeholder for "no input" and never read.
 *
 * Diagonals are pre-normalised so diagonal WASD doesn't move sqrt(2) faster
 * than cardinals (matches Source's wishdir normalisation). */
static const float _LOCAL_MOVE_X[9] = {
    0.0f,
    0.0f,
    0.70710678f,
    1.0f,
    0.70710678f,
    0.0f,
    -0.70710678f,
    -1.0f,
    -0.70710678f,
};
static const float _LOCAL_MOVE_Y[9] = {
    0.0f,
    1.0f,
    0.70710678f,
    0.0f,
    -0.70710678f,
    -1.0f,
    -0.70710678f,
    0.0f,
    0.70710678f,
};

static void process_movement(Dust2Env* env, const int32_t* actions, StepStats* ss, StepStats* es) {
    StaticData* sd = env->sd;
    GameState*  g  = &env->game;

    for (int i = 0; i < N_AGENTS; i++) {
        AgentState* a = &g->agents[i];
        if (!a->alive)
            continue;

        int move_dir = actions[i * ACTION_DIM + 0];
        count_action(ss->action_move, es->action_move, move_dir, 9);

        /* Crouch toggle (head 6) */
        int crouch_act = actions[i * ACTION_DIM + 6];
        if (crouch_act == 1 && a->crouch_cd == 0) {
            a->is_crouching = !a->is_crouching;
            a->crouch_cd    = CROUCH_COOLDOWN_TICKS;
        }
        count_action(ss->action_crouch, es->action_crouch, crouch_act, 2);

        if (move_dir == 0) {
            a->vx = 0.0f;
            a->vy = 0.0f;
            continue;
        }

        if (move_dir < 1 || move_dir > 8 || a->area_idx < 0) {
            if (a->team == 0) {
                ss->blocked_moves_t++;
                es->blocked_moves_t++;
            } else {
                ss->blocked_moves_ct++;
                es->blocked_moves_ct++;
            }
            a->vx = 0.0f;
            a->vy = 0.0f;
            continue;
        }

        /* Per-tick world displacement for this movement input.
         *
         * Human path: move_dir is a facing-local bin (see _wasd_to_local_bin).
         *   The exact world direction = local (fx,fy) rotated by aim_rad, so
         *   holding W while turning with the mouse moves continuously where
         *   the camera is pointing (no 45° compass snap).
         *
         *   World rotation matches cs2_render.h update_camera convention:
         *     forward_world = (cos(aim), sin(aim))
         *     right_world   = (-sin(aim),  cos(aim))
         *
         * Bot path: move_dir indexes the nav.py-baked world-space bins in
         *   sd->delta_x/delta_y — unchanged, keeps RL action semantics intact.
         */
        float move_speed = get_move_speed(a);
        float scale      = move_speed / 250.0f;
        float dx_world, dy_world;
        if (a->human_controlled) {
            float fx = _LOCAL_MOVE_X[move_dir];
            float fy = _LOCAL_MOVE_Y[move_dir];
            float ca = cosf(a->aim_rad);
            float sa = sinf(a->aim_rad);
            /* Per-tick step: 15.625 u at wishspeed 250 (MOVE_SPEED*DT). */
            const float step = 15.625f;
            dx_world         = (fy * ca - fx * sa) * step * scale;
            dy_world         = (fy * sa + fx * ca) * step * scale;
        } else {
            dx_world = sd->delta_x[move_dir] * scale;
            dy_world = sd->delta_y[move_dir] * scale;
        }
        float tx = a->x + dx_world;
        float ty = a->y + dy_world;

        int gx = (int)((tx - sd->grid_x_min) * sd->grid_inv_cell);
        int gy = (int)((ty - sd->grid_y_min) * sd->grid_inv_cell);
        if (gx < 0 || gx >= sd->grid_w || gy < 0 || gy >= sd->grid_h) {
            if (a->team == 0) {
                ss->blocked_moves_t++;
                es->blocked_moves_t++;
            } else {
                ss->blocked_moves_ct++;
                es->blocked_moves_ct++;
            }
            a->vx = 0.0f;
            a->vy = 0.0f;
            continue;
        }

        int target_idx = sd->raster_grid[gy * sd->grid_w + gx];
        /* Allow same-area movement (scale < 1 may not leave current area cell) */
        if (target_idx < 0 ||
            (target_idx != a->area_idx && !sd->adjacency[a->area_idx * sd->N + target_idx])) {
            if (a->team == 0) {
                ss->blocked_moves_t++;
                es->blocked_moves_t++;
            } else {
                ss->blocked_moves_ct++;
                es->blocked_moves_ct++;
            }
            a->vx = 0.0f;
            a->vy = 0.0f;
            continue;
        }

        a->x        = tx;
        a->y        = ty;
        a->area_idx = target_idx;
        a->vx       = dx_world;
        a->vy       = dy_world;

        /* Only audible footstep if NOT crouching */
        if (!a->is_crouching)
            a->is_moving = 1;
    }
}
