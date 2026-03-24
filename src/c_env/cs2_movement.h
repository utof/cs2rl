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

static void process_movement(Dust2Env* env, const int32_t* actions, StepStats* ss,
                             StepStats* es) {
    StaticData* sd = env->sd;
    GameState*  g  = &env->game;

    for (int i = 0; i < N_AGENTS; i++) {
        AgentState* a = &g->agents[i];
        if (!a->alive) continue;

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
            if (a->team == 0) { ss->blocked_moves_t++; es->blocked_moves_t++; }
            else               { ss->blocked_moves_ct++; es->blocked_moves_ct++; }
            a->vx = 0.0f; a->vy = 0.0f;
            continue;
        }

        float scale = get_move_speed(a) / 250.0f;
        float tx = a->x + sd->delta_x[move_dir] * scale;
        float ty = a->y + sd->delta_y[move_dir] * scale;

        int gx = (int)((tx - sd->grid_x_min) * sd->grid_inv_cell);
        int gy = (int)((ty - sd->grid_y_min) * sd->grid_inv_cell);
        if (gx < 0 || gx >= sd->grid_w || gy < 0 || gy >= sd->grid_h) {
            if (a->team == 0) { ss->blocked_moves_t++; es->blocked_moves_t++; }
            else               { ss->blocked_moves_ct++; es->blocked_moves_ct++; }
            a->vx = 0.0f; a->vy = 0.0f;
            continue;
        }

        int target_idx = sd->raster_grid[gy * sd->grid_w + gx];
        /* Allow same-area movement (scale < 1 may not leave current area cell) */
        if (target_idx < 0 ||
            (target_idx != a->area_idx && !sd->adjacency[a->area_idx * sd->N + target_idx])) {
            if (a->team == 0) { ss->blocked_moves_t++; es->blocked_moves_t++; }
            else               { ss->blocked_moves_ct++; es->blocked_moves_ct++; }
            a->vx = 0.0f; a->vy = 0.0f;
            continue;
        }

        a->x        = tx;
        a->y        = ty;
        a->area_idx = target_idx;
        a->vx       = sd->delta_x[move_dir] * scale;
        a->vy       = sd->delta_y[move_dir] * scale;

        /* Only audible footstep if NOT crouching */
        if (!a->is_crouching) a->is_moving = 1;
    }
}
