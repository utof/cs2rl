#pragma once
#include "cs2_types.h"

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
        int         move_dir;
        int         shoot;
        int         use;
        int         last;
        int         gx;
        int         gy;
        int         target_idx;
        float       tx;
        float       ty;

        if (!a->alive) {
            continue;
        }

        move_dir = actions[i * ACTION_DIM + 0];
        shoot    = actions[i * ACTION_DIM + 1];
        use      = actions[i * ACTION_DIM + 2];
        last     = actions[i * ACTION_DIM + 3];

        count_action(ss->action_move, es->action_move, move_dir, 9);
        count_action(ss->action_shoot, es->action_shoot, shoot, 2);
        count_action(ss->action_use, es->action_use, use, 2);
        count_action(ss->action_last, es->action_last, last, 2);

        if (move_dir == 0) {
            continue;
        }

        if (move_dir < 0 || move_dir > 8 || a->area_idx < 0) {
            if (a->team == 0) {
                ss->blocked_moves_t++;
                es->blocked_moves_t++;
            } else {
                ss->blocked_moves_ct++;
                es->blocked_moves_ct++;
            }
            continue;
        }

        tx = a->x + sd->delta_x[move_dir];
        ty = a->y + sd->delta_y[move_dir];
        gx = (int)((tx - sd->grid_x_min) * sd->grid_inv_cell);
        gy = (int)((ty - sd->grid_y_min) * sd->grid_inv_cell);
        if (gx < 0 || gx >= sd->grid_w || gy < 0 || gy >= sd->grid_h) {
            if (a->team == 0) {
                ss->blocked_moves_t++;
                es->blocked_moves_t++;
            } else {
                ss->blocked_moves_ct++;
                es->blocked_moves_ct++;
            }
            continue;
        }

        target_idx = sd->raster_grid[gy * sd->grid_w + gx];
        if (target_idx < 0 || !sd->adjacency[a->area_idx * sd->N + target_idx]) {
            if (a->team == 0) {
                ss->blocked_moves_t++;
                es->blocked_moves_t++;
            } else {
                ss->blocked_moves_ct++;
                es->blocked_moves_ct++;
            }
            continue;
        }

        a->x        = tx;
        a->y        = ty;
        a->area_idx = target_idx;
        /* Smooth turn: clamp facing change to max_turn_speed radians/tick */
        {
            float target = sd->dir_facing[move_dir];
            float diff   = target - a->facing;
            /* Normalise diff to [-π, π] */
            while (diff > (float)M_PI)
                diff -= 2.0f * (float)M_PI;
            while (diff < -(float)M_PI)
                diff += 2.0f * (float)M_PI;
            if (fabsf(diff) <= sd->max_turn_speed) {
                a->facing = target;
            } else {
                a->facing += (diff > 0.0f ? 1.0f : -1.0f) * sd->max_turn_speed;
            }
        }
        a->is_moving = 1;
    }
}
