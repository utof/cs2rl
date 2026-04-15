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

/* CS2 / CSGO competitive default movement cvars. Values verified against
 * Source SDK (movevars_shared) and widely cited movement analyses
 * (adrianb.io, SourceRuns wiki). At 16 Hz the ramp still takes ~0.18s
 * (≈3 ticks) to reach weapon max speed — same wall clock as 64 Hz Source.
 * The formula is frametime-correct; only the number of discrete speed
 * steps during the ramp differs. */
#define SV_ACCELERATE_CS 5.5f  /* ground acceleration coefficient              */
#define SV_FRICTION_CS   5.2f  /* ground friction coefficient                  */
#define SV_STOPSPEED_CS  80.0f /* friction floor speed — amplifies decel below */

/* Sim tick duration (seconds). The env runs at 16 Hz. Keeping this local
 * avoids a cross-header coupling for a single constant used only here. */
#define DT_SIM_MOVE (1.0f / 16.0f)

/* Facing-local unit vectors for 8-bin movement input: (right, forward) in
 * the agent's own frame. Order matches _wasd_to_local_bin in cs2_input.h:
 *   0=none, 1=W, 2=WD, 3=D, 4=SD, 5=S, 6=SA, 7=A, 8=WA
 * Applied to both humans (via WASD) and bots (via policy action). Index 0
 * is a placeholder and never read.
 *
 * Diagonals are pre-normalised so diagonal input doesn't move sqrt(2) faster
 * than cardinal input (matches Source's wishdir normalisation). */
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

/* ── process_movement ─────────────────────────────────────────────────────
 *
 * Source-style ground movement applied uniformly to every alive agent:
 *
 *   1. Friction bleeds current velocity (even when no input key is held),
 *      with a stopspeed floor that amplifies decel at low speeds.
 *   2. Accelerate adds velocity toward the wishdir, capping the projection
 *      onto wishdir at the agent's current weapon wishspeed. Adding along
 *      a perpendicular direction is unclamped — the mechanism behind CS's
 *      air-strafe, applied on the ground here too for consistency.
 *   3. Position integrates as p += v * dt, subject to the existing nav
 *      raster / adjacency collision test.
 *
 * Wishdir for every agent is a facing-local 8-bin rotated into world by the
 * agent's current facing angle (humans: aim_rad from mouse; bots: a->facing
 * from the aim action). This is a deliberate change from the previous
 * world-compass bin semantics: it means action 1 = "move forward" for any
 * agent, matching real CS "W + mouse-look" behaviour, and it changes bot RL
 * action semantics accordingly.
 *
 * a->vx / a->vy now carry per-SECOND velocity (units/s) across ticks so the
 * ramp-up and friction decay have somewhere to live. Observations normalise
 * by 250 u/s (max wishspeed), so this produces values in ~[0,1] correctly. */
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

        int valid_dir = (move_dir >= 1 && move_dir <= 8);
        if (a->area_idx < 0) {
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

        /* ── Step 1: friction (always, regardless of input). */
        float vel_x = a->vx;
        float vel_y = a->vy;
        float speed = sqrtf(vel_x * vel_x + vel_y * vel_y);
        if (speed > 0.0f) {
            float control  = speed < SV_STOPSPEED_CS ? SV_STOPSPEED_CS : speed;
            float drop     = control * SV_FRICTION_CS * DT_SIM_MOVE;
            float newspeed = speed - drop;
            if (newspeed < 0.0f)
                newspeed = 0.0f;
            float frac  = newspeed / speed;
            vel_x      *= frac;
            vel_y      *= frac;
        }

        /* ── Step 2: accelerate toward facing-local wishdir (if any input).
         * Humans sync aim_rad to the mouse every frame; bots use their
         * discrete a->facing from the aim action. Both rotate the same
         * local (right, forward) vector into world space, matching the
         * cs2_render.h camera convention:
         *   forward_world = (cos, sin),  right_world = (-sin, cos). */
        if (valid_dir) {
            float facing = a->human_controlled ? a->aim_rad : a->facing;
            float fx     = _LOCAL_MOVE_X[move_dir];
            float fy     = _LOCAL_MOVE_Y[move_dir];
            float ca     = cosf(facing);
            float sa     = sinf(facing);
            float wx     = fy * ca - fx * sa;
            float wy     = fy * sa + fx * ca;

            float wishspeed    = get_move_speed(a);
            float currentspeed = vel_x * wx + vel_y * wy;
            float addspeed     = wishspeed - currentspeed;
            if (addspeed > 0.0f) {
                float accelspeed = SV_ACCELERATE_CS * DT_SIM_MOVE * wishspeed;
                if (accelspeed > addspeed)
                    accelspeed = addspeed;
                vel_x += accelspeed * wx;
                vel_y += accelspeed * wy;
            }
        }

        /* ── Step 3: integrate position, then test nav-grid / adjacency. */
        float tx = a->x + vel_x * DT_SIM_MOVE;
        float ty = a->y + vel_y * DT_SIM_MOVE;

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
        /* Allow same-area movement (sub-cell step may not cross a boundary). */
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
        a->vx       = vel_x;
        a->vy       = vel_y;

        /* Footstep audibility: moving at ≥16 u/s (≈1 u/tick) and not crouching.
         * Uses speed rather than "input pressed" so agents coasting on residual
         * post-friction velocity still emit footsteps — matches CS behaviour. */
        float sp_sq = vel_x * vel_x + vel_y * vel_y;
        if (!a->is_crouching && sp_sq > (16.0f * 16.0f))
            a->is_moving = 1;
    }
}
