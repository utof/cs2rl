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
 * Source SDK (movevars_shared, gamemovement.cpp) and widely cited movement
 * analyses (adrianb.io, SourceRuns wiki). At 16 Hz the ground ramp still
 * takes ~0.18s (≈3 ticks) to reach weapon max speed — same wall clock as
 * 64 Hz Source. The air formula is tickrate-invariant because `addspeed`
 * is capped by a fixed wishspeed (30), not a per-tick rate. */
#define SV_ACCELERATE_CS    5.5f        /* ground acceleration coefficient               */
#define SV_FRICTION_CS      5.2f        /* ground friction coefficient                   */
#define SV_STOPSPEED_CS     80.0f       /* friction floor — amplifies decel below it     */
#define SV_AIRACCELERATE_CS 12.0f       /* air acceleration coefficient                  */
#define SV_AIR_MAX_WISHSPD  30.0f       /* wishspeed clamp for air addspeed — bhop key   */
#define SV_GRAVITY_CS       800.0f      /* units/s² downward                             */
#define SV_JUMP_IMPULSE_CS  301.993377f /* sqrt(2 * g * 57u) — 57u target jump height */

/* Sim tick duration (seconds). The env runs at 16 Hz. Keeping this local
 * avoids a cross-header coupling for a single constant used only here. */
#define DT_SIM_MOVE (1.0f / 16.0f)

/* Resolve an attempted XY position against the nav-mesh raster.
 *
 * Returns the destination area index if (tx, ty) is walkable from a->area_idx
 * (i.e. on-grid, on-mesh, and the same area or an adjacency-graph neighbour),
 * or -1 if the attempted step hits a wall / leaves the mesh.
 *
 * Shared by the axis-split wall-slide logic in process_movement: we first try
 * the full diagonal step, and on block we retry each axis separately — so
 * touching a wall while moving diagonally into it preserves the tangential
 * velocity component instead of producing a full stop. */
static inline int _resolve_xy_collision(StaticData* sd, const AgentState* a, float tx, float ty) {
    int gx = (int)((tx - sd->grid_x_min) * sd->grid_inv_cell);
    int gy = (int)((ty - sd->grid_y_min) * sd->grid_inv_cell);
    if (gx < 0 || gx >= sd->grid_w || gy < 0 || gy >= sd->grid_h)
        return -1;
    int target_idx = sd->raster_grid[gy * sd->grid_w + gx];
    if (target_idx < 0)
        return -1;
    if (target_idx != a->area_idx && !sd->adjacency[a->area_idx * sd->N + target_idx])
        return -1;
    return target_idx;
}

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

        /* Crouch (head 6) — hold-to-crouch, matching CS's +duck behaviour.
         * crouch_act directly sets the state each tick: holding the key
         * keeps you crouched, releasing stands you back up. */
        int crouch_act  = actions[i * ACTION_DIM + 6];
        a->is_crouching = (crouch_act == 1) ? 1 : 0;
        count_action(ss->action_crouch, es->action_crouch, crouch_act, 2);

        int valid_dir = (move_dir >= 1 && move_dir <= 8);
        int jump_act  = actions[i * ACTION_DIM + 7];
        count_action(ss->action_jump, es->action_jump, jump_act, 2);

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
            a->vz = 0.0f;
            continue;
        }

        /* Tick the jump cooldown regardless of input — so even if the bot
         * spams jump it can't re-jump faster than jump_cd allows. We use 0
         * cooldown in v1 for full bhop parity with CS's competitive default
         * behaviour (the anti-bhop 1.1x clamp is not modelled yet). */
        if (a->jump_cd > 0)
            a->jump_cd--;

        /* Compute world-space wishdir from the facing-local bin + agent
         * facing. Shared by both ground and air accel paths so W+D in the
         * air behaves symmetrically with W+D on the ground. */
        float wx = 0.0f, wy = 0.0f;
        if (valid_dir) {
            float facing = a->human_controlled ? a->aim_rad : a->facing;
            float fx     = _LOCAL_MOVE_X[move_dir];
            float fy     = _LOCAL_MOVE_Y[move_dir];
            float ca     = cosf(facing);
            float sa     = sinf(facing);
            wx           = fy * ca - fx * sa;
            wy           = fy * sa + fx * ca;
        }

        float vel_x     = a->vx;
        float vel_y     = a->vy;
        float vel_z     = a->vz;
        float wishspeed = get_move_speed(a);

        /* Jump initiation — press-edge only (integer bin). Gated on ground,
         * no cooldown, not crouching. Matches Source's CheckJumpButton: the
         * impulse overwrites vz (not additive), and horizontal velocity is
         * preserved for running-jump continuity. */
        if (jump_act == 1 && !a->is_airborne && a->jump_cd == 0 && !a->is_crouching) {
            vel_z          = SV_JUMP_IMPULSE_CS;
            a->is_airborne = 1;
            /* jump_cd stays 0 to permit bhop re-jumps on landing tick. */
        }

        if (!a->is_airborne) {
            /* ── Ground path ───────────────────────────────────────────────
             * 1) Friction (always, regardless of input — releasing keys
             *    smoothly decelerates instead of halting).
             * 2) Ground accel toward wishdir, capping the projection onto
             *    wishdir at the agent's weapon wishspeed. */
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
            if (valid_dir) {
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
        } else {
            /* ── Air path ──────────────────────────────────────────────────
             * Source's leapfrog gravity: half-step before accel, half-step
             * after integration. Keeps apex timing accurate at our coarse
             * 16 Hz tick (full-step introduces ~3% apex-height error).
             *
             * Air accelerate is the bhop/air-strafe mechanic: the
             * per-tick addspeed budget is capped by `min(wishspeed, 30)`,
             * so once your velocity's projection onto wishdir reaches 30
             * u/s, no more speed is added in that direction — but motion
             * perpendicular to velocity keeps accelerating, which is how
             * strafing into mouse sweeps compounds forward speed. The
             * accel *rate* uses the full uncapped wishspeed. */
            vel_z -= 0.5f * SV_GRAVITY_CS * DT_SIM_MOVE;

            if (valid_dir) {
                float wishspd_capped =
                    wishspeed < SV_AIR_MAX_WISHSPD ? wishspeed : SV_AIR_MAX_WISHSPD;
                float currentspeed = vel_x * wx + vel_y * wy;
                float addspeed     = wishspd_capped - currentspeed;
                if (addspeed > 0.0f) {
                    float accelspeed = SV_AIRACCELERATE_CS * DT_SIM_MOVE * wishspeed;
                    if (accelspeed > addspeed)
                        accelspeed = addspeed;
                    vel_x += accelspeed * wx;
                    vel_y += accelspeed * wy;
                }
            }
        }

        /* ── Integrate position. XY/nav-grid collision logic is shared
         * between ground and air. Z integrates freely (flat world, z=0
         * ground plane); landing is resolved after. */
        float tx = a->x + vel_x * DT_SIM_MOVE;
        float ty = a->y + vel_y * DT_SIM_MOVE;
        float tz = a->z + vel_z * DT_SIM_MOVE;

        /* Axis-split collision / wall sliding.
         *
         * First try the full diagonal step. If that's blocked, retry each
         * axis in isolation and keep whichever component is unobstructed —
         * so hugging a wall while pressing forward slides along the wall
         * (tangential velocity preserved, normal velocity zeroed) instead
         * of producing a dead-stop. Only when BOTH single-axis moves also
         * fail (inside-corner or agent already wedged) do we register a
         * blocked move and zero horizontal velocity entirely. */
        int target_idx = _resolve_xy_collision(sd, a, tx, ty);
        if (target_idx >= 0) {
            a->x        = tx;
            a->y        = ty;
            a->area_idx = target_idx;
        } else {
            int idx_x = _resolve_xy_collision(sd, a, tx, a->y);
            int idx_y = _resolve_xy_collision(sd, a, a->x, ty);
            if (idx_x >= 0) {
                /* Slide along X: tangent to a horizontal wall. */
                a->x        = tx;
                a->area_idx = idx_x;
                vel_y       = 0.0f;
            } else if (idx_y >= 0) {
                /* Slide along Y: tangent to a vertical wall. */
                a->y        = ty;
                a->area_idx = idx_y;
                vel_x       = 0.0f;
            } else {
                /* Fully stuck: count as a blocked move and stop. */
                if (a->team == 0) {
                    ss->blocked_moves_t++;
                    es->blocked_moves_t++;
                } else {
                    ss->blocked_moves_ct++;
                    es->blocked_moves_ct++;
                }
                vel_x = 0.0f;
                vel_y = 0.0f;
            }
        }

        /* Apply second half of gravity (leapfrog split) so velocity at
         * start of next tick is correctly phase-aligned with position. */
        if (a->is_airborne)
            vel_z -= 0.5f * SV_GRAVITY_CS * DT_SIM_MOVE;

        /* Landing: the world is currently flat at z=0. When the airborne
         * agent's integrated z dips to or below the floor with non-positive
         * vz, snap to ground and clear airborne state. */
        a->z  = tz;
        a->vx = vel_x;
        a->vy = vel_y;
        a->vz = vel_z;
        if (a->is_airborne && a->z <= 0.0f && a->vz <= 0.0f) {
            a->z           = 0.0f;
            a->vz          = 0.0f;
            a->is_airborne = 0;
            /* No cooldown — bhop-style re-jump permitted on landing tick. */
        }

        /* Footstep audibility: moving at ≥16 u/s on the ground, not crouching.
         * Airborne agents don't emit footsteps (they're in the air). */
        float sp_sq = vel_x * vel_x + vel_y * vel_y;
        if (!a->is_crouching && !a->is_airborne && sp_sq > (16.0f * 16.0f))
            a->is_moving = 1;
    }
}
