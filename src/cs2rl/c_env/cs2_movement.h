#pragma once
#include "cs2_types.h"
#include "cs2_weapons.h"
#include "cs2_terrain.h"

/* Interpolated surface z at (x,y) on area_idx — same quad the renderer draws.
 * Cliff-guard Δz still uses centroids_z (top), not this. */
static inline float _surface_z(const StaticData* sd, int area_idx, float x, float y) {
    return demo_terrain_z(area_idx, x, y, sd->N, sd->area_bounds, sd->centroids_z, sd->is_ramp);
}

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
/* Source sv_stepsize default — maximum grounded up-step height before cliff guard
 * rejects the move. Must match MAX_STEP_HEIGHT = 18.0 in src/cs2rl/map.py (L9 prune). */
#define SV_MAX_STEP_HEIGHT_CS 18.0f

/* Tolerance (world units) for the is_airborne re-evaluation: an agent is treated as
 * airborne when (z > terrain_z + SV_AIRBORNE_EPS_CS) OR (vz != 0). Without this ε,
 * float noise on z (e.g., from a ground-snap that didn't land exactly on terrain_z
 * due to FP rounding) would flicker is_airborne every tick. Spec §7 risks: raise to
 * 2.0 if standing-still flicker is observed during real training. Tuning this is a
 * one-line change — guard is a named constant precisely so future ops finds it. */
#define SV_AIRBORNE_EPS_CS 1.0f

/* Sim tick duration (seconds). The env runs at 16 Hz. Keeping this local
 * avoids a cross-header coupling for a single constant used only here. */
#define DT_SIM_MOVE (1.0f / 16.0f)

/* Raster cell at (x,y). floorf, not (int)cast: C truncates toward zero, so
 * x ∈ (grid_x_min − cell, grid_x_min) would look like cell 0 and walk
 * through the west/south exterior wall. */
static inline int _raster_at(const StaticData* sd, float x, float y) {
    int gx = (int)floorf((x - sd->grid_x_min) * sd->grid_inv_cell);
    int gy = (int)floorf((y - sd->grid_y_min) * sd->grid_inv_cell);
    if (gx < 0 || gx >= sd->grid_w || gy < 0 || gy >= sd->grid_h)
        return -1;
    return sd->raster_grid[gy * sd->grid_w + gx];
}

/* True if (x,y) is inside area idx's room quad. NULL bounds (dust2) = skip.
 * Inclusive on the max edge so a portal at x=x1 stays in both rooms.
 * idx < 0 is outside — do not treat a miss as inside. */
static inline int _in_area_aabb(const StaticData* sd, int idx, float x, float y) {
    if (sd->area_bounds == NULL)
        return 1;
    if (idx < 0)
        return 0;
    const float* b = sd->area_bounds + idx * 4;
    return (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]);
}

/* Room that contains (x,y). Raster label first (stable on 16-aligned
 * interiors). If that label's quad misses — later rooms overwrite the
 * 16u column that straddles 750/820/1100/1170 — search the other rooms.
 * Returns -1 for true exterior overshoot (catwalk x∈[816,820)). */
static inline int _area_at(const StaticData* sd, float x, float y) {
    int idx = _raster_at(sd, x, y);
    if (idx < 0)
        return -1;
    if (sd->area_bounds == NULL || _in_area_aabb(sd, idx, x, y))
        return idx;
    for (int j = 0; j < sd->N; j++) {
        if (j != idx && _in_area_aabb(sd, j, x, y))
            return j;
    }
    return -1;
}

/* Resolve an attempted XY position against the nav-mesh raster.
 *
 * Returns the destination area index if (tx, ty) is walkable from a->area_idx
 * (i.e. on-grid, on-mesh, and the same area or an adjacency-graph neighbour),
 * or -1 if the attempted step hits a wall / leaves the mesh.
 *
 * Shared by the axis-split wall-slide logic in process_movement: we first try
 * the full diagonal step, and on block we retry each axis separately — so
 * touching a wall while moving diagonally into it preserves the tangential
 * velocity component instead of producing a full stop.
 *
 * Hull (area_bounds != NULL only): the four axis offsets at AGENT_HULL_RADIUS
 * must also resolve via _area_at (raster, then containing room if the
 * label's quad misses — later rooms own the 16u portal column). That
 * keeps the 12u cylinder out of the 8u exterior wall without sealing
 * ramps. Adjacency / cliff stay center-only. Dust2 skips this. */
static inline int _resolve_xy_collision(StaticData* sd, const AgentState* a, float tx, float ty) {
    int target_idx = _area_at(sd, tx, ty);
    if (target_idx < 0)
        return -1;
    if (target_idx != a->area_idx && !sd->adjacency[a->area_idx * sd->N + target_idx])
        return -1;
    /* L11 cliff guard: block grounded up-steps where Δz > SV_MAX_STEP_HEIGHT_CS into a
     * non-ramp target. Mirrors the Python adjacency post-prune (map.py L9) so nav-distance
     * shaping stays consistent with movement enforcement.
     * Down-steps (dz < 0) are always allowed — the ground-snap + airborne paths handle them.
     * Ramp targets are always allowed — they are the explicit walk-up affordance.
     * Note: only is_ramp[target_idx] matters; the SOURCE area's ramp flag is irrelevant.
     * Walking off a ramp onto a cliff is still blocked by Δz; walking from a non-ramp onto
     * a ramp is allowed (canonical "ascend the ramp" case).
     * Airborne agents bypass this guard — they are off-ground; the landing rule (L5) resolves
     * where they touch down. The axis-split sliding block in process_movement calls this
     * helper for each retry, so the guard is inherited for free by all diagonal/axis cases. */
    if (!a->is_airborne) {
        float dz = sd->centroids_z[target_idx] - sd->centroids_z[a->area_idx];
        if (dz > SV_MAX_STEP_HEIGHT_CS && !sd->is_ramp[target_idx]) {
            return -1; /* cliff: reject — agent slides or stops via axis-split in caller */
        }
    }
    /* Simple-map rooms publish area_bounds. Dust2 leaves it NULL so thin
     * nav areas stay point-collided (a 12u hull would seal corridors <24u). */
    if (sd->area_bounds != NULL) {
        const float r     = AGENT_HULL_RADIUS;
        const float hx[4] = {tx + r, tx - r, tx, tx};
        const float hy[4] = {ty, ty, ty + r, ty - r};
        for (int k = 0; k < 4; k++) {
            if (_area_at(sd, hx[k], hy[k]) < 0)
                return -1;
        }
    }
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

        int move_dir = actions[i * ACTION_DIM + HEAD_MOVE];
        count_action(ss->action_move, es->action_move, move_dir, 9);

        /* Crouch (head 6) — hold-to-crouch, matching CS's +duck behaviour.
         * crouch_act directly sets the state each tick: holding the key
         * keeps you crouched, releasing stands you back up. */
        int crouch_act = actions[i * ACTION_DIM + HEAD_CROUCH];
        /* W5 (#156): crouch_enabled is a SIM invariant, not just a policy mask.
         * compute_masks still masks the bin, but env_step takes raw actions —
         * scripted bots (#152), BC replay and tests all bypass the mask, so the
         * enforcement has to live here. Zeroed at the READ, i.e. BEFORE both the
         * is_crouching write below and the count_action feed, so the histogram
         * reports EFFECTIVE actions: gate readers treat action histograms as
         * ground truth of sim behaviour, and a disabled press that still counts
         * would poison that reading. Zeroing after the is_crouching write would
         * green the histogram while leaving the agent crouched; zeroing after
         * count_action would stop the crouch but keep counting attempts. Both
         * are covered by tests/test_stance_flags.py. */
        if (!sd->crouch_enabled)
            crouch_act = 0;
        a->is_crouching = (crouch_act == 1) ? 1 : 0;
        count_action(ss->action_crouch, es->action_crouch, crouch_act, 2);

        int valid_dir = (move_dir >= 1 && move_dir <= 8);
        int jump_act  = actions[i * ACTION_DIM + HEAD_JUMP];
        /* W5 (#156): same invariant as the crouch guard above, but note the
         * geometry is inverted — jump's state effect is ~55 lines DOWN, at the
         * `jump_act == 1 && !a->is_airborne` impulse. "Guard before its state
         * effects" would therefore admit an insertion just above that impulse,
         * which is AFTER this count_action and would leave the jump histogram
         * counting ATTEMPTED jumps forever. Hence: zero at the READ, above the
         * feed. tests/test_stance_flags.py asserts action_jump_1 == 0 under a
         * forced jump at jump_enabled=0 precisely to keep that placement
         * observable — a vel_z-only test passes with the weak placement. */
        if (!sd->jump_enabled)
            jump_act = 0;
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
            /* world = fy * forward + fx * right, with forward = (ca, sa) and
             * right = (sa, -ca) — the CLOCKWISE perpendicular, since yaw is
             * CCW in this x-east/y-north frame. F9 (2026-07-06 adversarial
             * review): the old form (wx = fy*ca - fx*sa; wy = fy*sa + fx*ca)
             * rotated fx onto the CCW/left perpendicular, so D (bin 3,
             * labelled right) moved world-LEFT and A world-RIGHT. Harmless
             * under relabeling-invariant self-play, but wrong for scripted
             * experts / BC demos / deploy key export. Pinned by
             * test_strafe_labels_match_geometry. */
            wx = fy * ca + fx * sa;
            wy = fy * sa - fx * ca;
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

        a->z  = tz;
        a->vx = vel_x;
        a->vy = vel_y;
        a->vz = vel_z;

        /* L5 ground-snap: when grounded, snap a->z to the current area's surface.
         * Ramps use bilinear demo_terrain_z (same quad as the renderer); flat
         * rooms / NULL bounds stay at centroids_z. Two cases:
         *   1. Agent walked onto a new area — pin z to that area's surface.
         *   2. Same area — no-op on flats; on ramps this follows the slope as xy moves.
         * MUST come AFTER _resolve_xy_collision axis-split: area_idx is updated there.
         * MUST come AFTER a->z = tz above: tz is the ballistic z; snap overrides it.
         * The landing block below gates on a->is_airborne, so grounded agents skip it. */
        if (!a->is_airborne) {
            a->z = _surface_z(sd, a->area_idx, a->x, a->y);
        }

        /* L5 landing rule: terrain z is per-area (interpolated on ramps). Agent has
         * touched the surface when integrated z dips at or below it with vz <= 0.
         * Pitfall: a->area_idx is already updated by _resolve_xy_collision, so this
         * is the LANDING area's surface — correct. */
        float terrain_z = _surface_z(sd, a->area_idx, a->x, a->y);
        if (a->is_airborne && a->z <= terrain_z && a->vz <= 0.0f) {
            a->z           = terrain_z;
            a->vz          = 0.0f;
            a->is_airborne = 0;
            /* No cooldown — bhop-style re-jump permitted on landing tick. */
        }

        /* L5 is_airborne ε-guard: an agent above terrain by more than SV_AIRBORNE_EPS_CS
         * (or with non-zero vz) is airborne. The ε prevents float-noise flicker when
         * standing still on an elevated platform (z == terrain_z exactly but tiny
         * integration error). Walking off a platform onto a lower-z area triggers
         * this on the next tick — gravity then pulls them down via the airborne path.
         * MUST come AFTER the landing block: otherwise we'd flip is_airborne back to 1
         * immediately after landing because tz is still mid-tick and vz has a remnant.
         * Pitfall: the `vz != 0.0f` exact-float compare is safe because vz is only ever
         * set to literal 0.0f (in the landing block above) when the grounded path is
         * entered. Modify with care: any future grounded-path mutation of vz (e.g., a
         * "stick to ground" damping) could silently flip agents to airborne via FP noise. */
        {
            float terrain_z_now = _surface_z(sd, a->area_idx, a->x, a->y);
            if (a->z > terrain_z_now + SV_AIRBORNE_EPS_CS || a->vz != 0.0f) {
                a->is_airborne = 1;
            }
        }

        /* Footstep audibility: moving at ≥16 u/s on the ground, not crouching.
         * Airborne agents don't emit footsteps (they're in the air). */
        float sp_sq = vel_x * vel_x + vel_y * vel_y;
        if (!a->is_crouching && !a->is_airborne && sp_sq > (16.0f * 16.0f))
            a->is_moving = 1;
    }
}
