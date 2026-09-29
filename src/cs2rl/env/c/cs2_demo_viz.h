/* cs2_demo_viz.h — Raylib-free aim-stick helpers for demo draw.
 *
 * Includes cs2_terrain.h (ramp quad / terrain_z). Binding must not include
 * this file — cs2_movement.h includes terrain only, so aim-stick stays out
 * of the env include graph. Do not include raylib from here.
 */
#pragma once
#include <math.h>
#include <stdint.h>
#include "cs2_terrain.h"

#define CS2_DEMO_VIZ_H 1

/* demo_aim_dir_sim — combat ray direction, no punch.
 *
 * What: d = (cos p · cos y, cos p · sin y, sin p). |d| = 1 by construction.
 * Why:  draw and tests share the combat formula without Raylib; punch is
 *       applied by the caller before passing yaw/pitch.
 * Pitfalls: does not clamp pitch. Recoil clamp ±1.5533 is the caller's job.
 */
static inline void demo_aim_dir_sim(float yaw, float pitch, float* dx, float* dy, float* dz) {
    float cos_p = cosf(pitch);
    *dx         = cos_p * cosf(yaw);
    *dy         = cos_p * sinf(yaw);
    *dz         = sinf(pitch);
}

/* demo_aim_stick_rl — head-origin aim stick in Raylib XYZ.
 *
 * What: start = (x, z+108, y); end = start + length * (dx, dz, dy).
 * Why:  other agents' sticks must tilt with pitch. Sim XY is Raylib XZ
 *       and sim Z is Raylib Y; the draw rig puts the head 108 above feet.
 * Pitfalls: 108 is the draw-rig head, not EYE_HEIGHT_STAND (48). Forgetting
 *           the dy/dz swap keeps the stick flat in Raylib Y.
 */
static inline void demo_aim_stick_rl(float x,
                                     float y,
                                     float z,
                                     float yaw,
                                     float pitch,
                                     float length,
                                     float start_xyz[3],
                                     float end_xyz[3]) {
    float dx, dy, dz;
    demo_aim_dir_sim(yaw, pitch, &dx, &dy, &dz);
    start_xyz[0] = x;
    start_xyz[1] = z + 108.0f;
    start_xyz[2] = y;
    end_xyz[0]   = start_xyz[0] + length * dx;
    end_xyz[1]   = start_xyz[1] + length * dz;
    end_xyz[2]   = start_xyz[2] + length * dy;
}
