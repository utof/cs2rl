/* cs2_demo_viz.h — Raylib-free demo geometry. Do not include from cs2_env.h.
 *
 * Demo / tests only. Must not include raylib.h or cs2_render.h so
 * binding.so stays display-free. Task 2 draws with these numbers;
 * this file is the math only.
 */
#pragma once
#include <math.h>

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

typedef struct {
    float x[4], y[4], z[4]; /* (x0,y0), (x1,y0), (x1,y1), (x0,y1) */
} DemoRampQuad;

/* Same "on the line" vs "strictly past it" as build_walls_from_nav. */
#define DEMO_EDGE_EPS 1.0f

/* demo_edge_covers_j — one neighbor vs one edge (the four halfspaces).
 *
 * What: 1 and clipped [out_lo, out_hi] if area j covers edge e of area i
 *       from the exterior (interior-reaches-across, then clip to i's span).
 * Why:  one copy of the if (e==0) blocks for ramp_quad, exterior gaps, and
 *       the lip pass. Longest-overlap and "all covers" both call this.
 * Pitfalls: does not skip j==i (caller must). Overlap must be > DEMO_EDGE_EPS
 *           after clip. Do not require exact bound equality.
 */
static inline int demo_edge_covers_j(int          area_i,
                                     int          j,
                                     int          edge /*0=W 1=E 2=S 3=N*/,
                                     const float* area_bounds,
                                     float*       out_lo,
                                     float*       out_hi) {
    const float EPS  = DEMO_EDGE_EPS;
    float       x0i  = area_bounds[area_i * 4 + 0];
    float       y0i  = area_bounds[area_i * 4 + 1];
    float       x1i  = area_bounds[area_i * 4 + 2];
    float       y1i  = area_bounds[area_i * 4 + 3];
    int         vert = (edge < 2);
    float       line = vert ? (edge == 0 ? x0i : x1i) : (edge == 2 ? y0i : y1i);
    float       slo  = vert ? y0i : x0i;
    float       shi  = vert ? y1i : x1i;
    float       x0j = area_bounds[j * 4 + 0], y0j = area_bounds[j * 4 + 1];
    float       x1j = area_bounds[j * 4 + 2], y1j = area_bounds[j * 4 + 3];
    int         covers = 0;
    float       lo = 0.0f, hi = 0.0f;

    if (edge == 0) { /* left: exterior x < line */
        if (x0j < line - EPS && x1j >= line - EPS) {
            lo     = y0j;
            hi     = y1j;
            covers = 1;
        }
    } else if (edge == 1) { /* right: exterior x > line */
        if (x1j > line + EPS && x0j <= line + EPS) {
            lo     = y0j;
            hi     = y1j;
            covers = 1;
        }
    } else if (edge == 2) { /* bottom: exterior y < line */
        if (y0j < line - EPS && y1j >= line - EPS) {
            lo     = x0j;
            hi     = x1j;
            covers = 1;
        }
    } else { /* top: exterior y > line */
        if (y1j > line + EPS && y0j <= line + EPS) {
            lo     = x0j;
            hi     = x1j;
            covers = 1;
        }
    }
    if (!covers)
        return 0;
    if (lo < slo)
        lo = slo;
    if (hi > shi)
        hi = shi;
    if (hi - lo <= EPS)
        return 0;
    *out_lo = lo;
    *out_hi = hi;
    return 1;
}

/* demo_edge_cover — longest-overlap neighbor on one edge.
 *
 * What: 1 and *out_j / *out_lo / *out_hi if some neighbor covers edge e of
 *       area i from the exterior. Longest overlap wins.
 * Why:  demo_ramp_quad needs one z per edge; walls reuse the same predicate
 *       per-j so exterior subtraction and lips share the four ifs.
 * Pitfalls: several neighbors can cover one edge — longest, not first-found.
 *           No neighbor → return 0 (caller uses this area's z).
 */
static inline int demo_edge_cover(int          area_i,
                                  int          edge /*0=W 1=E 2=S 3=N*/,
                                  int          n_areas,
                                  const float* area_bounds,
                                  int*         out_j,
                                  float*       out_lo,
                                  float*       out_hi) {
    float best_ol = 0.0f;
    int   best_j  = -1;
    float best_lo = 0.0f, best_hi = 0.0f;
    int   j;
    for (j = 0; j < n_areas; j++) {
        float lo, hi;
        if (j == area_i)
            continue;
        if (!demo_edge_covers_j(area_i, j, edge, area_bounds, &lo, &hi))
            continue;
        float ol = hi - lo;
        if (ol > best_ol) {
            best_ol = ol;
            best_j  = j;
            best_lo = lo;
            best_hi = hi;
        }
    }
    if (best_j < 0)
        return 0;
    *out_j  = best_j;
    *out_lo = best_lo;
    *out_hi = best_hi;
    return 1;
}

/* demo_ramp_quad — sloped floor from neighbor edge zs.
 *
 * What: four corners (x0,y0), (x1,y0), (x1,y1), (x0,y1). Each edge takes
 *       the covering neighbor with the longest overlap (EPS=1, same
 *       interior-reaches-across test as build_walls_from_nav). That
 *       neighbor's centroids_z is the edge z; no neighbor → this area's
 *       z. Slope along X if |z_e-z_w| >= |z_n-z_s|, else Y.
 * Why:  ramps must connect the low room to the high room instead of a
 *       flat tile at Y=0 (or disappearing).
 * Pitfalls: several neighbors can cover one edge — longest overlap, not
 *           first-found. Isolated edges keep this area's z. Do not require
 *           exact bound equality; the wall helper already treats a
 *           straddle as coverage.
 */
static inline void demo_ramp_quad(int           area_i,
                                  int           n_areas,
                                  const float*  area_bounds, /* n*4: x0,y0,x1,y1 */
                                  const float*  centroids_z,
                                  DemoRampQuad* out) {
    float x0 = area_bounds[area_i * 4 + 0];
    float y0 = area_bounds[area_i * 4 + 1];
    float x1 = area_bounds[area_i * 4 + 2];
    float y1 = area_bounds[area_i * 4 + 3];
    float z_w, z_e, z_s, z_n;
    {
        int   j;
        float lo, hi;
        z_w = demo_edge_cover(area_i, 0, n_areas, area_bounds, &j, &lo, &hi) ? centroids_z[j]
                                                                             : centroids_z[area_i];
        z_e = demo_edge_cover(area_i, 1, n_areas, area_bounds, &j, &lo, &hi) ? centroids_z[j]
                                                                             : centroids_z[area_i];
        z_s = demo_edge_cover(area_i, 2, n_areas, area_bounds, &j, &lo, &hi) ? centroids_z[j]
                                                                             : centroids_z[area_i];
        z_n = demo_edge_cover(area_i, 3, n_areas, area_bounds, &j, &lo, &hi) ? centroids_z[j]
                                                                             : centroids_z[area_i];
    }

    out->x[0] = x0;
    out->y[0] = y0;
    out->x[1] = x1;
    out->y[1] = y0;
    out->x[2] = x1;
    out->y[2] = y1;
    out->x[3] = x0;
    out->y[3] = y1;

    if (fabsf(z_e - z_w) >= fabsf(z_n - z_s)) {
        out->z[0] = z_w;
        out->z[1] = z_e;
        out->z[2] = z_e;
        out->z[3] = z_w;
    } else {
        out->z[0] = z_s;
        out->z[1] = z_s;
        out->z[2] = z_n;
        out->z[3] = z_n;
    }
}
