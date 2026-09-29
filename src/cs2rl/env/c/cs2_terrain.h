/* cs2_terrain.h — Raylib-free ramp AABB / interpolated surface z.
 *
 * Shared by process_movement and the demo draw. Binding includes this
 * via cs2_movement.h. Aim-stick helpers stay in cs2_demo_viz.h so the
 * env include graph does not compile them.
 */
#pragma once
#include <math.h>
#include <stdint.h>

typedef struct {
    float x[4], y[4], z[4]; /* (x0,y0), (x1,y0), (x1,y1), (x0,y1) */
} DemoRampQuad;

/* Same "on the line" vs "strictly past it" as build_solids_from_rooms. */
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
 *       interior-reaches-across test as build_solids_from_rooms). That
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

/* demo_terrain_z — interpolated surface z at (x,y) on area i.
 *
 * What: flat rooms / missing bounds / non-ramp → centroids_z[i]. Ramps
 *       call demo_ramp_quad then bilinear on the four corners; u,v clamped
 *       to [0,1].
 * Why:  process_movement snaps a->z to this so grounded agents sit on the
 *       same slope the renderer draws (not the top-of-ramp centroid).
 * Pitfalls: area_bounds==NULL (dust2 / make_cs2_map) is a no-op via the
 *           data, not a special-case map. area_i<0 returns 0 (do not index
 *           centroids_z[-1]). Degenerate AABB (dx/dy==0) uses u/v=0.
 *           Cliff-guard Δz still uses the top centroid — not this helper.
 */
static inline float demo_terrain_z(int           area_i,
                                   float         x,
                                   float         y,
                                   int           n_areas,
                                   const float*  area_bounds, /* NULL allowed */
                                   const float*  centroids_z,
                                   const int8_t* is_ramp) {
    if (area_i < 0)
        return 0.0f;
    if (area_bounds == NULL || is_ramp == NULL || !is_ramp[area_i])
        return centroids_z[area_i];

    DemoRampQuad q;
    demo_ramp_quad(area_i, n_areas, area_bounds, centroids_z, &q);

    float x0 = q.x[0], y0 = q.y[0];
    float x1 = q.x[1], y1 = q.y[2];
    float dx = x1 - x0;
    float dy = y1 - y0;
    float u  = (dx != 0.0f) ? (x - x0) / dx : 0.0f;
    float v  = (dy != 0.0f) ? (y - y0) / dy : 0.0f;
    if (u < 0.0f)
        u = 0.0f;
    else if (u > 1.0f)
        u = 1.0f;
    if (v < 0.0f)
        v = 0.0f;
    else if (v > 1.0f)
        v = 1.0f;

    /* corners: (x0,y0)=0, (x1,y0)=1, (x1,y1)=2, (x0,y1)=3 */
    return (1.0f - u) * (1.0f - v) * q.z[0] + u * (1.0f - v) * q.z[1] + u * v * q.z[2] +
           (1.0f - u) * v * q.z[3];
}
