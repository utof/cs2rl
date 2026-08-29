/* cs2_solids.h — ONE list of solid faces baked from the room quads.
 *
 * Why this file exists
 * --------------------
 * The simple training map used to describe its walls twice: cs2_render.h
 * derived draw cubes from the room quads, while movement and line-of-sight
 * consulted an unrelated 16u occupancy raster. The two drifted — agents
 * walked through drawn walls and shot through drawn corners. Everything now
 * bakes into sd->wall_list once. draw_walls() already reads that list;
 * wiring movement and LoS onto solid_sweep_xy / solid_ray_clear is the
 * follow-up step, which is why those two queries live here and not in the
 * renderer.
 *
 * Raylib-free ON PURPOSE. binding.so may include this header; it must never
 * pull raylib.h or cs2_render.h, so the render-only WALL_DEPTH draw offset
 * stays in draw_walls() and SOLID_WALL_HEIGHT is a local copy of the render
 * WALL_HEIGHT. If those two ever need to differ, they are already separate
 * constants; if they must stay equal, keep this comment honest.
 *
 * What a "solid" is
 * -----------------
 * Each nav area is an axis-aligned rectangle [x0,y0,x1,y1] in
 * sd->area_bounds[idx*4+0..3], with terrain z in sd->centroids_z[idx]. Per
 * edge of each room we ask which neighbours cover that edge from the
 * exterior (demo_edge_covers_j in cs2_terrain.h — the same four halfspaces
 * the ramp quad uses), then split the edge into:
 *
 *   exterior leftover  no neighbour at all  -> full-height wall
 *                                             z0=0, height=SOLID_WALL_HEIGHT+zi
 *   portal             covered, adjacency   -> NOTHING (walkable doorway),
 *                      true                   including ramp connectors AND
 *                                             connected drops such as
 *                                             catwalk->CT-ramp
 *   lip                covered, adjacency   -> z0=min(zi,zj), height=|zi-zj|
 *                      false, zi > zj         (the catwalk face over the
 *                                             bombsite; keeps the overlook
 *                                             open above the drop)
 *   divider            covered, adjacency   -> full-height wall, emitted once
 *                      false, no height       (see the i<j note below)
 *                      difference
 *
 * Adjacency is checked BEFORE the height test, so "the nav graph says you can
 * walk here" always wins over "there is a drop here". The two must agree or
 * nav shaping will steer agents into a face they cannot pass.
 *
 * Walls sit on the ground (z0=0), NOT on centroids_z: a wall floated up to a
 * bombsite's z leaves a triangular void underneath it that you can see and
 * shoot through.
 *
 * Pitfalls
 * --------
 *  - The bake stores the TRUE room edge. The old build_walls_from_nav baked
 *    `eline = line ± WALL_DEPTH/2` so the 8u draw cube sat outside walkable
 *    tile; that offset is now applied by draw_walls() alone. Re-introducing
 *    it here would move every collision plane 4u off the room quad.
 *  - `dust2` (sd->area_bounds == NULL) has no room quads. The bake is a
 *    no-op there and both queries answer "free" / "clear" so the legacy
 *    raster path keeps owning that map.
 *  - build_solids_from_rooms is the ONLY place that allocates wall_list.
 *    It frees any previous list first, so it is safe to re-bake.
 */
#pragma once

#include <limits.h> /* INT_MAX — the wall_list capacity bound            */
#include <math.h>
#include <stdint.h>
#include <stdio.h> /* fprintf — the bake's only failure channel         */
#include <stdlib.h>

#include "cs2_terrain.h" /* demo_edge_covers_j — the shared edge predicate */
#include "cs2_types.h"   /* StaticData, Wall, WallList                     */

/* "On the line" vs "strictly past it". This is DEMO_EDGE_EPS, not a copy of
 * it: the coverage predicate (demo_edge_covers_j) and the gap/lip tests below
 * MUST use the same tolerance, or an interval this file calls "covered" can be
 * one the predicate rejected — a face that exists on one side of the edge and
 * not the other. The alias keeps the name local to this header's vocabulary
 * while making drift impossible.
 *
 * Scope of that argument: it covers the ALONG-EDGE uses only (the Pass A gap
 * tests, which compare positions on the edge line against intervals the
 * predicate produced). The bake also spends this same number as a VERTICAL
 * tolerance — "is this drop flush?" at the zi/zj comparisons in Pass B — and
 * that is an independent second use that merely happens to want the same
 * magnitude. Nothing ties it to DEMO_EDGE_EPS. If you ever need a different
 * flush threshold, split out a SOLID_Z_EPS for those two comparisons; do not
 * read the paragraph above as forbidding it. */
#define SOLID_EPS DEMO_EDGE_EPS

/* Copy of cs2_render.h WALL_HEIGHT. Do not include the render header here.
 * A drift between the two only changes how tall exterior walls are drawn vs
 * how tall they block, so it would show up as shooting over a drawn wall. */
#define SOLID_WALL_HEIGHT 128.0f

/* Collision-capsule height above the agent's feet. Matches the 96u
 * DrawCylinder body in cs2_render.h draw_agents; crouching does not shrink
 * the collision hull in v1 (only the eye height moves, see cs2_combat.h). */
#define SOLID_AGENT_HEIGHT 96.0f

/* Wall.kind — what the face is, which is what decides how it is DRAWN.
 * Collision and LoS treat all three identically. */
#define SOLID_KIND_EXTERIOR 0 /* void behind it: draw cube pushed WALL_DEPTH/2 out */
#define SOLID_KIND_LIP      1 /* height drop between rooms: draw centred on the edge */
#define SOLID_KIND_DIVIDER  2 /* flush but unconnected rooms: draw centred on the edge */

/* One sweep result. `t` is the fraction along the query segment at which the
 * expanded plane is crossed; (nx, ny) is the crossed face's plane normal
 * ORIENTED ALONG THE DIRECTION OF TRAVEL, so it always points the way the
 * mover was heading.
 *
 * That is NOT the face's stored outward normal: solid_sweep_xy derives the
 * sign from dx/dy and never reads w->nx/ny, so this is -Wall.nx/ny whenever
 * the mover approaches from the side opposite the room that emitted the face.
 * Reachable today — the real map's lip is emitted by the catwalk with ny=+1,
 * so an agent in the bombsite walking south into it gets ny=-1.
 * Slide response (v - (v·n)n) is invariant to the sign of n, so callers that
 * only project do not care. A future positional depenetration caller does:
 * pushing along a normal read as "the face's outward normal" would shove the
 * agent INTO the wall half the time. Use Wall.nx/ny if you need the face's
 * own orientation. */
typedef struct {
    float t;      /* hit fraction in [0,1) along the query */
    float nx, ny; /* unit outward normal in sim XY */
} SolidHit;

/* ── bake ────────────────────────────────────────────────────────────────── */

/* Edge-coverage interval plus the covering neighbour (the lip pass needs j). */
typedef struct {
    float lo, hi;
    int   j;
} SolidIv;

/* qsort comparator on interval start. static inline so a TU that includes
 * this header for the queries alone does not trip -Wunused-function. */
static inline int _solid_iv_cmp(const void* a, const void* b) {
    float al = ((const SolidIv*)a)->lo;
    float bl = ((const SolidIv*)b)->lo;
    return (al > bl) - (al < bl);
}

/* _solid_emit — append one axis-aligned face to the list.
 *
 * What: (a, b) is the span along the edge; `line` is the fixed coordinate.
 * Why:  exterior walls, lips and dividers differ only in height/z0/kind, so
 *       they share one emit and one capacity check.
 * Pitfalls: silently drops the face when the list is full — capacity is
 *           sized for the worst case in build_solids_from_rooms, so a drop
 *           here means that bound is wrong, not that the map is unusual.
 *           That "worst case" headroom exists DURING the bake only: the tail
 *           of build_solids_from_rooms reallocs down, leaving
 *           capacity == count. So this is not a general-purpose appender —
 *           call it on a finished list and it drops every single face
 *           without a word. An incremental appender needs its own grow step
 *           first; do not reach for the capacity check here as if it were
 *           one.
 *           Degenerate (b <= a) spans are skipped; every caller already
 *           filters at SOLID_EPS, so this only guards future callers.
 */
static inline void _solid_emit(WallList* wl,
                               int       is_vertical,
                               float     line,
                               float     a,
                               float     b,
                               float     height,
                               float     z0,
                               float     nx,
                               float     ny,
                               int32_t   kind) {
    Wall w;
    if (wl->count >= wl->capacity)
        return;
    if (b - a <= 0.0f)
        return;
    if (is_vertical) {
        w.x0 = line;
        w.y0 = a;
        w.x1 = line;
        w.y1 = b;
    } else {
        w.x0 = a;
        w.y0 = line;
        w.x1 = b;
        w.y1 = line;
    }
    w.height               = height;
    w.z0                   = z0;
    w.nx                   = nx;
    w.ny                   = ny;
    w.kind                 = kind;
    wl->walls[wl->count++] = w;
}

/* free_solids — release the baked list and reset the header.
 *
 * What: free + NULL + count=0 + capacity=0.
 * Why:  a freed pointer with a stale count is a use-after-free waiting for
 *       the next draw_walls / solid_sweep_xy call.
 * Pitfalls: idempotent by design — c_close may run after an explicit free.
 */
static inline void free_solids(StaticData* sd) {
    if (sd == NULL)
        return;
    if (sd->wall_list.walls != NULL)
        free(sd->wall_list.walls);
    sd->wall_list.walls    = NULL;
    sd->wall_list.count    = 0;
    sd->wall_list.capacity = 0;
}

/* build_solids_from_rooms — bake every room face into sd->wall_list.
 *
 * What: see the file header for the exterior / portal / lip / divider split.
 *       Reads sd->N, sd->area_bounds, sd->centroids_z, sd->is_ramp and
 *       sd->adjacency; writes only sd->wall_list.
 * Why:  one list so movement, LoS and the renderer cannot disagree.
 *
 * Pitfalls:
 *  - ONLY malloc site for wall_list. Frees the previous list first, so
 *    calling it twice is safe; calling it on a StaticData whose wall_list
 *    was never zeroed frees a garbage pointer.
 *  - Cannot report failure to its caller. An OOM leaves an EMPTY list, which
 *    every query reads as "no solids anywhere" — collision and LoS turn off
 *    map-wide instead of erroring. It shouts on stderr; if you ever give
 *    this function a return value, that is the first thing to propagate.
 *  - A connected interval emits NOTHING, whatever the height difference.
 *    Checking adjacency first is the whole point: map.py exempts ramp
 *    endpoints from cliff pruning, so catwalk(128)↔CT-ramp(64) is a
 *    walkable 64u drop, and a lip there would be a 64u wall across an edge
 *    nav shaping actively routes through.
 *  - A covered edge with a ramp on EITHER side emits neither a lip nor a
 *    divider. The T-ramp / CT-ramp / stairs are the walk-up affordance; any
 *    face on their covered edges would wall the connector off. The guard has
 *    to look at both rooms because the two sides share one line: testing
 *    only the emitting room let the mirrored orientations through — the LOW
 *    room being the ramp still got a lip dropped over its top, and a divider
 *    still landed on a ramp face whenever the ramp held the higher array
 *    index (the i<jj tie-break made the bug index-order dependent). Pass A
 *    still walls off the uncovered leftovers.
 *  - Only the higher side of a drop emits the lip (zi > zj + SOLID_EPS), so
 *    catwalk↔bombsite is ONE face, not two coincident cubes that z-fight.
 *    The lower side deliberately emits nothing.
 *  - The divider case (covered + not adjacent + no height difference) is
 *    what stops a pruned edge from silently becoming a doorway. It cannot
 *    trigger on SIMPLE_ROOMS today: map.py only prunes adjacency when
 *    |Δz| > MAX_STEP_HEIGHT, which lands in the lip branch instead. It is
 *    emitted once, by the lower-indexed room, for the same anti-duplicate
 *    reason as the lip.
 *  - Not covered: a pruned ramp edge leaves a hole, because neither side of
 *    it emits. map.py exempts ramp endpoints from cliff pruning, so a pruned
 *    ramp edge cannot exist today; revisit if that rule changes.
 */
static inline void build_solids_from_rooms(StaticData* sd) {
    WallList*    wl;
    const float* ab;
    SolidIv*     covs;
    Wall*        shrunk;
    size_t       slots;
    int          N, i, e, j, k;

    if (sd == NULL)
        return;

    wl = &sd->wall_list;
    if (wl->walls != NULL)
        free(wl->walls);
    wl->walls    = NULL;
    wl->count    = 0;
    wl->capacity = 0;

    ab = sd->area_bounds;
    N  = sd->N;
    /* dust2 / any map without room quads: nothing to bake, no allocation. */
    if (ab == NULL || sd->centroids_z == NULL || N <= 0)
        return;

    /* Worst case per edge: (ncov + 1) exterior gaps + ncov lips, ncov < N,
     * so 2N+1 per edge and 4*(2N+1) = 8N+4 per room. 8N*(N+1) covers it.
     * This bound is O(N^2) and wildly loose — 2448 slots (88 KB) for the 33
     * faces the 17-room map actually bakes, and ~12 MB at N=200. It is a
     * transient peak only: the tail of this function reallocs down to
     * wl->count. Sizing it exactly up front would mean running the whole
     * double loop twice.
     *
     * size_t on purpose. `N * 8 * (N + 1)` in int is UB past N≈16k, not a
     * large number, and wl->capacity is an int so a slot count above INT_MAX
     * could not be stored truthfully anyway. Both go down the failure path
     * below rather than wrapping to a small capacity and letting _solid_emit
     * drop faces into a too-small buffer. N=17 today; this is insurance. */
    slots     = (size_t)N * 8u * ((size_t)N + 1u);
    wl->walls = (slots > (size_t)INT_MAX) ? NULL : (Wall*)malloc(slots * sizeof(Wall));
    if (wl->walls == NULL) {
        /* Loud on purpose. A failed bake leaves walls=NULL/count=0, which is
         * bit-for-bit what "this map has no room quads" (dust2) looks like:
         * solid_sweep_xy then returns 0 and solid_ray_clear returns 1 for
         * EVERY query, so collision and line-of-sight quietly vanish
         * map-wide. The only symptom is agents strolling through drawn walls
         * and shooting through drawn corners — nobody debugging that would
         * arrive at malloc. There is no error code to return (callers are
         * make_client and the env init path, neither of which can fail), so
         * stderr is the whole channel. */
        fprintf(stderr,
                "cs2_solids: wall-list bake could not allocate %zu faces for N=%d rooms; "
                "collision and LoS are now DISABLED for this map\n",
                slots,
                N);
        wl->capacity = 0;
        return;
    }
    wl->capacity = (int)slots;

    /* Scratch coverage list. Second allocation inside the single bake
     * function, freed before every return path below; wall_list itself is
     * still allocated exactly once, here. */
    covs = (SolidIv*)malloc((size_t)N * sizeof(SolidIv));
    if (covs == NULL) {
        /* Same silent-disable hazard as above, same reason to shout. */
        fprintf(stderr,
                "cs2_solids: wall-list bake could not allocate the %d-room coverage scratch; "
                "collision and LoS are now DISABLED for this map\n",
                N);
        free(wl->walls);
        wl->walls    = NULL;
        wl->capacity = 0;
        return;
    }

    for (i = 0; i < N; i++) {
        float x0i = ab[i * 4 + 0], y0i = ab[i * 4 + 1];
        float x1i = ab[i * 4 + 2], y1i = ab[i * 4 + 3];
        float zi     = sd->centroids_z[i];
        int   ramp_i = (sd->is_ramp != NULL) ? (sd->is_ramp[i] != 0) : 0;

        /* 4 edges: 0=west(x=x0i), 1=east(x=x1i), 2=south(y=y0i), 3=north(y=y1i). */
        for (e = 0; e < 4; e++) {
            int   is_vertical = (e < 2);
            float line        = is_vertical ? (e == 0 ? x0i : x1i) : (e == 2 ? y0i : y1i);
            float seg_lo      = is_vertical ? y0i : x0i;
            float seg_hi      = is_vertical ? y1i : x1i;
            /* Outward normal of this edge, away from room i. */
            float nx   = is_vertical ? ((e == 0) ? -1.0f : 1.0f) : 0.0f;
            float ny   = is_vertical ? 0.0f : ((e == 2) ? -1.0f : 1.0f);
            int   ncov = 0;
            float cursor;

            /* Collect ALL covering neighbours, not just the longest overlap:
             * the catwalk's bombsite edge is bombsite + CT-ramp, and merging
             * first would lose that split (and with it one of the two lips). */
            for (j = 0; j < N; j++) {
                float lo, hi;
                if (j == i)
                    continue;
                if (!demo_edge_covers_j(i, j, e, ab, &lo, &hi))
                    continue;
                covs[ncov].lo = lo;
                covs[ncov].hi = hi;
                covs[ncov].j  = j;
                ncov++;
            }
            qsort(covs, (size_t)ncov, sizeof(SolidIv), _solid_iv_cmp);

            /* Pass A — exterior: subtract the union of covered intervals and
             * wall off whatever is left, from the ground up. */
            cursor = seg_lo;
            for (k = 0; k < ncov; k++) {
                if (covs[k].lo > cursor + SOLID_EPS)
                    _solid_emit(wl,
                                is_vertical,
                                line,
                                cursor,
                                covs[k].lo,
                                SOLID_WALL_HEIGHT + zi,
                                0.0f,
                                nx,
                                ny,
                                SOLID_KIND_EXTERIOR);
                if (covs[k].hi > cursor)
                    cursor = covs[k].hi;
            }
            if (cursor < seg_hi - SOLID_EPS)
                _solid_emit(wl,
                            is_vertical,
                            line,
                            cursor,
                            seg_hi,
                            SOLID_WALL_HEIGHT + zi,
                            0.0f,
                            nx,
                            ny,
                            SOLID_KIND_EXTERIOR);

            /* Pass B — covered intervals: portal, lip, or divider. */
            for (k = 0; k < ncov; k++) {
                int   jj        = covs[k].j;
                float zj        = sd->centroids_z[jj];
                int   ramp_j    = (sd->is_ramp != NULL) ? (sd->is_ramp[jj] != 0) : 0;
                int   connected = (sd->adjacency != NULL) ? (sd->adjacency[i * N + jj] != 0) : 1;

                /* Portal — the nav graph says i<->jj is walkable, so NOTHING
                 * may be emitted here, not even a lip. map.py exempts ramp
                 * endpoints from cliff pruning (src/map.py, the L9 post-prune
                 * loop), so catwalk(z=128) stays adjacent to CT-ramp(z=64)
                 * across a 64u drop; a lip on that interval would put an
                 * impassable face exactly where nav-distance shaping is
                 * steering agents. The same guard covers any future map with
                 * a connected step of 2..18u — walkable per
                 * SV_MAX_STEP_HEIGHT_CS, but > SOLID_EPS, so the drop test
                 * alone would have walled it off. */
                if (connected)
                    continue;
                /* A ramp is the walk-up affordance: its faces are how you get
                 * on and off it, so a covered edge with a ramp on EITHER side
                 * emits neither a lip nor a divider even when the adjacency
                 * is pruned. Both sides matter because they name the SAME
                 * line — `ramp_i` alone left the mirrored orientations open:
                 * a non-ramp room standing above a ramp still dropped a lip
                 * onto the ramp top, and the divider branch's `i < jj`
                 * tie-break meant a flush pruned edge was walled off or not
                 * depending purely on which room got the lower array index.
                 * (Pass A still walls off the parts of a ramp edge that no
                 * neighbour covers.) */
                if (ramp_i || ramp_j)
                    continue;

                if (zi > zj + SOLID_EPS) {
                    /* This room stands above an unconnected neighbour: emit
                     * the drop face, open above it. */
                    _solid_emit(wl,
                                is_vertical,
                                line,
                                covs[k].lo,
                                covs[k].hi,
                                zi - zj, /* == fabsf(zi - zj) in this branch */
                                zj,      /* == min(zi, zj) in this branch    */
                                nx,
                                ny,
                                SOLID_KIND_LIP);
                } else if (zj <= zi + SOLID_EPS && i < jj) {
                    /* Flush but unconnected: neither side has a drop to
                     * express, so a doorway would appear out of nowhere. */
                    _solid_emit(wl,
                                is_vertical,
                                line,
                                covs[k].lo,
                                covs[k].hi,
                                SOLID_WALL_HEIGHT + zi,
                                0.0f,
                                nx,
                                ny,
                                SOLID_KIND_DIVIDER);
                }
                /* else: zj > zi + EPS — the higher room jj owns that lip.
                 * Nothing to emit here. */
            }
        }
    }

    free(covs);

    /* Shrink to fit, so the steady-state cost is the faces that exist rather
     * than the quadratic upper bound allocated above. Still ONE allocation
     * site for wall_list — realloc keeps the same block identity.
     * Pitfalls: realloc(p, 0) is allowed to free and return NULL, so count==0
     *           is handled separately; and a failed shrink is not an error,
     *           the oversized block stays valid and capacity keeps describing
     *           it truthfully. */
    if (wl->count == 0) {
        free(wl->walls);
        wl->walls    = NULL;
        wl->capacity = 0;
    } else {
        shrunk = (Wall*)realloc(wl->walls, (size_t)wl->count * sizeof(Wall));
        if (shrunk != NULL) {
            wl->walls    = shrunk;
            wl->capacity = wl->count;
        }
    }
}

/* ── queries ─────────────────────────────────────────────────────────────── */

/* _solid_is_horizontal — seg orientation, the one rule every consumer uses.
 *
 * What: 1 for a face that extends along X (y0 == y1), 0 for one along Y.
 * Why:  the bake stores an EXACT axis-aligned unit normal — horizontal faces
 *       get ny = ±1 and nx = 0, vertical ones the reverse — so testing the
 *       normal is exact. The old form compared the endpoints against a 1.0f
 *       magic threshold, which is a length tolerance standing in for an
 *       orientation test, and draw_walls had a second copy of the same
 *       constant.
 * Pitfalls: only valid for walls that came out of _solid_emit. A hand-built
 *           Wall with a zeroed normal classifies as vertical.
 */
static inline int _solid_is_horizontal(const Wall* w) {
    return w->ny != 0.0f;
}

/* _solid_slab_overlaps — does the vertical span [lo, hi] touch the face?
 * Inclusive on [w->z0, w->z0 + w->height]: standing exactly on top of a
 * 128u lip must still be blocked from stepping off it. */
static inline int _solid_slab_overlaps(const Wall* w, float lo, float hi) {
    return !(lo > w->z0 + w->height || hi < w->z0);
}

/* solid_sweep_xy — capsule sweep against the baked faces.
 *
 * What: 1 if the agent capsule [z, z + SOLID_AGENT_HEIGHT] overlaps a face's
 *       vertical slab AND the radius-r XY disk crosses that face's plane,
 *       expanded by r, at some t in [0, 1). `hit` (may be NULL) gets the
 *       smallest such t and that face's outward normal.
 * Why:  movement asks "does this step leave the room through a solid?".
 *       One list, so what blocks you is what is drawn.
 *
 * CALLER CONTRACT — on a hit you MUST slide along `hit->nx/ny` and re-sweep.
 * A plain dest-reject ("blocked, so stay put") freezes wall-hugging movement,
 * and not as an edge case: the moment the start sits inside ANY face's
 * r-band, that face reports t = 0 and wins the min-t race unconditionally,
 * even when a nearer face lies ahead in the travel direction. So an agent
 * that ends up within r of a wall — dropping from z=65 to z=64 next to a 64u
 * lip is enough, since that activates the lip's slab underneath it — would
 * not be able to move ALONG that wall at all; every step comes back blocked
 * and the agent is pinned until something else moves it. The intended
 * response is the usual iterated slide, v' = v - (v·n)n, then sweep again;
 * that is the whole reason SolidHit carries a normal instead of just a t.
 *
 * Corollary: "nearest hit wins" holds only for a start OUTSIDE every r-band.
 * Inside one, an infinitesimal change of input flips which face is reported
 * — from (5,50), a 0.01u sidestep swaps the west face at t=0 for the north
 * face at t=0.95. That is correct for slide response and wrong to lean on as
 * a nearest-surface query; do not build a raycast on top of this (use
 * solid_ray_clear).
 *
 * Pitfalls:
 *  - t < 0 means the mover STARTS inside the r-band, i.e. its hull already
 *    overlaps the face's expanded plane. That is two different situations
 *    and they must not share an answer:
 *      (a) still on the approach side of the face itself, heading at it —
 *          a genuine hit, reported clamped to t = 0. Blanket-rejecting t<0
 *          used to open a hole exactly r wide (12u) along the approach side
 *          of EVERY face: an agent that got within 12u — e.g. by jumping
 *          off the catwalk and landing next to its lip — could then walk
 *          straight through it.
 *      (b) already at or past the face plane in the direction of travel —
 *          a MISS. v1 has no depenetration: an agent that somehow ended up
 *          inside a wall keeps moving rather than being teleported out.
 *          Standing exactly on the plane counts as (b) in both directions,
 *          so an agent pinned on a wall line is never frozen.
 *    Consequence of (b): a mover already past a face can keep going further
 *    past it, but is blocked from coming back through — the return trip is
 *    case (a). This state IS reachable in normal play, so it is a documented
 *    behaviour and not a "can't happen": _solid_slab_overlaps is inclusive,
 *    so anyone airborne over a lip at z inside [z0, z0 + height] is past that
 *    face's plane while the face is still live for them. One-way is the right
 *    answer there — you cleared the lip, you landed past it. What you must
 *    NOT do is lean on it as a general escape hatch from a geometry bug; it
 *    exists because v1 has no depenetration, not as a movement feature.
 *  - Motion parallel to a face never hits it, by construction.
 *  - The hull is a SQUARE at the span ends, not the round disk the "radius-r
 *    XY disk" wording above suggests. The span test is `cross < lo - r ||
 *    cross > hi + r`: extending the ends by r along the face while the plane
 *    test extends by r along the normal is a Minkowski sum with an r-square,
 *    so within r of a face endpoint the face blocks out to r*sqrt(2) (~17u at
 *    AGENT_HULL_RADIUS=12) on the diagonal instead of 12u. Only ever
 *    conservative — it can never let the agent tunnel — but it means door
 *    jambs are square-cornered and clip a little sooner than the stated 12u
 *    radius implies. Rounding the corners means a per-endpoint distance test;
 *    v1 does not need it.
 *  - Axis-aligned faces only. A diagonal seg would be mis-classified by
 *    _solid_is_horizontal; the bake never produces one.
 *  - Faces are planes, not 8u boxes: the draw cube is WALL_DEPTH thick and
 *    offset, the collision plane is not.
 */
static inline int solid_sweep_xy(
    const StaticData* sd, float x0, float y0, float x1, float y1, float z, float r, SolidHit* hit) {
    const WallList* wl;
    float           dx, dy;
    float           best_t = 2.0f, best_nx = 0.0f, best_ny = 0.0f;
    int             i, found = 0;

    if (sd == NULL)
        return 0;
    wl = &sd->wall_list;
    if (wl->walls == NULL || wl->count <= 0)
        return 0;

    dx = x1 - x0;
    dy = y1 - y0;

    for (i = 0; i < wl->count; i++) {
        const Wall* w = &wl->walls[i];
        float       lo, hi, plane, t, cross, n;

        if (!_solid_slab_overlaps(w, z, z + SOLID_AGENT_HEIGHT))
            continue;

        if (!_solid_is_horizontal(w)) {
            /* Vertical face at x = w->x0, spanning y. */
            if (dx > -1e-6f && dx < 1e-6f)
                continue;
            lo = (w->y0 < w->y1) ? w->y0 : w->y1;
            hi = (w->y0 < w->y1) ? w->y1 : w->y0;
            /* Approach side: the disk touches the plane r units early. */
            n     = (dx > 0.0f) ? 1.0f : -1.0f;
            plane = w->x0 - n * r;
            t     = (plane - x0) / dx;
            if (t >= 1.0f)
                continue;
            if (t < 0.0f) {
                /* Started inside the r-band: hit at t=0 unless already at or
                 * past the face plane itself (see pitfall (a)/(b) above). */
                if ((dx > 0.0f) ? (x0 >= w->x0) : (x0 <= w->x0))
                    continue;
                t = 0.0f;
            }
            cross = y0 + t * dy;
            if (cross < lo - r || cross > hi + r)
                continue;
            if (t < best_t) {
                best_t  = t;
                best_nx = n;
                best_ny = 0.0f;
                found   = 1;
            }
        } else {
            /* Horizontal face at y = w->y0, spanning x. */
            if (dy > -1e-6f && dy < 1e-6f)
                continue;
            lo    = (w->x0 < w->x1) ? w->x0 : w->x1;
            hi    = (w->x0 < w->x1) ? w->x1 : w->x0;
            n     = (dy > 0.0f) ? 1.0f : -1.0f;
            plane = w->y0 - n * r;
            t     = (plane - y0) / dy;
            if (t >= 1.0f)
                continue;
            if (t < 0.0f) {
                /* Same clamp as the vertical branch; see the pitfalls above. */
                if ((dy > 0.0f) ? (y0 >= w->y0) : (y0 <= w->y0))
                    continue;
                t = 0.0f;
            }
            cross = x0 + t * dx;
            if (cross < lo - r || cross > hi + r)
                continue;
            if (t < best_t) {
                best_t  = t;
                best_nx = 0.0f;
                best_ny = n;
                found   = 1;
            }
        }
    }

    if (!found)
        return 0;
    if (hit != NULL) {
        hit->t  = best_t;
        hit->nx = best_nx;
        hit->ny = best_ny;
    }
    return 1;
}

/* _solid_span_range — t-window in [0,1] where p0 + t*d stays inside [lo,hi].
 *
 * Only used by the collinear (ray parallel to and on the face plane) branch
 * of solid_ray_clear. Returns 0 when the ray never overlaps the span.
 */
static inline int
_solid_span_range(float p0, float d, float lo, float hi, float* out_a, float* out_b) {
    float ta, tb, tmp;
    if (d > -1e-6f && d < 1e-6f) {
        if (p0 < lo || p0 > hi)
            return 0;
        *out_a = 0.0f;
        *out_b = 1.0f;
        return 1;
    }
    ta = (lo - p0) / d;
    tb = (hi - p0) / d;
    if (ta > tb) {
        tmp = ta;
        ta  = tb;
        tb  = tmp;
    }
    if (ta < 0.0f)
        ta = 0.0f;
    if (tb > 1.0f)
        tb = 1.0f;
    if (ta > tb)
        return 0;
    *out_a = ta;
    *out_b = tb;
    return 1;
}

/* solid_ray_clear — 0 if a face blocks the segment, 1 if it is clear.
 *
 * What: 2D segment/plane intersection, then the interpolated hit z must fall
 *       inside [w->z0, w->z0 + w->height] (inclusive) for the face to block.
 * Why:  LoS and bullets must be stopped by exactly the geometry that is
 *       drawn — and must NOT be stopped by a lip they are looking over,
 *       which is what makes the catwalk an overlook rather than a wall.
 *
 * z0/z1 are EYE heights (cs2_combat.h: 48 standing, 24 crouched, above the
 * agent's feet), not feet z. Passing feet z silently lets shots through the
 * bottom of a lip.
 *
 * Pitfalls:
 *  - A ray lying exactly in a face's plane counts as blocked. Tolerance is
 *    1e-3 world units, so this only fires on genuinely collinear queries
 *    (an agent standing dead on a wall line), not on near misses.
 *  - Endpoint-inclusive in t: a ray that just reaches a face is blocked.
 *  - No area/room filtering. Every baked face is tested; sd->wall_list is
 *    a few dozen segs on the simple map.
 */
static inline int
solid_ray_clear(const StaticData* sd, float x0, float y0, float z0, float x1, float y1, float z1) {
    const WallList* wl;
    float           dx, dy, dz;
    int             i;

    if (sd == NULL)
        return 1;
    wl = &sd->wall_list;
    if (wl->walls == NULL || wl->count <= 0)
        return 1;

    dx = x1 - x0;
    dy = y1 - y0;
    dz = z1 - z0;

    for (i = 0; i < wl->count; i++) {
        const Wall* w   = &wl->walls[i];
        float       top = w->z0 + w->height;
        float       lo, hi, line, t, cross, hz;
        float       along0, alongd, span0, spand;
        int         horizontal = _solid_is_horizontal(w);

        if (!horizontal) {
            line = w->x0;
            lo   = (w->y0 < w->y1) ? w->y0 : w->y1;
            hi   = (w->y0 < w->y1) ? w->y1 : w->y0;
            /* "along" = the axis the ray must cross to reach the plane,
             * "span"  = the axis the face extends along. */
            along0 = x0;
            alongd = dx;
            span0  = y0;
            spand  = dy;
        } else {
            line   = w->y0;
            lo     = (w->x0 < w->x1) ? w->x0 : w->x1;
            hi     = (w->x0 < w->x1) ? w->x1 : w->x0;
            along0 = y0;
            alongd = dy;
            span0  = x0;
            spand  = dx;
        }

        if (alongd > -1e-6f && alongd < 1e-6f) {
            /* Parallel. Only a ray lying in the plane can block. */
            float ta, tb, za, zb, zmin, zmax;
            if (fabsf(along0 - line) > 1e-3f)
                continue;
            if (!_solid_span_range(span0, spand, lo, hi, &ta, &tb))
                continue;
            za   = z0 + ta * dz;
            zb   = z0 + tb * dz;
            zmin = (za < zb) ? za : zb;
            zmax = (za < zb) ? zb : za;
            if (zmax >= w->z0 && zmin <= top)
                return 0;
            continue;
        }

        t = (line - along0) / alongd;
        if (t < 0.0f || t > 1.0f)
            continue;
        cross = span0 + t * spand;
        if (cross < lo || cross > hi)
            continue;
        hz = z0 + t * dz;
        if (hz >= w->z0 && hz <= top)
            return 0;
    }

    return 1;
}
