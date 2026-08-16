/* cs2_render.h — Raylib 3D FPS renderer for cs2rl. Phase 6. */
#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "raylib.h"
#include "rlgl.h" /* rlSetClipPlanes — raylib's default far is 1000u, we need more */
#include "cs2_types.h"
/* line_of_sight_2d for the --fog flag in draw_agents. cs2_combat.h is also
 * pulled in transitively through cs2_env.h in the demo TU, but include it
 * directly so the fog feature doesn't depend on header order. */
#include "cs2_combat.h"
/* demo_decay_punch — render-frame view-kick. Header is raylib-free; do
 * not include this from cs2_env.h (binding stays display-free). */
#include "cs2_demo_events.h"

#define PLAYER_EYE_HEIGHT 64.0f  /* eye height above agent.z in world units */
#define WALL_HEIGHT       128.0f /* wall extrusion height                   */
#define WALL_DEPTH        8.0f   /* wall thickness                          */
#define MOUSE_SENSITIVITY 0.002f /* rad/px — tuned with raw per-frame pixel deltas */
#define WINDOW_W          1280
#define WINDOW_H          720

/* Demo-juice audio (P0). Voices live in src/c_env/demo_assets/ because
 * src/c_env/resources is a pufferlib symlink in the parent tree (and is
 * gitignored). build.zig copies the four WAVs to zig-out/bin/resources/. */
#define DEMO_VOICE_SHOT  0
#define DEMO_VOICE_FOOT  1
#define DEMO_VOICE_PLANT 2
#define DEMO_VOICE_BEEP  3
#define DEMO_VOICE_N     4
/* Rotating LoadSoundAlias pool: overlapping shots/feet would cut off if we
 * PlaySound the same Sound twice. Unloaded in c_close, not per PlaySound. */
#define DEMO_ALIAS_N     24
#define DEMO_KILL_FEED_N 4

/* ── AgentSnapshot — interpolation state per agent ─────────────────────── */
typedef struct {
    float x, y, z, facing;
    int   hp, alive, team, has_bomb;
} AgentSnapshot;

/* ── Client — per-window render state ───────────────────────────────────── */
typedef struct {
    int           width, height;
    Camera3D      camera;
    Font          font;
    int           human_agent_idx; /* -1 = spectate */
    float         yaw;             /* radians, accumulated from mouse */
    float         pitch;           /* radians, clamped to ±1.553 (≈±89°) */
    double        last_step_time;  /* GetTime() at last sim tick */
    AgentSnapshot prev[N_AGENTS];
    AgentSnapshot curr[N_AGENTS];
    Vector2       last_mouse;
    int           mouse_init;
    int           mouse_captured;
    /* Render-frame key-edge latches. Raylib's IsKeyPressed only returns true
     * on the single render frame the key transitions down, but input is
     * sampled by human_input at the sim tick (16 Hz — every ~4 render
     * frames). Without a latch, press-edges that land between sim ticks are
     * silently dropped. Each per-frame poll sets the latch; the sim-tick
     * sampler consumes it. */
    int jump_pending;
    /* Fog-of-war debug toggle: when set, draw_agents skips rendering any
     * agent the human's own agent can't see via line_of_sight_2d (the same
     * raycast the C env uses for combat / obs / memory). Lets a human
     * "play as the bot" — see only what the trained policy sees in obs.
     * Set via --fog CLI flag in cs2_demo.c. */
    int fog_enabled;
    /* Snapshot-diff audio. AgentSnapshot cannot drive this (no
     * fired_this_tick / is_airborne / bomb_ticks_left). */
    int    audio_ok;                  /* 1 iff InitAudioDevice + IsAudioDeviceReady */
    float  master_volume;             /* default 1.0; [ / ] nudge via SetMasterVolume */
    double last_footstep_t[N_AGENTS]; /* GetTime() of last *played* foot */
    Sound  snd[DEMO_VOICE_N];
    int    snd_ok[DEMO_VOICE_N];
    Sound  alias_pool[DEMO_ALIAS_N];
    int    alias_used[DEMO_ALIAS_N];
    int    alias_cursor;
    /* Applied once on a local shot in the 16 Hz block. Each render
     * frame decays these and adds them to camera look only — never
     * written into yaw/pitch/aim_rad (human_input copies those). */
    float punch_pitch;
    float punch_yaw;
    /* Alive 1→0 edges recorded on the sim tick. Drawn with a 3 s fade. */
    int    kill_feed_n;
    int    kill_feed_idx[DEMO_KILL_FEED_N];
    int    kill_feed_team[DEMO_KILL_FEED_N];
    double kill_feed_t[DEMO_KILL_FEED_N];
} Client;

/* ─────────────────────────────────────────────────────────────────────────
 * Internal helpers
 * ───────────────────────────────────────────────────────────────────────── */

static float _lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

/* Shortest-angle lerp for radians */
static float _lerp_angle(float a, float b, float t) {
    float diff = b - a;
    while (diff > (float)M_PI)
        diff -= 2.0f * (float)M_PI;
    while (diff < -(float)M_PI)
        diff += 2.0f * (float)M_PI;
    return a + diff * t;
}

/* Copy current AgentState → AgentSnapshot array */
static void _copy_agents_to_snapshot(Dust2Env* env, AgentSnapshot* snap) {
    for (int i = 0; i < N_AGENTS; i++) {
        AgentState* a    = &env->game.agents[i];
        snap[i].x        = a->x;
        snap[i].y        = a->y;
        snap[i].z        = a->z;
        snap[i].facing   = a->facing;
        snap[i].hp       = a->hp;
        snap[i].alive    = a->alive;
        snap[i].team     = a->team;
        snap[i].has_bomb = a->has_bomb;
    }
}

/* ── Wall derivation from nav geometry ──────────────────────────────────
 *
 * Each nav area is an axis-aligned rectangle with bounds [x0,y0,x1,y1] in
 * area_bounds[idx*4+0..3]. For each of the 4 edges of every area, we want
 * to emit a wall only where no other area sits on the exterior side — i.e.
 * the outer boundary of the union of all areas. Everything internal (even
 * when coverage is partial, or when areas overlap in 2D) must not become a
 * wall, since agents can walk across those boundaries.
 *
 * Algorithm (per edge):
 *   1. The edge runs along a single varying axis over [seg_lo, seg_hi].
 *   2. For each other area j, compute the sub-interval of the edge that j
 *      covers from the exterior side (j's interior reaches strictly past
 *      the edge line). Both "j abuts the edge from outside" and "j overlaps
 *      across the edge line" count as coverage.
 *   3. Subtract the union of those intervals from [seg_lo, seg_hi].
 *   4. Emit the remaining gaps as wall segments.
 *
 * Called once from make_client().
 */

/* Edge-coverage interval, used while subtracting neighbor coverage. */
typedef struct {
    float lo, hi;
} _WallIv;

static int _wall_iv_cmp(const void* a, const void* b) {
    float al = ((const _WallIv*)a)->lo;
    float bl = ((const _WallIv*)b)->lo;
    return (al > bl) - (al < bl);
}

static void build_walls_from_nav(StaticData* sd, const float* area_bounds) {
    WallList* wl = &sd->wall_list;
    /* Worst case: each edge can be broken into up to N segments by N-1 gaps.
     * Pad generously — this array is only alive for the lifetime of the demo. */
    wl->capacity = sd->N * 4 * (sd->N + 1);
    wl->walls    = (Wall*)malloc(wl->capacity * sizeof(Wall));
    wl->count    = 0;

    /* Tolerance for "on the edge line" vs. "strictly past it". Nav bounds are
     * stored as exact floats on a coarse grid, so 1.0 unit is safely below any
     * real spatial feature but above float round-off. */
    const float EPS = 1.0f;

    _WallIv* covs = (_WallIv*)malloc((size_t)sd->N * sizeof(_WallIv));

    for (int i = 0; i < sd->N; i++) {
        float x0i = area_bounds[i * 4 + 0], y0i = area_bounds[i * 4 + 1];
        float x1i = area_bounds[i * 4 + 2], y1i = area_bounds[i * 4 + 3];

        /* 4 edges: 0=left(x=x0i), 1=right(x=x1i), 2=bottom(y=y0i), 3=top(y=y1i). */
        for (int e = 0; e < 4; e++) {
            int   is_vertical = (e < 2);
            float line        = is_vertical ? (e == 0 ? x0i : x1i) : (e == 2 ? y0i : y1i);
            float seg_lo      = is_vertical ? y0i : x0i;
            float seg_hi      = is_vertical ? y1i : x1i;

            /* Collect every other area that covers this edge from the exterior
             * side. "Exterior" is the halfspace opposite the area's interior:
             *   left  edge → x <  x0i
             *   right edge → x >  x1i
             *   bottom    → y <  y0i
             *   top       → y >  y1i
             * Area j covers the edge line iff j's interior reaches across it
             * (so j either abuts from outside, x0j==line, or straddles it). */
            int ncov = 0;
            for (int j = 0; j < sd->N; j++) {
                if (j == i)
                    continue;
                float x0j = area_bounds[j * 4 + 0], y0j = area_bounds[j * 4 + 1];
                float x1j = area_bounds[j * 4 + 2], y1j = area_bounds[j * 4 + 3];

                int   covers = 0;
                float lo = 0.f, hi = 0.f;

                if (e == 0) { /* left: exterior x<line */
                    if (x0j < line - EPS && x1j >= line - EPS) {
                        lo     = y0j;
                        hi     = y1j;
                        covers = 1;
                    }
                } else if (e == 1) { /* right: exterior x>line */
                    if (x1j > line + EPS && x0j <= line + EPS) {
                        lo     = y0j;
                        hi     = y1j;
                        covers = 1;
                    }
                } else if (e == 2) { /* bottom: exterior y<line */
                    if (y0j < line - EPS && y1j >= line - EPS) {
                        lo     = x0j;
                        hi     = x1j;
                        covers = 1;
                    }
                } else { /* top: exterior y>line */
                    if (y1j > line + EPS && y0j <= line + EPS) {
                        lo     = x0j;
                        hi     = x1j;
                        covers = 1;
                    }
                }

                if (!covers)
                    continue;
                if (lo < seg_lo)
                    lo = seg_lo;
                if (hi > seg_hi)
                    hi = seg_hi;
                if (hi - lo > EPS) {
                    covs[ncov].lo = lo;
                    covs[ncov].hi = hi;
                    ncov++;
                }
            }

            qsort(covs, (size_t)ncov, sizeof(_WallIv), _wall_iv_cmp);

            /* Push the wall segment WALL_DEPTH/2 into the exterior halfspace
             * so its visible cube sits entirely outside the walkable area.
             * Without this offset the cube is centred on the nav-area
             * boundary and overlaps ~4u of walkable tile, letting the player
             * visually clip into walls as they approach. Sign matches the
             * edge type (see e=0..3 layout above): left/bottom push negative,
             * right/top push positive along the normal axis. */
            float ofs     = WALL_DEPTH * 0.5f;
            float shift_x = is_vertical ? ((e == 0) ? -ofs : ofs) : 0.0f;
            float shift_y = is_vertical ? 0.0f : ((e == 2) ? -ofs : ofs);
            float eline   = line + (is_vertical ? shift_x : shift_y);

            /* Walk sorted coverage intervals; emit the gaps as walls. */
            float cursor = seg_lo;
            for (int k = 0; k < ncov; k++) {
                float lo = covs[k].lo, hi = covs[k].hi;
                if (lo > cursor + EPS) {
                    if (wl->count < wl->capacity) {
                        Wall w;
                        if (is_vertical) {
                            w = (Wall){eline, cursor, eline, lo, WALL_HEIGHT};
                        } else {
                            w = (Wall){cursor, eline, lo, eline, WALL_HEIGHT};
                        }
                        wl->walls[wl->count++] = w;
                    }
                }
                if (hi > cursor)
                    cursor = hi;
            }
            if (cursor < seg_hi - EPS && wl->count < wl->capacity) {
                Wall w;
                if (is_vertical) {
                    w = (Wall){eline, cursor, eline, seg_hi, WALL_HEIGHT};
                } else {
                    w = (Wall){cursor, eline, seg_hi, eline, WALL_HEIGHT};
                }
                wl->walls[wl->count++] = w;
            }
        }
    }

    free(covs);
}

/* ── Snapshot helpers (called by cs2_demo.c around each sim tick) ─────── */

void snapshot_prev(Client* client, Dust2Env* env) {
    memcpy(client->prev, client->curr, sizeof(client->curr));
}

void snapshot_curr(Client* client, Dust2Env* env) {
    _copy_agents_to_snapshot(env, client->curr);
}

/* ── Demo juice audio helpers ─────────────────────────────────────────────
 *
 * Resource walk (no extras-c, no ChangeDirectory):
 *   GetApplicationDirectory() + resources/     (sibling of the binary)
 *   GetApplicationDirectory() + ../resources
 *   same two slots with demo_assets/           (source dir name)
 * GetApplicationDirectory() keeps a trailing slash; we still tolerate a
 * missing one so "bin"+"resources" cannot glue. Missing WAV: log once,
 * skip that voice. InitAudioDevice is void — failure is IsAudioDeviceReady.
 */

static const char* DEMO_VOICE_FILES[DEMO_VOICE_N] = {
    "gunshot.wav",
    "footstep.wav",
    "plant.wav",
    "beep.wav",
};

/* Join appdir + folder + file into out. folder may be "resources" or
 * "../resources". appdir usually ends in '/'. */
static void
_demo_join_res(char* out, size_t n, const char* appdir, const char* folder, const char* file) {
    size_t la    = strlen(appdir);
    int    slash = (la > 0 && appdir[la - 1] != '/' && appdir[la - 1] != '\\');
    if (slash)
        snprintf(out, n, "%s/%s/%s", appdir, folder, file);
    else
        snprintf(out, n, "%s%s/%s", appdir, folder, file);
}

/* Resolve one voice path. Returns 1 and writes out[] on the first hit.
 * resource_dir (when non-NULL and non-empty) is tried first as the
 * directory that contains the WAVs; miss falls through to the appdir walk. */
static int _demo_find_voice(char* out, size_t n, const char* file, const char* resource_dir) {
    const char* appdir    = GetApplicationDirectory();
    const char* folders[] = {"resources", "../resources", "demo_assets", "../demo_assets"};
    int         i;
    if (resource_dir != NULL && resource_dir[0]) {
        snprintf(out, n, "%s/%s", resource_dir, file);
        if (FileExists(out))
            return 1;
    }
    for (i = 0; i < 4; i++) {
        _demo_join_res(out, n, appdir, folders[i], file);
        if (FileExists(out))
            return 1;
    }
    return 0;
}

/* Load the four voices. Each miss is logged once here (init-time only). */
static void _demo_load_voices(Client* cl, const char* resource_dir) {
    char path[1024];
    int  i;
    for (i = 0; i < DEMO_VOICE_N; i++) {
        cl->snd_ok[i] = 0;
        if (!_demo_find_voice(path, sizeof(path), DEMO_VOICE_FILES[i], resource_dir)) {
            TraceLog(LOG_WARNING,
                     "cs2_demo: missing voice %s (searched resources/ + demo_assets/)",
                     DEMO_VOICE_FILES[i]);
            continue;
        }
        cl->snd[i] = LoadSound(path);
        if (!IsSoundValid(cl->snd[i])) {
            TraceLog(LOG_WARNING, "cs2_demo: failed to load %s from %s", DEMO_VOICE_FILES[i], path);
            continue;
        }
        cl->snd_ok[i] = 1;
    }
}

/* Official raylib audio_sound_positioning pan/attenuate (not Steam Audio).
 * Listener is cl->camera. Event is sim (x,y,z) → raylib (x, z, y).
 * max_dist is scaled to this map (~2000 u); the official sample uses 1.0
 * because its scene is a 10-unit grid. */
static void _demo_set_spatial(Client* cl, Sound snd, float sx, float sy, float sz, float max_dist) {
    float px   = sx;
    float py   = sz + PLAYER_EYE_HEIGHT;
    float pz   = sy;
    float dx   = px - cl->camera.position.x;
    float dy   = py - cl->camera.position.y;
    float dz   = pz - cl->camera.position.z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    float att  = 1.0f / (1.0f + dist / max_dist);
    if (att < 0.0f)
        att = 0.0f;
    if (att > 1.0f)
        att = 1.0f;

    float inv = (dist > 1e-4f) ? (1.0f / dist) : 0.0f;
    float ndx = dx * inv, ndy = dy * inv, ndz = dz * inv;

    float fx = cl->camera.target.x - cl->camera.position.x;
    float fy = cl->camera.target.y - cl->camera.position.y;
    float fz = cl->camera.target.z - cl->camera.position.z;
    float fl = sqrtf(fx * fx + fy * fy + fz * fz);
    if (fl > 1e-4f) {
        fx /= fl;
        fy /= fl;
        fz /= fl;
    }

    /* right = cross(up, forward). Camera up is (0,1,0) in this demo. */
    float ux = cl->camera.up.x, uy = cl->camera.up.y, uz = cl->camera.up.z;
    float rx = uy * fz - uz * fy;
    float ry = uz * fx - ux * fz;
    float rz = ux * fy - uy * fx;
    float rl = sqrtf(rx * rx + ry * ry + rz * rz);
    if (rl > 1e-4f) {
        rx /= rl;
        ry /= rl;
        rz /= rl;
    }

    float fdot = fx * ndx + fy * ndy + fz * ndz;
    if (fdot < 0.0f)
        att *= (1.0f + fdot * 0.5f);

    float pan = 0.5f + 0.5f * (ndx * rx + ndy * ry + ndz * rz);
    if (pan < 0.0f)
        pan = 0.0f;
    if (pan > 1.0f)
        pan = 1.0f;

    SetSoundVolume(snd, att);
    SetSoundPan(snd, pan);
}

/* Play one voice at a sim-space point via the next alias slot. */
static void _demo_play_at(Client* cl, int voice, float x, float y, float z, float max_dist) {
    int i;
    if (!cl->audio_ok || voice < 0 || voice >= DEMO_VOICE_N || !cl->snd_ok[voice])
        return;
    i = cl->alias_cursor;
    if (cl->alias_used[i])
        UnloadSoundAlias(cl->alias_pool[i]);
    cl->alias_pool[i] = LoadSoundAlias(cl->snd[voice]);
    cl->alias_used[i] = 1;
    cl->alias_cursor  = (i + 1) % DEMO_ALIAS_N;
    if (!IsSoundValid(cl->alias_pool[i]))
        return;
    _demo_set_spatial(cl, cl->alias_pool[i], x, y, z, max_dist);
    PlaySound(cl->alias_pool[i]);
}

/* ── make_client / c_close ───────────────────────────────────────────────
 *
 * area_bounds: float[N*4] = [x0, y0, x1, y1] per area — from nav_data.h
 * resource_dir: directory that contains the WAVs; may be NULL
 */
Client* make_client(Dust2Env*    env,
                    int          human_agent_idx,
                    const float* area_bounds,
                    const char*  resource_dir) {
    Client* cl          = (Client*)calloc(1, sizeof(Client));
    cl->width           = WINDOW_W;
    cl->height          = WINDOW_H;
    cl->human_agent_idx = human_agent_idx;
    cl->mouse_init      = 0;
    cl->mouse_captured  = 0;

    InitWindow(cl->width, cl->height, "cs2rl 250326");
    SetTargetFPS(60);
    DisableCursor();
    /* Push the far clip plane out. raylib's default projection caps at
     * 1000 units, so on a ~2000-u-wide map (de_dust2) anything past
     * ~1000 u from the camera clips to the background — the "void veil"
     * players see mid-range. 8000 is well beyond the map diagonal and
     * still leaves plenty of depth-buffer resolution at our agent scale. */
    rlSetClipPlanes(0.1, 8000.0);
    /* Reset virtual cursor to center so first-frame delta is zero */
    // SetMousePosition(cl->width / 2, cl->height / 2);

    /* Init camera pointing in +X direction */
    cl->yaw   = 0.0f;
    cl->pitch = 0.0f;

    cl->camera.up         = (Vector3){0.0f, 1.0f, 0.0f};
    cl->camera.fovy       = 70.0f;
    cl->camera.projection = CAMERA_PERSPECTIVE;
    /* position/target updated each frame in update_camera() */
    cl->camera.position = (Vector3){0.0f, PLAYER_EYE_HEIGHT, 0.0f};
    cl->camera.target   = (Vector3){1.0f, PLAYER_EYE_HEIGHT, 0.0f};

    /* Derive walls from nav adjacency */
    build_walls_from_nav(env->sd, area_bounds);

    /* Init snapshots from current state */
    _copy_agents_to_snapshot(env, cl->curr);
    memcpy(cl->prev, cl->curr, sizeof(cl->curr));

    cl->last_step_time = GetTime();
    cl->master_volume  = 1.0f;
    /* First foot must not be dropped: calloc leaves 0, and GetTime() is
     * near 0 right after InitWindow, so 0 would eat the first ~333 ms. */
    {
        int i;
        for (i = 0; i < N_AGENTS; i++)
            cl->last_footstep_t[i] = -1.0e9;
    }

    /* InitAudioDevice is void. Failure is the ready check — same skip
     * path as a missing WAV (log once, no voices). */
    InitAudioDevice();
    cl->audio_ok = IsAudioDeviceReady() ? 1 : 0;
    if (!cl->audio_ok) {
        TraceLog(LOG_WARNING, "cs2_demo: audio device not ready; skipping voices");
    } else {
        SetMasterVolume(cl->master_volume);
        _demo_load_voices(cl, resource_dir);
    }

    env->client = (struct Client*)cl;
    return cl;
}

void c_close(Dust2Env* env) {
    if (env->client) {
        Client* cl = (Client*)env->client;
        int     i;
        /* Aliases first (they share sample data), then source Sounds.
         * CloseAudioDevice BEFORE CloseWindow — raylib tears audio down
         * against the still-alive context. */
        for (i = 0; i < DEMO_ALIAS_N; i++) {
            if (cl->alias_used[i]) {
                UnloadSoundAlias(cl->alias_pool[i]);
                cl->alias_used[i] = 0;
            }
        }
        if (cl->audio_ok) {
            for (i = 0; i < DEMO_VOICE_N; i++) {
                if (cl->snd_ok[i]) {
                    UnloadSound(cl->snd[i]);
                    cl->snd_ok[i] = 0;
                }
            }
        }
        CloseAudioDevice();
        if (env->sd && env->sd->wall_list.walls) {
            free(env->sd->wall_list.walls);
            env->sd->wall_list.walls = NULL;
        }
        EnableCursor();
        CloseWindow();
        free(env->client);
        env->client = NULL;
    }
}

/* ── update_camera — called every render frame (not just sim ticks) ─────
 *
 * Reads mouse delta, updates yaw/pitch, syncs camera to human agent position.
 * For spectate (human_agent_idx < 0), camera follows agent 0.
 *
 * View-kick: punch is applied on the sim tick. Here we sample look =
 * (yaw,pitch)+punch first so a just-applied kick is visible at full
 * strength, then decay with GetFrameTime() (previous frame's dt).
 * Decaying first would spend last-frame dt on a punch that did not
 * exist then. Do not write punch into cl->yaw / cl->pitch.
 */
static void update_camera(Client* cl, Dust2Env* env, float alpha) {
    (void)env;

    /* Manual center-warp mouse input. Rationale:
     *   On WSL2 / some X11 configs, GLFW_CURSOR_DISABLED does NOT actually
     *   lock the cursor to the window — the pointer drifts out and leaves
     *   the window entirely. DisableCursor() alone is unreliable here.
     *
     *   Instead we every frame: read pos, take delta from window center,
     *   then SetMousePosition(center). This keeps the pointer pinned even
     *   when the backend's lock fails, and gives per-frame pixel deltas
     *   that match the MOUSE_SENSITIVITY tuning (rad/px).
     *
     *   Click-to-capture semantics are preserved: only warp while captured.
     *   On any 0→1 capture transition, warp once first so the pre-capture
     *   cursor position doesn't leak into the first delta. */
    const int cx = cl->width / 2;
    const int cy = cl->height / 2;

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && !cl->mouse_captured) {
        DisableCursor(); /* best-effort; ignored if backend can't lock */
        HideCursor();
        cl->mouse_captured = 1;
        SetMousePosition(cx, cy); /* seed baseline — next frame delta = 0 */
    }

    if (IsKeyPressed(KEY_ESCAPE) && cl->mouse_captured) {
        EnableCursor();
        ShowCursor();
        cl->mouse_captured = 0;
    }

    Vector2 md = {0};
    if (cl->mouse_captured && IsWindowFocused()) {
        Vector2 pos = GetMousePosition();
        md.x        = pos.x - (float)cx;
        md.y        = pos.y - (float)cy;
        SetMousePosition(cx, cy);
    }

    cl->yaw   += md.x * MOUSE_SENSITIVITY;
    cl->pitch -= md.y * MOUSE_SENSITIVITY;

    /* Latch press-edges for inputs that fire once per key-press — sampled
     * every render frame, consumed at the next sim tick by human_input.
     * Without this, a ~60 Hz render loop vs 16 Hz sim tick drops 3 out of
     * 4 jump inputs because raylib's per-frame event (IsKeyPressed /
     * GetMouseWheelMove) fires on exactly one render frame.
     *
     * Jump = mousewheel-down. GetMouseWheelMove() returns the wheel delta
     * for this frame (positive = up, negative = down); we latch any
     * negative tick as a jump request. */
    if (GetMouseWheelMove() < 0.0f)
        cl->jump_pending = 1;
    /* Clamp pitch to ±89° in radians */
    if (cl->pitch > 1.5533f)
        cl->pitch = 1.5533f;
    if (cl->pitch < -1.5533f)
        cl->pitch = -1.5533f;

    int   idx = (cl->human_agent_idx >= 0) ? cl->human_agent_idx : 0;
    float px  = _lerp(cl->prev[idx].x, cl->curr[idx].x, alpha);
    float py  = _lerp(cl->prev[idx].y, cl->curr[idx].y, alpha);
    float pz  = _lerp(cl->prev[idx].z, cl->curr[idx].z, alpha);

    /* Map sim(x,y,z) → Raylib(x,z_rl,y_rl) */
    float eye_x = px;
    float eye_y = pz + PLAYER_EYE_HEIGHT; /* Raylib Y = height */
    float eye_z = py;                     /* Raylib Z = sim Y  */

    /* Look dir = aim + sim punch. Clamp the *look* pitch only. */
    float look_yaw   = cl->yaw + env->game.agents[idx].punch_yaw;
    float look_pitch = cl->pitch + env->game.agents[idx].punch_pitch;
    if (look_pitch > 1.5533f)
        look_pitch = 1.5533f;
    if (look_pitch < -1.5533f)
        look_pitch = -1.5533f;

    float dir_x = cosf(look_yaw) * cosf(look_pitch);
    float dir_y = sinf(look_pitch);
    float dir_z = sinf(look_yaw) * cosf(look_pitch);

    cl->camera.position = (Vector3){eye_x, eye_y, eye_z};
    cl->camera.target   = (Vector3){eye_x + dir_x, eye_y + dir_y, eye_z + dir_z};
}

/* ── draw_floor ─────────────────────────────────────────────────────────── */
static void draw_floor(Dust2Env* env) {
    StaticData* sd = env->sd;
    for (int i = 0; i < sd->N; i++) {
        float cx = sd->centroid_xy[i * 2 + 0];
        float cy = sd->centroid_xy[i * 2 + 1];
        /* Color by bombsite */
        Color c;
        if (sd->bombsite_by_idx[i])
            c = (Color){180, 100, 30, 200}; /* orange = bombsite  */
        else
            c = (Color){80, 80, 80, 200};   /* grey   = other     */
        /* Simple fixed-size tile; real bounds come from nav_data.h in cs2_demo */
        DrawPlane((Vector3){cx, 0.0f, cy}, (Vector2){256.0f, 256.0f}, c);
    }
}

/* ── draw_walls ──────────────────────────────────────────────────────────── */
static void draw_walls(Dust2Env* env) {
    WallList* wl = &env->sd->wall_list;
    for (int i = 0; i < wl->count; i++) {
        Wall* w   = &wl->walls[i];
        float cx  = (w->x0 + w->x1) * 0.5f;
        float cy  = (w->y0 + w->y1) * 0.5f;
        float len = sqrtf((w->x1 - w->x0) * (w->x1 - w->x0) + (w->y1 - w->y0) * (w->y1 - w->y0));
        /* Wall midpoint; Raylib Y=up */
        Vector3 pos = {cx, w->height * 0.5f, cy};
        /* Axis-aligned: horizontal wall = extends along X, vertical = extends along Z */
        int   horizontal = fabsf(w->y1 - w->y0) < 1.0f;
        float wx         = horizontal ? len : WALL_DEPTH;
        float wz         = horizontal ? WALL_DEPTH : len;
        DrawCube(pos, wx, w->height, wz, (Color){140, 140, 160, 255});
        DrawCubeWires(pos, wx, w->height, wz, (Color){80, 80, 100, 255});
    }
}

/* ── draw_agents ─────────────────────────────────────────────────────────── */
static void draw_agents(Dust2Env* env, Client* cl, float alpha) {
    /* Fog-of-war debug: only draw agents the human's own agent has line of
     * sight to. Uses the SAME line_of_sight_2d the C env uses for combat /
     * obs / memory, so what the human sees ≡ what a deployed bot sees in
     * its obs vector. Spectate mode (human_agent_idx<0) renders everyone
     * — there's no "viewer" to filter from. */
    AgentState* viewer = (cl->fog_enabled && cl->human_agent_idx >= 0)
                             ? &env->game.agents[cl->human_agent_idx]
                             : NULL;

    for (int i = 0; i < N_AGENTS; i++) {
        if (i == cl->human_agent_idx)
            continue; /* don't draw own body in first-person */

        if (viewer != NULL) {
            AgentState* target = &env->game.agents[i];
            /* Use the agent's CURRENT (sim) position for the LoS test, not the
             * interpolated render position — the LoS test mirrors the per-tick
             * gate the C env applies at sim-tick rate. Interpolating between
             * prev/curr could flicker the visibility at room boundaries. */
            if (viewer->area_idx < 0 || target->area_idx < 0 ||
                !line_of_sight_2d(env->sd, viewer->x, viewer->y, target->x, target->y))
                continue; /* fogged: skip rendering */
        }

        float x     = _lerp(cl->prev[i].x, cl->curr[i].x, alpha);
        float y     = _lerp(cl->prev[i].y, cl->curr[i].y, alpha);
        float z     = _lerp(cl->prev[i].z, cl->curr[i].z, alpha);
        float fa    = _lerp_angle(cl->prev[i].facing, cl->curr[i].facing, alpha);
        int   alive = cl->curr[i].alive;

        /* Team color: T=orange, CT=blue; dead agents are dark */
        Color body_col, head_col;
        if (!alive) {
            body_col = head_col = (Color){40, 40, 40, 128};
        } else if (cl->curr[i].team == 0) {
            body_col = (Color){220, 120, 30, 255}; /* T = orange */
            head_col = (Color){255, 160, 60, 255};
        } else {
            body_col = (Color){30, 80, 200, 255}; /* CT = blue */
            head_col = (Color){60, 130, 255, 255};
        }

        /* Agent rig offsets from feet: body extends 0→96, head at 108, bomb
         * marker at 130. Agent z (ground or airborne) becomes the feet level,
         * so a jumping agent visibly rises with their velocity. */
        DrawCylinder((Vector3){x, z, y}, 12.0f, 12.0f, 96.0f, 8, body_col);
        DrawSphere((Vector3){x, z + 108.0f, y}, 16.0f, head_col);

        /* Aim direction line from head */
        if (alive) {
            float aim_x = x + cosf(fa) * 60.0f;
            float aim_z = y + sinf(fa) * 60.0f;
            DrawLine3D((Vector3){x, z + 108.0f, y},
                       (Vector3){aim_x, z + 108.0f, aim_z},
                       (Color){255, 255, 0, 200});
        }

        /* Bomb indicator above carrier */
        if (cl->curr[i].has_bomb && alive) {
            DrawSphere((Vector3){x, z + 130.0f, y}, 8.0f, YELLOW);
        }
    }
}

/* ── draw_bomb ────────────────────────────────────────────────────────────── */
static void draw_bomb(Dust2Env* env, Client* cl, float alpha) {
    (void)cl;
    (void)alpha;
    GameState* g = &env->game;
    if (g->bomb_is_dropped || g->bomb_planted) {
        float bx = g->bomb_x;
        float by = g->bomb_y;
        float bz = g->bomb_z;
        /* Pulse red when planted */
        Color col;
        if (g->bomb_planted) {
            float pulse = 0.5f + 0.5f * sinf((float)g->tick * 0.5f);
            col         = (Color){255, (uint8_t)(30 * (1.0f - pulse)), 0, 255};
        } else {
            col = YELLOW;
        }
        DrawSphere((Vector3){bx, bz + 10.0f, by}, 12.0f, col);
        DrawSphereWires((Vector3){bx, bz + 10.0f, by}, 12.0f, 8, 8, (Color){255, 255, 0, 200});
    }
}

/* ── draw_hud ────────────────────────────────────────────────────────────── */
static void draw_hud(Client* cl, Dust2Env* env) {
    GameState*  g   = &env->game;
    int         idx = (cl->human_agent_idx >= 0) ? cl->human_agent_idx : 0;
    AgentState* a   = &g->agents[idx];

    /* Crosshair — two short lines, screen-center. Stays put while punch
     * offsets the camera look (brief aim/crosshair disagreement is OK). */
    int cx = cl->width / 2, cy = cl->height / 2;
    DrawLine(cx - 10, cy, cx + 10, cy, WHITE);
    DrawLine(cx, cy - 10, cx, cy + 10, WHITE);

    /* HP: red below 25, otherwise green; number on the bar. */
    int   hp     = a->alive ? a->hp : 0;
    Color hp_col = (hp < 25) ? (Color){200, 0, 0, 255} : (Color){0, 200, 0, 255};
    DrawRectangle(10, cl->height - 30, 200, 20, DARKGRAY);
    DrawRectangle(10, cl->height - 30, hp * 2, 20, hp_col);
    DrawText(TextFormat("HP: %d", hp), 15, cl->height - 28, 16, WHITE);

    /* Weapon name + clip/reserve. */
    int         slot     = (int)a->weapon_slot;
    int         ammo     = (slot >= 0 && slot < 3) ? a->ammo_clip[slot] : 0;
    int         resrv    = (slot >= 0 && slot < 3) ? a->ammo_reserve[slot] : 0;
    const char* wnames[] = {"RIFLE", "PISTOL", "KNIFE"};
    const char* wname    = (slot >= 0 && slot < 3) ? wnames[slot] : "???";
    DrawText(
        TextFormat("%s %d/%d", wname, ammo, resrv), cl->width - 150, cl->height - 30, 16, WHITE);

    /* Bomb clock M:SS only while planted. ticks/16 = whole seconds. */
    if (g->bomb_planted) {
        int secs = g->bomb_ticks_left / 16;
        if (secs < 0)
            secs = 0;
        DrawText(
            TextFormat("BOMB: %d:%02d", secs / 60, secs % 60), cl->width / 2 - 55, 10, 24, RED);
    }

    /* Round clock M:SS. */
    int rt = g->round_ticks_left / 16;
    if (rt < 0)
        rt = 0;
    DrawText(
        TextFormat("%d:%02d", rt / 60, rt % 60), cl->width / 2 - 20, cl->height - 55, 20, WHITE);

    /* Kill feed: last 4 alive 1→0 rows, 3 s fade. T#n / CT#n. */
    {
        double now = GetTime();
        int    k;
        int    row = 0;
        for (k = 0; k < cl->kill_feed_n; k++) {
            float age = (float)(now - cl->kill_feed_t[k]);
            if (age < 0.0f || age >= 3.0f)
                continue;
            float       fade = 1.0f - age / 3.0f;
            int         t0   = (cl->kill_feed_team[k] == 0);
            const char* tag  = t0 ? "T" : "CT";
            Color       col  = t0 ? (Color){220, 120, 30, 255} : (Color){60, 130, 255, 255};
            col              = Fade(col, fade);
            DrawText(TextFormat("%s#%d", tag, cl->kill_feed_idx[k]),
                     cl->width - 90,
                     36 + row * 20,
                     18,
                     col);
            row++;
        }
    }

    /* Human / spectate indicator — keep existing banner. */
    if (cl->human_agent_idx >= 0)
        DrawText("HUMAN CONTROL", 10, 10, 18, GREEN);
    else
        DrawText("SPECTATE", 10, 10, 18, GRAY);
}

/* ── c_render — full implementation ─────────────────────────────────────── */
void c_render(Client* cl, Dust2Env* env) {
    double now   = GetTime();
    float  alpha = (float)((now - cl->last_step_time) * 16.0);
    if (alpha > 1.0f)
        alpha = 1.0f;

    /* Volume nudge is per render frame so a tap between 16 Hz ticks is
     * not dropped. IsKeyPressed, not IsKeyDown — spec says nudge. */
    if (IsKeyPressed(KEY_LEFT_BRACKET)) {
        cl->master_volume -= 0.1f;
        if (cl->master_volume < 0.0f)
            cl->master_volume = 0.0f;
        if (cl->audio_ok)
            SetMasterVolume(cl->master_volume);
    } else if (IsKeyPressed(KEY_RIGHT_BRACKET)) {
        cl->master_volume += 0.1f;
        if (cl->master_volume > 1.0f)
            cl->master_volume = 1.0f;
        if (cl->audio_ok)
            SetMasterVolume(cl->master_volume);
    }

    update_camera(cl, env, alpha);

    BeginDrawing();
    ClearBackground((Color){20, 20, 30, 255});

    BeginMode3D(cl->camera);
    draw_floor(env);
    draw_walls(env);
    draw_agents(env, cl, alpha);
    draw_bomb(env, cl, alpha);
    EndMode3D();

    draw_hud(cl, env);
    DrawFPS(10, cl->height - 20);
    EndDrawing();
}
