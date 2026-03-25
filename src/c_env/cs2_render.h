/* cs2_render.h — Raylib 3D FPS renderer for cs2rl. Phase 6. */
#pragma once
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "raylib.h"
#include "cs2_types.h"

#define PLAYER_EYE_HEIGHT 64.0f  /* eye height above agent.z in world units */
#define WALL_HEIGHT       128.0f /* wall extrusion height                   */
#define WALL_DEPTH        8.0f   /* wall thickness                          */
#define MOUSE_SENSITIVITY 0.002f /* mouse sensitivity for camera rotation   */
#define WINDOW_W          1280
#define WINDOW_H          720

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

/* ── Wall derivation from nav adjacency ──────────────────────────────────
 *
 * Each nav area has bounds [x0, y0, x1, y1] stored in area_bounds[idx*4+0..3].
 * We check each of the 4 edges of each area. An edge is a wall if no adjacent
 * area's bounds share (overlap) that edge's segment.
 *
 * Called once from make_client().
 */
static void build_walls_from_nav(StaticData* sd, const float* area_bounds) {
    WallList* wl = &sd->wall_list;
    wl->capacity = sd->N * 4;
    wl->walls    = (Wall*)malloc(wl->capacity * sizeof(Wall));
    wl->count    = 0;

    for (int i = 0; i < sd->N; i++) {
        float x0i = area_bounds[i * 4 + 0], y0i = area_bounds[i * 4 + 1];
        float x1i = area_bounds[i * 4 + 2], y1i = area_bounds[i * 4 + 3];

        /* 4 edges of area i: (left, right, bottom, top) as (ax0,ay0,ax1,ay1) */
        float edges[4][4] = {
            {x0i, y0i, x0i, y1i}, /* left   */
            {x1i, y0i, x1i, y1i}, /* right  */
            {x0i, y0i, x1i, y0i}, /* bottom */
            {x0i, y1i, x1i, y1i}, /* top    */
        };

        for (int e = 0; e < 4; e++) {
            float ex0 = edges[e][0], ey0 = edges[e][1];
            float ex1 = edges[e][2], ey1 = edges[e][3];

            /* Check if any adjacent area shares (touches) this edge */
            int shared = 0;
            for (int j = 0; j < sd->N && !shared; j++) {
                if (j == i)
                    continue;
                if (!sd->adjacency[i * sd->N + j])
                    continue;
                float x0j = area_bounds[j * 4 + 0], y0j = area_bounds[j * 4 + 1];
                float x1j = area_bounds[j * 4 + 2], y1j = area_bounds[j * 4 + 3];
                /* Adjacent area j shares the edge if it touches the same line segment */
                int touches_x = (fabsf(x0j - ex0) < 1.0f && fabsf(x1j - ex1) < 1.0f) ||
                                (fabsf(x0j - ex0) < 1.0f && fabsf(x1j - ex0) < 1.0f) ||
                                (x0j <= ex0 + 1.0f && x1j >= ex1 - 1.0f);
                int touches_y = (fabsf(y0j - ey0) < 1.0f && fabsf(y1j - ey1) < 1.0f) ||
                                (fabsf(y0j - ey0) < 1.0f && fabsf(y1j - ey0) < 1.0f) ||
                                (y0j <= ey0 + 1.0f && y1j >= ey1 - 1.0f);
                /* Vertical edge: x values match, y range must overlap */
                if (fabsf(ex0 - ex1) < 1.0f) { /* vertical edge */
                    shared = (fabsf(x0j - ex0) < 1.0f || fabsf(x1j - ex0) < 1.0f) && touches_y;
                } else {                       /* horizontal edge */
                    shared = (fabsf(y0j - ey0) < 1.0f || fabsf(y1j - ey0) < 1.0f) && touches_x;
                }
                (void)touches_x;
                (void)touches_y; /* suppress unused warnings if needed */
            }

            if (!shared && wl->count < wl->capacity) {
                Wall w                 = {ex0, ey0, ex1, ey1, WALL_HEIGHT};
                wl->walls[wl->count++] = w;
            }
        }
    }
}

/* ── Snapshot helpers (called by cs2_demo.c around each sim tick) ─────── */

void snapshot_prev(Client* client, Dust2Env* env) {
    memcpy(client->prev, client->curr, sizeof(client->curr));
}

void snapshot_curr(Client* client, Dust2Env* env) {
    _copy_agents_to_snapshot(env, client->curr);
}

/* ── make_client / c_close ───────────────────────────────────────────────
 *
 * area_bounds: float[N*4] = [x0, y0, x1, y1] per area — from nav_data.h
 */
Client* make_client(Dust2Env* env, int human_agent_idx, const float* area_bounds) {
    Client* cl          = (Client*)calloc(1, sizeof(Client));
    cl->width           = WINDOW_W;
    cl->height          = WINDOW_H;
    cl->human_agent_idx = human_agent_idx;

    InitWindow(cl->width, cl->height, "cs2rl - Phase 6");
    SetTargetFPS(60);
    DisableCursor();

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
    env->client        = (struct Client*)cl;
    return cl;
}

void c_close(Dust2Env* env) {
    if (env->client) {
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
 */
static void update_camera(Client* cl, Dust2Env* env, float alpha) {

    Vector2 delta  = GetMouseDelta();
    cl->yaw       += delta.x * MOUSE_SENSITIVITY;
    cl->pitch     -= delta.y * MOUSE_SENSITIVITY;
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

    /* Direction from yaw/pitch */
    float dir_x = cosf(cl->yaw) * cosf(cl->pitch);
    float dir_y = sinf(cl->pitch);
    float dir_z = sinf(cl->yaw) * cosf(cl->pitch);

    cl->camera.position = (Vector3){eye_x, eye_y, eye_z};
    cl->camera.target   = (Vector3){eye_x + dir_x, eye_y + dir_y, eye_z + dir_z};
}

/* Stub c_render — replaced in Task 7 */
void c_render(Client* cl, Dust2Env* env) {
    float t     = (float)GetTime();
    float alpha = (float)((t - cl->last_step_time) * 16.0);
    if (alpha > 1.0f)
        alpha = 1.0f;
    update_camera(cl, env, alpha);

    BeginDrawing();
    ClearBackground((Color){30, 30, 40, 255});
    BeginMode3D(cl->camera);
    /* TODO: draw calls in Task 7 */
    DrawGrid(20, 100.0f);
    EndMode3D();
    DrawFPS(10, 10);
    if (cl->human_agent_idx >= 0)
        DrawText("HUMAN CONTROL", 10, 30, 20, GREEN);
    EndDrawing();
}
