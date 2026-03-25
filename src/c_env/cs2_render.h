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
    /* Reset virtual cursor to center so first-frame delta is zero */
    SetMousePosition(cl->width / 2, cl->height / 2);

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

    /* Manual center-warp: more reliable than GetMouseDelta() in WSL2/X11
     * where GLFW_CURSOR_DISABLED may not lock the cursor properly. */
    Vector2 pos = GetMousePosition();
    float   mdx = pos.x - (float)(cl->width / 2);
    float   mdy = pos.y - (float)(cl->height / 2);
    SetMousePosition(cl->width / 2, cl->height / 2);
    cl->yaw   += mdx * MOUSE_SENSITIVITY;
    cl->pitch -= mdy * MOUSE_SENSITIVITY;
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
    for (int i = 0; i < N_AGENTS; i++) {
        if (i == cl->human_agent_idx)
            continue; /* don't draw own body in first-person */

        float x     = _lerp(cl->prev[i].x, cl->curr[i].x, alpha);
        float y     = _lerp(cl->prev[i].y, cl->curr[i].y, alpha);
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

        /* Body cylinder: base at floor, top 96 units */
        DrawCylinder((Vector3){x, 0.0f, y}, 12.0f, 12.0f, 96.0f, 8, body_col);
        /* Head sphere: at 108 units */
        DrawSphere((Vector3){x, 108.0f, y}, 16.0f, head_col);

        /* Aim direction line from head */
        if (alive) {
            float aim_x = x + cosf(fa) * 60.0f;
            float aim_z = y + sinf(fa) * 60.0f;
            DrawLine3D((Vector3){x, 108.0f, y},
                       (Vector3){aim_x, 108.0f, aim_z},
                       (Color){255, 255, 0, 200});
        }

        /* Bomb indicator above carrier */
        if (cl->curr[i].has_bomb && alive) {
            DrawSphere((Vector3){x, 130.0f, y}, 8.0f, YELLOW);
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

    /* Crosshair */
    int cx = cl->width / 2, cy = cl->height / 2;
    DrawLine(cx - 10, cy, cx + 10, cy, WHITE);
    DrawLine(cx, cy - 10, cx, cy + 10, WHITE);

    /* Health bar */
    int hp = a->alive ? a->hp : 0;
    DrawRectangle(10, cl->height - 30, 200, 20, DARKGRAY);
    DrawRectangle(10, cl->height - 30, hp * 2, 20, (Color){0, 200, 0, 255});
    DrawText(TextFormat("HP: %d", hp), 15, cl->height - 28, 16, WHITE);

    /* Weapon / ammo */
    int         slot     = (int)a->weapon_slot;
    int         ammo     = (slot >= 0 && slot < 3) ? a->ammo_clip[slot] : 0;
    int         resrv    = (slot >= 0 && slot < 3) ? a->ammo_reserve[slot] : 0;
    const char* wnames[] = {"RIFLE", "PISTOL", "KNIFE"};
    const char* wname    = (slot >= 0 && slot < 3) ? wnames[slot] : "???";
    DrawText(
        TextFormat("%s %d/%d", wname, ammo, resrv), cl->width - 150, cl->height - 30, 16, WHITE);

    /* Bomb timer */
    if (g->bomb_planted) {
        int secs = g->bomb_ticks_left / 16;
        DrawText(TextFormat("BOMB: %ds", secs), cl->width / 2 - 40, 10, 24, RED);
    }

    /* Round timer */
    int rt = g->round_ticks_left / 16;
    DrawText(
        TextFormat("%d:%02d", rt / 60, rt % 60), cl->width / 2 - 20, cl->height - 55, 20, WHITE);

    /* Human / spectate indicator */
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
