#pragma once
#include <stdint.h>
#include <string.h>
#include <math.h>

/* ── Constants ───────────────────────────────────────────────────────────── */
#define TICK_RATE          16
#define MOVE_SPEED         250
#define DT                 (1.0f / TICK_RATE)
#define LASER_DAMAGE       100
#define LASER_RANGE        3000
#define LASER_RANGE_SQ     (LASER_RANGE * LASER_RANGE)
#define SHOOT_COOLDOWN     10
#define BOMB_PLANT_TIME    ((int)(1.2f * TICK_RATE))
#define BOMB_DEFUSE_TIME   (10 * TICK_RATE)
#define BOMB_DEFUSE_KIT    (5  * TICK_RATE)
#define BOMB_TIMER         ((int)(40.0f * TICK_RATE))
#define ROUND_TIME         ((int)(40.0f * TICK_RATE))
#define FOOTSTEP_RADIUS    800
#define GUNSHOT_RADIUS     2000
#define BOMB_BEEP_RADIUS   1500
#define ENEMY_MEMORY_TICKS 32
#define TEAM_SIZE          5
#define N_AGENTS           10
#define OBS_DIM            71
#define STALE_MEMORY_TICK  (-9999)
#define INVALID_AREA_IDX   (-1)
#define PBRS_GAMMA         0.99f

/* ── Static data (owned by Python numpy arrays, pointer shared across instances) ── */
typedef struct {
    int      N;               /* nav area count                                   */
    int8_t*  vis_matrix;      /* [N*N]           area visibility, row-major        */
    int32_t* raster_grid;     /* [grid_h*grid_w]  pos->area_idx, -1=off mesh       */
    int8_t*  adjacency;       /* [N*N]           nav graph connectivity            */
    float*   centroid_xy;     /* [N*2]           idx-indexed: centroid_xy[i*2+0]=x */
    int32_t* area_ids;        /* [N]             idx -> raw area_id                */
    int8_t*  bombsite_mask;   /* [max_area_id+1]  area_id-indexed (for _potential) */
    int8_t*  bombsite_by_idx; /* [N]              idx-indexed (for step hot path)  */
    int      grid_w, grid_h, max_area_id;
    float    grid_x_min, grid_y_min, grid_inv_cell;
    float    inv_x_range, inv_y_range, x_offset, y_offset;
    int32_t  t_spawns[15];   int n_t_spawns;
    int32_t  ct_spawns[5];   int n_ct_spawns;
} StaticData;

/* ── Per-agent state ── */
typedef struct {
    float    x, y, z;
    int32_t  area_idx;           /* 0-based index; INVALID_AREA_IDX=-1            */
    float    facing;             /* radians, 0=+X                                 */
    int32_t  hp;
    int32_t  shoot_cd;
    int8_t   alive;
    int8_t   has_bomb;
    int8_t   has_kit;
    int8_t   team;               /* 0=T, 1=CT                                     */
    int8_t   is_moving;          /* set by movement; read by sound system          */
    int8_t   fired_this_tick;    /* set by shoot; read by sound system             */
    int32_t  enemy_mem_idx[5];   /* area_idx of last known pos; INVALID_AREA_IDX  */
    int32_t  enemy_mem_tick[5];  /* tick when recorded; STALE_MEMORY_TICK=-9999   */
} AgentState;

/* ── Game state ── */
typedef struct {
    int32_t    tick;
    int32_t    round_ticks_left;
    AgentState agents[10];        /* agents[0..4]=T, agents[5..9]=CT              */
    int8_t     bomb_planted;
    int8_t     round_over;
    int32_t    winner;            /* 0=T, 1=CT, -1=ongoing                        */
    int32_t    bomb_carrier_id;   /* agent index 0-4 (T side only)                */
    int32_t    bomb_area_idx;     /* INVALID_AREA_IDX until planted               */
    float      bomb_x, bomb_y, bomb_z;
    int32_t    bomb_ticks_left;
    int32_t    bomb_being_planted_by; /* agent index or -1 */
    int32_t    bomb_plant_ticks;
    int32_t    bomb_being_defused_by; /* agent index or -1 */
    int32_t    bomb_defuse_ticks;
} GameState;

/* ── Full environment (one per parallel instance) ── */
typedef struct {
    StaticData* sd;                    /* shared pointer, never freed by C        */
    GameState   game;
    float       observations[N_AGENTS * OBS_DIM];
    float       rewards[N_AGENTS];
    int8_t      terminals[N_AGENTS];
    int8_t      truncations[N_AGENTS]; /* always 0                                */
    int32_t     actions[N_AGENTS * 4]; /* [move_dir, shoot, interact, reserved]   */
    float       team_spirit;
    float       delta_x[9];
    float       delta_y[9];
    float       dir_facing[9];
    uint32_t    rng;
} Dust2Env;

/* ── Public interface ────────────────────────────────────────────────────── */
void env_init(Dust2Env* env, StaticData* sd, uint32_t seed, float team_spirit);
void env_reset(Dust2Env* env);
void env_step(Dust2Env* env);
void env_close(Dust2Env* env);
