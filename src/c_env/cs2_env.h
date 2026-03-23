#pragma once
#include <stdint.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Constants ───────────────────────────────────────────────────────────── */
#define TEAM_SIZE 5
#define N_AGENTS 10
#define OBS_DIM 72
#define ACTION_DIM 4
#define INVALID_AREA_IDX (-1)

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
    float*   bombsite_dist;   /* [max_area_id+1]  area_id-indexed shortest dist    */
    int      grid_w, grid_h, max_area_id;
    float    grid_x_min, grid_y_min, grid_inv_cell;
    float    inv_x_range, inv_y_range, x_offset, y_offset;
    float    bombsite_dist_scale;
    int32_t  laser_damage;
    float    laser_range;
    float    laser_range_sq;
    int32_t  shoot_cooldown;
    int32_t  bomb_plant_time;
    int32_t  bomb_defuse_time;
    int32_t  bomb_defuse_kit;
    int32_t  bomb_timer;
    int32_t  round_time;
    float    footstep_radius_sq;
    float    gunshot_radius_sq;
    int32_t  enemy_memory_ticks;
    int32_t  stale_memory_tick;
    float    pbrs_gamma;
    float    delta_x[9];
    float    delta_y[9];
    float    dir_facing[9];
    int32_t  t_spawns[15];
    int      n_t_spawns;
    int32_t  ct_spawns[5];
    int      n_ct_spawns;
    float    max_turn_speed; /* max facing change per tick (radians)       */
} StaticData;

/* ── Per-agent state ── */
typedef struct {
    float   x, y, z;
    int32_t area_idx; /* 0-based index; INVALID_AREA_IDX=-1            */
    float   facing;   /* radians, 0=+X                                 */
    int32_t hp;
    int32_t shoot_cd;
    int8_t  alive;
    int8_t  has_bomb;
    int8_t  has_kit;
    int8_t  team;              /* 0=T, 1=CT                                     */
    int8_t  is_moving;         /* set by movement; read by sound system          */
    int8_t  fired_this_tick;   /* set by shoot; read by sound system             */
    int32_t enemy_mem_idx[5];  /* area_idx of last known pos; INVALID_AREA_IDX  */
    int32_t enemy_mem_tick[5]; /* tick when recorded; STALE_MEMORY_TICK=-9999   */
} AgentState;

/* ── Game state ── */
typedef struct {
    int32_t    tick;
    int32_t    round_ticks_left;
    AgentState agents[10]; /* agents[0..4]=T, agents[5..9]=CT              */
    int8_t     bomb_planted;
    int8_t     round_over;
    int32_t    winner;          /* 0=T, 1=CT, -1=ongoing                        */
    int32_t    bomb_carrier_id; /* agent index 0-4 (T side only)                */
    int32_t    bomb_area_idx;   /* INVALID_AREA_IDX until planted               */
    float      bomb_x, bomb_y, bomb_z;
    int32_t    bomb_ticks_left;
    int32_t    bomb_being_planted_by; /* agent index or -1 */
    int32_t    bomb_plant_ticks;
    int32_t    bomb_being_defused_by; /* agent index or -1 */
    int32_t    bomb_defuse_ticks;
    int8_t     bombsite_entered[5]; /* per-T-agent flag: 1 if entered bombsite this round */
} GameState;

/* ── Per-step stats exported for Python-side episode aggregation ─────────── */
typedef struct {
    int32_t bomb_planted;
    int32_t bomb_defused;
    int32_t kills_t;
    int32_t kills_ct;
    int32_t blocked_moves_t;
    int32_t blocked_moves_ct;
    int32_t winner;
    int32_t winner_t;
    int32_t winner_ct;
    int32_t timed_out;
    int32_t alive_t_end;
    int32_t alive_ct_end;
    int32_t round_length;
    int32_t action_move[9];
    int32_t action_shoot[2];
    int32_t action_use[2];
    int32_t action_last[2];
} StepStats;

/* ── Full environment (one per parallel instance) ── */
typedef struct {
    StaticData* sd; /* shared pointer, never freed by C        */
    GameState   game;
    StepStats   step_stats;
    StepStats   episode_stats;
    float       observations[N_AGENTS * OBS_DIM];
    float       rewards[N_AGENTS];
    int8_t      terminals[N_AGENTS];
    int8_t      truncations[N_AGENTS]; /* always 0                                */
    float       team_spirit;
    uint32_t    rng;
} Dust2Env;

/* ── Public interface ────────────────────────────────────────────────────── */
void env_init(Dust2Env* env, StaticData* sd, uint32_t seed, float team_spirit);
void env_reset(Dust2Env* env);
void env_step(Dust2Env* env, const int32_t* actions);
void env_close(Dust2Env* env);
