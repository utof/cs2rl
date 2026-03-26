/* src/c_env/cs2_types.h */
#pragma once
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Constants ─────────────────────────────────────────────────────────── */
#define TEAM_SIZE             5
#define N_AGENTS              10
#define OBS_DIM               104
#define ACTION_DIM            7
#define ACTION_MASK_DIM       36 /* 9+16+2+2+3+2+2 */
#define WEAPON_SWITCH_TICKS   8  /* ~0.5s at 16 Hz */
#define CROUCH_COOLDOWN_TICKS 7  /* ~0.4s at 16 Hz */
#define INVALID_AREA_IDX      (-1)

/* ── Weapon definition (compile-time table in cs2_weapons.h) ── */
typedef struct {
    int     type;           /* WEAPON_RIFLE=0, WEAPON_PISTOL=1, WEAPON_KNIFE=2 */
    float   base_damage;
    float   armor_pen;      /* 0.0-1.0 */
    int32_t cycle_ticks;
    int32_t mag_size;       /* -1 for knife (infinite) */
    int32_t reserve_mags;   /* -1 for knife */
    int32_t reload_ticks;
    float   move_speed;     /* units/second at this weapon */
    float   range_modifier; /* damage falloff per 500 units */
} WeaponDef;

/* ── Renderer wall geometry ── */
typedef struct {
    float x0, y0, x1, y1; /* segment endpoints in world space (sim XY coords) */
    float height;         /* extrusion height in world units */
} Wall;

typedef struct {
    Wall* walls;
    int   count;
    int   capacity;
} WallList;

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
    /* ── Phase 5: reward weights (defaults match prior hardcoded values) ── */
    float reward_win;            /* ±applied per alive agent at round end */
    float reward_kill;           /* per kill */
    float reward_death;          /* per death (stored positive, applied negative) */
    float reward_bombsite_entry; /* one-time bonus for T bomb-carrier entering bombsite */
    float reward_plant_bonus;    /* bomb plant completion */
    float reward_plant_base;     /* base objective-action reward on plant (mirrors reward_defuse) */
    float reward_plant_progress_scale; /* per-tick plant progress */
    float reward_plant_interrupted;    /* interrupted-plant penalty (stored positive) */
    float reward_defuse;               /* defuse completion */
    float reward_shot_penalty;         /* per shot fired (stored positive, applied negative) */
    float reward_ct_survival;          /* per-tick CT survival micro-reward */
    float reward_inaction;             /* per-tick penalty for alive agent choosing move=0 */
    float pbrs_alive_weight;           /* alive-delta coefficient in _potential */
    float pbrs_hp_weight;              /* hp-delta coefficient (default 0.002 = 1/500) */
    float pbrs_site_weight;            /* site-presence coefficient */
    float pbrs_bomb_progress_weight;   /* bomb-closeness scale in _potential */
    float pbrs_nav_weight_t;           /* T-side nav approach weight */
    float pbrs_nav_weight_ct;          /* CT-side nav approach weight */
    /* Phase 6: renderer wall list — populated by build_walls_from_nav(), C-demo only */
    WallList wall_list;
} StaticData;

/* ── Per-agent state ── */
typedef struct {
    float   x, y, z;
    int32_t area_idx; /* 0-based index; INVALID_AREA_IDX=-1            */
    float   facing;   /* radians, 0=+X                                 */
    int32_t hp;
    int32_t fire_cd;  /* was shoot_cd */
    int8_t  alive;
    int8_t  has_bomb;
    int8_t  has_kit;
    int8_t  team;              /* 0=T, 1=CT                                     */
    int8_t  is_moving;         /* set by movement; read by sound system          */
    int8_t  fired_this_tick;   /* set by shoot; read by sound system             */
    int8_t  _pad0[2];          /* explicit padding to int32 boundary */
    int32_t enemy_mem_idx[5];  /* area_idx of last known pos; INVALID_AREA_IDX  */
    int32_t enemy_mem_tick[5]; /* tick when recorded; STALE_MEMORY_TICK=-9999   */
    /* Phase 4b additions — appended to preserve existing field offsets */
    float   vx, vy;
    int8_t  is_crouching;
    int8_t  _pad1[3];
    int32_t crouch_cd;
    int32_t armor;
    int8_t  has_helmet;
    int8_t  weapon_slot;        /* 0=rifle, 1=pistol, 2=knife */
    int8_t  weapon_slot_target; /* pending slot after switch completes */
    int8_t  _pad2[1];
    int32_t ammo_clip[3];
    int32_t ammo_reserve[3];
    int32_t reload_ticks;
    int32_t switch_ticks;
    /* Phase 6 additions — appended to preserve existing field offsets */
    uint8_t human_controlled; /* 1 = process_movement uses aim_rad, ignores actions[1] */
    int8_t  _pad3[3];         /* explicit padding: 1 + 3 = 4 bytes → float alignment   */
    float   aim_rad;          /* continuous facing angle set by human input (radians)   */
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
    int8_t     bombsite_entered[5];   /* per-T-agent flag: 1 if entered bombsite this round */
    int8_t     bomb_is_dropped;       /* 1 when bomb on ground */
    int8_t     _pad_gs[2];            /* pad to 4-byte boundary */
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
    int32_t action_aim[16];
    int32_t action_reload[2];
    int32_t action_weapon[3];
    int32_t action_crouch[2];
    /* ── Phase 5: episode-level reward component accumulators ── */
    float reward_win;      /* cumulative win/loss reward (all agents) */
    float reward_kills;    /* cumulative kill rewards */
    float reward_deaths;   /* cumulative death penalties */
    float reward_bomb;     /* entry + plant-progress + plant-base + plant-bonus + defuse */
    float reward_pbrs;     /* total PBRS contribution (all agents, all ticks) */
    float reward_shots;    /* total shot penalties */
    float reward_survival; /* total CT survival micro-rewards */
    float reward_inaction; /* total inaction penalties */
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
    int8_t      masks[N_AGENTS * ACTION_MASK_DIM];
    /* Phase 6: renderer client — NULL during training, set by make_client() */
    struct Client* client;
} Dust2Env;

/* ── RNG utility (available to all headers) ─────────────────────────────── */
static inline uint32_t xorshift32(uint32_t* state) {
    uint32_t x  = *state;
    x          ^= x << 13;
    x          ^= x >> 17;
    x          ^= x << 5;
    return (*state = x);
}
