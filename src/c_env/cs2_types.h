/* src/c_env/cs2_types.h */
#pragma once
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Sim recoil v1 (#120): one-pole punch decay. tau=0.08 s.
 * demo_decay_punch must call this — do not fork the formula. */
static inline float recoil_decay_punch(float prev, float dt) {
    return prev * expf(-dt / 0.08f);
}

/* ── Constants ─────────────────────────────────────────────────────────── */
#define TEAM_SIZE 5
#define N_AGENTS  10
/* Horizontal body radius. Must match DrawCylinder in cs2_render.h. Collision
 * is a point on the raster; without this hull the 12u mesh sits inside the
 * 8u exterior wall (and C (int)truncation lets x∈(-cell,0) look like cell 0).
 * Axis samples only — corners can still clip ~5u. Not a cliff/adjacency test. */
#define AGENT_HULL_RADIUS 12.0f
/* Batch 3.5 (#24): self block now carries pitch sin/cos at obs[11..12]; all
 * downstream obs slots shifted +2. SIM_OBS_VERSION below tracks sim's internal
 * obs schema, distinct from deploy's frozen v2-105dim (gh #34 suspension). */
/* ── Observation block layout (SINGLE SOURCE OF TRUTH) ───────────────────────
 * The per-agent obs vector is four contiguous blocks:
 *   self     [0  .. 28)  OBS_SELF_SIZE                       scalar self-state
 *   teammate [28 .. 56)  OBS_TEAMMATE_COUNT × _STRIDE (4×7)  nearest teammates
 *   enemy    [56 .. 96)  OBS_ENEMY_COUNT    × _STRIDE (5×8)  dist-sorted enemies
 *   global   [96 .. 110) OBS_GLOBAL_SIZE                     round/bomb scalars
 * Each block base is DERIVED from the preceding block's width, so the block
 * boundaries (28/56/96) live in exactly ONE place. Consumers:
 *   - cs2_observations.h writes every slot through these bases (never bare ints);
 *   - env_init() in cs2_env.h asserts the blocks tile OBS_DIM exactly;
 *   - scripts/sync_action_spec.py parses the *_SIZE/*_STRIDE/*_COUNT literals
 *     into src/_obs_spec.py so Python masking / demo-zeroing code imports
 *     OBS_BLOCKS instead of hardcoding block boundaries.
 * Pitfall: adding a feature => bump the OWNING block's *_SIZE (or *_STRIDE) AND
 * the OBS_DIM literal below. If they drift, the env_init tiling assert fires at
 * startup and the generator raises at codegen time.
 * Batch 6 Task 2.5 (spec R9/D4): self block 25 → 28 — appended goal-direction
 * slots [sin(rel_bearing), cos(rel_bearing), dist/map_diag] to the nearest
 * bombsite, written in cs2_observations.h. Downstream bases shifted +3. */
#define OBS_SELF_BASE       0
#define OBS_SELF_SIZE       28
#define OBS_TEAMMATE_BASE   (OBS_SELF_BASE + OBS_SELF_SIZE)
#define OBS_TEAMMATE_STRIDE 7
#define OBS_TEAMMATE_COUNT  4
#define OBS_ENEMY_BASE      (OBS_TEAMMATE_BASE + OBS_TEAMMATE_STRIDE * OBS_TEAMMATE_COUNT)
#define OBS_ENEMY_STRIDE    8
#define OBS_ENEMY_COUNT     5
#define OBS_GLOBAL_BASE     (OBS_ENEMY_BASE + OBS_ENEMY_STRIDE * OBS_ENEMY_COUNT)
#define OBS_GLOBAL_SIZE     14
#define OBS_DIM             110
#define ACTION_DIM                                                                                 \
    7  /* Batch 3: HEAD_AIM removed; aim is now a continuous head, see AIM_DIM below */
#define ACTION_MASK_DIM                                                                            \
    22 /* Batch 3: sum(ACTION_HEAD_SIZES) = 9+2+2+3+2+2+2 = 22 (was 38 with the 16-bin aim) */
/* Batch 3.5: 2D Gaussian (Δyaw + absolute pitch). Order in continuous_actions buffer:
 * [i*AIM_DIM+0] = Δyaw  — accumulated via wrap_pi(facing + clamped Δyaw).
 * [i*AIM_DIM+1] = pitch — ABSOLUTE (v1c, gh #36): a->pitch = clamp(value, ±π/2)
 *                         each tick. NOT a delta — the policy picks where to
 *                         look directly. Avoids the saturation pathology where
 *                         random-init μ drift locked pitch at ±π/2 within ~16
 *                         ticks. Effective range is ±max_turn_speed (~±π/4 = ±45°)
 *                         since policy output is bounded by tanh*max_turn_speed. */
#define AIM_DIM 2
/* Sim-side obs schema version (NOT used at runtime — pure documentation;
 * future sim refactors bump this when obs schema changes. Deploy export
 * literals stay frozen at v2-105dim per gh #34.) */
#define SIM_OBS_VERSION       "sim-v2-110dim"
#define WEAPON_SWITCH_TICKS   8 /* ~0.5s at 16 Hz */
#define CROUCH_COOLDOWN_TICKS 7 /* ~0.4s at 16 Hz */
#define INVALID_AREA_IDX      (-1)

/* ── Action head spec (single source of truth for names, sizes, order) ── */
/* Batch 3: HEAD_AIM removed entirely from the discrete head enum.
 * Continuous aim lives in a separate float buffer (see binding.c env_step
 * second arg) and is NOT one of these heads. Renumbered offsets propagate
 * through cs2_env.h (mask emit), cs2_movement.h, cs2_combat.h, cs2_bomb.h,
 * and cs2_input.h. */
enum ActionHead {
    HEAD_MOVE   = 0,
    HEAD_SHOOT  = 1,
    HEAD_RELOAD = 2,
    HEAD_WEAPON = 3,
    HEAD_USE    = 4,
    HEAD_CROUCH = 5,
    HEAD_JUMP   = 6,
};

static const int   ACTION_HEAD_SIZES[] = {9, 2, 2, 3, 2, 2, 2};
static const char* ACTION_HEAD_NAMES[] = {
    "move",
    "shoot",
    "reload",
    "weapon",
    "use",
    "crouch",
    "jump",
};

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
    float z0;             /* sim z of wall base; 0 = ground */
} Wall;

typedef struct {
    Wall* walls;
    int   count;
    int   capacity;
} WallList;

/* ── Static data (owned by Python numpy arrays, pointer shared across instances) ── */
typedef struct {
    int      N;           /* nav area count                                   */
    int8_t*  vis_matrix;  /* [N*N]           area visibility, row-major        */
    int32_t* raster_grid; /* [grid_h*grid_w]  pos->area_idx, -1=off mesh       */
    int8_t*  adjacency;   /* [N*N]           nav graph connectivity            */
    float*   centroid_xy; /* [N*2]           idx-indexed: centroid_xy[i*2+0]=x */
    /* T2 (verticality): per-area terrain elevation and ramp flag.
     * centroids_z[i] = z of area i's ground surface (0.0 = flat).
     * is_ramp[i]     = 1 if area i is a ramp/stairs — exempts the area from the
     *                  cliff guard in cs2_movement.h _resolve_xy_collision.
     * Field order MUST stay in sync with:
     *   - StaticDataC._fields_ in cs2_env.py  (ctypes mirror)
     *   - PyArg_ParseTuple format string in binding.c py_init()
     * Mismatch silently corrupts all pointer fields that follow. */
    float*   centroids_z;     /* [N]             idx-indexed: terrain z per area   */
    int32_t* area_ids;        /* [N]             idx -> raw area_id                */
    int8_t*  bombsite_mask;   /* [max_area_id+1]  area_id-indexed (for _potential) */
    int8_t*  bombsite_by_idx; /* [N]              idx-indexed (for step hot path)  */
    int8_t*  is_ramp;         /* [N]              idx-indexed: 1=ramp/stairs       */
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
    float reward_win; /* ±applied per alive agent at round end (legacy; superseded by
                       * per-mechanism fields below when Batch 1 routing is active) */
    /* Batch 1 (RL overhaul): differential win rewards by outcome mechanism.
     * Supersedes the symmetric reward_win at round end. Routing determined by
     * win_by_detonation / win_by_defuse flags in StepStats (set in
     * compute_rewards round-over block). Defaults set Python-side in Cs2Env.
     *
     * Pitfall: these must stay BEFORE wall_list. StaticDataC now overlays
     * wall_list + area_bounds after this prefix (offsets asserted in
     * cs2_env.py). Do not insert fields here without updating that overlay. */
    float reward_win_t_detonation;   /* default 5.0 — T wins by bomb detonation */
    float reward_win_t_elimination;  /* default 3.0 — T wins by eliminating all CT (no plant) */
    float reward_win_ct_defuse;      /* default 5.0 — CT wins by defusing a planted bomb */
    float reward_win_ct_timeout;     /* default 4.0 — CT wins by round timer (bomb not planted) */
    float reward_win_ct_elimination; /* default 3.0 — CT wins by eliminating all T pre-plant */
    float reward_kill;               /* per kill */
    float reward_death;              /* per death (stored positive, applied negative) */
    float reward_bombsite_entry;     /* one-time bonus for T bomb-carrier entering bombsite */
    float reward_plant_bonus;        /* bomb plant completion */
    float reward_plant_base; /* base objective-action reward on plant (mirrors reward_defuse) */
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
    /* Ramp interpolation AABB. After wall_list. StaticDataC appends wall_list
     * then these so Python can publish the room quad after env_init.
     * Measured gcc offsetof: wall_list=480, area_bounds=496, owned=504
     * (no pad after pbrs_nav_weight_ct). NULL → centroids_z (dust2). */
    const float* area_bounds;       /* [N*4] x0,y0,x1,y1; NULL = no interpolation */
    int          area_bounds_owned; /* 1 if C malloc'd it; train/play set 0 */
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
    /* Phase 7 additions (jump / vertical movement) — appended, never reorder. */
    float   vz;          /* per-second vertical velocity (units/s)                    */
    int8_t  is_airborne; /* 1 when z > 0 or vz != 0 — skips ground friction           */
    int8_t  _pad4[3];    /* pad to int32 boundary                                     */
    int32_t jump_cd;     /* ticks until another jump press is honoured (0 = bhop OK)  */
    /* Batch 3.5 additions (pitch / 3D combat) — appended, never reorder. */
    float pitch; /* radians, 0 = horizontal; ABSOLUTE per env_step (v1c, gh #36) */
    /* Sim recoil v1 (#120): view-kick, appended, never reorder earlier fields.
     * Shared by the hit ray and cs2_demo camera when Dust2Env.recoil_enabled.
     * memset on spawn / env_reset zeros them. Do not write these into facing,
     * aim_rad, or stored pitch — add them at the ray / look site only. */
    float punch_pitch;
    float punch_yaw;
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
    /* Batch 2: round-fixed designated bomb carrier (T-side index 0..4).
     * Distinct from bomb_carrier_id, which is the *dynamic* possession
     * tracker (reassigned on drop+auto-pickup in cs2_bomb.h:111). This
     * field is set ONLY in env_reset and is the round's stable identity
     * signal. Consumed by compute_observations to emit the role bit at
     * obs[OBS_GLOBAL_BASE+13] (=109 since the Batch 6 bearing slots; was 106,
     * and 104 before the T4 pitch insertion). Pitfall: must stay in the int32_t block
     * before bombsite_entered to keep ctypes alignment in sync — see GameStateC mirror in
     * cs2_env.py. */
    int32_t round_designated_carrier_id;
    int8_t  bombsite_entered[5]; /* per-T-agent flag: 1 if entered bombsite this round */
    int8_t  bomb_is_dropped;     /* 1 when bomb on ground */
    int8_t  _pad_gs[2];          /* pad to 4-byte boundary */
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
    /* action_last[2] removed (F13, 2026-07-06 adversarial review): it had no
     * corresponding action head (legacy of a pre-Batch-3 "switch to last
     * weapon" concept), no writer, and exported permanently-zero metrics.
     * Mirror struct in cs2_env.py StepStatsC + its sizeof assert were
     * updated in the same change — touch both or the ctypes overlay shifts. */
    /* Batch 3: continuous-aim Δyaw stats (replaces 16-bin action_aim histogram).
     * Sum + sum-of-squares + count enables Welford-style mean/var recovery
     * Python-side without storing the full rollout. mean = sum / count;
     * var = (sq_sum / count) - mean².
     * No explicit pad — three int32-aligned fields (4+4+4=12B) follow
     * the int32-aligned `action_use[2]` cleanly. `_pad_ss_wins[2]` at
     * end of struct still pads to 4-byte boundary as before. */
    float   aim_delta_sum;
    float   aim_delta_sq_sum;
    int32_t aim_delta_count;
    /* Batch 3.5: pitch Welford triple (mirrors yaw fields above). Same int32-aligned
     * layout: 4+4+4=12B. Pitch_log_std non-collapse is the spec's load-bearing
     * acceptance signal — diagnostics need their own surface. */
    float   aim_delta_pitch_sum;
    float   aim_delta_pitch_sq_sum;
    int32_t aim_delta_pitch_count;
    int32_t action_reload[2];
    int32_t action_weapon[3];
    int32_t action_crouch[2];
    int32_t action_jump[2]; /* Phase 7: jump action head counter (appended) */
    /* ── Phase 5: episode-level reward component accumulators ── */
    float reward_win;      /* cumulative win/loss reward (all agents) */
    float reward_kills;    /* cumulative kill rewards */
    float reward_deaths;   /* cumulative death penalties */
    float reward_bomb;     /* entry + plant-progress + plant-base + plant-bonus + defuse */
    float reward_pbrs;     /* total PBRS contribution (all agents, all ticks) */
    float reward_shots;    /* total shot penalties */
    float reward_survival; /* total CT survival micro-rewards */
    float reward_inaction; /* total inaction penalties */
    /* Batch 1 (RL overhaul): round-end win classification flags.
     * Cleared by round_reset (Task 2). Set by compute_rewards round-over
     * block (Task 3). Consumed Python-side by split_into_channels to route
     * reward_win into the objective channel on detonation/defuse, combat
     * channel on elimination/timeout. */
    int8_t win_by_detonation; /* 1 when round ended because bomb detonated (T wins) */
    int8_t win_by_defuse;     /* 1 when round ended because bomb was defused (CT wins) */
    int8_t _pad_ss_wins[2];   /* pad to 4-byte boundary for ctypes alignment */
    /* Observe-only plant timestamp (instrumentation 2026-08-15).
     * Written in cs2_bomb.h at plant completion from g->tick (same clock as
     * round_length). 0 = never planted. clear_stats memsets the whole
     * struct, so env_reset starts this at 0. Do not reorder earlier fields
     * — ctypes overlay + sizeof asserts must stay in lockstep. */
    int32_t plant_tick; /* g->tick at plant completion; 0 = never planted */
} StepStats;

/* ── Full environment (one per parallel instance) ── */
typedef struct {
    StaticData* sd; /* Python-owned; env_close frees area_bounds only if owned */
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
    /* Sim recoil v1 (#120): env-wide physics switch, after client.
     * 0 = today's hitscan (train / make_env default); 1 = punch on the hit ray
     * (cs2_demo). Not a binding.init argument — that 69-arg FMT is a footgun.
     * make_env writes this after Dust2EnvC.from_address. env_reset memsets
     * GameState only, so the flag survives mid-round reset. */
    int32_t recoil_enabled;
} Dust2Env;

/* ── Angle utilities ────────────────────────────────────────────────────── */
/* Batch 3: wrap a radian value into [-π, +π].
 * Used by env_step to keep `a->facing` bounded after applying Δyaw.
 * Loop form (vs `fmodf`) is fine — typical input is `a->facing + clamped`
 * where clamped ≤ π/4, so worst case is one iteration. */
static inline float wrap_pi(float x) {
    while (x > (float)M_PI)
        x -= 2.0f * (float)M_PI;
    while (x < -(float)M_PI)
        x += 2.0f * (float)M_PI;
    return x;
}

/* ── RNG utility (available to all headers) ─────────────────────────────── */
static inline uint32_t xorshift32(uint32_t* state) {
    uint32_t x  = *state;
    x          ^= x << 13;
    x          ^= x >> 17;
    x          ^= x << 5;
    return (*state = x);
}
