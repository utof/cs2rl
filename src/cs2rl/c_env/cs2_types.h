/* src/cs2rl/c_env/cs2_types.h */
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
/* Horizontal body radius. Must match DrawCylinder in cs2_render.h.
 * When area_bounds is set this is a 12u axis hull; dust2 (NULL bounds)
 * stays a point on the raster. Without the hull the 12u mesh sits inside
 * the 8u exterior wall. Axis samples only — corners can still clip ~5u.
 * Not a cliff/adjacency test. */
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
 *     into src/cs2rl/_obs_spec.py so Python masking / demo-zeroing code imports
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
 * literals stay frozen at v2-105dim per gh #34.)
 * sim-v3 (R0-E.1, #130): enemy slots +0/+1 (rel-pos) and +5/+6 (bearing) are
 * FACING-RELATIVE (rotated by -facing; memory fallback too). Same dim count.
 * Deploy OBS_VERSION untouched — the deploy sidecar is still absolute, so
 * policies trained after this bump must NOT be exported. */
#define SIM_OBS_VERSION       "sim-v3-110dim"
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

/* ── Solid face geometry (shared by movement, LoS and the renderer) ──
 *
 * Baked by build_solids_from_rooms() in cs2_solids.h from the room quads.
 * Axis-aligned only: either x0==x1 (vertical seg) or y0==y1 (horizontal).
 *
 * The endpoints are the TRUE room edge, never the ±WALL_DEPTH/2 line the
 * demo cube is drawn on — collision and the draw offset must not disagree.
 * draw_walls() re-applies that offset using (nx, ny) and `kind`.
 *
 * Pitfall: WallC in cs2_env.py DOES mirror this struct field for field —
 * it is the pointee type of WallListC.walls, so a stale mirror makes every
 * Python-side walls[i] read the wrong bytes (and silently: ctypes cannot
 * see the C layout). Appending a field here means appending it there too;
 * the _Static_assert below and the matching ctypes.sizeof(WallC) assert in
 * cs2_env.py are what turn a forgotten update into a build/import error.
 * Separately: fields may NOT be inserted into StaticData before wall_list,
 * whose byte offset cs2_env.py also asserts.
 */
typedef struct {
    float x0, y0, x1, y1; /* segment endpoints in world space (sim XY coords) */
    float height;         /* extrusion height in world units */
    float z0;             /* sim z of face base; 0 = ground */
    /* Unit outward normal of the owning room's edge (axis-aligned: exactly
     * one of nx/ny is ±1, the other 0). Points away from the room that
     * emitted the face — i.e. into the void for an exterior wall, and down
     * onto the lower room for a lip. */
    float   nx, ny;
    int32_t kind; /* SOLID_KIND_* in cs2_solids.h — drives the draw offset */
} Wall;

/* 8 floats + 1 int32, all 4-byte aligned → 36 with no padding on every
 * target we build for. Kept in lock-step with WallC in cs2_env.py, which
 * asserts the same number from the ctypes side.
 * Pitfall: we build with -std=c99, where glibc's <sys/cdefs.h> replaces
 * _Static_assert with a negative-bitfield trick — a failure here reports
 * "bit-field '__error_if_negative' has negative width", not the message
 * below. Same line number, so read this line and ignore the wording. */
_Static_assert(sizeof(Wall) == 36, "Wall layout changed — update WallC in cs2_env.py");

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
     *   - SD_PREFIX_FIELDS below              (per-field layout table: every
     *     field before wall_list needs a row, in this same order)
     *   - the ten-pointer argument list in binding.c py_init() — POINTER
     *     fields only; scalars travel in the packed buffer
     * A mismatch between the first two changes the layout hash on one side
     * only, so binding.init refuses to copy anything (spec 2026-08-31 §2 W2);
     * a missing SD_PREFIX_FIELDS row fails tests/test_static_data_layout.py.
     * For the ten POINTER fields the order also decides which numpy array each
     * one receives. cs2_env.py derives the order it PASSES from the mirror
     * (_SD_POINTER_FIELDS), so the call site follows a reorder on its own —
     * but py_init's list is hand-written and does not, and nothing catches
     * that: a consistent reorder leaves both layout hashes equal, and the
     * Python-side pointer guard compares sets, not order. Reorder pointer
     * fields in all three places or in none. */
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
    /* Rung 0 (spec 2026-08-29 §2.1 / R0-E.2) + Rung 1a (spec 2026-08-30 T2a):
     * sim-level knobs, all int32, packed by name like every other prefix field.
     * Inserted BEFORE wall_list so the two pointer-ish tail fields stay last
     * and their offsets move together on both sides (StaticDataC mirrors this
     * same position).
     *   n_active_per_team — agents per team that spawn (1..TEAM_SIZE). Slots
     *                       >= n are "parked": participating=0, alive=0,
     *                       area_idx=INVALID_AREA_IDX, enemy_mem_idx[*]=
     *                       INVALID_AREA_IDX (NOT the zero-init value, which
     *                       is a valid area); env_init asserts >= 1.
     *   pin_pitch         — 1 => continuous_actions[i*AIM_DIM+1] is ignored,
     *                       a->pitch stays 0 (flat maps; R0-E.2). Declared
     *                       here in Rung 0; the consumer lands in a later task.
     *   crouch_enabled    — 0 => the sim IGNORES HEAD_CROUCH entirely
     *                       (W5, #156): process_movement zeroes crouch_act at
     *                       the read, so the agent never crouches and the
     *                       crouch histogram never counts the press, on EVERY
     *                       path — including raw env_step callers that bypass
     *                       the mask (scripted bots #152, BC replay, tests).
     *                       compute_masks still masks HEAD_CROUCH bin 1; that
     *                       is now an optimisation (don't spend policy
     *                       probability mass on a bin the sim drops), not the
     *                       mechanism.
     *   jump_enabled      — 0 => the sim IGNORES HEAD_JUMP entirely. Exact
     *                       mirror of crouch_enabled, same W5 guard, same
     *                       surviving mask (Rung 1a shrinks the action space to
     *                       the aim problem; a jumping agent also leaves the
     *                       pinned-pitch hit band, gh #150).
     * PITFALL: cs2_demo.c load_nav_data memsets StaticData and assigns by
     * name — it must set n_active_per_team=TEAM_SIZE, pin_pitch=0 and
     * crouch_enabled=jump_enabled=1 or the demo silently parks everyone /
     * disables crouch / disables jump. */
    int32_t n_active_per_team;
    int32_t pin_pitch;
    int32_t crouch_enabled;
    int32_t jump_enabled;
    /* Baked solid faces (cs2_solids.h). build_solids_from_rooms() is the ONLY
     * allocation site; env_close() and c_close() both free it via free_solids.
     * Per-env: binding.c puts Dust2Env and StaticData in one calloc, so this
     * list is never shared between envs. */
    WallList wall_list;
    /* Ramp interpolation AABB. After wall_list. StaticDataC appends wall_list
     * then this so Python can publish the room quad after env_init.
     * Measured offsetof: wall_list=496, area_bounds=512 (no pad after
     * pbrs_nav_weight_ct, which ends at 480). These are DOCUMENTATION ONLY —
     * binding.c publishes the real offsetof and cs2_env.py asserts the mirror
     * against it, which is what actually keeps the two sides together. The
     * quoted pair went stale once already (it still read 480/496 after Rung 0
     * appended three int32 knobs), so trust the assert, not this line.
     * Sizing note: the sim-knob block above is 8-aligned as a whole, so
     * jump_enabled (Rung 1a, the 4th int32) landed in the pad that already sat
     * between crouch_enabled and wall_list — sizeof(StaticData) stayed 520 and
     * neither offset moved. A FIFTH int32 knob will move both.
     * NULL → centroids_z (dust2).
     *
     * Ownership: BORROWED, always. The two writers are cs2_env.py (a numpy
     * array kept alive in Cs2Env._refs) and make_client (nav_data.h statics).
     * C never allocates it, so nothing here may free it — env_close frees
     * wall_list and nothing else. There used to be an `area_bounds_owned`
     * flag beside this pointer for a C-malloc'd variant that never shipped;
     * it was written by Python, read by no C code, and freed by nobody. */
    const float* area_bounds; /* [N*4] x0,y0,x1,y1; NULL = no interpolation */
} StaticData;

/* ── StaticData prefix layout table (X-macro) ─────────────────────────────────
 *
 * WHAT: one row per StaticData field in [0, offsetof(StaticData, wall_list)) —
 * the "prefix", i.e. everything Python publishes through binding.init — in
 * declaration order. Consumed by py_static_data_layout() in binding.c, which
 * turns each row into (name, offset, size, canonical type name) and hashes the
 * result. tests/test_static_data_layout.py compares that hash against the same
 * quadruples derived by ctypes introspection of StaticDataC in cs2_env.py.
 *
 * X(type, name, is_array):
 *   type     — the field's declared type, EXACTLY as spelled in the struct
 *              above. For an array field this is the ELEMENT type (`float` for
 *              `float delta_x[9]`), because that is what sizeof() needs to
 *              recover the count. binding.c maps these spellings to the
 *              canonical vocabulary the ctypes side introspects (`int` and
 *              `int32_t` both land on c_int, because ctypes folds
 *              c_int32 into c_int and so cannot tell them apart either).
 *   name     — the field name. Stringified for the hash, and fed to offsetof /
 *              sizeof, so key and value cannot disagree.
 *   is_array — 1 for the fixed-size array fields, 0 otherwise. NOT inferred
 *              from `sizeof(field) != sizeof(type)`: that inference silently
 *              mis-labels a one-element array as a scalar, and the resulting
 *              canonical name would then disagree with ctypes forever.
 *
 * Both `type` and `is_array` are checked against the real struct member at
 * compile time by struct sd_prefix_type_checks, right below this table — a row
 * that describes its field wrongly does not build.
 *
 * WHY a separate list rather than generating the struct from this macro: the
 * struct above carries ~90 lines of interleaved field documentation that a
 * macro body (every line backslash-continued) would mangle, and rewriting a
 * 520-byte layout to add a hash is a worse trade than maintaining one extra
 * name list. What this list must never contain is a hand-written offset or
 * size — those are all offsetof/sizeof EXPRESSIONS in binding.c, so the numbers
 * come from the compiler that laid the struct out. (cs2_types.h has already had
 * one pair of hand-quoted offsets go stale; see the wall_list comment above.)
 *
 * PITFALL: adding a prefix field here but not to StaticDataC in cs2_env.py (or
 * vice versa) fails tests/test_static_data_layout.py. Adding it to the struct
 * and to NEITHER is caught by the sizeof(StaticData) guard — unless the new
 * field fits entirely inside existing padding, which has happened once already
 * (jump_enabled, see the wall_list comment). So: struct, mirror, and this table,
 * in the same commit, always. */
#define SD_PREFIX_FIELDS(X)                                                                        \
    X(int, N, 0)                                                                                   \
    X(int8_t*, vis_matrix, 0)                                                                      \
    X(int32_t*, raster_grid, 0)                                                                    \
    X(int8_t*, adjacency, 0)                                                                       \
    X(float*, centroid_xy, 0)                                                                      \
    X(float*, centroids_z, 0)                                                                      \
    X(int32_t*, area_ids, 0)                                                                       \
    X(int8_t*, bombsite_mask, 0)                                                                   \
    X(int8_t*, bombsite_by_idx, 0)                                                                 \
    X(int8_t*, is_ramp, 0)                                                                         \
    X(float*, bombsite_dist, 0)                                                                    \
    X(int, grid_w, 0)                                                                              \
    X(int, grid_h, 0)                                                                              \
    X(int, max_area_id, 0)                                                                         \
    X(float, grid_x_min, 0)                                                                        \
    X(float, grid_y_min, 0)                                                                        \
    X(float, grid_inv_cell, 0)                                                                     \
    X(float, inv_x_range, 0)                                                                       \
    X(float, inv_y_range, 0)                                                                       \
    X(float, x_offset, 0)                                                                          \
    X(float, y_offset, 0)                                                                          \
    X(float, bombsite_dist_scale, 0)                                                               \
    X(int32_t, laser_damage, 0)                                                                    \
    X(float, laser_range, 0)                                                                       \
    X(float, laser_range_sq, 0)                                                                    \
    X(int32_t, shoot_cooldown, 0)                                                                  \
    X(int32_t, bomb_plant_time, 0)                                                                 \
    X(int32_t, bomb_defuse_time, 0)                                                                \
    X(int32_t, bomb_defuse_kit, 0)                                                                 \
    X(int32_t, bomb_timer, 0)                                                                      \
    X(int32_t, round_time, 0)                                                                      \
    X(float, footstep_radius_sq, 0)                                                                \
    X(float, gunshot_radius_sq, 0)                                                                 \
    X(int32_t, enemy_memory_ticks, 0)                                                              \
    X(int32_t, stale_memory_tick, 0)                                                               \
    X(float, pbrs_gamma, 0)                                                                        \
    X(float, delta_x, 1)                                                                           \
    X(float, delta_y, 1)                                                                           \
    X(float, dir_facing, 1)                                                                        \
    X(int32_t, t_spawns, 1)                                                                        \
    X(int, n_t_spawns, 0)                                                                          \
    X(int32_t, ct_spawns, 1)                                                                       \
    X(int, n_ct_spawns, 0)                                                                         \
    X(float, max_turn_speed, 0)                                                                    \
    X(float, reward_win, 0)                                                                        \
    X(float, reward_win_t_detonation, 0)                                                           \
    X(float, reward_win_t_elimination, 0)                                                          \
    X(float, reward_win_ct_defuse, 0)                                                              \
    X(float, reward_win_ct_timeout, 0)                                                             \
    X(float, reward_win_ct_elimination, 0)                                                         \
    X(float, reward_kill, 0)                                                                       \
    X(float, reward_death, 0)                                                                      \
    X(float, reward_bombsite_entry, 0)                                                             \
    X(float, reward_plant_bonus, 0)                                                                \
    X(float, reward_plant_base, 0)                                                                 \
    X(float, reward_plant_progress_scale, 0)                                                       \
    X(float, reward_plant_interrupted, 0)                                                          \
    X(float, reward_defuse, 0)                                                                     \
    X(float, reward_shot_penalty, 0)                                                               \
    X(float, reward_ct_survival, 0)                                                                \
    X(float, reward_inaction, 0)                                                                   \
    X(float, pbrs_alive_weight, 0)                                                                 \
    X(float, pbrs_hp_weight, 0)                                                                    \
    X(float, pbrs_site_weight, 0)                                                                  \
    X(float, pbrs_bomb_progress_weight, 0)                                                         \
    X(float, pbrs_nav_weight_t, 0)                                                                 \
    X(float, pbrs_nav_weight_ct, 0)                                                                \
    X(int32_t, n_active_per_team, 0)                                                               \
    X(int32_t, pin_pitch, 0)                                                                       \
    X(int32_t, crouch_enabled, 0)                                                                  \
    X(int32_t, jump_enabled, 0)

/* ── Table-row type == struct-member type (compile-time) ──────────────────────
 *
 * WHAT: one check per SD_PREFIX_FIELDS row, asserting that the row's `type`
 * column is the type the StaticData member above ACTUALLY has. A mismatch is a
 * compile error naming the field; the table row that caused it shows up in the
 * macro-expansion note under the error.
 *
 * WHY: offset and size in the layout table are offsetof/sizeof EXPRESSIONS, so
 * the compiler owns them and no one can get them wrong. The type column is not
 * — it is a hand-written string, and without this check nothing compares it to
 * the struct. Editing `int32_t stale_memory_tick` to `float` in the STRUCT ONLY
 * (leaving this table and StaticDataC in cs2_env.py saying int32_t) left all 71
 * layout tests, the layout hash, and the sizeof(StaticData) guard green — the
 * struct held a float that both descriptions called an int. A struct-only type
 * edit must fail HERE, not silently reinterpret the bytes Python packs into the
 * prefix (spec 2026-08-31 §2 W2); a pointer element-type change (`int8_t*` →
 * `int32_t*`, same 8-byte field) is an out-of-bounds read that no runtime test
 * in this tree would name.
 *
 * `int` and `int32_t` are deliberately COMPATIBLE here on this target — that is
 * the same folding ctypes does (c_int32 IS c_int), which the whole two-sided
 * design rests on, so this check must not be stricter than the hash it guards.
 *
 * The is_array arm compares `ctype[]` against the member's array type, so a
 * wrong is_array flag also fails to compile rather than producing a plausible
 * `arr_<base>_<count>` — the one part of the row the hash could not police.
 *
 * PITFALL: GNU builtins, not C11 `_Static_assert`, because build.zig pins
 * `-std=c99`; both are available in clang (zig cc) in any -std mode. This block
 * is C-only — `__builtin_types_compatible_p` does not exist in C++, and no C++
 * TU includes this header today. Deliberately unguarded by `#ifdef __GNUC__`: a
 * compiler that cannot run the check should fail loudly, not skip a layout
 * guard in silence. */
#define SD_ROW_TYPE_MATCHES_0(ctype, f)                                                            \
    __builtin_types_compatible_p(ctype, __typeof__(((StaticData*)0)->f))
#define SD_ROW_TYPE_MATCHES_1(ctype, f)                                                            \
    __builtin_types_compatible_p(ctype[], __typeof__(((StaticData*)0)->f))
/* Two levels so `is_array` is expanded before it is pasted onto the name. */
#define SD_ROW_TYPE_MATCHES_(is_array, ctype, f) SD_ROW_TYPE_MATCHES_##is_array(ctype, f)
#define SD_ROW_TYPE_MATCHES(is_array, ctype, f)  SD_ROW_TYPE_MATCHES_(is_array, ctype, f)

/* A negative array size is the C99 way to fail a compile-time predicate. The
 * member name IS the diagnostic, since a negative-size error quotes it. */
#define SD_ROW_TYPE_CHECK(ctype, f, is_array)                                                      \
    char sd_table_type_must_match_struct_##f[SD_ROW_TYPE_MATCHES(is_array, ctype, f) ? 1 : -1];

/* Type declaration only — never defined, never instantiated, costs no bytes. */
struct sd_prefix_type_checks {
    SD_PREFIX_FIELDS(SD_ROW_TYPE_CHECK)
};

#undef SD_ROW_TYPE_CHECK
#undef SD_ROW_TYPE_MATCHES
#undef SD_ROW_TYPE_MATCHES_
#undef SD_ROW_TYPE_MATCHES_1
#undef SD_ROW_TYPE_MATCHES_0

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
    /* Rung 0 (spec 2026-08-29 §2.1): 1 for the n_active_per_team slots per team
     * that spawned this round, 0 for parked slots. Written by spawn_team for
     * active slots and by env_reset for parked ones, right after the two
     * spawn_team calls. Read by the deliberately alive-AGNOSTIC loops in
     * cs2_rewards.h (terminal win payout, PBRS), which would otherwise pay a
     * parked row as a team member.
     * PITFALL: `participating` is NOT redundant with `alive`. A parked slot and
     * a killed slot are both alive=0, but only the killed one is owed team
     * reward. Any new loop that ignores `alive` on purpose must gate on this.
     * PITFALL: explicit pad — AgentStateC mirrors both this and _pad5. */
    int8_t participating;
    int8_t _pad5[3];
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
     * tracker (reassigned on drop+auto-pickup; pickup is in process_bomb, cs2_bomb.h). This
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
    /* ── Rung 0 R0-A (spec 2026-08-29 §3): combat instrumentation ──
     * Written into BOTH step_stats and episode_stats at the accumulation
     * site (there is no ss→es merge). Shooter counters are per round fired
     * by a participating && alive agent; the *_facing/_on_target/_hit/
     * _stance_blocked subsets are scored against the nearest visible
     * participating enemy snapshotted BEFORE process_combat (cs2_env.h), so
     * a same-tick kill cannot make a shot "unscored". mutual_vis pair
     * counters use vis10[i][j] && vis10[j][i] (DDA is not symmetric) over
     * participating && alive opposing pairs; agent_ticks_with_visible_enemy
     * is ONE-directional (i sees any j), by design. min_enemy_distance is 2D
     * and counted regardless of
     * visibility; sentinel 1e30f (never INFINITY — -ffast-math) set in
     * clear_stats; converted per episode in cs2_env.py _build_terminal_info.
     * reward_win_t/ct: one-sided terminal payouts (diagnostic; reward_win
     * stays the cross-team sum). Appended — never reorder. reward_win_ct is
     * the struct tail: binding.c py_struct_sizes() and cs2_env.py
     * _C_OFFSET_FIELDS anchor on it. */
    int32_t shots_fired;
    int32_t shots_with_enemy_in_los;
    int32_t shots_facing_enemy;
    int32_t shots_on_target;
    int32_t shots_hit;
    int32_t shots_stance_blocked;
    int32_t mutual_vis_pair_ticks;
    int32_t agent_ticks_with_visible_enemy;
    float   damage_dealt;
    float   min_enemy_distance;
    float   reward_win_t;
    float   reward_win_ct;
} StepStats;

/* ── Full environment (one per parallel instance) ── */
typedef struct {
    /* Not owned by this struct: binding.c allocates Dust2Env and its
     * StaticData in one calloc, so the block dies with the capsule.
     * env_close frees sd->wall_list and nothing else — area_bounds is
     * borrowed from Python/nav (see the StaticData field comment). */
    StaticData* sd;
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
     * (cs2_demo). Not reachable through binding.init: that call carries the
     * StaticData prefix, and this flag lives on Dust2Env instead.
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
