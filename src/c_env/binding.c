/* src/c_env/binding.c — Python C API bridge for the Dust2 C environment.
 *
 * Exposes: binding.init(...) -> capsule
 *          binding.reset(capsule)
 *          binding.step(capsule, actions_array, continuous_actions_array)
 *          binding.close(capsule)
 *          binding.get_buffers(capsule) -> (obs_ptr, rew_ptr, term_ptr, trunc_ptr)
 */
#define PY_ARRAY_UNIQUE_SYMBOL cs2rl_binding_ARRAY_API
#define NPY_NO_DEPRECATED_API  NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stddef.h> /* offsetof — used by py_struct_sizes below */
#include <stdlib.h>
#include <string.h>
#include "cs2_env.h"
#ifdef CS2_DEMO_VIZ_H
#error "binding must not include cs2_demo_viz.h (aim-stick stays out of the env graph)"
#endif

/* Single heap allocation holding both Dust2Env and its StaticData.
 * env is first field so &benv == &benv->env — Python's env_ptr extracts this address. */
typedef struct {
    Dust2Env   env;
    StaticData sd;
    int        closed;
} BindingEnv;

static void capsule_destructor(PyObject* cap) {
    BindingEnv* benv = (BindingEnv*)PyCapsule_GetPointer(cap, NULL);
    if (benv) {
        if (!benv->closed)
            env_close(&benv->env);
        free(benv);
    }
}

/* ── binding.init(...) -> PyCapsule ── */
static PyObject* py_init(PyObject* self, PyObject* args) {
    PyObject *vis_matrix_o, *raster_grid_o, *adjacency_o, *centroid_xy_o;
    /* T2 (verticality): centroids_z immediately after centroid_xy; is_ramp immediately
     * after bombsite_by_idx.  Field order here MUST match:
     *   - StaticData struct in cs2_types.h   (C canonical source)
     *   - StaticDataC._fields_ in cs2_env.py (ctypes mirror)
     * Mismatch silently hands area_ids data to is_ramp ptr (etc.). */
    PyObject*    centroids_z_o;
    PyObject *   area_ids_o, *bombsite_mask_o, *bombsite_by_idx_o;
    PyObject*    is_ramp_o;
    PyObject*    bombsite_dist_o;
    int          N, grid_w, grid_h, max_area_id;
    float        grid_x_min, grid_y_min, grid_inv_cell;
    float        inv_x_range, inv_y_range, x_offset, y_offset, bombsite_dist_scale;
    int          laser_damage;
    float        laser_range, laser_range_sq;
    int          shoot_cooldown, bomb_plant_time, bomb_defuse_time;
    int          bomb_defuse_kit, bomb_timer, round_time;
    float        footstep_radius_sq, gunshot_radius_sq;
    int          enemy_memory_ticks, stale_memory_tick;
    float        pbrs_gamma;
    PyObject *   delta_x_o, *delta_y_o, *dir_facing_o, *t_spawns_o;
    int          n_t_spawns;
    PyObject*    ct_spawns_o;
    int          n_ct_spawns;
    float        max_turn_speed;
    unsigned int seed;
    float        team_spirit;

    /* Phase 5 reward weights */
    float reward_win, reward_kill, reward_death, reward_bombsite_entry;
    float reward_plant_bonus, reward_plant_base, reward_plant_progress_scale;
    float reward_plant_interrupted, reward_defuse, reward_shot_penalty;
    float reward_ct_survival, reward_inaction;
    float pbrs_alive_weight, pbrs_hp_weight, pbrs_site_weight;
    float pbrs_bomb_progress_weight, pbrs_nav_weight_t, pbrs_nav_weight_ct;

    /* Batch 1 (RL overhaul): per-outcome win magnitudes — appended after reward_win
     * to match the StaticData field order in cs2_types.h. */
    float reward_win_t_detonation, reward_win_t_elimination;
    float reward_win_ct_defuse, reward_win_ct_timeout, reward_win_ct_elimination;

    /* Rung 0 (spec 2026-08-29 §2.1 / R0-E.2): sim knobs. `int`, not int32_t —
     * PyArg_ParseTuple's "i" writes an int and nothing else is safe here. */
    int n_active_per_team, pin_pitch, crouch_enabled;

    /* 72-arg format string — positions match StaticDataC._fields_ order from cs2_env.py.
     * T2 (verticality): added centroids_z_o after centroid_xy_o (pos 4) and is_ramp_o
     * after bombsite_by_idx_o (pos 8), for 10 O args total instead of 8.
     * Total: 10O + 4i + 8f + i + 2f + 6i + 2f + 2i + f + 4O + i + O + i + f + I + f
     *      + f(reward_win) + 5f(Batch1) + 17f(Phase5-rest) + 3i(Rung 0) = 72 args.
     * CRITICAL: positions must stay in sync with StaticDataC._fields_ in cs2_env.py
     * and StaticData in cs2_types.h — mismatch silently corrupts pointer assignments. */
    static const char FMT[] =
        "OOOOOO"            /* 0-5:  vis_matrix, raster_grid, adjacency, centroid_xy,
                                      centroids_z, area_ids */
        "OOOO"              /* 6-9:  bombsite_mask, bombsite_by_idx, is_ramp, bombsite_dist */
        "iiii"              /* 10-13: N, grid_w, grid_h, max_area_id */
        "ffffffff"          /* 14-21: grid_x_min, grid_y_min, grid_inv_cell,
                                       inv_x_range, inv_y_range, x_offset, y_offset, bombsite_dist_scale */
        "i"                 /* 22: laser_damage */
        "ff"                /* 23-24: laser_range, laser_range_sq */
        "iiiiii"            /* 25-30: shoot_cooldown, bomb_plant_time, bomb_defuse_time,
                                       bomb_defuse_kit, bomb_timer, round_time */
        "ff"                /* 31-32: footstep_radius_sq, gunshot_radius_sq */
        "ii"                /* 33-34: enemy_memory_ticks, stale_memory_tick */
        "f"                 /* 35: pbrs_gamma */
        "OOOO"              /* 36-39: delta_x, delta_y, dir_facing, t_spawns (all arrays) */
        "i"                 /* 40: n_t_spawns */
        "O"                 /* 41: ct_spawns (array) */
        "i"                 /* 42: n_ct_spawns */
        "f"                 /* 43: max_turn_speed */
        "I"                 /* 44: seed (unsigned int) */
        "f"                 /* 45: team_spirit */
        "f"                 /* 46: reward_win (legacy symmetric) */
        "fffff"             /* 47-51: Batch 1 per-mechanism win magnitudes */
        "fffffffffffffffff" /* 52-68: 17 remaining Phase-5 reward weights
                                       (reward_kill through pbrs_nav_weight_ct) */
        "iii";              /* 69-71: n_active_per_team, pin_pitch, crouch_enabled (Rung 0) */

    if (!PyArg_ParseTuple(args,
                          FMT,
                          &vis_matrix_o,
                          &raster_grid_o,
                          &adjacency_o,
                          &centroid_xy_o,
                          &centroids_z_o, /* T2: terrain z per area (float32[N]) */
                          &area_ids_o,
                          &bombsite_mask_o,
                          &bombsite_by_idx_o,
                          &is_ramp_o, /* T2: ramp flag per area (int8[N]) */
                          &bombsite_dist_o,
                          &N,
                          &grid_w,
                          &grid_h,
                          &max_area_id,
                          &grid_x_min,
                          &grid_y_min,
                          &grid_inv_cell,
                          &inv_x_range,
                          &inv_y_range,
                          &x_offset,
                          &y_offset,
                          &bombsite_dist_scale,
                          &laser_damage,
                          &laser_range,
                          &laser_range_sq,
                          &shoot_cooldown,
                          &bomb_plant_time,
                          &bomb_defuse_time,
                          &bomb_defuse_kit,
                          &bomb_timer,
                          &round_time,
                          &footstep_radius_sq,
                          &gunshot_radius_sq,
                          &enemy_memory_ticks,
                          &stale_memory_tick,
                          &pbrs_gamma,
                          &delta_x_o,
                          &delta_y_o,
                          &dir_facing_o,
                          &t_spawns_o,
                          &n_t_spawns,
                          &ct_spawns_o,
                          &n_ct_spawns,
                          &max_turn_speed,
                          &seed,
                          &team_spirit,
                          &reward_win,
                          &reward_win_t_detonation,
                          &reward_win_t_elimination,
                          &reward_win_ct_defuse,
                          &reward_win_ct_timeout,
                          &reward_win_ct_elimination,
                          &reward_kill,
                          &reward_death,
                          &reward_bombsite_entry,
                          &reward_plant_bonus,
                          &reward_plant_base,
                          &reward_plant_progress_scale,
                          &reward_plant_interrupted,
                          &reward_defuse,
                          &reward_shot_penalty,
                          &reward_ct_survival,
                          &reward_inaction,
                          &pbrs_alive_weight,
                          &pbrs_hp_weight,
                          &pbrs_site_weight,
                          &pbrs_bomb_progress_weight,
                          &pbrs_nav_weight_t,
                          &pbrs_nav_weight_ct,
                          &n_active_per_team,
                          &pin_pitch,
                          &crouch_enabled))
        return NULL;

    BindingEnv* benv = (BindingEnv*)calloc(1, sizeof(BindingEnv));
    if (!benv)
        return PyErr_NoMemory();

    StaticData* sd  = &benv->sd;
    sd->N           = N;
    sd->vis_matrix  = (int8_t*)PyArray_DATA((PyArrayObject*)vis_matrix_o);
    sd->raster_grid = (int32_t*)PyArray_DATA((PyArrayObject*)raster_grid_o);
    sd->adjacency   = (int8_t*)PyArray_DATA((PyArrayObject*)adjacency_o);
    sd->centroid_xy = (float*)PyArray_DATA((PyArrayObject*)centroid_xy_o);
    /* T2 (verticality): store terrain-z and ramp-flag pointers.
     * Python side passes centroids_z as float32 and is_ramp as int8 (bool converted
     * via .astype(np.int8) before passing — see cs2_env.py _refs).  The numpy arrays
     * are kept alive by self._refs on the Cs2Env instance; these raw pointers are valid
     * for the env's lifetime as long as the Python-side Cs2Env stays alive. */
    sd->centroids_z     = (float*)PyArray_DATA((PyArrayObject*)centroids_z_o);
    sd->area_ids        = (int32_t*)PyArray_DATA((PyArrayObject*)area_ids_o);
    sd->bombsite_mask   = (int8_t*)PyArray_DATA((PyArrayObject*)bombsite_mask_o);
    sd->bombsite_by_idx = (int8_t*)PyArray_DATA((PyArrayObject*)bombsite_by_idx_o);
    sd->is_ramp         = (int8_t*)PyArray_DATA((PyArrayObject*)is_ramp_o);
    sd->bombsite_dist   = (float*)PyArray_DATA((PyArrayObject*)bombsite_dist_o);

    sd->grid_w              = grid_w;
    sd->grid_h              = grid_h;
    sd->max_area_id         = max_area_id;
    sd->grid_x_min          = grid_x_min;
    sd->grid_y_min          = grid_y_min;
    sd->grid_inv_cell       = grid_inv_cell;
    sd->inv_x_range         = inv_x_range;
    sd->inv_y_range         = inv_y_range;
    sd->x_offset            = x_offset;
    sd->y_offset            = y_offset;
    sd->bombsite_dist_scale = bombsite_dist_scale;

    sd->laser_damage       = (int32_t)laser_damage;
    sd->laser_range        = laser_range;
    sd->laser_range_sq     = laser_range_sq;
    sd->shoot_cooldown     = (int32_t)shoot_cooldown;
    sd->bomb_plant_time    = (int32_t)bomb_plant_time;
    sd->bomb_defuse_time   = (int32_t)bomb_defuse_time;
    sd->bomb_defuse_kit    = (int32_t)bomb_defuse_kit;
    sd->bomb_timer         = (int32_t)bomb_timer;
    sd->round_time         = (int32_t)round_time;
    sd->footstep_radius_sq = footstep_radius_sq;
    sd->gunshot_radius_sq  = gunshot_radius_sq;
    sd->enemy_memory_ticks = (int32_t)enemy_memory_ticks;
    sd->stale_memory_tick  = (int32_t)stale_memory_tick;
    sd->pbrs_gamma         = pbrs_gamma;

    memcpy(sd->delta_x, PyArray_DATA((PyArrayObject*)delta_x_o), 9 * sizeof(float));
    memcpy(sd->delta_y, PyArray_DATA((PyArrayObject*)delta_y_o), 9 * sizeof(float));
    memcpy(sd->dir_facing, PyArray_DATA((PyArrayObject*)dir_facing_o), 9 * sizeof(float));

    sd->n_t_spawns = n_t_spawns;
    memcpy(sd->t_spawns,
           PyArray_DATA((PyArrayObject*)t_spawns_o),
           (size_t)n_t_spawns * sizeof(int32_t));
    sd->n_ct_spawns = n_ct_spawns;
    memcpy(sd->ct_spawns,
           PyArray_DATA((PyArrayObject*)ct_spawns_o),
           (size_t)n_ct_spawns * sizeof(int32_t));

    sd->max_turn_speed = max_turn_speed;

    sd->reward_win = reward_win;
    /* Batch 1 (RL overhaul): per-outcome win magnitudes */
    sd->reward_win_t_detonation     = reward_win_t_detonation;
    sd->reward_win_t_elimination    = reward_win_t_elimination;
    sd->reward_win_ct_defuse        = reward_win_ct_defuse;
    sd->reward_win_ct_timeout       = reward_win_ct_timeout;
    sd->reward_win_ct_elimination   = reward_win_ct_elimination;
    sd->reward_kill                 = reward_kill;
    sd->reward_death                = reward_death;
    sd->reward_bombsite_entry       = reward_bombsite_entry;
    sd->reward_plant_bonus          = reward_plant_bonus;
    sd->reward_plant_base           = reward_plant_base;
    sd->reward_plant_progress_scale = reward_plant_progress_scale;
    sd->reward_plant_interrupted    = reward_plant_interrupted;
    sd->reward_defuse               = reward_defuse;
    sd->reward_shot_penalty         = reward_shot_penalty;
    sd->reward_ct_survival          = reward_ct_survival;
    sd->reward_inaction             = reward_inaction;
    sd->pbrs_alive_weight           = pbrs_alive_weight;
    sd->pbrs_hp_weight              = pbrs_hp_weight;
    sd->pbrs_site_weight            = pbrs_site_weight;
    sd->pbrs_bomb_progress_weight   = pbrs_bomb_progress_weight;
    sd->pbrs_nav_weight_t           = pbrs_nav_weight_t;
    sd->pbrs_nav_weight_ct          = pbrs_nav_weight_ct;
    /* Rung 0 knobs — range-validated Python-side (Cs2Env.__init__ raises
     * ValueError) and asserted again in env_init, which is the only guard the
     * non-Python callers (cs2_demo.c) get. */
    sd->n_active_per_team = n_active_per_team;
    sd->pin_pitch         = pin_pitch;
    sd->crouch_enabled    = crouch_enabled;

    env_init(&benv->env, sd, (uint32_t)seed, team_spirit);

    PyObject* cap = PyCapsule_New(benv, NULL, capsule_destructor);
    if (!cap) {
        free(benv);
        return NULL;
    }
    return cap;
}

/* ── binding.reset(capsule) -> None ── */
static PyObject* py_reset(PyObject* self, PyObject* args) {
    PyObject* cap;
    if (!PyArg_ParseTuple(args, "O", &cap))
        return NULL;
    Dust2Env* env = (Dust2Env*)PyCapsule_GetPointer(cap, NULL);
    if (!env) {
        PyErr_SetString(PyExc_ValueError, "invalid capsule");
        return NULL;
    }
    env_reset(env);
    Py_RETURN_NONE;
}

/* ── binding.step(capsule, actions_array, continuous_actions_array) -> None ── */
/* Batch 3: continuous_actions is a numpy float32 array shape
 * (N_AGENTS, AIM_DIM). Caller is responsible for dtype + contiguity
 * (Cs2Env.step in cs2_env.py converts/coerces before invoking). The two
 * action buffers are kept SEPARATE — no bit-cast — so the int32 discrete
 * heads and float32 [Δyaw, Δpitch] don't share alignment hazards.
 *
 * Batch 3.5 (#24): cont buffer shape is validated here before PyArray_DATA
 * dereference — see validation block below. */
static PyObject* py_step(PyObject* self, PyObject* args) {
    PyObject *cap, *actions_o, *cont_o;
    if (!PyArg_ParseTuple(args, "OOO", &cap, &actions_o, &cont_o))
        return NULL;
    Dust2Env* env = (Dust2Env*)PyCapsule_GetPointer(cap, NULL);
    if (!env) {
        PyErr_SetString(PyExc_ValueError, "invalid capsule");
        return NULL;
    }
    /* Batch 3.5 (#24): defensively validate continuous_actions shape.
     * Without this, a stale caller passing (N_AGENTS, 1) after the AIM_DIM 1→2
     * bump silently reads OOB at [i*AIM_DIM+1]. PyErr_Format gives a clear deploy-
     * time error instead of a NaN-flooded training run or a segfault.
     * Defense-in-depth: cs2_env.py::_prepare_continuous_actions:779 already raises
     * ValueError on shape mismatch; this check catches callers that bypass the
     * Python wrapper (direct binding consumers, regression tests with raw shapes,
     * future C-only consumers). */
    PyArrayObject* cont_arr = (PyArrayObject*)cont_o;
    if (PyArray_NDIM(cont_arr) != 2 || PyArray_DIM(cont_arr, 0) != N_AGENTS ||
        PyArray_DIM(cont_arr, 1) != AIM_DIM || PyArray_TYPE(cont_arr) != NPY_FLOAT32) {
        PyErr_Format(PyExc_ValueError,
                     "continuous_actions must be (N_AGENTS=%d, AIM_DIM=%d) float32; "
                     "got shape (%ld, %ld) dtype=%d",
                     N_AGENTS,
                     AIM_DIM,
                     (long)PyArray_DIM(cont_arr, 0),
                     (long)PyArray_DIM(cont_arr, 1),
                     PyArray_TYPE(cont_arr));
        return NULL;
    }
    env_step(env,
             (const int32_t*)PyArray_DATA((PyArrayObject*)actions_o),
             (const float*)PyArray_DATA(cont_arr));
    Py_RETURN_NONE;
}

/* ── binding.close(capsule) -> None ── */
static PyObject* py_close(PyObject* self, PyObject* args) {
    PyObject* cap;
    if (!PyArg_ParseTuple(args, "O", &cap))
        return NULL;
    BindingEnv* benv = (BindingEnv*)PyCapsule_GetPointer(cap, NULL);
    if (benv && !benv->closed) {
        env_close(&benv->env);
        benv->closed = 1;
    }
    Py_RETURN_NONE;
}

/* ── binding.get_buffers(capsule) -> (obs_ptr, rew_ptr, term_ptr, trunc_ptr) ── */
static PyObject* py_get_buffers(PyObject* self, PyObject* args) {
    PyObject* cap;
    if (!PyArg_ParseTuple(args, "O", &cap))
        return NULL;
    Dust2Env* env = (Dust2Env*)PyCapsule_GetPointer(cap, NULL);
    if (!env) {
        PyErr_SetString(PyExc_ValueError, "invalid capsule");
        return NULL;
    }
    return Py_BuildValue("(KKKK)",
                         (unsigned long long)(uintptr_t)env->observations,
                         (unsigned long long)(uintptr_t)env->rewards,
                         (unsigned long long)(uintptr_t)env->terminals,
                         (unsigned long long)(uintptr_t)env->truncations);
}

/* ── binding.get_masks(capsule) -> int (pointer as int) ── */
static PyObject* py_get_masks(PyObject* self, PyObject* args) {
    PyObject* cap;
    if (!PyArg_ParseTuple(args, "O", &cap))
        return NULL;
    Dust2Env* env = (Dust2Env*)PyCapsule_GetPointer(cap, NULL);
    if (!env) {
        PyErr_SetString(PyExc_ValueError, "invalid capsule");
        return NULL;
    }
    return PyLong_FromUnsignedLongLong((unsigned long long)(uintptr_t)env->masks);
}

/* ── binding.struct_sizes() -> dict ──
 * The single source of truth for the ctypes mirrors in cs2_env.py.
 *
 * WHAT: sizeof of every struct Python overlays with ctypes, an offsetof anchor
 * on each of those structs' LAST field, plus the two team-size macros nav.py
 * duplicates.
 *
 * WHY: cs2_env.py used to carry hand-measured literals (164/1708/204/6832 and
 * offsets 476/480/496) refreshed by compiling a throwaway printf TU by hand.
 * They rotted on every appended field, and a stale literal turns a real layout
 * drift into a confusing assert about the wrong number. These values come from
 * the same compiler invocation that laid the structs out, so they cannot lie.
 *
 * KEY NAMING — three conventions, do not invent a fourth:
 *   "<Struct>"                 -> sizeof(<Struct>)             e.g. "StaticData"
 *   "<Struct>_<field>_offset"  -> offsetof(<Struct>, <field>)   e.g.
 *                                 "StaticData_wall_list_offset"
 *   "<MACRO>"                  -> a bare cs2_types.h macro      e.g. "TEAM_SIZE"
 *
 * PITFALL: append a key here for every struct/macro a ctypes mirror depends on
 * — a struct with no key here is unguarded, and a struct with ONLY a sizeof key
 * is guarded against the wrong thing: sizeof is blind to a field inserted
 * mid-struct in C but appended at the tail of the mirror. Every struct needs a
 * tail offsetof too, per TAIL ANCHORS below.
 *
 * The converse is enforced, not just asked
 * for: every key published here MUST be consumed by the _C_SIZES guard in
 * cs2_env.py, because test_struct_sizes_keys_are_all_consumed (in
 * tests/test_struct_sizes.py) asserts set(struct_sizes()) == _C_SIZE_KEYS_CHECKED.
 * A key added here and never compared there fails that test instead of sitting
 * unguarded. Sizes use the "n" (Py_ssize_t) format because sizeof yields size_t;
 * the macros use "i" (plain int). */
/* Py_UNUSED(ignored), not `args`: this is METH_NOARGS, so CPython passes NULL
 * as the second argument rather than an empty tuple. Naming it `args` invites a
 * later `PyArg_ParseTuple(args, ...)` to be added here, which would deref NULL.
 * Py_UNUSED mangles the name so it cannot be referenced at all. */
static PyObject* py_struct_sizes(PyObject* self, PyObject* Py_UNUSED(ignored)) {
    (void)self;
    return Py_BuildValue(
        "{s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:i,s:i}",
        "AgentState",
        (Py_ssize_t)sizeof(AgentState),
        "GameState",
        (Py_ssize_t)sizeof(GameState),
        "StepStats",
        (Py_ssize_t)sizeof(StepStats),
        "Dust2Env",
        (Py_ssize_t)sizeof(Dust2Env),
        "StaticData",
        (Py_ssize_t)sizeof(StaticData),
        "Wall",
        (Py_ssize_t)sizeof(Wall),
        "WallList",
        (Py_ssize_t)sizeof(WallList),
        /* TAIL ANCHORS — one offsetof per mirrored struct, on that struct's
         * LAST field. sizeof alone cannot see a field inserted mid-struct here
         * but appended at the end of the ctypes mirror: the total stays equal,
         * every guard passes, and Python reads the wrong bytes forever. The
         * last field's offset does move under that edit, so it catches it.
         * Rule: a new field goes in the SAME position in the C struct and in
         * the mirror. Appending at the tail (the usual case) shifts the anchor
         * on both sides by the same amount, which is exactly what we want —
         * the anchor then names the NEW last field, so update the key here and
         * the entry in _C_OFFSET_FIELDS (cs2_env.py) together.
         * StaticData carries three anchors, not one: wall_list/area_bounds sit
         * after a long run of float reward weights, and pbrs_nav_weight_ct pins
         * the end of that run so a drift inside it is localised. */
        "StaticData_pbrs_nav_weight_ct_offset",
        (Py_ssize_t)offsetof(StaticData, pbrs_nav_weight_ct),
        "StaticData_wall_list_offset",
        (Py_ssize_t)offsetof(StaticData, wall_list),
        "StaticData_area_bounds_offset",
        (Py_ssize_t)offsetof(StaticData, area_bounds),
        "AgentState__pad5_offset",
        (Py_ssize_t)offsetof(AgentState, _pad5),
        /* GameState's last field is the explicit tail pad, not a "real"
         * field. offsetof on a pad array is legal, and the rule is uniform:
         * anchor the LAST field. Picking bomb_is_dropped instead would miss
         * a field slipped in between it and the pad on one side only. */
        "GameState__pad_gs_offset",
        (Py_ssize_t)offsetof(GameState, _pad_gs),
        "StepStats_reward_win_ct_offset",
        (Py_ssize_t)offsetof(StepStats, reward_win_ct),
        "Dust2Env_recoil_enabled_offset",
        (Py_ssize_t)offsetof(Dust2Env, recoil_enabled),
        "Wall_kind_offset",
        (Py_ssize_t)offsetof(Wall, kind),
        "WallList_capacity_offset",
        (Py_ssize_t)offsetof(WallList, capacity),
        "TEAM_SIZE",
        TEAM_SIZE,
        "N_AGENTS",
        N_AGENTS);
}

/* ── binding.static_data_scalars(capsule) -> dict ──
 * Read every scalar StaticData field back out of a live env.
 *
 * WHAT: EVERY plain-number field of StaticData (int / int32_t / float), keyed
 * by its C field name — all 55 of them, not a curated subset. Excluded, because
 * they are not scalars: the pointer fields, the fixed arrays (delta_x, delta_y,
 * dir_facing, t_spawns, ct_spawns) and the nested wall_list / area_bounds,
 * which struct_sizes() covers with offsetof keys instead.
 *
 * That completeness is ENFORCED, not merely documented:
 * test_static_data_scalars_covers_every_scalar_field in
 * tests/test_struct_sizes.py compares this dict's key set against the
 * scalar-typed fields of the StaticDataC ctypes mirror. Appending a field to
 * cs2_types.h + the mirror and forgetting this function fails that test.
 *
 * WHY: struct_sizes() cannot detect a mis-ordered PyArg_ParseTuple FMT string
 * in py_init — two floats swapped still parse, still have identical sizes, and
 * silently feed reward_kill into reward_death. Tests push distinct sentinels
 * through Cs2Env and read them back here, so a transposition fails loudly.
 *
 * FMT ARG NUMBERING — stated once, used everywhere in this file and in the
 * positional comments on cs2_env.py's binding.init() call: FMT arg numbers are
 * 0-INDEXED (arg 0 is vis_matrix). CPython's own PyArg_ParseTuple failures are
 * 1-indexed ("argument 22 must be..."), so when cross-referencing a real error
 * message subtract 1 from what CPython printed to land on the comment's number.
 *
 * PITFALL: the capsule must be cast to BindingEnv*, NOT Dust2Env*. py_reset /
 * py_step / py_get_masks cast to Dust2Env* because env is BindingEnv's first
 * field, so both casts "work" — but only BindingEnv* can reach ->sd, which
 * lives after the embedded Dust2Env. A Dust2Env* cast here would compile and
 * then read whatever follows the env. The capsule is created unnamed
 * (PyCapsule_New(benv, NULL, ...) in py_init), hence the NULL name below.
 *
 * PITFALL: floats are widened to double for PyFloat_FromDouble; comparisons on
 * the Python side must use pytest.approx, since e.g. 0.123f != 0.123.
 *
 * PITFALL: add fields ONLY through the SD_INT / SD_FLOAT macros. They stringify
 * the field name, so the key and the value it carries cannot disagree. Do not
 * hand-roll a Py_BuildValue("{s:d,s:d,...}") with 52 pairs: that is the same
 * footgun as py_init's 72-arg FMT (which cs2_env.py extends only under
 * protest, and only at the tail), where one misplaced format char silently
 * mislabels every field after it — and a key/value swap is invisible to the
 * completeness test above. */

/* Store `v` under `key`, stealing the reference. Returns -1 with a Python
 * exception already set if `v` is NULL (allocation failed) or the insert
 * failed, so callers only ever have to test the return value.
 * Helper for SD_INT / SD_FLOAT; nothing else should call it. */
static int sd_dict_set(PyObject* d, const char* key, PyObject* v) {
    if (!v)
        return -1;
    int rc = PyDict_SetItemString(d, key, v);
    Py_DECREF(v);
    return rc;
}

/* #f stringifies the field name, so key and value are the same token — a
 * transposition inside this function is impossible by construction.
 * Casts are explicit (int32_t -> long, float -> double) rather than relying on
 * the implicit conversion, matching the style of the rest of this file.
 *
 * PITFALL: these macros are NOT self-contained. They capture three things from
 * the enclosing scope and only compile inside a function that provides all
 * three: a `PyObject* d` (the dict being built), a `const StaticData* sd` (the
 * struct being read), and a `fail:` label that owns d's cleanup. That is why
 * they are #undef'd immediately after py_static_data_scalars below — moving a
 * SD_* line outside that function is a compile error, not a silent misread, and
 * the #undef keeps it that way. */
#define SD_INT(f)                                                                                  \
    do {                                                                                           \
        if (sd_dict_set(d, #f, PyLong_FromLong((long)sd->f)) < 0)                                  \
            goto fail;                                                                             \
    } while (0)
#define SD_FLOAT(f)                                                                                \
    do {                                                                                           \
        if (sd_dict_set(d, #f, PyFloat_FromDouble((double)sd->f)) < 0)                             \
            goto fail;                                                                             \
    } while (0)

/* Implements binding.static_data_scalars(); see the "binding.static_data_scalars"
 * doc block above sd_dict_set for WHAT/WHY/PITFALLs (the helper and the two
 * macros sit between the two, so the docs are not directly overhead). */
static PyObject* py_static_data_scalars(PyObject* self, PyObject* args) {
    (void)self;
    PyObject* cap;
    if (!PyArg_ParseTuple(args, "O", &cap))
        return NULL;
    BindingEnv* benv = (BindingEnv*)PyCapsule_GetPointer(cap, NULL);
    if (!benv) {
        PyErr_SetString(PyExc_ValueError, "invalid env capsule");
        return NULL;
    }
    const StaticData* sd = &benv->sd;
    PyObject*         d  = PyDict_New();
    if (!d)
        return NULL;
    /* Listed in cs2_types.h declaration order so the two can be diffed
     * top-to-bottom; the dict is unordered, only the key set is contractual. */
    /* Nav/raster geometry (FMT args 10–21): map-derived, not tunable. */
    SD_INT(N);
    SD_INT(grid_w);
    SD_INT(grid_h);
    SD_INT(max_area_id);
    SD_FLOAT(grid_x_min);
    SD_FLOAT(grid_y_min);
    SD_FLOAT(grid_inv_cell);
    SD_FLOAT(inv_x_range);
    SD_FLOAT(inv_y_range);
    SD_FLOAT(x_offset);
    SD_FLOAT(y_offset);
    SD_FLOAT(bombsite_dist_scale);
    /* Weapon + round timing constants, forwarded from nav.py. */
    SD_INT(laser_damage);
    SD_FLOAT(laser_range);
    SD_FLOAT(laser_range_sq);
    SD_INT(shoot_cooldown);
    SD_INT(bomb_plant_time);
    SD_INT(bomb_defuse_time);
    SD_INT(bomb_defuse_kit);
    SD_INT(bomb_timer);
    SD_INT(round_time);
    SD_FLOAT(footstep_radius_sq);
    SD_FLOAT(gunshot_radius_sq);
    SD_INT(enemy_memory_ticks);
    SD_INT(stale_memory_tick);
    SD_FLOAT(pbrs_gamma);
    /* Spawn-array lengths. The arrays themselves are not scalars, so only
     * their counts appear here; a wrong count is a buffer overrun in C. */
    SD_INT(n_t_spawns);
    SD_INT(n_ct_spawns);
    SD_FLOAT(max_turn_speed);
    /* Reward weights and PBRS coefficients (FMT args 46–68). This run of
     * same-width floats is exactly where a transposed FMT string hides. */
    SD_FLOAT(reward_win);
    SD_FLOAT(reward_win_t_detonation);
    SD_FLOAT(reward_win_t_elimination);
    SD_FLOAT(reward_win_ct_defuse);
    SD_FLOAT(reward_win_ct_timeout);
    SD_FLOAT(reward_win_ct_elimination);
    SD_FLOAT(reward_kill);
    SD_FLOAT(reward_death);
    SD_FLOAT(reward_bombsite_entry);
    SD_FLOAT(reward_plant_bonus);
    SD_FLOAT(reward_plant_base);
    SD_FLOAT(reward_plant_progress_scale);
    SD_FLOAT(reward_plant_interrupted);
    SD_FLOAT(reward_defuse);
    SD_FLOAT(reward_shot_penalty);
    SD_FLOAT(reward_ct_survival);
    SD_FLOAT(reward_inaction);
    SD_FLOAT(pbrs_alive_weight);
    SD_FLOAT(pbrs_hp_weight);
    SD_FLOAT(pbrs_site_weight);
    SD_FLOAT(pbrs_bomb_progress_weight);
    SD_FLOAT(pbrs_nav_weight_t);
    SD_FLOAT(pbrs_nav_weight_ct);
    /* Rung 0 sim knobs (FMT args 69-71). */
    SD_INT(n_active_per_team);
    SD_INT(pin_pitch);
    SD_INT(crouch_enabled);
    return d;
fail:
    Py_DECREF(d);
    return NULL;
}
#undef SD_INT
#undef SD_FLOAT

static PyMethodDef binding_methods[] = {
    {"init", py_init, METH_VARARGS, "Init env, return capsule"},
    {"reset", py_reset, METH_VARARGS, "Reset env"},
    {"step", py_step, METH_VARARGS, "Step env"},
    {"close", py_close, METH_VARARGS, "Close env"},
    {"get_buffers", py_get_buffers, METH_VARARGS, "Get buffer addresses as ints"},
    {"get_masks", py_get_masks, METH_VARARGS, "Get masks buffer address as int"},
    {"struct_sizes", py_struct_sizes, METH_NOARGS, "sizeof/offsetof of the C structs"},
    {"static_data_scalars",
     py_static_data_scalars,
     METH_VARARGS,
     "Read scalar StaticData fields from a live env capsule"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef binding_module = {
    PyModuleDef_HEAD_INIT, "binding", NULL, -1, binding_methods};

PyMODINIT_FUNC PyInit_binding(void) {
    import_array1(NULL);
    return PyModule_Create(&binding_module);
}
