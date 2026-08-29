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

    /* 69-arg format string — positions match StaticDataC._fields_ order from cs2_env.py.
     * T2 (verticality): added centroids_z_o after centroid_xy_o (pos 4) and is_ramp_o
     * after bombsite_by_idx_o (pos 8), for 10 O args total instead of 8.
     * Total: 10O + 4i + 8f + i + 2f + 6i + 2f + 2i + f + 4O + i + O + i + f + I + f
     *      + f(reward_win) + 5f(Batch1) + 17f(Phase5-rest) = 69 args.
     * CRITICAL: positions must stay in sync with StaticDataC._fields_ in cs2_env.py
     * and StaticData in cs2_types.h — mismatch silently corrupts pointer assignments. */
    static const char FMT[] =
        "OOOOOO"             /* 0-5:  vis_matrix, raster_grid, adjacency, centroid_xy,
                                       centroids_z, area_ids */
        "OOOO"               /* 6-9:  bombsite_mask, bombsite_by_idx, is_ramp, bombsite_dist */
        "iiii"               /* 10-13: N, grid_w, grid_h, max_area_id */
        "ffffffff"           /* 14-21: grid_x_min, grid_y_min, grid_inv_cell,
                                        inv_x_range, inv_y_range, x_offset, y_offset, bombsite_dist_scale */
        "i"                  /* 22: laser_damage */
        "ff"                 /* 23-24: laser_range, laser_range_sq */
        "iiiiii"             /* 25-30: shoot_cooldown, bomb_plant_time, bomb_defuse_time,
                                        bomb_defuse_kit, bomb_timer, round_time */
        "ff"                 /* 31-32: footstep_radius_sq, gunshot_radius_sq */
        "ii"                 /* 33-34: enemy_memory_ticks, stale_memory_tick */
        "f"                  /* 35: pbrs_gamma */
        "OOOO"               /* 36-39: delta_x, delta_y, dir_facing, t_spawns (all arrays) */
        "i"                  /* 40: n_t_spawns */
        "O"                  /* 41: ct_spawns (array) */
        "i"                  /* 42: n_ct_spawns */
        "f"                  /* 43: max_turn_speed */
        "I"                  /* 44: seed (unsigned int) */
        "f"                  /* 45: team_spirit */
        "f"                  /* 46: reward_win (legacy symmetric) */
        "fffff"              /* 47-51: Batch 1 per-mechanism win magnitudes */
        "fffffffffffffffff"; /* 52-68: 17 remaining Phase-5 reward weights
                                        (reward_kill through pbrs_nav_weight_ct) */

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
                          &pbrs_nav_weight_ct))
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

static PyMethodDef binding_methods[] = {
    {"init", py_init, METH_VARARGS, "Init env, return capsule"},
    {"reset", py_reset, METH_VARARGS, "Reset env"},
    {"step", py_step, METH_VARARGS, "Step env"},
    {"close", py_close, METH_VARARGS, "Close env"},
    {"get_buffers", py_get_buffers, METH_VARARGS, "Get buffer addresses as ints"},
    {"get_masks", py_get_masks, METH_VARARGS, "Get masks buffer address as int"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef binding_module = {
    PyModuleDef_HEAD_INIT, "binding", NULL, -1, binding_methods};

PyMODINIT_FUNC PyInit_binding(void) {
    import_array1(NULL);
    return PyModule_Create(&binding_module);
}
