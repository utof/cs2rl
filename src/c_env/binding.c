/* src/c_env/binding.c — Python C API bridge for the Dust2 C environment.
 *
 * Exposes: binding.init(...) -> capsule
 *          binding.reset(capsule)
 *          binding.step(capsule, actions_array)
 *          binding.close(capsule)
 *          binding.get_buffers(capsule) -> (obs_ptr, rew_ptr, term_ptr, trunc_ptr)
 */
#define PY_ARRAY_UNIQUE_SYMBOL cs2rl_binding_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <string.h>
#include "dust2_env.h"

/* Single heap allocation holding both Dust2Env and its StaticData.
 * env is first field so &benv == &benv->env — Python's env_ptr extracts this address. */
typedef struct {
    Dust2Env   env;
    StaticData sd;
    int        closed;
} BindingEnv;

static void capsule_destructor(PyObject *cap) {
    BindingEnv *benv = (BindingEnv *)PyCapsule_GetPointer(cap, NULL);
    if (benv) {
        if (!benv->closed) env_close(&benv->env);
        free(benv);
    }
}

/* ── binding.init(...) -> PyCapsule ── */
static PyObject *py_init(PyObject *self, PyObject *args) {
    PyObject *vis_matrix_o, *raster_grid_o, *adjacency_o, *centroid_xy_o;
    PyObject *area_ids_o, *bombsite_mask_o, *bombsite_by_idx_o, *bombsite_dist_o;
    int       N, grid_w, grid_h, max_area_id;
    float     grid_x_min, grid_y_min, grid_inv_cell;
    float     inv_x_range, inv_y_range, x_offset, y_offset, bombsite_dist_scale;
    int       laser_damage;
    float     laser_range, laser_range_sq;
    int       shoot_cooldown, bomb_plant_time, bomb_defuse_time;
    int       bomb_defuse_kit, bomb_timer, round_time;
    float     footstep_radius_sq, gunshot_radius_sq;
    int       enemy_memory_ticks, stale_memory_tick;
    float     pbrs_gamma;
    PyObject *delta_x_o, *delta_y_o, *dir_facing_o, *t_spawns_o;
    int       n_t_spawns;
    PyObject *ct_spawns_o;
    int       n_ct_spawns;
    float     max_turn_speed;
    unsigned int seed;
    float     team_spirit;

    /* 44-arg format string — positions match StaticDataC._fields_ order from wrapper.py */
    static const char FMT[] =
        "OOOOOOOO"   /* 0-7:  vis_matrix, raster_grid, adjacency, centroid_xy,
                               area_ids, bombsite_mask, bombsite_by_idx, bombsite_dist */
        "iiii"       /* 8-11: N, grid_w, grid_h, max_area_id */
        "ffffffff"   /* 12-19: grid_x_min, grid_y_min, grid_inv_cell,
                                inv_x_range, inv_y_range, x_offset, y_offset, bombsite_dist_scale */
        "i"          /* 20: laser_damage */
        "ff"         /* 21-22: laser_range, laser_range_sq */
        "iiiiii"     /* 23-28: shoot_cooldown, bomb_plant_time, bomb_defuse_time,
                                bomb_defuse_kit, bomb_timer, round_time */
        "ff"         /* 29-30: footstep_radius_sq, gunshot_radius_sq */
        "ii"         /* 31-32: enemy_memory_ticks, stale_memory_tick */
        "f"          /* 33: pbrs_gamma */
        "OOOO"       /* 34-37: delta_x, delta_y, dir_facing, t_spawns (all arrays) */
        "i"          /* 38: n_t_spawns */
        "O"          /* 39: ct_spawns (array) */
        "i"          /* 40: n_ct_spawns */
        "f"          /* 41: max_turn_speed */
        "I"          /* 42: seed (unsigned int) */
        "f";         /* 43: team_spirit */

    if (!PyArg_ParseTuple(args, FMT,
            &vis_matrix_o, &raster_grid_o, &adjacency_o, &centroid_xy_o,
            &area_ids_o, &bombsite_mask_o, &bombsite_by_idx_o, &bombsite_dist_o,
            &N, &grid_w, &grid_h, &max_area_id,
            &grid_x_min, &grid_y_min, &grid_inv_cell,
            &inv_x_range, &inv_y_range, &x_offset, &y_offset, &bombsite_dist_scale,
            &laser_damage, &laser_range, &laser_range_sq,
            &shoot_cooldown, &bomb_plant_time, &bomb_defuse_time,
            &bomb_defuse_kit, &bomb_timer, &round_time,
            &footstep_radius_sq, &gunshot_radius_sq,
            &enemy_memory_ticks, &stale_memory_tick,
            &pbrs_gamma,
            &delta_x_o, &delta_y_o, &dir_facing_o, &t_spawns_o,
            &n_t_spawns, &ct_spawns_o, &n_ct_spawns,
            &max_turn_speed, &seed, &team_spirit))
        return NULL;

    BindingEnv *benv = (BindingEnv *)calloc(1, sizeof(BindingEnv));
    if (!benv) return PyErr_NoMemory();

    StaticData *sd  = &benv->sd;
    sd->N           = N;
    sd->vis_matrix  = (int8_t  *)PyArray_DATA((PyArrayObject *)vis_matrix_o);
    sd->raster_grid = (int32_t *)PyArray_DATA((PyArrayObject *)raster_grid_o);
    sd->adjacency   = (int8_t  *)PyArray_DATA((PyArrayObject *)adjacency_o);
    sd->centroid_xy = (float   *)PyArray_DATA((PyArrayObject *)centroid_xy_o);
    sd->area_ids    = (int32_t *)PyArray_DATA((PyArrayObject *)area_ids_o);
    sd->bombsite_mask   = (int8_t *)PyArray_DATA((PyArrayObject *)bombsite_mask_o);
    sd->bombsite_by_idx = (int8_t *)PyArray_DATA((PyArrayObject *)bombsite_by_idx_o);
    sd->bombsite_dist   = (float  *)PyArray_DATA((PyArrayObject *)bombsite_dist_o);

    sd->grid_w           = grid_w;
    sd->grid_h           = grid_h;
    sd->max_area_id      = max_area_id;
    sd->grid_x_min       = grid_x_min;
    sd->grid_y_min       = grid_y_min;
    sd->grid_inv_cell    = grid_inv_cell;
    sd->inv_x_range      = inv_x_range;
    sd->inv_y_range      = inv_y_range;
    sd->x_offset         = x_offset;
    sd->y_offset         = y_offset;
    sd->bombsite_dist_scale = bombsite_dist_scale;

    sd->laser_damage        = (int32_t)laser_damage;
    sd->laser_range         = laser_range;
    sd->laser_range_sq      = laser_range_sq;
    sd->shoot_cooldown      = (int32_t)shoot_cooldown;
    sd->bomb_plant_time     = (int32_t)bomb_plant_time;
    sd->bomb_defuse_time    = (int32_t)bomb_defuse_time;
    sd->bomb_defuse_kit     = (int32_t)bomb_defuse_kit;
    sd->bomb_timer          = (int32_t)bomb_timer;
    sd->round_time          = (int32_t)round_time;
    sd->footstep_radius_sq  = footstep_radius_sq;
    sd->gunshot_radius_sq   = gunshot_radius_sq;
    sd->enemy_memory_ticks  = (int32_t)enemy_memory_ticks;
    sd->stale_memory_tick   = (int32_t)stale_memory_tick;
    sd->pbrs_gamma          = pbrs_gamma;

    memcpy(sd->delta_x,    PyArray_DATA((PyArrayObject *)delta_x_o),    9 * sizeof(float));
    memcpy(sd->delta_y,    PyArray_DATA((PyArrayObject *)delta_y_o),    9 * sizeof(float));
    memcpy(sd->dir_facing, PyArray_DATA((PyArrayObject *)dir_facing_o), 9 * sizeof(float));

    sd->n_t_spawns  = n_t_spawns;
    memcpy(sd->t_spawns,  PyArray_DATA((PyArrayObject *)t_spawns_o),  (size_t)n_t_spawns  * sizeof(int32_t));
    sd->n_ct_spawns = n_ct_spawns;
    memcpy(sd->ct_spawns, PyArray_DATA((PyArrayObject *)ct_spawns_o), (size_t)n_ct_spawns * sizeof(int32_t));

    sd->max_turn_speed = max_turn_speed;

    env_init(&benv->env, sd, (uint32_t)seed, team_spirit);

    PyObject *cap = PyCapsule_New(benv, NULL, capsule_destructor);
    if (!cap) { free(benv); return NULL; }
    return cap;
}

/* ── binding.reset(capsule) -> None ── */
static PyObject *py_reset(PyObject *self, PyObject *args) {
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    Dust2Env *env = (Dust2Env *)PyCapsule_GetPointer(cap, NULL);
    if (!env) { PyErr_SetString(PyExc_ValueError, "invalid capsule"); return NULL; }
    env_reset(env);
    Py_RETURN_NONE;
}

/* ── binding.step(capsule, actions_array) -> None ── */
static PyObject *py_step(PyObject *self, PyObject *args) {
    PyObject *cap, *actions_o;
    if (!PyArg_ParseTuple(args, "OO", &cap, &actions_o)) return NULL;
    Dust2Env *env = (Dust2Env *)PyCapsule_GetPointer(cap, NULL);
    if (!env) { PyErr_SetString(PyExc_ValueError, "invalid capsule"); return NULL; }
    env_step(env, (const int32_t *)PyArray_DATA((PyArrayObject *)actions_o));
    Py_RETURN_NONE;
}

/* ── binding.close(capsule) -> None ── */
static PyObject *py_close(PyObject *self, PyObject *args) {
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    BindingEnv *benv = (BindingEnv *)PyCapsule_GetPointer(cap, NULL);
    if (benv && !benv->closed) { env_close(&benv->env); benv->closed = 1; }
    Py_RETURN_NONE;
}

/* ── binding.get_buffers(capsule) -> (obs_ptr, rew_ptr, term_ptr, trunc_ptr) ── */
static PyObject *py_get_buffers(PyObject *self, PyObject *args) {
    PyObject *cap;
    if (!PyArg_ParseTuple(args, "O", &cap)) return NULL;
    Dust2Env *env = (Dust2Env *)PyCapsule_GetPointer(cap, NULL);
    if (!env) { PyErr_SetString(PyExc_ValueError, "invalid capsule"); return NULL; }
    return Py_BuildValue("(KKKK)",
        (unsigned long long)(uintptr_t)env->observations,
        (unsigned long long)(uintptr_t)env->rewards,
        (unsigned long long)(uintptr_t)env->terminals,
        (unsigned long long)(uintptr_t)env->truncations);
}

static PyMethodDef binding_methods[] = {
    {"init",        py_init,        METH_VARARGS, "Init env, return capsule"},
    {"reset",       py_reset,       METH_VARARGS, "Reset env"},
    {"step",        py_step,        METH_VARARGS, "Step env"},
    {"close",       py_close,       METH_VARARGS, "Close env"},
    {"get_buffers", py_get_buffers, METH_VARARGS, "Get buffer addresses as ints"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef binding_module = {
    PyModuleDef_HEAD_INIT, "binding", NULL, -1, binding_methods};

PyMODINIT_FUNC PyInit_binding(void) {
    import_array1(NULL);
    return PyModule_Create(&binding_module);
}
