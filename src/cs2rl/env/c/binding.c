/* src/cs2rl/env/c/binding.c — Python C API bridge for the Dust2 C environment.
 *
 * Exposes: binding.init(...) -> capsule
 *          binding.reset(capsule)
 *          binding.step(capsule, actions_array, continuous_actions_array)
 *          binding.close(capsule)
 *          binding.get_buffers(capsule) -> (obs_ptr, rew_ptr, term_ptr, trunc_ptr)
 */
#define PY_ARRAY_UNIQUE_SYMBOL cs2rl_binding_ARRAY_API
#define NPY_NO_DEPRECATED_API  NPY_1_7_API_VERSION
/* REQUIRED by py_init's "y#" argument, and it must precede <Python.h>. Since
 * 3.10 a "#" format without this macro is not "int instead of Py_ssize_t" —
 * PyArg_ParseTuple raises SystemError outright, so binding.init would fail on
 * every call. It is a compile-time-invisible runtime break; do not drop it. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stddef.h> /* offsetof — used by py_struct_sizes below */
#include <stdio.h>  /* snprintf — used by py_static_data_layout below */
#include <stdlib.h>
#include <string.h>
#include "cs2_env.h"
#include "cs2_sha256.h" /* layout hash; included by binding.c only */
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

/* Defined further down, next to the SD_PREFIX_FIELDS table it walks and the
 * type vocabulary it needs. Declared here because py_init consumes the digest
 * as a precondition, and moving 200 lines up would separate that machinery
 * from the doc block that explains it. */
static int sd_layout_digest(char out_hex[65], PyObject* fields);
/* Likewise defined next to the SD_PREFIX_FIELDS walk: py_init hands it the
 * positional arguments that follow the four scalars and it stores them. */
static int sd_bind_pointers(StaticData* sd, PyObject* args, Py_ssize_t first);

/* ── binding.init(...) -> PyCapsule ──
 *
 * init(buffer, layout_hash, seed, team_spirit,
 *      vis_matrix, raster_grid, adjacency, centroid_xy, centroids_z,
 *      area_ids, bombsite_mask, bombsite_by_idx, is_ramp, bombsite_dist)
 *
 * WHAT: `buffer` is StaticData's prefix — [0, offsetof(StaticData, wall_list)) —
 * packed field-by-field by _pack_static_data() in cs2_env.py out of the
 * StaticDataC ctypes mirror. `layout_hash` is that mirror's layout digest.
 * `seed` and `team_spirit` are NOT StaticData fields (env_init takes them
 * directly), so they keep argument slots.
 *
 * The TEN pointer fields also keep argument slots and are never packed (they
 * are the arguments after team_spirit, in StaticData's pointer-field order): C must
 * end up holding the numpy buffers' own addresses — kept alive for the env's
 * lifetime by Cs2Env._refs — and an address packed into a transient bytes
 * object would dangle the moment that object was collected.
 *
 * WHY a buffer instead of the 73-arg PyArg_ParseTuple format string this
 * replaces: that string tied Python's arguments to C's fields BY POSITION and
 * said nothing about names or types, so inserting a field mid-struct on one
 * side silently shifted every field after it. Packing by name moves the
 * agreement onto a comparison of DECLARATIONS — the layout hash, which the C
 * compiler and ctypes derive independently of each other.
 *
 * THREE PRECONDITIONS, in this order, ALL BEFORE THE MEMCPY. The order is
 * normative, not incidental:
 *
 *   1. `layout_hash` must equal THIS BUILD's own digest. That comparison is the
 *      only RUNTIME check that the .so and the installed cs2_env.py describe
 *      the same struct. It matters because the extension is a gitignored local
 *      artifact, so "headers edited, .so not rebuilt" is the realistic failure,
 *      and tests/env/c/test_static_data_layout.py only compares the two at TEST time.
 *      It runs FIRST because a buffer packed against a different declaration
 *      has an unknown prefix size, which would make check 2 compare the length
 *      against the wrong number.
 *   2. buffer LENGTH >= prefix size. The hash compares declarations, not the
 *      buffer that arrived; a short buffer is an out-of-bounds read the hash
 *      cannot see.
 *   3. only then the memcpy, and only then the ten pointer assignments.
 *
 * MEMCPY SCOPE is [0, offsetof(StaticData, wall_list)), never sizeof(StaticData):
 * wall_list is C-owned (build_solids_from_rooms allocates it, env_close frees
 * it) and area_bounds, which follows it, is published by Python AFTER init
 * through the ctypes overlay. An unscoped copy would clobber both.
 *
 * ORDER of memcpy vs pointer assignment is normative for the mirror-image
 * reason: the copied range CONTAINS the ten pointer slots, because they live in
 * the prefix. Assigning first and copying second would overwrite every pointer
 * with the buffer's zeros and hand env_init ten NULLs.
 *
 * NOT CHECKED HERE, deliberately: that the ten arguments are numpy arrays of
 * the right dtype and length. PyArray_DATA is taken on trust exactly as it was
 * before this rewrite — Cs2Env.__init__ builds all ten itself — and adding
 * validation here is a separate change with its own behavioural surface. */
static PyObject* py_init(PyObject* self, PyObject* args) {
    (void)self;
    const char* buffer;
    Py_ssize_t  buffer_len;
    const char* layout_hash;
    /* The ten pointer arguments are NOT named here. Which argument fills which
     * StaticData field is decided by sd_bind_pointers, which walks the same
     * SD_PREFIX_FIELDS table the layout hash is built from: the Nth pointer row
     * receives the Nth argument after team_spirit. cs2_env.py passes them in
     * StaticDataC's pointer order (_SD_POINTER_FIELDS), and the hash already
     * proves StaticDataC and SD_PREFIX_FIELDS list the same rows in the same
     * order, so no second hand-written list exists to fall out of step. */
    PyObject*    head;
    unsigned int seed;
    float        team_spirit;

    /* "y#sIf" parses the leading four only; PyArg_ParseTuple would reject the
     * pointer arguments as surplus, so parse a slice and leave them to
     * sd_bind_pointers (which checks the count). */
    if (PyTuple_GET_SIZE(args) < 4) {
        PyErr_SetString(PyExc_TypeError, "init() takes at least 4 arguments");
        return NULL;
    }
    head = PyTuple_GetSlice(args, 0, 4);
    if (!head)
        return NULL;
    if (!PyArg_ParseTuple(head, "y#sIf", &buffer, &buffer_len, &layout_hash, &seed, &team_spirit)) {
        Py_DECREF(head);
        return NULL;
    }
    /* buffer and layout_hash borrow from args' items, which outlive `head`. */
    Py_DECREF(head);

    /* Precondition 1 — the incoming hash must be CONSUMED, not just accepted.
     * A parameter nothing compares is the same defect as a layout table nothing
     * knocks out: every test still passes while the guard does nothing. */
    char c_layout_hash[65];
    if (sd_layout_digest(c_layout_hash, NULL) != 0)
        return NULL;
    /* ASCII ONLY in this format string. PyErr_Format goes through
     * PyUnicode_FromFormatV, which raises ValueError on a non-ASCII byte -- so
     * an em dash here would replace the diagnosis with a confusing codec error
     * at exactly the moment someone needs to read it. */
    if (strcmp(layout_hash, c_layout_hash) != 0) {
        PyErr_Format(PyExc_RuntimeError,
                     "StaticData layout hash mismatch: this build of binding says %s, the caller "
                     "packed its buffer against %s. The built extension and cs2_env.py describe "
                     "different structs; rebuild with "
                     "`uv run --with \"ziglang>=0.14.0,<0.15\" python setup.py build_ext "
                     "--inplace`, and if that does not fix it, run "
                     "tests/env/c/test_static_data_layout.py to see which field disagrees.",
                     c_layout_hash,
                     layout_hash);
        return NULL;
    }

    /* Precondition 2 — LENGTH. >=, not ==: the hash has already established
     * that both sides agree on prefix_size, so a longer buffer is a caller that
     * sent the whole struct rather than a caller that is confused, and the copy
     * below reads only the prefix either way. Shorter is an OOB read. */
    const size_t prefix_size = offsetof(StaticData, wall_list);
    if (buffer_len < (Py_ssize_t)prefix_size) {
        PyErr_Format(PyExc_ValueError,
                     "StaticData buffer is %zd bytes, need at least %zu "
                     "(offsetof(StaticData, wall_list))",
                     buffer_len,
                     prefix_size);
        return NULL;
    }

    BindingEnv* benv = (BindingEnv*)calloc(1, sizeof(BindingEnv));
    if (!benv)
        return PyErr_NoMemory();

    StaticData* sd = &benv->sd;
    /* Precondition 3 / step one: the scoped copy, FIRST. Everything Python
     * owns — all 56 scalars, delta_x/delta_y/dir_facing and the two spawn
     * arrays — arrives in this single memcpy. The spawn arrays' unused tail
     * slots read as zero because cs2_env.py packs from a fresh (zeroed)
     * StaticDataC, which reproduces exactly what the old partial memcpy left in
     * this calloc'd struct. */
    memcpy(sd, buffer, prefix_size);

    /* Step two: the ten borrowed pointers, AFTER the copy that would have
     * NULLed them. Python keeps every one of these arrays alive in Cs2Env._refs
     * for the env's lifetime; C never frees them. */
    if (sd_bind_pointers(sd, args, 4) != 0) {
        free(benv);
        return NULL;
    }

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
     * Defense-in-depth: cs2_env.py::_prepare_continuous_actions already raises
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
    /* #164: the bomb lifecycle is one state (GameState.bomb) that only the
     * cs2_bomb.h transitions write. A hand-assembled state written through the
     * Python overlay between steps (tests, scripted setups) is checked here,
     * the one door those writes pass through before the sim reads them, so an
     * impossible bomb state raises instead of being stepped. */
    const char* bomb_err = bomb_state_error(&env->game, env->sd);
    if (bomb_err) {
        PyErr_Format(PyExc_ValueError, "invalid GameState.bomb before step: %s", bomb_err);
        return NULL;
    }
    env_step(env,
             (const int32_t*)PyArray_DATA((PyArrayObject*)actions_o),
             (const float*)PyArray_DATA(cont_arr));
    Py_RETURN_NONE;
}

/* ── binding.give_bomb(capsule, agent_index) -> None ──
 * The one sanctioned way to choose who carries the bomb from outside the sim
 * (#164): the same bomb_give transition env_reset and pickup use, after
 * bomb_give_error's checks (a live, participating T and an unplanted bomb;
 * ValueError otherwise). It changes possession only: round_designated_carrier_id
 * (the role bit) and the agent's weapon are the caller's, and masks stay as the
 * last step computed them until the next step. */
static PyObject* py_give_bomb(PyObject* self, PyObject* args) {
    (void)self;
    PyObject* cap;
    int       agent;
    if (!PyArg_ParseTuple(args, "Oi", &cap, &agent))
        return NULL;
    Dust2Env* env = (Dust2Env*)PyCapsule_GetPointer(cap, NULL);
    if (!env) {
        PyErr_SetString(PyExc_ValueError, "invalid capsule");
        return NULL;
    }
    const char* err = bomb_give_error(&env->game, agent);
    if (err) {
        PyErr_Format(PyExc_ValueError, "give_bomb(%d): %s", agent, err);
        return NULL;
    }
    bomb_give(&env->game, agent);
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
 * on each of those structs' LAST field, plus the two team-size macros env/nav.py
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
 * tests/env/c/test_struct_sizes.py) asserts set(struct_sizes()) == _C_SIZE_KEYS_CHECKED.
 * A key added here and never compared there fails that test instead of sitting
 * unguarded. Sizes use the "n" (Py_ssize_t) format because sizeof yields size_t;
 * the macros use "i" (plain int). */
/* Py_UNUSED(ignored), not `args`: this is METH_NOARGS, so CPython passes NULL
 * as the second argument rather than an empty tuple. Naming it `args` invites a
 * later `PyArg_ParseTuple(args, ...)` to be added here, which would deref NULL.
 * Py_UNUSED mangles the name so it cannot be referenced at all. */
static PyObject* py_struct_sizes(PyObject* self, PyObject* Py_UNUSED(ignored)) {
    (void)self;
    return Py_BuildValue("{s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n,"
                         "s:n,s:i,s:i,s:i,s:i,"
                         "s:i,s:i,s:i,s:i,s:i}",
                         "AgentState",
                         (Py_ssize_t)sizeof(AgentState),
                         "BombState",
                         (Py_ssize_t)sizeof(BombState),
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
                          * anchor the LAST field. Picking bombsite_entered instead would miss
                          * a field slipped in between it and the pad on one side only. */
                         "GameState__pad_gs_offset",
                         (Py_ssize_t)offsetof(GameState, _pad_gs),
                         "GameState_bomb_offset",
                         (Py_ssize_t)offsetof(GameState, bomb),
                         "BombState_z_offset",
                         (Py_ssize_t)offsetof(BombState, z),
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
                         N_AGENTS,
                         /* BombPhase values (cs2_types.h), mirrored by cs2_env.BombPhase. */
                         "BOMB_CARRIED",
                         BOMB_CARRIED,
                         "BOMB_PLANTING",
                         BOMB_PLANTING,
                         "BOMB_DROPPED",
                         BOMB_DROPPED,
                         "BOMB_PLANTED",
                         BOMB_PLANTED,
                         "BOMB_DEFUSING",
                         BOMB_DEFUSING,
                         "BOMB_DEFUSED",
                         BOMB_DEFUSED,
                         "BOMB_DETONATED",
                         BOMB_DETONATED);
}

/* ── binding.static_data_layout() -> dict ──
 * The C compiler's own account of StaticData's prefix layout, plus a hash of it.
 *
 * WHAT: {"format", "preamble", "prefix_size", "fields", "hash"}, where "fields"
 * is a tuple of (name, offset, size, canonical_type) — one entry per row of
 * SD_PREFIX_FIELDS (cs2_types.h), in declaration order — and "hash" is the
 * sha256 of the serialisation described under SD_LAYOUT_FORMAT below.
 *
 * WHY a NEW entry point and not three more struct_sizes() keys: struct_sizes()'s
 * key set is asserted set-equal to cs2_env._C_SIZE_KEYS_CHECKED by
 * test_struct_sizes_keys_are_all_consumed, i.e. every key it publishes must be
 * consumed by a hand-written tuple in cs2_env.py. That guard is exactly right
 * for a handful of sizeofs and anchors and exactly wrong for 71 per-field rows.
 *
 * WHY it exists at all: struct_sizes() measures 7 sizeofs and 3 StaticData
 * offsets. That is blind to a field-for-field disagreement between the C struct
 * and the ctypes mirror that happens to preserve both — two same-width fields
 * swapped, or an int declared where the mirror says float. This function makes
 * the C side state its layout per field so the Python side can compare against
 * its own introspection of StaticDataC rather than against nothing.
 *
 * WHAT IT DOES NOT COVER, stated plainly: this compares two DECLARATIONS. It
 * cannot see a value-routing mistake — Python assigning jump_enabled into the
 * crouch_enabled field packs the wrong number into a correctly-described slot,
 * and every quadruple here still matches. That failure mode belongs to
 * static_data_scalars() and the two-env sentinel scheme in
 * tests/env/c/test_struct_sizes.py, which is why neither is retired.
 * It also covers the PREFIX ONLY (up to wall_list): the tail is held by the
 * three offset anchors in py_struct_sizes above.
 *
 * PITFALL: the canonical type names are NOT the C spellings. `int` and
 * `int32_t` both become "c_int" because ctypes folds c_int32 into c_int and the
 * two sides have to be able to agree; pointers become "ptr_<elem>" and arrays
 * "arr_<elem>_<count>". Never widen that mapping to make a failing comparison
 * pass — the type column is the only part of the hash that sees an int/float
 * swap between two 4-byte fields.
 */

/* Serialisation hashed by BOTH sides. Bump the version tag if the line format
 * changes, so a stale .so fails on the tag rather than on an opaque hex diff.
 * Lines, each '\n'-terminated:
 *     <format tag>
 *     <preamble>
 *     prefix_size=<offsetof(StaticData, wall_list)>
 *     <name>|<offset>|<size>|<canonical type>      (once per prefix field)
 * The Python counterpart is static_data_layout() in cs2_env.py. */
#define SD_LAYOUT_FORMAT "cs2rl-static-data-layout-v1"

/* C's FIXED expectation about how ctypes must be laying StaticDataC out.
 * Python does NOT hardcode this string: it DERIVES its own from live
 * introspection (hasattr(StaticDataC, "_pack_"), getattr(..., "_layout_")). Two
 * hardcoded constants would compare nothing. `pack=unset` matters because
 * setting _pack_ on Linux silently switches ctypes to MSVC layout rules, which
 * would move fields without changing any single field's declared type. */
#define SD_LAYOUT_PREAMBLE "struct=StaticData;pack=unset;layout=gcc-sysv"

/* Declared C type spelling -> the name ctypes introspection produces for the
 * same field. Written ONCE, here: the Python side derives its names structurally
 * (ctype.__name__, "ptr_"/"arr_" prefixes) and asserts they land in this same
 * vocabulary. A type used in SD_PREFIX_FIELDS but missing here makes
 * static_data_layout() raise, which is the intended direction — a new field type
 * must be a conscious decision on both sides, not a silently unhashed column. */
typedef struct {
    const char* c_type;
    const char* canonical;
} SdTypeName;

static const SdTypeName SD_TYPE_NAMES[] = {
    {"int", "c_int"},
    {"int32_t", "c_int"}, /* ctypes: c_int32 IS c_int, so these are one name */
    {"int8_t", "c_byte"}, /* ctypes: c_int8 IS c_byte */
    {"float", "c_float"},
    {"int8_t*", "ptr_c_byte"},
    {"int32_t*", "ptr_c_int"},
    {"float*", "ptr_c_float"},
};

/* Compare two C type spellings ignoring spaces, so that a reformat of
 * SD_PREFIX_FIELDS from `int8_t*` to `int8_t *` (clang-format's
 * PointerAlignment could do it) does not turn every pointer field into an
 * unknown type. */
static int sd_type_spelling_eq(const char* a, const char* b) {
    for (;;) {
        while (*a == ' ')
            a++;
        while (*b == ' ')
            b++;
        if (*a != *b)
            return 0;
        if (*a == '\0')
            return 1;
        a++;
        b++;
    }
}

/* Write the canonical type name for one field into `out` (0 on success, -1 if
 * `c_type` is not in the vocabulary above or the array arithmetic is nonsense).
 * `field_size`/`elem_size` are sizeof expressions from the caller, so the array
 * COUNT in the name is compiler-derived rather than copied out of the struct. */
static int sd_canonical_type(const char* c_type,
                             int         is_array,
                             size_t      field_size,
                             size_t      elem_size,
                             char*       out,
                             size_t      out_cap) {
    const char* base = NULL;
    size_t      i;
    int         n;
    for (i = 0; i < sizeof(SD_TYPE_NAMES) / sizeof(SD_TYPE_NAMES[0]); i++) {
        if (sd_type_spelling_eq(SD_TYPE_NAMES[i].c_type, c_type)) {
            base = SD_TYPE_NAMES[i].canonical;
            break;
        }
    }
    if (!base)
        return -1;
    if (!is_array)
        n = snprintf(out, out_cap, "%s", base);
    else if (elem_size == 0 || field_size % elem_size != 0)
        return -1;
    else
        /* %lu, not %zu: the Windows CRT historically ignores the z length
         * modifier. Every count here is < 100, so the narrowing is safe. */
        n = snprintf(out, out_cap, "arr_%s_%lu", base, (unsigned long)(field_size / elem_size));
    return (n < 0 || (size_t)n >= out_cap) ? -1 : 0;
}

/* Walk SD_PREFIX_FIELDS once: hash the canonical serialisation into `out_hex`
 * (64 lowercase hex chars + NUL) and, when `fields` is non-NULL, append one
 * (name, offset, size, canonical type) tuple per row to that list.
 *
 * ONE walk with TWO callers, and that is the point. py_static_data_layout needs
 * the rows, because the test that reports WHICH field drifted compares them
 * against ctypes; py_init needs only the digest, to reject a buffer packed
 * against a different declaration. A second copy of this loop for the second
 * caller could drift from the first, and then py_init would reject exactly the
 * buffers cs2_env.py packs correctly — a guard that fires on healthy trees is
 * worse than no guard, because it gets deleted.
 *
 * Returns 0, or -1 with a Python exception already set. On failure `fields` is
 * left to the caller: it owns the list either way. */
static int sd_layout_digest(char out_hex[65], PyObject* fields) {
    Cs2Sha256 h;
    char      line[256];
    int       n;

    cs2_sha256_init(&h);
    n = snprintf(line,
                 sizeof(line),
                 "%s\n%s\nprefix_size=%lu\n",
                 SD_LAYOUT_FORMAT,
                 SD_LAYOUT_PREAMBLE,
                 (unsigned long)offsetof(StaticData, wall_list));
    if (n < 0 || (size_t)n >= sizeof(line)) {
        PyErr_SetString(PyExc_RuntimeError, "static_data_layout: header line overflow");
        return -1;
    }
    cs2_sha256_update(&h, line, (size_t)n);

/* One SD_PREFIX_FIELDS row -> one hashed line and, when the caller asked for
 * them, one tuple entry. offsetof and sizeof are evaluated here, so the numbers
 * are the compiler's, never the table's. #f stringifies the field name, so the
 * name in the hash and the field the offset was taken from cannot disagree.
 *
 * The `if (fields)` arm is what lets py_init share this walk without paying for
 * a list it would immediately throw away. The HASH is built unconditionally —
 * the two callers must never be able to hash different things. */
#define SD_LAYOUT_ROW(ctype, f, is_array)                                                          \
    do {                                                                                           \
        char      canon[64];                                                                       \
        size_t    off = offsetof(StaticData, f);                                                   \
        size_t    fsz = sizeof(((StaticData*)0)->f);                                               \
        PyObject* row;                                                                             \
        if (sd_canonical_type(#ctype, (is_array), fsz, sizeof(ctype), canon, sizeof(canon)) !=     \
            0) {                                                                                   \
            PyErr_Format(PyExc_RuntimeError,                                                       \
                         "static_data_layout: field '%s' declared '%s' has no canonical type "     \
                         "name; add it to SD_TYPE_NAMES in binding.c and to "                      \
                         "_CANONICAL_SCALAR_CTYPES in cs2_env.py",                                 \
                         #f,                                                                       \
                         #ctype);                                                                  \
            return -1;                                                                             \
        }                                                                                          \
        n = snprintf(line,                                                                         \
                     sizeof(line),                                                                 \
                     "%s|%lu|%lu|%s\n",                                                            \
                     #f,                                                                           \
                     (unsigned long)off,                                                           \
                     (unsigned long)fsz,                                                           \
                     canon);                                                                       \
        if (n < 0 || (size_t)n >= sizeof(line)) {                                                  \
            PyErr_Format(                                                                          \
                PyExc_RuntimeError, "static_data_layout: line overflow for field '%s'", #f);       \
            return -1;                                                                             \
        }                                                                                          \
        cs2_sha256_update(&h, line, (size_t)n);                                                    \
        if (fields) {                                                                              \
            row = Py_BuildValue("(snns)", #f, (Py_ssize_t)off, (Py_ssize_t)fsz, canon);            \
            if (!row)                                                                              \
                return -1;                                                                         \
            if (PyList_Append(fields, row) != 0) {                                                 \
                Py_DECREF(row);                                                                    \
                return -1;                                                                         \
            }                                                                                      \
            Py_DECREF(row);                                                                        \
        }                                                                                          \
    } while (0);

    SD_PREFIX_FIELDS(SD_LAYOUT_ROW)
#undef SD_LAYOUT_ROW

    cs2_sha256_final_hex(&h, out_hex);
    return 0;
}

/* Does this C type spelling (from an SD_PREFIX_FIELDS row) end in `*`? */
static int sd_type_is_pointer(const char* c_type) {
    size_t n = strlen(c_type);
    while (n > 0 && c_type[n - 1] == ' ')
        n--;
    return n > 0 && c_type[n - 1] == '*';
}

/* Store args[first..] into StaticData's pointer fields: the Nth row of
 * SD_PREFIX_FIELDS whose type is a pointer receives the Nth argument. This is
 * the ONLY place the argument order is defined, and it is derived from the
 * table, so a reorder of the struct moves it with it. Python derives its own
 * passing order from StaticDataC (_SD_POINTER_FIELDS); the layout hash proves
 * both describe the same rows in the same order, and
 * tests/env/c/test_static_data_layout.py reads the stored addresses back.
 *
 * The arguments are taken on trust as numpy arrays (see py_init). The count is
 * checked before anything is written. The slot is filled with memcpy of a
 * void*, not a typed store: the macro body is expanded for every row, and
 * `sd->f = (ctype)p` does not compile for the scalar and array rows. Returns 0, or -1 with an
 * exception set. */
static int sd_bind_pointers(StaticData* sd, PyObject* args, Py_ssize_t first) {
    Py_ssize_t expected = 0;
    Py_ssize_t k        = 0;

#define SD_COUNT_PTR(ctype, f, is_array)                                                           \
    if (sd_type_is_pointer(#ctype))                                                                \
        expected++;
    SD_PREFIX_FIELDS(SD_COUNT_PTR)
#undef SD_COUNT_PTR

    if (PyTuple_GET_SIZE(args) != first + expected) {
        PyErr_Format(PyExc_TypeError,
                     "init() takes %zd pointer arguments (one per pointer field of StaticData), "
                     "got %zd",
                     expected,
                     PyTuple_GET_SIZE(args) - first);
        return -1;
    }

#define SD_STORE_PTR(ctype, f, is_array)                                                           \
    if (sd_type_is_pointer(#ctype)) {                                                              \
        void* p = PyArray_DATA((PyArrayObject*)PyTuple_GET_ITEM(args, first + k));                 \
        memcpy((char*)sd + offsetof(StaticData, f), &p, sizeof(p));                                \
        k++;                                                                                       \
    }
    SD_PREFIX_FIELDS(SD_STORE_PTR)
#undef SD_STORE_PTR
    return 0;
}

/* METH_NOARGS — see the Py_UNUSED note above py_struct_sizes for why the second
 * parameter is named this way. */
static PyObject* py_static_data_layout(PyObject* self, PyObject* Py_UNUSED(ignored)) {
    (void)self;
    char      hex[65];
    PyObject* tuple;
    PyObject* fields = PyList_New(0);
    if (!fields)
        return NULL;
    if (sd_layout_digest(hex, fields) != 0) {
        Py_DECREF(fields);
        return NULL;
    }
    tuple = PyList_AsTuple(fields);
    Py_DECREF(fields);
    if (!tuple)
        return NULL;
    /* s:N hands `tuple` to the dict and steals the reference, including on
     * failure — nothing left to clean up here either way. */
    return Py_BuildValue("{s:s,s:s,s:n,s:N,s:s}",
                         "format",
                         SD_LAYOUT_FORMAT,
                         "preamble",
                         SD_LAYOUT_PREAMBLE,
                         "prefix_size",
                         (Py_ssize_t)offsetof(StaticData, wall_list),
                         "fields",
                         tuple,
                         "hash",
                         hex);
}

/* ── binding.static_data_scalars(capsule) -> dict ──
 * Read every scalar StaticData field back out of a live env.
 *
 * WHAT: EVERY plain-number field of StaticData (int / int32_t / float), keyed
 * by its C field name — all 56 of them, not a curated subset. Excluded, because
 * they are not scalars: the pointer fields, the fixed arrays (delta_x, delta_y,
 * dir_facing, t_spawns, ct_spawns) and the nested wall_list / area_bounds,
 * which struct_sizes() covers with offsetof keys instead.
 *
 * That completeness is ENFORCED, not merely documented:
 * test_static_data_scalars_covers_every_scalar_field in
 * tests/env/c/test_struct_sizes.py compares this dict's key set against the
 * scalar-typed fields of the StaticDataC ctypes mirror. Appending a field to
 * cs2_types.h + the mirror and forgetting this function fails that test.
 *
 * WHY: neither struct_sizes() nor the layout hash can detect a VALUE routed
 * into the wrong field — both describe DECLARATIONS, and two floats swapped
 * between the named assignments in cs2_env.py's `static_data` mapping keep
 * every size, offset and type identical while silently feeding reward_kill into
 * reward_death. Tests push distinct sentinels through Cs2Env and read them back
 * here, so a transposition fails loudly.
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
 * hand-roll a Py_BuildValue("{s:d,s:d,...}") with 56 pairs: that is the same
 * footgun as the 73-arg PyArg_ParseTuple format string py_init carried before
 * spec 2026-08-31 §2 W2, where one misplaced format char silently mislabelled
 * every field after it — and a key/value swap is invisible to the completeness
 * test above. */

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
    /* Nav/raster geometry: map-derived, not tunable. */
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
    /* Weapon + round timing constants, forwarded from env/nav.py. */
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
    /* Reward weights and PBRS coefficients. This run of same-width floats is
     * exactly where a transposed pair of packing assignments hides. */
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
    /* Rung 0 sim knobs + Rung 1a jump_enabled. */
    SD_INT(n_active_per_team);
    SD_INT(pin_pitch);
    SD_INT(crouch_enabled);
    SD_INT(jump_enabled);
    return d;
fail:
    Py_DECREF(d);
    return NULL;
}
#undef SD_INT
#undef SD_FLOAT

/* ── binding.bake_solids(capsule) -> int ──
 * Run build_solids_from_rooms on this env's StaticData and return wall_list.count.
 *
 * WHAT: the ONE Python entry into cs2_solids.h's bake. The training path never
 *       bakes (env_step does not query solids today; only make_client in
 *       cs2_render.h does), so without this a Python test cannot ask "does this
 *       map survive the solids bake?" (spec 2026-08-29 §8, tests/env/test_arena_duel.py).
 * WHY:  build_solids_from_rooms is static inline in a header — not reachable by
 *       ctypes — and the alternative (re-deriving the face rules in Python)
 *       would test a copy, not the bake.
 * PITFALLS: re-baking frees the previous list first (build_solids_from_rooms
 *       does that itself); env_close / c_close free it, so a baked env leaks
 *       nothing extra. Reads sd->area_bounds, which cs2_env.py installs AFTER
 *       binding.init — a map without room quads (dust2) bakes 0 faces, which is
 *       indistinguishable from a failed malloc (that path prints to stderr). */
static PyObject* py_bake_solids(PyObject* self, PyObject* args) {
    (void)self;
    PyObject* cap;
    if (!PyArg_ParseTuple(args, "O", &cap))
        return NULL;
    BindingEnv* benv = (BindingEnv*)PyCapsule_GetPointer(cap, NULL);
    if (!benv) {
        PyErr_SetString(PyExc_ValueError, "invalid env capsule");
        return NULL;
    }
    build_solids_from_rooms(&benv->sd);
    return PyLong_FromLong((long)benv->sd.wall_list.count);
}

/* ── binding.solid_ray_clear(capsule, x0, y0, z0, x1, y1, z1) -> int ──
 * 1 if no baked face blocks the segment, 0 if one does (cs2_solids.h
 * solid_ray_clear, the LoS/bullet query the demo uses). z0/z1 are EYE heights.
 *
 * WHY: lets tests check a spawn→spawn lane through the REAL face list rather
 *      than a Python re-implementation of the segment/plane test.
 * PITFALL: with an empty list (never baked, dust2, failed malloc) every ray is
 *      "clear" — pair any clear-lane assert with a must-block control ray. */
static PyObject* py_solid_ray_clear(PyObject* self, PyObject* args) {
    (void)self;
    PyObject* cap;
    float     x0, y0, z0, x1, y1, z1;
    if (!PyArg_ParseTuple(args, "Offffff", &cap, &x0, &y0, &z0, &x1, &y1, &z1))
        return NULL;
    BindingEnv* benv = (BindingEnv*)PyCapsule_GetPointer(cap, NULL);
    if (!benv) {
        PyErr_SetString(PyExc_ValueError, "invalid env capsule");
        return NULL;
    }
    return PyLong_FromLong((long)solid_ray_clear(&benv->sd, x0, y0, z0, x1, y1, z1));
}

static PyMethodDef binding_methods[] = {
    {"init", py_init, METH_VARARGS, "Init env, return capsule"},
    {"reset", py_reset, METH_VARARGS, "Reset env"},
    {"step", py_step, METH_VARARGS, "Step env"},
    {"give_bomb", py_give_bomb, METH_VARARGS, "Hand the bomb to a live T agent (cs2_bomb.h)"},
    {"close", py_close, METH_VARARGS, "Close env"},
    {"get_buffers", py_get_buffers, METH_VARARGS, "Get buffer addresses as ints"},
    {"get_masks", py_get_masks, METH_VARARGS, "Get masks buffer address as int"},
    {"struct_sizes", py_struct_sizes, METH_NOARGS, "sizeof/offsetof of the C structs"},
    {"static_data_layout",
     py_static_data_layout,
     METH_NOARGS,
     "Per-field layout of StaticData's prefix, plus its sha256"},
    {"static_data_scalars",
     py_static_data_scalars,
     METH_VARARGS,
     "Read scalar StaticData fields from a live env capsule"},
    {"bake_solids",
     py_bake_solids,
     METH_VARARGS,
     "Bake sd->wall_list from the room quads; returns count"},
    {"solid_ray_clear",
     py_solid_ray_clear,
     METH_VARARGS,
     "1 if no baked face blocks the segment (x0,y0,z0)->(x1,y1,z1)"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef binding_module = {
    PyModuleDef_HEAD_INIT, "binding", NULL, -1, binding_methods};

PyMODINIT_FUNC PyInit_binding(void) {
    import_array1(NULL);
    return PyModule_Create(&binding_module);
}
