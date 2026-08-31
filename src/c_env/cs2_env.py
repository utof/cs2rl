# src/c_env/cs2_env.py — PufferEnv subclass backed by binding.c C API bridge.

import ctypes
import sys
from dataclasses import dataclass
from pathlib import Path

import gymnasium
import numpy as np
import pufferlib

import nav
from _action_spec import ACTION_DIM, ACTION_HEAD_SIZES, ACTION_MASK_DIM, AIM_DIM
from map import make_cs2_map
from nav import N_AGENTS, OBS_DIM, TEAM_SIZE

_DIR = Path(__file__).parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))
import binding                         # noqa: E402

# ── Viz dataclasses (used by snapshot_state) ─────────────────────────────────


@dataclass
class VizAgentState:
    agent_id: int
    team: int
    pos: np.ndarray
    facing: float
    pitch: float                       # Batch 3.5: aim pitch (radians, 0=horizontal, ±π/2 bounds).
    hp: int
    alive: bool
    has_bomb: bool
    has_kit: bool


@dataclass
class VizGameState:
    agents: list
    bomb_planted: bool
    bomb_pos: np.ndarray


# ── ctypes struct definitions (read/write overlay for _c_env) ─────────────────


class WallC(ctypes.Structure):
    # Field-for-field mirror of `Wall` in cs2_types.h, and the pointee type of
    # WallListC.walls. Nothing in the repo dereferences walls[i] from Python
    # today — the baked list is produced and consumed entirely in C — so this
    # mirror exists to already be right the first time something does. Should
    # it drift from the C struct before then, that first read would land on
    # the wrong bytes and ctypes has no way to notice: no exception, no shape
    # mismatch, just plausible-looking garbage coordinates. The bake added
    # nx/ny/kind (24 B -> 36 B) for the draw-offset rule in draw_walls.
    # Pitfall: append only, and append on BOTH sides. The C side carries a
    # matching _Static_assert(sizeof(Wall) == 36); the ctypes.sizeof(WallC)
    # assert further down is the other half of that pair, and the pair is
    # what actually catches the drift.
    _fields_ = [
        ("x0", ctypes.c_float),
        ("y0", ctypes.c_float),
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("height", ctypes.c_float),
        ("z0", ctypes.c_float),
        ("nx", ctypes.c_float),        # unit outward normal of the owning
        ("ny", ctypes.c_float),        #   room's edge (exactly one is ±1)
        ("kind", ctypes.c_int32),      # SOLID_KIND_* in cs2_solids.h
    ]


class WallListC(ctypes.Structure):
    # Mirrors C WallList: ptr + count + capacity. Appended on StaticDataC
    # after the binding.init prefix so area_bounds can follow at C offsets.
    _fields_ = [
        ("walls", ctypes.POINTER(WallC)),
        ("count", ctypes.c_int),
        ("capacity", ctypes.c_int),
    ]


class StaticDataC(ctypes.Structure):
    # fmt: off  -- YAPF aligns standalone comments to trailing-comment column; suppress here
    _fields_ = [
        ("N", ctypes.c_int),
        ("vis_matrix", ctypes.POINTER(ctypes.c_int8)),
        ("raster_grid", ctypes.POINTER(ctypes.c_int32)),
        ("adjacency", ctypes.POINTER(ctypes.c_int8)),
        ("centroid_xy", ctypes.POINTER(ctypes.c_float)),
        # T2 (verticality): per-area terrain elevation and ramp flag.
        # Field order MUST stay in sync with:
        #   - StaticData struct in cs2_types.h  (C canonical source)
        #   - PyArg_ParseTuple format string in binding.c py_init()
        # Mismatch here silently corrupts all pointer fields that follow.
        ("centroids_z", ctypes.POINTER(ctypes.c_float)),               # float32[N] — terrain z per area  # noqa: E501
        ("area_ids", ctypes.POINTER(ctypes.c_int32)),
        ("bombsite_mask", ctypes.POINTER(ctypes.c_int8)),
        ("bombsite_by_idx", ctypes.POINTER(ctypes.c_int8)),
        # is_ramp: MapData bool is converted to int8 in Cs2Env.__init__ before passing
        ("is_ramp", ctypes.POINTER(ctypes.c_int8)),                    # int8[N] — 1=ramp/stairs
        ("bombsite_dist", ctypes.POINTER(ctypes.c_float)),
        ("grid_w", ctypes.c_int),
        ("grid_h", ctypes.c_int),
        ("max_area_id", ctypes.c_int),
        ("grid_x_min", ctypes.c_float),
        ("grid_y_min", ctypes.c_float),
        ("grid_inv_cell", ctypes.c_float),
        ("inv_x_range", ctypes.c_float),
        ("inv_y_range", ctypes.c_float),
        ("x_offset", ctypes.c_float),
        ("y_offset", ctypes.c_float),
        ("bombsite_dist_scale", ctypes.c_float),
        ("laser_damage", ctypes.c_int32),
        ("laser_range", ctypes.c_float),
        ("laser_range_sq", ctypes.c_float),
        ("shoot_cooldown", ctypes.c_int32),
        ("bomb_plant_time", ctypes.c_int32),
        ("bomb_defuse_time", ctypes.c_int32),
        ("bomb_defuse_kit", ctypes.c_int32),
        ("bomb_timer", ctypes.c_int32),
        ("round_time", ctypes.c_int32),
        ("footstep_radius_sq", ctypes.c_float),
        ("gunshot_radius_sq", ctypes.c_float),
        ("enemy_memory_ticks", ctypes.c_int32),
        ("stale_memory_tick", ctypes.c_int32),
        ("pbrs_gamma", ctypes.c_float),
        ("delta_x", ctypes.c_float * 9),
        ("delta_y", ctypes.c_float * 9),
        ("dir_facing", ctypes.c_float * 9),
        ("t_spawns", ctypes.c_int32 * 15),
        ("n_t_spawns", ctypes.c_int),
        ("ct_spawns", ctypes.c_int32 * 5),
        ("n_ct_spawns", ctypes.c_int),
        ("max_turn_speed", ctypes.c_float),
        # legacy symmetric — superseded by per-mechanism fields (Batch 1)
        ("reward_win", ctypes.c_float),
        # Batch 1 (RL overhaul): per-outcome win magnitudes (Task 3).
        # Must stay in same order as StaticData in cs2_types.h.
        ("reward_win_t_detonation", ctypes.c_float),                   # default 5.0
        ("reward_win_t_elimination", ctypes.c_float),                  # default 3.0
        ("reward_win_ct_defuse", ctypes.c_float),                      # default 5.0
        ("reward_win_ct_timeout", ctypes.c_float),                     # default 4.0
        ("reward_win_ct_elimination", ctypes.c_float),                 # default 3.0
        ("reward_kill", ctypes.c_float),
        ("reward_death", ctypes.c_float),
        ("reward_bombsite_entry", ctypes.c_float),
        ("reward_plant_bonus", ctypes.c_float),
        ("reward_plant_base", ctypes.c_float),
        ("reward_plant_progress_scale", ctypes.c_float),
        ("reward_plant_interrupted", ctypes.c_float),
        ("reward_defuse", ctypes.c_float),
        ("reward_shot_penalty", ctypes.c_float),
        ("reward_ct_survival", ctypes.c_float),
        ("reward_inaction", ctypes.c_float),
        ("pbrs_alive_weight", ctypes.c_float),
        ("pbrs_hp_weight", ctypes.c_float),
        ("pbrs_site_weight", ctypes.c_float),
        ("pbrs_bomb_progress_weight", ctypes.c_float),
        ("pbrs_nav_weight_t", ctypes.c_float),
        ("pbrs_nav_weight_ct", ctypes.c_float),
        # Rung 0 (spec 2026-08-29): FMT positions 69-71, before wall_list in C.
        ("n_active_per_team", ctypes.c_int32),
        ("pin_pitch", ctypes.c_int32),
        ("crouch_enabled", ctypes.c_int32),
        # Rung 1a (spec 2026-08-30 T2a): FMT position 72, same block.
        ("jump_enabled", ctypes.c_int32),
        # After the binding.init prefix: overlay C wall_list then ramp bounds.
        # These two stay LAST, matching cs2_types.h. New scalars go above them
        # (and at the same spot in the C struct); the offset anchors below are
        # what proves the two sides moved together, so no hand-measured
        # offsetof literals are quoted here any more — they rot on every insert.
        ("wall_list", WallListC),
        ("area_bounds", ctypes.POINTER(ctypes.c_float)),
    ]
    # fmt: on


class AgentStateC(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("area_idx", ctypes.c_int32),
        ("facing", ctypes.c_float),
        ("hp", ctypes.c_int32),
        ("fire_cd", ctypes.c_int32),
        ("alive", ctypes.c_int8),
        ("has_bomb", ctypes.c_int8),
        ("has_kit", ctypes.c_int8),
        ("team", ctypes.c_int8),
        ("is_moving", ctypes.c_int8),
        ("fired_this_tick", ctypes.c_int8),
        ("_pad0", ctypes.c_int8 * 2),
        ("enemy_mem_idx", ctypes.c_int32 * 5),
        ("enemy_mem_tick", ctypes.c_int32 * 5),
                                                       # Phase 4b additions
        ("vx", ctypes.c_float),
        ("vy", ctypes.c_float),
        ("is_crouching", ctypes.c_int8),
        ("_pad1", ctypes.c_int8 * 3),
        ("crouch_cd", ctypes.c_int32),
        ("armor", ctypes.c_int32),
        ("has_helmet", ctypes.c_int8),
        ("weapon_slot", ctypes.c_int8),
        ("weapon_slot_target", ctypes.c_int8),
        ("_pad2", ctypes.c_int8 * 1),
        ("ammo_clip", ctypes.c_int32 * 3),
        ("ammo_reserve", ctypes.c_int32 * 3),
        ("reload_ticks", ctypes.c_int32),
        ("switch_ticks", ctypes.c_int32),
                                                       # Phase 6 additions
        ("human_controlled", ctypes.c_uint8),
        ("_pad3", ctypes.c_int8 * 3),
        ("aim_rad", ctypes.c_float),
                                                       # Phase 7 additions (jump)
        ("vz", ctypes.c_float),
        ("is_airborne", ctypes.c_int8),
        ("_pad4", ctypes.c_int8 * 3),
        ("jump_cd", ctypes.c_int32),
                                                       # Batch 3.5: pitch — appended, mirror cs2_types.h AgentState.
        ("pitch", ctypes.c_float),
                                                       # Sim recoil v1 (#120): punch — appended, never reorder.
                                                       # Shared by the hit ray / demo camera when recoil_enabled.
                                                       # Do not write these into facing / aim_rad / stored pitch.
        ("punch_pitch", ctypes.c_float),
        ("punch_yaw", ctypes.c_float),
                                                       # Rung 0: 0 for parked slots (spec §2.1).
                                                       # NOT the same as alive=0 — see cs2_types.h.
        ("participating", ctypes.c_int8),
        ("_pad5", ctypes.c_int8 * 3),
    ]


class GameStateC(ctypes.Structure):
    _fields_ = [
        ("tick", ctypes.c_int32),
        ("round_ticks_left", ctypes.c_int32),
        ("agents", AgentStateC * 10),
        ("bomb_planted", ctypes.c_int8),
        ("round_over", ctypes.c_int8),
        ("winner", ctypes.c_int32),
        ("bomb_carrier_id", ctypes.c_int32),
        ("bomb_area_idx", ctypes.c_int32),
        ("bomb_x", ctypes.c_float),
        ("bomb_y", ctypes.c_float),
        ("bomb_z", ctypes.c_float),
        ("bomb_ticks_left", ctypes.c_int32),
        ("bomb_being_planted_by", ctypes.c_int32),
        ("bomb_plant_ticks", ctypes.c_int32),
        ("bomb_being_defused_by", ctypes.c_int32),
        ("bomb_defuse_ticks", ctypes.c_int32),
                                                                       # Batch 2: round-fixed designated carrier — mirrors C GameState.  # noqa: E501
                                                                       # See cs2_types.h for why/pitfalls; insertion order matters for alignment.  # noqa: E501
        ("round_designated_carrier_id", ctypes.c_int32),
        ("bombsite_entered", ctypes.c_int8 * 5),
        ("bomb_is_dropped", ctypes.c_int8),
        ("_pad_gs", ctypes.c_int8 * 2),
    ]


class StepStatsC(ctypes.Structure):
    _fields_ = [
        ("bomb_planted", ctypes.c_int32),
        ("bomb_defused", ctypes.c_int32),
        ("kills_t", ctypes.c_int32),
        ("kills_ct", ctypes.c_int32),
        ("blocked_moves_t", ctypes.c_int32),
        ("blocked_moves_ct", ctypes.c_int32),
        ("winner", ctypes.c_int32),
        ("winner_t", ctypes.c_int32),
        ("winner_ct", ctypes.c_int32),
        ("timed_out", ctypes.c_int32),
        ("alive_t_end", ctypes.c_int32),
        ("alive_ct_end", ctypes.c_int32),
        ("round_length", ctypes.c_int32),
        ("action_move", ctypes.c_int32 * 9),
        ("action_shoot", ctypes.c_int32 * 2),
        ("action_use", ctypes.c_int32 * 2),
                                                                       # action_last removed (F13) — dead legacy counter, see cs2_types.h.  # noqa: E501
                                                                       # Batch 3: continuous-aim Δyaw stats (mirror C StepStats fields).  # noqa: E501
                                                                       # Replaces the 16-bin action_aim histogram (64B) with a Welford-style  # noqa: E501
                                                                       # triple (sum + sq_sum + count = 12B). No explicit _pad_aim_delta —  # noqa: E501
                                                                       # the three int32-aligned fields slot in cleanly between action_use  # noqa: E501
                                                                       # and action_reload. See cs2_types.h StepStats comment.  # noqa: E501
        ("aim_delta_sum", ctypes.c_float),
        ("aim_delta_sq_sum", ctypes.c_float),
        ("aim_delta_count", ctypes.c_int32),
                                                                       # Batch 3.5: pitch Welford triple (mirror cs2_types.h StepStats fields).
        ("aim_delta_pitch_sum", ctypes.c_float),
        ("aim_delta_pitch_sq_sum", ctypes.c_float),
        ("aim_delta_pitch_count", ctypes.c_int32),
        ("action_reload", ctypes.c_int32 * 2),
        ("action_weapon", ctypes.c_int32 * 3),
        ("action_crouch", ctypes.c_int32 * 2),
        ("action_jump", ctypes.c_int32 * 2),
        ("reward_win", ctypes.c_float),
        ("reward_kills", ctypes.c_float),
        ("reward_deaths", ctypes.c_float),
        ("reward_bomb", ctypes.c_float),
        ("reward_pbrs", ctypes.c_float),
        ("reward_shots", ctypes.c_float),
        ("reward_survival", ctypes.c_float),
        ("reward_inaction", ctypes.c_float),
                                                                       # Batch 1 (RL overhaul): round-end win classification flags.  # noqa: E501
                                                                       # Cleared by round_reset (Task 2). Set by compute_rewards (Task 3).  # noqa: E501
                                                                       # Consumed by split_into_channels to route reward_win:  # noqa: E501
                                                                       #   detonation/defuse → objective channel; else → combat channel.  # noqa: E501
        ("win_by_detonation", ctypes.c_int8),                          # 1 when bomb detonated (T wins)
        ("win_by_defuse", ctypes.c_int8),                              # 1 when bomb was defused (CT wins)
        ("_pad_ss_wins", ctypes.c_int8 * 2),                           # pad to 4-byte boundary
        ("plant_tick", ctypes.c_int32),                                # g->tick at plant; 0 = never planted
                                                                       # R0-A (spec 2026-08-29 §3) — appended, never reorder. Mirrors the
                                                                       # combat-instrumentation tail of C StepStats (cs2_types.h); semantics
                                                                       # documented there. reward_win_ct is the tail anchor (_C_OFFSET_FIELDS).
        ("shots_fired", ctypes.c_int32),
        ("shots_with_enemy_in_los", ctypes.c_int32),
        ("shots_facing_enemy", ctypes.c_int32),
        ("shots_on_target", ctypes.c_int32),
        ("shots_hit", ctypes.c_int32),
        ("shots_stance_blocked", ctypes.c_int32),
        ("mutual_vis_pair_ticks", ctypes.c_int32),
        ("agent_ticks_with_visible_enemy", ctypes.c_int32),
        ("damage_dealt", ctypes.c_float),
        ("min_enemy_distance", ctypes.c_float),                        # 1e30 sentinel = no pair coexisted
        ("reward_win_t", ctypes.c_float),
        ("reward_win_ct", ctypes.c_float),
    ]


class StepStatsView:
    """Dict-like zero-copy view over a ctypes StepStatsC struct.

    Cs2Env.step() returns a reference to this wrapper in info[0]["step_stats"]
    when include_step_stats_in_info=True. The wrapper proxies __getitem__ to
    attribute access on the underlying struct, so downstream consumers (e.g.
    split_into_channels in src/train_helpers_batch1.py) can read fields by
    name with zero per-tick allocation.

    ndim is set to 0 so split_into_channels's length-1 squeeze (which only
    fires for ndim==1 structured arrays) does not trigger on this wrapper.
    """
    __slots__ = ("_ss", )
    ndim = 0                           # class attr so split_into_channels's ndim==1 check is False

    def __init__(self, ss):
        self._ss = ss

    def __getitem__(self, key):
        return getattr(self._ss, key)

    def get(self, key, default=None):
        return getattr(self._ss, key, default)


class Dust2EnvC(ctypes.Structure):
    _fields_ = [
        ("sd", ctypes.POINTER(StaticDataC)),
        ("game", GameStateC),
        ("step_stats", StepStatsC),
        ("episode_stats", StepStatsC),
        ("observations", ctypes.c_float * (N_AGENTS * OBS_DIM)),
        ("rewards", ctypes.c_float * N_AGENTS),
        ("terminals", ctypes.c_int8 * N_AGENTS),
        ("truncations", ctypes.c_int8 * N_AGENTS),
        ("team_spirit", ctypes.c_float),
        ("rng", ctypes.c_uint32),
        ("masks", ctypes.c_int8 * (N_AGENTS * ACTION_MASK_DIM)),
        ("client", ctypes.c_void_p),                                   # Client* (NULL in training)
                                                                       # Sim recoil v1 (#120): after client, not a binding.init arg.
                                                                       # make_env writes this after from_address; env_reset does not
                                                                       # clear it (memsets GameState only). 0=hitscan, 1=punch on ray.
        ("recoil_enabled", ctypes.c_int32),
    ]


# ── ctypes mirror ↔ C layout guard ──────────────────────────────────────────
# Every struct above is overlaid byte-for-byte on memory the C extension owns,
# so a mirror that drifts from cs2_types.h reads garbage SILENTLY — ctypes
# cannot see the real C layout. These asserts pin each mirror to the C
# compiler's own sizeof/offsetof, published by binding.struct_sizes().
#
# Why not literals: this block used to carry hand-measured numbers
# (164/1708/204/6832; offsets 476/480/496) plus a running changelog explaining
# each delta. They rotted on every appended field and could only be refreshed
# by compiling a throwaway printf TU by hand, so a genuine drift surfaced as a
# confusing assert about the wrong number. The history lives in git; the
# numbers now come from the same compiler invocation that laid the structs out.
#
# If one of these raises at import time the MIRROR is wrong, not the assert —
# fix _fields_ above. Adding a struct? Add a sizeof key AND a tail offsetof key
# to py_struct_sizes() in binding.c, plus an entry in each matching tuple below
# (_C_SIZE_MIRRORS and _C_OFFSET_FIELDS). BOTH halves of that slip are caught,
# so neither a mirror nor a key can sit unguarded:
#   - mirror declared here, no key in binding.c ->
#     test_every_ctypes_mirror_is_size_guarded enumerates the ctypes.Structure
#     subclasses DEFINED IN THIS MODULE and compares them to _C_SIZE_MIRRORS.
#     (An earlier version of this comment claimed this direction was
#     undetectable. It is not: the module namespace is the second, independent
#     list of mirrors — that is what makes the census possible.)
#   - key in binding.c, never consumed here ->
#     test_struct_sizes_keys_are_all_consumed asserts
#     set(binding.struct_sizes()) == _C_SIZE_KEYS_CHECKED.
# Both live in tests/test_struct_sizes.py.
#
# Pitfall: struct_sizes() reads the CURRENTLY BUILT .so. Editing src/c_env/*.h
# without rebuilding (`python setup.py build_ext --inplace`) compares a new
# mirror against a stale binary — it can pass on a broken tree or fail on a
# correct one. Rebuild first, then trust this.
_C_SIZES = binding.struct_sizes()
# fmt: off  -- one pair per line; YAPF repacks these into unreadable columns
# (struct_sizes() key -> ctypes mirror) — sizeof pairs.
_C_SIZE_MIRRORS = (
    ("AgentState", AgentStateC),
    ("GameState", GameStateC),
    ("StepStats", StepStatsC),
    ("Dust2Env", Dust2EnvC),
    ("StaticData", StaticDataC),
    ("WallList", WallListC),
    ("Wall", WallC),
)
# (ctypes mirror, field name, struct_sizes() key) — offsetof anchors.
#
# EVERY mirror's TAIL field is anchored here, not just StaticData's. sizeof is
# blind to the drift that costs the most: a field inserted mid-struct in
# cs2_types.h but appended at the END of the mirror keeps sizeof identical, so
# the size guard above passes and every field after the insertion point reads
# the wrong bytes, silently, forever. The last field's offset DOES move under
# that edit. Swapping two same-width fields is the same story — invisible to
# sizeof, visible to the anchor when the swap reaches the tail.
#
# RULE when adding a field: it goes in the SAME position in the C struct
# (cs2_types.h) and in the mirror above. Appending — the usual case — makes it
# the new tail, so retarget that struct's anchor to it in BOTH py_struct_sizes()
# (binding.c) and the tuple below, in the same commit. Leaving the anchor on the
# old tail still works (both sides shift together) but stops guarding the tail.
#
# StaticData carries three anchors instead of one: wall_list/area_bounds are
# appended after a long run of float reward weights, and pbrs_nav_weight_ct pins
# the end of that run, so a drift inside the run is localised rather than only
# surfacing at the very end.
_C_OFFSET_FIELDS = (
    (StaticDataC, "pbrs_nav_weight_ct", "StaticData_pbrs_nav_weight_ct_offset"),
    (StaticDataC, "wall_list", "StaticData_wall_list_offset"),
    (StaticDataC, "area_bounds", "StaticData_area_bounds_offset"),
    (AgentStateC, "_pad5", "AgentState__pad5_offset"),
    # GameState's tail is its explicit pad array, not a "real" field. offsetof
    # on a pad is legal, and the rule is uniform: anchor the LAST field. Picking
    # bomb_is_dropped instead would miss a field slipped in between it and the
    # pad on one side only.
    (GameStateC, "_pad_gs", "GameState__pad_gs_offset"),
    (StepStatsC, "reward_win_ct", "StepStats_reward_win_ct_offset"),
    (Dust2EnvC, "recoil_enabled", "Dust2Env_recoil_enabled_offset"),
    (WallC, "kind", "Wall_kind_offset"),
    (WallListC, "capacity", "WallList_capacity_offset"),
)
# (struct_sizes() key -> Python value) — bare macros nav.py re-declares in
# Python. Pin them to the header: the reward views below slice
# rewards[:TEAM_SIZE], so a drift would mis-attribute every team-spirit term
# rather than crash.
_C_MACROS = (
    ("TEAM_SIZE", TEAM_SIZE),
    ("N_AGENTS", N_AGENTS),
)
# fmt: on

for _name, _mirror in _C_SIZE_MIRRORS:
    # _mirror.__name__ rather than f"{_name}C": the "C" suffix is a convention
    # the tuple does not enforce, so concatenating it would print a class name
    # that may not exist. __name__ is always the class actually compared.
    # RuntimeError, not assert: python -O strips asserts and this is the only
    # layout guard outside the test suite.
    if _C_SIZES[_name] != ctypes.sizeof(_mirror):
        raise RuntimeError(f"{_mirror.__name__} size mismatch (struct_sizes key {_name!r}): "
                           f"ctypes {ctypes.sizeof(_mirror)} vs C {_C_SIZES[_name]}")
del _name, _mirror
for _mirror, _field, _key in _C_OFFSET_FIELDS:
    # _mirror.__name__, not a hard-coded class: the tuple spans every mirror
    # now, so the message must name the one that actually drifted.
    if getattr(_mirror, _field).offset != _C_SIZES[_key]:
        raise RuntimeError(
            f"{_mirror.__name__}.{_field} offset mismatch (struct_sizes key {_key!r}): "
            f"ctypes {getattr(_mirror, _field).offset} vs C {_C_SIZES[_key]}")
del _mirror, _field, _key
for _macro, _py_value in _C_MACROS:
    if _py_value != _C_SIZES[_macro]:
        raise RuntimeError(f"{_macro} mismatch: nav {_py_value} vs C {_C_SIZES[_macro]}")
del _macro, _py_value

# Every struct_sizes() key this module actually compares, derived from the three
# tuples above rather than re-listed by hand (a hand-written copy would be the
# next thing to rot). tests/test_struct_sizes.py asserts
# set(binding.struct_sizes()) == _C_SIZE_KEYS_CHECKED, which turns "published a
# key in binding.c and forgot to consume it here" from a silent unguarded field
# into a failing test. Exported for that test; nothing in the sim reads it.
_C_SIZE_KEYS_CHECKED = (frozenset(_n for _n, _ in _C_SIZE_MIRRORS)
                        | frozenset(_k for _, _, _k in _C_OFFSET_FIELDS)
                        | frozenset(_m for _m, _ in _C_MACROS))

# ctypes helper to extract raw pointer from PyCapsule
_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

# Module-level cache for map data
_ENV_CACHE: dict = {}

# ── Zero-sum reward symmetrization (spec 2026-08-01 §4.3) ─────────────────────


def symmetrize_rewards(rewards, n_active_per_team=TEAM_SIZE):
    """Rewrite a 10-agent reward vector to be exactly zero-sum, in place.

    WHAT: for agent i on team A facing team B,
        r_i' = 0.5 * ( r_i - mean_{j in ACTIVE(B)}(r_j) )
    Agents 0..TEAM_SIZE-1 are T, TEAM_SIZE..N_AGENTS-1 are CT (same split the
    C CT-survival loop uses, src/c_env/cs2_rewards.h:225). Only the first
    n_active_per_team slots of each team are read or written; the parked
    remainder (Rung 0, spec 2026-08-29 §2.1) is left untouched at exactly 0.0.

    WHY: the shared self-play policy is paid for private, non-zero-sum
    per-team subsidies (the CT survival drip, the timeout-win bonus); that
    gradient leaks through the shared trunk and shows up on the T side as
    ever-later plants. Subtracting the opponent's mean restores the exactly
    zero-sum structure under which shared-policy self-play is known to work.
    mean (not sum) keeps per-agent scale comparable; the 0.5x keeps total
    reward scale from ~doubling, which would interact with the return-norm
    patch and the BC-warm-started critic. Subtracting the opponent's PBRS
    term is itself potential-based (-phi_B(s)), so PBRS policy invariance
    survives.

    PITFALLS:
    1. The divisor is a FIXED ROSTER SIZE (n_active_per_team, TEAM_SIZE by
       default) — do not "fix" it to a per-tick alive-count. Dividing each
       team's mean by its own alive count breaks the exact zero-sum property
       this function exists to provide: the sum over the 2n written rows is
       0.5*(S_A - n*mean_B) + 0.5*(S_B - n*mean_A), and the two half-terms
       cancel ONLY because both means are scaled by the same constant n. The
       C team_spirit loop right below the PBRS block IS alive-gated
       (src/c_env/cs2_rewards.h:247), so mirroring it here looks like the
       obvious consistency fix; it would silently destroy zero-sum.
       Consequence to carry into analysis, not a wart to repair: late-round
       with n-1 dead CTs, the lone survivor's stall drip is attenuated to 1/n
       before subtraction, so subsidy cancellation is WEAKEST exactly in the
       stall-heavy end-of-round window the A2 experiment targets (review
       finding 7).
    2. BOTH team means must be read before EITHER slice is written — writing
       the T slice first makes the CT half depend on transformed values.
    3. In place, and the caller's array is what matters: on the vecenv path
       this array is the PufferLib shared reward buffer the trainer reads,
       and the returned tuple is ignored by the Multiprocessing backend.
    4. Safe to mutate the zero-copy C view: env_step memsets env->rewards
       (src/c_env/cs2_env.h:339) before accumulating this tick's terms, and
       nothing reads the reward array ahead of that memset — the calls that
       precede it (update_enemy_memory, compute_observations) touch obs and
       memory, not rewards. The C episode / step stat channels are separate
       accumulators, which is exactly why those channels stay PRE-transform
       (spec §4.3 analysis caveat).
    5. No all-zero fast path. `if not rewards.any(): return` looks free but is
       strictly harmful: measured 5000/5000 ticks carry a nonzero reward,
       because the PBRS loop adds a term for every PARTICIPATING agent, dead or
       alive (cs2_rewards.h, the `if (!g->agents[i].participating) continue;`
       block). The guard would never fire and would tax every step with an
       extra full-array scan.
    6. Rung 0 (spec 2026-08-29 §2.1): BOTH means and BOTH written slices are
       restricted to the n_active_per_team ACTIVE slots. Getting either half
       of that wrong is a real bug, not a cosmetic one — the first Rung 0
       version kept the TEAM_SIZE divisor and wrote all 10 rows, which broke
       two things at once:
         - PARKED rows arrived at 0.0 and left at -0.5*mean_opponent, so the
           "parked slots get zero reward every tick" contract held only as
           long as a trainer-side participating mask covered for it;
         - ACTIVE rows had the opponent-mean subtraction attenuated by
           n/TEAM_SIZE — at n=1 an agent got 0.5*(r_0 - r_5/5) where the
           transform is defined as 0.5*(r_0 - r_5), i.e. a 5x weaker subsidy
           cancellation. No mask repairs that: it is the UNMASKED rows that
           are wrong.
       Zero-sum stays EXACT in the active-only form, which is why restricting
       the slices is the fix and not a violation of PITFALL 1: the written
       rows sum to 0.5*(S_T - n*mean_ct) + 0.5*(S_CT - n*mean_t) =
       0.5*(S_T - S_CT) + 0.5*(S_CT - S_T) = 0 (since n*mean_ct == S_CT and
       n*mean_t == S_T), and the parked rows contribute 0 because they are
       never written. What PITFALL 1 forbids is a divisor that varies per
       TICK (an alive count), not one that is constant for the whole run.
    """
    n = n_active_per_team
    # Guard the now-public parameter: n > TEAM_SIZE would fold T rows into the
    # CT mean and write past the roster; n < 1 gives a mean of an empty slice.
    if not 1 <= n <= TEAM_SIZE:
        raise ValueError(f"n_active_per_team must be in 1..{TEAM_SIZE}, got {n}")
    # Both means BEFORE either write (PITFALL 2). Sliced, not masked: at the
    # default n == TEAM_SIZE, rewards[TEAM_SIZE:TEAM_SIZE + n] is the identical
    # view to the old rewards[TEAM_SIZE:], so .mean() reduces in the same order
    # and the pre-Rung-0 float results are reproduced bit for bit
    # (tests/test_parked_agents.py::test_symmetrize_default_matches_pre_rung0_bitwise).
    mean_t = rewards[:n].mean()
    mean_ct = rewards[TEAM_SIZE:TEAM_SIZE + n].mean()
    rewards[:n] = 0.5 * (rewards[:n] - mean_ct)
    rewards[TEAM_SIZE:TEAM_SIZE + n] = 0.5 * (rewards[TEAM_SIZE:TEAM_SIZE + n] - mean_t)
    return rewards


# ── Cs2Env ────────────────────────────────────────────────────────────────────


class Cs2Env(pufferlib.PufferEnv):

    def __init__(
            self,
            seed=0,
            team_spirit=0.0,
            buf=None,
            nav_graph=None,
            auto_reset=True,
            map_data=None,
            reward_win=1.0,
            reward_kill=0.3,
            reward_death=0.1,
            reward_bombsite_entry=0.3,
            reward_plant_bonus=3.0,
            reward_plant_base=0.2,
            reward_plant_progress_scale=0.05,
            reward_plant_interrupted=0.1,
            reward_defuse=0.2,
            reward_shot_penalty=0.005,
            reward_ct_survival=0.001,
            reward_inaction=0.0005,
            pbrs_alive_weight=0.3,
            pbrs_hp_weight=0.002,
            pbrs_site_weight=0.2,
            pbrs_bomb_progress_weight=0.3,
            pbrs_nav_weight_t=0.04,
            pbrs_nav_weight_ct=0.15,
                                                                                # MUST equal the training discount (build_train_config "gamma") —  # noqa: E501
                                                                                # PBRS F(s,s') = γ_pbrs·φ(s') − φ(s) is policy-invariant only when  # noqa: E501
                                                                                # γ_pbrs == γ (finding 2, 2026-07-06 review; was 0.99 vs 0.999).  # noqa: E501
                                                                                # Drift-guarded by test_pbrs_gamma_matches_training_gamma.  # noqa: E501
            pbrs_gamma=0.999,
                                                                                # Batch 1 (RL overhaul): per-outcome win magnitudes.  # noqa: E501
                                                                                # These supersede the symmetric reward_win at round end.  # noqa: E501
                                                                                # Defaults chosen to make detonation/defuse > timeout > elimination.  # noqa: E501
            reward_win_t_detonation=5.0,
            reward_win_t_elimination=3.0,
            reward_win_ct_defuse=5.0,
            reward_win_ct_timeout=4.0,
            reward_win_ct_elimination=3.0,
            include_step_stats_in_info: bool = False,                           # Task 6a (utof/cs2rl#7)  # noqa: E501
            reward_symmetrize: bool = False,                                    # spec 2026-08-01 §4.3  # noqa: E501
            recoil: bool = False,                                               # #120: punch on ray; default off (today's hitscan)  # noqa: E501
            n_active_per_team: int = TEAM_SIZE,                                 # Rung 0 §2.1: agents per team that spawn  # noqa: E501
            pin_pitch: int = 0,                                                 # Rung 0 R0-E.2: ignore pitch action  # noqa: E501
            crouch_enabled: int = 1,                                            # Rung 0 R0-E.2: mask crouch when 0  # noqa: E501
            jump_enabled: int = 1,                                              # Rung 1a T2a: mask jump when 0  # noqa: E501
            round_time: int
        | None = None,                                                          # Rung 0 R0-G: ticks per round; None ⇒ nav.ROUND_TIME  # noqa: E501
            laser_range: float
        | None = None,                                                          # Rung 0 R0-G: hitscan reach; None ⇒ nav.LASER_RANGE  # noqa: E501
            max_turn_speed: float
        | None = None,                                                          # Rung 0 R0-G: rad/tick aim clamp; None ⇒ nav.MAX_TURN_SPEED_RAD  # noqa: E501
    ):
        self.single_observation_space = gymnasium.spaces.Box(low=-5.0,
                                                             high=5.0,
                                                             shape=(OBS_DIM, ),
                                                             dtype=np.float32)
        self.single_action_space = gymnasium.spaces.MultiDiscrete(list(ACTION_HEAD_SIZES))
        self.num_agents = N_AGENTS
        super().__init__(buf)

        self.nav_graph = nav_graph
        self.map_data = map_data
        self._auto_reset = bool(auto_reset)
        self._uses_external_buffers = buf is not None

        # team_spirit may be float or multiprocessing.Value
        self._team_spirit_shared = team_spirit if hasattr(team_spirit, "value") else None
        if self._team_spirit_shared is not None:
            init_team_spirit = float(team_spirit.value)
        elif team_spirit is None:
            init_team_spirit = 0.0
        else:
            init_team_spirit = float(team_spirit)

        md = map_data

        def _arr(a, dtype):
            return np.ascontiguousarray(a.astype(dtype, copy=False).flatten())

        vis_matrix = _arr(md.vis_matrix, np.int8)
        raster_grid = _arr(md.grid, np.int32)
        adjacency = _arr(md.adjacency, np.int8)
        centroid_xy = _arr(md.centroids, np.float32)
        # T2 (verticality): centroids_z must be float32 (C side reads as float*).
        # _arr coerces dtype via astype, but if MapData ever passes the wrong dtype
        # we want a loud error here, not silent coercion poisoning the C pointer.
        centroids_z = _arr(md.centroids_z, np.float32)
        if centroids_z.dtype != np.float32:
            raise TypeError(f"centroids_z float32 conversion failed: dtype={centroids_z.dtype}; "
                            "this would dangle the C-side sd->centroids_z pointer")
        area_ids = _arr(md.area_ids, np.int32)
        bombsite_mask = _arr(md.bombsite_mask, np.int8)
        bombsite_by_idx = _arr(md.bombsite_by_idx, np.int8)
        # T2 (verticality): is_ramp is bool in MapData but C reads int8*.
        # Convert explicitly — numpy bool layout is platform-dependent.
        # Pitfall: do NOT pass md.is_ramp directly — bool dtype may not
        # be 1-byte on all platforms; int8 is guaranteed portable.
        # np.ascontiguousarray guards against stride surprises post-astype.
        is_ramp_int8 = np.ascontiguousarray(md.is_ramp.astype(np.int8))
        if is_ramp_int8.dtype != np.int8 or is_ramp_int8.itemsize != 1:
            raise ValueError(f"is_ramp int8 conversion failed: dtype={is_ramp_int8.dtype}, "
                             f"itemsize={is_ramp_int8.itemsize}; "
                             "this would dangle the C-side sd->is_ramp pointer")
        bombsite_dist = _arr(md.bombsite_dist, np.float32)

        inv_x = 2.0 / (md.x_max - md.x_min)
        inv_y = 2.0 / (md.y_max - md.y_min)
        x_off = (md.x_max + md.x_min) / (md.x_max - md.x_min)
        y_off = (md.y_max + md.y_min) / (md.y_max - md.y_min)

        id2idx = {int(aid): i for i, aid in enumerate(md.area_ids)}
        t_spawns = np.array([id2idx[a] for a in md.t_spawn_areas], dtype=np.int32)
        ct_spawns = np.array([id2idx[a] for a in md.ct_spawn_areas], dtype=np.int32)
        delta_x = np.array([float(nav._DELTA_VECTORS[k][0]) for k in range(9)], dtype=np.float32)
        delta_y = np.array([float(nav._DELTA_VECTORS[k][1]) for k in range(9)], dtype=np.float32)
        dir_facing = np.array([float(nav._DIR_FACING[k]) for k in range(9)], dtype=np.float32)

        # Keep refs alive — prevents GC of backing numpy arrays.
        # T2 (verticality): centroids_z and is_ramp_int8 added here so the C pointers
        # sd->centroids_z and sd->is_ramp remain valid for the env's lifetime.
        # is_ramp_int8 is a NEW array (result of .astype); it would be collected
        # immediately if not held here — the C pointer would then dangle.
        self._refs = [
            vis_matrix,
            raster_grid,
            adjacency,
            centroid_xy,
            centroids_z,
            area_ids,
            bombsite_mask,
            bombsite_by_idx,
            is_ramp_int8,
            bombsite_dist,
            t_spawns,
            ct_spawns,
            delta_x,
            delta_y,
            dir_facing,
        ]

        # Rung 0 (spec 2026-08-29 §2.1): validate BEFORE binding.init. env_init
        # asserts the same range in C, and a failed C assert aborts the whole
        # process — inside a Puffer worker that is a silent death with no
        # traceback. Raising here turns a bad training config into an ordinary
        # Python error. pin_pitch / crouch_enabled / jump_enabled are flags, so
        # any truthy value normalises to 1 rather than being rejected.
        # Reject non-integers rather than truncating (int(2.9) == 2 would
        # silently park a different roster than the config asked for).
        if int(n_active_per_team) != n_active_per_team:
            raise ValueError(f"n_active_per_team must be an integer, got {n_active_per_team!r}")
        n_active_per_team = int(n_active_per_team)
        if not 1 <= n_active_per_team <= TEAM_SIZE:
            raise ValueError(f"n_active_per_team must be in 1..{TEAM_SIZE}, "
                             f"got {n_active_per_team}")
        self.n_active_per_team = n_active_per_team
        self.pin_pitch = int(bool(pin_pitch))
        self.crouch_enabled = int(bool(crouch_enabled))
        self.jump_enabled = int(bool(jump_enabled))

        # Rung 0 R0-G: env knobs. None ⇒ the nav.py constant, so demo/test/
        # deploy callers that never pass them keep today's values byte-for-byte
        # (sim fingerprints at defaults must not move). Validated here, not in
        # C, for the same reason as n_active_per_team above: a C assert kills a
        # forked Puffer worker silently. round_time is rejected (not truncated)
        # when non-integral — int(2.5) would run a different episode length
        # than the config recorded.
        # PITFALL: laser_range_sq (FMT 24) is derived from _laser_range below;
        # never accept it as a separate kwarg or the range check and the
        # damage falloff would disagree.
        if round_time is None:
            round_time = nav.ROUND_TIME
        if int(round_time) != round_time:
            raise ValueError(f"round_time must be an integer tick count, got {round_time!r}")
        self._round_time = int(round_time)
        self._laser_range = float(nav.LASER_RANGE if laser_range is None else laser_range)
        self._max_turn_speed = float(nav.MAX_TURN_SPEED_RAD if max_turn_speed is
                                     None else max_turn_speed)
        if self._round_time <= 0:
            raise ValueError(f"round_time must be > 0, got {self._round_time}")
        if not self._laser_range > 0.0:                # `not >` also rejects NaN
            raise ValueError(f"laser_range must be > 0, got {self._laser_range}")
        if not self._max_turn_speed > 0.0:
            raise ValueError(f"max_turn_speed must be > 0, got {self._max_turn_speed}")

        # Call binding.init() — positional order matches C format string.
        # T2 (verticality): centroids_z inserted at pos 4 (after centroid_xy);
        # is_ramp_int8 inserted at pos 8 (after bombsite_by_idx).
        # CRITICAL: positions must stay in sync with StaticDataC._fields_ in this file
        # and StaticData in cs2_types.h — mismatch silently corrupts pointer assignments.
        self._capsule = binding.init(
            vis_matrix,
            raster_grid,
            adjacency,
            centroid_xy,                                               # 0-3
            centroids_z,                                               # 4: T2 terrain z per area
            area_ids,
            bombsite_mask,
            bombsite_by_idx,
            is_ramp_int8,                                              # 8: T2 ramp (int8)
            bombsite_dist,                                             # 9
            int(md.N),
            int(md.grid.shape[1]),
            int(md.grid.shape[0]),                                     # 10-12: N, grid_w, grid_h
            int(md.area_ids.max()),                                    # 13: max_area_id
            float(md.grid_x_min),
            float(md.grid_y_min),                                      # 14-15
            float(1.0 / md.grid_cell_size),                            # 16: grid_inv_cell
            float(inv_x),
            float(inv_y),
            float(x_off),
            float(y_off),                                              # 17-20
            float(md.bombsite_dist_scale),                             # 21
            int(nav.LASER_DAMAGE),                                     # 22
            float(self._laser_range),                                  # R0-G knob
            float(self._laser_range * self._laser_range),              # 23-24
            int(nav.SHOOT_COOLDOWN),
            int(nav.BOMB_PLANT_TIME),                                  # 25-26
            int(nav.BOMB_DEFUSE_TIME),
            int(nav.BOMB_DEFUSE_KIT),                                  # 27-28
            int(nav.BOMB_TIMER),
            int(self._round_time),                                     # 29-30 (30: R0-G knob)
            float(nav.FOOTSTEP_RADIUS * nav.FOOTSTEP_RADIUS),          # 31
            float(nav.GUNSHOT_RADIUS * nav.GUNSHOT_RADIUS),            # 32
            int(nav.ENEMY_MEMORY_TICKS),
            int(nav.STALE_MEMORY_TICK),                                # 33-34
            float(pbrs_gamma),                                         # 35: pbrs_gamma
            delta_x,
            delta_y,
            dir_facing,
            t_spawns,                                                  # 36-39
            int(len(md.t_spawn_areas)),                                # 40: n_t_spawns
            ct_spawns,                                                 # 41
            int(len(md.ct_spawn_areas)),                               # 42: n_ct_spawns
            float(self._max_turn_speed),                               # 43: R0-G knob
            int(seed) & 0xFFFFFFFF,                                    # 44: seed (uint32)
            float(init_team_spirit),                                   # 45
            float(reward_win),                                         # 46
            float(reward_win_t_detonation),                            # 47: Batch 1 per-mechanism
            float(reward_win_t_elimination),                           # 48
            float(reward_win_ct_defuse),                               # 49
            float(reward_win_ct_timeout),                              # 50
            float(reward_win_ct_elimination),                          # 51
            float(reward_kill),                                        # 52
            float(reward_death),                                       # 53
            float(reward_bombsite_entry),                              # 54
            float(reward_plant_bonus),                                 # 55
            float(reward_plant_base),                                  # 56
            float(reward_plant_progress_scale),                        # 57
            float(reward_plant_interrupted),                           # 58
            float(reward_defuse),                                      # 59
            float(reward_shot_penalty),                                # 60
            float(reward_ct_survival),                                 # 61
            float(reward_inaction),                                    # 62
            float(pbrs_alive_weight),                                  # 63
            float(pbrs_hp_weight),                                     # 64
            float(pbrs_site_weight),                                   # 65
            float(pbrs_bomb_progress_weight),                          # 66
            float(pbrs_nav_weight_t),                                  # 67
            float(pbrs_nav_weight_ct),                                 # 68
            n_active_per_team,                                         # 69: Rung 0
            self.pin_pitch,                                            # 70: Rung 0
            self.crouch_enabled,                                       # 71: Rung 0
            self.jump_enabled,                                         # 72: Rung 1a
        )

        # ctypes overlay of the C-allocated Dust2Env (tests + snapshot only)
        # BindingEnv has env as first field, so capsule ptr == &env
        env_ptr = _PyCapsule_GetPointer(self._capsule, None)
        self._c_env = Dust2EnvC.from_address(env_ptr)

        # Rung 0 §2.2: every env (workers AND the parent driver_env, which is
        # never reset) proves the C side saw the same knob the trainer will mask
        # rows by. Read sd, not game.agents — __init__ never resets, so
        # `participating` is still all-zero here. RuntimeError, not assert:
        # python -O strips asserts and this guard runs outside the test suite
        # (same reason as the import-time layout guard above).
        _sc = binding.static_data_scalars(self._capsule)
        if _sc["n_active_per_team"] != n_active_per_team:
            raise RuntimeError(
                f"C StaticData.n_active_per_team is {_sc['n_active_per_team']}, expected "
                f"{n_active_per_team} — binding.init's FMT string and the call above disagree")

        # Sim recoil v1 (#120): write AFTER the overlay, not via binding.init
        # (the 73-arg FMT is a footgun; extend it only at the tail). env_init memsets
        # Dust2Env so this starts 0; env_reset memsets GameState only, so the
        # flag survives reset. Train / Modal stay off unless a later card
        # passes recoil=True into make_env.
        self._c_env.recoil_enabled = 1 if recoil else 0

        # Room AABB for ramp interpolation. Not a binding.init arg (it is a
        # pointer into a Python-owned buffer, not a scalar). make_simple_map fills area_bounds from SIMPLE_ROOMS;
        # make_cs2_map leaves None so interpolation stays centroids_z.
        # Ownership: `ab` stays in self._refs for the life of this Cs2Env and C
        # only borrows the pointer. Dropping that ref while the env is alive
        # frees the buffer under the sim; C will not (and must not) free it.
        if getattr(md, "area_bounds", None) is not None:
            ab = np.ascontiguousarray(np.asarray(md.area_bounds, dtype=np.float32).reshape(-1))
            self._refs.append(ab)
            self._c_env.sd.contents.area_bounds = ab.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Zero-copy NumPy views into C buffers
        obs_ptr, rew_ptr, term_ptr, trunc_ptr = binding.get_buffers(self._capsule)
        masks_ptr = binding.get_masks(self._capsule)
        self._masks_view = np.frombuffer(
            (ctypes.c_int8 * (N_AGENTS * ACTION_MASK_DIM)).from_address(masks_ptr),
            dtype=np.int8,
        ).reshape(N_AGENTS, ACTION_MASK_DIM)
        self._obs_view = np.frombuffer(
            (ctypes.c_float * (N_AGENTS * OBS_DIM)).from_address(obs_ptr),
            dtype=np.float32,
        ).reshape(N_AGENTS, OBS_DIM)
        self._rew_view = np.frombuffer((ctypes.c_float * N_AGENTS).from_address(rew_ptr),
                                       dtype=np.float32)
        self._term_view = np.frombuffer((ctypes.c_bool * N_AGENTS).from_address(term_ptr),
                                        dtype=np.bool_)
        self._trunc_view = np.frombuffer((ctypes.c_bool * N_AGENTS).from_address(trunc_ptr),
                                         dtype=np.bool_)

        if not self._uses_external_buffers:
            self.observations = self._obs_view
            self.rewards = self._rew_view
            self.terminals = self._term_view
            self.truncations = self._trunc_view

        self._actions_shape = (N_AGENTS, ACTION_DIM)
        self._actions_scratch = np.zeros(self._actions_shape, dtype=np.int32)
        # Batch 3: continuous-aim Δyaw scratch buffer (N_AGENTS, AIM_DIM=1) float32.
        # Reused per-step when the caller passes None (default zero) or a
        # non-contiguous / wrong-dtype array. Owning the scratch here means we
        # never allocate on the hot path; copy-into is the worst case.
        self._cont_actions_shape = (N_AGENTS, AIM_DIM)
        self._cont_actions_scratch = np.zeros(self._cont_actions_shape, dtype=np.float32)
        # Batch 3 (T5b): Optional shared-memory view onto a parent-process
        # cont-action buffer. Set by _attach_cont_action_view() when the env
        # is constructed by the Multiprocessing vecenv path; the worker reads
        # whatever the trainer wrote into the shared RawArray BEFORE step.
        # When set, _prepare_continuous_actions(None) returns this view
        # instead of the (always-zero) scratch — so MP workers actually see
        # the policy's Δyaw sample. Serial backend continues to bypass this
        # via the per-env step wrapper and a Python attr stash on the
        # vecenv (see _patch_trainer_with_hybrid_aim in src/train.py).
        self._cont_action_view = None
        # Hold the parent-process RawArray to keep it from being GC'd if the
        # caller passes it transiently (it is also kept alive on the trainer
        # side, but defensive double-anchoring is cheap).
        self._cont_action_shm_ref = None
        # F8 (2026-07-06 adversarial review): outbound action-mask view.
        # Reverse direction of the cont-action shm — the ENV writes, the
        # TRAINER reads. When _attach_mask_view() installs a view onto a
        # parent-process RawArray, step()/reset() copy the C-computed masks
        # (self._masks_view) into it so the trainer can mask sampling for the
        # obs it just received. None (default) = no copy, zero cost — legacy
        # callers and tests that read env._masks_view directly are unaffected.
        self._mask_out_view = None
        self._mask_shm_ref = None
        self._terminal_rewards = np.empty(N_AGENTS, dtype=np.float32)
        self._terminal_terminals = np.empty(N_AGENTS, dtype=bool)
        self._terminal_truncations = np.empty(N_AGENTS, dtype=bool)
        self._empty_infos = []
        self._include_step_stats_in_info = bool(include_step_stats_in_info)
        # Spec 2026-08-01 §4.3: Python-layer zero-sum transform, applied at the
        # very end of step(). No StaticDataC field and no C rebuild — the
        # binding cannot be rebuilt on this box (issue #101).
        self._reward_symmetrize = bool(reward_symmetrize)
        # Task 6a: optional per-tick step_stats view in info (see utof/cs2rl#7).
        # Zero cost when flag is off; constructed once at init when flag is on.
        # _nonterminal_infos is a pre-built list[dict] reused every non-terminal
        # step to avoid per-tick allocation — the view object is a stable proxy
        # over the ctypes struct, so the same reference is safe to return each tick.
        if self._include_step_stats_in_info:
            self._step_stats_view = StepStatsView(self._c_env.step_stats)
            self._nonterminal_infos = [{"step_stats": self._step_stats_view}]
        else:
            self._step_stats_view = None
            self._nonterminal_infos = self._empty_infos

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None):
        # seed accepted for API compatibility; C RNG is set at init time
        self._sync_team_spirit()
        binding.reset(self._capsule)
        self._sync_outputs()
        if self._mask_out_view is not None:
            # F8: env_reset recomputes masks in C; publish them so the trainer
            # masks the very first sample of the episode too.
            np.copyto(self._mask_out_view, self._masks_view)
        return self.observations, self._empty_infos

    def step(self, actions, continuous_actions=None):
        """Step the env one tick.

        Batch 3: actions is (N_AGENTS, ACTION_DIM=7) int32 — discrete heads.
        continuous_actions is (N_AGENTS, AIM_DIM=1) float32 — Δyaw rad. If
        None (legacy callers, tests, render path), a zero buffer is supplied
        so RL agents do not turn. Wrong shape raises ValueError before the
        C call to prevent OOB reads.

        Pitfall: shape strictness is critical — the C side does no bounds
        check on continuous_actions[i*AIM_DIM+0]; a (10,2) buffer would
        silently read garbage from a misaligned slot.
        """
        self._sync_team_spirit()
        actions_arr = self._prepare_actions(actions)
        cont_arr = self._prepare_continuous_actions(continuous_actions)
        binding.step(self._capsule, actions_arr, cont_arr)
        self._sync_outputs()
        infos = self._empty_infos
        rewards = self.rewards
        terminals = self.terminals
        truncations = self.truncations
        if bool(self._c_env.game.round_over):
            summary = self._build_terminal_info()
            if self._include_step_stats_in_info:
                # Task 6a: merge step_stats into the terminal summary dict so
                # consumers see both round-end stats and per-tick stats in one dict.
                summary["step_stats"] = self._step_stats_view
            infos = [summary]
            if self._auto_reset:
                if not self._uses_external_buffers:
                    np.copyto(self._terminal_rewards, self._rew_view)
                    np.copyto(self._terminal_terminals, self._term_view)
                    np.copyto(self._terminal_truncations, self._trunc_view)
                    rewards = self._terminal_rewards
                    terminals = self._terminal_terminals
                    truncations = self._terminal_truncations
                binding.reset(self._capsule)
                self._sync_observations()
        elif self._include_step_stats_in_info:
            # Task 6a: non-terminal tick — return the pre-built singleton info list
            # (no per-tick allocation; the StepStatsView proxies the live struct).
            infos = self._nonterminal_infos
        if self._mask_out_view is not None:
            # F8: publish the masks for the observation being returned. On the
            # auto-reset branch binding.reset already recomputed masks for the
            # fresh spawn state (env_reset calls compute_masks), so this copy
            # is correct in both the mid-round and the round-rollover case.
            np.copyto(self._mask_out_view, self._masks_view)
        if self._reward_symmetrize:
            # Applied LAST so it covers all four buffer paths: external buffer
            # (vecenv — `rewards` is self.rewards, already synced from C by
            # _sync_outputs above and untouched by binding.reset), the
            # non-external terminal snapshot (_terminal_rewards), and both
            # non-terminal cases. Terminal ticks carry the win/loss magnitudes,
            # so they MUST be symmetrized too (spec §4.3).
            # n_active_per_team, not TEAM_SIZE: parked rows must leave this tick
            # at exactly 0.0, and the active rows' opponent-mean subtraction must
            # not be attenuated by n/TEAM_SIZE (PITFALL 6 on the function).
            symmetrize_rewards(rewards, self.n_active_per_team)
        return self.observations, rewards, terminals, truncations, infos

    def set_team_spirit(self, value: float):
        self._c_env.team_spirit = float(value)

    def close(self):
        binding.close(self._capsule)

    @property
    def round_time(self):
        """Ticks per round as handed to C (FMT position 30).

        R0-G: reflects the ``round_time`` kwarg, not the module constant —
        trainer code that sizes horizons / stat windows off this property
        would otherwise disagree with the env when the knob is set.
        """
        return self._round_time

    def snapshot_state(self):
        g = self._c_env.game
        agents = []
        for i in range(N_AGENTS):
            agent = g.agents[i]
            agents.append(
                VizAgentState(
                    agent_id=i,
                    team=int(agent.team),
                    pos=np.array([agent.x, agent.y, agent.z], dtype=np.float32),
                    facing=float(agent.facing),
                    pitch=float(agent.pitch),                                    # Batch 3.5: 3D aim direction.
                    hp=int(agent.hp),
                    alive=bool(agent.alive),
                    has_bomb=bool(agent.has_bomb),
                    has_kit=bool(agent.has_kit),
                ))
        return VizGameState(
            agents=agents,
            bomb_planted=bool(g.bomb_planted),
            bomb_pos=np.array([g.bomb_x, g.bomb_y, g.bomb_z], dtype=np.float32),
        )

    def _sync_team_spirit(self):
        if self._team_spirit_shared is not None:
            self._c_env.team_spirit = float(self._team_spirit_shared.value)

    def _prepare_actions(self, actions):
        if (isinstance(actions, np.ndarray) and actions.dtype == np.int32
                and actions.shape == self._actions_shape and actions.flags.c_contiguous):
            return actions

        actions_arr = np.asarray(actions, dtype=np.int32)
        if actions_arr.shape != self._actions_shape:
            actions_arr = actions_arr.reshape(self._actions_shape)
        if actions_arr.flags.c_contiguous:
            return actions_arr

        np.copyto(self._actions_scratch, actions_arr, casting="no")
        return self._actions_scratch

    def _attach_cont_action_view(self, raw_shm, env_idx):
        """Attach a numpy view onto a parent-process RawArray slice (Batch 3 T5b).

        Called by the env factory when the Multiprocessing vecenv path is
        live. The trainer allocates a single ``multiprocessing.RawArray('f',
        num_envs * N_AGENTS * AIM_DIM)`` BEFORE forking workers; each worker
        receives the same raw buffer (forked file mapping, no pipe transfer)
        and we carve out this env's per-agent slice as a numpy view.

        After attach, ``_prepare_continuous_actions(None)`` returns this view
        instead of the all-zero scratch — so the trainer can write Δyaw into
        the shm slice on the main process and the worker's ``Cs2Env.step``
        will see it on the very next tick.

        Pitfall: ``raw_shm`` MUST have at least
        ``(env_idx + 1) * N_AGENTS * AIM_DIM`` float32 slots. We do NOT
        revalidate the global length here (we don't know num_envs); a too-
        small allocation will manifest as np.frombuffer raising or as
        out-of-range data. The caller (env factory in src/train.py) owns
        the sizing.
        """
        if raw_shm is None:
            return
        per_env = N_AGENTS * AIM_DIM
        # ctypes float = 4 bytes; np.frombuffer with offset/count addresses
        # the slice without copying. np.frombuffer DOES set the resulting
        # array's .base to raw_shm (so the buffer is technically reachable),
        # but env_factory's local reference to raw_shm goes out of scope
        # right after this method returns. Keeping _cont_action_shm_ref as
        # an explicit anchor on self protects against future code that might
        # replace _cont_action_view (e.g. with a reshape or a slice) and
        # accidentally flatten the .base chain — at which point the OS could
        # reclaim the mapping and the next read would be UB.
        flat = np.frombuffer(raw_shm, dtype=np.float32, count=per_env, offset=env_idx * per_env * 4)
        self._cont_action_view = flat.reshape(self._cont_actions_shape)
        self._cont_action_shm_ref = raw_shm

    def _attach_mask_view(self, raw_shm, env_idx):
        """Attach the outbound action-mask shm slice (F8, env→trainer direction).

        Mirror image of ``_attach_cont_action_view``: the trainer allocates one
        ``multiprocessing.RawArray('b', num_envs * N_AGENTS * ACTION_MASK_DIM)``
        BEFORE forking workers; each env carves out its (N_AGENTS,
        ACTION_MASK_DIM) int8 slice and copies the C-side masks into it at the
        end of every step()/reset(). The trainer's main-process view over the
        same bytes is read right after vecenv.recv() — by which point the
        worker has finished its step, so the masks always correspond to the
        observation batch just received (recv is the synchronisation point).

        Pitfall: sizing is the caller's job, exactly as for the cont-action
        attach — ``raw_shm`` must hold at least (env_idx+1)*N_AGENTS*
        ACTION_MASK_DIM bytes.
        """
        if raw_shm is None:
            return
        per_env = N_AGENTS * ACTION_MASK_DIM
        flat = np.frombuffer(raw_shm, dtype=np.int8, count=per_env, offset=env_idx * per_env)
        self._mask_out_view = flat.reshape(N_AGENTS, ACTION_MASK_DIM)
        self._mask_shm_ref = raw_shm

    def _prepare_continuous_actions(self, cont):
        """Coerce caller-supplied Δyaw buffer to (N_AGENTS, AIM_DIM) float32 contiguous.

        Batch 3: shape mismatch ALWAYS raises (silent reshape would conceal a
        bug given AIM_DIM=1 — e.g. a (10,) accidentally passed as (1,10) would
        slip through). dtype mismatch is forgiven via cast. None → cached
        zero scratch (RL training path before the policy is wired uses this),
        OR if ``_attach_cont_action_view`` has installed a shared-memory
        view (Batch 3 T5b — Multiprocessing vecenv path), that view is
        returned instead so workers actually consume the trainer's Δyaw.
        """
        if cont is None:
            if self._cont_action_view is not None:
                # MP vecenv path: trainer wrote into the parent-process shm;
                # the view is the workers' window onto the same physical
                # bytes. Returning the view directly means the C binding
                # reads the live data on this tick.
                return self._cont_action_view
            # Reuse the pre-zeroed scratch — zeroing every step would be wasteful.
            # Tests that mutate this scratch via env.step(...) MUST pass an
            # explicit buffer; the scratch is treated as read-only zeroes here.
            return self._cont_actions_scratch
        if not isinstance(cont, np.ndarray):
            cont = np.asarray(cont, dtype=np.float32)
        if cont.shape != self._cont_actions_shape:
            raise ValueError(f"continuous_actions shape {cont.shape} != "
                             f"{self._cont_actions_shape} (expected (N_AGENTS, AIM_DIM))")
        if cont.dtype == np.float32 and cont.flags.c_contiguous:
            return cont
        # Fall through: cast/copy into scratch. casting="unsafe" allows
        # float64 → float32 (RL policies sometimes emit float32 already, but
        # numpy generic dtypes from `np.zeros((10,1))` default to float64).
        np.copyto(self._cont_actions_scratch, cont.astype(np.float32, copy=False), casting="unsafe")
        return self._cont_actions_scratch

    def _sync_outputs(self):
        if not self._uses_external_buffers:
            return
        self._sync_observations()
        np.copyto(self.rewards, self._rew_view)
        np.copyto(self.terminals, self._term_view)
        np.copyto(self.truncations, self._trunc_view)

    def _sync_observations(self):
        if self._uses_external_buffers:
            np.copyto(self.observations, self._obs_view)

    def _build_terminal_info(self):
        stats = self._c_env.episode_stats
        summary = {
            "bomb_planted": int(stats.bomb_planted),
            "bomb_defused": int(stats.bomb_defused),
            "kills_t": int(stats.kills_t),
            "kills_ct": int(stats.kills_ct),
            "blocked_moves_t": int(stats.blocked_moves_t),
            "blocked_moves_ct": int(stats.blocked_moves_ct),
        }
        for idx in range(9):
            summary[f"action_move_{idx}"] = int(stats.action_move[idx])
        for idx in range(2):
            summary[f"action_shoot_{idx}"] = int(stats.action_shoot[idx])
            summary[f"action_use_{idx}"] = int(stats.action_use[idx])
            # action_last_* removed (F13): the counter had no writer and no
            # corresponding action head — it exported permanent zeros.
        # Batch 3: continuous-aim stats — emit Welford triple instead of
        # 16-bin histogram. Consumers that previously summed action_aim_*
        # to derive total turns should now use aim_delta_count; mean/var
        # via standard formulas. mean = sum / count;
        # var = sq_sum / count - mean².
        summary["aim_delta_sum"] = float(stats.aim_delta_sum)
        summary["aim_delta_sq_sum"] = float(stats.aim_delta_sq_sum)
        summary["aim_delta_count"] = int(stats.aim_delta_count)
        # Batch 3.5 (#24): pitch Welford triple (mirrors yaw fields above).
        # Consumers compute mean/var/std the same way: mean = sum / count;
        # var = sq_sum / count - mean²; std = sqrt(max(0, var)).
        # Pitch_log_std non-collapse is the spec's load-bearing acceptance
        # signal — diagnostics need their own surface.
        summary["aim_delta_pitch_sum"] = float(stats.aim_delta_pitch_sum)
        summary["aim_delta_pitch_sq_sum"] = float(stats.aim_delta_pitch_sq_sum)
        summary["aim_delta_pitch_count"] = int(stats.aim_delta_pitch_count)
        for idx in range(2):
            summary[f"action_reload_{idx}"] = int(stats.action_reload[idx])
        for idx in range(3):
            summary[f"action_weapon_{idx}"] = int(stats.action_weapon[idx])
        for idx in range(2):
            summary[f"action_crouch_{idx}"] = int(stats.action_crouch[idx])
        for idx in range(2):
            summary[f"action_jump_{idx}"] = int(stats.action_jump[idx])
        summary.update({
            "winner": int(stats.winner),
            "winner_t": int(stats.winner_t),
            "winner_ct": int(stats.winner_ct),
            "timed_out": int(stats.timed_out),
            "alive_t_end": int(stats.alive_t_end),
            "alive_ct_end": int(stats.alive_ct_end),
            "round_length": int(stats.round_length),
                                                                                         # plant_tick is the observe-only C field (0 = never planted).
                                                                                         # Win-type flags are the existing episode_stats ints; export them
                                                                                         # here so compute_game_metrics can re-key rates without turning
                                                                                         # on include_step_stats_in_info or merging per-tick step_stats.
            "plant_tick": int(stats.plant_tick),
            "win_by_detonation": int(stats.win_by_detonation),
            "win_by_defuse": int(stats.win_by_defuse),
        })
        summary.update({
            "reward_win": float(stats.reward_win),
            "reward_kills": float(stats.reward_kills),
            "reward_deaths": float(stats.reward_deaths),
            "reward_bomb": float(stats.reward_bomb),
            "reward_pbrs": float(stats.reward_pbrs),
            "reward_shots": float(stats.reward_shots),
            "reward_survival": float(stats.reward_survival),
            "reward_inaction": float(stats.reward_inaction),
        })
                                                                                         # R0-A combat instrumentation. min_enemy_distance is converted HERE,
                                                                                         # per episode: PufferLib's mean_and_log averages the window before
                                                                                         # compute_game_metrics sees it, so one 1e30 sentinel in a 40-episode
                                                                                         # window would average to ~2.5e28. Emit (sum, valid) and let
                                                                                         # compute_game_metrics divide the two window means.
        _med = float(stats.min_enemy_distance)
        _valid = 1 if _med < 1e29 else 0
        summary.update({
            "shots_fired": int(stats.shots_fired),
            "shots_with_enemy_in_los": int(stats.shots_with_enemy_in_los),
            "shots_facing_enemy": int(stats.shots_facing_enemy),
            "shots_on_target": int(stats.shots_on_target),
            "shots_hit": int(stats.shots_hit),
            "shots_stance_blocked": int(stats.shots_stance_blocked),
            "damage_dealt": float(stats.damage_dealt),
            "mutual_vis_pair_ticks": int(stats.mutual_vis_pair_ticks),
            "agent_ticks_with_visible_enemy": int(stats.agent_ticks_with_visible_enemy),
            "min_enemy_distance_sum": _med if _valid else 0.0,
            "min_enemy_distance_valid": _valid,
            "reward_win_t": float(stats.reward_win_t),
            "reward_win_ct": float(stats.reward_win_ct),
        })
        return summary


# ── make_env ──────────────────────────────────────────────────────────────────


def make_env(
        seed=0,
        team_spirit=0.0,
        auto_reset=True,
        buf=None,
        map_data=None,
        reward_win=1.0,
        reward_kill=0.3,
        reward_death=0.1,
        reward_bombsite_entry=0.3,
        reward_plant_bonus=3.0,
        reward_plant_base=0.2,
        reward_plant_progress_scale=0.05,
        reward_plant_interrupted=0.1,
        reward_defuse=0.2,
        reward_shot_penalty=0.005,
        reward_ct_survival=0.001,
        reward_inaction=0.0005,
        pbrs_alive_weight=0.3,
        pbrs_hp_weight=0.002,
        pbrs_site_weight=0.2,
        pbrs_bomb_progress_weight=0.3,
        pbrs_nav_weight_t=0.04,
        pbrs_nav_weight_ct=0.15,
        pbrs_gamma=0.999,                                              # must equal training gamma — see Cs2Env.__init__ note
                                                                       # Batch 1 (RL overhaul): per-outcome win magnitudes (Task 3).  # noqa: E501
    reward_win_t_detonation=5.0,
        reward_win_t_elimination=3.0,
        reward_win_ct_defuse=5.0,
        reward_win_ct_timeout=4.0,
        reward_win_ct_elimination=3.0,
        include_step_stats_in_info: bool = False,                      # Task 6a (utof/cs2rl#7)
        reward_symmetrize: bool = False,                               # spec 2026-08-01 §4.3
        recoil: bool = False,                                          # #120: punch on ray; default off
        n_active_per_team: int = TEAM_SIZE,                            # Rung 0 §2.1: agents per team that spawn
        pin_pitch: int = 0,                                            # Rung 0 R0-E.2: ignore pitch action
        crouch_enabled: int = 1,                                       # Rung 0 R0-E.2: mask crouch when 0
        jump_enabled: int = 1,                                         # Rung 1a T2a: mask jump when 0
        round_time: int | None = None,                                 # Rung 0 R0-G: None ⇒ nav.ROUND_TIME
        laser_range: float | None = None,                              # Rung 0 R0-G: None ⇒ nav.LASER_RANGE
        max_turn_speed: float | None = None,                           # Rung 0 R0-G: None ⇒ nav.MAX_TURN_SPEED_RAD
):
    """Load map data and return a ready-to-use Cs2Env."""
    if map_data is None:
        key = (nav.NAV_PATH, nav.CACHE_PATH)
        md = _ENV_CACHE.get(key)
        if md is None:
            md = make_cs2_map(nav.NAV_PATH, nav.CACHE_PATH)
            _ENV_CACHE[key] = md
    else:
        md = map_data
    return Cs2Env(
        seed=seed,
        team_spirit=team_spirit,
        buf=buf,
        nav_graph=md.nav_graph,
        auto_reset=auto_reset,
        map_data=md,
        reward_win=reward_win,
        reward_kill=reward_kill,
        reward_death=reward_death,
        reward_bombsite_entry=reward_bombsite_entry,
        reward_plant_bonus=reward_plant_bonus,
        reward_plant_base=reward_plant_base,
        reward_plant_progress_scale=reward_plant_progress_scale,
        reward_plant_interrupted=reward_plant_interrupted,
        reward_defuse=reward_defuse,
        reward_shot_penalty=reward_shot_penalty,
        reward_ct_survival=reward_ct_survival,
        reward_inaction=reward_inaction,
        pbrs_alive_weight=pbrs_alive_weight,
        pbrs_hp_weight=pbrs_hp_weight,
        pbrs_site_weight=pbrs_site_weight,
        pbrs_bomb_progress_weight=pbrs_bomb_progress_weight,
        pbrs_nav_weight_t=pbrs_nav_weight_t,
        pbrs_nav_weight_ct=pbrs_nav_weight_ct,
        pbrs_gamma=pbrs_gamma,
        reward_win_t_detonation=reward_win_t_detonation,
        reward_win_t_elimination=reward_win_t_elimination,
        reward_win_ct_defuse=reward_win_ct_defuse,
        reward_win_ct_timeout=reward_win_ct_timeout,
        reward_win_ct_elimination=reward_win_ct_elimination,
        include_step_stats_in_info=include_step_stats_in_info,
        reward_symmetrize=reward_symmetrize,
        recoil=recoil,
        n_active_per_team=n_active_per_team,
        pin_pitch=pin_pitch,
        crouch_enabled=crouch_enabled,
        jump_enabled=jump_enabled,
        round_time=round_time,
        laser_range=laser_range,
        max_turn_speed=max_turn_speed,
    )
