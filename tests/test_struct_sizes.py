"""binding.struct_sizes() / binding.static_data_scalars() — the layout oracles.

WHY: cs2_env.py used to carry hand-measured sizeof literals (164/1708/204/6832)
and hand-measured offsetof literals (476/480/496) with a comment forbidding
"invented" pads. Every field appended to a C struct made those literals rot, and
the only way to refresh them was to compile a throwaway printf TU by hand.
struct_sizes() asks the same compiler that laid the structs out, so the ctypes
mirrors in cs2_env.py are pinned to the C headers automatically.

sizeof alone cannot catch a value packed under the *wrong field name* in
Cs2Env.__init__ — swapping two floats keeps every size identical while silently
feeding reward_kill into reward_death. static_data_scalars() closes that hole by
reading scalar StaticData fields back out of a live env, so a test can push
distinct sentinels through Cs2Env and check where they landed.

PITFALL: every struct change on this branch must extend BOTH dicts, or the
new field is unguarded.
"""

import ctypes
import dataclasses
import sys
from pathlib import Path

import pytest

# Repo convention (same two sys.path.insert lines at the top of
# tests/test_binding.py): `binding` is a C extension living in src/c_env, so
# that directory must be on sys.path before the import. conftest.py only adds
# src/. Anchored by name, not by line number — line anchors rot.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "c_env"))
# I001 is suppressed, not fixed: yapf snaps these trailing `noqa` comments to its
# spaces_before_comment stops while ruff's isort wants one space, and the two then
# fight forever (gh#97; same waiver as tests/test_env_config.py:21).
import binding                         # noqa: E402, I001

from c_env.cs2_env import (                                                                   # noqa: E402
    _C_SIZE_KEYS_CHECKED, AgentStateC, Dust2EnvC, GameStateC, StaticDataC, StepStatsC, WallC,
    WallListC, make_env,
)
from env_config import (                                                                      # noqa: E402
    KNOB_FIELDS, REWARD_FIELDS, EnvConfig, RewardWeights,
)


def _is_scalar_ctype(ctype):
    """Is `ctype` a plain number that static_data_scalars() can return?

    Defined by EXCLUSION, and that is the whole point. This used to be an
    allowlist — `(c_int, c_int32, c_float)` — which fails OPEN: the first
    StaticData field declared c_int8 / c_uint8 / c_uint32 / c_double would be
    silently absent from the `expected` set in
    test_static_data_scalars_covers_every_scalar_field, so that test would stop
    demanding it and the field would ship unguarded. That is exactly the decay
    this module exists to stop, and it is not hypothetical: Dust2Env already
    uses int32_t for a flag and AgentState uses int8/uint8 flags heavily.
    Enumerating what is EXCLUDED cannot fail that way — anything not matching
    one of the three aggregate shapes is demanded, whether or not anyone
    anticipated the type.

    Excluded, because none of them is expressible as a single number:
      - ctypes._Pointer subclasses — POINTER(c_int8), POINTER(c_float), ...
        The leading underscore is unavoidable: it is the only structural test
        for "is a pointer". `hasattr(t, "_type_")` is NOT a substitute, because
        simple types carry _type_ too (ctypes.c_float._type_ == "f").
      - ctypes.Array subclasses — c_float * 9, c_int32 * 15, ...
      - ctypes.Structure subclasses — the nested WallListC.

    Deliberately NOT excluded: everything else, including a hypothetical
    ctypes.Union field. Such a field would be *demanded* and the test would fail
    loudly, forcing a conscious decision here — the correct failure direction
    for a guard whose job is to refuse to be forgotten.
    """
    return not issubclass(ctype, (ctypes._Pointer, ctypes.Array, ctypes.Structure))


# Every scalar StaticData field, derived from the mirror (which the sizeof /
# offsetof asserts pin to the C header). Never re-listed by hand: a hand-written
# copy is the next thing to rot.
_STATIC_DATA_SCALARS = tuple(
    (name, ctype) for name, ctype in StaticDataC._fields_ if _is_scalar_ctype(ctype))

# Partition those scalars by whether the env CONFIG can set them. Anything the
# config exposes is sentinel-testable (test_static_data_scalars_round_trip
# pushes a distinct value through it); anything it does not is map-derived or a
# nav.py constant and is checked against that source instead.
#
# The settable surface is the CONFIG surface, not make_env's signature: after
# spec 2026-09-03 the weights and knobs sit behind make_env's **legacy, and
# inspect.signature would see eight names (none a weight), empty these tuples,
# and let the sweep below pass while iterating nothing. Derive from the
# dataclass fields instead and pin the census so an emptied partition fails on
# the count before the values. KNOB_FIELDS is itself derived from
# fields(EnvConfig), so a knob added to the dataclass arrives here for free.
_CONFIG_NAMES = frozenset(f.name
                          for f in dataclasses.fields(RewardWeights)) | frozenset(KNOB_FIELDS)
_FLOAT_KWARG_SCALARS = tuple(
    sorted(n for n, t in _STATIC_DATA_SCALARS if n in _CONFIG_NAMES and t is ctypes.c_float))
_INT_KWARG_SCALARS = tuple(
    sorted(n for n, t in _STATIC_DATA_SCALARS if n in _CONFIG_NAMES and t is not ctypes.c_float))
_NON_KWARG_SCALARS = tuple(sorted(n for n, _ in _STATIC_DATA_SCALARS if n not in _CONFIG_NAMES))


def test_sentinel_partition_census_is_pinned():
    """26 float kwarg scalars (23 weights + pbrs_gamma, laser_range, max_turn_speed),
    5 int (n_active_per_team, pin_pitch, crouch_enabled, jump_enabled, round_time),
    25 non-kwarg. reward_symmetrize and recoil have no StaticData slot and belong
    to neither kwarg tuple. Knock-out: delete one RewardWeights field and the
    first assert fires — the float count drops to 25 (by construction the
    non-kwarg count then rises to 26, but that assert is never reached)."""
    assert len(_FLOAT_KWARG_SCALARS) == 26, _FLOAT_KWARG_SCALARS
    assert len(_INT_KWARG_SCALARS) == 5, _INT_KWARG_SCALARS
    assert len(_NON_KWARG_SCALARS) == 25, _NON_KWARG_SCALARS
    assert set(_INT_KWARG_SCALARS) == {
        "n_active_per_team", "pin_pitch", "crouch_enabled", "jump_enabled", "round_time"
    }


# Int kwargs whose C field REJECTS the generic 101, 102, ... run below. There
# are two shapes and they need different treatment:
#
#   - RANGE-constrained. n_active_per_team is validated to 1..TEAM_SIZE (Cs2Env
#     raises ValueError), but any in-range value round-trips unchanged, so ONE
#     value serves every env config. Those live here.
#   - FLAG-constrained. pin_pitch / crouch_enabled / jump_enabled are
#     normalised by Cs2Env with int(bool(...)), so 101 would come back as 1 and
#     the round trip would fail on a perfectly healthy build. Only 0 and 1
#     survive, which means three flags CANNOT get three distinct values inside a
#     single env — see _BOOL_SENTINEL_CONFIGS below for what replaces that.
#
# Overriding beats excluding them: an excluded field is an UNGUARDED packing
# assignment, which is the exact hole this module exists to close.
#
# ADDING AN INT KWARG: if it accepts arbitrary ints, add nothing — the fallback
# covers it automatically. If it is range-constrained, add it here; if it is a
# 0/1 flag, add it to EVERY dict in _BOOL_SENTINEL_CONFIGS.
# test_int_sentinels_are_usable is what tells you which case you are in.
_INT_SENTINEL_OVERRIDES = {
    "n_active_per_team": 3,
}

# Flag kwargs, one assignment per env config. The round trip builds one env per
# dict and reads every field back from both, so a flag is identified not by a
# single value (impossible — there are only two) but by its VECTOR of read-back
# values across the configs. Those vectors must be pairwise DISTINCT, which is
# what keeps a transposition between two flags visible: a swap then changes at
# least one config's read-back.
#
# WHAT A TRANSPOSITION LOOKS LIKE NOW (spec 2026-08-31 §2 W2): it used to be two
# swapped positions in py_init's PyArg_ParseTuple format string. That string is
# gone; Python packs StaticData by name instead, so the same bug is now two
# swapped NAMED PACKING ASSIGNMENTS in Cs2Env.__init__ — "crouch_enabled":
# self.jump_enabled. The layout hash that replaced the format string cannot see
# it, because it compares DECLARATIONS and this puts a wrong value into a
# correctly described slot. This scheme is still the only thing that catches it,
# which is why W2 retired none of it.
#
# The vectors are deliberately NOT complementary. Complementarity would force
# every flag into {(0,1), (1,0)} and, by pigeonhole, two of the three would
# collide — that pair could then be swapped invisibly. With
# crouch = (0,0), pin_pitch = (0,1), jump = (1,0) all three are distinct.
#
# Two configs address 2² = 4 vectors, so this scheme holds up to FOUR flags; a
# fifth needs a third dict here, which costs one more env construction in
# test_static_data_scalars_round_trip and nothing else.
_BOOL_SENTINEL_CONFIGS = (
    {
        "pin_pitch": 0,
        "crouch_enabled": 0,
        "jump_enabled": 1
    },
    {
        "pin_pitch": 1,
        "crouch_enabled": 0,
        "jump_enabled": 0
    },
)
_BOOL_KWARG_SCALARS = tuple(sorted({name for cfg in _BOOL_SENTINEL_CONFIGS for name in cfg}))

# One distinct sentinel per settable field, generated from the sorted field list
# so a newly added field automatically gets one. Distinctness is the only
# property that matters: it is what makes a swapped pair of named packing
# assignments visible. Floats get 0.101, 0.102, ... — not exactly representable in float32,
# hence pytest.approx on the way back (see the widening PITFALL in binding.c).
# Ints get 101, 102, ... unless _INT_SENTINEL_OVERRIDES names them.
# Do NOT sentinel with the defaults instead: several collide
# (reward_win_t_detonation and reward_win_ct_defuse are both 5.0, reward_death
# and reward_plant_interrupted both 0.1), so a transposition between a colliding
# pair would stay invisible.
#
# _SENTINELS_BASE is everything whose sentinel is the same in every config (all
# floats, plus the non-flag ints); the enumerate() index still runs over the
# FULL sorted int list so skipping the flags cannot make two fallbacks collide.
# _SENTINEL_CONFIGS layers one flag assignment on top of that base per env.
_SENTINELS_BASE = {name: 0.101 + 0.001 * i for i, name in enumerate(_FLOAT_KWARG_SCALARS)}
_SENTINELS_BASE.update({
    name: _INT_SENTINEL_OVERRIDES.get(name, 101 + i)
    for i, name in enumerate(_INT_KWARG_SCALARS) if name not in _BOOL_KWARG_SCALARS
})
_SENTINEL_CONFIGS = tuple({**_SENTINELS_BASE, **flags} for flags in _BOOL_SENTINEL_CONFIGS)


def _config_from_field_kwargs(kwargs):
    kw = dict(kwargs)
    weights = {name: kw.pop(name) for name in REWARD_FIELDS if name in kw}
    return EnvConfig(rewards=RewardWeights(**weights), **kw)


def test_config_from_field_kwargs_partitions_flat_reward_and_knob_names():
    from env_config import EnvConfig, RewardWeights
    payload = {"reward_kill": 1.0, "n_active_per_team": 3}
    cfg = _config_from_field_kwargs(payload)
    assert payload == {"reward_kill": 1.0, "n_active_per_team": 3}
    assert isinstance(cfg, EnvConfig)
    assert cfg.rewards == RewardWeights(reward_kill=1.0)
    assert cfg.n_active_per_team == 3
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'reward_kil'"):
        _config_from_field_kwargs({"reward_kil": 1.0})


# Per-int-field vector of sentinels across the configs — the object whose
# pairwise distinctness test_int_sentinels_are_usable asserts. Non-flag ints get
# a constant vector (3, 3) / (101, 101) / ...; flags get the 2-vectors above.
# None marks a flag named by SOME but not all of _BOOL_SENTINEL_CONFIGS: it
# would be left at its make_env default in the config that omits it, silently
# weakening the guard. Recorded rather than raised so the diagnosis lands in
# test_int_sentinels_are_usable instead of as a KeyError during collection.
_INT_SENTINEL_VECTORS = {
    name: tuple(cfg.get(name) for cfg in _SENTINEL_CONFIGS)
    for name in _INT_KWARG_SCALARS
}

# StaticData scalars that are tunables — the ones a training config sweeps.
# Every one must stay reachable from make_env or it drops out of the sentinel
# sweep above; test_every_tunable_scalar_is_a_make_env_kwarg enforces that.
_TUNABLE_PREFIXES = ("reward_", "pbrs_")


def test_struct_sizes_match_ctypes_mirrors():
    """Every ctypes mirror in cs2_env.py must agree with the C compiler.

    If this fails, the mirror drifted from cs2_types.h — fix the mirror, never
    the assert.

    Honest scope: this is a DUPLICATE of the `_C_SIZES` assert block cs2_env.py
    runs at import time, comparing the same oracle to the same mirror. It
    therefore passes and fails in exactly the same conditions, and on a drifting
    tree this module fails during collection (the import above) before this test
    ever reports. It cannot detect a dishonest struct_sizes(); nothing here can,
    because there is no second, independent measurement of the C layout. Its
    value is defense-in-depth: if the cs2_env.py block is ever deleted, weakened,
    or bypassed (e.g. `python -O`, which strips asserts), this keeps the
    comparison running inside the test suite. Real independent checks live in
    test_struct_sizes_exposes_team_constants (literal 5 / 2*TEAM_SIZE) and
    test_static_data_scalars_round_trip (sentinels through the packed buffer).
    The complementary "is every mirror even guarded" question is answered by
    test_every_ctypes_mirror_is_size_guarded.
    """
    sizes = binding.struct_sizes()
    assert sizes["AgentState"] == ctypes.sizeof(AgentStateC)
    assert sizes["GameState"] == ctypes.sizeof(GameStateC)
    assert sizes["StepStats"] == ctypes.sizeof(StepStatsC)
    assert sizes["Dust2Env"] == ctypes.sizeof(Dust2EnvC)
    assert sizes["StaticData"] == ctypes.sizeof(StaticDataC)
    assert sizes["Wall"] == ctypes.sizeof(WallC)
    assert sizes["WallList"] == ctypes.sizeof(WallListC)


def test_struct_offsets_match_ctypes_mirrors():
    """StaticData's tail offsets are the load-bearing ones.

    wall_list / area_bounds sit after a long run of float reward weights;
    inserting a field before them shifts both and silently repoints every
    Python-side walls[i] read. cs2_env.py asserts these at import time against
    the C offsetof instead of the literals 476/480/496 it used to carry.

    Same honest scope as test_struct_sizes_match_ctypes_mirrors above: this
    duplicates the import-time `_C_SIZES` offset loop rather than checking it
    independently, and exists so the comparison survives that block being
    weakened or removed.
    """
    sizes = binding.struct_sizes()
    assert sizes["StaticData_pbrs_nav_weight_ct_offset"] == StaticDataC.pbrs_nav_weight_ct.offset
    assert sizes["StaticData_wall_list_offset"] == StaticDataC.wall_list.offset
    assert sizes["StaticData_area_bounds_offset"] == StaticDataC.area_bounds.offset


def test_struct_sizes_exposes_team_constants():
    """TEAM_SIZE/N_AGENTS are duplicated in nav.py; pin them to the C macros.

    Parked-agent work reduces the *effective* team size without changing the C
    macro, so a drift between nav.TEAM_SIZE and the header would mis-slice every
    per-team reward view in cs2_env.py.
    """
    from nav import N_AGENTS, TEAM_SIZE
    sizes = binding.struct_sizes()
    assert sizes["TEAM_SIZE"] == 5
    assert sizes["TEAM_SIZE"] == TEAM_SIZE
    assert sizes["N_AGENTS"] == N_AGENTS == 2 * TEAM_SIZE


def test_static_data_scalars_round_trip(simple_map):
    """Distinct sentinels in → same sentinels out, per named field.

    This is the VALUE-ROUTING guard. Cs2Env.__init__ names every StaticData
    field it packs, and writing the wrong value under a correct name is
    invisible to sizeof, to the offset anchors, and to the layout hash — all
    three describe the struct, not the values put into it. Two swapped floats
    keep every size, offset and type identical while feeding reward_kill into
    reward_death. (Before spec 2026-08-31 §2 W2 the same bug wore a different
    costume: two swapped positions in py_init's 73-arg PyArg_ParseTuple format
    string. The costume changed; the exposure did not, which is why this test
    survived the migration unchanged.)

    EXHAUSTIVE over the settable fields. Every StaticData scalar make_env
    exposes as a keyword argument (_SENTINEL_CONFIGS, derived by intersecting
    the EnvConfig/RewardWeights field names with the mirror — not hand-listed)
    gets its own distinct value and is asserted back by name. An earlier version
    checked 4 of the 24 and defended that as "the fields with a distinguishable
    expected value"; that premise was wrong — make_env exposes all of them — and it left
    20 of the 23 consecutive same-width reward/PBRS floats, the exact run the
    code itself calls the hiding place for a transposition, unchecked.

    TWO ENVS, one per entry in _BOOL_SENTINEL_CONFIGS. The 0/1 flag kwargs
    (pin_pitch / crouch_enabled / jump_enabled) cannot all hold distinct values
    at once — int(bool(...)) leaves two, and there are three flags — so a
    single-env sweep would have to leave one of them unguarded, which is the
    hole this module exists to close. Instead each flag is pinned by its vector
    of read-back values ACROSS the configs; those vectors are pairwise distinct
    (asserted in test_int_sentinels_are_usable), so any transposition between
    two flag positions still shows up in at least one config. Everything else
    (all floats, the non-flag ints) carries the same sentinel in both configs
    and is simply checked twice.

    Fields make_env does NOT expose are checked against their real source
    instead: laser_range_sq against the laser_range sentinel it is derived
    from (round_time / laser_range / max_turn_speed became kwargs in R0-G and
    are sentinel-checked like the rest) and the map-derived ones
    (bombsite_dist_scale, N, grid_w/grid_h, max_area_id, spawn counts) against
    the fixture map. grid_w/grid_h are
    adjacent same-width ints, so only a value comparison separates them.

    REMAINING GAP, stated plainly. These 16 non-kwarg scalars (plus the derived
    laser_range_sq, checked above) are covered only
    by key presence (test_static_data_scalars_covers_every_scalar_field), not by
    value: grid_x_min, grid_y_min, grid_inv_cell, inv_x_range, inv_y_range,
    x_offset, y_offset, shoot_cooldown, bomb_plant_time, bomb_defuse_time,
    bomb_defuse_kit, bomb_timer, footstep_radius_sq, gunshot_radius_sq,
    enemy_memory_ticks, stale_memory_tick. They are not settable, so a value
    check would have to recompute the implementation's own formula (the
    geometry) or restate a nav.py constant (the timings) — weaker than a
    sentinel, and for the geometry partly degenerate, since a symmetric fixture
    map can make x_offset == y_offset. (nav.BOMB_TIMER == nav.ROUND_TIME == 640
    at defaults; the round_time sentinel now separates the two.) Closing this
    properly means sentinels, which means kwargs; out of scope here, and
    deliberately not papered over.
    """
    import nav

    # Sequential, not two live envs at once: nothing here needs them to coexist,
    # and one env at a time keeps a failure attributable to a single config.
    for cfg_i, sentinels in enumerate(_SENTINEL_CONFIGS):
        env = make_env(map_data=simple_map, config=_config_from_field_kwargs(sentinels))
        try:
            sc = binding.static_data_scalars(env._capsule)
            # ── every settable field, one sentinel each, in this config ──
            absent = sorted(name for name in sentinels if name not in sc)
            assert not absent, (
                f"config {cfg_i}: make_env kwargs with no static_data_scalars() key: {absent}; add "
                "SD_INT/SD_FLOAT for them in src/c_env/binding.c and rebuild (see "
                "test_static_data_scalars_covers_every_scalar_field)")
            wrong = {}
            for name, sent in sentinels.items():
                if sc[name] != pytest.approx(sent):
                    # Which sentinel DID land here? For a swapped pair of packing
                    # assignments that names the swap partner outright, which is
                    # the whole diagnosis. Flags share values within a config, so
                    # the partner is a hint there, not a unique identification —
                    # the config INDEX is the other half of the diagnosis.
                    partner = next(
                        (other for other, v in sentinels.items() if sc[name] == pytest.approx(v)),
                        None)
                    wrong[name] = (sent, sc[name], partner)
            assert not wrong, (
                f"config {cfg_i} ({_BOOL_SENTINEL_CONFIGS[cfg_i]}): sentinel landed in the wrong "
                "StaticData field — two of the named assignments in the `static_data` mapping in "
                "Cs2Env.__init__ (src/c_env/cs2_env.py) carry each other's values. "
                f"{{field: (sent, got, whose_sentinel_got_is)}} = {wrong}")
            # R0-G (Task 11): round_time / laser_range / max_turn_speed are now
            # make_env kwargs, so they are in the config and were checked above.
            # laser_range_sq is NOT a kwarg — it is derived from the laser_range
            # sentinel inside Cs2Env.__init__, so check the derivation rather than
            # a nav constant. The None ⇒ nav.py default path is covered by
            # tests/test_env_knobs.py::test_default_knobs_match_nav_constants.
            assert sc["laser_range_sq"] == pytest.approx(sentinels["laser_range"]**2)
            assert sc["laser_damage"] == nav.LASER_DAMAGE
            # Map-derived scalars. These are the only non-reward, non-nav values in
            # StaticData, and bombsite_dist_scale is wedged between y_offset
            # (float) and laser_damage (int), exactly where a transposition
            # would hide. `> 0.0` used to be the check here; any positive garbage
            # passed it. Cs2Env forwards md.bombsite_dist_scale verbatim, so compare
            # against the map. float() round-trips through C float32, hence approx.
            assert sc["bombsite_dist_scale"] == pytest.approx(simple_map.bombsite_dist_scale)
            assert sc["N"] == simple_map.N
            assert sc["grid_w"] == simple_map.grid.shape[1]
            assert sc["grid_h"] == simple_map.grid.shape[0]
            assert sc["max_area_id"] == int(simple_map.area_ids.max())
            assert sc["n_t_spawns"] == len(simple_map.t_spawn_areas)
            assert sc["n_ct_spawns"] == len(simple_map.ct_spawn_areas)
        finally:
            env.close()


def test_static_data_scalars_rejects_non_capsule():
    """A wrong argument must raise, not read a wild pointer.

    py_static_data_scalars casts the capsule to BindingEnv* and dereferences
    ->sd; handing it an int (a common copy/paste from get_buffers' int return)
    would otherwise segfault the whole test session.
    """
    with pytest.raises((ValueError, TypeError)):
        binding.static_data_scalars(12345)


def test_struct_sizes_keys_are_all_consumed():
    """Every key struct_sizes() publishes must actually be compared somewhere.

    WHY: the cs2_env.py guard loops over hand-written Python-side tuples. A key
    added to py_struct_sizes() in binding.c without a matching entry there is
    published and then never compared — the struct looks guarded (it has a key!)
    while nothing checks it. That is invisible in review and stays broken until a
    layout drift corrupts memory at runtime.

    _C_SIZE_KEYS_CHECKED is derived from those tuples, not re-listed by hand, so
    this equality is a real check on binding.c and not a copy compared to itself.

    If this fails after you added a key: consume it in the matching tuple in
    cs2_env.py (_C_SIZE_MIRRORS / _C_OFFSET_FIELDS / _C_MACROS). Deleting the key
    is the other valid fix; loosening this assert is not.
    """
    assert set(binding.struct_sizes()) == _C_SIZE_KEYS_CHECKED, (
        "struct_sizes() keys and the cs2_env.py layout guard disagree; "
        f"published-but-unchecked={sorted(set(binding.struct_sizes()) - _C_SIZE_KEYS_CHECKED)}, "
        f"checked-but-missing={sorted(_C_SIZE_KEYS_CHECKED - set(binding.struct_sizes()))}")


def test_static_data_scalars_covers_every_scalar_field(simple_map):
    """static_data_scalars() must expose EVERY scalar StaticData field.

    WHY: this is the enforcement behind the "extend this dict for every new
    scalar" instruction in binding.c. Without it the rule is prose, and prose
    invariants decay across a long sequence of field-adding tasks: the dict would
    quietly stay at whatever subset someone last found interesting, and a new
    field's packing assignment would be unguarded with nothing complaining.

    The expectation is computed from StaticDataC._fields_ rather than pinned to a
    literal count, so it tracks the mirror automatically. The mirror is itself
    pinned to the C header by the sizeof/offsetof asserts above, which is what
    makes it a legitimate oracle for "what scalars does StaticData have".

    "Scalar" is decided by _is_scalar_ctype, which EXCLUDES pointers / arrays /
    nested structs rather than listing the accepted number types. That direction
    matters: an allowlist would quietly drop a future c_int8 or c_double field
    out of `expected`, and this test would go green on an unguarded field. See
    _is_scalar_ctype's docstring.

    If this fails: add SD_INT/SD_FLOAT(<field>) to py_static_data_scalars in
    src/c_env/binding.c, rebuild the .so, and add a value assert for the field to
    test_static_data_scalars_round_trip.

    PITFALL: a stale .so is the likeliest cause of a surprise failure here — the
    mirror is read from source, the dict from the built binary. Rebuild with
    `python setup.py build_ext --inplace` before believing this test.
    """
    expected = {name for name, _ in _STATIC_DATA_SCALARS}
    env = make_env(map_data=simple_map)
    try:
        exposed = set(binding.static_data_scalars(env._capsule))
    finally:
        env.close()
    assert exposed == expected, (
        f"missing from static_data_scalars(): {sorted(expected - exposed)}; "
        f"exposed but not a scalar in StaticDataC: {sorted(exposed - expected)}")


def test_every_ctypes_mirror_is_size_guarded():
    """Every ctypes.Structure defined in cs2_env.py must be size-guarded.

    WHY: the other direction — a struct_sizes() key nobody consumes — is caught
    by test_struct_sizes_keys_are_all_consumed. This is its mirror image: a new
    ctypes mirror added to cs2_env.py with no entry in _C_SIZE_MIRRORS gets
    overlaid on C-owned memory with NOTHING pinning it to the C layout, so it
    reads garbage silently. A comment in cs2_env.py used to assert this slip was
    undetectable ("invisible from both ends"). It is not: the module namespace
    is itself a second, independent list of the mirrors, so the two lists can be
    compared. That comment has been corrected.

    Scope: this checks each mirror is *claimed* by _C_SIZE_MIRRORS. Whether the
    claim is true is what the sizeof assert in cs2_env.py checks; together the
    two make "declared a mirror and forgot to guard it" impossible.

    `__module__` filtering is what keeps this honest: it counts only classes
    DEFINED in cs2_env.py, so a ctypes.Structure imported from elsewhere (none
    today) is not spuriously demanded, and ctypes.Structure itself is excluded.

    If this fails after you added a mirror: add a key to py_struct_sizes() in
    src/c_env/binding.c and the (key, mirror) pair to _C_SIZE_MIRRORS. Deleting
    the mirror is the other valid fix; deleting this assert is not.
    """
    from c_env import cs2_env
    from c_env.cs2_env import _C_SIZE_MIRRORS
    defined = {
        obj.__name__
        for obj in vars(cs2_env).values() if isinstance(obj, type)
        and issubclass(obj, ctypes.Structure) and obj.__module__ == cs2_env.__name__
    }
    guarded = {mirror.__name__ for _, mirror in _C_SIZE_MIRRORS}
    assert defined == guarded, (
        "ctypes mirrors in cs2_env.py and the _C_SIZE_MIRRORS size guard disagree; "
        f"declared-but-unguarded={sorted(defined - guarded)}, "
        f"guarded-but-not-defined-here={sorted(guarded - defined)}")
    # The struct_sizes() key convention ("AgentState" -> AgentStateC). Not
    # load-bearing for the asserts above, but it is what would catch a mis-paired
    # entry between two structs that happen to share a size.
    mispaired = [(key, mirror.__name__) for key, mirror in _C_SIZE_MIRRORS
                 if mirror.__name__ != key + "C"]
    assert not mispaired, ("struct_sizes() key must be the mirror's name minus the trailing 'C'; "
                           f"mismatched (key, mirror) pairs: {mispaired}")


def test_every_tunable_scalar_is_a_make_env_kwarg():
    """Reward/PBRS scalars must stay reachable from make_env.

    WHY: test_static_data_scalars_round_trip is exhaustive over the fields
    make_env exposes and silently skips the ones it does not. So a new `reward_*`
    field added to cs2_types.h, the ctypes mirror, the `static_data` packing
    mapping and static_data_scalars() — but NOT to make_env — would drop
    straight into the unchecked-by-value set with every other test still green,
    re-opening the transposition hole in exactly the same-width-float run where
    that hole lives. This makes the omission fail instead.

    Scope: tunables only (reward_*, pbrs_*). Map-derived geometry and nav.py
    constants are deliberately not kwargs; see the round-trip docstring's
    REMAINING GAP paragraph.
    """
    unreachable = sorted(name for name in _NON_KWARG_SCALARS if name.startswith(_TUNABLE_PREFIXES))
    assert not unreachable, (
        f"tunable StaticData scalars not exposed by make_env: {unreachable}; add them as "
        "keyword arguments to make_env and Cs2Env.__init__ so the sentinel sweep in "
        "test_static_data_scalars_round_trip covers them")


def test_int_sentinels_are_usable():
    """The int-sentinel tables must stay live and collision-free.

    WHY: _INT_SENTINEL_OVERRIDES and _BOOL_SENTINEL_CONFIGS are hand-written maps
    keyed by field name, so they rot in three directions and all of them fail
    SILENTLY.
      - A stale key (field renamed or dropped from make_env) simply stops
        applying; the field it was protecting then gets 101 back and the round
        trip fails with a confusing "expected 101, got 1" instead of pointing
        here. Worse, if the rename landed with a same-shaped replacement, nothing
        would point at these tables at all.
      - A flag named by only SOME of the configs keeps its make_env DEFAULT in
        the config that omits it, so its vector is half-unguarded — and nothing
        downstream notices, because a default that happens to match still
        round-trips.
      - Two fields sharing a sentinel VECTOR blinds the transposition check the
        whole sentinel scheme exists for: swap those two packing assignments and
        every assert still passes. This is the assert that made the third flag
        (jump_enabled) possible — with only 0 and 1 available per env, the
        distinctness that matters is across configs, not within one.
    None of the three is visible from test_static_data_scalars_round_trip, which
    only ever asserts value-in == value-out.
    """
    named = set(_INT_SENTINEL_OVERRIDES) | set(_BOOL_KWARG_SCALARS)
    stale = sorted(named - set(_INT_KWARG_SCALARS))
    assert not stale, (
        "_INT_SENTINEL_OVERRIDES / _BOOL_SENTINEL_CONFIGS name non-kwarg / non-int StaticData "
        f"fields: {stale}; drop the entry or fix the name")
    both = sorted(set(_INT_SENTINEL_OVERRIDES) & set(_BOOL_KWARG_SCALARS))
    assert not both, (
        f"fields in BOTH _INT_SENTINEL_OVERRIDES and _BOOL_SENTINEL_CONFIGS: {both}; the "
        "per-config value wins and the override is dead code — keep each field in exactly one")
    partial = sorted(n for n, vec in _INT_SENTINEL_VECTORS.items() if None in vec)
    assert not partial, (
        f"flags missing from at least one _BOOL_SENTINEL_CONFIGS dict: {partial}; every dict must "
        "name every flag, or the omitting config silently tests the make_env default instead")
    collisions = {}
    for name, vec in _INT_SENTINEL_VECTORS.items():
        collisions.setdefault(vec, []).append(name)
    clashing = {vec: names for vec, names in collisions.items() if len(names) > 1}
    assert not clashing, (
        f"int sentinel vectors are not pairwise distinct: {clashing}; a transposition between two "
        "fields sharing a vector would be invisible in EVERY config (add a config dict to "
        "_BOOL_SENTINEL_CONFIGS if you have run out of 2-vectors)")


def test_static_data_scalars_round_trip_sim_knobs(simple_map):
    """The four int32 sim knobs, read back by name.

    Three are Rung 0 (n_active_per_team, pin_pitch, crouch_enabled); the fourth
    is Rung 1a's jump_enabled.

    Redundant with the sentinel sweep by construction — and deliberately so.
    The sweep derives its kwargs from the dataclass fields; this test names them
    literally as an independent read. It also documents the intended tuple shape:
    n_active is a count, the other three are flags.

    The flags are set to a combination NOT used by either _BOOL_SENTINEL_CONFIGS
    entry (jump and pin_pitch both 1), so this is a genuinely independent read
    rather than a third copy of a config the sweep already ran.
    """
    env = make_env(map_data=simple_map,
                   config=_config_from_field_kwargs({
                       "n_active_per_team": 3,
                       "pin_pitch": 1,
                       "crouch_enabled": 0,
                       "jump_enabled": 1,
                   }))
    try:
        sc = binding.static_data_scalars(env._capsule)
        assert sc["n_active_per_team"] == 3
        assert sc["pin_pitch"] == 1
        assert sc["crouch_enabled"] == 0
        assert sc["jump_enabled"] == 1
    finally:
        env.close()


def test_n_active_per_team_out_of_range_rejected(simple_map):
    """Out-of-range n_active_per_team must raise, not abort the process.

    env_init asserts 1 <= n_active_per_team <= TEAM_SIZE, and a C assert kills
    the interpreter — which in a Puffer vecenv means a worker dying with no
    traceback. Cs2Env therefore validates first and raises ValueError, so a bad
    training config surfaces as a normal Python error. Both ends of the range are
    checked: 0 is the memset-zero value (the failure mode the C assert exists
    for) and 6 is the "someone typed the real team size wrong" case.
    """
    for bad in (0, 6, 2.9):                            # 2.9: non-integers must be rejected, not truncated
        with pytest.raises(ValueError):
            make_env(map_data=simple_map,
                     config=_config_from_field_kwargs({"n_active_per_team": bad}))
