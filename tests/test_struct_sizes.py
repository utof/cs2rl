"""binding.struct_sizes() / binding.static_data_scalars() — the layout oracles.

WHY: cs2_env.py used to carry hand-measured sizeof literals (164/1708/204/6832)
and hand-measured offsetof literals (476/480/496) with a comment forbidding
"invented" pads. Every field appended to a C struct made those literals rot, and
the only way to refresh them was to compile a throwaway printf TU by hand.
struct_sizes() asks the same compiler that laid the structs out, so the ctypes
mirrors in cs2_env.py are pinned to the C headers automatically.

sizeof alone cannot catch a *mis-ordered* PyArg_ParseTuple FMT string in
binding.c py_init() — swapping two floats keeps every size identical while
silently feeding reward_kill into reward_death. static_data_scalars() closes
that hole by reading scalar StaticData fields back out of a live env, so a test
can push distinct sentinels through Cs2Env and check where they landed.

PITFALL: every struct/FMT change on this branch must extend BOTH dicts, or the
new field is unguarded.
"""

import ctypes
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

# Repo convention (same two sys.path.insert lines at the top of
# tests/test_binding.py): `binding` is a C extension living in src/c_env, so
# that directory must be on sys.path before the import. conftest.py only adds
# src/. Anchored by name, not by line number — line anchors rot.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "c_env"))
import binding                         # noqa: E402

from c_env.cs2_env import (                                                                   # noqa: E402
    _C_SIZE_KEYS_CHECKED, AgentStateC, Dust2EnvC, GameStateC, StaticDataC, StepStatsC, WallC,
    WallListC, make_env,
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

# Partition those scalars by whether make_env can set them. Anything make_env
# exposes is sentinel-testable (test_static_data_scalars_round_trip pushes a
# distinct value through it); anything it does not is map-derived or a nav.py
# constant and is checked against that source instead.
_MAKE_ENV_PARAMS = frozenset(inspect.signature(make_env).parameters)
_FLOAT_KWARG_SCALARS = tuple(
    sorted(n for n, t in _STATIC_DATA_SCALARS if n in _MAKE_ENV_PARAMS and t is ctypes.c_float))
_INT_KWARG_SCALARS = tuple(
    sorted(n for n, t in _STATIC_DATA_SCALARS if n in _MAKE_ENV_PARAMS and t is not ctypes.c_float))
_NON_KWARG_SCALARS = tuple(sorted(n for n, _ in _STATIC_DATA_SCALARS if n not in _MAKE_ENV_PARAMS))

# Int kwargs whose C field REJECTS the generic 101, 102, ... run below, with the
# in-range value to use instead. Rung 0 (spec 2026-08-29 §2.1) added the first
# three int kwargs and all three are constrained: n_active_per_team is validated
# to 1..TEAM_SIZE (Cs2Env raises ValueError), and pin_pitch / crouch_enabled are
# flags Cs2Env normalises with int(bool(...)), so a 101 would come back as 1 and
# the round trip would fail on a perfectly healthy build.
#
# Overriding beats excluding them: an excluded field is an UNGUARDED FMT
# position, which is the exact hole this module exists to close. Distinctness is
# the only property the scheme needs, and 3 / 1 / 0 are pairwise distinct — a
# transposition among the three "iii" positions is still visible.
#
# ADDING AN INT KWARG: if it accepts arbitrary ints, add nothing — the fallback
# covers it automatically. If it is range- or flag-constrained, add it here;
# test_int_sentinels_are_usable is what tells you which case you are in.
_INT_SENTINEL_OVERRIDES = {
    "n_active_per_team": 3,
    "pin_pitch": 1,
    "crouch_enabled": 0,
}

# One distinct sentinel per settable field, generated from the sorted field list
# so a newly added field automatically gets one. Distinctness is the only
# property that matters: it is what makes a swapped pair in py_init's FMT string
# visible. Floats get 0.101, 0.102, ... — not exactly representable in float32,
# hence pytest.approx on the way back (see the widening PITFALL in binding.c).
# Ints get 101, 102, ... unless _INT_SENTINEL_OVERRIDES names them.
# Do NOT sentinel with the defaults instead: several collide
# (reward_win_t_detonation and reward_win_ct_defuse are both 5.0, reward_death
# and reward_plant_interrupted both 0.1), so a transposition between a colliding
# pair would stay invisible.
_SENTINELS = {name: 0.101 + 0.001 * i for i, name in enumerate(_FLOAT_KWARG_SCALARS)}
_SENTINELS.update({
    name: _INT_SENTINEL_OVERRIDES.get(name, 101 + i)
    for i, name in enumerate(_INT_KWARG_SCALARS)
})

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
    test_static_data_scalars_round_trip (sentinels through the FMT string).
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

    This is the FMT-order guard: py_init's 72-arg PyArg_ParseTuple string is the
    only thing tying Cs2Env's kwargs to StaticData's fields, and a transposition
    there is invisible to sizeof — two swapped floats parse fine and keep every
    size identical while feeding reward_kill into reward_death.

    EXHAUSTIVE over the settable fields. Every StaticData scalar make_env
    exposes as a keyword argument (_SENTINELS, derived by intersecting the
    make_env signature with the mirror — not hand-listed) gets its own distinct
    value and is asserted back by name. An earlier version checked 4 of the 24
    and defended that as "the fields with a distinguishable expected value";
    that premise was wrong — make_env exposes all of them — and it left 20 of
    the 23 consecutive same-width reward/PBRS floats, the exact run the code
    itself calls the hiding place for a transposition, unchecked.

    Fields make_env does NOT expose are checked against their real source
    instead: the nav.py constants Cs2Env forwards (round_time, max_turn_speed,
    laser_*) and the map-derived ones (bombsite_dist_scale, N, grid_w/grid_h,
    max_area_id, spawn counts) against the fixture map. grid_w/grid_h are
    adjacent same-width ints, so only a value comparison separates them.

    REMAINING GAP, stated plainly. These 16 non-kwarg scalars are covered only
    by key presence (test_static_data_scalars_covers_every_scalar_field), not by
    value: grid_x_min, grid_y_min, grid_inv_cell, inv_x_range, inv_y_range,
    x_offset, y_offset, shoot_cooldown, bomb_plant_time, bomb_defuse_time,
    bomb_defuse_kit, bomb_timer, footstep_radius_sq, gunshot_radius_sq,
    enemy_memory_ticks, stale_memory_tick. They are not settable, so a value
    check would have to recompute the implementation's own formula (the
    geometry) or restate a nav.py constant (the timings) — weaker than a
    sentinel, and for the geometry partly degenerate, since a symmetric fixture
    map can make x_offset == y_offset. Note also nav.BOMB_TIMER ==
    nav.ROUND_TIME == 640 today, so bomb_timer and round_time are mutually
    indistinguishable by value however they are checked. Closing this properly
    means sentinels, which means kwargs; out of scope here, and deliberately not
    papered over.
    """
    import nav

    env = make_env(map_data=simple_map, **_SENTINELS)
    try:
        sc = binding.static_data_scalars(env._capsule)
        # ── every settable field, one distinct sentinel each ──
        absent = sorted(name for name in _SENTINELS if name not in sc)
        assert not absent, (f"make_env kwargs with no static_data_scalars() key: {absent}; add "
                            "SD_INT/SD_FLOAT for them in src/c_env/binding.c and rebuild (see "
                            "test_static_data_scalars_covers_every_scalar_field)")
        wrong = {}
        for name, sent in _SENTINELS.items():
            if sc[name] != pytest.approx(sent):
                # Which sentinel DID land here? For a transposed FMT string that
                # names the swap partner outright, which is the whole diagnosis.
                partner = next(
                    (other for other, v in _SENTINELS.items() if sc[name] == pytest.approx(v)),
                    None)
                wrong[name] = (sent, sc[name], partner)
        assert not wrong, (
            "sentinel landed in the wrong StaticData field — py_init's FMT string in "
            "src/c_env/binding.c is out of order with the binding.init() call in "
            f"Cs2Env.__init__. {{field: (sent, got, whose_sentinel_got_is)}} = {wrong}")
        # Constants forwarded verbatim from nav.py.
        assert sc["round_time"] == nav.ROUND_TIME == 640
        assert sc["max_turn_speed"] == pytest.approx(nav.MAX_TURN_SPEED_RAD)
        assert sc["max_turn_speed"] == pytest.approx(np.pi / 4)
        assert sc["laser_range"] == pytest.approx(float(nav.LASER_RANGE))
        assert sc["laser_range"] == pytest.approx(3000.0)
        assert sc["laser_range_sq"] == pytest.approx(3000.0 * 3000.0)
        assert sc["laser_damage"] == nav.LASER_DAMAGE
        # Map-derived scalars. These are the only non-reward, non-nav values in
        # StaticData, and bombsite_dist_scale is FMT arg 21 — wedged between
        # y_offset (float) and laser_damage (int), exactly where a transposition
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
    field's FMT position would be unguarded with nothing complaining.

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
    field added to cs2_types.h, the FMT string, the ctypes mirror and
    static_data_scalars() — but NOT to make_env — would drop straight into the
    unchecked-by-value set with every other test still green, re-opening the
    FMT-transposition hole in exactly the same-width-float run where that hole
    lives. This makes the omission fail instead.

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
    """The int-sentinel table must stay live and collision-free.

    WHY: _INT_SENTINEL_OVERRIDES is a hand-written map keyed by field name, so it
    rots in two directions and both fail SILENTLY.
      - A stale key (field renamed or dropped from make_env) simply stops
        applying; the field it was protecting then gets 101 back and the round
        trip fails with a confusing "expected 101, got 1" instead of pointing
        here. Worse, if the rename landed with a same-shaped replacement, nothing
        would point at this table at all.
      - Two overrides sharing a value blinds the transposition check the whole
        sentinel scheme exists for: swap those two FMT positions and every assert
        still passes.
    Neither is visible from test_static_data_scalars_round_trip, which only ever
    asserts value-in == value-out.
    """
    stale = sorted(set(_INT_SENTINEL_OVERRIDES) - set(_INT_KWARG_SCALARS))
    assert not stale, (
        f"_INT_SENTINEL_OVERRIDES names non-kwarg / non-int StaticData fields: {stale}; "
        "drop the entry or fix the name")
    int_sentinels = [_SENTINELS[name] for name in _INT_KWARG_SCALARS]
    assert len(set(int_sentinels)) == len(int_sentinels), (
        f"int sentinels are not pairwise distinct: "
        f"{dict(zip(_INT_KWARG_SCALARS, int_sentinels, strict=True))}; "
        "a transposition between two equal-valued fields would be invisible")


def test_static_data_scalars_round_trip_rung0_knobs(simple_map):
    """The three Rung 0 "iii" FMT positions (69-71), read back by name.

    Redundant with the sentinel sweep by construction — and deliberately so.
    The sweep derives its kwargs from inspect.signature(make_env), so it silently
    stops covering these the moment they leave make_env's signature (Task 4+
    touches the same call chain). This test names them literally, so that
    removal fails loudly. It also documents the intended tuple shape for the
    next task: n_active is a count, the other two are flags.
    """
    env = make_env(map_data=simple_map, n_active_per_team=3, pin_pitch=1, crouch_enabled=0)
    try:
        sc = binding.static_data_scalars(env._capsule)
        assert sc["n_active_per_team"] == 3
        assert sc["pin_pitch"] == 1
        assert sc["crouch_enabled"] == 0
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
    for bad in (0, 6):
        with pytest.raises(ValueError):
            make_env(map_data=simple_map, n_active_per_team=bad)
