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

# The ctypes types StaticDataC uses for plain numbers. c_int32 IS c_int on every
# platform this builds for, so the tuple has a duplicate — harmless, and it keeps
# the intent readable if that ever stops being true. Anything not in here is a
# pointer, a fixed array, or the nested WallListC, none of which
# static_data_scalars() can or should return.
_SCALAR_CTYPES = (ctypes.c_int, ctypes.c_int32, ctypes.c_float)


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

    This is the FMT-order guard: py_init's 69-arg PyArg_ParseTuple string is the
    only thing tying Cs2Env's kwargs to StaticData's fields, and a transposition
    there is invisible to sizeof. Values chosen so no two fields share a number.
    Non-configurable fields (round_time, max_turn_speed, laser_range) are checked
    against the nav.py constants Cs2Env forwards, and the map-derived ones
    (bombsite_dist_scale, N, grid_w/grid_h, spawn counts) against the fixture map
    — grid_w/grid_h in particular are adjacent same-width ints, so only a value
    comparison separates them.

    Not exhaustive by design: this test covers the fields with a distinguishable
    expected value. test_static_data_scalars_covers_every_scalar_field is what
    guarantees the dict itself stays complete.
    """
    import nav

    env = make_env(map_data=simple_map,
                   reward_kill=0.123,
                   reward_death=0.456,
                   reward_win_t_elimination=7.75,
                   pbrs_gamma=0.789)
    try:
        sc = binding.static_data_scalars(env._capsule)
        # Sentinels routed through make_env kwargs.
        assert sc["reward_kill"] == pytest.approx(0.123)
        assert sc["reward_death"] == pytest.approx(0.456)
        assert sc["reward_win_t_elimination"] == pytest.approx(7.75)
        assert sc["pbrs_gamma"] == pytest.approx(0.789)
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

    If this fails: add SD_INT/SD_FLOAT(<field>) to py_static_data_scalars in
    src/c_env/binding.c, rebuild the .so, and add a value assert for the field to
    test_static_data_scalars_round_trip.

    PITFALL: a stale .so is the likeliest cause of a surprise failure here — the
    mirror is read from source, the dict from the built binary. Rebuild with
    `python setup.py build_ext --inplace` before believing this test.
    """
    expected = {name for name, ctype in StaticDataC._fields_ if ctype in _SCALAR_CTYPES}
    env = make_env(map_data=simple_map)
    try:
        exposed = set(binding.static_data_scalars(env._capsule))
    finally:
        env.close()
    assert exposed == expected, (
        f"missing from static_data_scalars(): {sorted(expected - exposed)}; "
        f"exposed but not a scalar in StaticDataC: {sorted(exposed - expected)}")
