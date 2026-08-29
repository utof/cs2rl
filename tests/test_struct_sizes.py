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

# Repo convention (mirrors tests/test_binding.py:7-9): `binding` is a C
# extension living in src/c_env, so that directory must be on sys.path before
# the import. conftest.py only adds src/.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "c_env"))
import binding                         # noqa: E402

from c_env.cs2_env import (                                                                  # noqa: E402
    AgentStateC, Dust2EnvC, GameStateC, StaticDataC, StepStatsC, WallC, WallListC, make_env,
)


def test_struct_sizes_match_ctypes_mirrors():
    """Every ctypes mirror in cs2_env.py must agree with the C compiler.

    If this fails, the mirror drifted from cs2_types.h — fix the mirror, never
    the assert. cs2_env.py runs the same comparison at import time, so a real
    drift usually surfaces as an ImportError long before this test runs; the
    test exists to keep the oracle itself honest (a struct_sizes() that returns
    garbage would make the import-time assert vacuous).
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
    against the nav.py constants Cs2Env forwards.
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
        assert sc["bombsite_dist_scale"] > 0.0
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
