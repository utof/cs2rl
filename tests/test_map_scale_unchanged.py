"""R0-F (#136): bombsite_dist sentinel fill leaves bombsite_dist_scale bit-identical.

WHAT: pins `bombsite_dist_scale` for the simple map and dust2 to the values
measured on the parent commit (1936393, before the sentinel fill), and checks
that unreachable areas get a FINITE 4×max sentinel instead of inf.

WHY: the C potential (cs2_rewards.h) used `isfinite(dist)` to skip unreachable
areas. Under -ffast-math `isfinite` folds to true and inf*scale = inf leaks
into the PBRS potential. The fix replaces inf with 4×max in map.py so that
closeness = 1 − 4 < 0 clamps to 0 — the same reward as the old skip — and
scale must still be computed from the finite entries FIRST, or the sentinel
would shrink it by 4× and silently rescale every nav reward.

PITFALL: the pinned constants are measured values, not derived; if a map
change legitimately moves them, re-measure on the parent commit and update.
"""
from pathlib import Path

import numpy as np
import pytest

from cs2rl.env.map import SIMPLE_ROOMS, make_simple_map

# Pinned on feat/rung1-duel @ 1936393 (before this task) via
#   UV_NO_SYNC=1 uv run python -c "from cs2rl.map import make_simple_map as m; print(repr(m().bombsite_dist_scale))"
SIMPLE_SCALE_BEFORE = 0.2                              # max hop 5.0 → 1/5
                                                       # Same, from make_cs2_map(nav.NAV_PATH, nav.CACHE_PATH): max hop 42 → 1/42.
                                                       # 273 entries were inf pre-change (area-id gaps + unreachable areas).
DUST2_SCALE_BEFORE = 0.023809523809523808              # == 1.0 / 42.0

# One extra room far away from everything: no adjacency ⇒ unreachable from the bombsite.
_UNREACHABLE_ROOM = (len(SIMPLE_ROOMS), 5000.0, 5000.0, 5100.0, 5100.0, 0.0, False)


def test_simple_map_scale_bit_identical():
    md = make_simple_map()
    assert md.bombsite_dist_scale == SIMPLE_SCALE_BEFORE
    assert np.isfinite(md.bombsite_dist).all()
    # Default simple map: every area reaches the bombsite, so max hop × scale == 1.
    assert md.bombsite_dist.max() * md.bombsite_dist_scale == pytest.approx(1.0)


def test_unreachable_area_gets_4x_sentinel():
    md = make_simple_map(rooms=list(SIMPLE_ROOMS) + [_UNREACHABLE_ROOM])
    scale_ref = make_simple_map().bombsite_dist_scale
    assert md.bombsite_dist_scale == scale_ref                         # computed from finite entries FIRST
    reach = md.bombsite_dist[:len(SIMPLE_ROOMS)]
    assert np.isfinite(md.bombsite_dist).all()
    assert md.bombsite_dist.dtype == np.float32
    assert md.bombsite_dist[-1] == pytest.approx(4.0 * reach.max())
    assert 1.0 - md.bombsite_dist[-1] * md.bombsite_dist_scale < 0.0   # closeness clamps to 0


def test_no_bombsites_yields_zero_scale_and_inf_array():
    md = make_simple_map(bombsites=[])
    assert md.bombsite_dist_scale == 0.0
    assert not np.isfinite(md.bombsite_dist).any()


def test_dust2_scale_bit_identical():
    from cs2rl.env import nav
    from cs2rl.env.map import make_cs2_map

    if not Path(nav.NAV_PATH).exists():
        pytest.skip("dust2 nav data absent")
    md = make_cs2_map(nav.NAV_PATH, nav.CACHE_PATH)
    assert md.bombsite_dist_scale == DUST2_SCALE_BEFORE
    assert np.isfinite(md.bombsite_dist).all()
    # Sentinel = 4 × max finite hop = 4 × 42 = 168.
    assert md.bombsite_dist.max() == pytest.approx(4.0 / DUST2_SCALE_BEFORE)
