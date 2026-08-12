"""tag_summary — analyzer-side conflict scoring (spec 2026-08-13 §4.5).

Synthetic rows, no filesystem. Pre-registered criterion under test:
conflict = median(within_i - cross_half_i) >= 0.1 with a 95% bootstrap CI
excluding 0 AND n >= 5 surviving epochs; drops: selfplay_active epochs, NaN
measurements, norm ratio outside [0.1, 10].
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analyze_tplant import tag_summary


def _row(step,
         cross_half,
         within,
         gt=1.0,
         gct=1.0,
         selfplay=0.0,
         mb="mb0",
         group="trunk",
         cross=None):
    return {
        "step": step,
        "tag/selfplay_active": selfplay,
        f"tag/cossim_cross/{group}/{mb}": cross_half if cross is None else cross,
        f"tag/cossim_cross_half/{group}/{mb}": cross_half,
        f"tag/cossim_within_t/{group}/{mb}": within,
        f"tag/cossim_within_ct/{group}/{mb}": within,
        f"tag/gnorm_t/{group}/{mb}": gt,
        f"tag/gnorm_ct/{group}/{mb}": gct,
    }


def test_conflict_detected_when_cross_half_well_below_within():
    rows = [_row(step=i * 1e5, cross_half=0.15, within=0.60) for i in range(40)]
    s = tag_summary(rows, dead_windows=[])
    r = s["trunk"]["mb0"]["healthy"]
    assert abs(r["conflict"] - 0.45) < 1e-9
    assert r["ci_low"] > 0
    assert r["verdict"] == "CONFLICT"


def test_no_conflict_when_cross_equals_within():
    rows = [_row(step=i * 1e5, cross_half=0.5, within=0.5) for i in range(40)]
    s = tag_summary(rows, dead_windows=[])
    assert s["trunk"]["mb0"]["healthy"]["verdict"] == "no conflict detected"


def test_conflict_is_median_of_paired_diffs_not_diff_of_medians():
    """Plan-review finding 5: point estimate and CI must be the SAME
    estimator. Construction where they differ: diffs alternate 0.3/0.3/0.0…
    with within/cross values whose independent medians differ from the
    paired median. 21 epochs: 14 diffs of 0.3, 7 of 0.0 → median(diffs) =
    0.3; but median(within)=0.5, median(cross)=0.35 → difference 0.15. The
    paired value is correct.
    """
    rows = []
    for i in range(21):
        if i % 3 == 2:
            rows.append(_row(step=i * 1e5, cross_half=0.8, within=0.8))               # diff 0.0
        else:
            rows.append(_row(step=i * 1e5, cross_half=0.2 + 0.01 * i,
                             within=0.5 + 0.01 * i))                                  # diff 0.3
    s = tag_summary(rows, dead_windows=[])
    r = s["trunk"]["mb0"]["healthy"]
    assert abs(r["conflict"] -
               0.3) < 1e-9, ("conflict must be median(within_i - cross_half_i), not "
                             "median(within) - median(cross_half)")


def test_drop_rules_nan_norm_ratio_selfplay():
    """One NaN epoch, one degenerate-norm epoch (ratio 200), one
    selfplay-contaminated epoch — all excluded BEFORE medians; 20 clean
    epochs survive.
    """
    rows = ([_row(step=i * 1e5, cross_half=0.15, within=0.60)
             for i in range(20)] + [_row(step=21e5, cross_half=float("nan"), within=0.6)] +
            [_row(step=22e5, cross_half=-0.9, within=0.6, gt=100.0, gct=0.5)] +
            [_row(step=23e5, cross_half=-0.9, within=0.6, selfplay=1.0)])
    s = tag_summary(rows, dead_windows=[])
    r = s["trunk"]["mb0"]["healthy"]
    assert math.isfinite(r["conflict"])
    assert r["n_epochs"] == 20


def test_min_n_guard_blocks_single_epoch_conflict():
    """Plan-review finding 7: with n=1 every bootstrap resample is the same
    value and any diff >= 0.1 would self-certify. n < 5 ⇒ insufficient
    data, never CONFLICT.
    """
    rows = [_row(step=1e5, cross_half=0.1, within=0.6)]
    s = tag_summary(rows, dead_windows=[])
    r = s["trunk"]["mb0"]["healthy"]
    assert r["n_epochs"] == 1
    assert r["verdict"] == "insufficient data (n=1)"


def test_dead_window_rows_split_into_dead_phase():
    rows = ([_row(step=i * 1e5, cross_half=0.5, within=0.55) for i in range(10)] +
            [_row(step=(10 + i) * 1e5, cross_half=-0.2, within=0.5) for i in range(10)])
    s = tag_summary(rows, dead_windows=[(9.5e5, 21e5)])
    assert s["trunk"]["mb0"]["healthy"]["n_epochs"] == 10
    assert s["trunk"]["mb0"]["dead"]["n_epochs"] == 10
    assert s["trunk"]["mb0"]["dead"]["conflict"] > 0.5
