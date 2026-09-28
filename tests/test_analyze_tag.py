"""tag_summary — analyzer-side conflict scoring (spec 2026-08-13 §4.5).

Synthetic rows, no filesystem. Pre-registered criterion under test:
conflict = median(within_i - cross_half_i) >= 0.1 with a 95% bootstrap CI
excluding 0 AND n >= 5 surviving epochs; drops: selfplay_active epochs, NaN
measurements, norm ratio outside [0.1, 10].
"""
import io
import math
from contextlib import redirect_stdout

from cs2rl.experiment.analyze_tplant import print_tag_report, tag_summary


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
                                                                                         # decoy: the FULL-size cross key is never the criterion, so it
                                                                                         # defaults to a value distinct from cross_half — a regex that
                                                                                         # over-matched `cossim_cross` would move every result and fail.
        f"tag/cossim_cross/{group}/{mb}": cross_half + 0.11 if cross is None else cross,
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
    estimator, so conflict must be the median of the PAIRED differences.

    Discriminating construction (a fixture where both estimators agree
    would let the wrong one pass): pairs (cross_half, within) =
    (0.5, 0.5), (0.45, 0.45), (0.0, 0.3), (0.1, 0.4), (0.6, 0.9).
    Paired diffs are [0, 0, 0.3, 0.3, 0.3] → median 0.3. But BOTH marginal
    medians land on the zero-diff pairs — median(cross_half) = 0.45 and
    median(within) = 0.45 — so a difference of independent medians returns
    exactly 0.0. The paired value is the correct one.

    No verdict/CI assertion here: with two zero diffs in five, ci_low can
    touch 0, which says nothing about the estimator property under test.
    """
    pairs = [(0.5, 0.5), (0.45, 0.45), (0.0, 0.3), (0.1, 0.4), (0.6, 0.9)]
    rows = [_row(step=i * 1e5, cross_half=c, within=w) for i, (c, w) in enumerate(pairs)]
    s = tag_summary(rows, dead_windows=[])
    r = s["trunk"]["mb0"]["healthy"]
    assert abs(r["conflict"] -
               0.3) < 1e-9, ("conflict must be median(within_i - cross_half_i) (=0.3 here), "
                             "not median(within) - median(cross_half) (=0.0 here)")


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


def test_all_epochs_dropped_reports_dropped_not_absent():
    """Review finding: "every epoch dropped" and "instrument never ran" are
    different facts and must not print the same line. _n_raw counts rows
    carrying tag measurements BEFORE any drop rule, so the report can tell
    a fully-contaminated run from a pre-instrument one.
    """
    rows = [_row(step=i * 1e5, cross_half=0.1, within=0.6, selfplay=1.0) for i in range(8)]
    s = tag_summary(rows, dead_windows=[])
    assert s["_n_raw"] == 8
    assert "trunk" not in s and "policy_heads" not in s

    out = io.StringIO()
    with redirect_stdout(out):
        print_tag_report(s)
    text = out.getvalue()
    assert "was --tag-diagnostic on?" not in text, (
        "an all-dropped run must not be reported as having no measurements")
    assert "dropped" in text and "8" in text


def test_no_tag_keys_at_all_reports_instrument_absent():
    """The pre-instrument run (e.g. the 30M A/B checkpoints) still prints
    the original hint — _n_raw == 0 is the distinguishing fact."""
    s = tag_summary([{"step": 1e5, "game/bomb_plant_rate": 0.5}], dead_windows=[])
    assert s["_n_raw"] == 0
    out = io.StringIO()
    with redirect_stdout(out):
        print_tag_report(s)
    assert "was --tag-diagnostic on?" in out.getvalue()


def test_vf_control_prints_even_when_all_pg_epochs_drop():
    """The vf control is measured over different params than the pg groups;
    a pg-side drop (degenerate norm ratio) must not swallow it."""
    rows = []
    for i in range(6):
        row = _row(step=i * 1e5, cross_half=0.1, within=0.6, gt=100.0, gct=0.5)
        row["tag/cossim_vf/mb0"] = -0.8
        rows.append(row)
    s = tag_summary(rows, dead_windows=[])
    assert "trunk" not in s
    assert abs(s["_vf"][("mb0", "healthy")] - (-0.8)) < 1e-9
    out = io.StringIO()
    with redirect_stdout(out):
        print_tag_report(s)
    assert "vf control mb0/healthy" in out.getvalue()


def test_dead_window_rows_split_into_dead_phase():
    rows = ([_row(step=i * 1e5, cross_half=0.5, within=0.55) for i in range(10)] +
            [_row(step=(10 + i) * 1e5, cross_half=-0.2, within=0.5) for i in range(10)])
    s = tag_summary(rows, dead_windows=[(9.5e5, 21e5)])
    assert s["trunk"]["mb0"]["healthy"]["n_epochs"] == 10
    assert s["trunk"]["mb0"]["dead"]["n_epochs"] == 10
    assert s["trunk"]["mb0"]["dead"]["conflict"] > 0.5


def _split_row(step, cross_half, within, group, mb="mb0", split_active=1.0):
    """A metrics row from a SPLIT run: same shape as _row plus split/active."""
    row = _row(step=step, cross_half=cross_half, within=within, mb=mb, group=group)
    row["split/active"] = split_active
    return row


def test_split_active_routes_heads_cells_to_the_structural_state():
    """Spec §5 test 5b: under a split run the policy_heads cross cos-sim is
    EXACTLY 0 by architecture (each team's gradient is zero on the other
    team's copy), so within − 0 would print a false CONFLICT. Rows carrying
    split/active == 1 must route their policy_heads cells to the dedicated
    structural verdict — while trunk cells keep normal verdicts, because the
    trunk is still shared and its conflict is the actual decision metric.
    """
    rows = []
    for i in range(30):
        rows.append(_split_row(i * 1e5, cross_half=0.0, within=0.55, group="policy_heads"))
        rows.append(_split_row(i * 1e5, cross_half=0.15, within=0.60, group="trunk"))
    # merge the per-group rows pairwise so each epoch is one row, as in a real run
    merged = []
    for a, b in zip(rows[::2], rows[1::2], strict=True):
        m = dict(a)
        m.update(b)
        merged.append(m)

    s = tag_summary(merged, dead_windows=[])
    heads = s["policy_heads"]["mb0"]["healthy"]
    assert heads["verdict"] == "structural (split run — cross≡0 by architecture)"
    assert heads["n_epochs"] == 30
    assert s["trunk"]["mb0"]["healthy"]["verdict"] == "CONFLICT"
    # _n_raw accounting is about "did the instrument run", NOT about structural
    # exclusion — a structurally-excluded cell is not a dropped one (re-review N9).
    assert s["_n_raw"] == 30


def test_rows_without_split_active_behave_exactly_as_before():
    """Spec §5 test 5b (regression half): a pre-Batch-7 run has no
    split/active key anywhere and must score identically to today.
    """
    rows = [
        _row(step=i * 1e5, cross_half=0.15, within=0.60, group="policy_heads") for i in range(30)
    ]
    s = tag_summary(rows, dead_windows=[])
    assert s["policy_heads"]["mb0"]["healthy"]["verdict"] == "CONFLICT"
    assert s["_structural_backstop"] is False


def test_all_zero_cross_backstop_warns_without_split_active():
    """Spec §3.4 backstop (belt-and-braces, warn-only): if every measured
    policy_heads cross_half is exactly 0.0 and no split/active key is present,
    the run is almost certainly a split run whose metrics predate — or lost —
    the labeling key. Warn rather than relabel: the analyzer must not
    silently decide a run's architecture from a numeric coincidence.
    """
    rows = [
        _row(step=i * 1e5, cross_half=0.0, within=0.55, group="policy_heads") for i in range(30)
    ]
    s = tag_summary(rows, dead_windows=[])
    assert s["_structural_backstop"] is True
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_tag_report(s)
    out = buf.getvalue()
    assert "cross_half is exactly 0.0" in out
    assert "--tct-split-heads" in out


def test_structural_verdict_prints_on_its_own_line():
    """Spec §5 test 5b: the structural state is REPORTED, on its own line, and
    says why the cell is excluded from the decision — not hidden, and not
    routed through the drop path whose _n_raw wording ("everything dropped")
    would misdescribe it.
    """
    rows = [
        _split_row(i * 1e5, cross_half=0.0, within=0.55, group="policy_heads") for i in range(30)
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_tag_report(tag_summary(rows, dead_windows=[]))
    out = buf.getvalue()
    assert "structural (split run — cross≡0 by architecture)" in out
    assert "trunk cells are the decision metric" in out


TRUNK_STRUCTURAL = "structural (trunk split — cross≡0 by architecture)"


def test_trunk_active_routes_trunk_cells_and_replaces_footer():
    rows = []
    for i in range(30):
        r = _split_row(i * 1e5, cross_half=0.0, within=0.55, group="policy_heads")
        r.update(_row(step=i * 1e5, cross_half=0.0, within=0.55, group="trunk"))
        r["split/trunk_active"] = 1.0
        rows.append(r)
    s = tag_summary(rows, dead_windows=[])
    assert s["trunk"]["mb0"]["healthy"]["verdict"] == TRUNK_STRUCTURAL
    assert s["_saw_trunk_active"] is True
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_tag_report(s)
    out = buf.getvalue()
    assert TRUNK_STRUCTURAL in out
    assert "no actor TAG cell is a decision metric" in out
    assert "trunk cells are the decision metric" not in out


def test_trunk_active_zero_keeps_batch7_footer():
    """Key present but 0.0 is heads-only after this batch (float(hasattr))."""
    rows = [
        _split_row(i * 1e5, cross_half=0.0, within=0.55, group="policy_heads") for i in range(30)
    ]
    for r in rows:
        r["split/trunk_active"] = 0.0
    s = tag_summary(rows, dead_windows=[])
    assert s["_saw_trunk_active"] is False
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_tag_report(s)
    assert "trunk cells are the decision metric" in buf.getvalue()


def test_trunk_active_key_replaces_footer_even_if_trunk_cells_dropped():
    """Predicate is the raw key scan, not a surviving labeled cell."""
    rows = [
        _split_row(i * 1e5, cross_half=0.0, within=0.55, group="policy_heads") for i in range(30)
    ]
    for r in rows:
        r["split/trunk_active"] = 1.0
        r["tag/selfplay_active"] = 1   # drop every epoch
    s = tag_summary(rows, dead_windows=[])
    assert s["_saw_trunk_active"] is True
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_tag_report(s)
    assert "trunk cells are the decision metric" not in buf.getvalue()
    assert "no actor TAG cell is a decision metric" in buf.getvalue()


def test_trunk_cross_zero_backstop_warns_without_trunk_active():
    """Spec §5.5b / §3.4 trunk backstop: warn-only, no relabel.

    WHAT: every surviving trunk cross_half is exactly 0.0 and no
    split/trunk_active key is present (absent, not 0.0). The report
    warns that the labeling key is missing; trunk cells keep a normal
    CONFLICT verdict.

    WHY: a trunk-split run whose metrics lost the key would otherwise
    print a false CONFLICT on the last remaining decision cell with no
    hint that the architecture is the cause. Relabeling from a numeric
    coincidence is forbidden (same contract as the heads backstop).

    PITFALL: these rows are heads-legacy (no split/active). The Batch 7
    "trunk cells are the decision metric" footer must stay off — it
    keys on a heads structural cell, not on a numeric trunk zero.
    """
    rows = []
    for i in range(30):
        r = _row(step=i * 1e5, cross_half=0.0, within=0.55, group="trunk")
        r.update(_row(step=i * 1e5, cross_half=0.15, within=0.60, group="policy_heads"))
        rows.append(r)
    assert "split/trunk_active" not in rows[0]
    assert "split/active" not in rows[0]
    s = tag_summary(rows, dead_windows=[])
    assert s["_trunk_structural_backstop"] is True
    assert s["_saw_trunk_active"] is False
    assert s["_structural_backstop"] is False
    assert s["trunk"]["mb0"]["healthy"]["verdict"] == "CONFLICT"
    assert s["trunk"]["mb0"]["healthy"]["verdict"] != TRUNK_STRUCTURAL
    assert s["policy_heads"]["mb0"]["healthy"]["verdict"] == "CONFLICT"
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_tag_report(s)
    out = buf.getvalue()
    assert "every measured trunk cross_half is exactly 0.0" in out
    assert "split/trunk_active" in out
    assert "--tct-split-trunk" in out
    assert TRUNK_STRUCTURAL not in out
    assert "no actor TAG cell is a decision metric" not in out
    assert "trunk cells are the decision metric" not in out
