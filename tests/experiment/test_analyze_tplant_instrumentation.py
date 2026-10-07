"""Analyzer t_plant + outcome mix (spec 2026-08-15 §3.5).

Synthetic rows only — pass `rows=` to analyze_run, no metrics.jsonl.
Mix assertions use pytest.approx so IEEE 0.6-0.1-0.4 (~0.0999…) is
accepted; production must not wash floats to make exact == pass.
"""
import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from cs2rl.experiment.analyze_tplant import analyze_run, outcome_mix, print_report, t_plant

CAP, BOMB, PMIN = 640.0, 640.0, 0.15
_MIX_KEYS = ("t_detonation", "ct_defuse", "timeout", "t_elimination", "ct_elimination")


def _analyze(rows):
    """analyze_run without touching disk (`rows=` skips load_rows)."""
    return analyze_run(
        Path("fake-run"),
        CAP,
        BOMB,
        PMIN,
        min_block=3,
        window_steps=5e6,
        dead_window_steps=1e6,
        dead_after_steps=15e6,
        rows=rows,
    )


def test_old_format_still_inverts():
    row = {"game/bomb_plant_rate": 0.5, "environment/round_length": 960.0}
    # (960 - 0.5*640)/0.5 - 640 = 640
    t = t_plant(row, CAP, BOMB, PMIN)
    assert t is not None, "p = 0.5 is above p_min, so the old-format row inverts"
    assert abs(t - 640.0) < 1e-9


def test_old_format_respects_p_min():
    row = {"game/bomb_plant_rate": 0.15, "environment/round_length": 960.0}
    assert t_plant(row, CAP, BOMB, PMIN) is None
    row2 = {"game/bomb_plant_rate": 0.0, "environment/round_length": 960.0}
    assert t_plant(row2, CAP, BOMB, PMIN) is None


def test_new_format_uses_m_over_p_not_raw_mean():
    row = {"game/bomb_plant_rate": 0.5, "game/plant_tick": 40.0}
    assert t_plant(row, CAP, BOMB, PMIN) == 80.0


def test_new_format_key_presence_not_truthiness():
    row = {"game/bomb_plant_rate": 0.5, "game/plant_tick": 0.0}
    assert t_plant(row, CAP, BOMB, PMIN) == 0.0


def test_new_format_respects_p_min():
    row = {"game/bomb_plant_rate": 0.15, "game/plant_tick": 40.0}
    assert t_plant(row, CAP, BOMB, PMIN) is None
    row2 = {"game/bomb_plant_rate": 0.0, "game/plant_tick": 0.0}
    assert t_plant(row2, CAP, BOMB, PMIN) is None


def test_environment_plant_tick_fallback():
    row = {"game/bomb_plant_rate": 0.5, "environment/plant_tick": 40.0}
    assert t_plant(row, CAP, BOMB, PMIN) == 80.0


def test_outcome_mix_is_rate_differences():
    # Live IEEE: 0.6-0.1-0.4 is 0.0999…, not 0.1. approx pins the
    # subtraction the analyzer actually runs; do not Decimal-wash.
    row = {
        "game/win_by_detonation": 0.2,
        "game/win_by_defuse": 0.1,
        "game/timeout_rate": 0.4,
        "game/win_rate_t": 0.4,
        "game/win_rate_ct": 0.6,
    }
    mix = outcome_mix(row)
    assert mix is not None, "the row carries game/win_by_detonation"
    assert mix["t_detonation"] == pytest.approx(0.2)
    assert mix["ct_defuse"] == pytest.approx(0.1)
    assert mix["timeout"] == pytest.approx(0.4)
    assert mix["t_elimination"] == pytest.approx(0.4 - 0.2)
    assert mix["ct_elimination"] == pytest.approx(0.6 - 0.1 - 0.4)


def test_outcome_mix_clamps_negatives_and_skips_old_rows():
    assert outcome_mix({"game/timeout_rate": 0.4}) is None

    # Present 0.0 is new-format (key presence). CT-elim 0.4-0.1-0.4 < 0.
    row = {
        "game/win_by_detonation": 0.0,
        "game/win_by_defuse": 0.1,
        "game/timeout_rate": 0.4,
        "game/win_rate_t": 0.0,
        "game/win_rate_ct": 0.4,
    }
    mix = outcome_mix(row)
    assert mix is not None, "the row carries game/win_by_detonation"
    assert mix["t_detonation"] == pytest.approx(0.0)
    assert mix["ct_defuse"] == pytest.approx(0.1)
    assert mix["timeout"] == pytest.approx(0.4)
    assert mix["t_elimination"] == pytest.approx(0.0)
    assert mix["ct_elimination"] == pytest.approx(0.0)

    # Missing siblings default to 0.0 after the presence gate.
    only_det = outcome_mix({"game/win_by_detonation": 0.0})
    assert only_det == {k: 0.0 for k in _MIX_KEYS}

    # T-elim clamp: 0.1 - 0.3 < 0.
    t_neg = outcome_mix({
        "game/win_by_detonation": 0.3,
        "game/win_rate_t": 0.1,
        "game/win_by_defuse": 0.0,
        "game/timeout_rate": 0.0,
        "game/win_rate_ct": 0.0,
    })
    assert t_neg is not None, "the row carries game/win_by_detonation"
    assert t_neg["t_elimination"] == pytest.approx(0.0)
    assert t_neg["t_detonation"] == pytest.approx(0.3)


def test_analyze_run_old_format_p_min_drops_t_plant_and_has_no_mix():
    rows = [{
        "run_id": "old",
        "epoch": i,
        "step": 1000 * (i + 1),
        "game/bomb_plant_rate": 0.15,
        "environment/round_length": 960.0,
    } for i in range(5)]
    r = _analyze(rows)
    assert r["blocks"] == []
    assert r["median_slope"] is None
    assert r["outcome_mix_mean"] is None


def test_analyze_run_mix_mean_is_mean_of_rows_not_mix_of_means():
    # max(0, ·) is the nonlinearity: row0 clamps ct_elim -0.1 → 0,
    # row1 has 0.3. Mean of mixes = 0.15. Mix-of-means is
    # 0.5 - 0.1 - 0.3 = 0.1. Keys must be the five print_report reads.
    rows = [
        {
            "run_id": "n",
            "epoch": 0,
            "step": 1000,
            "game/bomb_plant_rate": 0.5,
            "game/plant_tick": 40.0,
            "game/win_by_detonation": 0.0,
            "game/win_by_defuse": 0.1,
            "game/timeout_rate": 0.4,
            "game/win_rate_t": 0.0,
            "game/win_rate_ct": 0.4,
        },
        {
            "run_id": "n",
            "epoch": 1,
            "step": 2000,
            "game/bomb_plant_rate": 0.5,
            "game/plant_tick": 40.0,
            "game/win_by_detonation": 0.0,
            "game/win_by_defuse": 0.1,
            "game/timeout_rate": 0.2,
            "game/win_rate_t": 0.0,
            "game/win_rate_ct": 0.6,
        },
    ]
    r = _analyze(rows)
    mix = r["outcome_mix_mean"]
    assert mix is not None, "both rows carry game/win_by_detonation"
    assert set(mix) == set(_MIX_KEYS)
    assert mix["ct_defuse"] == pytest.approx(0.1)
    assert mix["timeout"] == pytest.approx(0.3)
    assert mix["ct_elimination"] == pytest.approx(0.15)
    assert mix["ct_elimination"] != pytest.approx(0.1)


def test_print_report_emits_mix_labels_when_present():
    rows = [{
        "run_id": "n",
        "epoch": 0,
        "step": 1000,
        "game/bomb_plant_rate": 0.5,
        "game/plant_tick": 40.0,
        "game/win_by_detonation": 0.2,
        "game/win_by_defuse": 0.1,
        "game/timeout_rate": 0.4,
        "game/win_rate_t": 0.4,
        "game/win_rate_ct": 0.6,
    }]
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(_analyze(rows), 5e6)
    mix_line = next(line for line in buf.getvalue().splitlines() if line.startswith("outcome mix:"))
    assert "t_detonation" in mix_line
    assert "ct_defuse" in mix_line
    assert "timeout" in mix_line
    assert "t_elim" in mix_line
    assert "ct_elim" in mix_line
    assert "t_elimination" not in mix_line
    assert "ct_elimination" not in mix_line


def test_print_report_omits_mix_when_absent():
    rows = [{
        "run_id": "old",
        "epoch": 0,
        "step": 1000,
        "game/bomb_plant_rate": 0.5,
        "environment/round_length": 960.0,
    }]
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(_analyze(rows), 5e6)
    assert "outcome mix:" not in buf.getvalue()
