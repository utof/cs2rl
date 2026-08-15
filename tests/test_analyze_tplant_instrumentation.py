import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from analyze_tplant import outcome_mix, t_plant

CAP, BOMB, PMIN = 640.0, 640.0, 0.15


def test_old_format_still_inverts():
    row = {"game/bomb_plant_rate": 0.5, "environment/round_length": 960.0}
    # (960 - 0.5*640)/0.5 - 640 = 640
    assert abs(t_plant(row, CAP, BOMB, PMIN) - 640.0) < 1e-9


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
    row = {
        "game/win_by_detonation": 0.2,
        "game/win_by_defuse": 0.1,
        "game/timeout_rate": 0.4,
        "game/win_rate_t": 0.4,
        "game/win_rate_ct": 0.6,
    }
    mix = outcome_mix(row)
    assert mix["t_detonation"] == 0.2
    assert mix["ct_defuse"] == 0.1
    assert mix["timeout"] == 0.4
    assert mix["t_elimination"] == 0.2
    assert mix["ct_elimination"] == 0.1


def test_outcome_mix_clamps_negatives_and_skips_old_rows():
    assert outcome_mix({"game/timeout_rate": 0.4}) is None
    row = {
        "game/win_by_detonation": 0.0,
        "game/win_by_defuse": 0.1,
        "game/timeout_rate": 0.4,
        "game/win_rate_t": 0.0,
        "game/win_rate_ct": 0.4,
    }
    mix = outcome_mix(row)
    assert mix["ct_elimination"] == 0.0
