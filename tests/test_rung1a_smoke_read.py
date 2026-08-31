"""scripts/rung1a_smoke_read.py — the frozen Rung 1a smoke rules on synthetic rows.

Rules source: docs/science-superpowers/preregistrations/2026-08-31-rung1a-smoke.md
(the pre-registration is frozen by sha256; if an expectation here disagrees with
it, the READER is wrong, never the prereg).

Every number below is hand-derivable from the fixture builders at the top. The
fixtures imitate the real metrics stream: 61 rows at 16,384 hero steps each
(final agent_steps 999,424), of which exactly the last 7 clear the 900,000
window floor, and a 9-bin `environment/action_move_*` histogram whose counts sum
to 320 = 2 agents x 160 ticks (they are COUNTS, not fractions — the reader must
divide, see test_move_bin0_is_a_fraction_not_a_raw_count).

PITFALL this file guards: a pre-flight that is silently unevaluable (key missing
from every row) must report SMOKE INVALID, never a vacuous "ok" — a vacuous ok
would let a harness defect be published as a scientific FAIL.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from rung1a_smoke_read import EXIT_CODES, main, read_run               # noqa: E402, I001

STEP = 16384                           # hero steps per epoch row (256 envs x 64 bptt x 1 active)
N_ROWS = 61                            # 61 * 16384 = 999,424 — the T4 budget as delivered
CAP = -2.9957                          # --aim-log-std-max
INIT = -3.1957                         # resolve_aim_log_std_init(CAP) = cap - 0.2
CFG = {"aim_log_std_max": CAP, "aim_log_std_init": INIT, "participating_timesteps": 1000000}


def _row(k, **over):
    """Row k (1-based) of a healthy Rung 1a run; `over` replaces top-level keys.

    Defaults are the "everything nominal, nothing learned" case: statue inert,
    hero-only participation, sigma parked at its init, 10 shots/episode with 1
    on target (hit rate 0.1) and no kills.
    """
    row = {
        "agent_steps": k * STEP,
        "environment/episodes": 100.0,
        "game/kills_per_episode": 0.0,
        "game/kills_ct": 0.0,
        "game/shots_fired": 10.0,
        "game/shots_on_target": 1.0,
        "losses/participating_rows": 16384.0,
        "policy/aim_log_std_yaw_raw": INIT,
    }
    row["environment/action_move_0"] = 160.0           # statue: 160 of 320 ticks
    for b in range(1, 9):
        row[f"environment/action_move_{b}"] = 20.0
    row.update(over)
    return row


def _rows(n=N_ROWS, window_over=None, all_over=None):
    """n rows; `all_over` patches every row, `window_over` only the last 7 (the
    ones inside the >= 900,000 window)."""
    rows = [_row(k, **(all_over or {})) for k in range(1, n + 1)]
    for row in rows[-7:]:
        row.update(window_over or {})
    return rows


def _write(tmp_path, rows, cfg=CFG, name="s0"):
    run = tmp_path / name
    run.mkdir(parents=True, exist_ok=True)
    if cfg is not None:
        (run / "config.json").write_text(json.dumps(cfg))
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run


def _read(tmp_path, rows, cfg=CFG):
    return read_run(_write(tmp_path, rows, cfg))


def _check(rep, n):
    return next(c for c in rep["checks"] if c["n"] == n)


# ── window & dedup ───────────────────────────────────────────────────────────
def test_window_is_the_last_seven_rows_over_900k(tmp_path):
    rep = _read(tmp_path, _rows())
    assert len(rep["rows"]) == N_ROWS
    assert [r["agent_steps"] for r in rep["window"]] == [k * STEP for k in range(55, 62)]


def test_window_caps_at_ten_rows(tmp_path):
    # A longer run (200 rows) has far more than 10 rows over the floor; the
    # window is the LAST 10, so an early-window row must not leak in.
    rep = _read(tmp_path, _rows(n=200))
    assert len(rep["window"]) == 10
    assert [r["agent_steps"] for r in rep["window"]] == [k * STEP for k in range(191, 201)]


def test_fewer_than_five_window_rows_is_smoke_invalid(tmp_path):
    # 58 rows ⇒ only rows 55..58 clear 900,000 ⇒ 4 < MIN_WINDOW_ROWS.
    rep = _read(tmp_path, _rows(n=58))
    assert rep["verdict"] == "SMOKE INVALID"
    assert any("only 4 rows in the window" in r for r in rep["invalid"])


def test_duplicate_agent_steps_are_deduped_last_wins(tmp_path):
    rows = _rows()
    replay = _row(61, **{"game/kills_per_episode": 9.0})               # post-resume replay of row 61
    rep = _read(tmp_path, rows + [replay])
    assert len(rep["rows"]) == N_ROWS                                  # not 62
    assert rep["rows"][-1]["game/kills_per_episode"] == 9.0            # last value, first position
    assert rep["window"][-1]["agent_steps"] == 61 * STEP


# ── episode weighting ────────────────────────────────────────────────────────
def test_episode_weighting_matches_hand_computation(tmp_path):
    """Non-uniform episode counts: the pooled value must differ from the plain
    row mean, and equal the hand-computed weighted value."""
    rows = _rows()
    # (episodes, kills/episode, fired, on_target) for the 7 window rows.
    win = [(257, 0.1, 10.0, 1.0), (4, 3.0, 2.0, 2.0), (252, 0.1, 10.0, 1.0), (8, 3.0, 2.0, 2.0),
           (240, 0.1, 10.0, 1.0), (6, 3.0, 2.0, 2.0), (250, 0.1, 10.0, 1.0)]
    for row, (ep, kills, fired, on_t) in zip(rows[-7:], win, strict=True):
        row.update({
            "environment/episodes": float(ep),
            "game/kills_per_episode": kills,
            "game/shots_fired": fired,
            "game/shots_on_target": on_t,
        })
    rep = _read(tmp_path, rows)
    episodes = sum(e for e, _, _, _ in win)                            # 1017
    kills = sum(e * k for e, k, _, _ in win) / episodes                # 153.9 / 1017
    fired = sum(e * f for e, _, f, _ in win)                           # 7526.0
    on_t = sum(e * t for e, _, _, t in win)                            # 826.0
    assert rep["agg"]["episodes"] == pytest.approx(1017.0)
    assert rep["agg"]["kills"] == pytest.approx(kills)                 # ≈ 0.1513
    assert rep["agg"]["hit_rate"] == pytest.approx(on_t / fired)       # ≈ 0.1097

    # The sparse rows (4/8/6 episodes) carry the 3.0-kill spikes; an unweighted
    # row mean would report ≈ 1.4 kills/episode and PASS the gate.
    plain = sum(k for _, k, _, _ in win) / len(win)
    assert plain > 1.0 and rep["agg"]["kills"] < 0.2


def test_zero_episodes_in_window_is_smoke_invalid(tmp_path):
    rep = _read(tmp_path, _rows(window_over={"environment/episodes": 0.0}))
    assert rep["verdict"] == "SMOKE INVALID"
    assert any("zero episodes" in r for r in rep["invalid"])


# ── pre-flights ──────────────────────────────────────────────────────────────
def test_all_preflights_pass_on_a_healthy_run(tmp_path):
    rep = _read(tmp_path, _rows())
    # Six checks: 1, 2, 3, 4a, 4b, 5 (pre-flight 4 is two assertions).
    assert [c["n"] for c in rep["checks"]] == [1, 2, 3, "4a", "4b", 5]
    assert [c["ok"] for c in rep["checks"]] == [True] * 6
    assert rep["invalid"] == []
    assert _check(rep, 3)["observed"] == "999424"


def test_preflight1_wrong_participating_rows_is_invalid(tmp_path):
    # 32,768 is the self-play value — the statue override never reached the trainer.
    rows = _rows()
    rows[40]["losses/participating_rows"] = 32768.0
    rep = _read(tmp_path, rows)
    assert not _check(rep, 1)["ok"]
    assert rep["verdict"] == "SMOKE INVALID"


def test_preflight1_ignores_the_first_row_without_losses(tmp_path):
    # mean_and_log() runs before self.losses is set, so row 0 carries no losses/*.
    rows = _rows()
    del rows[0]["losses/participating_rows"]
    rep = _read(tmp_path, rows)
    assert _check(rep, 1)["ok"] and "60/60" in _check(rep, 1)["observed"]


def test_preflight1_key_missing_everywhere_is_invalid_not_vacuously_ok(tmp_path):
    rows = _rows()
    for row in rows:
        del row["losses/participating_rows"]
    rep = _read(tmp_path, rows)
    assert not _check(rep, 1)["ok"] and "key drift" in _check(rep, 1)["observed"]
    assert rep["verdict"] == "SMOKE INVALID"


def test_preflight2_opponent_team_key_is_invalid(tmp_path):
    rows = _rows()
    rows[3]["self_play/opponent_team"] = 1.0
    rep = _read(tmp_path, rows)
    assert not _check(rep, 2)["ok"]
    assert rep["verdict"] == "SMOKE INVALID"


def test_preflight3_short_budget_is_invalid(tmp_path):
    """A run that stopped at 940,000 steps: the window is full (6 rows >= 900,000)
    so ONLY pre-flight 3 fires — the budget shortfall is not masked by a
    too-small window."""
    rows = _rows(n=54)                                                  # up to 884,736
    tail = [_row(1, agent_steps=900_000 + i * 8_000) for i in range(6)] # 900,000 … 940,000
    rep = _read(tmp_path, rows + tail)
    assert len(rep["window"]) == 6
    assert not _check(rep, 3)["ok"] and _check(rep, 3)["observed"] == "940000"
    assert rep["verdict"] == "SMOKE INVALID"
    assert [c["ok"] for c in rep["checks"]] == [True, True, False, True, True, True]


def test_preflight4a_statue_kill_is_invalid(tmp_path):
    rows = _rows()
    rows[10]["game/kills_ct"] = 0.01
    rep = _read(tmp_path, rows)
    assert not _check(rep, "4a")["ok"]
    assert rep["verdict"] == "SMOKE INVALID"


def test_preflight4b_moving_opponent_is_invalid(tmp_path):
    # A live opponent's bin-0 fraction is ~0.10; here row 20 has 32/320 = 0.1.
    rows = _rows()
    rows[20]["environment/action_move_0"] = 32.0
    for b in range(1, 9):
        rows[20][f"environment/action_move_{b}"] = 36.0
    rep = _read(tmp_path, rows)
    assert not _check(rep, "4b")["ok"] and "0.1000" in _check(rep, "4b")["observed"]
    assert rep["verdict"] == "SMOKE INVALID"


def test_move_bin0_is_a_fraction_not_a_raw_count(tmp_path):
    """The counts are per-episode COUNTS: a raw `>= 0.45` on action_move_0 would
    pass for a fully mobile opponent (32 >> 0.45). Only the fraction catches it."""
    rows = _rows()
    for row in rows:
        row["environment/action_move_0"] = 32.0
        for b in range(1, 9):
            row[f"environment/action_move_{b}"] = 36.0
    rep = _read(tmp_path, rows)
    assert not _check(rep, "4b")["ok"]


def test_preflight4b_missing_histogram_is_invalid(tmp_path):
    rows = _rows()
    for row in rows:
        for b in range(9):
            del row[f"environment/action_move_{b}"]
    rep = _read(tmp_path, rows)
    assert not _check(rep, "4b")["ok"] and "key drift" in _check(rep, "4b")["observed"]
    assert rep["verdict"] == "SMOKE INVALID"


def test_preflight5_missing_sigma_key_is_invalid(tmp_path):
    rows = _rows()
    for row in rows:
        del row["policy/aim_log_std_yaw_raw"]
    rep = _read(tmp_path, rows)
    assert not _check(rep, 5)["ok"] and rep["verdict"] == "SMOKE INVALID"
    assert rep["sigma_capped"] is False                # absent ≠ violated


def test_missing_config_provenance_is_invalid_not_a_crash(tmp_path):
    rep = _read(tmp_path, _rows(), cfg=None)
    assert rep["verdict"] == "SMOKE INVALID"
    assert any("config.json unreadable" in r for r in rep["invalid"])


def test_missing_metrics_file_is_invalid_not_a_crash(tmp_path):
    run = tmp_path / "empty"
    run.mkdir()
    (run / "config.json").write_text(json.dumps(CFG))
    rep = read_run(run)
    assert rep["verdict"] == "SMOKE INVALID"
    assert any("metrics.jsonl unreadable" in r for r in rep["invalid"])


# ── verdict routing ──────────────────────────────────────────────────────────
def test_pass_route(tmp_path):
    rep = _read(
        tmp_path,
        _rows(window_over={
            "game/kills_per_episode": 0.6,
            "game/shots_on_target": 5.0,               # 5/10 = 0.5 >= 0.4
        }))
    assert rep["verdict"] == "PASS"
    assert "Rung 1b" in rep["next_step"]
    assert EXIT_CODES[rep["verdict"]] == 0


def test_pass_needs_both_kills_and_hit_rate(tmp_path):
    # kills clears 0.5 but the hit rate does not — not a PASS, and not covered
    # by a sigma-movement branch either.
    rep = _read(tmp_path, _rows(window_over={"game/kills_per_episode": 0.6}))
    assert rep["verdict"] == "FAIL-unrouted"


def test_fail_aim_route_kills_low_sigma_moved(tmp_path):
    rep = _read(
        tmp_path,
        _rows(window_over={
            "game/kills_per_episode": 0.2,
            "policy/aim_log_std_yaw_raw": INIT - 0.15,
        }))
    assert rep["verdict"] == "FAIL-aim"
    assert "rec 1" in rep["next_step"]
    assert rep["agg"]["sigma_move"] == pytest.approx(0.15)


def test_untrained_route_no_kills_no_sigma_movement(tmp_path):
    rep = _read(tmp_path, _rows(window_over={"policy/aim_log_std_yaw_raw": INIT - 0.05}))
    assert rep["verdict"] == "FAIL-aim-head-untrained"
    assert "rec 6" in rep["next_step"]


def test_sigma_capped_route(tmp_path):
    """raw above the cap ⇒ the movement routing is void: SIGMA-CAPPED, kills
    alone, next step rec 4(b). NOT a SMOKE INVALID."""
    rep = _read(tmp_path, _rows(window_over={"policy/aim_log_std_yaw_raw": CAP + 0.5}))
    assert not _check(rep, 5)["ok"] and rep["sigma_capped"] is True
    assert rep["invalid"] == []
    assert rep["verdict"].startswith("FAIL SIGMA-CAPPED")
    assert "rec 4(b)" in rep["next_step"]


def test_sigma_capped_does_not_block_a_pass(tmp_path):
    # The PASS rule never mentions sigma; a capped sigma with real kills still PASSes.
    rep = _read(
        tmp_path,
        _rows(
            window_over={
                "policy/aim_log_std_yaw_raw": CAP + 0.5,
                "game/kills_per_episode": 0.6,
                "game/shots_on_target": 5.0,
            }))
    assert rep["sigma_capped"] is True and rep["verdict"] == "PASS"


def test_zero_shots_fired_gives_na_hit_rate_not_a_crash(tmp_path):
    rep = _read(tmp_path, _rows(window_over={"game/shots_fired": 0.0, "game/shots_on_target": 0.0}))
    assert rep["agg"]["hit_rate"] is None
    assert rep["verdict"] == "FAIL-aim-head-untrained"


# ── CLI surface ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("window_over,expected", [
    ({
        "game/kills_per_episode": 0.6,
        "game/shots_on_target": 5.0
    }, 0),
    ({
        "game/kills_per_episode": 0.2,
        "policy/aim_log_std_yaw_raw": INIT - 0.15
    }, 1),
    ({
        "environment/episodes": 0.0
    }, 2),
])
def test_main_exit_codes(tmp_path, capsys, window_over, expected):
    run = _write(tmp_path, _rows(window_over=window_over))
    assert main([str(run)]) == expected
    out = capsys.readouterr().out
    assert out.rstrip().splitlines()[-1].startswith("VERDICT: ")
    assert "pre-flight 1" in out and "pre-flight 5" in out


def test_script_runs_as_a_subprocess(tmp_path):
    """stdlib-only: the reader must run with no third-party import available."""
    run = _write(tmp_path, _rows())
    script = Path(__file__).resolve().parent.parent / "scripts" / "rung1a_smoke_read.py"
    proc = subprocess.run([sys.executable, str(script), str(run)],
                          capture_output=True,
                          text=True,
                          check=False)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "VERDICT: FAIL-aim-head-untrained" in proc.stdout
