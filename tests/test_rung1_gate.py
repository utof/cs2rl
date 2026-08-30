"""scripts/rung1_gate.py — spec 2026-08-29 §5 gate arithmetic on synthetic metrics rows.

Every number below is hand-derivable from GOOD (see the comment on it); if a
threshold in spec §5 changes, change rung1_gate.GATES and these expectations
together.

PITFALL (Task 15 ruling): the fixture config.json carries BOTH
`total_timesteps = 5 * PT` (what PufferLib receives at n_active=1) and
`participating_timesteps = PT`. The window must be computed on the
participating budget — a gate keyed on total_timesteps would see W = ∅ on
every real run and report every seed as a structural failure.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from rung1_gate import gate_report, main, seed_metrics                 # noqa: E402

PT = 1000.0                            # participating_timesteps (config.json) → window starts at agent_steps >= 900


def _row(step,
         episodes,
         fired,
         facing,
         on_target,
         hit,
         kills,
         used_past=0.0,
         eval_t=None,
         eval_ct=None):
    """One metrics.jsonl row in the shape train.py writes (`step`/`agent_steps`
    via pufferl logs, compute_game_metrics `game/*` window means, R0-A
    `environment/episodes`, Task 13 `eval/*`, self_play/used_past)."""
    r = {
        "run_id": "r",
        "step": step,
        "agent_steps": step,
        "environment/episodes": episodes,
        "game/shots_fired": fired,
        "game/shots_facing_enemy": facing,
        "game/shots_on_target": on_target,
        "game/shots_hit": hit,
        "game/kills_per_episode": kills,
        "self_play/used_past": used_past
    }
    if eval_t is not None:
        r["eval/win_vs_random_as_t"] = eval_t
        r["eval/win_vs_random_as_ct"] = eval_ct
    return r


# Steps are distinct except the deliberate replay: dedupe_resume_rows keys on
# (run_id, step), so two synthetic rows sharing a step would collapse to one.
# W after dedupe = rows 900 (ep 10), 950 (ep 30), 1000-replay (ep 10):
#   episodes 50; kills (10·1 + 30·0.8 + 10·1)/50 = 0.88
#   shots_fired (200 + 600 + 400)/50 = 24
#   hit/facing (80 + 240 + 160)/(160 + 480 + 320) = 480/960 = 0.5
#   facing/fired 960/(200 + 600 + 400) = 0.8;  on_target/facing (100+300+200)/960 = 0.625
#   hit/on_target 480/600 = 0.8
#   eval = LAST row in W carrying the key = the replayed step-1000 row → 0.9 / 0.9
GOOD = [
    _row(850, 10, 20, 18, 12, 10, 1.0, eval_t=0.5, eval_ct=0.5),       # < 0.9·PT — excluded
    _row(920, 10, 20, 16, 10, 8, 1.0, used_past=0.5),                  # past-policy row — excluded
    _row(900, 10, 20, 16, 10, 8, 1.0, eval_t=0.95, eval_ct=0.90),
    _row(950, 30, 20, 16, 10, 8, 0.8),
    _row(1000, 10, 20, 16, 10, 8, 1.0, eval_t=1.0, eval_ct=0.95),
    _row(1000, 10, 40, 32, 20, 16, 1.0, eval_t=0.9, eval_ct=0.9),      # R0-C replay: last wins
]


def test_seed_metrics_window_dedupe_and_episode_weighting():
    m = seed_metrics(GOOD, PT)
    assert "fail" not in m
    assert m["rows"] == 3
    assert m["episodes"] == pytest.approx(50.0)
    assert m["kills_per_episode"] == pytest.approx(0.88)
    assert m["shots_fired"] == pytest.approx(24.0)
    assert m["hit_per_facing"] == pytest.approx(0.5)
    assert m["facing_per_fired"] == pytest.approx(0.8)
    assert m["on_target_per_facing"] == pytest.approx(0.625)
    assert m["hit_per_on_target"] == pytest.approx(0.8)
    assert m["eval/win_vs_random_as_t"] == pytest.approx(0.9)
    assert m["eval/win_vs_random_as_ct"] == pytest.approx(0.9)


def test_seed_metrics_missing_used_past_key_counts_as_zero():
    rows = [{k: v for k, v in r.items() if k != "self_play/used_past"} for r in GOOD]
    assert seed_metrics(rows, PT)["rows"] == 4         # the used_past=0.5 row is now in W


def test_seed_metrics_window_is_participating_not_total():
    """Same rows, window computed on 5·PT: everything is below 0.9·5000 → structural fail."""
    m = seed_metrics(GOOD, 5 * PT)
    assert "fail" in m and "rows in W" in m["fail"]


@pytest.mark.parametrize(
    "rows, why",
    [
        (GOOD[:4], "rows in W"),                                       # only 2 rows ≥ 900
        ([r for r in GOOD if "eval/win_vs_random_as_t" not in r] +
         [_row(990, 10, 20, 16, 10, 8, 1.0),
          _row(995, 10, 20, 16, 10, 8, 1.0)], "no eval row"),          # W = 950, 990, 995
        ([_row(900 + i, 10, 5, 4, 3, 2, 0.1, eval_t=1.0, eval_ct=1.0)
          for i in range(3)], "shots_fired"),                          # 5 < 10
        ([_row(900 + i, 10, 20, 0, 0, 0, 0.0, eval_t=1.0, eval_ct=1.0)
          for i in range(3)], "zero denominator"),                     # shots_facing_enemy == 0
    ])
def test_seed_metrics_structural_failures(rows, why):
    m = seed_metrics(rows, PT)
    assert "fail" in m and why in m["fail"], m


def _write_run(run_dir, rows):
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({
            "total_timesteps": 5 * PT,                 # raw budget PufferLib sees at n_active=1
            "participating_timesteps": PT,             # the --timesteps the gate must key on
        }))
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _neg(hit):                         # negative control: same shape, hit count per row set explicitly
    return [_row(900 + 50 * i, 10, 20, 16, 10, hit, 0.2, eval_t=0.3, eval_ct=0.3) for i in range(3)]


def test_gate_report_pass_and_median_with_failed_seeds_as_zero(tmp_path):
    for s in range(5):
        _write_run(tmp_path / f"rung1-s{s}", GOOD)
    for s in range(2):
        _write_run(tmp_path / f"rung1-neg-s{s}", _neg(2))              # 2/16 = 0.125
    rep = gate_report(tmp_path, seeds=range(5), neg_seeds=range(2))
    assert rep["verdict"] == "PASS"
    assert rep["median"]["hit_per_facing"] == pytest.approx(0.5)
    assert rep["control_hit_per_facing"] == pytest.approx(0.125)
                                                                       # Two crashed seeds (< 3 rows in W) enter the median as 0.0 and still lose the vote 3:2 …
    for s in (1, 3):
        d = tmp_path / f"rung1-s{s}"
        (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in GOOD[:4]))
    rep = gate_report(tmp_path, seeds=range(5), neg_seeds=range(2))
    assert rep["verdict"] == "PASS" and "fail" in rep["treatment"][1]
                                                                       # … three do not.
    (tmp_path / "rung1-s0" / "metrics.jsonl").write_text("".join(
        json.dumps(r) + "\n" for r in GOOD[:4]))
    rep = gate_report(tmp_path, seeds=range(5), neg_seeds=range(2))
    assert rep["verdict"] == "FAIL" and rep["median"]["hit_per_facing"] == 0.0


def test_gate_report_negative_control_invalidates(tmp_path):
    for s in range(5):
        _write_run(tmp_path / f"rung1-s{s}", GOOD)
    for s in range(2):
        _write_run(tmp_path / f"rung1-neg-s{s}", _neg(7.5))            # 7.5/16 = 0.469 ≥ 0.45
    rep = gate_report(tmp_path, seeds=range(5), neg_seeds=range(2))
    assert rep["verdict"] == "INVALID"
    assert all(rep["ok"].values())                                     # treatment alone would pass


def test_gate_report_missing_seed_dir_is_structural_failure(tmp_path):
    for s in range(4):                 # rung1-s4 never ran
        _write_run(tmp_path / f"rung1-s{s}", GOOD)
    rep = gate_report(tmp_path, seeds=range(5), neg_seeds=())
    assert "fail" in rep["treatment"][4] and "FileNotFoundError" in rep["treatment"][4]["fail"]
    assert rep["verdict"] == "PASS"    # 4:1 still carries the median
    assert rep["control_hit_per_facing"] is None and rep["control_ok"]


def test_main_exit_status_and_verdict_line(tmp_path, capsys):
    for s in range(5):
        _write_run(tmp_path / f"rung1-s{s}", GOOD)
    assert main([str(tmp_path), "--neg-seeds"]) == 0   # no control → PASS
    assert "VERDICT: PASS" in capsys.readouterr().out
    (tmp_path / "rung1-s2" / "metrics.jsonl").write_text("")
    (tmp_path / "rung1-s3" / "metrics.jsonl").write_text("")
    (tmp_path / "rung1-s4" / "metrics.jsonl").write_text("")
    assert main([str(tmp_path), "--neg-seeds"]) == 1
    out = capsys.readouterr().out
    assert "VERDICT: FAIL" in out and "FAIL: only 0 rows in W" in out


def test_script_is_runnable_as_cli(tmp_path):
    """`uv run python scripts/rung1_gate.py <dir>` — the documented invocation."""
    gate = Path(__file__).resolve().parent.parent / "scripts" / "rung1_gate.py"
    r = subprocess.run(
        [sys.executable, str(gate),
         str(tmp_path), "--seeds", "0", "--neg-seeds"],
        capture_output=True,
        text=True,
        timeout=120)
    assert r.returncode == 1, r.stderr
    assert "VERDICT: FAIL" in r.stdout and "FileNotFoundError" in r.stdout
