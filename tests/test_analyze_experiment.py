"""Unit tests for scripts/analyze_experiment.py."""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
ANALYZER = REPO / "scripts" / "analyze_experiment.py"


def _make_metrics(run_dir: Path, rows: list[dict]) -> None:
    metrics = run_dir / "metrics.jsonl"
    metrics.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _make_spec(run_dir: Path, **kwargs) -> None:
    defaults = {
        "run_id": run_dir.name,
        "hypothesis": "test hypothesis",
        "change_type": "hyperparameter",
        "change_summary": "test change",
        "baseline_run_id": None,
        "commit": "abc1234",
        "branch": f"exp/{run_dir.name}",
        "budget_steps": 2_000_000,
        "wall_time_sec": 100,
        "env_fingerprint": {
            "obs_dim": 104,
            "action_head_sizes": [9, 16, 2, 2, 3, 2, 2, 2],
            "reward_terms": ["a", "b"],
            "c_env_sha": "def",
            "train_py_sha": "abc",
            "train_config_hash": "hhh",
            "behavior_hash": "bhash",
        },
    }
    defaults.update(kwargs)
    (run_dir / "spec.json").write_text(json.dumps(defaults))


def _run_analyzer(run_id: str, experiments_root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable,
         str(ANALYZER), run_id, "--experiments-root",
         str(experiments_root)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_analyze_insufficient_samples(tmp_path):
    """sample_count < 8 → verdict=inconclusive, trend=insufficient_samples."""
    exp_root = tmp_path / "experiments"
    run_dir = exp_root / "150426-0-test"
    run_dir.mkdir(parents=True)
    _make_spec(run_dir)
    rows = [{
        "step": i * 163_840,
        "epoch": i,
        "winner_t": 0.5,
        "winner_ct": 0.5,
        "approx_kl": 0.01
    } for i in range(5)]
    _make_metrics(run_dir, rows)

    r = _run_analyzer("150426-0-test", exp_root)
    assert r.returncode == 0, r.stderr

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["sample_count"] == 5
    assert summary["trend"] == "insufficient_samples"
    assert summary["verdict"] == "inconclusive"


def test_analyze_detects_nan(tmp_path):
    """NaN in any metric → verdict=bug, any_nan=true."""
    exp_root = tmp_path / "experiments"
    run_dir = exp_root / "150426-0-nan"
    run_dir.mkdir(parents=True)
    _make_spec(run_dir)
    rows = [{
        "step": i * 163_840,
        "epoch": i,
        "winner_t": 0.5,
        "approx_kl": 0.01
    } for i in range(10)]
    rows[7]["winner_t"] = float("nan")
    _make_metrics(run_dir, rows)

    r = _run_analyzer("150426-0-nan", exp_root)
    assert r.returncode == 0

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["stability"]["any_nan"] is True
    assert summary["verdict"] == "bug"


def test_analyze_missing_metrics(tmp_path):
    """Missing metrics.jsonl → verdict=bug, sample_count=0."""
    exp_root = tmp_path / "experiments"
    run_dir = exp_root / "150426-0-missing"
    run_dir.mkdir(parents=True)
    _make_spec(run_dir)

    r = _run_analyzer("150426-0-missing", exp_root)
    assert r.returncode == 0

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["verdict"] == "bug"
    assert summary["sample_count"] == 0


def test_analyze_writes_skeleton_sentinel(tmp_path):
    """analysis.md gets written with the skeleton sentinel on first line."""
    exp_root = tmp_path / "experiments"
    run_dir = exp_root / "150426-0-skel"
    run_dir.mkdir(parents=True)
    _make_spec(run_dir)
    _make_metrics(run_dir, [{
        "step": i * 163_840,
        "epoch": i,
        "winner_t": 0.5,
        "approx_kl": 0.01
    } for i in range(10)])

    r = _run_analyzer("150426-0-skel", exp_root)
    assert r.returncode == 0

    content = (run_dir / "analysis.md").read_text()
    assert content.startswith("<!-- ANALYZER_SKELETON -->")
    assert "What was tested" in content
    assert "Verdict" in content


def test_analyze_idempotent_preserves_edited_analysis(tmp_path):
    """Re-running the analyzer does NOT overwrite analysis.md if sentinel was removed."""
    exp_root = tmp_path / "experiments"
    run_dir = exp_root / "150426-0-idem"
    run_dir.mkdir(parents=True)
    _make_spec(run_dir)
    _make_metrics(run_dir, [{
        "step": i * 163_840,
        "epoch": i,
        "winner_t": 0.5,
        "approx_kl": 0.01
    } for i in range(10)])

    _run_analyzer("150426-0-idem", exp_root)
    analysis_path = run_dir / "analysis.md"
    original = analysis_path.read_text()
    edited = original.replace("<!-- ANALYZER_SKELETON -->\n", "") + "\n\nMy edit.\n"
    analysis_path.write_text(edited)

    _run_analyzer("150426-0-idem", exp_root)
    assert analysis_path.read_text() == edited


def test_analyze_never_touches_results_jsonl(tmp_path):
    """analyzer does NOT append to or modify results.jsonl."""
    exp_root = tmp_path / "experiments"
    run_dir = exp_root / "150426-0-noledger"
    run_dir.mkdir(parents=True)
    _make_spec(run_dir)
    _make_metrics(run_dir, [{
        "step": i * 163_840,
        "epoch": i,
        "winner_t": 0.5,
        "approx_kl": 0.01
    } for i in range(10)])
    ledger = exp_root / "results.jsonl"
    ledger.write_text('{"run_id":"other","verdict":"keep"}\n')
    mtime_before = ledger.stat().st_mtime_ns

    _run_analyzer("150426-0-noledger", exp_root)

    assert ledger.stat().st_mtime_ns == mtime_before
    assert ledger.read_text() == '{"run_id":"other","verdict":"keep"}\n'
