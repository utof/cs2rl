"""Tests for scripts/reconcile_experiments.py."""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
RECONCILE = REPO / "scripts" / "reconcile_experiments.py"


def _run(exp_root: Path, *args: str, input_text: str = "") -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RECONCILE), "--experiments-root",
         str(exp_root), *args],
        capture_output=True,
        text=True,
        timeout=10,
        input=input_text,
    )


def test_list_stale_runs(tmp_path):
    """--list shows runs with in-progress STATUS."""
    exp_root = tmp_path / "experiments"
    exp_root.mkdir()
    (exp_root / "results.jsonl").touch()

    cases = [("training", "150426-0-a"), ("done", "150426-1-b"), ("analyzing", "150426-2-c")]
    for status, name in cases:
        d = exp_root / name
        d.mkdir()
        (d / "STATUS.txt").write_text(status)

    r = _run(exp_root, "--list")
    assert r.returncode == 0
    assert "150426-0-a" in r.stdout
    assert "150426-2-c" in r.stdout
    assert "150426-1-b" not in r.stdout


def test_orphan_ledger_rows_detected(tmp_path):
    """--list-orphans shows run dirs with STATUS=done but no ledger row."""
    exp_root = tmp_path / "experiments"
    exp_root.mkdir()
    (exp_root / "results.jsonl").write_text(json.dumps({"run_id": "150426-0-tracked"}) + "\n")
    for rid in ("150426-0-tracked", "150426-1-orphan"):
        d = exp_root / rid
        d.mkdir()
        (d / "STATUS.txt").write_text("done")
        (d / "summary.json").write_text(json.dumps({"run_id": rid, "verdict": "keep"}))

    r = _run(exp_root, "--list-orphans")
    assert r.returncode == 0
    assert "150426-1-orphan" in r.stdout
    assert "150426-0-tracked" not in r.stdout


def test_stale_lock_detected(tmp_path):
    exp_root = tmp_path / "experiments"
    exp_root.mkdir()
    (exp_root / ".lock").write_text("99999999 old")
    r = _run(exp_root, "--list-locks")
    assert r.returncode == 0
    assert "lock" in r.stdout.lower()


def test_mark_failed_updates_status_and_appends_ledger(tmp_path):
    exp_root = tmp_path / "experiments"
    exp_root.mkdir()
    (exp_root / "results.jsonl").touch()
    rid = "150426-0-kill"
    d = exp_root / rid
    d.mkdir()
    (d / "STATUS.txt").write_text("training")

    r = _run(exp_root, "--mark-failed", rid, "--reason", "manual_cleanup")
    assert r.returncode == 0

    status = (d / "STATUS.txt").read_text()
    assert status.startswith("failed")
    assert "manual_cleanup" in status

    ledger_rows = [
        json.loads(ln) for ln in (exp_root / "results.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    assert any(r.get("run_id") == rid and r.get("verdict") == "bug" for r in ledger_rows)
