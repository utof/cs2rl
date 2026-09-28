#!/usr/bin/env python
"""Reconcile stale experiment state.

Detects and helps clean up:
- Run dirs with in-progress STATUS.txt (started/built/training/analyzing)
- Run dirs with STATUS=done but no matching ledger row
- Stale lock files

Offers listing and flagged-action modes; no confirmation prompts in MVP —
user runs these one at a time.

From a worktree, put its own src/ first: `env PYTHONPATH=<checkout>/src python
scripts/reconcile_experiments.py ...` (it imports cs2rl.experiment.lib).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cs2rl.experiment import lib as exp_lib

IN_PROGRESS_STATUSES = {"started", "built", "training", "analyzing"}


def list_stale(exp_root: Path) -> list[tuple[str, str]]:
    """Return [(run_id, status), ...] for run dirs with in-progress status."""
    out = []
    for d in sorted(exp_root.iterdir()):
        if not d.is_dir():
            continue
        status, _ = exp_lib.read_status(d)
        if status in IN_PROGRESS_STATUSES:
            out.append((d.name, status))
    return out


def list_orphans(exp_root: Path) -> list[str]:
    """Return run_ids with STATUS=done but no ledger row."""
    ledger = exp_root / "results.jsonl"
    if not ledger.exists():
        ledger.touch()
    known = {r.get("run_id") for r in exp_lib.ledger_read(ledger)}
    out = []
    for d in sorted(exp_root.iterdir()):
        if not d.is_dir():
            continue
        status, _ = exp_lib.read_status(d)
        if status == "done" and d.name not in known:
            out.append(d.name)
    return out


def list_locks(exp_root: Path) -> list[str]:
    lock = exp_root / ".lock"
    if lock.exists():
        return [f"lock: {lock} contents={lock.read_text().strip()}"]
    return []


def mark_failed(exp_root: Path, run_id: str, reason: str) -> None:
    d = exp_root / run_id
    if not d.exists():
        raise RuntimeError(f"run dir not found: {d}")
    exp_lib.write_status(d, "failed", reason=reason)
    ledger = exp_root / "results.jsonl"
    row = {
        "run_id": run_id,
        "timestamp": exp_lib.utc_now_iso(),
        "verdict": "bug",
        "notes": f"reconciled: {reason}",
    }
    sp = d / "summary.json"
    if sp.exists():
        try:
            existing = json.loads(sp.read_text())
            row = {**existing, **row}
        except json.JSONDecodeError:
            pass
    exp_lib.ledger_upsert(ledger, row)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experiments-root", type=Path, default=Path("outputs/experiments"))
    p.add_argument("--list", action="store_true", help="List stale in-progress runs")
    p.add_argument("--list-orphans", action="store_true", help="List done runs without ledger row")
    p.add_argument("--list-locks", action="store_true", help="Show lock file if present")
    p.add_argument(
        "--mark-failed",
        type=str,
        metavar="RUN_ID",
        help="Mark a run as failed + add bug row",
    )
    p.add_argument(
        "--reason",
        type=str,
        default="manual",
        help="Reason to record with --mark-failed",
    )
    p.add_argument("--remove-lock", action="store_true", help="Delete stale .lock")

    args = p.parse_args()
    root = args.experiments_root

    did_something = False
    if args.list:
        for rid, status in list_stale(root):
            print(f"{rid}\t{status}")
        did_something = True
    if args.list_orphans:
        for rid in list_orphans(root):
            print(rid)
        did_something = True
    if args.list_locks:
        for line in list_locks(root):
            print(line)
        did_something = True
    if args.mark_failed:
        mark_failed(root, args.mark_failed, args.reason)
        print(f"marked {args.mark_failed} as failed (reason={args.reason})")
        did_something = True
    if args.remove_lock:
        lock = root / ".lock"
        if lock.exists():
            lock.unlink()
            print(f"removed {lock}")
        did_something = True

    if not did_something:
        p.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
