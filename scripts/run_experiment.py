#!/usr/bin/env python
"""Run one RL experiment end-to-end. See spec §5.1.

Usage:
  run_experiment.py --tag <short-tag> --hypothesis "<1-line>" \\
                    --change-type <hp|reward|arch|env|monitoring> \\
                    --changed-files "<space-sep paths>" \\
                    [--timesteps N] [--resume <ckpt>]

  run_experiment.py --reappend-ledger <run_id>
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import exp_lib

# Repo root is parameterizable via CS2RL_REPO_ROOT env var so tests can point
# this whole script at a fake repo. Without the override, falls back to the
# real repo (script's parent-of-parent). ALL other paths derive from this, so
# tests that set the env var redirect every filesystem + git operation.
REPO_ROOT = Path(os.environ.get("CS2RL_REPO_ROOT", Path(__file__).parent.parent)).resolve()
EXPERIMENTS_DIR = REPO_ROOT / "outputs" / "experiments"
CHECKPOINTS_DIR = REPO_ROOT / "outputs" / "checkpoints"
LOCK_PATH = EXPERIMENTS_DIR / ".lock"
LEDGER_PATH = EXPERIMENTS_DIR / "results.jsonl"

DEFAULT_TIMESTEPS = 2_000_000
MIN_FREE_DISK_GB = 5


class PreconditionError(RuntimeError):
    pass


# ── Preconditions ─────────────────────────────────────────────────────────


def _current_branch() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return r.stdout.strip()


def _working_tree_changes() -> list[str]:
    """Return list of tracked files changed vs HEAD (includes both staged + unstaged).

    Uses `git diff --name-only HEAD` rather than plain `git diff` because the
    subagent may have staged its edits already; plain `git diff` only shows
    unstaged, which would miss staged changes and fail the declared-files check.
    Gitignored paths are NOT in this output by git's design.
    """
    r = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def _disk_free_gb(path: Path) -> float:
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)


def check_preconditions(changed_files: list[str]) -> None:
    """Raise PreconditionError if any precondition fails."""
    branch = _current_branch()
    if branch != "main":
        raise PreconditionError(f"not on main (current: {branch})")

    if LOCK_PATH.exists():
        raise PreconditionError(
            f"lock file present: {LOCK_PATH} — another run in progress or stale")

    diff = _working_tree_changes()
    declared = set(changed_files)
    actual = set(diff)
    if declared != actual:
        extra = actual - declared
        missing = declared - actual
        msg = ("--changed-files mismatch vs git diff (tracked files only). "
               f"declared={sorted(declared)} actual={sorted(actual)}")
        if extra:
            msg += f" UNDECLARED CHANGES: {sorted(extra)}"
        if missing:
            msg += f" MISSING: {sorted(missing)}"
        raise PreconditionError(msg)

    free = _disk_free_gb(REPO_ROOT)
    if free < MIN_FREE_DISK_GB:
        raise PreconditionError(f"disk too low: {free:.1f} GB free, need ≥ {MIN_FREE_DISK_GB}")


# ── Lock file ─────────────────────────────────────────────────────────────


def acquire_lock() -> None:
    """Atomic-create lock file. Fails if already exists."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with LOCK_PATH.open("x") as f:                 # x = exclusive create
            f.write(f"{os.getpid()} {exp_lib.utc_now_iso()}\n")
    except FileExistsError as err:
        raise PreconditionError(f"lock file already exists: {LOCK_PATH}") from err


def release_lock() -> None:
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


# ── Main entry point (skeleton — full flow in Task 6) ──────────────────────


def main_run(args) -> int:
    changed_files = [f for f in args.changed_files.split() if f]
    check_preconditions(changed_files)

    # The rest of the flow — run_id resolve, fingerprint, branch, train,
    # analyze, ledger — comes in Task 6.
    # For now, if we reach here, preconditions passed.

    if os.environ.get("CS2RL_FAKE_TRAIN") == "1":
        # Test harness — exit cleanly without actually running anything.
        return 0

    raise NotImplementedError("full flow implemented in Task 6")


def main_reappend_ledger(args) -> int:
    raise NotImplementedError("reappend-ledger implemented in Task 7")


def main() -> int:
    p = argparse.ArgumentParser()

    p.add_argument("--tag", type=str)
    p.add_argument("--hypothesis", type=str)
    p.add_argument("--change-type", type=str, choices=["hp", "reward", "arch", "env", "monitoring"])
    p.add_argument("--changed-files", type=str, default="")
    p.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    p.add_argument("--resume", type=str, default=None)

    p.add_argument("--reappend-ledger", type=str, metavar="RUN_ID", default=None)

    args = p.parse_args()

    try:
        if args.reappend_ledger:
            return main_reappend_ledger(args)
        if not all([args.tag, args.hypothesis, args.change_type]):
            p.error("--tag, --hypothesis, --change-type are required")
        return main_run(args)
    except PreconditionError as e:
        print(f"PRECONDITION FAILED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
