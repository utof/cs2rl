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
import json
import os
import shutil
import subprocess
import sys
import time
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


# ── Main entry point ───────────────────────────────────────────────────────


def _python_cmd(train_py: Path) -> list[str]:
    """Pick interpreter: in test mode (CS2RL_REPO_ROOT set) use sys.executable,
    else use `uv run python` so project deps are resolved via uv.

    Why the split: production runs need torch/pufferlib, which live in the uv
    environment. Tests, however, run against a fake repo tmp_path with no
    pyproject.toml — `uv run` would fail there. The env var CS2RL_REPO_ROOT is
    already the test-mode signal the script uses everywhere else, so we reuse
    it as the "use sys.executable directly" flag.
    """
    if os.environ.get("CS2RL_REPO_ROOT"):
        return [sys.executable, str(train_py)]
    return ["uv", "run", "python", str(train_py)]


def _read_baseline() -> str | None:
    """Read baseline run_id from outputs/experiments/baseline.txt if present.

    Returns None when the file is absent or empty. The baseline file is
    managed by promote_baseline.py; missing = no baseline comparison.
    """
    p = EXPERIMENTS_DIR / "baseline.txt"
    if not p.exists():
        return None
    text = p.read_text().strip()
    return text or None


def _append_ledger(run_dir: Path) -> None:
    """Append (or upsert) the run's summary.json row into results.jsonl.

    No-ops if summary.json is missing (caller should have already written a
    bug-case summary in that situation). Uses exp_lib.ledger_upsert for
    atomicity and single-row-per-run_id guarantee.
    """
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return
    row = json.loads(summary_path.read_text())
    exp_lib.ledger_upsert(LEDGER_PATH, row)


def _write_bug_summary(run_dir: Path, spec_json: dict, start_wall: float, reason: str) -> None:
    """Minimal summary for catastrophic early-failure cases.

    Called when training crashes or an exception escapes before the analyzer
    can run. Marks verdict=bug so the ledger reflects the failure, and keeps
    the schema compatible with normal summaries (downstream readers don't
    need to special-case).
    """
    summary = {
        **spec_json,
        "timestamp": exp_lib.utc_now_iso(),
        "wall_time_sec": int(time.time() - start_wall),
        "sample_count": 0,
        "terminal_metrics": {},
        "trend": "insufficient_samples",
        "stability": {
            "max_kl": 0,
            "max_clipfrac": 0,
            "any_nan": False,
            "dead_run_triggered": False
        },
        "delta_vs_baseline": {},
        "verdict": "bug",
        "notes": f"FILL_ME (auto-set bug; reason={reason})",
        "analysis_path": str(run_dir / "analysis.md"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2))


def _truncate_log(log_path: Path, max_bytes: int) -> None:
    """Truncate a log file to head+tail if it exceeds max_bytes.

    Preserves the first and last quarter (so we see both the startup banner
    and the tail with any crash output), joined by a TRUNCATED marker. No-ops
    if the file is absent or already under budget.
    """
    if not log_path.exists() or log_path.stat().st_size <= max_bytes:
        return
    data = log_path.read_bytes()
    head = data[:max_bytes // 4]
    tail = data[-max_bytes // 4:]
    log_path.write_bytes(head + b"\n...TRUNCATED...\n" + tail)


def main_run(args) -> int:
    changed_files = [f for f in args.changed_files.split() if f]
    check_preconditions(changed_files)
    acquire_lock()

    run_id: str | None = None
    run_dir: Path | None = None
    original_branch: str | None = None
    start_wall = time.time()
    ledger_appended = False
    spec_json: dict = {}

    try:
        original_branch = _current_branch()
        run_id = exp_lib.resolve_run_id(args.tag, checkpoints_root=CHECKPOINTS_DIR)
        ckpt_dir = CHECKPOINTS_DIR / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        run_dir = EXPERIMENTS_DIR / run_id
        if run_dir.exists():
            raise PreconditionError(f"experiment dir already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        exp_lib.write_status(run_dir, "started")
        (run_dir / "spec.md").write_text(f"# {run_id}\n\n"
                                         f"**Hypothesis:** {args.hypothesis}\n\n"
                                         f"**Change type:** {args.change_type}\n\n"
                                         f"**Changed files:** {changed_files}\n\n"
                                         f"**Budget steps:** {args.timesteps}\n\n"
                                         f"**Resume from:** {args.resume or '(none)'}\n")

        train_py = REPO_ROOT / "src" / "train.py"
        rewards_h = REPO_ROOT / "src" / "c_env" / "cs2_rewards.h"
        env_c = REPO_ROOT / "src" / "c_env" / "cs2_env.c"
        c_env_sha = exp_lib.path_last_commit_sha(REPO_ROOT / "src" / "c_env")
        train_py_sha = exp_lib.path_last_commit_sha(train_py)

        dump_cmd = _python_cmd(train_py) + [
            "--dump-config",
            "--checkpoint-dir",
            str(ckpt_dir),
        ]
        dump_r = subprocess.run(
            dump_cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if dump_r.returncode != 0:
            raise RuntimeError(f"--dump-config failed: {dump_r.stderr}")
        cfg_hash = exp_lib.config_hash(ckpt_dir / "config.json")

        env_fp = exp_lib.env_fingerprint(
            train_py_path=train_py,
            rewards_h_path=rewards_h,
            env_c_path=env_c,
        )
        env_fp["c_env_sha"] = c_env_sha
        env_fp["train_py_sha"] = train_py_sha
        env_fp["train_config_hash"] = cfg_hash
        env_fp["behavior_hash"] = exp_lib.behavior_hash(env_fp, cfg_hash)

        spec_json = {
            "run_id": run_id,
            "hypothesis": args.hypothesis,
            "change_type": args.change_type,
            "change_summary": f"{args.change_type}: see spec.md",
            "baseline_run_id": _read_baseline(),
            "commit": "",
            "branch": "",
            "budget_steps": args.timesteps,
            "wall_time_sec": 0,
            "env_fingerprint": env_fp,
        }

        exp_branch = f"exp/{run_id}"
        subprocess.run(["git", "checkout", "-b", exp_branch],
                       cwd=REPO_ROOT,
                       check=True,
                       capture_output=True)

        if any(f.startswith("src/c_env/") for f in changed_files):
            exp_lib.write_status(run_dir, "building")
            zig_r = subprocess.run(
                ["uv", "run", "--no-sync", "zig", "build"],
                cwd=REPO_ROOT / "src" / "c_env",
                capture_output=True,
                text=True,
            )
            if zig_r.returncode != 0:
                sync_r = subprocess.run(
                    ["uv", "sync"],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                )
                if sync_r.returncode != 0:
                    raise RuntimeError(
                        f"zig build failed: {zig_r.stderr}\nuv sync also failed: {sync_r.stderr}")
        exp_lib.write_status(run_dir, "built")

        if changed_files:
            subprocess.run(["git", "add", *changed_files],
                           cwd=REPO_ROOT,
                           check=True,
                           capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", f"exp({run_id}): {args.hypothesis}"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
            )
        commit_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        (run_dir / "commit.txt").write_text(commit_sha)
        spec_json["commit"] = commit_sha
        spec_json["branch"] = exp_branch

        # Step 9: Train — use --checkpoint_dir (not --name) to bypass
        # resolve_run_name's double-prefix logic inside train.py.
        exp_lib.write_status(run_dir, "training")
        train_log = run_dir / "train.log"
        train_cmd = _python_cmd(train_py) + [
            "--train",
            "--timesteps",
            str(args.timesteps),
            "--checkpoint_dir",
            str(ckpt_dir),
        ]
        if args.resume:
            train_cmd.extend(["--resume", args.resume])
        with train_log.open("w") as logf:
            train_r = subprocess.run(
                train_cmd,
                cwd=REPO_ROOT,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
        if train_r.returncode != 0:
            exp_lib.write_status(run_dir, "failed", reason="training_crashed")
            _write_bug_summary(run_dir, spec_json, start_wall, reason="training_crashed")
            _append_ledger(run_dir)
            ledger_appended = True
            return 1

        _truncate_log(train_log, max_bytes=10 * 1024 * 1024)

        src_metrics = ckpt_dir / "metrics.jsonl"
        if src_metrics.exists():
            shutil.copy(src_metrics, run_dir / "metrics.jsonl")

        exp_lib.write_status(run_dir, "analyzing")
        spec_json["wall_time_sec"] = int(time.time() - start_wall)
        (run_dir / "spec.json").write_text(json.dumps(spec_json, sort_keys=True, indent=2))
        analyzer = Path(__file__).parent / "analyze_experiment.py"
        ar = subprocess.run(
            [sys.executable,
             str(analyzer), run_id, "--experiments-root",
             str(EXPERIMENTS_DIR)],
            capture_output=True,
            text=True,
        )
        if ar.returncode != 0:
            raise RuntimeError(f"analyzer failed: {ar.stderr}")

        _append_ledger(run_dir)
        ledger_appended = True

        exp_lib.write_status(run_dir, "done")

        summary_text = (run_dir / "summary.json").read_text()
        print(summary_text)
        return 0

    except PreconditionError:
        raise
    except Exception as e:
        if run_dir is not None:
            exp_lib.write_status(run_dir, "failed", reason=str(e)[:200])
            if not ledger_appended:
                try:
                    _write_bug_summary(run_dir, spec_json, start_wall, reason=str(e)[:200])
                    _append_ledger(run_dir)
                    ledger_appended = True
                except Exception as le:
                    print(f"WARN: failed to append bug row to ledger: {le}", file=sys.stderr)
        print(f"EXPERIMENT FAILED: {e}", file=sys.stderr)
        return 1
    finally:
        if original_branch is not None:
            try:
                current = _current_branch()
                if current != original_branch:
                    subprocess.run(["git", "checkout", original_branch],
                                   cwd=REPO_ROOT,
                                   check=False,
                                   capture_output=True)
            except Exception as ce:
                print(f"WARN: failed to restore branch {original_branch}: {ce}", file=sys.stderr)
            try:
                diff = _working_tree_changes()
                if diff:
                    print(f"WARN: tree not clean after checkout; diff: {diff}", file=sys.stderr)
            except Exception:
                pass
        release_lock()


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
