"""Shared helpers for the experiment-runner scripts.

Used by: run_experiment.py, analyze_experiment.py, reconcile_experiments.py,
promote_baseline.py.

Deliberately stdlib-only so these helpers start fast. Do NOT import from
src/ (would pull torch + pufferlib and make every CLI call slow).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

# ── Run-ID resolution ──────────────────────────────────────────────────────


def resolve_run_id(tag: str, checkpoints_root: Path) -> str:
    """Return DDMMYY-N-<tag> where N = count of existing dirs with today's prefix.

    Duplicates src/train.py::resolve_run_name so this script doesn't need to
    import the full training stack.
    """
    today_prefix = date.today().strftime("%d%m%y")
    count = 0
    if checkpoints_root.exists():
        search_prefix = f"{today_prefix}-"
        count = sum(1 for d in checkpoints_root.iterdir()
                    if d.is_dir() and d.name.startswith(search_prefix))
    return f"{today_prefix}-{count}-{tag}"


# ── Status file ────────────────────────────────────────────────────────────


def write_status(run_dir: Path, status: str, reason: str | None = None) -> None:
    """Write STATUS.txt in run_dir. Status on line 1, optional reason on line 2."""
    content = status if reason is None else f"{status}\n{reason}"
    (run_dir / "STATUS.txt").write_text(content)


def read_status(run_dir: Path) -> tuple[str, str | None]:
    path = run_dir / "STATUS.txt"
    if not path.exists():
        return ("missing", None)
    lines = path.read_text().splitlines()
    if not lines:
        return ("empty", None)
    return (lines[0], lines[1] if len(lines) >= 2 else None)


# ── Env fingerprint ────────────────────────────────────────────────────────

_OBS_DIM_RE = re.compile(r"^\s*OBS_DIM\s*=\s*(\d+)", re.MULTILINE)
_ACTION_HEAD_RE = re.compile(r"^\s*ACTION_HEAD_SIZES\s*=\s*\(([^)]+)\)", re.MULTILINE)
_REWARD_IDENT_RE = re.compile(r"\b(\w+_(?:reward|penalty|bonus|cost|value))\b")
_DEFINE_RE = re.compile(r"^\s*#define\s+([A-Z][A-Z0-9_]+)", re.MULTILINE)


def env_fingerprint(
    *,
    train_py_path: Path,
    rewards_h_path: Path,
    env_c_path: Path,
) -> dict:
    """Capture the env-shape fingerprint by grepping the named files.

    Returns {"obs_dim", "action_head_sizes", "reward_terms"}. Does NOT populate
    "c_env_sha" — that's the orchestrator's job (see run_experiment.py step 4
    of §5.1 in the design spec), which injects it before calling behavior_hash.
    Kept split so this helper stays pure: regex over file contents, no git.
    """
    train_text = train_py_path.read_text()
    obs_m = _OBS_DIM_RE.search(train_text)
    if obs_m is None:
        raise RuntimeError(f"OBS_DIM not found in {train_py_path}")
    obs_dim = int(obs_m.group(1))

    head_m = _ACTION_HEAD_RE.search(train_text)
    if head_m is None:
        raise RuntimeError(f"ACTION_HEAD_SIZES not found in {train_py_path}")
    action_head_sizes = [int(n.strip()) for n in head_m.group(1).split(",") if n.strip()]

    reward_terms: set[str] = set()
    for p in (rewards_h_path, env_c_path):
        if not p.exists():
            continue
        text = p.read_text()
        reward_terms.update(_REWARD_IDENT_RE.findall(text))
        reward_terms.update(_DEFINE_RE.findall(text))

    if not reward_terms:
        raise RuntimeError(f"Reward-term grep returned empty from {rewards_h_path} + {env_c_path}. "
                           "Grep pattern may be out of date with new env naming.")

    return {
        "obs_dim": obs_dim,
        "action_head_sizes": action_head_sizes,
        "reward_terms": sorted(reward_terms),
    }


def path_last_commit_sha(path: Path) -> str:
    """Return the SHA of the last commit touching `path`. Empty string if untracked."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--",
         str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


# ── Hashing ────────────────────────────────────────────────────────────────


def config_hash(config_path: Path) -> str:
    """sha1 of the sorted JSON dump of the config file contents.

    Reads config.json, re-sorts keys, hashes. Returns 'unknown' if the file is
    missing or malformed — analyzer callers should handle this.
    """
    try:
        data = json.loads(config_path.read_text())
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha1(canonical.encode()).hexdigest()
    except Exception:
        return "unknown"


def behavior_hash(env_fp: dict, train_config_hash: str) -> str:
    """sha1 of config hash + env shape. Primary key for cross-run comparison."""
    payload = {
        "train_config_hash": train_config_hash,
        "obs_dim": env_fp["obs_dim"],
        "action_head_sizes": env_fp["action_head_sizes"],
        "reward_terms": sorted(env_fp["reward_terms"]),
        "c_env_sha": env_fp.get("c_env_sha", ""),
    }
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(canonical.encode()).hexdigest()


# ── Ledger upsert ──────────────────────────────────────────────────────────


def ledger_upsert(ledger_path: Path, row: dict) -> None:
    """Replace the line for row['run_id'] in ledger, or append if not present.

    Atomic: writes to ledger.tmp, renames. Touches the ledger file if missing.
    """
    run_id = row["run_id"]
    existing_lines: list[str] = []
    if ledger_path.exists():
        existing_lines = ledger_path.read_text().splitlines()

    replaced = False
    out_lines: list[str] = []
    for ln in existing_lines:
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            out_lines.append(ln)       # pass through malformed lines untouched
            continue
        if rec.get("run_id") == run_id:
            out_lines.append(json.dumps(row, sort_keys=True))
            replaced = True
        else:
            out_lines.append(ln)
    if not replaced:
        out_lines.append(json.dumps(row, sort_keys=True))

    # Atomic rename: build sibling .tmp path (can't use with_suffix — it strips the
    # existing suffix instead of appending, which breaks for suffix-less names).
    tmp_path = ledger_path.parent / (ledger_path.name + ".tmp")
    tmp_path.write_text("\n".join(out_lines) + "\n")
    os.replace(tmp_path, ledger_path)


def ledger_read(ledger_path: Path) -> list[dict]:
    """Return all rows as dicts. Skips malformed lines."""
    if not ledger_path.exists():
        return []
    rows = []
    for ln in ledger_path.read_text().splitlines():
        if not ln.strip():
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return rows


# ── Misc ───────────────────────────────────────────────────────────────────

SKELETON_SENTINEL = "<!-- ANALYZER_SKELETON -->"


def utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
