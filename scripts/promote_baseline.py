#!/usr/bin/env python
"""Promote a run to current baseline. USER-ONLY — reads confirmation from /dev/tty.

Any agent (PI, subagent) invoking this will fail because /dev/tty isn't available
in non-interactive contexts. Additionally add this script to the deny list in
`.claude/settings.local.json`.

Usage:
  uv run python scripts/promote_baseline.py <run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent.parent / "outputs" / "experiments"


def require_tty_confirmation(prompt: str, expected: str) -> bool:
    """Read confirmation from /dev/tty directly.

    Fails if /dev/tty isn't available (piped, heredoc, non-interactive). This
    is the first line of defense against an agent driving the promotion —
    stdin redirection can't satisfy a /dev/tty read, so `echo CONFIRM | ...`
    or `Bash(... <<< CONFIRM)` both fail here. Pair this with the bash deny
    list in .claude/settings.local.json for belt-and-braces.
    """
    try:
        with open("/dev/tty") as tty_in, open("/dev/tty", "w") as tty_out:
            tty_out.write(prompt)
            tty_out.flush()
            line = tty_in.readline().strip()
            return line == expected
    except OSError:
        print(
            "ERROR: /dev/tty is not available. This script requires an interactive "
            "terminal — it cannot be driven by an agent or a pipe.",
            file=sys.stderr,
        )
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    args = p.parse_args()

    run_dir = EXPERIMENTS_DIR / args.run_id
    if not run_dir.exists():
        print(f"run dir not found: {run_dir}", file=sys.stderr)
        return 2
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        print(f"summary.json not found: {summary_path}", file=sys.stderr)
        return 2
    summary = json.loads(summary_path.read_text())
    verdict = summary.get("verdict")
    if verdict != "keep":
        # Refuse to promote anything not explicitly marked "keep" — silently promoting
        # a bug/inconclusive run would poison every subsequent --resume flow.
        print(f"refusing to promote — verdict={verdict!r} (must be 'keep')", file=sys.stderr)
        return 2

    baseline_path = EXPERIMENTS_DIR / "baseline.txt"
    current = baseline_path.read_text().strip() if baseline_path.exists() else "(none)"

    prompt = (f"Promote {args.run_id} to baseline?\n"
              f"Current baseline: {current}\n"
              f"Type 'CONFIRM {args.run_id}' (no quotes) to proceed: ")
    if not require_tty_confirmation(prompt, f"CONFIRM {args.run_id}"):
        print("aborted", file=sys.stderr)
        return 1

    baseline_path.write_text(f"{args.run_id}\n")
    print(f"promoted: baseline.txt = {args.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
