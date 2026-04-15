#!/usr/bin/env python
"""Read <run_dir>/metrics.jsonl, write summary.json + skeleton analysis.md.

Never touches results.jsonl — that's run_experiment.py's sole job (see spec §5.2).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import exp_lib

KEY_METRICS = [
    "winner_t",
    "winner_ct",
    "kills_t",
    "kills_ct",
    "bomb_planted",
    "approx_kl",
    "clipfrac",
    "ret_mean",
    "explained_variance",
    "SPS",
]

MIN_SAMPLES_FOR_TREND = 8

SKELETON = """<!-- ANALYZER_SKELETON -->
# Analysis: {run_id}

## What was tested
(1 sentence)

## What actually happened
(2–3 sentences with specific numbers from summary.json terminal_metrics)

## Verdict
(keep / discard / inconclusive / bug — with one-line reason)

## Generalization
(free-text, 1–2 sentences: will this survive env changes on the roadmap? why?)

## Next hypotheses
- (up to 3 bullets for the queue)

## Optional: Causal narrative
(only if the result required multi-step debugging or reveals a surprising interaction)
"""


def _linreg_slope(xs: list[float], ys: list[float]) -> float:
    """Simple OLS slope. Falls back to index-based xs when xs are degenerate.

    Degenerate xs = all identical (e.g., logger writes epoch=0 repeatedly). In that
    case we use sample index as x, which still lets us detect monotonic trends in y.
    Returns 0.0 only when there's genuinely nothing to regress (n < 2, or ys constant).
    """
    n = len(xs)
    if n < 2:
        return 0.0
    x_mean = sum(xs) / n
    den = sum((x - x_mean)**2 for x in xs)
    if den == 0:
        xs = list(range(n))
        x_mean = sum(xs) / n
        den = sum((x - x_mean)**2 for x in xs)
        if den == 0:
            return 0.0
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    return num / den


def _classify_slope(ys: list[float], slope: float) -> str:
    """Classify a trend as improving/degrading/stable based on slope magnitude
    relative to the signal's own scale."""
    if not ys:
        return "stable"
    y_range = max(ys) - min(ys)
    if y_range == 0:
        return "stable"
    swing_ratio = abs(slope * len(ys)) / (y_range + 1e-9)
    if swing_ratio < 0.2:
        return "stable"
    return "improving" if slope > 0 else "degrading"


def analyze(run_dir: Path, ledger_path: Path | None = None) -> dict:
    """Compute summary.json contents from run_dir artifacts.

    `ledger_path` is read ONLY to resolve baseline deltas; it is never written.
    """
    spec_path = run_dir / "spec.json"
    if not spec_path.exists():
        raise RuntimeError(f"Missing spec.json in {run_dir}")
    spec = json.loads(spec_path.read_text())

    metrics_path = run_dir / "metrics.jsonl"
    rows: list[dict] = []
    if metrics_path.exists():
        for ln in metrics_path.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass

    sample_count = len(rows)

    max_kl = max((r.get("approx_kl", 0.0) or 0.0 for r in rows), default=0.0)
    max_clipfrac = max((r.get("clipfrac", 0.0) or 0.0 for r in rows), default=0.0)
    any_nan = any(isinstance(v, float) and math.isnan(v) for r in rows for v in r.values())
    dead_run = False

    stability = {
        "max_kl": max_kl,
        "max_clipfrac": max_clipfrac,
        "any_nan": any_nan,
        "dead_run_triggered": dead_run,
    }

    if rows:
        last = rows[-1]
        terminal = {k: last.get(k) for k in KEY_METRICS if k in last}
    else:
        terminal = {}

    trend: dict | str
    if sample_count < MIN_SAMPLES_FOR_TREND:
        trend = "insufficient_samples"
    else:
        xs = [float(r.get("epoch", i)) for i, r in enumerate(rows)]
        trend = {}
        if any("winner_t" in r and "winner_ct" in r for r in rows):
            bal = [abs((r.get("winner_t", 0.5)) - (r.get("winner_ct", 0.5))) for r in rows]
            trend["winner_balance"] = _classify_slope(bal, _linreg_slope(xs, bal))
        if any("kills_t" in r and "kills_ct" in r for r in rows):
            kt = [(r.get("kills_t", 0.0) or 0.0) + (r.get("kills_ct", 0.0) or 0.0) for r in rows]
            trend["kills_total"] = _classify_slope(kt, _linreg_slope(xs, kt))
        if any("approx_kl" in r for r in rows):
            ks = [r.get("approx_kl", 0.0) or 0.0 for r in rows]
            trend["approx_kl"] = _classify_slope(ks, _linreg_slope(xs, ks))

    baseline_run_id = spec.get("baseline_run_id")
    delta = {}
    if baseline_run_id and ledger_path and ledger_path.exists():
        baseline_rows = [
            r for r in exp_lib.ledger_read(ledger_path) if r.get("run_id") == baseline_run_id
        ]
        if baseline_rows:
            baseline = baseline_rows[0]
            bhash_now = spec.get("env_fingerprint", {}).get("behavior_hash", "")
            bhash_base = baseline.get("env_fingerprint", {}).get("behavior_hash", "")
            delta["baseline_run_id_used"] = baseline_run_id
            delta["baseline_behavior_hash_match"] = (bhash_now == bhash_base and bhash_now != "")
            bterm = baseline.get("terminal_metrics", {})
            if "winner_t" in bterm and "winner_t" in terminal:
                delta["winner_balance_delta"] = (abs(terminal.get("winner_t", 0.5) - 0.5) -
                                                 abs(bterm.get("winner_t", 0.5) - 0.5))
            if "kills_t" in bterm and "kills_t" in terminal:
                delta["kills_total_delta"] = (
                    (terminal.get("kills_t", 0) + terminal.get("kills_ct", 0)) -
                    (bterm.get("kills_t", 0) + bterm.get("kills_ct", 0)))
            if "bomb_planted" in bterm and "bomb_planted" in terminal:
                delta["bomb_planted_delta"] = (terminal.get("bomb_planted", 0) -
                                               bterm.get("bomb_planted", 0))

    if any_nan or sample_count == 0:
        verdict = "bug"
    elif sample_count < MIN_SAMPLES_FOR_TREND:
        verdict = "inconclusive"
    else:
        verdict = "keep"

    summary = {
        "run_id": spec["run_id"],
        "timestamp": exp_lib.utc_now_iso(),
        "branch": spec.get("branch", ""),
        "commit": spec.get("commit", ""),
        "baseline_run_id": baseline_run_id,
        "hypothesis": spec.get("hypothesis", ""),
        "change_type": spec.get("change_type", ""),
        "change_summary": spec.get("change_summary", ""),
        "env_fingerprint": spec.get("env_fingerprint", {}),
        "budget_steps": spec.get("budget_steps", 0),
        "wall_time_sec": spec.get("wall_time_sec", 0),
        "sample_count": sample_count,
        "terminal_metrics": terminal,
        "trend": trend,
        "stability": stability,
        "delta_vs_baseline": delta,
        "verdict": verdict,
        "notes": "FILL_ME",
        "analysis_path": str(run_dir / "analysis.md"),
    }
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("outputs/experiments"),
        help="Root dir for experiments (default: outputs/experiments)",
    )
    args = p.parse_args()

    run_dir = args.experiments_root / args.run_id
    if not run_dir.exists():
        print(f"run dir not found: {run_dir}", file=sys.stderr)
        return 2

    ledger_path = args.experiments_root / "results.jsonl"
    summary = analyze(run_dir, ledger_path=ledger_path)

    (run_dir / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2))

    # Write skeleton analysis.md ONLY if the sentinel is still present anywhere in
    # the file — its absence signals the subagent has taken ownership. Scanning the
    # whole file (rather than just line 1) is robust against autoformatter-inserted
    # blank lines or a copied-out skeleton that left the sentinel mid-file.
    analysis_path = run_dir / "analysis.md"
    should_write_skeleton = True
    if analysis_path.exists():
        text = analysis_path.read_text()
        if text.strip() and exp_lib.SKELETON_SENTINEL not in text:
            should_write_skeleton = False
    if should_write_skeleton:
        analysis_path.write_text(SKELETON.format(run_id=args.run_id))

    return 0


if __name__ == "__main__":
    sys.exit(main())
