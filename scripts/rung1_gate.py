#!/usr/bin/env python
"""Rung 1 gate — spec 2026-08-29 §5, plus the §4 negative-control invalidation.

Reads the seed dirs written by scripts/run_rung1.sh
(<OUT_ROOT>/rung1-s<k>/{config.json,metrics.jsonl}, <OUT_ROOT>/rung1-neg-s<k>/...)
and prints a per-seed + median table with ONE verdict: PASS / FAIL / INVALID.
Exit status 0 only on PASS. Usage:

    UV_NO_SYNC=1 uv run python scripts/rung1_gate.py outputs/checkpoints/rung1

Rules (all from spec §5 — change the spec first, then this file):
  W       rows with agent_steps >= 0.9 * participating_timesteps (config.json;
          the PARTICIPATING --timesteps budget, §2.2 — NOT total_timesteps, which
          is 5x larger at n_active=1) AND self_play/used_past == 0.0 (a missing
          key means self-play was off, i.e. 0.0). Rows are passed through
          analyze_tplant.dedupe_resume_rows first (R0-C resume replays).
  seed    fails STRUCTURALLY if |W| < 3, no row in W carries both
          eval/win_vs_random_as_t and _as_ct, the episode-weighted
          game/shots_fired < 10, or a gated ratio has a zero denominator.
  ratios  game/* are window MEANS per episode (compute_game_metrics), so a
          ratio over W is (sum_W mean_row * episodes_row) for numerator and
          denominator separately, then divide — never a mean of per-row ratios.
  eval/*  the LAST row in W carrying the key.
  median  across seeds; a structurally failed seed enters every median as 0.0
          (worst case). The spec says such a seed *fails*; dropping it from the
          median would let three good seeds carry two crashed ones.
  control (§4) INVALID if the negative-control median hit/facing is >= 0.45 or
          within 0.05 of the treatment median: the sigma cap is then not the
          binding constraint and the result licenses no E.3/E.4 attribution.
          INVALID replaces PASS only — a failing treatment stays FAIL.

PITFALL: the table is NOT the PR's §5 table until every seed has finished
(run_rung1.sh's done marker is <dir>/DONE); a half-finished seed has an
empty W and shows as a structural failure, which drags the medians to 0.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_tplant import dedupe_resume_rows, load_rows               # noqa: E402

WINDOW_FRAC = 0.9
MIN_WINDOW_ROWS = 3
MIN_SHOTS_FIRED = 10.0
EPISODES_KEY = "environment/episodes"
EVAL_T, EVAL_CT = "eval/win_vs_random_as_t", "eval/win_vs_random_as_ct"
# (column, threshold, comparison) — spec §5 bullets, in table order.
GATES = (
    ("kills_per_episode", 0.5, ">"),
    ("hit_per_facing", 0.45, ">"),
    ("facing_per_fired", 0.7, ">"),
    (EVAL_T, 0.90, ">="),
    (EVAL_CT, 0.90, ">="),
)
REPORT_ONLY = ("on_target_per_facing", "hit_per_on_target", "shots_fired", "episodes", "rows")
NEG_CONTROL_MAX = 0.45
NEG_CONTROL_MARGIN = 0.05


def gate_window(rows, participating_timesteps):
    """Spec §5 W. `rows` must already be deduped (see seed_metrics).
    PITFALL: keyed on agent_steps (participating counter) vs the PARTICIPATING
    budget — never total_timesteps."""
    lo = WINDOW_FRAC * participating_timesteps
    return [
        r for r in rows if float(r.get("agent_steps", -1.0)) >= lo
        and float(r.get("self_play/used_past", 0.0)) == 0.0
    ]


def weighted_sum(rows, key):
    """sum_W mean_row * episodes_row — one side of an episode-weighted ratio."""
    return sum(float(r.get(key, 0.0)) * float(r.get(EPISODES_KEY, 0.0)) for r in rows)


def ratio(rows, num_key, den_key):
    """Episode-weighted num/den over W; None on a zero denominator (caller decides)."""
    den = weighted_sum(rows, den_key)
    return weighted_sum(rows, num_key) / den if den > 0 else None


def seed_metrics(rows, participating_timesteps):
    """Gate columns for one seed, or {"fail": reason} on a structural failure.
    Dedupes here (idempotent) so callers may pass raw or load_rows() output."""
    window = gate_window(dedupe_resume_rows(list(rows)), participating_timesteps)
    if len(window) < MIN_WINDOW_ROWS:
        return {"fail": f"only {len(window)} rows in W (need {MIN_WINDOW_ROWS})"}
    ev = [r for r in window if EVAL_T in r and EVAL_CT in r]
    if not ev:
        return {"fail": "no eval row in W"}
    episodes = sum(float(r.get(EPISODES_KEY, 0.0)) for r in window)
    if episodes <= 0:
        return {"fail": "zero episodes in W"}
    shots_fired = weighted_sum(window, "game/shots_fired") / episodes
    if shots_fired < MIN_SHOTS_FIRED:
        return {"fail": f"episode-weighted game/shots_fired {shots_fired:.2f} < {MIN_SHOTS_FIRED}"}
    m = {
        "kills_per_episode": weighted_sum(window, "game/kills_per_episode") / episodes,
        "hit_per_facing": ratio(window, "game/shots_hit", "game/shots_facing_enemy"),
        "facing_per_fired": ratio(window, "game/shots_facing_enemy", "game/shots_fired"),
        "on_target_per_facing": ratio(window, "game/shots_on_target", "game/shots_facing_enemy"),
        "hit_per_on_target": ratio(window, "game/shots_hit", "game/shots_on_target"),
        EVAL_T: float(ev[-1][EVAL_T]),
        EVAL_CT: float(ev[-1][EVAL_CT]),
        "shots_fired": shots_fired,
        "episodes": episodes,
        "rows": len(window),
    }
    for col, _, _ in GATES:
        if m[col] is None:
            return {"fail": f"zero denominator in {col}"}
    return m


def load_seed(run_dir):
    """seed_metrics over a run dir; a missing dir/file/key is a structural
    failure (reported in the table), not a crash of the whole gate."""
    try:
        cfg = json.loads((run_dir / "config.json").read_text())
        return seed_metrics(load_rows(run_dir), float(cfg["participating_timesteps"]))
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        return {"fail": f"{type(exc).__name__}: {exc}"}


def _median(per_seed, col):
    return statistics.median([0.0 if "fail" in m else float(m[col]) for m in per_seed])


def evaluate_arm(per_seed):
    """(medians, per-gate ok) for one arm; failed seeds count as 0.0 everywhere.
    REPORT_ONLY columns get medians too (same 0.0 rule) but never a verdict."""
    med = {col: _median(per_seed, col) for col in [c for c, _, _ in GATES] + list(REPORT_ONLY)}
    ok = {col: (med[col] > thr) if op == ">" else (med[col] >= thr) for col, thr, op in GATES}
    return med, ok


def gate_report(out_root, seeds, neg_seeds, prefix="rung1"):
    """Full report dict for <out_root>/<prefix>-s<k> (treatment) and
    <prefix>-neg-s<k> (control). Empty neg_seeds skips the §4 check."""
    out_root = Path(out_root)
    treat = {s: load_seed(out_root / f"{prefix}-s{s}") for s in seeds}
    neg = {s: load_seed(out_root / f"{prefix}-neg-s{s}") for s in neg_seeds}
    if not treat:
        raise ValueError("gate_report needs at least one treatment seed")
    med, ok = evaluate_arm(list(treat.values()))
    verdict = "PASS" if all(ok.values()) else "FAIL"
    neg_hit = _median(list(neg.values()), "hit_per_facing") if neg else None
    control_ok = neg_hit is None or (neg_hit < NEG_CONTROL_MAX
                                     and abs(neg_hit - med["hit_per_facing"]) > NEG_CONTROL_MARGIN)
    if verdict == "PASS" and not control_ok:
        verdict = "INVALID"
    return {
        "treatment": treat,
        "control": neg,
        "median": med,
        "ok": ok,
        "control_hit_per_facing": neg_hit,
        "control_ok": control_ok,
        "verdict": verdict
    }


def _fmt(v):
    if v is None:
        return "n/a"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def print_report(rep, prefix="rung1", file=None):
    """Human table + verdict. `file=None` → sys.stdout resolved at CALL time
    (a default bound at import would bypass pytest's capsys)."""
    file = file or sys.stdout
    cols = [c for c, _, _ in GATES] + list(REPORT_ONLY)
    head = f"{'seed':<14}" + "".join(f"{c.replace('eval/win_vs_random_', 'eval_'):>22}"
                                     for c in cols)
    print(head, file=file)
    for label, arm in ((prefix, rep["treatment"]), (f"{prefix}-neg", rep["control"])):
        for s, m in arm.items():
            name = f"{label}-s{s}"
            if "fail" in m:
                print(f"{name:<14}FAIL: {m['fail']}", file=file)
            else:
                print(f"{name:<14}" + "".join(f"{_fmt(m[c]):>22}" for c in cols), file=file)
    print(f"{'median':<14}" + "".join(f"{_fmt(rep['median'].get(c)):>22}" for c in cols), file=file)
    for col, thr, op in GATES:
        print(
            f"  {col:<28} median {rep['median'][col]:.3f} {op} {thr} -> "
            f"{'ok' if rep['ok'][col] else 'FAIL'}",
            file=file)
    if rep["control_hit_per_facing"] is not None:
        print(
            f"  negative control hit/facing median {rep['control_hit_per_facing']:.3f} "
            f"(must be < {NEG_CONTROL_MAX} and > {NEG_CONTROL_MARGIN} from treatment) -> "
            f"{'ok' if rep['control_ok'] else 'INVALIDATES'}",
            file=file)
    print(f"VERDICT: {rep['verdict']}", file=file)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_root", type=Path, help="OUT_ROOT passed to scripts/run_rung1.sh")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument(
        "--neg-seeds",
        type=int,
        nargs="*",
        default=[0, 1],
        help="negative-control seeds (pass the flag with no values to skip the §4 check)")
    ap.add_argument("--prefix", default="rung1", help="run_seed label prefix in run_rung1.sh")
    args = ap.parse_args(argv)
    rep = gate_report(args.out_root, args.seeds, args.neg_seeds, prefix=args.prefix)
    print_report(rep, prefix=args.prefix)
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
