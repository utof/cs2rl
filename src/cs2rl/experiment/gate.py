#!/usr/bin/env python
"""Rung 1 gate — spec 2026-08-29 §5, plus the §4 negative-control invalidation.

Reads the seed dirs written by scripts/run_rung1.sh
(<OUT_ROOT>/rung1-s<k>/{config.json,metrics.jsonl}, <OUT_ROOT>/rung1-neg-s<k>/...)
and prints a per-seed + median table with ONE verdict: PASS / FAIL / INVALID.
Exit status: 0 PASS, 1 FAIL, 2 INVALID. Usage:

    UV_NO_SYNC=1 uv run python -m cs2rl.experiment.gate outputs/checkpoints/rung1

From a worktree, put its own src/ first:

    env UV_NO_SYNC=1 PYTHONPATH=<checkout>/src uv run python -m cs2rl.experiment.gate ...

Rules (all from spec §5 — change the spec first, then this file):
  W       rows with agent_steps >= 0.9 * participating_timesteps (config.json;
          the PARTICIPATING --timesteps budget, §2.2 — NOT total_timesteps, which
          is 5x larger at n_active=1) AND self_play/used_past == 0.0 (a missing
          key means self-play was off, i.e. 0.0). Rows are passed through
          analyze_tplant.dedupe_resume_rows first (R0-C resume replays).
  seed    is INCOMPLETE (the experiment did not deliver a judgeable window:
          dir/config.json/metrics.jsonl missing, |W| < 3, no row in W carrying
          both eval/win_vs_random_as_t and _as_ct, zero episodes in W,
          game/shots_fired absent from every W row — key drift) or FAILS the gate on its
          own merits (episode-weighted game/shots_fired < 10, a gated ratio
          with a zero denominator). Both are "fail" rows in the table; only
          the first kind invalidates the verdict (below).
  ratios  game/* are window MEANS per episode (compute_game_metrics), so a
          ratio over W is (sum_W mean_row * episodes_row) for numerator and
          denominator separately, then divide — never a mean of per-row ratios.
  eval/*  the LAST row in W carrying the key.
  median  across seeds; a failed seed (either kind) enters every gated median
          as 0.0 (worst case) — never silently dropped. The report states how
          many seeds contributed to each arm's medians.
  INVALID (exit 2) whenever ANY requested seed — treatment or control — is
          INCOMPLETE: spec §5 is a median across 5 seeds and §4's invalidation
          test needs both controls, so a crashed/missing/half-run seed means
          the experiment was not performed as specified. PITFALL this fixes:
          a missing control used to enter the control median as 0.0 hit/facing,
          which is exactly the value that makes the §4 check pass — so a sweep
          whose controls never ran reported PASS. The seed(s) and reasons are
          named in the report.
  control (§4) INVALID if the negative-control median hit/facing is >= 0.45 or
          within 0.05 of the treatment median: the sigma cap is then not the
          binding constraint and the result licenses no E.3/E.4 attribution.
          This clause replaces PASS only — a treatment failing on its merits
          stays FAIL (the experiment ran; the treatment lost).
  report  spec §5 "Report, not gated" columns are printed per seed and as a
          cross-seed median but never judged: ratios are episode-weighted like
          the gated ones; per-row scalars (losses/*, policy/*, game/*_ticks,
          min_enemy_distance) are the median over W rows; losses/approx_kl is
          the p90 over W rows; eval/win_vs_oracle is the last W row carrying
          it. A key absent from every row prints n/a — never a crash.

PITFALL: the table is NOT the PR's §5 table until every seed has finished
(run_rung1.sh's done marker is <dir>/DONE); a half-finished seed has an
empty W, is INCOMPLETE, and makes the verdict INVALID.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from cs2rl.experiment.analyze_tplant import dedupe_resume_rows, load_rows

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
# Report-only columns printed in the main table (derived from gated inputs).
REPORT_ONLY = ("on_target_per_facing", "hit_per_on_target", "shots_fired", "episodes", "rows")
# Spec §5 "Report, not gated" — (column, kind, args). kind:
#   ratio   episode-weighted num/den over W (None on zero denominator)
#   median  median of the raw row value over W rows that carry the key
#   p90     90th percentile of the raw row value over W rows that carry the key
#   last    the last W row carrying the key
# Keys are the ones train.py emits: compute_game_metrics `game/<counter>`
# window means, pufferl's `losses/<k>` prefix over trainer losses
# (`entropy/shoot` is the per-head entropy of ACTION_HEAD_NAMES[1]), the
# `policy/aim_log_std_yaw` diagnostic, eval_baselines' `eval/win_vs_oracle`.
REPORT_EXTRA = (
    ("stance_blocked_per_facing", "ratio", ("game/shots_stance_blocked",
                                            "game/shots_facing_enemy")),
    ("los_per_fired", "ratio", ("game/shots_with_enemy_in_los", "game/shots_fired")),
    ("losses/entropy/shoot", "median", ("losses/entropy/shoot", )),
    ("policy/aim_log_std_yaw", "median", ("policy/aim_log_std_yaw", )),
    ("losses/approx_kl_p90", "p90", ("losses/approx_kl", )),
    ("losses/effective_alpha", "median", ("losses/effective_alpha", )),
    ("losses/empty_minibatches", "median", ("losses/empty_minibatches", )),
    ("game/mutual_vis_pair_ticks", "median", ("game/mutual_vis_pair_ticks", )),
    ("game/agent_ticks_with_visible_enemy", "median", ("game/agent_ticks_with_visible_enemy", )),
    ("game/min_enemy_distance", "median", ("game/min_enemy_distance", )),
    ("game/min_enemy_distance_valid_frac", "median", ("game/min_enemy_distance_valid_frac", )),
    ("eval/win_vs_oracle", "last", ("eval/win_vs_oracle", )),
)
NEG_CONTROL_MAX = 0.45
NEG_CONTROL_MARGIN = 0.05
EXIT_CODES = {"PASS": 0, "FAIL": 1, "INVALID": 2}


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


def _incomplete(reason):
    """A seed whose window cannot be judged (run missing / crashed / too short)."""
    return {"fail": reason, "incomplete": True}


def _row_values(rows, key):
    return [float(r[key]) for r in rows if key in r and r[key] is not None]


def _p90(values):
    """Nearest-rank p90 (no interpolation — 3-row windows would otherwise
    invent a value between two real ones)."""
    v = sorted(values)
    return v[min(len(v) - 1, int(math.ceil(0.9 * len(v))) - 1)]


def report_extra(window):
    """Spec §5 report-only columns over W (see REPORT_EXTRA). Missing keys → None."""
    out = {}
    for col, kind, args in REPORT_EXTRA:
        if kind == "ratio":
            # n/a unless some W row carries the numerator (a counter absent from
            # every row would otherwise print a fake 0.000).
            out[col] = ratio(window, *args) if _row_values(window, args[0]) else None
        else:
            vals = _row_values(window, args[0])
            if not vals:
                out[col] = None
            elif kind == "median":
                out[col] = statistics.median(vals)
            elif kind == "p90":
                out[col] = _p90(vals)
            elif kind == "last":
                out[col] = vals[-1]
    return out


def seed_metrics(rows, participating_timesteps):
    """Gate columns for one seed, or {"fail": reason[, "incomplete": True]}.
    `incomplete` marks a window that cannot be judged (→ INVALID verdict);
    a plain "fail" is a gate failure on the seed's own merits (→ 0.0 in medians).
    Dedupes here (idempotent) so callers may pass raw or load_rows() output."""
    window = gate_window(dedupe_resume_rows(list(rows)), participating_timesteps)
    if len(window) < MIN_WINDOW_ROWS:
        return _incomplete(f"only {len(window)} rows in W (need {MIN_WINDOW_ROWS})")
    ev = [r for r in window if EVAL_T in r and EVAL_CT in r]
    if not ev:
        return _incomplete("no eval row in W")
    episodes = sum(float(r.get(EPISODES_KEY, 0.0)) for r in window)
    if episodes <= 0:
        return _incomplete("zero episodes in W")
    # Key-drift tripwire: every game/* counter rides the same compute_game_metrics
    # row, so shots_fired absent from EVERY W row means the metrics schema moved,
    # not that the agent never fired. Without this guard weighted_sum reads 0.0
    # and the seed FAILS "on its merits" — a misleading "treatment lost" verdict.
    if not _row_values(window, "game/shots_fired"):
        return _incomplete("game/shots_fired absent from every W row (metrics key drift?)")
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
    m.update(report_extra(window))
    return m


def load_seed(run_dir):
    """seed_metrics over a run dir; a missing dir/file/key is a structural
    failure (reported in the table), not a crash of the whole gate."""
    try:
        cfg = json.loads((run_dir / "config.json").read_text())
        return seed_metrics(load_rows(run_dir), float(cfg["participating_timesteps"]))
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        return _incomplete(f"{type(exc).__name__}: {exc}")


def _median(per_seed, col):
    """Cross-seed median of a GATED/REPORT_ONLY column; failed seeds → 0.0."""
    return statistics.median([0.0 if "fail" in m else float(m[col]) for m in per_seed])


def _median_optional(per_seed, col):
    """Cross-seed median of a REPORT_EXTRA column over the seeds that have it;
    None when no seed does (missing key → n/a, not 0.0: these are diagnostics,
    a fake 0.0 would read as a real measurement)."""
    vals = [float(m[col]) for m in per_seed if "fail" not in m and m.get(col) is not None]
    return statistics.median(vals) if vals else None


def incomplete_seeds(arm):
    """{seed: reason} for the INCOMPLETE seeds of one arm (dict seed → metrics)."""
    return {s: m["fail"] for s, m in arm.items() if m.get("incomplete")}


def evaluate_arm(per_seed):
    """(medians, per-gate ok) for one arm; failed seeds count as 0.0 everywhere.
    REPORT_ONLY columns get medians too (same 0.0 rule) but never a verdict;
    REPORT_EXTRA medians skip seeds lacking the key."""
    med = {col: _median(per_seed, col) for col in [c for c, _, _ in GATES] + list(REPORT_ONLY)}
    med.update({col: _median_optional(per_seed, col) for col, _, _ in REPORT_EXTRA})
    ok = {col: (med[col] > thr) if op == ">" else (med[col] >= thr) for col, thr, op in GATES}
    return med, ok


def gate_report(out_root, seeds, neg_seeds, prefix="rung1"):
    """Full report dict for <out_root>/<prefix>-s<k> (treatment) and
    <prefix>-neg-s<k> (control). Empty neg_seeds skips the §4 check.
    Verdict precedence: INVALID (any incomplete seed, either arm) >
    FAIL (treatment medians miss a gate) > INVALID (§4 control clause) > PASS."""
    out_root = Path(out_root)
    treat = {s: load_seed(out_root / f"{prefix}-s{s}") for s in seeds}
    neg = {s: load_seed(out_root / f"{prefix}-neg-s{s}") for s in neg_seeds}
    if not treat:
        raise ValueError("gate_report needs at least one treatment seed")
    med, ok = evaluate_arm(list(treat.values()))
    neg_med, _ = evaluate_arm(list(neg.values())) if neg else ({}, {})
    verdict = "PASS" if all(ok.values()) else "FAIL"
    neg_hit = neg_med["hit_per_facing"] if neg else None
    control_ok = neg_hit is None or (neg_hit < NEG_CONTROL_MAX
                                     and abs(neg_hit - med["hit_per_facing"]) > NEG_CONTROL_MARGIN)
    if verdict == "PASS" and not control_ok:
        verdict = "INVALID"
    incomplete = {
        "treatment": incomplete_seeds(treat),
        "control": incomplete_seeds(neg),
    }
    if incomplete["treatment"] or incomplete["control"]:
        verdict = "INVALID"
    contributed = {
        "treatment": (sum(1 for m in treat.values() if "fail" not in m), len(treat)),
        "control": (sum(1 for m in neg.values() if "fail" not in m), len(neg)),
    }
    return {
        "treatment": treat,
        "control": neg,
        "median": med,
        "control_median": neg_med,
        "ok": ok,
        "control_hit_per_facing": neg_hit,
        "control_ok": control_ok,
        "incomplete": incomplete,
        "contributed": contributed,
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
    # Spec §5 report-only diagnostics: one line per metric, one column per seed.
    seeds = [(f"{prefix}-s{s}", m) for s, m in rep["treatment"].items()]
    neg = [(f"{prefix}-neg-s{s}", m) for s, m in rep["control"].items()]
    print("report-only (not gated; per-seed over W, then cross-seed median):", file=file)
    print(f"{'metric':<38}" + "".join(f"{n:>14}" for n, _ in seeds) + f"{'median':>14}" +
          "".join(f"{n:>14}" for n, _ in neg),
          file=file)
    for col, _, _ in REPORT_EXTRA:
        cells = [_fmt(None if "fail" in m else m.get(col)) for _, m in seeds]
        cells.append(_fmt(rep["median"].get(col)))
        cells += [_fmt(None if "fail" in m else m.get(col)) for _, m in neg]
        print(f"{col:<38}" + "".join(f"{c:>14}" for c in cells), file=file)
    for col, thr, op in GATES:
        print(
            f"  {col:<28} median {rep['median'][col]:.3f} {op} {thr} -> "
            f"{'ok' if rep['ok'][col] else 'FAIL'}",
            file=file)
    for arm in ("treatment", "control"):
        got, want = rep["contributed"][arm]
        if want:
            print(f"  {arm} seeds contributing to medians: {got}/{want}" +
                  ("" if got == want else " (failed seeds enter every median as 0.0)"),
                  file=file)
    if rep["control_hit_per_facing"] is not None:
        print(
            f"  negative control hit/facing median {rep['control_hit_per_facing']:.3f} "
            f"(must be < {NEG_CONTROL_MAX} and > {NEG_CONTROL_MARGIN} from treatment) -> "
            f"{'ok' if rep['control_ok'] else 'INVALIDATES'}",
            file=file)
    for arm in ("treatment", "control"):
        for s, why in rep["incomplete"][arm].items():
            print(f"  INCOMPLETE {arm} seed {s}: {why} -> verdict INVALID", file=file)
    print(f"VERDICT: {rep['verdict']}", file=file)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cs2rl.experiment.gate",
                                 description=__doc__.splitlines()[0])
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
    return EXIT_CODES[rep["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
