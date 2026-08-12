#!/usr/bin/env python
"""t_plant drift analysis for the CT stall-subsidy A/B (spec 2026-08-01 §5).

WHAT: reads one or more run dirs' metrics.jsonl and reports, per run:
  - per-block within-block t_plant drift slope (ticks/epoch) + the median
    across blocks — the PRIMARY A/B readout (baseline: +16..64/epoch),
  - dead windows (contiguous >1M-step spans with plant rate ≤ p_min) and
    whether any occur after 15M steps (secondary failure criterion),
  - 5M-window plant-rate table + run mean,
  - CT-pressure confound controls: CT kills/epoch-row, CT win rate, timeout
    rate, and CT win-by-elimination share. Pre-registered interpretation:
    a flat slope WITH collapsed CT pressure reads as "CT went passive" — a
    confound, NOT a success.

WHY t_plant is derived, not logged: the env logs mean round_length and
plant rate p per epoch row. Inverting
    round_length = p*(t_plant + BOMB_TIMER) + (1-p)*cap
gives
    t_plant = (round_length - (1-p)*cap) / p - BOMB_TIMER.
Assumptions (spec §5, stated, not checked): planted rounds run to
detonation — defuses / post-plant T eliminations bias t_plant low (both
~0 today; their growth shows up in the CT-pressure controls) — and the
1/p factor amplifies noise at low p, so rows with p <= p_min are excluded
and slopes are fit only within blocks of consecutive kept rows.

PITFALLS:
  - Absolute t_plant levels are NOT comparable across different round caps
    (the (1-p)*cap term shifts the constant): never compare against the
    roundtime-1280 run without passing --cap 1280 for that run.
  - Crash-resumed runs append to the same metrics.jsonl under a NEW run_id
    with epoch reset to 0 (the box's GPU falls off the bus under thermal
    load — resume seams are expected, not exceptional). Rows are grouped
    by run_id and blocks NEVER span a seam, so slopes stay within-segment
    honest. Step totals concatenate segments.
  - Wall-clock is deliberately absent from the report: A2's symmetrize
    transform costs ~33µs/step, so cross-arm wall-clock comparisons are
    meaningless and pre-registered as out of scope.

Usage:
  uv run python scripts/analyze_tplant.py outputs/checkpoints/<run-a> [<run-b> ...]
  uv run python scripts/analyze_tplant.py --cap 1280 outputs/checkpoints/roundtime-1280-run
"""

import argparse
import json
import sys
from pathlib import Path

BOMB_TIMER_DEFAULT = 640               # ticks; src/nav.py BOMB_TIMER — keep in sync
CAP_DEFAULT = 640                      # ticks; src/c_env/nav_data.h CFG_ROUND_TIME


def load_rows(run_dir: Path):
    """Yield metric rows (dicts) from <run_dir>/metrics.jsonl in file order.

    File order == chronological order across resume seams (train.py appends);
    malformed lines are skipped loudly on stderr rather than crashing — a
    hard power cut can tear the final line of the previous segment.
    """
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — is this a run dir?")
    with path.open() as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] {path}:{i}: torn/malformed line skipped", file=sys.stderr)


def t_plant(row, cap, bomb_timer, p_min):
    """Derived plant latency for one epoch row, or None if p too low / keys missing."""
    p = row.get("game/bomb_plant_rate")
    rl = row.get("environment/round_length")
    if p is None or rl is None or p <= p_min:
        return None
    return (rl - (1.0 - p) * cap) / p - bomb_timer


def slope(xs, ys):
    """Least-squares slope of ys on xs. Requires len >= 3 (else None)."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx)**2 for x in xs)
    if denom == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / denom


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def segment_rows(rows):
    """Split chronological rows into per-run_id segments (resume seams).

    Consecutive-grouping (not a global groupby) on purpose: run_id is a
    random string, and grouping globally would merge segments out of order
    if a run_id ever repeated.
    """
    segments = []
    cur_id, cur = object(), []
    for row in rows:
        rid = row.get("run_id")
        if rid != cur_id:
            if cur:
                segments.append((cur_id, cur))
            cur_id, cur = rid, []
        cur.append(row)
    if cur:
        segments.append((cur_id, cur))
    return segments


def analyze_run(run_dir: Path, cap: float, bomb_timer: float, p_min: float, min_block: int,
                window_steps: float, dead_window_steps: float, dead_after_steps: float):
    rows = list(load_rows(run_dir))
    if not rows:
        raise ValueError(f"{run_dir}: metrics.jsonl is empty")
    segments = segment_rows(rows)

    # ── blocks + slopes (never spanning a resume seam) ──────────────────
    blocks = []                        # list of (seg_idx, [(epoch, t_plant)])
    for si, (_, seg) in enumerate(segments):
        cur = []
        for row in seg:
            tp = t_plant(row, cap, bomb_timer, p_min)
            if tp is None:
                if len(cur) >= min_block:
                    blocks.append((si, cur))
                cur = []
            else:
                cur.append((row.get("epoch", 0), tp))
        if len(cur) >= min_block:
            blocks.append((si, cur))
    block_slopes = []
    for si, blk in blocks:
        s = slope([e for e, _ in blk], [t for _, t in blk])
        if s is not None:
            block_slopes.append((si, blk[0][0], blk[-1][0], len(blk), s))

    # ── dead windows over concatenated step axis ────────────────────────
    # Steps are cumulative WITHIN a segment; concatenate by offsetting each
    # segment by the running total so windows and "after 15M" are global.
    dead_windows = []
    offset = 0.0
    prev_step = 0.0
    win_start = None
    total_steps = 0.0
    for _, seg in segments:
        seg_last = 0.0
        for row in seg:
            step = offset + row.get("step", 0)
            seg_last = row.get("step", 0)
            p = row.get("game/bomb_plant_rate") or 0.0
            if p <= p_min:
                if win_start is None:
                    win_start = prev_step
            else:
                if win_start is not None and step - win_start >= dead_window_steps:
                    dead_windows.append((win_start, step))
                win_start = None
            prev_step = step
        offset += seg_last
    total_steps = offset
    if win_start is not None and prev_step - win_start >= dead_window_steps:
        dead_windows.append((win_start, prev_step))
    late_dead = [w for w in dead_windows if w[1] > dead_after_steps]

    # ── plant-rate windows + CT-pressure controls ───────────────────────
    plant_windows = {}                                                                             # window_idx -> [p, ...]
    ct = {"kills_ct": [], "win_ct": [], "timeout": [], "plant": []}
    offset = 0.0
    for _, seg in segments:
        seg_last = 0.0
        for row in seg:
            step = offset + row.get("step", 0)
            seg_last = row.get("step", 0)
            p = row.get("game/bomb_plant_rate")
            if p is not None:
                plant_windows.setdefault(int(step // window_steps), []).append(p)
                ct["plant"].append(p)
            for src, dst in (("environment/kills_ct", "kills_ct"), ("game/win_rate_ct", "win_ct"),
                             ("game/timeout_rate", "timeout")):
                v = row.get(src)
                if v is not None:
                    ct[dst].append(v)
        offset += seg_last

    mean = lambda v: sum(v) / len(v) if v else None                                            # noqa: E731
    win_ct_mean, timeout_mean = mean(ct["win_ct"]), mean(ct["timeout"])
                                                                                               # CT wins by timeout ARE timeouts (T never wins one); elimination share
                                                                                               # is the remainder of CT wins. Guard div-by-zero when CT never wins.
    elim_share = (None if not win_ct_mean else max(0.0, win_ct_mean - (timeout_mean or 0.0)) /
                  win_ct_mean)

    return {
        "run": run_dir.name,
        "segments": len(segments),
        "rows": len(rows),
        "total_steps": total_steps,
        "blocks": [(f"seg{si}", e0, e1, n, s) for si, e0, e1, n, s in block_slopes],
        "median_slope": median([s for *_, s in block_slopes]),
        "dead_windows": dead_windows,
        "late_dead_windows": late_dead,
        "plant_windows": {
            k: mean(v)
            for k, v in sorted(plant_windows.items())
        },
        "plant_run_mean": mean(ct["plant"]),
        "ct_kills_mean": mean(ct["kills_ct"]),
        "ct_win_rate_mean": win_ct_mean,
        "timeout_rate_mean": timeout_mean,
        "ct_win_by_elim_share": elim_share,
    }


def print_report(r, window_steps):
    fmt = lambda v, spec=".3f": ("n/a" if v is None else format(v, spec))                                    # noqa: E731
    print(f"\n== {r['run']} ==")
    print(f"rows={r['rows']}  segments={r['segments']} "
          f"(resume seams: {r['segments'] - 1})  total_steps={r['total_steps']:.3g}")
    print(f"PRIMARY  median within-block t_plant slope: {fmt(r['median_slope'], '+.2f')} "
          f"ticks/epoch  (baseline reference: +16..64)")
    for seg, e0, e1, n, s in r["blocks"]:
        print(f"  block {seg} epochs {e0}-{e1} (n={n}): {s:+.2f} ticks/epoch")
    if not r["blocks"]:
        print("  (no plant blocks above p_min — t_plant undefined)")
    lw = r["late_dead_windows"]
    print(f"dead windows >=1M steps: {len(r['dead_windows'])} total, "
          f"{len(lw)} after 15M steps {'← FAIL criterion' if lw else '(pass)'}")
    for a, b in r["dead_windows"]:
        print(f"  dead [{a:.3g} .. {b:.3g}] ({(b - a):.3g} steps)")
    print(f"plant rate: run mean {fmt(r['plant_run_mean'])}; per {window_steps:.0g}-step window:")
    for k, v in r["plant_windows"].items():
        print(f"  [{k * window_steps:.3g} .. {(k + 1) * window_steps:.3g}): {v:.3f}")
    print("CT pressure (confound controls — collapse here voids a flat slope):")
    print(
        f"  kills_ct mean {fmt(r['ct_kills_mean'])}  win_rate_ct {fmt(r['ct_win_rate_mean'])}  "
        f"timeout_rate {fmt(r['timeout_rate_mean'])}  CT-win-by-elim share {fmt(r['ct_win_by_elim_share'])}"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument(
        "--cap",
        type=float,
        default=CAP_DEFAULT,
        help="round-time cap in ticks (MUST match the run's build; 1280 for the roundtime-1280 run)"
    )
    ap.add_argument("--bomb-timer", type=float, default=BOMB_TIMER_DEFAULT)
    ap.add_argument("--p-min",
                    type=float,
                    default=0.15,
                    help="rows with plant rate <= this are excluded (1/p noise amplification)")
    ap.add_argument("--min-block",
                    type=int,
                    default=3,
                    help="minimum consecutive kept rows for a slope fit")
    ap.add_argument("--window-steps", type=float, default=5e6)
    ap.add_argument("--dead-window-steps", type=float, default=1e6)
    ap.add_argument("--dead-after-steps", type=float, default=15e6)
    args = ap.parse_args(argv)
    for d in args.run_dirs:
        print_report(
            analyze_run(d, args.cap, args.bomb_timer, args.p_min, args.min_block, args.window_steps,
                        args.dead_window_steps, args.dead_after_steps), args.window_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
