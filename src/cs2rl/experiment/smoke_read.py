#!/usr/bin/env python
"""Rung 1a smoke gate reader — the FROZEN pre-registration of 2026-08-31.

Historical rules: 2026-08-31 Rung 1a registration (gh#152).
Current public summary: docs/validation.md#rung-1a-smoke-reader.
The constants and synthetic-row tests retain the frozen decision rules.

WHAT: reads ONE run dir (`metrics.jsonl` + `config.json`), prints the five
pre-flight assertions with their observed values, the episode-weighted window
aggregates, and exactly one verdict line. Usage:

    UV_NO_SYNC=1 uv run python -m cs2rl.experiment.smoke_read outputs/checkpoints/rung1a/s0

From a worktree, put its own src/ first:

    env UV_NO_SYNC=1 PYTHONPATH=<checkout>/src uv run python -m cs2rl.experiment.smoke_read ...

Exit status: 0 PASS, 1 FAIL (any routing), 2 SMOKE INVALID.

WHY a new reader instead of the Rung 1 gate (cs2rl/experiment/gate.py): that gate encodes the Rung 1
sweep's rules — a MIN_SHOTS_FIRED floor, two `eval/win_vs_random_*` completeness
requirements and the negative-control invalidation clause — none of which apply
to a 1-seed run against a noop statue (there is no control arm, and the fixed
baselines shoot back so their win rates are not the object of study). What IS
reused are its *conventions*, because they are properties of the metrics stream
rather than of that gate: dedup of replayed rows (gate_window's precondition,
done in seed_metrics), a missing `self_play/used_past` reading as 0.0
(gate_window), and episode weighting (weighted_sum / ratio).

WHY EPISODE WEIGHTING (the single most important thing in this file): every
`game/*` value on a row is the MEAN OVER THE EPISODES THAT FINISHED INSIDE THAT
ROW's WINDOW (`cs2rl.train.metrics.compute_game_metrics` + pufferl's window flush), and
the per-row episode count is wildly non-uniform — all 256 envs stay round-
synchronised, so consecutive rows carry e.g. 257, 4, 252, 8 episodes. A plain
mean over rows therefore gives a near-empty 4-episode row the same weight as a
257-episode one, which is how a 3-kill fluke in a sparse row can move the
headline number by an order of magnitude. Every aggregate below is
sum_rows(value * episodes) / sum_rows(episodes), i.e. the value pooled over
episodes, and ratios pool numerator and denominator SEPARATELY before dividing
(never a mean of per-row ratios).

WHY `action_move_*` NEEDS A DIVISION (pre-flight 4): the
`environment/action_move_<bin>` keys are per-episode COUNTS, not fractions —
observed on smoke-v1c as 9 bins summing to 320 = 2 agents x 160 ticks. A literal
`action_move_0 >= 0.5` on the raw count would pass with no statue override in
place at all (a live opponent easily emits >0.5 counts of bin 0). The assertion
is on the FRACTION `action_move_0 / sum(action_move_*)`: the inert statue
contributes ~0.5 of all move actions by itself, while the live-baseline bin-0
fraction is ~0.10, so >= 0.45 discriminates "statue" from "opponent that moves"
even if the hero never stands still.

PITFALLS
  - Pre-flights 1, 2 and 4 are evaluated over EVERY deduped row of the run (the
    prereg says "every row" / "any row"), not just the window: a statue that
    came alive at epoch 3 must invalidate the smoke even if it was inert at the
    end. Pre-flight 5 is window-only, as written. Pre-flight 1 skips the run's
    first row on purpose — `mean_and_log()` runs before `self.losses` is set
    (`cs2rl.train.trainer.Cs2PuffeRL.train`) so row 0 carries no `losses/*` at all; the rule is
    "every row CARRYING the key".
  - A key missing from EVERY row is a harness/key-drift finding, not a passing
    assertion: those paths report SMOKE INVALID rather than vacuously ok.
  - Pre-flight 5 is the one assertion whose failure is NOT invalidating. A raw
    sigma above the cap means the gradient exists and pushed sigma into the
    clamp-dead zone, so the sigma-MOVEMENT routing is void while the run itself
    is still interpretable; the reader reports SIGMA-CAPPED and routes on kills
    alone (next step rec 4(b)), per prereg pre-flight 5.
  - Nothing here reads an outcome to decide a threshold. If a number below has
    to change, change the pre-registration first (and the read stops being
    confirmatory).
"""
import argparse
import json
import re
import sys
from pathlib import Path

RUN_DIR_DEFAULT = Path("outputs/checkpoints/rung1a/s0")

# ── Frozen constants (prereg "Primary analysis" + "Decision rule") ───────────
PARTICIPATING_TIMESTEPS = 1_000_000    # spec §2 T4 `--timesteps`
WINDOW_FRAC = 0.9                      # window: agent_steps >= 900_000
WINDOW_MAX_ROWS = 10                   # ...last up-to-10 of them
MIN_WINDOW_ROWS = 5                    # fewer ⇒ SMOKE INVALID
FINAL_AGENT_STEPS_FRAC = 0.95          # pre-flight 3: >= 950_000
EXPECTED_PARTICIPATING_ROWS = 16384.0  # 256 envs x 64 bptt x 1 active
MOVE_BIN0_MIN_FRAC = 0.45              # pre-flight 4 (statue inert)
KILLS_PASS = 0.5                       # PASS rule, both required
HIT_RATE_PASS = 0.4
SIGMA_MOVED_MIN = 0.1                  # |raw - init| that counts as "moved"
KILLS_UNTRAINED_MAX = 0.1              # kills below this + no sigma move ⇒ rec 6

EPISODES_KEY = "environment/episodes"
KILLS_KEY = "game/kills_per_episode"
ON_TARGET_KEY = "game/shots_on_target"
FIRED_KEY = "game/shots_fired"
KILLS_CT_KEY = "game/kills_ct"
PART_ROWS_KEY = "losses/participating_rows"
OPPONENT_TEAM_KEY = "self_play/opponent_team"
USED_PAST_KEY = "self_play/used_past"
SIGMA_RAW_KEY = "policy/aim_log_std_yaw_raw"
MOVE_KEY_RE = re.compile(r"^environment/action_move_(\d+)$")

PASS = "PASS"
FAIL = "FAIL"
INVALID = "SMOKE INVALID"
EXIT_CODES = {PASS: 0, FAIL: 1, INVALID: 2}


def load_rows(run_dir):
    """Metric rows from <run_dir>/metrics.jsonl in file order.

    Malformed lines are skipped with a loud stderr note rather than crashing the
    read (a hard power cut can tear the last line); a missing file raises
    FileNotFoundError, which read_run turns into a SMOKE INVALID reason.
    """
    path = Path(run_dir) / "metrics.jsonl"
    rows = []
    with path.open() as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[read] skipping malformed {path}:{i}: {exc}", file=sys.stderr)
    return rows


def dedupe_by_agent_steps(rows):
    """Last row wins per `agent_steps`, keeping the FIRST occurrence's position.

    Mirrors analyze_tplant.dedupe_resume_rows (keyed there on (run_id, step)):
    a resume replays the epochs that the pre-crash segment already logged, and
    the post-resume row is the one trained on state that survived. Keyed on
    agent_steps here because that is the counter the window is defined on, so
    dedup and window can never disagree about what "the same epoch" means.
    """
    out, idx = [], {}
    for row in rows:
        key = row.get("agent_steps")
        if key in idx:
            out[idx[key]] = row
        else:
            idx[key] = len(out)
            out.append(row)
    return out


def gate_window(rows):
    """Deduped rows with agent_steps >= 0.9 * 1M and self-play NOT using a past
    policy, last up-to-10. A missing `self_play/used_past` reads 0.0 (self-play
    off ⇒ the hero never faced a frozen opponent) — the run this reader is
    registered for is `--no-self-play`, so that key is absent from every row and
    the filter is a no-op; it is kept because the convention, not the run,
    defines the window."""
    lo = WINDOW_FRAC * PARTICIPATING_TIMESTEPS
    w = [
        r for r in rows
        if float(r.get("agent_steps", -1.0)) >= lo and float(r.get(USED_PAST_KEY, 0.0)) == 0.0
    ]
    return w[-WINDOW_MAX_ROWS:]


def rows_with(rows, key):
    """Rows carrying `key` with a non-null value (the prereg evaluates each
    assertion "over rows that carry the named key")."""
    return [r for r in rows if r.get(key) is not None]


def weighted_sum(rows, key):
    """sum_rows(value * episodes) — one side of an episode-weighted aggregate.
    See the module docstring for WHY the episode weight is mandatory."""
    return sum(float(r.get(key, 0.0)) * float(r.get(EPISODES_KEY, 0.0)) for r in rows)


def move_bin0_fraction(row):
    """`action_move_0 / sum(action_move_*)` for one row, or None if the row
    carries no move-histogram keys (or an all-zero histogram — an env that
    logged the bins but recorded no actions says nothing about the statue)."""
    bins = {k: float(v) for k, v in row.items() if MOVE_KEY_RE.match(k) and v is not None}
    total = sum(bins.values())
    if not bins or total <= 0.0 or "environment/action_move_0" not in bins:
        return None
    return bins["environment/action_move_0"] / total


def _check(n, name, ok, observed, invalidating=True):
    return {"n": n, "name": name, "ok": ok, "observed": observed, "invalidating": invalidating}


def run_preflights(rows, window, cap):
    """The five frozen pre-flight assertions, in order, as _check dicts.

    `rows` is every deduped row of the run (assertions 1, 2, 4 and the
    final-step read); `window` is the gate window (assertion 5 only). `cap` is
    the run's `aim_log_std_max` from config.json — the clamp the raw sigma must
    stay under; None (no provenance) makes assertion 5 unevaluable rather than
    silently true. Assertion 5 is flagged non-invalidating (module docstring).
    """
    checks = []

    # 1 — hero-only participation. Under `--opponent noop` + n_active=1 the
    # buffer-wide sum is 256 envs x 64 bptt x 1 hero row; the self-play value is
    # twice that (both teams train), so this is the tripwire for "the statue
    # override did not reach the trainer".
    part = rows_with(rows, PART_ROWS_KEY)
    vals = sorted({float(r[PART_ROWS_KEY]) for r in part})
    if not part:
        checks.append(
            _check(
                1, f"{PART_ROWS_KEY} == {EXPECTED_PARTICIPATING_ROWS:.0f} in every row"
                " carrying the key", False,
                f"key absent from all {len(rows)} rows (metrics key drift?)"))
    else:
        bad = [r for r in part if float(r[PART_ROWS_KEY]) != EXPECTED_PARTICIPATING_ROWS]
        checks.append(
            _check(
                1, f"{PART_ROWS_KEY} == {EXPECTED_PARTICIPATING_ROWS:.0f} in every row"
                " carrying the key", not bad, f"{len(part) - len(bad)}/{len(part)} rows equal; "
                f"distinct values {[f'{v:.0f}' for v in vals]}"))

    # 2 — self-play really is off, so the hero team is constant (T) and
    # `game/kills_ct` below is unambiguously "kills BY the statue".
    seen = rows_with(rows, OPPONENT_TEAM_KEY)
    checks.append(
        _check(2, f"no {OPPONENT_TEAM_KEY} key in any row", not seen,
               f"{len(seen)}/{len(rows)} rows carry the key"))

    # 3 — the budget was actually delivered (the T3 hero-only budget fix gives
    # 999,424 participating steps for --timesteps 1000000).
    steps = [float(r["agent_steps"]) for r in rows if r.get("agent_steps") is not None]
    final = max(steps) if steps else 0.0
    floor = FINAL_AGENT_STEPS_FRAC * PARTICIPATING_TIMESTEPS
    checks.append(
        _check(3, f"final agent_steps >= {floor:.0f}",
               bool(steps) and final >= floor,
               f"{final:.0f}" if steps else "no row carries agent_steps"))

    # 4a — a statue cannot kill. Kills are keyed on the KILLER (cs2_rewards.h),
    # the hero is T, so `game/kills_ct` > 0 means the opponent acted.
    kct = rows_with(rows, KILLS_CT_KEY)
    name4a = f"statue inert: {KILLS_CT_KEY} == 0 in every row"
    if not kct:
        checks.append(
            _check("4a", name4a, False,
                   f"key absent from all {len(rows)} rows (metrics key drift?)"))
    else:
        worst = max(float(r[KILLS_CT_KEY]) for r in kct)
        checks.append(_check("4a", name4a, worst == 0.0, f"max over {len(kct)} rows = {worst:.4f}"))

    # 4b — ...and it does not MOVE. Fraction, not raw count: see module docstring.
    fracs = [f for f in (move_bin0_fraction(r) for r in rows) if f is not None]
    name4b = (f"statue inert: action_move_0 / sum(action_move_*) >= {MOVE_BIN0_MIN_FRAC}"
              " in every row")
    if not fracs:
        checks.append(
            _check(
                "4b", name4b, False, f"no usable action_move_* histogram in any of"
                f" {len(rows)} rows (metrics key drift?)"))
    else:
        checks.append(
            _check("4b", name4b,
                   min(fracs) >= MOVE_BIN0_MIN_FRAC,
                   f"min over {len(fracs)} rows = {min(fracs):.4f} (max {max(fracs):.4f})"))

    # 5 — sigma still inside the clamp band, so |raw - init| is a real
    # measurement rather than a censored one. NOT invalidating: a violation
    # voids the sigma routing only (⇒ SIGMA-CAPPED, route on kills alone).
    sig = rows_with(window, SIGMA_RAW_KEY)
    name5 = f"sigma measurable: {SIGMA_RAW_KEY} <= {cap} in every window row"
    if cap is None:
        checks.append(_check(5, name5, False, "run cap unknown (config.json provenance missing)"))
    elif not sig:
        # Absent ≠ violated: with no sigma at all neither the assertion nor the
        # routing can be evaluated, which is a harness finding ⇒ invalidating.
        checks.append(
            _check(5, name5, False, f"key absent from all {len(window)} window rows"
                   " (metrics key drift?)"))
    else:
        worst = max(float(r[SIGMA_RAW_KEY]) for r in sig)
        checks.append(
            _check(5,
                   name5,
                   worst <= cap,
                   f"max over {len(sig)} window rows = {worst:.4f}",
                   invalidating=False))
    return checks


def aggregates(window, init):
    """Episode-weighted window aggregates + sigma movement (all prereg §Decision
    rule quantities). `hit_rate` is None on a zero shots_fired denominator (it
    then cannot clear its threshold, and printing 0.000 would read as a
    measurement); `sigma_move` is None when no window row carries the raw key.
    """
    episodes = sum(float(r.get(EPISODES_KEY, 0.0)) for r in window)
    fired = weighted_sum(window, FIRED_KEY)
    on_target = weighted_sum(window, ON_TARGET_KEY)
    sig = [float(r[SIGMA_RAW_KEY]) for r in rows_with(window, SIGMA_RAW_KEY)]
    return {
        "episodes": episodes,
        "kills": weighted_sum(window, KILLS_KEY) / episodes if episodes > 0 else None,
        "shots_fired": fired,
        "shots_on_target": on_target,
        "hit_rate": on_target / fired if fired > 0 else None,
        "sigma_move": max(abs(v - init) for v in sig) if sig else None,
        "sigma_min": min(sig) if sig else None,
        "sigma_max": max(sig) if sig else None,
    }


def route(agg, sigma_capped):
    """(verdict, next-step sentence) from the prereg decision rule.

    Branch order is the prereg's: the PASS rule first (it never mentions sigma,
    so it survives a SIGMA-CAPPED run untouched), then — for a non-PASS run —
    rec 4(b) if sigma is capped, else the two sigma-movement branches.

    The prereg's routing table does not cover every non-PASS corner (e.g. kills
    in [0.1, 0.5) with sigma NOT moving, or kills >= 0.5 with hit rate < 0.4).
    Those exit as FAIL-unrouted rather than being quietly folded into a
    neighbouring branch: an unroutable result is a fact about the pre-registered
    rules and belongs in the ledger for the controller to adjudicate.
    """
    kills = agg["kills"]
    hit = agg["hit_rate"]
    moved = agg["sigma_move"] is not None and agg["sigma_move"] >= SIGMA_MOVED_MIN
    if kills is not None and kills >= KILLS_PASS and hit is not None and hit >= HIT_RATE_PASS:
        return PASS, "H1 supported — next step: Rung 1b (random-walker opponent) spec"
    if sigma_capped:
        return (f"{FAIL} SIGMA-CAPPED: gradient present, exploration-starved",
                "sigma routing void, routed on kills alone — next step: rec 4(b)"
                " (omega_max / sigma-band redesign)")
    if kills is not None and kills < KILLS_UNTRAINED_MAX and not moved:
        return (f"{FAIL}-aim-head-untrained",
                "the aim head is not being trained — next step: rec 6 (discretised"
                " relative yaw bins), NOT reward changes")
    if kills is not None and kills < KILLS_PASS and moved:
        return (f"{FAIL}-aim", "next step: rec 1 (forced shoot + hit +0.1); if rec 1's own"
                " smoke fails, rec 4(b) before rec 2")
    return (f"{FAIL}-unrouted",
            "result falls outside the pre-registered routing branches — controller"
            " adjudicates in the ledger")


def read_run(run_dir):
    """Full report dict for one run dir. Never raises on a data problem: every
    structural defect becomes an `invalid` reason, so the caller always gets a
    printable report and a verdict."""
    run_dir = Path(run_dir)
    rep = {
        "run_dir": run_dir,
        "raw_rows": 0,
        "rows": [],
        "window": [],
        "invalid": [],
        "checks": [],
        "agg": None,
        "cap": None,
        "init": None,
        "sigma_capped": False,
        "verdict": INVALID,
        "next_step": "",
    }
    try:
        cfg = json.loads((run_dir / "config.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        rep["invalid"].append(f"config.json unreadable: {type(exc).__name__}: {exc}")
        cfg = {}
    try:
        raw = load_rows(run_dir)
    except OSError as exc:
        rep["invalid"].append(f"metrics.jsonl unreadable: {type(exc).__name__}: {exc}")
        raw = []
    rep["raw_rows"] = len(raw)
    rep["rows"] = rows = dedupe_by_agent_steps(raw)
    rep["window"] = window = gate_window(rows)

    # Provenance the routing needs. Both are written by train() into config.json
    # (`aim_log_std_init` is DERIVED from the cap — never re-derive it here, or
    # config and reader can drift; cs2rl.policy.resolve_aim_log_std_init).
    rep["cap"] = cfg.get("aim_log_std_max")
    rep["init"] = cfg.get("aim_log_std_init")
    if rep["cap"] is None:
        rep["invalid"].append("config.json lacks aim_log_std_max (pre-flight 5 not evaluable)")
    if rep["init"] is None:
        rep["invalid"].append("config.json lacks aim_log_std_init (sigma movement not evaluable)")

    if not rows:
        rep["invalid"].append("no metric rows")
        return rep
    if len(window) < MIN_WINDOW_ROWS:
        rep["invalid"].append(f"only {len(window)} rows in the window (need {MIN_WINDOW_ROWS}); "
                              f"agent_steps >= {WINDOW_FRAC * PARTICIPATING_TIMESTEPS:.0f}")

    cap = None if rep["cap"] is None else float(rep["cap"])
    rep["checks"] = run_preflights(rows, window, cap)
    for c in rep["checks"]:
        if not c["ok"] and c["invalidating"]:
            rep["invalid"].append(f"pre-flight {c['n']} failed: {c['name']} — {c['observed']}")
        elif not c["ok"]:
            rep["sigma_capped"] = True

    if rep["window"] and rep["init"] is not None:
        rep["agg"] = agg = aggregates(window, float(rep["init"]))
        # Fail-closed harness guard (rung1_gate's "zero episodes in W"): with no
        # finished episodes every episode-weighted aggregate is 0/0, so there is
        # nothing to route on.
        if not agg["episodes"] > 0:
            rep["invalid"].append("zero episodes in the window (aggregates undefined)")

    if rep["invalid"]:
        rep["verdict"] = INVALID
        rep["next_step"] = "fix the harness defect and relaunch fresh — the smoke did not run" \
                           " as registered; a pre-flight failure disconfirms nothing"
    else:
        rep["verdict"], rep["next_step"] = route(rep["agg"], rep["sigma_capped"])
    return rep


def _fmt(v, spec=".4f"):
    return "n/a" if v is None else format(v, spec)


def print_report(rep, file=None):
    """Human-readable report. `file=None` resolves sys.stdout at CALL time (a
    default bound at import would bypass pytest's capsys)."""
    file = file or sys.stdout
    print(f"rung1a smoke read — {rep['run_dir']}", file=file)
    print(
        f"  rows: {rep['raw_rows']} in metrics.jsonl, {len(rep['rows'])} after "
        f"dedup-by-agent_steps; window {len(rep['window'])} rows "
        f"(agent_steps >= {WINDOW_FRAC * PARTICIPATING_TIMESTEPS:.0f}, last "
        f"<= {WINDOW_MAX_ROWS}, need >= {MIN_WINDOW_ROWS})",
        file=file)
    print(
        f"  provenance: aim_log_std_max={rep['cap']} aim_log_std_init={rep['init']}"
        " (config.json)",
        file=file)
    for c in rep["checks"]:
        print(f"  pre-flight {c['n']}: {c['name']}", file=file)
        print(f"      observed {c['observed']} -> {'ok' if c['ok'] else 'FAIL'}", file=file)
    agg = rep["agg"]
    if agg:
        print(
            f"  window aggregates (episode-weighted over {len(rep['window'])} rows, "
            f"{agg['episodes']:.0f} episodes):",
            file=file)
        print(
            f"      weighted {KILLS_KEY:<26} = {_fmt(agg['kills'])}"
            f"   (PASS needs >= {KILLS_PASS})",
            file=file)
        print(
            f"      weighted on_target/fired{'':<9} = {_fmt(agg['hit_rate'])}"
            f"   (PASS needs >= {HIT_RATE_PASS};"
            f" sums {agg['shots_on_target']:.1f}/{agg['shots_fired']:.1f})",
            file=file)
        print(
            f"      max |sigma_raw - init|{'':<11} = {_fmt(agg['sigma_move'])}"
            f"   ('moved' means >= {SIGMA_MOVED_MIN}; raw in"
            f" [{_fmt(agg['sigma_min'])}, {_fmt(agg['sigma_max'])}])",
            file=file)
    for reason in rep["invalid"]:
        print(f"  INVALID: {reason}", file=file)
    if rep["sigma_capped"]:
        print(
            "  SIGMA-CAPPED: gradient present, exploration-starved — the sigma-movement"
            " routing is void",
            file=file)
    print(f"VERDICT: {rep['verdict']} — {rep['next_step']}", file=file)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cs2rl.experiment.smoke_read",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("run_dir",
                    nargs="?",
                    type=Path,
                    default=RUN_DIR_DEFAULT,
                    help="run dir holding metrics.jsonl + config.json "
                    f"(default {RUN_DIR_DEFAULT})")
    args = ap.parse_args(argv)
    rep = read_run(args.run_dir)
    print_report(rep)
    return EXIT_CODES[FAIL if rep["verdict"].startswith(FAIL) else rep["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
