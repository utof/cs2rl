#!/usr/bin/env python
"""Duel evaluation of a policy checkpoint against the #152 L2 duel bots (Rung 1c's instrument).

WHAT
----
Plays a checkpoint against ``eval.duel_bot`` opponents (oracle_duel's ``PROFILE[,dial=value...]``
specs, default ``OPPONENTS``) in oracle_duel's harness: ``run_duel`` over oracle_statue's
``run_check``, the arena duel with one agent a side, pin_pitch, crouch and jump off, the bots
deciding from their observation rows. The policy is loaded and played as aim_eval does it
(``aim_eval.load_checkpoint``: built with the run's config.json, a sim-v3 encoder widened onto
the sim-v4 obs; ``aim_eval.build_policy_hero``), by default in the mode the Rung 1b-2 gate reads:
aim ``mean``, LSTM ``carried`` through the round. The cone and the hit model default to the
run's config.json settings, as in aim_eval; ``--accuracy`` and ``--hitboxes`` override them.

Per opponent it also plays two reference heroes on the same seed and side: ``perfect`` (the
``DuelBotParams()`` bot) and ``mirror`` (the opponent's own spec), so the RL gap is read off the
same seeds. A cell is one hero x one opponent x one side. It prints one row per cell and writes
every number, per-round rows included, to ``--json``. Exit status 0 whenever the run completes:
it has no verdict (a Rung 1c gate reads the JSON).

    env PYTHONPATH=<checkout>/src UV_NO_SYNC=1 CUDA_VISIBLE_DEVICES= .venv/bin/python \\
        -m cs2rl.experiment.duel_eval <run>/checkpoints/<checkpoint>.pt --json out.json

PER CELL
--------
Every number comes from the C state through ``oracle_duel.DuelRecorder``, whose docstring holds
the definitions of FAST, ENGAGEMENT, FIRST SHOT and TIME TO DAMAGE; nothing here re-implements
combat.
  * outcomes (win, loss, double: both died on one tick, timeout) and the hero's share of the
    decided rounds with its 95% Wilson interval (``oracle_duel.summary``), and time to kill,
    the median tick on which the opponent died;
  * per side (``hero_`` / ``opp_`` keys): fired, hit, hit per fired; fast per fired, the share
    of shots fired at or above 0.34 x vmax (research G2, the counter-strafe failure rate), and
    hit per fired of the fast and of the slow shots; the first-shot hit rate per engagement
    with its Wilson interval, and shots per kill (G3); time to damage, its median over the
    rounds whose first hit came under 1 s, and, of the rounds with a first hit, the share
    whose first hit came at or over 1 s, the ones Leetify excludes (G4); hp lost and lifetime per round, lifetime being the death
    tick, else the round's last tick (G5); and the moving-while-visible-not-firing share, the
    share of idle engagement ticks that were fast (G5, the ADAD signature);
  * ``accounting_ok``: each side's fired, hit and fast shots sum to the env's StepStats totals,
    and no visibility slot went unmatched (oracle_duel's accounting check).

SIDES
-----
``--sides`` defaults to T. The Rung 1b-2 runs trained the learner on T only (under a scripted
``--opponent`` SelfPlayManager keeps its initial hero team, T, for the whole run), and the obs
is not side-agnostic: obs[24] is the team flag, and position and facing are absolute. A policy
trained on both sides can be read with ``--sides T CT``; the CT cells run at seed + 10000,
oracle_duel's arrangement.

PITFALLS
--------
* The bots aim their pitch (hit_geometry), a Rung 1 policy's pitch head is masked. The arena
  runs pin_pitch=1, which holds every agent's pitch at 0, so neither side has an advantage.
* Each policy cell forks the torch RNG and seeds it with its arrangement's seed, so cells
  reproduce in any order (aim_eval's rule). The bots draw from their own Generators.
* Never writes to ``outputs/``.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from cs2rl.eval.duel_bot import TICK_RATE
from cs2rl.experiment import aim_eval
from cs2rl.experiment import oracle_duel as D

# The Rung 1c zero-shot opponents: the CS2 profiles at the L2 default style (stop), and the
# hard bot in the two other moving styles.
OPPONENTS = ("easy,style=stop", "normal,style=stop", "hard,style=stop", "hard,style=run",
             "hard,style=adad", "expert,style=stop")
REFERENCE_HEROES = ("perfect", "mirror")
SIDES = D.SIDES
# Leetify's time to damage leaves out a first hit at or over 1 s ("trigger discipline plays").
TTD_CUTOFF_TICKS = TICK_RATE


def policy_hero(policy, aim: str, lstm: str, bptt_horizon: int):
    """A ``run_duel`` hero factory: aim_eval's policy hero for the mode, whatever the row.

    ``PolicyActor`` fills every row and run_check keeps its side's rows, so the same actor
    plays T or CT."""
    aim_eval.build_policy_hero(policy, aim, lstm, bptt_horizon)        # rejects an unknown mode
    return lambda row, rng: aim_eval.build_policy_hero(policy, aim, lstm, bptt_horizon)


def _rate(k, n):
    return k / n if n else None


def _median(values):
    return float(np.median(values)) if values else None


def _side_summary(rows, who: str, kills: int, died_key: str) -> dict:
    """One side's quantities (the module docstring's list), unprefixed."""

    def col(k):
        return [r[f"{who}_{k}"] for r in rows]

    fired, hit, fast, fast_hit = (sum(col(k)) for k in ("fired", "hit", "fast", "fast_hit"))
    firsts = [
        h for f, h in zip(col("first_fire"), col("first_fire_hit"), strict=True) if f is not None
    ]
    ttd = [t for t in col("first_hit") if t is not None]
    quick = [t for t in ttd if t < TTD_CUTOFF_TICKS]
    idle = sum(col("idle"))
    return dict(hit_per_fired=_rate(hit, fired),
                fast_per_fired=_rate(fast, fired),
                hit_per_fast=_rate(fast_hit, fast),
                hit_per_slow=_rate(hit - fast_hit, fired - fast),
                first_shots=len(firsts),
                first_shot_hits=sum(firsts),
                first_shot_hit_rate=_rate(sum(firsts), len(firsts)),
                first_shot_ci=D.wilson(sum(firsts), len(firsts)),
                shots_per_kill=_rate(fired, kills),
                ttd_rounds=len(ttd),
                ttd_median=_median(quick),
                ttd_1s_share=_rate(len(ttd) - len(quick), len(ttd)),
                hp_lost_mean=float(np.mean(col("hp_lost"))),
                lifetime_mean=float(
                    np.mean([r["ticks"] if r[died_key] is None else r[died_key] for r in rows])),
                idle_ticks=idle,
                moving_idle_share=_rate(sum(col("idle_fast")), idle))


def cell_summary(res: dict) -> dict:
    """oracle_duel.summary of the cell's rows plus each side's quantities (module docstring)
    and the accounting check."""
    rows = res["rows"]
    s = D.summary(rows)
    for who, kills, died_key in (("hero", s["win"] + s["double"], "death_tick"),
                                 ("opp", s["loss"] + s["double"], "kill_tick")):
        s.update((f"{who}_{k}", v) for k, v in _side_summary(rows, who, kills, died_key).items())
    acc = res["accounting"]
    s["accounting"] = acc
    s["accounting_ok"] = D.shots_accounted(acc) and all(a["unmatched"] == 0 for a in acc)
    return s


def run_eval(policy,
             bptt_horizon: int,
             rounds: int = 200,
             seed: int = 0,
             opponents=OPPONENTS,
             sides=("T", ),
             aim: str = "mean",
             lstm: str = "carried",
             accuracy: bool | None = None,
             hitboxes: bool | None = None,
             references=REFERENCE_HEROES) -> dict:
    """Every opponent x side x (the policy + each reference hero); returns the JSON dict."""
    import torch
    t0 = time.perf_counter()
    make_policy = policy_hero(policy, aim, lstm, bptt_horizon)
    cells = []
    for spec in opponents:
        opp = D.parse_bot(spec)
        heroes = [("policy", make_policy)]
        heroes += [(h, D.parse_bot(spec if h == "mirror" else h)) for h in references]
        for side in sides:
            k = SIDES.index(side)
            for name, hero in heroes:
                c0 = time.perf_counter()
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed + k * D.CT_SEED_OFFSET)
                    res = D.run_duel(hero, opp, rounds, seed, True, accuracy, hitboxes, sides=(k, ))
                cell = {"opponent": spec, "hero": name, "side": side, **cell_summary(res)}
                cell.update(accuracy=res["accuracy"],
                            hitboxes=res["hitboxes"],
                            wall_s=time.perf_counter() - c0,
                            per_round=res["rows"])
                cells.append(cell)
    return {
        "rounds": rounds,
        "seed": seed,
        "sides": list(sides),
        "aim": aim,
        "lstm": lstm,
        "bptt_horizon": bptt_horizon,
        "accuracy": cells[0]["accuracy"],
        "hitboxes": cells[0]["hitboxes"],
        "cells": cells,
        "wall_s": time.perf_counter() - t0
    }


def _fmt(v, spec):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def format_table(out: dict) -> str:
    """One row per cell; the per-round rows are in the JSON."""
    wo = max(len(c["opponent"]) for c in out["cells"])
    src = f", from {out['accuracy_source']}" if out.get("accuracy_source") else ""
    hb_src = f", from {out['hitboxes_source']}" if out.get("hitboxes_source") else ""
    lines = [
        f"duel_eval: {out['rounds']} rounds per cell, seed {out['seed']}, sides "
        f"{' '.join(out['sides'])}, policy aim {out['aim']} lstm {out['lstm']}; cone "
        f"{'ON' if out['accuracy'] else 'off'}{src}; hit model "
        f"{'capsule hitboxes' if out['hitboxes'] else 'legacy ellipsoid'}{hb_src}",
        "share = hero wins / decided [95% Wilson]; fast = fired at >= 0.34 x vmax; 1st = "
        "first-shot hit rate; ttd = time to damage (ticks, median under 1 s; >=1s its share); "
        "mov = moving share of idle visible ticks; life, hp- = per-round means",
        f"{'opponent':<{wo}} {'hero':<7} side W/L/D/TO        share  [95% CI]         hit/f "
        f"fast/f opp-fast 1st   [95% CI]       ttd  >=1s  ttk   hp-   life  mov   opp-mov acct",
    ]
    for c in out["cells"]:
        lo, hi = c["hero_ci"]
        flo, fhi = c["hero_first_shot_ci"]
        wldt = f"{c['win']}/{c['loss']}/{c['double']}/{c['timeout']}"
        lines.append(
            f"{c['opponent']:<{wo}} {c['hero']:<7} {c['side']:<4} {wldt:<15} "
            f"{_fmt(c['hero_share'], '.4f')} [{_fmt(lo, '.4f')}, {_fmt(hi, '.4f')}] "
            f"{_fmt(c['hero_hit_per_fired'], '.3f')} {_fmt(c['hero_fast_per_fired'], '5.3f')}  "
            f"{_fmt(c['opp_fast_per_fired'], '5.3f')}    "
            f"{_fmt(c['hero_first_shot_hit_rate'], '.3f')} "
            f"[{_fmt(flo, '.3f')}, {_fmt(fhi, '.3f')}] "
            f"{_fmt(c['hero_ttd_median'], '4.1f')} {_fmt(c['hero_ttd_1s_share'], '5.3f')} "
            f"{_fmt(c['ttk_median'], '4.1f')} {c['hero_hp_lost_mean']:5.1f} "
            f"{c['hero_lifetime_mean']:5.1f} {_fmt(c['hero_moving_idle_share'], '5.3f')} "
            f"{_fmt(c['opp_moving_idle_share'], '5.3f')}   {'ok' if c['accounting_ok'] else 'XX'}")
        if not c["accounting_ok"]:
            lines.append(f"  <-- WARNING: the per-side shots do not sum to the env totals, or a "
                         f"visibility slot went unmatched: {c['accounting']}")
    lines.append(f"wall {out['wall_s']:.1f} s")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cs2rl.experiment.duel_eval",
                                 description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="a policy .pt; its run's config.json must sit beside it")
    ap.add_argument("--opponents",
                    nargs="+",
                    default=list(OPPONENTS),
                    help="duel bot specs, oracle_duel's PROFILE[,dial=value...] (default: "
                    f"{' '.join(OPPONENTS)})")
    ap.add_argument("--rounds", type=int, default=200, help="rounds per cell (default 200)")
    ap.add_argument("--seed", type=int, default=0, help="env, bot and torch seed (default 0)")
    ap.add_argument("--sides",
                    nargs="+",
                    choices=SIDES,
                    default=["T"],
                    help="the hero's sides (default T: the side Rung 1b trained on)")
    ap.add_argument("--references",
                    nargs="*",
                    choices=REFERENCE_HEROES,
                    default=list(REFERENCE_HEROES),
                    help="the reference heroes per opponent (default: perfect mirror; none: "
                    "the policy cells only)")
    ap.add_argument("--aim", choices=("sample", "mean"), default="mean", help="default mean")
    ap.add_argument("--lstm",
                    choices=("carried", "zeroed"),
                    default="carried",
                    help="default carried")
    ap.add_argument("--json", type=Path, help="write the full result (per-round rows too) here")
    ap.add_argument("--accuracy",
                    type=int,
                    choices=(0, 1),
                    default=None,
                    help="the movement-inaccuracy cone: 1 on, 0 off. Default: the run's "
                    "config.json setting, as in aim_eval.")
    ap.add_argument("--hitboxes",
                    type=int,
                    choices=(0, 1),
                    default=None,
                    help="the hit model: 1 the capsule hitboxes, 0 the legacy ellipsoid. "
                    "Default: the run's config.json setting, as in aim_eval.")
    args = ap.parse_args(argv)
    for spec in args.opponents:
        try:
            D.parse_bot(spec)
        except ValueError as e:
            ap.error(str(e))
    policy, cfg = aim_eval.load_checkpoint(args.checkpoint)
    if args.accuracy is None:
        accuracy, source = aim_eval.trained_accuracy(cfg), "the run's config.json"
    else:
        accuracy, source = bool(args.accuracy), "--accuracy"
    if args.hitboxes is None:
        hitboxes, hb_source = aim_eval.trained_hitboxes(cfg), "the run's config.json"
    else:
        hitboxes, hb_source = bool(args.hitboxes), "--hitboxes"
    out = run_eval(policy,
                   int(cfg["bptt_horizon"]),
                   rounds=args.rounds,
                   seed=args.seed,
                   opponents=args.opponents,
                   sides=args.sides,
                   aim=args.aim,
                   lstm=args.lstm,
                   accuracy=accuracy,
                   hitboxes=hitboxes,
                   references=args.references)
    out.update(checkpoint=str(args.checkpoint),
               accuracy_source=source,
               hitboxes_source=hb_source,
               env_mismatches=aim_eval.env_mismatches(cfg, out["accuracy"], out["hitboxes"]))
    for m in out["env_mismatches"]:
        print(f"WARNING: the run trained a different env, {m}")
    print(format_table(out))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
