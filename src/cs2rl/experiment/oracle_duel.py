#!/usr/bin/env python
"""Oracle-vs-oracle duel check: layer L2 of the #152 cheat-bot ladder.

WHAT
----
Two ``eval.duel_bot`` bots duel in oracle_statue's arena (``run_check``: the Rung 1a env, one
agent a side, pin_pitch, crouch and jump off, the cone and the hit model as ``--accuracy`` and
``--hitboxes`` say). The hero plays ``--rounds`` rounds on the T row, then ``--rounds`` on the
CT row against the same opponent, so a side advantage cancels in the hero's share and shows in
the T share. It prints one row per round, the per-side totals and one PASS/FAIL line. Exit
status 0 = PASS, 1 = FAIL.

    env UV_NO_SYNC=1 PYTHONPATH=<checkout>/src .venv/bin/python -m cs2rl.experiment.oracle_duel
    ... -m cs2rl.experiment.oracle_duel --hero hard,style=stop --opponent hard,style=run --expect hero

A bot is ``PROFILE[,dial=value...]``: a CS2 profile of ``duel_bot.PROFILES`` or ``perfect``
(``DuelBotParams()``), then any ``DuelBotParams`` field (``lead=0``, ``reaction_ticks=3``,
``style=adad``). The opponent may also be ``walker``, oracle_tracker's L1 walker (it never
shoots). The default is the mirror ``hard,style=stop`` (DEFAULT_BOT).

Each row holds the arrangement's seed, the round, both spawns, the hero's side, the outcome
for the hero (win, loss, double: both died on one tick, timeout), the tick it ended, and each
side's shots: fired, hit, fired at or above 0.34 x vmax ("fast", the cone's start). A round
replays with the same --seed and arrangement and ``--rounds`` past its index: the env's RNG and
the bots' Generators advance through the run. The hero-on-CT arrangement runs at seed + 10000.

THE VERDICT
-----------
  * accounting: in each arrangement each side's fired, hit and fast shots, read off the C state
    (``DuelRecorder``), sum to the env's StepStats totals; no unmatched visibility slot;
  * winnable: at most MAX_TIMEOUT_SHARE of the rounds time out;
  * symmetric: the 95% Wilson interval of the T side's share of the decided rounds contains 0.5;
  * ``--expect``: ``even`` (the default when the two bots are the same): the hero's share's
    interval contains 0.5; ``hero`` / ``opponent``: it lies above / below 0.5; ``any``: no check.
A decided round has one survivor; a double kill and a time-out are draws.

WHAT L2 CERTIFIES, WHAT IT DOES NOT: the "L2" paragraph of src/cs2rl/experiment/CONTEXT.md.

PITFALLS
--------
* run_check names its rows hero (T) and statue (CT): here its ``hero`` argument is whichever
  actor plays the T row, and ``per_episode``'s ``hero_xy`` / ``statue_xy`` are the T and CT spawns.
* The bots decide from their observation rows (as a trainer's would) unless --ground-truth.
* The bots' pitch is aimed (hit_geometry); a Rung 1 policy's is pinned. Moot under pin_pitch=1.
* Never writes to outputs/.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from typing import Any

import numpy as np
from scipy.stats import binomtest

from cs2rl.eval.duel_bot import ACCURACY_THRESHOLD_FRAC, PROFILES, DuelBotActor, DuelBotParams
from cs2rl.experiment import oracle_statue as L0

SIDES = ("T", "CT")
ROWS = (L0.HERO, L0.STATUE)            # the T row and the CT row
DEFAULT_BOT = "hard,style=stop"
WALKER = "walker"
CT_SEED_OFFSET = 10_000
MAX_TIMEOUT_SHARE = 0.05


def parse_bot(spec: str):
    """``PROFILE[,dial=value...]`` -> DuelBotParams, or WALKER. Raises ValueError."""
    name, *dials = spec.split(",")
    if name == WALKER and not dials:
        return WALKER
    if name != "perfect" and name not in PROFILES:
        raise ValueError(f"unknown bot {name!r}: one of perfect, {', '.join(PROFILES)}, {WALKER}")
    params = DuelBotParams() if name == "perfect" else PROFILES[name]
    types = {f.name: type(getattr(params, f.name)) for f in dataclasses.fields(params)}
    kw: dict[str, Any] = {}
    for d in dials:
        key, _, value = d.partition("=")
        if key not in types:
            raise ValueError(f"unknown dial {key!r}: one of {', '.join(types)}")
        kw[key] = bool(int(value)) if types[key] is bool else types[key](value)
    return dataclasses.replace(params, **kw)


def _actor(spec, row: int, rng: np.random.Generator, obs_only: bool):
    """The actor for ``row``: a DuelBotParams bot, the L1 walker, or ``spec(row, rng)`` when
    ``spec`` is a factory (cs2rl.experiment.duel_eval passes its policy hero this way)."""
    if callable(spec):
        return spec(row, rng)
    if spec == WALKER:
        from cs2rl.eval.walker import WalkerActor
        return WalkerActor(rng, [row])
    return DuelBotActor([row], spec, rng, obs_only)


# The per-side quantities DuelRecorder keeps for every round; run_duel copies each into the
# round's row as hero_<key> and opp_<key>.
PER_SIDE = ("fired", "hit", "fast", "fast_hit", "first_fire", "first_fire_hit", "first_hit",
            "hp_lost", "idle", "idle_fast")


class DuelRecorder:
    """``run_check``'s ``on_step``: per round, each side's shots, the tick each side died, and
    the per-side quantities the Rung 1c duel evaluation reads (cs2rl.experiment.duel_eval).

    Read off the C state after each step (``st``) and the state the actors saw (``st_prev``,
    the reset state on tick 1). Index 0 is the T side. The definitions, one place each:

      * FIRED: the side's ``fired_this_tick`` is set. HIT: the other side's hp fell this tick
        (one enemy each, so a drop is the other side's hit).
      * FAST (``fast``): the side's planar speed after the move is at or above 0.34 x vmax,
        vmax being the move speed of the weapon it holds (``weapon_slot``), computed in
        float32 as cs2_combat.h computes its ``shots_fired_fast``. ``fast`` counts the fired
        ticks that were fast, ``fast_hit`` those that also hit. WHY the held weapon: a policy
        can switch to its pistol (240 u/s against the rifle's 220); a rifle threshold
        counted a pistol shot at 81.19 u/s fast that C did not (lane J, 2026-10-11).
      * ENGAGEMENT: the ticks whose observed state ``st_prev`` has both sides alive. The arena
        has 2D line of sight on every tick and the sim has no field of view
        (cs2_combat.h build_vis_matrix), so a round is exactly one engagement, from tick 1
        to the round's last tick. KNOWN LIMIT: on a map with occlusion, gate on visibility
        and split a round into its engagements.
      * FIRST SHOT: the side's first fired tick of the engagement (``first_fire``, None if it
        never fired); ``first_fire_hit``: that tick was a HIT.
      * TIME TO DAMAGE: from the observed state that first showed the enemy (the reset
        state, tick 0) to the tick of the side's first HIT, so ``first_hit`` is it in ticks.
      * ``hp_lost``: the hp the side lost over the round (hp is clamped at 0 in C).
      * IDLE: an engagement tick on which the side did not fire (``idle``); ``idle_fast``:
        those that were FAST, the moving-while-visible-not-firing ticks.
    """

    def __init__(self, move_speeds):
        """``move_speeds``: each weapon slot's move speed (binding.weapon_defs())."""
        self.rounds: list[dict] = []
        self.thr = np.float32(ACCURACY_THRESHOLD_FRAC) * np.asarray(move_speeds, np.float32)

    def fast(self, st, i) -> bool:
        """Row ``i``'s planar speed in ``st`` is at or above 0.34 x its weapon's move speed
        (float32, as in C)."""
        vx, vy = np.float32(st["vx"][i]), np.float32(st["vy"][i])
        return bool(np.sqrt(vx * vx + vy * vy) >= self.thr[st["weapon_slot"][i]])

    def __call__(self, tick, st_prev, st):
        if tick == 1:
            self.rounds.append({
                **{
                    k: [0, 0]
                    for k in ("fired", "hit", "fast", "fast_hit", "hp_lost", "idle", "idle_fast")
                },
                **{
                    k: [None, None]
                    for k in ("died", "first_fire", "first_fire_hit", "first_hit")
                },
            })
        r = self.rounds[-1]
        r["ticks"] = tick
        engaged = bool(st_prev["alive"][ROWS[0]] and st_prev["alive"][ROWS[1]])
        for s, i in enumerate(ROWS):
            other = ROWS[1 - s]
            fired, fast = bool(st["fired_this_tick"][i]), self.fast(st, i)
            hit = bool(st["hp"][other] < st_prev["hp"][other])
            if fired:
                r["fired"][s] += 1
                r["fast"][s] += fast
                r["fast_hit"][s] += fast and hit
                if engaged and r["first_fire"][s] is None:
                    r["first_fire"][s], r["first_fire_hit"][s] = tick, hit
            elif engaged:
                r["idle"][s] += 1
                r["idle_fast"][s] += fast
            r["hit"][s] += hit
            if hit and r["first_hit"][s] is None:
                r["first_hit"][s] = tick
            r["hp_lost"][s] += max(int(st_prev["hp"][i]) - int(st["hp"][i]), 0)
            if st_prev["alive"][i] and not st["alive"][i]:
                r["died"][s] = tick


def _outcome(hero_died, opp_died) -> str:
    if hero_died is None:
        return "timeout" if opp_died is None else "win"
    return "loss" if opp_died is None else "double"


def run_duel(hero,
             opponent,
             rounds: int = 500,
             seed: int = 0,
             obs_only: bool = True,
             accuracy: bool | None = None,
             hitboxes: bool | None = None,
             sides=(0, 1)) -> dict:
    """The arrangements in ``sides`` (0: the hero on the T row at ``seed``, 1: on the CT row at
    ``seed + CT_SEED_OFFSET``); returns {"rows": per-round rows, "accounting": one entry per
    arrangement played, ...}. ``hero`` and ``opponent`` are what ``_actor`` takes."""
    from cs2rl.env.c import binding
    rows, accounting, env_flags = [], [], (None, None)
    for hero_side, arr_seed in ((0, seed), (1, seed + CT_SEED_OFFSET)):
        if hero_side not in sides:
            continue
        hero_rng, opp_rng = np.random.default_rng([arr_seed,
                                                   0]), np.random.default_rng([arr_seed, 1])
        bots = [
            _actor(hero, ROWS[hero_side], hero_rng, obs_only),
            _actor(opponent, ROWS[1 - hero_side], opp_rng, obs_only)
        ]
        t_bot, ct_bot = (bots if hero_side == 0 else bots[::-1])
        rec = DuelRecorder([w["move_speed"] for w in binding.weapon_defs()])
        res = L0.run_check(episodes=rounds,
                           seed=arr_seed,
                           hero=t_bot,
                           opponent=ct_bot,
                           on_step=rec,
                           accuracy=accuracy,
                           hitboxes=hitboxes)
        env_flags = (res["accuracy"], res["hitboxes"])
        accounting.append({
            k: (sum(r[k][0] for r in rec.rounds), sum(r[k][1] for r in rec.rounds), res[env_key])
            for k, env_key in (("fired", "shots_fired"), ("hit", "shots_hit"), ("fast",
                                                                                "shots_fired_fast"))
        })
        accounting[-1]["unmatched"] = res["unmatched_vis_slots"]
        for ep, r in zip(res["per_episode"], rec.rounds, strict=True):
            h, o = hero_side, 1 - hero_side
            rows.append({
                "seed": arr_seed,
                "episode": ep["episode"],
                "t_xy": ep["hero_xy"],
                "ct_xy": ep["statue_xy"],
                "hero_side": SIDES[h],
                "ticks": r["ticks"],
                "outcome": _outcome(r["died"][h], r["died"][o]),
                "t_won": r["died"][1] is not None and r["died"][0] is None,
                "ct_won": r["died"][0] is not None and r["died"][1] is None,
                "kill_tick": r["died"][o],
                "death_tick": r["died"][h],
                **{
                    f"{who}_{k}": r[k][s]
                    for who, s in (("hero", h), ("opp", o))
                    for k in PER_SIDE
                }
            })
    return {
        "rows": rows,
        "accounting": accounting,
        "rounds": rounds,
        "seed": seed,
        "obs_only": obs_only,
        "accuracy": env_flags[0],
        "hitboxes": env_flags[1]
    }


def wilson(k: int, n: int) -> tuple[float, float]:
    """The 95% Wilson score interval of k successes in n (scipy); (nan, nan) for n = 0."""
    if n == 0:
        return (math.nan, math.nan)
    ci = binomtest(k, n).proportion_ci(method="wilson")
    return (float(ci.low), float(ci.high))


def summary(rows: list[dict]) -> dict:
    """The hero's and the T side's shares of the decided rounds, with Wilson intervals, and
    each side's totals."""
    n = {o: sum(r["outcome"] == o for r in rows) for o in ("win", "loss", "double", "timeout")}
    decided = n["win"] + n["loss"]
    t_wins = sum(r["t_won"] for r in rows)
    out = {
        **n, "rounds": len(rows),
        "decided": decided,
        "t_wins": t_wins,
        "hero_share": n["win"] / decided if decided else math.nan,
        "hero_ci": wilson(n["win"], decided),
        "t_share": t_wins / decided if decided else math.nan,
        "t_ci": wilson(t_wins, decided)
    }
    for who in ("hero", "opp"):
        for k in ("fired", "hit", "fast"):
            out[f"{who}_{k}"] = sum(r[f"{who}_{k}"] for r in rows)
    kills = [r["kill_tick"] for r in rows if r["kill_tick"] is not None]
    deaths = [r["death_tick"] for r in rows if r["death_tick"] is not None]
    out["ttk_median"] = float(np.median(kills)) if kills else math.nan
    out["ttd_median"] = float(np.median(deaths)) if deaths else math.nan
    return out


def shots_accounted(accounting) -> bool:
    """In every arrangement each side's fired, hit and fast shots sum to the env's totals."""
    return all(a[k][0] + a[k][1] == a[k][2] for a in accounting for k in ("fired", "hit", "fast"))


def verdict(res: dict, expect: str) -> tuple[bool, list[tuple[str, bool, str]]]:
    """(passed, [(name, ok, detail), ...]) — see THE VERDICT in the module docstring."""
    s = summary(res["rows"])
    acc = res["accounting"]
    sums_ok = shots_accounted(acc)
    detail = "; ".join(f"{k} {a[k][0]}+{a[k][1]}={a[k][2]}" for a in acc
                       for k in ("fired", "hit", "fast"))
    lo, hi = s["t_ci"]
    hlo, hhi = s["hero_ci"]
    # No decided round (a perfect mirror: every round a double kill) leaves both shares nan,
    # and nan fails both share checks: nothing was shown about who wins.
    none = "no decided round: every round a draw" if s["decided"] == 0 else None
    checks = [
        ("per-side shots sum to the env totals", sums_ok, detail),
        ("unmatched vis slots == 0", all(a["unmatched"] == 0
                                         for a in acc), str([a["unmatched"] for a in acc])),
        (f"timeouts <= {MAX_TIMEOUT_SHARE:.2f} of rounds", s["timeout"]
         <= MAX_TIMEOUT_SHARE * s["rounds"], f"{s['timeout']}/{s['rounds']}"),
        ("T share 95% CI contains 0.5", lo <= 0.5 <= hi, none
         or f"{s['t_share']:.4f} [{lo:.4f}, {hi:.4f}]"),
    ]
    rule = {"even": hlo <= 0.5 <= hhi, "hero": hlo > 0.5, "opponent": hhi < 0.5}
    if expect in rule:
        checks.append((f"hero share 95% CI: {expect}", rule[expect], none
                       or f"{s['hero_share']:.4f} [{hlo:.4f}, {hhi:.4f}]"))
    return all(ok for _, ok, _ in checks), checks


def format_report(res: dict, hero: str, opponent: str, expect: str) -> str:
    """The per-round table, the per-side totals and the verdict."""
    s = summary(res["rows"])
    passed, checks = verdict(res, expect)
    lines = [
        f"── duel (#152 L2): hero {hero} vs opponent {opponent} — arena-duel, n_active=1, "
        f"pin_pitch=1, cone {'on' if res['accuracy'] else 'off'}, "
        f"{'capsule hitboxes' if res['hitboxes'] else 'legacy ellipsoid'}, "
        f"{'obs-only' if res['obs_only'] else 'ground-truth'} bots ──",
        " seed    ep  T (x, y)      CT (x, y)    hero  outcome  tick  hero f/h/fast  opp f/h/fast"
    ]
    for r in res["rows"]:
        lines.append(
            f"{r['seed']:>5} {r['episode']:>5}  ({r['t_xy'][0]:4.0f},{r['t_xy'][1]:4.0f})"
            f"  ({r['ct_xy'][0]:4.0f},{r['ct_xy'][1]:4.0f})  {r['hero_side']:>4}  {r['outcome']:<7}"
            f"  {r['ticks']:>4}  {r['hero_fired']:>4}/{r['hero_hit']}/{r['hero_fast']:<5}"
            f"  {r['opp_fired']:>4}/{r['opp_hit']}/{r['opp_fast']}")
    for who, p, wins, deaths in (("hero", "hero", s["win"], s["loss"] + s["double"]),
                                 ("opponent", "opp", s["loss"], s["win"] + s["double"])):
        fired, hit, fast = s[p + "_fired"], s[p + "_hit"], s[p + "_fast"]
        lines.append(f"{who:<9} wins {wins}  deaths {deaths}  shots fired {fired}  hit {hit} "
                     f"({hit / max(fired, 1):.3f})  fast {fast} ({fast / max(fired, 1):.3f})")
    lines += [
        f"rounds {s['rounds']} ({res['rounds']} per side, seed {res['seed']}): decided {s['decided']}, "
        f"double kills {s['double']}, timeouts {s['timeout']}",
        f"time to kill (hero kills) median {s['ttk_median']:.1f}   time to death median "
        f"{s['ttd_median']:.1f} ticks", ""
    ]
    lines += [f"  [{'ok' if ok else 'XX'}] {name:<40} {detail}" for name, ok, detail in checks]
    lines += [
        "", "PASS — the duel is winnable, symmetric and accounted for"
        if passed else "FAIL — see the [XX] checks"
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cs2rl.experiment.oracle_duel",
                                 description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hero", default=DEFAULT_BOT, help=f"the hero bot (default {DEFAULT_BOT})")
    ap.add_argument("--opponent", help="the opponent bot, or walker (default: the hero's spec)")
    ap.add_argument("--rounds", type=int, default=500, help="rounds per side (default 500)")
    ap.add_argument("--seed", type=int, default=0, help="env and bot seed (default 0)")
    ap.add_argument("--expect",
                    choices=("even", "hero", "opponent", "any"),
                    help="the hero's share (default: even for a mirror, else any)")
    ap.add_argument("--ground-truth",
                    action="store_true",
                    help="the bots read the C state instead of their observation rows")
    ap.add_argument("--accuracy",
                    type=int,
                    choices=(0, 1),
                    default=None,
                    help="the movement-inaccuracy cone (EnvConfig.accuracy): 1 on, 0 off. "
                    "Default: EnvConfig.accuracy.")
    ap.add_argument("--hitboxes",
                    type=int,
                    choices=(0, 1),
                    default=None,
                    help="the hit model (EnvConfig.hitboxes): 1 the capsule hitboxes, 0 the legacy "
                    "ellipsoid. Default: EnvConfig.hitboxes.")
    args = ap.parse_args(argv)
    opponent = args.opponent or args.hero
    try:
        hero, opp = parse_bot(args.hero), parse_bot(opponent)
    except ValueError as e:
        ap.error(str(e))
    if hero == WALKER:
        ap.error("the hero must be a duel bot")
    expect = args.expect or ("even" if hero == opp else "any")
    res = run_duel(hero, opp, args.rounds, args.seed, not args.ground_truth,
                   None if args.accuracy is None else bool(args.accuracy),
                   None if args.hitboxes is None else bool(args.hitboxes))
    print(format_report(res, args.hero, opponent, expect))
    return 0 if verdict(res, expect)[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
