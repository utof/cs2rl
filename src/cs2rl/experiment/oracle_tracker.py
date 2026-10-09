#!/usr/bin/env python
"""Oracle-vs-walker tracking check: layer L1 of the #152 cheat-bot ladder.

WHAT
----
oracle_statue's harness with the statue replaced by a moving target. The env, the
hero (``OracleActor``, or ``ObsOracleActor`` under ``--obs-only``), the loop and the
L0 thresholds are oracle_statue's; the CT row is driven by
``eval.walker.WalkerActor`` (a random move direction held for 4..16 ticks, never
shooting or turning). It prints one row per episode, then the totals and one PASS/FAIL
line. Exit status 0 = PASS, 1 = FAIL.

    env UV_NO_SYNC=1 PYTHONPATH=<checkout>/src .venv/bin/python -m cs2rl.experiment.oracle_tracker
    env UV_NO_SYNC=1 PYTHONPATH=<checkout>/src .venv/bin/python -m cs2rl.experiment.oracle_tracker --obs-only

Each row holds the seed, the episode index, the spawn (both positions, the distance
and the hero's opening yaw error), the time to kill, the episode's shots, and the
walker's motion. A run is deterministic for a seed, but an episode cannot be replayed
alone: the C RNG and the walker's Generator both advance through the run. To replay
episode k, rerun with the same ``--seed`` and ``--episodes k+1``; the spawn columns
identify the row.

THE VERDICT
-----------
The L0 checks (``oracle_statue.verdict``: kill rate, median TTK, no stance-blocked
shot, and under ``--obs-only`` the obs-encoding checks) plus two walker-motion checks
read off the C state:

  * ``walker moving_frac``: the share of the walker's live post-step ticks on which
    its planar speed exceeded ``oracle_statue.OPP_MOVING_SPEED``;
  * ``walker net_disp median``: the median over episodes of the 2D distance from the
    walker's spawn to its last live position.

WHY the motion checks: the oracle kills a statue and the walker alike (200/200 at
median TTK 10 against both, 2026-10-09), so kills alone would pass a walker that never
moves. A statue reads 0 on both checks, and so does a walker whose move bins are all
0. The displacement floor also fails a walker that redraws its direction every tick:
it moves on every tick but jitters in place (median 21 u against the walker's 94 u).

WHAT L1 CERTIFIES
-----------------
That the oracle's turn-and-fire loop keeps killing a target that walks around the
arena at the walker's speeds. Under ``--obs-only`` it also certifies that the enemy
block of the obs tracks a moving target: the obs-only hero must pass with no blind
tick, and on 2026-10-09 it matched the ground-truth hero's TTK and shot count in every
episode.

WHAT IT DOES NOT CERTIFY
------------------------
* LEAD. The walker is too slow to need it. Its p90 speed is 220 u/s, about 14 u per
  tick, less than HIT_HALF_WIDTH (16 u, cs2_combat.h), so a shot aimed where the walker
  was still hits. Measured on 2026-10-09: ``OracleActor`` with its velocity lead
  switched off (``eval.baselines.TICK_DT = 0``) fired and hit exactly as often as
  with it (703 shots, 685 hits; the 18 misses match the statue run's 18), and
  ``ObsOracleActor``, which has no lead, matched it in every episode. A PASS here says
  nothing about leading a target; that needs a target fast enough that an unled shot
  misses.
* Occlusion, the last-known-position memory, a target that shoots back, crouching,
  jumping: the arena has permanent 2D LoS and the walker does none of the rest.
* Anything about a POLICY. Like L0, this is a precondition on the env and the obs.

PITFALLS
--------
* ``run_check``'s summary names stay "statue" (``statue_xy``, ``STATUE``): they name
  the CT row, which the walker now drives.
* The motion is read from the C state after each step, so the reset tick is not in
  it, and a kill ends the episode's sample: an episode whose kill lands on tick TTK
  contributes TTK - 1 ticks.
* This script never writes to ``outputs/`` and never touches training state.
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from cs2rl.eval.walker import WalkerActor
from cs2rl.experiment import oracle_statue as L0

# Walker-motion floors. Deliberately NOT CLI-configurable, like L0's thresholds.
# KNOWN LIMIT: wall stalls count as not moving, so this is calibrated to the default WalkerActor (a strafe-only
# walker reads 0.874 and fails); recalibrate before changing the walker's bins or holds (WalkerParams exists now; this walker uses none).
MIN_MOVING_FRAC = 0.90
MIN_NET_DISP_MEDIAN = 50.0             # u


def build_walker(seed: int) -> WalkerActor:
    """The opponent: ``WalkerActor`` on the CT row only, seeded from the run's seed."""
    return WalkerActor(np.random.default_rng(seed), [L0.STATUE])


def run_tracker(episodes: int = 200, seed: int = 0, obs_only: bool = False, opponent=None) -> dict:
    """``oracle_statue.run_check`` against the walker (or ``opponent``, for knock-outs)."""
    if opponent is None:
        opponent = build_walker(seed)
    return L0.run_check(episodes=episodes, seed=seed, obs_only=obs_only, opponent=opponent)


def verdict(res: dict) -> tuple[bool, list[tuple[str, bool, str]]]:
    """(passed, [(name, ok, detail), ...]): the L0 checks plus the walker-motion checks."""
    _, checks = L0.verdict(res)
    checks = checks + [
        (f"walker moving_frac >= {MIN_MOVING_FRAC:.2f}", res["opp_moving_frac"]
         >= MIN_MOVING_FRAC, f"{res['opp_moving_frac']:.3f}"),
        (f"walker net_disp median >= {MIN_NET_DISP_MEDIAN:.0f} u", res["opp_net_disp_median"]
         >= MIN_NET_DISP_MEDIAN, f"{res['opp_net_disp_median']:.1f}"),
    ]
    return all(ok for _, ok, _ in checks), checks


def episode_rows(res: dict) -> list[str]:
    """The per-episode table: a header line, then one line per episode."""
    lines = [
        "seed    ep  hero (x, y)    walker (x, y)   dist u  yaw deg  ttk  fired  hit  moving/alive  disp u"
    ]
    for p in res["per_episode"]:
        ttk = "-" if p["ttk"] is None else str(p["ttk"])
        lines.append(
            f"{res['seed']:>4} {p['episode']:>5}  ({p['hero_xy'][0]:5.0f},{p['hero_xy'][1]:5.0f})"
            f"  ({p['statue_xy'][0]:5.0f},{p['statue_xy'][1]:5.0f})  {p['spawn_dist']:6.1f}"
            f"  {math.degrees(p['yaw_err']):+7.1f}  {ttk:>3}  {p['shots_fired']:>5}"
            f"  {p['shots_hit']:>3}  {p['opp_moving_ticks']:>6}/{p['opp_alive_ticks']:<5}"
            f"  {p['opp_net_disp']:6.1f}")
    return lines


def format_report(res: dict) -> str:
    """The per-episode table, the totals every check rests on, and the verdict."""
    passed, checks = verdict(res)
    fired = max(res["shots_fired"], 1)
    lines = [
        f"── oracle vs walker (#152 L1) — arena-duel, n_active=1, pin_pitch=1, crouch=0, jump=0, "
        f"{'OBS-ONLY hero' if res['obs_only'] else 'ground-truth hero'} ──",
        *episode_rows(res),
        "",
        f"episodes                 {res['episodes']}  (seed {res['seed']}, round_time {res['round_time']})",
        f"kill rate                {res['kill_rate']:.3f}  ({res['kills']}/{res['episodes']})",
        f"time-to-kill (ticks)     median {res['ttk_median']:.1f}   p90 {res['ttk_p90']:.1f}   "
        f"censored {res['ttk_censored']}",
        f"shots_fired              {res['shots_fired']}   shots_hit {res['shots_hit']}  "
        f"({res['shots_hit'] / fired:.3f} of fired)",
        f"walker motion            moving_frac {res['opp_moving_frac']:.3f}   "
        f"net_disp median {res['opp_net_disp_median']:.1f} u",
        f"unmatched vis slots      {res['unmatched_vis_slots']}",
    ]
    if res["obs_only"]:
        lines += [
            f"obs blind ticks          {res['obs_blind_ticks']}",
            f"obs slot inconsistency   {res['obs_inconsistent_slots']}",
            f"obs-decoded rz           {L0._rz_range(res['obs_rz_min'], res['obs_rz_max'])} u  "
            f"(C rz of the observed states "
            f"{L0._rz_range(res['observed_rz_min'], res['observed_rz_max'])} u)",
        ]
    lines.append("")
    for name, ok, detail in checks:
        lines.append(f"  [{'ok' if ok else 'XX'}] {name:<34} {detail}")
    lines.append("")
    lines.append(
        "PASS — the oracle tracks and kills a walking target (lead NOT certified)" if passed else
        "FAIL — see the [XX] checks: no kill, no walk, or (--obs-only) an obs-encoding check")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cs2rl.experiment.oracle_tracker",
                                 description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200, help="rounds to play (default 200)")
    ap.add_argument("--seed", type=int, default=0, help="env + walker seed (default 0)")
    ap.add_argument("--obs-only",
                    action="store_true",
                    help="aim from the hero's observation vector instead of the C state "
                    "(oracle_statue's ObsOracleActor); the verdict adds L0's obs-encoding checks")
    args = ap.parse_args(argv)
    res = run_tracker(episodes=args.episodes, seed=args.seed, obs_only=args.obs_only)
    print(format_report(res))
    return 0 if verdict(res)[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
