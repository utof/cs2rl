#!/usr/bin/env python
"""Oracle-vs-statue solvability check — the precondition for reading Rung 1a §3.

WHAT
----
Drives the EXACT Rung 1a environment (ARENA_DUEL_V1, ``n_active_per_team=1``,
``round_time=160``, ``pin_pitch=1``, ``crouch_enabled=0``) with two scripted
actors and no learning anywhere in the loop:

  * hero   — agent 0 (T): ``eval_baselines.OracleActor``. Per tick it reads the
             enemy's position from the live C state, turns the continuous Δyaw
             toward that bearing (clamped to ±``sd->max_turn_speed``, exactly as
             ``env_step`` will clamp it) and pulls the trigger whenever the enemy
             is visible, in range, and the weapon is off cooldown.
  * statue — agent 5 (CT): ``eval_baselines.IdleActor``. Every discrete head at
             bin 0 and Δyaw = Δpitch = 0, so it never moves, turns, or fires.

It reports kill rate, time-to-kill quantiles and the R0-A shot counters, then
prints one PASS/FAIL line. Exit status 0 = PASS, 1 = FAIL.

    UV_NO_SYNC=1 uv run python scripts/oracle_statue_check.py
    UV_NO_SYNC=1 uv run python scripts/oracle_statue_check.py --statue-z 24

WHY
---
A Rung 1a §3 verdict is a statement about a LEARNED policy, and it is only
meaningful if the environment is solvable at all. If a scripted actor with
ground-truth positions, perfect aim, no exploration cost and a target that
stands still cannot get a kill, then a §3 FAIL says nothing about PPO, the
reward shaping or the aim head — it says the sim is broken, and every learning
number measured on it is void. This script is the instrument that separates
those two readings.

It is a PRECONDITION, not a baseline. Passing it licenses interpreting a §3
FAIL as a learning result; it does NOT predict that a policy will pass, and a
policy losing to a statue is not evidence against the sim once this passes.

THE ELEVATED-STATUE VARIANT (``--statue-z``)
--------------------------------------------
``--statue-z OFF`` holds the statue OFF world units above its spawn surface for
every tick of the round. The hold has to be re-applied before every step:
``process_movement`` snaps a grounded agent back to the area surface, and an
airborne one is pulled down by gravity, so a one-shot write would decay within
a couple of ticks (see ``_hold_statue_above_ground``).

What ``--statue-z 24`` tests, exactly. The shot the v1c ellipsoid (gh #150) was
introduced for is one whose vertical offset from the shooter's eye is ±24 u —
the offset a crouched target presents (``TORSO_OFFSET_CROUCH`` 24 against
``EYE_HEIGHT_STAND`` 48). Raising a STANDING statue by 24 u reproduces that same
|rz| against the combat ray, so it exercises the ellipsoid's vertical term
in-env, end to end, through the real action path.

What it does NOT test: the ``HIT_HALF_HEIGHT_CROUCH = 27`` branch. The statue is
standing, so the gate uses ``HIT_HALF_HEIGHT_STAND = 36``; ``crouch_enabled=0``
masks the crouch head, so a genuinely crouched target is unreachable through the
Rung 1a action space and this variant deliberately does not fake one. Read a
pass as "the ellipsoid's vertical extent works in the live sim at |rz| ≈ 22",
never as "crouched targets are killable".

One tick of leapfrog gravity runs between the hold and ``process_combat``, so
the |rz| the ray actually sees is ``OFF − 1.5625`` (g = 800, dt = 1/16). The
summary prints the MEASURED offset rather than assuming it.

PITFALLS
--------
* ``auto_reset=False`` is mandatory: with auto-reset the C ``episode_stats`` are
  cleared on the terminal tick and every counter below reads 0.
* ``vis_prev`` must be threaded tick to tick. ``OracleActor`` only fires at an
  enemy that was visible in the PREVIOUS tick's observation; feeding it ``None``
  every tick silently degrades it into a walking, non-firing actor that times
  out — which looks exactly like a broken sim. ``unmatched_vis_slots`` in the
  summary is the tripwire for that thread going wrong (it must be 0).
* Importing ``eval_baselines`` pulls in torch (``PolicyActor`` needs it). Nothing
  here uses it, but the import cost is real; that is the price of reusing the
  evaluator's actors instead of writing a second oracle that can drift from it.
* This script never writes to ``outputs/`` and never touches training state.
"""
# src/ has to be on sys.path before the repo imports near the bottom of this
# block can resolve, so they cannot sit at the top of the file. E402 is
# suppressed file-wide rather than per-line because a per-line suppression puts
# ruff's isort and yapf's trailing-comment aligner in a permanent fight over
# which column the comment belongs in.
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from c_env.cs2_env import TEAM_SIZE
from eval_baselines import BaselineEvaluator, IdleActor, OracleActor, vis_from_obs

# Rung 1a env preset — these MUST mirror the smoke run's env knobs. Changing one
# here without changing the smoke makes the precondition test a different env.
ROUND_TIME = 160
N_ACTIVE_PER_TEAM = 1
PIN_PITCH = 1
CROUCH_ENABLED = 0

# At n_active_per_team=1 env_reset parks every slot with (i % TEAM_SIZE) >= 1,
# so exactly two agents spawn: row 0 (T) and row TEAM_SIZE (CT).
HERO = 0
STATUE = TEAM_SIZE

# PASS thresholds (task brief). Deliberately NOT CLI-configurable: this is a
# gate, and a gate whose threshold is an argument is not a gate.
PASS_MIN_KILL_RATE = 0.90
PASS_MAX_MEDIAN_TTK = 120.0

# cs2_movement.h: leapfrog applies half a gravity step before the position
# update, so a held-still airborne agent loses this much z before combat runs.
GRAVITY_SAG_PER_TICK = 0.5 * 800.0 * (1.0 / 16.0)**2


def build_env(seed: int, round_time: int = ROUND_TIME):
    """The Rung 1a arena env, built for instrument use (auto_reset off).

    WHY the knobs are hard-coded rather than passed through: the whole point of
    the check is that it runs the env the smoke runs. A caller that wants a
    different env is asking a different question and should say so in code.
    """
    from c_env.cs2_env import make_env
    from map import make_arena_duel_map
    return make_env(map_data=make_arena_duel_map(),
                    n_active_per_team=N_ACTIVE_PER_TEAM,
                    pin_pitch=PIN_PITCH,
                    crouch_enabled=CROUCH_ENABLED,
                    round_time=round_time,
                    auto_reset=False,
                    seed=seed)


def _hold_statue_above_ground(env, ground_z: float, offset: float) -> None:
    """Re-place the statue ``offset`` units above its spawn surface, this tick.

    Must be called BEFORE every ``env.step``. Two sim behaviours make a one-shot
    write useless (cs2_movement.h):
      * a grounded agent has ``a->z`` snapped to ``_surface_z`` every tick, which
        would undo the lift immediately — hence ``is_airborne = 1``;
      * an airborne agent accumulates gravity, so without ``vz = 0`` and a fresh
        ``z`` each tick it would arc back down within ~10 ticks and the
        experiment would silently become the ground experiment.

    PITFALL: the value the combat ray sees is ``offset − GRAVITY_SAG_PER_TICK``,
    not ``offset`` — process_movement runs between this write and
    process_combat. Callers measure the realised offset instead of assuming it.
    """
    a = env._c_env.game.agents[STATUE]
    a.z = ground_z + offset
    a.vz = 0.0
    a.is_airborne = 1


def _episode(env, ev, oracle, statue, statue_z, round_time):
    """One round. Returns (ttk or None, realised |rz| samples, unmatched slots).

    ``ttk`` is the tick index on which the statue's ``alive`` flag flipped to 0,
    counting from 1 (``g->tick`` after the k-th step is exactly k), or None if it
    survived the round. Per-episode shot counters are left in the env's
    ``episode_stats`` for the caller to read before the next reset clears them.
    """
    obs, _ = env.reset()
    oracle.reset()
    statue.reset()
    ground_z = float(env._c_env.game.agents[STATUE].z)
    st = ev.reader.read().snapshot()
    vis_prev = None
    ttk = None
    rz_samples = []
    unmatched_total = 0
    hero_rows = slice(0, TEAM_SIZE)

    # round_time + 1 for the same reason BaselineEvaluator uses it: the timeout
    # terminal lands ON the last tick, and a loop that falls through without a
    # terminal means the env's round timer is broken, which must not read as a
    # quiet "no kill".
    for tick in range(1, int(round_time) + 2):
        if statue_z:
            _hold_statue_above_ground(env, ground_z, statue_z)

        act, cont = statue.act(obs, st, vis_prev, env)                 # all-zero rows
        a_hero, c_hero = oracle.act(obs, st, vis_prev, env)
        act[hero_rows], cont[hero_rows] = a_hero[hero_rows], c_hero[hero_rows]

        obs, _rew, term, trunc, _info = env.step(act, cont)
        st = ev.reader.read().snapshot()
        vis_prev, unmatched = vis_from_obs(obs, st, ev.map_diag)
        unmatched_total += int(unmatched)

        if st["alive"][HERO] and st["alive"][STATUE]:
            # Both standing, so rz reduces to the plain z difference: this is
            # exactly the `tgt_rz` cs2_combat.h computed on this tick.
            rz_samples.append(float(st["z"][STATUE] - st["z"][HERO]))
        if ttk is None and not st["alive"][STATUE]:
            ttk = tick
        if term.any() or trunc.any():
            break
    else:
        raise RuntimeError(f"episode did not terminate within round_time+1={round_time + 1} "
                           "ticks — the env's timeout terminal is broken")
    return ttk, rz_samples, unmatched_total


def run_check(episodes: int = 200,
              seed: int = 0,
              statue_z: float = 0.0,
              round_time: int = ROUND_TIME) -> dict:
    """Play ``episodes`` oracle-vs-statue rounds; return the summary dict.

    The env, the actors and the derived sim constants are built once and reused
    across episodes (the C RNG carries over, so consecutive episodes draw
    different spawn rows — that variety is the point, the arena's four spawn rows
    per side are what make the opening turn non-constant).

    WHY ``BaselineEvaluator`` is constructed here without ever calling
    ``evaluate``/``run_pair``: it is the single place that derives
    ``max_turn_speed``, ``laser_range``, ``map_diag`` and the ``NavHelper`` /
    ``StateReader`` from a live env. Re-deriving them here would create a second
    copy of the ``map_diag`` formula that can silently drift from the evaluator's.
    We do not use its episode driver because it has no time-to-kill and splits
    episodes half-and-half across sides; this instrument keeps the hero on T.
    """
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}")
    env = build_env(seed, round_time)
    try:
        ev = BaselineEvaluator(env, episodes=2, seed=seed)             # constants only, see docstring
        oracle = OracleActor(np.random.default_rng(seed), ev.max_turn_speed, ev.laser_range, ev.nav)
        statue = IdleActor()

        ttks, rz_min, rz_max, unmatched = [], None, None, 0
        totals = dict.fromkeys(("shots_fired", "shots_with_enemy_in_los", "shots_facing_enemy",
                                "shots_on_target", "shots_hit", "shots_stance_blocked"), 0)
        kills = 0
        for _ in range(episodes):
            ttk, rz_samples, unmatched_ep = _episode(env, ev, oracle, statue, statue_z, round_time)
            unmatched += unmatched_ep
            if rz_samples:
                rz_min = min(rz_samples) if rz_min is None else min(rz_min, min(rz_samples))
                rz_max = max(rz_samples) if rz_max is None else max(rz_max, max(rz_samples))
            es = env._c_env.episode_stats
            for k in totals:
                totals[k] += int(getattr(es, k))
            if ttk is not None:
                kills += 1
            ttks.append(ttk)
    finally:
        env.close()

    # Censored TTK: an episode with no kill enters the quantiles as round_time+1,
    # never as a dropped sample. Dropping them would let a run that kills in 10 %
    # of rounds report a beautiful median — the exact failure this gate exists to
    # catch. `ttk_median_killed` is reported alongside for diagnosis only.
    censored = np.array([round_time + 1 if t is None else t for t in ttks], dtype=float)
    killed = [t for t in ttks if t is not None]
    return {
        "episodes": episodes,
        "seed": seed,
        "statue_z": statue_z,
        "round_time": round_time,
        "kills": kills,
        "kill_rate": kills / episodes,
        "ttk_median": float(np.median(censored)),
        "ttk_p90": float(np.percentile(censored, 90)),
        "ttk_min": int(min(killed)) if killed else None,
        "ttk_median_killed": float(np.median(killed)) if killed else None,
        "ttk_censored": episodes - kills,
        "rz_min": rz_min,
        "rz_max": rz_max,
        "unmatched_vis_slots": unmatched,
        **totals,
    }


def verdict(res: dict) -> tuple[bool, list[tuple[str, bool, str]]]:
    """(passed, [(name, ok, detail), ...]) — the gate, evaluated on a summary.

    Three checks. The first two are the brief's gate. The third
    (``shots_stance_blocked == 0``) is trivially true on flat ground and is the
    entire content of the ``--statue-z`` variant: it says the vertical offset the
    shots were taken against stayed inside the target's vertical semi-axis, so a
    kill there is the v1c ellipsoid working and not the offset quietly being
    ignored.
    """
    kill_ok = res["kill_rate"] >= PASS_MIN_KILL_RATE
    ttk_ok = res["ttk_median"] < PASS_MAX_MEDIAN_TTK
    checks = [
        (f"kill_rate >= {PASS_MIN_KILL_RATE:.2f}", kill_ok, f"{res['kill_rate']:.3f}"),
        (f"median_ttk < {PASS_MAX_MEDIAN_TTK:.0f}", ttk_ok, f"{res['ttk_median']:.1f}"),
        ("shots_stance_blocked == 0", res["shots_stance_blocked"] == 0,
         str(res["shots_stance_blocked"])),
    ]
    return all(ok for _, ok, _ in checks), checks


def format_summary(res: dict) -> str:
    """Human-readable report — every number the verdict rests on, plus context."""
    passed, checks = verdict(res)
    rz = "n/a"
    if res["rz_min"] is not None:
        rz = f"{res['rz_min']:+.2f} .. {res['rz_max']:+.2f}"
    fired = max(res["shots_fired"], 1)
    lines = [
        "── oracle vs statue — arena-duel, n_active=1, pin_pitch=1, crouch=0 ──",
        f"episodes                 {res['episodes']}  (seed {res['seed']}, "
        f"round_time {res['round_time']})",
        f"statue z offset          {res['statue_z']:+.1f} u requested; "
        f"realised rz at combat {rz} u",
        f"kill rate                {res['kill_rate']:.3f}  ({res['kills']}/{res['episodes']})",
        f"time-to-kill (ticks)     median {res['ttk_median']:.1f}   p90 {res['ttk_p90']:.1f}   "
        f"min {res['ttk_min']}   censored {res['ttk_censored']}",
        f"  median over kills only {res['ttk_median_killed']}",
        f"shots_fired              {res['shots_fired']}",
        f"shots_with_enemy_in_los  {res['shots_with_enemy_in_los']}  "
        f"({res['shots_with_enemy_in_los'] / fired:.3f} of fired)",
        f"shots_on_target          {res['shots_on_target']}  "
        f"({res['shots_on_target'] / fired:.3f} of fired)",
        f"shots_hit                {res['shots_hit']}  ({res['shots_hit'] / fired:.3f} of fired)",
        f"shots_stance_blocked     {res['shots_stance_blocked']}",
        f"unmatched vis slots      {res['unmatched_vis_slots']}"
        f"{'   <-- WARNING: visibility recovery is broken, oracle targeting is suspect' if res['unmatched_vis_slots'] else ''}",
        "",
    ]
    for name, ok, detail in checks:
        lines.append(f"  [{'ok' if ok else 'XX'}] {name:<28} {detail}")
    lines.append("")
    lines.append("PASS — environment is solvable; a Rung 1a §3 FAIL is a learning result" if passed
                 else "FAIL — a scripted perfect-aim actor cannot kill a stationary target; "
                 "Rung 1a §3 verdicts on this env are void")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes",
                    type=int,
                    default=200,
                    help="rounds to play (default 200; the gate is a rate, so fewer is noisier)")
    ap.add_argument("--seed", type=int, default=0, help="env + actor seed (default 0)")
    ap.add_argument("--statue-z",
                    type=float,
                    default=0.0,
                    help="hold the statue this many world units above its spawn surface every "
                    "tick (default 0 = on the ground). 24 reproduces the |rz| of a crouched "
                    "target against a STANDING hitbox — see the module docstring for exactly "
                    "what that does and does not prove.")
    ap.add_argument("--round-time",
                    type=int,
                    default=ROUND_TIME,
                    help=f"ticks per round (default {ROUND_TIME}, the Rung 1a value)")
    args = ap.parse_args(argv)

    res = run_check(episodes=args.episodes,
                    seed=args.seed,
                    statue_z=args.statue_z,
                    round_time=args.round_time)
    print(format_summary(res))
    return 0 if verdict(res)[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
