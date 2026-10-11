#!/usr/bin/env python
"""Save rounds of a rung's arena harness as a replay to watch: one command to save, one to view.

    python -m cs2rl.experiment.replay CHECKPOINT.pt --vs heldout-24-48 --rounds 5
    python -m cs2rl.experiment.replay oracle --vs walker-4-16
    then run the "view:" line it prints, from any directory

WHAT
----
Plays ``--rounds`` rounds of oracle_statue's arena harness (``run_check``: the Rung 1a env,
jump off, the movement-inaccuracy cone as ``--accuracy`` says) with the hero on the T row
and an opponent from ``aim_eval.OPPONENTS`` on the CT row, and records every tick with
``cs2rl.viz.render.ReplayRecorder``. The hero is a
checkpoint (``aim_eval.load_checkpoint``: its run's config.json must sit beside it) played
like aim_eval's policy cells (``--aim``, ``--lstm``), or ``oracle`` / ``obs-oracle``,
oracle_statue's two scripted heroes. It prints one row per round, writes the replay (one
Rerun recording per round, in one .rrd) and a per-tick text dump beside it (``.txt``), and
prints the command that opens the replay. Exit status 0 whenever the rounds complete.

The text dump is the headless check: per round its outcome, then per tick the hero's and
the opponent's positions, the hero's aim error to the opponent in degrees against the hit
half-window ``asin(16/d)`` (``aim_eval.aim_geometry``), the fires, the hits and the
opponent's hp. An agent can read it, or name a round and tick for the owner to watch.

FIDELITY
--------
Round k is episode k of the ``run_check`` call aim_eval makes for the same cell, seed and
cone setting: the same opponent factory, the same hero, the same torch seeding for a policy
hero, and episodes do not depend on how many follow. Recording only reads the snapshots run_check
already takes, and draws no random numbers. tests/experiment/test_replay.py pins both (the
aim_eval cell only where the Rung 1a checkpoint is present).

PITFALLS
--------
* A checkpoint trained on another env prints the same WARNING lines aim_eval prints
  (``aim_eval.env_mismatches``); the replay still plays the harness's env.
* The cone (``EnvConfig.accuracy``): without ``--accuracy`` a checkpoint plays its run's own
  config.json setting (``aim_eval.trained_accuracy``: a run from before the key trained
  without it), as aim_eval does, and a scripted hero the EnvConfig default (on). The .txt
  header names the setting the rounds ran.
* Writes under ``outputs/recordings/`` (gitignored) unless ``--out`` says otherwise.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from cs2rl.experiment import aim_eval
from cs2rl.experiment import oracle_statue as L0
from cs2rl.spec.paths import RECORDINGS_DIR

SCRIPTED_HEROES = dict(aim_eval.REFERENCE_HEROES)      # name -> run_check obs_only


class TickTable:
    """``run_check``'s ``on_step``: the per-tick text dump of the hero against the opponent."""

    HEADER = ("tick   hero_x   hero_y  facing    opp_x    opp_y    dist  aim_err  window  fire"
              "  hit  opp_hp")

    def __init__(self, hero: int = L0.HERO, opp: int = L0.STATUE):
        self.hero, self.opp = hero, opp
        self.rounds: list[list[str]] = []

    def __call__(self, tick, st_prev, st):
        if tick == 1:
            self.rounds.append([self.HEADER, self._row(0, st_prev, st_prev, False, False)])
        h, o = self.hero, self.opp
        hit = bool(st["hp"][o] < st_prev["hp"][o])
        self.rounds[-1].append(self._row(tick, st_prev, st, bool(st["fired_this_tick"][h]), hit))

    def _row(self, tick, st_prev, st, fired, hit) -> str:
        h, o = self.hero, self.opp
        e, _w, d = aim_eval.aim_geometry(st_prev, st, h, o)
        # A mirror of aim_eval.MissRecorder's half-window (P2 would share it with aim_eval).
        half = math.degrees(math.asin(min(aim_eval.HIT_HALF_WIDTH / max(d, 1e-6), 1.0)))
        return (
            f"{tick:>4} {st['x'][h]:8.1f} {st['y'][h]:8.1f} {math.degrees(st['facing'][h]):7.1f}"
            f" {st['x'][o]:8.1f} {st['y'][o]:8.1f} {d:7.1f} {math.degrees(e[0]):+8.2f}"
            f" {half:7.2f}  {'F' if fired else '.':>4} {'H' if hit else '.':>4} {int(st['hp'][o]):>7}"
        )


def default_path(hero: str, opponent: str, seed: int) -> Path:
    """``outputs/recordings/<hero label>-vs-<opponent>-seed<seed>.rrd``.

    A checkpoint's label is its file stem after the nearest ancestor directory not named
    "checkpoints" (the run's directory in both the local and the Modal layouts).
    """
    if hero in SCRIPTED_HEROES:
        label = hero
    else:
        p = Path(hero)
        run = next((a.name for a in p.parents if a.name != "checkpoints"), "")
        label = f"{run}-{p.stem}" if run else p.stem
    return RECORDINGS_DIR / f"{label}-vs-{opponent}-seed{seed}.rrd"


def play(hero: str,
         opponent: str,
         rounds: int,
         seed: int,
         on_step,
         aim: str = "sample",
         lstm: str = "carried",
         accuracy: bool | None = None) -> tuple[dict, list[str]]:
    """``(run_check summary, warnings)``: ``rounds`` rounds of ``hero`` against ``opponent``.

    The run_check call aim_eval.run_eval makes for this cell: a scripted hero runs as a
    reference row does, a checkpoint as a policy cell does, inside its own torch stream
    seeded with ``seed``. ``accuracy`` is run_check's cone switch; None plays a checkpoint's
    own config.json setting and, for a scripted hero, the EnvConfig default.
    """
    make_opp = aim_eval.OPPONENTS[opponent]
    if hero in SCRIPTED_HEROES:
        res = L0.run_check(rounds,
                           seed,
                           obs_only=SCRIPTED_HEROES[hero],
                           opponent=make_opp(seed),
                           on_step=on_step,
                           accuracy=accuracy)
        return res, []
    import torch
    policy, cfg = aim_eval.load_checkpoint(hero)
    actor = aim_eval.build_policy_hero(policy, aim, lstm, int(cfg["bptt_horizon"]))
    if accuracy is None:
        accuracy = aim_eval.trained_accuracy(cfg)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        res = L0.run_check(rounds,
                           seed,
                           opponent=make_opp(seed),
                           hero=actor,
                           on_step=on_step,
                           accuracy=accuracy)
    return res, aim_eval.env_mismatches(cfg, accuracy)


def save(hero: str,
         opponent: str,
         rounds: int,
         seed: int,
         out: Path,
         aim: str = "sample",
         lstm: str = "carried",
         command: str = "",
         accuracy: bool | None = None) -> dict:
    """Play and record; write ``out`` (.rrd) and ``out`` with suffix .txt. Returns the summary
    with ``warnings``, ``replay`` and ``text`` added."""
    from cs2rl.env.map import make_arena_duel_map
    from cs2rl.viz.render import ReplayRecorder

    table = TickTable()
    with ReplayRecorder(out, make_arena_duel_map(), hero=L0.HERO, opp=L0.STATUE) as rec:

        def on_step(tick, st_prev, st):
            rec(tick, st_prev, st)
            table(tick, st_prev, st)

        res, warnings = play(hero, opponent, rounds, seed, on_step, aim, lstm, accuracy)
    hero_desc = hero if hero in SCRIPTED_HEROES else f"{hero} (aim {aim}, lstm {lstm})"
    lines = [
        f"# {command}" if command else "#",
        f"# hero T{L0.HERO} = {hero_desc}; opponent CT{L0.STATUE} = {opponent}; seed {seed}"
        f"; movement-inaccuracy cone {'on' if res['accuracy'] else 'off'}",
        "# aim_err: hero's yaw error to the opponent (bearing - facing), deg; window: the hit"
        " half-window asin(16/d), deg; F fired, H the opponent lost hp"
    ]
    for k, (ep, rows) in enumerate(zip(res["per_episode"], table.rounds, strict=True), start=1):
        lines += ["", _round_line(k, ep), *rows]
    text = out.with_suffix(".txt")
    text.write_text("\n".join(lines) + "\n")
    return {**res, "warnings": warnings, "replay": out, "text": text}


def _round_line(k: int, ep: dict) -> str:
    ttk = "no kill" if ep["ttk"] is None else f"kill at tick {ep['ttk']}"
    return (f"round {k}: {ttk}, {ep['shots_fired']} shots, {ep['shots_hit']} hits; spawn distance "
            f"{ep['spawn_dist']:.0f} u, opening yaw error {math.degrees(ep['yaw_err']):+.0f} deg")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cs2rl.experiment.replay",
                                 description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hero",
                    help="a policy .pt (its run's config.json beside it), or "
                    f"{' / '.join(SCRIPTED_HEROES)}")
    ap.add_argument("--vs",
                    default="walker-4-16",
                    choices=list(aim_eval.OPPONENTS),
                    help="the opponent (default walker-4-16)")
    ap.add_argument("--rounds", type=int, default=5, help="rounds to record (default 5)")
    ap.add_argument("--seed", type=int, default=0, help="env, walker and torch seed (default 0)")
    ap.add_argument("--aim",
                    choices=("sample", "mean"),
                    default="sample",
                    help="a policy hero's aim: the sampled Gaussian or its mean (default sample)")
    ap.add_argument("--lstm",
                    choices=("carried", "zeroed"),
                    default="carried",
                    help="a policy hero's LSTM state: carried all round, or zeroed every "
                    "bptt_horizon ticks as in training (default carried)")
    ap.add_argument("--accuracy",
                    type=int,
                    choices=(0, 1),
                    default=None,
                    help="the movement-inaccuracy cone (EnvConfig.accuracy): 1 on, 0 off. "
                    "Default: a checkpoint's run's own config.json setting (a run without the "
                    "key trained without the cone), as aim_eval plays it; EnvConfig.accuracy "
                    "for a scripted hero")
    ap.add_argument("--out",
                    type=Path,
                    help="the .rrd to write (default under outputs/recordings/)")
    args = ap.parse_args(argv)
    if args.rounds < 1:
        ap.error(f"--rounds {args.rounds}: record at least 1 round")
    from cs2rl.viz.render import view_command
    out = args.out or default_path(args.hero, args.vs, args.seed)
    t0 = time.perf_counter()
    res = save(args.hero,
               args.vs,
               args.rounds,
               args.seed,
               out,
               args.aim,
               args.lstm,
               command="python -m cs2rl.experiment.replay " +
               " ".join(sys.argv[1:] if argv is None else argv),
               accuracy=None if args.accuracy is None else bool(args.accuracy))
    for w in res["warnings"]:
        print(f"WARNING: the run trained a different env, {w}")
    for k, ep in enumerate(res["per_episode"], start=1):
        print(_round_line(k, ep))
    print(f"saved {res['replay']} and {res['text'].name} in {time.perf_counter() - t0:.1f} s")
    print(f"view: {view_command(res['replay'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
