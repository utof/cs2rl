#!/usr/bin/env python
"""Aim evaluation of a policy checkpoint against scripted opponents, with the miss decomposition.

WHAT
----
Plays a checkpoint as ``eval.baselines.PolicyActor`` on the hero's T row of oracle_statue's
arena harness (``run_check``: the Rung 1a env, jump off) against every opponent in
``OPPONENTS``, in every mode in ``MODES``:

  * aim ``sample``: the Gaussian draw, as training and BaselineEvaluator play it; aim ``mean``:
    cont = mu (``PolicyActor(mean_aim=True)``), the discrete heads still sampled;
  * LSTM ``carried`` for the whole round, or ``zeroed`` every ``bptt_horizon`` forwards
    (the run's config.json), counted across rounds as training counts its rollout steps:
    ``Cs2PuffeRL.evaluate`` zeroes ``lstm_h``/``lstm_c`` at the start of every rollout,
    and a rollout is ``bptt_horizon`` steps of every row.

Per opponent it also runs oracle_statue's two scripted heroes on the same seed:
``OracleActor`` (ground truth, one-tick lead) for the RL gap, and ``ObsOracleActor`` (aims
where the obs says the target was, no lead) as the one-tick-lag reference. It prints one row
per cell and writes every number, per-episode rows included, to ``--json``. Exit status is
always 0: this is an instrument, not a gate.

    env PYTHONPATH=<checkout>/src UV_NO_SYNC=1 CUDA_VISIBLE_DEVICES= .venv/bin/python \\
        -m cs2rl.experiment.aim_eval outputs/checkpoints/rung1a/s0/rung1a-s0.pt --json out.json

WHY
---
To say WHY a policy misses a moving target before any architecture change is chosen
(research C1-C4): aim NOISE (the sampled Gaussian; the mean mode removes it), LAG (the
crosshair trails a target in proportion to its angular speed), FIRE DISCIPLINE (firing while
off target), or the MEMORY HORIZON (training never carries the LSTM past ``bptt_horizon``
ticks; the zeroed mode plays it that way).

THE MISS DECOMPOSITION
----------------------
Read off the C state after each step: positions and facing as ``process_combat`` used them.
  * e: the aim error (bearing - facing, elevation - pitch) from the hero's eye to the
    opponent's torso, in units of the hit half-window ``asin(16/d)`` (d = the 2D distance; the
    test cs2_combat.h's ``shots_on_target`` applies), so ``|e_yaw| < 1`` is on target.
  * w: the line of sight's angular velocity over the tick (rad/tick): the change of
    (bearing, elevation) from the state before the step to the one after it.
  * along = e . w/|w|: + means the crosshair trails the target (lag), - that it leads.
    across = the signed perpendicular component. With pitch pinned on flat ground every
    elevation term is 0, so across is 0 and along is +-e_yaw. Below ``MIN_ANGULAR_SPEED``
    the motion has no direction and a sample has no along/across.
  * per fire: |e_yaw| median and p90, the share fired outside the window, along and across
    means, and hit/fired for fires up to and after episode tick ``bptt_horizon`` (C4);
  * per tick: ``lag_ticks``, the least-squares slope of along (rad) on |w| (rad/tick), whose
    unit is ticks. Aiming at where the target was one tick ago reads 1 (ObsOracleActor); a
    one-tick velocity lead reads about 0 (OracleActor). The fit starts at the round's first
    on-target tick: the opening turn is acquisition, not tracking, and its errors are tens of
    half-windows wide in a random direction (with them, the ObsOracleActor reference read
    0.07 +- 0.36 instead of 1 against the hold-4..16 walker, on the pre-#157 env).

TRIPWIRES in every cell: ``on_target_recount`` and ``hit_recount`` (|e_yaw| < 1 at a fire,
and a fire on which the opponent lost hp) must equal the C counters ``shots_on_target`` and
``shots_hit``; a mismatch means this module's geometry or fire attribution has drifted from
the sim's.

PITFALLS
--------
* Every tick of the arena has 2D line of sight, so nothing here is gated on visibility.
  KNOWN LIMIT: on a map with occlusion, gate on the pre-step visibility (the obs reads a target
  killed this tick as unseen, which would drop every killing shot).
* Spawns are not paired across cells: the C RNG also rolls hit locations, so two cells on
  one seed draw the same first spawn and then diverge. Compare cells as samples.
* ``HIT_HALF_WIDTH`` mirrors cs2_combat.h; tests/experiment/test_aim_eval.py pins it.
* This script never writes to ``outputs/`` and never touches training state.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cs2rl.eval.baselines import (
    EYE_CROUCH,
    EYE_STAND,
    TORSO_CROUCH,
    TORSO_STAND,
    IdleActor,
    PolicyActor,
    wrap_pi,
)
from cs2rl.eval.walker import WalkerActor
from cs2rl.experiment import oracle_statue as L0

HIT_HALF_WIDTH = 16.0                  # u, MIRROR of cs2_combat.h

# rad/tick. A walker crossing at 1 u/tick at the arena's longest 360 u spawn moves 2.8e-3.
MIN_ANGULAR_SPEED = 1e-3


def _walker(hold_min, hold_max):
    return lambda seed: WalkerActor(np.random.default_rng(seed), [L0.STATUE], hold_min, hold_max)


# name -> factory(seed) -> opponent actor for the CT row. "heldout" names the family the
# Rung 1b training mix leaves out (long straight runs). The plan's 2..8 training family also
# stands still on a quarter of its holds, which WalkerActor cannot do yet; this one never stops.
OPPONENTS = {
    "statue": lambda seed: IdleActor(),
    "walker-4-16": _walker(4, 16),
    "walker-2-8": _walker(2, 8),
    "heldout-24-48": _walker(24, 48),
}
MODES = (("sample", "carried"), ("mean", "carried"), ("sample", "zeroed"), ("mean", "zeroed"))
REFERENCE_HEROES = (("oracle", False), ("obs-oracle", True))           # (name, run_check obs_only)


class LstmZeroedEvery:
    """A ``PolicyActor`` whose LSTM state is zeroed every ``period`` forwards, as in training.

    The count runs across rounds, as training's runs across episodes: a round shorter than
    ``period`` shifts where the next one is zeroed. ``PolicyActor.reset`` is the zeroing: it
    rebuilds a zero ``lstm_h``/``lstm_c`` (and ``done``, which is 0 mid-round here).
    """

    name = "policy"

    def __init__(self, actor, period: int):
        self.actor, self.period, self.forwards = actor, int(period), 0

    def reset(self):
        self.actor.reset()

    def act(self, obs, st, vis_prev, env):
        if self.forwards and self.forwards % self.period == 0:
            self.actor.reset()
        self.forwards += 1
        return self.actor.act(obs, st, vis_prev, env)


def build_policy_hero(policy, aim: str, lstm: str, bptt_horizon: int):
    """The hero for one mode: ``aim`` in (sample, mean), ``lstm`` in (carried, zeroed)."""
    if aim not in ("sample", "mean") or lstm not in ("carried", "zeroed"):
        raise ValueError(f"unknown mode {(aim, lstm)}")
    actor = PolicyActor(policy, "cpu", mean_aim=aim == "mean")
    return actor if lstm == "carried" else LstmZeroedEvery(actor, bptt_horizon)


def aim_geometry(st_prev, st, hero, opp):
    """``(e, w, d)``: the aim error and the line of sight's angular velocity, both (yaw, pitch)
    in rad, and the 2D distance, for ``hero`` aiming at ``opp`` in the post-step state ``st``.
    """

    def los(s):
        dx, dy = s["x"][opp] - s["x"][hero], s["y"][opp] - s["y"][hero]
        eye = s["z"][hero] + (EYE_CROUCH if s["is_crouching"][hero] else EYE_STAND)
        torso = s["z"][opp] + (TORSO_CROUCH if s["is_crouching"][opp] else TORSO_STAND)
        d = math.hypot(dx, dy)
        return math.atan2(dy, dx), math.atan2(torso - eye, d), d

    bearing0, elev0, _ = los(st_prev)
    bearing, elev, d = los(st)
    e = (wrap_pi(bearing - st["facing"][hero]), elev - st["pitch"][hero])
    w = (wrap_pi(bearing - bearing0), elev - elev0)
    return e, w, d


def decompose(e, w):
    """``(along, across)`` of the error ``e`` against the angular motion ``w``, or None when
    ``|w| < MIN_ANGULAR_SPEED``. along > 0: the crosshair trails the target."""
    speed = math.hypot(w[0], w[1])
    if speed < MIN_ANGULAR_SPEED:
        return None
    ux, uy = w[0] / speed, w[1] / speed
    return e[0] * ux + e[1] * uy, ux * e[1] - uy * e[0]


class MissRecorder:
    """``run_check``'s ``on_step``: every fire's and every tick's aim error of the hero."""

    def __init__(self, hero: int = L0.HERO, opp: int = L0.STATUE):
        self.hero, self.opp = hero, opp
        self.fires = []                # (episode tick, |e_yaw|, along, across, hit), e in half-windows
        self.track = []                # (|w| rad/tick, along rad), from acquisition, with a direction
        self.acquired = False          # this round's crosshair has been on target

    def __call__(self, tick, st_prev, st):
        h, o = self.hero, self.opp
        if not (st_prev["alive"][h] and st_prev["alive"][o]):
            return
        e, w, d = aim_geometry(st_prev, st, h, o)
        split = decompose(e, w)
        half = math.asin(min(HIT_HALF_WIDTH / max(d, 1e-6), 1.0))
        self.acquired = (self.acquired and tick > 1) or abs(e[0]) < half
        if split is not None and self.acquired:
            self.track.append((math.hypot(w[0], w[1]), split[0]))
        if st["fired_this_tick"][h]:
            along, across = (math.nan, math.nan) if split is None else split
            self.fires.append((tick, abs(e[0]) / half, along / half, across / half,
                               bool(st["hp"][o] < st_prev["hp"][o])))

    def summary(self, horizon: int) -> dict:
        f = np.array(self.fires, dtype=float).reshape(-1, 5)
        tick, abs_e, along, across, hit = f.T
        moving = ~np.isnan(along)

        def rate(sel):
            return float(hit[sel].mean()) if sel.any() else None

        out: dict[str, float | int | None] = {
            "fires": len(f),
            "abs_err_median": float(np.median(abs_e)) if len(f) else None,
            "abs_err_p90": float(np.percentile(abs_e, 90)) if len(f) else None,
            "off_window_frac": float((abs_e >= 1).mean()) if len(f) else None,
            "moving_fires": int(moving.sum()),
            "along_mean": float(along[moving].mean()) if moving.any() else None,
            "across_mean": float(across[moving].mean()) if moving.any() else None,
            "hit_per_fired_early": rate(tick <= horizon),
            "hit_per_fired_late": rate(tick > horizon),
            "late_fires": int((tick > horizon).sum()),
            "on_target_recount": int((abs_e < 1).sum()),
            "hit_recount": int(hit.sum()),
            "track_ticks": len(self.track),
            "lag_ticks": None,
            "lag_ticks_se": None,
        }
        if len(self.track) >= 10:
            x, y = np.array(self.track).T
            if np.ptp(x) > 0:
                (slope, _), cov = np.polyfit(x, y, 1, cov=True)
                out["lag_ticks"], out["lag_ticks_se"] = float(slope), float(math.sqrt(cov[0, 0]))
        return out


def _cell(opponent, hero, aim, lstm, res, rec, horizon) -> dict:
    """One output row: the cell's labels, run_check's totals and the recorder's summary."""
    row = dict(opponent=opponent, hero=hero, aim=aim, lstm=lstm)
    row.update((k, res[k])
               for k in ("episodes", "kills", "kill_rate", "ttk_median", "ttk_p90", "shots_fired",
                         "shots_hit", "shots_on_target", "opp_moving_frac", "per_episode"))
    row["hit_per_fired"] = res["shots_hit"] / max(res["shots_fired"], 1)
    row.update(rec.summary(horizon))
    return row


def run_eval(policy,
             bptt_horizon: int,
             episodes: int = 100,
             seed: int = 0,
             opponents=None,
             modes=MODES) -> dict:
    """Every opponent x (reference heroes + every mode) on ``seed``; returns the JSON dict."""
    t0 = time.perf_counter()
    cells, refs = [], []
    for name, make in (OPPONENTS if opponents is None else opponents).items():
        for hero, obs_only in REFERENCE_HEROES:
            rec = MissRecorder()
            res = L0.run_check(episodes, seed, obs_only=obs_only, opponent=make(seed), on_step=rec)
            refs.append(_cell(name, hero, None, None, res, rec, bptt_horizon))
        oracle = refs[-len(REFERENCE_HEROES)]
        for aim, lstm in modes:
            hero = build_policy_hero(policy, aim, lstm, bptt_horizon)
            rec = MissRecorder()
            # Each cell draws from its own torch stream, so cells reproduce in any order.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                res = L0.run_check(episodes, seed, opponent=make(seed), hero=hero, on_step=rec)
            row = _cell(name, "policy", aim, lstm, res, rec, bptt_horizon)
            row["gap_to_oracle"] = {
                "kill_rate": oracle["kill_rate"] - row["kill_rate"],
                "ttk_median": row["ttk_median"] - oracle["ttk_median"],
                "hit_per_fired": oracle["hit_per_fired"] - row["hit_per_fired"],
            }
            cells.append(row)
    return {
        "episodes": episodes,
        "seed": seed,
        "bptt_horizon": bptt_horizon,
        "references": refs,
        "cells": cells,
        "wall_s": time.perf_counter() - t0
    }


def load_checkpoint(path) -> tuple:
    """``(policy, config)``: the checkpoint, built with its run's config.json (beside it).

    ``aim_log_std_max`` and ``pin_pitch`` are run properties, not checkpoint state
    (``load_policy_from_checkpoint``), so a checkpoint without its config.json is refused.
    """
    from cs2rl.policy import load_policy_from_checkpoint
    path = Path(path)
    cfg = json.loads((path.parent / "config.json").read_text())
    policy = load_policy_from_checkpoint(path,
                                         "cpu",
                                         aim_log_std_max=cfg["aim_log_std_max"],
                                         pin_pitch=bool(cfg["pin_pitch"]))
    return policy, cfg


def env_mismatches(cfg) -> list[str]:
    """The run's env knobs that differ from the arena preset every cell plays."""
    preset = {
        "n_active_per_team": L0.N_ACTIVE_PER_TEAM,
        "pin_pitch": L0.PIN_PITCH,
        "crouch_enabled": L0.CROUCH_ENABLED,
        "jump_enabled": L0.JUMP_ENABLED,
        "round_time_ticks": L0.ROUND_TIME
    }
    return [f"{k}: run {cfg.get(k)} vs harness {v}" for k, v in preset.items() if cfg.get(k) != v]


def _fmt(v, spec):
    return "-" if v is None else format(v, spec)


def format_table(out: dict) -> str:
    """One row per reference hero and per policy cell."""
    lines = [
        f"aim_eval: {out['episodes']} episodes per cell, seed {out['seed']}, "
        f"bptt_horizon {out['bptt_horizon']}; e in hit half-windows asin(16/d)",
        f"{'opponent':<14} {'hero':<10} {'aim':<6} {'lstm':<7} kills  ttk50 ttk90 hit/f "
        f" |e|50 |e|90 off   along across lag(t)  se   ticks  hit/f<=H hit/f>H gap_k  gap_ttk"
    ]
    for r in out["references"] + out["cells"]:
        gap = r.get("gap_to_oracle", {})
        lines.append(
            f"{r['opponent']:<14} {r['hero']:<10} {r['aim'] or '-':<6} {r['lstm'] or '-':<7} "
            f"{r['kills']:>3}/{r['episodes']:<3} {r['ttk_median']:5.1f} {r['ttk_p90']:5.1f} "
            f"{r['hit_per_fired']:.3f}  {_fmt(r['abs_err_median'], '5.2f')} "
            f"{_fmt(r['abs_err_p90'], '5.2f')} {_fmt(r['off_window_frac'], '.2f')} "
            f"{_fmt(r['along_mean'], '+6.2f')} {_fmt(r['across_mean'], '+5.2f')} "
            f"{_fmt(r['lag_ticks'], '+6.2f')} {_fmt(r['lag_ticks_se'], '5.2f')} {r['track_ticks']:>6} "
            f"{_fmt(r['hit_per_fired_early'], '8.3f')} {_fmt(r['hit_per_fired_late'], '7.3f')} "
            f"{_fmt(gap.get('kill_rate'), '+.2f'):>6} {_fmt(gap.get('ttk_median'), '+6.1f')}")
        if (r["on_target_recount"], r["hit_recount"]) != (r["shots_on_target"], r["shots_hit"]):
            lines.append(f"  <-- WARNING: recount on_target {r['on_target_recount']} hit "
                         f"{r['hit_recount']} vs C {r['shots_on_target']} {r['shots_hit']}")
    lines.append(f"wall {out['wall_s']:.1f} s")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cs2rl.experiment.aim_eval",
                                 description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="a policy .pt; its run's config.json must sit beside it")
    ap.add_argument("--episodes", type=int, default=100, help="rounds per cell (default 100)")
    ap.add_argument("--seed", type=int, default=0, help="env, walker and torch seed (default 0)")
    ap.add_argument("--json", type=Path, help="write the full result (per-episode rows too) here")
    args = ap.parse_args(argv)
    policy, cfg = load_checkpoint(args.checkpoint)
    out = run_eval(policy, int(cfg["bptt_horizon"]), episodes=args.episodes, seed=args.seed)
    out.update(checkpoint=str(args.checkpoint),
               env_mismatches=env_mismatches(cfg),
               aim_log_std=getattr(policy, "aim_log_std", torch.zeros(0)).detach().tolist())
    for m in out["env_mismatches"]:
        print(f"WARNING: the run trained a different env, {m}")
    print(format_table(out))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
