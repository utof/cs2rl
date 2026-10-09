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
  * per tick: ``lag_ticks``, the median over tracking ticks of along / |w| (rad over rad/tick),
    i.e. how many ticks of the line of sight's motion the crosshair trails by; its standard
    error comes from resampling whole episodes. Aiming at where the target was one tick ago
    reads exactly 1 (ObsOracleActor); a one-tick velocity lead reads near 0 (OracleActor: 0.08
    to 0.16 against the three walkers on 2026-10-09).
    ``lag_by_speed`` gives the same ratio per quintile of |w|: flat means the lag is
    proportional to the angular speed. Tracking starts at the round's first on-target tick:
    the opening turn is acquisition, and its errors are tens of half-windows wide.
    WHY a median ratio, not a least-squares slope: the slope is carried by the few ticks with
    the largest |w|. For the Rung 1a checkpoint against walker-4-16 (sampled aim, 100 rounds)
    it read 1.65 while every |w| quintile's median ratio read 2.9 to 3.3.

TRIPWIRES in every cell: ``on_target_recount`` and ``hit_recount`` (|e_yaw| < 1 at a fire,
and a fire on which the opponent lost hp) must equal the C counters ``shots_on_target`` and
``shots_hit``; a mismatch means this module's geometry or fire attribution has drifted from
the sim's.

PITFALLS
--------
* Every tick of the arena has 2D line of sight, so nothing here is gated on visibility.
  KNOWN LIMIT: on a map with occlusion, gate on the pre-step visibility (the obs reads a target
  killed this tick as unseen, which would drop every killing shot).
* Spawns are not paired across cells: the env's one RNG also rolls hit locations
  (cs2_combat.h), so two cells on one seed share their first few spawns and then diverge (2 to
  9 shared with the oracle's cell on 2026-10-09). Compare cells as samples.
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
    ACTION_DIM,
    AIM_DIM,
    EYE_CROUCH,
    EYE_STAND,
    TORSO_CROUCH,
    TORSO_STAND,
    IdleActor,
    PolicyActor,
    wrap_pi,
)
from cs2rl.eval.walker import H_MOVE, HELD_OUT, TRAIN_MIX, RandomWalker, WalkerActor, WalkerParams
from cs2rl.experiment import oracle_statue as L0

HIT_HALF_WIDTH = 16.0                  # u, MIRROR of cs2_combat.h

# rad/tick. A walker crossing at 1 u/tick at the arena's longest 360 u spawn moves 2.8e-3.
MIN_ANGULAR_SPEED = 1e-3
MIN_TRACK_TICKS = 10                   # fewer tracking ticks than this: no lag reading
BOOTSTRAP_DRAWS = 200


class _FamilyWalker:
    """One ``WalkerParams`` family on the CT row, in the eval.baselines actor shape: WalkerActor's
    ``act`` over a one-family ``RandomWalker`` mix, because WalkerActor takes no mix. Each
    ``reset()`` redraws the family's ``p_stop`` and ``duty`` inside their ranges."""

    def __init__(self, rng, params: WalkerParams):
        self.core = RandomWalker(1, rng, mix=(params, ))

    def reset(self):
        self.core.reset()

    def act(self, obs, st, vis_prev, env):
        act = np.zeros((len(obs), ACTION_DIM), dtype=np.int32)
        act[L0.STATUE, H_MOVE] = self.core.step()[0]
        return act, np.zeros((len(obs), AIM_DIM), dtype=np.float32)


def _walker(params: WalkerParams):
    """factory(seed) -> the CT row's walker for one family. A family that never stops and always
    presses is WalkerActor's hold walker, which oracle_tracker plays (the same draws on the same
    seed); any other family plays through ``_FamilyWalker``."""
    if params.p_stop == (0.0, 0.0) and params.duty == (1.0, 1.0):
        return lambda seed: WalkerActor(np.random.default_rng(seed), [L0.STATUE], *params.hold)
    return lambda seed: _FamilyWalker(np.random.default_rng(seed), params)


# name -> factory(seed) -> opponent actor for the CT row: the Rung 1b training families
# (eval.walker.TRAIN_MIX: a statue, a fast walker, a stop-and-go walker) and HELD_OUT, the long
# straight runs training leaves out. tests/experiment/test_aim_eval.py pins the names to them.
OPPONENTS = {
    "statue": lambda seed: IdleActor(),
    "walker-4-16": _walker(TRAIN_MIX[1]),
    "walker-2-8-stop": _walker(TRAIN_MIX[2]),
    "heldout-24-48": _walker(HELD_OUT),
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
        self.fires = []                # (episode tick, |e_yaw|, along, across, hit), half-windows
        self.track = []                # (episode, |w| rad/tick, along rad) once acquired, moving
        self.episode = -1
        self.acquired = False          # this round's crosshair has been on target

    def __call__(self, tick, st_prev, st):
        h, o = self.hero, self.opp
        self.episode += tick == 1
        if not (st_prev["alive"][h] and st_prev["alive"][o]):
            return
        e, w, d = aim_geometry(st_prev, st, h, o)
        split = decompose(e, w)
        half = math.asin(min(HIT_HALF_WIDTH / max(d, 1e-6), 1.0))
        self.acquired = (self.acquired and tick > 1) or abs(e[0]) < half
        if split is not None and self.acquired:
            self.track.append((self.episode, math.hypot(w[0], w[1]), split[0]))
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

        out: dict[str, float | int | list | None] = {
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
            "lag_by_speed": None,
        }
        if len(self.track) >= MIN_TRACK_TICKS:
            episode, speed, along_rad = np.array(self.track).T
            ratio = along_rad / speed
            out["lag_ticks"] = float(np.median(ratio))
            out["lag_ticks_se"] = _episode_bootstrap_se(ratio, episode)
            out["lag_by_speed"] = _by_speed_quintile(speed, ratio)
        return out


def _episode_bootstrap_se(values, episode) -> float:
    """The spread of ``median(values)`` over resamples of whole episodes.

    WHY episodes, not ticks: consecutive ticks of a round are strongly correlated, so a
    per-tick standard error understates the uncertainty. Seeded, so a cell reproduces.
    """
    groups = [values[episode == e] for e in np.unique(episode)]
    rng = np.random.default_rng(0)
    medians = [
        np.median(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    return float(np.std(medians))


def _by_speed_quintile(speed, ratio) -> list:
    """``[|w| median, along/|w| median, ticks]`` per quintile of |w|: the lag curve (research
    C3). A flat curve is a lag proportional to the target's angular speed."""
    edges = np.quantile(speed, np.linspace(0.0, 1.0, 6))
    idx = np.clip(np.searchsorted(edges, speed, side="right") - 1, 0, 4)
    return [[
        float(np.median(speed[idx == k])),
        float(np.median(ratio[idx == k])),
        int((idx == k).sum())
    ] for k in range(5) if (idx == k).any()]


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
    rows = out["references"] + out["cells"]
    wo = max(len(r["opponent"]) for r in rows)
    lines = [
        f"aim_eval: {out['episodes']} episodes per cell, seed {out['seed']}, "
        f"bptt_horizon {out['bptt_horizon']}; e in hit half-windows asin(16/d)",
        f"{'opponent':<{wo}} {'hero':<10} {'aim':<6} {'lstm':<7} kills  ttk50 ttk90 hit/f "
        f" |e|50 |e|90 off   along across lag(t)  se   ticks  hit/f<=H hit/f>H gap_k  gap_ttk"
    ]
    for r in rows:
        gap = r.get("gap_to_oracle", {})
        lines.append(
            f"{r['opponent']:<{wo}} {r['hero']:<10} {r['aim'] or '-':<6} {r['lstm'] or '-':<7} "
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
    # The aim log-sigma parameters by state_dict name, before forward()'s clamp: one
    # ``aim_log_std``, or ``aim_log_std_t`` and ``aim_log_std_ct`` with split heads.
    log_std = {k: v.tolist() for k, v in policy.state_dict().items() if "aim_log_std" in k}
    out.update(checkpoint=str(args.checkpoint),
               env_mismatches=env_mismatches(cfg),
               aim_log_std=log_std)
    for m in out["env_mismatches"]:
        print(f"WARNING: the run trained a different env, {m}")
    print(format_table(out))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
