#!/usr/bin/env python
"""Oracle-vs-statue solvability check — the precondition for reading Rung 1a §3.

WHAT
----
Drives the EXACT Rung 1a environment (ARENA_DUEL_V1, ``n_active_per_team=1``,
``round_time=160``, ``pin_pitch=1``, ``crouch_enabled=0``) with two scripted
actors and no learning anywhere in the loop:

  * hero   — agent 0 (T): ``eval.baselines.OracleActor``. Per tick it reads the
             enemy's position from the live C state, turns the continuous Δyaw
             toward that bearing (clamped to ±``sd->max_turn_speed``, exactly as
             ``env_step`` will clamp it) and pulls the trigger whenever the enemy
             is visible, in range, and the weapon is off cooldown.
  * statue — agent 5 (CT): ``eval.baselines.IdleActor``. Every discrete head at
             bin 0 and Δyaw = Δpitch = 0, so it never moves, turns, or fires.

It reports kill rate, time-to-kill quantiles and the R0-A shot counters, then
prints one PASS/FAIL line. Exit status 0 = PASS, 1 = FAIL.

    UV_NO_SYNC=1 uv run python -m cs2rl.experiment.oracle_statue
    UV_NO_SYNC=1 uv run python -m cs2rl.experiment.oracle_statue --statue-z 24
    UV_NO_SYNC=1 uv run python -m cs2rl.experiment.oracle_statue --obs-only

From a worktree, put its own src/ first: ``env UV_NO_SYNC=1
PYTHONPATH=<checkout>/src uv run python -m cs2rl.experiment.oracle_statue ...``.

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

THE OBS-ONLY VARIANT (``--obs-only``)
------------------------------------
The default oracle reads the enemy's position out of the live C state, so it
proves the env is solvable BY ACTIONS and nothing more. It would keep passing at
200/200 with the enemy block of the observation vector encoded wrongly — rotated
by the wrong sign, or normalised by the wrong constant — because it never looks
at it. A policy has only the obs, so that bug class turns a solvable env into an
unlearnable one while this gate stays green.

``--obs-only`` swaps ``OracleActor`` for ``ObsOracleActor``, which derives the
enemy's bearing, distance and height offset from ``obs[hero]`` (the same array
the network is fed) and keeps every other decision — the ±max_turn_speed yaw
clamp, the range test, the fire/reload discipline — byte-for-byte identical. A
divergence between the two modes therefore isolates the obs encoding.

The kill-rate and TTK bars are the same ones: this is not a second gate with a
second threshold. It does add three checks that exist only in this mode, and
they are part of the EXIT CODE rather than advisory prints (see ``verdict``):
the enemy slot's two encodings of the relative position must agree, the rz
decoded from slot +2 must match the rz measured off the C state, and the blind
ticks must not exceed one per episode. They carry the mode, because the failures
they see are invisible to the kill rate — mutation-tested, both at 1.000 kills:
``EN_DIST`` pointed one slot over, and ``OBS_Z_SCALE`` halved.

Run it as ``--obs-only --episodes 200 --seed 0`` and expect the same verdict; a
mode that kills on ground truth and not on obs is a finding about
``cs2_observations.h``, not about the sim's combat.

WHAT ``--obs-only`` DOES NOT CERTIFY
  * The VISIBILITY GATE. ``can_see`` (+3) and ``alive`` (+4) hold the same value
    on every tick of this env — permanent 2D LoS, and a target that never dies
    mid-round — so an enemy block gated on the wrong one of the two decodes
    identically and passes 20/20 with 0 inconsistent slots. Only an env with
    occlusion (or a target that dies while unseen) can test that flag.
  * Anything the actor does not read: the last-known-position memory branch
    (never exercised at 100 % LoS), the weapon-state slots (taken from the C
    state on purpose — see ``ObsOracleActor``), and every non-enemy block.
  * That a POLICY can learn from the obs. This says the enemy block carries the
    geometry needed to aim; it says nothing about scale, normalisation or
    conditioning of the rest of the vector.

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
* Importing ``eval.baselines`` pulls in torch (``PolicyActor`` needs it). Nothing
  here uses it, but the import cost is real; that is the price of reusing the
  evaluator's actors instead of writing a second oracle that can drift from it.
* ``env.reset()`` returns an ALL-ZERO obs (compute_observations has not run
  yet), so ``--obs-only`` is necessarily blind on tick 1 of every round and
  stands still for it. That is one tick of TTK, reported as ``obs blind ticks``
  (expected value: exactly one per episode — more means the hero lost sight of
  the statue mid-round, which is a different failure than a bad encoding).
* This script never writes to ``outputs/`` and never touches training state.
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from cs2rl.env.c.cs2_env import N_AGENTS, TEAM_SIZE
from cs2rl.eval.baselines import (
    ACTION_DIM,
    AIM_DIM,
    EYE_CROUCH,
    EYE_STAND,
    H_MOVE,
    H_RELOAD,
    H_SHOOT,
    TORSO_STAND,
    BaselineEvaluator,
    IdleActor,
    OracleActor,
    vis_from_obs,
    wrap_pi,
)
from cs2rl.spec.obs import OBS_BLOCKS, OBS_ENEMY_COUNT, OBS_ENEMY_STRIDE

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

# ── observation-vector layout, for --obs-only ────────────────────────────────
# Block bounds come from the GENERATED spec (src/cs2rl/spec/obs.py, regenerated from
# the OBS_* macros in cs2_types.h by scripts/sync_action_spec.py) — never
# hardcode 56 / 5 / 8 here.
OBS_ENEMY_BASE = OBS_BLOCKS["enemy"][0]

# Sub-slot field order inside ONE enemy slot, mirroring the writer in
# cs2_observations.h ("Enemies (OBS_ENEMY_BASE ..): 5 × 8"):
#   +0/+1  (rx, ry) / map_diag, rotated by -facing  (+x ahead, +y left)
#   +2     (enemy z - self z) / 128            [written only when can_see]
#   +3     can_see (alive-gated vis10)   +4  alive
#   +5/+6  sin/cos of the facing-relative bearing  [only when can_see]
#   +7     2D distance / map_diag                  [only when can_see]
# No generated constant exists for the field order, so this is the mirror that
# can drift. That is the point: --obs-only is its tripwire, because the
# ground-truth mode would keep passing 200/200 with every one of these wrong.
EN_RX, EN_RY, EN_DZ, EN_SEE, EN_ALIVE, EN_SIN, EN_COS, EN_DIST = range(OBS_ENEMY_STRIDE)

# Self-block slot the obs actor needs: obs[13] = is_crouching (cs2_observations.h).
# crouch_enabled=0 pins it to 0 in this env; decoded anyway so the eye height is
# never a silent assumption.
OBS_SELF_CROUCH = 13

# Every z-delta in the obs (self +5, teammate +2, enemy +2) is normalised by
# 128 u. MIRROR of cs2_observations.h — same silent-drift hazard as the geometry
# constants in eval.baselines.
OBS_Z_SCALE = 128.0

# The enemy slot encodes the same relative position TWICE — as the rotated
# (rx, ry) pair and as bearing sin/cos + distance — written by different lines
# of C. The actor steers by the bearing pair only, so a disagreement between the
# two is an encoding bug that a kill-rate check alone cannot see; it is counted
# and reported. Tolerances are loose enough that a float32 round-trip never
# trips them (1e-2 rad ≈ 0.6°, 1e-3 · map_diag ≈ 0.36 u on the arena) and tight
# enough to catch a sign flip, a wrong rotation or a wrong normaliser.
OBS_BEARING_TOL_RAD = 1e-2
OBS_DIST_TOL_NORM = 1e-3
# How far the rz decoded from slot +2 may sit from the rz measured off the C
# state before the report calls it a disagreement. Generous (0.5 u against a
# 36 u semi-axis) because the two samples are one tick apart; a wrong
# normaliser or a dropped term misses by tens of units, not by half of one.
OBS_RZ_TOL = 0.5
# Below this 2D separation the bearing is numerically meaningless (atan2 of two
# quantisation residues), so the cross-check is skipped rather than trusted.
OBS_BEARING_MIN_DIST = 1.0

# ── FAIL diagnostics ─────────────────────────────────────────────────────────
# Per-episode spawn geometry kept for episodes that produced no kill, and how
# many of them are printed individually. A total wipeout must not bury the
# verdict line under 200 rows, and past ~10 the aggregate quantiles printed
# above the list are the more useful read anyway.
FAILURE_GEOMETRY_KEYS = ("hero_xy", "statue_xy", "spawn_dist", "yaw_err")
FAIL_DETAIL_LIMIT = 10


def build_env(seed: int, round_time: int = ROUND_TIME):
    """The Rung 1a arena env, built for instrument use (auto_reset off).

    WHY the knobs are hard-coded rather than passed through: the whole point of
    the check is that it runs the env the smoke runs. A caller that wants a
    different env is asking a different question and should say so in code.
    """
    from cs2rl.env.c.cs2_env import make_env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_arena_duel_map
    return make_env(config=EnvConfig(n_active_per_team=N_ACTIVE_PER_TEAM,
                                     pin_pitch=PIN_PITCH,
                                     crouch_enabled=CROUCH_ENABLED,
                                     round_time=round_time),
                    map_data=make_arena_duel_map(),
                    auto_reset=False,
                    seed=seed)


class ObsOracleActor:
    """``OracleActor`` with the enemy located from the OBSERVATION VECTOR.

    WHAT IS DIFFERENT from ``eval.baselines.OracleActor`` — and only this:

      * target choice, bearing, range and height offset come from ``obs[i]``
        (the array the policy is fed), not from the C ``AgentState``;
      * no nav graph. Obs carries no area indices, so an enemy that is not
        visible is approached in a straight line toward its last-known-position
        slot — exactly what ``OracleActor`` does when ``next_hop_xy`` returns
        None. In the arena both agents are in permanent 2D LoS, so this branch
        is nearly dead; ``obs blind ticks`` in the summary says how often it (or
        the do-nothing branch below) was reached.

    EVERYTHING ELSE IS DELIBERATELY IDENTICAL: the Δyaw clamp to
    ±max_turn_speed, the ``laser_range`` test, and the fire/reload discipline —
    which is read from the C state (``fire_cd`` / ``reload_ticks`` /
    ``switch_ticks`` / ammo) in BOTH actors on purpose. Those fields are partly
    observable (obs[17] ammo, obs[19] reload, obs[21] fire cooldown; switch is
    not), but decoding them here would put a second suspect in the room: the
    whole value of this actor is that a divergence from the ground-truth mode
    can only be the enemy encoding.

    NOT AN EVAL BASELINE. Like ``IdleActor`` this is instrument code; it is a
    strictly worse fighter than ``OracleActor`` (no velocity lead, no nav) and
    exists to test the obs, not to score a policy against.

    PITFALLS
      * One-tick velocity lead is GONE. ``OracleActor`` extrapolates the target
        by v·dt because the aim command lands after movement; obs carries no
        enemy velocity, so this actor aims where the enemy WAS. Harmless here
        (the target is a statue) and a real handicap against anything that
        moves — do not reuse it as a moving-target oracle.
      * ``obs[b + EN_DZ]`` is visibility-gated in C: 0 means "invisible", not
        "same height". Only the ``can_see`` branch below reads it.
      * Enemy stance is NOT in the obs, so the target's torso offset is assumed
        standing (``TORSO_STAND``). True for Rung 1a (``crouch_enabled=0``);
        revisit before reusing this with the crouch head unmasked.
      * The decoded height offset does NOT influence whether a shot lands.
        ``pin_pitch=1`` makes the env ignore the pitch command, and the range
        test never binds at ``laser_range`` 3000 on a 360 u arena, so a WRONG
        ``EN_DZ`` decode would still kill 200/200. That is why the decoded rz is
        recorded and reported against the C-measured rz instead of being trusted
        because the kills came in.
      * Counters are RUN totals, not per-episode — ``reset`` deliberately does
        not clear them (see its docstring).
    """

    name = "obs-oracle"

    def __init__(self, max_turn_speed, laser_range, map_diag, rows):
        self.mts = max_turn_speed
        self.laser_range = laser_range
        self.map_diag = map_diag
        # Which agent rows this actor is responsible for. `_episode` splices
        # only the hero's rows out of the returned arrays, so filling the rest
        # would be dead work whose diagnostics (blind ticks) would still land in
        # the counters and misreport the hero.
        self.rows = list(rows)
        self.blind_ticks = 0
        self.inconsistent_slots = 0
        # Range of the eye-to-torso rz DECODED from enemy slot +2, over the
        # targets actually engaged. Reported next to the same quantity measured
        # off the C state, which is the only way this mode says anything about
        # the z encoding — see the pitfall about pin_pitch above.
        self.rz_min = None
        self.rz_max = None

    def reset(self):
        """Per-episode hook — intentionally a no-op.

        There is no per-episode state, and the diagnostic counters/ranges are
        accumulated over the whole RUN: they are reported once, next to totals
        like ``shots_fired``, and clearing them here would silently report only
        the last episode's.
        """

    def _decode_slot(self, o, s, eye_z_offset):
        """One enemy slot → ``(seen, dist3d, rel_bearing, dist2d, rz)`` or None.

        None means "this slot carries no usable target": a dead/empty slot, or
        an invisible enemy the agent has no memory of (the C writer leaves such
        a slot at its memset zeros, so an all-zero rel-pos IS the sentinel).

        ``rz`` is the eye-to-torso vertical offset the combat ray will see:
        ``(enemy_z - self_z) + TORSO_STAND - eye``, the same quantity
        ``OracleActor`` builds from ground truth. It is 0 on the memory branch
        because the C writer leaves +2 unwritten there — the actor is only
        walking then, and walks level.
        """
        b = OBS_ENEMY_BASE + s * OBS_ENEMY_STRIDE
        if o[b + EN_ALIVE] < 0.5:
            return None
        rx = float(o[b + EN_RX]) * self.map_diag
        ry = float(o[b + EN_RY]) * self.map_diag
        if o[b + EN_SEE] < 0.5:
            if rx == 0.0 and ry == 0.0:
                return None            # alive, unseen, no memory
            d2 = math.hypot(rx, ry)
            return False, d2, math.atan2(ry, rx), d2, 0.0
        dist2d = float(o[b + EN_DIST]) * self.map_diag
        rel = math.atan2(float(o[b + EN_SIN]), float(o[b + EN_COS]))
        rz = float(o[b + EN_DZ]) * OBS_Z_SCALE + TORSO_STAND - eye_z_offset
        self._cross_check(rx, ry, dist2d, rel)
        return True, math.hypot(dist2d, rz), rel, dist2d, rz

    def _cross_check(self, rx, ry, dist2d, rel):
        """Count slots whose two encodings of the same relative position differ.

        Not used for control — purely a tripwire, see OBS_BEARING_TOL_RAD.
        """
        d2 = math.hypot(rx, ry)
        if abs(d2 - dist2d) > OBS_DIST_TOL_NORM * self.map_diag:
            self.inconsistent_slots += 1
        elif (d2 >= OBS_BEARING_MIN_DIST
              and abs(wrap_pi(math.atan2(ry, rx) - rel)) > OBS_BEARING_TOL_RAD):
            self.inconsistent_slots += 1

    def act(self, obs, st, vis_prev, env):
        """Same signature and same return shapes as every eval-baselines actor.

        ``vis_prev`` and ``env`` are accepted and ignored: visibility here comes
        from the obs slot's own ``can_see`` flag, which is the point of the mode.
        ``st`` is used ONLY for this agent's own weapon state (see class doc).
        """
        act = np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32)
        cont = np.zeros((N_AGENTS, AIM_DIM), dtype=np.float32)

        for i in self.rows:
            if not st["alive"][i]:
                continue
            o = obs[i]
            eye = EYE_CROUCH if o[OBS_SELF_CROUCH] >= 0.5 else EYE_STAND
            best = None
            for s in range(OBS_ENEMY_COUNT):
                cand = self._decode_slot(o, s, eye)
                if cand is None:
                    continue
                # Same preference as OracleActor: any visible enemy beats any
                # invisible one, ties break on distance. Shooting an enemy the
                # sim says we cannot see is a guaranteed miss that still burns
                # the fire cooldown.
                better = (best is None or (cand[0] and not best[0])
                          or (cand[0] == best[0] and cand[1] < best[1]))
                if better:
                    best = cand
            if best is None:
                # Nothing in the obs to aim at or walk toward. Standing still is
                # the honest response — the ground-truth oracle would use the
                # nav graph here, and faking that would import the ground truth
                # this mode exists to exclude.
                self.blind_ticks += 1
                continue

            # `rel` is ALREADY the facing-relative bearing (C writes it as
            # wrap_pi(atan2(dy, dx) - facing)), so it IS the Δyaw command — no
            # subtraction of our own facing, which is exactly the step R0-E.1
            # moved out of the policy and into the encoder.
            seen, dist3d, rel, dist2d, rz = best
            engage = seen and dist3d <= self.laser_range
            if engage:
                self.rz_min = rz if self.rz_min is None else min(self.rz_min, rz)
                self.rz_max = rz if self.rz_max is None else max(self.rz_max, rz)
                tgt_pitch = math.atan2(rz, dist2d)
            else:
                tgt_pitch = 0.0        # level look while walking
                act[i, H_MOVE] = 1     # facing-local bin 1 = forward
            cont[i, 0] = float(np.clip(rel, -self.mts, self.mts))
            cont[i, 1] = float(np.clip(tgt_pitch, -math.pi / 2, math.pi / 2))

            ready = (st["fire_cd"][i] == 0 and st["reload_ticks"][i] == 0
                     and st["switch_ticks"][i] == 0)
            has_ammo = st["ammo_clip"][i] > 0
            if engage and ready and has_ammo:
                act[i, H_SHOOT] = 1
            elif not has_ammo and st["reload_ticks"][i] == 0 and st["ammo_reserve"][i] > 0:
                act[i, H_RELOAD] = 1
        return act, cont


def _build_hero_actor(ev, seed: int, obs_only: bool):
    """The hero's actor for this run — ground-truth oracle, or its obs-only twin.

    The obs actor is scoped to the hero's rows (``range(TEAM_SIZE)``) because
    ``_episode`` splices only those into the step; filling the statue's row too
    would be dead work whose blind-tick count would still land in the totals the
    summary attributes to the hero.
    """
    if obs_only:
        return ObsOracleActor(ev.max_turn_speed, ev.laser_range, ev.map_diag, range(TEAM_SIZE))
    return OracleActor(np.random.default_rng(seed), ev.max_turn_speed, ev.laser_range, ev.nav)


def _hold_statue_above_ground(env, ground_z: float, offset: float) -> None:
    """Re-place the statue ``offset`` units above its spawn surface, this tick.

    Must be called BEFORE every ``env.step``. Two sim behaviours make a one-shot
    write useless (cs2_movement.h):
      * a grounded agent has ``a->z`` snapped to ``_surface_z`` every tick, which
        would undo the lift immediately — hence ``is_airborne = 1``;
      * an airborne agent accumulates gravity, so without ``vz = 0`` and a fresh
        ``z`` each tick it would arc back down within ~10 ticks and the
        experiment would silently become the ground experiment.

    PITFALL: the value the combat ray sees is ``offset − 1.5625``, not
    ``offset`` — process_movement runs between this write and process_combat,
    and its leapfrog applies half a gravity step first (cs2_movement.h:
    ``0.5 · 800 · (1/16)² = 1.5625``). Callers measure the realised offset
    instead of assuming it.
    """
    a = env._c_env.game.agents[STATUE]
    a.z = ground_z + offset
    a.vz = 0.0
    a.is_airborne = 1


def _episode(env, ev, oracle, statue, statue_z, round_time):
    """One round. Returns a dict: ttk, realised rz samples, unmatched slots, spawn geometry.

    ``ttk`` is the tick index on which the statue's ``alive`` flag flipped to 0,
    counting from 1 (``g->tick`` after the k-th step is exactly k), or None if it
    survived the round. Per-episode shot counters are left in the env's
    ``episode_stats`` for the caller to read before the next reset clears them.

    The spawn geometry (``spawn_dist``, ``yaw_err`` and both xy pairs, all read
    at reset before the first step) is what the FAIL report prints. WHY it is
    captured on every episode rather than re-derived for the failures: the C RNG
    advances with the run, so there is no way to replay episode 137's spawn
    after the fact — either it is recorded as it happens or it is gone.
    """
    obs, _ = env.reset()
    oracle.reset()
    statue.reset()
    ground_z = float(env._c_env.game.agents[STATUE].z)
    st = ev.reader.read().snapshot()
    spawn_dx = float(st["x"][STATUE] - st["x"][HERO])
    spawn_dy = float(st["y"][STATUE] - st["y"][HERO])
    geometry = {
        "hero_xy": (float(st["x"][HERO]), float(st["y"][HERO])),
        "statue_xy": (float(st["x"][STATUE]), float(st["y"][STATUE])),
        "spawn_dist": math.hypot(spawn_dx, spawn_dy),
        "yaw_err": float(wrap_pi(math.atan2(spawn_dy, spawn_dx) - st["facing"][HERO])),
    }
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
    return {"ttk": ttk, "rz_samples": rz_samples, "unmatched": unmatched_total, **geometry}


def run_check(episodes: int = 200,
              seed: int = 0,
              statue_z: float = 0.0,
              round_time: int = ROUND_TIME,
              obs_only: bool = False) -> dict:
    """Play ``episodes`` oracle-vs-statue rounds; return the summary dict.

    ``obs_only`` swaps the ground-truth ``OracleActor`` for ``ObsOracleActor``
    (see ``_build_hero_actor`` and that class's docstring) and is the ONLY thing
    that differs between the two modes: same env, same seed, same statue, same
    loop, same thresholds.

    ``obs_blind_ticks`` / ``obs_inconsistent_slots`` come back as None (not 0)
    in ground-truth mode: that actor never reads an obs slot, so it cannot make
    a claim about them either way.

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
        oracle = _build_hero_actor(ev, seed, obs_only)
        statue = IdleActor()

        ttks, rz_min, rz_max, unmatched = [], None, None, 0
        totals = dict.fromkeys(("shots_fired", "shots_with_enemy_in_los", "shots_facing_enemy",
                                "shots_on_target", "shots_hit", "shots_stance_blocked"), 0)
        kills = 0
        failures = []
        for ep in range(episodes):
            r = _episode(env, ev, oracle, statue, statue_z, round_time)
            unmatched += r["unmatched"]
            if r["rz_samples"]:
                ep_lo, ep_hi = min(r["rz_samples"]), max(r["rz_samples"])
                rz_min = ep_lo if rz_min is None else min(rz_min, ep_lo)
                rz_max = ep_hi if rz_max is None else max(rz_max, ep_hi)
            es = env._c_env.episode_stats
            for k in totals:
                totals[k] += int(getattr(es, k))
            if r["ttk"] is not None:
                kills += 1
            else:
                failures.append({"episode": ep, **{k: r[k] for k in FAILURE_GEOMETRY_KEYS}})
            ttks.append(r["ttk"])
    finally:
        env.close()

    # Censored TTK: an episode with no kill enters the quantiles as round_time+1,
    # never as a dropped sample. Dropping them would let a run that kills in 10 %
    # of rounds report a beautiful median — the exact failure this gate exists to
    # catch. `ttk_median_killed` is reported alongside for diagnosis only.
    censored = np.array([round_time + 1 if t is None else t for t in ttks], dtype=float)
    killed = [t for t in ttks if t is not None]
    # The obs diagnostics live on ObsOracleActor only, which _build_hero_actor
    # returns exactly when obs_only is set.
    obs_actor = oracle if isinstance(oracle, ObsOracleActor) else None
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
        "obs_only": obs_only,
        "obs_blind_ticks": obs_actor.blind_ticks if obs_actor is not None else None,
        "obs_inconsistent_slots": obs_actor.inconsistent_slots if obs_actor is not None else None,
        "obs_rz_min": obs_actor.rz_min if obs_actor is not None else None,
        "obs_rz_max": obs_actor.rz_max if obs_actor is not None else None,
        "failures": failures,
        **totals,
    }


def _rz_range(lo, hi) -> str:
    """``"+lo .. +hi"``, or "n/a" when nothing was sampled. Shared by both rz rows.

    Either endpoint missing means the range is empty (they are filled and
    cleared together), so a half-populated pair reads "n/a" rather than raising
    on the format — ``verdict`` renders this detail for hand-built summaries too.
    """
    return "n/a" if lo is None or hi is None else f"{lo:+.2f} .. {hi:+.2f}"


def _rz_disagrees(res: dict) -> bool:
    """Does the rz DECODED from enemy slot +2 differ from the one measured in C?

    They are the same physical quantity (eye-to-torso vertical offset) and the
    obs sample is at most one tick older, which on a flat arena holding a
    motionless statue is no difference at all. A gap past ``OBS_RZ_TOL`` means
    the slot is encoded or normalised wrongly — which nothing else in this run
    would notice, because ``pin_pitch=1`` makes the pitch it feeds inert and the
    range test never binds (see ``ObsOracleActor``). Vacuously False when either
    side has no samples.

    PITFALL: the two ranges are drawn from DIFFERENT tick populations — the obs
    range accumulates only on ticks where the actor engages, the C range on every
    tick where both agents are alive. Comparing them is exact here only because
    the statue's height is constant over a round; against a target whose height
    varies (a jumping or crouching opponent) the two would differ legitimately
    and this would warn on a correct encoding. Re-scope it to a common tick set
    before reusing it on such a target.
    """
    obs_lo, obs_hi = res.get("obs_rz_min"), res.get("obs_rz_max")
    c_lo, c_hi = res.get("rz_min"), res.get("rz_max")
    if obs_lo is None or obs_hi is None or c_lo is None or c_hi is None:
        return False
    return abs(obs_lo - c_lo) > OBS_RZ_TOL or abs(obs_hi - c_hi) > OBS_RZ_TOL


def verdict(res: dict) -> tuple[bool, list[tuple[str, bool, str]]]:
    """(passed, [(name, ok, detail), ...]) — the gate, evaluated on a summary.

    Three checks in ground-truth mode. The first two are the brief's gate. The
    third (``shots_stance_blocked == 0``) is trivially true on flat ground and is
    the entire content of the ``--statue-z`` variant: it says the vertical offset
    the shots were taken against stayed inside the target's vertical semi-axis,
    so a kill there is the v1c ellipsoid working and not the offset quietly being
    ignored.

    ``--obs-only`` adds three more (r1-I1). They existed before as ``<-- WARNING``
    markers in the report that left the exit code at 0 — so a controller
    scripting the documented CLI contract ("exit 0 = PASS") read GREEN on exactly
    the encoding bugs the mode was built to catch. Both mutations that motivated
    this keep the kill rate at 1.000 and are seen by nothing else: ``EN_DIST``
    pointed at slot +2 (131 inconsistent slots over 20 episodes), and
    ``OBS_Z_SCALE`` halved (obs rz +11.22 against a C rz of +22.44).

    Every obs field is read through ``.get``, so the three extra checks are
    skipped for any summary that is not an obs-only run — a ground-truth summary,
    which carries the obs keys as None, and a partial dict built by hand in a
    test both evaluate the ground-truth three and nothing else.
    """
    kill_ok = res["kill_rate"] >= PASS_MIN_KILL_RATE
    ttk_ok = res["ttk_median"] < PASS_MAX_MEDIAN_TTK
    checks = [
        (f"kill_rate >= {PASS_MIN_KILL_RATE:.2f}", kill_ok, f"{res['kill_rate']:.3f}"),
        (f"median_ttk < {PASS_MAX_MEDIAN_TTK:.0f}", ttk_ok, f"{res['ttk_median']:.1f}"),
        ("shots_stance_blocked == 0", res["shots_stance_blocked"] == 0,
         str(res["shots_stance_blocked"])),
    ]
    if res.get("obs_only"):
        # `or 0` rather than a default: these arrive as None from a summary whose
        # obs_only flag was set by hand, and None does not compare with int.
        inconsistent = res.get("obs_inconsistent_slots") or 0
        blind = res.get("obs_blind_ticks") or 0
        episodes = res.get("episodes") or 0
        # The blind-tick bound is `<= episodes`, not `== episodes`: exactly one
        # blind tick per episode is structural (env.reset returns an all-zero
        # obs). MORE than that means the hero lost the statue mid-round, which
        # makes the kill numbers a statement about LoS rather than about the
        # encoding — the same condition the report has always warned on.
        checks += [
            ("obs_inconsistent_slots == 0", inconsistent == 0, str(inconsistent)),
            (f"|obs_rz - C rz| <= {OBS_RZ_TOL}", not _rz_disagrees(res),
             _rz_range(res.get("obs_rz_min"), res.get("obs_rz_max"))),
            (f"obs_blind_ticks <= {episodes}", blind <= episodes, str(blind)),
        ]
    return all(ok for _, ok, _ in checks), checks


def _failure_lines(res: dict) -> list[str]:
    """Spawn geometry of the episodes that produced no kill (FAIL path only).

    WHY (M2-lite): "kill rate 0.985" and "kill rate 0.000" are the same verdict
    line but completely different bugs — one spawn row that the opening turn
    cannot cover in time, versus a sim in which nothing can die. The aggregate
    counters above cannot separate them, and the C RNG makes the failing rounds
    unreplayable after the fact, so the geometry is printed here or lost.

    Empty when nothing was censored: a FAIL on median TTK alone (every round
    killed, just slowly) has no failing episode to describe.
    """
    fails = res["failures"]
    if not fails:
        return []
    dist = np.array([f["spawn_dist"] for f in fails], dtype=float)
    yaw_deg = np.degrees(np.abs(np.array([f["yaw_err"] for f in fails], dtype=float)))
    lines = [
        "",
        f"failing episodes (no kill)  {len(fails)} of {res['episodes']}",
        f"  spawn distance 2D (u)    min {dist.min():.1f}   median {np.median(dist):.1f}   "
        f"max {dist.max():.1f}",
        f"  |initial yaw error| deg  min {yaw_deg.min():.1f}   median {np.median(yaw_deg):.1f}   "
        f"max {yaw_deg.max():.1f}",
    ]
    for f in fails[:FAIL_DETAIL_LIMIT]:
        lines.append(
            f"  ep {f['episode']:>4}   hero ({f['hero_xy'][0]:.0f}, {f['hero_xy'][1]:.0f})"
            f" -> statue ({f['statue_xy'][0]:.0f}, {f['statue_xy'][1]:.0f})   "
            f"dist {f['spawn_dist']:.1f} u   yaw err {math.degrees(f['yaw_err']):+.1f} deg")
    if len(fails) > FAIL_DETAIL_LIMIT:
        lines.append(f"  ... {len(fails) - FAIL_DETAIL_LIMIT} more")
    return lines


def format_summary(res: dict) -> str:
    """Human-readable report — every number the verdict rests on, plus context."""
    passed, checks = verdict(res)
    rz = _rz_range(res["rz_min"], res["rz_max"])
    fired = max(res["shots_fired"], 1)
    # M3: a run with zero kills has no min and no over-kills median. Printing
    # the bare None read as a crashed/missing statistic; say what it is.
    ttk_min = "n/a (no kills)" if res["ttk_min"] is None else str(res["ttk_min"])
    ttk_med_killed = ("n/a (no kills)"
                      if res["ttk_median_killed"] is None else f"{res['ttk_median_killed']:.1f}")
    # M1: shots_facing_enemy (yaw error < 45°, cs2_combat.h) is the DENOMINATOR
    # of the Rung 1 §5 aim criterion hit/facing > 0.45 — it was collected and
    # then thrown away here, which made the criterion unreadable off this report.
    facing = res["shots_facing_enemy"]
    hit_over_facing = "n/a" if facing == 0 else f"{res['shots_hit'] / facing:.3f}"
    lines = [
        "── oracle vs statue — arena-duel, n_active=1, pin_pitch=1, crouch=0 ──",
        f"episodes                 {res['episodes']}  (seed {res['seed']}, "
        f"round_time {res['round_time']})",
        f"statue z offset          {res['statue_z']:+.1f} u requested; "
        f"realised rz at combat {rz} u",
        f"kill rate                {res['kill_rate']:.3f}  ({res['kills']}/{res['episodes']})",
        f"time-to-kill (ticks)     median {res['ttk_median']:.1f}   p90 {res['ttk_p90']:.1f}   "
        f"min {ttk_min}   censored {res['ttk_censored']}",
        f"  median over kills only {ttk_med_killed}",
        f"shots_fired              {res['shots_fired']}",
        f"shots_with_enemy_in_los  {res['shots_with_enemy_in_los']}  "
        f"({res['shots_with_enemy_in_los'] / fired:.3f} of fired)",
        f"shots_facing_enemy       {facing}  ({facing / fired:.3f} of fired)   "
        f"hit/facing {hit_over_facing}  (Rung 1 §5 wants > 0.45)",
        f"shots_on_target          {res['shots_on_target']}  "
        f"({res['shots_on_target'] / fired:.3f} of fired)",
        f"shots_hit                {res['shots_hit']}  ({res['shots_hit'] / fired:.3f} of fired)",
        f"shots_stance_blocked     {res['shots_stance_blocked']}",
        f"unmatched vis slots      {res['unmatched_vis_slots']}"
        f"{'   <-- WARNING: visibility recovery is broken, oracle targeting is suspect' if res['unmatched_vis_slots'] else ''}",
    ]
    if res["obs_only"]:
        blind, inconsistent = res["obs_blind_ticks"], res["obs_inconsistent_slots"]
        lines += [
            "aim source               OBSERVATION VECTOR (enemy block +2/+3/+5/+6/+7), "
            "ground truth NOT read",
            f"obs blind ticks          {blind}  (expected {res['episodes']}: the all-zero obs "
            f"env.reset returns, one per episode)"
            f"{'   <-- hero lost the statue mid-round' if blind > res['episodes'] else ''}",
            f"obs slot inconsistency   {inconsistent}"
            f"{'   <-- WARNING: rel-pos and bearing/distance encodings disagree' if inconsistent else ''}",
            f"obs-decoded rz           {_rz_range(res['obs_rz_min'], res['obs_rz_max'])} u  "
            f"(enemy slot +2; must match the realised rz above)"
            f"{'   <-- WARNING: the obs z-delta is not what the C state says' if _rz_disagrees(res) else ''}",
        ]
    if not passed:
        lines += _failure_lines(res)
    lines.append("")
    for name, ok, detail in checks:
        lines.append(f"  [{'ok' if ok else 'XX'}] {name:<28} {detail}")
    lines.append("")
    lines.append("PASS — environment is solvable; a Rung 1a §3 FAIL is a learning result" if passed
                 else "FAIL — a scripted perfect-aim actor cannot kill a stationary target; "
                 "Rung 1a §3 verdicts on this env are void")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cs2rl.experiment.oracle_statue",
                                 description=__doc__.split("\n\n")[0],
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
    ap.add_argument("--obs-only",
                    action="store_true",
                    help="locate the enemy from the hero's OBSERVATION VECTOR instead of the "
                    "live C state, keeping every other decision identical. Same PASS bar: a "
                    "mode that kills on ground truth but not here is an obs-encoding bug, which "
                    "is invisible to the default run — see the module docstring.")
    args = ap.parse_args(argv)

    res = run_check(episodes=args.episodes,
                    seed=args.seed,
                    statue_z=args.statue_z,
                    round_time=args.round_time,
                    obs_only=args.obs_only)
    print(format_summary(res))
    return 0 if verdict(res)[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
