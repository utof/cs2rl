#!/usr/bin/env python
"""Hash a deterministic N-step rollout on simple_map — the bit-identity oracle.

WHAT: builds one Cs2Env (seed 7, otherwise default kwargs, simple_map), steps
`--steps` ticks with seeded random discrete actions sampled THROUGH the action
masks, and prints a single sha256 hex digest over the concatenated
mask/obs/reward/terminal stream.

TWO MODES, TWO ORACLES — run BOTH, they are complementary, and their hashes are
unrelated to each other (only parent-vs-HEAD comparisons within one mode mean
anything):
  --aim-mode random (default)  uniform aim noise. Nothing is ever on target, so
      this covers spawn placement (the 5-area Fisher-Yates branch), movement,
      nav/PBRS shaping, the mask buffer, the round timer and — at --steps 700 —
      round end + auto-reset. It does NOT cover combat.
  --aim-mode track             aim is steered at each agent's nearest alive
      enemy, and both teams spawn in the map's LoS-open corridor band, so
      agents actually shoot each other. 4 deaths inside 200 steps, first
      damage on tick 0. Use this for ANY change to hitscan, damage, death or
      the visible-enemy scan (e.g. process_combat's nearest_vis_enemy).

WHY: behaviour-neutral sim refactors (spec §6: the parked-agent /
`n_active_per_team` change at n_active=5) must be proven bit-identical to the
parent commit. Run this script at the parent commit and at HEAD with the .so
rebuilt at each; the two hashes must match. Later sim edits (fire-mask fix,
crouch mask, facing-relative bearing) are *expected* to move the hash — that is
the same oracle used in the other direction.

WHAT IS HASHED, and why each piece:
  - the action-mask buffer, BEFORE each step. The sim never reads `env->masks`
    (masks are policy-facing only), so a mask-only change would leave obs and
    rewards untouched and be invisible to the oracle without this.
  - observations after reset and after every step (the agent-visible state).
  - rewards and terminals after every step (the learning signal).
Truncations are deliberately left out: they are redundant with terminals on
this path and hashing them buys nothing over the reward/terminal stream.
Config (steps / seed / n_active) is deliberately NOT hashed, so that a HEAD run
with `--n-active 5` can be compared directly against a parent run that predates
the flag.

PITFALLS:
  - Rebuild the .so before each run:
    `PY_ZIG=... UV_NO_SYNC=1 uv run python setup.py build_ext --inplace`.
    A stale binding silently fingerprints the wrong commit.
  - Uses `make_simple_map()`, NOT the baked CS2 arena. The arena has a single
    spawn area per side, so all five agents stack on one point and the
    comparison degenerates (spec §8).
  - The RNG stream is mask-dependent by construction: a head whose mask is
    all-zero for an agent consumes no draw. That is intentional — it amplifies
    mask regressions into a hash change — but it also means an unrelated mask
    change reshuffles every subsequent action. Expect large diffs, not small.
  - `auto_reset` is left at its default (True), so a round ending mid-rollout
    rolls straight into a fresh round. That path is deterministic (the C RNG
    carries on) and worth covering, but it does mean the fingerprint also
    covers env_reset.
  - COVERAGE: `--aim-mode random` does not exercise combat. Measured at
    42706ff, over 700 ticks (a full ROUND_TIME=640 round plus rollover) every
    agent stays at 100 HP, kills_t == kills_ct == 0, and the bomb is never
    planted; ~126 shoot actions in the first 200 ticks do zero damage because
    the aim is a random walk. A change confined to the hitscan/damage/death
    path would NOT move the random-mode hash — use --aim-mode track for
    combat-path edits, and quote BOTH hashes when claiming bit-identity.
  - The default 200 steps sees no terminal in either mode; use --steps 700 when
    the change under test could touch round end or reset.
  - Read the stderr counters, not just the hash. `deaths` / `damage_ticks` are
    the combat-coverage evidence: if they are 0 the run proves nothing about
    combat no matter which --aim-mode was passed.

PARENT FINGERPRINTS (seed 7, rng-seed 123, default kwargs). Measured with the
binding built from src/c_env at 88c1d5d. This commit touches only scripts/, so
these are equally the fingerprints of THIS commit's tree — the sim is identical.
Task 3 must reproduce all three with `--n-active 5`, rebuilding its own .so:
    --steps 200                  c925f1a9a85d1446906989610c325599a59b405440871f1be58ce068e3382950
    --steps 700                  9174cc6eeaf0a6fd2ae3da8ee93fe67ef1405e658d77c999d32d671ebf56c5a5
    --steps 200 --aim-mode track cb43ab925b02cc469c6a9368c0e57a5b805bf842594ec3f4279645a056412e15
                                 (deaths=4, damage_ticks=200, min_hp=14)
    --steps 700 --aim-mode track ad3801aae0e242f3941a945caeab97da2f18a224e3c49f9f5424f9ad9c027ac2
                                 (deaths=9, damage_ticks=699, terminal_ticks=1
                                  — the only run that covers combat AND round
                                  end + auto-reset together)

Usage:
    UV_NO_SYNC=1 uv run python scripts/sim_fingerprint.py [--steps 200] [--n-active 5]
    UV_NO_SYNC=1 uv run python scripts/sim_fingerprint.py --aim-mode track [--n-active 5]
"""
import argparse
import hashlib
import math
import sys

import numpy as np

from cs2rl.c_env.cs2_env import make_env
from cs2rl.env.config import EnvConfig
from cs2rl.env.map import SIMPLE_ROOMS, make_simple_map
from cs2rl.env.nav import N_AGENTS, TEAM_SIZE
from cs2rl.spec.action import ACTION_DIM, ACTION_HEAD_SIZES, ACTION_MASK_DIM, AIM_DIM

# The mask buffer is the discrete heads laid end to end. If a head is ever added
# or resized without ACTION_MASK_DIM following, the per-head slicing below would
# read the wrong columns and the oracle would compare garbage — fail loudly.
# `raise`, not `assert`: python -O would strip an assert and the oracle would
# silently hash garbage.
if sum(ACTION_HEAD_SIZES) != ACTION_MASK_DIM or len(ACTION_HEAD_SIZES) != ACTION_DIM:
    raise RuntimeError(f"mask layout drift: sum(ACTION_HEAD_SIZES)={sum(ACTION_HEAD_SIZES)} "
                       f"!= ACTION_MASK_DIM={ACTION_MASK_DIM} or len != ACTION_DIM={ACTION_DIM}; "
                       "regenerate src/cs2rl/spec/action.py")


def sample_masked_actions(masks, rng):
    """Draw one legal discrete action per (agent, head) from the mask buffer.

    Head-major, agent-minor iteration order — arbitrary, but it must never
    change: the order fixes the RNG consumption sequence and therefore the
    fingerprint. An all-zero mask row for a head yields action 0 and consumes
    no randomness (the sim clamps illegal actions anyway).
    """
    act = np.zeros((N_AGENTS, ACTION_DIM), np.int32)
    off = 0
    for head_idx, n in enumerate(ACTION_HEAD_SIZES):
        for agent in range(N_AGENTS):
            allowed = np.flatnonzero(masks[agent, off:off + n])
            act[agent, head_idx] = rng.choice(allowed) if allowed.size else 0
        off += n
    return act


# ── track-mode spawn areas ────────────────────────────────────────────────────
# WHY these differ from the default spawn lists: on simple_map the two spawn
# CLUSTERS (T areas 0-4 at x<512, CT areas 8-12 at x>1500) are joined only by
# the southern corridor band (areas 5/13/6/14/7 at y 192-416). No T-spawn cell
# has line of sight to any CT-spawn cell — the straight line crosses off-mesh
# dead space, and line_of_sight_2d treats that as a wall. Measured: with aim
# tracking AND the shoot head forced on, 700 ticks fire 198 shots for zero
# damage; agents that walk straight at the enemy just pile against the x=512 /
# x=1500 spawn walls (movement collision zeroes velocity, it does not slide),
# so random play never reaches the corridor either.
#
# The minimum deterministic fix is to spawn both teams INSIDE that corridor
# band, where LoS is open: 5→13→6→14→7 are pairwise adjacent, so the raycast
# crosses only connected rooms. Room GEOMETRY is untouched (same SIMPLE_ROOMS)
# — only which areas the two teams spawn in. Everything else (env kwargs, seed,
# action sampling) is identical to random mode.
#
# PITFALLS:
#   - These lists are shorter than TEAM_SIZE, so spawn_team takes its
#     `n_spawns < TEAM_SIZE` branch (one modulo draw per agent) instead of the
#     Fisher-Yates permutation the 5-area default lists take. Track mode
#     therefore covers the OTHER spawn branch; random mode still covers the
#     permutation one. Neither mode covers both — run both.
#   - Two areas per team means teammates stack on a shared area centroid. That
#     is harmless for combat (same-team pairs are never scanned) but it does
#     make track mode a weak witness for per-agent spawn placement. Random mode
#     is the oracle for that.
#   - Do NOT add the catwalk (15) or stairs (16): they share the y=192 edge
#     with the bombsite but are NOT adjacent to it (full-height wall, see
#     line_of_sight_2d in cs2_combat.h), so agents there would have no LoS.
TRACK_T_SPAWNS = [5, 13]               # T-corridor, T-ramp
TRACK_CT_SPAWNS = [7, 14]              # CT-corridor, CT-ramp


def build_map(aim_mode):
    """simple_map for the given aim mode — identical rooms, differing spawns.

    random mode uses make_simple_map()'s defaults so its hash stays comparable
    against any commit that predates --aim-mode.
    """
    if aim_mode == "track":
        return make_simple_map(rooms=SIMPLE_ROOMS,
                               t_spawns=TRACK_T_SPAWNS,
                               ct_spawns=TRACK_CT_SPAWNS)
    return make_simple_map()


# Hitbox geometry mirrored from process_combat (src/cs2rl/c_env/cs2_combat.h): the
# hitscan ray starts at the shooter's EYE and the perpendicular-distance gate is
# measured against the target's TORSO point. Track mode aims eye→torso so a
# converged aim gives perp ≈ 0 and the shot connects.
# PITFALL: hand-mirrored constants. If cs2_combat.h changes them, track mode
# aims slightly off — that does NOT corrupt any hash (the aim is still a pure
# function of state), but the combat coverage this mode exists for degrades
# silently. Re-measure the kill counter after any EYE_HEIGHT_*/TORSO_OFFSET_*
# or hitbox-geometry edit.
EYE_HEIGHT_STAND = 48.0
EYE_HEIGHT_CROUCH = 24.0
TORSO_OFFSET_STAND = 48.0
TORSO_OFFSET_CROUCH = 24.0


def track_aim_actions(game, max_turn):
    """Continuous aim that steers every alive agent onto its nearest alive enemy.

    WHAT: returns the (N_AGENTS, AIM_DIM) float32 continuous-action buffer for
    one tick. Column 0 is a Δyaw (the C step clamps it to ±max_turn_speed and
    wrap_pi's the result); column 1 is an ABSOLUTE pitch (NOT a delta — see the
    Batch 3.5 v1c comment in cs2_env.h). Both are pre-clamped here to
    ±max_turn_speed, the range a tanh-squashed policy head can actually emit.

    WHY: with random aim nothing is ever on target, so 700 ticks of rollout land
    zero damage and the fingerprint is blind to the whole hitscan/damage/death
    path. Deterministic tracking makes agents shoot each other, so the hash
    covers combat too. See --aim-mode in the module docstring.

    Determinism: this is a pure function of the C game state — it draws no
    randomness, so the RNG stream in track mode is the discrete sampler's alone.
    That is why the two modes have different fingerprints; they are separate
    oracles, not two measurements of one.

    PITFALL: `game` must be the LIVE ctypes overlay (`env._c_env.game`), read at
    the top of the tick. A stale copy would aim at last tick's positions and
    still hash deterministically — a silent coverage loss, not a crash.
    """
    cont = np.zeros((N_AGENTS, AIM_DIM), np.float32)
    for i in range(N_AGENTS):
        a = game.agents[i]
        if not a.alive:
            continue                   # dead agents' aim is ignored by the C step
        eye_z = a.z + (EYE_HEIGHT_CROUCH if a.is_crouching else EYE_HEIGHT_STAND)

        # Team layout is fixed: agents [0, TEAM_SIZE) are T, [TEAM_SIZE, N_AGENTS) are CT
        # — same `en_start` split process_combat uses to scan for targets.
        # Unlike process_combat there is NO vis/laser_range filter here: an
        # occluded nearest enemy is still tracked. The stderr `deaths` counter
        # is the guard — if it drops to 0, track mode has stopped proving combat.
        en_start = TEAM_SIZE if a.team == 0 else 0
        best, best_d2 = None, None
        for j in range(en_start, en_start + TEAM_SIZE):
            en = game.agents[j]
            if not en.alive:
                continue
            rx, ry = en.x - a.x, en.y - a.y
            torso_z = en.z + (TORSO_OFFSET_CROUCH if en.is_crouching else TORSO_OFFSET_STAND)
            rz = torso_z - eye_z
            d2 = rx * rx + ry * ry + rz * rz
            if best_d2 is None or d2 < best_d2:
                best, best_d2 = (rx, ry, rz), d2
        if best is None:
            continue                   # team wiped: Δyaw 0 holds yaw; pitch snaps level
        rx, ry, rz = best

        # Shortest signed rotation onto the target bearing, wrapped into [-π, π]
        # BEFORE clamping — without the wrap, a target 10° clockwise across the
        # ±π seam reads as a 350° turn and the clamp sends the agent the long way.
        dyaw = math.atan2(ry, rx) - a.facing
        dyaw = (dyaw + math.pi) % (2.0 * math.pi) - math.pi
        cont[i, 0] = max(-max_turn, min(max_turn, dyaw))
        cont[i, 1] = max(-max_turn, min(max_turn, math.atan2(rz, math.hypot(rx, ry))))
    return cont


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n-active",
                    type=int,
                    default=None,
                    help="n_active_per_team; omit on commits that predate the kwarg")
    ap.add_argument("--seed", type=int, default=7, help="env seed (sanity checks only)")
    ap.add_argument("--rng-seed",
                    type=int,
                    default=123,
                    help="action-sampling seed (sanity checks only)")
    ap.add_argument("--aim-mode",
                    choices=("random", "track"),
                    default="random",
                    help="random: uniform aim noise (never hits — no combat coverage). "
                    "track: aim at the nearest alive enemy, so the hash covers the "
                    "hitscan/damage/death path. Different modes => different hashes.")
    a = ap.parse_args()

    # PITFALL: `--n-active` omitted must build the SAME env as before, so the
    # None arm is a bare `EnvConfig()`, never `EnvConfig(n_active_per_team=<the
    # field default>)`. Spelling the default here would also be a restated
    # default that tests/test_no_restated_env_defaults.py fails on — as the
    # first draft of this very comment was, by writing the number.
    config = EnvConfig() if a.n_active is None else EnvConfig(n_active_per_team=a.n_active)
    env = make_env(config=config, seed=a.seed, map_data=build_map(a.aim_mode))
    rng = np.random.default_rng(a.rng_seed)
    h = hashlib.sha256()

    # Live ctypes overlay onto the C GameState + StaticData. Read-only here:
    # track mode derives aim from it, and the combat counters below read HP and
    # alive flags straight from it rather than trusting the reward stream.
    game = env._c_env.game
    max_turn = float(env._c_env.sd.contents.max_turn_speed)

    obs, _ = env.reset()
    h.update(np.ascontiguousarray(obs).tobytes())

    # Activity counters — printed to stderr so stdout stays a bare hash that can
    # be diffed/piped. A rollout with all-zero rewards would hash fine yet prove
    # nothing, so surface enough to see the episode actually ran. `deaths` and
    # `damage_ticks` are the combat-coverage evidence: if they are 0 the hash
    # says nothing about the hitscan/damage/death path (see PITFALLS).
    n_terminal_ticks, n_nonzero_reward_ticks, reward_abs_sum = 0, 0, 0.0
    n_deaths, n_damage_ticks, min_hp = 0, 0, 100
    # Counting alive→dead EDGES (not a final headcount) so deaths across an
    # auto-reset round rollover still accumulate.
    prev_alive = [bool(game.agents[i].alive) for i in range(N_AGENTS)]

    for _ in range(a.steps):
        masks = np.asarray(env._masks_view)            # (N_AGENTS, ACTION_MASK_DIM) int8
        h.update(np.ascontiguousarray(masks).tobytes())
        act = sample_masked_actions(masks, rng)

        # Order is load-bearing: the discrete sampler consumes RNG first. In
        # random mode the uniform draw follows it; in track mode there is no
        # draw at all, so the two modes' RNG streams diverge immediately.
        if a.aim_mode == "track":
            cont = track_aim_actions(game, max_turn)
        else:
            cont = rng.uniform(-0.5, 0.5, size=(N_AGENTS, AIM_DIM)).astype(np.float32)
        obs, rew, term, _trunc, _info = env.step(act, cont)
        h.update(np.ascontiguousarray(obs).tobytes())
        h.update(np.ascontiguousarray(rew).tobytes())
        h.update(np.ascontiguousarray(term).tobytes())
        n_terminal_ticks += int(np.any(term))
        n_nonzero_reward_ticks += int(np.any(rew != 0))
        reward_abs_sum += float(np.abs(rew).sum())

        alive = [bool(game.agents[i].alive) for i in range(N_AGENTS)]
        n_deaths += sum(1 for i in range(N_AGENTS) if prev_alive[i] and not alive[i])
        prev_alive = alive
        hps = [int(game.agents[i].hp) for i in range(N_AGENTS) if alive[i]]
        if hps:
            min_hp = min(min_hp, min(hps))
        n_damage_ticks += int(any(hp < 100 for hp in hps))

    env.close()
    print(
        f"steps={a.steps} aim_mode={a.aim_mode} terminal_ticks={n_terminal_ticks} "
        f"nonzero_reward_ticks={n_nonzero_reward_ticks} "
        f"reward_abs_sum={reward_abs_sum:.6f} "
        f"deaths={n_deaths} damage_ticks={n_damage_ticks} min_hp={min_hp}",
        file=sys.stderr)
    print(h.hexdigest())


if __name__ == "__main__":
    main()
