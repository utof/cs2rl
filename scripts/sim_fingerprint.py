#!/usr/bin/env python
"""Hash a deterministic N-step random rollout on simple_map — the bit-identity oracle.

WHAT: builds one Cs2Env (seed 7, otherwise default kwargs, simple_map), steps
`--steps` ticks with seeded random discrete actions sampled THROUGH the action
masks plus uniform continuous aim deltas, and prints a single sha256 hex digest
over the concatenated mask/obs/reward/terminal stream.

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
  - COVERAGE, measured at 42706ff: random play never lands a shot. Over 700
    ticks (a full ROUND_TIME=640 round plus rollover) every agent stays at
    100 HP, kills_t == kills_ct == 0, and the bomb is never planted; ~126
    shoot actions in the first 200 ticks do zero damage because the aim is a
    random walk. So this oracle pins spawn placement, movement, aim, nav/PBRS
    shaping, the mask buffer, the round timer and (at --steps 700) round end +
    auto-reset — it does NOT exercise the damage, death or bomb paths. A change
    confined to hitscan damage would NOT move the hash. The default 200 steps
    additionally sees no terminal at all; use --steps 700 when the change under
    test could touch round end or reset.

Usage:
    UV_NO_SYNC=1 uv run python scripts/sim_fingerprint.py [--steps 200] [--n-active 5]
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _action_spec import ACTION_DIM, ACTION_HEAD_SIZES, ACTION_MASK_DIM, AIM_DIM # noqa: E402
from c_env.cs2_env import make_env                                               # noqa: E402
from map import make_simple_map                                                  # noqa: E402
from nav import N_AGENTS                                                         # noqa: E402

# The mask buffer is the discrete heads laid end to end. If a head is ever added
# or resized without ACTION_MASK_DIM following, the per-head slicing below would
# read the wrong columns and the oracle would compare garbage — fail loudly.
assert sum(ACTION_HEAD_SIZES) == ACTION_MASK_DIM, (
    f"mask layout drift: sum(ACTION_HEAD_SIZES)={sum(ACTION_HEAD_SIZES)} "
    f"!= ACTION_MASK_DIM={ACTION_MASK_DIM}; regenerate src/_action_spec.py")
assert len(ACTION_HEAD_SIZES) == ACTION_DIM


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
    a = ap.parse_args()

    kw = {} if a.n_active is None else {"n_active_per_team": a.n_active}
    env = make_env(seed=a.seed, map_data=make_simple_map(), **kw)
    rng = np.random.default_rng(a.rng_seed)
    h = hashlib.sha256()

    obs, _ = env.reset()
    h.update(np.ascontiguousarray(obs).tobytes())

    # Activity counters — printed to stderr so stdout stays a bare hash that can
    # be diffed/piped. A rollout with all-zero rewards would hash fine yet prove
    # nothing, so surface enough to see the episode actually ran.
    n_terminal_ticks, n_nonzero_reward_ticks, reward_abs_sum = 0, 0, 0.0

    for _ in range(a.steps):
        masks = np.asarray(env._masks_view)            # (N_AGENTS, ACTION_MASK_DIM) int8
        h.update(np.ascontiguousarray(masks).tobytes())
        act = sample_masked_actions(masks, rng)
        cont = rng.uniform(-0.5, 0.5, size=(N_AGENTS, AIM_DIM)).astype(np.float32)
        obs, rew, term, _trunc, _info = env.step(act, cont)
        h.update(np.ascontiguousarray(obs).tobytes())
        h.update(np.ascontiguousarray(rew).tobytes())
        h.update(np.ascontiguousarray(term).tobytes())
        n_terminal_ticks += int(np.any(term))
        n_nonzero_reward_ticks += int(np.any(rew != 0))
        reward_abs_sum += float(np.abs(rew).sum())

    env.close()
    print(
        f"steps={a.steps} terminal_ticks={n_terminal_ticks} "
        f"nonzero_reward_ticks={n_nonzero_reward_ticks} "
        f"reward_abs_sum={reward_abs_sum:.6f}",
        file=sys.stderr)
    print(h.hexdigest())


if __name__ == "__main__":
    main()
