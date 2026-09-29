#!/usr/bin/env python3
"""Generate BC demonstration .npz files from the scripted bomber expert
(Batch 6 Task 3 — spec §7 schema, R3 carrier-only, R8 obs masking).

One .npz per episode under outputs/demos/ (git-ignored; never commit demos).
Each episode: fresh env on the SIMPLE map, one T agent (0-4) made the bomb
carrier, ScriptedBomber walks it to the bombsite and plants through the real
action interface; we record the per-tick (obs, discrete, continuous, done)
stream FOR THE CARRIER ROW ONLY (spec R3 — the 9 idle agents' transitions
would teach "stand still").

Masking decision (spec R8): the teammate + enemy obs blocks are ZEROED in the
recorded obs (mask-to-zero, NOT randomize-idle-agents). Rationale: during
demo-gen the other 9 agents are frozen at spawn, so those blocks describe a
static world that never occurs at RL time; zeroing makes the clone's walk
depend only on self + goal-direction + global state. Zero is also exactly
what those slots hold for dead/invisible agents, so the masked obs stay
in-distribution. Block boundaries come from spec.obs.OBS_BLOCKS (generated
from cs2_types.h) — NEVER hardcoded, they shifted in Task 2.5 (25/53/93 →
28/56/96) and will shift again.

Determinism note (measured, plan Task 1 RESULT): spawns are deterministic
per-area centroids; seeds only permute which agent gets which of the 5 T
areas. N seeds × 5 carrier slots therefore yields ~5 unique trajectories
duplicated across seeds — expected and accepted (the Task 2.5 bombsite-bearing
obs is the generalization answer, not spawn diversity). `spawn_area` metadata
+ the printed coverage summary make this visible per demo set.

Recording pitfalls handled here (see ScriptedBomber docstring for the yield
contract):
  * env.reset() does NOT populate observations (it only zeroes the buffer;
    compute_observations runs inside env.step) — a priming zero-action step
    runs after the carrier pokes so the first recorded obs is real, not zeros.
  * The generator's yielded (disc, cont) buffers AND env.observations are
    REUSED every tick — everything recorded is .copy()'d.
  * auto_reset=False — a silent mid-capture round reset would splice rounds.
  * Hard in-loop ≤ROUND_TIME cutoff: episodes that don't plant in budget are
    DISCARDED entirely (Gate 0 discipline), counted and reported.

Schema (spec §7) per .npz:
  obs                float32[T, OBS_DIM]  carrier row, teammate/enemy zeroed
  discrete_actions   int64[T, ACTION_DIM] (move, shoot, reload, weapon, use, crouch, jump)
  continuous_actions float32[T, AIM_DIM]  (Δyaw, absolute pitch)
  dones              bool[T]              True only on the plant-completing tick
  + self-identifying metadata: OBS_DIM, ACTION_DIM, AIM_DIM, seed, carrier_idx,
    spawn_area, bombsite_area, tick_count, git_sha, map ("simple_v1").
    train_bc.py must assert these against its live constants before training
    (spec R7 — the --resume path has no shape guard).

Usage (from a worktree, prefix `env PYTHONPATH=<checkout>/src`):
    uv run python -m cs2rl.bc_demos --seeds 10 --out outputs/demos
"""
import argparse
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from cs2rl.c_env.cs2_env import make_env
from cs2rl.env.map import make_simple_map
from cs2rl.env.nav import N_AGENTS, ROUND_TIME, TEAM_SIZE
from cs2rl.eval.scripted_expert import ScriptedBomber, setup_bomb_carrier
from cs2rl.spec.action import ACTION_DIM, AIM_DIM
from cs2rl.spec.obs import OBS_BLOCKS, OBS_DIM

# This file is <repo>/src/cs2rl/bc_demos.py: parents[2] is the checkout root, the
# same root train_bc.DEFAULT_DEMO_DIR is built on (pinned equal by
# tests/test_demo_format.py). Deliberately not imported from train_bc: demo
# generation must not depend on the trainer.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Self-identifying map tag stored in every demo. The demo distribution is only
# valid for BC → PPO on this exact map (per-map obs normalization + geometry);
# train_bc.py should refuse demos whose map tag it does not expect.
MAP_NAME = "simple_v1"


def _git_sha() -> str:
    """C-env git sha at generation time (spec §7). Demos are only trustworthy
    against the env revision that produced them — obs layout is a moving
    target (107→110 happened between Tasks 2 and 3 of this very batch)."""
    return subprocess.run(["git", "rev-parse", "HEAD"],
                          cwd=REPO_ROOT,
                          capture_output=True,
                          text=True,
                          check=True).stdout.strip()


def generate_episode(seed: int, carrier_idx: int, git_sha: str) -> dict | None:
    """Run one seeded expert episode and return the demo arrays + metadata,
    or None if the episode must be discarded (no path / not planted within
    the ROUND_TIME budget).

    A FRESH env is built per episode: Cs2Env.reset(seed=...) ignores its seed
    argument (the C RNG is seeded once at init), so per-episode determinism
    requires constructing with make_env(seed=...) — same pattern as the Gate 0
    measurement (`git show 9b9bf2f:scripts/measure_budget.py`) and
    evaluate_checkpoint.
    """
    env = make_env(seed=seed, map_data=make_simple_map(), auto_reset=False)
    try:
        env.reset(seed=seed)
        setup_bomb_carrier(env, carrier_idx)

        # Priming step: populate env.observations (reset only zeroes the
        # buffer) AND let the carrier pokes (bomb, knife, role bit) land in
        # the obs before the first recorded tick. Costs 1 tick of the round
        # budget; the bomber budget below is reduced accordingly.
        env.step(np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32),
                 np.zeros((N_AGENTS, AIM_DIM), dtype=np.float32))

        agent = env._c_env.game.agents[carrier_idx]
        spawn_area = int(env.map_data.area_ids[agent.area_idx])

        bomber = ScriptedBomber(env, carrier_idx, max_ticks=ROUND_TIME - 1)
        if not bomber.path:
            return None

        tm = slice(*OBS_BLOCKS["teammate"])
        en = slice(*OBS_BLOCKS["enemy"])
        obs_buf, disc_buf, cont_buf = [], [], []
        for disc, cont in bomber.run():
            # Belt-and-braces cutoff on top of the bomber's own max_ticks:
            # nothing recorded past the round budget, ever.
            if len(obs_buf) >= ROUND_TIME:
                break
            obs = env.observations[carrier_idx].astype(np.float32, copy=True)
            obs[tm] = 0.0              # spec R8: frozen idle agents must not
            obs[en] = 0.0              # become walk features (see module doc)
            obs_buf.append(obs)
                                       # .astype(int64) / .astype(float32) copy the reused buffers AND pin
                                       # the on-disk dtypes from spec §7 in one step.
            disc_buf.append(disc[carrier_idx].astype(np.int64))
            cont_buf.append(cont[carrier_idx].astype(np.float32))

        if not bomber.planted:
            return None                # over budget or stuck — discard (Gate 0 discipline)

        tick_count = len(obs_buf)
        dones = np.zeros(tick_count, dtype=bool)
        dones[-1] = True                               # plant completed on the last recorded tick
        return {
            "obs": np.stack(obs_buf),
            "discrete_actions": np.stack(disc_buf),
            "continuous_actions": np.stack(cont_buf),
            "dones": dones,
            "OBS_DIM": OBS_DIM,
            "ACTION_DIM": ACTION_DIM,
            "AIM_DIM": AIM_DIM,
            "seed": seed,
            "carrier_idx": carrier_idx,
            "spawn_area": spawn_area,
            "bombsite_area": int(bomber.path[-1]),
            "tick_count": tick_count,
            "git_sha": git_sha,
            "map": MAP_NAME,
        }
    finally:
        env.close()


def generate_demos(n_seeds: int, out_dir: Path, start_seed: int = 0) -> dict:
    """Generate n_seeds × TEAM_SIZE carrier-slot episodes into out_dir.
    Returns summary stats (also printed): kept/discarded counts, tick
    distribution, per-spawn-area coverage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    git_sha = _git_sha()

    kept, discarded = 0, 0
    tick_counts = []
    spawn_coverage = Counter()
    for seed in range(start_seed, start_seed + n_seeds):
        for carrier_idx in range(TEAM_SIZE):
            demo = generate_episode(seed, carrier_idx, git_sha)
            if demo is None:
                discarded += 1
                print(f"[gen_bc_demos] DISCARD seed={seed} carrier={carrier_idx} "
                      f"(not planted within {ROUND_TIME} ticks)")
                continue
            path = out_dir / f"demo_s{seed:04d}_c{carrier_idx}.npz"
            np.savez(path, **demo)
            kept += 1
            tick_counts.append(demo["tick_count"])
            spawn_coverage[demo["spawn_area"]] += 1

    ticks = np.array(tick_counts) if tick_counts else np.array([0])
    stats = {
        "kept": kept,
        "discarded": discarded,
        "ticks_min": int(ticks.min()),
        "ticks_median": int(np.median(ticks)),
        "ticks_max": int(ticks.max()),
        "spawn_coverage": dict(spawn_coverage),
    }
    print(f"[gen_bc_demos] kept {kept} episodes, discarded {discarded} "
          f"→ {out_dir} (map={MAP_NAME}, OBS_DIM={OBS_DIM}, sha={git_sha[:9]})")
    print(f"[gen_bc_demos] ticks/episode: min {stats['ticks_min']} / "
          f"median {stats['ticks_median']} / max {stats['ticks_max']} (budget {ROUND_TIME})")
    # Deterministic-spawn reality check (plan Task 1 RESULT): expect exactly
    # the 5 T-spawn areas here regardless of seed count.
    print(f"[gen_bc_demos] spawn-area coverage (deterministic centroids — "
          f"~5 unique trajectories expected): {stats['spawn_coverage']}")
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m cs2rl.bc_demos",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("--seeds",
                        type=int,
                        default=10,
                        help="number of seeds (episodes = seeds × 5 carriers)")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "demos")
    args = parser.parse_args(argv)
    stats = generate_demos(args.seeds, args.out, start_seed=args.start_seed)
    # Non-zero exit if everything was discarded — a silent empty demo dir
    # would only fail much later inside train_bc.py.
    return 0 if stats["kept"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
