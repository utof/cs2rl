#!/usr/bin/env python
"""CS2 RL Sim — training entry point.

Usage:
  python train.py --smoke       # sanity check: 1000 steps, no crash, print steps/sec
  python train.py --train       # full PPO self-play training
  python train.py --record      # run 1 episode, save rerun recording
  python train.py --eval --checkpoint checkpoints/latest.zip
"""

import argparse
import time
import numpy as np
from sim import Dust2Env

# ── SECTION: Smoke Test ────────────────────────────────────────────────────

def smoke_test():
    print("[Smoke] Initialising environment...")
    env = Dust2Env()
    obs, _ = env.reset(seed=42)

    assert len(obs) == 10, f"Expected 10 agents, got {len(obs)}"
    for aid, ob in obs.items():
        assert ob.shape == (71,), f"{aid}: shape {ob.shape} != (71,)"
        assert np.isfinite(ob).all(), f"{aid}: NaN in initial obs"

    print("[Smoke] Running 1000 steps...")
    t0 = time.time()
    step_count = 0

    # Cache action spaces to avoid repeated re-seeding overhead
    action_spaces = {aid: env.action_space(aid) for aid in env.possible_agents}

    for step_n in range(1000):
        actions = {aid: action_spaces[aid].sample() for aid in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)

        for aid, ob in obs.items():
            assert np.isfinite(ob).all(), f"NaN at step {step_n} agent {aid}"

        step_count += 1

        if all(terms.get(aid, False) for aid in env.possible_agents):
            obs, _ = env.reset()
            action_spaces = {aid: env.action_space(aid) for aid in env.possible_agents}

    elapsed = time.time() - t0
    sps = step_count / elapsed

    print(f"SMOKE TEST PASSED — {sps:.0f} steps/sec")
    if sps < 500:
        print("WARNING: Very slow (<500 steps/sec). Profile step() with cProfile.")
    elif sps < 5000:
        print("INFO: Acceptable speed. Target is >5000 for training.")


# ── SECTION: CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",      action="store_true")
    parser.add_argument("--train",      action="store_true")
    parser.add_argument("--record",     action="store_true")
    parser.add_argument("--eval",       action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--timesteps",  type=int, default=10_000_000)
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
    elif args.train:
        print("Training not implemented yet — run --smoke first")
    elif args.record:
        print("Recording not implemented yet")
    elif args.eval:
        print("Eval not implemented yet")
    else:
        parser.print_help()
