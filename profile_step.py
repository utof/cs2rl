"""Isolated profiler for Dust2Env.step() — finds per-function bottlenecks.

Run with:
    uv run python profile_step.py
"""
import cProfile
import pstats
import io
import time
import numpy as np
from sim import Dust2Env

def run_steps(n=2000):
    env = Dust2Env()
    obs, _ = env.reset(seed=42)
    action_spaces = {aid: env.action_space(aid) for aid in env.possible_agents}
    for _ in range(n):
        actions = {aid: action_spaces[aid].sample() for aid in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        if all(terms.get(aid, False) for aid in env.possible_agents):
            obs, _ = env.reset()
            action_spaces = {aid: env.action_space(aid) for aid in env.possible_agents}

if __name__ == "__main__":
    print("Warming up environment (vis matrix build/load)...")
    _warmup = Dust2Env()
    del _warmup

    print("Profiling 2000 steps...")
    pr = cProfile.Profile()
    pr.enable()
    run_steps(2000)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(40)
    print(s.getvalue())

    # Also print by tottime (self time) to find leaf bottlenecks
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
    ps2.print_stats(30)
    print("=== BY SELF TIME ===")
    print(s2.getvalue())
