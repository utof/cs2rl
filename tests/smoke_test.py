"""Smoke test for C environment — asserts SPS, shapes, no NaN."""
import time
import numpy as np
import pytest
from pathlib import Path


def test_c_env_smoke():
    from c_env.wrapper import make_env

    so  = Path("c_env/dust2_env.so")
    src = Path("c_env/dust2_env.c")
    assert so.exists(), "dust2_env.so missing — run make -C c_env/"
    assert so.stat().st_mtime >= src.stat().st_mtime, \
        "dust2_env.so is older than dust2_env.c — run make -C c_env/"

    env = make_env(seed=42)
    obs, _ = env.reset()

    assert obs.shape == (10, 71), f"Expected obs shape (10, 71), got {obs.shape}"
    assert np.isfinite(obs).all(), "NaN/Inf in initial obs"

    actions = np.zeros((10, 4), dtype=np.int32)
    t0 = time.time()
    for step in range(1000):
        obs, rew, terms, truncs, _ = env.step(actions)
        assert np.isfinite(obs).all(),  f"NaN/Inf in obs at step {step}"
        assert np.isfinite(rew).all(),  f"NaN/Inf in rewards at step {step}"
        if terms.any():
            obs, _ = env.reset()

    sps = 1000 / (time.time() - t0)
    print(f"\nC env SPS: {sps:.0f}")
    assert sps >= 50_000, f"SPS {sps:.0f} below 50_000 threshold"
