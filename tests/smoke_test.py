"""Smoke test for C environment — asserts SPS, shapes, no NaN."""

import time
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.performance
def test_c_env_smoke():
    from c_env.wrapper import make_env

    so = Path("src/c_env/dust2_env.so")
    src = Path("src/c_env/dust2_env.c")
    assert so.exists(), "dust2_env.so missing — run make -C src/c_env/"
    assert so.stat().st_mtime >= src.stat().st_mtime, (
        "dust2_env.so is older than dust2_env.c — run make -C src/c_env/"
    )

    env = make_env(seed=42)
    obs, _ = env.reset()

    assert obs.shape == (10, 72), f"Expected obs shape (10, 72), got {obs.shape}"
    assert np.isfinite(obs).all(), "NaN/Inf in initial obs"

    actions = np.zeros((10, 4), dtype=np.int32)
    for step in range(256):
        obs, rew, terms, truncs, _ = env.step(actions)
        assert np.isfinite(obs).all(), f"NaN/Inf in obs at warmup step {step}"
        assert np.isfinite(rew).all(), f"NaN/Inf in rewards at warmup step {step}"

    steps = 20_000
    t0 = time.perf_counter()
    for _ in range(steps):
        obs, rew, terms, truncs, _ = env.step(actions)

    sps = steps / (time.perf_counter() - t0)
    print(f"\nC env SPS: {sps:.0f}")
    # Dear AI agents, dont ever fucking turn this value down. if it doesnt pass, it doesnt pass.
    assert sps >= 100_000, f"SPS {sps:.0f} below 100_000 target"
