"""Smoke test for C environment — asserts SPS, shapes, no NaN."""

import time

import numpy as np
import pytest


@pytest.mark.performance
def test_c_env_smoke():
    import glob as _glob

    from c_env.cs2_env import make_env
    so_files = _glob.glob("src/c_env/binding.cpython-*.so")
    assert so_files, ("binding.cpython-*.so missing — run: "
                      "uv run python setup.py build_ext --inplace")

    env = make_env(seed=42)
    obs, _ = env.reset()

    expected_shape = (env.num_agents, env.single_observation_space.shape[0])
    assert obs.shape == expected_shape, f"Expected obs shape {expected_shape}, got {obs.shape}"
    assert np.isfinite(obs).all(), "NaN/Inf in initial obs"

    actions = np.zeros((env.num_agents, env.single_action_space.shape[0]), dtype=np.int32)
    WARMUP_STEPS = 256
    for step in range(WARMUP_STEPS):
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
    assert sps >= 300_000, f"SPS {sps:.0f} below 300_000 target"
