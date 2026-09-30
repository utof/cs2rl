"""Smoke test for C environment — asserts SPS, shapes, no NaN."""

import time

import numpy as np
import pytest


@pytest.mark.performance
def test_c_env_smoke():
    from cs2rl.env.c import SOURCE_DIR
    from cs2rl.env.c.cs2_env import make_env

    # From the package, never the cwd: a relative glob found nothing from any other directory.
    so_files = list(SOURCE_DIR.glob("binding.cpython-*.so"))
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


def test_batch1_smoke_runs_without_nan():
    """Batch 1 integration smoke (plan §Task 10).

    Wires the full Batch 1 stack (selfplay patch, return-norm patch, symlog
    rewards, Welford normalization, event-mask aggregation, prio_probs
    boost, target_entropy schedule, log_alpha reset, metric exposure) and
    runs one evaluate()+train() round. Catches the failure modes that
    individual unit tests can't: NaN propagation through the rollout
    buffer, attribute pre-init missing on real trainers, and stat
    interactions across the patches.

    Deliberately tiny (num_envs=32, Serial backend) so it stays under ~30s
    and can run in CI as a per-PR gate.
    """
    import math

    from tests._helpers.trainer_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:

        # ── Run one full round ──────────────────────────────────────────
        trainer.evaluate()
        trainer.train()

        # ── No NaN/Inf in the rollout reward buffer ─────────────────────
        rewards = trainer.rewards.detach().cpu()
        import torch
        assert torch.isfinite(rewards).all(), (
            "Batch 1 smoke: trainer.rewards contains NaN/Inf — symlog/Welford"
            " pipeline likely produced a divergent value")

        # ── All Batch 1 metrics populated and finite ────────────────────
        for name in (
                "_batch1_max_entropy",
                "_batch1_current_target_entropy",
                "_batch1_log_alpha",
                "_batch1_effective_alpha",
                "_batch1_std_combat",
                "_batch1_std_objective",
                "_batch1_std_positional",
                "_batch1_event_oversample_fraction",
                "_batch1_grad_norm",
        ):
            assert hasattr(trainer, name), f"Batch 1 smoke: missing {name}"
            v = getattr(trainer, name)
            assert math.isfinite(v), f"Batch 1 smoke: {name}={v} not finite"

        # ── Channel stds non-negative ───────────────────────────────────
        # During warmup (count < min_count=1000) WelfordStd returns the
        # prior_std=1.0 floor. With ~num_envs * bptt_horizon ticks per
        # evaluate() round (~2048 here) the warmup boundary is crossed and
        # the actual running std takes over. That value can be small (most
        # channels are zero on most ticks), so the only universal invariant
        # is non-negative.
        assert trainer._batch1_std_combat >= 0.0
        assert trainer._batch1_std_objective >= 0.0
        assert trainer._batch1_std_positional >= 0.0
    finally:
        cleanup()
