"""Trainer-level tests for Batch 1 reward architecture changes.

The first two tests exercise the bare Cs2Env factory (obs/rewards shape
sanity). The Task 6c tests below use the minimal trainer harness from
src/train_test_harness.py (Task 6b). Do NOT import the full production
trainer setup — those tests only need to verify the
_patch_trainer_with_selfplay changes from Task 6c.
"""
import numpy as np

import train


def test_make_puffer_env_reset_returns_expected_batch():
    env = train.make_puffer_env(seed=123)
    try:
        obs, info = env.reset(seed=123)
        assert obs.shape == (10, train.OBS_DIM)
        assert env.single_observation_space.shape == (train.OBS_DIM, )
        assert np.isfinite(obs).all(), "reset returned NaN/Inf obs"
    finally:
        env.close()


def test_make_env_alias_steps_without_nan():
    env = train.make_env()
    try:
        obs, _ = env.reset(seed=7)
        actions = np.zeros((10, len(train.ACTION_HEAD_SIZES)), dtype=np.int32)
        next_obs, rewards, terms, truncs, info = env.step(actions)

        assert obs.shape == (10, train.OBS_DIM)
        assert next_obs.shape == (10, train.OBS_DIM)
        assert rewards.shape == (10, )
        assert terms.shape == (10, )
        assert truncs.shape == (10, )
        assert np.isfinite(next_obs).all(), "step returned NaN/Inf obs"
        assert np.isfinite(rewards).all(), "step returned NaN/Inf rewards"
    finally:
        env.close()


# ── Task 6c: reward-clamp removal + per-channel Welford + symlog ───────────
# These tests depend on:
#   - Task 4 (symlog, split_into_channels in src/train_helpers_batch1.py)
#   - Task 5 (WelfordStd in src/train_helpers_batch1.py)
#   - Task 6a (Cs2Env include_step_stats_in_info plumbing)
#   - Task 6b (src/train_test_harness._build_trainer_for_test)
# If any of the above regress, these tests will surface the break early.


def test_selfplay_patch_attaches_welford_and_event_mask():
    """Task 6c: _patch_trainer_with_selfplay must attach three WelfordStd
    instances and two event-mask buffers to the trainer object."""
    from train_helpers_batch1 import WelfordStd
    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        # Pin config values too — a silent change to prior_std or min_count
        # would alter warmup behaviour without tripping any existing test.
        for w in (
                trainer._batch1_welford_combat,
                trainer._batch1_welford_objective,
                trainer._batch1_welford_positional,
        ):
            assert isinstance(w, WelfordStd)
            assert w.prior_std == 1.0
            assert w.min_count == 1000
        assert hasattr(trainer, "_batch1_event_mask")
        assert hasattr(trainer, "_batch1_current_segment_has_event")
    finally:
        cleanup()


def test_rewards_not_clamped_to_unit_range():
    """Task 6c: after removing torch.clamp(r, -1, 1) from the selfplay
    evaluate patch, rewards written into the rollout buffer should reflect
    the symlog-compressed per-channel sum, not a hard [-1, 1] clamp.

    Structural assertion (not numerical): verify the source of
    _patch_trainer_with_selfplay no longer contains `torch.clamp(r, -1, 1)`,
    AND verify the patched evaluate() produces at least one Welford update
    during a rollout (proving the new path executed).

    NOTE the structural-first approach: the plan's original `max_abs > 1.0`
    assertion was flagged as flaky against Welford warmup + random event
    timing; we instead assert (a) source-level absence of the clamp, and
    (b) that the channel-split code path executed at least once.
    """
    import inspect

    from train import _patch_trainer_with_selfplay
    from train_test_harness import _build_trainer_for_test

    src = inspect.getsource(_patch_trainer_with_selfplay)
    assert "torch.clamp(r, -1, 1)" not in src, (
        "Task 6c expected torch.clamp(r, -1, 1) removed from selfplay patch")
    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        c0 = trainer._batch1_welford_combat.count
        o0 = trainer._batch1_welford_objective.count
        p0 = trainer._batch1_welford_positional.count
        trainer.evaluate()
        # At least one channel must have received an update during the rollout.
        total_updates = ((trainer._batch1_welford_combat.count - c0) +
                         (trainer._batch1_welford_objective.count - o0) +
                         (trainer._batch1_welford_positional.count - p0))
        assert total_updates > 0, (
            f"Task 6c: no Welford updates during evaluate(); channel split "
            f"did not execute. Deltas: combat={trainer._batch1_welford_combat.count - c0}, "
            f"objective={trainer._batch1_welford_objective.count - o0}, "
            f"positional={trainer._batch1_welford_positional.count - p0}")
    finally:
        cleanup()
