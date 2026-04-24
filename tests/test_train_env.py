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


# ── Task 7: segment-level event-mask aggregation ──────────────────────────
# These tests exercise the two code paths added by Task 7:
#   (a) per-tick OR of bomb_planted into _batch1_current_segment_has_event
#       (one bool per agent row), gated on step_stats being present.
#   (b) flush of _batch1_current_segment_has_event → _batch1_event_mask at
#       the segment boundary (every bptt_horizon ticks), keyed on the OLD
#       ep_indices (before they get re-assigned for the next segment).
#
# Why two tests: (a) is a deterministic unit-level check that directly writes
# the live accumulator and verifies the flush happens at the boundary — no
# dependence on random bomb-plant timing. (b) is an integration check that
# stubs the recv() payload so bomb_planted=1 appears in step_stats, verifying
# the detection branch. We need both because the live accumulator could be
# correct without detection, and detection could be correct without a flush.


def test_event_mask_flushed_and_reset_at_segment_boundary():
    """Task 7: pre-set the live accumulator for one agent row to True, then
    run a full evaluate() round. After the segment closes, the corresponding
    segment row in _batch1_event_mask must be True and the live accumulator
    must be reset back to False.

    Independent of bomb_planted plumbing — this exercises only the flush.
    """

    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        # Force accumulator True for agent row 0 BEFORE evaluate() runs.
        # We cache the segment index that row 0 will flush into — this is
        # the value of trainer.ep_indices[0] at the moment the boundary hits,
        # but at start-of-rollout it's trivially 0 (free_idx starts at 0 and
        # ep_indices was initialised to arange(total_agents)).
        pre_seg_idx = int(trainer.ep_indices[0].item())
        trainer._batch1_current_segment_has_event.zero_()
        trainer._batch1_current_segment_has_event[0] = True
        trainer._batch1_event_mask.zero_()

        trainer.evaluate()

        # After one full evaluate() round all 320 agent rows flushed once, so
        # the pre_seg_idx slot must reflect our True write.
        assert bool(trainer._batch1_event_mask[pre_seg_idx].item()), (
            "Task 7: live accumulator True for row 0 did not flush into "
            f"_batch1_event_mask[{pre_seg_idx}] at segment boundary.")
        # Live accumulator for row 0 must have been reset (it would only come
        # back True if a real bomb_planted event happened for env 0 during
        # the rollout, which is possible but not guaranteed — so we assert a
        # weaker property: at least one row is False, proving the reset path
        # is not a no-op. With a 64-tick rollout and random actions most envs
        # see zero plants, so most rows will be False.)
        assert not trainer._batch1_current_segment_has_event.all().item(), (
            "Task 7: all live-accumulator rows True after evaluate() — the "
            "reset at segment boundary appears to be missing.")
    finally:
        cleanup()


def test_event_mask_detects_injected_bomb_planted():
    """Task 7: when step_stats['bomb_planted']==1 arrives for any env during
    the rollout, the event_mask must contain at least one True after the
    segment flushes.

    Strategy: wrap vecenv.recv() so that on the first call, env 0's
    step_stats is a dict substitute with bomb_planted=1. Real step_stats are
    StepStatsView proxies, but the evaluate() loop uses plain .get('bomb_planted')
    which works on any Mapping. This avoids driving the C env to plant a bomb.
    """

    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        # Baseline — ensure no stale event bits.
        trainer._batch1_event_mask.zero_()
        trainer._batch1_current_segment_has_event.zero_()

        vecenv = trainer.vecenv
        real_recv = vecenv.recv
        injected = {"triggered": False}

        class _StubStepStats:
            """Dict-like stand-in. evaluate() calls ss.get('bomb_planted', 0)
            which works on plain dicts. Other fields must also be gettable
            because split_into_channels reads several StepStats fields; we
            default everything else to 0 via a permissive .get."""

            def get(self, key, default=0):
                if key == "bomb_planted":
                    return 1
                return default

            def __getitem__(self, key):
                return self.get(key, 0)

            ndim = 0

        def stub_recv():
            o, r, d, t, infos, ids, m = real_recv()
            if not injected["triggered"] and len(infos) > 0:
                # Replace env 0's info with a bomb_planted=1 payload. Leave
                # other envs untouched so the rest of the rollout is normal.
                infos = list(infos)
                infos[0] = {"step_stats": _StubStepStats()}
                injected["triggered"] = True
            return o, r, d, t, infos, ids, m

        vecenv.recv = stub_recv
        trainer.evaluate()

        assert injected["triggered"], "stub never ran — recv wrapping failed"
        assert trainer._batch1_event_mask.any().item(), (
            "Task 7: injected bomb_planted=1 in env 0 did not propagate into "
            "_batch1_event_mask after segment flush.")
    finally:
        cleanup()
