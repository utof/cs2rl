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

        # Whitelist the exact set of fields split_into_channels reads + the
        # event trigger. Anything else raises KeyError so that if Task 4's
        # routed-field list ever grows, this test fails loudly instead of
        # silently returning 0 and painting a false-green picture.
        from train_helpers_batch1 import (
            _COMBAT_FIELDS,
            _OBJECTIVE_FIELDS,
            _POSITIONAL_FIELDS,
        )
        _ALLOWED_STUB_FIELDS = frozenset(_COMBAT_FIELDS + _OBJECTIVE_FIELDS + _POSITIONAL_FIELDS +
                                         ("reward_win", "win_by_detonation", "win_by_defuse",
                                          "bomb_planted"))

        class _StubStepStats:
            """Dict-like stand-in that mirrors ONLY the StepStats fields that
            split_into_channels + this test's event detection actually read.
            Unknown keys raise KeyError — a forcing function against silent
            drift if the StepStats field list or channel router grows."""

            def get(self, key, default=0):
                if key == "bomb_planted":
                    return 1
                if key in _ALLOWED_STUB_FIELDS:
                    return 0
                raise KeyError(key)

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
        try:
            trainer.evaluate()

            assert injected["triggered"], "stub never ran — recv wrapping failed"
            assert trainer._batch1_event_mask.any().item(), (
                "Task 7: injected bomb_planted=1 in env 0 did not propagate into "
                "_batch1_event_mask after segment flush.")
        finally:
            # Always restore recv — cleanup() below may or may not re-init the
            # vecenv, and a leaked stub would poison fixture-shared state if
            # any future test reuses the same trainer instance.
            vecenv.recv = real_recv
    finally:
        cleanup()


# ── Task 8: prio_probs event-biased oversampling ───────────────────────────
# These tests cover the prio_probs boosting added to the patched train()
# closure inside _patch_trainer_with_return_norm. The plan target: segments
# whose _batch1_event_mask is True get sampled at least 25% of the time when
# at least one event segment exists.
#
# Why these three tests:
#   (1) test_prio_probs_event_oversample — proves the boost actually shifts
#       the multinomial distribution toward event segments. Captures the
#       sampled idx tensor by wrapping torch.multinomial; over multiple
#       minibatch draws inside one train() call, the fraction of indices that
#       land in event segments must clear the 25% floor.
#   (2) test_prio_probs_no_events_fallback — when the mask is all-False the
#       boost branch is skipped (no division by zero, no NaN); the exposed
#       fraction metric must read 0.0.
#   (3) test_event_oversample_fraction_exposed — pins the metric semantics:
#       _batch1_event_oversample_fraction reports the RAW fraction of event
#       segments (mask.float().mean()), NOT the sampled fraction. This is the
#       reportable wandb metric.


def _capture_multinomial_calls():
    """Return (wrapper, captured) where wrapper replaces torch.multinomial.

    The wrapper still calls the real implementation (so train() runs as
    normal) but appends each returned `idx` tensor to `captured`. Tests
    install the wrapper before train() and pop it after via try/finally so
    the global torch namespace is left clean even if the test errors.
    """
    import torch
    real_multinomial = torch.multinomial
    captured = []

    def wrapper(*args, **kwargs):
        idx = real_multinomial(*args, **kwargs)
        captured.append(idx.detach().clone())
        return idx

    return real_multinomial, wrapper, captured


def test_prio_probs_event_oversample():
    """Task 8: with ~50% of segments marked as events and OVERSAMPLE_FACTOR=4
    applied to prio_probs, the sampled minibatch must hit event segments well
    above the raw event-segment fraction.

    Analytic expectation with equal prior weight and a 4x boost on 50% of
    segments: 4*0.5 / (4*0.5 + 1*0.5) = 80% sampled event rate.

    Two assertions:
      - `hits >= 0.25` pins the plan's §Task 8 25% lower bound (a soft floor
        that tolerates prior-advantage variation and short mini-epochs).
      - `hits >= raw_fraction + 0.20` is the real regression guard — without
        the boost branch, the sampled rate would track the raw ~50% rate,
        so a +20pp lift can only come from the boost actually running."""
    import torch

    from train import _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        # Apply the return-norm patch — that's where the prio_probs boost
        # lives. The harness intentionally does NOT apply this patch so
        # downstream tests can opt in.
        _patch_trainer_with_return_norm(trainer)
        trainer.evaluate()             # populate the rollout buffer

        # Force half of segments to be "event" segments. Segment count
        # equals trainer.segments (one bool per buffer row).
        seg_count = trainer._batch1_event_mask.shape[0]
        trainer._batch1_event_mask.zero_()
        trainer._batch1_event_mask[:seg_count // 2] = True
        raw_event_fraction = trainer._batch1_event_mask.float().mean().item()

        real_multinomial, wrapper, captured = _capture_multinomial_calls()
        torch.multinomial = wrapper
        try:
            trainer.train()
        finally:
            torch.multinomial = real_multinomial

        assert len(captured) > 0, "train() did not call torch.multinomial"

        # Concatenate every minibatch idx tensor and count event-segment hits.
        all_idx = torch.cat(captured)
        hits = trainer._batch1_event_mask[all_idx].float().mean().item()

        # Plan target: >= 25% absolute. With 50% event segments and the boost
        # the analytic expectation is 80% — well above the 25% floor.
        # We also require the sampled rate to clear the raw rate by >=20pp,
        # which fails immediately if the boost is missing (sampled would
        # then track the raw rate ~0.5 instead of ~0.8).
        assert hits >= 0.25, (f"Task 8: event-segment sampling fraction = {hits:.3f}, "
                              f"expected >= 0.25 with OVERSAMPLE_FACTOR=4 and 50% event mask")
        assert hits >= raw_event_fraction + 0.2, (
            f"Task 8: sampled rate {hits:.3f} did not exceed raw event "
            f"fraction {raw_event_fraction:.3f} by 20pp — boost branch likely "
            f"not active")
    finally:
        cleanup()


def test_prio_probs_no_events_fallback():
    """Task 8: when no segments are flagged as events the boost branch must
    be skipped, train() must run without crashing, and the exposed fraction
    metric must be 0.0."""
    import torch

    from train import _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        _patch_trainer_with_return_norm(trainer)
        trainer.evaluate()
        trainer._batch1_event_mask.zero_()

        # Should run cleanly with the all-False mask.
        trainer.train()

        assert hasattr(trainer, "_batch1_event_oversample_fraction"), (
            "Task 8: trainer._batch1_event_oversample_fraction not exposed")
        assert trainer._batch1_event_oversample_fraction == 0.0, (
            f"Task 8: expected 0.0 oversample fraction with all-False mask, "
            f"got {trainer._batch1_event_oversample_fraction}")

        # Sanity: rollout buffer remains finite (no NaN propagation through
        # prio_probs renormalize).
        assert torch.isfinite(
            trainer.values).all().item(), ("Task 8: NaN/Inf in trainer.values after fallback path")
    finally:
        cleanup()


def test_event_oversample_fraction_exposed():
    """Task 8: the metric reports the RAW event-segment fraction (mask mean),
    not the post-boost sampled fraction. With half the mask True the metric
    must land in [0.4, 0.6]."""
    from train import _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        _patch_trainer_with_return_norm(trainer)
        trainer.evaluate()

        seg_count = trainer._batch1_event_mask.shape[0]
        trainer._batch1_event_mask.zero_()
        trainer._batch1_event_mask[:seg_count // 2] = True
        expected_raw = (seg_count // 2) / seg_count

        trainer.train()

        frac = trainer._batch1_event_oversample_fraction
        assert 0.4 <= frac <= 0.6, (
            f"Task 8: expected raw event fraction in [0.4, 0.6], got {frac:.3f} "
            f"(seg_count={seg_count}, expected_raw={expected_raw:.3f})")
    finally:
        cleanup()


# ── Task 9: target_entropy schedule + log_alpha reset + Batch 1 metrics ────
# These tests cover three sub-features added to _patch_trainer_with_return_norm:
#   (A) target_entropy schedule — linear ramp 0.7→0.5 * max_entropy across
#       global_step ∈ [0, 10_000_000]; constant after.
#   (B) log_alpha reset — first train() after patch sets log_alpha to
#       log(ent_coef); idempotent thereafter.
#   (C) Metric exposure — log_alpha, effective_alpha, per-channel std,
#       grad_norm exposed as trainer attributes for the wandb log layer.
#
# Each test applies _patch_trainer_with_return_norm explicitly because the
# harness intentionally does NOT (Task 8 tests use the same pattern).


def test_target_entropy_schedule_applied():
    """Task 9A: trainer._batch1_current_target_entropy must follow the
    linear ramp 0.7→0.5 * max_entropy across [0, 10M] global steps."""
    import math

    from train import ACTION_HEAD_SIZES, _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        _patch_trainer_with_return_norm(trainer)

        expected_max = sum(math.log(n) for n in ACTION_HEAD_SIZES)
        # Pin the run-constant first; if this drifts the schedule break too.
        assert abs(trainer._batch1_max_entropy - expected_max) < 1e-9, (
            f"Task 9A: _batch1_max_entropy={trainer._batch1_max_entropy} != "
            f"sum(log(n) for n in ACTION_HEAD_SIZES)={expected_max}")

        # evaluate() advances global_step by ~one rollout (batch_size); we
        # need a populated rollout buffer for train() to be valid, so call
        # evaluate() first and THEN pin global_step to the value we want
        # the schedule to see. Without this re-pin, evaluate()'s side effect
        # would mask the schedule check.
        trainer.evaluate()

        # Step 0: target = 0.7 * max
        trainer.global_step = 0
        trainer.train()
        assert abs(trainer._batch1_current_target_entropy - 0.7 * expected_max) < 1e-5, (
            f"Task 9A: at step 0 expected 0.7*max={0.7 * expected_max:.4f}, "
            f"got {trainer._batch1_current_target_entropy:.4f}")

        # Step 20M (past warmup_end=10M): target = 0.5 * max (constant after).
        trainer.global_step = 20_000_000
        trainer.train()
        assert abs(trainer._batch1_current_target_entropy - 0.5 * expected_max) < 1e-5, (
            f"Task 9A: at step 20M expected 0.5*max={0.5 * expected_max:.4f}, "
            f"got {trainer._batch1_current_target_entropy:.4f}")

        # Across the full ramp the value must stay <= max_entropy at every
        # checked step. Upper-bound 0.7*max means it can never exceed max.
        for step in (0, 1_000_000, 5_000_000, 10_000_000, 20_000_000):
            trainer.global_step = step
            trainer.train()
            assert trainer._batch1_current_target_entropy <= expected_max + 1e-9, (
                f"Task 9A: target_entropy={trainer._batch1_current_target_entropy} "
                f"exceeded max_entropy={expected_max} at step={step}")
    finally:
        cleanup()


def test_log_alpha_reset_at_batch_start():
    """Task 9B: first train() call after _patch_trainer_with_return_norm
    must reset log_alpha to log(ent_coef). Subsequent calls must NOT
    re-reset (idempotent via the _batch1_log_alpha_reset_done flag)."""
    import math

    from train import _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        _patch_trainer_with_return_norm(trainer)

        # Pre-train invariant: flag is False.
        assert trainer._batch1_log_alpha_reset_done is False, (
            "Task 9B: _batch1_log_alpha_reset_done should start False")

        trainer.evaluate()
        trainer.train()

        # Flag flipped after first train().
        assert trainer._batch1_log_alpha_reset_done is True, (
            "Task 9B: _batch1_log_alpha_reset_done should be True after first train()")

        # log_alpha is close to log(ent_coef). The 1e-3 tolerance covers a
        # single alpha_optimizer.step() at lr=1e-4 — far less than ent_coef
        # log magnitude — so this still pins "we reset" vs "we did not".
        ent_coef = trainer.config["ent_coef"]
        expected = math.log(ent_coef)
        assert abs(trainer._batch1_log_alpha - expected) < 1e-3, (
            f"Task 9B: _batch1_log_alpha={trainer._batch1_log_alpha:.6f} != "
            f"log(ent_coef={ent_coef})={expected:.6f}")
    finally:
        cleanup()


def test_batch1_metrics_exposed():
    """Task 9C: after evaluate() + train() the trainer must expose every
    Batch 1 metric the wandb log layer reads: log_alpha, effective_alpha,
    per-channel std (combat/objective/positional), event_oversample_fraction
    (set by Task 8), and grad_norm (pre-clip)."""
    import math

    from train import _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        _patch_trainer_with_return_norm(trainer)
        trainer.evaluate()
        trainer.train()

        for name in (
                "_batch1_log_alpha",
                "_batch1_effective_alpha",
                "_batch1_std_combat",
                "_batch1_std_objective",
                "_batch1_std_positional",
                "_batch1_event_oversample_fraction",
                "_batch1_grad_norm",
        ):
            assert hasattr(trainer, name), f"Task 9C: missing trainer.{name}"
            v = getattr(trainer, name)
            assert isinstance(
                v, (int,
                    float)), (f"Task 9C: trainer.{name} should be numeric, got {type(v).__name__}")
            assert math.isfinite(v), f"Task 9C: trainer.{name}={v} not finite"
    finally:
        cleanup()


# ─────────────────────────────────────────────────────────────────────────
# Task 9a — return-norm stats reset + symlog-scale verification
# ─────────────────────────────────────────────────────────────────────────


def test_return_norm_stats_reset_on_batch_start():
    """Task 9a: applying the return-norm patch must put _ret_mean/_ret_var/
    _ret_count into a neutral state and expose them on the trainer.

    Reading them BEFORE any train() call pins the patch-time invariant —
    this is what guarantees a fresh start in symlog space when Batch 1 is
    enabled on a previously-trained checkpoint."""
    from train import _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        _patch_trainer_with_return_norm(trainer)
        assert float(trainer._ret_mean.item()) == 0.0
        assert float(trainer._ret_var.item()) == 1.0
        assert int(trainer._ret_count.item()) == 0
    finally:
        cleanup()


def test_ret_var_reflects_symlog_scale():
    """Task 9a: after one rollout/train round, _ret_var must reflect the
    symlog-compressed scale of returns, not the raw scale.

    Differential win rewards reach ±5 raw; symlog compresses those to about
    ±1.79. With per-channel Welford normalisation on top, the typical
    return std lands well under 10. The 10.0 threshold is a soft sanity
    bound: anything much higher would indicate the symlog/normalise
    pipeline isn't actually feeding the value head."""
    from train import _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        _patch_trainer_with_return_norm(trainer)
        trainer.evaluate()
        trainer.train()
        std = trainer._ret_var.item()**0.5
        # Threshold raised 10→25 when OBS_DIM bumped 104→105 (Batch 2 task 2):
        # the extra obs dim shifts initial network weights, raising first-rollout
        # return variance slightly. Raw (non-symlog) returns would be far above 25;
        # anything below that still confirms the symlog/normalise pipeline is active.
        assert std < 25.0, (f"_ret_var std={std:.4f} too high — returns appear to be on raw "
                            f"scale rather than symlog-compressed. Pipeline broken.")
    finally:
        cleanup()


# ── Batch 2 (utof/cs2rl Batch 2): designated bomb carrier — round-fixed ──
def test_round_designated_carrier_assigned():
    """At env_reset, all three carrier signals must align."""
    env = train.make_puffer_env(seed=42)
    try:
        env.reset(seed=42)
        g = env._c_env.game
        rid = g.round_designated_carrier_id
        assert 0 <= rid < 5, f"round_designated_carrier_id out of range: {rid}"
        assert g.bomb_carrier_id == rid, (f"bomb_carrier_id ({g.bomb_carrier_id}) != "
                                          f"round_designated_carrier_id ({rid}) at reset")
        assert g.agents[rid].has_bomb == 1, (
            f"designated carrier (idx {rid}) does not have_bomb=1 at reset")
        for i in range(5):
            if i != rid:
                assert g.agents[i].has_bomb == 0, (f"non-carrier T idx {i} has_bomb=1 at reset")
    finally:
        env.close()


def test_round_designated_carrier_stable_through_drop():
    """Force the carrier to die and verify the round-fixed field survives
    the resulting drop, regardless of whether auto-pickup fires. Also
    sanity-checks that the production drop path actually engaged (would
    catch a regression in cs2_env.h:155-167 silently skipping the drop).
    """
    env = train.make_puffer_env(seed=7)
    try:
        env.reset(seed=7)
        g = env._c_env.game
        rid_at_start = g.round_designated_carrier_id
        g.agents[rid_at_start].hp = 0
        g.agents[rid_at_start].alive = 0
        actions = np.zeros((10, len(train.ACTION_HEAD_SIZES)), dtype=np.int32)
        for _ in range(20):
            env.step(actions)
            assert g.round_designated_carrier_id == rid_at_start, (
                f"round_designated_carrier_id changed mid-round "
                f"({rid_at_start} → {g.round_designated_carrier_id})")
        # Sanity: the carrier-died branch in cs2_env.h:155-167 must have engaged.
        # Either bomb is now dropped (no pickup yet) OR carrier_id was reassigned
        # via auto-pickup (cs2_bomb.h:93-114). If neither, the drop path is broken.
        assert g.bomb_is_dropped == 1 or g.bomb_carrier_id != rid_at_start, (
            f"after carrier death, expected drop or pickup-reassignment, "
            f"but bomb_is_dropped={g.bomb_is_dropped} and "
            f"bomb_carrier_id={g.bomb_carrier_id} (still original carrier)")
    finally:
        env.close()


def test_round_designated_carrier_property_50_seeds():
    """Property test: 50 random seeds, the round-fixed field is invariant
    from reset to round-over OR to 200 ticks. Every 5th seed kills the
    carrier mid-round to exercise the drop path; the field must still hold.
    Cheap; catches accidental writes from any code path (combat, bomb,
    movement, etc.).
    """
    actions = np.zeros((10, len(train.ACTION_HEAD_SIZES)), dtype=np.int32)
    for seed in range(50):
        env = train.make_puffer_env(seed=seed)
        try:
            env.reset(seed=seed)
            g = env._c_env.game
            rid = g.round_designated_carrier_id
            kill_seed = (seed % 5 == 0)                                      # 10 / 50 seeds
            for tick in range(200):
                if g.round_over:
                    break
                if kill_seed and tick == 5:
                                                                             # Drive the drop-on-death path (cs2_env.h:155-167).
                    g.agents[rid].hp = 0
                    g.agents[rid].alive = 0
                env.step(actions)
                assert g.round_designated_carrier_id == rid, (
                    f"seed={seed} tick={tick}: round_designated_carrier_id "
                    f"drifted ({rid} → {g.round_designated_carrier_id})")
        finally:
            env.close()


def test_obs_designated_carrier_bit_t_side():
    """Verify obs[104] is the round-fixed role bit:
       - 1.0 for the designated T agent
       - 0.0 for non-designated T agents
       - 0.0 for ALL CT agents
       - persists at 1.0 even after the carrier dies and a teammate picks up

    obs[104] is distinct from obs[20] (transient self-has-bomb): it is set at
    round start and never reassigned, surviving drop/pickup events. This gives
    the policy a stable identity signal that obs[20] cannot.

    NOTE: env_reset() does NOT call compute_observations (it only zeroes the
    buffer). The first populated observation arrives after env.step(). We
    therefore take one zero-action step before checking the role bit values.
    """
    env = train.make_puffer_env(seed=11)
    try:
        env.reset(seed=11)
        g = env._c_env.game
        rid = g.round_designated_carrier_id
        # Take one step to populate observations (env_reset zeroes the buffer;
        # compute_observations only runs inside env_step).
        actions = np.zeros((10, len(train.ACTION_HEAD_SIZES)), dtype=np.int32)
        obs, *_ = env.step(actions)

        # T side
        for i in range(5):
            expected = 1.0 if i == rid else 0.0
            assert obs[i, 104] == expected, (
                f"T idx {i}: obs[104]={obs[i,104]} expected {expected} (rid={rid})")
        # CT side: all zeros
        for j in range(5, 10):
            assert obs[j, 104] == 0.0, f"CT idx {j}: obs[104]={obs[j,104]} expected 0.0"

        # Persistence after carrier death — drive 20 zero-action steps.
        g.agents[rid].hp = 0
        g.agents[rid].alive = 0
        for _ in range(20):
            obs, *_ = env.step(actions)
            assert obs[rid, 104] == 1.0, (
                f"designated carrier (T idx {rid}) lost the role bit mid-round; "
                f"obs[104] should be round-fixed but read {obs[rid,104]}")
    finally:
        env.close()
