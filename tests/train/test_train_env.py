"""Trainer-level tests for Batch 1 reward architecture changes.

The first two tests exercise the bare Cs2Env factory (obs/rewards shape
sanity). The Task 6c tests below use the minimal trainer harness from
src/cs2rl/train_test_harness.py (Task 6b). Do NOT import the full production
trainer setup — those tests only need to verify the Task 6c changes to the
self-play state (``Cs2PuffeRL._init_selfplay`` / ``evaluate``, gh#168 W2b).
"""
import numpy as np

from cs2rl import policy as policy_mod
from cs2rl.env.c.cs2_env import make_env
from cs2rl.spec import action as spec_action
from cs2rl.spec import obs as spec_obs
from cs2rl.train import envs as train_envs


def test_make_env_reset_returns_expected_batch():
    env = make_env(seed=123)
    try:
        obs, info = env.reset(seed=123)
        assert obs.shape == (10, spec_obs.OBS_DIM)
        assert env.single_observation_space.shape == (spec_obs.OBS_DIM, )
        assert np.isfinite(obs).all(), "reset returned NaN/Inf obs"
    finally:
        env.close()


def test_make_env_alias_steps_without_nan():
    env = train_envs.make_env()
    try:
        obs, _ = env.reset(seed=7)
        actions = np.zeros((10, len(spec_action.ACTION_HEAD_SIZES)), dtype=np.int32)
        next_obs, rewards, terms, truncs, info = env.step(actions)

        assert obs.shape == (10, spec_obs.OBS_DIM)
        assert next_obs.shape == (10, spec_obs.OBS_DIM)
        assert rewards.shape == (10, )
        assert terms.shape == (10, )
        assert truncs.shape == (10, )
        assert np.isfinite(next_obs).all(), "step returned NaN/Inf obs"
        assert np.isfinite(rewards).all(), "step returned NaN/Inf rewards"
    finally:
        env.close()


def test_compute_game_metrics_surfaces_new_keys_without_backfilling_plant_tick():
    from cs2rl.train.metrics import compute_game_metrics
    old = {
        "environment/winner_t": 0.4,
        "environment/winner_ct": 0.6,
        "environment/timed_out": 0.2,
        "environment/kills_t": 1.0,
        "environment/kills_ct": 2.0,
        "environment/bomb_planted": 0.3,
        "environment/bomb_defused": 0.1,
        "environment/round_length": 500.0,
        "environment/reward_win": 0.0,
        "environment/reward_kills": 0.5,
        "environment/reward_deaths": -0.2,
        "environment/reward_bomb": 1.1,
        "environment/reward_pbrs": 0.3,
        "environment/reward_shots": -0.01,
        "environment/reward_survival": 0.2,
        "environment/reward_inaction": -0.05,
    }
    out = compute_game_metrics(old)
    assert out["game/win_rate_t"] == 0.4
    assert out["game/kills_per_episode"] == 3.0
    assert out["game/kills_t"] == 1.0
    assert out["game/kills_ct"] == 2.0
    assert out["game/defuse_rate"] == 0.1
    assert "game/reward/win" not in out                # R0-A (#128): dropped, nets ~0 by identity
    assert out["game/reward/kills"] == 0.5
    assert out["game/reward/deaths"] == -0.2
    assert out["game/reward/bomb"] == 1.1
    assert out["game/reward/pbrs"] == 0.3
    assert out["game/reward/shots"] == -0.01
    assert out["game/reward/survival"] == 0.2
    assert out["game/reward/inaction"] == -0.05
    assert "game/plant_tick" not in out
    assert "game/win_by_detonation" not in out
    assert "game/win_by_defuse" not in out

    new = dict(old)
    new["environment/plant_tick"] = 40.0
    new["environment/win_by_detonation"] = 0.2
    new["environment/win_by_defuse"] = 0.1
    out2 = compute_game_metrics(new)
    assert out2["game/plant_tick"] == 40.0
    assert out2["game/win_by_detonation"] == 0.2
    assert out2["game/win_by_defuse"] == 0.1


def test_make_env_default_keeps_step_stats_off():
    import inspect
    sig = inspect.signature(make_env)
    assert sig.parameters["include_step_stats_in_info"].default is False
    env = make_env(seed=0)
    try:
        assert env._include_step_stats_in_info is False
    finally:
        env.close()


# ── Task 6c: reward-clamp removal + per-channel Welford + symlog ───────────
# These tests depend on:
#   - Task 4 (symlog, split_into_channels in cs2rl.train.rewards)
#   - Task 5 (WelfordStd in cs2rl.train.rewards)
#   - Task 6a (Cs2Env include_step_stats_in_info plumbing)
#   - Task 6b (tests._helpers.trainer_harness._build_trainer_for_test)
# If any of the above regress, these tests will surface the break early.


def test_selfplay_init_attaches_welford_and_event_mask():
    """Task 6c: Cs2PuffeRL._init_selfplay must attach three WelfordStd
    instances and two event-mask buffers to the trainer object."""
    from cs2rl.train.rewards import WelfordStd
    from tests._helpers.trainer_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        # Pin config values too — a silent change to prior_std or min_count
        # would alter warmup behaviour without tripping any existing test.
        for w in (
                trainer._welford_combat,
                trainer._welford_objective,
                trainer._welford_positional,
        ):
            assert isinstance(w, WelfordStd)
            assert w.prior_std == 1.0
            assert w.min_count == 1000
        assert hasattr(trainer, "_event_mask")
        assert hasattr(trainer, "_current_segment_has_event")
    finally:
        cleanup()


def test_rewards_not_clamped_to_unit_range():
    """Task 6c: after removing torch.clamp(r, -1, 1) from the selfplay
    evaluate patch, rewards written into the rollout buffer should reflect
    the symlog-compressed per-channel sum, not a hard [-1, 1] clamp.

    Structural assertion (not numerical): verify the source of
    Cs2PuffeRL (evaluate() included) no longer contains `torch.clamp(r, -1, 1)`,
    AND verify evaluate() produces at least one Welford update
    during a rollout (proving the new path executed).

    NOTE the structural-first approach: the plan's original `max_abs > 1.0`
    assertion was flagged as flaky against Welford warmup + random event
    timing; we instead assert (a) source-level absence of the clamp, and
    (b) that the channel-split code path executed at least once.
    """
    import inspect

    from cs2rl.train.trainer import Cs2PuffeRL
    from tests._helpers.trainer_harness import _build_trainer_for_test

    src = inspect.getsource(Cs2PuffeRL)
    assert "torch.clamp(r, -1, 1)" not in src, (
        "Task 6c expected torch.clamp(r, -1, 1) removed from the self-play evaluate body")
    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        c0 = trainer._welford_combat.count
        o0 = trainer._welford_objective.count
        p0 = trainer._welford_positional.count
        trainer.evaluate()
        # At least one channel must have received an update during the rollout.
        total_updates = ((trainer._welford_combat.count - c0) +
                         (trainer._welford_objective.count - o0) +
                         (trainer._welford_positional.count - p0))
        assert total_updates > 0, (
            f"Task 6c: no Welford updates during evaluate(); channel split "
            f"did not execute. Deltas: combat={trainer._welford_combat.count - c0}, "
            f"objective={trainer._welford_objective.count - o0}, "
            f"positional={trainer._welford_positional.count - p0}")
    finally:
        cleanup()


# ── Task 7: segment-level event-mask aggregation ──────────────────────────
# These tests exercise the two code paths added by Task 7:
#   (a) per-tick OR of bomb_planted into _current_segment_has_event
#       (one bool per agent row), gated on step_stats being present.
#   (b) flush of _current_segment_has_event → _event_mask at
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
    segment row in _event_mask must be True and the live accumulator
    must be reset back to False.

    Independent of bomb_planted plumbing — this exercises only the flush.
    """

    from tests._helpers.trainer_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        # Force accumulator True for agent row 0 BEFORE evaluate() runs.
        # We cache the segment index that row 0 will flush into — this is
        # the value of trainer.ep_indices[0] at the moment the boundary hits,
        # but at start-of-rollout it's trivially 0 (free_idx starts at 0 and
        # ep_indices was initialised to arange(total_agents)).
        pre_seg_idx = int(trainer.ep_indices[0].item())
        trainer._current_segment_has_event.zero_()
        trainer._current_segment_has_event[0] = True
        trainer._event_mask.zero_()

        trainer.evaluate()

        # After one full evaluate() round all 320 agent rows flushed once, so
        # the pre_seg_idx slot must reflect our True write.
        assert bool(trainer._event_mask[pre_seg_idx].item()), (
            "Task 7: live accumulator True for row 0 did not flush into "
            f"_event_mask[{pre_seg_idx}] at segment boundary.")
        # Live accumulator for row 0 must have been reset (it would only come
        # back True if a real bomb_planted event happened for env 0 during
        # the rollout, which is possible but not guaranteed — so we assert a
        # weaker property: at least one row is False, proving the reset path
        # is not a no-op. With a 64-tick rollout and random actions most envs
        # see zero plants, so most rows will be False.)
        assert not trainer._current_segment_has_event.all().item(), (
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

    from tests._helpers.trainer_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        # Baseline — ensure no stale event bits.
        trainer._event_mask.zero_()
        trainer._current_segment_has_event.zero_()

        vecenv = trainer.vecenv
        real_recv = vecenv.recv
        injected = {"triggered": False}

        # Whitelist the exact set of fields split_into_channels reads + the
        # event trigger. Anything else raises KeyError so that if Task 4's
        # routed-field list ever grows, this test fails loudly instead of
        # silently returning 0 and painting a false-green picture.
        from cs2rl.train.rewards import _COMBAT_FIELDS, _OBJECTIVE_FIELDS, _POSITIONAL_FIELDS
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
            assert trainer._event_mask.any().item(), (
                "Task 7: injected bomb_planted=1 in env 0 did not propagate into "
                "_event_mask after segment flush.")
        finally:
            # Always restore recv — cleanup() below may or may not re-init the
            # vecenv, and a leaked stub would poison fixture-shared state if
            # any future test reuses the same trainer instance.
            vecenv.recv = real_recv
    finally:
        cleanup()


# ── Task 8: prio_probs event-biased oversampling ───────────────────────────
# These tests cover the prio_probs boosting added to the replacement train()
# body (Cs2PuffeRL.train in src/cs2rl/train/trainer.py since gh#168 W2a — the harness
# trainer is a Cs2PuffeRL since gh#168 W1.5).
# The plan target: segments
# whose _event_mask is True get sampled at least 25% of the time when
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
#       _event_oversample_fraction reports the RAW fraction of event
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

    from tests._helpers.trainer_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        # The prio_probs boost lives in the return-norm train() body, which
        # the harness trainer carries by construction (gh#168 W1.5).
        trainer.evaluate()             # populate the rollout buffer

        # Force half of segments to be "event" segments. Segment count
        # equals trainer.segments (one bool per buffer row).
        seg_count = trainer._event_mask.shape[0]
        trainer._event_mask.zero_()
        trainer._event_mask[:seg_count // 2] = True
        raw_event_fraction = trainer._event_mask.float().mean().item()

        real_multinomial, wrapper, captured = _capture_multinomial_calls()
        torch.multinomial = wrapper
        try:
            trainer.train()
        finally:
            torch.multinomial = real_multinomial

        assert len(captured) > 0, "train() did not call torch.multinomial"

        # Concatenate every minibatch idx tensor and count event-segment hits.
        all_idx = torch.cat(captured)
        hits = trainer._event_mask[all_idx].float().mean().item()

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

    from tests._helpers.trainer_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        trainer.evaluate()
        trainer._event_mask.zero_()

        # Should run cleanly with the all-False mask.
        trainer.train()

        assert hasattr(trainer, "_event_oversample_fraction"), (
            "Task 8: trainer._event_oversample_fraction not exposed")
        assert trainer._event_oversample_fraction == 0.0, (
            f"Task 8: expected 0.0 oversample fraction with all-False mask, "
            f"got {trainer._event_oversample_fraction}")

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
    from tests._helpers.trainer_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        trainer.evaluate()

        seg_count = trainer._event_mask.shape[0]
        trainer._event_mask.zero_()
        trainer._event_mask[:seg_count // 2] = True
        expected_raw = (seg_count // 2) / seg_count

        trainer.train()

        frac = trainer._event_oversample_fraction
        assert 0.4 <= frac <= 0.6, (
            f"Task 8: expected raw event fraction in [0.4, 0.6], got {frac:.3f} "
            f"(seg_count={seg_count}, expected_raw={expected_raw:.3f})")
    finally:
        cleanup()


# ── Task 9: target_entropy schedule + log_alpha reset + Batch 1 metrics ────
# These tests cover three sub-features of the return-norm machinery
# (Cs2PuffeRL._init_return_norm + Cs2PuffeRL.train, src/cs2rl/train/trainer.py, gh#168 W2a):
#   (A) target_entropy schedule — linear ramp 0.7→0.5 * max_entropy across
#       global_step ∈ [0, 10_000_000]; constant after.
#   (B) log_alpha reset — first train() after construction sets log_alpha to
#       log(ent_coef); idempotent thereafter.
#   (C) Metric exposure — log_alpha, effective_alpha, per-channel std,
#       grad_norm exposed as trainer attributes for the wandb log layer.
#
# gh#168 W1.5: the harness applies that patch itself (Cs2PuffeRL.__init__), so
# none of these tests applies it; each reads the attributes straight off the
# built trainer (Task 8 tests are the same shape).


def test_target_entropy_schedule_applied():
    """Task 9A: trainer._current_target_entropy must follow the
    linear ramp warmup_frac→base_frac * max_entropy across [0, warmup_steps]
    global steps. Fracs are config-driven since the finding-4-residual fix
    (defaults 0.5→0.35; see test_entropy_target_config_threading) — this
    test reads them from trainer.config and additionally proves a custom
    config value is honored by the live train() recompute.

    Batch 3 (T5): the max_entropy expectation now includes the closed-form
    Gaussian aim head entropy at σ_max = exp(LOG_STD_MAX). Pre-Batch-3 this
    pin was sum(log(n) for n in ACTION_HEAD_SIZES) ≈ 6.7616; the Gaussian
    term at AIM_DIM=1 / σ_max=0.5 adds 0.5·log(2πe·σ_max²) ≈ 0.7258, giving
    ≈ 7.4874. The continuous term can be NEGATIVE for σ < 1/√(2πe) — but
    here we evaluate at σ_max where it is positive, so target_entropy stays
    > 0 across the whole ramp. Do NOT add an entropy >= 0 assertion to this
    test (or anywhere else); negative continuous entropy is expected at
    LOG_STD_INIT=0.1 and not a bug.
    """
    import math

    from cs2rl.policy import LOG_STD_MAX
    from cs2rl.spec.action import ACTION_HEAD_SIZES, AIM_DIM
    from tests._helpers.trainer_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:

        expected_max_discrete = sum(math.log(n) for n in ACTION_HEAD_SIZES)
        # Closed-form Normal entropy at σ = exp(LOG_STD_MAX), summed across
        # AIM_DIM. Mirrors the calc inside Cs2PuffeRL._init_return_norm
        # so this pin tracks the production formula exactly.
        sigma_max = math.exp(LOG_STD_MAX)
        expected_max_continuous = AIM_DIM * 0.5 * math.log(2 * math.pi * math.e * sigma_max**2)
        expected_max = expected_max_discrete + expected_max_continuous
        # Pin the run-constant first; if this drifts the schedule break too.
        assert abs(trainer._max_entropy -
                   expected_max) < 1e-9, (f"Task 9A: _max_entropy={trainer._max_entropy} != "
                                          f"discrete+continuous max={expected_max}")

        # evaluate() advances global_step by ~one rollout (batch_size); we
        # need a populated rollout buffer for train() to be valid, so call
        # evaluate() first and THEN pin global_step to the value we want
        # the schedule to see. Without this re-pin, evaluate()'s side effect
        # would mask the schedule check.
        trainer.evaluate()

        warmup_frac = trainer.config["entropy_target_warmup_frac"]
        base_frac = trainer.config["entropy_target_base_frac"]

        # Step 0: target = warmup_frac * max
        trainer.global_step = 0
        trainer.train()
        assert abs(trainer._current_target_entropy - warmup_frac * expected_max) < 1e-5, (
            f"Task 9A: at step 0 expected {warmup_frac}*max={warmup_frac * expected_max:.4f}, "
            f"got {trainer._current_target_entropy:.4f}")

        # Step 20M (past warmup_end=10M): target = base_frac * max (constant after).
        trainer.global_step = 20_000_000
        trainer.train()
        assert abs(trainer._current_target_entropy - base_frac * expected_max) < 1e-5, (
            f"Task 9A: at step 20M expected {base_frac}*max={base_frac * expected_max:.4f}, "
            f"got {trainer._current_target_entropy:.4f}")

        # Across the full ramp the value must stay <= max_entropy at every
        # checked step. Upper-bound warmup_frac*max means it can never exceed max.
        for step in (0, 1_000_000, 5_000_000, 10_000_000, 20_000_000):
            trainer.global_step = step
            trainer.train()
            assert trainer._current_target_entropy <= expected_max + 1e-9, (
                f"Task 9A: target_entropy={trainer._current_target_entropy} "
                f"exceeded max_entropy={expected_max} at step={step}")

        # Config threading end-to-end: a custom frac set on the live config
        # must be picked up by the next train() call's schedule recompute.
        trainer.config["entropy_target_warmup_frac"] = 0.42
        trainer.global_step = 0
        trainer.train()
        assert abs(trainer._current_target_entropy - 0.42 * expected_max) < 1e-5, (
            f"custom entropy_target_warmup_frac not honored: expected "
            f"{0.42 * expected_max:.4f}, got {trainer._current_target_entropy:.4f}")
    finally:
        cleanup()


def test_log_alpha_reset_at_batch_start():
    """Task 9B: the first train() call on a freshly constructed trainer (the
    return-norm state is seeded by Cs2PuffeRL._init_return_norm, gh#168 W2a;
    the harness trainer is a Cs2PuffeRL since W1.5)
    must reset log_alpha to log(ent_coef). Subsequent calls must NOT
    re-reset (idempotent via the _log_alpha_reset_done flag)."""
    import math

    from tests._helpers.trainer_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:

        # Pre-train invariant: flag is False.
        assert trainer._log_alpha_reset_done is False, (
            "Task 9B: _log_alpha_reset_done should start False")

        trainer.evaluate()
        trainer.train()

        # Flag flipped after first train().
        assert trainer._log_alpha_reset_done is True, (
            "Task 9B: _log_alpha_reset_done should be True after first train()")

        # log_alpha is close to log(ent_coef). The 1e-3 tolerance covers a
        # single alpha_optimizer.step() at lr=1e-4 — far less than ent_coef
        # log magnitude — so this still pins "we reset" vs "we did not".
        ent_coef = trainer.config["ent_coef"]
        expected = math.log(ent_coef)
        assert abs(trainer._log_alpha -
                   expected) < 1e-3, (f"Task 9B: _log_alpha={trainer._log_alpha:.6f} != "
                                      f"log(ent_coef={ent_coef})={expected:.6f}")
    finally:
        cleanup()


def test_training_metrics_exposed():
    """Task 9C: after evaluate() + train() the trainer must expose every
    Batch 1 metric the wandb log layer reads: log_alpha, effective_alpha,
    per-channel std (combat/objective/positional), event_oversample_fraction
    (set by Task 8), and grad_norm (pre-clip)."""
    import math

    from tests._helpers.trainer_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        trainer.evaluate()
        trainer.train()

        for name in (
                "_log_alpha",
                "_effective_alpha",
                "_std_combat",
                "_std_objective",
                "_std_positional",
                "_event_oversample_fraction",
                "_grad_norm",
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
    """Task 9a: Cs2PuffeRL._init_return_norm (run by __init__, gh#168 W2a) must
    put _ret_mean/_ret_var/_ret_count into a neutral state and expose them on
    the trainer.

    Reading them BEFORE any train() call pins the patch-time invariant —
    this is what guarantees a fresh start in symlog space when Batch 1 is
    enabled on a previously-trained checkpoint."""
    from tests._helpers.trainer_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
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
    from tests._helpers.trainer_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        trainer.evaluate()
        trainer.train()
        std = trainer._ret_var.item()**0.5
        # The harness produces ~32 envs × 4 rollout rounds per train() call,
        # so Welford min_count=1000 is never crossed and per-channel std
        # stays at prior=1.0 — symlog operates on raw gamma-discounted
        # channel sums, which empirically land in the 5-15 range. The
        # 10.0 boundary was flaky right at the upper edge. 50.0 still
        # catches a runaway pipeline without flaking on warmup-bound
        # harness arithmetic. (Cherry of stranded Batch 1 commit 8c1226c.)
        assert std < 50.0, (f"_ret_var std={std:.4f} too high — returns appear to be on raw "
                            f"scale rather than symlog-compressed. Pipeline broken.")
    finally:
        cleanup()


# ── Batch 2 (utof/cs2rl Batch 2): designated bomb carrier — round-fixed ──
def test_round_designated_carrier_assigned():
    """At env_reset, all three carrier signals must align."""
    env = make_env(seed=42)
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
    env = make_env(seed=7)
    try:
        env.reset(seed=7)
        g = env._c_env.game
        rid_at_start = g.round_designated_carrier_id
        g.agents[rid_at_start].hp = 0
        g.agents[rid_at_start].alive = 0
        actions = np.zeros((10, len(spec_action.ACTION_HEAD_SIZES)), dtype=np.int32)
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
    actions = np.zeros((10, len(spec_action.ACTION_HEAD_SIZES)), dtype=np.int32)
    for seed in range(50):
        env = make_env(seed=seed)
        try:
            env.reset(seed=seed)
            g = env._c_env.game
            rid = g.round_designated_carrier_id
            kill_seed = (seed % 5 == 0)                                      # 10 / 50 seeds
            for tick in range(200):
                if g.round_over:
                    break
                if kill_seed and tick == 5:
                                                                             # Drive the drop-on-death path (cs2_env.h:155-167).  # noqa: E501
                    g.agents[rid].hp = 0
                    g.agents[rid].alive = 0
                env.step(actions)
                assert g.round_designated_carrier_id == rid, (
                    f"seed={seed} tick={tick}: round_designated_carrier_id "
                    f"drifted ({rid} → {g.round_designated_carrier_id})")
        finally:
            env.close()


def test_obs_designated_carrier_bit_t_side():
    """Verify obs[109] (OBS_GLOBAL_BASE+13; was 106 before the Batch 6
    bombsite-bearing slots) is the round-fixed role bit:
       - 1.0 for the designated T agent
       - 0.0 for non-designated T agents
       - 0.0 for ALL CT agents
       - persists at 1.0 even after the carrier dies and a teammate picks up

    obs[109] is distinct from obs[22] (transient self-has-bomb): it is set at
    round start and never reassigned, surviving drop/pickup events. This gives
    the policy a stable identity signal that obs[22] cannot.

    NOTE: env_reset() does NOT call compute_observations (it only zeroes the
    buffer). The first populated observation arrives after env.step(). We
    therefore take one zero-action step before checking the role bit values.
    """
    env = make_env(seed=11)
    try:
        env.reset(seed=11)
        g = env._c_env.game
        rid = g.round_designated_carrier_id
        # Take one step to populate observations (env_reset zeroes the buffer;
        # compute_observations only runs inside env_step).
        actions = np.zeros((10, len(spec_action.ACTION_HEAD_SIZES)), dtype=np.int32)
        obs, *_ = env.step(actions)

        # T side
        for i in range(5):
            expected = 1.0 if i == rid else 0.0
            assert obs[i, 109] == expected, (
                f"T idx {i}: obs[109]={obs[i,109]} expected {expected} (rid={rid})")
        # CT side: all zeros
        for j in range(5, 10):
            assert obs[j, 109] == 0.0, f"CT idx {j}: obs[109]={obs[j,109]} expected 0.0"

        # Persistence after carrier death — drive 20 zero-action steps.
        g.agents[rid].hp = 0
        g.agents[rid].alive = 0
        for _ in range(20):
            obs, *_ = env.step(actions)
            assert obs[rid, 109] == 1.0, (
                f"designated carrier (T idx {rid}) lost the role bit mid-round; "
                f"obs[109] should be round-fixed but read {obs[rid,109]}")
    finally:
        env.close()


def test_post_pickup_plant_mask_unmasked():
    """After carrier dies and a teammate auto-picks up the bomb, the new
    holder's HEAD_USE plant action must be unmasked when standing at a
    bombsite. Closes mega-spec §Risks gap and verifies the dynamic
    has_bomb gate (not the round-fixed role bit) drives the plant mask.

    Sequencing rationale: the drop-on-death logic at cs2_env.h:154-167
    runs inside env_step. process_bomb (which contains the auto-pickup
    loop, cs2_bomb.h:93-114) is called in the SAME env_step immediately
    after the drop. Because T-agents share a tight spawn cluster, another
    T is almost always within the 32-unit pickup radius, so drop + pickup
    typically complete atomically in one step. The test therefore:

      Step 1 — kill the carrier and step; confirm the bomb is no longer
               with the original carrier (either still dropped OR already
               picked up by a nearby teammate).
      Step 2 — if bomb is still dropped (rare), teleport a teammate onto
               it and step so the pickup loop fires; either way, identify
               the new_holder as whoever now has has_bomb==1.
      Step 3 — scan bombsite areas; teleport the new_holder to each and
               step until HEAD_USE+1 is unmasked.
    """
    env = make_env(seed=21)
    try:
        env.reset(seed=21)
        g = env._c_env.game
        rid = g.round_designated_carrier_id
        actions = np.zeros((10, len(spec_action.ACTION_HEAD_SIZES)), dtype=np.int32)

        # Step 1: kill the carrier; env_step performs drop-on-death and (if a
        # teammate is within 32 units) the auto-pickup atomically in the same
        # call — bomb_is_dropped may go 0→1→0 internally in one tick.
        g.agents[rid].hp = 0
        g.agents[rid].alive = 0
        env.step(actions)

        # After step 1 the original carrier must not still hold the bomb.
        assert g.agents[rid].has_bomb == 0, (
            f"original carrier (idx {rid}) still has_bomb after death+step; "
            f"drop-on-death at cs2_env.h:157-169 may be broken")

        # Step 2 (conditional): if the bomb is still in the air (no teammate
        # was within 32 units), teleport the next-T teammate onto the drop
        # location so the pickup loop fires on the following step.
        if g.bomb_is_dropped:
            # Pick the first ALIVE non-carrier T. Hardcoding (rid + 1) % 5 is
            # brittle: that agent could itself have died on the same tick (e.g.
            # multi-kill seeds). Iterating + alive-check removes the seed
            # dependency.
            candidate = next((i for i in range(5) if i != rid and g.agents[i].alive), None)
            assert candidate is not None, (
                "no alive T teammate available to receive the dropped bomb; "
                "all 5 T-agents died on the same tick (test-setup edge case)")
            g.agents[candidate].x = g.bomb_x
            g.agents[candidate].y = g.bomb_y
            env.step(actions)

        # Identify the new bomb holder (whoever now has has_bomb==1 among
        # alive T-agents).
        new_holder = None
        for i in range(5):
            if g.agents[i].has_bomb == 1 and g.agents[i].alive:
                new_holder = i
                break
        assert new_holder is not None, (
            "No alive T-agent holds the bomb after drop+pickup sequence; "
            "auto-pickup loop at cs2_bomb.h:93-114 may be broken or all "
            "T-agents died during the sequence")
        assert new_holder != rid, (f"bomb ended up back with the original carrier (idx {rid}); "
                                   "expected a teammate to receive it after drop+pickup")

        # Step 3: find a bombsite area and teleport new_holder there; verify
        # HEAD_USE+1 (plant action) is unmasked by the dynamic has_bomb gate.
        # We scan all sd.N area indices (not capped) to locate bombsite areas,
        # then step only once we land on one. Bombsite indices on Dust2 start
        # around idx 1320 so a small cap like 200 would miss them entirely.
        # We limit the number of env.step calls (not area scans) to 20 so the
        # test terminates even if every bombsite area somehow fails to unmask.
        sd = env._c_env.sd.contents
        # Offset into the flat mask row for HEAD_USE action slot 1 (plant).
        # ACTION_HEAD_SIZES = (move, shoot, reload, weapon, use, crouch, jump)
        # → USE head starts at sum of the first FOUR sizes (9+2+2+3 = 16).
        # N1 (2026-07-06 verification): this used to be [:5] = 18, which put
        # the assertion on crouch-press (index 19) — always 1 for alive
        # agents, so the test was vacuously green. Derive the offset from the
        # head-name index instead of a hardcoded count so a future head
        # reorder can't silently re-vacuous it.
        use_head_idx = spec_action.ACTION_HEAD_NAMES.index("use")
        use_mask_offset = sum(spec_action.ACTION_HEAD_SIZES[:use_head_idx])
        assert use_mask_offset == 16, "USE head offset drifted; check ACTION_HEAD_SIZES order"
        masks_open = False
        steps_taken = 0
        for ai in range(sd.N):
            if not sd.bombsite_by_idx[ai]:
                continue
            g.agents[new_holder].area_idx = ai
            env.step(actions)
            steps_taken += 1
            if env._masks_view[new_holder, use_mask_offset + 1] == 1:
                masks_open = True
                break
            if steps_taken >= 20:
                break
        assert masks_open, (f"tried {steps_taken} bombsite areas (scanned all {sd.N} indices); "
                            f"post-pickup carrier (idx {new_holder}) on a bombsite did NOT have "
                            f"HEAD_USE+1 unmasked. Dynamic has_bomb gate at cs2_env.h:253-263 "
                            f"may be broken. new_holder.has_bomb={g.agents[new_holder].has_bomb}, "
                            f"new_holder.alive={g.agents[new_holder].alive}, "
                            f"bomb_planted={g.bomb_planted}, round_over={g.round_over}")
    finally:
        env.close()


# ── Batch 2 task 4: OBS_DIM constant-consistency ─────────────────────────────
def test_obs_dim_constant_consistency():
    """nav.OBS_DIM must agree with env.single_observation_space.shape[0].
    A drift here means the C ↔ Python boundary is misconfigured. The
    ctypes-vs-C layout asserts in cs2_env.py (the `_C_SIZES` block, fed by
    binding.struct_sizes()) catch struct-size drift; this test is the
    higher-level constant-agreement check.
    train.OBS_DIM is no third declaration: it and nav.OBS_DIM are the one
    generated spec.obs.OBS_DIM, so comparing them proves nothing. It is checked
    against the env by test_make_env_reset_returns_expected_batch. The
    TEAM_SIZE literals are cross-checked here too.
    """
    from cs2rl.env import nav
    from cs2rl.train import config as train_config
    assert nav.OBS_DIM == 110, f"nav.OBS_DIM is {nav.OBS_DIM}, expected 110 for Batch 6 Task 2.5"
    # Rung 0 (spec 2026-08-29 §2.2): train.TEAM_SIZE is a bare literal (train_shared
    # must stay free of the nav import), so it needs a drift guard —
    # it divides the participating-step budget and builds the per-row
    # participation vector.
    from cs2rl.env.c import cs2_env
    assert train_config.TEAM_SIZE == nav.TEAM_SIZE == cs2_env.TEAM_SIZE, (
        f"train.TEAM_SIZE ({train_config.TEAM_SIZE}) / nav.TEAM_SIZE ({nav.TEAM_SIZE}) / "
        f"cs2_env.TEAM_SIZE ({cs2_env.TEAM_SIZE}) disagree")
    env = make_env(seed=0)
    try:
        assert env.single_observation_space.shape == (nav.OBS_DIM, ), (
            f"env.single_observation_space.shape={env.single_observation_space.shape} "
            f"!= ({nav.OBS_DIM},)")
    finally:
        env.close()


def test_team_size_literals_agree():
    """env_config.TEAM_SIZE and train_shared.TEAM_SIZE are literals (both leaves
    refuse to import nav just to read a 5). This is the cross-check that makes
    the literals safe (spec 2026-09-03 §2.1)."""
    from cs2rl.env import config as env_config
    from cs2rl.env import nav
    from cs2rl.env.c.cs2_env import TEAM_SIZE as c_team
    from cs2rl.train import config as train_config
    assert env_config.TEAM_SIZE == train_config.TEAM_SIZE == nav.TEAM_SIZE == c_team


def test_obs_blocks_tile_obs_dim():
    """OBS_BLOCKS (generated from cs2_types.h OBS_* macros) must tile [0, OBS_DIM)
    with no gaps/overlaps, in order. This is the Python mirror of the env_init
    tiling assert in cs2_env.h. Demo-zeroing / masking code slices obs via this
    table (obs[start:stop]) instead of hardcoding 28/56/96 — a drift here would
    silently zero the wrong obs slice, so pin the boundaries explicitly.
    Catches the regen-not-run footgun (edit cs2_types.h, forget the generator)."""
    from cs2rl.spec import obs as spec

    # Named boundaries are the ones the upcoming demo code depends on.
    assert spec.OBS_BLOCKS["self"] == (0, 28)
    assert spec.OBS_BLOCKS["teammate"] == (28, 56)
    assert spec.OBS_BLOCKS["enemy"] == (56, 96)
    assert spec.OBS_BLOCKS["global"] == (96, 110)

    # Structural invariant: contiguous, ordered, covers exactly [0, OBS_DIM).
    prev_stop = 0
    for name, (start, stop) in spec.OBS_BLOCKS.items():
        assert start == prev_stop, f"block {name!r} starts at {start}, expected {prev_stop} (gap/overlap)"
        assert stop > start, f"block {name!r} is empty/inverted: {(start, stop)}"
        prev_stop = stop
    assert prev_stop == spec.OBS_DIM, (f"blocks end at {prev_stop} but OBS_DIM={spec.OBS_DIM}")

    # Per-entity sub-structure matches the block widths (count × stride).
    tm_start, tm_stop = spec.OBS_BLOCKS["teammate"]
    assert tm_stop - tm_start == spec.OBS_TEAMMATE_COUNT * spec.OBS_TEAMMATE_STRIDE
    en_start, en_stop = spec.OBS_BLOCKS["enemy"]
    assert en_stop - en_start == spec.OBS_ENEMY_COUNT * spec.OBS_ENEMY_STRIDE


# ── Batch 3 (utof/cs2rl Batch 3): continuous-aim H-PPO ──


def test_action_spec_aim_is_gaussian_2d():
    """`spec/action.py` exports the discrete/continuous split:
       - DISCRETE_HEAD_SPEC has 7 categorical entries, sum(sizes) = 22
       - CONTINUOUS_HEAD_SPEC has 1 gaussian entry, dim 2 (Δyaw + Δpitch, Batch 3.5)
       - AIM_DIM == 2
       - Backwards-compat ACTION_DIM is 7, ACTION_MASK_DIM is 22.
    Catches the regen-not-run footgun (modifying cs2_types.h without
    re-running scripts/sync_action_spec.py)."""
    from cs2rl.spec import action as spec
    assert spec.AIM_DIM == 2, f"AIM_DIM={spec.AIM_DIM}, expected 2 for Batch 3.5"
    assert spec.ACTION_DIM == 7, f"ACTION_DIM={spec.ACTION_DIM}, expected 7 (HEAD_AIM removed)"
    assert spec.ACTION_MASK_DIM == 22, f"ACTION_MASK_DIM={spec.ACTION_MASK_DIM}, expected 22"
    assert len(spec.DISCRETE_HEAD_SPEC) == 7, (
        f"DISCRETE_HEAD_SPEC has {len(spec.DISCRETE_HEAD_SPEC)} entries, expected 7")
    assert all(t == "categorical" for _, t, _ in spec.DISCRETE_HEAD_SPEC), (
        f"DISCRETE_HEAD_SPEC has non-categorical entries: {spec.DISCRETE_HEAD_SPEC}")
    assert spec.CONTINUOUS_HEAD_SPEC == (("aim", "gaussian", 2), ), (
        f"CONTINUOUS_HEAD_SPEC={spec.CONTINUOUS_HEAD_SPEC}, expected gaussian/dim=2")


def test_continuous_aim_action_consumed():
    """Env step with continuous_actions[i,0]=0.1 advances facing by exactly 0.1 rad.

    Batch 3: validates that the float buffer plumbed through binding.step
    actually drives env_step's wrap_pi(facing + clamped) branch. Uses the
    designated bomb carrier (RL agent, not human_controlled) to ensure the
    continuous branch fires.
    """
    from cs2rl.spec import action as spec
    env = make_env(seed=42)
    try:
        env.reset(seed=42)
        g = env._c_env.game
        i = g.round_designated_carrier_id
        f0 = g.agents[i].facing
        actions = np.zeros((10, spec.ACTION_DIM), dtype=np.int32)
        cont = np.zeros((10, spec.AIM_DIM), dtype=np.float32)
        cont[i, 0] = 0.1
        env._c_env                     # noqa: B018  (touch to ensure overlay is live)
        env.step(actions, cont)
        assert g.agents[i].alive == 1
        delta = g.agents[i].facing - f0
        if delta > np.pi:
            delta -= 2 * np.pi
        if delta < -np.pi:
            delta += 2 * np.pi
        assert abs(delta -
                   0.1) < 1e-5, (f"facing advanced by {delta} rad, expected ~0.1 (clamped+wrapped)")
    finally:
        env.close()


def test_continuous_aim_clamped_at_max_turn():
    """Δyaw=10.0 is clamped to max_turn_speed=π/4 ≈ 0.7854 rad.

    Batch 3: verifies the fminf/fmaxf clamp inside env_step. Without it
    the policy could turn arbitrarily fast and break collision/visibility
    invariants.
    """
    from cs2rl.spec import action as spec
    env = make_env(seed=42)
    try:
        env.reset(seed=42)
        g = env._c_env.game
        sd = env._c_env.sd.contents
        i = g.round_designated_carrier_id
        f0 = g.agents[i].facing
        actions = np.zeros((10, spec.ACTION_DIM), dtype=np.int32)
        cont = np.zeros((10, spec.AIM_DIM), dtype=np.float32)
        cont[i, 0] = 10.0
        env.step(actions, cont)
        delta = g.agents[i].facing - f0
        if delta > np.pi:
            delta -= 2 * np.pi
        if delta < -np.pi:
            delta += 2 * np.pi
        assert abs(delta - sd.max_turn_speed) < 1e-5, (
            f"clamp failed: delta={delta}, expected {sd.max_turn_speed}")
    finally:
        env.close()


def test_continuous_aim_facing_wraps_around_pi():
    """Starting facing=π−0.1, Δyaw=+0.2 wraps to ~−π+0.1.

    Batch 3: verifies wrap_pi keeps facing bounded to [-π, +π]. Without
    wrap_pi, facing would drift to π+0.1 and break downstream consumers
    that assume the bounded representation (renderer, observation
    normalisation, etc.).
    """
    from cs2rl.spec import action as spec
    env = make_env(seed=42)
    try:
        env.reset(seed=42)
        g = env._c_env.game
        i = g.round_designated_carrier_id
        g.agents[i].facing = np.pi - 0.1
        actions = np.zeros((10, spec.ACTION_DIM), dtype=np.int32)
        cont = np.zeros((10, spec.AIM_DIM), dtype=np.float32)
        cont[i, 0] = 0.2
        env.step(actions, cont)
        expected = -np.pi + 0.1
        assert abs(g.agents[i].facing - expected) < 1e-5, (
            f"wrap failed: facing={g.agents[i].facing}, expected {expected}")
    finally:
        env.close()


def test_step_stats_aim_delta_tracking():
    """100-tick rollout with Δyaw=0.05; assert sum/sq_sum/count tracking.

    Batch 3: validates that StepStats's Welford triple is populated so
    Python-side Δyaw mean/var can be recovered for telemetry without
    storing the full rollout. Resets the fields before the loop because
    other tests in the same env instance may have populated them.
    """
    from cs2rl.spec import action as spec
    env = make_env(seed=42)
    try:
        env.reset(seed=42)
        actions = np.zeros((10, spec.ACTION_DIM), dtype=np.int32)
        cont = np.full((10, spec.AIM_DIM), 0.05, dtype=np.float32)
        ss = env._c_env.episode_stats
        ss.aim_delta_sum = 0.0
        ss.aim_delta_sq_sum = 0.0
        ss.aim_delta_count = 0
        g = env._c_env.game
        for _ in range(100):
            if g.round_over:
                break
            env.step(actions, cont)
        assert ss.aim_delta_count > 0, "no aim_delta updates recorded"
        assert ss.aim_delta_sum > 0, f"sum should be positive, got {ss.aim_delta_sum}"
        mean = ss.aim_delta_sum / ss.aim_delta_count
        assert abs(mean - 0.05) < 1e-5, f"mean Δyaw={mean}, expected 0.05"
    finally:
        env.close()


# ── Batch 3 Task 4: Hybrid policy forward shape + log_std clamp tests ──────
# These exercise build_policy()'s new Gaussian aim head: the 4-tuple return
# from forward(), the µ shape (B, AIM_DIM) bounded by max_turn_speed via
# tanh*scale, and the [LOG_STD_MIN, LOG_STD_MAX] clamp that prevents σ
# collapse / explosion. Trainer-side wiring (PPO ratio over both factors)
# arrives in Task 5; these tests stay green regardless of trainer state.
def test_policy_forward_emits_mu_and_logstd():
    """build_policy returns a hybrid policy whose forward emits (logits[7],
    mu_aim, log_std, value). mu_aim shape (B, 1), log_std broadcasts to
    same. AIM_DIM bump to 2 (Batch 3.5) is a single-line change here."""
    import torch

    from cs2rl.spec import action as spec
    env = make_env(seed=0)
    try:
        policy = policy_mod.build_policy(env, device='cpu')
        x = torch.zeros((1, spec_obs.OBS_DIM))
        logits, mu_aim, log_std, value = policy.forward(x, state={})
        assert len(logits) == spec.ACTION_DIM == 7, (
            f"got {len(logits)} discrete heads, expected 7")
        assert mu_aim.shape == (1, spec.AIM_DIM), (
            f"mu_aim.shape={mu_aim.shape}, expected (1, {spec.AIM_DIM})")
        assert log_std.shape == mu_aim.shape, (
            f"log_std.shape={log_std.shape} should broadcast to mu shape")
        # μ is already tanh*max_turn_speed scaled, so |μ| ≤ max_turn_speed.
        sd = env._c_env.sd.contents
        assert mu_aim.abs().max().item() <= sd.max_turn_speed + 1e-6, (
            f"|mu_aim| exceeds max_turn_speed: {mu_aim.abs().max().item()}")
        # value head still emits a single scalar per agent.
        assert value.shape == (1, 1), f"value.shape={value.shape}, expected (1, 1)"
    finally:
        env.close()


def test_logstd_clamp_lower():
    """Push aim_log_std → -∞ via direct write; forward()'s output must
    be ≥ LOG_STD_MIN after the clamp. Defends σ collapse — without the
    clamp the Normal entropy would diverge to −∞ and pin the SAC-α loop."""
    import torch
    env = make_env(seed=0)
    try:
        policy = policy_mod.build_policy(env, device='cpu')
        with torch.no_grad():
            policy.aim_log_std.fill_(-100.0)                            # exp(-100) ≈ 0
        x = torch.zeros((1, spec_obs.OBS_DIM))
        _, _, log_std, _ = policy.forward(x, state={})
        assert log_std.min().item() >= policy_mod.LOG_STD_MIN - 1e-6, (
            f"log_std={log_std.min().item()} below LOG_STD_MIN={policy_mod.LOG_STD_MIN}")
    finally:
        env.close()


def test_logstd_clamp_upper():
    """Push aim_log_std → +∞; forward() output must be ≤ LOG_STD_MAX.
    Defends σ explosion — uncapped σ would dominate the policy and
    negate any μ signal the network learns."""
    import torch
    env = make_env(seed=0)
    try:
        policy = policy_mod.build_policy(env, device='cpu')
        with torch.no_grad():
            policy.aim_log_std.fill_(100.0)
        x = torch.zeros((1, spec_obs.OBS_DIM))
        _, _, log_std, _ = policy.forward(x, state={})
        assert log_std.max().item() <= policy_mod.LOG_STD_MAX + 1e-6, (
            f"log_std={log_std.max().item()} above LOG_STD_MAX={policy_mod.LOG_STD_MAX}")
    finally:
        env.close()


# ── Batch 3 Task 5: hybrid-aim trainer integration tests ─────────────────────
#
# These two tests exercise the trainer-side wiring that T5 added on top of
# T4's HybridPolicy:
#   - test_hybrid_sample_writes_two_buffers: a single get_action_and_value()
#       call must yield BOTH a finite int discrete action (7 heads) AND a
#       finite float Δyaw bounded by max_turn_speed. If either buffer is
#       silently zero/garbage the rollout would write a corrupt PPO target
#       and the training run would diverge invisibly.
#   - test_hybrid_loss_clip_applies_per_factor: feeds _hybrid_ppo_loss
#       inputs that produce an out-of-clip discrete ratio AND an in-clip
#       continuous ratio, asserting the per-factor clip (Fan et al. 2019)
#       triggers on one head and not the other. This is the "L8 spec lock"
#       — if a future refactor goes back to a single shared ratio this
#       test goes red.
#
# Both tests use the bare make_puffer_env path (no PufferLib trainer) so
# they are fast and don't depend on the test harness.


def test_hybrid_sample_writes_two_buffers():
    """T5: hybrid sampler must emit finite int discrete action AND finite
    float Δaim within ±max_turn_speed. Batch 3.5: AIM_DIM=2 (Δyaw + Δpitch)."""
    import torch

    from cs2rl.spec.action import AIM_DIM
    env = make_env(seed=0)
    try:
        policy = policy_mod.build_policy(env, device='cpu')
        x = torch.zeros((1, spec_obs.OBS_DIM))
        action, cont_action, lp, ent, val, _ = policy.get_action_and_value(x)
        assert action.shape == (1, 7), f"action.shape={action.shape}"
        assert action.dtype in (torch.int64, torch.long), \
            f"action.dtype={action.dtype}"
        assert torch.isfinite(action.float()).all()
        assert cont_action.shape == (1, AIM_DIM), \
            f"cont_action.shape={cont_action.shape}"
        assert cont_action.dtype == torch.float32
        assert torch.isfinite(cont_action).all()
        sd = env._c_env.sd.contents
        assert cont_action.abs().max().item() <= sd.max_turn_speed + 1e-5, (
            f"|cont_action|={cont_action.abs().max().item()} > max_turn_speed")
        assert lp.shape == (1, ) and ent.shape == (1, ), \
            f"lp.shape={lp.shape}, ent.shape={ent.shape}"
        assert torch.isfinite(lp).all() and torch.isfinite(ent).all()
        assert val.shape == (1, 1), f"val.shape={val.shape}"
    finally:
        env.close()


def test_hybrid_sample_logits_returns_per_factor_halves():
    """Fix #1: _hybrid_sample_logits returns 6-tuple
    (action, cont_action, log_prob_d, log_prob_c, entropy_d, entropy_c).

    The per-factor halves it returns must equal what the old rebuild
    pattern (constructing Categorical + Normal a second time at the
    rollout site) would produce — bit-equivalent because the math is
    identical and the inputs are deterministic given (logits, action).

    This test pins the new contract so a future change that re-summed the
    halves at return time (or worse, dropped an entropy slot) would go red.
    Without this assertion the rollout would silently feed a wrong
    log_probs_d / log_probs_c into _hybrid_ppo_loss and PPO updates would
    diverge invisibly.
    """
    import torch

    from cs2rl.policy import _hybrid_sample_logits

    torch.manual_seed(42)
    B = 16
    head_sizes = (9, 2, 2, 3, 2, 2, 2)
    logits_list = [torch.randn(B, n) for n in head_sizes]
    mu_aim = torch.zeros(B, 1)
    log_std_aim = torch.full((1, ), -2.30)             # log(0.1)

    ret = _hybrid_sample_logits(
        (logits_list, mu_aim, log_std_aim, None),
        max_turn_speed=0.7853981633974483,             # π/4
    )
    assert len(ret) == 6, f"Expected 6-tuple, got {len(ret)}-tuple"
    action, cont_action, lp_d, lp_c, ent_d, ent_c = ret

    # Shape checks
    assert action.shape == (B, len(head_sizes))
    assert cont_action.shape == (B, 1)
    assert lp_d.shape == (B, ) and lp_c.shape == (B, )
    assert ent_d.shape == (B, ) and ent_c.shape == (B, )
    assert torch.isfinite(lp_d).all() and torch.isfinite(lp_c).all()

    # Bit-equivalence with the rebuild pattern that the rollout caller
    # used to do (and which Fix #1 deletes). Same math, same inputs →
    # same bits. allclose with atol=0 is the strongest assertion.
    rebuild_lp_d = sum(
        torch.distributions.Categorical(logits=lg).log_prob(action[..., i])
        for i, lg in enumerate(logits_list))
    sigma = torch.exp(log_std_aim).expand_as(mu_aim)
    rebuild_lp_c = (torch.distributions.Normal(mu_aim, sigma).log_prob(cont_action).sum(-1))
    assert torch.allclose(lp_d, rebuild_lp_d, atol=1e-7), \
        f"log_prob_d drift: max diff {(lp_d - rebuild_lp_d).abs().max().item()}"
    assert torch.allclose(lp_c, rebuild_lp_c, atol=1e-7), \
        f"log_prob_c drift: max diff {(lp_c - rebuild_lp_c).abs().max().item()}"

    # Joint log-prob = sum of halves (spec L8 independence)
    assert torch.allclose(lp_d + lp_c, rebuild_lp_d + rebuild_lp_c, atol=1e-7)


def test_hybrid_loss_clip_applies_per_factor():
    """T5: _hybrid_ppo_loss applies the PPO clip independently per factor.

    Constructs a minibatch where:
      - mb_old_logp_d is set so ratio_d = exp(1) ≈ 2.72 — outside clip band
      - mb_old_logp_c is set so ratio_c ≈ exp(0.05) ≈ 1.05 — inside band
    With clip_coef=0.2 the discrete factor MUST be clamped to [0.8, 1.2]
    in the loss while the continuous factor stays unclamped. The asserts
    inspect the returned ratios directly to verify the input setup, then
    check that pg_loss is finite (i.e. the loss path didn't NaN out).
    """
    import torch

    env = make_env(seed=0)
    try:
        policy = policy_mod.build_policy(env, device='cpu')
        B = 4
        mb_obs = torch.zeros((B, spec_obs.OBS_DIM))
        mb_actions = torch.zeros((B, 7), dtype=torch.int64)
        mb_cont_actions = torch.zeros((B, 1), dtype=torch.float32)
        mb_advantages = torch.ones(B)

        # Seed old log-probs so we know what ratio_d / ratio_c will be.
        # First evaluate the new policy on the inputs to get the canonical
        # new_logp values; then offset by the desired ratio sign-flip.
        with torch.no_grad():
            logits_list, mu_aim, log_std_aim, _ = policy(mb_obs, state={})
            dists_d = [torch.distributions.Categorical(logits=lg) for lg in logits_list]
            new_logp_d_seed = sum(d.log_prob(mb_actions[..., i]) for i, d in enumerate(dists_d))
            sigma = torch.exp(log_std_aim).expand_as(mu_aim)
            dist_c = torch.distributions.Normal(mu_aim, sigma)
            new_logp_c_seed = dist_c.log_prob(mb_cont_actions).sum(-1)

        # ratio = exp(new_logp - old_logp): subtract delta from new_logp to set ratio.
        mb_old_logp_d = new_logp_d_seed - 1.0          # ratio_d = e^1 ≈ 2.72 → outside clip
        mb_old_logp_c = new_logp_c_seed - 0.05         # ratio_c ≈ 1.05 → inside clip

        from cs2rl.train.update import _hybrid_ppo_loss
        pg_loss, entropy, new_value, new_logp_total, ratio_d, ratio_c, _lg = _hybrid_ppo_loss(
            policy,
            mb_obs,
            mb_actions,
            mb_cont_actions,
            mb_old_logp_d,
            mb_old_logp_c,
            mb_advantages,
            clip_coef=0.2,
            state={},
        )
        assert (ratio_d > 1.2).all(), \
            f"ratio_d={ratio_d} should exceed 1.2 (outside clip)"
        assert ((ratio_c > 0.8) & (ratio_c < 1.2)).all(), \
            f"ratio_c={ratio_c} should be in (0.8, 1.2) (inside clip)"
        assert torch.isfinite(pg_loss).all()
        assert torch.isfinite(entropy).all()
    finally:
        env.close()


def test_hybrid_ppo_loss_matches_torch_distributions_reference():
    """Fix #3: _hybrid_ppo_loss now uses hand-rolled F.log_softmax + analytic
    Normal in place of torch.distributions.{Categorical,Normal}. The new
    new_logp_d / new_logp_c (and consequently ratio_d / ratio_c) must match
    what a torch.distributions reference implementation would compute, to
    fp32 noise tolerance. This pins the new contract — a future change that
    accidentally drops a term (e.g. forgets the -½ log 2π constant in the
    Normal log-prob) would silently shift all PPO ratios and go red here.
    """
    import torch

    from cs2rl import policy as policy_mod
    from cs2rl.spec import obs as spec_obs

    env = make_env(seed=0)
    try:
        policy = policy_mod.build_policy(env, device='cpu')
        torch.manual_seed(13)
        B = 8
        mb_obs = torch.randn((B, spec_obs.OBS_DIM)) * 0.5
        mb_actions = torch.randint(0, 2, (B, 7), dtype=torch.int64)
        mb_cont_actions = (torch.rand(B, 1) - 0.5) * 0.4               # within ±π/4
        mb_advantages = torch.randn(B)
        mb_old_logp_d = torch.zeros(B)
        mb_old_logp_c = torch.zeros(B)

        from cs2rl.train.update import _hybrid_ppo_loss
        pg_loss, entropy, new_value, new_logp_total, ratio_d, ratio_c, _lg = (_hybrid_ppo_loss(
            policy,
            mb_obs,
            mb_actions,
            mb_cont_actions,
            mb_old_logp_d,
            mb_old_logp_c,
            mb_advantages,
            clip_coef=0.2,
            state={}))

        # ── Reference: re-run the same math with torch.distributions ──
        # Both paths must observe the SAME policy state (no parameter mutation
        # between calls), so we re-evaluate logits/mu/log_std fresh under
        # no_grad and feed them to the reference.
        with torch.no_grad():
            ref_logits, ref_mu, ref_log_std, _ = policy(mb_obs, state={})
            ref_dists_d = [
                torch.distributions.Categorical(logits=lg, validate_args=False) for lg in ref_logits
            ]
            ref_logp_d = sum(d.log_prob(mb_actions[..., i]) for i, d in enumerate(ref_dists_d))
            ref_sigma = torch.exp(ref_log_std).expand_as(ref_mu)
            ref_dist_c = torch.distributions.Normal(ref_mu, ref_sigma, validate_args=False)
            ref_logp_c = ref_dist_c.log_prob(mb_cont_actions).sum(-1)

        # The hand-rolled new_logp_d/c are computed inside _hybrid_ppo_loss but
        # not directly returned; we recover them from ratios since old_logp=0.
        # ratio = exp(new_logp - 0) ⇒ new_logp = log(ratio).
        recovered_new_logp_d = ratio_d.log()
        recovered_new_logp_c = ratio_c.log()

        assert torch.allclose(
            recovered_new_logp_d, ref_logp_d,
            atol=1e-5), (f"new_logp_d drift: max diff "
                         f"{(recovered_new_logp_d - ref_logp_d).abs().max().item():.2e}")
        assert torch.allclose(
            recovered_new_logp_c, ref_logp_c,
            atol=1e-5), (f"new_logp_c drift: max diff "
                         f"{(recovered_new_logp_c - ref_logp_c).abs().max().item():.2e}")
        # Joint = sum of halves
        assert torch.allclose(new_logp_total, ref_logp_d + ref_logp_c, atol=1e-5)
        # pg_loss + entropy + value sane
        assert torch.isfinite(pg_loss).all()
        assert torch.isfinite(entropy).all()
        assert torch.isfinite(new_value).all()
    finally:
        env.close()


def test_pbrs_gamma_matches_training_gamma():
    """Finding 2 (2026-07-06 adversarial review): PBRS shaping used
    γ_pbrs = 0.99 while training used γ = 0.999. With F(s,s') =
    γ_pbrs·φ(s') − φ(s), any γ_pbrs ≠ γ breaks the Ng et al. policy-
    invariance guarantee — the residual (γ_pbrs − γ)·φ ≈ −0.009·φ per tick
    penalized dwelling in high-φ states. This drift guard pins the LIVE
    env value (read from C static data through the production factory)
    against build_train_config's gamma: if either side changes without the
    other, this fails and points here.
    """
    from types import SimpleNamespace

    import pytest

    from cs2rl.env.config import EnvConfig
    from cs2rl.train import config as train_config
    from cs2rl.train import resume as train_resume

    args = SimpleNamespace(seed=0, timesteps=1_000, checkpoint_dir="/tmp/unused", device="cpu")
    cfg = train_config.build_train_config(args, batch_size=1024, bptt_horizon=64)

    env = make_env(seed=0)
    try:
        live_pbrs_gamma = float(env._c_env.sd.contents.pbrs_gamma)
    finally:
        env.close()
    assert live_pbrs_gamma == pytest.approx(
        cfg["gamma"]), (f"pbrs_gamma={live_pbrs_gamma} != training gamma={cfg['gamma']}; "
                        f"PBRS is only policy-invariant when they match — update the "
                        f"cs2_env defaults (Cs2Env.__init__ AND make_env) or thread "
                        f"pbrs_gamma explicitly")

    # R0-J (Task 14): config records BOTH gammas. Default path: pbrs_gamma
    # follows gamma. Explicit --pbrs-gamma that differs is allowed (an
    # experiment that deliberately breaks invariance) and recorded verbatim.
    assert cfg["pbrs_gamma"] == cfg["gamma"] == 0.999
    args2 = SimpleNamespace(seed=0,
                            timesteps=1_000,
                            checkpoint_dir="/tmp/unused",
                            device="cpu",
                            gamma=0.999,
                            pbrs_gamma=0.99)
    cfg2 = train_config.build_train_config(args2, batch_size=1024, bptt_horizon=64)
    assert cfg2["gamma"] == 0.999 and cfg2["pbrs_gamma"] == 0.99
    args3 = SimpleNamespace(seed=0,
                            timesteps=1_000,
                            checkpoint_dir="/tmp/unused",
                            device="cpu",
                            gamma=0.99,
                            pbrs_gamma=None)
    cfg3 = train_config.build_train_config(args3, batch_size=1024, bptt_horizon=64)
    assert cfg3["gamma"] == 0.99 and cfg3["pbrs_gamma"] == 0.99
    assert train_config.env_config_from_args(args3).pbrs_gamma == 0.99
    assert "gamma" not in train_resume.RESUME_CONFIG_ALLOWLIST
    assert "pbrs_gamma" not in train_resume.RESUME_CONFIG_ALLOWLIST

    # N3 fix: make_puffer_env must expose pbrs_gamma for per-experiment
    # overrides (previously the training γ could not be threaded through
    # without a signature change).
    env = make_env(seed=0, config=EnvConfig(pbrs_gamma=0.5))
    try:
        assert float(env._c_env.sd.contents.pbrs_gamma) == pytest.approx(0.5)
    finally:
        env.close()


def test_entropy_target_config_threading():
    """Finding 4 residual (2026-07-06 adversarial review): the entropy-target
    schedule fracs were hardcoded (0.7→0.5·max held after 10M steps) — high
    enough that even with a live pg gradient the α controller steers the
    policy toward near-uniform forever. They are now config keys with lower
    defaults (0.5→0.35·max), threaded through _scheduled_target_entropy so
    both the patch-time seed and the per-train()-call recompute read the
    same source. Pins: key names + defaults in build_train_config, custom
    values honored, missing keys fall back, and the base target stays ABOVE
    the hard entropy floor (0.3·max) so the floor branch (clamp α ≥ 0.5)
    can never fight the controller.
    """
    from types import SimpleNamespace

    import pytest

    from cs2rl.train import config as train_config
    from cs2rl.train import update as train_update

    args = SimpleNamespace(seed=0, timesteps=1_000, checkpoint_dir="/tmp/unused", device="cpu")
    cfg = train_config.build_train_config(args, batch_size=1024, bptt_horizon=64)
    assert cfg["entropy_target_warmup_frac"] == 0.5
    assert cfg["entropy_target_base_frac"] == 0.35
    assert cfg["entropy_target_warmup_steps"] == 10_000_000
    # Floor consistency: base frac must stay above the 0.3·max hard floor.
    assert cfg["entropy_target_base_frac"] > 0.3

    max_ent = 8.0
    assert train_update._scheduled_target_entropy(cfg, 0, max_ent) == pytest.approx(0.5 * max_ent)
    assert train_update._scheduled_target_entropy(cfg, 10_000_000, max_ent) == \
        pytest.approx(0.35 * max_ent)
    assert train_update._scheduled_target_entropy(cfg, 30_000_000, max_ent) == \
        pytest.approx(0.35 * max_ent)

    custom = {
        "entropy_target_warmup_frac": 0.42,
        "entropy_target_base_frac": 0.21,
        "entropy_target_warmup_steps": 100,
    }
    assert train_update._scheduled_target_entropy(custom, 0, 10.0) == pytest.approx(4.2)
    assert train_update._scheduled_target_entropy(custom, 50, 10.0) == pytest.approx(3.15)
    assert train_update._scheduled_target_entropy(custom, 100, 10.0) == pytest.approx(2.1)

    # Missing keys → same defaults as build_train_config (config .get fallback).
    assert train_update._scheduled_target_entropy({}, 0, max_ent) == pytest.approx(0.5 * max_ent)


def test_hybrid_ppo_loss_normalizes_advantages():
    """Finding 1 (2026-07-06 adversarial review): _hybrid_ppo_loss must
    normalize advantages (stock-PufferLib style: (adv - mean) / (std + 1e-8),
    scaled by the prio-IS weight) INSIDE the loss. The fork previously fed
    raw advantages to the pg term and orphaned the normalization block at
    the call site — the runtime probe showed pg_loss linear in adv
    (pg_loss(10·A) = 10·pg_loss(A)), i.e. no normalization anywhere on the
    gradient path. With sparse rewards this left the policy gradient ≈ 0
    and let the entropy objective drag the policy to uniform (the 30M
    degenerate run). Normalizing inside the loss makes the property
    directly assertable: pg_loss must be invariant to advantage scale.

    Also pins the prio-IS contract: mb_prio=None and mb_prio=1 are
    equivalent; a non-uniform mb_prio must actually reweight the loss
    (previously it was computed and discarded).
    """
    import torch

    from cs2rl import policy as policy_mod
    from cs2rl.spec import obs as spec_obs

    env = make_env(seed=0)
    try:
        policy = policy_mod.build_policy(env, device='cpu')
        torch.manual_seed(7)
        B = 32
        mb_obs = torch.randn((B, spec_obs.OBS_DIM)) * 0.5
        mb_actions = torch.randint(0, 2, (B, 7), dtype=torch.int64)
        mb_cont_actions = (torch.rand(B, 1) - 0.5) * 0.4
        mb_advantages = torch.randn(B)
        mb_old_logp_d = torch.zeros(B)
        mb_old_logp_c = torch.zeros(B)

        from cs2rl.train.update import _hybrid_ppo_loss

        def loss_of(adv, prio=None):
            pg_loss, *_ = _hybrid_ppo_loss(
                policy,
                mb_obs,
                mb_actions,
                mb_cont_actions,
                mb_old_logp_d,
                mb_old_logp_c,
                adv,
                clip_coef=0.2,
                state={},
                mb_prio=prio,
            )
            return pg_loss

        base = loss_of(mb_advantages)

        # Scale invariance — the R1 probe property that failed pre-fix
        # (raw path gave exactly 100× here).
        scaled = loss_of(mb_advantages * 100.0)
        assert torch.allclose(base, scaled, rtol=1e-4), \
            f"pg_loss not scale-invariant: {base.item():.6f} vs {scaled.item():.6f}"

        # Idempotence: feeding already-stock-normalized advantages must give
        # the same loss (normalizing a normalized tensor is a no-op).
        pre_norm = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
        assert torch.allclose(base, loss_of(pre_norm), rtol=1e-4)

        # prio-IS: uniform prio ≡ no prio; non-uniform prio must change the loss.
        assert torch.allclose(base, loss_of(mb_advantages, prio=torch.ones(B)), rtol=1e-6)
        skew = torch.linspace(0.2, 2.0, B)
        assert not torch.allclose(base, loss_of(mb_advantages, prio=skew), rtol=1e-3), \
            "non-uniform mb_prio had no effect on pg_loss — prio-IS weight still discarded"
    finally:
        env.close()


# ── LSTM-BPTT correctness (rollout ↔ training consistency) ──────────────────
#
# Background: `use_rnn: True` means PufferLib's rollout threads LSTM state
# tick-by-tick through forward_eval (state carried across the 64-tick
# bptt_horizon segment, zeroed at each evaluate() start, reset on done).
# The PPO update re-evaluates the SAME 64-tick segments via forward() on
# (segments, bptt_horizon, OBS_DIM) batches. If forward() processes each
# tick statelessly (LSTM seq-len 1, zero state — the pre-fix behaviour),
# the recomputed log-probs/values systematically diverge from what the
# rollout stored: importance ratios ≠ 1 before any gradient step, and the
# LSTM's recurrent weights never receive through-time gradients (the LSTM
# degenerates to an expensive MLP layer). These tests pin the fixed
# contract: forward() must run true BPTT over the time dimension and match
# the stepwise forward_eval rollout exactly (fp32 tolerance).


def test_policy_forward_bptt_matches_stepwise_rollout():
    """forward() on a (B, T, OBS) segment must reproduce, tick for tick, what
    forward_eval() produces when threading LSTM state stepwise over the same
    sequence (zero initial state, no dones). This is THE consistency property
    PPO needs: rollout stores logprobs/values from forward_eval; the update
    recomputes them via forward(). Pre-fix, forward() was stateless per tick
    and this diverges from t=1 onwards."""
    import torch

    env = make_env(seed=0)
    try:
        policy = policy_mod.build_policy(env, device='cpu')
        policy.eval()
        torch.manual_seed(0)
        B, T = 3, 6
        x_seq = torch.randn(B, T, spec_obs.OBS_DIM)

        # Stepwise rollout path: forward_eval threads lstm_h/lstm_c via state.
        state = {"done": torch.zeros(B)}
        step_logits, step_mu, step_value = [], [], []
        with torch.no_grad():
            for t in range(T):
                logits, mu, _log_std, value = policy.forward_eval(x_seq[:, t, :], state)
                step_logits.append(torch.cat(logits, dim=-1))
                step_mu.append(mu)
                step_value.append(value)

        # Training path: one forward() call over the whole segment.
        with torch.no_grad():
            logits_b, mu_b, _log_std_b, value_b = policy.forward(x_seq, state={})
        flat_logits = torch.cat(logits_b, dim=-1)      # (B*T, sum(heads))

        for t in range(T):
            for b in range(B):
                flat_idx = b * T + t                                                          # row-major (B, T) flatten
                assert torch.allclose(flat_logits[flat_idx], step_logits[t][b], atol=1e-5), (
                    f"logits diverge at b={b} t={t}: "
                    f"max diff {(flat_logits[flat_idx] - step_logits[t][b]).abs().max():.2e}")
                assert torch.allclose(mu_b[flat_idx], step_mu[t][b],
                                      atol=1e-5), (f"mu_aim diverges at b={b} t={t}")
                assert torch.allclose(value_b[flat_idx], step_value[t][b],
                                      atol=1e-5), (f"value diverges at b={b} t={t}")
    finally:
        env.close()


def test_policy_forward_bptt_carries_memory():
    """Two sequences with identical final-tick obs but different histories
    must produce different final-tick outputs — i.e. forward() actually
    propagates LSTM state across ticks. Pre-fix (stateless per-tick LSTM)
    the final-tick outputs are bitwise identical and this test fails."""
    import torch

    env = make_env(seed=0)
    try:
        policy = policy_mod.build_policy(env, device='cpu')
        policy.eval()
        torch.manual_seed(1)
        T = 5
        last_obs = torch.randn(1, spec_obs.OBS_DIM)
        hist_a = torch.zeros(1, T - 1, spec_obs.OBS_DIM)
        hist_b = torch.randn(1, T - 1, spec_obs.OBS_DIM)
        seq_a = torch.cat([hist_a, last_obs.unsqueeze(1)], dim=1)      # (1, T, OBS)
        seq_b = torch.cat([hist_b, last_obs.unsqueeze(1)], dim=1)

        with torch.no_grad():
            _, _, _, value_a = policy.forward(seq_a, state={})
            _, _, _, value_b = policy.forward(seq_b, state={})
        # Final tick = flat row T-1 (row-major (B=1, T) flatten).
        diff = (value_a[T - 1] - value_b[T - 1]).abs().max().item()
        assert diff > 1e-6, (
            f"final-tick value identical ({diff:.2e}) despite different histories — "
            f"forward() is not carrying LSTM state across ticks (no BPTT)")
    finally:
        env.close()


def test_policy_forward_bptt_resets_on_terminal():
    """state['terminals'] (B, T) must reset the LSTM state at done ticks,
    mirroring the rollout's (1-done)*state masking in _forward_core. A done
    at tick k for row 0 means ticks k..T-1 of row 0 must equal a fresh
    zero-state forward() over just that suffix; rows without dones must be
    unaffected by the masking path."""
    import torch

    env = make_env(seed=0)
    try:
        policy = policy_mod.build_policy(env, device='cpu')
        policy.eval()
        torch.manual_seed(2)
        B, T, k = 2, 6, 3
        x_seq = torch.randn(B, T, spec_obs.OBS_DIM)
        terminals = torch.zeros(B, T)
        terminals[0, k] = 1.0          # row 0 episode ends before tick k

        with torch.no_grad():
            _, _, _, value_masked = policy.forward(x_seq, state={"terminals": terminals})
            # Reference A: row 0's suffix from a fresh zero state.
            _, _, _, value_suffix = policy.forward(x_seq[0:1, k:, :], state={})
            # Reference B: the full batch with no terminals at all.
            _, _, _, value_plain = policy.forward(x_seq, state={})

        for t in range(k, T):
            assert torch.allclose(
                value_masked[0 * T + t], value_suffix[t - k],
                atol=1e-5), (f"row 0 tick {t}: masked output != fresh-suffix output — "
                             f"terminal reset not applied in BPTT")
        for t in range(T):
            assert torch.allclose(
                value_masked[1 * T + t], value_plain[1 * T + t],
                atol=1e-5), (f"row 1 tick {t}: no-done row was perturbed by the masking path")
    finally:
        env.close()


def test_train_path_logprobs_match_rollout():
    """End-to-end: after one real evaluate() rollout, re-evaluating the FULL
    buffer through _hybrid_ppo_loss (the exact training-path forward, with
    state carrying mb_terminals) must reproduce the rollout-stored logprobs
    and values. This is the importance-ratio==1-at-epoch-start invariant;
    pre-fix the stateless training forward breaks it by construction."""
    import torch

    from cs2rl.train.update import _hybrid_ppo_loss
    from tests._helpers.trainer_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=False)
    try:
        trainer.evaluate()
        state = dict(
            action=trainer.actions,
            lstm_h=None,
            lstm_c=None,
            terminals=trainer.terminals,
        )
        with torch.no_grad():
            _pg, _ent, newvalue, newlogprob, _rd, _rc, _lg = _hybrid_ppo_loss(
                trainer.policy,
                trainer.observations,
                trainer.actions,
                trainer.cont_actions,
                trainer.logprobs_d,
                trainer.logprobs_c,
                torch.zeros_like(trainer.logprobs),
                0.15,
                state,
                                                                                                   # F8: the harness rollout samples MASKED; re-evaluating the
                                                                                                   # buffer must use the same stored masks or the logprobs
                                                                                                   # diverge by construction (that divergence is itself pinned
                                                                                                   # by test_sampler_and_loss_mask_consistency's control case).
                mb_masks=trainer.action_masks,
            )
        newlogprob = newlogprob.reshape(trainer.logprobs.shape)
        newvalue = newvalue.reshape(trainer.values.shape)
        lp_diff = (newlogprob - trainer.logprobs).abs().max().item()
        v_diff = (newvalue - trainer.values).abs().max().item()
        assert lp_diff < 1e-3, (f"training-path logprobs diverge from rollout by {lp_diff:.4f} — "
                                f"PPO ratios != 1 at epoch start (rollout/training LSTM mismatch)")
        assert v_diff < 1e-3, (f"training-path values diverge from rollout by {v_diff:.4f}")
    finally:
        cleanup()


# ── F8 (2026-07-06 adversarial review): action-mask consumption ──────────────
# The C env computed per-agent action masks every tick since Batch 2, but no
# trainer-path code ever consumed them. F8 threads them through:
#   compute_masks (C, also at reset)  →  Cs2Env._attach_mask_view shm slice
#   →  trainer._action_mask_view_main  →  _hybrid_sample_logits(mask=...)
#   →  trainer.action_masks rollout buffer  →  _hybrid_ppo_loss(mb_masks=...)
# The tests below pin each contract. The former USE-mask test above was
# vacuous (N1: asserted the crouch slot); its offset is now derived + pinned.


def test_hybrid_sample_logits_respects_masks():
    """Masked bins must never be sampled, their probability mass must be 0 in
    log_prob, and a fully-no-op'd (dead-agent-style) row must have discrete
    entropy exactly 0. Reference distribution: torch Categorical on -inf
    masked logits (the sampler uses finfo.min/2 instead of -inf to keep
    entropy NaN-free — same distribution, different fill)."""
    import torch

    from cs2rl.policy import _MASK_HEAD_SLICES, _hybrid_sample_logits

    torch.manual_seed(7)
    B = 64
    head_sizes = spec_action.ACTION_HEAD_SIZES
    mask_dim = sum(head_sizes)
    logits_list = [torch.randn(B, n) for n in head_sizes]
    mu_aim = torch.zeros(B, 2)
    log_std_aim = torch.full((2, ), -2.30)

    # Row 0: dead-agent pattern — only bin 0 of each head valid.
    # Other rows: random masks with bin 0 always valid (C invariant).
    mask = (torch.rand(B, mask_dim) > 0.4)
    for (lo, _hi) in _MASK_HEAD_SLICES:
        mask[:, lo] = True
    mask[0] = False
    for (lo, _hi) in _MASK_HEAD_SLICES:
        mask[0, lo] = True

    for trial in range(20):
        action, _cont, lp_d, _lp_c, ent_d, _ent_c = _hybrid_sample_logits(
            (logits_list, mu_aim, log_std_aim, None),
            max_turn_speed=0.7853981633974483,
            mask=mask,
        )
        for h, (lo, _hi) in enumerate(_MASK_HEAD_SLICES):
            picked_valid = mask[torch.arange(B), lo + action[:, h]]
            assert picked_valid.all(), (f"trial {trial}: head {h} sampled a masked bin at rows "
                                        f"{(~picked_valid).nonzero().flatten().tolist()}")
        # Dead-style row: exactly one valid bin per head ⇒ H = 0, action = no-ops.
        assert (action[0] == 0).all(), f"dead-style row sampled non-no-op: {action[0]}"
        assert abs(ent_d[0].item()) < 1e-5, f"dead-style row entropy_d={ent_d[0].item()}"
        assert torch.isfinite(lp_d).all() and torch.isfinite(ent_d).all()

    # log_prob reference vs torch.distributions on -inf-masked logits.
    fixed_action = action
    _a, _c, lp_d_eval, _lpc, _ed, _ec = _hybrid_sample_logits(
        (logits_list, mu_aim, log_std_aim, None),
        action=fixed_action,
        continuous_action=_cont,
        mask=mask,
    )
    ref = 0.0
    for h, (lo, hi) in enumerate(_MASK_HEAD_SLICES):
        ref_logits = logits_list[h].masked_fill(~mask[:, lo:hi], float("-inf"))
        ref = ref + torch.distributions.Categorical(logits=ref_logits).log_prob(fixed_action[:, h])
    assert torch.allclose(lp_d_eval, ref,
                          atol=1e-5), (f"masked log_prob_d drift vs reference: "
                                       f"{(lp_d_eval - ref).abs().max().item():.2e}")


def test_sampler_and_loss_mask_consistency():
    """Rollout sampler and PPO-update loss must see the SAME masked
    distribution: feeding the sampler's (action, logp) back into
    _hybrid_ppo_loss with the same mb_masks must give ratio ≈ 1 exactly.
    This is the invariant that breaks silently if masking is applied on one
    side only (the failure mode the F8 docstrings warn about)."""
    import torch

    from cs2rl.policy import _hybrid_sample_logits
    from cs2rl.train.update import _hybrid_ppo_loss

    torch.manual_seed(11)
    B = 32
    head_sizes = spec_action.ACTION_HEAD_SIZES
    mask_dim = sum(head_sizes)
    logits_list = [torch.randn(B, n) for n in head_sizes]
    mu_aim = torch.randn(B, 2) * 0.1
    log_std_aim = torch.full((2, ), -2.30)
    value = torch.zeros(B, 1)

    from cs2rl.policy import _MASK_HEAD_SLICES
    mask = (torch.rand(B, mask_dim) > 0.3)
    for (lo, _hi) in _MASK_HEAD_SLICES:
        mask[:, lo] = True

    class _StubPolicy:

        def __call__(self, mb_obs, state):
            return logits_list, mu_aim, log_std_aim, value

    action, cont, lp_d, lp_c, _, _ = _hybrid_sample_logits(
        (logits_list, mu_aim, log_std_aim, value),
        max_turn_speed=0.7853981633974483,
        mask=mask,
    )
    pg_loss, entropy, _v, _lp_tot, ratio_d, ratio_c, _lg = _hybrid_ppo_loss(
        _StubPolicy(),
        torch.zeros(B, 4),                                                                      # obs unused by the stub
        action,
        cont,
        lp_d,
        lp_c,
        torch.randn(B),
        clip_coef=0.2,
        state={},
        mb_masks=mask,
    )
    assert torch.allclose(ratio_d, torch.ones(B),
                          atol=1e-5), (f"ratio_d != 1 with identical mask on both sides: "
                                       f"max dev {(ratio_d - 1).abs().max().item():.2e}")
    assert torch.allclose(ratio_c, torch.ones(B), atol=1e-5)
    assert torch.isfinite(pg_loss).all() and torch.isfinite(entropy).all()
                                                                                                # F16: the loss returns the logits it computed (7th element) so the
                                                                                                # trainer skips the redundant diagnostic forward. With mb_masks given
                                                                                                # they must be the MASKED logits — masked bins pushed to huge negatives.
    from cs2rl.policy import _MASK_HEAD_SLICES as _slices
    for h, (lo, hi) in enumerate(_slices):
        head_mask = mask[:, lo:hi]
        if (~head_mask).any():
            assert (_lg[h][~head_mask]
                    < -1e30).all(), (f"head {h}: returned logits not masked — F16 diagnostics "
                                     f"would report the unmasked distribution")

    # Control: DROP the mask on the loss side → ratios must deviate wherever
    # a mask bit was 0 (proves the consistency requirement is load-bearing).
    _pg2, _e2, _v2, _lpt2, ratio_d_unmasked, _rc2, _lg2 = _hybrid_ppo_loss(
        _StubPolicy(),
        torch.zeros(B, 4),
        action,
        cont,
        lp_d,
        lp_c,
        torch.randn(B),
        clip_coef=0.2,
        state={},
        mb_masks=None,
    )
    assert not torch.allclose(ratio_d_unmasked, torch.ones(B), atol=1e-3), (
        "unmasked loss over masked rollout produced ratio 1 — mask had no "
        "distributional effect; test premises are broken")


def test_env_publishes_masks_after_reset_and_step():
    """Cs2Env._attach_mask_view + the step()/reset() copy-out: the shm slice
    must mirror _masks_view after reset (fresh spawn masks — the pre-F8 code
    had NO mask computation at reset at all) and after a step that kills an
    agent (dead row = per-head no-ops, the C invariant the sampler needs)."""
    from multiprocessing import RawArray

    from cs2rl.spec.action import ACTION_MASK_DIM

    env = make_env(seed=0)
    try:
        n_agents = 10
        shm = RawArray("b", n_agents * ACTION_MASK_DIM)
        env._attach_mask_view(shm, 0)
        view = np.frombuffer(shm, dtype=np.int8).reshape(n_agents, ACTION_MASK_DIM)

        env.reset()
        assert (view == env._masks_view).all(), "shm != _masks_view after reset"
        assert view.sum() > n_agents, (
            "masks after reset look empty — env_reset no longer calls compute_masks?")
        # All agents alive at spawn: move head fully valid.
        assert view[:, :9].all(), "alive agents should have the full move head valid"

        g = env._c_env.game
        g.agents[0].hp = 0
        g.agents[0].alive = 0
        actions = np.zeros((n_agents, len(spec_action.ACTION_HEAD_SIZES)), dtype=np.int32)
        env.step(actions)
        assert (view == env._masks_view).all(), "shm != _masks_view after step"
        offs = np.cumsum((0, ) + tuple(spec_action.ACTION_HEAD_SIZES))[:-1]
        dead = view[0]
        assert dead.sum() == len(spec_action.ACTION_HEAD_SIZES), (
            f"dead agent should have exactly one valid bin per head, got {dead.tolist()}")
        assert all(dead[o] == 1
                   for o in offs), (f"dead agent per-head no-ops not set: {dead.tolist()}")
    finally:
        env.close()


# ── F13 (2026-07-06 adversarial review): USE counter wired ───────────────────
def test_action_use_counter_wired():
    """action_use was declared + exported since Batch 2 but had NO writer in C
    (USE is processed in process_bomb, which never called count_action) — the
    W&B metric was permanently 0 and misleading when diagnosing plant
    behaviour. Now counted like every other head: intent of alive agents.
    (action_last was removed outright: no head, no writer, dead legacy.)"""
    env = make_env(seed=0)
    try:
        env.reset()
        use_head = list(spec_action.ACTION_HEAD_NAMES).index("use")
        acts = np.zeros((10, len(spec_action.ACTION_HEAD_SIZES)), dtype=np.int32)
        acts[:, use_head] = 1
        for _ in range(5):
            env.step(acts)
        ss = env._c_env.step_stats
        es = env._c_env.episode_stats
        alive = sum(1 for i in range(10) if env._c_env.game.agents[i].alive)
        assert list(ss.action_use) == [
            0, alive
        ], (f"per-tick USE counter wrong: {list(ss.action_use)} (alive={alive})")
        assert list(es.action_use) == [0, 5 * alive
                                       ], (f"episode USE counter wrong: {list(es.action_use)}")
        info = env._build_terminal_info()
        assert "action_use_1" in info and "action_last_0" not in info, (
            "terminal info should export action_use_* and no longer export action_last_*")
    finally:
        env.close()


# ── F9 (2026-07-06 adversarial review): strafe labels match geometry ─────────
def test_strafe_labels_match_geometry():
    """With facing = 0 (+X) in this x-east/y-north, CCW-yaw frame:
    W (bin 1) must move +X, D (bin 3, 'right') must move -Y (clockwise
    perpendicular = geometric right), A (bin 7, 'left') +Y, S (bin 5) -X.
    The pre-F9 basis rotated the strafe axis CCW, so A/D were mirrored vs
    their labels — invisible to self-play (relabeling-invariant) but wrong
    for scripted experts, BC demos, and deploy key export."""
    import math

    from cs2rl.env.c.cs2_env import make_env

    env = make_env(seed=0, auto_reset=False)
    try:
        g = env._c_env.game
        sd = env._c_env.sd.contents
        # An open bombsite-area centroid (R4 recipe) — flat, no walls nearby.
        id2idx = {int(aid): i for i, aid in enumerate(env.map_data.area_ids)}
        open_area = next(aid for aid in env.map_data.area_ids
                         if sd.bombsite_by_idx[id2idx[int(aid)]])
        cx, cy = env.map_data.centroids[open_area][:2]

        expected = {1: 0.0, 3: -90.0, 5: 180.0, 7: 90.0}                                      # bin → world angle (deg)
        for move_bin, want_deg in expected.items():
            env.reset()
            a = g.agents[0]
            a.alive, a.hp = 1, 100
            a.x, a.y, a.z = float(cx), float(cy), 0.0
            a.area_idx = id2idx[int(open_area)]
            a.facing = 0.0
            a.aim_rad = 0.0
            a.vx = a.vy = a.vz = 0.0
            acts = np.zeros((10, 7), dtype=np.int64)
            acts[0, 0] = move_bin
            env.step(acts)
            speed = math.hypot(a.vx, a.vy)
            assert speed > 1.0, f"bin {move_bin}: agent did not move (v={a.vx},{a.vy})"
            got_deg = math.degrees(math.atan2(a.vy, a.vx))
            diff = (got_deg - want_deg + 180.0) % 360.0 - 180.0
            assert abs(diff) < 1.0, (
                f"bin {move_bin}: velocity angle {got_deg:.1f}° != expected {want_deg:.1f}° "
                f"(A/D strafe basis regressed — see F9)")
    finally:
        env.close()


# ── F10 (2026-07-06 adversarial review): enemy-slot sort must not leak ───────
def test_enemy_slot_sort_does_not_leak_invisible_rank():
    """Enemy obs slots are distance-sorted, but the sort key must only use
    information the policy legitimately has: true distance when visible,
    last-known memory position when not, sentinel-far otherwise. Pre-F10 the
    key was the TRUE distance for all 5 enemies unconditionally, so an unseen
    enemy walking closer visibly reordered the slots (rank leak).

    Invisibility here is forced via area_idx = -1 — build_vis_matrix
    short-circuits off-mesh agents to can_see=0 regardless of position."""
    from cs2rl.env.c.cs2_env import make_env
    from cs2rl.spec.obs import OBS_BLOCKS, OBS_ENEMY_STRIDE

    env = make_env(seed=0, auto_reset=False)
    try:
        g = env._c_env.game
        obs_view = env._obs_view
        enemy_base = OBS_BLOCKS["enemy"][0]
        observer = g.agents[0]         # T slot 0; enemies are 5-9

        def slot_flags():
            """(can_see, alive) per enemy obs slot of agent 0."""
            return [(int(obs_view[0, enemy_base + s * OBS_ENEMY_STRIDE + 3]),
                     int(obs_view[0, enemy_base + s * OBS_ENEMY_STRIDE + 4])) for s in range(5)]

        env.reset()
        # Wipe observer's enemy memory so the sentinel branch is exercised.
        for m in range(5):
            observer.enemy_mem_idx[m] = -1             # INVALID_AREA_IDX
                                                       # Enemy 5: alive, ON-mesh, right next to the observer (trivial LoS).
        vis_enemy = g.agents[5]
        vis_enemy.alive, vis_enemy.hp = 1, 100
        vis_enemy.x, vis_enemy.y = observer.x + 40.0, observer.y
        vis_enemy.area_idx = observer.area_idx
                                                       # Enemies 6-9: alive but OFF-mesh (area_idx=-1 → can_see forced 0).
                                                       # Enemy 6 sits CLOSER than the visible one — pre-F10 it stole slot 0.
        for j in range(6, 10):
            g.agents[j].alive, g.agents[j].hp = 1, 100
            g.agents[j].area_idx = -1
            g.agents[j].x, g.agents[j].y = observer.x + 500.0, observer.y
        g.agents[6].x = observer.x + 5.0

        acts = np.zeros((10, 7), dtype=np.int64)
        env.step(acts)
        flags = slot_flags()
        assert flags[0] == (1, 1), (
            f"slot 0 must hold the VISIBLE enemy (can_see=1); got slots {flags} — "
            f"a closer invisible enemy outranked it (F10 leak)")
        assert all(cs == 0 for cs, _ in flags[1:]), f"only one enemy is visible: {flags}"

        # Unseen movement must not reorder: drag the invisible enemy through
        # the observer and re-step several times — slot 0 stays the visible one.
        for new_dx in (2.0, 1.0, 0.5):
            g.agents[6].x = observer.x + new_dx
            env.step(acts)
            assert slot_flags()[0] == (1, 1), (
                f"invisible enemy at dx={new_dx} reordered the slots — rank leak")
    finally:
        env.close()
