"""gh#90 regression tests — KL early-stop metrics scaling + break granularity.

Background (root-caused 2026-08-01, run checkpoints-20260801-022606): every
``losses/*`` metric used to accumulate ``value / self.total_minibatches``, but
the target_kl early-stop ``break`` sat BEFORE the logging block — so when the
update loop exited after k of N minibatches, every logged loss was silently
scaled by k/N. In that run "importance=0.0167" was really ratio=1.0 with k=1.
Additionally the flattened minibatch loop (all update_epochs collapsed into
one ``range``) aborted passes over data never visited — harsher than standard
PPO, which finishes the current epoch before stopping.

Contract pinned here (implemented in Cs2PuffeRL.train, src/trainer.py):
  1. losses/* are normalized by the EXECUTED minibatch count, not the planned
     total — so a truncated update reports true per-minibatch means.
  2. ``losses["minibatches_run"]`` reports the executed count.
  3. The KL break is gated to update-epoch boundaries: a KL trip mid-epoch
     finishes the current pass (total_minibatches // update_epochs minibatches)
     before stopping, and can never truncate epoch 0 mid-pass.

Uses the minimal harness (src/train_test_harness.py), whose trainer is
Cs2PuffeRL (gh#168 W1.5) whose train() is the return-norm body (a method since
gh#168 W2a), same pattern as tests/test_train_env.py. One harness build serves all scenarios
(builds cost ~5s each on the VM).
"""


def _run_train_once(trainer):
    """One evaluate()+train() cycle; force the log-flush path so the losses
    dict lands on trainer.losses regardless of wall-clock timing."""
    trainer.evaluate()
    trainer.last_log_time = 0.0
    trainer.train()
    return trainer.losses


def test_kl_break_metrics_and_granularity():
    from train_test_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        update_epochs = int(trainer.config["update_epochs"])
        total_mb = int(trainer.total_minibatches)
        mbs_per_epoch = max(1, total_mb // max(1, update_epochs))
        assert total_mb > mbs_per_epoch, (
            "harness config must have update_epochs > 1 for the granularity "
            "assertions below to discriminate anything")

        # ── Scenario A: no KL trip → all planned minibatches run ────────────
        trainer.config["target_kl"] = 1e9
        losses = _run_train_once(trainer)
        assert losses["minibatches_run"] == total_mb, (
            f"untripped update must run all {total_mb} minibatches, "
            f"ran {losses['minibatches_run']}")
        # First update after a fresh rollout: new policy == rollout policy,
        # so the joint importance ratio is ~1.0. Under the old bug this
        # metric read k/total_mb instead.
        assert abs(losses["importance"] -
                   1.0) < 0.05, (f"importance should be ~1.0 on-policy, got {losses['importance']}")

        # ── Scenario B: KL trips on mb 0 → finish epoch 0, then stop ────────
        # approx_kl = mean((r-1)-log r) >= 0 pointwise, so a negative target
        # trips deterministically on the very first minibatch.
        trainer.config["target_kl"] = -1.0
        losses = _run_train_once(trainer)
        assert losses["minibatches_run"] == mbs_per_epoch, (
            f"KL trip on mb 0 must still finish epoch 0 "
            f"({mbs_per_epoch} minibatches), ran {losses['minibatches_run']}")
        # The k/N-scaling regression: even with a truncated update, the
        # logged importance is a true mean over EXECUTED minibatches (~1.0
        # right after a rollout), not scaled down by k/total_minibatches.
        assert abs(losses["importance"] -
                   1.0) < 0.05, (f"truncated update must not scale losses/* by k/N; "
                                 f"importance={losses['importance']}")
        # entropy/total normalization uses the same divisor — pin one more
        # key so a partial fix (importance only) can't pass.
        assert abs(losses["entropy/total"] - losses["entropy"]) < 1e-6, (
            "entropy and entropy/total accumulate the same quantity and must "
            "agree under the executed-count divisor")
    finally:
        cleanup()


def test_clipfrac_halves_and_event_fraction_are_logged():
    from train_test_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        trainer.config["target_kl"] = 1e9
        losses = _run_train_once(trainer)
        assert "clipfrac_d" in losses
        assert "clipfrac_c" in losses
        assert "event_oversample_fraction" in losses
        assert 0.0 <= losses["clipfrac_d"] <= 1.0
        assert 0.0 <= losses["clipfrac_c"] <= 1.0
        assert 0.0 <= losses["event_oversample_fraction"] <= 1.0
        # This harness sets include_step_stats_in_info=True, so the fraction
        # is NOT the production #100-closed zero. Do not assert == 0.0 here.
        # Exact equality pins the persist AFTER the `_mb_run` divisor: a
        # pre-divisor write would be divided by executed minibatches and
        # no longer match the per-call Task 8 scalar.
        assert isinstance(losses["event_oversample_fraction"], float)
        assert isinstance(trainer._batch1_event_oversample_fraction, float)
        assert (losses["event_oversample_fraction"] == trainer._batch1_event_oversample_fraction)
        assert "clipfrac" in losses
    finally:
        cleanup()


def test_self_play_used_past_metric():
    from train import self_play_used_past_metric

    class _T:
        _selfplay_used_past = True

    class _F:
        _selfplay_used_past = False

    class _U:
        pass

    # Persist filter drops non-floats; a raw bool would still pass `== 1.0`.
    used = self_play_used_past_metric(_T())
    unused = self_play_used_past_metric(_F())
    missing = self_play_used_past_metric(_U())
    assert isinstance(used, float) and used == 1.0
    assert isinstance(unused, float) and unused == 0.0
    assert isinstance(missing, float) and missing == 0.0


def test_self_play_used_past_is_assigned_on_outer_logs():
    """The persist site is the outer logs dict, not trainer.losses."""
    import inspect

    import train
    src = inspect.getsource(train)
    assert 'logs["self_play/used_past"] = self_play_used_past_metric(trainer)' in src
    assert src.index('logs["self_play/pool_size"]') < src.index('logs["self_play/used_past"]')
