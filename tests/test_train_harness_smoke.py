"""Smoke tests for the Batch-1 trainer test harness (Task 6b, utof/cs2rl#8).

These tests pin the attribute surface that Task 6c and Tasks 7-11 will consume
from a harness-built trainer. If this surface changes (PufferLib API drift or
our own patches rename things), these tests fail loudly at import-time for
downstream tasks rather than partway through their own logic.

Both tests construct a tiny Serial-backend trainer and run ONE evaluate()
round. Wall time per test must stay under ~5s; if it regresses past ~10s,
revisit the segments / num_envs defaults in _build_trainer_for_test.
"""


def test_build_trainer_for_test_smoke():
    """Harness produces a trainer with the Batch-1-consumed attribute surface.

    Covers the patch-off (no self-play) path. This is the mode used by Tasks
    7-11, which exercise trainer internals without the self-play opponent
    override.
    """
    from train_test_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=False)
    try:
        # Attribute surface consumed by Task 6c + Tasks 7-11. If any of these
        # disappear (PufferLib rename, upstream refactor), Batch 1 breaks.
        # Note: PuffeRL 3.0 exposes ``total_agents`` not ``num_envs`` — the
        # task-spec list has been reconciled with the real upstream surface.
        # ``full_rows`` is asserted AFTER evaluate() because it's zeroed inside
        # evaluate() and only set to its final value on exit.
        for attr in (
                "evaluate",
                "train",
                "rewards",
                "vecenv",
                "segments",
                "total_agents",
                "config",
                "lstm_h",
                "lstm_c",
                "global_step",
                "ep_lengths",
                "ep_indices",
                "epoch",
                "profile",
                "amp_context",
                "policy",
                                       # close is required by harness cleanup(); pinned so a PufferLib
                                       # rename silently no-ops the cleanup is caught here instead.
                "close",
        ):
            assert hasattr(trainer, attr), f"trainer missing attribute: {attr}"

        # One evaluate() round must succeed and advance global_step. This is
        # the minimal liveness check Task 6c needs before it can remove the
        # reward clamp and assert clipping-free behaviour.
        gs_before = trainer.global_step
        trainer.evaluate()
        assert trainer.global_step > gs_before, (
            f"evaluate() did not advance global_step (before={gs_before}, "
            f"after={trainer.global_step})")
        # full_rows is set inside evaluate(); it exists now.
        assert hasattr(trainer, "full_rows"), "trainer missing attribute: full_rows"
    finally:
        cleanup()


def test_build_trainer_for_test_with_selfplay():
    """Harness can engage self-play — required for Task 6c.

    Task 6c removed torch.clamp(r, -1, 1) from the self-play evaluate body
    (now Cs2PuffeRL.evaluate, gh#168 W2b), so the test it ships needs a
    trainer with self-play engaged. Here we just confirm evaluate() runs
    end-to-end on a tiny rollout.
    """
    from train_test_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        # Cs2PuffeRL.evaluate is the self-play rollout (gh#168 W2b). If it
        # raises (e.g. past_policy loading NRE), self-play is broken for the
        # harness — Task 6c can't proceed until this passes.
        trainer.evaluate()
    finally:
        cleanup()
