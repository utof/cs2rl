# tests/integration/test_pufferlib_pin.py
#
# Canary tests for the PufferLib dependency surface (gh #85).
#
# WHY: Cs2PuffeRL overrides PuffeRL's evaluate()/train()/save_checkpoint()
# and still extends rollout storage and vecenv plumbing via the hybrid-aim
# patcher. This is a de-facto fork of the hot loops pinned to upstream 3.0.0.
# Upstream 4.0 deletes pufferlib.vector and pufferlib.pytorch and rewrites the
# trainer, so an accidental version bump would produce confusing failures.
# These tests turn that into one clear failure pointing at the upgrade playbook
# and pin the upstream invariant the hybrid-aim patcher still relies on.

import pufferlib


def test_pufferlib_version_pinned():
    """One clear failure on an accidental pufferlib bump.

    If this fails you (or a relock) changed the pufferlib version. Upgrading is
    a deliberate migration, not a bump: 4.0 deletes pufferlib.vector/pytorch
    (both imported by src/cs2rl/train.py), moves the trainer to torch_pufferl, and
    adds native self-play. Read the playbook in gh #85 before touching the pin
    in pyproject.toml.
    """
    # NB: 3.0.0 ships __version__ as the FLOAT 3.0 (not a string) — compare
    # via str() so this survives either representation without false alarms.
    assert str(pufferlib.__version__) == "3.0", (
        f"pufferlib version changed to {pufferlib.__version__!r} (expected '3.0'). "
        "The trainer subclass and remaining hybrid-aim patch target 3.0.0 "
        "internals. See gh #85 for the upgrade playbook before proceeding.")


def test_segments_equals_total_agents_invariant():
    """Pin segments == total_agents on a real harness-built trainer.

    This equality makes BPTT zero-initial-state EXACT (each agent row fills
    exactly one buffer segment per evaluate(); see Dust2Policy.forward /
    _lstm_bptt in src/cs2rl/train.py). Upstream only enforces <=; our equality holds
    by construction in compute_batch_dims. The construction-time assert in
    Cs2PuffeRL._init_return_norm (src/cs2rl/trainer.py, gh#168 W2a) guards production;
    this test guards the harness/config path and documents the invariant where
    reviewers look.
    """
    from tests._helpers.trainer_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=False)
    try:
        assert trainer.segments == trainer.total_agents, (
            f"segments={trainer.segments} total_agents={trainer.total_agents} — "
            "BPTT zero-init exactness broken; see gh #85 and the assert in "
            "Cs2PuffeRL._init_return_norm.")
    finally:
        cleanup()
