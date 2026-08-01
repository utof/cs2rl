"""Warm-start entropy mode — trainer-contract tests (spec 2026-08-01 §5.2).

Uses the minimal harness + explicit _patch_trainer_with_return_norm, same
pattern as tests/test_kl_break_metrics.py. Config keys are injected into
trainer.config before patching (ordering is actually indifferent — the patch
init block only seeds attributes; the keys are read per train() call — but
before-patching keeps the setup unambiguous).

PITFALL: trainer.losses is a defaultdict(float) — losses["warmstart_phase"]
== 0 would be vacuously true on a missing key. Always assert membership
BEFORE any value comparison against 0/WS_GRACE.

PITFALL (spec finding 8): assert on trainer._batch1_effective_alpha and
losses["log_alpha"], NOT losses["alpha"] — losses/alpha logs RAW alpha,
which sits at ent_coef=0.1 during grace by design (Task 9B reset).
"""
import math

import pytest

# "log_alpha did not move" tolerance. NOT 0/1e-9: log_alpha is a float32
# tensor, so log_alpha.item() round-trips log(0.1) to ~3.2e-8 of the float64
# math.log(0.1) the assertions compare against. 1e-7 absorbs that while
# staying three orders of magnitude below real controller motion — Adam at
# lr=1e-4 moves log_alpha ~1e-4 per minibatch (~7e-4 over one update at the
# harness's 7 minibatches), which is what test_mode_off_is_unchanged_behavior
# asserts on the other side of the contract.
_FROZEN_TOL = 1e-7


def _run_train_once(trainer):
    trainer.evaluate()
    trainer.last_log_time = 0.0
    trainer.train()
    return trainer.losses


def _build_ws_trainer(num_envs=32, **ws_overrides):
    from train import _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=num_envs, with_selfplay=True)
    trainer.config["warmstart_entropy"] = True
    trainer.config["warmstart_alpha_ceiling"] = 0.0
    trainer.config.update(ws_overrides)
    _patch_trainer_with_return_norm(trainer)
    return trainer, cleanup


def test_grace_pins_effective_alpha_and_freezes_log_alpha():
    trainer, cleanup = _build_ws_trainer(warmstart_grace_steps=10**12,
                                         warmstart_ramp_steps=10_000_000)
    try:
        losses = _run_train_once(trainer)
        # defaultdict(float): membership first!
        assert "warmstart_phase" in losses
        assert losses["warmstart_phase"] == 0
        assert trainer._batch1_effective_alpha <= 1e-8, (
            "grace must ceiling effective alpha to the (0.0) ceiling; "
            f"got {trainer._batch1_effective_alpha}")
        # alpha optimizer paused: log_alpha stays at the Task 9B reset value
        assert abs(losses["log_alpha"] - math.log(trainer.config["ent_coef"])) < _FROZEN_TOL
        losses2 = _run_train_once(trainer)
        assert abs(losses2["log_alpha"] - math.log(trainer.config["ent_coef"])) < _FROZEN_TOL
        # Value pin, not mere presence: h_over_h0 is an ABSOLUTE ratio and
        # must stay ~1 over two updates. If it ever migrates above the gh#90
        # divisor loop it gets divided by the executed-minibatch count (7 on
        # this harness), landing near 0.14 — well outside rel=0.5.
        assert "warmstart_h_over_h0" in losses2
        assert losses2["warmstart_h_over_h0"] == pytest.approx(1.0, rel=0.5)
        # the anchor source must be maintained every update, unconditionally
        assert trainer._batch1_last_entropy_mean is not None
    finally:
        cleanup()


def test_grace_zero_anchors_on_second_update_and_ramps():
    trainer, cleanup = _build_ws_trainer(warmstart_grace_steps=0, warmstart_ramp_steps=10**12)
    try:
        # no prior update -> no anchor source yet -> still GRACE
        losses1 = _run_train_once(trainer)
        assert losses1["warmstart_phase"] == 0
        h_after_1 = trainer._batch1_last_entropy_mean
        # anchors to update 1's mean H
        losses2 = _run_train_once(trainer)
        assert losses2["warmstart_phase"] == 1
        assert abs(trainer._batch1_warmstart_h_anchor - h_after_1) < 1e-9
        # ramp_steps is huge so the mirrored target sits ~at the anchor
        assert abs(trainer._batch1_current_target_entropy - h_after_1) < 1e-3
        # COUPLING: the assertion above only proves the MIRROR (the wandb
        # trace) was overridden. This one proves the CONSUMED target — the
        # _t9_target_entropy that alpha_loss is actually computed from — was
        # overridden too, which is the half that steers training.
        # alpha_loss = mean(log_alpha * (H - target)). With the target anchored
        # at the previous update's mean H, (H - target) ~ 0, so alpha_loss ~ 0
        # (measured -0.020). Drop the consumed override while keeping the
        # mirror and the target reverts to the Task 9A schedule, which this
        # early in warmup reads ~0.4997*max = 4.10 nats against H ~ 1.55:
        # alpha_loss jumps to +5.89 (measured). The bound sits at 1.151
        # (|log 0.1| * 0.5): 5x below the broken value, 59x above the correct
        # one — verified by breaking the override and watching this fail.
        assert abs(losses2["alpha_loss"]) < abs(math.log(trainer.config["ent_coef"])) * 0.5, (
            "consumed target was not overridden — alpha_loss "
            f"{losses2['alpha_loss']} implies the Task 9A schedule is still "
            "driving the controller during RAMP")
    finally:
        cleanup()


def test_mode_off_is_unchanged_behavior():
    from train import _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        # no warmstart keys at all
        _patch_trainer_with_return_norm(trainer)
        losses = _run_train_once(trainer)
        assert "warmstart_phase" not in losses
        assert "warmstart_h_over_h0" not in losses
        # controller is live: with target far from H the alpha optimizer steps
        assert abs(losses["log_alpha"] - math.log(trainer.config["ent_coef"])) > 1e-6, (
            "with the mode off the alpha optimizer must actually step")
    finally:
        cleanup()
