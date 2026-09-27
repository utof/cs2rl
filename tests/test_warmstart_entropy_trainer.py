"""Warm-start entropy mode — trainer-contract tests (spec 2026-08-01 §5.2).

Uses the minimal harness, whose trainer is Cs2PuffeRL (gh#168 W1.5), whose
train() is the return-norm body (a method since gh#168 W2a), same pattern as
tests/test_kl_break_metrics.py. Config keys are injected into trainer.config
AFTER construction; that is fine because Cs2PuffeRL._init_return_norm only
seeds attributes and the warmstart_* keys are read per train() call.

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
    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=num_envs, with_selfplay=True)
    trainer.config["warmstart_entropy"] = True
    trainer.config["warmstart_alpha_ceiling"] = 0.0
    trainer.config.update(ws_overrides)
    return trainer, cleanup


def _force_floor_above_entropy(trainer, floor=1e6):
    """Raise the trainer's entropy floor far above any achievable entropy (gh#96).

    WHAT: sets `trainer._entropy_floor` so `current_entropy.item() <
    self._entropy_floor` in Cs2PuffeRL.train is true by an explicit margin,
    not by the harness's numbers.

    WHY a margin and not a precondition: on this harness the floor trips
    NATURALLY. Measured (num_envs=32, with_selfplay=True, mode off, PR #261
    review): losses["entropy"] ~1.60 nats against _entropy_floor = 0.3 * 8.21
    = 2.46, so entropy_floor_fires is 7/7 and _batch1_effective_alpha is 0.5
    with no forcing at all, and deleting the write below leaves both floor
    tests green. The helper pins those two tests to the below-floor arm however
    the harness policy's entropy drifts; the proof that the body READS
    self._entropy_floor is the opposite direction,
    test_floor_below_entropy_leaves_alpha_unclamped_when_mode_off. (An earlier
    docstring here claimed the harness sits near max entropy and the condition
    never trips naturally; that was false before W2a too.)

    WHY a write after construction and not a config knob: the floor is computed
    once in Cs2PuffeRL._init_return_norm as `0.3 * max_entropy`, from the
    module-level ACTION_HEAD_SIZES / LOG_STD_MAX — never from config.

    PITFALL: until gh#168 W2a the floor was a closure cell of the patched
    train() body and this helper rewrote it through `__closure__`; W2a made it
    the instance attribute `_entropy_floor` (declared in `_init_return_norm`,
    pinned by tests/test_trainer_composition.py's derived constructor surface and the O4
    construction snapshot). The attribute must exist BEFORE the write: a
    renamed attribute would otherwise create a dead one and this helper would
    silently no-op, which is what the assert below turns into a failure.
    """
    assert hasattr(trainer, "_entropy_floor"), (
        "trainer has no _entropy_floor: Cs2PuffeRL._init_return_norm renamed it; update this "
        "helper to match, or the floor write below would create a dead attribute")
    trainer._entropy_floor = float(floor)


def test_floor_stays_disarmed_during_grace_even_below_floor():
    """gh#96: the min=0.5 floor clamp must NOT re-arm inside the warm-start window.

    Forces the collapse condition (entropy below floor) while GRACE is active. If a
    refactor drops the `_ws_floor_active and` guard, effective_alpha jumps from the
    0.0 ceiling to 0.5 — a ~500x discontinuity mid-window (spec finding 2) — and
    the ceiling assertion below fails.
    """
    trainer, cleanup = _build_ws_trainer(warmstart_grace_steps=10**12,
                                         warmstart_ramp_steps=10_000_000)
    try:
        _force_floor_above_entropy(trainer)
        losses = _run_train_once(trainer)
        assert "warmstart_phase" in losses
        assert losses["warmstart_phase"] == 0, "test precondition: must still be in GRACE"
        assert trainer._batch1_effective_alpha <= 1e-8, (
            "floor re-armed during GRACE: effective_alpha "
            f"{trainer._batch1_effective_alpha} left the ceiling despite "
            "_ws_floor_active being False")
    finally:
        cleanup()


def test_floor_clamps_effective_alpha_when_mode_off():
    """gh#96: the other half of the gate — with the mode off the floor still bites.

    Same forced collapse condition as the GRACE test, but no warmstart keys, so
    _ws_floor_active stays True and the clamp must raise effective_alpha to >=0.5.
    Without this, the GRACE test above would also pass on a build where the floor
    clamp was deleted outright.
    """
    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        _force_floor_above_entropy(trainer)
        losses = _run_train_once(trainer)
        assert "warmstart_phase" not in losses, "test precondition: mode must be off"
        assert trainer._batch1_effective_alpha >= 0.5, (
            "floor clamp did not fire with the mode off: effective_alpha "
            f"{trainer._batch1_effective_alpha} < 0.5")
    finally:
        cleanup()


def test_floor_below_entropy_leaves_alpha_unclamped_when_mode_off():
    """Positive control for the floor READ (PR #261 review): the floor forced BELOW the
    entropy, and the clamp must not fire.

    The two tests above force the floor up, but on this harness the floor trips on its
    own (entropy ~1.60 nats < 0.3 * max_entropy = 2.46; see _force_floor_above_entropy),
    so both stay green for a body that never reads self._entropy_floor and clamps
    unconditionally, or that compares against a copy taken at construction. This
    direction is what such a body cannot pass: with _entropy_floor = -1.0 the
    comparison `current_entropy < floor` is false on every minibatch, so
    entropy_floor_fires must be 0 and effective alpha must stay at the raw alpha
    (exp(log_alpha) ~ ent_coef = 0.1 after the Task 9B reset), i.e. < 0.5.

    Measured mutants of `if _ws_floor_active and current_entropy.item() <
    self._entropy_floor:` in Cs2PuffeRL.train: comparison replaced by `True` -> this
    test red (fires 7, alpha 0.5), the mode-off test above green; replaced by `False`
    -> the mode-off test above red (alpha ~0.1), this one green. The pair pins both
    constant replacements; neither alone does.
    """
    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        assert hasattr(trainer, "_entropy_floor"), (
            "trainer has no _entropy_floor: Cs2PuffeRL._init_return_norm renamed it")
        trainer._entropy_floor = -1.0
        losses = _run_train_once(trainer)
        assert "warmstart_phase" not in losses, "test precondition: mode must be off"
        # defaultdict(float): membership first, or a missing key reads as 0 == 0.
        assert "entropy_floor_fires" in losses
        assert losses["entropy_floor_fires"] == 0, (
            f"floor fired {losses['entropy_floor_fires']}x with _entropy_floor = -1.0: "
            "Cs2PuffeRL.train is not comparing against self._entropy_floor")
        assert trainer._batch1_effective_alpha < 0.5, (
            "floor clamp fired with the floor below the entropy: effective_alpha "
            f"{trainer._batch1_effective_alpha}")
    finally:
        cleanup()


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
        assert h_after_1 is not None
        losses2 = _run_train_once(trainer)
        assert losses2["warmstart_phase"] == 1
        anchor = trainer._batch1_warmstart_h_anchor
        assert anchor is not None
        assert abs(anchor - h_after_1) < 1e-9
        # ramp_steps is huge so the mirrored target sits ~at the anchor
        target = trainer._batch1_current_target_entropy
        assert target is not None
        assert abs(target - h_after_1) < 1e-3
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
    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=32, with_selfplay=True)
    try:
        # no warmstart keys at all
        losses = _run_train_once(trainer)
        assert "warmstart_phase" not in losses
        assert "warmstart_h_over_h0" not in losses
        # controller is live: with target far from H the alpha optimizer steps
        assert abs(losses["log_alpha"] - math.log(trainer.config["ent_coef"])) > 1e-6, (
            "with the mode off the alpha optimizer must actually step")
    finally:
        cleanup()
