"""The SAC-style alpha step on a real trainer: a non-finite alpha loss does not step alpha."""
import math

import pytest
import torch

from tests._helpers.trainer_harness import _build_trainer_for_test

pytestmark = pytest.mark.training

_ENTROPY_FILL = {"nan": float("nan"), "posinf": float("inf"), "neginf": float("-inf")}


@pytest.mark.parametrize("fault", [*_ENTROPY_FILL, "target_nan"])
def test_nonfinite_alpha_loss_leaves_log_alpha_finite(monkeypatch, fault):
    """One non-finite alpha-loss input in update 0; update 1 trains alpha and the policy.

    #353: the alpha step runs before the NaN guard. When it stepped on a non-finite alpha
    loss, log_alpha and its Adam moments went NaN and stayed NaN, the entropy bonus
    ``-effective_alpha * entropy`` was NaN in every later minibatch, and the NaN guard
    skipped every later policy step: at fc6ce11, on this trainer, one NaN-entropy
    minibatch left 0 of the policy's 27 parameter tensors changed in each of the next
    three updates.

    The entropy faults fill the first minibatch's entropy with NaN, +inf or -inf (+inf
    makes the alpha loss -inf, not NaN). ``target_nan`` makes update 0's entropy target
    NaN, the alpha loss's other input; entropy and the policy loss stay finite, so the
    policy still steps in update 0.
    """
    from cs2rl.train import trainer as trainer_module
    from cs2rl.train.resume import seed_everything

    seed_everything(0)
    trainer, cleanup = _build_trainer_for_test(num_envs=4)
    try:
        real_loss = trainer_module._hybrid_ppo_loss
        real_target = trainer_module._scheduled_target_entropy
        calls = {"loss": 0, "target": 0}

        def faulty_loss(*args, **kwargs):
            out = real_loss(*args, **kwargs)
            calls["loss"] += 1
            if fault in _ENTROPY_FILL and calls["loss"] == 1:
                return (out[0], out[1] * 0.0 + _ENTROPY_FILL[fault], *out[2:])
            return out

        def faulty_target(*args, **kwargs):
            # Construction made its call already; the first one here is update 0's.
            calls["target"] += 1
            if fault == "target_nan" and calls["target"] == 1:
                return float("nan")
            return real_target(*args, **kwargs)

        monkeypatch.setattr(trainer_module, "_hybrid_ppo_loss", faulty_loss)
        monkeypatch.setattr(trainer_module, "_scheduled_target_entropy", faulty_target)
        log_alpha = trainer._log_alpha_tensor
        # Steps alpha takes in update 0: the minibatches whose alpha loss is finite.
        expected_steps = 0 if fault == "target_nan" else 2
        for update in range(2):
            trainer.evaluate()
            trainer.total_minibatches = 3
            trainer.config["target_kl"] = None
            trainer.last_log_time = 0.0
            before = [p.detach().clone() for p in trainer.policy.parameters()]
            trainer.train()
            losses = trainer.losses
            assert "alpha_loss" in losses and losses["minibatches_run"] == 3
            # The fault reached the alpha loss in update 0 only (it is logged as a mean).
            assert math.isfinite(losses["alpha_loss"]) == (update == 1)
            assert torch.isfinite(log_alpha).all(), f"log_alpha {log_alpha.item()} after {update}"
            # No non-finite gradient is left on log_alpha for a later step to read.
            assert log_alpha.grad is None or torch.isfinite(log_alpha.grad).all()
            state = trainer._alpha_optimizer.state.get(log_alpha, {})
            steps = state["step"].item() if state else 0
            assert steps == expected_steps, f"alpha took {steps} steps by update {update}"
            assert all(
                torch.isfinite(state[k]).all() for k in ("exp_avg", "exp_avg_sq") if k in state)
            expected_steps += 3
            if update == 1:
                changed = sum(not torch.equal(b, p.detach())
                              for b, p in zip(before, trainer.policy.parameters(), strict=True))
                assert changed, "update 1 did not step the policy"
        assert calls["target"] == 2 and calls["loss"] == 6
    finally:
        cleanup()
