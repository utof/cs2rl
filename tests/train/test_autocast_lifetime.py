"""Real PPO updates must restore the caller's thread-local autocast scope."""
from contextlib import nullcontext

import pytest
import torch

from tests._helpers.trainer_harness import _build_trainer_for_test

pytestmark = pytest.mark.training


def _autocast_state():
    """Read nesting with a balanced native increment/decrement pair."""
    torch.autocast_increment_nesting()
    nesting = torch.autocast_decrement_nesting()
    return (nesting, torch.is_autocast_cache_enabled(),
            tuple((torch.is_autocast_enabled(device), torch.get_autocast_dtype(device))
                  for device in ("cpu", "cuda")))


@pytest.mark.parametrize("precision", ["default", "cpu_bfloat16", "outer_cpu"])
@pytest.mark.parametrize(
    "path",
    ["normal", "empty", "nonfinite", "nonfinite_entropy", "forward_error", "backward_error"])
def test_train_restores_autocast_scope(monkeypatch, precision, path):
    """Check real forward/backward, skip and exception paths, including an outer owner.

    CPU autocast is explicitly supplied because PuffeRL constructs a CUDA context
    even for CPU trainers. The default arm retains that production context.
    Repair leaked test-thread state in finally so a regression cannot pollute
    later tests or turn the expected assertion failure into a secondary error.
    """
    from cs2rl.train import trainer as trainer_module

    initial = _autocast_state()
    trainer, cleanup = _build_trainer_for_test(num_envs=4, seed=73)
    try:
        trainer.evaluate()
        assert _autocast_state() == initial
        trainer.total_minibatches = 2
        trainer.config["target_kl"] = None
        if precision != "default":
            trainer.amp_context = torch.amp.autocast("cpu",
                                                     dtype=torch.bfloat16,
                                                     enabled=precision == "cpu_bfloat16")
        if path == "empty":
            trainer.participating.zero_()

        forward_states, backward_states, value_dtypes = [], [], []
        original_loss = trainer_module._hybrid_ppo_loss

        def observe_loss(*args, **kwargs):
            """Inject faults at the real loss boundary; keep normal math intact."""
            forward_states.append(_autocast_state())
            if path == "forward_error":
                raise RuntimeError("autocast forward sentinel")
            result = original_loss(*args, **kwargs)
            value_dtypes.append(result[2].dtype)
            if path == "nonfinite":
                return (result[0] * float("nan"), *result[1:])
            if path == "nonfinite_entropy":
                return (result[0], result[1] * float("nan"), *result[2:])
            return result

        def observe_backward(gradient):
            """Observe both optimizers' backward passes and optionally raise."""
            backward_states.append(_autocast_state())
            if path == "backward_error":
                raise RuntimeError("autocast backward sentinel")
            return gradient

        monkeypatch.setattr(trainer_module, "_hybrid_ppo_loss", observe_loss)
        trainer._log_alpha_tensor.register_hook(observe_backward)
        next(trainer.policy.parameters()).register_hook(observe_backward)
        outer = (torch.amp.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False)
                 if precision == "outer_cpu" else nullcontext())
        with outer:
            before = _autocast_state()
            if path.endswith("_error"):
                with pytest.raises(RuntimeError, match="autocast .* sentinel"):
                    trainer.train()
            else:
                trainer.last_log_time = 0
                trainer.train()
            assert _autocast_state() == before
            assert all(state == before for state in backward_states)
            if path == "empty":
                assert not forward_states
                assert trainer.losses["empty_minibatches"] == 2
                assert trainer.losses["minibatches_run"] == 0
                assert not trainer.optimizer.state
            else:
                assert forward_states
                assert all(state[0] == before[0] + 1 for state in forward_states)
                if precision != "default":
                    assert all(state[2][0][0] == (precision == "cpu_bfloat16")
                               for state in forward_states)
                if value_dtypes:
                    expected = torch.bfloat16 if precision == "cpu_bfloat16" else torch.float32
                    assert all(dtype == expected for dtype in value_dtypes)
                if path in ("normal", "nonfinite", "nonfinite_entropy"):
                    assert trainer.losses["minibatches_run"] == 2
                    assert trainer.losses["empty_minibatches"] == 0
                    assert bool(trainer.optimizer.state) == (path == "normal")
                if path in ("normal", "nonfinite"):
                    # alpha steps before the NaN guard and the policy step, so a minibatch
                    # skipped for a non-finite policy loss whose alpha loss is finite still
                    # steps it: one alpha step per minibatch run.
                    alpha_state = trainer._alpha_optimizer.state[trainer._log_alpha_tensor]
                    assert alpha_state["step"].item() == 2
                if path == "nonfinite_entropy":
                    # A NaN entropy makes the alpha loss NaN too: alpha never steps and
                    # log_alpha keeps its finite value (#353).
                    assert not trainer._alpha_optimizer.state
                    assert torch.isfinite(trainer._log_alpha_tensor).all()
                if path == "normal":
                    assert backward_states
                    # A second call also checks reuse of the same autocast object.
                    trainer.train()
                    assert _autocast_state() == before
        assert _autocast_state() == initial
    finally:
        cleanup()
        current = _autocast_state()[0]
        for _ in range(max(0, current - initial[0])):
            torch.autocast_decrement_nesting()
        torch.clear_autocast_cache()
        torch.set_autocast_cache_enabled(initial[1])
        for device, (enabled, dtype) in zip(("cpu", "cuda"), initial[2], strict=True):
            torch.set_autocast_enabled(device, enabled)
            torch.set_autocast_dtype(device, dtype)
