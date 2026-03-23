"""Tests for _patch_trainer_with_timing."""
import time
import pytest


class _FakeTrainer:
    """Minimal stand-in for PuffeRL — only needs evaluate() and train()."""

    def __init__(self):
        self._evaluate_called = False
        self._train_called = False

    def evaluate(self):
        time.sleep(0.01)  # simulate work
        self._evaluate_called = True

    def train(self):
        time.sleep(0.005)
        self._train_called = True
        return {"SPS": 1000}


def test_timing_patch_records_positive_values():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from train import _patch_trainer_with_timing

    trainer = _FakeTrainer()
    _patch_trainer_with_timing(trainer)

    trainer.evaluate()
    trainer.train()

    assert trainer._timing["collect_ms"] > 0, "collect_ms should be positive"
    assert trainer._timing["update_ms"] > 0, "update_ms should be positive"
    assert trainer._evaluate_called
    assert trainer._train_called


def test_timing_patch_preserves_train_return_value():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from train import _patch_trainer_with_timing

    trainer = _FakeTrainer()
    _patch_trainer_with_timing(trainer)

    trainer.evaluate()
    logs = trainer.train()
    # Verify the original SPS value is preserved
    assert logs["SPS"] == 1000
    # Verify timing keys were injected with positive values
    assert logs["timing/collect_ms"] > 0
    assert logs["timing/update_ms"] > 0
    # Verify no other unexpected keys were added
    assert set(logs.keys()) == {"SPS", "timing/collect_ms", "timing/update_ms"}
