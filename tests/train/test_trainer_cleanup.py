"""The trainer owns its utilization thread, including incomplete construction."""
import pytest

from cs2rl.train.trainer import Cs2PuffeRL
from tests._helpers.trainer_harness import _build_trainer_for_test


def test_dashboard_failure_during_base_init_stops_thread(monkeypatch, tmp_path):
    """PuffeRL starts utilization before print_dashboard, which can raise on output."""
    error = BrokenPipeError("dashboard stream closed")
    utilization = None

    def fail(self, *a, **kw):
        nonlocal utilization
        # Retain the thread even if constructor cleanup finishes it before we assert.
        utilization = self.utilization
        raise error

    monkeypatch.setattr(Cs2PuffeRL, "print_dashboard", fail)
    try:
        with pytest.raises(BrokenPipeError) as caught:
            _build_trainer_for_test(num_envs=16)
        assert caught.value is error
        assert utilization is not None, "failure must occur after utilization starts"
        assert utilization.stopped
        utilization.join(timeout=5)
        assert not utilization.is_alive()
    finally:
        # Keep the regression's original-red run from hanging the interpreter.
        if utilization is not None:
            utilization.stop()
            utilization.join(timeout=5)


@pytest.mark.parametrize("method", ["close", "close_resources"])
def test_vector_close_failure_still_stops_thread(monkeypatch, method):
    """A raised vector shutdown must not strand PuffeRL's non-daemon thread."""
    trainer, cleanup = _build_trainer_for_test(num_envs=16)
    error = KeyboardInterrupt("shutdown interrupted")

    def fail():
        raise error

    try:
        with monkeypatch.context() as patch:
            patch.setattr(trainer.vecenv, "close", fail)
            with pytest.raises(KeyboardInterrupt) as caught:
                getattr(trainer, method)()
            assert caught.value is error
            assert trainer.utilization.stopped
            trainer.utilization.join(timeout=5)
            assert not trainer.utilization.is_alive()
    finally:
        cleanup()


def test_failure_before_utilization_creation_keeps_original_error(monkeypatch):
    """Protecting base construction must also handle its earliest failure sites."""
    from pufferlib.pufferl import PuffeRL

    error = RuntimeError("before utilization")

    def fail(*a, **kw):
        raise error

    monkeypatch.setattr(PuffeRL, "__init__", fail)
    with pytest.raises(RuntimeError) as caught:
        Cs2PuffeRL({},
                   None,
                   None,
                   cont_action_view_main=None,
                   mask_view_main=None,
                   participating_rows=None,
                   self_play_mgr=None)
    assert caught.value is error


def test_normal_close_reuses_upstream_checkpoint_and_stops_once(monkeypatch):
    """The failure fallback must not repeat shutdown on the successful path."""
    from pathlib import Path

    trainer, cleanup = _build_trainer_for_test(num_envs=16)
    stopped = []
    stop = trainer.utilization.stop

    def tracked_stop():
        stopped.append(True)
        stop()

    try:
        with monkeypatch.context() as patch:
            patch.setattr(trainer.utilization, "stop", tracked_stop)
            path = trainer.close()
        assert stopped == [True]
        assert Path(path).is_file()
    finally:
        cleanup()
