"""The trainer owns its utilization thread, including incomplete construction."""
import threading

import pytest
from pufferlib.pufferl import Utilization

from cs2rl.train.trainer import Cs2PuffeRL
from tests._helpers.trainer_harness import _build_trainer_for_test


def test_dashboard_failure_during_base_init_stops_thread(monkeypatch, tmp_path):
    """PuffeRL starts utilization before print_dashboard, which can raise on output."""
    error = BrokenPipeError("dashboard stream closed")
    before = set(threading.enumerate())

    def fail(*a, **kw):
        raise error

    monkeypatch.setattr(Cs2PuffeRL, "print_dashboard", fail)
    try:
        with pytest.raises(BrokenPipeError) as caught:
            _build_trainer_for_test(num_envs=16)
        assert caught.value is error
        threads = [
            t for t in threading.enumerate() if isinstance(t, Utilization) and t not in before
        ]
        assert threads, "failure must occur after utilization starts"
        assert all(t.stopped for t in threads)
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
    finally:
        # Keep the regression's original-red run from hanging the interpreter.
        for thread in threading.enumerate():
            if isinstance(thread, Utilization) and thread not in before:
                thread.stop()
                thread.join(timeout=5)


def test_vector_close_failure_still_stops_thread(monkeypatch):
    """A raised vector shutdown must not strand PuffeRL's non-daemon thread."""
    trainer, cleanup = _build_trainer_for_test(num_envs=16)
    error = KeyboardInterrupt("shutdown interrupted")

    def fail():
        raise error

    try:
        with monkeypatch.context() as patch:
            patch.setattr(trainer.vecenv, "close", fail)
            with pytest.raises(KeyboardInterrupt) as caught:
                trainer.close()
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
