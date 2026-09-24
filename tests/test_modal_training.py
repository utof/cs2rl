"""Behavior tests for scripts.modal_runner.training: the attempt, publication, supervision.

One of the per-module runner test files (RUNNER_TEST_FILES in
tests/modal_runner_tables.py). Before you add, move or delete a test here, or
add a helper, read THE PLACEMENT RULE FOR RUNNER TESTS in
tests/test_modal_packaging.py: which file a test belongs in, what the change
costs in the seam manifest, and where helpers go.

Deterministic patch-binding controls live in test_modal_patch_bindings.py;
interruption tests here retain their original negative assertions.
"""
import ast
import dataclasses
import io
import json
import os
import signal
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # scripts/ is a namespace package; tests import scripts.modal_runner
    # the same way the later CLIs will. Do not rely on the editable install.
    sys.path.insert(0, str(ROOT))

import scripts.modal_runner as mrl                                     # noqa: E402, I001
from scripts.modal_runner import core, state, training                 # noqa: E402, I001
from tests.modal_patch_binding_campaign import binding_target          # noqa: E402, I001
from tests.modal_test_helpers import (                                 # noqa: E402
    FakeChild, FakeRegistry, _aware, _make_manifest, _minimal_completed_tree, _no_torch)

# ── Run result: the explicit result schema ─────────────────────────────────


def test_run_result_schema_is_explicit():
    result = training.RunResult(
        schema_version=1,
        status=core.Status.COMPLETED,
        exit_code=0,
        started_at="2026-08-13T12:00:00+00:00",
        finished_at="2026-08-13T12:01:00+00:00",
        artifact_root="/artifacts/runs/ok-id",
        checkpoint_sha256="a" * 64,
        metrics_row_count=2,
        last_step=29_982_720,
    )
    payload = result.to_dict()
    assert payload["schema_version"] == 1
    assert payload["status"] == "completed"
    assert payload["exit_code"] == 0
    assert payload["checkpoint_sha256"] == "a" * 64
    assert payload["last_step"] == 29_982_720


# ── Spawn: new session, tee to the log sink, redelivery ────────────────────


def _prepared_source(tmp_path: Path, **overrides) -> core.PreparedSource:
    source_dir = tmp_path / "src"
    source_dir.mkdir(exist_ok=True)
    prepared = core.PreparedSource(
        source_dir=source_dir,
        child_env={
            "PATH": "/usr/bin",
            "OMP_NUM_THREADS": "1"
        },
        train_command=["/opt/cs2rl/.venv/bin/python", "src/train.py", "--train"],
        heartbeat=None,
        config_hash="d" * 64,
    )
    for key, value in overrides.items():
        setattr(prepared, key, value)
    return prepared


def _advance_to_building(run_root, attempt_id="attempt-a", *, lock):
    state.transition_status(run_root,
                            core.Status.PREPARING,
                            now=_aware(),
                            attempt_id=attempt_id,
                            lock=lock)
    return state.transition_status(run_root,
                                   core.Status.BUILDING,
                                   now=_aware(),
                                   attempt_id=attempt_id,
                                   lock=lock)


def _training_kwargs(tmp_path: Path, **overrides):
    run_root = tmp_path / "run"
    run_root.mkdir(exist_ok=True)
    lock = threading.Lock()
    _advance_to_building(run_root, lock=lock)
    child = overrides.pop("child", FakeChild(stdout=b"ok\n"))
    launches: list[tuple[tuple, dict]] = []

    def default_factory(*args, **kwargs):
        launches.append((args, kwargs))
        return child

    kwargs = {
        "registry": FakeRegistry(),
        "attempt_id": "attempt-a",
        "run_root": run_root,
        "prepared": _prepared_source(tmp_path),
        "commit": lambda: None,
        "lock": lock,
        "now": lambda: _aware(),
        "process_factory": default_factory,
        "sleep": lambda _seconds: None,
        "log_sink": io.StringIO(),
    }
    kwargs.update(overrides)
    kwargs["_launches"] = launches
    kwargs["_child"] = child
    return kwargs


def test_training_child_starts_in_new_session_without_shell(tmp_path):
    kwargs = _training_kwargs(tmp_path)
    launches = kwargs.pop("_launches")
    kwargs.pop("_child")
    prepared = kwargs["prepared"]
    mrl.execute_training_attempt(**kwargs)
    assert len(launches) == 1
    args, kw = launches[0]
    command = args[0] if args else kw.get("args")
    assert list(command) == prepared.train_command
    assert kw["start_new_session"] is True
    assert kw["shell"] is False
    assert kw["cwd"] == os.fspath(prepared.source_dir)
    assert kw["env"] == prepared.child_env


def test_stdout_stderr_are_teed_to_log_sink_without_truncation(tmp_path):
    payload_out = ("OUT" + ("x" * 200_000) + "END\n").encode()
    payload_err = ("ERR" + ("y" * 200_000) + "FIN\n").encode()
    child = FakeChild(stdout=payload_out, stderr=payload_err)

    class CaptureSink:

        def __init__(self):
            self.parts: list[str] = []

        def write(self, data):
            self.parts.append(data)

        def flush(self):
            return None

        def close(self):
            return None

        def getvalue(self):
            return "".join(self.parts)

    sink = CaptureSink()
    kwargs = _training_kwargs(tmp_path, child=child, log_sink=sink)
    kwargs.pop("_launches")
    kwargs.pop("_child")
    (kwargs["run_root"] / "train.log").write_text("already here\n")
    mrl.execute_training_attempt(**kwargs)
    text = sink.getvalue()
    assert "OUT" in text and "END" in text
    assert "ERR" in text and "FIN" in text
    assert text.count("x") == 200_000
    assert text.count("y") == 200_000
    leftover = (kwargs["run_root"] / "train.log").read_text()
    assert leftover.startswith("already here\n")
    assert leftover.count("x") == 200_000
    assert leftover.endswith("FIN\n") or "FIN\n" in leftover
    assert leftover.count("y") >= 200_000


def test_same_attempt_redelivery_invokes_subprocess_once(tmp_path):
    commits: list[str] = []
    child = FakeChild(stdout=b"first-delivery\n")
    kwargs = _training_kwargs(
        tmp_path,
        child=child,
        commit=lambda: commits.append("commit"),
    )
    launches = kwargs.pop("_launches")
    kwargs.pop("_child")
    first = mrl.execute_training_attempt(**kwargs)
    assert first != mrl.REDELIVERED
    assert len(launches) == 1
    status_after_first = (kwargs["run_root"] / mrl.STATUS_FILENAME).read_bytes()
    commits_after_first = list(commits)

    def must_not_launch(*_args, **_kwargs):
        raise AssertionError("redelivered container must not start training")

    kwargs["process_factory"] = must_not_launch
    second = mrl.execute_training_attempt(**kwargs)
    assert second == mrl.REDELIVERED
    assert len(launches) == 1
    assert (kwargs["run_root"] / mrl.STATUS_FILENAME).read_bytes() == status_after_first
    assert commits == commits_after_first


# ── Training loop: heartbeat and checkpoint commits ────────────────────────


class _FakeClock:

    def __init__(self):
        self._now = _aware()
        self._lock = threading.Lock()

    def now(self):
        with self._lock:
            return self._now

    def advance(self, seconds: float):
        with self._lock:
            self._now += timedelta(seconds=seconds)
            return self._now


def _consume_training_kwargs(kwargs):
    kwargs.pop("_launches", None)
    kwargs.pop("_child", None)
    return kwargs


def _run_attempt_in_thread(kwargs):
    finished = threading.Event()
    boxed: list[object] = []

    def runner():
        try:
            boxed.append(mrl.execute_training_attempt(**kwargs))
        except Exception as err:
            boxed.append(err)
        finally:
            finished.set()

    thread = threading.Thread(target=runner)
    thread.start()
    return thread, finished, boxed


def test_heartbeat_commits_every_60s_while_training(tmp_path):
    clock = _FakeClock()
    child = FakeChild(hold=True)
    beat_times: list = []

    def commit():
        beat_times.append(clock.now())

    def wait(event: threading.Event, seconds: float) -> bool:
        clock.advance(seconds)
        return event.wait(0.01)

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            commit=commit,
            now=clock.now,
            wait=wait,
        ))
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        deadline = time.monotonic() + 5.0
        while len(beat_times) < 6 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(beat_times) >= 6
        for earlier, later in zip(beat_times, beat_times[1:], strict=False):
            assert later - earlier <= timedelta(seconds=60)
    finally:
        child.release()
        assert finished.wait(timeout=2.0)
        thread.join(timeout=2.0)
        assert not thread.is_alive()


def _write_policy_checkpoint(run_root: Path, value: float) -> Path:
    import torch

    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "dust2_policy.pt"
    torch.save({"weight": torch.tensor([value])}, ckpt)
    return ckpt


def test_stable_checkpoint_gets_sidecar_and_joint_commit(tmp_path):
    child = FakeChild(hold=True)
    events: list[tuple] = []
    settle_seen = threading.Event()

    def commit():
        sidecar = kwargs["run_root"] / "checkpoints" / "dust2_policy.pt.meta.json"
        ckpt = kwargs["run_root"] / "checkpoints" / "dust2_policy.pt"
        events.append(("commit", sidecar.is_file(), ckpt.is_file()))

    def fake_sleep(seconds: float) -> None:
        if seconds >= 1.0 and ckpt.is_file() and not settle_seen.is_set():
            settle_seen.set()
            _write_policy_checkpoint(kwargs["run_root"], 2.0)

    kwargs = _consume_training_kwargs(
        _training_kwargs(tmp_path, child=child, commit=commit, sleep=fake_sleep))
    ckpt = _write_policy_checkpoint(kwargs["run_root"], 1.0)
    first_digest = core.sha256_file(ckpt)
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    sidecar = kwargs["run_root"] / "checkpoints" / "dust2_policy.pt.meta.json"
    try:
        deadline = time.monotonic() + 5.0
        while not sidecar.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sidecar.is_file()
        meta = json.loads(sidecar.read_text())
        stable_digest = core.sha256_file(ckpt)
        assert stable_digest != first_digest
        assert meta["sha256"] == stable_digest
        assert meta["size"] == ckpt.stat().st_size
        assert meta["mtime_ns"] == ckpt.stat().st_mtime_ns
        assert meta["validated_at"]
        assert any(kind == "commit" and has_side and has_ckpt
                   for kind, has_side, has_ckpt in events)
    finally:
        child.release()
        assert finished.wait(timeout=2.0)
        thread.join(timeout=2.0)


def test_torn_checkpoint_does_not_publish_sidecar(tmp_path):
    child = FakeChild(hold=True)
    settle_calls = threading.Event()

    def fake_sleep(seconds: float) -> None:
        if seconds >= 1.0:
            settle_calls.set()

    kwargs = _consume_training_kwargs(_training_kwargs(tmp_path, child=child, sleep=fake_sleep))
    ckpt_dir = kwargs["run_root"] / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "dust2_policy.pt").write_bytes(b"torn-not-a-checkpoint")
    sidecar = ckpt_dir / "dust2_policy.pt.meta.json"
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        assert settle_calls.wait(timeout=2.0)
        time.sleep(0.05)
        assert not sidecar.exists()
    finally:
        child.release()
        assert finished.wait(timeout=2.0)
        thread.join(timeout=2.0)


def test_interrupt_publishes_sidecar_after_unstable_live_saves(tmp_path):
    """Live PufferLib rewrites dust2_policy.pt every epoch (~0.5s).

    The 1s settle window never elapses while the child is alive. After SIGINT
    the file is stable and finalize must still publish the sidecar, or resume
    cannot validate the parent.
    """
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    rewrites = {"n": 0}

    def fake_sleep(seconds: float) -> None:
        hooks["sleep"](seconds)
        if seconds >= 1.0 and child.poll() is None:
            rewrites["n"] += 1
            _write_policy_checkpoint(kwargs["run_root"], float(rewrites["n"]))

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=fake_sleep,
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    ckpt = _write_policy_checkpoint(kwargs["run_root"], 0.0)
    sidecar = kwargs["run_root"] / "checkpoints" / "dust2_policy.pt.meta.json"
    thread, finished, boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, _term = _wait_until_handlers(hooks["installed"], hooks["originals"])
        deadline = time.monotonic() + 2.0
        while rewrites["n"] < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert rewrites["n"] >= 2
        assert not sidecar.exists()
        int_handler(signal.SIGINT, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert json.loads(
        (kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())["status"] == "interrupted"
    deadline = time.monotonic() + 2.0
    while not sidecar.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sidecar.is_file()
    meta = json.loads(sidecar.read_text())
    assert meta["sha256"] == core.sha256_file(ckpt)
    assert meta["size"] == ckpt.stat().st_size


# ── Runner-interpreter checkpoint validation: publish and interrupt ────────
#
# The Modal runner process and the training child are DIFFERENT interpreters.
# The runner is the image's standalone /usr/local/bin/python (only uv + modal);
# torch lives exclusively in the PREBUILT_PYTHON venv that runs train.py.
# Verified in a live container on 2026-08-14:
#   runner_executable=/usr/local/bin/python  runner_torch=MISSING
# Every test above runs on a laptop where `import torch` succeeds, so none of
# them can see this. These do: they force the torch-less runner condition.


def _publish(run_root: Path):
    commits: list[int] = []
    outcome = training.publish_stable_checkpoint(
        run_root,
        now=_aware,
        commit=lambda: commits.append(1),
        sleep=lambda _seconds: None,
        last_published=None,
    )
    return outcome, commits


def test_publish_validates_via_prebuilt_interpreter_when_runner_lacks_torch(tmp_path, monkeypatch):
    """A valid checkpoint must still publish when the runner cannot import torch."""
    run_root = tmp_path / "run"
    run_root.mkdir()
    ckpt = _write_policy_checkpoint(run_root, 1.0)
    _no_torch(monkeypatch, prebuilt=sys.executable)

    outcome, commits = _publish(run_root)

    sidecar = ckpt.with_name("dust2_policy.pt.meta.json")
    assert sidecar.is_file()
    assert outcome.reason is None
    assert outcome.generation == (ckpt.stat().st_mtime_ns, ckpt.stat().st_size)
    assert len(commits) == 1
    assert json.loads(sidecar.read_text())["sha256"] == core.sha256_file(ckpt)


def test_prebuilt_validation_still_rejects_a_torn_checkpoint(tmp_path, monkeypatch):
    """The fallback must not become a rubber stamp: garbage still fails to load."""
    run_root = tmp_path / "run"
    (run_root / "checkpoints").mkdir(parents=True)
    (run_root / "checkpoints" / "dust2_policy.pt").write_bytes(b"torn-not-a-checkpoint")
    _no_torch(monkeypatch, prebuilt=sys.executable)

    outcome, commits = _publish(run_root)

    assert not (run_root / "checkpoints" / "dust2_policy.pt.meta.json").exists()
    assert outcome.generation is None
    assert "not weights-only loadable" in outcome.reason
    assert commits == []


def test_publish_reason_names_the_missing_interpreter(tmp_path, monkeypatch):
    """No torch and no prebuilt venv: skipping is fine, skipping SILENTLY is not."""
    run_root = tmp_path / "run"
    run_root.mkdir()
    _write_policy_checkpoint(run_root, 1.0)
    _no_torch(monkeypatch, prebuilt=str(tmp_path / "nonexistent" / "python"))

    outcome, commits = _publish(run_root)

    assert not (run_root / "checkpoints" / "dust2_policy.pt.meta.json").exists()
    assert "nonexistent" in outcome.reason
    assert commits == []


def test_interrupt_without_publishable_checkpoint_writes_a_reason_file(tmp_path, monkeypatch):
    """finalize must leave evidence on the Volume, before its commit, of WHY there
    is no sidecar. Three T4 runs were burned on a silently swallowed skip."""
    _no_torch(monkeypatch, prebuilt=str(tmp_path / "nonexistent" / "python"))
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    commits: list[bool] = []

    def commit() -> None:
        commits.append(
            (kwargs["run_root"] / "checkpoints" / core.CHECKPOINT_PUBLISH_REASON_NAME).is_file())

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            commit=commit,
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    _write_policy_checkpoint(kwargs["run_root"], 1.0)
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, _term = _wait_until_handlers(hooks["installed"], hooks["originals"])
        int_handler(signal.SIGINT, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)

    reason_path = kwargs["run_root"] / "checkpoints" / core.CHECKPOINT_PUBLISH_REASON_NAME
    assert reason_path.is_file()
    payload = json.loads(reason_path.read_text())
    assert "nonexistent" in payload["reason"]
    assert payload["at"]
    # Written BEFORE a commit, or it never reaches the Volume.
    assert any(commits)


def test_interrupt_commits_status_even_if_prebuilt_load_hangs(tmp_path, monkeypatch):
    """SIGINT finalize must persist STATUS before any hung PREBUILT_PYTHON load.

    Modal preemption grace is ~30s and Function-timeout slack is seconds. A
    120s weights-only load inside finalize can lose both sidecar and STATUS.
    The watcher thread is exempt so its 50ms poll cannot stall this test.
    """
    release = threading.Event()

    def hanging_load(_path):
        if threading.current_thread().name == "cs2rl-checkpoint-watch":
            raise mrl.ValidationError("watcher must not hang the interrupt path")
        if not release.wait(timeout=10.0):
            raise mrl.ValidationError("test timed out waiting to release the hung load")

    monkeypatch.setattr(*binding_target("interrupt-loader"), hanging_load)
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    commits: list[str | None] = []

    def commit() -> None:
        status_path = kwargs["run_root"] / mrl.STATUS_FILENAME
        if not status_path.is_file():
            commits.append(None)
            return
        commits.append(json.loads(status_path.read_text())["status"])

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            commit=commit,
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    _write_policy_checkpoint(kwargs["run_root"], 1.0)
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, _term = _wait_until_handlers(hooks["installed"], hooks["originals"])
        handler_thread = threading.Thread(target=int_handler,
                                          args=(signal.SIGINT, None),
                                          daemon=True)
        handler_thread.start()
        deadline = time.monotonic() + 2.0
        interrupted = False
        while time.monotonic() < deadline:
            status_path = kwargs["run_root"] / mrl.STATUS_FILENAME
            if (status_path.is_file()
                    and json.loads(status_path.read_text())["status"] == "interrupted"):
                interrupted = True
                break
            time.sleep(0.01)
        assert interrupted, "STATUS must become interrupted while the prebuilt load is still hung"
        assert "interrupted" in commits
    finally:
        release.set()
        child.release()
        assert finished.wait(timeout=2.0)
        thread.join(timeout=2.0)


def test_checkpoint_watcher_threads_generation_into_last_published(tmp_path, monkeypatch):
    """A watcher-only typo on PublishOutcome.generation is swallowed every 50ms.

    Direct publish tests cannot see that: they assert .generation on the
    function return, not on the value the thread feeds back as last_published.
    """
    seen: list[tuple[int, int] | None] = []
    generation = (111, 222)

    def fake_publish(*_args, last_published=None, **_kwargs):
        seen.append(last_published)
        return training.PublishOutcome(generation)

    monkeypatch.setattr(*binding_target("watcher-publisher"), fake_publish)
    stop, watcher = training._start_checkpoint_watcher(
        run_root=tmp_path,
        now=_aware,
        commit=lambda: None,
        sleep=lambda _seconds: None,
    )
    try:
        deadline = time.monotonic() + 2.0
        while len(seen) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(seen) >= 2
        assert seen[0] is None
        assert seen[1] == generation
    finally:
        stop.set()
        watcher.join(timeout=2.0)


# ── Attempt supervision: SIGINT / KeyboardInterrupt / SIGTERM cleanup ──────


def _signal_hooks(child, *, release_on=signal.SIGKILL):
    """Recording fakes for the attempt's signal seam: handlers, killpg, getpgid, sleep.

    `fake_getpgid` is the identity, so the child passes the `_signal_process_group`
    guard's `pgid!=pid` clause, and `fake_killpg` records instead of signalling.

    The precondition sits here because every test that drives an attempt into
    its kill path takes its fakes from this helper, so one assertion covers
    them all. (The §2a guard test calls `_signal_process_group` directly and
    builds its own.) The guard also refuses the runner's OWN group: if this
    session's process group happened to equal the fake child's pid (4242 by
    default), each of those tests would see its kills refused and fail for a
    reason unrelated to its subject, so this stops at the cause instead.
    """
    assert os.getpgrp() != child.pid, (
        f"the runner's own process group ({os.getpgrp()}) equals the fake child's pid "
        f"({child.pid}), so the §2a guard in `_signal_process_group` refuses to signal it by "
        "design and this kill-path test cannot run in this session")
    originals = {signal.SIGINT: object(), signal.SIGTERM: object()}
    installed: dict[int, object] = dict(originals)
    kills: list[int] = []
    slept: list[float] = []

    def fake_signal(sig, handler):
        previous = installed.get(sig, originals.get(sig))
        installed[sig] = handler
        return previous

    def fake_getpgid(pid):
        return pid

    def fake_killpg(_pgid, sig):
        kills.append(sig)
        if release_on is not None and sig == release_on:
            child.release()

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    return {
        "originals": originals,
        "installed": installed,
        "kills": kills,
        "slept": slept,
        "signal_signal": fake_signal,
        "getpgid": fake_getpgid,
        "killpg": fake_killpg,
        "sleep": fake_sleep,
    }


def _wait_until_handlers(installed, originals, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        int_handler = installed.get(signal.SIGINT)
        term_handler = installed.get(signal.SIGTERM)
        if (int_handler is not None and int_handler is not originals[signal.SIGINT]
                and term_handler is not None and term_handler is not originals[signal.SIGTERM]):
            return int_handler, term_handler
        time.sleep(0.01)
    raise AssertionError("signal handlers were not installed around the child")


def test_sigint_and_sigterm_share_cleanup_and_restore_handlers(tmp_path):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    order: list[object] = []

    def stop_and_join():
        order.append("heartbeat_stopped")
        order.append(json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())["status"])

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    kwargs["prepared"] = _prepared_source(tmp_path,
                                          heartbeat=SimpleNamespace(stop_and_join=stop_and_join))
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, term_handler = _wait_until_handlers(hooks["installed"], hooks["originals"])
        assert int_handler is term_handler
        int_handler(signal.SIGINT, None)
        term_handler(signal.SIGTERM, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert hooks["installed"][signal.SIGINT] is hooks["originals"][signal.SIGINT]
    assert hooks["installed"][signal.SIGTERM] is hooks["originals"][signal.SIGTERM]
    persisted = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "interrupted"
    assert persisted["attempt_id"] == "attempt-a"
    assert order[0] == "heartbeat_stopped"
    assert order[1] == "training"
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]


def test_keyboard_interrupt_uses_same_cleanup(tmp_path):
    child = FakeChild(hold=True)

    def exploding_wait(timeout=None):
        raise KeyboardInterrupt

    child.wait = exploding_wait
    hooks = _signal_hooks(child)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    result = mrl.execute_training_attempt(**kwargs)
    assert result != mrl.REDELIVERED
    assert hooks["installed"][signal.SIGINT] is hooks["originals"][signal.SIGINT]
    assert hooks["installed"][signal.SIGTERM] is hooks["originals"][signal.SIGTERM]
    persisted = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "interrupted"
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]


def test_child_receives_term_then_kill_after_grace(tmp_path):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child, release_on=signal.SIGKILL)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        _handler, term_handler = _wait_until_handlers(hooks["installed"], hooks["originals"])
        term_handler(signal.SIGTERM, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]
    assert training.TERM_GRACE_SECONDS in child.wait_timeouts


def test_cleanup_closes_log_before_final_commit(tmp_path):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    events: list[str] = []

    class RecordingSink(io.StringIO):

        def close(self):
            events.append("log_closed")
            super().close()

    def commit():
        events.append("commit")

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            commit=commit,
            log_sink=RecordingSink(),
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, _term = _wait_until_handlers(hooks["installed"], hooks["originals"])
        int_handler(signal.SIGINT, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert "log_closed" in events
    assert "commit" in events[events.index("log_closed") + 1:]


def test_failed_cleanup_commit_does_not_let_redelivery_write(tmp_path):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    committed: list[dict] = []

    def commit():
        payload = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
        if payload["status"] == "interrupted":
            raise RuntimeError("volume commit failed")
        committed.append(payload)

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            commit=commit,
            now=lambda: _aware(),
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, _term = _wait_until_handlers(hooks["installed"], hooks["originals"])
        int_handler(signal.SIGINT, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert committed
    assert committed[-1]["status"] == "training"
    last = state.RunStatus.from_dict(committed[-1])
    derived = state.derive_status(last, now=_aware(minute=5))
    assert derived.stale is True
    assert derived.status is core.Status.INTERRUPTED
    before = (kwargs["run_root"] / mrl.STATUS_FILENAME).read_bytes()
    commits_before = list(committed)

    def must_not_launch(*_args, **_kwargs):
        raise AssertionError("redelivered container must not start training")

    kwargs["process_factory"] = must_not_launch
    assert mrl.execute_training_attempt(**kwargs) == mrl.REDELIVERED
    assert (kwargs["run_root"] / mrl.STATUS_FILENAME).read_bytes() == before
    assert committed == commits_before


def test_post_spawn_failure_kills_child_and_writes_terminal_status(tmp_path):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    (kwargs["run_root"] / core.TRAIN_LOG_NAME).mkdir()
    with pytest.raises(OSError):
        mrl.execute_training_attempt(**kwargs)
    persisted = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] in {"failed", "interrupted"}
    assert persisted["attempt_id"] == "attempt-a"
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]
    assert child.poll() is not None


def test_term_grace_is_deadline_not_mandatory_sleep(tmp_path):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child, release_on=signal.SIGTERM)
    started = time.monotonic()
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        _handler, term_handler = _wait_until_handlers(hooks["installed"], hooks["originals"])
        term_handler(signal.SIGTERM, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert time.monotonic() - started < 5.0
    assert hooks["kills"] == [signal.SIGTERM]
    assert 15.0 not in hooks["slept"]
    persisted = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "interrupted"


def _record_hash_after_terminal(monkeypatch, run_root: Path) -> list[str]:
    hashed: list[str] = []

    def wrapped_validate(path):
        del path
        status = json.loads((run_root / mrl.STATUS_FILENAME).read_text())["status"]
        if status != "training":
            hashed.append("validate")
        raise mrl.ValidationError("test stub: skip torch")

    def wrapped_hash(path):
        del path
        status = json.loads((run_root / mrl.STATUS_FILENAME).read_text())["status"]
        if status != "training":
            hashed.append("hash")
        return "00" * 32

    monkeypatch.setattr(*binding_target("terminal-validator"), wrapped_validate)
    monkeypatch.setattr(*binding_target("terminal-hasher"), wrapped_hash)
    return hashed


def test_interrupt_uses_sidecar_digest_and_skips_torch_hash(tmp_path, monkeypatch):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    run_root = kwargs["run_root"]
    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir()
    sidecar_digest = "ab" * 32
    (ckpt_dir / mrl.CHECKPOINT_SIDECAR_NAME).write_text(
        json.dumps({
            "sha256": sidecar_digest,
            "size": 13,
            "mtime_ns": 1,
            "validated_at": "2026-08-13T00:00:00+00:00",
        }) + "\n")
    hashed = _record_hash_after_terminal(monkeypatch, run_root)
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, _term = _wait_until_handlers(hooks["installed"], hooks["originals"])
        int_handler(signal.SIGINT, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert hashed == []
    payload = json.loads((run_root / core.RESULT_FILENAME).read_text())
    assert payload["status"] == "interrupted"
    assert payload["checkpoint_sha256"] == sidecar_digest


def test_interrupt_without_sidecar_leaves_checkpoint_hash_null(tmp_path, monkeypatch):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    run_root = kwargs["run_root"]
    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "dust2_policy.pt").write_bytes(b"do-not-load-me")
    hashed = _record_hash_after_terminal(monkeypatch, run_root)
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, _term = _wait_until_handlers(hooks["installed"], hooks["originals"])
        int_handler(signal.SIGINT, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert hashed == []
    payload = json.loads((run_root / core.RESULT_FILENAME).read_text())
    assert payload["status"] == "interrupted"
    assert payload["checkpoint_sha256"] is None


def test_checkpoint_watcher_stops_before_terminal_status(tmp_path, monkeypatch):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child, release_on=None)
    watcher_stop: dict[str, threading.Event | None] = {"event": None}
    at_terminal: list[tuple[str, bool]] = []
    real_start = training._start_checkpoint_watcher
    real_transition = state.transition_status

    def wrapped_start(**kwargs):
        stop, thread = real_start(**kwargs)
        watcher_stop["event"] = stop
        return stop, thread

    def wrapped_transition(run_root, next_status, **kwargs):
        if next_status in mrl.TERMINAL_STATUSES:
            event = watcher_stop["event"]
            at_terminal.append((next_status.value, event is not None and event.is_set()))
        return real_transition(run_root, next_status, **kwargs)

    monkeypatch.setattr(*binding_target("attempt-watcher"), wrapped_start)
    monkeypatch.setattr(*binding_target("attempt-transition"), wrapped_transition)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, _term = _wait_until_handlers(hooks["installed"], hooks["originals"])
        int_handler(signal.SIGINT, None)
        child.release()
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert at_terminal == [("interrupted", True)]
    persisted = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "interrupted"


# A hard timeout, not belt-and-braces: if `spy_start`'s `_target is _tee_stream`
# filter ever stops matching, the wrapper never fires, the SIGTERM is never sent,
# FakeChild(hold=True) is never released, and the attempt's wait loop spins
# forever -- so the `assert fired` guard below is UNREACHABLE and this becomes an
# unbounded hang instead of a failure. The repo configures no default timeout.
# W3b makes this reachable: anything that stops `training._tee_stream` being the
# identical object passed as `target` (a re-export wrapper, functools.partial, a
# Thread subclass) breaks identity SILENTLY, where a rename would at least raise
# AttributeError.
@pytest.mark.timeout(30)
def test_real_sigterm_in_tee_window_never_joins_unstarted_thread(tmp_path, monkeypatch):
    """A real SIGTERM inside the tee-thread start window must not strand the run.

    gh#217. `finalize` (nested in `training._run_training_attempt`) joins
    `tee_threads` unconditionally, so for
    as long as that list could hold a not-yet-started thread, a signal arriving
    there raised `RuntimeError: cannot join thread before it is started` out of
    `on_signal` and past `transition_status`: STATUS.json stuck on `training`, no
    result.json, child already dead. This is gh#217's demonstration 3 checked in —
    `execute_training_attempt` on the MAIN thread with the REAL `signal.signal`, a
    helper thread firing a REAL `os.kill(os.getpid(), SIGTERM)`, and a
    `threading.Thread.start` wrapper filtered on `_target is _tee_stream` that puts
    the signal in the window deterministically rather than relying on machine load.

    Three pitfalls this test is built around, each of which fails silently:
      * `_target` is captured BEFORE delegating to the real `start()`. CPython's
        `Thread.run()` does `del self._target, self._args, self._kwargs` in its
        `finally` when a thread finishes (it is run(), not _bootstrap_inner --
        gh#217's draft said otherwise), and a tee thread over an empty FakeChild
        stream can finish before `start()` returns — read afterwards it is None and
        the filter never matches.
      * `killpg`/`getpgid` are FAKE, and that is a safety requirement rather than a
        preference. `finalize` signals the child's process group using FakeChild's
        default `pid=4242`; with the real ones, a machine where pid 4242 happens to
        exist gets a genuine SIGTERM, and `_signal_process_group`'s
        `except ProcessLookupError` makes it silent on every machine where it does
        not.
      * A no-op SIGTERM handler is installed around the call. The runner's `finally`
        restores whatever disposition was in force on entry, so without this a
        signal arriving after that restore reaches pytest's default disposition and
        kills the session instead of failing the test.

    Disclosed blind spots — this test gates the defect, not one specific repair:
      * A fix that guards the join site (`if thread.ident is not None`) instead of
        reordering passes, and so does one that clears `tee_threads` before joining.
        Nothing in this suite excludes either; both are disclosed deliberately.
      * Do NOT "correct" the spy's predicate to `is_alive()`. gh#217 measured
        `ident` to be an unreliable proxy for "join will raise" when used as a
        PRODUCTION guard, because `_bootstrap_inner` sets `_ident` before
        `_started` while `join` gates on `_started`. As a SPY it is exactly right: a
        never-started thread has `ident is None`, and once `start()` has returned
        `_started` is already set. `is_alive()` silently changes what is detected —
        a started-and-already-finished thread is not alive.
      * It drives one deterministic point inside the window. It does not prove the
        window is shut at every instruction.
      * Under the FIX, the residual pre-first-`append` window is the only state this
        test ever observes -- measured, `len(tee_threads) == 0` at the join and the
        spy's log is `[('cs2rl-preflight-heartbeat', False),
        ('cs2rl-checkpoint-watch', False)]`, i.e. zero tee-thread joins. So the
        outcome assertions are made from inside the accepted residual, and what goes
        UNTESTED is the state the fix creates: `tee_threads` non-empty and holding
        only started threads. An earlier wording had this exactly backwards.
    """
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    entered_finalize = threading.Event()
    killpg_hook = hooks["killpg"]

    def killpg_marking_finalize(pgid, sig):
        # finalize kills the child (its `_signal_process_group` call) before it
        # joins tee_threads, so this is the earliest in-handler observable available.
        # The handler runs on the main thread, so the spin below cannot observe
        # this flag until the handler has already returned or raised.
        killpg_hook(pgid, sig)
        entered_finalize.set()

    real_start = threading.Thread.start
    real_join = threading.Thread.join
    joined_unstarted: list[str] = []
    fired: list[str] = []
    killers: list[threading.Thread] = []

    def spy_join(self, timeout=None):
        # The primary assertion. Outcome assertions alone do NOT carry this test:
        # measured on unfixed source with join swallowing the RuntimeError, the
        # window is still wide open yet STATUS reaches `interrupted` and
        # result.json is written. Only this spy sees the unstarted join.
        if self.ident is None:
            joined_unstarted.append(self.name)
        return real_join(self, timeout)

    def deliver_sigterm() -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    def spy_start(self):
        target = getattr(self, "_target", None)
        if fired or target is not training._tee_stream:
            return real_start(self)
        # Fire once. A fire-every-start shim measures the same, because the first
        # fire aborts the loop before the second start() — this is readability.
        fired.append(self.name)
        killer = threading.Thread(target=deliver_sigterm, name="b0-sigterm-source", daemon=True)
        real_start(killer)
        killers.append(killer)
        deadline = time.monotonic() + 5.0
        while not entered_finalize.is_set():
            if time.monotonic() > deadline:
                child.release()
                raise AssertionError("SIGTERM was never handled inside the start window")
            time.sleep(0.001)
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", spy_start)
    monkeypatch.setattr(threading.Thread, "join", spy_join)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=killpg_marking_finalize,
            getpgid=hooks["getpgid"],
        ))
    run_root = kwargs["run_root"]
    previous_term = signal.signal(signal.SIGTERM, lambda *_args: None)
    previous_int = signal.getsignal(signal.SIGINT)
    try:
        result = mrl.execute_training_attempt(**kwargs)
    finally:
        child.release()
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)
        if previous_int is not None:
            signal.signal(signal.SIGINT, previous_int)
        for killer in killers:
            real_join(killer, 2.0)
    assert fired, "the _tee_stream start wrapper never fired; the window was never opened"
    assert joined_unstarted == []
    assert result != mrl.REDELIVERED
    persisted = json.loads((run_root / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "interrupted"
    assert (run_root / core.RESULT_FILENAME).is_file()
    payload = json.loads((run_root / core.RESULT_FILENAME).read_text())
    assert payload["status"] == persisted["status"]
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.parametrize("case", ["pgid-le-1", "own-group", "pgid-ne-pid", "control"])
def test_signal_process_group_refuses_groups_a_live_child_cannot_have(case, capsys):
    """`_signal_process_group` refuses the three groups a live child cannot lead.

    Spec §2a. The guard exists because an agent's script once ran
    `killpg(getpgid(1), SIGTERM)` for real, which is `kill(-1, SIGTERM)`, and
    ended the user's desktop session. Each refusal case sets up exactly ONE of
    the three conditions:
      * `pgid-le-1`   -- pid 1, identity getpgid: the incident's exact shape;
      * `own-group`   -- pid = the runner's own group, identity getpgid;
      * `pgid-ne-pid` -- pid P, getpgid returning P + 1 (a fake, or a reused pid);
      * `control`     -- pid P, identity getpgid: must SIGTERM P and print nothing.
    The case ids are shell-safe spellings; the stderr tokens are `pgid<=1`,
    `own-group` and `pgid!=pid`.

    WHY one condition per case. A case where two conditions hold stays green
    when one of their clauses is deleted, because the other still refuses: the
    shadowed clause would ship untested. So `pid == pgid` everywhere except the
    `pgid-ne-pid` case, and `P = os.getpgrp() + 4242` differs from the runner's
    group by construction. If the runner's group were 1, `pgid-le-1` would also
    be `own-group` (and below 1, `own-group` would also be `pgid<=1`), hence
    the first assertion, which names that condition instead of letting a
    knock-out stay green for a reason nobody can see.

    WHY each case checks its own token AND the absence of the other two. A
    guard printing one line that names all three conditions for every refusal
    would pass a "contains my token" check. Only the exclusion pins WHICH clause
    refused.

    WHY the three refusal children are held (`hold=True`) and the control's is
    not. A guard that prints the line and skips the SIGTERM but forgets to
    `return` falls through to the escalation. With an exited child, the first
    `poll()` returns and all three cases stay green. On a live child the same
    guard reaches `killpg(pgid, SIGKILL)` with the refused pgid: in the
    `pgid-le-1` case that is `kill(-1, SIGKILL)`. A held FakeChild is that live
    child: `poll()` is None and the grace `wait` times out at once (no
    wall-clock wait), so the missing `return` records a SIGKILL and
    `kills == []` goes red. The control keeps an exited child, so it records
    the SIGTERM alone.

    SAFETY: every case passes a recording `killpg` and a fake `sleep`, so
    nothing here signals even with a clause removed. The knock-outs (delete one
    clause, or drop the refusal's `return`; run only the affected case nodes;
    see red) are run by hand and are not committed.
    """
    assert os.getpgrp() > 1, (
        f"the runner's own process group is {os.getpgrp()}, not above 1, so the `pgid-le-1` and "
        "`own-group` cases would both hold two conditions and could not tell the clauses apart")
    P = os.getpgrp() + 4242

    def identity(pid):
        return pid

    child, getpgid, token = {
        "pgid-le-1": (FakeChild(pid=1, hold=True), identity, "pgid<=1"),
        "own-group": (FakeChild(pid=os.getpgrp(), hold=True), identity, "own-group"),
        "pgid-ne-pid": (FakeChild(pid=P, hold=True), lambda pid: pid + 1, "pgid!=pid"),
        "control": (FakeChild(pid=P), identity, None),
    }[case]
    kills: list[tuple[int, int]] = []
    slept: list[float] = []

    training._signal_process_group(child,
                                   killpg=lambda pgid, sig: kills.append((pgid, sig)),
                                   getpgid=getpgid,
                                   sleep=slept.append)

    refusals = [
        line for line in capsys.readouterr().err.splitlines()
        if line.startswith("cs2rl: refusing to signal process group")
    ]
    if token is None:
        assert kills == [(P, signal.SIGTERM)]
        assert refusals == []
        return
    assert kills == []
    assert len(refusals) == 1, refusals
    assert token in refusals[0]
    others = {"pgid<=1", "own-group", "pgid!=pid"} - {token}
    assert not [other for other in others if other in refusals[0]], refusals[0]


# ── Kill seam: the static safety clauses and the ProcessControl tripwire ──


class _KillSeamClauses:
    """The clauses `test_kill_seam_static_safety` enforces: one checker and one plant table each.

    gh#163 spec §4.8 criterion 6. A checker takes `{relative path: source text}`
    and returns `(problems, examined)`: what fails the clause, and what it looked
    at, which the test requires to hold the clause's population on the real tree.
    A plant has the same shape as the real sources, `{path: source text}`; it is
    parsed and never run, and must fail its clause. Nothing here imports, calls
    or opens anything: it reads only the text it is handed.

    WHY A CLASS, AND WHY HERE. Nested inside the test, the checkers counted
    towards the test's own complexity, because ruff's C901 and complexipy both
    count a nested def inside its parent (50 / 222 at the W5 types commit). As
    methods each is measured, and read, on its own. The class is ONE governed
    seam name (tests/fixtures/modal_test_seam_manifest.json and
    GOVERNED_NAME_COUNT in tests/test_modal_packaging.py): module-level checker
    functions or plant tables would each be another, so keep every checker,
    table and plant inside it.

    ADDING A CLAUSE (the W5 execute commit adds (iii) and (vi)): write its
    checker as a method here, plus a `clause_<n>` method that returns
    `_clause(...)` with its plants; list it in `clauses()`; and add its label to
    the test's pinned list. Each plant carries its own path, so a clause that
    reads both training.py and tests/ (as (iii) will) plants into either.
    """

    TRAINING = "scripts/modal_runner/training.py"
    PLANT_TEST = "tests/test_kill_seam_plant.py"
    FIELDS = ("spawn", "getpgid", "killpg", "install_signal")
    REAL_SYSTEM = {
        "spawn": "subprocess.Popen",
        "getpgid": "os.getpgid",
        "killpg": "os.killpg",
        "install_signal": "signal.signal",
    }
    OS_MODULES = ("os", "posix")
    IMPORT_CALLS = ("__import__", "import_module")
    BANNED = frozenset({"killpg", "getpgid", "kill"})
    GUARDED_CALLS = ("killpg", "getpgid")

    @classmethod
    def clauses(cls):
        """Every clause, in label order. The test pins the labels: dropping one here is red."""
        return [cls.clause_i(), cls.clause_ii(), cls.clause_iv(), cls.clause_v(), cls.clause_vii()]

    @staticmethod
    def _clause(label, name, *, reads, check, population, populated, plants):
        """One clause. `reads` names the source sets it checks: "training", "tests" or both."""
        return SimpleNamespace(label=label,
                               name=f"{label} {name}",
                               reads=reads,
                               check=check,
                               population=population,
                               populated=populated,
                               plants=plants)

    @staticmethod
    def _at(path, sources):
        """`{plant: source}` as `{plant: {path: source}}`: every plant planted at `path`."""
        return {plant: {path: source} for plant, source in sources.items()}

    # ── Shared AST readers

    @staticmethod
    def _last_name(expr):
        """The callee spelling a clause keys on: a bare name, or the last attribute."""
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Attribute):
            return expr.attr
        return None

    @staticmethod
    def _is_docstring(statement):
        return (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str))

    @staticmethod
    def _process_control_classes(tree):
        return [
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ProcessControl"
        ]

    # ── (i) no field defaults

    @classmethod
    def clause_i(cls):
        fields_block = ("    spawn: Callable[..., object]\n"
                        "    getpgid: Callable[[int], int]\n"
                        "    killpg: Callable[[int, int], None]\n")
        return cls._clause(
            "(i)",
            "ProcessControl's fields have no defaults",
            reads=("training", ),
            check=cls.no_field_defaults,
            population="the four ProcessControl fields",
            populated=lambda seen: tuple(seen) == cls.FIELDS,
            plants=cls._at(
                cls.TRAINING, {
                    "a default on one field":
                    ("class ProcessControl:\n" + fields_block +
                     "    install_signal: Callable[..., object] = signal.signal\n"),
                    "a field(default=...)":
                    ("class ProcessControl:\n" + fields_block +
                     "    install_signal: Callable[..., object] = field(default=signal.signal)\n"),
                }))

    @classmethod
    def no_field_defaults(cls, sources):
        """(i). A default is how the incident's draft handed out the real functions.

        `examined` is the annotated field names, in order.
        """
        problems, examined = [], []
        for rel, text in sources.items():
            for process_control in cls._process_control_classes(ast.parse(text)):
                fields, wrong = cls._fields_and_problems(rel, process_control)
                examined.extend(fields)
                problems.extend(wrong)
        return problems, examined

    @classmethod
    def _fields_and_problems(cls, rel, process_control):
        """One ProcessControl class body: its annotated fields, and what breaks clause (i).

        Only fields, methods and the docstring may appear: any other statement
        (a plain `x = ...`) would be a class attribute no field check reads.
        """
        fields, problems = [], []
        for statement in process_control.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                fields.append(statement.target.id)
                if statement.value is not None:
                    problems.append(f"{rel}:{statement.lineno} field {statement.target.id} "
                                    "has a default")
            elif not (cls._is_docstring(statement)
                      or isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))):
                problems.append(f"{rel}:{statement.lineno} a class-body "
                                f"{type(statement).__name__}, not a field or a method")
        if tuple(fields) != cls.FIELDS:
            problems.append(f"{rel}: the fields are {fields}, not {list(cls.FIELDS)}; if that is "
                            "deliberate, update every clause here and the tripwire's calls in "
                            "test_process_control_tripwire_poisons_system")
        return fields, problems

    # ── (ii) four keywords per construction

    @classmethod
    def clause_ii(cls):
        return cls._clause(
            "(ii)",
            "every ProcessControl(...) under tests/ passes four keywords",
            reads=("tests", ),
            check=cls.four_keyword_constructions,
            population="the tripwire's construction in tests/conftest.py",
            populated=lambda seen: any(where.startswith("tests/conftest.py:") for where in seen),
            plants=cls._at(
                cls.PLANT_TEST, {
                    "a ** splat":
                    "control = training.ProcessControl(**fields)\n",
                    "a missing field":
                    "ProcessControl(spawn=f, getpgid=g, killpg=k)\n",
                    "positional fields":
                    "ProcessControl(f, g, k, s)\n",
                    "a splat through an import alias":
                    ("from scripts.modal_runner.training import ProcessControl as PC\n"
                     "PC(spawn=f, **rest)\n"),
                    "a splat through an assignment alias":
                    "PC = training.ProcessControl\nPC(*parts)\n",
                }))

    @classmethod
    def _process_control_spellings(cls, tree):
        """`ProcessControl` and every import or assignment alias of it in `tree`."""
        spellings = {"ProcessControl"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                spellings |= {
                    alias.asname
                    for alias in node.names if alias.name == "ProcessControl" and alias.asname
                }
            elif isinstance(node, ast.Assign) and cls._last_name(node.value) == "ProcessControl":
                spellings |= {target.id for target in node.targets if isinstance(target, ast.Name)}
        return spellings

    @classmethod
    def _constructions(cls, tree):
        """Every call in `tree` whose callee is ProcessControl or an alias of it."""
        spellings = cls._process_control_spellings(tree)
        return [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and cls._last_name(node.func) in spellings
        ]

    @classmethod
    def _four_keywords(cls, call):
        """Whether `call` passes exactly the four fields as keywords, and nothing else."""
        names = sorted(k.arg for k in call.keywords if k.arg is not None)
        splat = any(k.arg is None for k in call.keywords)
        return not call.args and not splat and names == sorted(cls.FIELDS)

    @classmethod
    def four_keyword_constructions(cls, sources):
        """(ii). No positional argument and no `*`/`**` splat, whose fields cannot be read
        statically (a builder is the likely place for one). `examined` is every construction.
        """
        problems, examined = [], []
        for rel, text in sources.items():
            for call in cls._constructions(ast.parse(text)):
                examined.append(f"{rel}:{call.lineno}")
                if not cls._four_keywords(call):
                    problems.append(f"{rel}:{call.lineno} {ast.unparse(call)[:120]}")
        return problems, examined

    # ── (iv) no load of the kill seam's OS functions under tests/

    @classmethod
    def clause_iv(cls):
        return cls._clause(
            "(iv)",
            "no load of os/posix killpg, getpgid or kill under tests/",
            reads=("tests", ),
            check=cls.kill_seam_loads,
            population="the real-SIGTERM test's os.kill(os.getpid(), SIGTERM)",
            populated=lambda seen: any(
                where.startswith("tests/test_modal_training.py:") for where in seen),
            plants=cls._at(
                cls.PLANT_TEST, {
                    "the real functions passed explicitly":
                    ("training.ProcessControl(spawn=fake, getpgid=os.getpgid, killpg=os.killpg,\n"
                     "                         install_signal=signal.signal)\n"),
                    "a module alias":
                    "import os as o\no.killpg(4242, 15)\n",
                    "an assignment alias":
                    "o = os\no.killpg(4242, 15)\n",
                    "an annotated assignment alias":
                    "o: object = training.os\no.getpgid(4242)\n",
                    "an os from-imported out of another module":
                    ("from scripts.modal_runner.training import os as tos\n"
                     "tos.getpgid(4242)\n"),
                    "a from-import":
                    "from posix import getpgid\n",
                    "a from-import of kill":
                    "from os import kill as k\n",
                    "a star import":
                    "from os import *\n",
                    "getattr":
                    'getattr(os, "killpg")(4242, 15)\n',
                    "getattr with a computed name":
                    'controls = {n: getattr(os, n) for n in ("getpgid", "killpg")}\n',
                    "an __import__ receiver":
                    '__import__("os").getpgid(1)\n',
                    "a sys.modules receiver":
                    'sys.modules["posix"].killpg(1, 15)\n',
                    "a module attribute's os":
                    ("training.ProcessControl(spawn=fake, getpgid=lambda pid: pid,\n"
                     "                         killpg=training.os.killpg, install_signal=s)\n"),
                    "a nested module attribute's os":
                    "mrl.training.os.getpgid(4242)\n",
                    "another module's os":
                    "subprocess.os.killpg(4242, 15)\n",
                    "kill of a group":
                    "os.kill(-4242, 15)\n",
                    "kill of a group through a module attribute's os":
                    "training.os.kill(-4242, 0)\n",
                    "kill of self through a module alias":
                    "import os as o\no.kill(o.getpid(), 0)\n",
                    "kill of self through posix":
                    "import posix\nposix.kill(posix.getpid(), 0)\n",
                    "kill of self through a module attribute's os":
                    "training.os.kill(training.os.getpid(), 0)\n",
                }))

    @classmethod
    def kill_seam_loads(cls, sources):
        """(iv). Passing the real functions explicitly (`ProcessControl(..., killpg=os.killpg,
        ...)`) satisfies (i), (ii) and the tripwire: only this clause stops it.

        `killpg` and `getpgid` may not be loaded from os or posix in any spelling
        `_is_os` recognises, as `<os>.X`, `getattr(<os>, "X")` or `getattr(<os>,
        <a computed name>)`, nor from-imported (`from os import X` or `*`). `kill`
        likewise, except the one allowed call, `os.kill(os.getpid(), ...)`: any
        other target could be `-pgid`, which is `killpg` by another name.
        `examined` is the allowed calls.
        """
        problems, examined = [], []
        for rel, text in sources.items():
            tree = ast.parse(text)
            names = cls._os_names(tree)
            allowed = cls._allowed_kills(tree)
            examined.extend(f"{rel}:{line}" for line in allowed.values())
            problems.extend(cls._banned_from_imports(rel, tree))
            for node in ast.walk(tree):
                loaded = cls._banned_load(node, names, allowed)
                if loaded is not None:
                    problems.append(f"{rel}:{loaded.lineno} {ast.unparse(loaded)}")
        return problems, examined

    @classmethod
    def _is_os(cls, expr, names):
        """Whether `expr` spells the os or posix module.

        A name in `names` (`_os_names`); any attribute named `os` or `posix`,
        which is how a test reaches a module's own import (`training.os`,
        `mrl.training.os`, `subprocess.os`, `os.path.os`); `__import__("os")`
        or `importlib.import_module("os")`; or a constant subscript such as
        `sys.modules["posix"]`.
        """
        if isinstance(expr, ast.Name):
            return expr.id in names
        if isinstance(expr, ast.Attribute):
            return expr.attr in cls.OS_MODULES
        named = None
        if isinstance(expr, ast.Call) and cls._last_name(expr.func) in cls.IMPORT_CALLS:
            named = expr.args[0] if expr.args else None
        elif isinstance(expr, ast.Subscript):
            named = expr.slice
        return isinstance(named, ast.Constant) and named.value in cls.OS_MODULES

    @classmethod
    def _os_names(cls, tree):
        """Every name `tree` binds to os or posix, anywhere in the file.

        The two module names; `import os as o`; `from <any module> import os
        [as o]`; and an assignment `o = <anything _is_os accepts>`, plain or
        annotated. The assignments are read in `ast.walk` order, so a chain
        (`o = os; p = o`) is followed when each link comes first in that order.
        """
        names = set(cls.OS_MODULES) | {
            alias.asname or alias.name
            for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names if alias.name in cls.OS_MODULES
        }
        for node in ast.walk(tree):
            targets, value = cls._assignment(node)
            if value is not None and cls._is_os(value, names):
                names |= {target.id for target in targets if isinstance(target, ast.Name)}
        return names

    @staticmethod
    def _assignment(node):
        """`(targets, value)` of a plain or annotated assignment with a value, else `([], None)`."""
        if isinstance(node, ast.Assign):
            return node.targets, node.value
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            return [node.target], node.value
        return [], None

    @staticmethod
    def _bare_os_call(expr, attr):
        """`expr` if it is a call `os.<attr>(...)` on the bare name `os`, else None."""
        if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute)
                and expr.func.attr == attr and isinstance(expr.func.value, ast.Name)
                and expr.func.value.id == "os"):
            return expr
        return None

    @classmethod
    def _allowed_kills(cls, tree):
        """`{id(the os.kill attribute): line}` for every `os.kill(os.getpid(), ...)`.

        Spelled exactly so, with the bare name `os` on both calls and no argument
        to `getpid`: an alias, `posix` or a module attribute's `os` is not the
        allowed shape (spec §4.8 criterion 6 (iv)), even with the same target.
        """
        allowed = {}
        for node in ast.walk(tree):
            kill = cls._bare_os_call(node, "kill")
            if kill is None or not kill.args:
                continue
            target = cls._bare_os_call(kill.args[0], "getpid")
            if target is not None and not (target.args or target.keywords):
                allowed[id(kill.func)] = kill.lineno
        return allowed

    @classmethod
    def _banned_from_imports(cls, rel, tree):
        """`from os|posix import killpg|getpgid|kill|*`, each alias one problem."""
        return [
            f"{rel}:{node.lineno} from {node.module} import {alias.name}" for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module in cls.OS_MODULES
            for alias in node.names if alias.name in cls.BANNED | {"*"}
        ]

    @classmethod
    def _banned_load(cls, node, names, allowed):
        """`node` if it loads a banned function from os or posix, else None.

        `<os>.X`, unless it is the callee of an allowed kill; `getattr(<os>,
        "X")`; and `getattr(<os>, <anything but a constant>)`, whose name cannot
        be read (`{n: getattr(os, n) for n in ("getpgid", "killpg")}`).
        """
        if isinstance(node, ast.Attribute):
            loads = (isinstance(node.ctx, ast.Load) and node.attr in cls.BANNED
                     and cls._is_os(node.value, names) and id(node) not in allowed)
            return node if loads else None
        if (isinstance(node, ast.Call) and cls._last_name(node.func) == "getattr"
                and len(node.args) >= 2 and cls._is_os(node.args[0], names)):
            attr = node.args[1]
            harmless = isinstance(attr, ast.Constant) and attr.value not in cls.BANNED
            return None if harmless else node
        return None

    # ── (v) system() builds exactly the real functions

    @classmethod
    def clause_v(cls):
        return cls._clause(
            "(v)",
            "ProcessControl.system() builds exactly the real functions",
            reads=("training", ),
            check=cls.system_builds_the_real_functions,
            population="one system() construction",
            populated=lambda seen: len(seen) == 1,
            plants=cls._at(
                cls.TRAINING, {
                    "a wrong function":
                    ("class ProcessControl:\n"
                     "    @classmethod\n"
                     "    def system(cls):\n"
                     "        return cls(spawn=subprocess.Popen, getpgid=os.getpgid,\n"
                     "                   killpg=os.killpg, install_signal=signal.getsignal)\n"),
                }))

    @classmethod
    def system_builds_the_real_functions(cls, sources):
        """(v). The values are compared as `ast.unparse` strings, so this test loads none
        of them and passes (iv) itself. `examined` is the one construction.
        """
        problems, examined = [], []
        for rel, text in sources.items():
            for process_control in cls._process_control_classes(ast.parse(text)):
                call = cls._system_return(process_control)
                if call is None:
                    problems.append(f"{rel}: ProcessControl.system is not one `return cls(...)`")
                    continue
                examined.append(f"{rel}:{call.lineno}")
                built = {k.arg: ast.unparse(k.value) for k in call.keywords}
                if (cls._last_name(call.func) not in ("cls", "ProcessControl") or call.args
                        or built != cls.REAL_SYSTEM):
                    problems.append(f"{rel}:{call.lineno} system() builds {ast.unparse(call)}")
        return problems, examined

    @classmethod
    def _system_return(cls, process_control):
        """The call `system()` returns, when there is exactly one `system` and its body
        (docstring aside) is the one statement `return <call>`; otherwise None."""
        systems = [
            node for node in process_control.body
            if isinstance(node, ast.FunctionDef) and node.name == "system"
        ]
        body = [s for s in systems[0].body if not cls._is_docstring(s)] if len(systems) == 1 else []
        only = body[0] if len(body) == 1 else None
        returned = only.value if isinstance(only, ast.Return) else None
        return returned if isinstance(returned, ast.Call) else None

    # ── (vii) killpg/getpgid called only inside the §2a guard

    @classmethod
    def clause_vii(cls):
        return cls._clause(
            "(vii)",
            "killpg/getpgid are called only inside _signal_process_group",
            reads=("training", ),
            check=cls.kill_calls_only_in_the_guard,
            population="the guard's own getpgid and killpg calls",
            populated=lambda seen: {"killpg", "getpgid"} <= set(seen),
            plants=cls._at(
                cls.TRAINING, {
                    "a direct kill in a new method":
                    ("class _LiveAttempt:\n"
                     "    def kill(self):\n"
                     "        os.killpg(os.getpgid(self.child.pid), signal.SIGTERM)\n"),
                }))

    @classmethod
    def kill_calls_only_in_the_guard(cls, sources):
        """(vii). A direct call anywhere else bypasses the guard, and the runtime layers
        govern ProcessControl, not a direct call. `examined` is the callee names found
        inside the guard.
        """
        problems, examined = [], []
        for rel, text in sources.items():
            tree = ast.parse(text)
            inside = cls._inside_the_guard(tree)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and cls._last_name(node.func) in cls.GUARDED_CALLS):
                    continue
                if id(node) in inside:
                    examined.append(cls._last_name(node.func))
                else:
                    problems.append(f"{rel}:{node.lineno} {ast.unparse(node)}")
        return problems, examined

    @staticmethod
    def _inside_the_guard(tree):
        """The ids of every node in the module-level `_signal_process_group`."""
        return {
            id(node)
            for guard in tree.body
            if isinstance(guard, ast.FunctionDef) and guard.name == "_signal_process_group"
            for node in ast.walk(guard)
        }


def test_kill_seam_static_safety():
    """No test, and no training code outside the §2a guard, can reach the real kill seam.

    gh#163 spec §4.8 criterion 6. On 2026-09-23 an agent's throwaway script ran
    the real `killpg(getpgid(1), SIGTERM)`, which is `kill(-1, SIGTERM)`, and
    ended the user's desktop session. The runtime layers (a ProcessControl
    without field defaults; the tests/conftest.py tripwire) each have a way
    round them that only a static check sees, so this test reads source by AST.
    It NEVER imports or calls what it checks: training.py is read through
    `training.__file__` (which is also how the reach floor credits this test to
    `training`), and every tests/*.py from disk.

    THE CLAUSES, each a checker with its plants in `_KillSeamClauses` above.
    A plant is source text that is parsed and never run, and must fail its
    clause: a clause no plant can fail is not evidence.
      (i)   ProcessControl's dataclass fields have no defaults, and are exactly
            the four fields.
      (ii)  Every ProcessControl(...) construction under tests/, through the
            name or an import or assignment alias of it, passes the four fields
            as four keywords: no positional argument and no `*`/`**` splat.
      (iv)  No load of os/posix `killpg` or `getpgid` under tests/, in any
            spelling the clause reads: through `os`, `posix`, an alias of
            either, a module attribute's `os` (`training.os`), `__import__`,
            `importlib.import_module` or `sys.modules[...]`; `from <m> import X`
            or `*`; `getattr` with that name or a computed one. `kill` likewise,
            except a call spelled exactly `os.kill(os.getpid(), ...)`.
      (v)   ProcessControl.system() builds exactly spawn=subprocess.Popen,
            getpgid=os.getpgid, killpg=os.killpg, install_signal=signal.signal.
      (vii) In training.py a call whose callee is named `killpg` or `getpgid`
            (a bare name, or the last attribute on any receiver) occurs only
            inside `_signal_process_group`, which holds the §2a guard.
    Each clause also asserts a non-empty population on the real tree, so a
    checker that silently examines nothing cannot pass: the four fields (i), the
    tripwire's own construction in tests/conftest.py (ii), the real-SIGTERM
    test's `os.kill(os.getpid(), SIGTERM)` (iv), the system() construction (v),
    and the guard's own getpgid and killpg calls (vii). The list of clause
    labels is pinned below, so a clause dropped from `_KillSeamClauses.clauses()`
    is red, not silently unchecked.

    ADDING A CLAUSE: see `_KillSeamClauses`. Clauses (iii) and (vi) join in the
    W5 execute commit: (iii) cannot hold until execute_training_attempt stops
    loading os.killpg/os.getpgid/signal.signal itself, and (vi) pins the
    `process=None` resolution that commit writes.

    RESIDUAL. The clauses read spellings, not values. Measured against (iv),
    these still pass: `vars(os)["killpg"]`, `os.__dict__["killpg"]`,
    `operator.attrgetter("killpg")(os)`, `inspect.getattr_static(os, "killpg")`,
    `importlib.import_module(<a variable>).killpg`,
    `sys.modules.get("posix").killpg`, an os bound some other way (a parameter
    default `def f(o=os)`, a walrus, a function's return value), ctypes' libc
    `killpg`, a shell `kill` in a subprocess, and code inside a string a child
    interpreter runs. A wrapper passes only when what it wraps does:
    `functools.partial(os.killpg, 0)` is caught, because it loads `os.killpg`.
    They are a backstop for the mistakes this branch has seen, not a sandbox.
    """
    training_file = Path(training.__file__).resolve()
    assert training_file.is_relative_to(ROOT), (
        f"scripts.modal_runner.training was imported from {training_file}, outside this checkout "
        f"({ROOT}), so these clauses would read another tree's training.py")
    sources = {
        "training": {
            _KillSeamClauses.TRAINING: training_file.read_text(encoding="utf-8")
        },
        "tests": {
            path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted((ROOT / "tests").rglob("*.py"))
        },
    }
    assert {"tests/conftest.py",
            "tests/test_modal_training.py"} <= set(sources["tests"]), sorted(sources["tests"])[:5]
    clauses = _KillSeamClauses.clauses()
    pinned = ["(i)", "(ii)", "(iv)", "(v)", "(vii)"]
    assert [clause.label for clause in clauses] == pinned, (
        "_KillSeamClauses.clauses() lost or gained a clause; a new one also adds its label here")
    for clause in clauses:
        real = {rel: text for scope in clause.reads for rel, text in sources[scope].items()}
        problems, examined = clause.check(real)
        assert clause.populated(examined), (
            f"{clause.name}: examined {examined!r}, not {clause.population}. The checker read "
            "nothing it exists to check, so its green would not be evidence")
        assert problems == [], f"{clause.name}: {problems}"
        assert clause.plants, f"{clause.name} has no plant, so nothing shows it can fail"
        for plant, planted in clause.plants.items():
            assert clause.check(planted)[0], (f"{clause.name}: the plant {plant!r} passed it, so "
                                              "the clause cannot fail and is not evidence")


def test_process_control_tripwire_poisons_system(_process_control_tripwire,
                                                 process_control_tripwire_error, monkeypatch):
    """Under pytest, ProcessControl.system() returns a control whose every field raises.

    gh#163 spec §2a layer 3, §4.8 criterion 6. `execute_training_attempt` is to
    resolve `process=None` to `ProcessControl.system()`, and the autouse fixture
    in tests/conftest.py patches `system` to return its poisoned control. So a
    test or client wrapper that forgets `process` fails loudly on a
    ProcessControlTripwire instead of spawning a real child or signalling a real
    process group.

    SAFETY, in order:
      * The FIRST statement calls `system()`, which only builds a dataclass, and
        asserts that it IS the fixture's poison. Under the spec's knock-out 4(k)
        (the fixture's one `setattr` line deleted) `system()` returns the real
        control: this assertion fails and nothing below runs.
      * The fields called are the FIXTURE'S OWN object's, never those of
        anything `system()` returned. Never change that.
      * The arguments are inert on the real functions too: `killpg(0, 0)` sends
        signal 0, which sends nothing; `getpgid(0)` reads the caller's group;
        `Popen([])` raises IndexError before it forks; `signal.signal(0, None)`
        rejects signal 0. So a real function slipped into the poison fails the
        type check below having done nothing.
    Then, before any field is called: the poison's type is a RuntimeError and
    neither a ValueError nor a ProcessLookupError, the two the attempt swallows
    (the handler install; getpgid/killpg); and the test's own
    `monkeypatch.undo()` leaves the poison in place, because the fixture patches
    through its own MonkeyPatch (tests/conftest.py). All four fields are called,
    because one call would pass a poison whose `spawn` is inert, and spawn is
    the first field an attempt calls; and the field set is compared with the
    dataclass's, so a fifth field cannot go unpoisoned. The resolution path
    itself (execute_training_attempt with `process` omitted) gets its own
    control when that path lands (W5 execute).
    """
    assert training.ProcessControl.system() is _process_control_tripwire, (
        "ProcessControl.system() is not the tests/conftest.py tripwire's poisoned control, so a "
        "forgotten `process` in this session would reach the REAL spawn and killpg. Nothing "
        "was called.")
    assert issubclass(process_control_tripwire_error, RuntimeError), (
        "the tripwire's poison is not a RuntimeError subclass (spec §4.8 criterion 6)")
    assert not issubclass(process_control_tripwire_error, (ValueError, ProcessLookupError)), (
        "the attempt swallows a ValueError from the handler install and a ProcessLookupError "
        "from getpgid/killpg, so a poison of either type would be silent exactly there")
    monkeypatch.undo()
    assert training.ProcessControl.system() is _process_control_tripwire, (
        "the test's own monkeypatch.undo() lifted the tripwire: the fixture must patch through "
        "its own pytest.MonkeyPatch.context(), not the test's monkeypatch. Nothing was called.")
    poisoned = _process_control_tripwire
    calls = {
        "killpg": lambda: poisoned.killpg(0, 0),
        "getpgid": lambda: poisoned.getpgid(0),
        "spawn": lambda: poisoned.spawn([]),
        "install_signal": lambda: poisoned.install_signal(0, None),
    }
    assert {field.name for field in dataclasses.fields(poisoned)} == set(calls)
    for field, call in calls.items():
        with pytest.raises(process_control_tripwire_error, match=f"ProcessControl.{field} was"):
            call()


# ── Attempt outcome: exit mapping and completion evidence ──────────────────


def test_exit_zero_fails_when_completion_evidence_invalid(tmp_path):
    events: list[str] = []

    class RecordingSink(io.StringIO):

        def close(self):
            events.append("log_closed")
            super().close()

    def commit():
        events.append("commit")

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=FakeChild(returncode=0, stdout=b"done\n"),
            commit=commit,
            log_sink=RecordingSink(),
            manifest=_make_manifest(),
        ))
    result = mrl.execute_training_attempt(**kwargs)
    assert result != mrl.REDELIVERED
    assert result.status is core.Status.FAILED
    assert result.reason == training.REASON_INVALID_EVIDENCE
    assert result.exit_code == 0
    persisted = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "failed"
    assert "log_closed" in events
    assert "commit" in events[events.index("log_closed") + 1:]


def test_exit_zero_with_valid_evidence_completes(tmp_path):
    run_root, manifest, effective, ckpt = _minimal_completed_tree(tmp_path)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=FakeChild(returncode=0, stdout=b"done\n"),
            manifest=manifest,
        ))
    result = mrl.execute_training_attempt(**kwargs)
    assert result.status is core.Status.COMPLETED
    assert result.reason is None
    assert result.exit_code == 0
    persisted = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "completed"
    payload = json.loads((kwargs["run_root"] / "result.json").read_text())
    assert payload["status"] == "completed"
    assert payload["exit_code"] == 0
    assert payload["last_step"] == effective
    assert payload["checkpoint_sha256"] == core.sha256_file(ckpt)


def test_dead_run_and_timeout_have_distinct_reasons(tmp_path):
    dead_root = tmp_path / "dead"
    dead_root.mkdir()
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            dead_root,
            child=FakeChild(returncode=3, stdout=b"dead\n"),
            manifest=_make_manifest(),
        ))
    (kwargs["run_root"] / "checkpoints").mkdir(exist_ok=True)
    (kwargs["run_root"] / "checkpoints" / "dust2_policy_dead.pt").write_bytes(b"autopsy")
    dead = mrl.execute_training_attempt(**kwargs)
    assert dead.status is core.Status.FAILED
    assert dead.reason == training.REASON_DEAD_RUN
    assert dead.exit_code == 3
    assert json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())["status"] == "failed"

    clock = _FakeClock()
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    timeout_root = tmp_path / "timeout"
    timeout_root.mkdir()
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            timeout_root,
            child=child,
            now=clock.now,
            timeout=timedelta(minutes=120),
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
            manifest=_make_manifest(),
        ))
    thread, finished, boxed = _run_attempt_in_thread(kwargs)
    try:
        _wait_until_handlers(hooks["installed"], hooks["originals"])
        clock.advance(120 * 60)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    timed_out = boxed[0]
    assert not isinstance(timed_out, Exception), timed_out
    assert timed_out.status is core.Status.INTERRUPTED
    assert timed_out.reason == training.REASON_TIMEOUT
    assert timed_out.reason != dead.reason
    assert json.loads(
        (kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())["status"] == "interrupted"
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]
