"""Behavior tests for scripts.modal_runner.training: the attempt, publication, supervision.

One of the per-module runner test files (RUNNER_TEST_FILES in
tests/modal_runner_tables.py). Before you add, move or delete a test here, or
add a helper, read THE PLACEMENT RULE FOR RUNNER TESTS in
tests/test_modal_packaging.py: which file a test belongs in, what the change
costs in the seam manifest, and where helpers go.

Deterministic patch-binding controls live in test_modal_patch_bindings.py;
interruption tests here retain their original negative assertions.
"""
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
