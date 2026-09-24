"""Behavior tests for scripts.modal_runner.state: run status, reservation, delivery.

One of the eight per-module runner test files (RUNNER_TEST_FILES in
tests/modal_runner_tables.py), split by module from the one unsplit runner test
file in W4. A test lives in the file of the module whose behaviour it tests:
the seam manifest (tests/fixtures/modal_test_seam_manifest.json) records that
placement, and the seam gate in tests/test_modal_packaging.py checks it from
below with the reach floor. Tests reach private library names through their
owning submodules; the package facade exposes the production caller surface.
Helpers reached by tests in two or more seam files live in
tests/modal_test_helpers.py, with ownership recomputed by classify_seam.
"""
import json
import sys
import threading
from datetime import UTC, timedelta
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # scripts/ is a namespace package; tests import scripts.modal_runner
    # the same way the later CLIs will. Do not rely on the editable install.
    sys.path.insert(0, str(ROOT))

import scripts.modal_runner as mrl                                                 # noqa: E402, I001
from scripts.modal_runner import core, state                                       # noqa: E402, I001
from tests.modal_test_helpers import FakeRegistry, _aware, _minimal_completed_tree # noqa: E402

# ── Run state: atomic writes, transitions, heartbeat, status, artifacts ────


def test_atomic_write_json_replaces_and_cleans_temp_on_failure(tmp_path):
    path = tmp_path / "STATUS.json"
    state.atomic_write_json(path, {"ok": True})
    assert json.loads(path.read_text()) == {"ok": True}

    def boom(src, dst):
        raise OSError("injected replace failure")

    with pytest.raises(OSError, match="injected"):
        state.atomic_write_json(path, {"ok": False}, replace=boom)
    assert json.loads(path.read_text()) == {"ok": True}
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "STATUS.json"]
    assert leftovers == []


def test_status_transitions_are_monotonic_and_attempt_owned(tmp_path):
    from datetime import datetime

    run_root = tmp_path / "run"
    run_root.mkdir()
    lock = threading.Lock()
    now = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    first = state.transition_status(run_root,
                                    core.Status.PREPARING,
                                    now=now,
                                    attempt_id="attempt-a",
                                    lock=lock)
    assert first.status is core.Status.PREPARING
    assert first.attempt_id == "attempt-a"
    state.transition_status(run_root,
                            core.Status.BUILDING,
                            now=now,
                            attempt_id="attempt-a",
                            lock=lock)
    state.transition_status(run_root,
                            core.Status.TRAINING,
                            now=now,
                            attempt_id="attempt-a",
                            lock=lock)
    done = state.transition_status(run_root,
                                   core.Status.COMPLETED,
                                   now=now,
                                   attempt_id="attempt-a",
                                   lock=lock)
    assert done.status is core.Status.COMPLETED
    # Idempotent same-terminal write by the original delivery.
    again = state.transition_status(run_root,
                                    core.Status.COMPLETED,
                                    now=now,
                                    attempt_id="attempt-a",
                                    lock=lock)
    assert again.status is core.Status.COMPLETED
    with pytest.raises(mrl.ValidationError):
        state.transition_status(run_root,
                                core.Status.TRAINING,
                                now=now,
                                attempt_id="attempt-a",
                                lock=lock)
    before = (run_root / "STATUS.json").read_bytes()
    # Redelivered delivery has no authority and must not touch the file.
    denied = state.transition_status(run_root,
                                     core.Status.FAILED,
                                     now=now,
                                     attempt_id="attempt-b",
                                     lock=lock)
    assert denied is None
    assert (run_root / "STATUS.json").read_bytes() == before


def _advance_to_training(run_root, attempt_id="a1", *, lock):
    state.transition_status(run_root,
                            core.Status.PREPARING,
                            now=_aware(),
                            attempt_id=attempt_id,
                            lock=lock)
    state.transition_status(run_root,
                            core.Status.BUILDING,
                            now=_aware(),
                            attempt_id=attempt_id,
                            lock=lock)
    return state.transition_status(run_root,
                                   core.Status.TRAINING,
                                   now=_aware(),
                                   attempt_id=attempt_id,
                                   lock=lock)


def test_heartbeat_refreshes_updated_at_under_lock(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    lock = threading.Lock()
    _advance_to_training(run_root, lock=lock)
    # Fake clock: a beat at each 60s mark must refresh updated_at.
    last = None
    for minute in (1, 2):
        now = _aware(minute=minute)
        last = state.write_heartbeat(run_root, now=now, attempt_id="a1", lock=lock)
        assert last is not None
        assert last.status is core.Status.TRAINING
        assert last.updated_at == now.isoformat()
        persisted = json.loads((run_root / "STATUS.json").read_text())
        assert persisted["updated_at"] == now.isoformat()
    assert last is not None


def test_blocked_heartbeat_cannot_clobber_completed(tmp_path):
    """Hold the lock, queue a beat, write completed, then release.

    The queued heartbeat must observe the terminal write and leave
    STATUS.json as completed. This is the interleaving an unlocked
    transition_status would lose: beat reads training, terminal write
    lands, beat writes training back.
    """
    run_root = tmp_path / "run"
    run_root.mkdir()
    lock = threading.Lock()
    _advance_to_training(run_root, lock=lock)

    lock.acquire()
    started = threading.Event()
    beat_status = []

    def beat():
        started.set()
        beat_status.append(
            state.write_heartbeat(run_root, now=_aware(minute=3), attempt_id="a1", lock=lock))

    worker = threading.Thread(target=beat)
    worker.start()
    try:
        assert started.wait(timeout=2.0)
        # started.set() races the acquire; park long enough to be blocked.
        threading.Event().wait(0.05)
        assert worker.is_alive()

        # Critical section is already held; do not re-enter the same Lock.
        written = state._transition_status_unlocked(run_root,
                                                    core.Status.COMPLETED,
                                                    now=_aware(minute=2),
                                                    attempt_id="a1")
        assert written is not None
        assert written.status is core.Status.COMPLETED
    finally:
        lock.release()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert json.loads((run_root / "STATUS.json").read_text())["status"] == "completed"
    assert beat_status and beat_status[0].status is core.Status.COMPLETED


def test_late_heartbeat_cannot_replace_terminal(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    lock = threading.Lock()
    stop = threading.Event()
    _advance_to_training(run_root, lock=lock)

    def heartbeat_loop():
        while not stop.is_set():
            state.write_heartbeat(run_root, now=_aware(minute=1), attempt_id="a1", lock=lock)
            stop.wait(0.01)

    worker = threading.Thread(target=heartbeat_loop)
    worker.start()
    # Terminal cleanup stops/joins the heartbeat, then transitions while
    # holding the shared lock. A delayed beat after join must no-op.
    stop.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    state.transition_status(run_root,
                            core.Status.COMPLETED,
                            now=_aware(minute=2),
                            attempt_id="a1",
                            lock=lock)
    beat = state.write_heartbeat(run_root, now=_aware(minute=3), attempt_id="a1", lock=lock)
    assert beat is not None
    assert beat.status is core.Status.COMPLETED
    assert json.loads((run_root / "STATUS.json").read_text())["status"] == "completed"


def test_derive_status_stale_after_five_minutes_does_not_mutate(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    written = _advance_to_training(run_root, lock=threading.Lock())
    before = (run_root / "STATUS.json").read_bytes()
    derived = state.derive_status(written, now=_aware(hour=12, minute=5))
    assert derived.stale is True
    assert derived.status is core.Status.INTERRUPTED
    assert (run_root / "STATUS.json").read_bytes() == before
    fresh = state.derive_status(written, now=_aware(minute=4, second=59))
    assert fresh.stale is False
    assert fresh.status is core.Status.TRAINING


def test_reservation_without_status_is_preparing_then_interrupted(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    reserved_at = _aware()
    (run_root / "reservation.json").write_text(
        json.dumps({
            "attempt_id": "a1",
            "created_at": reserved_at.isoformat()
        }))
    early = state.derive_run_view(run_root, now=_aware(minute=4))
    assert early.status is core.Status.PREPARING
    assert early.reason == "no-heartbeat"
    assert early.stale is False
    late = state.derive_run_view(run_root, now=_aware(minute=5))
    assert late.status is core.Status.INTERRUPTED
    assert late.reason == "no-heartbeat"
    assert late.stale is True
    assert not (run_root / "STATUS.json").exists()


def test_list_run_artifacts_keeps_unknown_trainer_files(tmp_path):
    run_root, manifest, _, _ = _minimal_completed_tree(tmp_path)
    extra = run_root / "checkpoints" / "notes.txt"
    extra.write_text("keep me\n")
    nested = run_root / "checkpoints" / "extra" / "weird.bin"
    nested.parent.mkdir()
    nested.write_bytes(b"\x00\x01")
    listed = {path.relative_to(run_root).as_posix() for path in state.list_run_artifacts(run_root)}
    assert "checkpoints/notes.txt" in listed
    assert "checkpoints/extra/weird.bin" in listed
    assert "checkpoints/dust2_policy.pt" in listed


# ── Run reservation: registry / artifact protocols, durable commit ─────────


class FakeArtifactIndex:
    """In-memory Volume: client PurePosixPath keys, durable only after commit."""

    def __init__(self):
        self._lock = threading.Lock()
        self.committed: dict[PurePosixPath, bytes] = {}
        self.staged: dict[PurePosixPath, bytes] = {}
        self.events: list[tuple[object, ...]] = []
        self.replace_after_read: dict[PurePosixPath, bytes | None] = {}

    def exists(self, path: PurePosixPath) -> bool:
        with self._lock:
            return path in self.committed

    def put_file(self, path: PurePosixPath, data: bytes) -> None:
        if not isinstance(path, PurePosixPath) or path.is_absolute():
            raise AssertionError(f"client Volume path must be relative PurePosixPath, got {path!r}")
        with self._lock:
            self.staged[path] = data
            self.events.append(("put_file", path))

    def commit(self) -> None:
        with self._lock:
            self.committed.update(self.staged)
            self.staged.clear()
            self.events.append(("commit", ))

    def read_file(self, path: PurePosixPath) -> bytes | None:
        with self._lock:
            if path in self.replace_after_read:
                current = self.committed.get(path)
                replacement = self.replace_after_read.pop(path)
                if replacement is None:
                    self.committed.pop(path, None)
                else:
                    self.committed[path] = replacement
                return current
            return self.committed.get(path)


def test_reserve_run_commits_reservation_immediately_after_dict_claim():
    """Winning claim must persist runs/<id>/reservation.json before any other upload."""
    registry = FakeRegistry()
    artifacts = FakeArtifactIndex()
    now = _aware()
    mrl.reserve_run(registry, artifacts, "ok-id", "attempt-a", now=now)

    reservation_path = mrl.RUNS_ROOT / "ok-id" / mrl.RESERVATION_FILENAME
    assert registry.events[0] == ("put_if_absent", "run:ok-id")
    assert artifacts.events == [("put_file", reservation_path), ("commit", )]
    claim = registry.get("run:ok-id")
    assert claim is not None
    assert claim["attempt_id"] == "attempt-a"
    assert claim["created_at"] == now.isoformat()
    payload = json.loads(artifacts.committed[reservation_path])
    assert payload["attempt_id"] == "attempt-a"
    assert payload["created_at"] == now.isoformat()
    # Reservation is the only Volume write: no source, checkpoint, or Function.
    assert all(event[0] in {"put_file", "commit"} for event in artifacts.events)
    assert artifacts.events[0][1] == reservation_path


# ── Run reservation: race, expired-Dict Volume fallback, upload failure ────


def test_concurrent_reserve_run_admits_exactly_one_attempt():
    registry = FakeRegistry()
    artifacts = FakeArtifactIndex()
    results: list[tuple[str, str]] = []
    barrier = threading.Barrier(2)

    def worker(attempt_id: str) -> None:
        barrier.wait()
        try:
            mrl.reserve_run(registry, artifacts, "ok-id", attempt_id, now=_aware())
            results.append(("ok", attempt_id))
        except mrl.ValidationError:
            results.append(("reject", attempt_id))

    threads = [
        threading.Thread(target=worker, args=("attempt-a", )),
        threading.Thread(target=worker, args=("attempt-b", )),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    wins = [attempt for status, attempt in results if status == "ok"]
    losses = [attempt for status, attempt in results if status == "reject"]
    assert len(wins) == 1
    assert len(losses) == 1
    winner = wins[0]
    assert registry.get("run:ok-id")["attempt_id"] == winner
    reservation = json.loads(artifacts.committed[mrl.RUNS_ROOT / "ok-id" /
                                                 mrl.RESERVATION_FILENAME])
    assert reservation["attempt_id"] == winner


def test_dict_miss_with_existing_volume_manifest_rejects_run():
    registry = FakeRegistry()
    artifacts = FakeArtifactIndex()
    artifacts.committed[mrl.RUNS_ROOT / "ok-id" / core.MANIFEST_FILENAME] = b"{}\n"
    with pytest.raises(mrl.ValidationError):
        mrl.reserve_run(registry, artifacts, "ok-id", "attempt-b", now=_aware())
    assert registry.get("run:ok-id") is None
    assert artifacts.events == []


def test_upload_failure_after_reservation_keeps_run_id_and_records_failure_code():
    registry = FakeRegistry()
    artifacts = FakeArtifactIndex()
    mrl.reserve_run(registry, artifacts, "ok-id", "attempt-a", now=_aware())

    def boom_upload() -> None:
        raise OSError("could not upload /secrets/key to sources/dead.tar.gz")

    with pytest.raises(OSError, match="could not upload"):
        mrl.finish_reservation(registry, artifacts, "ok-id", "attempt-a", upload=boom_upload)
    claim = registry.get("run:ok-id")
    assert claim is not None
    assert claim["attempt_id"] == "attempt-a"
    assert claim["failure_code"] == "upload_failed"
    assert "secret" not in json.dumps(claim)
    assert "sources/dead.tar.gz" not in json.dumps(claim)
    assert mrl.RUNS_ROOT / "ok-id" / mrl.RESERVATION_FILENAME in artifacts.committed
    with pytest.raises(mrl.ValidationError):
        mrl.reserve_run(registry, artifacts, "ok-id", "attempt-b", now=_aware())


def test_expired_dict_still_rejects_when_volume_reservation_exists():
    registry = FakeRegistry()
    artifacts = FakeArtifactIndex()
    mrl.reserve_run(registry, artifacts, "ok-id", "attempt-a", now=_aware())

    def boom_upload() -> None:
        raise OSError("source upload failed")

    with pytest.raises(OSError):
        mrl.finish_reservation(registry, artifacts, "ok-id", "attempt-a", upload=boom_upload)
    # Seven inactive days evict the Dict lease; the Volume reservation remains.
    registry.expire("run:ok-id")
    assert registry.get("run:ok-id") is None
    with pytest.raises(mrl.ValidationError):
        mrl.reserve_run(registry, artifacts, "ok-id", "attempt-b", now=_aware())
    assert registry.get("run:ok-id") is None


# ── Attempt delivery: redelivery / idempotent terminal behavior ────────────


def test_first_attempt_claim_owns_canonical_state_writes(tmp_path):
    registry = FakeRegistry()
    artifacts = FakeArtifactIndex()
    mrl.reserve_run(registry, artifacts, "ok-id", "attempt-a", now=_aware())
    run_root = tmp_path / "run"
    run_root.mkdir()
    lock = threading.Lock()

    def train() -> str:
        status = state.transition_status(
            run_root,
            core.Status.PREPARING,
            now=_aware(),
            attempt_id="attempt-a",
            lock=lock,
        )
        assert status is not None
        (run_root / "result.json").write_text("{}\n")
        (run_root / "train.log").write_text("ok\n")
        (run_root / "checkpoints").mkdir()
        (run_root / "checkpoints" / "dust2_policy.pt").write_bytes(b"ckpt")
        artifacts.put_file(mrl.RUNS_ROOT / "ok-id" / "result.json", b"{}\n")
        artifacts.commit()
        return "trained"

    result = state.deliver_attempt(registry, artifacts, attempt_id="attempt-a", train=train)
    assert result == "trained"
    assert mrl.claim_attempt(registry, "attempt-a") is False
    persisted = json.loads((run_root / "STATUS.json").read_text())
    assert persisted["attempt_id"] == "attempt-a"
    assert persisted["status"] == "preparing"
    assert (run_root / "result.json").is_file()
    assert (run_root / "train.log").is_file()
    assert (run_root / "checkpoints" / "dust2_policy.pt").is_file()
    claim = registry.get("attempt:attempt-a")
    assert claim is not None
    assert claim["attempt_id"] == "attempt-a"


def test_same_input_redelivery_returns_without_train_or_writes(tmp_path):
    registry = FakeRegistry()
    artifacts = FakeArtifactIndex()
    mrl.reserve_run(registry, artifacts, "ok-id", "attempt-a", now=_aware())
    run_root = tmp_path / "run"
    run_root.mkdir()
    lock = threading.Lock()

    def train() -> str:
        state.transition_status(
            run_root,
            core.Status.PREPARING,
            now=_aware(),
            attempt_id="attempt-a",
            lock=lock,
        )
        (run_root / "result.json").write_text('{"status":"completed"}\n')
        (run_root / "train.log").write_text("first\n")
        (run_root / "checkpoints").mkdir()
        (run_root / "checkpoints" / "dust2_policy.pt").write_bytes(b"ckpt")
        artifacts.put_file(mrl.RUNS_ROOT / "ok-id" / "result.json", b"{}\n")
        artifacts.commit()
        return "trained"

    assert state.deliver_attempt(registry, artifacts, attempt_id="attempt-a",
                                 train=train) == "trained"
    before_status = (run_root / "STATUS.json").read_bytes()
    before_result = (run_root / "result.json").read_bytes()
    before_log = (run_root / "train.log").read_bytes()
    before_ckpt = (run_root / "checkpoints" / "dust2_policy.pt").read_bytes()
    before_events = list(artifacts.events)
    before_committed = dict(artifacts.committed)

    def should_not_run() -> str:
        raise AssertionError("training callback must not run on redelivery")

    assert state.deliver_attempt(registry, artifacts, attempt_id="attempt-a",
                                 train=should_not_run) == "redelivered"
    assert (run_root / "STATUS.json").read_bytes() == before_status
    assert (run_root / "result.json").read_bytes() == before_result
    assert (run_root / "train.log").read_bytes() == before_log
    assert (run_root / "checkpoints" / "dust2_policy.pt").read_bytes() == before_ckpt
    assert artifacts.events == before_events
    assert artifacts.committed == before_committed


def test_different_attempt_cannot_reach_remote_wrapper():
    registry = FakeRegistry()
    artifacts = FakeArtifactIndex()
    mrl.reserve_run(registry, artifacts, "ok-id", "attempt-a", now=_aware())
    with pytest.raises(mrl.ValidationError):
        mrl.reserve_run(registry, artifacts, "ok-id", "attempt-b", now=_aware())
    assert registry.get("run:ok-id")["attempt_id"] == "attempt-a"
    # Loser never received a Function delivery, so no attempt:<id> claim exists.
    assert registry.get("attempt:attempt-b") is None
    assert registry.get("attempt:attempt-a") is None


# ── Heartbeat loop: a transient commit error does not stop it ──────────────


def test_heartbeat_loop_survives_transient_commit_error(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    lock = threading.Lock()
    state.transition_status(run_root,
                            core.Status.PREPARING,
                            now=_aware(),
                            attempt_id="a1",
                            lock=lock)
    recovered = threading.Event()
    commits = {"n": 0}

    def flaky_commit():
        commits["n"] += 1
        if commits["n"] == 1:
            raise RuntimeError("volume commit blip")
        recovered.set()

    def wait(event: threading.Event, _seconds: float) -> bool:
        return event.wait(0.01)

    worker = state.start_heartbeat_worker(
        run_root=run_root,
        attempt_id="a1",
        lock=lock,
        now=_aware,
        commit=flaky_commit,
        interval=timedelta(seconds=60),
        wait=wait,
    )
    try:
        assert recovered.wait(timeout=2.0)
        assert worker.thread.is_alive()
        assert commits["n"] >= 2
    finally:
        worker.stop_and_join()
    assert not worker.thread.is_alive()
