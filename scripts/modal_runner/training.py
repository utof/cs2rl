"""Training attempt execution and checkpoint publication."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath

from . import checkpoint, core, state
from .checkpoint import _iter_metrics_steps, validate_completed_run
from .core import (
    CHECKPOINT_NAME,
    CHECKPOINT_PUBLISH_REASON_NAME,
    CHECKPOINT_SIDECAR_NAME,
    DEAD_CHECKPOINT_NAME,
    HEARTBEAT_INTERVAL,
    RESULT_FILENAME,
    SCHEMA_VERSION,
    TRAIN_LOG_NAME,
    CompletionEvidence,
    LockLike,
    Manifest,
    PreparedSource,
    Registry,
    Status,
    ValidationError,
)
from .state import _stop_heartbeat, atomic_write_json, deliver_attempt, start_heartbeat_worker


@dataclass(frozen=True)
class RunResult:
    """Terminal result.json (design §7). Never written by a losing delivery."""

    schema_version: int
    status: Status
    exit_code: int
    started_at: str
    finished_at: str
    artifact_root: str
    checkpoint_sha256: str | None
    metrics_row_count: int
    last_step: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "artifact_root": self.artifact_root,
            "checkpoint_sha256": self.checkpoint_sha256,
            "metrics_row_count": self.metrics_row_count,
            "last_step": self.last_step,
        }


CHECKPOINT_SETTLE_SECONDS = 1.0
TERM_GRACE_SECONDS = 15.0
DEAD_RUN_EXIT_CODE = 3
POLL_INTERVAL_SECONDS = 0.05
REASON_SIGNAL = "signal"
REASON_TIMEOUT = "timeout"
REASON_DEAD_RUN = "dead_run"
REASON_INVALID_EVIDENCE = "invalid_evidence"
REASON_NONZERO_EXIT = "nonzero_exit"
REASON_ERROR = "error"


@dataclass(frozen=True)
class TrainingAttemptResult:
    """Winner-only outcome. Losers return REDELIVERED, not this type."""

    status: Status
    reason: str | None = None
    exit_code: int | None = None


class _UnusedArtifacts:
    """deliver_attempt ignores artifacts on every path; refuse accidental writes."""

    def exists(self, path: PurePosixPath) -> bool:
        del path
        return False

    def put_file(self, path: PurePosixPath, data: bytes) -> None:
        raise RuntimeError(f"losing delivery must not write {path}")

    def commit(self) -> None:
        raise RuntimeError("losing delivery must not commit")

    def read_file(self, path: PurePosixPath) -> bytes | None:
        del path
        return None


@dataclass(frozen=True)
class PublishOutcome:
    """Result of one publish attempt. `reason is None` iff a sidecar was written.

    The reason exists because every call site wraps this in `except Exception:
    pass`. Returning WHY a generation was skipped is the only way a skip is
    visible from outside the container.
    """

    generation: tuple[int, int] | None
    reason: str | None = None


def _tee_stream(src: object, sinks: Sequence[object]) -> None:
    """Copy one child stream to every sink. Never slice or cap the payload."""
    if src is None:
        return
    read = getattr(src, "read", None)
    if read is None:
        return
    while True:
        chunk = read(65536)
        if not chunk:
            break
        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk,
                                                                     (bytes, bytearray)) else chunk
        for sink in sinks:
            if sink is None:
                continue
            try:
                sink.write(text)
                flush = getattr(sink, "flush", None)
                if flush is not None:
                    flush()
            except ValueError:
                continue


def execute_training_attempt(
    *,
    registry: Registry,
    attempt_id: str,
    run_root: Path,
    prepared: PreparedSource,
    commit: Callable[[], None],
    lock: LockLike,
    now: Callable[[], datetime],
    process_factory: Callable[..., object] = subprocess.Popen,
    sleep: Callable[[float], None] | None = None,
    wait: Callable[[threading.Event, float], bool] | None = None,
    log_sink: object | None = None,
    start_heartbeat: Callable[..., object] | None = None,
    killpg: Callable[[int, int], None] | None = None,
    getpgid: Callable[[int], int] | None = None,
    signal_signal: Callable[..., object] | None = None,
    manifest: Manifest | None = None,
    timeout: timedelta | None = None,
    already_claimed: bool = False,
) -> object:
    """Claim this delivery, then run the training child at most once.

    A same-input loser returns `redelivered` without writing STATUS, committing,
    or invoking the process factory. The original delivery is the only canonical
    writer. SIGINT, KeyboardInterrupt, and SIGTERM share one cleanup path.
    already_claimed skips the inner put_if_absent when the caller already won.
    """

    def train() -> object:
        return _run_training_attempt(
            attempt_id=attempt_id,
            run_root=Path(run_root),
            prepared=prepared,
            commit=commit,
            lock=lock,
            now=now,
            process_factory=process_factory,
            sleep=time.sleep if sleep is None else sleep,
            wait=wait,
            log_sink=log_sink,
            start_heartbeat=start_heartbeat,
            killpg=os.killpg if killpg is None else killpg,
            getpgid=os.getpgid if getpgid is None else getpgid,
            signal_signal=signal.signal if signal_signal is None else signal_signal,
            manifest=manifest,
            timeout=timeout,
        )

    if already_claimed:
        return train()
    return deliver_attempt(
        registry,
        _UnusedArtifacts(),
        attempt_id=attempt_id,
        train=train,
    )


def _checkpoint_generation(stat_result: os.stat_result) -> tuple[int, int]:
    return (stat_result.st_mtime_ns, stat_result.st_size)


def publish_stable_checkpoint(
    run_root: Path,
    *,
    now: Callable[[], datetime],
    commit: Callable[[], None],
    sleep: Callable[[float], None],
    last_published: tuple[int, int] | None = None,
) -> PublishOutcome:
    """Publish sidecar+commit only for a stable, weights-only-loadable generation.

    A changing mtime/size across the settle window is skipped. A torn file is
    load-rejected and must not produce a sidecar. Every skip carries a reason;
    "already published" is reported as a skip with no reason, since the sidecar
    for that generation does exist.
    """
    ckpt = Path(run_root) / "checkpoints" / CHECKPOINT_NAME
    if not ckpt.is_file():
        return PublishOutcome(last_published, f"no checkpoint at {ckpt}")
    first = _checkpoint_generation(ckpt.stat())
    if first == last_published:
        return PublishOutcome(last_published)
    sleep(CHECKPOINT_SETTLE_SECONDS)
    if not ckpt.is_file():
        return PublishOutcome(last_published, f"checkpoint vanished during settle: {ckpt}")
    second_stat = ckpt.stat()
    second = _checkpoint_generation(second_stat)
    if second != first:
        return PublishOutcome(
            last_published,
            f"checkpoint still changing across the {CHECKPOINT_SETTLE_SECONDS}s settle "
            f"window: {first} -> {second}")
    try:
        checkpoint.validate_local_checkpoint(ckpt)
    except ValidationError as err:
        return PublishOutcome(last_published, str(err))
    payload = {
        "sha256": core.sha256_file(ckpt),
        "size": second_stat.st_size,
        "mtime_ns": second_stat.st_mtime_ns,
        "validated_at": now().isoformat(),
    }
    atomic_write_json(ckpt.with_name(CHECKPOINT_SIDECAR_NAME), payload)
    commit()
    return PublishOutcome(second)


def _start_checkpoint_watcher(
    *,
    run_root: Path,
    now: Callable[[], datetime],
    commit: Callable[[], None],
    sleep: Callable[[float], None],
) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()

    def guarded_commit() -> None:
        if stop.is_set():
            return
        commit()

    def loop() -> None:
        last: tuple[int, int] | None = None
        while not stop.is_set():
            try:
                last = publish_stable_checkpoint(
                    run_root,
                    now=now,
                    commit=guarded_commit,
                    sleep=sleep,
                    last_published=last,
                ).generation
            except Exception:
                # Reasons are deliberately dropped here: this loop runs every
                # 50ms while training, so it must never write or log per skip.
                # finalize records the one reason that matters (the last one).
                pass
            # Real short poll: do not consume the injected heartbeat wait/clock.
            if stop.wait(0.05):
                break

    thread = threading.Thread(target=loop, name="cs2rl-checkpoint-watch", daemon=True)
    thread.start()
    return stop, thread


def _record_publish_reason(
    run_root: Path,
    reason: str | None,
    *,
    now: Callable[[], datetime],
) -> bool:
    """Keep the on-Volume note consistent with the last publish attempt.

    The file's presence means "this run produced no usable checkpoint, here is
    why"; a later success removes it. Returns whether the note changed on disk,
    so a caller that runs after the terminal commit knows to commit again.
    Never raises: losing the run's terminal write to a note would be worse.
    """
    path = Path(run_root) / "checkpoints" / CHECKPOINT_PUBLISH_REASON_NAME
    try:
        if reason is None:
            existed = path.is_file()
            path.unlink(missing_ok=True)
            return existed
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"reason": reason, "at": now().isoformat()})
        print(f"checkpoint sidecar not published: {reason}", file=sys.stderr, flush=True)
        return True
    except Exception:
        return False


def _publish_and_note(
    run_root: Path,
    *,
    now: Callable[[], datetime],
    commit: Callable[[], None],
    sleep: Callable[[float], None],
    commit_note: bool,
) -> None:
    """One publish attempt that can never raise and never skips silently.

    commit_note=False for the call inside finalize, whose terminal commit
    persists the note anyway; True for the retries that run after it.
    """
    try:
        reason = publish_stable_checkpoint(run_root, now=now, commit=commit, sleep=sleep).reason
    except Exception as err:           # noqa: BLE001 - a broken publish must not lose the run
        reason = f"publish raised {type(err).__name__}: {err}"
    if _record_publish_reason(run_root, reason, now=now) and commit_note:
        try:
            commit()
        except Exception:
            pass


def _close_log_sink(log_sink: object | None) -> None:
    if log_sink is None:
        return
    closer = getattr(log_sink, "close", None)
    if closer is not None:
        closer()


def _is_dead_run(run_root: Path, returncode: int | None) -> bool:
    if returncode == DEAD_RUN_EXIT_CODE:
        return True
    return (Path(run_root) / "checkpoints" / DEAD_CHECKPOINT_NAME).is_file()


def _map_child_exit(
    run_root: Path,
    returncode: int | None,
    manifest: Manifest | None,
) -> tuple[Status, str | None]:
    if returncode == 0:
        if manifest is None:
            return Status.FAILED, REASON_INVALID_EVIDENCE
        try:
            validate_completed_run(run_root, manifest)
        except ValidationError:
            return Status.FAILED, REASON_INVALID_EVIDENCE
        return Status.COMPLETED, None
    if _is_dead_run(run_root, returncode):
        return Status.FAILED, REASON_DEAD_RUN
    return Status.FAILED, REASON_NONZERO_EXIT


def _metrics_summary(run_root: Path) -> tuple[int, int | None]:
    try:
        steps = _iter_metrics_steps(Path(run_root) / "checkpoints" / "metrics.jsonl")
    except (ValidationError, OSError):
        return 0, None
    return len(steps), steps[-1]


def _optional_checkpoint_sha256(run_root: Path) -> str | None:
    """Sidecar digest only. Non-completed paths must not torch-load or rehash."""
    sidecar = Path(run_root) / "checkpoints" / CHECKPOINT_SIDECAR_NAME
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    digest = payload.get("sha256")
    if not isinstance(digest, str) or not digest:
        return None
    return digest


def _write_run_result(
    run_root: Path,
    *,
    status: Status,
    exit_code: int | None,
    started_at: str,
    finished_at: str,
    evidence: CompletionEvidence | None,
) -> None:
    if evidence is not None:
        digest = evidence.checkpoint_sha256
        last_step = evidence.last_step
        row_count, _ = _metrics_summary(run_root)
    else:
        digest = _optional_checkpoint_sha256(run_root)
        row_count, last_step = _metrics_summary(run_root)
    result = RunResult(
        schema_version=SCHEMA_VERSION,
        status=status,
        exit_code=-1 if exit_code is None else exit_code,
        started_at=started_at,
        finished_at=finished_at,
        artifact_root=str(run_root),
        checkpoint_sha256=digest,
        metrics_row_count=row_count,
        last_step=last_step,
    )
    atomic_write_json(Path(run_root) / RESULT_FILENAME, result.to_dict())


def _signal_process_group(
    child: object | None,
    *,
    killpg: Callable[[int, int], None],
    getpgid: Callable[[int], int],
    sleep: Callable[[float], None],
) -> None:
    if child is None:
        return
    pid = getattr(child, "pid", None)
    if pid is None:
        return
    try:
        pgid = getpgid(pid)
        killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    poll = getattr(child, "poll", None)
    if poll is not None and poll() is not None:
        return
    child_wait = getattr(child, "wait", None)
    if child_wait is not None:
        try:
            child_wait(timeout=TERM_GRACE_SECONDS)
        except (subprocess.TimeoutExpired, Exception, KeyboardInterrupt):
            # Deadline elapsed, or wait itself was interrupted — fall through
            # to poll/KILL. Cleanup must not resurrect KeyboardInterrupt.
            pass
    else:
        deadline = time.monotonic() + TERM_GRACE_SECONDS
        while True:
            if poll is not None and poll() is not None:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sleep(min(POLL_INTERVAL_SECONDS, remaining))
    if poll is not None and poll() is not None:
        return
    try:
        killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _run_training_attempt(
    *,
    attempt_id: str,
    run_root: Path,
    prepared: PreparedSource,
    commit: Callable[[], None],
    lock: LockLike,
    now: Callable[[], datetime],
    process_factory: Callable[..., object],
    sleep: Callable[[float], None],
    wait: Callable[[threading.Event, float], bool] | None,
    log_sink: object | None,
    start_heartbeat: Callable[..., object] | None,
    killpg: Callable[[int, int], None],
    getpgid: Callable[[int], int],
    signal_signal: Callable[..., object],
    manifest: Manifest | None,
    timeout: timedelta | None,
) -> object:
    run_root.mkdir(parents=True, exist_ok=True)
    state.transition_status(run_root, Status.TRAINING, now=now(), attempt_id=attempt_id, lock=lock)
    commit()
    wait_fn = wait if wait is not None else (lambda event, seconds: event.wait(seconds))
    heartbeat = prepared.heartbeat
    if heartbeat is None:
        starter = start_heartbeat if start_heartbeat is not None else start_heartbeat_worker
        heartbeat = starter(
            run_root=run_root,
            attempt_id=attempt_id,
            lock=lock,
            now=now,
            commit=commit,
            interval=HEARTBEAT_INTERVAL,
            wait=wait_fn,
        )
    ckpt_stop, ckpt_thread = _start_checkpoint_watcher(
        run_root=run_root,
        now=now,
        commit=commit,
        sleep=sleep,
    )
    child: object | None = None
    prev_int: object | None = None
    prev_term: object | None = None
    cleaned = False
    cleanup_lock = threading.Lock()
    heartbeat_stopped = False
    watcher_stopped = False
    started_at = now()
    final_result: TrainingAttemptResult | None = None
    owned_log: object | None = None
    tee_threads: list[threading.Thread] = []

    def stop_heartbeat_once() -> None:
        nonlocal heartbeat_stopped
        if heartbeat_stopped:
            return
        heartbeat_stopped = True
        _stop_heartbeat(heartbeat)

    def stop_watcher_once() -> None:
        nonlocal watcher_stopped
        if watcher_stopped:
            return
        watcher_stopped = True
        ckpt_stop.set()
        ckpt_thread.join(timeout=5.0)

    def finalize(
        status: Status,
        reason: str | None,
        exit_code: int | None,
        *,
        kill_child: bool,
    ) -> None:
        nonlocal cleaned, final_result
        with cleanup_lock:
            if cleaned:
                return
            cleaned = True
        if kill_child:
            _signal_process_group(child, killpg=killpg, getpgid=getpgid, sleep=sleep)
        for thread in tee_threads:
            thread.join(timeout=5.0)
        stop_heartbeat_once()
        stop_watcher_once()
        # Completed/failed still publish here so the sidecar shares the terminal
        # commit. Interrupted must not: the 120s prebuilt load would sit in the
        # SIGTERM handler before STATUS, and Modal's preemption/timeout kill
        # windows are ~30s / a handful of seconds. The post-finalize retry
        # publishes after STATUS is durable (live T4: sidecar validated_at was
        # ~7s after result.json finished_at).
        if status is not Status.INTERRUPTED:
            _publish_and_note(run_root, now=now, commit=commit, sleep=sleep, commit_note=False)
        _close_log_sink(owned_log)
        _close_log_sink(log_sink)
        evidence: CompletionEvidence | None = None
        if status is Status.COMPLETED and manifest is not None:
            try:
                evidence = validate_completed_run(run_root, manifest)
            except ValidationError:
                status = Status.FAILED
                reason = REASON_INVALID_EVIDENCE
        try:
            state.transition_status(run_root, status, now=now(), attempt_id=attempt_id, lock=lock)
            _write_run_result(
                run_root,
                status=status,
                exit_code=exit_code,
                started_at=started_at.isoformat(),
                finished_at=now().isoformat(),
                evidence=evidence,
            )
            commit()
        except Exception:
            # A failed Volume commit must not become a cross-container rewrite.
            pass
        final_result = TrainingAttemptResult(status=status, reason=reason, exit_code=exit_code)

    def on_signal(_signum: int, _frame: object) -> None:
        finalize(Status.INTERRUPTED, REASON_SIGNAL, None, kill_child=True)

    try:
        child = process_factory(
            prepared.train_command,
            cwd=os.fspath(prepared.source_dir),
            env=prepared.child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            shell=False,
        )
        try:
            prev_int = signal_signal(signal.SIGINT, on_signal)
            prev_term = signal_signal(signal.SIGTERM, on_signal)
        except ValueError:
            # signal.signal is main-thread-only; tests may run the child loop
            # on a worker thread and inject a fake installer instead.
            prev_int = None
            prev_term = None
        owned_log = (run_root / TRAIN_LOG_NAME).open("a", encoding="utf-8")
        shared_sinks: list[object] = [owned_log]
        if log_sink is not None:
            shared_sinks.append(log_sink)
        threads = [
            threading.Thread(target=_tee_stream,
                             args=(getattr(child, "stdout", None), [*shared_sinks, sys.stdout]),
                             daemon=True),
            threading.Thread(target=_tee_stream,
                             args=(getattr(child, "stderr", None), [*shared_sinks, sys.stderr]),
                             daemon=True),
        ]
        # Publish each thread only once it is started. finalize() joins tee_threads
        # unconditionally, and Thread.join() raises RuntimeError on a thread that was
        # never started; with the old extend-then-start, a SIGINT/SIGTERM landing in
        # that window raised out of on_signal before transition_status, stranding the
        # run in TRAINING with a dead child and no result.json (gh#217). Append per
        # thread rather than extending after the loop: start() can itself raise
        # ("can't start new thread"), and a failure on the second thread must not
        # leave the first started but unpublished -- joinable, but never joined,
        # because finalize only ever joins what is in tee_threads.
        #
        # The invariant is one-directional -- every thread in tee_threads has been
        # started, but not every started thread is yet in it. A signal in the residual
        # window between building `threads` and the first append finds the list empty,
        # so the tee threads are never joined and the attempt is not guaranteed to
        # write anything to train.log -- a thread started earlier in that window may
        # still get some output through before finalize closes the sinks with
        # `_close_log_sink`. That is the accepted trade: the threads are daemon=True so
        # they never hold the process open, finalize's join is their only consumer, and
        # the child is being killed anyway -- whereas publishing first costs a run with
        # no terminal status at all. Do not "fix" this by moving the append back above
        # start().
        #
        # It is only survivable because _tee_stream guards each sink's write/flush with
        # `except ValueError: continue`. Threads that start inside the residual window
        # run against sinks finalize has already closed; that handler is what keeps
        # this a no-op instead of an unraised-in-thread exception. Do not delete it as
        # dead defensive code.
        for thread in threads:
            thread.start()
            tee_threads.append(thread)
        timed_out = False
        child_wait = getattr(child, "wait", None)
        child_poll = getattr(child, "poll", None)
        while True:
            if child_poll is not None and child_poll() is not None:
                break
            if timeout is not None and now() - started_at >= timeout:
                timed_out = True
                break
            if child_wait is None:
                sleep(POLL_INTERVAL_SECONDS)
                continue
            try:
                child_wait(timeout=POLL_INTERVAL_SECONDS)
            except subprocess.TimeoutExpired:
                continue
        if not cleaned:
            if timed_out:
                finalize(Status.INTERRUPTED, REASON_TIMEOUT, None, kill_child=True)
            else:
                exit_code = getattr(child, "returncode", None)
                mapped, reason = _map_child_exit(run_root, exit_code, manifest)
                finalize(mapped, reason, exit_code, kill_child=False)
        # Signal-handler finalize cannot reliably torch.load/sleep. Retry on
        # the main thread now that the child wait loop has returned.
        _publish_and_note(run_root, now=now, commit=commit, sleep=sleep, commit_note=True)
        if final_result is not None:
            return final_result
        return TrainingAttemptResult(
            status=Status.FAILED,
            reason=REASON_NONZERO_EXIT,
            exit_code=getattr(child, "returncode", None),
        )
    except KeyboardInterrupt:
        finalize(Status.INTERRUPTED, REASON_SIGNAL, None, kill_child=True)
        _publish_and_note(run_root, now=now, commit=commit, sleep=sleep, commit_note=True)
        if final_result is not None:
            return final_result
        raise
    except Exception:
        finalize(Status.FAILED, REASON_ERROR, None, kill_child=True)
        _publish_and_note(run_root, now=now, commit=commit, sleep=sleep, commit_note=True)
        raise
    finally:
        if prev_int is not None:
            signal_signal(signal.SIGINT, prev_int)
        if prev_term is not None:
            signal_signal(signal.SIGTERM, prev_term)
        stop_watcher_once()
        stop_heartbeat_once()
        _close_log_sink(owned_log)
