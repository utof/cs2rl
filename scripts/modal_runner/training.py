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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath

from . import checkpoint, core, state
from .checkpoint import iter_metrics_steps, validate_completed_run
from .core import (
    CHECKPOINT_NAME,
    CHECKPOINT_PUBLISH_REASON_NAME,
    CHECKPOINT_SIDECAR_NAME,
    DEAD_CHECKPOINT_NAME,
    HEARTBEAT_INTERVAL,
    RESULT_FILENAME,
    SCHEMA_VERSION,
    TRAIN_LOG_NAME,
    AttemptContext,
    CompletionEvidence,
    Manifest,
    PreparedSource,
    Registry,
    Status,
    ValidationError,
)
from .state import atomic_write_json, deliver_attempt, start_heartbeat_worker, stop_heartbeat


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


@dataclass(frozen=True)
class ProcessControl:
    """The training child's OS: start it, find and signal its group, install signal handlers.

    NO FIELD DEFAULTS, DELIBERATELY (gh#163). This is the second of the kill
    seam's three safety layers: the process-group guard in
    `_signal_process_group`, no real-OS defaults here, and the tests/conftest.py
    tripwire. On 2026-09-23 a throwaway script built a draft of this class whose
    fields defaulted to the real OS functions, faked only `spawn`, and called
    `killpg(getpgid(1), SIGTERM)`: that is `kill(-1, SIGTERM)`, and it ended the
    user's desktop session. Without defaults a half-faked ProcessControl is a
    TypeError when it is built, never a live `killpg`. The real functions come
    only from `system()`.

    Tests build all four fields as keywords, never through a `*`/`**` splat,
    which a static check cannot read. `test_kill_seam_static_safety`
    (tests/test_modal_training.py) enforces that and the rest of the kill-seam
    rules by AST; under pytest the tests/conftest.py tripwire makes `system()`
    return a control whose every field raises, so a test that forgets its own
    control fails loudly instead of spawning or signalling.

    PITFALLS. Never give a field a default. Read `system` only as a call
    inside `execute_training_attempt`'s body: a module-level alias or a default
    argument captures the real factory at import and bypasses the tripwire's
    patch of the class attribute. `killpg` and `getpgid` are called only by
    `_signal_process_group`, which holds the process-group guard; a direct call
    anywhere else in this module bypasses it.
    """

    spawn: Callable[..., object]
    getpgid: Callable[[int], int]
    killpg: Callable[[int, int], None]
    install_signal: Callable[..., object]

    @classmethod
    def system(cls) -> ProcessControl:
        """The real OS functions, read when this is called, not at import.

        SAFETY: never call a field of what this returns outside the production
        attempt. Outside pytest, `killpg` here is the real `os.killpg`.
        """
        return cls(spawn=subprocess.Popen,
                   getpgid=os.getpgid,
                   killpg=os.killpg,
                   install_signal=signal.signal)


def execute_training_attempt(
    *,
    attempt: AttemptContext,
    prepared: PreparedSource,
    registry: Registry,
    manifest: Manifest | None = None,
    timeout: timedelta | None = None,
    process: ProcessControl | None = None,
    log_sink: object | None = None,
    already_claimed: bool = False,
) -> object:
    """Claim this delivery, then run the training child at most once.

    A same-input loser returns `redelivered` without writing STATUS, committing,
    or starting the child. The original delivery is the only canonical writer.
    SIGINT, KeyboardInterrupt, and SIGTERM share one cleanup path.
    already_claimed skips the inner put_if_absent when the caller already won.

    `attempt` is the delivery prepare ran under (the same lock, run_root, Volume
    and clock; gh#163 W5). `process` is the child's OS: spawn, the
    process-group signals, the handler install. Production passes none, and
    None resolves to `ProcessControl.system()`, the real functions, inside
    `train()`: after the claim, at call time. Tests pass a control whose
    `spawn`, `getpgid` and `killpg` are fakes (`_training_kwargs` in
    tests/test_modal_training.py builds one); its `install_signal` may be the
    real `signal.signal`, which only installs this attempt's own handlers.

    PITFALL: that `ProcessControl.system()` call is the module's one read of
    `system`, and it must stay a call in this function's body. A module-level
    alias or a default value would capture the real factory at import, where
    the tests/conftest.py tripwire, which patches the class attribute, never
    sees it: a test that forgot `process` would then spawn and signal for real.
    `test_kill_seam_static_safety` clause (vi) pins the spelling and
    `test_process_control_tripwire_guards_the_resolution_path` drives the path.
    """

    def train() -> object:
        control = ProcessControl.system() if process is None else process
        return _run_training_attempt(
            attempt=attempt,
            prepared=prepared,
            process=control,
            manifest=manifest,
            timeout=timeout,
            log_sink=log_sink,
        )

    if already_claimed:
        return train()
    return deliver_attempt(
        registry,
        _UnusedArtifacts(),
        attempt_id=attempt.attempt_id,
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
        steps = iter_metrics_steps(Path(run_root) / "checkpoints" / "metrics.jsonl")
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
    """SIGTERM the child's process group; SIGKILL it if the child outlives the grace period.

    The group, not the pid: the child is spawned with `start_new_session=True`, so
    everything it forks receives the same signal, unless it has left the group
    (`setsid`/`setpgid`). The SIGKILL escalation keys on the child alone: a
    grandchild that ignores SIGTERM survives if the child exits within the grace
    period.

    THE GUARD. The two `killpg` calls below are the runner's only `killpg`
    calls, the only place it signals a process group, so the guard lives here.
    It refuses, with one stderr line naming the condition, to signal a group
    that a live or unreaped child cannot have. The conditions are checked in
    this order, and the first that holds is the one reported:
      * `pgid!=pid` -- a `start_new_session` child leads its own group until it
        is reaped, so its pgid IS its pid. Anything else is a fake, or a pid
        reused after reaping by a process that does not lead its own group.
      * `pgid<=1` -- `killpg(1, sig)` is `kill(-1, sig)`: every process the
        user owns. `killpg(0, sig)` is the caller's own group.
      * `own-group` -- `pgid == os.getpgrp()`, the runner's own group.
    None of these can hold for a live or unreaped child. So the one change
    production can see is that a reaped pid reused by a non-leader is now
    refused, where before the guard its group was signalled.

    WHY it exists (2026-09-23). An agent's throwaway script built a draft
    `ProcessControl` with only `spawn` faked and called its
    `killpg(getpgid(1), SIGTERM)` directly, on the real OS functions. That is
    `kill(-1, SIGTERM)`, and it ended the user's desktop session. The call never
    went through this function, so this guard would not have stopped it; a
    `ProcessControl` with no real-OS defaults (gh#163 W5) answers the direct
    call. This guard closes the same shape on the one real path: the real
    `killpg`/`getpgid` that `ProcessControl.system()` supplies when
    `execute_training_attempt` is given no `process`.

    WHAT the guard covers. When the real functions reach it, it still refuses
    pid 1's group, the runner's own group and any pid that does not lead its
    own group. It does NOT cover a fake pid (FakeChild's default 4242, or any
    other) that happens to be a live process leading its own group: that
    passes all three conditions and receives the real signal. So no test may
    hand it the real functions. Since W5 none does: the training test builder
    always passes a ProcessControl whose `getpgid` and `killpg` are fakes, and
    the `tests/conftest.py` tripwire makes the real one fail loudly under
    pytest.

    RESIDUAL.
      * Accepted: a reaped pid reused by a process that leads its own group
        (any `start_new_session` child, a login shell, a daemon) has
        `pgid == pid` and is signalled. In production that needs the wait loop
        (`child_poll()`, or `child_wait()`, which is `Popen.wait` and reaps
        too) to reap the child and its pid to be reused before
        `finalize(kill_child=True)` runs, which is negligible.
      * Closed by W5 (not by this guard) except in the tripwire's declared
        windows: a test that forgot the fakes while its fake pid was a live
        process leading its own group would have sent a real signal (above).
        The builder's fakes and the conftest tripwire close it. The three
        windows tests/conftest.py declares unpoisoned stay open: a test body
        that is the first import of the package, code that runs at
        collection, and module-, class- or session-scoped fixtures.

    PITFALL: do NOT close the accepted residual by polling before `getpgid`. Today
    a zombie's `getpgid` succeeds, so the SIGTERM still reaches grandchildren
    left in its group. Polling first reaps the zombie, `getpgid` then raises
    ProcessLookupError, and the grandchildren escape. That is a behaviour change.

    The lookup and the SIGTERM are separate `try`s so that a refusal can never
    be mistaken for, or swallowed as, a lookup failure. The refusal line never
    raises: this runs on `finalize`'s path, and a closed or broken stderr must
    not cost the run its terminal STATUS and result.json.
    """
    if child is None:
        return
    pid = getattr(child, "pid", None)
    if pid is None:
        return
    try:
        pgid = getpgid(pid)
    except ProcessLookupError:
        return
    if pgid != pid:
        refused = "pgid!=pid"
    elif pgid <= 1:
        refused = "pgid<=1"
    elif pgid == os.getpgrp():
        refused = "own-group"
    else:
        refused = None
    if refused is not None:
        try:
            print(f"cs2rl: refusing to signal process group {pgid} of pid {pid}: {refused}",
                  file=sys.stderr,
                  flush=True)
        except Exception:
            # Never raises (like `_record_publish_reason`): losing finalize's terminal write to a
            # diagnostic line would be worse than losing the line.
            pass
        return
    try:
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


@dataclass(eq=False)
class _LiveAttempt:
    """What the cleanup paths of one running attempt share: the child, handlers, tees and log.

    gh#210 (gh#163 W5). These were locals of `_run_training_attempt`, shared
    through `nonlocal` by three closures (`finalize` and the two once-guards;
    the fourth, `on_signal`, only called `finalize`); as fields and methods
    each part can be read, and measured, on its own. State, not a
    collaborator: mutable, compared by identity, and never handed out of the
    attempt.

    Three paths reach `finalize`: the normal exit (`finish`), the SIGINT/SIGTERM
    handler (`on_signal`, installed by `install_handlers`), and the `except` arms
    of `_run_training_attempt`. The first to take `cleanup_lock` does the work;
    the others return at `if self.cleaned`. `release` is the `finally` arm and
    runs on every path.

    CALL ORDER (`_run_training_attempt`): `spawn`, `install_handlers`,
    `start_tees`, `wait_for_exit`, `finish`, then `publish(commit_note=True)`;
    `release` last, always. Each step assumes the ones before it ran:
      * `spawn` comes BEFORE `install_handlers`. Reversed, a signal during the
        spawn would finalize with `child` still None (nothing to kill), and the
        child spawned afterwards would run unsupervised while STATUS says
        interrupted.
      * `start_tees` and `wait_for_exit` read `child`: with it None,
        `start_tees` tees nothing, and `wait_for_exit` without a timeout polls
        forever.
      * `finish` is the wait loop's end: it maps the exit and calls
        `finalize`, unless a handler already has. `finalize` is the once-only
        terminal write that every path shares. Similar names, different jobs.

    PITFALLS.
      * `finalize` can run INSIDE a signal handler, on the main thread, between
        any two statements after `install_handlers`: inside `start_tees`
        (gh#217), inside the wait loop. Every field it reads must be usable from
        construction on (`child` None, `tee_threads` empty), and a method may
        publish into such a field only a value that is ready (`start_tees`
        appends a thread only once it is started).
      * `killpg`/`getpgid` are reached only through `_signal_process_group`, the
        process-group guard. Never call `self.process.killpg` or `getpgid` here,
        nor hand them to anything but that guard (`test_kill_seam_static_safety`
        clause (vii)).
      * Every clock read is `self.attempt.clock.<now|sleep>` and every commit
        `self.attempt.volume.commit`: in a test's fake-clock run a stray
        `_utc_now()` stamps the real time and a stray `time.sleep` really
        sleeps. (`test_failed_cleanup_commit_does_not_let_redelivery_write` and
        the timeout half of `test_dead_run_and_timeout_have_distinct_reasons`
        go red when these reads become `core._utc_now`.)
      * `threading` is read through this module at call time (`threading.Thread`
        in `start_tees`; `_run_training_attempt` builds `cleanup_lock`): the
        binding campaign's attempt rows replace `training.threading`.
    """

    attempt: AttemptContext
    process: ProcessControl
    manifest: Manifest | None
    log_sink: object | None
    heartbeat: object
    ckpt_stop: threading.Event
    ckpt_thread: threading.Thread
    started_at: datetime
    cleanup_lock: threading.Lock
    tee_threads: list[threading.Thread] = field(default_factory=list)
    child: object | None = None
    prev_int: object | None = None
    prev_term: object | None = None
    owned_log: object | None = None
    cleaned: bool = False
    heartbeat_stopped: bool = False
    watcher_stopped: bool = False
    final_result: TrainingAttemptResult | None = None

    def stop_heartbeat_once(self) -> None:
        if self.heartbeat_stopped:
            return
        self.heartbeat_stopped = True
        stop_heartbeat(self.heartbeat)

    def stop_watcher_once(self) -> None:
        if self.watcher_stopped:
            return
        self.watcher_stopped = True
        self.ckpt_stop.set()
        self.ckpt_thread.join(timeout=5.0)

    def finalize(
        self,
        status: Status,
        reason: str | None,
        exit_code: int | None,
        *,
        kill_child: bool,
    ) -> None:
        """End the attempt once: stop the child, then persist the terminal STATUS and result.json.

        Whichever path takes `cleanup_lock` first does this; the later ones
        return. In order:
          1. with `kill_child`, signal the child's group through the guard;
          2. join the tee threads (5 s each);
          3. stop the heartbeat, then the checkpoint watcher (each once);
          4. unless `status` is INTERRUPTED, publish a stable checkpoint's
             sidecar (the comment below says why not on INTERRUPTED);
          5. close train.log and the caller's `log_sink`;
          6. for COMPLETED with a manifest, re-validate the completion
             evidence: a failure downgrades the run to FAILED with
             `REASON_INVALID_EVIDENCE`;
          7. write the terminal STATUS and result.json, and commit the Volume;
          8. record `final_result`.
        Step 7 shares one `try`: a failed write or Volume commit is swallowed,
        because it must not become a cross-container rewrite. Step 8 runs
        either way, so the attempt still returns the terminal status even when
        the Volume does not hold it (unpinned by any test: gh#238).
        """
        with self.cleanup_lock:
            if self.cleaned:
                return
            self.cleaned = True
        if kill_child:
            _signal_process_group(self.child,
                                  killpg=self.process.killpg,
                                  getpgid=self.process.getpgid,
                                  sleep=self.attempt.clock.sleep)
        for thread in self.tee_threads:
            thread.join(timeout=5.0)
        self.stop_heartbeat_once()
        self.stop_watcher_once()
        # Completed/failed still publish here so the sidecar shares the terminal
        # commit. Interrupted must not: the 120s prebuilt load would sit in the
        # SIGTERM handler before STATUS, and Modal's preemption/timeout kill
        # windows are ~30s / a handful of seconds. The post-finalize retry
        # publishes after STATUS is durable (live T4: sidecar validated_at was
        # ~7s after result.json finished_at).
        if status is not Status.INTERRUPTED:
            self.publish(commit_note=False)
        _close_log_sink(self.owned_log)
        _close_log_sink(self.log_sink)
        evidence: CompletionEvidence | None = None
        if status is Status.COMPLETED and self.manifest is not None:
            try:
                evidence = validate_completed_run(self.attempt.run_root, self.manifest)
            except ValidationError:
                status = Status.FAILED
                reason = REASON_INVALID_EVIDENCE
        try:
            state.transition_status(self.attempt.run_root,
                                    status,
                                    now=self.attempt.clock.now(),
                                    attempt_id=self.attempt.attempt_id,
                                    lock=self.attempt.lock)
            _write_run_result(
                self.attempt.run_root,
                status=status,
                exit_code=exit_code,
                started_at=self.started_at.isoformat(),
                finished_at=self.attempt.clock.now().isoformat(),
                evidence=evidence,
            )
            self.attempt.volume.commit()
        except Exception:
            # A failed Volume commit must not become a cross-container rewrite.
            pass
        self.final_result = TrainingAttemptResult(status=status, reason=reason, exit_code=exit_code)

    def publish(self, *, commit_note: bool) -> None:
        """One `_publish_and_note` over this attempt's run_root, clock and Volume; never raises.

        `commit_note=False` only inside `finalize`, whose terminal commit
        carries the note; True for the retries after it (see
        `_publish_and_note`).
        """
        _publish_and_note(self.attempt.run_root,
                          now=self.attempt.clock.now,
                          commit=self.attempt.volume.commit,
                          sleep=self.attempt.clock.sleep,
                          commit_note=commit_note)

    def on_signal(self, _signum: int, _frame: object) -> None:
        self.finalize(Status.INTERRUPTED, REASON_SIGNAL, None, kill_child=True)

    def spawn(self, prepared: PreparedSource) -> None:
        """Start the child in its own session (its pgid is its pid), without a shell."""
        self.child = self.process.spawn(
            prepared.train_command,
            cwd=os.fspath(prepared.source_dir),
            env=prepared.child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            shell=False,
        )

    def install_handlers(self) -> None:
        """Route SIGINT and SIGTERM to `on_signal`, keeping the previous handlers for `release`.

        `self.on_signal` builds a new bound-method object on every read, so it
        is read once and that one object installed for both signals: the tests
        check that the two handlers are the same object.
        """
        handler = self.on_signal
        try:
            self.prev_int = self.process.install_signal(signal.SIGINT, handler)
            self.prev_term = self.process.install_signal(signal.SIGTERM, handler)
        except ValueError:
            # signal.signal is main-thread-only; tests may run the child loop
            # on a worker thread and inject a fake installer instead.
            self.prev_int = None
            self.prev_term = None

    def start_tees(self) -> None:
        """Open train.log and start one thread per child stream copying it there and to the sinks."""
        self.owned_log = (self.attempt.run_root / TRAIN_LOG_NAME).open("a", encoding="utf-8")
        shared_sinks: list[object] = [self.owned_log]
        if self.log_sink is not None:
            shared_sinks.append(self.log_sink)
        child = self.child
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
            self.tee_threads.append(thread)

    def wait_for_exit(self, timeout: timedelta | None) -> bool:
        """Poll the child until it exits or `timeout` has passed; True means it timed out."""
        child_wait = getattr(self.child, "wait", None)
        child_poll = getattr(self.child, "poll", None)
        while True:
            if child_poll is not None and child_poll() is not None:
                return False
            if timeout is not None and self.attempt.clock.now() - self.started_at >= timeout:
                return True
            if child_wait is None:
                self.attempt.clock.sleep(POLL_INTERVAL_SECONDS)
                continue
            try:
                child_wait(timeout=POLL_INTERVAL_SECONDS)
            except subprocess.TimeoutExpired:
                continue

    def finish(self, timed_out: bool) -> None:
        """Finalize from the wait loop, unless a signal handler already has.

        A timeout kills the child and is INTERRUPTED; an exit is mapped from its
        code and the completion evidence, and leaves the (exited) child alone.
        """
        if self.cleaned:
            return
        if timed_out:
            self.finalize(Status.INTERRUPTED, REASON_TIMEOUT, None, kill_child=True)
            return
        exit_code = getattr(self.child, "returncode", None)
        mapped, reason = _map_child_exit(self.attempt.run_root, exit_code, self.manifest)
        self.finalize(mapped, reason, exit_code, kill_child=False)

    def release(self) -> None:
        """The `finally` arm: restore the handlers, stop the watcher and heartbeat, close the log."""
        if self.prev_int is not None:
            self.process.install_signal(signal.SIGINT, self.prev_int)
        if self.prev_term is not None:
            self.process.install_signal(signal.SIGTERM, self.prev_term)
        self.stop_watcher_once()
        self.stop_heartbeat_once()
        _close_log_sink(self.owned_log)


def _run_training_attempt(
    *,
    attempt: AttemptContext,
    prepared: PreparedSource,
    process: ProcessControl,
    manifest: Manifest | None,
    timeout: timedelta | None,
    log_sink: object | None,
) -> object:
    """The attempt, in order: TRAINING, heartbeat, watcher, then the supervised child.

    gh#210. Enter TRAINING and commit; take the prepared heartbeat or start
    the fallback; start the checkpoint watcher; build the live state; spawn,
    install the handlers, start the tees, wait (with the timeout), map the
    exit and finalize; retry the publish; return the result (`_LiveAttempt`'s
    CALL ORDER says why in that order). The arms:
      * KeyboardInterrupt: finalize INTERRUPTED (the child is killed), retry
        the publish, and RETURN the INTERRUPTED result; it re-raises only if
        no result was recorded.
      * Any other exception: finalize FAILED (the child is killed), retry the
        publish, and re-raise.
      * `finally`: `release` restores the handlers, stops the watcher and the
        heartbeat, and closes train.log, on every path.

    PITFALLS.
      * Keep the TRAINING `state.transition_status(...)` and the
        `_start_checkpoint_watcher(...)` call in THIS function, each reference
        on one line: the binding campaign (tests/test_modal_patch_bindings.py,
        `_capture_consumer`) rewrites those loads in this function's source.
      * That rewrite recompiles this function without the module's `from
        __future__ import annotations`, so every annotation in its signature
        must name something this module binds at run time.
    """
    attempt.run_root.mkdir(parents=True, exist_ok=True)
    state.transition_status(attempt.run_root,
                            Status.TRAINING,
                            now=attempt.clock.now(),
                            attempt_id=attempt.attempt_id,
                            lock=attempt.lock)
    attempt.volume.commit()
    heartbeat = prepared.heartbeat
    if heartbeat is None:
        heartbeat = start_heartbeat_worker(
            run_root=attempt.run_root,
            attempt_id=attempt.attempt_id,
            lock=attempt.lock,
            now=attempt.clock.now,
            commit=attempt.volume.commit,
            interval=HEARTBEAT_INTERVAL,
            wait=attempt.clock.wait,
        )
    ckpt_stop, ckpt_thread = _start_checkpoint_watcher(
        run_root=attempt.run_root,
        now=attempt.clock.now,
        commit=attempt.volume.commit,
        sleep=attempt.clock.sleep,
    )
    live = _LiveAttempt(
        attempt=attempt,
        process=process,
        manifest=manifest,
        log_sink=log_sink,
        heartbeat=heartbeat,
        ckpt_stop=ckpt_stop,
        ckpt_thread=ckpt_thread,
        started_at=attempt.clock.now(),
        cleanup_lock=threading.Lock(),
    )
    try:
        live.spawn(prepared)
        live.install_handlers()
        live.start_tees()
        timed_out = live.wait_for_exit(timeout)
        live.finish(timed_out)
        # Signal-handler finalize cannot reliably torch.load/sleep. Retry on
        # the main thread now that the child wait loop has returned.
        live.publish(commit_note=True)
        if live.final_result is not None:
            return live.final_result
        return TrainingAttemptResult(
            status=Status.FAILED,
            reason=REASON_NONZERO_EXIT,
            exit_code=getattr(live.child, "returncode", None),
        )
    except KeyboardInterrupt:
        live.finalize(Status.INTERRUPTED, REASON_SIGNAL, None, kill_child=True)
        live.publish(commit_note=True)
        if live.final_result is not None:
            return live.final_result
        raise
    except Exception:
        live.finalize(Status.FAILED, REASON_ERROR, None, kill_child=True)
        live.publish(commit_note=True)
        raise
    finally:
        live.release()
