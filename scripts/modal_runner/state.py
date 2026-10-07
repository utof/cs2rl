"""Run status, reservations, and heartbeat persistence."""
from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .core import (
    HEARTBEAT_INTERVAL,
    MANIFEST_FILENAME,
    RESERVATION_FILENAME,
    RUNS_ROOT,
    SCHEMA_VERSION,
    STATUS_FILENAME,
    LockLike,
    Registry,
    Status,
    ValidationError,
)
from .request import validate_run_id

TERMINAL_STATUSES = frozenset({
    Status.COMPLETED,
    Status.FAILED,
    Status.INTERRUPTED,
    Status.BUILD_FAILED,
})

# Linear lifecycle. Terminals have no outbound edges except the same-terminal
# idempotent write handled in transition_status.
_ALLOWED_TRANSITIONS: dict[Status, frozenset[Status]] = {
    Status.PREPARING:
    frozenset({Status.BUILDING, Status.BUILD_FAILED, Status.INTERRUPTED, Status.FAILED}),
    Status.BUILDING:
    frozenset({Status.TRAINING, Status.BUILD_FAILED, Status.INTERRUPTED, Status.FAILED}),
    Status.TRAINING:
    frozenset({Status.COMPLETED, Status.FAILED, Status.INTERRUPTED}),
}


@dataclass(frozen=True)
class RunStatus:
    """Persisted STATUS.json. attempt_id is the sole writer identity."""

    schema_version: int
    status: Status
    attempt_id: str
    updated_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "attempt_id": self.attempt_id,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunStatus:
        # Any, not object: the values are parsed JSON and these coercions are the check. A wrong
        # type raises TypeError or ValueError, which derive_run_view_from_bytes reports as corrupt.
        return cls(
            schema_version=int(payload["schema_version"]),
            status=Status(str(payload["status"])),
            attempt_id=str(payload["attempt_id"]),
            updated_at=str(payload["updated_at"]),
        )


STALE_AFTER = timedelta(minutes=5)


class ArtifactIndex(Protocol):
    """Client Volume metadata and reservation writes. Paths are PurePosixPath.

    exists is committed-object metadata (Volume.iterdir). put_file stages;
    commit flushes via client batch_upload(force=False), which already persists.
    Volume.commit() is the in-container mounted-volume API only. Never pass
    /artifacts/... here.
    """

    def exists(self, path: PurePosixPath) -> bool:
        """True if a committed Volume object exists at the client path."""
        ...

    def put_file(self, path: PurePosixPath, data: bytes) -> None:
        """Stage bytes at a client Volume path. Durable only after commit."""
        ...

    def commit(self) -> None:
        """Persist staged uploads via client batch_upload(force=False)."""
        ...

    def read_file(self, path: PurePosixPath) -> bytes | None:
        """Committed object bytes, or None if missing. Never /artifacts/..."""
        ...


FAILURE_UPLOAD = "upload_failed"
ALLOWED_FAILURE_CODES = frozenset({FAILURE_UPLOAD})

REDELIVERED = "redelivered"


@dataclass(frozen=True)
class DerivedStatus:
    """Client-side view. Never written back to STATUS.json."""

    status: Status
    stale: bool
    reason: str | None = None


@dataclass
class HeartbeatWorker:
    """Independent STATUS.json refresher. stop_and_join before every terminal write."""

    stop: threading.Event
    thread: threading.Thread

    def stop_and_join(self, timeout: float = 5.0) -> None:
        self.stop.set()
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise RuntimeError("heartbeat worker did not stop")


def atomic_write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    replace: Callable[[str, str], None] = os.replace,
) -> None:
    """Write JSON via a sibling temp file, then atomically replace.

    The temp is always unlinked on failure so a crashed replace cannot leave a
    `.STATUS.json.tmp-*` that a later reader might mistake for canonical state.
    `replace` is injectable so tests can prove that cleanup path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        tmp.write_text(json.dumps(dict(payload), sort_keys=True, indent=2) + "\n")
        replace(str(tmp), str(path))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def read_status(run_root: Path) -> RunStatus | None:
    path = run_root / STATUS_FILENAME
    if not path.is_file():
        return None
    return RunStatus.from_dict(json.loads(path.read_text()))


def transition_status(
    run_root: Path,
    next_status: Status,
    *,
    now: datetime,
    attempt_id: str,
    lock: LockLike,
) -> RunStatus | None:
    """Advance STATUS.json if this attempt owns the run.

    Returns None when a different attempt already owns canonical state — the
    redelivered container must not write, commit, or train. Same-terminal
    writes by the original attempt are idempotent. `lock` is required and
    must be the same lock the heartbeat uses so a beat cannot clobber a
    terminal write.
    """
    with lock:
        return _transition_status_unlocked(run_root, next_status, now=now, attempt_id=attempt_id)


def _transition_status_unlocked(
    run_root: Path,
    next_status: Status,
    *,
    now: datetime,
    attempt_id: str,
) -> RunStatus | None:
    run_root = Path(run_root)
    current = read_status(run_root)
    if current is not None and current.attempt_id != attempt_id:
        return None
    if current is None:
        if next_status is not Status.PREPARING:
            raise ValidationError(f"first status must be preparing, got {next_status.value}")
    elif current.status is next_status:
        if next_status in TERMINAL_STATUSES:
            return current
        raise ValidationError(f"nonterminal status {next_status.value} is already current")
    elif next_status not in _ALLOWED_TRANSITIONS.get(current.status, frozenset()):
        raise ValidationError(
            f"illegal status transition {current.status.value} -> {next_status.value}")
    status = RunStatus(
        schema_version=SCHEMA_VERSION,
        status=next_status,
        attempt_id=attempt_id,
        updated_at=now.isoformat(),
    )
    atomic_write_json(run_root / STATUS_FILENAME, status.to_dict())
    return status


def run_registry_key(run_id: str) -> str:
    """Dict key for the provisional run lease."""
    return f"run:{run_id}"


def attempt_registry_key(attempt_id: str) -> str:
    """Dict key for the remote same-input attempt claim."""
    return f"attempt:{attempt_id}"


def _reservation_path(run_id: str) -> PurePosixPath:
    return RUNS_ROOT / run_id / RESERVATION_FILENAME


def _manifest_path(run_id: str) -> PurePosixPath:
    return RUNS_ROOT / run_id / MANIFEST_FILENAME


def _volume_has_run(artifacts: ArtifactIndex, run_id: str) -> bool:
    """True if a durable reservation or manifest already occupies this run."""
    return artifacts.exists(_reservation_path(run_id)) or artifacts.exists(_manifest_path(run_id))


def record_run_failure(
    registry: Registry,
    run_id: str,
    attempt_id: str,
    failure_code: str,
) -> None:
    """Annotate the winning claim. Never delete it; never store exception text.

    failure_code is allowlisted so a caught upload error cannot leak a path
    or secret into the Dict. set_existing still refuses a mismatched attempt.
    """
    if failure_code not in ALLOWED_FAILURE_CODES:
        raise ValidationError(f"unknown failure code: {failure_code!r}")
    validate_run_id(run_id)
    validate_run_id(attempt_id)
    key = run_registry_key(run_id)
    current = registry.get(key)
    if current is None or current.get("attempt_id") != attempt_id:
        raise ValidationError(f"run id is not claimed by this attempt: {run_id}")
    updated = dict(current)
    updated["lifecycle"] = "failed"
    updated["failure_code"] = failure_code
    registry.set_existing(key, updated)


def finish_reservation(
    registry: Registry,
    artifacts: ArtifactIndex,
    run_id: str,
    attempt_id: str,
    *,
    upload: Callable[[], None],
) -> None:
    """Run the first post-reservation upload; keep the claim if it fails.

    Reservation already committed. A later source/checkpoint failure must
    not free the run ID. `artifacts` is unused here — uploads go through
    the caller — but the signature keeps the same adapter pair as reserve_run.
    """
    del artifacts
    try:
        upload()
    except Exception:
        record_run_failure(registry, run_id, attempt_id, FAILURE_UPLOAD)
        raise


def reserve_run(
    registry: Registry,
    artifacts: ArtifactIndex,
    run_id: str,
    attempt_id: str,
    *,
    now: datetime | None = None,
) -> None:
    """Atomically claim run_id, then durably commit reservation.json.

    The Dict put is only a seven-day mutex. The Volume reservation is the
    durable boundary and must land before any source/checkpoint upload or
    Function schedule. Reject when either a live Dict claim or a committed
    reservation/manifest exists — an expired Dict cannot reuse a Volume
    record. A lost put_if_absent is a hard reject — never overwrite
    another attempt's claim.
    """
    validate_run_id(run_id)
    validate_run_id(attempt_id)
    if _volume_has_run(artifacts, run_id):
        raise ValidationError(f"run id already reserved: {run_id}")
    stamp = (now if now is not None else datetime.now(UTC)).isoformat()
    claim: dict[str, object] = {"attempt_id": attempt_id, "created_at": stamp}
    if not registry.put_if_absent(run_registry_key(run_id), claim):
        raise ValidationError(f"run id already claimed: {run_id}")
    payload = {"attempt_id": attempt_id, "created_at": stamp}
    artifacts.put_file(
        _reservation_path(run_id),
        (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode(),
    )
    artifacts.commit()


def claim_attempt(registry: Registry, attempt_id: str) -> bool:
    """Atomically claim this same-input delivery. True iff first writer.

    Maps to put_if_absent on attempt:<attempt_id>. Modal restarts a
    preempted Function on the same input; the loser must not train.
    """
    validate_run_id(attempt_id)
    return registry.put_if_absent(attempt_registry_key(attempt_id), {"attempt_id": attempt_id})


def deliver_attempt(
    registry: Registry,
    artifacts: ArtifactIndex,
    *,
    attempt_id: str,
    train: Callable[[], object],
) -> object:
    """Run train() only for the winning attempt claim.

    A same-input loser returns 'redelivered' without calling train, writing
    canonical state, or committing the Volume. artifacts is unused on the
    loser path on purpose — the callback is the only writer.
    """
    del artifacts
    if not claim_attempt(registry, attempt_id):
        return REDELIVERED
    return train()


def write_heartbeat(
    run_root: Path,
    *,
    now: datetime,
    attempt_id: str,
    lock: LockLike,
) -> RunStatus | None:
    """Refresh updated_at if this attempt still owns a nonterminal run.

    The shared lock is the same one terminal cleanup holds. Taking it after
    a terminal write means we observe COMPLETED/FAILED/... and return it
    unchanged — a late beat cannot resurrect `training`.
    """
    with lock:
        current = read_status(Path(run_root))
        if current is None or current.attempt_id != attempt_id:
            return None
        if current.status in TERMINAL_STATUSES:
            return current
        status = RunStatus(
            schema_version=current.schema_version,
            status=current.status,
            attempt_id=current.attempt_id,
            updated_at=now.isoformat(),
        )
        atomic_write_json(Path(run_root) / STATUS_FILENAME, status.to_dict())
        return status


def load_volume_json(raw: bytes, message: str = "corrupt volume json") -> object:
    """Parse Volume bytes as JSON, normalising every failure to ValidationError.

    WHY the `message` parameter: a caller that re-raises ValidationError
    untouched (the reservation branch of `derive_run_view_from_bytes`, which
    must let "corrupt volume timestamp" escape) has no other way to attach its
    own operator-facing wording. The STATUS branch wraps and so does not need
    it, but passes it anyway so the two calls read alike.

    PITFALLS:
      * ValidationError subclasses ValueError, so a caller's
        `except (TypeError, ValueError, KeyError)` silently swallows and
        re-labels whatever this raises. That is fine when the wrap says the
        same thing, and a regression when it does not — see the asymmetry
        documented on `derive_run_view_from_bytes`.
      * `raw` is bytes, not str: UnicodeDecodeError is a real outcome here and
        is deliberately in the caught tuple.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as err:
        raise ValidationError(message) from err


def _parse_iso8601(value: str) -> datetime:
    """Parse an ISO-8601 stamp written by this module into a datetime.

    The only two callers are `derive_status` (STATUS `updated_at`) and the
    reservation branch of `derive_run_view_from_bytes` (`created_at`), so a
    truncated or garbled stamp fails one way rather than once per adapter.

    PITFALL: the "corrupt volume timestamp" message only reaches an operator
    where the caller re-raises ValidationError unchanged. In the STATUS branch
    of `derive_run_view_from_bytes` it is deliberately re-labelled
    "corrupt volume status json" — a bad `updated_at` is a corrupt STATUS file.
    """
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError) as err:
        raise ValidationError("corrupt volume timestamp") from err


def derive_status(status: RunStatus, *, now: datetime) -> DerivedStatus:
    """Map a persisted status to the client view. Does not write."""
    if status.status in TERMINAL_STATUSES:
        return DerivedStatus(status=status.status, stale=False)
    age = now - _parse_iso8601(status.updated_at)
    if age >= STALE_AFTER:
        return DerivedStatus(status=Status.INTERRUPTED, stale=True, reason="stale")
    return DerivedStatus(status=status.status, stale=False)


def derive_run_view_from_bytes(
    status_bytes: bytes | None,
    reservation_bytes: bytes | None,
    *,
    now: datetime,
) -> DerivedStatus:
    """Derive the client-visible run view from raw STATUS/reservation bytes.

    The one status algorithm. `None` means "no such object on the Volume".
    STATUS.json wins when present; otherwise a reservation stands in for a run
    that crashed after reserving but before its first heartbeat.

    WHY bytes and not paths: the status client, the launch validator and the
    sidecar backfill all reach the same judgement through this function, so
    staleness cannot drift between them. Each caller only has to produce bytes.

    Operator-facing messages, all ValidationError:
      * "corrupt volume status json"      — STATUS present but unparsable, not
        a mapping, missing a field, or carrying a bad `updated_at`.
      * "corrupt volume reservation json" — reservation unparsable, not a
        mapping, or missing `created_at`.
      * "corrupt volume timestamp"        — reservation `created_at` present
        but not ISO-8601.
      * "no STATUS.json or reservation.json" — both absent. Every caller
        checks for that case before calling in, so this generic wording is
        usually replaced: `collect_status` and backfill name the run id.
        `prior_checkpoint_or_raise` is the
        exception — it says a bare "parent run was not found" with no id.

    PITFALL — the two branches are deliberately asymmetric. The STATUS branch
    has no `except ValidationError: raise`, so a bad `updated_at` is reported
    as a corrupt STATUS file. The reservation branch has one, so a bad
    `created_at` keeps the distinct "corrupt volume timestamp". Neither is an
    oversight. Against the two pre-unification *Volume* adapter copies this
    matches on six of seven inputs; the reservation timestamp is a deliberate
    divergence (those copies re-wrapped it as "corrupt volume reservation
    json", because ValidationError is a ValueError and their generic handler
    swallowed it), made because spec §2.2 asks for corrupt timestamps to
    surface distinctly.

    Every spelling is pinned by the message table in
    tests/modal/test_modal_protocol.py. Adding or removing either clause silently
    changes what an operator sees; change the pinned table first.
    """
    if status_bytes is not None:
        try:
            payload = load_volume_json(status_bytes, "corrupt volume status json")
            if not isinstance(payload, dict):
                raise TypeError("STATUS.json is not a JSON object")
            return derive_status(RunStatus.from_dict(payload), now=now)
        except (TypeError, ValueError, KeyError) as err:
            raise ValidationError("corrupt volume status json") from err
    if reservation_bytes is None:
        raise ValidationError("no STATUS.json or reservation.json")
    try:
        payload = load_volume_json(reservation_bytes, "corrupt volume reservation json")
        if not isinstance(payload, dict):
            raise TypeError("reservation.json is not a JSON object")
        created = _parse_iso8601(str(payload["created_at"]))
    except ValidationError:
        raise
    except (TypeError, ValueError, KeyError) as err:
        raise ValidationError("corrupt volume reservation json") from err
    if now - created >= STALE_AFTER:
        return DerivedStatus(status=Status.INTERRUPTED, stale=True, reason="no-heartbeat")
    return DerivedStatus(status=Status.PREPARING, stale=False, reason="no-heartbeat")


def start_heartbeat_worker(
    *,
    run_root: Path,
    attempt_id: str,
    lock: LockLike,
    now: Callable[[], datetime],
    commit: Callable[[], None] | None = None,
    interval: timedelta = HEARTBEAT_INTERVAL,
    wait: Callable[[threading.Event, float], bool] | None = None,
) -> HeartbeatWorker:
    """Write + commit immediately, then every `interval`, until stop_and_join.

    `wait(event, seconds)` is injectable so tests can advance a fake clock
    instead of sleeping a real minute. The default is Event.wait.
    """
    stop = threading.Event()
    wait_fn = wait if wait is not None else (lambda event, seconds: event.wait(seconds))

    def loop() -> None:
        while not stop.is_set():
            try:
                write_heartbeat(run_root, now=now(), attempt_id=attempt_id, lock=lock)
                if commit is not None:
                    commit()
            except Exception:
                # One Volume.commit() blip must not kill the daemon.
                pass
            if wait_fn(stop, interval.total_seconds()):
                break

    thread = threading.Thread(target=loop, name="cs2rl-preflight-heartbeat", daemon=True)
    thread.start()
    return HeartbeatWorker(stop=stop, thread=thread)


def stop_heartbeat(heartbeat: object | None) -> None:
    if heartbeat is None:
        return
    stop = getattr(heartbeat, "stop_and_join", None)
    if stop is not None:
        stop()
