"""The runner's shared vocabulary: what two or more package modules read.

That is the Volume and run-directory layout below, which stays here whole even
where one module reads a given name, so the layout reads as one contract; the
Status enum, ValidationError, the path and hash helpers, and the protocols and
records that cross module boundaries. A name only one module reads lives in
that module (see WHERE A NEW NAME GOES in __init__.py).
"""
from __future__ import annotations

import enum
import hashlib
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Protocol

VOLUME_NAME = "cs2rl-training-artifacts"
REGISTRY_NAME = "cs2rl-training-run-registry"
VOLUME_MOUNT = Path("/artifacts")
SOURCES_ROOT = PurePosixPath("sources")
INPUTS_ROOT = PurePosixPath("inputs")
RUNS_ROOT = PurePosixPath("runs")
# File names inside a source bundle and a run directory.
PROVENANCE_NAME = ".cs2rl-provenance.json"
STATUS_FILENAME = "STATUS.json"
RESERVATION_FILENAME = "reservation.json"
MANIFEST_FILENAME = "manifest.json"
TRAIN_LOG_NAME = "train.log"
RESULT_FILENAME = "result.json"
CHECKPOINT_NAME = "dust2_policy.pt"
CHECKPOINT_SIDECAR_NAME = "dust2_policy.pt.meta.json"
CHECKPOINT_PUBLISH_REASON_NAME = "dust2_policy.pt.publish_reason.json"
DEAD_CHECKPOINT_NAME = "dust2_policy_dead.pt"

# The image venv that holds torch/numpy/PufferLib and runs cs2rl.train.__main__. It is NOT
# the interpreter this module runs under on the container: a Modal function runs
# on the image's standalone python (/usr/local/bin/python from add_python=), which
# has only uv + the modal client. checkpoint.py's `_assert_weights_only_loadable`
# shells out to it when in-process torch is unavailable, and commands.py builds
# the install, train and CUDA-probe commands on it; both read it as
# `core.PREBUILT_PYTHON`, one of the package's qualified seams (see __init__.py).
PREBUILT_PYTHON = "/opt/cs2rl/.venv/bin/python"


class ValidationError(ValueError):
    """Locally-detectable invalid launch or observe request.

    Raised before any Dict/Volume write or GPU invocation. Message is operator
    facing; do not put secret values in it.
    """


class Status(enum.StrEnum):
    """Run STATUS.json states. Terminal vs nonterminal is a partition."""

    PREPARING = "preparing"
    BUILDING = "building"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    BUILD_FAILED = "build_failed"


def mounted_path(relative: PurePosixPath) -> Path:
    """Translate a client Volume path to the in-container mount path.

    Client APIs take `sources/...`; the container opens `/artifacts/sources/...`.
    Absolute paths and any `..` part are rejected so a bad relative cannot
    escape the mount even if a later caller stringifies carelessly.
    """
    if not isinstance(relative, PurePosixPath):
        raise ValidationError(f"mounted_path expects PurePosixPath, got {type(relative)!r}")
    if relative.is_absolute():
        raise ValidationError(f"refusing absolute client path: {relative}")
    if any(part in {"..", ""} for part in relative.parts):
        raise ValidationError(f"refusing client path with '..' or empty part: {relative}")
    return VOLUME_MOUNT.joinpath(*relative.parts)


def sha256_file(path: Path) -> str:
    """Stream a file's SHA-256. Used for archives and checkpoints."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileProvenance:
    """Content-addressed local file destined for inputs/sha256/<digest>.pt."""

    sha256: str
    size: int
    client_path: PurePosixPath
    mount_path: Path


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Manifest:
    """Minimum manifest.json contract from design §5.

    effective_map is authoritative. Do not store live config's `env` field: it
    is only a label, which `cs2rl.train.config.build_train_config` derives as
    `cs2-<map>` from the `--map` that `cs2rl.train.__main__` resolves (`cs2-dust2`
    when the `map` attribute is missing or empty). The map the runner validated is
    effective_map.
    """

    schema_version: int
    run_id: str
    attempt_id: str
    commit: str
    tree: str
    source_archive_sha256: str
    modal_version: str
    image_digest: str
    effective_map: str
    gpu: str
    cpu_request: int
    cpu_soft_limit: int
    memory_request_mib: int
    memory_hard_limit_mib: int
    vec_workers: int
    timeout_minutes: int
    training_argv: list[str]
    requested_timesteps: int
    effective_timesteps: int
    batch_size: int
    seed: int
    created_at: str
    resume_sha256: str | None
    resume_size: int | None
    resume_source_path: str | None
    runner_commit: str
    config_hash: str
    thread_caps: list[str]
    resumed_from_run_id: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


HEARTBEAT_INTERVAL = timedelta(seconds=60)


class Registry(Protocol):
    """Atomic run/attempt claims. Maps to a Modal Dict.

    put_if_absent is Dict.put(key, value, skip_if_exists=True). Do not
    synthesize this from contains() plus put() — two launchers can both
    observe a miss and both write.
    """

    def put_if_absent(self, key: str, value: Mapping[str, object]) -> bool:
        """Insert key only if absent. True iff this caller created it."""
        ...

    def get(self, key: str) -> Mapping[str, object] | None:
        """Return a copy of the stored claim, or None on a Dict miss."""
        ...

    def set_existing(self, key: str, value: Mapping[str, object]) -> None:
        """Overwrite a claim only when stored attempt_id still matches.

        Maps to a normal Dict.put after get. Not compare-and-swap; it
        only refuses to clobber a different attempt's record.
        """
        ...


class LockLike(Protocol):

    def __enter__(self) -> object:
        ...

    def __exit__(self, *exc: object) -> None:
        ...


def sha256_bytes(data: bytes) -> str:
    """SHA-256 of an in-memory payload (normalized config JSON)."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class CompletionEvidence:
    last_step: int
    checkpoint_sha256: str
    config_hash: str


@dataclass
class PreparedSource:
    """Extracted project ready for remaining preflight / training handoff."""

    source_dir: Path
    child_env: dict[str, str] = field(repr=False)
    train_command: list[str]
    heartbeat: object | None = None
    config_hash: str | None = None
    _staging: tempfile.TemporaryDirectory[str] | None = field(default=None,
                                                              repr=False,
                                                              compare=False)

    def __repr__(self) -> str:
        # child_env may hold WANDB_API_KEY; only key names are safe to show.
        return (f"PreparedSource(source_dir={self.source_dir!r}, "
                f"child_env_keys={sorted(self.child_env)!r}, "
                f"train_command={self.train_command!r}, "
                f"config_hash={self.config_hash!r})")


def _utc_now() -> datetime:
    """The real wall clock, timezone-aware: `Clock.now`'s default.

    A module function, not a lambda default, for two reasons: the package-shape
    `import-time` clause rejects a lambda in a dataclass default, and one named
    object lets preflight and training default to the same clock.
    """
    return datetime.now(UTC)


def _event_wait(event: threading.Event, seconds: float) -> bool:
    """`event.wait(seconds)`: `Clock.wait`'s default, the fallback heartbeat's interruptible wait.

    Returns True iff the event was set (the heartbeat was told to stop) before
    the timeout. A module function for the same reasons as `_utc_now`.
    """
    return event.wait(seconds)


@dataclass(frozen=True)
class Clock:
    """Every time-dependent effect of an attempt goes through here.

    `now` stamps STATUS writes, heartbeats and checkpoint sidecars. `sleep` is
    BOTH the checkpoint settle window and the SIGTERM grace poll: one callable
    on purpose, because tests rely on a single recording `sleep` seeing both.
    `wait` is the fallback heartbeat's interruptible wait. A test swaps one
    field with `dataclasses.replace(clock, sleep=...)` and keeps the others.

    PITFALL, when defaults are resolved (gh#163): all three are bound
    when this module is imported. A test that monkeypatches the global
    `time.sleep` therefore does NOT reach `Clock().sleep`; inject a Clock. The
    W5 census found no test that patches `time.sleep` in any spelling.
    """

    now: Callable[[], datetime] = _utc_now
    sleep: Callable[[float], None] = time.sleep
    wait: Callable[[threading.Event, float], bool] = _event_wait


# Moved here from preflight.py in W5 (gh#163): `AttemptContext` holds one, and
# both preflight and training read that, so it is shared vocabulary now.
class ReloadingVolume(Protocol):
    """In-container Volume handle. reload before STATUS writes; commit after them."""

    def reload(self) -> None:
        ...

    def commit(self) -> None:
        ...


@dataclass(frozen=True)
class AttemptContext:
    """One delivery of one run: who writes its state, where, under which lock, on which clock.

    Both phases take the same one. Production builds it once and hands it to
    prepare and then to training, so "the same lock, the same run_root, the
    same Volume" is one object instead of three pairs of arguments that must
    agree. Prepare's status transitions are `transition_status(run_root,
    status, now=clock.now(), attempt_id=..., lock=...)`. PREPARING and
    BUILD_FAILED are each followed by `volume.commit()`, and so are training's
    TRAINING and terminal transitions. BUILDING is NOT: no commit of its own
    follows it, and the next commit of the Volume carries it (a heartbeat
    beat, the manifest rewrite's, or the next transition's). Keep it so when
    prepare's phases change: a commit added after BUILDING changes prepare's
    effect order, which the gh#163 refactor keeps identical to the runner's
    before it (no test pins that order yet: gh#236).

    PITFALL: it holds the Volume, not a bare `commit`. Prepare must reload and
    commit the SAME Volume, and a separate `commit` argument could name a
    different one; the tests' RecordingVolume.reload() discards uncommitted
    writes, so that split is a real hazard. In a test, change one field with
    `dataclasses.replace(ctx, volume=...)`: it keeps the lock, and a fresh
    AttemptContext with a new lock passes most tests while testing less.
    """

    attempt_id: str
    run_root: Path
    lock: LockLike
    volume: ReloadingVolume
    clock: Clock = field(default_factory=Clock)
