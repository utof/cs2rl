"""Shared runner values, protocols, and data records."""
from __future__ import annotations

import enum
import hashlib
import re
import tarfile
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import Protocol

VOLUME_NAME = "cs2rl-training-artifacts"
REGISTRY_NAME = "cs2rl-training-run-registry"
VOLUME_MOUNT = Path("/artifacts")
SOURCES_ROOT = PurePosixPath("sources")
INPUTS_ROOT = PurePosixPath("inputs")
RUNS_ROOT = PurePosixPath("runs")

# The image venv that holds torch/numpy/PufferLib and runs train.py. It is NOT
# the interpreter this module runs under on the container: a Modal function runs
# on the image's standalone python (/usr/local/bin/python from add_python=), which
# has only uv + the modal client. checkpoint.py's `_assert_weights_only_loadable`
# shells out to it when in-process torch is unavailable, and commands.py builds
# the install, train and CUDA-probe commands on it; both read it as
# `core.PREBUILT_PYTHON`, one of the package's qualified seams (see __init__.py).
PREBUILT_PYTHON = "/opt/cs2rl/.venv/bin/python"
# Cap on the out-of-process weights-only load. Generous for a ~2.5 MB policy;
# a hung interpreter must not stall the interrupt path's terminal write.
PREBUILT_LOAD_TIMEOUT_SECONDS = 120.0

ALLOWED_MAPS = frozenset({"simple", "dust2", "arena-duel"})            # R0-J: arena-duel (Task 12 map)
ALLOWED_GPUS = frozenset({"T4", "L4", "A10"})
ALLOWED_NUM_ENVS = frozenset({16, 32, 64, 128, 256})
ALLOWED_CPU_CORES = frozenset({4, 8, 16})
DEFAULT_GPU = "T4"
DEFAULT_NUM_ENVS = 256
DEFAULT_CPU_CORES = 8
DEFAULT_MEMORY_MIB = 16384
DEFAULT_VEC_WORKERS = 8
DEFAULT_TIMEOUT_MINUTES = 120
DEFAULT_SAVE_EVERY_SECONDS = 300
MIN_MEMORY_MIB = 8192
MAX_MEMORY_MIB = 32768
MIN_TIMEOUT_MINUTES = 1
MAX_TIMEOUT_MINUTES = 360
MIN_SAVE_EVERY_SECONDS = 60
MAX_SAVE_EVERY_SECONDS = 300
AGENTS_PER_ENV = 10
BPTT_HORIZON = 64
MIN_BATCH_SIZE = 8192

# Run IDs name Volume paths. The regex is the entire contract: start with an
# alnum so `.hidden` / `-flag-like` IDs cannot be confused with options, then
# at most 79 more alnum/dot/underscore/hyphen (80 total). Slash, `..`,
# whitespace, and shell metacharacters are all excluded by construction.
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
# Modal Secret names are operator-supplied identifiers, not secret values.
# Reuse the run-id grammar so a value cannot smuggle path/shell syntax.
_SECRET_NAME_RE = _RUN_ID_RE

# Exact live train.py long-option names and their arities. Prefixes are NOT
# listed: argparse would resolve `--devi` → `--device`, which is exactly the
# footgun this table exists to close. `--num-envs` is the runner spelling and
# is intentionally absent so train-args using it fail as unknown.
LIVE_TRAIN_OPTION_ARITY: dict[str, int] = {
    "--dust2": 0,
    "--map": 1,                                        # R0-H (Task 12): live --map {simple,dust2,arena-duel}
    "--smoke": 0,
    "--train": 0,
    "--record": 0,
    "--eval": 0,
    "--checkpoint": 1,
    "--resume": 1,
    "--resume-run": 1,                                 # R0-C (#134) full-state resume
    "--run-id": 1,
    "--checkpoint-interval": 1,
    "--timesteps": 1,
    "--num_envs": 1,
    "--seed": 1,
    "--device": 1,
    "--save_every_sec": 1,
    "--checkpoint_dir": 1,
    "--checkpoint-dir": 1,
    "--dump-config": 0,
    "--vec-backend": 1,
    "--vec-num-workers": 1,
    "--vec-overwork": 0,
    "--record-out": 1,
    "--record-policy": 1,
    "--eval-episodes": 1,
    "--eval-interval": 1,                              # R0-I (Task 13): in-training fixed-baseline eval cadence
    "--eval-policy": 1,
    "--name": 1,
    "--wandb": 0,
    "--wandb-project": 1,
    "--wandb-entity": 1,
    "--no-self-play": 0,
    "--no-dead-run-abort": 0,
    "--n-active-per-team": 1,
    "--pin-pitch": 1,                                  # R0-E.2 (#131)
    "--crouch-enabled": 1,                             # R0-E.2 (#131)
    "--jump-enabled": 1,                               # Rung 1a T2b
    "--opponent": 1,                                   # Rung 1a T3: {self,noop} statue opponent
    "--round-time-ticks": 1,                           # R0-G
    "--laser-range": 1,                                # R0-G
    "--max-turn-speed": 1,                             # R0-G
    "--aim-entropy-bonus": 1,                          # R0-E.4 (#131)
    "--aim-log-std-max": 1,                            # R0-E.3 (#131)
    "--gamma": 1,                                      # R0-J (Task 14)
    "--pbrs-gamma": 1,                                 # R0-J (Task 14)
    "--warmstart-entropy": 0,
    "--warmstart-grace-steps": 1,
    "--warmstart-ramp-steps": 1,
    "--warmstart-alpha-ceiling": 1,
    "--reward-symmetrize": 0,
    "--tag-diagnostic": 0,
    "--tag-every": 1,
    "--tct-split-heads": 0,
    "--tct-split-trunk": 0,
                                                       # Reward weights: dest is underscore, CLI is hyphen. Mirror the generator
                                                       # in src/train.py (`--{_rw_name.replace('_', '-')}`) so a new weight is a
                                                       # one-line add here, not a silent "unknown option" after a train.py bump.
    "--reward-win": 1,
    "--reward-kill": 1,
    "--reward-death": 1,
    "--reward-bombsite-entry": 1,
    "--reward-plant-bonus": 1,
    "--reward-plant-base": 1,
    "--reward-plant-progress-scale": 1,
    "--reward-plant-interrupted": 1,
    "--reward-defuse": 1,
    "--reward-shot-penalty": 1,
    "--reward-ct-survival": 1,
    "--reward-inaction": 1,
    "--reward-win-t-detonation": 1,
    "--reward-win-t-elimination": 1,
    "--reward-win-ct-defuse": 1,
    "--reward-win-ct-timeout": 1,
    "--reward-win-ct-elimination": 1,
    "--pbrs-alive-weight": 1,
    "--pbrs-hp-weight": 1,
    "--pbrs-site-weight": 1,
    "--pbrs-bomb-progress-weight": 1,
    "--pbrs-nav-weight-t": 1,
    "--pbrs-nav-weight-ct": 1,
}
LIVE_TRAIN_OPTIONS = frozenset(LIVE_TRAIN_OPTION_ARITY)

# Flags the runner injects (or whose mode it owns). Presence in --train-args
# is always a hard error, even when the live name is spelled exactly.
RUNNER_OWNED_TRAIN_FLAGS = frozenset({
    "--train",
    "--resume",
    "--resume-run",                                    # R0-C: local run-dir resume — the runner owns paths/ids
    "--run-id",
    "--name",
    "--checkpoint-dir",
    "--checkpoint_dir",
    "--device",
    "--save_every_sec",
    "--vec-backend",
    "--vec-num-workers",
    "--vec-overwork",
    "--dump-config",
    "--smoke",
    "--record",
    "--eval",
    "--dust2",
    "--map",                                           # R0-H: the runner owns map choice (effective_map); Task 14 emits it
    "--num_envs",
})

# Launch-only flags. status/download must reject these rather than ignore them
# — ignoring would hide an operator mistake and look like a successful observe.
RUN_ONLY_OPTIONS = frozenset({
    "--git-sha",
    "--map",
    "--gpu",
    "--cpu-cores",
    "--memory-mib",
    "--num-envs",
    "--vec-workers",
    "--timeout-minutes",
    "--save-every-seconds",
    "--train-args",
    "--resume-local-checkpoint",
    "--resume-run-id",
    "--wandb-secret-name",
    "--action",
})


class ValidationError(ValueError):
    """Locally-detectable invalid launch or observe request.

    Raised before any Dict/Volume write or GPU invocation. Message is operator
    facing; do not put secret values in it.
    """


class Action(enum.StrEnum):
    RUN = "run"
    STATUS = "status"
    DOWNLOAD = "download"


class Status(enum.StrEnum):
    """Run STATUS.json states. Terminal vs nonterminal is a partition."""

    PREPARING = "preparing"
    BUILDING = "building"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    BUILD_FAILED = "build_failed"


TERMINAL_STATUSES = frozenset({
    Status.COMPLETED,
    Status.FAILED,
    Status.INTERRUPTED,
    Status.BUILD_FAILED,
})
NONTERMINAL_STATUSES = frozenset({
    Status.PREPARING,
    Status.BUILDING,
    Status.TRAINING,
})


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


_COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

_SAFE_TAR_TYPES = {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE}

PROVENANCE_NAME = ".cs2rl-provenance.json"


@dataclass(frozen=True)
class SourceProvenance:
    """Content-addressed source snapshot: commit, tree, and archive digest."""

    commit: str
    tree: str
    archive_sha256: str
    archive_path: Path


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


# argv[1] is the checkpoint path. Kept out of the -c source so no filename can
# ever be interpolated into executed code.
_PREBUILT_LOAD_SOURCE = (
    "import sys, torch; torch.load(sys.argv[1], map_location='cpu', weights_only=True)")

STATUS_FILENAME = "STATUS.json"
SCHEMA_VERSION = 1

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
    def from_dict(cls, payload: Mapping[str, object]) -> RunStatus:
        return cls(
            schema_version=int(payload["schema_version"]),
            status=Status(str(payload["status"])),
            attempt_id=str(payload["attempt_id"]),
            updated_at=str(payload["updated_at"]),
        )


@dataclass(frozen=True)
class Manifest:
    """Minimum manifest.json contract from design §5.

    effective_map is authoritative. Do not store live config's `env` field: it
    is only a label, which src/train_config.py derives as `cs2-<map>` from
    train.py's resolved `--map` (`cs2-dust2` when the `map` attribute is
    missing or empty). The map the runner validated is effective_map.
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


HEARTBEAT_INTERVAL = timedelta(seconds=60)
STALE_AFTER = timedelta(minutes=5)
RESERVATION_FILENAME = "reservation.json"
MANIFEST_FILENAME = "manifest.json"


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


class LockLike(Protocol):

    def __enter__(self) -> object:
        ...

    def __exit__(self, *exc: object) -> None:
        ...


@dataclass(frozen=True)
class DerivedStatus:
    """Client-side view. Never written back to STATUS.json."""

    status: Status
    stale: bool
    reason: str | None = None


@dataclass(frozen=True)
class CheckpointVerdict:
    """Outcome of `verify_checkpoint`. `reason` is a token, never a sentence.

    Callers own the wording: launch maps the token through
    `run_modal._LAUNCH_CHECKPOINT_ERRORS` to a parent-checkpoint sentence,
    status reporting collapses it to a `checkpoint_loadable` bool. Keeping the
    sentence out of here is what stops a third error vocabulary appearing.

    Invariant: `ok=True` implies `reason is None` and both `checkpoint_bytes`
    and `digest` are set; `ok=False` implies both are `None`. The payload field
    is the *checkpoint* bytes, never the sidecar's — launch uploads them under
    `INPUTS_ROOT/sha256/{digest}.pt`, so carrying the sidecar here would ship
    the metadata as the weights.
    """

    ok: bool
    reason: str | None
    checkpoint_bytes: bytes | None
    digest: str | None


def sha256_bytes(data: bytes) -> str:
    """SHA-256 of an in-memory payload (normalized config JSON)."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class CompletionEvidence:
    last_step: int
    checkpoint_sha256: str
    config_hash: str


UV_BIN = "/usr/local/bin/uv"
# PREBUILT_PYTHON, the image venv's interpreter, is defined with the path
# constants at the top of this module rather than here.
TRAIN_SCRIPT = "src/train.py"

# Child env is an allowlist, not a denylist: Modal/image leftovers (tokens,
# extra WANDB_* creds, host thread caps) must not leak into uv/train.
_PRESERVED_CHILD_ENV_KEYS = frozenset({
    "PATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "HOME",
    "TMPDIR",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "LC_NUMERIC",
    "LC_TIME",
    "LC_COLLATE",
    "LC_MONETARY",
    "LC_MESSAGES",
    "LC_PAPER",
    "LC_NAME",
    "LC_ADDRESS",
    "LC_TELEPHONE",
    "LC_MEASUREMENT",
    "LC_IDENTIFICATION",
})
_THREAD_CAP_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

# Minimum CUDA tensors: 2-step x 1-agent, enough for compute_puff_advantage
# to dispatch without training-sized allocations.
CUDA_PROBE_SOURCE = """
import torch
import pufferlib.pufferl as pufferl
assert torch.cuda.is_available(), "cuda is not available"
assert pufferl.ADVANTAGE_CUDA, "ADVANTAGE_CUDA is false"
values = torch.zeros((2, 1), device="cuda")
rewards = torch.zeros((2, 1), device="cuda")
terminals = torch.zeros((2, 1), device="cuda")
ratio = torch.ones((2, 1), device="cuda")
advantages = torch.zeros((2, 1), device="cuda")
pufferl.compute_puff_advantage(
    values, rewards, terminals, ratio, advantages, 0.99, 0.95, 1.0, 1.0)
torch.cuda.synchronize()
""".strip()


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


class ReloadingVolume(Protocol):
    """In-container Volume handle. reload before STATUS writes; commit after them."""

    def reload(self) -> None:
        ...

    def commit(self) -> None:
        ...


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


TRAIN_LOG_NAME = "train.log"
RESULT_FILENAME = "result.json"
CHECKPOINT_NAME = "dust2_policy.pt"
CHECKPOINT_SIDECAR_NAME = "dust2_policy.pt.meta.json"
CHECKPOINT_PUBLISH_REASON_NAME = "dust2_policy.pt.publish_reason.json"
DEAD_CHECKPOINT_NAME = "dust2_policy_dead.pt"
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
