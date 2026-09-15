"""Modal-free request, path, and (later) artifact helpers for the optional runner.

WHY this module exists separately from scripts/run_modal.py: status/download and
all validation must work without importing or hydrating a Modal App. Importing
the App would build the CUDA image. This file therefore uses only the standard
library at module scope; Torch is imported lazily inside checkpoint validation
(Task 3), and Modal is never imported here.

PITFALLS:
  * Live train.py argparse accepts prefixes (`--devi` → `--device`). The runner
    must NOT. Only exact long-option names from the mirrored live set are legal.
  * Client Volume APIs take root-relative PurePosixPath (`runs/...`); the
    container sees the same object at `/artifacts/runs/...`. Mixing the two
    namespaces silently talks to the wrong path.
  * Live config.json currently writes env=cs2-dust2 even for the simple map.
    effective_map is the runner's source of truth and is never derived from that
    field.
"""
from __future__ import annotations

import enum
import gzip
import hashlib
import io
import json
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
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
# has only uv + the modal client. Defined here, above validate_local_checkpoint,
# because that validator shells out to it when in-process torch is unavailable.
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


@dataclass(frozen=True)
class ResumeRequest:
    """At most one resume source. Omitting both starts from scratch."""

    local_checkpoint: Path | None = None
    prior_run_id: str | None = None

    def __post_init__(self):
        if self.local_checkpoint is not None and self.prior_run_id is not None:
            raise ValidationError(
                "resume-local-checkpoint and resume-run-id are mutually exclusive")
        if self.prior_run_id is not None:
            validate_run_id(self.prior_run_id)


@dataclass(frozen=True)
class ArtifactClientRequest:
    """status/download: run id only. No launch options, no App hydration."""

    action: Action
    run_id: str


@dataclass(frozen=True)
class RunRequest:
    """Validated launch request. effective_map is required; never inferred."""

    run_id: str
    git_sha: str
    effective_map: str
    gpu: str = DEFAULT_GPU
    cpu_cores: int = DEFAULT_CPU_CORES
    memory_mib: int = DEFAULT_MEMORY_MIB
    num_envs: int = DEFAULT_NUM_ENVS
    vec_workers: int = DEFAULT_VEC_WORKERS
    timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES
    save_every_seconds: int = DEFAULT_SAVE_EVERY_SECONDS
    train_args: tuple[str, ...] = ()
    wandb_secret_name: str | None = None
    resume: ResumeRequest = field(default_factory=ResumeRequest)
    timesteps: int = 0

    @property
    def batch_size(self) -> int:
        """Live compute_batch_dims rule: num_envs * 10 agents * 64 BPTT."""
        return self.num_envs * AGENTS_PER_ENV * BPTT_HORIZON

    @property
    def cpu_request_limit(self) -> tuple[int, int]:
        """Equal request/soft-limit tuple for Function.with_options(cpu=...)."""
        return (self.cpu_cores, self.cpu_cores)

    @property
    def memory_request_limit(self) -> tuple[int, int]:
        """Equal request/hard-limit tuple for Function.with_options(memory=...)."""
        return (self.memory_mib, self.memory_mib)

    def training_argv(self, run_root: Path, remote_resume: str | None = None) -> list[str]:
        """Assemble the exact live argv for this request under run_root."""
        return _assemble_train_argv(self, run_root, remote_resume, dump_config=False)

    def __post_init__(self):
        validate_run_id(self.run_id)
        if self.effective_map not in ALLOWED_MAPS:
            raise ValidationError(
                f"map must be one of {sorted(ALLOWED_MAPS)}, got {self.effective_map!r}")
        if self.gpu not in ALLOWED_GPUS:
            raise ValidationError(f"gpu must be one of {sorted(ALLOWED_GPUS)}, got {self.gpu!r}")
        if self.num_envs not in ALLOWED_NUM_ENVS:
            raise ValidationError(
                f"num_envs must be one of {sorted(ALLOWED_NUM_ENVS)}, got {self.num_envs!r}")
        if self.cpu_cores not in ALLOWED_CPU_CORES:
            raise ValidationError(
                f"cpu_cores must be one of {sorted(ALLOWED_CPU_CORES)}, got {self.cpu_cores!r}")
        if not isinstance(self.memory_mib,
                          int) or not (MIN_MEMORY_MIB <= self.memory_mib <= MAX_MEMORY_MIB):
            raise ValidationError(
                f"memory_mib must be in [{MIN_MEMORY_MIB}, {MAX_MEMORY_MIB}], got {self.memory_mib!r}"
            )
        if not isinstance(self.timeout_minutes, int) or not (
                MIN_TIMEOUT_MINUTES <= self.timeout_minutes <= MAX_TIMEOUT_MINUTES):
            raise ValidationError(
                f"timeout_minutes must be in [{MIN_TIMEOUT_MINUTES}, {MAX_TIMEOUT_MINUTES}], "
                f"got {self.timeout_minutes!r}")
        if not isinstance(self.save_every_seconds, int) or not (
                MIN_SAVE_EVERY_SECONDS <= self.save_every_seconds <= MAX_SAVE_EVERY_SECONDS):
            raise ValidationError(f"save_every_seconds must be in [{MIN_SAVE_EVERY_SECONDS}, "
                                  f"{MAX_SAVE_EVERY_SECONDS}], got {self.save_every_seconds!r}")
        if not isinstance(self.vec_workers, int) or self.vec_workers < 1:
            raise ValidationError(f"vec_workers must be a positive int, got {self.vec_workers!r}")
        if self.vec_workers > self.cpu_cores:
            raise ValidationError(
                f"vec_workers {self.vec_workers} exceeds CPU soft limit {self.cpu_cores}")
        if self.num_envs % self.vec_workers != 0:
            raise ValidationError(
                f"vec_workers {self.vec_workers} must divide num_envs {self.num_envs}")
        batch_size = self.num_envs * AGENTS_PER_ENV * BPTT_HORIZON
        if batch_size < MIN_BATCH_SIZE:
            raise ValidationError(
                f"batch_size {batch_size} is below PufferLib floor {MIN_BATCH_SIZE}")
        if not isinstance(self.train_args, tuple) or not all(
                isinstance(token, str) for token in self.train_args):
            raise ValidationError("train_args must be a tuple of strings")
        steps = validate_train_args(self.train_args)
        object.__setattr__(self, "timesteps", steps)
        wandb_requested = "--wandb" in self.train_args
        if wandb_requested and self.wandb_secret_name is None:
            raise ValidationError("--wandb requires --wandb-secret-name")
        if self.wandb_secret_name is not None and not wandb_requested:
            raise ValidationError("--wandb-secret-name requires --wandb in --train-args")
        if self.wandb_secret_name is not None:
            validate_secret_name(self.wandb_secret_name)
        if not isinstance(self.resume, ResumeRequest):
            raise ValidationError(f"resume must be a ResumeRequest, got {type(self.resume)!r}")


def validate_run_id(value: str) -> str:
    """Return value if it is a safe Volume path component, else raise."""
    if not isinstance(value, str) or _RUN_ID_RE.fullmatch(value) is None:
        raise ValidationError(f"invalid run id: {value!r}")
    if ".." in value:
        # The regex already forbids `/` but `..` as a substring of an otherwise
        # legal id (a..b) is still a path-escape lookalike. Reject it.
        raise ValidationError(f"invalid run id: {value!r}")
    return value


def validate_secret_name(value: str) -> str:
    """Return a Modal Secret *name* (never a secret value)."""
    if not isinstance(value, str) or _SECRET_NAME_RE.fullmatch(value) is None:
        raise ValidationError(f"invalid Modal Secret name: {value!r}")
    return value


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


def parse_train_args(raw: str) -> tuple[str, ...]:
    """Split `--train-args` with shlex so punctuation stays argv data.

    Never pass the result to a shell. `org/name;rm -rf` is one token, not a
    command. Callers that already have a list should skip this and validate
    directly.
    """
    if not isinstance(raw, str):
        raise ValidationError(f"--train-args must be a string, got {type(raw)!r}")
    try:
        return tuple(shlex.split(raw))
    except ValueError as err:
        raise ValidationError(f"invalid --train-args quoting: {err}") from err


def _split_long_option(token: str) -> tuple[str, str | None]:
    """Canonicalize `--flag=value` to (`--flag`, `value`). No prefix matching."""
    if not token.startswith("--") or token == "--":
        raise ValidationError(f"unconsumed positional token: {token!r}")
    if "=" in token:
        name, value = token.split("=", 1)
        return name, value
    return token, None


def _option_value(name: str, eq_value: str | None, tokens: Sequence[str],
                  index: int) -> tuple[str, int]:
    """Return (value, index_of_value_token) for an arity-1 option."""
    if eq_value is not None:
        return eq_value, index
    next_index = index + 1
    if next_index >= len(tokens) or tokens[next_index].startswith("--"):
        raise ValidationError(f"{name} requires a value")
    return tokens[next_index], next_index


def validate_train_args(argv: Sequence[str]) -> int:
    """Consume each exact known option + arity; return the sole --timesteps.

    Rejects: unknown names, argparse-style prefixes, runner-owned flags,
    positionals, missing/zero/duplicate timesteps. Ownership is checked only
    AFTER the name matches the live set, so `--num-envs` fails as unknown
    rather than as a special case — same path as `--devi`.
    """
    timesteps: int | None = None
    index = 0
    while index < len(argv):
        token = argv[index]
        name, eq_value = _split_long_option(token)
        if name not in LIVE_TRAIN_OPTION_ARITY:
            raise ValidationError(f"unknown or abbreviated option: {name}")
        if name in RUNNER_OWNED_TRAIN_FLAGS:
            raise ValidationError(f"runner-owned option not allowed in --train-args: {name}")
        arity = LIVE_TRAIN_OPTION_ARITY[name]
        if arity == 0:
            if eq_value is not None:
                raise ValidationError(f"{name} does not take a value")
            index += 1
            continue
        value, value_index = _option_value(name, eq_value, argv, index)
        if name == "--timesteps":
            if timesteps is not None:
                raise ValidationError("--timesteps supplied more than once")
            try:
                steps = int(value)
            except ValueError as err:
                raise ValidationError(f"--timesteps must be a positive int, got {value!r}") from err
            if steps <= 0:
                raise ValidationError(f"--timesteps must be a positive int, got {value!r}")
            timesteps = steps
        index = value_index + 1
    if timesteps is None:
        raise ValidationError("--timesteps must be supplied exactly once")
    return timesteps


def build_run_request(
    *,
    run_id: str,
    git_sha: str,
    effective_map: str,
    gpu: str = DEFAULT_GPU,
    cpu_cores: int = DEFAULT_CPU_CORES,
    memory_mib: int = DEFAULT_MEMORY_MIB,
    num_envs: int = DEFAULT_NUM_ENVS,
    vec_workers: int = DEFAULT_VEC_WORKERS,
    timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES,
    save_every_seconds: int = DEFAULT_SAVE_EVERY_SECONDS,
    train_args: str | Sequence[str] = (),
    wandb_secret_name: str | None = None,
    resume_local_checkpoint: str | Path | None = None,
    resume_run_id: str | None = None,
) -> RunRequest:
    """Validate structured launch fields and return a frozen request.

    effective_map is required (no default in the signature on purpose). Resource
    defaults match the design: T4 / 8 CPU / 16384 MiB / 256 envs / 8 workers /
    120 min / 300 s save. train_args defaults to () so a missing --timesteps
    cannot inherit a 30M scientific budget.
    """
    validate_run_id(run_id)
    if effective_map not in ALLOWED_MAPS:
        raise ValidationError(f"map must be one of {sorted(ALLOWED_MAPS)}, got {effective_map!r}")
    if gpu not in ALLOWED_GPUS:
        raise ValidationError(f"gpu must be one of {sorted(ALLOWED_GPUS)}, got {gpu!r}")
    if num_envs not in ALLOWED_NUM_ENVS:
        raise ValidationError(
            f"num_envs must be one of {sorted(ALLOWED_NUM_ENVS)}, got {num_envs!r}")
    if cpu_cores not in ALLOWED_CPU_CORES:
        raise ValidationError(
            f"cpu_cores must be one of {sorted(ALLOWED_CPU_CORES)}, got {cpu_cores!r}")
    if not isinstance(memory_mib, int) or not (MIN_MEMORY_MIB <= memory_mib <= MAX_MEMORY_MIB):
        raise ValidationError(
            f"memory_mib must be in [{MIN_MEMORY_MIB}, {MAX_MEMORY_MIB}], got {memory_mib!r}")
    if not isinstance(timeout_minutes,
                      int) or not (MIN_TIMEOUT_MINUTES <= timeout_minutes <= MAX_TIMEOUT_MINUTES):
        raise ValidationError(
            f"timeout_minutes must be in [{MIN_TIMEOUT_MINUTES}, {MAX_TIMEOUT_MINUTES}], "
            f"got {timeout_minutes!r}")
    if not isinstance(save_every_seconds, int) or not (MIN_SAVE_EVERY_SECONDS <= save_every_seconds
                                                       <= MAX_SAVE_EVERY_SECONDS):
        raise ValidationError(f"save_every_seconds must be in [{MIN_SAVE_EVERY_SECONDS}, "
                              f"{MAX_SAVE_EVERY_SECONDS}], got {save_every_seconds!r}")
    if not isinstance(vec_workers, int) or vec_workers < 1:
        raise ValidationError(f"vec_workers must be a positive int, got {vec_workers!r}")
    if vec_workers > cpu_cores:
        raise ValidationError(f"vec_workers {vec_workers} exceeds CPU soft limit {cpu_cores}")
    if num_envs % vec_workers != 0:
        raise ValidationError(f"vec_workers {vec_workers} must divide num_envs {num_envs}")
    batch_size = num_envs * AGENTS_PER_ENV * BPTT_HORIZON
    if batch_size < MIN_BATCH_SIZE:
        raise ValidationError(f"batch_size {batch_size} is below PufferLib floor {MIN_BATCH_SIZE}")
    parsed_train_args = (parse_train_args(train_args)
                         if isinstance(train_args, str) else tuple(train_args))
    timesteps = validate_train_args(parsed_train_args)
    wandb_requested = "--wandb" in parsed_train_args
    if wandb_requested and wandb_secret_name is None:
        raise ValidationError("--wandb requires --wandb-secret-name")
    if wandb_secret_name is not None and not wandb_requested:
        raise ValidationError("--wandb-secret-name requires --wandb in --train-args")
    if wandb_secret_name is not None:
        validate_secret_name(wandb_secret_name)
    if resume_local_checkpoint is not None and resume_run_id is not None:
        raise ValidationError("resume-local-checkpoint and resume-run-id are mutually exclusive")
    if resume_run_id is not None:
        validate_run_id(resume_run_id)
    if timesteps < batch_size:
        raise ValidationError(f"--timesteps {timesteps} is below one full batch ({batch_size})")
    resume = ResumeRequest(
        local_checkpoint=Path(resume_local_checkpoint) if resume_local_checkpoint else None,
        prior_run_id=resume_run_id,
    )
    return RunRequest(
        run_id=run_id,
        git_sha=git_sha,
        effective_map=effective_map,
        gpu=gpu,
        cpu_cores=cpu_cores,
        memory_mib=memory_mib,
        num_envs=num_envs,
        vec_workers=vec_workers,
        timeout_minutes=timeout_minutes,
        save_every_seconds=save_every_seconds,
        train_args=parsed_train_args,
        wandb_secret_name=wandb_secret_name,
        resume=resume,
        timesteps=timesteps,
    )


def parse_artifact_client_request(argv: Sequence[str]) -> ArtifactClientRequest:
    """Parse status/download argv. Any launch-only option is a hard error.

    The observe client must never look like it accepted --gpu/--train-args/etc.
    Ignoring unknown flags would hide a mistaken `modal_artifacts.py status`
    invocation that the operator thought launched something.
    """
    if not argv:
        raise ValidationError("missing action")
    try:
        action = Action(argv[0])
    except ValueError as err:
        raise ValidationError(f"unknown action: {argv[0]!r}") from err
    if action is Action.RUN:
        raise ValidationError("run is not an artifact-client action")
    run_id: str | None = None
    tokens = list(argv[1:])
    while tokens:
        token = tokens.pop(0)
        name, eq_value = (token.split("=", 1) + [None])[:2] if token.startswith("--") else (token,
                                                                                            None)
        if name == "--run-id":
            if eq_value is None:
                if not tokens:
                    raise ValidationError("--run-id requires a value")
                eq_value = tokens.pop(0)
            run_id = validate_run_id(eq_value)
            continue
        if name in RUN_ONLY_OPTIONS:
            raise ValidationError(f"{action.value} rejects run-only option {name}")
        raise ValidationError(f"{action.value} rejects option {token!r}")
    if run_id is None:
        raise ValidationError(f"{action.value} requires --run-id")
    return ArtifactClientRequest(action=action, run_id=run_id)


_COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git with a list argv. Never a shell string; cwd is the target repo."""
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def validate_clean_head(repo: Path, requested_sha: str) -> str:
    """Require requested_sha to be the clean HEAD commit of repo.

    Checks, in order: 40-hex shape, `cat-file -e <sha>^{commit}`, equality with
    `rev-parse HEAD`, then both unstaged and staged diffs vs that SHA. Untracked
    files are intentionally ignored — they are not shipped (git archive).
    Returns the lowercase canonical SHA.
    """
    if not isinstance(requested_sha, str) or _COMMIT_SHA_RE.fullmatch(requested_sha) is None:
        raise ValidationError(f"git-sha must be 40 hex chars, got {requested_sha!r}")
    canonical = requested_sha.lower()
    probe = _run_git(repo, "cat-file", "-e", f"{canonical}^{{commit}}")
    if probe.returncode != 0:
        raise ValidationError(f"git-sha is not a commit object: {canonical}")
    head = _run_git(repo, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise ValidationError(f"cannot resolve HEAD in {repo}: {head.stderr.strip()}")
    head_sha = head.stdout.strip()
    if head_sha != canonical:
        raise ValidationError(f"git-sha {canonical} is not HEAD ({head_sha})")
    unstaged = _run_git(repo, "diff", "--quiet", canonical, "--")
    if unstaged.returncode != 0:
        raise ValidationError("tracked working tree does not match git-sha")
    staged = _run_git(repo, "diff", "--cached", "--quiet", canonical, "--")
    if staged.returncode != 0:
        raise ValidationError("index does not match git-sha")
    return canonical


_SAFE_TAR_TYPES = {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE}


def _reject_unsafe_tar_member(member: tarfile.TarInfo) -> None:
    """Fail closed on anything git archive should never emit for our sources."""
    name = member.name
    if member.issym() or member.islnk():
        raise ValidationError(f"archive member is a link: {name}")
    if member.isfifo() or member.ischr() or member.isblk():
        raise ValidationError(f"archive member is a special file: {name}")
    if member.type not in _SAFE_TAR_TYPES:
        raise ValidationError(f"archive member has unsafe type {member.type!r}: {name}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise ValidationError(f"archive member escapes destination: {name}")


def safe_extract_git_archive(archive: Path, destination: Path) -> None:
    """Extract a git tar only after every member has passed the safety scan.

    The producer is trusted git, but this is the last check before bytes land
    on disk. Scan first, then extract — do not extract-and-rollback.
    """
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        for member in members:
            _reject_unsafe_tar_member(member)
        # 3.12 filter='data' is a second belt: no links, no absolute paths.
        tar.extractall(destination, members=members, filter="data")


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


def _staging_members(staging: Path) -> list[tuple[str, Path]]:
    """Return (posix-relpath, path) pairs including parent dirs, sorted."""
    members: dict[str, Path] = {}
    for path in staging.rglob("*"):
        relative = PurePosixPath(*path.relative_to(staging).parts)
        members[str(relative)] = path
        parent = relative.parent
        while parent.parts:
            key = str(parent)
            members.setdefault(key, staging.joinpath(*parent.parts))
            parent = parent.parent
    return sorted(members.items(), key=lambda item: item[0])


def _repack_deterministic(staging: Path, destination: Path) -> None:
    """Write a gzip tar with frozen metadata so the digest is reproducible.

    uid/gid/mtime/names are zeroed. Dirs and git-executable files are 0755;
    everything else is 0644. gzip mtime=0 and no header filename, so two
    machines packaging the same commit produce the same bytes.
    """
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for name, path in _staging_members(staging):
                    info = tarfile.TarInfo(name=name)
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    if path.is_dir():
                        info.type = tarfile.DIRTYPE
                        info.mode = 0o755
                        tar.addfile(info)
                        continue
                    info.type = tarfile.REGTYPE
                    info.mode = 0o755 if path.stat().st_mode & stat.S_IXUSR else 0o644
                    info.size = path.stat().st_size
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)


def create_source_bundle(repo: Path, sha: str, destination: Path) -> SourceProvenance:
    """Package a clean HEAD commit into a content-addressed gzip tar.

    Staging lives under TemporaryDirectory — never inside the repo — so a crash
    cannot leave a dirty tree or a partial archive next to source.
    """
    canonical = validate_clean_head(repo, sha)
    tree_proc = _run_git(repo, "rev-parse", f"{canonical}^{{tree}}")
    if tree_proc.returncode != 0:
        raise ValidationError(f"cannot resolve tree for {canonical}: {tree_proc.stderr.strip()}")
    tree = tree_proc.stdout.strip()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cs2rl-src-") as tmp:
        tmp_path = Path(tmp)
        raw_tar = tmp_path / "git.tar"
        packed = tmp_path / "packed.tar.gz"
        archive = _run_git(repo, "archive", "--format=tar", f"--output={raw_tar}", canonical)
        if archive.returncode != 0:
            raise ValidationError(f"git archive failed: {archive.stderr.strip()}")
        staging = tmp_path / "tree"
        staging.mkdir()
        safe_extract_git_archive(raw_tar, staging)
        sidecar = {"commit": canonical, "tree": tree}
        (staging / PROVENANCE_NAME).write_text(json.dumps(sidecar, sort_keys=True) + "\n")
        _repack_deterministic(staging, packed)
        destination.write_bytes(packed.read_bytes())
    return SourceProvenance(
        commit=canonical,
        tree=tree,
        archive_sha256=sha256_file(destination),
        archive_path=destination,
    )


@dataclass(frozen=True)
class FileProvenance:
    """Content-addressed local file destined for inputs/sha256/<digest>.pt."""

    sha256: str
    size: int
    client_path: PurePosixPath
    mount_path: Path


def _import_torch() -> object:
    """Import torch, or raise ImportError. A seam, not a convenience wrapper.

    Tests monkeypatch this to reproduce the container runner's torch-less
    interpreter; without the seam the whole prebuilt fallback below is
    unreachable from a laptop, which is exactly how it stayed broken.
    """
    import torch
    return torch


# argv[1] is the checkpoint path. Kept out of the -c source so no filename can
# ever be interpolated into executed code.
_PREBUILT_LOAD_SOURCE = (
    "import sys, torch; torch.load(sys.argv[1], map_location='cpu', weights_only=True)")


def _assert_weights_only_loadable(path: Path) -> None:
    """Prove `path` is a weights-only-loadable torch checkpoint.

    In-process when torch is importable (laptop, tests, training child). On the
    Modal container the runner interpreter has NO torch — every dependency lives
    in the PREBUILT_PYTHON venv — so the load is delegated to that interpreter.
    Verified in a live container: runner_torch=MISSING, prebuilt subprocess=ok.

    PITFALL: never soften a failure here into "skip". A checkpoint that cannot
    be proven loadable must raise, so the caller records a reason instead of
    silently publishing nothing (gh: the three-T4-run sidecar hunt).
    """
    try:
        torch = _import_torch()
    except ImportError:
        pass
    else:
        try:
            torch.load(path, map_location="cpu", weights_only=True)
        except Exception as err:
            raise ValidationError(f"checkpoint is not weights-only loadable: {path}") from err
        return
    if not Path(PREBUILT_PYTHON).is_file():
        raise ValidationError(f"cannot validate {path}: torch is not importable and the prebuilt "
                              f"interpreter {PREBUILT_PYTHON} does not exist")
    try:
        completed = subprocess.run(
            [PREBUILT_PYTHON, "-c", _PREBUILT_LOAD_SOURCE,
             os.fspath(path)],
            capture_output=True,
            text=True,
            timeout=PREBUILT_LOAD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as err:
        raise ValidationError(f"cannot validate {path}: {PREBUILT_PYTHON} did not finish within "
                              f"{PREBUILT_LOAD_TIMEOUT_SECONDS}s") from err
    except OSError as err:
        raise ValidationError(f"cannot validate {path}: {PREBUILT_PYTHON} failed to run: "
                              f"{err}") from err
    if completed.returncode != 0:
        raise ValidationError(f"checkpoint is not weights-only loadable: {path}: "
                              f"{completed.stderr.strip()[-400:]}")


def validate_local_checkpoint(path: Path) -> FileProvenance:
    """Weights-only load, hash, and map a local checkpoint to the Volume path.

    Torch is never imported at module scope: that would pull CUDA/pynvml into
    every status/download invocation and into the import-boundary tests.
    """
    path = Path(path)
    if not path.is_file():
        raise ValidationError(f"checkpoint is not a readable file: {path}")
    _assert_weights_only_loadable(path)
    digest = sha256_file(path)
    client_path = INPUTS_ROOT / "sha256" / f"{digest}.pt"
    return FileProvenance(
        sha256=digest,
        size=path.stat().st_size,
        client_path=client_path,
        mount_path=mounted_path(client_path),
    )


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

    effective_map is authoritative. Do not store live config's `env` field —
    that currently says cs2-dust2 even for the simple map.
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


def _read_status(run_root: Path) -> RunStatus | None:
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
    current = _read_status(run_root)
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


FAILURE_UPLOAD = "upload_failed"
ALLOWED_FAILURE_CODES = frozenset({FAILURE_UPLOAD})


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


REDELIVERED = "redelivered"


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
        current = _read_status(Path(run_root))
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


def _load_volume_json(raw: bytes, message: str = "corrupt volume json") -> object:
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
    local Path wrapper all reach the same judgement through this function, so
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
        usually replaced: `collect_status` and backfill name the run id and the
        Path wrapper names the run root. `prior_checkpoint_or_raise` is the
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

    The third caller, the local Path wrapper `derive_run_view`, is not a
    fourth vocabulary but a newcomer to this one: before unification it parsed
    inline and raised raw JSONDecodeError / KeyError / TypeError, producing
    none of these labelled messages. Spec §2.2 declares that under "Behaviour
    change (stated, not wrapped)", and
    test_path_derive_run_view_corrupt_status_is_validation_error pins it.

    Every spelling is pinned by the message table in
    tests/test_modal_protocol.py. Adding or removing either clause silently
    changes what an operator sees; change the pinned table first.
    """
    if status_bytes is not None:
        try:
            payload = _load_volume_json(status_bytes, "corrupt volume status json")
            return derive_status(RunStatus.from_dict(payload), now=now)
        except (TypeError, ValueError, KeyError) as err:
            raise ValidationError("corrupt volume status json") from err
    if reservation_bytes is None:
        raise ValidationError("no STATUS.json or reservation.json")
    try:
        payload = _load_volume_json(reservation_bytes, "corrupt volume reservation json")
        created = _parse_iso8601(str(payload["created_at"]))
    except ValidationError:
        raise
    except (TypeError, ValueError, KeyError) as err:
        raise ValidationError("corrupt volume reservation json") from err
    if now - created >= STALE_AFTER:
        return DerivedStatus(status=Status.INTERRUPTED, stale=True, reason="no-heartbeat")
    return DerivedStatus(status=Status.PREPARING, stale=False, reason="no-heartbeat")


def derive_run_view(run_root: Path, *, now: datetime) -> DerivedStatus:
    """Derive status from STATUS.json or, if missing, reservation.json.

    A crash after the durable reservation but before the first STATUS write
    looks like preparing/no-heartbeat for five minutes, then interrupted.
    """
    run_root = Path(run_root)
    status_path = run_root / STATUS_FILENAME
    reservation_path = run_root / RESERVATION_FILENAME
    status_bytes = status_path.read_bytes() if status_path.is_file() else None
    reservation_bytes = reservation_path.read_bytes() if reservation_path.is_file() else None
    if status_bytes is None and reservation_bytes is None:
        raise ValidationError(f"no STATUS.json or reservation.json under {run_root}")
    return derive_run_view_from_bytes(status_bytes, reservation_bytes, now=now)


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


def _load_checkpoint_weights(buf: object, **kwargs: object) -> object:
    """Default `load` for `verify_checkpoint`: weights-only torch.load to CPU.

    torch is imported lazily so that importing this module — which the laptop-
    side status client does — does not pull in torch or touch CUDA.

    PITFALL: `**kwargs` is accepted and ignored. `verify_checkpoint` calls its
    `load` with `map_location`/`weights_only` spelled out; this function
    swallows them and hard-codes the same values rather than forwarding, so it
    cannot be talked into loading with weaker settings. Do not "simplify" it to
    `torch.load(buf, **kwargs)`.

    That is a property of this default only, not a security boundary. The
    `load=` parameter is a full replacement: an injected callable (the test
    fakes discard their kwargs outright) skips the weights-only check
    altogether. Injection is for tests; production must not pass `load=`.
    """
    import torch
    return torch.load(buf, map_location="cpu", weights_only=True)


def verify_checkpoint(
    sidecar_bytes: bytes | None,
    checkpoint_bytes: bytes | None,
    sidecar_reread_bytes: bytes | None,
    *,
    load: Callable[..., object] = _load_checkpoint_weights,
) -> CheckpointVerdict:
    """Decide whether a run's checkpoint is trustworthy, from raw bytes.

    Seven ordered checks; the first failure short-circuits, so a caller never
    pays for `torch.load` on bytes already known to be the wrong size. Returns
    a verdict instead of raising because two callers want two different things
    from the same judgement (a bool for status, an exception for launch).

    `sidecar_reread_bytes` is a *second* read of the same sidecar path, taken
    after the checkpoint read. Comparing it to the first read is how a sidecar
    rewritten underneath an in-flight verification is caught.

    Reason tokens, and only these (`reason is None` on success):
      * "missing_sidecar"     — the first sidecar read returned None.
      * "corrupt_sidecar"     — sidecar JSON did not parse, or is not a dict.
      * "missing_checkpoint"  — the checkpoint read returned None.
      * "stale_size"          — sidecar `size` != len(checkpoint bytes). Names
        the usual cause: a sidecar left behind by an earlier, shorter write.
      * "digest_mismatch"     — sidecar `sha256` != sha256 of the bytes read.
      * "not_loadable"        — `load(...)` raised, i.e. the bytes are not a
        weights-only-loadable torch payload.
      * "replaced"            — the sidecar reread is missing, unparsable, or
        parses to a different object than the first read: someone published a
        new sidecar while we were verifying, so neither read can be trusted.

    PITFALLS:
      * Torn-write detection compares *parsed JSON*, not raw bytes, so a
        reformatted-but-equal sidecar is not "replaced". Matching pre-
        unification adapter behaviour; do not tighten it to a byte compare.
      * A sidecar is mandatory. There is no orphan-checkpoint mode here:
        `sidecar_bytes is None` is "missing_sidecar", full stop. That is why
        `modal_backfill_sidecar` — whose whole job is a checkpoint with no
        sidecar yet — deliberately does not call this and runs its own local
        `torch.load` instead.
      * Being a live run is not this function's business. Launch gates on
        terminal-or-stale *before* calling; folding that in would make
        `collect_status` lie about a healthy running job's checkpoint.
      * Adding an eighth token means updating `_LAUNCH_CHECKPOINT_ERRORS` in
        scripts/run_modal.py, which indexes this token directly — an unmapped
        token escapes launch as a bare KeyError. Nothing checks that for you.
        test_launch_checkpoint_errors_is_total compares that map against the
        hand-written PROTOCOL_TOKENS tuple, which keeps those two in step, but
        the tuple is not derived from this function: a token added here and
        nowhere else leaves that test green. Update all three by hand.
    """

    def fail(reason: str) -> CheckpointVerdict:
        return CheckpointVerdict(ok=False, reason=reason, checkpoint_bytes=None, digest=None)

    if sidecar_bytes is None:
        return fail("missing_sidecar")
    try:
        sidecar = _load_volume_json(sidecar_bytes)
    except ValidationError:
        return fail("corrupt_sidecar")
    if not isinstance(sidecar, dict):
        return fail("corrupt_sidecar")
    if checkpoint_bytes is None:
        return fail("missing_checkpoint")
    if int(sidecar.get("size", -1)) != len(checkpoint_bytes):
        return fail("stale_size")
    digest = sha256_bytes(checkpoint_bytes)
    if sidecar.get("sha256") != digest:
        return fail("digest_mismatch")
    try:
        load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    except Exception:
        return fail("not_loadable")
    if sidecar_reread_bytes is None:
        return fail("replaced")
    try:
        sidecar_b = _load_volume_json(sidecar_reread_bytes)
    except ValidationError:
        return fail("replaced")
    if sidecar_b != sidecar:
        return fail("replaced")
    return CheckpointVerdict(
        ok=True,
        reason=None,
        checkpoint_bytes=checkpoint_bytes,
        digest=digest,
    )


def sha256_bytes(data: bytes) -> str:
    """SHA-256 of an in-memory payload (normalized config JSON)."""
    return hashlib.sha256(data).hexdigest()


def normalize_config_for_transport(config: Mapping[str, object]) -> dict[str, object]:
    """Drop only the run-local checkpoint data_dir; everything else must match.

    data_dir is the one path the trainer rewrites to the mounted run directory.
    Any other drift is a real config mismatch and must fail completion.
    """
    return {key: value for key, value in config.items() if key != "data_dir"}


@dataclass(frozen=True)
class CompletionEvidence:
    last_step: int
    checkpoint_sha256: str
    config_hash: str


def _iter_metrics_steps(metrics_path: Path) -> list[int]:
    """Parse every nonblank JSONL row; pin the live `step` key."""
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        raise ValidationError(f"metrics file missing or empty: {metrics_path}")
    steps: list[int] = []
    with metrics_path.open() as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValidationError(f"malformed metrics line {line_no}") from err
            if "step" not in row:
                raise ValidationError(f"metrics line {line_no} missing live key 'step'")
            try:
                step = int(row["step"])
            except (TypeError, ValueError) as err:
                raise ValidationError(f"metrics line {line_no} has non-integer step") from err
            if step < 0:
                raise ValidationError(f"metrics line {line_no} has negative step {step}")
            if steps and step < steps[-1]:
                raise ValidationError(
                    f"metrics step not monotonic at line {line_no}: {steps[-1]} -> {step}")
            steps.append(step)
    if not steps:
        raise ValidationError(f"metrics file has no rows: {metrics_path}")
    return steps


def validate_completed_run(run_root: Path, manifest: Manifest) -> CompletionEvidence:
    """Accept a terminal run only with loadable ckpt, matching config, and enough steps.

    last `step` is compared to effective_timesteps = floor(requested/batch)*batch,
    never to the raw request. A torn file's earlier maximum is ignored — we use
    the last row only after proving the whole file is monotonic.
    """
    run_root = Path(run_root)
    ckpt_dir = run_root / "checkpoints"
    ckpt = ckpt_dir / "dust2_policy.pt"
    validate_local_checkpoint(ckpt)
    config_path = ckpt_dir / "config.json"
    if not config_path.is_file():
        raise ValidationError("missing checkpoints/config.json")
    config = json.loads(config_path.read_text())
    normalized = normalize_config_for_transport(config)
    config_hash = sha256_bytes(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode())
    if config_hash != manifest.config_hash:
        raise ValidationError("config hash does not match manifest")
    if manifest.effective_timesteps < manifest.batch_size:
        raise ValidationError("effective_timesteps is less than one full batch")
    expected = (manifest.requested_timesteps // manifest.batch_size) * manifest.batch_size
    if manifest.effective_timesteps != expected:
        raise ValidationError(
            f"effective_timesteps {manifest.effective_timesteps} != floor formula {expected}")
    steps = _iter_metrics_steps(ckpt_dir / "metrics.jsonl")
    last_step = steps[-1]
    if last_step < manifest.effective_timesteps:
        raise ValidationError(
            f"last metrics step {last_step} < effective_timesteps {manifest.effective_timesteps}")
    return CompletionEvidence(
        last_step=last_step,
        checkpoint_sha256=sha256_file(ckpt),
        config_hash=config_hash,
    )


def list_run_artifacts(run_root: Path) -> list[Path]:
    """Every file under the run, including unknown trainer outputs.

    Download must not whitelist the minimum schema and drop extras.
    """
    run_root = Path(run_root)
    return sorted(path for path in run_root.rglob("*") if path.is_file())


def _assemble_train_argv(
    request: RunRequest,
    run_root: Path,
    remote_resume: str | None,
    *,
    dump_config: bool,
) -> list[str]:
    """Build the live argv. Never a shell string; every token is already split.

    Order is part of the contract (tests pin it): mode flag, `--map
    <effective_map>`, the already-validated user train-args, then each
    runner-owned flag exactly once. --resume is appended only when the caller
    supplies a remote path — the runner owns that flag, so a user --resume
    never reaches this function.

    R0-J (Task 14): the map is ALWAYS emitted as `--map` (never the `--dust2`
    alias): train.py resolves `--map` over `--dust2`, so an alias here would
    be the one flag whose presence changes nothing — and `simple` was the
    implicit no-flag default, which is exactly the kind of silent default the
    runner exists to pin. A user --map/--dust2 in train-args is rejected by
    validate_train_args, so the count here is exactly one.
    """
    argv: list[str] = ["--dump-config"] if dump_config else ["--train"]
    argv.extend(["--map", request.effective_map])
    argv.extend(request.train_args)
    argv.extend([
        "--num_envs",
        str(request.num_envs),
        "--checkpoint-dir",
        f"{run_root}/checkpoints",
        "--device",
        "cuda",
        "--save_every_sec",
        str(request.save_every_seconds),
        "--vec-backend",
        "multiprocessing",
        "--vec-num-workers",
        str(request.vec_workers),
    ])
    if remote_resume is not None:
        argv.extend(["--resume", remote_resume])
    return argv


def build_train_argv(request: RunRequest, remote_resume: str | None) -> list[str]:
    """Live training argv with checkpoint-dir under the mounted run root."""
    run_root = mounted_path(RUNS_ROOT / request.run_id)
    return _assemble_train_argv(request, run_root, remote_resume, dump_config=False)


def build_dump_config_argv(request: RunRequest, remote_resume: str | None) -> list[str]:
    """Same owned flags as training, but --dump-config and no --train.

    Used for the cheap remote config-hash preflight. remote_resume is accepted
    so a resumed run fingerprints the same --resume path training will load.
    """
    run_root = mounted_path(RUNS_ROOT / request.run_id)
    return _assemble_train_argv(request, run_root, remote_resume, dump_config=True)


UV_BIN = "/usr/local/bin/uv"
# PREBUILT_PYTHON is defined with the path constants at the top of this module —
# validate_local_checkpoint needs it and is defined long before this point.
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


def _is_preserved_child_env_key(key: str) -> bool:
    return key in _PRESERVED_CHILD_ENV_KEYS or key.startswith("LC_")


def build_child_env(
    parent: Mapping[str, str],
    *,
    wandb_enabled: bool = False,
    wandb_api_key: str | None = None,
) -> dict[str, str]:
    """Allowlisted runtime/build env plus forced single-thread BLAS caps.

    WANDB_* never comes from the parent. When W&B is on, only WANDB_API_KEY
    from the attached Secret is added — callers must not log or persist it.
    """
    env = {key: value for key, value in parent.items() if _is_preserved_child_env_key(key)}
    env.update(_THREAD_CAP_ENV)
    if wandb_enabled:
        if not wandb_api_key:
            raise ValidationError("WANDB_API_KEY is required when W&B is enabled")
        env["WANDB_API_KEY"] = wandb_api_key
    return env


def build_install_command(source_dir: str | Path) -> list[str]:
    """Install only the extracted project into the prebuilt image venv."""
    return [
        UV_BIN,
        "pip",
        "install",
        "--python",
        PREBUILT_PYTHON,
        "--no-deps",
        "--no-build-isolation",
        os.fspath(source_dir),
    ]


def build_train_command(argv: Sequence[str]) -> list[str]:
    """Prebuilt interpreter + live script + already-split argv. Never a shell."""
    return [PREBUILT_PYTHON, TRAIN_SCRIPT, *argv]


def build_dump_config_command(request: RunRequest, remote_resume: str | None) -> list[str]:
    """Cheap --dump-config invocation with the same owned flags as training."""
    return build_train_command(build_dump_config_argv(request, remote_resume))


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


def build_cuda_probe_command() -> list[str]:
    """Prebuilt interpreter running the CUDA/PufferLib advantage probe string."""
    return [PREBUILT_PYTHON, "-c", CUDA_PROBE_SOURCE]


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


def _verify_extracted_provenance(source_dir: Path, expected_commit: str,
                                 expected_tree: str) -> None:
    sidecar_path = source_dir / PROVENANCE_NAME
    if not sidecar_path.is_file():
        raise ValidationError(f"missing provenance sidecar: {sidecar_path}")
    payload = json.loads(sidecar_path.read_text())
    commit = str(payload.get("commit", ""))
    tree = str(payload.get("tree", ""))
    if commit != expected_commit or tree != expected_tree:
        raise ValidationError(f"provenance sidecar mismatch: commit {commit} tree {tree} "
                              f"!= expected {expected_commit} {expected_tree}")


def _stop_heartbeat(heartbeat: object | None) -> None:
    if heartbeat is None:
        return
    stop = getattr(heartbeat, "stop_and_join", None)
    if stop is not None:
        stop()


def _validate_remote_resume(path: Path, expected_sha256: str | None) -> FileProvenance:
    """Weights-only load the mounted checkpoint and pin its content hash."""
    provenance = validate_local_checkpoint(path)
    if expected_sha256 is not None and provenance.sha256 != expected_sha256:
        raise ValidationError(f"resume sha256 {provenance.sha256} != expected {expected_sha256}")
    return provenance


def _hash_dumped_config(run_root: Path) -> str:
    config_path = Path(run_root) / "checkpoints" / "config.json"
    if not config_path.is_file():
        raise ValidationError("dump-config did not write checkpoints/config.json")
    config = json.loads(config_path.read_text())
    normalized = normalize_config_for_transport(config)
    return sha256_bytes(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode())


def prepare_remote_source(
    *,
    volume: ReloadingVolume,
    archive_path: Path,
    expected_archive_sha256: str,
    expected_commit: str,
    expected_tree: str,
    request: RunRequest,
    run_root: Path,
    attempt_id: str,
    lock: LockLike,
    run: Callable[..., subprocess.CompletedProcess[object]] = subprocess.run,
    start_heartbeat: Callable[..., object] | None = None,
    ephemeral_parent: Path | None = None,
    parent_env: Mapping[str, str] | None = None,
    wandb_api_key: str | None = None,
    now: Callable[[], datetime] | None = None,
    remote_resume: str | Path | None = None,
    expected_resume_sha256: str | None = None,
    manifest: Manifest | None = None,
    on_ready: Callable[[PreparedSource], object] | None = None,
) -> PreparedSource:
    """Reload, verify, extract, install, dump, probe; hand off a live heartbeat.

    Reload first so Volume.reload() cannot drop an uncommitted STATUS write.
    PREPARING is committed before the heartbeat starts. Fallible project and
    C-extension work happens after that durable status exists so a failure
    can persist build_failed. Static image-build errors stay in the CLI and
    never reach this function. Success transfers heartbeat ownership to the
    caller; every exception path stops/joins first, then writes the terminal
    state under the same lock.
    """
    now_fn = now if now is not None else (lambda: datetime.now(UTC))
    run_root = Path(run_root)
    heartbeat: object | None = None
    staging: tempfile.TemporaryDirectory[str] | None = None
    if start_heartbeat is None:
        start_heartbeat = start_heartbeat_worker
    try:
        volume.reload()
        archive_path = Path(archive_path)
        if not archive_path.is_file():
            raise ValidationError(f"source archive missing after Volume.reload(): {archive_path}")
        digest = sha256_file(archive_path)
        if digest != expected_archive_sha256:
            raise ValidationError(
                f"source archive sha256 {digest} != expected {expected_archive_sha256}")
        if _read_status(run_root) is None:
            transition_status(run_root,
                              Status.PREPARING,
                              now=now_fn(),
                              attempt_id=attempt_id,
                              lock=lock)
            volume.commit()
        heartbeat = start_heartbeat(
            run_root=run_root,
            attempt_id=attempt_id,
            lock=lock,
            now=now_fn,
            commit=volume.commit,
        )
        if ephemeral_parent is not None:
            Path(ephemeral_parent).mkdir(parents=True, exist_ok=True)
        staging = tempfile.TemporaryDirectory(prefix="cs2rl-src-", dir=ephemeral_parent)
        source_dir = Path(staging.name)
        safe_extract_git_archive(archive_path, source_dir)
        _verify_extracted_provenance(source_dir, expected_commit, expected_tree)
        if manifest is not None:
            atomic_write_json(run_root / MANIFEST_FILENAME, manifest.to_dict())
            volume.commit()
        transition_status(run_root, Status.BUILDING, now=now_fn(), attempt_id=attempt_id, lock=lock)
        child_env = build_child_env(
            parent_env if parent_env is not None else os.environ,
            wandb_enabled=request.wandb_secret_name is not None,
            wandb_api_key=wandb_api_key,
        )
        cwd = os.fspath(source_dir)
        run(build_install_command(source_dir), cwd=cwd, shell=False, env=child_env, check=True)
        resume_str = os.fspath(remote_resume) if remote_resume is not None else None
        if resume_str is not None:
            _validate_remote_resume(Path(resume_str), expected_resume_sha256)
        run(
            build_dump_config_command(request, resume_str),
            cwd=cwd,
            shell=False,
            env=child_env,
            check=True,
        )
        config_hash = _hash_dumped_config(run_root)
        if manifest is not None:
            atomic_write_json(
                run_root / MANIFEST_FILENAME,
                replace(manifest, config_hash=config_hash).to_dict(),
            )
            volume.commit()
        run(build_cuda_probe_command(), cwd=cwd, shell=False, env=child_env, check=True)
        prepared = PreparedSource(
            source_dir=source_dir,
            child_env=child_env,
            train_command=build_train_command(build_train_argv(request, resume_str)),
            heartbeat=heartbeat,
            config_hash=config_hash,
            _staging=staging,
        )
        if on_ready is not None:
            on_ready(prepared)
        return prepared
    except Exception:
        try:
            _stop_heartbeat(heartbeat)
        except Exception:
            # Do not hide the original preflight error.
            pass
        current = _read_status(run_root)
        if current is not None and current.attempt_id == attempt_id:
            try:
                transition_status(run_root,
                                  Status.BUILD_FAILED,
                                  now=now_fn(),
                                  attempt_id=attempt_id,
                                  lock=lock)
                volume.commit()
            except Exception:
                # Do not hide the original preflight error.
                pass
        if staging is not None:
            staging.cleanup()
        raise


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


@dataclass(frozen=True)
class PublishOutcome:
    """Result of one publish attempt. `reason is None` iff a sidecar was written.

    The reason exists because every call site wraps this in `except Exception:
    pass`. Returning WHY a generation was skipped is the only way a skip is
    visible from outside the container.
    """

    generation: tuple[int, int] | None
    reason: str | None = None


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
        validate_local_checkpoint(ckpt)
    except ValidationError as err:
        return PublishOutcome(last_published, str(err))
    payload = {
        "sha256": sha256_file(ckpt),
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
    transition_status(run_root, Status.TRAINING, now=now(), attempt_id=attempt_id, lock=lock)
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
            transition_status(run_root, status, now=now(), attempt_id=attempt_id, lock=lock)
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
        # leave the first started but unjoinable.
        #
        # The invariant is one-directional -- every thread in tee_threads has been
        # started, but not every started thread is yet in it. A signal in the residual
        # window between building `threads` and the first append finds the list empty,
        # so the tee threads are never joined and the attempt writes nothing to
        # train.log. That is the accepted trade: the threads are daemon=True so they
        # never hold the process open, finalize's join is their only consumer, and the
        # child is being killed anyway -- whereas publishing first costs a run with no
        # terminal status at all. Do not "fix" this by moving the append back above
        # start().
        #
        # It is only survivable because _tee_stream guards each sink's write/flush with
        # `except ValueError: continue` (`:2175-2181` at this commit). Threads that
        # start inside the residual window run against sinks finalize has already
        # closed; that handler is what keeps this a no-op instead of an
        # unraised-in-thread exception. Do not delete it as dead defensive code.
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
