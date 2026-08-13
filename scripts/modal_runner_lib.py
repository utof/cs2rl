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
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

VOLUME_NAME = "cs2rl-training-artifacts"
REGISTRY_NAME = "cs2rl-training-run-registry"
VOLUME_MOUNT = Path("/artifacts")
SOURCES_ROOT = PurePosixPath("sources")
INPUTS_ROOT = PurePosixPath("inputs")
RUNS_ROOT = PurePosixPath("runs")

ALLOWED_MAPS = frozenset({"simple", "dust2"})
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
    "--smoke": 0,
    "--train": 0,
    "--record": 0,
    "--eval": 0,
    "--checkpoint": 1,
    "--resume": 1,
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
    "--eval-policy": 1,
    "--name": 1,
    "--wandb": 0,
    "--wandb-project": 1,
    "--wandb-entity": 1,
    "--no-self-play": 0,
    "--no-dead-run-abort": 0,
    "--warmstart-entropy": 0,
    "--warmstart-grace-steps": 1,
    "--warmstart-ramp-steps": 1,
    "--warmstart-alpha-ceiling": 1,
    "--reward-symmetrize": 0,
    "--tag-diagnostic": 0,
    "--tag-every": 1,
    "--tct-split-heads": 0,
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
    "--name",
    "--checkpoint-dir",
    "--checkpoint_dir",
    "--device",
    "--save_every_sec",
    "--vec-backend",
    "--vec-num-workers",
    "--dump-config",
    "--smoke",
    "--record",
    "--eval",
    "--dust2",
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
    return tuple(shlex.split(raw))


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
    train_args: str | Sequence[str] = "--timesteps 30000000 --seed 2",
    wandb_secret_name: str | None = None,
    resume_local_checkpoint: str | Path | None = None,
    resume_run_id: str | None = None,
) -> RunRequest:
    """Validate structured launch fields and return a frozen request.

    effective_map is required (no default in the signature on purpose). Resource
    defaults match the design: T4 / 8 CPU / 16384 MiB / 256 envs / 8 workers /
    120 min / 300 s save.
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


def _assemble_train_argv(
    request: RunRequest,
    run_root: Path,
    remote_resume: str | None,
    *,
    dump_config: bool,
) -> list[str]:
    """Build the live argv. Never a shell string; every token is already split.

    Order is part of the contract (tests pin it): mode flag, optional --dust2,
    the already-validated user train-args, then each runner-owned flag exactly
    once. --resume is appended only when the caller supplies a remote path —
    the runner owns that flag, so a user --resume never reaches this function.
    """
    argv: list[str] = ["--dump-config"] if dump_config else ["--train"]
    if request.effective_map == "dust2":
        argv.append("--dust2")
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
