"""Runner request parsing and validation."""
from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .commands import _assemble_train_argv
from .core import (
    _RUN_ID_RE,
    _SECRET_NAME_RE,
    AGENTS_PER_ENV,
    ALLOWED_CPU_CORES,
    ALLOWED_GPUS,
    ALLOWED_MAPS,
    ALLOWED_NUM_ENVS,
    BPTT_HORIZON,
    DEFAULT_CPU_CORES,
    DEFAULT_GPU,
    DEFAULT_MEMORY_MIB,
    DEFAULT_NUM_ENVS,
    DEFAULT_SAVE_EVERY_SECONDS,
    DEFAULT_TIMEOUT_MINUTES,
    DEFAULT_VEC_WORKERS,
    LIVE_TRAIN_OPTION_ARITY,
    MAX_MEMORY_MIB,
    MAX_SAVE_EVERY_SECONDS,
    MAX_TIMEOUT_MINUTES,
    MIN_BATCH_SIZE,
    MIN_MEMORY_MIB,
    MIN_SAVE_EVERY_SECONDS,
    MIN_TIMEOUT_MINUTES,
    RUN_ONLY_OPTIONS,
    RUNNER_OWNED_TRAIN_FLAGS,
    Action,
    ValidationError,
)


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
