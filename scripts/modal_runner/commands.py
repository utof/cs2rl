"""Training commands and child process environments."""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from . import core
from .core import (
    _PRESERVED_CHILD_ENV_KEYS,
    _THREAD_CAP_ENV,
    CUDA_PROBE_SOURCE,
    RUNS_ROOT,
    TRAIN_SCRIPT,
    UV_BIN,
    ValidationError,
    mounted_path,
)

if TYPE_CHECKING:
    from .request import RunRequest


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
        core.PREBUILT_PYTHON,
        "--no-deps",
        "--no-build-isolation",
        os.fspath(source_dir),
    ]


def build_train_command(argv: Sequence[str]) -> list[str]:
    """Prebuilt interpreter + live script + already-split argv. Never a shell."""
    return [core.PREBUILT_PYTHON, TRAIN_SCRIPT, *argv]


def build_dump_config_command(request: RunRequest, remote_resume: str | None) -> list[str]:
    """Cheap --dump-config invocation with the same owned flags as training."""
    return build_train_command(build_dump_config_argv(request, remote_resume))


def build_cuda_probe_command() -> list[str]:
    """Prebuilt interpreter running the CUDA/PufferLib advantage probe string."""
    return [core.PREBUILT_PYTHON, "-c", CUDA_PROBE_SOURCE]
