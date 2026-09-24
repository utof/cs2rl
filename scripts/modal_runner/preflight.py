"""Remote source preparation before a training attempt."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from . import checkpoint, core, state
from .checkpoint import normalize_config_for_transport
from .commands import (
    build_child_env,
    build_cuda_probe_command,
    build_dump_config_command,
    build_install_command,
    build_train_argv,
    build_train_command,
)
from .core import (
    MANIFEST_FILENAME,
    PROVENANCE_NAME,
    AttemptContext,
    FileProvenance,
    Manifest,
    PreparedSource,
    Status,
    ValidationError,
    sha256_bytes,
)
from .source import safe_extract_git_archive
from .state import atomic_write_json, read_status, start_heartbeat_worker, stop_heartbeat

if TYPE_CHECKING:
    from .request import RunRequest


@dataclass(frozen=True)
class ExpectedSource:
    """The source bundle prepare must find on the Volume, and the provenance it must carry.

    `archive_sha256` is checked after `Volume.reload()`; `commit` and `tree`
    are checked against the extracted bundle's provenance sidecar. One value
    because they are one fact, the client's bundle, and are checked together.
    """

    archive_path: Path
    archive_sha256: str
    commit: str
    tree: str


@dataclass(frozen=True)
class RemoteResume:
    """The checkpoint an attempt resumes from, on the Volume mount, and its pinned hash.

    One value because `sha256` means nothing without `path`; no resume is
    `None`, not a RemoteResume with an empty path. PITFALL:
    `_validate_remote_resume(path, expected_sha256)` keeps its positional
    signature (the binding campaign's prepare-validator consumer and
    `test_patch_target_dichotomy` call it that way), so the caller unpacks
    this; never pass the object in.
    """

    path: Path
    sha256: str | None = None


@dataclass(frozen=True)
class PreflightHost:
    """What prepare takes from the container. Production uses the defaults.

    `run` runs the install, dump-config and CUDA-probe commands. `parent_env`
    is what the child environment is built from (None: `os.environ`, read
    when prepare runs). `ephemeral_parent` is where the source is extracted
    (None: the system temp dir). `start_heartbeat` starts the preflight
    heartbeat (None: `state.start_heartbeat_worker`, resolved when prepare
    runs). Tests replace these; production never passes a PreflightHost.

    PITFALL, when defaults are resolved (gh#163 spec §4.2): `run` is bound to
    `subprocess.run` at import, exactly as prepare's own `run` default is, so
    a monkeypatch of the global `subprocess.run` does not reach it; inject
    `run` instead. The three None defaults are resolved at call time.
    """

    run: Callable[..., subprocess.CompletedProcess[object]] = subprocess.run
    parent_env: Mapping[str, str] | None = None
    ephemeral_parent: Path | None = None
    start_heartbeat: Callable[..., object] | None = None


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


def _validate_remote_resume(path: Path, expected_sha256: str | None) -> FileProvenance:
    """Weights-only load the mounted checkpoint and pin its content hash."""
    provenance = checkpoint.validate_local_checkpoint(path)
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


def _verify_archive_then_enter_preparing(attempt: AttemptContext, source: ExpectedSource) -> None:
    """Phase 1: reload the Volume, verify the uploaded archive, and enter PREPARING.

    Reload FIRST: Volume.reload() replaces the mount, so a STATUS written
    before it could be dropped, and the archive appears on the mount only
    after it. The archive is then trusted by its hash alone. PREPARING is
    written only when no STATUS exists yet, and committed before this returns,
    so it is durable before the orchestrator starts the heartbeat.
    """
    attempt.volume.reload()
    archive_path = source.archive_path
    if not archive_path.is_file():
        raise ValidationError(f"source archive missing after Volume.reload(): {archive_path}")
    digest = core.sha256_file(archive_path)
    if digest != source.archive_sha256:
        raise ValidationError(f"source archive sha256 {digest} != expected {source.archive_sha256}")
    if read_status(attempt.run_root) is None:
        state.transition_status(attempt.run_root,
                                Status.PREPARING,
                                now=attempt.clock.now(),
                                attempt_id=attempt.attempt_id,
                                lock=attempt.lock)
        attempt.volume.commit()


def _extract_verified_source(attempt: AttemptContext, source: ExpectedSource, source_dir: Path,
                             manifest: Manifest | None) -> None:
    """Phase 3: extract the archive into `source_dir`, check its provenance, write the manifest.

    PITFALL: `source_dir` is the orchestrator's staging directory, and the
    orchestrator creates it, not this phase. If this phase created it and then
    raised (a provenance mismatch), the orchestrator would never receive it,
    the failure arm could not remove it, and the extracted tree would survive
    until a finalizer ran (gh#163 spec §4.5, the resource-acquisition rule).
    """
    safe_extract_git_archive(source.archive_path, source_dir)
    _verify_extracted_provenance(source_dir, source.commit, source.tree)
    if manifest is not None:
        atomic_write_json(attempt.run_root / MANIFEST_FILENAME, manifest.to_dict())
        attempt.volume.commit()


def _build_in_source(
    attempt: AttemptContext,
    request: RunRequest,
    source_dir: Path,
    *,
    resume: RemoteResume | None,
    manifest: Manifest | None,
    wandb_api_key: str | None,
    host: PreflightHost,
) -> tuple[dict[str, str], str | None, str]:
    """Phase 4, BUILDING: install, validate the resume, dump and hash the config, then probe.

    Returns `(child_env, resume_str, config_hash)` for the PreparedSource the
    orchestrator builds. The order is the contract: the install runs before
    the resume is validated, the resume is validated before the cheap
    dump-config run, the dumped config is hashed and the manifest rewritten
    with that hash, and the CUDA probe runs last. `host.parent_env=None` reads
    `os.environ` here, at call time. `resume` is unpacked into
    `_validate_remote_resume`'s positional arguments, never passed whole (its
    PITFALL, on RemoteResume).
    """
    state.transition_status(attempt.run_root,
                            Status.BUILDING,
                            now=attempt.clock.now(),
                            attempt_id=attempt.attempt_id,
                            lock=attempt.lock)
    child_env = build_child_env(
        host.parent_env if host.parent_env is not None else os.environ,
        wandb_enabled=request.wandb_secret_name is not None,
        wandb_api_key=wandb_api_key,
    )
    cwd = os.fspath(source_dir)
    host.run(build_install_command(source_dir), cwd=cwd, shell=False, env=child_env, check=True)
    resume_str = os.fspath(resume.path) if resume is not None else None
    if resume is not None:
        _validate_remote_resume(resume.path, resume.sha256)
    host.run(
        build_dump_config_command(request, resume_str),
        cwd=cwd,
        shell=False,
        env=child_env,
        check=True,
    )
    config_hash = _hash_dumped_config(attempt.run_root)
    if manifest is not None:
        atomic_write_json(
            attempt.run_root / MANIFEST_FILENAME,
            replace(manifest, config_hash=config_hash).to_dict(),
        )
        attempt.volume.commit()
    host.run(build_cuda_probe_command(), cwd=cwd, shell=False, env=child_env, check=True)
    return child_env, resume_str, config_hash


def _fail_preflight(attempt: AttemptContext, heartbeat: object | None,
                    staging: tempfile.TemporaryDirectory[str] | None) -> None:
    """The failure arm: stop the heartbeat, write BUILD_FAILED, remove the extracted source.

    In this order, which is the pre-W5 arm's: the heartbeat is stopped and
    joined first, then BUILD_FAILED is written under the attempt's lock and
    committed, but only while this attempt still owns STATUS, then the staging
    directory is cleaned up. `heartbeat` and `staging` are None when the
    failure came before the orchestrator acquired them. The caller re-raises
    the original error afterwards, so the heartbeat stop and the BUILD_FAILED
    write each swallow their own exception rather than replace it.
    """
    try:
        stop_heartbeat(heartbeat)
    except Exception:
        # Do not hide the original preflight error.
        pass
    current = read_status(attempt.run_root)
    if current is not None and current.attempt_id == attempt.attempt_id:
        try:
            state.transition_status(attempt.run_root,
                                    Status.BUILD_FAILED,
                                    now=attempt.clock.now(),
                                    attempt_id=attempt.attempt_id,
                                    lock=attempt.lock)
            attempt.volume.commit()
        except Exception:
            # Do not hide the original preflight error.
            pass
    if staging is not None:
        staging.cleanup()


def prepare_remote_source(
    *,
    attempt: AttemptContext,
    request: RunRequest,
    source: ExpectedSource,
    resume: RemoteResume | None = None,
    manifest: Manifest | None = None,
    wandb_api_key: str | None = None,
    host: PreflightHost | None = None,
) -> PreparedSource:
    """Reload, verify, extract, install, dump, probe; hand off a live heartbeat.

    Reload first so Volume.reload() cannot drop an uncommitted STATUS write.
    PREPARING is committed before the heartbeat starts. Fallible project and
    C-extension work happens after that durable status exists so a failure
    can persist build_failed. Static image-build errors stay in the CLI and
    never reach this function. Success transfers heartbeat ownership to the
    caller; every exception path stops/joins first, then writes the terminal
    state under the same lock.

    The body is the phase sequence and nothing more: phase 1 verifies the
    archive and enters PREPARING; phase 2, here, starts the heartbeat; phase 3
    extracts and verifies the source and writes the manifest; phase 4 is
    BUILDING; phase 5, here, builds the PreparedSource. `_fail_preflight` is
    the failure arm. `attempt` is the one AttemptContext production also hands
    to training (same lock, run_root and Volume); `host=None` is production's
    `PreflightHost()`.

    PITFALL, the resource-acquisition rule (gh#163 spec §4.5): every
    statement that acquires something the failure arm must release,
    `start_heartbeat(...)` and `tempfile.TemporaryDirectory(...)`, is assigned
    HERE, in this function's own scope, before any phase that can fail uses
    it. Moving either into a phase function makes a failure in that phase
    leak it: the arm sees only what this scope holds. Knock-out (j) of the
    spec pins the staging half.

    `on_ready` was removed in W5: nothing in production passed it.
    """
    host = PreflightHost() if host is None else host
    start_heartbeat = (start_heartbeat_worker
                       if host.start_heartbeat is None else host.start_heartbeat)
    heartbeat: object | None = None
    staging: tempfile.TemporaryDirectory[str] | None = None
    try:
        _verify_archive_then_enter_preparing(attempt, source)
        heartbeat = start_heartbeat(
            run_root=attempt.run_root,
            attempt_id=attempt.attempt_id,
            lock=attempt.lock,
            now=attempt.clock.now,
            commit=attempt.volume.commit,
        )
        if host.ephemeral_parent is not None:
            host.ephemeral_parent.mkdir(parents=True, exist_ok=True)
        staging = tempfile.TemporaryDirectory(prefix="cs2rl-src-", dir=host.ephemeral_parent)
        source_dir = Path(staging.name)
        _extract_verified_source(attempt, source, source_dir, manifest)
        child_env, resume_str, config_hash = _build_in_source(attempt,
                                                              request,
                                                              source_dir,
                                                              resume=resume,
                                                              manifest=manifest,
                                                              wandb_api_key=wandb_api_key,
                                                              host=host)
        return PreparedSource(
            source_dir=source_dir,
            child_env=child_env,
            train_command=build_train_command(build_train_argv(request, resume_str)),
            heartbeat=heartbeat,
            config_hash=config_hash,
            _staging=staging,
        )
    except Exception:
        _fail_preflight(attempt, heartbeat, staging)
        raise
