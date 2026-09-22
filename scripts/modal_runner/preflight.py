"""Remote source preparation before a training attempt."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
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
    FileProvenance,
    LockLike,
    Manifest,
    PreparedSource,
    ReloadingVolume,
    Status,
    ValidationError,
    sha256_bytes,
)
from .source import safe_extract_git_archive
from .state import _read_status, atomic_write_json, start_heartbeat_worker

if TYPE_CHECKING:
    from .request import RunRequest


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


# Formatter exemption for `prepare_remote_source` only. Its bytes are pinned to
# the relocation oracle (`scripts/modal_runner_lib.py` at 2bb32ac) by
# `test_relocation_combined_contract` in tests/test_modal_relocation.py, which
# allows no edit beyond the declared `core.sha256_file` and
# `state.transition_status` qualifiers. The `state.` prefix moved the open paren
# of all three `transition_status` calls six columns right: two calls keep their
# pre-move continuation indent, and the third now exceeds the column limit. yapf
# would re-indent and re-wrap them. Delete this pragma pair once that byte pin is
# retired, then let yapf format the body. The blank line after `disable` keeps
# this comment out of the pinned segment.
# yapf: disable

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
        digest = core.sha256_file(archive_path)
        if digest != expected_archive_sha256:
            raise ValidationError(
                f"source archive sha256 {digest} != expected {expected_archive_sha256}")
        if _read_status(run_root) is None:
            state.transition_status(run_root,
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
        state.transition_status(run_root, Status.BUILDING, now=now_fn(), attempt_id=attempt_id, lock=lock)
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
                state.transition_status(run_root,
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

# yapf: enable
