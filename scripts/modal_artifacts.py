"""Client-only Modal artifact status/download.

WHY this is not scripts/run_modal.py: importing the launch App constructs the
CUDA Image and can hydrate named objects. Observing a run must stay
read-only and must not import that module.

PITFALLS:
  * Volume.from_name(..., create_if_missing=False) only. Never objects.create.
  * Client Volume APIs take root-relative runs/... — never /artifacts/...
  * Launch-only flags are rejected rather than ignored.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import modal                                           # noqa: E402, I001
import scripts.modal_runner_lib as mrl                 # noqa: E402, I001

REPO_ROOT = _REPO_ROOT
DEFAULT_DOWNLOAD_ROOT = REPO_ROOT / "outputs" / "modal"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modal_artifacts.py")
    subparsers = parser.add_subparsers(dest="action", required=True)
    for name in ("status", "download"):
        child = subparsers.add_parser(name)
        child.add_argument("--run-id", required=True)
    return parser


def _client_path(path: PurePosixPath) -> str:
    if path.is_absolute() or any(part in {"..", ""} for part in path.parts):
        raise mrl.ValidationError(f"refusing Volume client path: {path}")
    text = path.as_posix()
    if text.startswith("/artifacts"):
        raise mrl.ValidationError(f"refusing mounted path as Volume client API: {text}")
    return text


def _read_volume_file(volume: object, remote: str) -> bytes | None:
    if remote.startswith("/artifacts"):
        raise mrl.ValidationError(f"refusing mounted path as Volume client API: {remote}")
    try:
        chunks = list(volume.read_file(remote))
    except (FileNotFoundError, OSError, KeyError):
        return None
    return b"".join(chunks)


def _lookup_volume(modal_module: object | None = None):
    modal_mod = modal if modal_module is None else modal_module
    try:
        return modal_mod.Volume.from_name(mrl.VOLUME_NAME, create_if_missing=False)
    except Exception:
        raise mrl.ValidationError("artifact volume is missing") from None


def _derive_run_view(volume: object, run_id: str, now: datetime) -> mrl.DerivedStatus:
    status_remote = _client_path(mrl.RUNS_ROOT / run_id / mrl.STATUS_FILENAME)
    reservation_remote = _client_path(mrl.RUNS_ROOT / run_id / mrl.RESERVATION_FILENAME)
    status_bytes = _read_volume_file(volume, status_remote)
    if status_bytes is not None:
        return mrl.derive_status(mrl.RunStatus.from_dict(json.loads(status_bytes)), now=now)
    reservation_bytes = _read_volume_file(volume, reservation_remote)
    if reservation_bytes is None:
        raise mrl.ValidationError(f"run not found: {run_id}")
    payload = json.loads(reservation_bytes)
    created = datetime.fromisoformat(str(payload["created_at"]))
    if now - created >= mrl.STALE_AFTER:
        return mrl.DerivedStatus(status=mrl.Status.INTERRUPTED, stale=True, reason="no-heartbeat")
    return mrl.DerivedStatus(status=mrl.Status.PREPARING, stale=False, reason="no-heartbeat")


def _checkpoint_loadable(volume: object, run_id: str) -> bool:
    sidecar_remote = _client_path(mrl.RUNS_ROOT / run_id / "checkpoints" /
                                  mrl.CHECKPOINT_SIDECAR_NAME)
    ckpt_remote = _client_path(mrl.RUNS_ROOT / run_id / "checkpoints" / mrl.CHECKPOINT_NAME)
    sidecar_a_bytes = _read_volume_file(volume, sidecar_remote)
    if sidecar_a_bytes is None:
        return False
    sidecar_a = json.loads(sidecar_a_bytes)
    ckpt_bytes = _read_volume_file(volume, ckpt_remote)
    if ckpt_bytes is None:
        return False
    if int(sidecar_a.get("size", -1)) != len(ckpt_bytes):
        return False
    if sidecar_a.get("sha256") != mrl.sha256_bytes(ckpt_bytes):
        return False
    try:
        import torch
        torch.load(io.BytesIO(ckpt_bytes), map_location="cpu", weights_only=True)
    except Exception:
        return False
    sidecar_b_bytes = _read_volume_file(volume, sidecar_remote)
    return sidecar_b_bytes is not None and json.loads(sidecar_b_bytes) == sidecar_a


def collect_status(
    run_id: str,
    *,
    modal_module: object | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Read-only status. Missing Volume fails without creating objects."""
    mrl.validate_run_id(run_id)
    volume = _lookup_volume(modal_module)
    stamp = now if now is not None else datetime.now(UTC)
    view = _derive_run_view(volume, run_id, stamp)
    return {
        "run_id": run_id,
        "status": view.status.value,
        "stale": view.stale,
        "reason": view.reason,
        "checkpoint_loadable": _checkpoint_loadable(volume, run_id),
    }


def _safe_run_relative(entry_path: str, run_id: str) -> PurePosixPath:
    relative = PurePosixPath(entry_path)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise mrl.ValidationError(f"refusing path escape: {entry_path}")
    root = mrl.RUNS_ROOT / run_id
    try:
        stripped = relative.relative_to(root)
    except ValueError as err:
        raise mrl.ValidationError(f"download path escapes run directory: {entry_path}") from err
    if any(part == ".." for part in stripped.parts):
        raise mrl.ValidationError(f"refusing path escape: {entry_path}")
    return stripped


def _write_download_file(staging: Path, relative: PurePosixPath, data: bytes) -> None:
    dest = staging.joinpath(*relative.parts)
    try:
        dest.resolve().relative_to(staging.resolve())
    except ValueError as err:
        raise mrl.ValidationError(f"download path escapes staging: {relative}") from err
    if dest.is_symlink() or dest.exists():
        raise mrl.ValidationError(f"refusing symlink or overwrite in staging: {relative}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.parent.is_symlink():
        raise mrl.ValidationError(f"refusing symlink parent in staging: {relative}")
    dest.write_bytes(data)


def download_run(
    run_id: str,
    *,
    dest_root: Path | None = None,
    modal_module: object | None = None,
) -> Path:
    """Download runs/<id>/ into dest_root/<id> via a sibling temp directory."""
    mrl.validate_run_id(run_id)
    dest_root = Path(dest_root) if dest_root is not None else DEFAULT_DOWNLOAD_ROOT
    dest = dest_root / run_id
    if dest.exists():
        raise mrl.ValidationError(f"refusing to overwrite existing download: {dest}")
    volume = _lookup_volume(modal_module)
    prefix = _client_path(mrl.RUNS_ROOT / run_id)
    try:
        entries = list(volume.iterdir(prefix, recursive=True))
    except Exception as err:
        raise mrl.ValidationError(f"run not found: {run_id}") from err
    dest_root.mkdir(parents=True, exist_ok=True)
    staging = dest_root / f".{run_id}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for entry in entries:
            entry_path = str(getattr(entry, "path", ""))
            entry_type = str(getattr(entry, "type", "file"))
            if entry_type.lower() in {"directory", "dir"}:
                continue
            relative = _safe_run_relative(entry_path, run_id)
            data = _read_volume_file(volume, _client_path(PurePosixPath(entry_path)))
            if data is None:
                raise mrl.ValidationError(f"missing download object: {entry_path}")
            _write_download_file(staging, relative, data)
        os.replace(str(staging), str(dest))
    except Exception:
        if staging.exists():
            for path in sorted(staging.rglob("*"), reverse=True):
                if path.is_dir() and not path.is_symlink():
                    path.rmdir()
                else:
                    path.unlink()
            staging.rmdir()
        raise
    return dest


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in {"-h", "--help"}:
        _parser().print_help()
        return 0
    request = mrl.parse_artifact_client_request(argv)
    if request.action is mrl.Action.STATUS:
        print(json.dumps(collect_status(request.run_id), indent=2, sort_keys=True))
    else:
        print(download_run(request.run_id))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except mrl.ValidationError as err:
        print(err, file=sys.stderr)
        raise SystemExit(2) from err
