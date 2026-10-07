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
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

_REPO_ROOT = Path(__file__).resolve().parents[1]
# A direct path launch starts with scripts/ on sys.path. Dotted imports and
# module-mode launches already have their caller's namespace path.
if __name__ == "__main__" and not __package__ and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import modal                           # noqa: E402, I001
import scripts.modal_runner as mrl     # noqa: E402, I001

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


def read_volume_file(volume: modal.Volume, remote: str) -> bytes | None:
    """Read a committed Volume object whole, or None if it is not there.

    Absent is a normal outcome for every protocol read here (no STATUS.json
    yet, no sidecar yet), so it is a return value rather than an exception;
    `derive_run_view_from_bytes` and `verify_checkpoint` both take `None` as
    meaningful input.

    PITFALLS:
      * Public on purpose. `modal_backfill_sidecar` imports it; this is the
        supported way to read a Volume object outside this module.
      * Client Volume APIs take root-relative `runs/...`. A `/artifacts/...`
        path is the *mounted* in-container spelling, so it is refused here
        rather than attempted — otherwise a wrong-API call would come back as
        an indistinguishable `None`. Callers that route through `_client_path`
        are already checked; this guard covers the ones that are not.
      * The SDK maps a missing file to FileNotFoundError before yielding bytes.
        Once any chunk arrives, a read failure is a failed transfer, not an
        absent artifact. Permission, transport and other errors propagate.
    """
    if remote.startswith("/artifacts"):
        raise mrl.ValidationError(f"refusing mounted path as Volume client API: {remote}")
    chunks: list[bytes] = []
    try:
        chunks.extend(volume.read_file(remote))
    except FileNotFoundError:
        if chunks:
            raise
        return None
    return b"".join(chunks)


def lookup_volume():
    """Look up the artifact Volume read-only, or raise if it does not exist.

    `create_if_missing=False` is the whole point: observing a run must never
    bring the Volume into existence, because a freshly created empty Volume
    would make every run look merely absent instead of making the mistake
    obvious. A missing Volume is an operator-facing ValidationError.

    PITFALL: from_name returns a lazy handle. Explicit public hydration puts
    missing-Volume translation at the lookup operation, before file APIs can
    report a missing path. Only the SDK's NotFoundError means absent here;
    permission, transport and unexpected failures retain their diagnostics.
    """
    try:
        volume = modal.Volume.from_name(mrl.VOLUME_NAME, create_if_missing=False)
        volume.hydrate()
    except modal.exception.NotFoundError as err:
        raise mrl.ValidationError("artifact volume is missing") from err
    return volume


class VolumeIndex:
    """Committed-object reads over a client Volume. Paths via `_client_path`."""

    def __init__(self, volume: modal.Volume):
        self._volume = volume

    def read_file(self, path: PurePosixPath) -> bytes | None:
        return read_volume_file(self._volume, _client_path(path))


def collect_status(
    run_id: str,
    *,
    now: datetime | None = None,
) -> dict[str, object]:
    """Read-only status. Missing Volume fails without creating objects.

    All judgement lives in the scripts/modal_runner package: this function only
    fetches bytes and hands them to `derive_run_view_from_bytes` (defined in
    scripts/modal_runner/state.py) / `verify_checkpoint` (defined in
    scripts/modal_runner/checkpoint.py), so the status client, the launch
    validator and the sidecar backfiller cannot drift apart on staleness or on
    what makes a checkpoint trustworthy.

    PITFALLS:
      * Absent STATUS.json *and* absent reservation.json means the run does not
        exist. `derive_run_view_from_bytes` refuses that case too, but with the
        generic "no STATUS.json or reservation.json"; raising here first is what
        puts the run id in the message. The exact string is pinned by
        test_collect_status_missing_run_message.
      * The sidecar is read twice on purpose. `verify_checkpoint` compares the
        two reads to catch a sidecar being rewritten underneath us mid-status.
    """
    mrl.validate_run_id(run_id)
    volume = lookup_volume()
    stamp = now if now is not None else datetime.now(UTC)
    index = VolumeIndex(volume)
    status_bytes = index.read_file(mrl.RUNS_ROOT / run_id / mrl.STATUS_FILENAME)
    reservation_bytes = index.read_file(mrl.RUNS_ROOT / run_id / mrl.RESERVATION_FILENAME)
    if status_bytes is None and reservation_bytes is None:
        raise mrl.ValidationError(f"run not found: {run_id}")
    view = mrl.derive_run_view_from_bytes(status_bytes, reservation_bytes, now=stamp)
    sidecar_path = mrl.RUNS_ROOT / run_id / "checkpoints" / mrl.CHECKPOINT_SIDECAR_NAME
    ckpt_path = mrl.RUNS_ROOT / run_id / "checkpoints" / mrl.CHECKPOINT_NAME
    sidecar_bytes = index.read_file(sidecar_path)
    ckpt_bytes = index.read_file(ckpt_path)
    reread_bytes = index.read_file(sidecar_path)
    verdict = mrl.verify_checkpoint(sidecar_bytes, ckpt_bytes, reread_bytes)
    return {
        "run_id": run_id,
        "status": view.status.value,
        "stale": view.stale,
        "reason": view.reason,
        "checkpoint_loadable": verdict.ok,
    }


def _entry_type_name(entry: object) -> str:
    entry_type = getattr(entry, "type", "file")
    name = getattr(entry_type, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(entry_type, int) and not isinstance(entry_type, bool):
        return {1: "FILE", 2: "DIRECTORY", 3: "SYMLINK"}.get(int(entry_type), str(entry_type))
    text = str(entry_type)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def _skip_download_entry(entry: object) -> bool:
    # FileEntry.type is FileEntryType IntEnum; str(type) is "fileentrytype.directory".
    entry_type = getattr(entry, "type", "file")
    name = getattr(entry_type, "name", None)
    kind = _entry_type_name(entry)
    if name in {"DIRECTORY", "directory", "dir"} or kind in {"DIRECTORY", "directory", "dir"}:
        return True
    if isinstance(entry_type, int) and not isinstance(entry_type, bool) and int(entry_type) == 2:
        return True
    if name in {"SYMLINK", "symlink"} or kind in {"SYMLINK", "symlink"}:
        raise mrl.ValidationError(f"refusing volume symlink: {getattr(entry, 'path', '')}")
    if isinstance(entry_type, int) and not isinstance(entry_type, bool) and int(entry_type) == 3:
        raise mrl.ValidationError(f"refusing volume symlink: {getattr(entry, 'path', '')}")
    return False


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
) -> Path:
    """Download runs/<id>/ into dest_root/<id> via a sibling temp directory."""
    mrl.validate_run_id(run_id)
    dest_root = Path(dest_root) if dest_root is not None else DEFAULT_DOWNLOAD_ROOT
    dest = dest_root / run_id
    if dest.exists():
        raise mrl.ValidationError(f"refusing to overwrite existing download: {dest}")
    volume = lookup_volume()
    prefix = _client_path(mrl.RUNS_ROOT / run_id)
    entries: list[object] = []
    try:
        entries.extend(volume.iterdir(prefix, recursive=True))
    except modal.exception.NotFoundError as err:
        # A missing path before any entries is an absent run; partial listing
        # failure must keep its original diagnosis and never publish a download.
        if entries:
            raise
        raise mrl.ValidationError(f"run not found: {run_id}") from err
    dest_root.mkdir(parents=True, exist_ok=True)
    staging = dest_root / f".{run_id}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for entry in entries:
            if _skip_download_entry(entry):
                continue
            entry_path = str(getattr(entry, "path", ""))
            relative = _safe_run_relative(entry_path, run_id)
            data = read_volume_file(volume, _client_path(PurePosixPath(entry_path)))
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
