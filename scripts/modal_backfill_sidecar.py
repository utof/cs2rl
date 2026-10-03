"""Client-only backfill of a run's checkpoint sidecar.

WHY this exists: --resume-run-id refuses any parent without
checkpoints/dust2_policy.pt.meta.json, and only the training container writes
one. If that container dies without reaching finalize — hard preemption, OOM
kill, node loss, or (before ae7dd7b) the torch-less-runner bug — the run is left
with a perfectly good checkpoint that can never be resumed, because the process
that could vouch for it no longer exists.

This tool vouches for it from the laptop instead. The checks are the same ones
the client already performs when validating a resume parent: the run must be
terminal, the checkpoint must load weights-only, and the sidecar must round-trip
unchanged. It writes exactly one small JSON file.

WHY not scripts/modal_artifacts.py: that module is documented read-only, and
observing a run must never be able to mutate it. WHY not scripts/run_modal.py:
importing the launch App constructs the CUDA Image.

PITFALLS:
  * Never overwrites. A sidecar that already exists is authoritative even if it
    disagrees with the checkpoint — a mismatch means someone else is writing,
    which is exactly when a second writer is most harmful.
  * mtime_ns is null, not fabricated. A client cannot observe the container's
    nanosecond mtime; nothing reads the field, and a plausible-looking guess
    would be indistinguishable from a real one later.
  * Refuses live runs. A run still writing dust2_policy.pt every epoch would
    get a sidecar describing a generation that no longer exists.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Bootstrap only the documented direct path launch; importing the client must
# leave the caller's namespace path untouched.
if __name__ == "__main__" and not __package__ and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.modal_artifacts as arts                 # noqa: E402, I001
import scripts.modal_runner as mrl                     # noqa: E402, I001


def _sidecar_payload(ckpt_bytes: bytes, *, now: datetime) -> dict[str, object]:
    """Same shape publish_stable_checkpoint writes, with honest provenance."""
    return {
        "sha256": mrl.sha256_bytes(ckpt_bytes),
        "size": len(ckpt_bytes),
        "mtime_ns": None,
        "validated_at": now.isoformat(),
        "backfilled": True,
    }


def backfill_sidecar(
    run_id: str,
    *,
    modal_module: object | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Publish a sidecar for a terminal run whose container never wrote one.

    Raises ValidationError and uploads nothing unless every guard passes.
    """
    mrl.validate_run_id(run_id)
    volume = arts.lookup_volume(modal_module)
    stamp = now if now is not None else datetime.now(UTC)

    # The still-active gate is the shared bytes-in protocol, not a private
    # helper borrowed from the read-only status client: one derivation rule for
    # "is this run finished?" means backfill and `modal_artifacts status` can
    # never disagree about STALE_AFTER or about which statuses are terminal.
    # WHY the both-absent check is here and not left to the protocol:
    # derive_run_view_from_bytes refuses this case too, but with the generic
    # "no STATUS.json or reservation.json", which never names the run. Raising
    # first attaches the run id, so a typo'd --run-id reads as a typo rather
    # than as a damaged volume. Keep it above every read that follows.
    status_remote = (mrl.RUNS_ROOT / run_id / mrl.STATUS_FILENAME).as_posix()
    reservation_remote = (mrl.RUNS_ROOT / run_id / mrl.RESERVATION_FILENAME).as_posix()
    status_bytes = arts.read_volume_file(volume, status_remote)
    reservation_bytes = arts.read_volume_file(volume, reservation_remote)
    if status_bytes is None and reservation_bytes is None:
        raise mrl.ValidationError(f"run not found: {run_id}")
    view = mrl.derive_run_view_from_bytes(status_bytes, reservation_bytes, now=stamp)
    if view.status not in mrl.TERMINAL_STATUSES and not view.stale:
        raise mrl.ValidationError(
            f"run {run_id} is still active ({view.status.value}); refusing to backfill")

    # Remote paths are formed here, from mrl.RUNS_ROOT, rather than through
    # modal_artifacts' path sanitiser: every component is either a module
    # constant or a run id that validate_run_id already rejected if it could
    # contain a separator, so nothing here can escape runs/<id>/. read_volume_file
    # still refuses a mounted /artifacts path, which is the leak that matters.
    sidecar_remote = (mrl.RUNS_ROOT / run_id / "checkpoints" /
                      mrl.CHECKPOINT_SIDECAR_NAME).as_posix()
    if arts.read_volume_file(volume, sidecar_remote) is not None:
        raise mrl.ValidationError(f"run {run_id} already has a checkpoint sidecar")

    ckpt_remote = (mrl.RUNS_ROOT / run_id / "checkpoints" / mrl.CHECKPOINT_NAME).as_posix()
    ckpt_bytes = arts.read_volume_file(volume, ckpt_remote)
    if ckpt_bytes is None:
        raise mrl.ValidationError(f"run {run_id} has no {mrl.CHECKPOINT_NAME} to vouch for")
    try:
        import torch
        torch.load(io.BytesIO(ckpt_bytes), map_location="cpu", weights_only=True)
    except ImportError as err:
        raise mrl.ValidationError("torch is required to backfill a sidecar") from err
    except Exception as err:
        raise mrl.ValidationError(f"run {run_id} checkpoint is not weights-only loadable") from err

    payload = _sidecar_payload(ckpt_bytes, now=stamp)
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    try:
        with volume.batch_upload(force=False) as batch:
            batch.put_file(io.BytesIO(body), sidecar_remote)
    except FileExistsError as err:
        raise mrl.ValidationError(f"run {run_id} gained a sidecar during backfill") from err

    written = arts.read_volume_file(volume, sidecar_remote)
    if written is None or mrl.load_volume_json(written) != payload:
        raise mrl.ValidationError(f"run {run_id} sidecar did not round-trip after upload")
    return {
        "run_id": run_id,
        "sidecar": sidecar_remote,
        "sha256": payload["sha256"],
        "size": payload["size"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="modal_backfill_sidecar.py",
        description="Publish a checkpoint sidecar for a terminal run that never got one.")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    report = backfill_sidecar(args.run_id)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
