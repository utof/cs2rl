"""Checkpoint loading, validation, and completion evidence."""
from __future__ import annotations

import io
import json
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

from . import core
from .core import (
    _PREBUILT_LOAD_SOURCE,
    INPUTS_ROOT,
    PREBUILT_LOAD_TIMEOUT_SECONDS,
    CheckpointVerdict,
    CompletionEvidence,
    FileProvenance,
    Manifest,
    ValidationError,
    mounted_path,
    sha256_bytes,
)
from .state import _load_volume_json


def _import_torch() -> object:
    """Import torch, or raise ImportError. A seam, not a convenience wrapper.

    Tests monkeypatch this to reproduce the container runner's torch-less
    interpreter; without the seam the whole prebuilt fallback below is
    unreachable from a laptop, which is exactly how it stayed broken.
    """
    import torch
    return torch


# Formatter exemption for `_assert_weights_only_loadable` only. Its bytes are
# pinned to the relocation oracle (`scripts/modal_runner_lib.py` at 2bb32ac) by
# `test_relocation_combined_contract` in tests/test_modal_relocation.py, which
# allows no edit beyond the declared `core.PREBUILT_PYTHON` qualifiers. One
# qualified line now exceeds the column limit and yapf would re-wrap it. Delete
# this pragma pair once that byte pin is retired, then let yapf format the body.
# The blank line after `disable` keeps this comment out of the pinned segment.
# yapf: disable

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
    if not Path(core.PREBUILT_PYTHON).is_file():
        raise ValidationError(f"cannot validate {path}: torch is not importable and the prebuilt "
                              f"interpreter {core.PREBUILT_PYTHON} does not exist")
    try:
        completed = subprocess.run(
            [core.PREBUILT_PYTHON, "-c", _PREBUILT_LOAD_SOURCE,
             os.fspath(path)],
            capture_output=True,
            text=True,
            timeout=PREBUILT_LOAD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as err:
        raise ValidationError(f"cannot validate {path}: {core.PREBUILT_PYTHON} did not finish within "
                              f"{PREBUILT_LOAD_TIMEOUT_SECONDS}s") from err
    except OSError as err:
        raise ValidationError(f"cannot validate {path}: {core.PREBUILT_PYTHON} failed to run: "
                              f"{err}") from err
    if completed.returncode != 0:
        raise ValidationError(f"checkpoint is not weights-only loadable: {path}: "
                              f"{completed.stderr.strip()[-400:]}")

# yapf: enable


def validate_local_checkpoint(path: Path) -> FileProvenance:
    """Weights-only load, hash, and map a local checkpoint to the Volume path.

    Torch is never imported at module scope: that would pull CUDA/pynvml into
    every status/download invocation and into the import-boundary tests.
    """
    path = Path(path)
    if not path.is_file():
        raise ValidationError(f"checkpoint is not a readable file: {path}")
    _assert_weights_only_loadable(path)
    digest = core.sha256_file(path)
    client_path = INPUTS_ROOT / "sha256" / f"{digest}.pt"
    return FileProvenance(
        sha256=digest,
        size=path.stat().st_size,
        client_path=client_path,
        mount_path=mounted_path(client_path),
    )


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


def normalize_config_for_transport(config: Mapping[str, object]) -> dict[str, object]:
    """Drop only the run-local checkpoint data_dir; everything else must match.

    data_dir is the one path the trainer rewrites to the mounted run directory.
    Any other drift is a real config mismatch and must fail completion.
    """
    return {key: value for key, value in config.items() if key != "data_dir"}


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
        checkpoint_sha256=core.sha256_file(ckpt),
        config_hash=config_hash,
    )
