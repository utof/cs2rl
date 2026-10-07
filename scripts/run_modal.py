"""Thin Modal launch App. status/download live in scripts/modal_artifacts.py.

WHY this file exists separately from the artifact client: importing this module
constructs the CUDA Image and registers the App. Observing a run must never
pay that cost or create named objects. All request validation lives in
scripts/modal_runner/ so this module stays an adapter.

PITFALLS:
  * The base Function must not declare a named Volume/Dict. First-run App
    hydration happens before those objects exist; attach the Volume only
    through Function.with_options after objects.create.
  * include_source=False on App and Function. Training source arrives as the
    content-addressed archive, not Modal automount.
  * Never call the undecorated Function. GPU/CPU/memory/Volume/Secret belong
    only on the configured variant.
  * Launch via with_options(...).spawn(), never .remote(). SYNC remote inputs
    are cancelled when the local client dies, even under `modal run --detach`.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import uuid
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

import modal

import scripts.modal_runner as mrl

_REPO_ROOT = Path(__file__).resolve().parents[1]
CUDA_IMAGE = ("nvidia/cuda:12.8.1-devel-ubuntu22.04@"
              "sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7")
IMAGE_DIGEST = "sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7"
PUFFERLIB_SDIST = (
    "https://files.pythonhosted.org/packages/7c/e1/5292f9b69c6263707b40ba04a87e6b9bcc177281d31092f77afd90c412f1/"
    "pufferlib-3.0.0.tar.gz#sha256=7df3a3e3f5f894d78d2a1f5374097890aec01473183e748abefe4f3faa10eaa9"
)

dependency_image = (modal.Image.from_registry(
    CUDA_IMAGE,
    add_python="3.12",
).apt_install("git", "build-essential").pip_install("uv==0.11.1", "ziglang==0.14.1").env({
    "TORCH_CUDA_ARCH_LIST":
    "7.5;8.6;8.9",
    "NO_OCEAN":
    "1",
}).uv_sync(
    str(_REPO_ROOT),
    frozen=True,
    uv_version="0.11.1",
    extra_options="--no-default-groups --no-install-package pufferlib",
).run_commands(
    "mkdir -p /opt/cs2rl && ln -s /.uv/.venv /opt/cs2rl/.venv",
    "/opt/cs2rl/.venv/bin/python -c \"import importlib.metadata as m, json, pathlib; "
    "inventory={d.metadata['Name'].lower():d.version for d in m.distributions()}; "
    "pathlib.Path('/tmp/locked-inventory.json').write_text("
    "json.dumps(inventory, sort_keys=True))\"",
    "uv pip install --python /opt/cs2rl/.venv/bin/python --no-deps "
    "setuptools==82.0.1 wheel==0.48.0 Cython==3.2.9 ziglang==0.14.1",
    "python3 -c \""
    "import hashlib, tarfile, urllib.request; "
    "from pathlib import Path; "
    f"spec={PUFFERLIB_SDIST!r}; "
    "url, _, digest = spec.partition('#sha256='); "
    "archive = Path('/tmp/pufferlib-3.0.0.tar.gz'); "
    "urllib.request.urlretrieve(url, archive); "
    "got = hashlib.sha256(archive.read_bytes()).hexdigest(); "
    "assert got == digest, got; "
    "tf = tarfile.open(archive); "
    "tf.extractall('/tmp', filter='data'); "
    "tf.close(); "
    "p = Path('/tmp/pufferlib-3.0.0/setup.py'); "
    "text = p.read_text(); "
    "old = 'c_extensions = []'; "
    "assert old in text; "
    "p.write_text(text.replace(old, old + chr(10) + 'c_extension_paths = []', 1))"
    "\" && "
    "CC=gcc CXX=g++ uv pip install --python /opt/cs2rl/.venv/bin/python "
    "--no-build-isolation --no-deps --no-binary pufferlib "
    "/tmp/pufferlib-3.0.0",
    "/opt/cs2rl/.venv/bin/python -c \"import importlib.metadata as m, json, pathlib; "
    "base=json.loads(pathlib.Path('/tmp/locked-inventory.json').read_text()); "
    "now={d.metadata['Name'].lower():d.version for d in m.distributions()}; "
    "allowed={'setuptools','wheel','cython','ziglang','pufferlib'}; "
    "assert {k:v for k,v in now.items() if k not in allowed} == "
    "{k:v for k,v in base.items() if k not in allowed}; "
    "assert set(now) <= set(base) | allowed; "
    "expected={'torch':'2.10.0','numpy':'2.4.3','pufferlib':'3.0.0',"
    "'setuptools':'82.0.1','wheel':'0.48.0','Cython':'3.2.9',"
    "'ziglang':'0.14.1'}; actual={k:m.version(k) for k in expected}; "
    "assert actual == expected, actual\"",
    "/opt/cs2rl/.venv/bin/python -c \"import importlib.util, pathlib, subprocess, "
    "sysconfig, torch; "
    "import pufferlib._C; so=importlib.util.find_spec('pufferlib._C').origin; "
    "header=pathlib.Path(sysconfig.get_paths()['include'])/'Python.h'; "
    "assert header.is_file(), header; "
    "nvcc=subprocess.check_output(['nvcc','--version'], text=True); "
    "assert 'release 12.8' in nvcc, nvcc; "
    "assert hasattr(torch.ops.pufferlib,'compute_puff_advantage'); "
    "elf=subprocess.check_output(['cuobjdump','--list-elf',so], text=True); "
    "assert all('sm_'+arch in elf for arch in ('75','86','89')), elf\"",
))
runner_image = dependency_image
for _runner_module in sorted((_REPO_ROOT / "scripts" / "modal_runner").glob("*.py")):
    runner_image = runner_image.add_local_file(
        str(_runner_module),
        f"/opt/app/scripts/modal_runner/{_runner_module.name}",
        copy=True,
    )
# Modal imports "run_modal"; this file imports scripts.modal_runner.
runner_image = runner_image.add_local_file(
    str(_REPO_ROOT / "scripts" / "run_modal.py"),
    "/opt/app/scripts/run_modal.py",
    copy=True,
).env({
    "PYTHONPATH": "/opt/app:/opt/app/scripts",
})

app = modal.App("cs2rl-training", include_source=False)


def resolve_launch_request(
    *,
    action: str | None = None,
    run_id: str | None = None,
    git_sha: str | None = None,
    map: str | None = None,
    gpu: str | None = None,
    cpu_cores: int | None = None,
    memory_mib: int | None = None,
    num_envs: int | None = None,
    vec_workers: int | None = None,
    timeout_minutes: int | None = None,
    save_every_seconds: int | None = None,
    train_args: str | None = None,
    resume_local_checkpoint: str | None = None,
    resume_run_id: str | None = None,
    wandb_secret_name: str | None = None,
) -> mrl.RunRequest:
    """Apply omitted-sentinel defaults, then validate. map has no default."""
    if action != mrl.Action.RUN.value:
        raise mrl.ValidationError("the only App action is run")
    if map is None:
        raise mrl.ValidationError(f"map is required and must be one of {sorted(mrl.ALLOWED_MAPS)}")
    if run_id is None:
        raise mrl.ValidationError("--run-id is required")
    if git_sha is None:
        raise mrl.ValidationError("--git-sha is required")
    if train_args is None:
        raise mrl.ValidationError("--train-args is required")
    return mrl.build_run_request(
        run_id=run_id,
        git_sha=git_sha,
        effective_map=map,
        gpu=mrl.DEFAULT_GPU if gpu is None else gpu,
        cpu_cores=mrl.DEFAULT_CPU_CORES if cpu_cores is None else cpu_cores,
        memory_mib=mrl.DEFAULT_MEMORY_MIB if memory_mib is None else memory_mib,
        num_envs=mrl.DEFAULT_NUM_ENVS if num_envs is None else num_envs,
        vec_workers=mrl.DEFAULT_VEC_WORKERS if vec_workers is None else vec_workers,
        timeout_minutes=mrl.DEFAULT_TIMEOUT_MINUTES if timeout_minutes is None else timeout_minutes,
        save_every_seconds=(mrl.DEFAULT_SAVE_EVERY_SECONDS
                            if save_every_seconds is None else save_every_seconds),
        train_args=train_args,
        wandb_secret_name=wandb_secret_name,
        resume_local_checkpoint=resume_local_checkpoint,
        resume_run_id=resume_run_id,
    )


def _client_volume_path(path: PurePosixPath) -> str:
    """Return a root-relative Volume path. Never /artifacts/..."""
    if not isinstance(path, PurePosixPath):
        raise mrl.ValidationError(f"client Volume path must be PurePosixPath, got {type(path)!r}")
    if path.is_absolute() or any(part in {"..", ""} for part in path.parts):
        raise mrl.ValidationError(f"refusing Volume client path: {path}")
    text = path.as_posix()
    if text.startswith("/artifacts"):
        raise mrl.ValidationError(f"refusing mounted path as Volume client API: {text}")
    return text


def _missing_iterdir_errors() -> tuple[type[BaseException], ...]:
    # Empty Volume prefixes raise NotFoundError on first launch.
    types: list[type[BaseException]] = [FileNotFoundError, OSError, KeyError]
    not_found = getattr(getattr(modal, "exception", None), "NotFoundError", None)
    if isinstance(not_found, type) and issubclass(not_found, BaseException):
        types.append(not_found)
    extra = getattr(modal, "NotFoundError", None)
    if isinstance(extra, type) and issubclass(extra, BaseException):
        types.append(extra)
    return tuple(dict.fromkeys(types))


def _iterdir_paths(volume: object, path: str) -> list[str]:
    try:
        return [str(entry.path) for entry in volume.iterdir(path, recursive=False)]
    except _missing_iterdir_errors():
        return []
    except Exception as err:
        if type(err).__name__ in {"NotFoundError", "FakeNotFoundError"}:
            return []
        raise


def _volume_has_client_path(volume: object, remote: str) -> bool:
    if remote in _iterdir_paths(volume, remote):
        return True
    parent = str(PurePosixPath(remote).parent)
    listed = _iterdir_paths(volume, "" if parent == "." else parent)
    return remote in listed or PurePosixPath(remote).name in listed


def _read_volume_file(volume: object, remote: str) -> bytes | None:
    try:
        chunks = list(volume.read_file(remote))
    except (FileNotFoundError, OSError, KeyError):
        return None
    return b"".join(chunks)


def _require_blob_match(remote: bytes, expected_size: int, expected_digest: str) -> None:
    if len(remote) != expected_size or mrl.sha256_bytes(remote) != expected_digest:
        raise mrl.ValidationError("remote blob does not match local size/hash")


class ModalVolumeIndex:
    """ArtifactIndex over a Modal Volume. Client paths only; never /artifacts."""

    def __init__(self, volume: object):
        self._volume = volume
        self._staged: list[tuple[PurePosixPath, bytes]] = []

    def exists(self, path: PurePosixPath) -> bool:
        return _volume_has_client_path(self._volume, _client_volume_path(path))

    def put_file(self, path: PurePosixPath, data: bytes) -> None:
        _client_volume_path(path)
        self._staged.append((path, data))

    def commit(self) -> None:
        # Client batch_upload already persists. Volume.commit() is mounted-only
        # and raises RuntimeError from the laptop after launch_run/reserve_run.
        if self._staged:
            with self._volume.batch_upload(force=False) as batch:
                for path, data in self._staged:
                    batch.put_file(io.BytesIO(data), _client_volume_path(path))
            self._staged.clear()

    def read_file(self, path: PurePosixPath) -> bytes | None:
        return _read_volume_file(self._volume, _client_volume_path(path))


class ModalDictRegistry:
    """Registry over a Modal Dict. put_if_absent is skip_if_exists=True."""

    def __init__(self, mapping: object):
        self._dict = mapping

    def put_if_absent(self, key: str, value: Mapping[str, object]) -> bool:
        return bool(self._dict.put(key, dict(value), skip_if_exists=True))

    def get(self, key: str) -> dict[str, object] | None:
        try:
            stored = self._dict.get(key)
        except KeyError:
            return None
        if stored is None:
            return None
        return dict(stored)

    def set_existing(self, key: str, value: Mapping[str, object]) -> None:
        current = self.get(key)
        if current is None or current.get("attempt_id") != value.get("attempt_id"):
            raise mrl.ValidationError("registry claim is not owned by this attempt")
        self._dict.put(key, dict(value))


def ensure_blob(volume: object, client_path: PurePosixPath, local_path: Path) -> None:
    """Reuse a digest path after streamed verify, or upload with force=False."""
    remote = _client_volume_path(client_path)
    local_path = Path(local_path)
    expected = local_path.read_bytes()
    expected_digest = mrl.sha256_bytes(expected)
    expected_size = len(expected)
    if _volume_has_client_path(volume, remote):
        existing = _read_volume_file(volume, remote)
        if existing is None:
            raise mrl.ValidationError(f"blob listed but unreadable: {remote}")
        _require_blob_match(existing, expected_size, expected_digest)
        return
    try:
        with volume.batch_upload(force=False) as batch:
            batch.put_file(os.fspath(local_path), remote)
    except FileExistsError:
        existing = _read_volume_file(volume, remote)
        if existing is None:
            raise mrl.ValidationError(
                f"concurrent blob create left no readable object: {remote}") from None
        _require_blob_match(existing, expected_size, expected_digest)


# Launch's private vocabulary: one `CheckpointVerdict.reason` token -> the
# sentence an operator sees when a --resume-run-id parent is unusable. The
# protocol deliberately returns tokens so status reporting can collapse them to
# a bool while launch says something actionable about *this* parent.
#
# This map stays TOTAL over `checkpoint.CHECKPOINT_REASON_TOKENS`, the
# protocol's one source: `test_launch_checkpoint_errors_is_total`
# (tests/modal/test_modal_client.py) compares the two sets, and an AST census ties
# that tuple to `verify_checkpoint`'s own `fail("...")` literals, so a token
# added to the protocol goes red until a row lands here (gh#197). Should one
# slip through anyway, `prior_checkpoint_or_raise` falls back to
# `_UNMAPPED_CHECKPOINT_ERROR` rather than raising a bare KeyError at the
# operator. The map itself is not derived from the tuple on purpose: each
# sentence is launch's own wording, reviewed row by row.
_LAUNCH_CHECKPOINT_ERRORS = {
    "missing_sidecar": "parent checkpoint sidecar missing",
    "corrupt_sidecar": "parent checkpoint sidecar is corrupt",
    "missing_checkpoint": "parent checkpoint missing",
    "stale_size": "parent checkpoint metadata is stale",
    "digest_mismatch": "parent checkpoint metadata is mismatched",
    "not_loadable": "parent checkpoint is not weights-only loadable",
    "replaced": "parent checkpoint was replaced during validation",
}

# The sentence for a reason token with no row above. Unreachable while the
# totality test is green; it exists so that a protocol/launch skew reaches the
# operator as the usual ValidationError, naming the token, not as a traceback.
_UNMAPPED_CHECKPOINT_ERROR = "parent checkpoint failed verification: {reason}"


def prior_checkpoint_or_raise(volume: object, parent_id: str, now: datetime) -> tuple[bytes, str]:
    """Validate a --resume-run-id parent and return its (checkpoint bytes, digest).

    The launch-side gate: refuses to start a child run unless the parent has
    finished (or gone stale) and its checkpoint still verifies. Raises
    ValidationError with an operator sentence on every rejection; returns only
    on success, so callers do not have to re-check anything.

    Order matters. The still-active check happens here, *before*
    `mrl.verify_checkpoint`, because "the parent is still running" is a launch
    policy and not a statement about the checkpoint — `collect_status` runs the
    same verification without it and must keep reporting live runs honestly.

    PITFALLS:
      * The sidecar is read twice (before and after the checkpoint) and both
        reads are handed to the protocol: that pair is what detects a sidecar
        republished mid-validation. Do not "optimise" the second read away.
      * The returned bytes are the checkpoint, not the sidecar. `launch_run`
        stages them as `{digest}.pt` and uploads them under
        `mrl.INPUTS_ROOT / "sha256"`, so swapping the two would ship the
        sidecar's metadata as the child run's weights.
      * The `verdict.checkpoint_bytes is None` guard after `verdict.ok` is
        unreachable by contract and exists only so a future protocol bug
        surfaces as the usual sentence instead of a TypeError downstream.
    """
    index = ModalVolumeIndex(volume)
    status_bytes = index.read_file(mrl.RUNS_ROOT / parent_id / mrl.STATUS_FILENAME)
    reservation_bytes = index.read_file(mrl.RUNS_ROOT / parent_id / mrl.RESERVATION_FILENAME)
    if status_bytes is None and reservation_bytes is None:
        raise mrl.ValidationError("parent run was not found")
    view = mrl.derive_run_view_from_bytes(status_bytes, reservation_bytes, now=now)
    if view.status not in mrl.TERMINAL_STATUSES and not view.stale:
        raise mrl.ValidationError("parent run is still active")
    sidecar_path = mrl.RUNS_ROOT / parent_id / "checkpoints" / mrl.CHECKPOINT_SIDECAR_NAME
    ckpt_path = mrl.RUNS_ROOT / parent_id / "checkpoints" / mrl.CHECKPOINT_NAME
    sidecar_bytes = index.read_file(sidecar_path)
    ckpt_bytes = index.read_file(ckpt_path)
    reread_bytes = index.read_file(sidecar_path)
    verdict = mrl.verify_checkpoint(sidecar_bytes, ckpt_bytes, reread_bytes)
    if not verdict.ok:
        # `ok=False` implies `reason` is a token (CheckpointVerdict's invariant); the
        # `or` only narrows the type, and a None would surface as "unknown" here.
        reason = verdict.reason or "unknown"
        sentence = _LAUNCH_CHECKPOINT_ERRORS.get(reason,
                                                 _UNMAPPED_CHECKPOINT_ERROR.format(reason=reason))
        raise mrl.ValidationError(sentence)
    if verdict.checkpoint_bytes is None or verdict.digest is None:
        raise mrl.ValidationError(_LAUNCH_CHECKPOINT_ERRORS["not_loadable"])
    return verdict.checkpoint_bytes, verdict.digest


def _not_found_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [FileNotFoundError, KeyError]
    not_found = getattr(getattr(modal, "exception", None), "NotFoundError", None)
    if isinstance(not_found, type) and issubclass(not_found, BaseException):
        types.append(not_found)
    extra = getattr(modal, "NotFoundError", None)
    if isinstance(extra, type) and issubclass(extra, BaseException):
        types.append(extra)
    return tuple(dict.fromkeys(types))


def _is_not_found_error(err: BaseException) -> bool:
    if isinstance(err, _not_found_types()):
        return True
    return type(err).__name__ in {"NotFoundError", "FakeNotFoundError"}


def _lookup_named(factory, name: str, *, missing: str):
    try:
        return factory.from_name(name, create_if_missing=False)
    except Exception as err:
        if _is_not_found_error(err):
            raise mrl.ValidationError(missing) from err
        raise


def _thread_cap_records() -> list[str]:
    return [f"{key}={value}" for key, value in sorted(mrl.THREAD_CAP_ENV.items())]


def _seed_from_train_args(train_args: tuple[str, ...]) -> int:
    """Return the live --seed, defaulting to `cs2rl.train.__main__`'s default of 1."""
    seed = 1
    index = 0
    tokens = list(train_args)
    while index < len(tokens):
        token = tokens[index]
        value: str | None = None
        if token == "--seed":
            if index + 1 >= len(tokens):
                raise mrl.ValidationError("--seed requires a value")
            value = tokens[index + 1]
            index += 2
        elif token.startswith("--seed="):
            value = token.split("=", 1)[1]
            index += 1
        else:
            index += 1
            continue
        try:
            seed = int(value)
        except ValueError as err:
            raise mrl.ValidationError(f"--seed must be an int, got {value!r}") from err
    return seed


def _effective_timesteps(request: mrl.RunRequest) -> int:
    return (request.timesteps // request.batch_size) * request.batch_size


def _require_pinned_image_digest(digest: str) -> str:
    """Fail if the live CUDA child digest or the payload digest drifted."""
    live = CUDA_IMAGE.rsplit("@", 1)[-1]
    if live != IMAGE_DIGEST:
        raise mrl.ValidationError(
            f"live CUDA image digest {live} drifted from pinned {IMAGE_DIGEST}")
    if digest != IMAGE_DIGEST:
        raise mrl.ValidationError(f"image digest {digest} drifted from pinned {IMAGE_DIGEST}")
    return IMAGE_DIGEST


def _launch_payload(
    request: mrl.RunRequest,
    *,
    attempt_id: str,
    git_sha: str,
    tree: str,
    source_archive_sha256: str,
    source_client: PurePosixPath,
    resume_client: PurePosixPath | None,
    resume_digest: str | None,
    resume_size: int | None,
    modal_version: str,
    wandb_enabled: bool,
    created_at: str,
) -> dict[str, object]:
    resume_mount = None if resume_client is None else str(mrl.mounted_path(resume_client))
    run_root = mrl.mounted_path(mrl.RUNS_ROOT / request.run_id)
    payload: dict[str, object] = {
        "run_id": request.run_id,
        "attempt_id": attempt_id,
        "git_sha": git_sha,
        "tree": tree,
        "source_archive_sha256": source_archive_sha256,
        "source_mount_path": str(mrl.mounted_path(source_client)),
        "resume_mount_path": resume_mount,
        "resume_sha256": resume_digest,
        "resume_size": resume_size,
        "resume_source_path": resume_mount,
        "effective_map": request.effective_map,
        "gpu": request.gpu,
        "cpu_request": request.cpu_cores,
        "cpu_soft_limit": request.cpu_cores,
        "memory_request_mib": request.memory_mib,
        "memory_hard_limit_mib": request.memory_mib,
        "num_envs": request.num_envs,
        "vec_workers": request.vec_workers,
        "timeout_minutes": request.timeout_minutes,
        "save_every_seconds": request.save_every_seconds,
        "train_args": list(request.train_args),
        "timesteps": request.timesteps,
        "training_argv": request.training_argv(run_root, resume_mount),
        "requested_timesteps": request.timesteps,
        "effective_timesteps": _effective_timesteps(request),
        "batch_size": request.batch_size,
        "seed": _seed_from_train_args(request.train_args),
        "created_at": created_at,
        "runner_commit": git_sha,
        "config_hash": "0" * 64,
        "image_digest": _require_pinned_image_digest(IMAGE_DIGEST),
        "modal_version": modal_version,
        "thread_caps": _thread_cap_records(),
        "resumed_from_run_id": request.resume.prior_run_id,
    }
    if wandb_enabled:
        payload["wandb_enabled"] = True
    return payload


def build_remote_manifest(payload: dict[str, object]) -> mrl.Manifest:
    """Materialize the design §5 Manifest from the Function payload."""
    digest = _require_pinned_image_digest(str(payload["image_digest"]))
    requested = int(payload["requested_timesteps"])
    batch_size = int(payload["batch_size"])
    effective = int(payload["effective_timesteps"])
    expected = (requested // batch_size) * batch_size
    if effective != expected:
        raise mrl.ValidationError(f"effective_timesteps {effective} != floor formula {expected}")
    resume_sha = payload.get("resume_sha256")
    resume_size = payload.get("resume_size")
    resume_source = payload.get("resume_source_path")
    config_hash = payload.get("config_hash")
    resumed_from = payload.get("resumed_from_run_id")
    return mrl.Manifest(
        schema_version=mrl.SCHEMA_VERSION,
        run_id=str(payload["run_id"]),
        attempt_id=str(payload["attempt_id"]),
        commit=str(payload["git_sha"]),
        tree=str(payload["tree"]),
        source_archive_sha256=str(payload["source_archive_sha256"]),
        modal_version=str(payload["modal_version"]),
        image_digest=digest,
        effective_map=str(payload["effective_map"]),
        gpu=str(payload["gpu"]),
        cpu_request=int(payload["cpu_request"]),
        cpu_soft_limit=int(payload["cpu_soft_limit"]),
        memory_request_mib=int(payload["memory_request_mib"]),
        memory_hard_limit_mib=int(payload["memory_hard_limit_mib"]),
        vec_workers=int(payload["vec_workers"]),
        timeout_minutes=int(payload["timeout_minutes"]),
        training_argv=[str(token) for token in payload["training_argv"]],
        requested_timesteps=requested,
        effective_timesteps=effective,
        batch_size=batch_size,
        seed=int(payload["seed"]),
        created_at=str(payload["created_at"]),
        resume_sha256=None if resume_sha is None else str(resume_sha),
        resume_size=None if resume_size is None else int(resume_size),
        resume_source_path=None if resume_source is None else str(resume_source),
        runner_commit=str(payload["runner_commit"]),
        config_hash=str(config_hash) if config_hash else "0" * 64,
        thread_caps=[str(item) for item in payload["thread_caps"]],
        resumed_from_run_id=None if resumed_from is None else str(resumed_from),
    )


def _remote_run_root(run_id: str) -> Path:
    return mrl.mounted_path(mrl.RUNS_ROOT / run_id)


def _request_from_payload(payload: dict[str, object]) -> mrl.RunRequest:
    train_args = tuple(str(token) for token in payload["train_args"])
    wandb_name = "attached" if payload.get("wandb_enabled") else None
    return mrl.build_run_request(
        run_id=str(payload["run_id"]),
        git_sha=str(payload["git_sha"]),
        effective_map=str(payload["effective_map"]),
        gpu=str(payload["gpu"]),
        cpu_cores=int(payload["cpu_request"]),
        memory_mib=int(payload["memory_request_mib"]),
        num_envs=int(payload["num_envs"]),
        vec_workers=int(payload["vec_workers"]),
        timeout_minutes=int(payload["timeout_minutes"]),
        save_every_seconds=int(payload["save_every_seconds"]),
        train_args=train_args,
        wandb_secret_name=wandb_name,
    )


def launch_run(
    request: mrl.RunRequest,
    *,
    repo: Path,
    app_obj: object | None = None,
    train_fn: object | None = None,
    modal_module: object | None = None,
    now: datetime | None = None,
    attempt_id: str | None = None,
    stdout: object | None = None,
) -> dict[str, object]:
    """Validate locally, then create/claim/upload and invoke the configured Function."""
    repo = Path(repo)
    stamp = now if now is not None else datetime.now(UTC)
    nonce = attempt_id if attempt_id is not None else uuid.uuid4().hex
    modal_mod = modal if modal_module is None else modal_module
    train = train_remote if train_fn is None else train_fn
    app_handle = app if app_obj is None else app_obj
    sink = sys.stdout if stdout is None else stdout

    canonical = mrl.validate_clean_head(repo, request.git_sha)
    local_ckpt = None
    if request.resume.local_checkpoint is not None:
        local_ckpt = mrl.validate_local_checkpoint(request.resume.local_checkpoint)

    secret = None
    if request.wandb_secret_name is not None:
        try:
            secret = modal_mod.Secret.from_name(request.wandb_secret_name)
        except Exception as err:
            if _is_not_found_error(err):
                raise mrl.ValidationError("requested W&B Secret is missing") from err
            raise

    prior_bytes: bytes | None = None
    prior_digest: str | None = None
    if request.resume.prior_run_id is not None:
        parent_volume = _lookup_named(
            modal_mod.Volume,
            mrl.VOLUME_NAME,
            missing="artifact volume is missing",
        )
        prior_bytes, prior_digest = prior_checkpoint_or_raise(parent_volume,
                                                              request.resume.prior_run_id, stamp)

    modal_mod.Volume.objects.create(mrl.VOLUME_NAME, allow_existing=True)
    modal_mod.Dict.objects.create(mrl.REGISTRY_NAME, allow_existing=True)
    volume = _lookup_named(modal_mod.Volume, mrl.VOLUME_NAME, missing="artifact volume is missing")
    registry_dict = _lookup_named(modal_mod.Dict,
                                  mrl.REGISTRY_NAME,
                                  missing="run registry is missing")
    registry = ModalDictRegistry(registry_dict)
    artifacts = ModalVolumeIndex(volume)
    mrl.reserve_run(registry, artifacts, request.run_id, nonce, now=stamp)

    uploaded: dict[str, object] = {}

    def upload() -> None:
        with tempfile.TemporaryDirectory(prefix="cs2rl-launch-") as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "source.tar.gz"
            provenance = mrl.create_source_bundle(repo, canonical, archive)
            source_client = mrl.SOURCES_ROOT / f"{provenance.archive_sha256}.tar.gz"
            ensure_blob(volume, source_client, archive)
            resume_client: PurePosixPath | None = None
            resume_digest: str | None = None
            resume_size: int | None = None
            if local_ckpt is not None:
                ensure_blob(volume, local_ckpt.client_path, Path(request.resume.local_checkpoint))
                resume_client = local_ckpt.client_path
                resume_digest = local_ckpt.sha256
                resume_size = local_ckpt.size
            elif prior_bytes is not None and prior_digest is not None:
                staged = tmp_path / f"{prior_digest}.pt"
                staged.write_bytes(prior_bytes)
                resume_client = mrl.INPUTS_ROOT / "sha256" / f"{prior_digest}.pt"
                ensure_blob(volume, resume_client, staged)
                resume_digest = prior_digest
                resume_size = len(prior_bytes)
            uploaded["provenance"] = provenance
            uploaded["source_client"] = source_client
            uploaded["resume_client"] = resume_client
            uploaded["resume_digest"] = resume_digest
            uploaded["resume_size"] = resume_size

    mrl.finish_reservation(registry, artifacts, request.run_id, nonce, upload=upload)
    provenance = uploaded["provenance"]
    payload = _launch_payload(
        request,
        attempt_id=nonce,
        git_sha=canonical,
        tree=provenance.tree,
        source_archive_sha256=provenance.archive_sha256,
        source_client=uploaded["source_client"],
        resume_client=uploaded["resume_client"],
        resume_digest=uploaded["resume_digest"],
        resume_size=uploaded["resume_size"],
        modal_version=str(modal_mod.__version__),
        wandb_enabled=secret is not None,
        created_at=stamp.isoformat(),
    )

    options: dict[str, object] = {
        "gpu": request.gpu,
        "cpu": request.cpu_request_limit,
        "memory": request.memory_request_limit,
        "timeout": request.timeout_minutes * 60,
        "volumes": {
            "/artifacts": volume
        },
    }
    if secret is not None:
        options["secrets"] = [secret]
    print(f"app_id={getattr(app_handle, 'app_id', None)} run_id={request.run_id}", file=sink)
    # ASYNC spawn, not SYNC remote. .remote() is FUNCTION_CALL_INVOCATION_TYPE_SYNC:
    # the Modal client cancels that input on SIGTERM/shutdown even under
    # `modal run --detach` (live: 150826-trunk-seed2-split, 21.3M/29.98M,
    # "Successfully canceled input"). --detach only keeps the App; Modal's
    # own Function CLI uses spawn when --detach is set. spawn() returns a
    # FunctionCall handle and does not wait, so the local entrypoint can exit
    # without owning the GPU input.
    handle = train.with_options(**options).spawn(payload)
    function_call_id = getattr(handle, "object_id", None)
    print(f"function_call_id={function_call_id}", file=sink)
    return {
        "status": "spawned",
        "run_id": request.run_id,
        "function_call_id": function_call_id,
    }


@app.function(
    image=runner_image,
    retries=0,
    single_use_containers=True,
    include_source=False,
)
def train_remote(payload: dict[str, object]) -> dict[str, object]:
    """Own the delivery claim before preparation; invoked via with_options(...).spawn.

    A duplicate must return before parsing the request, preparing source or
    writing artifacts. Only this winner hands an attempt to the executor.
    """
    volume = modal.Volume.from_name(mrl.VOLUME_NAME, create_if_missing=False)
    registry = ModalDictRegistry(modal.Dict.from_name(mrl.REGISTRY_NAME, create_if_missing=False))
    attempt_id = str(payload["attempt_id"])
    run_id = str(payload["run_id"])
    if not mrl.claim_attempt(registry, attempt_id):
        return {"status": mrl.REDELIVERED, "run_id": run_id}
    request = _request_from_payload(payload)
    run_root = _remote_run_root(request.run_id)
    # One attempt for both phases: the same lock, run_root and Volume, on the
    # default (real) clock.
    attempt = mrl.AttemptContext(attempt_id=attempt_id,
                                 run_root=run_root,
                                 lock=threading.Lock(),
                                 volume=volume)
    resume = payload.get("resume_mount_path")
    resume_sha = payload.get("resume_sha256")
    manifest = build_remote_manifest(payload)
    prepared = mrl.prepare_remote_source(
        attempt=attempt,
        request=request,
        source=mrl.ExpectedSource(
            archive_path=Path(str(payload["source_mount_path"])),
            archive_sha256=str(payload["source_archive_sha256"]),
            commit=str(payload["git_sha"]),
            tree=str(payload["tree"]),
        ),
        resume=None if resume is None else mrl.RemoteResume(
            path=Path(str(resume)), sha256=None if resume_sha is None else str(resume_sha)),
        wandb_api_key=os.environ.get("WANDB_API_KEY") if payload.get("wandb_enabled") else None,
        manifest=manifest,
    )
    if prepared.config_hash is not None:
        manifest = replace(manifest, config_hash=prepared.config_hash)
    result = mrl.execute_training_attempt(
        attempt=attempt,
        prepared=prepared,
        manifest=manifest,
        timeout=timedelta(minutes=request.timeout_minutes),
    )
    return {
        "status": result.status.value,
        "run_id": request.run_id,
        "reason": result.reason,
    }


@app.local_entrypoint()
def main(
    action: str | None = None,
    run_id: str | None = None,
    git_sha: str | None = None,
    map: str | None = None,
    gpu: str | None = None,
    cpu_cores: int | None = None,
    memory_mib: int | None = None,
    num_envs: int | None = None,
    vec_workers: int | None = None,
    timeout_minutes: int | None = None,
    save_every_seconds: int | None = None,
    train_args: str | None = None,
    resume_local_checkpoint: str | None = None,
    resume_run_id: str | None = None,
    wandb_secret_name: str | None = None,
) -> None:
    """Launch-only local entrypoint. status/download are not registered here."""
    request = resolve_launch_request(
        action=action,
        run_id=run_id,
        git_sha=git_sha,
        map=map,
        gpu=gpu,
        cpu_cores=cpu_cores,
        memory_mib=memory_mib,
        num_envs=num_envs,
        vec_workers=vec_workers,
        timeout_minutes=timeout_minutes,
        save_every_seconds=save_every_seconds,
        train_args=train_args,
        resume_local_checkpoint=resume_local_checkpoint,
        resume_run_id=resume_run_id,
        wandb_secret_name=wandb_secret_name,
    )
    launch_run(request, repo=Path(__file__).resolve().parents[1])
