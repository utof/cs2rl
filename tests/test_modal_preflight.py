"""Behavior tests for scripts.modal_runner.preflight: prepare_remote_source, in order.

One of the eight per-module runner test files (RUNNER_TEST_FILES in
tests/modal_runner_tables.py), split by module from the one unsplit runner test
file in W4. A test lives in the file of the module whose behaviour it tests:
the seam manifest (tests/fixtures/modal_test_seam_manifest.json) records that
placement, and the seam gate in tests/test_modal_packaging.py checks it from
below with the reach floor. Tests reach private library names through their
owning submodules; the package facade exposes the production caller surface.
Helpers reached by tests in two or more seam files live in
tests/modal_test_helpers.py, with ownership recomputed by classify_seam.
"""
import json
import os
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # scripts/ is a namespace package; tests import scripts.modal_runner
    # the same way the later CLIs will. Do not rely on the editable install.
    sys.path.insert(0, str(ROOT))

import scripts.modal_runner as mrl                                                       # noqa: E402, I001
from scripts.modal_runner import checkpoint, commands, core, state                       # noqa: E402, I001
from tests.modal_patch_binding_campaign import binding_target                            # noqa: E402, I001
from tests.modal_test_helpers import (                                                   # noqa: E402
    _aware, _git, _init_source_repo, _make_manifest, _noop_heartbeat, _valid_run_kwargs,
    _write_dumped_config)

# ── Remote prepare: reload, verify, extract, install, resume, dump steps ───


class RecordingVolume:
    """Materializes the uploaded archive only on reload, like Volume.reload().

    commit() snapshots run_root. reload() restores that snapshot and drops
    uncommitted STATUS.json — Volume.reload() replaces the mount.
    """

    def __init__(self, src_archive: Path, dest_archive: Path, run_root: Path | None = None):
        self.events: list[str] = []
        self._src = src_archive
        self._dest = dest_archive
        self._run_root = Path(run_root) if run_root is not None else None
        self._committed_run_root: dict[Path, bytes] = {}

    def reload(self) -> None:
        self.events.append("reload")
        if not self._dest.exists():
            self._dest.parent.mkdir(parents=True, exist_ok=True)
            self._dest.write_bytes(self._src.read_bytes())
        self._restore_run_root()

    def commit(self) -> None:
        self.events.append("commit")
        self._snapshot_run_root()

    def _snapshot_run_root(self) -> None:
        if self._run_root is None or not self._run_root.exists():
            return
        snapshot: dict[Path, bytes] = {}
        for path in self._run_root.rglob("*"):
            if path.is_file():
                snapshot[path.relative_to(self._run_root)] = path.read_bytes()
        self._committed_run_root = snapshot

    def _restore_run_root(self) -> None:
        if self._run_root is None:
            return
        if self._run_root.exists():
            for path in self._run_root.rglob("*"):
                if not path.is_file():
                    continue
                if path.relative_to(self._run_root) not in self._committed_run_root:
                    path.unlink()
        for rel, data in self._committed_run_root.items():
            dest = self._run_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)


def _source_bundle(tmp_path: Path):
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", f"{sha}^{{tree}}")
    client_archive = tmp_path / "client.tar.gz"
    provenance = mrl.create_source_bundle(repo, sha, client_archive)
    mount_archive = tmp_path / "artifacts" / "sources" / f"{provenance.archive_sha256}.tar.gz"
    return sha, tree, client_archive, mount_archive, provenance


def _preflight_kwargs(tmp_path: Path, **overrides):
    sha, tree, client_archive, mount_archive, provenance = _source_bundle(tmp_path)
    run_root = tmp_path / "run"
    run_root.mkdir()
    kwargs = {
        "volume": RecordingVolume(client_archive, mount_archive, run_root),
        "archive_path": mount_archive,
        "expected_archive_sha256": provenance.archive_sha256,
        "expected_commit": sha,
        "expected_tree": tree,
        "request": mrl.build_run_request(**_valid_run_kwargs(run_id="ok-id")),
        "run_root": run_root,
        "attempt_id": "attempt-a",
        "lock": threading.Lock(),
        "ephemeral_parent": tmp_path / "ephemeral",
        "parent_env": {
            "PATH": "/usr/bin",
            "HOME": "/home/modal",
            "WANDB_API_KEY": "parent-secret"
        },
        "now": lambda: _aware(),
        "start_heartbeat": _noop_heartbeat,
    }
    kwargs.update(overrides)
    return kwargs


def test_recording_volume_reload_restores_committed_run_root(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    src = tmp_path / "src.tar.gz"
    src.write_bytes(b"archive")
    dest = tmp_path / "dest.tar.gz"
    volume = RecordingVolume(src, dest, run_root)
    (run_root / mrl.STATUS_FILENAME).write_text("uncommitted\n")
    volume.reload()
    assert dest.read_bytes() == b"archive"
    assert not (run_root / mrl.STATUS_FILENAME).exists()
    (run_root / mrl.STATUS_FILENAME).write_text("preparing\n")
    (run_root / "keep.txt").write_text("committed\n")
    volume.commit()
    (run_root / mrl.STATUS_FILENAME).write_text("dirty\n")
    (run_root / "extra.txt").write_text("uncommitted\n")
    volume.reload()
    assert (run_root / mrl.STATUS_FILENAME).read_text() == "preparing\n"
    assert (run_root / "keep.txt").read_text() == "committed\n"
    assert not (run_root / "extra.txt").exists()


def test_prepare_reloads_before_status_write_and_commits_before_heartbeat(tmp_path):
    order: list[object] = []

    def fake_run(cmd, **kwargs):
        if "--dump-config" in list(cmd):
            _write_dumped_config(run_root)
        return subprocess.CompletedProcess(cmd, 0)

    def start_heartbeat(**kwargs):
        order.append("heartbeat")
        status = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
        assert status["status"] == "preparing"
        return SimpleNamespace(stop_and_join=lambda: None)

    kwargs = _preflight_kwargs(tmp_path, run=fake_run, start_heartbeat=start_heartbeat)
    volume = kwargs["volume"]
    run_root = kwargs["run_root"]
    orig_reload = volume.reload
    orig_commit = volume.commit

    def tracking_reload():
        order.append(("reload", (run_root / mrl.STATUS_FILENAME).exists()))
        orig_reload()

    def tracking_commit():
        order.append("commit")
        orig_commit()

    volume.reload = tracking_reload
    volume.commit = tracking_commit
    mrl.prepare_remote_source(**kwargs)
    assert order[0] == ("reload", False)
    assert order[1] == "commit"
    assert order[2] == "heartbeat"
    assert Path(mrl.STATUS_FILENAME) in volume._committed_run_root


def test_prepare_reloads_verifies_extracts_then_installs(tmp_path):
    recorded: list[tuple[list[str], dict]] = []
    heartbeat_events: list[str] = []

    def start_heartbeat(**kwargs):
        heartbeat_events.append("start")
        return SimpleNamespace(stop_and_join=lambda: heartbeat_events.append("stopped"))

    def fake_run(cmd, **kwargs):
        assert heartbeat_events == ["start"]
        recorded.append((list(cmd), kwargs))
        if "--dump-config" in list(cmd):
            _write_dumped_config(run_root)
        return subprocess.CompletedProcess(cmd, 0)

    kwargs = _preflight_kwargs(tmp_path, run=fake_run, start_heartbeat=start_heartbeat)
    volume = kwargs["volume"]
    run_root = kwargs["run_root"]
    prepared = mrl.prepare_remote_source(**kwargs)
    assert volume.events[0] == "reload"
    assert recorded, "install command was never invoked"
    install_cmd, install_kwargs = recorded[0]
    assert install_cmd == commands.build_install_command(prepared.source_dir)
    assert install_kwargs["cwd"] == os.fspath(prepared.source_dir)
    assert install_kwargs["shell"] is False
    assert install_kwargs["env"]["PATH"] == "/usr/bin"
    assert install_kwargs["env"]["OMP_NUM_THREADS"] == "1"
    assert "WANDB_API_KEY" not in install_kwargs["env"]
    assert (prepared.source_dir / "readme.txt").read_text() == "hello\n"
    sidecar = json.loads((prepared.source_dir / core.PROVENANCE_NAME).read_text())
    assert sidecar["commit"] == kwargs["expected_commit"]
    assert sidecar["tree"] == kwargs["expected_tree"]
    assert prepared.source_dir.is_relative_to(tmp_path / "ephemeral")
    assert not mount_is_extract_root(prepared.source_dir, kwargs["archive_path"])


def mount_is_extract_root(source_dir: Path, archive_path: Path) -> bool:
    return source_dir == archive_path.parent or archive_path.parent in source_dir.parents


def test_prepare_rejects_archive_hash_mismatch(tmp_path):
    kwargs = _preflight_kwargs(tmp_path, expected_archive_sha256="0" * 64, run=lambda *a, **k: None)
    with pytest.raises(mrl.ValidationError):
        mrl.prepare_remote_source(**kwargs)
    # Hash is checked after reload; the archive must not be trusted blindly.
    assert kwargs["volume"].events[0] == "reload"
    assert not (kwargs["run_root"] / mrl.STATUS_FILENAME).exists()


def test_prepare_rejects_provenance_sidecar_mismatch(tmp_path):
    kwargs = _preflight_kwargs(tmp_path, expected_commit="f" * 40, run=lambda *a, **k: None)
    with pytest.raises(mrl.ValidationError):
        mrl.prepare_remote_source(**kwargs)


def test_prepare_reads_archive_only_after_volume_reload(tmp_path):
    kwargs = _preflight_kwargs(tmp_path, run=lambda *a, **k: None)

    class BlindVolume:
        events: list[str] = []

        def reload(self) -> None:
            self.events.append("reload")

        def commit(self) -> None:
            self.events.append("commit")

    kwargs["volume"] = BlindVolume()
    with pytest.raises((mrl.ValidationError, FileNotFoundError, OSError)):
        mrl.prepare_remote_source(**kwargs)
    assert kwargs["volume"].events[0] == "reload"


def test_prepare_validates_resume_then_dumps_and_hashes_config(tmp_path):
    import torch

    ckpt = tmp_path / "artifacts" / "inputs" / "sha256" / "warm.pt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"weight": torch.tensor([1.0, 2.0])}, ckpt)
    digest = core.sha256_file(ckpt)
    recorded: list[tuple[list[str], dict]] = []
    validated: list[Path] = []

    def fake_run(cmd, **kwargs):
        recorded.append((list(cmd), kwargs))
        cmd_list = list(cmd)
        if "--dump-config" in cmd_list:
            # Resume must already have been accepted before the cheap dump.
            assert validated == [ckpt]
            _write_dumped_config(kwargs_run_root)
        return subprocess.CompletedProcess(cmd, 0)

    validate_target, validate_name = binding_target("prepare-validator")
    orig_validate = getattr(validate_target, validate_name)

    def tracking_validate(path):
        validated.append(Path(path))
        return orig_validate(path)

    kwargs_run_root = tmp_path / "run"
    manifest = _make_manifest(config_hash="0" * 64, run_id="ok-id")
    kwargs = _preflight_kwargs(
        tmp_path,
        run=fake_run,
        start_heartbeat=_noop_heartbeat,
        remote_resume=str(ckpt),
        expected_resume_sha256=digest,
        manifest=manifest,
    )
    kwargs_run_root = kwargs["run_root"]
    monkey_validate = tracking_validate
    setattr(validate_target, validate_name, monkey_validate)
    try:
        prepared = mrl.prepare_remote_source(**kwargs)
    finally:
        setattr(validate_target, validate_name, orig_validate)
    assert validated == [ckpt]
    assert len(recorded) >= 2
    dump_cmd, dump_kwargs = recorded[1]
    request = kwargs["request"]
    assert dump_cmd == commands.build_dump_config_command(request, str(ckpt))
    assert dump_kwargs["cwd"] == os.fspath(prepared.source_dir)
    assert dump_kwargs["shell"] is False
    assert dump_kwargs["env"]["OMP_NUM_THREADS"] == "1"
    dumped = json.loads((kwargs["run_root"] / "checkpoints" / "config.json").read_text())
    expected_hash = mrl.sha256_bytes(
        json.dumps(checkpoint.normalize_config_for_transport(dumped),
                   sort_keys=True,
                   separators=(",", ":")).encode())
    payload = json.loads((kwargs["run_root"] / core.MANIFEST_FILENAME).read_text())
    assert payload["config_hash"] == expected_hash
    assert prepared.config_hash == expected_hash
    assert "data_dir" not in checkpoint.normalize_config_for_transport(dumped)


def test_prepare_rejects_resume_hash_mismatch(tmp_path):
    import torch

    ckpt = tmp_path / "warm.pt"
    torch.save({"weight": torch.tensor([1.0])}, ckpt)
    kwargs = _preflight_kwargs(
        tmp_path,
        run=lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0),
        start_heartbeat=_noop_heartbeat,
        remote_resume=str(ckpt),
        expected_resume_sha256="0" * 64,
        manifest=_make_manifest(),
    )
    with pytest.raises(mrl.ValidationError):
        mrl.prepare_remote_source(**kwargs)


def test_prepare_rejects_non_checkpoint_resume(tmp_path):
    ckpt = tmp_path / "warm.pt"
    ckpt.write_text("not a checkpoint\n")
    kwargs = _preflight_kwargs(
        tmp_path,
        run=lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0),
        start_heartbeat=_noop_heartbeat,
        remote_resume=str(ckpt),
        expected_resume_sha256=core.sha256_file(ckpt),
        manifest=_make_manifest(),
    )
    with pytest.raises(mrl.ValidationError):
        mrl.prepare_remote_source(**kwargs)


# ── Preflight: command order, build failure, secrets, heartbeat ────────────


def test_prepare_records_install_dump_probe_then_launch(tmp_path):
    recorded: list[list[str]] = []
    launched: list[object] = []

    def fake_run(cmd, **kwargs):
        recorded.append(list(cmd))
        if "--dump-config" in list(cmd):
            _write_dumped_config(kwargs_run_root)
        return subprocess.CompletedProcess(cmd, 0)

    def on_ready(prepared):
        launched.append(prepared)
        fake_run(
            prepared.train_command,
            cwd=os.fspath(prepared.source_dir),
            shell=False,
            env=prepared.child_env,
        )

    kwargs = _preflight_kwargs(
        tmp_path,
        run=fake_run,
        start_heartbeat=_noop_heartbeat,
        on_ready=on_ready,
        manifest=_make_manifest(run_id="ok-id"),
    )
    kwargs_run_root = kwargs["run_root"]
    prepared = mrl.prepare_remote_source(**kwargs)
    assert launched and launched[0] is prepared
    assert recorded[0] == commands.build_install_command(prepared.source_dir)
    assert recorded[1] == commands.build_dump_config_command(kwargs["request"], None)
    assert recorded[2] == commands.build_cuda_probe_command()
    assert recorded[3] == prepared.train_command
    assert prepared.train_command == commands.build_train_command(
        commands.build_train_argv(kwargs["request"], None))
    assert prepared.heartbeat is not None


def test_preflight_failure_stops_heartbeat_then_writes_build_failed(tmp_path):
    order: list[str] = []
    status_at_stop: list[str] = []

    def start_heartbeat(**_kwargs):

        def stop_and_join():
            order.append("stop")
            status_path = kwargs["run_root"] / mrl.STATUS_FILENAME
            status_at_stop.append(json.loads(status_path.read_text())["status"])

        return SimpleNamespace(stop_and_join=stop_and_join)

    def fake_run(cmd, **_kwargs):
        if list(cmd)[:3] == ["/usr/local/bin/uv", "pip", "install"]:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    kwargs = _preflight_kwargs(tmp_path, run=fake_run, start_heartbeat=start_heartbeat)
    with pytest.raises(subprocess.CalledProcessError):
        mrl.prepare_remote_source(**kwargs)
    persisted = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
    assert order == ["stop"]
    assert status_at_stop == ["building"]
    assert persisted["status"] == "build_failed"
    assert persisted["attempt_id"] == "attempt-a"


def test_preflight_failure_keeps_build_failed_when_heartbeat_stop_raises(tmp_path):

    def start_heartbeat(**_kwargs):

        def stop_and_join(timeout: float = 5.0):
            raise RuntimeError("heartbeat worker did not stop")

        return SimpleNamespace(stop_and_join=stop_and_join)

    def fake_run(cmd, **_kwargs):
        if list(cmd)[:3] == ["/usr/local/bin/uv", "pip", "install"]:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    kwargs = _preflight_kwargs(tmp_path, run=fake_run, start_heartbeat=start_heartbeat)
    with pytest.raises(subprocess.CalledProcessError):
        mrl.prepare_remote_source(**kwargs)
    persisted = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "build_failed"
    assert persisted["attempt_id"] == "attempt-a"
    ephemeral = kwargs["ephemeral_parent"]
    assert not ephemeral.exists() or not any(ephemeral.iterdir())


def test_prepare_does_not_persist_wandb_secret(tmp_path):
    secret = "secret-from-modal"
    recorded_envs: list[dict[str, str]] = []

    def fake_run(cmd, **kwargs):
        recorded_envs.append(dict(kwargs["env"]))
        if "--dump-config" in list(cmd):
            _write_dumped_config(kwargs_run_root)
        return subprocess.CompletedProcess(cmd, 0)

    request = mrl.build_run_request(**_valid_run_kwargs(
        run_id="ok-id",
        train_args="--timesteps 163840 --wandb",
        wandb_secret_name="wandb",
    ))
    kwargs = _preflight_kwargs(
        tmp_path,
        run=fake_run,
        start_heartbeat=_noop_heartbeat,
        request=request,
        wandb_api_key=secret,
        manifest=_make_manifest(run_id="ok-id"),
    )
    kwargs_run_root = kwargs["run_root"]
    prepared = mrl.prepare_remote_source(**kwargs)
    assert all(env["WANDB_API_KEY"] == secret for env in recorded_envs)
    assert prepared.child_env["WANDB_API_KEY"] == secret
    assert secret not in repr(prepared)
    for path in kwargs["run_root"].rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(errors="ignore")


def test_heartbeat_commits_throughout_blocked_preflight(tmp_path):

    class Clock:

        def __init__(self):
            self._now = _aware()
            self._lock = threading.Lock()

        def now(self):
            with self._lock:
                return self._now

        def advance(self, seconds: float):
            with self._lock:
                self._now += timedelta(seconds=seconds)
                return self._now

    clock = Clock()
    beat_times: list = []

    def wait(event: threading.Event, seconds: float) -> bool:
        clock.advance(seconds)
        return event.wait(0.01)

    def start_heartbeat(**kwargs):
        return state.start_heartbeat_worker(
            run_root=kwargs["run_root"],
            attempt_id=kwargs["attempt_id"],
            lock=kwargs["lock"],
            now=clock.now,
            commit=kwargs["commit"],
            interval=timedelta(seconds=60),
            wait=wait,
        )

    def fake_run(cmd, **kwargs):
        cmd_list = list(cmd)
        if cmd_list[:3] == ["/usr/local/bin/uv", "pip", "install"]:
            started = clock.now()
            deadline = time.monotonic() + 5.0
            while clock.now() - started < timedelta(minutes=5, seconds=1):
                if time.monotonic() > deadline:
                    raise TimeoutError("fake clock did not advance during blocked install")
                time.sleep(0.01)
            return subprocess.CompletedProcess(cmd, 0)
        if "--dump-config" in cmd_list:
            _write_dumped_config(kwargs_run_root)
        return subprocess.CompletedProcess(cmd, 0)

    kwargs = _preflight_kwargs(
        tmp_path,
        run=fake_run,
        start_heartbeat=start_heartbeat,
        now=clock.now,
        manifest=_make_manifest(run_id="ok-id"),
    )
    kwargs_run_root = kwargs["run_root"]
    volume = kwargs["volume"]
    orig_commit = volume.commit

    def recording_commit():
        beat_times.append(clock.now())
        orig_commit()

    volume.commit = recording_commit
    prepared = mrl.prepare_remote_source(**kwargs)
    assert prepared.heartbeat is not None
    assert prepared.heartbeat.thread.is_alive()
    prepared.heartbeat.stop_and_join()
    assert not prepared.heartbeat.thread.is_alive()
    assert len(beat_times) >= 6
    for earlier, later in zip(beat_times, beat_times[1:], strict=False):
        assert later - earlier <= timedelta(seconds=60)
    status = state.RunStatus.from_dict(
        json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text()))
    derived = state.derive_status(status, now=clock.now())
    assert derived.stale is False
