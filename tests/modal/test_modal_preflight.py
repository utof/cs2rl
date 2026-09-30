"""Behavior tests for scripts.modal_runner.preflight: prepare_remote_source, in order.

One of the per-module runner test files (RUNNER_TEST_FILES in
tests/modal/modal_runner_tables.py). Before you add, move or delete a test here, or
add a helper, read THE PLACEMENT RULE FOR RUNNER TESTS in
tests/modal/test_modal_packaging.py: which file a test belongs in, what the change
costs in the seam manifest, and where helpers go.
"""
import ast
import dataclasses
import json
import os
import subprocess
import threading
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.conftest import REPO_ROOT

ROOT = REPO_ROOT

import scripts.modal_runner as mrl                                                       # noqa: E402, I001
from scripts.modal_runner import checkpoint, commands, core, preflight, state            # noqa: E402, I001
from tests.modal.modal_patch_binding_campaign import binding_target                      # noqa: E402, I001
from tests.modal.modal_test_helpers import (                                             # noqa: E402
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


def _preflight_kwargs(tmp_path: Path, **overrides) -> dict[str, Any]:
    """prepare_remote_source's keyword arguments over a real source bundle, from flat overrides.

    Call sites pass flat keys (`run=`, `now=`, `expected_commit=`, ...); this
    assembles them into the collaborators prepare takes (gh#163 W5): `attempt`
    (an AttemptContext over a RecordingVolume), `request`, `source` (an
    ExpectedSource) and `host` (a PreflightHost), plus `resume`, `manifest` and
    `wandb_api_key` only when overridden, so an absent one keeps prepare's own
    default.

    Every known key is taken with `overrides.pop`, and a leftover raises
    TypeError: a misspelt or retired key (`on_ready`) stays loud, as it was
    when the flat dict went straight to prepare. The mapping from key to field
    is hand-written, so `test_preflight_kwargs_routes_every_override` checks
    that every key a call site passes reaches its field; a key popped here and
    then dropped fails there instead of quietly testing the default.

    PITFALLS. `remote_resume` is stored as `Path(remote_resume)`, the one
    converted key (RemoteResume.path is a Path; call sites pass `str(ckpt)`);
    every other value is stored as passed. `expected_resume_sha256` without
    `remote_resume` raises: the hash names no file on its own. Without a `run`
    override the host's `run` raises AssertionError naming the missing
    override; it is never PreflightHost's default, the real `subprocess.run`,
    which would run the install and probe commands on the machine running the
    tests. Every call site passes one.
    The result is typed `dict[str, Any]` because tests reach test-double
    members through it (`kwargs["attempt"].volume.events` on a
    RecordingVolume), which the collaborator types do not declare.
    """
    known = ("request", "now", "run", "start_heartbeat", "expected_archive_sha256",
             "expected_commit", "remote_resume", "expected_resume_sha256", "manifest",
             "wandb_api_key")
    given = {key: overrides.pop(key) for key in known if key in overrides}
    if overrides:
        raise TypeError(f"unknown override(s): {sorted(overrides)}")
    if "expected_resume_sha256" in given and "remote_resume" not in given:
        raise TypeError("expected_resume_sha256 needs remote_resume: the hash names no file alone")
    sha, tree, client_archive, mount_archive, provenance = _source_bundle(tmp_path)
    run_root = tmp_path / "run"
    run_root.mkdir()

    def run_not_overridden(cmd, **_kwargs):
        raise AssertionError(f"prepare ran {cmd!r} through the default host run: pass run= to "
                             "_preflight_kwargs, or the real subprocess.run would run it")

    host = preflight.PreflightHost(
        run=given.get("run", run_not_overridden),
        parent_env={
            "PATH": "/usr/bin",
            "HOME": "/home/modal",
            "WANDB_API_KEY": "parent-secret"
        },
        ephemeral_parent=tmp_path / "ephemeral",
        start_heartbeat=given.get("start_heartbeat", _noop_heartbeat),
    )
    request = (given["request"] if "request" in given else mrl.build_run_request(
        **_valid_run_kwargs(run_id="ok-id")))
    attempt = core.AttemptContext(
        attempt_id="attempt-a",
        run_root=run_root,
        lock=threading.Lock(),
        volume=RecordingVolume(client_archive, mount_archive, run_root),
        clock=core.Clock(now=given.get("now", lambda: _aware())),
    )
    source = preflight.ExpectedSource(
        archive_path=mount_archive,
        archive_sha256=given.get("expected_archive_sha256", provenance.archive_sha256),
        commit=given.get("expected_commit", sha),
        tree=tree,
    )
    kwargs: dict[str, Any] = {
        "attempt": attempt,
        "request": request,
        "source": source,
        "host": host
    }
    if "remote_resume" in given:
        kwargs["resume"] = preflight.RemoteResume(path=Path(given["remote_resume"]),
                                                  sha256=given.get("expected_resume_sha256"))
    kwargs.update({key: given[key] for key in ("manifest", "wandb_api_key") if key in given})
    return kwargs


def test_preflight_kwargs_routes_every_override(tmp_path):
    """Every flat key a call site passes to `_preflight_kwargs` reaches the field prepare reads.

    gh#163 W5. The builder assembles flat overrides into collaborators
    by hand-written code, and several tests assert that something is ABSENT
    from a recorder they injected; such an assertion goes vacuous, still
    green, if the builder stops routing its key. So, both ways:
      * the keys call sites pass, enumerated by AST over the modal test files,
        must equal the keys of `routes`. A call is read through the bare name,
        an attribute `x._preflight_kwargs`, or an `import ... as` or plain
        `name = ...` alias. A key a call site passes that `routes` lacks fails,
        and so does a `routes` entry that no call site passes;
      * each key, passed as a sentinel, must come back by identity at the
        field `routes` names (the collaborator field that replaced the flat
        key). `remote_resume` alone compares by equality: the
        builder converts it, because RemoteResume.path is a Path and call
        sites pass `str(ckpt)`.
    A call site whose keys cannot be read statically (a `**` splat, a second
    positional argument) fails too, and so does any other reference to the
    builder: its name loaded anywhere but as a callee or a plain alias's value
    (`functools.partial(_preflight_kwargs, ...)`, a tuple assignment), or the
    name as a string (`getattr(module, "_preflight_kwargs")`). RESIDUAL: a
    name computed at run time (a concatenated string, `vars()` with a variable
    key) is not seen. This test's own calls and strings are not call sites.
    Without `run=`, the built host's `run` must raise, never be the real
    `subprocess.run`; that is checked last.

    THE PLANTS are synthetic call sites, parsed and never run, each passing an
    unmapped key through one spelling: each must fail the key equality or be
    reported as a problem, or the enumeration would not be evidence. Keep
    `routes` and the plants INSIDE this function: a module-level name in this
    file is a governed seam name and moves GOVERNED_NAME_COUNT
    (tests/modal/test_modal_packaging.py). The keys that are only READ back from the
    built kwargs and passed at no call site (`run_root`, `volume`,
    `archive_path`, `expected_tree`) are not here, by the same two-way rule:
    each would be an entry no call site passes. The enumerator (`last_name`,
    `call_site_keys`) is duplicated in `test_training_kwargs_routes_every_override`
    (tests/modal/test_modal_training.py), which asserts the two copies are AST-equal:
    change both together.
    """
    routes = {
        "expected_archive_sha256": lambda built: built["source"].archive_sha256,
        "expected_commit": lambda built: built["source"].commit,
        "expected_resume_sha256": lambda built: built["resume"].sha256,
        "manifest": lambda built: built["manifest"],
        "now": lambda built: built["attempt"].clock.now,
        "remote_resume": lambda built: built["resume"].path,
        "request": lambda built: built["request"],
        "run": lambda built: built["host"].run,
        "start_heartbeat": lambda built: built["host"].start_heartbeat,
        "wandb_api_key": lambda built: built["wandb_api_key"],
    }
    builder = "_preflight_kwargs"
    this_test = "test_preflight_kwargs_routes_every_override"

    def last_name(expr):
        """The callee spelling a call site is matched on: a bare name, or the last attribute."""
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Attribute):
            return expr.attr
        return None

    def call_site_keys(sources):
        """({key: [file:line, ...]}, problems) over every builder call outside this test.

        A reference to the builder that is not read as a call (see the
        docstring above) is a problem, so a spelling this cannot read fails
        rather than hiding its keys.
        """
        keys, problems = {}, []
        for rel, text in sources.items():
            tree = ast.parse(text)
            spellings = {builder}
            # ids of the loads a spelling may occupy: callees and plain alias values
            read = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    spellings |= {
                        alias.asname
                        for alias in node.names if alias.name == builder and alias.asname
                    }
                elif isinstance(node, ast.Assign) and last_name(node.value) == builder:
                    spellings |= {t.id for t in node.targets if isinstance(t, ast.Name)}
                    read.add(id(node.value))
            own = {
                id(node)
                for top in tree.body if isinstance(top, ast.FunctionDef) and top.name == this_test
                for node in ast.walk(top)
            }
            for node in ast.walk(tree):
                if (not isinstance(node, ast.Call) or last_name(node.func) not in spellings
                        or id(node) in own):
                    continue
                read.add(id(node.func))
                where = f"{rel}:{node.lineno}"
                if len(node.args) != 1 or any(k.arg is None for k in node.keywords):
                    problems.append(f"{where} {ast.unparse(node)[:100]}")
                for keyword in node.keywords:
                    if keyword.arg is not None:
                        keys.setdefault(keyword.arg, []).append(where)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Name, ast.Attribute)):
                    stray = last_name(node) in spellings and isinstance(node.ctx, ast.Load)
                elif isinstance(node, ast.Constant):
                    stray = node.value == builder
                else:
                    continue
                if stray and id(node) not in read and id(node) not in own:
                    problems.append(f"{rel}:{node.lineno} not a call: {ast.unparse(node)[:100]}")
        return keys, problems

    sources = {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for pattern in ("test_modal_*.py", "modal_test_helpers.py",
                        "modal_patch_binding_campaign.py")
        for path in sorted((ROOT / "tests").rglob(pattern))
    }
    assert {
        "tests/modal/test_modal_preflight.py", "tests/modal/modal_test_helpers.py",
        "tests/modal/modal_patch_binding_campaign.py"
    } <= set(sources), sorted(sources)[:5]
    keys, problems = call_site_keys(sources)
    assert problems == [], (
        f"call sites whose override keys cannot be read statically: {problems}. Call the "
        "builder by its name, an attribute or a plain alias, and pass every override as a "
        "keyword, so this test can check that it is routed")
    unrouted = sorted(set(keys) - set(routes))
    unused = sorted(set(routes) - set(keys))
    assert not unrouted and not unused, (
        f"call sites pass {unrouted} ({ {key: keys[key] for key in unrouted} }), which "
        f"`routes` does not map, and `routes` maps {unused}, which no call site passes. A new "
        "key needs a field in `_preflight_kwargs` and an entry here; a key nobody passes any "
        "more comes out of both")

    plants = {
        "the bare name":
        "_preflight_kwargs(tmp_path, on_ready=f)\n",
        "an attribute":
        "preflight_tests._preflight_kwargs(tmp_path, on_ready=f)\n",
        "an import alias":
        ("from tests.modal.test_modal_preflight import _preflight_kwargs as build\n"
         "build(tmp_path, on_ready=f)\n"),
        "an assignment alias":
        "build = _preflight_kwargs\nbuild(tmp_path, on_ready=f)\n",
        "functools.partial":
        "build = functools.partial(_preflight_kwargs, on_ready=f)\nbuild(tmp_path)\n",
        "getattr by name":
        "getattr(preflight_tests, '_preflight_kwargs')(tmp_path, on_ready=f)\n",
        "a tuple-assignment alias":
        "build, _ = _preflight_kwargs, None\nbuild(tmp_path, on_ready=f)\n",
    }
    for plant, source in plants.items():
        planted, planted_problems = call_site_keys({**sources, "tests/test_modal_plant.py": source})
        assert planted_problems or set(planted) != set(routes), (
            f"a call site passing an unmapped key through {plant} left the key sets equal and "
            "reported no problem, so the enumeration cannot see that spelling and its green is "
            "not evidence")
    _, splat = call_site_keys({"tests/test_modal_plant.py": "_preflight_kwargs(tmp_path, **k)\n"})
    assert splat, "a `**` splat call site was not reported, so its keys would go unchecked"

    resume = str(tmp_path / "warm.pt")
    sentinels: dict[str, object] = {key: object() for key in routes}
    sentinels["remote_resume"] = resume
    built = _preflight_kwargs(tmp_path, **sentinels)
    assert set(built) == {
        "attempt", "request", "source", "host", "resume", "manifest", "wandb_api_key"
    }, sorted(built)
    for key, route in routes.items():
        if key == "remote_resume":
            # Equality, not identity: the one key the builder converts (call sites pass
            # str(ckpt); RemoteResume.path is a Path), so it stores a new object.
            assert route(built) == Path(resume), f"{key} does not reach its field"
        else:
            assert route(built) is sentinels[key], f"{key} does not reach its field"
    # A directory each, so a builder that stopped raising would build there and fail on
    # DID NOT RAISE, rather than on colliding with the build above.
    leftover, hash_alone = tmp_path / "leftover", tmp_path / "hash-alone"
    leftover.mkdir()
    hash_alone.mkdir()
    with pytest.raises(TypeError, match=r"unknown override\(s\): \['on_ready'\]"):
        _preflight_kwargs(leftover, on_ready=lambda prepared: None)
    with pytest.raises(TypeError, match="needs remote_resume"):
        _preflight_kwargs(hash_alone, expected_resume_sha256="0" * 64)
    # Without run=, the host's run is loud, never the real subprocess.run. The program named
    # does not exist, so even a builder that fell back to the real one would execute nothing.
    no_run = tmp_path / "no-run"
    no_run.mkdir()
    with pytest.raises(AssertionError, match="pass run="):
        _preflight_kwargs(no_run)["host"].run(["/nonexistent/cs2rl-never-run"])


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
    volume = kwargs["attempt"].volume
    run_root = kwargs["attempt"].run_root
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
    volume = kwargs["attempt"].volume
    run_root = kwargs["attempt"].run_root
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
    assert sidecar["commit"] == kwargs["source"].commit
    assert sidecar["tree"] == kwargs["source"].tree
    assert prepared.source_dir.is_relative_to(tmp_path / "ephemeral")
    assert not mount_is_extract_root(prepared.source_dir, kwargs["source"].archive_path)


def mount_is_extract_root(source_dir: Path, archive_path: Path) -> bool:
    return source_dir == archive_path.parent or archive_path.parent in source_dir.parents


def test_prepare_rejects_archive_hash_mismatch(tmp_path):
    kwargs = _preflight_kwargs(tmp_path, expected_archive_sha256="0" * 64, run=lambda *a, **k: None)
    with pytest.raises(mrl.ValidationError):
        mrl.prepare_remote_source(**kwargs)
    # Hash is checked after reload; the archive must not be trusted blindly.
    assert kwargs["attempt"].volume.events[0] == "reload"
    assert not (kwargs["attempt"].run_root / mrl.STATUS_FILENAME).exists()


def test_prepare_rejects_provenance_sidecar_mismatch(tmp_path):
    kwargs = _preflight_kwargs(tmp_path, expected_commit="f" * 40, run=lambda *a, **k: None)
    # Bound on purpose (the resource-acquisition rule in prepare_remote_source's PITFALL): the
    # live traceback keeps the failing frame, and with it the TemporaryDirectory whose finalizer
    # would otherwise erase the extracted tree when the block exits, so the cleanup assertion
    # below could not see a skipped cleanup.
    with pytest.raises(mrl.ValidationError, match="provenance sidecar") as excinfo: # noqa: F841
        mrl.prepare_remote_source(**kwargs)
    ephemeral = kwargs["host"].ephemeral_parent
    assert not ephemeral.exists() or not any(ephemeral.iterdir())
    persisted = json.loads((kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "build_failed"


def test_prepare_reads_archive_only_after_volume_reload(tmp_path):
    kwargs = _preflight_kwargs(tmp_path, run=lambda *a, **k: None)

    class BlindVolume:
        events: list[str] = []

        def reload(self) -> None:
            self.events.append("reload")

        def commit(self) -> None:
            self.events.append("commit")

    kwargs["attempt"] = dataclasses.replace(kwargs["attempt"], volume=BlindVolume())
    with pytest.raises((mrl.ValidationError, FileNotFoundError, OSError)):
        mrl.prepare_remote_source(**kwargs)
    assert kwargs["attempt"].volume.events[0] == "reload"


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
    kwargs_run_root = kwargs["attempt"].run_root
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
    dumped = json.loads((kwargs["attempt"].run_root / "checkpoints" / "config.json").read_text())
    expected_hash = mrl.sha256_bytes(
        json.dumps(checkpoint.normalize_config_for_transport(dumped),
                   sort_keys=True,
                   separators=(",", ":")).encode())
    payload = json.loads((kwargs["attempt"].run_root / core.MANIFEST_FILENAME).read_text())
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
    with pytest.raises(mrl.ValidationError, match="resume sha256"):
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
    with pytest.raises(mrl.ValidationError, match="weights-only loadable"):
        mrl.prepare_remote_source(**kwargs)


# ── Preflight: command order, build failure, secrets, heartbeat ────────────


def test_prepare_records_install_dump_probe_in_order(tmp_path):
    recorded: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        recorded.append(list(cmd))
        if "--dump-config" in list(cmd):
            _write_dumped_config(kwargs_run_root)
        return subprocess.CompletedProcess(cmd, 0)

    kwargs = _preflight_kwargs(
        tmp_path,
        run=fake_run,
        start_heartbeat=_noop_heartbeat,
        manifest=_make_manifest(run_id="ok-id"),
    )
    kwargs_run_root = kwargs["attempt"].run_root
    prepared = mrl.prepare_remote_source(**kwargs)
    assert recorded[0] == commands.build_install_command(prepared.source_dir)
    assert recorded[1] == commands.build_dump_config_command(kwargs["request"], None)
    assert recorded[2] == commands.build_cuda_probe_command()
    assert prepared.train_command == commands.build_train_command(
        commands.build_train_argv(kwargs["request"], None))
    assert prepared.heartbeat is not None


def test_preflight_failure_stops_heartbeat_then_writes_build_failed(tmp_path):
    order: list[str] = []
    status_at_stop: list[str] = []

    def start_heartbeat(**_kwargs):

        def stop_and_join():
            order.append("stop")
            status_path = kwargs["attempt"].run_root / mrl.STATUS_FILENAME
            status_at_stop.append(json.loads(status_path.read_text())["status"])

        return SimpleNamespace(stop_and_join=stop_and_join)

    def fake_run(cmd, **_kwargs):
        if list(cmd)[:3] == ["/usr/local/bin/uv", "pip", "install"]:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    kwargs = _preflight_kwargs(tmp_path, run=fake_run, start_heartbeat=start_heartbeat)
    with pytest.raises(subprocess.CalledProcessError):
        mrl.prepare_remote_source(**kwargs)
    persisted = json.loads((kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_text())
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
    # Bound on purpose (the resource-acquisition rule in prepare_remote_source's PITFALL): the
    # live traceback keeps the failing frame, and with it the TemporaryDirectory whose finalizer
    # would otherwise erase the extracted tree when the block exits, so the cleanup assertion
    # below could not see a skipped cleanup.
    with pytest.raises(subprocess.CalledProcessError) as excinfo:      # noqa: F841
        mrl.prepare_remote_source(**kwargs)
    persisted = json.loads((kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "build_failed"
    assert persisted["attempt_id"] == "attempt-a"
    ephemeral = kwargs["host"].ephemeral_parent
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
    kwargs_run_root = kwargs["attempt"].run_root
    prepared = mrl.prepare_remote_source(**kwargs)
    assert all(env["WANDB_API_KEY"] == secret for env in recorded_envs)
    assert prepared.child_env["WANDB_API_KEY"] == secret
    assert secret not in repr(prepared)
    for path in kwargs["attempt"].run_root.rglob("*"):
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
    kwargs_run_root = kwargs["attempt"].run_root
    volume = kwargs["attempt"].volume
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
        json.loads((kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_text()))
    derived = state.derive_status(status, now=clock.now())
    assert derived.stale is False
