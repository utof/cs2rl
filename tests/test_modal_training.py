"""Behavior tests for scripts.modal_runner.training: the attempt, publication, supervision.

One of the per-module runner test files (RUNNER_TEST_FILES in
tests/modal_runner_tables.py). Before you add, move or delete a test here, or
add a helper, read THE PLACEMENT RULE FOR RUNNER TESTS in
tests/test_modal_packaging.py: which file a test belongs in, what the change
costs in the seam manifest, and where helpers go.

THE KILL SEAM. A test that drives the training attempt takes its arguments
from `_training_kwargs`, whose ProcessControl always has fake `spawn`,
`getpgid` and `killpg` (a client test's execute wrapper builds its own the
same way). Never hand the attempt the real functions and never read
`ProcessControl.system`: a test that leaves `process` out meets the autouse
tripwire in tests/conftest.py, which makes `system()` raise under pytest.
`test_kill_seam_static_safety` checks these rules by AST and says why each
exists (its clauses live in `_KillSeamClauses`); read it, and the tripwire,
before you touch the seam. A test that fires a signal handler from the test
thread does so through `_interrupt_in_production_order`, which releases the
held child only after the last handler returns (production order, gh#243),
unless the test is a key of `_SIGNAL_HOOKS_HANDWRITTEN_ALLOWLIST` (the two
grace tests call `term_handler` by hand with a releasing `killpg`);
`test_signal_tests_fire_handlers_in_production_order` enforces that by AST.

Deterministic patch-binding controls live in test_modal_patch_bindings.py.
The interruption tests here assert what the attempt reads and commits, and
`test_interrupt_without_sidecar_leaves_checkpoint_hash_null` carries a positive
control for the terminal-validator patch (gh#211).
"""
import ast
import dataclasses
import inspect
import io
import json
import os
import signal
import sys
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
from scripts.modal_runner import core, state, training                                   # noqa: E402, I001
from tests.modal_patch_binding_campaign import binding_target                            # noqa: E402, I001
from tests.modal_test_helpers import (                                                   # noqa: E402
    FakeChild, FakeRegistry, _aware, _make_manifest, _minimal_completed_tree, _no_torch,
    _noop_heartbeat)

# ── Run result: the explicit result schema ─────────────────────────────────


def test_run_result_schema_is_explicit():
    result = training.RunResult(
        schema_version=1,
        status=core.Status.COMPLETED,
        exit_code=0,
        started_at="2026-08-13T12:00:00+00:00",
        finished_at="2026-08-13T12:01:00+00:00",
        artifact_root="/artifacts/runs/ok-id",
        checkpoint_sha256="a" * 64,
        metrics_row_count=2,
        last_step=29_982_720,
    )
    payload = result.to_dict()
    assert payload["schema_version"] == 1
    assert payload["status"] == "completed"
    assert payload["exit_code"] == 0
    assert payload["checkpoint_sha256"] == "a" * 64
    assert payload["last_step"] == 29_982_720


# ── Spawn: new session, tee to the log sink, redelivery ────────────────────


def _prepared_source(tmp_path: Path, **overrides) -> core.PreparedSource:
    source_dir = tmp_path / "src"
    source_dir.mkdir(exist_ok=True)
    prepared = core.PreparedSource(
        source_dir=source_dir,
        child_env={
            "PATH": "/usr/bin",
            "OMP_NUM_THREADS": "1"
        },
        train_command=["/opt/cs2rl/.venv/bin/python", "-m", "cs2rl.train", "--train"],
        heartbeat=None,
        config_hash="d" * 64,
    )
    for key, value in overrides.items():
        setattr(prepared, key, value)
    return prepared


def _advance_to_building(run_root, attempt_id="attempt-a", *, lock):
    state.transition_status(run_root,
                            core.Status.PREPARING,
                            now=_aware(),
                            attempt_id=attempt_id,
                            lock=lock)
    return state.transition_status(run_root,
                                   core.Status.BUILDING,
                                   now=_aware(),
                                   attempt_id=attempt_id,
                                   lock=lock)


def _training_kwargs(tmp_path: Path, **overrides) -> dict[str, Any]:
    """execute_training_attempt's keyword arguments, from flat overrides, run_root in BUILDING.

    Call sites pass flat keys (`child=`, `now=`, `killpg=`, ...); this
    assembles them into the collaborators the attempt takes (gh#163 W5):
    `attempt` (an AttemptContext over a commit-only Volume and a Clock),
    `prepared`, `registry`, `process` (a ProcessControl) and `log_sink`, plus
    `manifest` and `timeout` only when overridden, so an absent one keeps the
    attempt's own default. The private keys `_launches` (the default spawn's
    record), `_child` and `_kills` (the default killpg's record, `(pgid, sig)`
    pairs) must be popped before the call: `_consume_training_kwargs` pops all
    three, and a test that pops by hand pops each.

    SAFETY: `process` ALWAYS has fake `spawn`, `getpgid` and `killpg`. `spawn`
    records and returns `child`; `getpgid` is the identity; `killpg` only
    records, into `_kills`. `install_signal` is NOT always a fake: it is the
    override `signal_signal` or the real `signal.signal`, which the real-SIGTERM
    test needs; it installs only the attempt's own handlers, and `release`
    restores the previous ones. So nothing built here reaches a real process
    group, and a test that drops `process` from the result meets the
    tests/conftest.py tripwire instead of the real functions (only
    `test_process_control_tripwire_guards_the_resolution_path` does so, on
    purpose). The default child has FakeChild's default pid, which
    tests/conftest.py checks once per session is not the session's own process
    group: the guard refuses that group, so a `kills == []` here would pass
    without testing anything.

    Every known key is taken with `overrides.pop`, and a leftover raises
    TypeError: a misspelt or retired key (`start_heartbeat`) stays loud, as it
    was when the flat dict went straight to the attempt. The mapping from key
    to field is hand-written, so `test_training_kwargs_routes_every_override`
    checks that every key a call site passes reaches its field; a key popped
    here and then dropped fails there instead of quietly testing the default.
    The `child` and `prepared` defaults are built only when not overridden, so
    an override makes the builder build no unused FakeChild or `src/`. The
    result is typed `dict[str, Any]` because tests reach test-double members
    through it and store replacements into it (`kwargs["process"] =
    dataclasses.replace(kwargs["process"], spawn=...)`).
    """
    known = ("child", "commit", "getpgid", "killpg", "log_sink", "manifest", "now", "prepared",
             "signal_signal", "sleep", "timeout", "wait")
    given = {key: overrides.pop(key) for key in known if key in overrides}
    if overrides:
        raise TypeError(f"unknown override(s): {sorted(overrides)}")
    run_root = tmp_path / "run"
    run_root.mkdir(exist_ok=True)
    lock = threading.Lock()
    _advance_to_building(run_root, lock=lock)
    child = given["child"] if "child" in given else FakeChild(stdout=b"ok\n")
    launches: list[tuple[tuple, dict]] = []
    kills: list[tuple[int, int]] = []

    def default_factory(*args, **kwargs):
        launches.append((args, kwargs))
        return child

    class Volume:
        """Commit-only: the attempt commits the Volume and never reloads it."""

        def __init__(self, commit):
            self.commit = commit

        def reload(self):
            raise RuntimeError("training must not reload the Volume")

    process = training.ProcessControl(
        spawn=default_factory,
        getpgid=given.get("getpgid", lambda pid: pid),
        killpg=given.get("killpg", lambda pgid, sig: kills.append((pgid, sig))),
        install_signal=given.get("signal_signal", signal.signal),
    )
    clock = core.Clock(
        now=given.get("now", lambda: _aware()),
        sleep=given.get("sleep", lambda _seconds: None),
        wait=given.get("wait", core._event_wait),
    )
    attempt = core.AttemptContext(
        attempt_id="attempt-a",
        run_root=run_root,
        lock=lock,
        volume=Volume(given.get("commit", lambda: None)),
        clock=clock,
    )
    kwargs: dict[str, Any] = {
        "attempt": attempt,
        "prepared": given["prepared"] if "prepared" in given else _prepared_source(tmp_path),
        "registry": FakeRegistry(),
        "process": process,
        "log_sink": given.get("log_sink", io.StringIO()),
    }
    kwargs.update({key: given[key] for key in ("manifest", "timeout") if key in given})
    kwargs["_launches"] = launches
    kwargs["_child"] = child
    kwargs["_kills"] = kills
    return kwargs


def test_training_kwargs_routes_every_override(tmp_path):
    """Every flat key a call site passes to `_training_kwargs` reaches the field the attempt reads.

    gh#163 W5. The builder assembles flat overrides into collaborators by
    hand-written code, and several tests assert that something is ABSENT from
    a recorder they injected (`kills == []`, `sleeps == []`); such an
    assertion goes vacuous, still green, if the builder stops routing its key.
    So, both ways:
      * the keys call sites pass, enumerated by AST over the modal test files,
        must equal the keys of `routes`. A call is read through the bare name,
        an attribute `x._training_kwargs`, or an `import ... as` or plain
        `name = ...` alias. A key a call site passes that `routes` lacks fails,
        and so does a `routes` entry that no call site passes;
      * each key, passed as a sentinel, must come back by identity at the
        field `routes` names (the collaborator field that replaced the flat
        key). No key is converted. One route is weaker than the rest:
        `child` is read back from the builder's own echo `_child`, not from
        what `process.spawn` returns, because calling the spawn factory here
        would run a builder fake (see SAFETY). A builder whose spawn returned
        some other child would pass this test; the attempt tests that assert
        on their own child fail instead.
    A call site whose keys cannot be read statically (a `**` splat, a second
    positional argument) fails too, and so does any other reference to the
    builder: its name loaded anywhere but as a callee or a plain alias's value
    (`functools.partial(_training_kwargs, ...)`, a tuple assignment), or the
    name as a string (`getattr(module, "_training_kwargs")`). RESIDUAL: a name
    computed at run time (a concatenated string, `vars()` with a variable key)
    is not seen. This test's own calls and strings are not call sites.

    SAFETY: the sentinels land in a ProcessControl and a Clock that are built
    and read back, never called; the builder's own recording fakes are never
    called here either. A builder that pops `sleep` and then drops it is
    caught here (and by the watcher tests that pass `sleep=`), not by the
    grace-period test, whose `15.0 not in hooks["slept"]` holds either way: a
    FakeChild has `wait`, so the grace period never reaches `sleep`.

    THE PLANTS are synthetic call sites, parsed and never run, each passing an
    unmapped key through one spelling: each must fail the key equality or be
    reported as a problem, or the enumeration would not be evidence. Keep
    `routes` and the plants INSIDE this function: a module-level name in this
    file is a governed seam name and moves GOVERNED_NAME_COUNT
    (tests/test_modal_packaging.py). The enumerator (`last_name`,
    `call_site_keys`) is duplicated in `test_preflight_kwargs_routes_every_override`
    (tests/test_modal_preflight.py) for the same reason, so this test asserts
    the two copies are AST-equal (`ast.dump`, docstrings included): change
    both together, or this goes red. A planted one-token edit of the other
    copy must make them differ, or the comparison would not be evidence.
    """
    routes = {
        "child": lambda built: built["_child"],
        "commit": lambda built: built["attempt"].volume.commit,
        "getpgid": lambda built: built["process"].getpgid,
        "killpg": lambda built: built["process"].killpg,
        "log_sink": lambda built: built["log_sink"],
        "manifest": lambda built: built["manifest"],
        "now": lambda built: built["attempt"].clock.now,
        "prepared": lambda built: built["prepared"],
        "signal_signal": lambda built: built["process"].install_signal,
        "sleep": lambda built: built["attempt"].clock.sleep,
        "timeout": lambda built: built["timeout"],
        "wait": lambda built: built["attempt"].clock.wait,
    }
    builder = "_training_kwargs"
    this_test = "test_training_kwargs_routes_every_override"

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
    assert {"tests/test_modal_training.py",
            "tests/test_modal_patch_bindings.py"} <= set(sources), sorted(sources)[:5]

    mirror_file = "tests/test_modal_preflight.py"
    mirror_test = "test_preflight_kwargs_routes_every_override"

    def enumerator_dumps(text, test_name):
        """{name: ast.dump} of the enumerator's two functions nested in `test_name`."""
        test = next(top for top in ast.parse(text).body
                    if isinstance(top, ast.FunctionDef) and top.name == test_name)
        return {
            node.name: ast.dump(node)
            for node in test.body
            if isinstance(node, ast.FunctionDef) and node.name in ("last_name", "call_site_keys")
        }

    ours = enumerator_dumps(sources["tests/test_modal_training.py"], this_test)
    assert set(ours) == {"last_name", "call_site_keys"}, sorted(ours)
    assert enumerator_dumps(sources[mirror_file], mirror_test) == ours, (
        f"the enumerator in {mirror_file}::{mirror_test} is no longer AST-equal to this test's "
        "`last_name`/`call_site_keys`: change both copies together")
    anchor = "if len(node.args) != 1 or"
    assert sources[mirror_file].count(anchor) == 1, f"plant anchor {anchor!r} is not unique"
    planted_mirror = sources[mirror_file].replace(anchor, "if len(node.args) != 2 or")
    assert enumerator_dumps(planted_mirror, mirror_test) != ours, (
        "a one-token edit of the other copy left the two enumerators equal, so the comparison "
        "above is not evidence")

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
        "key needs a field in `_training_kwargs` and an entry here; a key nobody passes any "
        "more comes out of both")

    plants = {
        "the bare name":
        "_training_kwargs(tmp_path, start_heartbeat=f)\n",
        "an attribute":
        "training_tests._training_kwargs(tmp_path, start_heartbeat=f)\n",
        "an import alias": ("from tests.test_modal_training import _training_kwargs as build\n"
                            "build(tmp_path, start_heartbeat=f)\n"),
        "an assignment alias":
        "build = _training_kwargs\nbuild(tmp_path, start_heartbeat=f)\n",
        "functools.partial":
        "build = functools.partial(_training_kwargs, start_heartbeat=f)\nbuild(tmp_path)\n",
        "getattr by name":
        "getattr(training_tests, '_training_kwargs')(tmp_path, start_heartbeat=f)\n",
        "a tuple-assignment alias":
        "build, _ = _training_kwargs, None\nbuild(tmp_path, start_heartbeat=f)\n",
    }
    for plant, source in plants.items():
        planted, planted_problems = call_site_keys({**sources, "tests/test_modal_plant.py": source})
        assert planted_problems or set(planted) != set(routes), (
            f"a call site passing an unmapped key through {plant} left the key sets equal and "
            "reported no problem, so the enumeration cannot see that spelling and its green is "
            "not evidence")
    _, splat = call_site_keys({"tests/test_modal_plant.py": "_training_kwargs(tmp_path, **k)\n"})
    assert splat, "a `**` splat call site was not reported, so its keys would go unchecked"

    sentinels: dict[str, object] = {key: object() for key in routes}
    built = _training_kwargs(tmp_path, **sentinels)
    assert set(built) == {
        "attempt", "prepared", "registry", "process", "log_sink", "manifest", "timeout",
        "_launches", "_child", "_kills"
    }, sorted(built)
    for key, route in routes.items():
        assert route(built) is sentinels[key], f"{key} does not reach its field"
    # A directory of its own, so a builder that stopped raising would build there and fail on
    # DID NOT RAISE, rather than on colliding with the build above.
    leftover = tmp_path / "leftover"
    leftover.mkdir()
    with pytest.raises(TypeError, match=r"unknown override\(s\): \['start_heartbeat'\]"):
        _training_kwargs(leftover, start_heartbeat=_noop_heartbeat)


def test_training_child_starts_in_new_session_without_shell(tmp_path):
    kwargs = _training_kwargs(tmp_path)
    launches = kwargs.pop("_launches")
    kwargs.pop("_child")
    kills = kwargs.pop("_kills")
    prepared = kwargs["prepared"]
    mrl.execute_training_attempt(**kwargs)
    assert len(launches) == 1
    args, kw = launches[0]
    command = args[0] if args else kw.get("args")
    assert list(command) == prepared.train_command
    assert kw["start_new_session"] is True
    assert kw["shell"] is False
    assert kw["cwd"] == os.fspath(prepared.source_dir)
    assert kw["env"] == prepared.child_env
    # A normal exit must signal nothing: the child has already exited. A finalize that killed
    # on every path would record a SIGTERM to the fake pid here.
    assert kills == []


def test_stdout_stderr_are_teed_to_log_sink_without_truncation(tmp_path):
    payload_out = ("OUT" + ("x" * 200_000) + "END\n").encode()
    payload_err = ("ERR" + ("y" * 200_000) + "FIN\n").encode()
    child = FakeChild(stdout=payload_out, stderr=payload_err)

    class CaptureSink:

        def __init__(self):
            self.parts: list[str] = []

        def write(self, data):
            self.parts.append(data)

        def flush(self):
            return None

        def close(self):
            return None

        def getvalue(self):
            return "".join(self.parts)

    sink = CaptureSink()
    kwargs = _training_kwargs(tmp_path, child=child, log_sink=sink)
    kwargs.pop("_launches")
    kwargs.pop("_child")
    kwargs.pop("_kills")
    (kwargs["attempt"].run_root / "train.log").write_text("already here\n")
    mrl.execute_training_attempt(**kwargs)
    text = sink.getvalue()
    assert "OUT" in text and "END" in text
    assert "ERR" in text and "FIN" in text
    assert text.count("x") == 200_000
    assert text.count("y") == 200_000
    leftover = (kwargs["attempt"].run_root / "train.log").read_text()
    assert leftover.startswith("already here\n")
    assert leftover.count("x") == 200_000
    assert leftover.endswith("FIN\n") or "FIN\n" in leftover
    assert leftover.count("y") >= 200_000


def test_same_attempt_redelivery_invokes_subprocess_once(tmp_path):
    commits: list[str] = []
    child = FakeChild(stdout=b"first-delivery\n")
    kwargs = _training_kwargs(
        tmp_path,
        child=child,
        commit=lambda: commits.append("commit"),
    )
    launches = kwargs.pop("_launches")
    kwargs.pop("_child")
    kwargs.pop("_kills")
    first = mrl.execute_training_attempt(**kwargs)
    assert first != mrl.REDELIVERED
    assert len(launches) == 1
    status_after_first = (kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_bytes()
    commits_after_first = list(commits)

    def must_not_launch(*_args, **_kwargs):
        raise AssertionError("redelivered container must not start training")

    kwargs["process"] = dataclasses.replace(kwargs["process"], spawn=must_not_launch)
    second = mrl.execute_training_attempt(**kwargs)
    assert second == mrl.REDELIVERED
    assert len(launches) == 1
    assert (kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_bytes() == status_after_first
    assert commits == commits_after_first


# ── Training loop: heartbeat and checkpoint commits ────────────────────────


class _FakeClock:

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


def _consume_training_kwargs(kwargs):
    kwargs.pop("_launches", None)
    kwargs.pop("_child", None)
    kwargs.pop("_kills", None)
    return kwargs


def _run_attempt_in_thread(kwargs, *, daemon: bool = False):
    """Run `execute_training_attempt(**kwargs)` on a thread; returns (thread, finished, boxed).

    `boxed` receives the result, or the Exception the attempt raised. `daemon`
    is for a test whose knock-out DEADLOCKS the attempt (gh#238 P2: a signal
    inside the once-gate): a stranded non-daemon thread would hold the pytest
    process open at exit, a daemon one is abandoned, and the test goes red at
    its `finished.wait(...)` instead of hanging the session.
    """
    finished = threading.Event()
    boxed: list[object] = []

    def runner():
        try:
            boxed.append(mrl.execute_training_attempt(**kwargs))
        except Exception as err:
            boxed.append(err)
        finally:
            finished.set()

    thread = threading.Thread(target=runner, daemon=daemon)
    thread.start()
    return thread, finished, boxed


def test_heartbeat_commits_every_60s_while_training(tmp_path):
    clock = _FakeClock()
    child = FakeChild(hold=True)
    beat_times: list = []

    def commit():
        beat_times.append(clock.now())

    def wait(event: threading.Event, seconds: float) -> bool:
        clock.advance(seconds)
        return event.wait(0.01)

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            commit=commit,
            now=clock.now,
            wait=wait,
        ))
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        deadline = time.monotonic() + 5.0
        while len(beat_times) < 6 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(beat_times) >= 6
        for earlier, later in zip(beat_times, beat_times[1:], strict=False):
            assert later - earlier <= timedelta(seconds=60)
    finally:
        child.release()
        assert finished.wait(timeout=2.0)
        thread.join(timeout=2.0)
        assert not thread.is_alive()


def _write_policy_checkpoint(run_root: Path, value: float) -> Path:
    import torch

    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "dust2_policy.pt"
    torch.save({"weight": torch.tensor([value])}, ckpt)
    return ckpt


def test_stable_checkpoint_gets_sidecar_and_joint_commit(tmp_path):
    child = FakeChild(hold=True)
    events: list[tuple] = []
    settle_seen = threading.Event()

    def commit():
        sidecar = kwargs["attempt"].run_root / "checkpoints" / "dust2_policy.pt.meta.json"
        ckpt = kwargs["attempt"].run_root / "checkpoints" / "dust2_policy.pt"
        events.append(("commit", sidecar.is_file(), ckpt.is_file()))

    def fake_sleep(seconds: float) -> None:
        if seconds >= 1.0 and ckpt.is_file() and not settle_seen.is_set():
            settle_seen.set()
            _write_policy_checkpoint(kwargs["attempt"].run_root, 2.0)

    kwargs = _consume_training_kwargs(
        _training_kwargs(tmp_path, child=child, commit=commit, sleep=fake_sleep))
    ckpt = _write_policy_checkpoint(kwargs["attempt"].run_root, 1.0)
    first_digest = core.sha256_file(ckpt)
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    sidecar = kwargs["attempt"].run_root / "checkpoints" / "dust2_policy.pt.meta.json"
    try:
        deadline = time.monotonic() + 5.0
        while not sidecar.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sidecar.is_file()
        meta = json.loads(sidecar.read_text())
        stable_digest = core.sha256_file(ckpt)
        assert stable_digest != first_digest
        assert meta["sha256"] == stable_digest
        assert meta["size"] == ckpt.stat().st_size
        assert meta["mtime_ns"] == ckpt.stat().st_mtime_ns
        assert meta["validated_at"]
        assert any(kind == "commit" and has_side and has_ckpt
                   for kind, has_side, has_ckpt in events)
    finally:
        child.release()
        assert finished.wait(timeout=2.0)
        thread.join(timeout=2.0)


def test_torn_checkpoint_does_not_publish_sidecar(tmp_path):
    child = FakeChild(hold=True)
    settle_calls = threading.Event()

    def fake_sleep(seconds: float) -> None:
        if seconds >= 1.0:
            settle_calls.set()

    kwargs = _consume_training_kwargs(_training_kwargs(tmp_path, child=child, sleep=fake_sleep))
    ckpt_dir = kwargs["attempt"].run_root / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "dust2_policy.pt").write_bytes(b"torn-not-a-checkpoint")
    sidecar = ckpt_dir / "dust2_policy.pt.meta.json"
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        assert settle_calls.wait(timeout=2.0)
        time.sleep(0.05)
        assert not sidecar.exists()
    finally:
        child.release()
        assert finished.wait(timeout=2.0)
        thread.join(timeout=2.0)


def test_interrupt_publishes_sidecar_after_unstable_live_saves(tmp_path):
    """Live PufferLib rewrites dust2_policy.pt every epoch (~0.5s).

    The 1s settle window never elapses while the child is alive. After SIGINT
    the file is stable and the attempt must still publish the sidecar, or
    resume cannot validate the parent. On INTERRUPTED `finalize` itself does
    not publish: the post-finalize retry is the publish the attempt
    guarantees (the watcher may get one in before `finalize` stops it).
    Production order (see `_signal_hooks`, ORDER).
    """
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    rewrites = {"n": 0}

    def fake_sleep(seconds: float) -> None:
        hooks["sleep"](seconds)
        if seconds >= 1.0 and child.poll() is None:
            rewrites["n"] += 1
            _write_policy_checkpoint(kwargs["attempt"].run_root, float(rewrites["n"]))

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=fake_sleep,
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    ckpt = _write_policy_checkpoint(kwargs["attempt"].run_root, 0.0)
    sidecar = kwargs["attempt"].run_root / "checkpoints" / "dust2_policy.pt.meta.json"

    def premise(_int_handler, _term_handler):
        # Two unstable live saves happened (`armed`) and none published: the
        # sidecar this test then asserts is the retry's, not a watcher's.
        assert not sidecar.exists()

    _interrupt_in_production_order(kwargs,
                                   child,
                                   hooks,
                                   signal.SIGINT,
                                   armed=lambda: rewrites["n"] >= 2,
                                   inspect=premise)
    assert json.loads(
        (kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_text())["status"] == "interrupted"
    deadline = time.monotonic() + 2.0
    while not sidecar.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sidecar.is_file()
    meta = json.loads(sidecar.read_text())
    assert meta["sha256"] == core.sha256_file(ckpt)
    assert meta["size"] == ckpt.stat().st_size


# ── Runner-interpreter checkpoint validation: publish and interrupt ────────
#
# The Modal runner process and the training child are DIFFERENT interpreters.
# The runner is the image's standalone /usr/local/bin/python (only uv + modal);
# torch lives exclusively in the PREBUILT_PYTHON venv that runs train.py.
# Verified in a live container on 2026-08-14:
#   runner_executable=/usr/local/bin/python  runner_torch=MISSING
# Every test above runs on a laptop where `import torch` succeeds, so none of
# them can see this. These do: they force the torch-less runner condition.


def _publish(run_root: Path):
    commits: list[int] = []
    outcome = training.publish_stable_checkpoint(
        run_root,
        now=_aware,
        commit=lambda: commits.append(1),
        sleep=lambda _seconds: None,
        last_published=None,
    )
    return outcome, commits


def test_publish_validates_via_prebuilt_interpreter_when_runner_lacks_torch(tmp_path, monkeypatch):
    """A valid checkpoint must still publish when the runner cannot import torch."""
    run_root = tmp_path / "run"
    run_root.mkdir()
    ckpt = _write_policy_checkpoint(run_root, 1.0)
    _no_torch(monkeypatch, prebuilt=sys.executable)

    outcome, commits = _publish(run_root)

    sidecar = ckpt.with_name("dust2_policy.pt.meta.json")
    assert sidecar.is_file()
    assert outcome.reason is None
    assert outcome.generation == (ckpt.stat().st_mtime_ns, ckpt.stat().st_size)
    assert len(commits) == 1
    assert json.loads(sidecar.read_text())["sha256"] == core.sha256_file(ckpt)


def test_prebuilt_validation_still_rejects_a_torn_checkpoint(tmp_path, monkeypatch):
    """The fallback must not become a rubber stamp: garbage still fails to load."""
    run_root = tmp_path / "run"
    (run_root / "checkpoints").mkdir(parents=True)
    (run_root / "checkpoints" / "dust2_policy.pt").write_bytes(b"torn-not-a-checkpoint")
    _no_torch(monkeypatch, prebuilt=sys.executable)

    outcome, commits = _publish(run_root)

    assert not (run_root / "checkpoints" / "dust2_policy.pt.meta.json").exists()
    assert outcome.generation is None
    assert "not weights-only loadable" in outcome.reason
    assert commits == []


def test_publish_reason_names_the_missing_interpreter(tmp_path, monkeypatch):
    """No torch and no prebuilt venv: skipping is fine, skipping SILENTLY is not."""
    run_root = tmp_path / "run"
    run_root.mkdir()
    _write_policy_checkpoint(run_root, 1.0)
    _no_torch(monkeypatch, prebuilt=str(tmp_path / "nonexistent" / "python"))

    outcome, commits = _publish(run_root)

    assert not (run_root / "checkpoints" / "dust2_policy.pt.meta.json").exists()
    assert "nonexistent" in outcome.reason
    assert commits == []


def test_interrupt_without_publishable_checkpoint_writes_a_reason_file(tmp_path, monkeypatch):
    """The attempt must leave evidence on the Volume, before a commit, of WHY there
    is no sidecar. Three T4 runs were burned on a silently swallowed skip.

    On INTERRUPTED, `finalize` does not publish, so the note is written by the
    post-finalize retry, and only the retry's own commit (`commit_note=True`)
    carries it to the Volume. Production order (see `_signal_hooks`): with the
    child released inside the handler, the retry could write the note before
    the terminal commit, and that commit would carry it even if the retry's
    own commit were gone. `test_publish_note_reaches_the_volume_in_the_right_commit`
    pins which commit carries the note on every exit path."""
    _no_torch(monkeypatch, prebuilt=str(tmp_path / "nonexistent" / "python"))
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    commits: list[bool] = []

    def commit() -> None:
        commits.append((kwargs["attempt"].run_root / "checkpoints" /
                        core.CHECKPOINT_PUBLISH_REASON_NAME).is_file())

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            commit=commit,
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    _write_policy_checkpoint(kwargs["attempt"].run_root, 1.0)
    _interrupt_in_production_order(kwargs, child, hooks, signal.SIGINT)

    reason_path = kwargs["attempt"].run_root / "checkpoints" / core.CHECKPOINT_PUBLISH_REASON_NAME
    assert reason_path.is_file()
    payload = json.loads(reason_path.read_text())
    assert "nonexistent" in payload["reason"]
    assert payload["at"]
    # Written BEFORE a commit, or it never reaches the Volume.
    assert any(commits)


def test_interrupt_commits_status_even_if_prebuilt_load_hangs(tmp_path, monkeypatch):
    """SIGINT finalize must persist STATUS before any hung PREBUILT_PYTHON load.

    Modal preemption grace is ~30s and Function-timeout slack is seconds. A
    120s weights-only load inside finalize can lose both sidecar and STATUS.
    The watcher thread is exempt so its 50ms poll cannot stall this test.
    """
    release = threading.Event()

    def hanging_load(_path):
        if threading.current_thread().name == "cs2rl-checkpoint-watch":
            raise mrl.ValidationError("watcher must not hang the interrupt path")
        if not release.wait(timeout=10.0):
            raise mrl.ValidationError("test timed out waiting to release the hung load")

    monkeypatch.setattr(*binding_target("interrupt-loader"), hanging_load)
    child = FakeChild(hold=True)
    # Opt-out (see `_SIGNAL_HOOKS_RELEASE_ALLOWLIST`): the child dies at SIGKILL.
    hooks = _signal_hooks(child, release_on=signal.SIGKILL)
    commits: list[str | None] = []

    def commit() -> None:
        status_path = kwargs["attempt"].run_root / mrl.STATUS_FILENAME
        if not status_path.is_file():
            commits.append(None)
            return
        commits.append(json.loads(status_path.read_text())["status"])

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            commit=commit,
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    _write_policy_checkpoint(kwargs["attempt"].run_root, 1.0)
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, _term = _wait_until_handlers(hooks["installed"], hooks["originals"])
        handler_thread = threading.Thread(target=int_handler,
                                          args=(signal.SIGINT, None),
                                          daemon=True)
        handler_thread.start()
        deadline = time.monotonic() + 2.0
        interrupted = False
        while time.monotonic() < deadline:
            status_path = kwargs["attempt"].run_root / mrl.STATUS_FILENAME
            if (status_path.is_file()
                    and json.loads(status_path.read_text())["status"] == "interrupted"):
                interrupted = True
                break
            time.sleep(0.01)
        assert interrupted, "STATUS must become interrupted while the prebuilt load is still hung"
        assert "interrupted" in commits
    finally:
        release.set()
        child.release()
        assert finished.wait(timeout=2.0)
        thread.join(timeout=2.0)


def test_checkpoint_watcher_threads_generation_into_last_published(tmp_path, monkeypatch):
    """A watcher-only typo on PublishOutcome.generation is swallowed every 50ms.

    Direct publish tests cannot see that: they assert .generation on the
    function return, not on the value the thread feeds back as last_published.
    """
    seen: list[tuple[int, int] | None] = []
    generation = (111, 222)

    def fake_publish(*_args, last_published=None, **_kwargs):
        seen.append(last_published)
        return training.PublishOutcome(generation)

    monkeypatch.setattr(*binding_target("watcher-publisher"), fake_publish)
    stop, watcher = training._start_checkpoint_watcher(
        run_root=tmp_path,
        now=_aware,
        commit=lambda: None,
        sleep=lambda _seconds: None,
    )
    try:
        deadline = time.monotonic() + 2.0
        while len(seen) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(seen) >= 2
        assert seen[0] is None
        assert seen[1] == generation
    finally:
        stop.set()
        watcher.join(timeout=2.0)


# ── Attempt supervision: SIGINT / KeyboardInterrupt / SIGTERM cleanup ──────


def _signal_hooks(child, *, release_on=None):
    """Recording fakes for the attempt's signal seam: handlers, killpg, getpgid, sleep.

    `fake_getpgid` is the identity, so the child passes the `_signal_process_group`
    guard's `pgid!=pid` clause, and `fake_killpg` records instead of signalling.

    The assertion below is the own-group precondition. The guard also
    refuses the runner's OWN process group, so if this session's group
    equalled the child's pid every kill would be refused: a test asserting a
    kill would fail for a reason unrelated to its subject, and one asserting
    that nothing was killed would pass without testing anything. It is not
    the only copy: tests/conftest.py checks the same precondition once per
    session for FakeChild's default pid, the pid of every child the modal
    tests hand a recording killpg (this helper's, `_training_kwargs`'
    defaults, the client wrappers'), and this repeats it for the child it is
    given, whatever its pid. (The process-group guard test,
    `test_signal_process_group_refuses_groups_a_live_child_cannot_have`, calls
    `_signal_process_group` directly and builds its own pids, apart from the
    session's group by construction.)

    ORDER (gh#211). In production the handler runs on the attempt's own (main)
    thread, so the wait loop is suspended until `finalize` returns: the
    attempt's tail (`finish`, the post-finalize publish retry, then
    `release()`, which restores the handlers, stops the watcher and the
    heartbeat, and closes train.log) always runs after the terminal STATUS,
    result.json and commit. A test that calls the handler from the test thread
    while the attempt waits on another thread loses that order if `killpg`
    releases the child: a `killpg` that releases at SIGKILL wakes the attempt
    thread in the middle of `finalize`, and its tail then races the rest of
    `finalize`. So the default is `release_on=None`: `killpg` records and
    releases nothing. A test that fires a handler from the test thread uses
    `_interrupt_in_production_order`, which releases the child only after the
    last handler returns. That is production order.
      * In production order the retry's commit follows every commit
        `finalize` makes. An assertion that means a commit inside `finalize`
        must single that commit out (by the STATUS it carries, say): "some
        commit after X" is satisfied by the retry's. Tighten such an
        assertion BEFORE converting its test, or the conversion quietly
        removes what it caught.
      * A releasing `killpg` is passed explicitly (`release_on=<signal>`) only
        by the tests in `_SIGNAL_HOOKS_RELEASE_ALLOWLIST`, where it models a
        child that dies at that signal; the static census
        `test_signal_tests_fire_handlers_in_production_order` enforces the
        list.
      * Where `finalize` runs on the attempt's own thread, `release_on` is
        inert, because the tail follows `finalize` whatever `killpg` does:
        `test_keyboard_interrupt_uses_same_cleanup`,
        `test_a_signal_while_taking_the_once_gate_returns_at_once`, the
        timeout half of `test_dead_run_and_timeout_have_distinct_reasons`, and
        every case of `test_publish_note_reaches_the_volume_in_the_right_commit`
        but `signal`.
    Production order is the default here (gh#243).
    """
    assert os.getpgrp() != child.pid, (
        f"the runner's own process group ({os.getpgrp()}) equals the fake child's pid "
        f"({child.pid}), so the process-group guard in `_signal_process_group` refuses to "
        "signal it by design and this kill-path test cannot run in this session")
    originals = {signal.SIGINT: object(), signal.SIGTERM: object()}
    installed: dict[int, object] = dict(originals)
    kills: list[int] = []
    slept: list[float] = []

    def fake_signal(sig, handler):
        previous = installed.get(sig, originals.get(sig))
        installed[sig] = handler
        return previous

    def fake_getpgid(pid):
        return pid

    def fake_killpg(_pgid, sig):
        kills.append(sig)
        if release_on is not None and sig == release_on:
            child.release()

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    return {
        "originals": originals,
        "installed": installed,
        "kills": kills,
        "slept": slept,
        "signal_signal": fake_signal,
        "getpgid": fake_getpgid,
        "killpg": fake_killpg,
        "sleep": fake_sleep,
    }


def _wait_until_handlers(installed, originals, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        int_handler = installed.get(signal.SIGINT)
        term_handler = installed.get(signal.SIGTERM)
        if (int_handler is not None and int_handler is not originals[signal.SIGINT]
                and term_handler is not None and term_handler is not originals[signal.SIGTERM]):
            return int_handler, term_handler
        time.sleep(0.01)
    raise AssertionError("signal handlers were not installed around the child")


def _interrupt_in_production_order(kwargs,
                                   child,
                                   hooks,
                                   *signals,
                                   armed=None,
                                   inspect=None,
                                   capture=False,
                                   daemon=False,
                                   timeout=2.0):
    """Run the attempt on a thread and fire `signals`' handlers from this thread in production order.

    Returns (boxed, raised). `boxed` is `_run_attempt_in_thread`'s list: the attempt's result, or
    the Exception it raised. `raised` holds what the handlers raised; it is non-empty only with
    `capture=True`.

    ORDER (see `_signal_hooks`): wait until the handlers are installed, then poll `armed()` true
    if given, then call `inspect(int_handler, term_handler)` if given, then call each handler in
    turn, then `child.release()`, then wait for the attempt. The child is released only after the
    last handler returns, so the attempt thread's tail (`finish`, the publish retry, `release()`)
    runs after `finalize`, as in production, where the handler runs on the attempt's own thread.
    Releasing the child BEFORE the first handler is not the racy order either: the child is then
    dead when `_signal_process_group` checks it after SIGTERM, so SIGKILL is skipped and `kills`
    is `[SIGTERM]` (measured: row 3 red 20 of 20 with no mutant). The racy order is a release
    INSIDE `finalize`, at the kill, which is what a non-None `release_on` does. Do not call this
    helper with hooks built by an allow-listed opt-out.

    `armed`: a predicate polled every 1 ms up to `timeout` before the first handler fires. The tee
    test passes `lambda: bool(child.wait_timeouts)`: handlers are installed BEFORE `start_tees`,
    and a handler fired earlier can finalize with no tee threads (gh#238 P1).
    `inspect`: called with the installed (SIGINT, SIGTERM) handler pair before any fires. The
    identity test RECORDS on it here, because after the helper returns `release()` has restored
    the two distinct originals, and asserts the record after the helper returns, so a helper
    that never calls `inspect` is red (knock-out K-INSPECT). The live-saves test checks its
    premise here, after `armed` and before the fire, as HEAD does.
    `capture`: a handler that raises stops the firing loop in every mode. By default the helper
    still releases the child and joins the thread, then RE-RAISES the first handler exception, so
    the test is red at the helper call, naming it. With `capture=True` it returns the exceptions in
    `raised` instead; only the hung-heartbeat test uses this, to keep its `raised == []` assertion.
    `daemon`: for a test whose knock-out deadlocks the attempt (see `_run_attempt_in_thread`).

    PITFALL: a caller whose `finally` must release something else (the tee test's streams) wraps
    this call in its own `try`/`finally`. Do not move those releases into this helper.
    """
    assert signals, "no signal to fire"
    thread, finished, boxed = _run_attempt_in_thread(kwargs, daemon=daemon)
    raised: list[BaseException] = []
    done = False
    try:
        int_handler, term_handler = _wait_until_handlers(hooks["installed"], hooks["originals"])
        if armed is not None:
            deadline = time.monotonic() + timeout
            while not armed():
                assert time.monotonic() < deadline, "the attempt never reached the armed state"
                time.sleep(0.001)
        if inspect is not None:
            inspect(int_handler, term_handler)
        handlers = {signal.SIGINT: int_handler, signal.SIGTERM: term_handler}
        for sig in signals:
            # Look up OUTSIDE the try: an unknown signal is a programming error and must raise
            # KeyError even under `capture=True`, not land in `raised`.
            handler = handlers[sig]
            try:
                handler(sig, None)
            except BaseException as err:               # noqa: BLE001 - re-raised below unless capture=True
                raised.append(err)
                break
        child.release()
        done = finished.wait(timeout=timeout)
    finally:
        child.release()
        thread.join(timeout=timeout)
    if raised and not capture:
        raise raised[0]
    assert done, "the attempt did not finish after the handler returned"
    return boxed, raised


# THE TWO ALLOW-LISTS of `test_signal_tests_fire_handlers_in_production_order`
# (gh#243). Both map a module-level test name to a one-line reason; the census
# reads this file by AST and requires every key to own a site of the kind it
# allows, so a stale name is red, not silently ignored. A key is a TEST name: a
# non-test helper wrapping an opt-out is red by construction (its owner is the
# helper, on no list). Adding a key here without the census's reason is the
# one way to reintroduce the racy order silently; write the reason first.

# Tests whose call to `_signal_hooks` passes a releasing `killpg`
# (`release_on=<signal>`). Each models a child that DIES at that signal and
# asserts nothing the attempt's tail can change, so the release inside
# `finalize` cannot race anything the test reads. None of them may also call
# `_interrupt_in_production_order`: the helper promises production order, and
# a releasing `killpg` under it is the racy order with extra steps (rule 1).
_SIGNAL_HOOKS_RELEASE_ALLOWLIST: dict[str, str] = {
    "test_child_receives_term_then_kill_after_grace":
    "the child dies at the kill; asserts the kill sequence and the grace wait",
    "test_term_grace_is_deadline_not_mandatory_sleep":
    "the child dies inside the grace window; asserts no mandatory sleep",
    "test_interrupt_commits_status_even_if_prebuilt_load_hangs":
    "the handler runs on a third thread and the retry blocks in the hung load until the "
    "test's `finally`; the release keeps the 'child exits at SIGKILL' scenario the issue names",
    "test_post_spawn_failure_kills_child_and_writes_terminal_status":
    "asserts `child.poll() is not None`; only the releasing kill makes it true",
    "test_real_sigterm_in_tee_window_never_joins_unstarted_thread":
    "the real handler runs on the main thread and the releasing kill is the only thing that "
    "ends `wait_for_exit`; the test's `finally` release comes after the attempt returns",
}

# Tests that wait for or fire the installed handlers BY HAND (a call to
# `_wait_until_handlers`, or a `hooks["installed"][sig](...)` fire, outside
# `_interrupt_in_production_order`). Each has a reason the helper does
# not fit; a test that fires from the test thread and is not here converts to
# the helper instead of joining this list.
_SIGNAL_HOOKS_HANDWRITTEN_ALLOWLIST: dict[str, str] = {
    "test_child_receives_term_then_kill_after_grace":
    "opt-out (release list): the helper promises production order, a releasing killpg is not",
    "test_term_grace_is_deadline_not_mandatory_sleep":
    "opt-out (release list): the helper promises production order, a releasing killpg is not",
    "test_interrupt_commits_status_even_if_prebuilt_load_hangs":
    "the handler runs on a third thread while the test polls STATUS from this one",
    "test_dead_run_and_timeout_have_distinct_reasons":
    "waits for installation only, fires nothing: the frozen clock ends the attempt",
    "test_a_signal_inside_finalize_does_not_finalize_again":
    "nested fire from inside `killpg`, on the thread running `finalize`, as production "
    "delivers it",
    "test_a_signal_while_taking_the_once_gate_returns_at_once":
    "fires from inside the gate double, on the attempt's thread",
}


def test_sigint_and_sigterm_share_cleanup_and_restore_handlers(tmp_path):
    """One handler object for both signals; `finalize` stops the heartbeat while STATUS is training.

    Production order (see `_signal_hooks`, ORDER): with the child released
    inside the handler, the attempt thread's `release()` also stops the
    heartbeat, racing `finalize`, and a `finalize` that no longer stopped it
    (mutant `STOP_HB_DEL`) passed 11 of 20 runs (measured 2026-09-24). That
    rate is the pre-gh#243 racy variant of this test (`_signal_hooks(child,
    release_on=signal.SIGKILL)`), not this test's own: under the
    production-order default the same mutant is red 20 of 20.
    """
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    order: list[object] = []

    def stop_and_join():
        order.append("heartbeat_stopped")
        order.append(
            json.loads((kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_text())["status"])

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    kwargs["prepared"] = _prepared_source(tmp_path,
                                          heartbeat=SimpleNamespace(stop_and_join=stop_and_join))
    seen: list[bool] = []

    def same_handler(int_handler, term_handler):
        # Recorded here, not asserted: after the helper returns, `release()` has
        # restored the two distinct originals, so the identity is only visible
        # while the handlers are installed. The one assertion below is both
        # the identity check and the proof that the helper called this once.
        seen.append(int_handler is term_handler)

    _interrupt_in_production_order(kwargs,
                                   child,
                                   hooks,
                                   signal.SIGINT,
                                   signal.SIGTERM,
                                   inspect=same_handler)
    assert seen == [True]
    assert hooks["installed"][signal.SIGINT] is hooks["originals"][signal.SIGINT]
    assert hooks["installed"][signal.SIGTERM] is hooks["originals"][signal.SIGTERM]
    persisted = json.loads((kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "interrupted"
    assert persisted["attempt_id"] == "attempt-a"
    assert order[0] == "heartbeat_stopped"
    assert order[1] == "training"
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]


def test_keyboard_interrupt_uses_same_cleanup(tmp_path):
    child = FakeChild(hold=True)

    def exploding_wait(timeout=None):
        raise KeyboardInterrupt

    child.wait = exploding_wait
    hooks = _signal_hooks(child)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    result = mrl.execute_training_attempt(**kwargs)
    assert result != mrl.REDELIVERED
    assert hooks["installed"][signal.SIGINT] is hooks["originals"][signal.SIGINT]
    assert hooks["installed"][signal.SIGTERM] is hooks["originals"][signal.SIGTERM]
    persisted = json.loads((kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "interrupted"
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]


def test_child_receives_term_then_kill_after_grace(tmp_path):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child, release_on=signal.SIGKILL)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        _handler, term_handler = _wait_until_handlers(hooks["installed"], hooks["originals"])
        term_handler(signal.SIGTERM, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]
    assert training.TERM_GRACE_SECONDS in child.wait_timeouts


def test_cleanup_closes_log_before_final_commit(tmp_path):
    """`finalize` closes the log sink before the commit that carries the terminal STATUS.

    Each commit is recorded with the STATUS it carries, so the assertion
    names the first commit that carries the terminal STATUS, which is
    `finalize`'s: "some commit after `log_closed`" would be satisfied by the
    retry's commit, which follows it in production order (see `_signal_hooks`,
    ORDER). With no terminal commit from `finalize` at all, the retry's would
    satisfy this too; `test_publish_note_reaches_the_volume_in_the_right_commit`
    catches that.
    """
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    events: list[str] = []

    class RecordingSink(io.StringIO):

        def close(self):
            events.append("log_closed")
            super().close()

    def commit():
        status_path = kwargs["attempt"].run_root / mrl.STATUS_FILENAME
        events.append(f"commit:{json.loads(status_path.read_text())['status']}")

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            commit=commit,
            log_sink=RecordingSink(),
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    _interrupt_in_production_order(kwargs, child, hooks, signal.SIGINT)
    assert "log_closed" in events
    assert events.index("log_closed") < events.index("commit:interrupted")


def test_failed_cleanup_commit_does_not_let_redelivery_write(tmp_path):
    """A failed terminal commit leaves the Volume at TRAINING, and a redelivery must not write.

    Production order (see `_signal_hooks`, ORDER): the retry's commit then
    always runs after the terminal STATUS, so it fails too and records nothing.

    The result assertion is what mutant T21 targets (gh#238): `finalize`
    records `final_result` OUTSIDE the terminal `try`, so the attempt still
    returns the INTERRUPTED result when the Volume commit fails. Moved inside
    that `try`, after the commit, the raise skips it and the attempt returns
    the FAILED / `REASON_NONZERO_EXIT` fallback instead.
    """
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    committed: list[dict] = []

    def commit():
        payload = json.loads((kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_text())
        if payload["status"] == "interrupted":
            raise RuntimeError("volume commit failed")
        committed.append(payload)

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            commit=commit,
            now=lambda: _aware(),
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    boxed, _ = _interrupt_in_production_order(kwargs, child, hooks, signal.SIGINT)
    assert len(boxed) == 1
    assert boxed[0] == training.TrainingAttemptResult(status=core.Status.INTERRUPTED,
                                                      reason=training.REASON_SIGNAL,
                                                      exit_code=None)
    assert committed
    assert committed[-1]["status"] == "training"
    last = state.RunStatus.from_dict(committed[-1])
    derived = state.derive_status(last, now=_aware(minute=5))
    assert derived.stale is True
    assert derived.status is core.Status.INTERRUPTED
    before = (kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_bytes()
    commits_before = list(committed)

    def must_not_launch(*_args, **_kwargs):
        raise AssertionError("redelivered container must not start training")

    kwargs["process"] = dataclasses.replace(kwargs["process"], spawn=must_not_launch)
    assert mrl.execute_training_attempt(**kwargs) == mrl.REDELIVERED
    assert (kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_bytes() == before
    assert committed == commits_before


def test_post_spawn_failure_kills_child_and_writes_terminal_status(tmp_path):
    child = FakeChild(hold=True)
    # Opt-out (see `_SIGNAL_HOOKS_RELEASE_ALLOWLIST`): `child.poll()` below is
    # non-None only because the child dies at SIGKILL.
    hooks = _signal_hooks(child, release_on=signal.SIGKILL)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    (kwargs["attempt"].run_root / core.TRAIN_LOG_NAME).mkdir()
    with pytest.raises(OSError):
        mrl.execute_training_attempt(**kwargs)
    persisted = json.loads((kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] in {"failed", "interrupted"}
    assert persisted["attempt_id"] == "attempt-a"
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]
    assert child.poll() is not None


def test_term_grace_is_deadline_not_mandatory_sleep(tmp_path):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child, release_on=signal.SIGTERM)
    started = time.monotonic()
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        _handler, term_handler = _wait_until_handlers(hooks["installed"], hooks["originals"])
        term_handler(signal.SIGTERM, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert time.monotonic() - started < 5.0
    assert hooks["kills"] == [signal.SIGTERM]
    assert 15.0 not in hooks["slept"]
    persisted = json.loads((kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "interrupted"


def _record_checkpoint_reads(monkeypatch, run_root: Path) -> list[tuple[str, str]]:
    """Record every checkpoint validation and hash an attempt makes, and when it made it.

    Returns a live list of `(kind, phase)` pairs. `kind` is "validate"
    (`validate_local_checkpoint`, stubbed to reject, so no publish gets as far
    as a sidecar) or "hash" (`core.sha256_file`, stubbed to a fixed digest).
    `phase` is one of:
      * "watcher": the checkpoint watcher's own thread, which validates every
        50 ms while training. It is known by its name, a copy of the `name=`
        literal in `training._start_checkpoint_watcher`; the assert below
        (read through the AST, so quote style does not matter) fails, naming
        the rename, if that literal changes. Otherwise the watcher's reads
        would land in "before-result" and blame `finalize`;
      * "before-result": any other thread, before result.json exists. On the
        interrupt path that is `finalize`, which must not load or hash the
        checkpoint: the prebuilt load can take 120 s, and Modal's kill window
        is about 30 s on preemption and seconds on a Function timeout;
      * "after-result": any other thread once result.json exists, which is the
        attempt's post-finalize publish retry.

    WHY phases by thread and result.json, not by STATUS (gh#211). The first
    form of this helper, `_record_hash_after_terminal`, recorded any call made
    once STATUS was no longer `training`, and its callers asserted it recorded
    nothing. But the retry runs after the terminal STATUS by design, so the
    no-sidecar caller's assertion held only when the tests' own thread order
    let the retry win a race with `finalize`; under CPU load it failed. And
    `finalize` publishes BEFORE its STATUS write, so a `finalize` that
    published on INTERRUPTED was invisible to it.

    Callers must run in production order (`_signal_hooks`, ORDER). If the
    child is released inside the handler, the retry can run while `finalize`
    is still going, and its read is recorded as "before-result".
    """
    watcher_name = "cs2rl-checkpoint-watch"
    source = ast.parse(inspect.getsource(training._start_checkpoint_watcher))
    names = [
        keyword.value.value for node in ast.walk(source) if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant)
    ]
    renamed = (f"the checkpoint watcher thread's `name=` is no longer the literal "
               f"{watcher_name!r} (found {names}): update _record_checkpoint_reads and the "
               "copy in test_interrupt_commits_status_even_if_prebuilt_load_hangs")
    assert names == [watcher_name], renamed
    reads: list[tuple[str, str]] = []

    def phase() -> str:
        if threading.current_thread().name == watcher_name:
            return "watcher"
        if (run_root / core.RESULT_FILENAME).is_file():
            return "after-result"
        return "before-result"

    def wrapped_validate(path):
        del path
        reads.append(("validate", phase()))
        raise mrl.ValidationError("test stub: skip torch")

    def wrapped_hash(path):
        del path
        reads.append(("hash", phase()))
        return "00" * 32

    monkeypatch.setattr(*binding_target("terminal-validator"), wrapped_validate)
    monkeypatch.setattr(*binding_target("terminal-hasher"), wrapped_hash)
    return reads


def test_interrupt_uses_sidecar_digest_and_skips_torch_hash(tmp_path, monkeypatch):
    """result.json takes the sidecar's digest; nothing loads or hashes the checkpoint.

    There is no checkpoint file, only its sidecar, so no publish (the
    watcher's or the retry's) gets as far as reading one, and any read at
    all is `finalize` hashing for result.json. Production order (see
    `_signal_hooks`).
    """
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    run_root = kwargs["attempt"].run_root
    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir()
    sidecar_digest = "ab" * 32
    (ckpt_dir / mrl.CHECKPOINT_SIDECAR_NAME).write_text(
        json.dumps({
            "sha256": sidecar_digest,
            "size": 13,
            "mtime_ns": 1,
            "validated_at": "2026-08-13T00:00:00+00:00",
        }) + "\n")
    reads = _record_checkpoint_reads(monkeypatch, run_root)
    _interrupt_in_production_order(kwargs, child, hooks, signal.SIGINT)
    assert reads == []
    payload = json.loads((run_root / core.RESULT_FILENAME).read_text())
    assert payload["status"] == "interrupted"
    assert payload["checkpoint_sha256"] == sidecar_digest


def test_interrupt_without_sidecar_leaves_checkpoint_hash_null(tmp_path, monkeypatch):
    """No sidecar: result.json's digest is null, and only the retry validates the checkpoint.

    Outside the watcher, the one read is the post-finalize retry's
    validation, after result.json exists. That read is also this test's
    positive control for the validator stub: one that stopped reaching its
    consumer (see `_record_checkpoint_reads`) records nothing and fails here.
    The hasher stub has no such control: unmutated code never hashes here. A
    read before result.json means `finalize` loaded or hashed the checkpoint,
    or the retry ran before `finalize` finished. Production order (see
    `_signal_hooks`).
    """
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    run_root = kwargs["attempt"].run_root
    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "dust2_policy.pt").write_bytes(b"do-not-load-me")
    reads = _record_checkpoint_reads(monkeypatch, run_root)
    _interrupt_in_production_order(kwargs, child, hooks, signal.SIGINT)
    outside_watcher = [read for read in reads if read != ("validate", "watcher")]
    assert outside_watcher == [("validate", "after-result")]
    payload = json.loads((run_root / core.RESULT_FILENAME).read_text())
    assert payload["status"] == "interrupted"
    assert payload["checkpoint_sha256"] is None


# A hard bound, not a margin: the `timeout` case runs the attempt on the test
# thread with a held child and a frozen clock, so a wait loop that stopped
# timing out would spin forever. The repo configures no default timeout.
@pytest.mark.timeout(30)
@pytest.mark.parametrize(("ending", "after_terminal"), [
    pytest.param("exit", [("failed", True), ("failed", True)], id="exit"),
    pytest.param("error", [("failed", True), ("failed", True)], id="error"),
    pytest.param("signal", [("interrupted", False), ("interrupted", True)], id="signal"),
    pytest.param("timeout", [("interrupted", False), ("interrupted", True)], id="timeout"),
    pytest.param("keyboard_interrupt", [("interrupted", False), ("interrupted", True)],
                 id="keyboard_interrupt"),
])
def test_publish_note_reaches_the_volume_in_the_right_commit(tmp_path, ending, after_terminal):
    """Which commit carries the publish note to the Volume, on each way an attempt ends.

    gh#211, gh#238 (mutants F11-F13). There is no checkpoint, so every
    `_publish_and_note` writes the note ("no checkpoint at ..."); the
    watcher's publishes drop their reasons by design. The attempt writes the
    note in `finalize`, with `commit_note=False` (and not at all on
    INTERRUPTED: a 120 s prebuilt load must not sit before the terminal
    STATUS), then again in the retry after it, with `commit_note=True`. The
    commit fake records `(STATUS, note on disk?)` at every commit.
    `after_terminal` is the record of the commits made once STATUS is
    terminal:
      * FAILED (`exit`: the child exits 1; `error`: the attempt raises after
        the spawn): the note written by `finalize` goes out with the terminal
        commit, and the retry commits its rewrite;
      * INTERRUPTED (`signal`, `timeout`, `keyboard_interrupt`): the terminal
        commit has no note, and only the retry's commit carries it.
    No commit made while STATUS is `training` may carry the note.

    What fails where: F11 (every publish `commit_note=False`) loses the last
    `True` everywhere; F12 (every publish `commit_note=True`) commits the note
    while STATUS is still `training` on the FAILED paths; F13 (`finalize`'s
    publish deleted) turns the FAILED terminal commit `False`; a `finalize`
    that publishes on INTERRUPTED turns that terminal commit `True`.

    Only `signal` runs the attempt on a thread, with the handler called from
    the test in production order (see `_signal_hooks`, ORDER). The other
    cases run the attempt on the test thread, where the order is production's
    already, so whether `killpg` releases the child is inert there (the
    attempt and `finalize` share the test thread, so the tail runs after
    `finalize` whatever `killpg` does); one default `_signal_hooks` serves
    every case only to keep one construction.
    """
    commits: list[tuple[str, bool]] = []

    def commit() -> None:
        run_root = kwargs["attempt"].run_root
        status = json.loads((run_root / mrl.STATUS_FILENAME).read_text())["status"]
        note = run_root / "checkpoints" / core.CHECKPOINT_PUBLISH_REASON_NAME
        commits.append((status, note.is_file()))

    child = FakeChild(returncode=1) if ending == "exit" else FakeChild(hold=True)
    if ending == "keyboard_interrupt":

        def exploding_wait(timeout=None):
            raise KeyboardInterrupt

        child.wait = exploding_wait
    hooks = _signal_hooks(child)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            commit=commit,
            timeout=timedelta(0) if ending == "timeout" else None,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    if ending == "signal":
        _interrupt_in_production_order(kwargs, child, hooks, signal.SIGINT)
    elif ending == "error":
        (kwargs["attempt"].run_root / core.TRAIN_LOG_NAME).mkdir()
        with pytest.raises(OSError):
            mrl.execute_training_attempt(**kwargs)
    else:
        mrl.execute_training_attempt(**kwargs)
    assert not any(note for status, note in commits if status == "training")
    assert [record for record in commits if record[0] != "training"] == after_terminal


@pytest.mark.timeout(30)
def test_a_signal_inside_finalize_does_not_finalize_again(tmp_path):
    """A second signal, handled while the first handler is inside `finalize`, returns at once.

    gh#211. In production every handler runs on the main thread, between
    bytecodes, so a SIGTERM that arrives while the SIGINT handler is
    signalling the child's group runs the handler again, nested, on the same
    thread. That nested `finalize` must fail the gate's acquire and return;
    the outer one then does the only terminal write. The fake `killpg`
    delivers the nested signal on the first SIGTERM, from inside
    `_signal_process_group`, which runs after the gate is taken. (A signal
    landing INSIDE the gate is
    `test_a_signal_while_taking_the_once_gate_returns_at_once`, gh#238 P2.)

    A gate released after use (mutant `GATE_RELEASED`: the lock released
    right after `cleaned = True`) would run the nested call in full: the
    nested `finalize` takes the released lock, so two SIGTERMs and two
    SIGKILLs, and a second terminal commit (a same-terminal STATUS write is
    idempotent, so the second one returns without an error). This test
    therefore requires `kills` to be one pair
    and, once STATUS is terminal, exactly two commits: the terminal one,
    without the note, then the retry's, with it (as in
    `test_publish_note_reaches_the_volume_in_the_right_commit`). The terminal
    commit count stands in for "one terminal transition": a second
    `transition_status` to the same terminal status writes nothing, and the
    census allows `binding_target("attempt-transition")` only in
    `test_checkpoint_watcher_stops_before_terminal_status`, which counts
    terminal transitions directly on the one-signal path. (Before gh#211
    only a racy assertion in the interrupt tests could see a late `cleaned`.)

    Production order (see `_signal_hooks`, ORDER). The timeout is a hard
    bound, not a margin: a `finalize` whose once-gate were a BLOCKING acquire
    would deadlock on the nested call instead of failing it.
    """
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    nested: list[str] = []
    commits: list[tuple[str, bool]] = []

    def reentrant_killpg(pgid, sig):
        hooks["killpg"](pgid, sig)
        if sig == signal.SIGTERM and not nested:
            nested.append("entered")
            handler = hooks["installed"][signal.SIGTERM]
            assert callable(handler), "the attempt's SIGTERM handler is not installed"
            handler(signal.SIGTERM, None)
            nested.append("returned")

    def commit() -> None:
        run_root = kwargs["attempt"].run_root
        status = json.loads((run_root / mrl.STATUS_FILENAME).read_text())["status"]
        note = run_root / "checkpoints" / core.CHECKPOINT_PUBLISH_REASON_NAME
        commits.append((status, note.is_file()))

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            commit=commit,
            sleep=hooks["sleep"],
            killpg=reentrant_killpg,
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    _interrupt_in_production_order(kwargs, child, hooks, signal.SIGINT)
    assert nested == ["entered", "returned"]
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]
    after_terminal = [record for record in commits if record[0] != "training"]
    assert after_terminal == [("interrupted", False), ("interrupted", True)]


@pytest.mark.timeout(30)
@pytest.mark.parametrize("ending", ["signal", "exit"])
def test_a_hung_heartbeat_does_not_strand_the_run_in_training(tmp_path, capsys, ending):
    """A heartbeat worker that will not stop costs one stderr line, never the terminal write.

    gh#238 P3. `finalize` stops the heartbeat (`HeartbeatWorker.stop_and_join`,
    which RAISES `RuntimeError("heartbeat worker did not stop")` after its 5 s
    join) before the terminal STATUS and result.json. Before the fix that step
    ran outside any `try`: the raise left the run in TRAINING with `cleaned`
    already set, so the `except` arms' second `finalize` returned at once and
    the run was stranded until Modal's kill. `stop_heartbeat_once` now swallows
    the failure and prints one stderr line (the worker is a daemon whose only
    shared write refuses terminal statuses, so abandoning it is safe: see its
    docstring for the one hang it does not cover).

    Two endings, because two paths reach `stop_heartbeat_once` first:
      * `signal`: the handler, called from the test thread in production order
        (see `_signal_hooks`, ORDER), reaches `finalize` → INTERRUPTED /
        `REASON_SIGNAL`. The helper runs with `capture=True`, so a RuntimeError
        escaping the handler is RECORDED in `raised`, not raised out of the
        test: `raised == []` is the first assertion, and the knock-out (`try`
        deleted, bare `stop_heartbeat`) goes red there.
      * `exit`: the child exits 0 at once with no manifest, on the test thread
        (as `test_keyboard_interrupt_uses_same_cleanup`), so `finish` reaches
        `finalize` → FAILED / `REASON_INVALID_EVIDENCE` (`_map_child_exit`).
        The same knock-out makes `execute_training_attempt` itself raise (the
        `except Exception` arm re-raises), red at the call line.
    The heartbeat is a `SimpleNamespace(stop_and_join=...)`, so no real worker
    starts; its fake raises on the FIRST call only and records every call:
    `release()` must not retry it (`heartbeat_stopped` is set before the call).
    The stderr assertion pins the PRODUCTION prefix, not only the worker's
    message: a swallow with no line (knock-out `HEARTBEAT_SILENT`) is red here.
    """
    calls: list[float | None] = []

    def stop_and_join(timeout=5.0):
        calls.append(timeout)
        if len(calls) == 1:
            raise RuntimeError("heartbeat worker did not stop")

    # `hooks` is bound ONCE, outside the if: bound in both arms, pyrefly joins
    # the two flows into a union of the dict's value types and flags every
    # `hooks["installed"][...]` read below as not subscriptable.
    child = FakeChild(hold=True) if ending == "signal" else FakeChild()
    hooks = _signal_hooks(child)
    if ending == "signal":
        expected = training.TrainingAttemptResult(status=core.Status.INTERRUPTED,
                                                  reason=training.REASON_SIGNAL,
                                                  exit_code=None)
    else:
        expected = training.TrainingAttemptResult(status=core.Status.FAILED,
                                                  reason=training.REASON_INVALID_EVIDENCE,
                                                  exit_code=0)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    kwargs["prepared"] = _prepared_source(tmp_path,
                                          heartbeat=SimpleNamespace(stop_and_join=stop_and_join))
    run_root = kwargs["attempt"].run_root
    if ending == "signal":
        boxed, raised = _interrupt_in_production_order(kwargs,
                                                       child,
                                                       hooks,
                                                       signal.SIGTERM,
                                                       capture=True)
        assert raised == []
    else:
        boxed = [mrl.execute_training_attempt(**kwargs)]
    persisted = json.loads((run_root / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == expected.status.value
    assert (run_root / core.RESULT_FILENAME).is_file()
    assert json.loads(
        (run_root / core.RESULT_FILENAME).read_text())["status"] == expected.status.value
    assert len(boxed) == 1
    assert boxed[0] == expected
    assert hooks["installed"][signal.SIGINT] is hooks["originals"][signal.SIGINT]
    assert hooks["installed"][signal.SIGTERM] is hooks["originals"][signal.SIGTERM]
    assert len(calls) == 1
    err = capsys.readouterr().err
    assert "cs2rl: heartbeat did not stop before the terminal write" in err
    assert "heartbeat worker did not stop" in err


@pytest.mark.timeout(30)
def test_a_signal_while_taking_the_once_gate_returns_at_once(tmp_path, monkeypatch):
    """A signal handled at the instant the once-gate is taken returns; it does not deadlock.

    gh#238 P2. CPython runs a signal handler between any two bytecodes of the
    thread that owns it, including between `finalize`'s first two statements.
    With the old `with self.cleanup_lock:` gate a nested `finalize` on the same
    thread blocked on the held non-reentrant lock forever: the run stayed in
    TRAINING with the child killed. The gate is now one NON-BLOCKING
    `acquire`, and the lock is never released afterwards, so a nested
    `finalize` fails the acquire and returns instead of waiting on a lock
    nobody will ever give back. (Mutant `GATE_BLOCKING`, a blocking
    `acquire()`, deadlocks here for exactly that reason: it waits forever on
    the lock the outer finalize still holds.)

    The double replaces `training.threading.Lock` (the sanctioned shape:
    tests/test_modal_patch_bindings.py's attempt rows), which training.py
    builds once, for `cleanup_lock`. On the FIRST acquire that returns True,
    reached through `acquire(...)` or `__enter__` (the same instant on the old
    and the new code), it calls the attempt's SIGTERM handler nested, before
    returning to its caller. `finalize` runs on the attempt's own thread (the
    child's `wait` raises KeyboardInterrupt, as in
    `test_keyboard_interrupt_uses_same_cleanup`), so `release_on` is inert and
    the nested handler is on the thread that holds the lock: exactly the
    production case. The attempt thread is a DAEMON, so a deadlock strands only
    it and `finished.wait(...)` goes red after 2 s instead of hanging the
    session. Run the knock-outs of this test as this single node: the stranded
    thread never reaches `release()`, so its watcher and heartbeat stay alive,
    idle, until the process exits.
    """
    child = FakeChild(hold=True)

    def exploding_wait(timeout=None):
        raise KeyboardInterrupt

    child.wait = exploding_wait
    hooks = _signal_hooks(child)
    fired: list[str] = []
    nested: list[str] = []
    commits: list[tuple[str, bool]] = []

    class _GateDouble:
        """A real Lock whose first successful acquire runs the SIGTERM handler, nested."""

        def __init__(self):
            self._lock = threading.Lock()

        def acquire(self, *args, **kwargs):
            taken = self._lock.acquire(*args, **kwargs)
            if taken and not fired:
                fired.append("fired")
                handler = hooks["installed"][signal.SIGTERM]
                assert callable(handler), "the attempt's SIGTERM handler is not installed"
                nested.append("entered")
                handler(signal.SIGTERM, None)
                nested.append("returned")
            return taken

        def release(self):
            self._lock.release()

        def __enter__(self):
            return self.acquire()

        def __exit__(self, *_exc):
            self.release()

    def commit() -> None:
        run_root = kwargs["attempt"].run_root
        status = json.loads((run_root / mrl.STATUS_FILENAME).read_text())["status"]
        note = run_root / "checkpoints" / core.CHECKPOINT_PUBLISH_REASON_NAME
        commits.append((status, note.is_file()))

    monkeypatch.setattr(
        training, "threading",
        SimpleNamespace(Thread=threading.Thread, Event=threading.Event, Lock=_GateDouble))
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            commit=commit,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    thread, finished, boxed = _run_attempt_in_thread(kwargs, daemon=True)
    try:
        assert finished.wait(timeout=2.0), "the nested finalize deadlocked on the once-gate"
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert fired == ["fired"]
    assert nested == ["entered", "returned"]
    assert len(boxed) == 1
    assert boxed[0] == training.TrainingAttemptResult(status=core.Status.INTERRUPTED,
                                                      reason=training.REASON_SIGNAL,
                                                      exit_code=None)
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]
    after_terminal = [record for record in commits if record[0] != "training"]
    assert after_terminal == [("interrupted", False), ("interrupted", True)]


class _BlockingStream:
    """A child stream whose `read` blocks until `release()`, then returns EOF.

    For `test_finalize_kills_the_child_before_joining_the_tees`: a tee thread
    over a `BytesIO` finishes before `finalize` runs, so whether the kill or
    the join comes first would depend on scheduling. Over this stream the tee
    thread stays alive until the test lets it go, at its own join. `release`
    is idempotent.
    """

    def __init__(self):
        self._released = threading.Event()

    def read(self, _size: int = -1) -> bytes:
        self._released.wait()
        return b""

    def release(self) -> None:
        self._released.set()


@pytest.mark.timeout(30)
def test_finalize_kills_the_child_before_joining_the_tees(tmp_path, monkeypatch):
    """`finalize` signals the child's group BEFORE it joins the tee threads.

    gh#238 P1 (no production change: this pins the order). A tee join waits
    up to 5 s per stream for a child that is still writing; with the kill
    first, the child is dying while the tees drain. Reversed, an INTERRUPTED
    run would spend its ~30 s preemption window waiting on a live child's
    output before it is even signalled.

    Both child streams are `_BlockingStream`s, so BOTH tee threads are alive
    at their join and the two-entry count is a property of the code, not of
    scheduling. Tee threads are identified at START, by object (`_target is
    training._tee_stream`, read before delegating: `Thread.run` deletes
    `_target` when the target returns, so a join-time read misses a finished
    tee, the gh#217 pitfall). At each tee join the spy records the kills so
    far, THEN releases both streams so no join waits out its timeout. The
    handler is called from the test thread in production order (see
    `_signal_hooks`, ORDER), once the attempt is in its wait loop: handlers
    are installed BEFORE `start_tees`, so a handler fired as soon as they
    appear could finalize with `tee_threads` still empty and see no join at
    all. Mutant `JOIN_BEFORE_KILL` (the join loop above the kill) records `[]`
    at both joins.
    """
    child = FakeChild(hold=True)
    out, err = _BlockingStream(), _BlockingStream()
    # FakeChild types its streams as BytesIO; a duck-typed stream is the point here.
    child.stdout = out                 # pyrefly: ignore[bad-assignment]
    child.stderr = err                 # pyrefly: ignore[bad-assignment]
    hooks = _signal_hooks(child)
    real_start = threading.Thread.start
    real_join = threading.Thread.join
    tee_threads_seen: set[threading.Thread] = set()
    joins: list[list[int]] = []

    def spy_start(self):
        if getattr(self, "_target", None) is training._tee_stream:
            tee_threads_seen.add(self)
        return real_start(self)

    def spy_join(self, timeout=None):
        if self in tee_threads_seen:
            joins.append(list(hooks["kills"]))
            out.release()
            err.release()
        return real_join(self, timeout)

    monkeypatch.setattr(threading.Thread, "start", spy_start)
    monkeypatch.setattr(threading.Thread, "join", spy_join)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    run_root = kwargs["attempt"].run_root
    try:
        # `armed`: the attempt is in its wait loop, so start_tees is done.
        _interrupt_in_production_order(kwargs,
                                       child,
                                       hooks,
                                       signal.SIGTERM,
                                       armed=lambda: bool(child.wait_timeouts))
    finally:
        out.release()
        err.release()
    assert len(joins) == 2
    for kills_at_join in joins:
        assert kills_at_join == [signal.SIGTERM, signal.SIGKILL]
    persisted = json.loads((run_root / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "interrupted"


def test_checkpoint_watcher_stops_before_terminal_status(tmp_path, monkeypatch):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    watcher_stop: dict[str, threading.Event | None] = {"event": None}
    at_terminal: list[tuple[str, bool]] = []
    real_start = training._start_checkpoint_watcher
    real_transition = state.transition_status

    def wrapped_start(**kwargs):
        stop, thread = real_start(**kwargs)
        watcher_stop["event"] = stop
        return stop, thread

    def wrapped_transition(run_root, next_status, **kwargs):
        if next_status in mrl.TERMINAL_STATUSES:
            event = watcher_stop["event"]
            at_terminal.append((next_status.value, event is not None and event.is_set()))
        return real_transition(run_root, next_status, **kwargs)

    monkeypatch.setattr(*binding_target("attempt-watcher"), wrapped_start)
    monkeypatch.setattr(*binding_target("attempt-transition"), wrapped_transition)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    _interrupt_in_production_order(kwargs, child, hooks, signal.SIGINT)
    assert at_terminal == [("interrupted", True)]
    persisted = json.loads((kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "interrupted"


# A hard timeout, not belt-and-braces: if `spy_start`'s `_target is _tee_stream`
# filter ever stops matching, the wrapper never fires, the SIGTERM is never sent,
# FakeChild(hold=True) is never released, and the attempt's wait loop spins
# forever -- so the `assert fired` guard below is UNREACHABLE and this becomes an
# unbounded hang instead of a failure. The repo configures no default timeout.
# W3b makes this reachable: anything that stops `training._tee_stream` being the
# identical object passed as `target` (a re-export wrapper, functools.partial, a
# Thread subclass) breaks identity SILENTLY, where a rename would at least raise
# AttributeError.
@pytest.mark.timeout(30)
def test_real_sigterm_in_tee_window_never_joins_unstarted_thread(tmp_path, monkeypatch):
    """A real SIGTERM inside the tee-thread start window must not strand the run.

    gh#217. `finalize` (`training._LiveAttempt.finalize`) joins
    `tee_threads` unconditionally, so for
    as long as that list could hold a not-yet-started thread, a signal arriving
    there raised `RuntimeError: cannot join thread before it is started` out of
    `on_signal` and past `transition_status`: STATUS.json stuck on `training`, no
    result.json, child already dead. This is gh#217's demonstration 3 checked in —
    `execute_training_attempt` on the MAIN thread with the REAL `signal.signal`, a
    helper thread firing a REAL `os.kill(os.getpid(), SIGTERM)`, and a
    `threading.Thread.start` wrapper filtered on `_target is _tee_stream` that puts
    the signal in the window deterministically rather than relying on machine load.

    Three pitfalls this test is built around, each of which fails silently:
      * `_target` is captured BEFORE delegating to the real `start()`. CPython's
        `Thread.run()` does `del self._target, self._args, self._kwargs` in its
        `finally` when a thread finishes (it is run(), not _bootstrap_inner --
        gh#217's draft said otherwise), and a tee thread over an empty FakeChild
        stream can finish before `start()` returns — read afterwards it is None and
        the filter never matches.
      * `killpg`/`getpgid` are FAKE, and that is a safety requirement rather than a
        preference. `finalize` signals the child's process group using FakeChild's
        default `pid=4242`; with the real ones, a machine where pid 4242 happens to
        exist gets a genuine SIGTERM, and `_signal_process_group`'s
        `except ProcessLookupError` makes it silent on every machine where it does
        not.
      * A no-op SIGTERM handler is installed around the call. The runner's `finally`
        restores whatever disposition was in force on entry, so without this a
        signal arriving after that restore reaches pytest's default disposition and
        kills the session instead of failing the test.

    Disclosed blind spots — this test gates the defect, not one specific repair:
      * A fix that guards the join site (`if thread.ident is not None`) instead of
        reordering passes, and so does one that clears `tee_threads` before joining.
        Nothing in this suite excludes either; both are disclosed deliberately.
      * Do NOT "correct" the spy's predicate to `is_alive()`. gh#217 measured
        `ident` to be an unreliable proxy for "join will raise" when used as a
        PRODUCTION guard, because `_bootstrap_inner` sets `_ident` before
        `_started` while `join` gates on `_started`. As a SPY it is exactly right: a
        never-started thread has `ident is None`, and once `start()` has returned
        `_started` is already set. `is_alive()` silently changes what is detected —
        a started-and-already-finished thread is not alive.
      * It drives one deterministic point inside the window. It does not prove the
        window is shut at every instruction.
      * Under the FIX, the residual pre-first-`append` window is the only state this
        test ever observes -- measured, `len(tee_threads) == 0` at the join and the
        spy's log is `[('cs2rl-preflight-heartbeat', False),
        ('cs2rl-checkpoint-watch', False)]`, i.e. zero tee-thread joins. So the
        outcome assertions are made from inside the accepted residual, and what goes
        UNTESTED is the state the fix creates: `tee_threads` non-empty and holding
        only started threads. An earlier wording had this exactly backwards.
    """
    child = FakeChild(hold=True)
    # Opt-out (see `_SIGNAL_HOOKS_RELEASE_ALLOWLIST`): the real handler runs on
    # the main thread and the child's death at SIGKILL is what ends `wait_for_exit`.
    hooks = _signal_hooks(child, release_on=signal.SIGKILL)
    entered_finalize = threading.Event()
    killpg_hook = hooks["killpg"]

    def killpg_marking_finalize(pgid, sig):
        # finalize kills the child (its `_signal_process_group` call) before it
        # joins tee_threads, so this is the earliest in-handler observable available.
        # The handler runs on the main thread, so the spin below cannot observe
        # this flag until the handler has already returned or raised.
        killpg_hook(pgid, sig)
        entered_finalize.set()

    real_start = threading.Thread.start
    real_join = threading.Thread.join
    joined_unstarted: list[str] = []
    fired: list[str] = []
    killers: list[threading.Thread] = []

    def spy_join(self, timeout=None):
        # The primary assertion. Outcome assertions alone do NOT carry this test:
        # measured on unfixed source with join swallowing the RuntimeError, the
        # window is still wide open yet STATUS reaches `interrupted` and
        # result.json is written. Only this spy sees the unstarted join.
        if self.ident is None:
            joined_unstarted.append(self.name)
        return real_join(self, timeout)

    def deliver_sigterm() -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    def spy_start(self):
        target = getattr(self, "_target", None)
        if fired or target is not training._tee_stream:
            return real_start(self)
        # Fire once. A fire-every-start shim measures the same, because the first
        # fire aborts the loop before the second start() — this is readability.
        fired.append(self.name)
        killer = threading.Thread(target=deliver_sigterm, name="b0-sigterm-source", daemon=True)
        real_start(killer)
        killers.append(killer)
        deadline = time.monotonic() + 5.0
        while not entered_finalize.is_set():
            if time.monotonic() > deadline:
                child.release()
                raise AssertionError("SIGTERM was never handled inside the start window")
            time.sleep(0.001)
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", spy_start)
    monkeypatch.setattr(threading.Thread, "join", spy_join)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=killpg_marking_finalize,
            getpgid=hooks["getpgid"],
        ))
    run_root = kwargs["attempt"].run_root
    previous_term = signal.signal(signal.SIGTERM, lambda *_args: None)
    previous_int = signal.getsignal(signal.SIGINT)
    try:
        result = mrl.execute_training_attempt(**kwargs)
    finally:
        child.release()
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)
        if previous_int is not None:
            signal.signal(signal.SIGINT, previous_int)
        for killer in killers:
            real_join(killer, 2.0)
    assert fired, "the _tee_stream start wrapper never fired; the window was never opened"
    assert joined_unstarted == []
    assert result != mrl.REDELIVERED
    persisted = json.loads((run_root / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "interrupted"
    assert (run_root / core.RESULT_FILENAME).is_file()
    payload = json.loads((run_root / core.RESULT_FILENAME).read_text())
    assert payload["status"] == persisted["status"]
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.parametrize("case", ["pgid-le-1", "own-group", "pgid-ne-pid", "control"])
def test_signal_process_group_refuses_groups_a_live_child_cannot_have(case, capsys):
    """`_signal_process_group` refuses the three groups a live child cannot lead.

    gh#163. The guard exists because an agent's script once ran
    `killpg(getpgid(1), SIGTERM)` for real, which is `kill(-1, SIGTERM)`, and
    ended the user's desktop session. Each refusal case sets up exactly ONE of
    the three conditions:
      * `pgid-le-1`   -- pid 1, identity getpgid: the incident's exact shape;
      * `own-group`   -- pid = the runner's own group, identity getpgid;
      * `pgid-ne-pid` -- pid P, getpgid returning P + 1 (a fake, or a reused pid);
      * `control`     -- pid P, identity getpgid: must SIGTERM P and print nothing.
    The case ids are shell-safe spellings; the stderr tokens are `pgid<=1`,
    `own-group` and `pgid!=pid`.

    WHY one condition per case. A case where two conditions hold stays green
    when one of their clauses is deleted, because the other still refuses: the
    shadowed clause would ship untested. So `pid == pgid` everywhere except the
    `pgid-ne-pid` case, and `P = os.getpgrp() + 4242` differs from the runner's
    group by construction. If the runner's group were 1, `pgid-le-1` would also
    be `own-group` (and below 1, `own-group` would also be `pgid<=1`), hence
    the first assertion, which names that condition instead of letting a
    knock-out stay green for a reason nobody can see.

    WHY each case checks its own token AND the absence of the other two. A
    guard printing one line that names all three conditions for every refusal
    would pass a "contains my token" check. Only the exclusion pins WHICH clause
    refused.

    WHY the three refusal children are held (`hold=True`) and the control's is
    not. A guard that prints the line and skips the SIGTERM but forgets to
    `return` falls through to the escalation. With an exited child, the first
    `poll()` returns and all three cases stay green. On a live child the same
    guard reaches `killpg(pgid, SIGKILL)` with the refused pgid: in the
    `pgid-le-1` case that is `kill(-1, SIGKILL)`. A held FakeChild is that live
    child: `poll()` is None and the grace `wait` times out at once (no
    wall-clock wait), so the missing `return` records a SIGKILL and
    `kills == []` goes red. The control keeps an exited child, so it records
    the SIGTERM alone.

    SAFETY: every case passes a recording `killpg` and a fake `sleep`, so
    nothing here signals even with a clause removed. The knock-outs (delete one
    clause, or drop the refusal's `return`; run only the affected case nodes;
    see red) are run by hand and are not committed.
    """
    assert os.getpgrp() > 1, (
        f"the runner's own process group is {os.getpgrp()}, not above 1, so the `pgid-le-1` and "
        "`own-group` cases would both hold two conditions and could not tell the clauses apart")
    P = os.getpgrp() + 4242

    def identity(pid):
        return pid

    child, getpgid, token = {
        "pgid-le-1": (FakeChild(pid=1, hold=True), identity, "pgid<=1"),
        "own-group": (FakeChild(pid=os.getpgrp(), hold=True), identity, "own-group"),
        "pgid-ne-pid": (FakeChild(pid=P, hold=True), lambda pid: pid + 1, "pgid!=pid"),
        "control": (FakeChild(pid=P), identity, None),
    }[case]
    kills: list[tuple[int, int]] = []
    slept: list[float] = []

    training._signal_process_group(child,
                                   killpg=lambda pgid, sig: kills.append((pgid, sig)),
                                   getpgid=getpgid,
                                   sleep=slept.append)

    refusals = [
        line for line in capsys.readouterr().err.splitlines()
        if line.startswith("cs2rl: refusing to signal process group")
    ]
    if token is None:
        assert kills == [(P, signal.SIGTERM)]
        assert refusals == []
        return
    assert kills == []
    assert len(refusals) == 1, refusals
    assert token in refusals[0]
    others = {"pgid<=1", "own-group", "pgid!=pid"} - {token}
    assert not [other for other in others if other in refusals[0]], refusals[0]


# ── Kill seam: the static safety clauses and the ProcessControl tripwire ──


class _KillSeamClauses:
    """The clauses `test_kill_seam_static_safety` enforces: one checker and one plant table each.

    gh#163 W5. A checker takes `{relative path: source text}`
    and returns `(problems, examined)`: what fails the clause, and what it looked
    at, which the test requires to hold the clause's population on the real tree.
    A plant has the same shape as the real sources, `{path: source text}`; it is
    parsed and never run, and must fail its clause. Nothing here imports, calls
    or opens anything: it reads only the text it is handed.

    WHY A CLASS, AND WHY HERE. Nested inside the test, the checkers counted
    towards the test's own complexity, because ruff's C901 and complexipy both
    count a nested def inside its parent (50 / 222 at the W5 types commit). As
    methods each is measured, and read, on its own. The class is ONE governed
    seam name (tests/fixtures/modal_test_seam_manifest.json and
    GOVERNED_NAME_COUNT in tests/test_modal_packaging.py): module-level checker
    functions or plant tables would each be another, so keep every checker,
    table and plant inside it.

    ADDING A CLAUSE: write its checker as a method here, plus a `clause_<n>`
    method that returns `_clause(...)` with its plants; list it in `clauses()`;
    and add its label and source sets to the test's pinned list. Each plant
    carries its own path, so a clause that reads both the runner and tests/ (as
    (iii) does) plants into either, and a runner-side plant names the module it
    stands in for (training.py, or another module of the package).
    """

    TRAINING = "scripts/modal_runner/training.py"
    # The "runner" source set is every module of the package and the Modal entry script;
    # a runner path is one under `scripts/`. Only training.py holds the allowances: system()'s
    # body, the process-group guard and its two handoffs.
    RUNNER_ENTRY = "scripts/run_modal.py"
    RUNNER_PLANT = "scripts/modal_runner/state.py"
    PLANT_TEST = "tests/test_kill_seam_plant.py"
    CONFTEST = "tests/conftest.py"
    TRIPWIRE = "_process_control_tripwire"
    FIELDS = ("spawn", "getpgid", "killpg", "install_signal")
    REAL_SYSTEM = {
        "spawn": "subprocess.Popen",
        "getpgid": "os.getpgid",
        "killpg": "os.killpg",
        "install_signal": "signal.signal",
    }
    OS_MODULES = ("os", "posix")
    SIGNAL_MODULES = ("signal", )
    SUBPROCESS_MODULES = ("subprocess", )
    IMPORT_CALLS = ("__import__", "import_module")
    BANNED = frozenset({"killpg", "getpgid", "kill"})
    BANNED_SIGNAL = frozenset({"signal"})
    BANNED_SUBPROCESS = frozenset({"Popen"})
    # (iii)'s real functions in training.py, as (modules, names): each is loaded only inside
    # ProcessControl.system()'s body.
    REAL_FUNCTIONS = (
        (OS_MODULES, BANNED),
        (SIGNAL_MODULES, BANNED_SIGNAL),
        (SUBPROCESS_MODULES, BANNED_SUBPROCESS),
    )
    GUARDED_CALLS = ("killpg", "getpgid")
    # The callees a `killpg=`/`getpgid=` keyword may hand those functions to, in (vii): the
    # guard itself, and system()'s own construction, `cls(...)`. PINNED: (vii)'s population
    # must find exactly one handoff to each name here, so a name added without a handoff, or
    # a second handoff to one of them, is red. Not `ProcessControl`: a re-wrap
    # `ProcessControl(killpg=self.process.killpg, ...)` would hand the real function to a
    # control that can then be called anywhere.
    GUARD_HANDOFFS = ("_signal_process_group", "cls")
    EXECUTE = "execute_training_attempt"
    RESOLUTION_CONTROL = "test_process_control_tripwire_guards_the_resolution_path"
    # (iii)'s exemptions, keyed (file, test): the only tests that may read `.system`.
    SYSTEM_READERS = (
        ("tests/test_modal_training.py", "test_process_control_tripwire_poisons_system"),
        ("tests/test_modal_training.py", RESOLUTION_CONTROL),
    )
    # (iv)'s exemptions, keyed (file, enclosing function): the only functions under tests/
    # that may load os/posix `killpg`, `getpgid` or `kill`. WHAT: each spawns its own child
    # with `start_new_session=True` and kills that group in a `finally`, so a red test cannot
    # leak ~900 MB vis-cache workers (gh#251 `_run_group`, gh#254). WHY it is not the seam:
    # the seam is Modal's ProcessControl in training.py, and a hygiene kill of the test's
    # own child group cannot bypass it, PROVIDED the file cannot reach the runner at all: an
    # exempt function may load anything banned, so one that imported the runner could hand
    # the real `os.killpg` to `ProcessControl(...)` in the clear. So `kill_seam_loads` also
    # makes a file that owns a row red on any import of `RUNNER_PACKAGES` (`_runner_imports`),
    # in every Import/ImportFrom spelling `_runner_imports` reads: `import
    # scripts.modal_runner.training`, `from scripts.modal_runner import training`, `from
    # scripts import modal_runner`, and the same for tests/modal_test_helpers.py, which binds
    # `training` and the package at module level (a one-hop re-export). WATCHED: (iv)'s population
    # requires every row here to have examined a banned load on the real tree, so a renamed
    # function, a deleted file or a dropped kill turns the test red, not silently green
    # (`hygiene_kills_and_the_sigterm_test`). PITFALLS: the key is the innermost enclosing
    # def's dotted name (`_enclosing_functions`), so a load at module level of an exempt
    # file, in any other function of it, in a same-named method (`T._run_group`) or in the
    # exempt def's own decorators or parameter defaults (evaluated at import) is red; and
    # `from os import killpg` is red even inside an exempt function. Never exempt by file.
    # A runner test that needs a hygiene kill cannot be listed here: give it a fake.
    KILL_HYGIENE_EXEMPT = frozenset({
        ("tests/test_arena_duel.py", "_run_group"),
        ("tests/test_arena_duel.py", "test_run_group_leaves_no_survivor"),
        ("tests/test_vis_pool_parent_death.py", "_kill_child_and_count_survivors"),
    })
    # The runner, as import targets: a file owning a KILL_HYGIENE_EXEMPT row may import
    # neither the package (any module of it), nor the Modal entry script, nor the shared
    # helper module, which binds `mrl`, `checkpoint`, `request` and `training` from the
    # runner at module level, so `from <helpers> import training` reaches ProcessControl in
    # one hop without spelling `scripts.`. PITFALL: the entry script's and the helper's
    # dotted names are DERIVED from their path constants, never spelled: any string constant
    # in this class or its test containing the entry script's dotted name is the
    # `_CLIENT_MODULES` seed of tests/test_modal_packaging.py (`_reaches_client_directly`, a
    # substring match that reads docstrings too), which reclassifies both to the client file
    # and turns the split red. The helper's is derived the same way so the two stay alike.
    HELPERS = "tests/modal_test_helpers.py"
    RUNNER_PACKAGES = tuple(
        ["scripts.modal_runner"] +
        [path.removesuffix(".py").replace("/", ".") for path in (RUNNER_ENTRY, HELPERS)])

    @classmethod
    def clauses(cls):
        """Every clause, in label order. The test pins the labels: dropping one here is red."""
        return [
            cls.clause_i(),
            cls.clause_ii(),
            cls.clause_iii(),
            cls.clause_iv(),
            cls.clause_v(),
            cls.clause_vi(),
            cls.clause_vii(),
            cls.clause_viii(),
        ]

    @staticmethod
    def _clause(label, name, *, reads, check, population, populated, plants, unpopulated=None):
        """One clause. `reads` names the source sets it checks: "training", "tests" or both.

        `unpopulated` (optional) is `{sample: examined}`, lists `populated` must reject: the
        plants of a population that pins something (an allow-set), which `check`'s plants
        cannot reach because they test problems, not the population.
        """
        return SimpleNamespace(label=label,
                               name=f"{label} {name}",
                               reads=reads,
                               check=check,
                               population=population,
                               populated=populated,
                               plants=plants,
                               unpopulated=unpopulated or {})

    @staticmethod
    def _at(path, sources):
        """`{plant: source}` as `{plant: {path: source}}`: every plant planted at `path`."""
        return {plant: {path: source} for plant, source in sources.items()}

    # ── Shared AST readers

    @staticmethod
    def _last_name(expr):
        """The callee spelling a clause keys on: a bare name, or the last attribute."""
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Attribute):
            return expr.attr
        return None

    @staticmethod
    def _is_docstring(statement):
        return (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str))

    @staticmethod
    def _process_control_classes(tree):
        return [
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ProcessControl"
        ]

    # ── (i) no field defaults

    @classmethod
    def clause_i(cls):
        fields_block = ("    spawn: Callable[..., object]\n"
                        "    getpgid: Callable[[int], int]\n"
                        "    killpg: Callable[[int, int], None]\n")
        return cls._clause(
            "(i)",
            "ProcessControl's fields have no defaults",
            reads=("training", ),
            check=cls.no_field_defaults,
            population="the four ProcessControl fields",
            populated=lambda seen: tuple(seen) == cls.FIELDS,
            plants=cls._at(
                cls.TRAINING, {
                    "a default on one field":
                    ("class ProcessControl:\n" + fields_block +
                     "    install_signal: Callable[..., object] = signal.signal\n"),
                    "a field(default=...)":
                    ("class ProcessControl:\n" + fields_block +
                     "    install_signal: Callable[..., object] = field(default=signal.signal)\n"),
                }))

    @classmethod
    def no_field_defaults(cls, sources):
        """(i). A default is how the incident's draft handed out the real functions.

        `examined` is the annotated field names, in order.
        """
        problems, examined = [], []
        for rel, text in sources.items():
            for process_control in cls._process_control_classes(ast.parse(text)):
                fields, wrong = cls._fields_and_problems(rel, process_control)
                examined.extend(fields)
                problems.extend(wrong)
        return problems, examined

    @classmethod
    def _fields_and_problems(cls, rel, process_control):
        """One ProcessControl class body: its annotated fields, and what breaks clause (i).

        Only fields, methods and the docstring may appear: any other statement
        (a plain `x = ...`) would be a class attribute no field check reads.
        """
        fields, problems = [], []
        for statement in process_control.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                fields.append(statement.target.id)
                if statement.value is not None:
                    problems.append(f"{rel}:{statement.lineno} field {statement.target.id} "
                                    "has a default")
            elif not (cls._is_docstring(statement)
                      or isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))):
                problems.append(f"{rel}:{statement.lineno} a class-body "
                                f"{type(statement).__name__}, not a field or a method")
        if tuple(fields) != cls.FIELDS:
            problems.append(f"{rel}: the fields are {fields}, not {list(cls.FIELDS)}; if that is "
                            "deliberate, update every clause here and the tripwire's calls in "
                            "test_process_control_tripwire_poisons_system")
        return fields, problems

    # ── (ii) four keywords per construction

    @classmethod
    def clause_ii(cls):
        return cls._clause(
            "(ii)",
            "every ProcessControl(...) under tests/ passes four keywords",
            reads=("tests", ),
            check=cls.four_keyword_constructions,
            population="the tripwire's construction in tests/conftest.py",
            populated=lambda seen: any(where.startswith("tests/conftest.py:") for where in seen),
            plants=cls._at(
                cls.PLANT_TEST, {
                    "a ** splat":
                    "control = training.ProcessControl(**fields)\n",
                    "a missing field":
                    "ProcessControl(spawn=f, getpgid=g, killpg=k)\n",
                    "positional fields":
                    "ProcessControl(f, g, k, s)\n",
                    "a splat through an import alias":
                    ("from scripts.modal_runner.training import ProcessControl as PC\n"
                     "PC(spawn=f, **rest)\n"),
                    "a splat through an assignment alias":
                    "PC = training.ProcessControl\nPC(*parts)\n",
                }))

    @classmethod
    def _process_control_spellings(cls, tree):
        """`ProcessControl` and every import or assignment alias of it in `tree`."""
        spellings = {"ProcessControl"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                spellings |= {
                    alias.asname
                    for alias in node.names if alias.name == "ProcessControl" and alias.asname
                }
            elif isinstance(node, ast.Assign) and cls._last_name(node.value) == "ProcessControl":
                spellings |= {target.id for target in node.targets if isinstance(target, ast.Name)}
        return spellings

    @classmethod
    def _constructions(cls, tree):
        """Every call in `tree` whose callee is ProcessControl or an alias of it."""
        spellings = cls._process_control_spellings(tree)
        return [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and cls._last_name(node.func) in spellings
        ]

    @classmethod
    def _four_keywords(cls, call):
        """Whether `call` passes exactly the four fields as keywords, and nothing else."""
        names = sorted(k.arg for k in call.keywords if k.arg is not None)
        splat = any(k.arg is None for k in call.keywords)
        return not call.args and not splat and names == sorted(cls.FIELDS)

    @classmethod
    def four_keyword_constructions(cls, sources):
        """(ii). No positional argument and no `*`/`**` splat, whose fields cannot be read
        statically (a builder is the likely place for one). `examined` is every construction.
        """
        problems, examined = [], []
        for rel, text in sources.items():
            for call in cls._constructions(ast.parse(text)):
                examined.append(f"{rel}:{call.lineno}")
                if not cls._four_keywords(call):
                    problems.append(f"{rel}:{call.lineno} {ast.unparse(call)[:120]}")
        return problems, examined

    # ── (iii) the real functions only in system(); `.system` read under tests/ only by the tripwire

    @classmethod
    def clause_iii(cls):
        pc = "class ProcessControl:\n"
        live = "class _LiveAttempt:\n    def kill(self):\n"
        system = (
            "    @classmethod\n"
            "    def system(cls):\n"
            "        return cls(spawn=subprocess.Popen, getpgid=os.getpgid, killpg=os.killpg,\n"
            "                   install_signal=signal.signal)\n")
        inert_kill = ("import os\n\n\ndef _never_called():\n"
                      "    os.killpg(os.getpgid(os.getpid()), 0)\n")
        return cls._clause(
            "(iii)", "os.killpg/os.getpgid/signal.signal/subprocess.Popen only inside "
            "ProcessControl.system() in training.py, across the runner; `.system` read under "
            "tests/ only by the two tripwire tests",
            reads=("runner", "tests"),
            check=cls.real_functions_only_in_system,
            population=("system()'s os.getpgid, os.killpg, signal.signal and subprocess.Popen, "
                        "and both exempt tests' calls"),
            populated=lambda seen: set(seen) == {
                "os.getpgid", "os.killpg", "signal.signal", "subprocess.Popen"
            } | {f"{file}::{test}"
                 for file, test in cls.SYSTEM_READERS},
            plants={
                **cls._at(
                    cls.TRAINING, {
                        "the pre-W5 fallback in execute": ("def execute_training_attempt(*, killpg=None):\n"
                                                           "    def train():\n"
                                                           "        return _run(killpg=os.killpg if killpg is None else killpg)\n"),
                        "a custom __init__ with real defaults": (pc + "    def __init__(self, spawn, getpgid=os.getpgid, killpg=os.killpg,\n"
                                                                 "                 install_signal=signal.signal):\n"
                                                                 "        pass\n"),
                        "a second factory": (pc + "    @classmethod\n"
                                             "    def with_spawn(cls, spawn):\n"
                                             "        return cls(spawn=spawn, getpgid=os.getpgid, killpg=os.killpg,\n"
                                             "                   install_signal=signal.signal)\n"),
                        "a __post_init__ swap": (pc + "    def __post_init__(self):\n"
                                                 '        object.__setattr__(self, "killpg", os.killpg)\n'),
                        "a default on system() itself": (pc + "    @classmethod\n"
                                                         "    def system(cls, killpg=os.killpg):\n"
                                                         "        return cls(spawn=subprocess.Popen, getpgid=os.getpgid,\n"
                                                         "                   killpg=killpg, install_signal=signal.signal)\n"),
                        "getattr":
                        live + '        getattr(os, "killpg")(self.child.pid, 15)\n',
                        "a kill by another name":
                        live + "        os.kill(-self.child.pid, signal.SIGTERM)\n",
                        "posix":
                        "import posix\nposix.killpg(4242, 15)\n",
                        "a direct signal.signal": ("class _LiveAttempt:\n    def install_handlers(self):\n"
                                                   "        signal.signal(signal.SIGTERM, self.on_signal)\n"),
                        "a module attribute's signal":
                        "install = core.signal.signal\n",
                        "a from-import":
                        "from os import killpg\n",
                        "a from-import of signal":
                        "from signal import signal as install\n",
                        "a direct Popen": ("class _LiveAttempt:\n    def spawn(self, prepared):\n"
                                           "        self.child = subprocess.Popen(\n"
                                           "            prepared.train_command,\n"
                                           "            start_new_session=True)\n"),
                        "a from-import of Popen":
                        "from subprocess import Popen\n",
                    }),
                **cls._at(
                    cls.RUNNER_PLANT, {
                        "an os.killpg in another runner module":
                        inert_kill,
                        "a system() in another runner module":
                        pc + system,
                        "a from-import of kill in another runner module":
                        "from os import kill\n",
                        "a Popen in another runner module":
                        "child = subprocess.Popen(command, start_new_session=True)\n",
                    }),
                "a signal.signal in the Modal entry script": {
                    cls.RUNNER_ENTRY: "signal.signal(signal.SIGTERM, on_term)\n",
                },
                **cls._at(
                    cls.PLANT_TEST, {
                        "the incident's half-fake":
                        "control = dataclasses.replace(training.ProcessControl.system(), spawn=fake)\n",
                        "an instance read":
                        'kwargs["process"].system()\n',
                        "a read through type()":
                        "type(control).system()\n",
                        "getattr":
                        'getattr(training.ProcessControl, "system")()\n',
                        "an unbound read":
                        "factory = training.ProcessControl.system\n",
                        "an exempt test's name in another file": ("def test_process_control_tripwire_poisons_system():\n"
                                                                  "    training.ProcessControl.system()\n"),
                    }),
                "an exempt test that no longer calls it": {
                    "tests/test_modal_training.py":
                    ("def test_process_control_tripwire_poisons_system():\n"
                     "    pass\n\n\n"
                     f"def {cls.RESOLUTION_CONTROL}():\n"
                     "    training.ProcessControl.system()\n"),
                },
            })

    @classmethod
    def real_functions_only_in_system(cls, sources):
        """(iii). In every runner module (a path under `scripts/`) the real os/posix
        `killpg`, `getpgid`, `kill`, `signal.signal` and `subprocess.Popen`
        (`REAL_FUNCTIONS`) are loaded only inside the body of training.py's
        `ProcessControl.system()`, in the spellings (iv) reads (`_banned_load`,
        `_banned_from_imports`): a direct Popen would spawn the real train command past the
        tripwire, which poisons only what `system()` returns, and the tripwire does not
        reach a raw `os.killpg` in any module at all. A `ProcessControl.system()` in another
        module earns no allowance. Under tests/, any read
        of an attribute named `system` on any receiver counts, because a
        ProcessControl instance or `type(control)` reaches the same classmethod: it is
        allowed only inside the `(file, test)` pairs of `SYSTEM_READERS`, and each pair
        whose file is read must still call it. `examined` is system()'s loads, unparsed,
        and `file::test` for each exempt test that calls it.
        """
        problems, examined = [], []
        for rel, text in sources.items():
            tree = ast.parse(text)
            if rel.startswith("scripts/"):
                found, seen = cls._real_function_loads(rel, tree)
            else:
                found, seen = cls._system_reads_in_tests(rel, tree)
            problems.extend(found)
            examined.extend(seen)
        return problems, examined

    @classmethod
    def _real_function_loads(cls, rel, tree):
        """(iii), one runner module: `(problems, examined)` for the real functions' loads.
        Only training.py's `ProcessControl.system()` body is allowed them."""
        inside = cls._system_body(tree) if rel == cls.TRAINING else set()
        problems, examined = [], []
        for modules, banned in cls.REAL_FUNCTIONS:
            problems.extend(cls._banned_from_imports(rel, tree, modules, banned))
            names = cls._module_names(tree, modules)
            for node in ast.walk(tree):
                loaded = cls._banned_load(node, names, {}, modules, banned)
                if loaded is None:
                    continue
                if id(loaded) in inside:
                    examined.append(ast.unparse(loaded))
                else:
                    problems.append(f"{rel}:{loaded.lineno} {ast.unparse(loaded)[:120]}")
        return problems, examined

    @classmethod
    def _system_body(cls, tree):
        """The ids of every node in the body of each module-level ProcessControl's `system`:
        not its decorators or parameter defaults, which run where the class is defined."""
        return {
            id(node)
            for process_control in cls._process_control_classes(tree)
            for method in process_control.body
            if isinstance(method, ast.FunctionDef) and method.name == "system"
            for statement in method.body for node in ast.walk(statement)
        }

    @classmethod
    def _system_reads(cls, tree):
        """Every read in `tree` of an attribute named `system`, on any receiver: `<x>.system`
        loaded, and `getattr(<x>, "system")`."""
        reads: list[ast.Attribute | ast.Call] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr == "system" and isinstance(node.ctx, ast.Load):
                    reads.append(node)
            elif (isinstance(node, ast.Call) and cls._last_name(node.func) == "getattr"
                  and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
                  and node.args[1].value == "system"):
                reads.append(node)
        return reads

    @classmethod
    def _system_reads_in_tests(cls, rel, tree):
        """(iii), one tests/ file: `(problems, examined)` for its `.system` reads."""
        exempt = {test for file, test in cls.SYSTEM_READERS if file == rel}
        owner = {
            id(node): top.name
            for top in tree.body if isinstance(top, ast.FunctionDef) and top.name in exempt
            for node in ast.walk(top)
        }
        callees = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        problems, calling = [], set()
        for read in cls._system_reads(tree):
            if id(read) not in owner:
                problems.append(f"{rel}:{read.lineno} {ast.unparse(read)[:120]}")
            elif id(read) in callees:
                calling.add(owner[id(read)])
        problems.extend(f"{rel}: {test} is exempt from (iii) but no longer calls `.system`; "
                        "drop its SYSTEM_READERS entry" for test in sorted(exempt - calling))
        return problems, [f"{rel}::{test}" for test in sorted(calling)]

    # ── (iv) no load of the kill seam's OS functions under tests/

    @classmethod
    def clause_iv(cls):
        rows = sorted(f"{file}::{function}" for file, function in cls.KILL_HYGIENE_EXEMPT)
        sigterm = "tests/test_modal_training.py:1"
        assert rows, "KILL_HYGIENE_EXEMPT is empty: delete the exemption machinery and its plants"
        # The exempt-file plants stand in one exempt (file, function): the first row, sorted.
        exempt_file, exempt_function = rows[0].split("::")
        return cls._clause(
            "(iv)", "no load of os/posix killpg, getpgid or kill under tests/, outside the "
            "KILL_HYGIENE_EXEMPT functions",
            reads=("tests", ),
            check=cls.kill_seam_loads,
            population=("the real-SIGTERM test's os.kill(os.getpid(), SIGTERM), and a banned "
                        "load in every KILL_HYGIENE_EXEMPT function"),
            populated=cls.hygiene_kills_and_the_sigterm_test,
            unpopulated={
                "the SIGTERM test alone": [sigterm],
                "an exempt row with no load": [sigterm] + rows[1:],
                "an exempt function's name in another file":
                [sigterm] + rows[1:] + [f"{cls.PLANT_TEST}::{exempt_function}"],
                "the exempt rows without the SIGTERM test":
                rows,
                "an exempt row as the SIGTERM test's file":
                [sigterm] + rows[1:] + [f"tests/test_modal_training.py::{exempt_function}"],
            },
            plants={
                **cls._at(
                    exempt_file, {
                        "a module-level load in an exempt file":
                        "os.killpg(1, 15)\n",
                        "a load in a non-exempt function of an exempt file":
                        "def _other():\n    os.killpg(1, 15)\n",
                        "a same-named method in an exempt file":
                        f"class T:\n    def {exempt_function}(self):\n        os.killpg(1, 15)\n",
                        "a load in an exempt function's parameter default":
                        f"def {exempt_function}(argv, kill=os.killpg):\n    pass\n",
                        "a load in an exempt function's decorator":
                        f"@functools.partial(os.killpg, 1)\ndef {exempt_function}():\n    pass\n",
                        "a from-import inside an exempt function":
                        f"def {exempt_function}():\n    from os import killpg\n",
                        "an exempt file importing a runner module":
                        "from scripts.modal_runner import training\n",
                        "an exempt file importing the runner package from scripts":
                        "from scripts import modal_runner\n",
                        "an exempt file importing a runner module as a dotted name":
                        "import scripts.modal_runner.training as t\n",
                        "an exempt file importing the Modal entry script":
                        f"import {cls.RUNNER_PACKAGES[1]}\n",
                        "an exempt file importing the runner through the helper module":
                        f"from {cls.RUNNER_PACKAGES[2]} import training\n",
                        "an exempt file importing the helper module from tests":
                        f"from tests import {cls.RUNNER_PACKAGES[2].split('.')[1]}\n",
                        "an exempt function importing the runner inside its body":
                        f"def {exempt_function}():\n    from scripts.modal_runner import training\n",
                    }),
                **cls._at(
                    cls.PLANT_TEST, {
                        "an exempt function's name in another file":
                        f"def {exempt_function}():\n    os.killpg(1, 15)\n",
                        "the real functions passed explicitly": ("training.ProcessControl(spawn=fake, getpgid=os.getpgid, killpg=os.killpg,\n"
                                                                 "                         install_signal=signal.signal)\n"),
                        "a module alias":
                        "import os as o\no.killpg(4242, 15)\n",
                        "an assignment alias":
                        "o = os\no.killpg(4242, 15)\n",
                        "an annotated assignment alias":
                        "o: object = training.os\no.getpgid(4242)\n",
                        "an os from-imported out of another module": ("from scripts.modal_runner.training import os as tos\n"
                                                                      "tos.getpgid(4242)\n"),
                        "a from-import":
                        "from posix import getpgid\n",
                        "a from-import of kill":
                        "from os import kill as k\n",
                        "a star import":
                        "from os import *\n",
                        "getattr":
                        'getattr(os, "killpg")(4242, 15)\n',
                        "getattr with a computed name":
                        'controls = {n: getattr(os, n) for n in ("getpgid", "killpg")}\n',
                        "an __import__ receiver":
                        '__import__("os").getpgid(1)\n',
                        "an import_module receiver":
                        'importlib.import_module("posix").killpg(4242, 15)\n',
                        "a sys.modules receiver":
                        'sys.modules["posix"].killpg(1, 15)\n',
                        "a module attribute's os": ("training.ProcessControl(spawn=fake, getpgid=lambda pid: pid,\n"
                                                    "                         killpg=training.os.killpg, install_signal=s)\n"),
                        "a nested module attribute's os":
                        "mrl.training.os.getpgid(4242)\n",
                        "another module's os":
                        "subprocess.os.killpg(4242, 15)\n",
                        "kill of a group":
                        "os.kill(-4242, 15)\n",
                        "kill of a group through a module attribute's os":
                        "training.os.kill(-4242, 0)\n",
                        "kill of self through a module alias":
                        "import os as o\no.kill(o.getpid(), 0)\n",
                        "kill of self through posix":
                        "import posix\nposix.kill(posix.getpid(), 0)\n",
                        "kill of self through a module attribute's os":
                        "training.os.kill(training.os.getpid(), 0)\n",
                    }),
            })

    @classmethod
    def hygiene_kills_and_the_sigterm_test(cls, seen):
        """(iv)'s population. `seen` holds the real-SIGTERM test's allowed call, as
        `tests/test_modal_training.py:<line>`, and one `<file>::<function>` per exempt load;
        it must hold the first and every KILL_HYGIENE_EXEMPT row. A row whose function no
        longer loads a banned name (renamed, its file deleted, its kill dropped) fails here,
        so the allow-set cannot go stale and stay green. The line form is matched on digits,
        so a `<file>::<function>` row in this file cannot stand in for the SIGTERM call."""
        prefix = "tests/test_modal_training.py:"
        sigterm = any(where.startswith(prefix) and where[len(prefix):].isdigit() for where in seen)
        rows = {f"{file}::{function}" for file, function in cls.KILL_HYGIENE_EXEMPT}
        return sigterm and rows <= set(seen)

    @classmethod
    def kill_seam_loads(cls, sources):
        """(iv). Passing the real functions explicitly (`ProcessControl(..., killpg=os.killpg,
        ...)`) satisfies (i), (ii) and the tripwire: only this clause stops it.

        `killpg` and `getpgid` may not be loaded from os or posix in any spelling
        `_is_module` recognises, as `<os>.X`, `getattr(<os>, "X")` or `getattr(<os>,
        <a computed name>)`, nor from-imported (`from os import X` or `*`). `kill`
        likewise, except the one allowed call, `os.kill(os.getpid(), ...)`: any
        other target could be `-pgid`, which is `killpg` by another name.

        EXEMPT: a load (not a from-import) whose innermost enclosing def is a
        `(file, function)` row of `KILL_HYGIENE_EXEMPT`, keyed by the dotted name
        `_enclosing_functions` computes; a module-level load in an exempt file is
        not inside any def, so it is red. The exemption is per function, not per
        load, so an exempt function could hand the real `os.killpg` to the runner
        unseen; therefore a file that owns any row is red on any import of
        `RUNNER_PACKAGES` (`_runner_imports`), anywhere in it. `examined` is the
        allowed calls, as `<file>:<line>`, and `<file>::<function>` for each
        exempt load, which the population reads back per row.
        """
        problems, examined = [], []
        for rel, text in sources.items():
            tree = ast.parse(text)
            names = cls._module_names(tree, cls.OS_MODULES)
            allowed = cls._allowed_kills(tree)
            owner = cls._enclosing_functions(tree)
            examined.extend(f"{rel}:{line}" for line in allowed.values())
            problems.extend(cls._banned_from_imports(rel, tree, cls.OS_MODULES, cls.BANNED))
            if any(file == rel for file, _ in cls.KILL_HYGIENE_EXEMPT):
                problems.extend(cls._runner_imports(rel, tree))
            for node in ast.walk(tree):
                loaded = cls._banned_load(node, names, allowed, cls.OS_MODULES, cls.BANNED)
                if loaded is None:
                    continue
                function = owner.get(id(loaded))
                if (rel, function) in cls.KILL_HYGIENE_EXEMPT:
                    examined.append(f"{rel}::{function}")
                else:
                    problems.append(f"{rel}:{loaded.lineno} {ast.unparse(loaded)}")
        return problems, examined

    @classmethod
    def _runner_imports(cls, rel, tree):
        """Every import of a `RUNNER_PACKAGES` module in `tree`, one problem each, anywhere
        in the file (module level or inside a def). `import scripts.modal_runner[.x] [as y]`
        and `from scripts.modal_runner[.x] import z` match on the dotted name, a package or
        any module under it; `from scripts import modal_runner` / `run_modal` and `from
        tests import modal_test_helpers` name the module as the alias under its parent. A
        relative import with a module (`from ..scripts.modal_runner import x`) is read on
        its `module` text like an absolute one; only the bare `from . import x` (`module`
        None) is not read. Import/ImportFrom only: see the test's RESIDUAL for what passes."""
        heads = {tuple(package.rsplit(".", 1)) for package in cls.RUNNER_PACKAGES}
        problems = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                hit = any(cls._under_runner(alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                hit = cls._under_runner(node.module) or any(
                    (node.module, alias.name) in heads for alias in node.names)
            else:
                continue
            if hit:
                problems.append(f"{rel}:{node.lineno} {ast.unparse(node)}: a file with a "
                                "KILL_HYGIENE_EXEMPT row may not import the runner or a "
                                "module that binds it (RUNNER_PACKAGES)")
        return problems

    @classmethod
    def _under_runner(cls, dotted):
        """Whether `dotted` is one of `RUNNER_PACKAGES` or a module under one."""
        return any(dotted == package or dotted.startswith(f"{package}.")
                   for package in cls.RUNNER_PACKAGES)

    @staticmethod
    def _enclosing_functions(tree):
        """`{id(node): dotted name of the innermost def whose BODY holds node}`, by a
        parent walk from the module down.

        The name is the enclosing classes' and defs' names joined with `.`, outermost
        first (`_run_group`, `TestX.test_y`, `test_y.inner`), without Python's
        `<locals>` marker. Only a def's `body` counts as inside it: its decorators,
        parameter defaults and annotations run where the def is bound, so they keep
        the enclosing scope. A class body adds its name to the path but is not a def:
        a node directly under a module-level class is absent, like a module-level one.
        ASYMMETRY, kept on purpose: a load inside a lambda or a comprehension in an
        exempt def inherits the def's name (exempt), while one inside a nested def
        is owned by `outer.inner` (red): a lambda or comprehension has no def name
        a KILL_HYGIENE_EXEMPT row could key on, a nested def does.
        """
        owner = {}
        stack = [(tree, "", False)]
        while stack:
            node, name, inside = stack.pop()
            scoped = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            body = {id(statement) for statement in node.body} if scoped else set()
            inner = f"{name}.{node.name}" if scoped and name else (node.name if scoped else name)
            is_def = scoped and not isinstance(node, ast.ClassDef)
            for child in ast.iter_child_nodes(node):
                if id(child) in body:
                    child_name, child_inside = inner, inside or is_def
                else:
                    child_name, child_inside = name, inside
                if child_inside:
                    owner[id(child)] = child_name
                stack.append((child, child_name, child_inside))
        return owner

    @classmethod
    def _is_module(cls, expr, names, modules):
        """Whether `expr` spells one of `modules` (os and posix, or signal).

        A name in `names` (`_module_names`); any attribute named after one of
        them, which is how a test reaches a module's own import (`training.os`,
        `mrl.training.os`, `subprocess.os`, `os.path.os`, `training.signal`);
        `__import__("os")` or `importlib.import_module("os")`; or a constant
        subscript such as `sys.modules["posix"]`.
        """
        if isinstance(expr, ast.Name):
            return expr.id in names
        if isinstance(expr, ast.Attribute):
            return expr.attr in modules
        named = None
        if isinstance(expr, ast.Call) and cls._last_name(expr.func) in cls.IMPORT_CALLS:
            named = expr.args[0] if expr.args else None
        elif isinstance(expr, ast.Subscript):
            named = expr.slice
        return isinstance(named, ast.Constant) and named.value in modules

    @classmethod
    def _module_names(cls, tree, modules):
        """Every name `tree` binds to one of `modules`, anywhere in the file.

        The module names themselves; `import os as o`; `from <any module>
        import os [as o]`; and an assignment `o = <anything _is_module
        accepts>`, plain or annotated. The assignments are read in `ast.walk`
        order, so a chain (`o = os; p = o`) is followed when each link comes
        first in that order. `from signal import signal as s` binds a function,
        not the module, but over-reading it only widens what the clauses
        reject.
        """
        names = set(modules) | {
            alias.asname or alias.name
            for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names if alias.name in modules
        }
        for node in ast.walk(tree):
            targets, value = cls._assignment(node)
            if value is not None and cls._is_module(value, names, modules):
                names |= {target.id for target in targets if isinstance(target, ast.Name)}
        return names

    @staticmethod
    def _assignment(node):
        """`(targets, value)` of a plain or annotated assignment with a value, else `([], None)`."""
        if isinstance(node, ast.Assign):
            return node.targets, node.value
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            return [node.target], node.value
        return [], None

    @staticmethod
    def _bare_os_call(expr, attr):
        """`expr` if it is a call `os.<attr>(...)` on the bare name `os`, else None."""
        if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute)
                and expr.func.attr == attr and isinstance(expr.func.value, ast.Name)
                and expr.func.value.id == "os"):
            return expr
        return None

    @classmethod
    def _allowed_kills(cls, tree):
        """`{id(the os.kill attribute): line}` for every `os.kill(os.getpid(), ...)`.

        Spelled exactly so, with the bare name `os` on both calls and no argument
        to `getpid`: an alias, `posix` or a module attribute's `os` is not the
        allowed shape (clause (iv)), even with the same target.
        """
        allowed = {}
        for node in ast.walk(tree):
            kill = cls._bare_os_call(node, "kill")
            if kill is None or not kill.args:
                continue
            target = cls._bare_os_call(kill.args[0], "getpid")
            if target is not None and not (target.args or target.keywords):
                allowed[id(kill.func)] = kill.lineno
        return allowed

    @staticmethod
    def _banned_from_imports(rel, tree, modules, banned):
        """`from <one of modules> import <a banned name>|*`, each alias one problem."""
        return [
            f"{rel}:{node.lineno} from {node.module} import {alias.name}" for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module in modules for alias in node.names
            if alias.name in banned | {"*"}
        ]

    @classmethod
    def _banned_load(cls, node, names, allowed, modules, banned):
        """`node` if it loads a `banned` function from one of `modules`, else None.

        `<m>.X`, unless its id is in `allowed` (the callee of an allowed kill);
        `getattr(<m>, "X")`; and `getattr(<m>, <anything but a constant>)`, whose
        name cannot be read (`{n: getattr(os, n) for n in ("getpgid", "killpg")}`).
        """
        if isinstance(node, ast.Attribute):
            loads = (isinstance(node.ctx, ast.Load) and node.attr in banned
                     and cls._is_module(node.value, names, modules) and id(node) not in allowed)
            return node if loads else None
        if (isinstance(node, ast.Call) and cls._last_name(node.func) == "getattr"
                and len(node.args) >= 2 and cls._is_module(node.args[0], names, modules)):
            attr = node.args[1]
            harmless = isinstance(attr, ast.Constant) and attr.value not in banned
            return None if harmless else node
        return None

    # ── (v) system() builds exactly the real functions

    @classmethod
    def clause_v(cls):
        return cls._clause(
            "(v)",
            "ProcessControl.system() builds exactly the real functions",
            reads=("training", ),
            check=cls.system_builds_the_real_functions,
            population="one system() construction",
            populated=lambda seen: len(seen) == 1,
            plants=cls._at(
                cls.TRAINING, {
                    "a wrong function":
                    ("class ProcessControl:\n"
                     "    @classmethod\n"
                     "    def system(cls):\n"
                     "        return cls(spawn=subprocess.Popen, getpgid=os.getpgid,\n"
                     "                   killpg=os.killpg, install_signal=signal.getsignal)\n"),
                }))

    @classmethod
    def system_builds_the_real_functions(cls, sources):
        """(v). The values are compared as `ast.unparse` strings, so this test loads none
        of them and passes (iv) itself. `examined` is the one construction.
        """
        problems, examined = [], []
        for rel, text in sources.items():
            for process_control in cls._process_control_classes(ast.parse(text)):
                call = cls._system_return(process_control)
                if call is None:
                    problems.append(f"{rel}: ProcessControl.system is not one `return cls(...)`")
                    continue
                examined.append(f"{rel}:{call.lineno}")
                built = {k.arg: ast.unparse(k.value) for k in call.keywords}
                if (cls._last_name(call.func) not in ("cls", "ProcessControl") or call.args
                        or built != cls.REAL_SYSTEM):
                    problems.append(f"{rel}:{call.lineno} system() builds {ast.unparse(call)}")
        return problems, examined

    @classmethod
    def _system_return(cls, process_control):
        """The call `system()` returns, when there is exactly one `system` and its body
        (docstring aside) is the one statement `return <call>`; otherwise None."""
        systems = [
            node for node in process_control.body
            if isinstance(node, ast.FunctionDef) and node.name == "system"
        ]
        body = [s for s in systems[0].body if not cls._is_docstring(s)] if len(systems) == 1 else []
        only = body[0] if len(body) == 1 else None
        returned = only.value if isinstance(only, ast.Return) else None
        return returned if isinstance(returned, ast.Call) else None

    # ── (vi) ProcessControl.system called exactly once, in execute_training_attempt's body

    @classmethod
    def clause_vi(cls):
        execute = "def execute_training_attempt(*, process=None"
        resolve = "        control = {} if process is None else process\n"
        return cls._clause(
            "(vi)", "ProcessControl.system is read once in training.py, called in "
            "execute_training_attempt's body",
            reads=("training", ),
            check=cls.system_read_once_in_execute,
            population="the one `ProcessControl.system()` call in execute_training_attempt",
            populated=lambda seen: len(seen) == 1,
            plants=cls._at(
                cls.TRAINING, {
                    "a module-scope binding called in the body":
                    ("_SYSTEM = ProcessControl.system\n\n\n" + execute + "):\n"
                     "    def train():\n" + resolve.format("_SYSTEM()") + "    return train()\n"),
                    "a default value": (execute + ", fallback=ProcessControl.system()):\n"
                                        "    return fallback if process is None else process\n"),
                    "a nested def's default":
                    (execute + "):\n"
                     "    def train(control=ProcessControl.system()):\n"
                     "        return control if process is None else process\n"
                     "    return train()\n"),
                    "a decorator": (execute + "):\n"
                                    "    @uses(ProcessControl.system())\n"
                                    "    def train():\n"
                                    "        return process\n"
                                    "    return train()\n"),
                    "the read in another top-level function":
                    ("def _resolve(process):\n"
                     "    return ProcessControl.system() if process is None else process\n\n\n" +
                     execute + "):\n    return _resolve(process)\n"),
                    "the attribute bound to a name":
                    (execute + "):\n"
                     "    factory = ProcessControl.system\n" +
                     resolve.format("factory()").removeprefix("    ")),
                    "getattr":
                    (execute + "):\n" +
                     resolve.format('getattr(ProcessControl, "system")()').removeprefix("    ")),
                    "an instance read":
                    "def execute_training_attempt(*, process):\n    return process.system()\n",
                    "a class alias": ("PC = ProcessControl\n\n\n" + execute + "):\n" +
                                      resolve.format("PC.system()").removeprefix("    ")),
                    "a second read":
                    (execute + "):\n" +
                     resolve.format("ProcessControl.system()").removeprefix("    ") +
                     "    spare = ProcessControl.system()\n"),
                    "no read":
                    execute + "):\n    return process\n",
                }))

    @classmethod
    def system_read_once_in_execute(cls, sources):
        """(vi). Without it the tripwire can be bypassed silently: a real factory captured
        at import (`_SYSTEM = ProcessControl.system`, or a default value) passes (i)-(v) and
        the tripwire test (whose patch it never sees), and a forgotten `process` then gets
        the real Popen and handlers. Every read of an attribute named `system` counts
        (`_system_reads`); the one allowed is a call `ProcessControl.system()`, on that bare
        name, inside the body of the module-level `execute_training_attempt`
        (`_body_nodes`), and there must be exactly one. `examined` is where it is.
        """
        problems, examined = [], []
        for rel, text in sources.items():
            tree = ast.parse(text)
            body = set()
            for top in tree.body:
                if isinstance(top, ast.FunctionDef) and top.name == cls.EXECUTE:
                    body |= cls._body_nodes(top)
            callees = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
            allowed = []
            for read in cls._system_reads(tree):
                if (isinstance(read, ast.Attribute) and id(read) in body and id(read) in callees
                        and isinstance(read.value, ast.Name) and read.value.id == "ProcessControl"):
                    allowed.append(f"{rel}:{read.lineno}")
                else:
                    problems.append(f"{rel}:{read.lineno} {ast.unparse(read)[:120]}")
            if len(allowed) != 1:
                problems.append(f"{rel}: {len(allowed)} `ProcessControl.system()` calls in "
                                f"{cls.EXECUTE}'s body, not exactly one")
            examined.extend(allowed)
        return problems, examined

    @staticmethod
    def _body_nodes(function):
        """The ids of every node in `function`'s body, nested defs, lambdas and classes
        followed into their bodies only.

        The resolution sits in the nested `train()`, so the whole body subtree
        counts. A default value, a decorator, an annotation or a base class does
        not: the function's own run at import, and a nested def's are evaluated
        where that def statement runs, which the rule keeps out on purpose.
        """
        found, pending = set(), list(function.body)
        while pending:
            node = pending.pop()
            found.add(id(node))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                pending.extend(node.body)
            elif isinstance(node, ast.Lambda):
                pending.append(node.body)
            else:
                pending.extend(ast.iter_child_nodes(node))
        return found

    # ── (vii) killpg/getpgid called only inside the process-group guard, across the runner

    @classmethod
    def clause_vii(cls):
        live = "class _LiveAttempt:\n    def kill(self):\n"
        stop = "def stop(process, child):\n"
        same_name = ("def _signal_process_group(child, *, killpg, getpgid):\n"
                     "    killpg(getpgid(child.pid), 15)\n")
        handoff = (stop + "    training._signal_process_group(child, killpg=process.killpg,\n"
                   "                                   getpgid=process.getpgid)\n")
        return cls._clause(
            "(vii)",
            "killpg/getpgid are called only inside training._signal_process_group, across the "
            "runner",
            reads=("runner", ),
            check=cls.kill_calls_only_in_the_guard,
            population=("the guard's own getpgid and killpg calls, and exactly one handoff to "
                        f"each of GUARD_HANDOFFS {cls.GUARD_HANDOFFS}"),
            populated=cls.guard_calls_and_pinned_handoffs,
            unpopulated={
                "a second handoff to one callee": ["getpgid", "killpg", "killpg"] +
                [f"handed to {callee}" for callee in cls.GUARD_HANDOFFS * 2],
                "an allowed callee with no handoff": ["getpgid", "killpg", "killpg"] +
                [f"handed to {callee}" for callee in cls.GUARD_HANDOFFS[1:]],
                "no guard calls": [f"handed to {callee}" for callee in cls.GUARD_HANDOFFS],
            },
            plants={
                **cls._at(
                    cls.RUNNER_PLANT, {
                        "a killpg call in another runner module": stop + "    process.killpg(child.pid, 15)\n",
                        "a guard of the same name in another runner module": same_name,
                        "a handoff to the guard from another runner module": handoff,
                    }),
                "an os.kill in the Modal entry script": {
                    cls.RUNNER_ENTRY: "os.kill(-child.pid, signal.SIGTERM)\n",
                },
                **cls._at(
                    cls.TRAINING, {
                        "a direct kill in a new method":
                        live + "        os.killpg(os.getpgid(self.child.pid), signal.SIGTERM)\n",
                        "a kill by another name":
                        live + "        os.kill(-self.child.pid, signal.SIGTERM)\n",
                        "a kill by another name through posix":
                        live + "        posix.kill(-self.child.pid, 15)\n",
                        "an aliased callee":
                        live + "        kp = self.process.killpg\n        kp(self.child.pid, 15)\n",
                        "getattr":
                        live + '        getattr(self.process, "killpg")(self.child.pid, 15)\n',
                        "handed to a helper that is not the guard": (live + "        _terminate(self.child, killpg=self.process.killpg,\n"
                                                                     "                   getpgid=self.process.getpgid)\n"),
                        "handed to a ProcessControl re-wrap": (live + "        return ProcessControl(spawn=self.process.spawn,\n"
                                                               "                              getpgid=self.process.getpgid,\n"
                                                               "                              killpg=self.process.killpg,\n"
                                                               "                              install_signal=self.process.install_signal)\n"),
                    }),
            })

    @classmethod
    def kill_calls_only_in_the_guard(cls, sources):
        """(vii). A direct call anywhere else bypasses the guard, and the runtime layers
        govern ProcessControl, not a direct call.

        Outside `_signal_process_group`, a load of anything named `killpg` or
        `getpgid` (a bare name, the last attribute on any receiver, or
        `getattr(<x>, "killpg")`) is a problem, so an aliased callee (`kp =
        self.process.killpg`) fails with the direct call. The one exception is
        handing it on as the same-named keyword to a `GUARD_HANDOFFS` callee:
        `_signal_process_group(child, killpg=self.process.killpg, ...)`, and
        system()'s own `cls(killpg=os.killpg, ...)`, which (iii) governs. A
        `ProcessControl(...)` re-wrap is not one. An os/posix `kill` is a problem
        too, since `kill(-pgid, s)` is `killpg` by another name. `examined` is the
        callee names of the calls found inside the guard, plus `handed to <callee>`
        once per call that hands one on, which the population pins to exactly one
        per `GUARD_HANDOFFS` entry. Every runner module is read; the guard and the
        handoffs are allowed in training.py only (`_guard_allowances`), so in any
        other module every such load is a problem.
        """
        problems, examined = [], []
        for rel, text in sources.items():
            tree = ast.parse(text)
            inside, handed, callees = cls._guard_allowances(rel, tree)
            examined.extend(f"handed to {callee}" for callee in callees)
            os_names = cls._module_names(tree, cls.OS_MODULES)
            for node in ast.walk(tree):
                if id(node) in inside:
                    if isinstance(node, ast.Call) and cls._last_name(
                            node.func) in cls.GUARDED_CALLS:
                        examined.append(cls._last_name(node.func))
                    continue
                loaded = cls._kill_load(node, os_names)
                if loaded is not None and id(loaded) not in handed:
                    problems.append(f"{rel}:{loaded.lineno} {ast.unparse(loaded)}")
        return problems, examined

    @classmethod
    def guard_calls_and_pinned_handoffs(cls, seen):
        """(vii)'s population: the guard's own getpgid and killpg calls, and exactly one
        handoff to each `GUARD_HANDOFFS` callee (`examined` holds `handed to <callee>` once per
        handing call). This is what watches the allow-set: a name added to it with no handoff,
        or a second handoff to a name in it, fails here."""
        handoffs = [
            entry.removeprefix("handed to ") for entry in seen if entry.startswith("handed to ")
        ]
        return {"killpg", "getpgid"} <= set(seen) and sorted(handoffs) == sorted(cls.GUARD_HANDOFFS)

    @classmethod
    def _guard_allowances(cls, rel, tree):
        """(vii), one runner module: `(inside, handed, callees)`, the guard's node ids and the
        handoffs' value ids and callees (`_inside_the_guard`, `_handed_to_the_guard`). Empty
        outside training.py: a `_signal_process_group` elsewhere is not the guard, and no
        other module may hand the functions on."""
        if rel != cls.TRAINING:
            return set(), set(), []
        return (cls._inside_the_guard(tree), *cls._handed_to_the_guard(tree))

    @classmethod
    def _kill_load(cls, node, os_names):
        """`node` if (vii) reads it as a load of a kill function, else None."""
        if isinstance(node, (ast.Name, ast.Attribute)):
            named = cls._last_name(node) in cls.GUARDED_CALLS
            os_kill = (isinstance(node, ast.Attribute) and node.attr == "kill"
                       and cls._is_module(node.value, os_names, cls.OS_MODULES))
            return node if isinstance(node.ctx, ast.Load) and (named or os_kill) else None
        if (isinstance(node, ast.Call) and cls._last_name(node.func) == "getattr"
                and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in cls.GUARDED_CALLS):
            return node
        return None

    @classmethod
    def _handed_to_the_guard(cls, tree):
        """`(ids, callees)`: the ids of the values handed on as `killpg=`/`getpgid=` to a
        `GUARD_HANDOFFS` callee, when the value's own last name is the keyword's
        (`killpg=<x>.killpg`); and that callee's name, once per call that hands one on."""
        ids, callees = set(), []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and cls._last_name(node.func) in cls.GUARD_HANDOFFS):
                continue
            handed = {
                id(keyword.value)
                for keyword in node.keywords
                if keyword.arg in cls.GUARDED_CALLS and cls._last_name(keyword.value) == keyword.arg
            }
            if handed:
                ids |= handed
                callees.append(cls._last_name(node.func))
        return ids, callees

    @staticmethod
    def _inside_the_guard(tree):
        """The ids of every node in the module-level `_signal_process_group`."""
        return {
            id(node)
            for guard in tree.body
            if isinstance(guard, ast.FunctionDef) and guard.name == "_signal_process_group"
            for node in ast.walk(guard)
        }

    # ── (viii) the tripwire is one autouse, function-scoped fixture in tests/conftest.py

    @classmethod
    def clause_viii(cls):
        fixture = (f"def {cls.TRIPWIRE}():\n"
                   "    poisoned = training.ProcessControl(spawn=p, getpgid=p, killpg=p,\n"
                   "                                       install_signal=p)\n"
                   "    yield poisoned\n")
        autouse = "@pytest.fixture(autouse=True)\n"
        nested = "if True:\n" + "".join(f"    {line}\n"
                                        for line in (autouse + fixture).splitlines())
        outside = ("poisoned = training.ProcessControl(\n"
                   "    spawn=p, getpgid=p, killpg=p, install_signal=p)\n\n\n"
                   f"{autouse}def {cls.TRIPWIRE}():\n"
                   "    yield poisoned\n")
        return cls._clause(
            "(viii)",
            f"{cls.TRIPWIRE} is one autouse, function-scoped fixture in {cls.CONFTEST} that "
            "builds the poison",
            reads=("tests", ),
            check=cls.tripwire_is_autouse,
            population=f"the one {cls.TRIPWIRE} definition, in {cls.CONFTEST}",
            populated=lambda seen: len(seen) == 1 and seen[0].startswith(f"{cls.CONFTEST}:"),
            unpopulated={
                "no definition": [],
                "one definition outside conftest": ["tests/test_modal_training.py:1"],
                "two definitions": [f"{cls.CONFTEST}:1", f"{cls.CONFTEST}:9"],
            },
            plants={
                **cls._at(
                    cls.CONFTEST, {
                        "autouse dropped": "@pytest.fixture\n" + fixture,
                        "autouse=False": "@pytest.fixture(autouse=False)\n" + fixture,
                        "a wider scope": '@pytest.fixture(autouse=True, scope="session")\n' + fixture,
                        "no decorator": fixture,
                        "a second decorator": autouse + "@wraps(poison)\n" + fixture,
                        "nested under a module-level statement": nested,
                        "the poison built outside it": outside,
                        "no definition": "",
                    }),
                "moved into a test file": {
                    cls.CONFTEST: "",
                    "tests/test_modal_training.py": autouse + fixture,
                },
                "shadowed by a narrower conftest": {
                    cls.CONFTEST: autouse + fixture,
                    "tests/fixtures/conftest.py": "@pytest.fixture\n" + fixture,
                },
            })

    @classmethod
    def tripwire_is_autouse(cls, sources):
        """(viii). The tripwire covers a test that does not name it only because it is
        `autouse`: dropped, or moved into a test file or a narrower conftest, it still reaches
        the tests that name it (both tripwire tests name it or fetch it), and every other
        test that drives the attempt silently loses it. So under tests/ there must be exactly
        one definition named `TRIPWIRE`, at module level in `CONFTEST`, decorated exactly
        `@pytest.fixture(autouse=True)` (function scope: a wider-scoped fixture patches once
        and is undone by nothing a test does), and the one ProcessControl construction, the
        poison, must be inside it. `examined` is that definition.
        """
        found = [(rel, node, tree) for rel, text in sources.items() for tree in [ast.parse(text)]
                 for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef,
                                      ast.AsyncFunctionDef)) and node.name == cls.TRIPWIRE]
        problems = [] if len(found) == 1 else [
            f"{len(found)} definitions of {cls.TRIPWIRE} under tests/, not one: "
            f"{[f'{rel}:{node.lineno}' for rel, node, _tree in found]}"
        ]
        for rel, node, tree in found:
            where = f"{rel}:{node.lineno}"
            decorators = [ast.unparse(decorator) for decorator in node.decorator_list]
            if rel != cls.CONFTEST or node not in tree.body:
                problems.append(f"{where} is not a module-level definition in {cls.CONFTEST}")
            if decorators != ["pytest.fixture(autouse=True)"]:
                problems.append(f"{where} is decorated {decorators}, not exactly "
                                "@pytest.fixture(autouse=True)")
            if len(cls._constructions(node)) != 1:
                problems.append(f"{where} does not build the one poisoned ProcessControl itself")
        return problems, [f"{rel}:{node.lineno}" for rel, node, _tree in found]


def test_kill_seam_static_safety():
    """No test, and no runner code outside the process-group guard, can reach the real kill seam.

    gh#163 W5. On 2026-09-23 an agent's throwaway script ran
    the real `killpg(getpgid(1), SIGTERM)`, which is `kill(-1, SIGTERM)`, and
    ended the user's desktop session. The runtime layers (a ProcessControl
    without field defaults; the tests/conftest.py tripwire) each have a way
    round them that only a static check sees, so this test reads source by AST.
    It NEVER imports or calls what it checks: training.py is read through
    `training.__file__` (which is also how the reach floor credits this test to
    `training`), the rest of the runner package from that file's directory,
    scripts/run_modal.py and every tests/*.py from disk. The "runner" source
    set (the package and scripts/run_modal.py) must hold every module that
    tests/modal_runner_tables.py declares, so a new module is read from the
    day it is declared.

    THE CLAUSES, each a checker with its plants in `_KillSeamClauses` above.
    A plant is source text that is parsed and never run, and must fail its
    clause: a clause no plant can fail is not evidence.
      (i)   ProcessControl's dataclass fields have no defaults, and are exactly
            the four fields.
      (ii)  Every ProcessControl(...) construction under tests/, through the
            name or an import or assignment alias of it, passes the four fields
            as four keywords: no positional argument and no `*`/`**` splat.
      (iii) In every runner module the real os/posix `killpg`, `getpgid` and
            `kill`, `signal.signal` and `subprocess.Popen` are loaded only
            inside training.py's ProcessControl.system(), in the spellings (iv)
            reads. Under tests/, an attribute named `system`
            is read (on any receiver: an instance or `type(control)` reaches the
            same classmethod) only inside the two tripwire tests, each exempt by
            (file, test), and each exempt test must still call it.
      (iv)  No load of os/posix `killpg` or `getpgid` under tests/, in any
            spelling the clause reads: through `os`, `posix`, an alias of
            either, a module attribute's `os` (`training.os`), `__import__`,
            `importlib.import_module` or `sys.modules[...]`; `from <m> import X`
            or `*`; `getattr` with that name or a computed one. `kill` likewise,
            except a call spelled exactly `os.kill(os.getpid(), ...)`. A load
            (never a from-import) inside a function listed in
            `KILL_HYGIENE_EXEMPT` is exempt, by (file, innermost enclosing
            def's dotted name): those functions kill a process group the test
            itself spawned, in a `finally`, so a red test leaks no vis-cache
            workers (gh#251, gh#254). That is hygiene of the test's own child,
            not the seam, which is Modal's ProcessControl: because the exemption
            is per function, not per load, a file that owns a row is also red
            on any import of `scripts.modal_runner`, scripts/run_modal.py or
            tests/modal_test_helpers.py, which re-exports the runner
            (`RUNNER_PACKAGES`, in every Import/ImportFrom spelling
            `_runner_imports` reads), so an exempt function cannot hand the
            real `os.killpg` to the runner.
            A load anywhere else in an exempt file, at module level or in
            another function, is red: never exempt by file.
      (v)   ProcessControl.system() builds exactly spawn=subprocess.Popen,
            getpgid=os.getpgid, killpg=os.killpg, install_signal=signal.signal.
      (vi)  In training.py an attribute named `system` is read exactly once: as
            the callee of a call `ProcessControl.system()` inside the body of
            `execute_training_attempt` (its nested `train()` included; its
            defaults and decorators are not body). That is the `process=None`
            resolution; a module-level alias or a default would capture the real
            factory at import, out of the tripwire's reach.
      (vii) In every runner module, outside training.py's
            `_signal_process_group` (which holds the process-group guard),
            nothing named `killpg` or `getpgid` is loaded, as a bare name, the
            last attribute on any receiver or through `getattr`, except, in
            training.py, handed on as the same-named keyword to that guard or
            to system()'s own `cls(...)` construction, once each
            (`GUARD_HANDOFFS`, pinned); and no os/posix `kill` is loaded.
      (viii) `_process_control_tripwire` is defined once under tests/, at
            module level in tests/conftest.py, decorated exactly
            `@pytest.fixture(autouse=True)`, and builds the poisoned
            ProcessControl itself. A test that does not name the tripwire gets
            it only through `autouse`, which nothing else declares.
    Each clause also asserts its population on the real tree, so a checker that
    silently examines nothing cannot pass: the four fields (i), the tripwire's
    own construction in tests/conftest.py (ii), system()'s four real-function
    loads and both exempt tests' `.system` calls (iii), the real-SIGTERM test's
    `os.kill(os.getpid(), SIGTERM)` and a banned load in every
    `KILL_HYGIENE_EXEMPT` function (iv), so a stale exemption row (function
    renamed, file deleted, kill dropped) is red rather than an unwatched
    allowance, the system() construction (v), the one
    `ProcessControl.system()` call (vi), and the guard's own getpgid and killpg
    calls plus exactly one handoff to each `GUARD_HANDOFFS` callee (vii), the
    one tripwire definition in tests/conftest.py (viii). A
    population that pins an allow-set or a count, as (iv)'s, (vii)'s and
    (viii)'s do, also carries `unpopulated` samples it must reject, so the pin
    itself can be seen to fail. The clause labels are pinned below together with each clause's
    source sets, so a clause dropped from `_KillSeamClauses.clauses()`, or
    narrowed from the runner back to training.py (its plants, which are handed
    to the checker directly, would stay red), is red, not silently unchecked.

    ADDING A CLAUSE: see `_KillSeamClauses`.

    RESIDUAL. The clauses read spellings, not values. Measured against (iv),
    these still pass: `vars(os)["killpg"]`, `os.__dict__["killpg"]`,
    `operator.attrgetter("killpg")(os)`, `inspect.getattr_static(os, "killpg")`,
    `importlib.import_module(<a variable>).killpg`,
    `sys.modules.get("posix").killpg`, an os bound some other way (a parameter
    default `def f(o=os)`, a walrus, a function's return value), ctypes' libc
    `killpg`, a shell `kill` in a subprocess, and code inside a string a child
    interpreter runs. A wrapper passes only when what it wraps does:
    `functools.partial(os.killpg, 0)` is caught, because it loads `os.killpg`.
    `_runner_imports` reads Import/ImportFrom spellings only: in an exempt
    file, `import scripts` then `scripts.modal_runner`, `from scripts import
    *`, `importlib.import_module`, `__import__` and `sys.modules` pass it.
    (i), (v) and (vi) read training.py only: a `ProcessControl.system` captured
    at import in another runner module passes (vi) (the tripwire still poisons
    it when called under pytest). (iii) and (vii) read the runner package and
    scripts/run_modal.py, not the rest of scripts/ or src/: the whole control
    handed out of the runner, to code that then calls its `killpg`, passes
    them. (viii) reads the declaration, not pytest's behaviour; the
    resolution-path control's first statement checks the behaviour. They are a
    backstop for the mistakes this branch has seen, not a sandbox.
    """
    training_file = Path(training.__file__).resolve()
    assert training_file.is_relative_to(ROOT), (
        f"scripts.modal_runner.training was imported from {training_file}, outside this checkout "
        f"({ROOT}), so these clauses would read another tree's training.py")
    runner_files = sorted(
        training_file.parent.glob("*.py")) + [ROOT / _KillSeamClauses.RUNNER_ENTRY]
    sources = {
        "training": {
            _KillSeamClauses.TRAINING: training_file.read_text(encoding="utf-8")
        },
        "runner": {
            path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
            for path in runner_files
        },
        "tests": {
            path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted((ROOT / "tests").rglob("*.py"))
        },
    }
    assert {"tests/conftest.py",
            "tests/test_modal_training.py"} <= set(sources["tests"]), sorted(sources["tests"])[:5]
    # The declared module list: every module in it must be in the runner set (iii) and (vii) read.
    from tests.modal_runner_tables import RUNNER_PATHS
    declared = {*RUNNER_PATHS, "scripts/modal_runner/__init__.py", _KillSeamClauses.RUNNER_ENTRY}
    assert declared <= set(sources["runner"]), (
        f"the runner source set lacks {sorted(declared - set(sources['runner']))}, so (iii) and "
        "(vii) would not read them")
    clauses = _KillSeamClauses.clauses()
    pinned = [("(i)", ("training", )), ("(ii)", ("tests", )), ("(iii)", ("runner", "tests")),
              ("(iv)", ("tests", )), ("(v)", ("training", )), ("(vi)", ("training", )),
              ("(vii)", ("runner", )), ("(viii)", ("tests", ))]
    assert [(clause.label, clause.reads) for clause in clauses] == pinned, (
        "_KillSeamClauses.clauses() lost or gained a clause, or a clause reads other source sets "
        "than pinned here (one narrowed to training.py would stop reading the rest of the runner "
        "while its plants stay red); a new clause also adds its label and reads here")
    for clause in clauses:
        real = {rel: text for scope in clause.reads for rel, text in sources[scope].items()}
        problems, examined = clause.check(real)
        assert clause.populated(examined), (
            f"{clause.name}: examined {examined!r}, not {clause.population}. The checker read "
            "nothing it exists to check, so its green would not be evidence")
        assert problems == [], f"{clause.name}: {problems}"
        for sample, seen in clause.unpopulated.items():
            assert not clause.populated(seen), (
                f"{clause.name}: its population check accepts {sample} ({seen!r}), so what it "
                "pins is not watched")
        assert clause.plants, f"{clause.name} has no plant, so nothing shows it can fail"
        for plant, planted in clause.plants.items():
            assert clause.check(planted)[0], (f"{clause.name}: the plant {plant!r} passed it, so "
                                              "the clause cannot fail and is not evidence")


def test_signal_tests_fire_handlers_in_production_order():
    """No test fires a handler from the test thread with a `killpg` that releases the child,
    unless it is on `_SIGNAL_HOOKS_RELEASE_ALLOWLIST`; hand-written choreography is listed too.

    gh#243. The racy order (a releasing `killpg` inside `finalize` while the
    handler runs on the test thread) let mutant `STOP_HB_DEL` pass 11 of 20
    runs of the identity test. `_signal_hooks` now defaults to a `killpg` that
    releases nothing and `_interrupt_in_production_order` is the one way to
    fire from the test thread. This census reads THIS file by AST and never
    imports what it checks, so a new test cannot quietly opt back into the
    racy order: it must appear on a list, with a reason.

    OWNER. A node's owner is the outermost module-level `def` enclosing it
    (nested defs and classes belong to their test). A node at module scope has
    owner `<module>`, on no list, so it is red; a non-test helper wrapping an
    opt-out is red the same way (rule 1 names the helper, not a test).

    THE RULES (each red names the owner and the rule):
      0. Every load of the bare names `_signal_hooks` and `_wait_until_handlers`
         is the callee of a call. `sh = _signal_hooks` is red: an alias is the
         one spelling rules 1 and 3 cannot see.
      1. A call to `_signal_hooks` whose `release_on` is anything but the
         literal `None` (an `IfExp`, a `Name`, an attribute, a signal), or that
         passes a `**` splat, is an OPT-OUT. Its owner must be a key of
         `_SIGNAL_HOOKS_RELEASE_ALLOWLIST`, and must not also call
         `_interrupt_in_production_order`: the helper promises production order.
      2. Every release-list key is a module-level `test_*` def here that owns
         an opt-out (a stale key is red).
      3. Every call to `_wait_until_handlers` outside the helper has its owner in
         `_SIGNAL_HOOKS_HANDWRITTEN_ALLOWLIST`.
      4. Every direct fire outside the helper likewise. A direct fire is a call
         whose callee is `hooks["installed"][sig]` (a subscript of a subscript
         keyed by the constant `"installed"`), or a name the same owner assigned
         from such a subscript (`handler = hooks["installed"][sig]; handler(...)`).
      5. Every hand-written-list key is a module-level `test_*` def here that
         owns a rule-3 or rule-4 site (a stale key is red).
    The real file must show exactly the pinned populations below (five opt-outs,
    twelve helper calls, four hand-written waits, two direct fires): a checker
    that examined nothing would otherwise be green. Every plant in `plants` is
    parsed, never run, and must produce exactly its named red.

    RESIDUAL (the census cannot see these):
      1. A hand-written fake `killpg` that calls `child.release()` itself: only
         `_signal_hooks` keywords are read, so a wrapper around `hooks["killpg"]`
         (like `killpg_marking_finalize`) is invisible. So is rebinding
         `hooks["killpg"] = <releasing wrapper>` after `_signal_hooks(child)`
         built default hooks, and so is any out-of-band release of the held
         child: `threading.Timer(0.02, child.release).start()`, or a FakeChild
         that dies on `poll`. The census does not see either; both were
         measured green with the racy test green 5 of 5 (gh#243 review).
      2. A direct fire spelled another way: `installed = hooks["installed"];
         installed[sig](...)`, `hooks["installed"].get(sig)(...)`, or a handler
         passed through another def.
      3. Aliases that are not a bare `Name` load: `module._signal_hooks`, or
         `globals()["_signal_hooks"]` (rule 0 catches the bare-name alias only).
      4. Other test files: this reads tests/test_modal_training.py only.
      5. A parametrized test where one arm opts out is allow-listed whole (none
         after gh#243 rewrote the hung-heartbeat test's hooks).
    """
    HELPER = "_interrupt_in_production_order"
    GOVERNED = {"_signal_hooks", "_wait_until_handlers"}

    def owner_of(stmt: ast.stmt) -> str:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return stmt.name
        return "<module>"

    def is_installed_subscript(node: ast.AST) -> bool:
        # `X["installed"][sig]`: a Subscript whose value is a Subscript keyed by "installed".
        return (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Subscript)
                and isinstance(node.value.slice, ast.Constant)
                and node.value.slice.value == "installed")

    def census(source: str, release: dict[str, str], handwritten: dict[str, str]):
        """Return (reds, population). A red is (rule, owner, why)."""
        tree = ast.parse(source)
        reds: list[tuple[int, str, str]] = []
        opt_out_owners: list[str] = []
        helper_owners: set[str] = set()
        wait_owners: list[str] = []
        fire_owners: list[str] = []
        module_tests = {
            stmt.name
            for stmt in tree.body
            if isinstance(stmt, ast.FunctionDef) and stmt.name.startswith("test_")
        }
        for stmt in tree.body:
            owner = owner_of(stmt)
            nodes = list(ast.walk(stmt))
            callee_ids = {id(n.func) for n in nodes if isinstance(n, ast.Call)}
            # Names this owner assigns from `X["installed"][sig]` (rule 4, second form).
            fire_aliases: set[str] = set()
            for n in nodes:
                if isinstance(n, ast.Assign) and is_installed_subscript(n.value):
                    fire_aliases |= {t.id for t in n.targets if isinstance(t, ast.Name)}
                elif (isinstance(n, ast.AnnAssign) and n.value is not None
                      and is_installed_subscript(n.value) and isinstance(n.target, ast.Name)):
                    fire_aliases.add(n.target.id)
            for n in nodes:
                # Rule 0: a governed name is loaded only as a callee.
                if (isinstance(n, ast.Name) and n.id in GOVERNED and isinstance(n.ctx, ast.Load)
                        and id(n) not in callee_ids):
                    reds.append((0, owner, f"loads `{n.id}` other than as a callee (an alias)"))
                if not isinstance(n, ast.Call):
                    continue
                func = n.func
                if isinstance(func, ast.Name) and func.id == HELPER:
                    helper_owners.add(owner)
                elif isinstance(func, ast.Name) and func.id == "_signal_hooks":
                    releasing = [
                        kw for kw in n.keywords
                        if kw.arg is None or (kw.arg == "release_on" and not (
                            isinstance(kw.value, ast.Constant) and kw.value.value is None))
                    ]
                    if releasing:
                        opt_out_owners.append(owner)
                        if owner not in release:
                            reds.append((1, owner, "passes `_signal_hooks` a releasing `killpg` "
                                         "(`release_on=...` or `**`) and is not on "
                                         "_SIGNAL_HOOKS_RELEASE_ALLOWLIST"))
                elif isinstance(func, ast.Name) and func.id == "_wait_until_handlers":
                    if owner != HELPER:
                        wait_owners.append(owner)
                        if owner not in handwritten:
                            reds.append((3, owner, "calls `_wait_until_handlers` by hand and is "
                                         "not on _SIGNAL_HOOKS_HANDWRITTEN_ALLOWLIST"))
                elif (is_installed_subscript(func)
                      or (isinstance(func, ast.Name) and func.id in fire_aliases)):
                    if owner != HELPER:
                        fire_owners.append(owner)
                        if owner not in handwritten:
                            reds.append((4, owner, "fires an installed handler by hand and is "
                                         "not on _SIGNAL_HOOKS_HANDWRITTEN_ALLOWLIST"))
        for owner in sorted(set(opt_out_owners) & helper_owners):
            reds.append((1, owner, f"calls {HELPER} with hooks built by a releasing `killpg`: "
                         "the helper must never run in the racy order"))
        for key in release:
            if key not in module_tests or key not in opt_out_owners:
                reds.append((2, key, "is on _SIGNAL_HOOKS_RELEASE_ALLOWLIST but is not a "
                             "module-level test_* that passes a releasing `killpg` (stale)"))
        for key in handwritten:
            if key not in module_tests or (key not in wait_owners and key not in fire_owners):
                reds.append((5, key, "is on _SIGNAL_HOOKS_HANDWRITTEN_ALLOWLIST but is not a "
                             "module-level test_* that waits for or fires a handler by hand "
                             "(stale)"))
        population = {
            "opt_outs": len(opt_out_owners),
            "helper_calls": len(helper_owners),
            "handwritten_waits": len(wait_owners),
            "direct_fires": len(fire_owners),
        }
        return reds, population

    real = Path(__file__).read_text(encoding="utf-8")
    reds, population = census(real, _SIGNAL_HOOKS_RELEASE_ALLOWLIST,
                              _SIGNAL_HOOKS_HANDWRITTEN_ALLOWLIST)
    assert reds == [], "\n".join(f"rule {rule}: {owner} {why}" for rule, owner, why in reds)
    # The pins: a checker that examined nothing would be green above. Adding
    # a converted test, opt-out or hand-written site moves one of these by one.
    assert population == {
        "opt_outs": 5,
        "helper_calls": 12,
        "handwritten_waits": 4,
        "direct_fires": 2
    }, population

    # Plants: source text that is parsed, never run. Signals are bare names so
    # no plant carries the text the acceptance grep counts. Each yields exactly
    # its named red as (rule, owner); K13's red must name the helper.
    RL, HL = _SIGNAL_HOOKS_RELEASE_ALLOWLIST, _SIGNAL_HOOKS_HANDWRITTEN_ALLOWLIST
    plants: dict[str, tuple[str, dict, dict, tuple[int, str], str | None]] = {
        "K2": ("def test_x(child): _signal_hooks(child, release_on=SIGKILL)\n", {}, {},
               (1, "test_x"), None),
        "K3": (real, {
            **RL, "test_cleanup_closes_log_before_final_commit": "no opt-out here"
        }, HL, (2, "test_cleanup_closes_log_before_final_commit"), None),
        "K4": ("def test_x(installed, originals): _wait_until_handlers(installed, originals)\n", {},
               {}, (3, "test_x"), None),
        "K5": (real, {
            k: v
            for k, v in RL.items() if k != "test_term_grace_is_deadline_not_mandatory_sleep"
        }, HL, (1, "test_term_grace_is_deadline_not_mandatory_sleep"), None),
        "K7": ('def test_x(hooks): hooks["installed"][SIGINT](SIGINT, None)\n', {}, {},
               (4, "test_x"), None),
        "K7b": ('def test_x(hooks):\n    h = hooks["installed"][SIGTERM]\n    h(SIGTERM, None)\n',
                {}, {}, (4, "test_x"), None),
        "K8": (real, RL, {
            **HL, "test_no_such_test": "nothing"
        }, (5, "test_no_such_test"), None),
        "K9":
        ("def test_x(child, arm): _signal_hooks(child, release_on=None if arm else SIGKILL)\n", {},
         {}, (1, "test_x"), None),
        "K10": ("def _held(child): return _signal_hooks(child, release_on=SIGKILL)\n", {}, {},
                (1, "_held"), None),
        "K12":
        ("def test_x(child, **kw): _signal_hooks(child, **kw)\n", {}, {}, (1, "test_x"), None),
        "K13": ("def test_x(child, k, c, h):\n    _signal_hooks(child, release_on=SIGKILL)\n"
                "    _interrupt_in_production_order(k, c, h, SIGINT)\n", {
                    "test_x": "opt-out"
                }, {}, (1, "test_x"), HELPER),
        "K14": ("def test_x(child):\n    sh = _signal_hooks\n    sh(child)\n", {}, {},
                (0, "test_x"), None),
        "K14b": ("def test_x(hooks):\n    w = _wait_until_handlers\n    w(hooks, hooks)\n", {}, {},
                 (0, "test_x"), None),
    }
    for plant, (source, release, handwritten, expected, why_contains) in plants.items():
        planted, _ = census(source, release, handwritten)
        assert [
            (rule, owner) for rule, owner, _why in planted
        ] == [expected], (f"plant {plant!r}: expected exactly the red {expected}, got {planted}")
        if why_contains is not None:
            assert why_contains in planted[0][2], (plant, planted)


def test_process_control_tripwire_poisons_system(_process_control_tripwire,
                                                 process_control_tripwire_error, monkeypatch):
    """Under pytest, ProcessControl.system() returns a control whose every field raises.

    gh#163 W5: the tripwire is the kill seam's third safety layer (see
    ProcessControl). `execute_training_attempt` resolves `process=None` to
    `ProcessControl.system()`, and the autouse fixture in tests/conftest.py
    patches `system` to return its poisoned control. So a
    test or client wrapper that forgets `process` fails loudly on a
    ProcessControlTripwire instead of spawning a real child or signalling a real
    process group.

    SAFETY, in order:
      * The FIRST statement calls `system()`, which only builds a dataclass, and
        asserts that it IS the fixture's poison. With the fixture's one
        `setattr` line deleted, `system()` returns the real control: this
        assertion fails and nothing below runs.
      * The fields called are the FIXTURE'S OWN object's, never those of
        anything `system()` returned. Never change that.
      * The arguments are inert on the real functions too: `killpg(0, 0)` sends
        signal 0, which sends nothing; `getpgid(0)` reads the caller's group;
        `Popen([])` raises IndexError before it forks; `signal.signal(0, None)`
        rejects signal 0. So a real function slipped into the poison fails the
        type check below having done nothing.
    Then, before any field is called: the poison's type is a RuntimeError and
    neither a ValueError nor a ProcessLookupError, the two the attempt swallows
    (the handler install; getpgid/killpg); and the test's own
    `monkeypatch.undo()` leaves the poison in place, because the fixture patches
    through its own MonkeyPatch (tests/conftest.py). All four fields are called,
    because one call would pass a poison whose `spawn` is inert, and spawn is
    the first field an attempt calls; and the field set is compared with the
    dataclass's, so a fifth field cannot go unpoisoned. The resolution path
    itself (execute_training_attempt with `process` omitted) has its own
    control, `test_process_control_tripwire_guards_the_resolution_path` below,
    which does not name the fixture, so it also pins that the fixture is
    autouse; this test names it, so it runs even with `autouse` dropped.
    """
    assert training.ProcessControl.system() is _process_control_tripwire, (
        "ProcessControl.system() is not the tests/conftest.py tripwire's poisoned control, so a "
        "forgotten `process` in this session would reach the REAL spawn and killpg. Nothing "
        "was called.")
    assert issubclass(process_control_tripwire_error, RuntimeError), (
        "the tripwire's poison is not a RuntimeError subclass, as tests/conftest.py declares it")
    assert not issubclass(process_control_tripwire_error, (ValueError, ProcessLookupError)), (
        "the attempt swallows a ValueError from the handler install and a ProcessLookupError "
        "from getpgid/killpg, so a poison of either type would be silent exactly there")
    monkeypatch.undo()
    assert training.ProcessControl.system() is _process_control_tripwire, (
        "the test's own monkeypatch.undo() lifted the tripwire: the fixture must patch through "
        "its own pytest.MonkeyPatch.context(), not the test's monkeypatch. Nothing was called.")
    poisoned = _process_control_tripwire
    calls = {
        "killpg": lambda: poisoned.killpg(0, 0),
        "getpgid": lambda: poisoned.getpgid(0),
        "spawn": lambda: poisoned.spawn([]),
        "install_signal": lambda: poisoned.install_signal(0, None),
    }
    assert {field.name for field in dataclasses.fields(poisoned)} == set(calls)
    for field, call in calls.items():
        with pytest.raises(process_control_tripwire_error, match=f"ProcessControl.{field} was"):
            call()


def test_process_control_tripwire_guards_the_resolution_path(request,
                                                             process_control_tripwire_error,
                                                             tmp_path):
    """execute_training_attempt with `process` omitted meets the tripwire, not the real OS.

    gh#163 W5: the tripwire's resolution-path control. The tripwire test
    above shows only that the fixture patched what it
    patched. This drives the path that matters: `process=None` resolves to
    `ProcessControl.system()` inside `train()`, which under pytest returns the
    fixture's poisoned control, so the attempt's first ProcessControl call,
    `spawn`, raises the dedicated type.

    IT DOES NOT NAME THE FIXTURE, on purpose, unlike the tripwire test above.
    Every other test that drives the attempt (the client wrappers, the binding
    rows, the rest of this file) meets the tripwire only because it is
    autouse; a test that names it would receive it either way. So this test
    sees what they see, and its first statement pins `autouse`.

    SAFETY, three belts. Never weaken any.
      1. The FIRST statement asserts that the fixture is active here although
         nothing requests it. With `autouse=True` dropped it fails before
         anything is called. A fixture moved into this file, still autouse,
         would pass it while the other files lose it: clause (viii) of the
         static test pins the fixture's home and decorator for that.
      2. The poisoned control is then fetched by name, and `system()` must BE
         it. That builds a dataclass and calls no field, so with the fixture's
         one setattr deleted this test fails before anything is called.
      3. The train command is `/nonexistent/cs2rl-tripwire-must-not-run`, so
         even a real spawn (an implementation that captured the real factory
         at import, which clause (vi) of the static test bans) fails to exec
         and runs nothing.
    Read statically: spawn is the first field the attempt calls
    (`_LiveAttempt.spawn`, before `install_handlers`). When it raises, the
    `except Exception` arm runs `finalize(kill_child=True)`, whose
    `_signal_process_group` returns at `child is None` before any getpgid or
    killpg; `release` restores no handler, since none was installed; and
    `deliver_attempt` re-raises the error unchanged. The prepared heartbeat is
    `_noop_heartbeat()`, so no fallback heartbeat thread starts. Every other
    collaborator is the builder's fake: only `process` is popped.
    """
    assert "_process_control_tripwire" in request.fixturenames, (
        "the tests/conftest.py tripwire is not active in a test that does not request it: it is "
        "no longer autouse, so every other test that drives the attempt has lost it. Nothing was "
        "called.")
    tripwire = request.getfixturevalue("_process_control_tripwire")
    assert training.ProcessControl.system() is tripwire, (
        "ProcessControl.system() is not the tests/conftest.py tripwire's poisoned control, so "
        "the attempt below would reach the REAL spawn and killpg. Nothing was called.")
    kwargs = _consume_training_kwargs(
        _training_kwargs(tmp_path,
                         prepared=_prepared_source(
                             tmp_path,
                             train_command=["/nonexistent/cs2rl-tripwire-must-not-run"],
                             heartbeat=_noop_heartbeat())))
    kwargs.pop("process")
    with pytest.raises(process_control_tripwire_error, match="ProcessControl.spawn was"):
        mrl.execute_training_attempt(**kwargs)


# ── Attempt outcome: exit mapping and completion evidence ──────────────────


def test_exit_zero_fails_when_completion_evidence_invalid(tmp_path):
    events: list[str] = []

    class RecordingSink(io.StringIO):

        def close(self):
            events.append("log_closed")
            super().close()

    def commit():
        status_path = kwargs["attempt"].run_root / mrl.STATUS_FILENAME
        events.append(f"commit:{json.loads(status_path.read_text())['status']}")

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=FakeChild(returncode=0, stdout=b"done\n"),
            commit=commit,
            log_sink=RecordingSink(),
            manifest=_make_manifest(),
        ))
    result = mrl.execute_training_attempt(**kwargs)
    assert result != mrl.REDELIVERED
    assert result.status is core.Status.FAILED
    assert result.reason == training.REASON_INVALID_EVIDENCE
    assert result.exit_code == 0
    persisted = json.loads((kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "failed"
    assert "log_closed" in events
    # The first commit carrying the terminal STATUS (finalize's), not any
    # later one: the retry's commit also follows `log_closed` (`_signal_hooks`,
    # ORDER).
    assert events.index("log_closed") < events.index("commit:failed")


def test_exit_zero_with_valid_evidence_completes(tmp_path):
    run_root, manifest, effective, ckpt = _minimal_completed_tree(tmp_path)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=FakeChild(returncode=0, stdout=b"done\n"),
            manifest=manifest,
        ))
    result = mrl.execute_training_attempt(**kwargs)
    assert result.status is core.Status.COMPLETED
    assert result.reason is None
    assert result.exit_code == 0
    persisted = json.loads((kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "completed"
    payload = json.loads((kwargs["attempt"].run_root / "result.json").read_text())
    assert payload["status"] == "completed"
    assert payload["exit_code"] == 0
    assert payload["last_step"] == effective
    assert payload["checkpoint_sha256"] == core.sha256_file(ckpt)


def test_dead_run_and_timeout_have_distinct_reasons(tmp_path):
    dead_root = tmp_path / "dead"
    dead_root.mkdir()
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            dead_root,
            child=FakeChild(returncode=3, stdout=b"dead\n"),
            manifest=_make_manifest(),
        ))
    (kwargs["attempt"].run_root / "checkpoints").mkdir(exist_ok=True)
    (kwargs["attempt"].run_root / "checkpoints" / "dust2_policy_dead.pt").write_bytes(b"autopsy")
    dead = mrl.execute_training_attempt(**kwargs)
    assert dead.status is core.Status.FAILED
    assert dead.reason == training.REASON_DEAD_RUN
    assert dead.exit_code == 3
    assert json.loads(
        (kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_text())["status"] == "failed"

    clock = _FakeClock()
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    timeout_root = tmp_path / "timeout"
    timeout_root.mkdir()
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            timeout_root,
            child=child,
            now=clock.now,
            timeout=timedelta(minutes=120),
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
            manifest=_make_manifest(),
        ))
    thread, finished, boxed = _run_attempt_in_thread(kwargs)
    try:
        _wait_until_handlers(hooks["installed"], hooks["originals"])
        clock.advance(120 * 60)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    timed_out = boxed[0]
    assert not isinstance(timed_out, Exception), timed_out
    assert timed_out.status is core.Status.INTERRUPTED
    assert timed_out.reason == training.REASON_TIMEOUT
    assert timed_out.reason != dead.reason
    assert json.loads(
        (kwargs["attempt"].run_root / mrl.STATUS_FILENAME).read_text())["status"] == "interrupted"
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]
