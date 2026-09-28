import os
import sys
from collections.abc import Generator, Iterable, Mapping
from pathlib import Path

import pytest

# The repo root, from this file's own location. Never `config.rootpath`, which
# narrows to a subdirectory when pytest is started from one.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def make_map():
    from map import make_simple_map
    return make_simple_map()


# Alias fixture used by test_map.py verticality tests (spec §3.10).
# Returns the same session-scoped instance as make_map.
@pytest.fixture(scope="session")
def simple_map(make_map):
    """Shared simple-map fixture (session-scoped, READ-ONLY).

    Mutating arrays on this fixture (adjacency, vis_matrix, centroids_z, is_ramp,
    ...) corrupts shared state across the entire test session because of session
    scope. If your test needs to mutate map data, build a fresh one with
    make_simple_map() inside the test instead of using this fixture.
    """
    return make_map


# ── The ProcessControl tripwire (gh#163; the kill seam's third safety layer) ──
#
# `execute_training_attempt(process=None)` resolves None to
# `ProcessControl.system()`, the REAL spawn/getpgid/killpg/signal functions.
# Production never passes `process`, and the client tests' wrappers forward
# production's keywords, so a test that forgets its own control would get the
# real ones without ever naming `system`. On 2026-09-23 a real
# `killpg(getpgid(1), SIGTERM)` (= `kill(-1, SIGTERM)`) ended the user's
# desktop session. Under pytest this fixture makes `system()` return a control
# whose every field raises, so that mistake fails loudly instead of spawning or
# signalling. `test_process_control_tripwire_poisons_system` and
# `test_process_control_tripwire_guards_the_resolution_path`
# (tests/test_modal_training.py) pin it.
#
# It covers every OTHER test only because it is `autouse`: dropped, or moved
# into a narrower conftest or a test file, the fixture still reaches a test
# that names it, and nothing else. So `autouse=True` is pinned twice: the
# resolution-path control does NOT name the fixture, and its first statement
# asserts the fixture is active in it anyway; and clause (viii) of
# test_kill_seam_static_safety requires, by AST, exactly one definition, here,
# decorated exactly `@pytest.fixture(autouse=True)` (no wider scope), that
# builds the poison itself.
#
# PITFALLS.
#   * The module is LOOKED UP in sys.modules, never imported: both fixtures below
#     act only in a session that has loaded the runner, and an import of it in
#     this file, even inside a fixture, would load it into every pytest session.
#   * The patch goes through the fixture's OWN `pytest.MonkeyPatch.context()`,
#     never the test's `monkeypatch`: a test body that calls
#     `monkeypatch.undo()` would otherwise restore the real `system` for the rest
#     of that test (tests/test_no_restated_env_defaults.py calls it).
#   * RESIDUAL: three windows are not poisoned, because the fixture is
#     function-scoped and needs the module already loaded.
#       1. A test whose own body is the first thing in the process to import the
#          package: the fixture found nothing to patch. Every test file that
#          drives the attempt imports the package at module scope, so collection
#          has loaded it before any fixture runs. Not every modal test file
#          does: three tests/test_modal_*.py files import it only inside test
#          bodies or not at all, and none of those reaches the attempt.
#       2. Code that runs at collection: module level, parametrize arguments.
#       3. Module-, class- and session-scoped fixtures, setup and teardown.
#     The only cover for all three is static, and partial: clause (iii) of
#     test_kill_seam_static_safety bans any `.system` read (on any receiver)
#     under tests/ outside the two tripwire tests. It does not see an attempt
#     driven with `process` forgotten from one of these windows; today no modal
#     test file has a higher-scoped fixture or drives the attempt at collection.
#   * The poison raises a RuntimeError subclass on purpose. The attempt swallows
#     a ValueError from the handler install and a ProcessLookupError from
#     getpgid/killpg, so a poison of either type would be silent exactly there.
#   * Build the poisoned control with all four fields as keywords: the static
#     safety test rejects a `*`/`**` splat in any ProcessControl(...) under tests/.

# The runner module both fixtures below look up. The tripwire's two tests fail
# if this name stops resolving (the tripwire then patches nothing), so the
# precondition fixture, which reads the same name, cannot go quietly blind.
_TRAINING_MODULE = "scripts.modal_runner.training"


class ProcessControlTripwire(RuntimeError):
    """Raised by every field of the poisoned ProcessControl the tripwire installs."""


def _process_control_poison(field):
    """A stand-in for ProcessControl.<field>: raises ProcessControlTripwire on any call."""

    def poisoned(*_args, **_kwargs):
        raise ProcessControlTripwire(
            f"ProcessControl.{field} was called under pytest: execute_training_attempt was "
            "called without process=... (or something else reached ProcessControl.system()). "
            "Pass a ProcessControl whose spawn, getpgid and killpg are fakes; the training test "
            "builder does.")

    return poisoned


@pytest.fixture(autouse=True)
def _process_control_tripwire():
    """Make `ProcessControl.system()` return a poisoned control; yield that control.

    Yields None, and patches nothing, when the training module is not loaded.
    The patch is undone when the test ends, by the fixture's own MonkeyPatch,
    which nothing the test does to its `monkeypatch` reaches. A knock-out that
    deletes the one `setattr` line (to see the two tripwire tests go red) may be
    run only on those two test nodes, because it switches the backstop off for
    the whole session.
    """
    training = sys.modules.get(_TRAINING_MODULE)
    if training is None:
        yield None
        return
    poisoned = training.ProcessControl(spawn=_process_control_poison("spawn"),
                                       getpgid=_process_control_poison("getpgid"),
                                       killpg=_process_control_poison("killpg"),
                                       install_signal=_process_control_poison("install_signal"))
    with pytest.MonkeyPatch.context() as tripwire_patch:
        tripwire_patch.setattr(training.ProcessControl, "system", staticmethod(lambda: poisoned))
        yield poisoned


@pytest.fixture
def process_control_tripwire_error():
    """The tripwire's exception type, so a test names it without importing conftest."""
    return ProcessControlTripwire


# ── The kill-path tests' own-group precondition ──
#
# The process-group guard in `_signal_process_group` refuses to signal the
# runner's OWN process group. Every modal test that hands the attempt a
# recording killpg -- the training test builder's default, `_signal_hooks`
# (tests/test_modal_training.py), the client tests' execute wrappers, the
# binding campaign -- pairs it with an identity getpgid and a FakeChild whose
# pid is FakeChild's default. In a session whose process group is that pid,
# the guard would refuse every one of their kills: a test asserting a kill
# would fail for a reason unrelated to its subject, and a test asserting that
# nothing was killed (`kills == []`) would pass without testing anything. So
# the session stops here, at the cause, before any test runs. The
# process-group guard test builds its own pids, apart from the session's group
# by construction, and `_signal_hooks` repeats the check for its own child.
#
# PITFALLS.
#   * FakeChild's default pid is stated here AND read from FakeChild whenever
#     tests/modal_test_helpers.py is loaded: a changed default fails the
#     comparison instead of leaving this check stale. The helpers are LOOKED
#     UP in sys.modules, like the runner above: importing them would import the
#     runner.
#   * It runs once per session, at the first test, and checks only if the
#     runner is loaded by then, keyed on `_TRAINING_MODULE` like the tripwire.
#     Collection has finished by then, and every test file that drives the
#     attempt imports the runner at module scope (the tripwire's RESIDUAL 1), so
#     a session that can reach a recording killpg is always checked.
_FAKE_CHILD_PID = 4242


@pytest.fixture(scope="session", autouse=True)
def _kill_path_own_group_precondition():
    """Fail every test of a session whose process group is FakeChild's default pid."""
    if _TRAINING_MODULE not in sys.modules:
        return
    helpers = sys.modules.get("tests.modal_test_helpers")
    if helpers is not None:
        default = helpers.FakeChild.__init__.__kwdefaults__["pid"]
        assert default == _FAKE_CHILD_PID, (
            f"FakeChild's default pid is {default}, but tests/conftest.py checks the session's "
            f"process group against {_FAKE_CHILD_PID}: update `_FAKE_CHILD_PID`")
    assert os.getpgrp() != _FAKE_CHILD_PID, (
        f"this session's process group is {os.getpgrp()}, FakeChild's default pid, so the "
        "process-group guard in `_signal_process_group` refuses every kill the modal kill-path "
        "tests record (their `kills == []` assertions would pass without testing anything). "
        "Run pytest from another process group, e.g. from a new shell.")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "performance: performance-sensitive tests excluded from default pytest runs",
    )
    # R0-D (#135): multi-minute subprocess training tests (test_seed_reproducible).
    # Unregistered markers are an error under --strict-markers.
    config.addinivalue_line(
        "markers",
        "slow: multi-minute subprocess/rollout tests; deselect with -m 'not slow'",
    )

    # Some tests (e.g. test_run_experiment.py::test_full_run_*) spawn the real
    # scripts/run_experiment.py subprocess, which enforces a >=5 GB free-disk
    # precondition on the repo root it's pointed at. The default pytest
    # tmp_path lives under /tmp on the system root partition, which on small
    # devices is regularly tight. So unless the user passed --basetemp or set
    # PYTEST_DEBUG_TEMPROOT, the temp ROOT moves to $HOME/.cache/cs2rl-pytest:
    # $HOME is the user's primary partition, with persistent free space. Keep
    # it on $HOME, not in the repo: on the main dev machine the repo drive is
    # fuseblk/NTFS, where chmod is a no-op, and chmod-based tests (e.g.
    # tests/test_pyrefly_gate.py) break there.
    #
    # Move the ROOT, never basetemp itself. Setting config.option.basetemp puts
    # pytest on its explicit-basetemp path, which rm_rf's that exact directory
    # the first time the session asks for a temp dir (in
    # TempPathFactory.getbasetemp, reached by the first tmp_path, tmpdir or
    # tmp_path_factory use), so two concurrent sessions deleted each other's
    # tmp_path trees mid-run (gh#219). With PYTEST_DEBUG_TEMPROOT pytest keeps
    # its default layout under the new root instead: each session gets its own
    # numbered <root>/pytest-of-<user>/pytest-<N>/, lock-protected while the
    # session runs, and only the oldest beyond the newest 3 are rotated away.
    # pytest reads the variable lazily, in getbasetemp, so setting it here is
    # early enough. Nested pytest sessions that tests spawn without --basetemp
    # inherit the variable through the environment, so they too get their own
    # numbered dir under the same root instead of wiping the outer session's.
    #
    # Why not ~/.pytest_tmp, the old pinned basetemp: conftests from before
    # gh#219, still present on other branches and checkouts, rm_rf that exact
    # directory whenever they run without --basetemp and a test asks for a
    # temp dir, and rm_rf ignores pytest's locks, so live sessions under a
    # root there would be deleted.
    # ~/.cache is the XDG default cache directory, and no old conftest touches
    # cs2rl-pytest. Nothing is created when PYTEST_DEBUG_TEMPROOT is already
    # set, so an explicitly redirected run leaves $HOME alone.
    # tests/test_pytest_tmp_isolation.py pins the per-session basetemp, the
    # default root and the explicit override.
    if not config.option.basetemp and "PYTEST_DEBUG_TEMPROOT" not in os.environ:
        temproot = Path(os.path.expanduser("~/.cache/cs2rl-pytest"))
        temproot.mkdir(parents=True, exist_ok=True)
        os.environ["PYTEST_DEBUG_TEMPROOT"] = str(temproot)


def pytest_collection_modifyitems(config, items):
    explicit_targets = {Path(str(arg)).as_posix() for arg in config.invocation_params.args}
    run_performance = any(
        target.endswith("tests/smoke_test.py") or target.endswith("smoke_test.py")
        for target in explicit_targets)
    if run_performance:
        return

    skip_performance = pytest.mark.skip(
        reason="performance test; run explicitly with `uv run pytest tests/smoke_test.py -q -s`")
    for item in items:
        if "performance" in item.keywords:
            item.add_marker(skip_performance)


# ── One module object per file ──
#
# A file imported under two names (`paths` and `src.paths`) is two module
# objects with two copies of the module's state, so a monkeypatch or an
# `except` aimed at one misses the other. After the last test (or after
# collection, under --collect-only) every session checks sys.modules for a repo
# file held under two names and fails if it finds one, the way pytest-cov fails
# a session on coverage: `session.testsfailed += 1` in a `pytest_runtestloop`
# wrapper. The static half is ruff's TID251 banned-api table in pyproject.toml.
# tests/test_one_module_object_per_file.py pins both the function and the hook.
#
# LIMITS.
#   (a) ruff cannot see a literal `importlib.import_module("...")` or
#       `__import__("...")`, and this guard sees one only when it executes in a
#       session that also loads the other spelling.
#   (b) the ban is enforced by the pre-commit hook, at commit time and on staged
#       files only, so `--no-verify`, merges, rebases and cherry-picks skip it.
#   (c) this is a snapshot of sys.modules at session end, so a second spelling
#       that is evicted before session end is invisible. `fake_modal` in
#       tests/test_modal_client.py pops `scripts.run_modal`,
#       `scripts.modal_artifacts` and `scripts.modal_backfill_sidecar`, so for
#       those three the ruff ban on their bare names is the only check.
#   Under `-x` with a failure, or with collection errors, pytest stops before
#   the check runs; that session fails anyway, and only the report is lost.
def files_under_two_module_names(modules: Mapping[str, object], root: Path,
                                 ignored_prefixes: Iterable[Path]) -> dict[Path, list[str]]:
    """Map every file under `root` that `modules` holds under two or more names to those names.

    Files are compared by resolved real path, so a symlink and its target are one
    file. A module without a string `__file__` (builtins, namespace packages) is
    skipped, and so is every file under one of `ignored_prefixes`. There are no
    name-based exclusions.

    PITFALL: the session passes the interpreter's `sys.prefix` and
    `sys.base_prefix` as `ignored_prefixes`, and both are needed. The stdlib and
    the venv hold legitimate aliases (`os.path` is `posixpath`), and
    `multiprocessing` registers `__main__` a second time as `__mp_main__`: under
    `.venv/bin/pytest` (what `uv run pytest` runs) that file is
    `<repo>/.venv/bin/pytest`, under the repo root and outside any
    site-packages directory, so only the prefix rule excludes it.
    """
    root = root.resolve()
    ignored = [prefix.resolve() for prefix in ignored_prefixes]
    names: dict[Path, list[str]] = {}
    # A copy: sys.modules can change size mid-iteration when another thread imports.
    for name, module in list(modules.items()):
        file = getattr(module, "__file__", None)
        if not isinstance(file, str):
            continue
        path = Path(file).resolve()
        if path.is_relative_to(root) and not any(path.is_relative_to(p) for p in ignored):
            names.setdefault(path, []).append(name)
    return {path: sorted(found) for path, found in names.items() if len(found) > 1}


@pytest.hookimpl(wrapper=True)
def pytest_runtestloop(session: pytest.Session) -> Generator[None, object, object]:
    """Fail the session, with a red report, if a repo file is loaded under two module names."""
    result = yield
    duplicates = files_under_two_module_names(sys.modules, REPO_ROOT,
                                              (Path(sys.prefix), Path(sys.base_prefix)))
    if duplicates:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_sep("=", "repo files loaded under two module names", red=True, bold=True)
            for path, names in duplicates.items():
                reporter.line(f"{path}: {names}", red=True)
            reporter.line(
                "Import each module under one name: `tests.X` for a module under tests/ (the "
                "name pytest collects it under); pyproject.toml's banned-api table names the "
                "spelling for src/, deploy/ and the Modal scripts.",
                red=True)
        session.testsfailed += 1
    return result
