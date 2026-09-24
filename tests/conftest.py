import os
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
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


# ── The ProcessControl tripwire (gh#163 spec §2a, layer 3) ────────────────
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
# PITFALLS.
#   * The module is LOOKED UP in sys.modules, never imported: any import of the
#     runner here, even inside the fixture, makes this file an importer that
#     tests/test_modal_packaging.py's runtime identity probe must then run (its
#     `bare_spelling_imports` census counts imports at any depth), and an
#     `importlib.import_module(<variable>)` would dodge that census through its
#     documented blind spot.
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


class ProcessControlTripwire(RuntimeError):
    """Raised by every field of the poisoned ProcessControl the tripwire installs."""


def _process_control_poison(field):
    """A stand-in for ProcessControl.<field>: raises ProcessControlTripwire on any call."""

    def poisoned(*_args, **_kwargs):
        raise ProcessControlTripwire(
            f"ProcessControl.{field} was called under pytest: execute_training_attempt was "
            "called without process=... (or something else reached ProcessControl.system()). "
            "Pass an all-fake ProcessControl; the training test builder does.")

    return poisoned


@pytest.fixture(autouse=True)
def _process_control_tripwire():
    """Make `ProcessControl.system()` return a poisoned control; yield that control.

    Yields None, and patches nothing, when the training module is not loaded.
    The patch is undone when the test ends, by the fixture's own MonkeyPatch,
    which nothing the test does to its `monkeypatch` reaches. Knock-out 4(k) of
    the spec deletes the one `setattr` line: it may be run only on the two
    tripwire test nodes, because it switches the backstop off for the whole
    session.
    """
    training = sys.modules.get("scripts.modal_runner.training")
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
