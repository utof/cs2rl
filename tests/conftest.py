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
