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
    # devices is regularly tight. Redirect basetemp to $HOME/.pytest_tmp
    # unless the user passed --basetemp explicitly. $HOME is the user's
    # primary partition and is the obvious place with persistent free space.
    if not config.option.basetemp:
        home_tmp = Path(os.path.expanduser("~/.pytest_tmp"))
        home_tmp.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(home_tmp)


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
