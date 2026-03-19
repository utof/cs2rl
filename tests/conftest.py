import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "performance: performance-sensitive tests excluded from default pytest runs",
    )


def pytest_collection_modifyitems(config, items):
    explicit_targets = {Path(str(arg)).as_posix() for arg in config.invocation_params.args}
    run_performance = any(
        target.endswith("tests/smoke_test.py") or target.endswith("smoke_test.py")
        for target in explicit_targets
    )
    if run_performance:
        return

    skip_performance = pytest.mark.skip(
        reason="performance test; run explicitly with `uv run pytest tests/smoke_test.py -q -s`"
    )
    for item in items:
        if "performance" in item.keywords:
            item.add_marker(skip_performance)
