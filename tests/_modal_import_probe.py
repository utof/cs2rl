"""pytest plugin: record which names `modal_runner_lib` resolved under, AND
enough session state to prove the import actually executed.

Loaded only via `-p tests._modal_import_probe` by test_modal_packaging.py,
never by the normal suite. The leading underscore keeps it out of collection
(`python_files` defaults to `test_*.py` and this repo configures no override --
there is no `[tool.pytest.ini_options]` in pyproject.toml and no pytest.ini /
tox.ini / setup.cfg at all, so the default stands).

WHY a plugin and not an import: `tests/test_eval_baselines.py` imports the
runner INSIDE a test body, so the second module object only comes into
existence when that test RUNS. Importing the test modules is not enough --
the session has to execute.

WHY `passed` is recorded and not just `modules`: the in-body import is the
SECOND-TO-LAST statement of its test, behind an import and four assertions.
Review demonstrated that making that test fail earlier for an unrelated reason
leaves the module census reading exactly `["scripts.modal_runner_lib"]` -- the
value the gate demands -- while the bare import sits unconverted and unreached.
A probe that only reports what it saw cannot distinguish "clean" from "never
got there". So it also reports which node ids reported `passed`, and the gate
requires the carrier test by name.

PITFALL: `report.passed` is true for the `setup` and `teardown` phases too, so
the `when == "call"` filter is load-bearing. Without it a SKIPPED test
contributes its passing setup report and the carrier assertion goes green over
a test body that never ran -- which is exactly positive control (b) in the task
report.
"""
import json
import os
import sys

# Module-level accumulator rather than a class: the plugin is loaded once per
# session by name, so there is exactly one instance of this list per process.
_PASSED = []


def pytest_runtest_logreport(report):
    """Collect the node id of every test whose BODY ran to completion."""
    if report.when == "call" and report.passed:
        _PASSED.append(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    """Dump the module census + session outcome to `MODAL_IMPORT_PROBE_OUT`.

    Silently inert when the variable is unset, so accidentally loading this
    plugin in a normal run writes nothing and changes no behaviour.

    The census matches on the LAST dotted segment, so it catches the bare
    spelling (`modal_runner_lib`) and the packaged one
    (`scripts.modal_runner_lib`) with one predicate, and would catch any third
    root someone adds to `sys.path` later.
    """
    out = os.environ.get("MODAL_IMPORT_PROBE_OUT")
    if not out:
        return
    keys = sorted(k for k in sys.modules if k.split(".")[-1] == "modal_runner_lib")
    payload = {
        "modules": keys,
        "passed": sorted(_PASSED),
        "exitstatus": int(exitstatus),
    }
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
