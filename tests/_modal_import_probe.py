"""pytest plugin: record which governed runner modules a session reached, under
which import spellings and as which module objects, AND enough session state to
prove the import actually executed.

Loaded only with `-p tests._modal_import_probe`, by the nested session of
`test_modal_runner_resolves_to_exactly_one_module_object` and by any full-suite
run that wants whole-session evidence; never by the normal suite. Test
discovery does not collect it: `python_files` defaults to `test_*.py` and
`*_test.py`, and this repo sets no override (checked 2026-09-23: no
`[tool.pytest.ini_options]` in pyproject.toml, no pytest.ini / tox.ini /
setup.cfg). The leading underscore only marks it private; the pattern is what
keeps it out. A path named on the command line bypasses that pattern, but this
module defines no tests, so such a run collects none (measured 2026-09-23).

No runner imports occur here: a module the session never reached must remain
visible as missing, so the probe cannot satisfy its own every-name clause. The
governed names come from the package directory's LISTING (`_governed_modules`),
which imports nothing and needs only the standard library.

WHY a plugin and not an import: `tests/test_eval_baselines.py` imports the
runner INSIDE a test body, so a module object that import creates only comes
into existence when that test RUNS. Importing the test modules is not enough --
the session has to execute.

WHY `passed` is recorded and not just the census: the in-body import is the
second-to-last statement of its test, and the other importers reach the runner
too, so a carrier that fails before its import can leave a clean census while
the line the gate exists for never ran. A probe that only reports what it saw
cannot distinguish "clean" from "never got there", so it also reports which
node ids passed their call phase, and the gate requires the carrier by name.
"""
import json
import os
import sys
from pathlib import Path


def _governed_modules(package_dir):
    """The dotted names of the package and of every `*.py` submodule in `package_dir`.

    Derived from the directory rather than written out, so a new module is
    governed the moment its file exists, with no list here to update. Only the
    listing is read: nothing is imported, so a module the session never
    reached still shows as missing. `*.py` only, so `__pycache__` and non-Python
    files do not count.

    PITFALL: the glob matches any entry named `*.py`, a directory included,
    as does `scripts/run_modal.py`'s mount loop. Such a directory is governed
    like a module no session can import, so clause 1 reports it (red).

    PITFALL: a wrong `package_dir` yields an empty set, and every clause of
    `assert_module_identity` then passes over nothing.
    `test_runtime_identity_requires_every_governed_module`
    (tests/test_modal_packaging.py) pins `_EXPECTED` to the modules
    RUNNER_MODULES declares, so that cannot happen silently.
    """
    names = set()
    for path in package_dir.glob("*.py"):
        stem = "" if path.name == "__init__.py" else f".{path.stem}"
        names.add(f"scripts.modal_runner{stem}")
    return frozenset(names)


_PACKAGE_DIR = Path(__file__).resolve().parents[1] / "scripts" / "modal_runner"
_EXPECTED = _governed_modules(_PACKAGE_DIR)
# Module-level accumulator rather than a class: the plugin is loaded once per
# session by name, so there is exactly one instance of this list per process.
_PASSED = []


def module_census():
    """Group every `sys.modules` spelling of a governed module under its
    canonical name, recording each spelling's object id and source file.

    Two routes into a group, and each catches a spelling the other cannot:

    - the bare twin `modal_runner[.<sub>]` maps to `scripts.` + its name, so a
      bare spelling is grouped even when its object has no `__file__`;
    - any other name whose `__file__` resolves to a governed module's file is
      grouped by that origin, so a second object loaded under an unrelated name
      (`core`, `pkg.core_alias`) is still seen.
      `test_runtime_identity_groups_a_second_object_by_its_source_file` is the
      control for this route.

    `forbidden_bare` lists the bare names separately, because a bare spelling
    is forbidden even when it aliases the canonical object.

    PITFALL: this reads `sys.modules` once, when it is called. An object that
    was created and later evicted from `sys.modules`, or loaded without ever
    being registered there, is invisible to it.
    """
    snapshot = dict(sys.modules)
    origins = {}
    for name in _EXPECTED:
        origin = getattr(snapshot.get(name), "__file__", None)
        if origin:
            origins[str(Path(origin).resolve())] = name
    identities = {name: [] for name in sorted(_EXPECTED)}
    bare = []
    for name, module in sorted(snapshot.items()):
        origin = getattr(module, "__file__", None)
        if name == "modal_runner" or name.startswith("modal_runner."):
            bare.append(name)
            canonical = "scripts." + name
        elif name in _EXPECTED:
            canonical = name
        else:
            canonical = origins.get(str(Path(origin).resolve())) if origin else None
        if canonical is not None and canonical in _EXPECTED:
            identities[canonical].append({"name": name, "object_id": id(module), "file": origin})
    return {
        "modules": sorted(_EXPECTED.intersection(snapshot)),
        "identities": identities,
        "forbidden_bare": bare
    }


def assert_module_identity(payload):
    """Fail unless `payload` (a `module_census()` result) shows every governed
    module reached, as one object, under one spelling.

    The clauses, in the order they report. Each is pinned by a
    `test_runtime_identity_*` control in tests/test_modal_packaging.py whose
    `match=` stops matching if that clause is deleted:

    1. every governed name is present (a module the session never reached);
    2. one object id per governed module (a second object under any spelling);
    3. one distinct object per governed name (one object bound under two);
    4. no bare spelling (even one that aliases the canonical object);
    5. no second spelling at all (a non-bare alias of the canonical object).

    There is no payload-schema clause and no "canonical entry present" clause:
    `module_census` builds one `identities` entry per governed name and files
    every present canonical name under itself, so neither could fire on its
    output, and clause 1 covers both.
    """
    assert set(payload["modules"]) == _EXPECTED, (
        "every governed module must be in sys.modules when the census runs: the package and "
        f"each entry matching *.py in {_PACKAGE_DIR}. Never reached: "
        f"{sorted(_EXPECTED - set(payload['modules']))}. A submodule imported only inside a "
        "function body is reached only if the session runs that function: import it at module "
        "scope in the package module that uses it, or have a test in the session run that "
        "path. A submodule imported only under `if TYPE_CHECKING:` is never imported at run "
        "time: import it at module scope in a module that reads it at run time, or move its "
        "names into a module that is imported. A submodule nothing imports is dead code: "
        "delete it.")
    for name in sorted(_EXPECTED):
        entries = payload["identities"][name]
        ids = {entry["object_id"] for entry in entries}
        assert len(ids) == 1, f"module object identity: {name}: {entries}"
    objects = {name: entries[0]["object_id"] for name, entries in payload["identities"].items()}
    assert len(set(objects.values())) == len(_EXPECTED), (
        f"distinct governed module object identities required: {objects}")
    assert not payload["forbidden_bare"], (
        f"forbidden bare import spellings: {payload['forbidden_bare']}")
    spellings = {
        name: [entry["name"] for entry in entries]
        for name, entries in payload["identities"].items()
    }
    competing = {name: names for name, names in spellings.items() if names != [name]}
    assert not competing, f"competing import spellings: {competing}"


def pytest_runtest_logreport(report):
    """Collect the node id of every test whose BODY ran to completion.

    PITFALL: `report.passed` is true for the `setup` and `teardown` phases too,
    so the `when == "call"` filter is load-bearing. Without it a SKIPPED test
    contributes its passing setup report, and the gate's carrier assertion goes
    green over a test body that never ran.
    """
    if report.when == "call" and report.passed:
        _PASSED.append(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    """Write the census, the passed call phases and the session's own exit
    status to `MODAL_IMPORT_PROBE_OUT`, and fail a clean session whose module
    identity is wrong.

    Silently inert when the variable is unset, so loading this plugin by
    accident in a normal run writes nothing and changes no behaviour.

    The payload's `exitstatus` is the session's status BEFORE this hook. An
    identity failure turns a clean session's 0 into 1 through
    `session.exitstatus`, which pytest returns as the process exit code. The
    gate reads `identity_error` together with that pre-hook `exitstatus` to
    tell "identity failed in a clean session" from "the session itself
    failed"; it never compares the pre-hook status with the process exit code.
    The failure is also written to stderr, because
    pytest's own summary is built from the original status and still reads
    "N passed" -- without that line a reader of the session output has nothing
    that says why it exited 1.
    """
    out = os.environ.get("MODAL_IMPORT_PROBE_OUT")
    if not out:
        return
    # Annotated because `identity_error` below is added as `str` or `None`,
    # neither of which the census's own values are.
    payload: dict[str, object] = {
        **module_census(), "passed": sorted(_PASSED),
        "exitstatus": int(exitstatus)
    }
    try:
        assert_module_identity(payload)
    except AssertionError as error:
        payload["identity_error"] = str(error)
        sys.stderr.write(f"\n{__name__}: module identity failed: {error}\n")
        if not exitstatus:
            session.exitstatus = 1
    else:
        payload["identity_error"] = None
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
