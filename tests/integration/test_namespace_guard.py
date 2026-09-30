"""The namespace guard in tests/conftest.py: `tests` and `scripts` are this checkout's own.

WHY (#207). tests/ has no __init__.py since pytest moved to importlib mode, so
`tests`, like `scripts`, is a PEP 420 namespace package: its __path__ is every
`tests/` directory without an __init__.py on sys.path, recomputed whenever
sys.path changes (a `tests/` WITH one, anywhere on sys.path, replaces it
outright). Another tree on sys.path (another checkout's root) then serves any
`tests.X` that this checkout lacks, silently. The guard has two halves: (c) stops a session whose
sys.path, at the start, holds either name outside this checkout's root, and (d)
fails a session that ends holding a `tests`/`scripts` module from anywhere but
the place its dotted name implies in this checkout.

1. Child pytest sessions, as in tests/test_checkout_resolution.py: each loads
the real conftest as a plugin (`-p tests.conftest`) over one planted test, with
a `pytest.ini` in its tmp dir so the repo's own config stays out of it.
  - (c) positive control: a tmp dir holding `tests/metrics_census_other.py` (no
    __init__.py) first on PYTHONPATH: the session stops before collection and
    names the entry and `tests`.
  - (d) positive control: the planted test appends that dir to sys.path, calls
    importlib.invalidate_caches() and imports the foreign module. The test
    passes, so exit 1 can only come from the guard, which must name `tests` and
    the module with their foreign paths.
  - negative control: no foreign dir, and a plant that imports real `tests.*`
    and `scripts.*` modules, passes.
2. The two functions on synthetic inputs, for what a child session cannot build
without writing into the checkout: the NESTED layouts, a foreign checkout UNDER
the root (<root>/.worktrees/x, where this repo's worktrees live) and a foreign
tree inside our own tests/ (<root>/tests/_sim/x/tests). A committed test never
writes into the checkout, so the root here is a tmp dir.

3. The same two end-to-end rows again on pytest-xdist workers (`-n 2`, #285). A serial child
cannot see what xdist takes away: a worker's failures and output never reach the
controller, and the test modules are imported in the workers. The conftest hook relays each
worker's facts through `workeroutput` and the controller judges them; these rows pin (d)
through that relay, and the negative control pins that nothing else turns red. They never
skip: without pytest-xdist in the environment they fail with the remedy.

PITFALL: every child PREPENDS to the inherited PYTHONPATH, never replaces it.
The inherited value is what put this checkout's src/ first, and replacing it
trips the checkout tripwire's (a) instead of the case under test.
"""
import os
import subprocess
import sys
import types
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import pytest

from tests.conftest import (
    REPO_ROOT,
    assert_child_had_xdist,
    namespace_entry_problems,
    namespace_modules_outside_their_package,
)

# Generous next to a child's ~2 s runtime; the bound exists so a wedged child
# fails with its output instead of hanging the suite.
_CHILD_TIMEOUT_S = 120
# The prefix pytest_sessionstart's UsageError carries.
_TRIPWIRE = "checkout tripwire (tests/conftest.py)"
# (d)'s report header, as tests/conftest.py writes it.
_REPORT_TITLE = "tests/scripts modules loaded from another tree"
_FOREIGN_MODULE = "metrics_census_other"
# The report of a worker that never delivered its findings, as tests/conftest.py writes it.
_LOST_TITLE = "session guards could not check an xdist worker"
_XDIST = ("-n", "2")


def _session(
    tmp_path: Path, plant: str, *first: Path,
    extra_args: tuple[str, ...] = ()) -> tuple[subprocess.CompletedProcess, str]:
    """Run a child session over `plant` with the `first` paths ahead of PYTHONPATH.

    `extra_args` go after the plant path, e.g. `("-n", "2")`.
    """
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_plant.py").write_text(plant)
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    # REPO_ROOT makes `tests.conftest` (and `scripts.*`) importable.
    entries = [*map(str, first), str(REPO_ROOT), os.environ.get("PYTHONPATH")]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, entries))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    child = subprocess.run([
        sys.executable, "-m", "pytest",
        str(tmp_path / "test_plant.py"), "-p", "tests.conftest", "-p", "no:cacheprovider", "-q",
        *extra_args
    ],
                           cwd=tmp_path,
                           env=env,
                           capture_output=True,
                           text=True,
                           timeout=_CHILD_TIMEOUT_S)
    assert_child_had_xdist(child)
    output = (f"exit {child.returncode}\n{child.stdout[-3000:]}\n"
              f"--- stderr ---\n{child.stderr[-3000:]}")
    return child, output


def _foreign_tree(tmp_path: Path) -> Path:
    """Another tree's root: a `tests/` holding a module this checkout lacks, no __init__.py."""
    other = tmp_path / "other"
    (other / "tests").mkdir(parents=True)
    (other / "tests" / f"{_FOREIGN_MODULE}.py").write_text("WHO = 'other'\n")
    return other


def test_a_foreign_tests_dir_on_the_path_stops_the_session_and_is_named(tmp_path):
    """(c), end to end: the entry and the name are named, and nothing is collected."""
    other = _foreign_tree(tmp_path)
    child, output = _session(tmp_path, "def test_plant():\n    pass\n", other)
    assert child.returncode == pytest.ExitCode.USAGE_ERROR, (
        f"the guard did not stop the session\n{output}")
    assert _TRIPWIRE in child.stderr, f"the session stopped, but not by the tripwire\n{output}"
    assert f"(c) sys.path entry {str(other)!r} holds `tests` ({other / 'tests'})" in child.stderr, (
        f"the message does not name the entry and `tests`\n{output}")
    assert "passed" not in child.stdout, f"the planted test ran: collection was not stopped\n{output}"


def _check_foreign_import_fails_the_session(tmp_path: Path, extra_args: tuple[str, ...]) -> None:
    """(d) end to end, serially or on workers: the planted import passes, the guard fails it."""
    other = _foreign_tree(tmp_path)
    plant = ("import importlib\nimport sys\n\n\ndef test_plant():\n"
             f"    sys.path.append({str(other)!r})\n"
             "    importlib.invalidate_caches()\n"
             f"    importlib.import_module('tests.{_FOREIGN_MODULE}')\n")
    child, output = _session(tmp_path, plant, extra_args=extra_args)
    assert "1 passed" in child.stdout, f"the planted import did not run as planned\n{output}"
    assert child.returncode == 1, f"the session did not fail\n{output}"
    assert _REPORT_TITLE in child.stdout, f"the session failed, but not by the guard\n{output}"
    for line in (f"tests: [{str(other / 'tests')!r}]",
                 f"tests.{_FOREIGN_MODULE}: [{str(other / 'tests' / f'{_FOREIGN_MODULE}.py')!r}]"):
        assert line in child.stdout, f"the report does not say {line!r}\n{output}"
    assert _LOST_TITLE not in child.stdout, f"a worker failed to report\n{output}"


def _check_own_modules_pass(tmp_path: Path, extra_args: tuple[str, ...]) -> None:
    """Neither half fires on this checkout's own `tests.*` and `scripts.*` modules."""
    plant = ("def test_plant():\n"
             "    import scripts.modal_runner.training\n"
             "    import tests.modal_test_helpers\n")
    child, output = _session(tmp_path, plant, extra_args=extra_args)
    assert child.returncode == 0, f"the session failed\n{output}"
    assert "1 passed" in child.stdout, f"the planted test did not run\n{output}"
    assert _TRIPWIRE not in child.stderr, f"the tripwire fired\n{output}"
    assert _REPORT_TITLE not in child.stdout, f"(d) reported this checkout's modules\n{output}"
    assert _LOST_TITLE not in child.stdout, f"a worker failed to report\n{output}"


def test_a_module_imported_from_a_foreign_tests_dir_fails_the_session_and_is_named(tmp_path):
    """(d), end to end: a foreign dir that reaches sys.path mid-session, after (c) ran.

    The import succeeds, which is the hazard itself: the planted test passes, so
    exit 1 can only come from the guard's `session.testsfailed += 1`, and the
    report must name both the merged `tests` portion and the module.
    """
    _check_foreign_import_fails_the_session(tmp_path, ())


def test_negative_control_this_checkouts_own_tests_and_scripts_modules_pass(tmp_path):
    """Neither half fires on this checkout's own `tests.*` and `scripts.*` modules."""
    _check_own_modules_pass(tmp_path, ())


def test_a_foreign_module_imported_on_an_xdist_worker_fails_the_session_and_is_named(tmp_path):
    """(d) at `-n 2`: the worker sees the foreign module, the controller must report it."""
    _check_foreign_import_fails_the_session(tmp_path, _XDIST)


def test_negative_control_this_checkouts_own_modules_pass_on_xdist_workers(tmp_path):
    """The negative control at `-n 2`: own modules are green and no worker is reported lost."""
    _check_own_modules_pass(tmp_path, _XDIST)


def _checkout(root: Path) -> Path:
    """A tree whose root holds both namespace names as directories, as a checkout's does."""
    (root / "tests").mkdir(parents=True)
    (root / "scripts").mkdir()
    return root


def _heads(problems: list[str]) -> list[str]:
    """Each (c) problem up to the found path's closing parenthesis: entry, name and path."""
    return [problem[:problem.index(").") + 1] for problem in problems]


@pytest.mark.parametrize(
    "case",
    ["tests-dir", "scripts-dir", "tests-module", "regular-package", "nested-checkout", "cwd"])
def test_the_start_check_names_each_foreign_entry_and_name(tmp_path, monkeypatch, case):
    """(c) on sys.path lists: every entry but the root itself that holds either name.

    `nested-checkout` is the nested layout: <root>/.worktrees/x is under the
    root, and is still another checkout, so "under the root" must not pass.
    `tests-module` is a module file, not a directory. `regular-package` is a
    `tests/` WITH an __init__.py: not a namespace portion, but worse, since it
    replaces our `tests` outright, so the check must not exempt it. `cwd` is
    the entry `''`, which the message must resolve to the directory it means.
    """
    root = _checkout(tmp_path / "root")
    entry = tmp_path / "other"
    # (name, what the entry holds under that name), in the order (c) checks them.
    if case == "nested-checkout":
        entry = _checkout(root / ".worktrees" / "x")
        expected = [("tests", "tests"), ("scripts", "scripts")]
    elif case == "tests-module":
        entry.mkdir()
        (entry / "tests.py").write_text("")
        expected = [("tests", "tests.py")]
    else:
        name = "scripts" if case == "scripts-dir" else "tests"
        (entry / name).mkdir(parents=True)
        if case == "regular-package":
            (entry / name / "__init__.py").write_text("")
        expected = [(name, name)]
    path_entry, shown = str(entry), repr(str(entry))
    if case == "cwd":
        monkeypatch.chdir(entry)
        # The cwd as the OS reports it, which is what '' means.
        entry = Path.cwd()
        path_entry, shown = "", f"'' ({entry})"
    problems = namespace_entry_problems(root, [str(root), path_entry])
    assert _heads(problems) == [
        f"(c) sys.path entry {shown} holds `{name}` ({entry / held})" for name, held in expected
    ], problems


def test_the_start_check_is_silent_on_the_root_and_on_entries_holding_neither_name(
        tmp_path, monkeypatch):
    """(c)'s negative control: the root under any spelling, near-miss names, a missing entry.

    `alias` is the root through a symlink, the shape of this drive's two mount
    names: only os.path.samefile knows it is the root. `plain` holds names that
    are not importable as `tests`/`scripts` (a `tests_extra/` directory, a
    suffixless `tests` file, `scripts.txt`).
    """
    root = _checkout(tmp_path / "root")
    alias = tmp_path / "alias"
    alias.symlink_to(root)
    plain = tmp_path / "plain"
    (plain / "tests_extra").mkdir(parents=True)
    (plain / "tests").write_text("")
    (plain / "scripts.txt").write_text("")
    monkeypatch.chdir(root)
    entries = [str(root), str(alias), f"{root}/.", "", str(plain), str(tmp_path / "missing")]
    assert namespace_entry_problems(root, entries) == []


def _module(name: str,
            file: Path | None = None,
            path: list[Path] | None = None) -> types.ModuleType:
    """A module object named `name` with `__file__` and `__path__` set only when given."""
    module = types.ModuleType(name)
    if file is not None:
        module.__file__ = str(file)
    if path is not None:
        module.__path__ = [str(p) for p in path]
    return module


@pytest.mark.parametrize("case", [
    "nested-checkout", "nested-in-own-package", "nested-scripts", "the-other-package",
    "misnamed-in-own-package", "another-tree"
])
def test_the_end_check_reports_a_module_outside_its_own_package_directory(tmp_path, case):
    """(d) on module maps: every place but the one the module's dotted name implies is named.

    `nested-checkout` is THE nested-layout control, the shape of the #207
    confirmation's plant: a worktree under the root (<root>/.worktrees/x) whose
    `tests/` merged into ours and served a module. Every path in it is under the
    root, so a check against the root itself passes it silently. It covers both
    places, the portion in `tests.__path__` and the module's `__file__`.
    `nested-in-own-package` is the same one level down, the #207 verifier's
    HOLE: a foreign tree at <root>/tests/_sim/x/tests is under our own tests/,
    so a check against the package directory passes it silently.
    `the-other-package` is a `tests` module under the root's scripts/: its OWN
    package directory decides, not either one. `misnamed-in-own-package` pins
    that the name decides the exact place: a module file or package `__init__`
    in our own tests/ under another name is not home.
    """
    root = _checkout(tmp_path / "root")
    nested = root / ".worktrees" / "x"
    if case == "nested-in-own-package":
        inner = root / "tests" / "_sim" / "x" / "tests"
        file = inner / "nested_evil.py"
        modules = {
            "tests": _module("tests", path=[root / "tests", inner]),
            "tests.nested_evil": _module("tests.nested_evil", file=file),
        }
        expected = {"tests": [str(inner)], "tests.nested_evil": [str(file)]}
    elif case == "misnamed-in-own-package":
        other, helper = root / "tests" / "beta.py", root / "tests" / "pkg" / "helper.py"
        modules = {
            "tests.alpha": _module("tests.alpha", file=other),
            "tests.pkg": _module("tests.pkg", file=helper, path=[root / "tests" / "pkg"]),
        }
        expected = {"tests.alpha": [str(other)], "tests.pkg": [str(helper)]}
    elif case == "nested-checkout":
        file = nested / "tests" / "metrics_census_nested.py"
        modules = {
            "tests": _module("tests", path=[root / "tests", nested / "tests"]),
            "tests.metrics_census_nested": _module("tests.metrics_census_nested", file=file),
        }
        expected = {"tests": [str(nested / "tests")], "tests.metrics_census_nested": [str(file)]}
    elif case == "nested-scripts":
        package = nested / "scripts" / "modal_runner"
        modules = {
            "scripts.modal_runner":
            _module("scripts.modal_runner", file=package / "__init__.py", path=[package])
        }
        expected = {"scripts.modal_runner": [str(package / "__init__.py"), str(package)]}
    elif case == "the-other-package":
        modules = {"tests.stray": _module("tests.stray", file=root / "scripts" / "stray.py")}
        expected = {"tests.stray": [str(root / "scripts" / "stray.py")]}
    else:
        file = tmp_path / "other" / "tests" / "elsewhere.py"
        modules = {"tests.elsewhere": _module("tests.elsewhere", file=file)}
        expected = {"tests.elsewhere": [str(file)]}
    assert namespace_modules_outside_their_package(modules, root) == expected


def test_the_end_check_is_silent_on_this_checkouts_own_modules(tmp_path):
    """(d)'s negative control: own modules, the root through a symlink, and other names.

    The root is passed through `alias` (the shape of the drive's second mount
    name) while most places use the real path, and one place goes through the
    alias: both sides are resolved, so neither is a false alarm. A package's
    `__init__` file and an extension module (`tests._native`, with the ABI-tagged
    suffix) are home too. `testsuite` only STARTS with `tests`; the top-level
    name decides. A module with no `__file__` and no `__path__` has no place to
    judge. `tests.linked` sits beside a symlinked `tests/linked/`: a module file
    is home by its own parent, never by the directory's target.
    """
    root = _checkout(tmp_path / "root")
    (root / "tests" / "_helpers").mkdir()
    (root / "scripts" / "modal_runner").mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root)
    runner = root / "scripts" / "modal_runner"
    (tmp_path / "elsewhere" / "linked").mkdir(parents=True)
    (root / "tests" / "linked").symlink_to(tmp_path / "elsewhere" / "linked")
    own = [
        _module("tests", path=[root / "tests"]),
        _module("tests.conftest", file=root / "tests" / "conftest.py"),
        _module("tests._helpers", path=[root / "tests" / "_helpers"]),
        _module("tests._native", file=root / "tests" / f"_native{EXTENSION_SUFFIXES[0]}"),
        _module("tests.no_place"),
        _module("tests.linked", file=root / "tests" / "linked.py"),
        _module("scripts", path=[alias / "scripts"]),
        _module("scripts.modal_runner", file=runner / "__init__.py", path=[runner]),
        _module("testsuite", file=tmp_path / "other" / "testsuite.py"),
        _module("numpy", file=tmp_path / "site" / "numpy" / "__init__.py"),
    ]
    modules = {module.__name__: module for module in own}
    assert namespace_modules_outside_their_package(modules, alias) == {}
