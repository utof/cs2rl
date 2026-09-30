"""The session guard in tests/conftest.py: no repo file is loaded under two module names.

`files_under_two_module_names` is tested on synthetic module maps. They pin the
prefix rule, which no session in a worktree exercises: a worktree's `.venv` is a
symlink to main's, outside the worktree root, so there `<root>/.venv/bin/pytest`
never resolves under the root at all.

The hook is tested in child pytest sessions that load the real conftest as a
plugin (`-p tests.conftest`, as tests/integration/test_pytest_tmp_isolation.py does) over a
planted test, in a normal session with the imports in the test body and under
--collect-only with the imports at module scope. The plant imports
`src/cs2rl/spec/paths.py`, a light module, as `cs2rl.spec.paths` and, in the two-name
case, also as bare `paths`. That second name is reachable because the child's
PYTHONPATH carries `src/cs2rl/spec` (prepended in both cases, so the two cases differ
only in the plant): the same thing a script-path launch of a package module does
to `sys.path[0]`, which is how the trap arises in practice.

Every child goes through `_session`. Two more kinds of row run the child on pytest-xdist
workers (`-n 2`), because a serial child cannot see what xdist takes away (#285): a
worker's `testsfailed` and terminal output never reach the controller, so the hook
relays each worker's facts through `workeroutput` and the controller judges their union.
  - dup: both names in one test, on one worker. The controller must still report it.
  - split: the two names in two files pinned to two workers by `xdist_group`, so NO
    process holds both. Only the union sees the duplicate. The row proves the split by
    reading each worker's id from the plant, not by parsing output.
  - negative control: one name, green.
  - sends-nothing: a plugin makes worker gw1 drop its findings, and the child must be
    red only through the controller's fail-closed "finished without reporting" report.
  - torch: the worker's thread cap must not import torch into a plant that never did.
The `-n 2` rows never skip: without pytest-xdist in the environment they fail with the
remedy (`assert_child_had_xdist`).
"""
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT, assert_child_had_xdist, files_under_two_module_names

# The hook's report header, as tests/conftest.py writes it.
_REPORT_TITLE = "repo files loaded under two module names"
# The report of a worker that never delivered its findings, as tests/conftest.py writes it.
_LOST_TITLE = "session guards could not check an xdist worker"
_XDIST = ("-n", "2")
# What the report says about the plants' shared file, when both names reached the judge.
_PATHS_REPORT = f"{REPO_ROOT / 'src' / 'cs2rl' / 'spec' / 'paths.py'}: ['cs2rl.spec.paths', 'paths']"
# Generous next to a child's ~2 s runtime; the bound exists so a wedged child
# fails with its output instead of hanging the suite.
_CHILD_TIMEOUT_S = 120


def _module(name: str, file: Path | None) -> types.ModuleType:
    """A module object named `name` whose `__file__` is `file` (absent when None)."""
    module = types.ModuleType(name)
    if file is not None:
        module.__file__ = str(file)
    return module


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    """(repo root, its in-root venv, the base interpreter's prefix) under tmp_path."""
    return tmp_path / "repo", tmp_path / "repo" / ".venv", tmp_path / "python"


@pytest.mark.parametrize("case", [
    "venv-bin-main-and-mp-main",
    "venv-site-packages-alias",
    "base-prefix-alias",
    "outside-the-root",
    "no-file",
    "one-name",
])
def test_the_guard_is_silent_on_files_it_must_not_report(tmp_path, case):
    """NEGATIVE CONTROL: none of these is a repo file under two names.

    The first two sit under an ignored prefix inside the root, which is main's
    layout: `.venv/bin/pytest` is `__main__` and `__mp_main__` when the suite
    runs as `uv run pytest`, and the venv's site-packages holds aliases like
    `os.path`/`posixpath`. Dropping the prefix rule turns both red.
    """
    root, venv, base = _layout(tmp_path)
    files = {
        "venv-bin-main-and-mp-main": {
            "__main__": venv / "bin" / "pytest",
            "__mp_main__": venv / "bin" / "pytest",
        },
        "venv-site-packages-alias": {
            "pkg.mod": venv / "lib" / "python3.12" / "site-packages" / "pkg" / "mod.py",
            "pkg.alias": venv / "lib" / "python3.12" / "site-packages" / "pkg" / "mod.py",
        },
        "base-prefix-alias": {
            "os.path": base / "lib" / "python3.12" / "posixpath.py",
            "posixpath": base / "lib" / "python3.12" / "posixpath.py",
        },
        "outside-the-root": {
            "other": tmp_path / "elsewhere" / "other.py",
            "elsewhere.other": tmp_path / "elsewhere" / "other.py",
        },
        "no-file": {
            "builtin_twin": None,
            "builtin_twin_alias": None,
        },
        "one-name": {
            "cs2rl.paths": root / "src" / "cs2rl" / "paths.py",
        },
    }[case]
    modules = {name: _module(name, file) for name, file in files.items()}
    assert files_under_two_module_names(modules, root, (venv, base)) == {}


def test_the_guard_reports_a_repo_file_under_two_names_with_every_name(tmp_path):
    """POSITIVE CONTROL: a repo file under two names is reported, with both names.

    A third name reaches the same file through a symlink, so the report also
    shows that files are compared by resolved path. The silent cases' modules
    ride along and must stay out of the report.
    """
    root, venv, base = _layout(tmp_path)
    real = root / "src" / "cs2rl" / "paths.py"
    real.parent.mkdir(parents=True)
    real.write_text("")
    link = root / "linked_paths.py"
    link.symlink_to(real)
    modules = {
        "cs2rl.paths": _module("cs2rl.paths", real),
        "paths": _module("paths", real),
        "linked_paths": _module("linked_paths", link),
        "__main__": _module("__main__", venv / "bin" / "pytest"),
        "__mp_main__": _module("__mp_main__", venv / "bin" / "pytest"),
        "builtin_twin": _module("builtin_twin", None),
    }
    assert files_under_two_module_names(modules, root, (venv, base)) == {
        real.resolve(): ["cs2rl.paths", "linked_paths", "paths"]
    }


def _plant(spellings: tuple[str, ...], *, at_module_scope: bool) -> str:
    """A test file that imports src/cs2rl/spec/paths.py under each of `spellings`.

    No `sys.path` edit: the child's PYTHONPATH makes both spellings importable.
    """
    imports = [f"import {name}" for name in spellings]
    if at_module_scope:
        lines = [*imports, "", "", "def test_plant():", "    pass"]
    else:
        lines = ["def test_plant():", *(f"    {line}" for line in imports)]
    return "\n".join(lines) + "\n"


def _session(tmp_path: Path, plant_files: dict[str, str], *extra_args:
             str) -> tuple[subprocess.CompletedProcess, str]:
    """Run a child session over the planted files (name -> source) with the real conftest loaded.

    `extra_args` go after the plant paths, e.g. `-n 2` or `--collect-only`. Returns the
    finished process and a printable dump of its exit code and output tails.
    """
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    for name, source in plant_files.items():
        (tmp_path / name).write_text(source)
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    # PREPENDED, never replacing the inherited value: that value is what put this
    # checkout's src/ first, and without it the child would import another
    # checkout's cs2rl. REPO_ROOT makes `tests.conftest` importable; src/cs2rl/spec
    # makes bare `paths` importable; tmp_path makes a plugin module planted there
    # importable by the child and by its xdist workers.
    entries = [
        str(tmp_path),
        str(REPO_ROOT),
        str(REPO_ROOT / "src" / "cs2rl" / "spec"),
        os.environ.get("PYTHONPATH")
    ]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, entries))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    argv = [sys.executable, "-m", "pytest", *(str(tmp_path / name) for name in plant_files)]
    argv += ["-p", "tests.conftest", "-p", "no:cacheprovider", "-q", *extra_args]
    child = subprocess.run(argv,
                           cwd=tmp_path,
                           env=env,
                           capture_output=True,
                           text=True,
                           timeout=_CHILD_TIMEOUT_S)
    assert_child_had_xdist(child)
    output = (f"exit {child.returncode}\n{child.stdout[-3000:]}\n"
              f"--- stderr ---\n{child.stderr[-2000:]}")
    return child, output


@pytest.mark.parametrize("collect_only", [False, True], ids=["session", "collect-only"])
@pytest.mark.parametrize("spellings", [("cs2rl.spec.paths", "paths"), ("cs2rl.spec.paths", )],
                         ids=["two-names", "one-name"])
def test_a_session_that_loads_a_repo_file_under_two_names_fails(tmp_path, collect_only, spellings):
    """The hook, end to end: two names fail the session and name the file; one name passes.

    `two-names` is the positive control and `one-name` the negative one. In the
    `session` rows the planted test itself passes, so exit 1 can only come from
    the guard's `session.testsfailed += 1`; the report line is asserted too, so a
    child that failed for any other reason cannot pass for the guard.
    """
    plant = {"test_plant.py": _plant(spellings, at_module_scope=collect_only)}
    child, output = _session(tmp_path, plant, *(["--collect-only"] if collect_only else []))
    ran = "1 test collected" if collect_only else "1 passed"
    assert ran in child.stdout, f"the planted test did not run as planned\n{output}"
    if len(spellings) == 2:
        assert child.returncode == 1, f"the session did not fail\n{output}"
        assert _REPORT_TITLE in child.stdout and _PATHS_REPORT in child.stdout, (
            f"the session failed without naming the file and both names\n{output}")
    else:
        assert child.returncode == 0, f"the session failed\n{output}"
        assert _REPORT_TITLE not in child.stdout, f"the guard reported one name\n{output}"


def test_two_names_in_one_test_on_an_xdist_worker_fail_the_session_and_are_named(tmp_path):
    """dup at `-n 2`: the worker holds both names, and only the relay lets the controller see it."""
    child, output = _session(
        tmp_path, {"test_plant.py": _plant(
            ("cs2rl.spec.paths", "paths"), at_module_scope=False)}, *_XDIST)
    assert "1 passed" in child.stdout, f"the planted test did not run as planned\n{output}"
    assert child.returncode == 1, f"the session did not fail\n{output}"
    assert _REPORT_TITLE in child.stdout and _PATHS_REPORT in child.stdout, (
        f"the session failed without naming the file and both names\n{output}")
    assert _LOST_TITLE not in child.stdout, f"a worker failed to report\n{output}"


def test_two_names_in_two_workers_fail_the_session_and_are_named(tmp_path):
    """split at `-n 2 --dist loadgroup`: no process holds both names, so only the union does.

    The two plants sit in two xdist groups, so they run on two workers, and each imports ONE
    spelling: a per-worker verdict (the natural way to add xdist support) finds nothing. Each
    plant records its worker's id, which proves the split without parsing output; a serial
    child would raise on `workerinput`, so this row cannot pass without workers.
    """
    plants, records = {}, {}
    for group, spelling in (("a", "cs2rl.spec.paths"), ("b", "paths")):
        records[group] = tmp_path / f"{group}.worker"
        plants[f"test_plant_{group}.py"] = (
            "from pathlib import Path\n\nimport pytest\n\n\n"
            f"@pytest.mark.xdist_group({group!r})\n"
            f"def test_plant_{group}(request):\n"
            f"    import {spelling}\n"
            f"    Path({str(records[group])!r}).write_text(request.config.workerinput['workerid'])\n"
        )
    child, output = _session(tmp_path, plants, *_XDIST, "--dist", "loadgroup")
    assert "2 passed" in child.stdout, f"the planted tests did not run as planned\n{output}"
    workers = {group: record.read_text() for group, record in records.items()}
    assert workers["a"] != workers[
        "b"], f"the plants shared a worker, so nothing was split\n{workers}\n{output}"
    assert child.returncode == 1, f"the session did not fail\n{output}"
    assert _REPORT_TITLE in child.stdout and _PATHS_REPORT in child.stdout, (
        f"the session failed without naming the file and both names\n{output}")
    assert _LOST_TITLE not in child.stdout, f"a worker failed to report\n{output}"


def test_one_name_on_xdist_workers_passes(tmp_path):
    """NEGATIVE CONTROL at `-n 2`: one name is green and no worker is reported lost."""
    child, output = _session(
        tmp_path, {"test_plant.py": _plant(("cs2rl.spec.paths", ), at_module_scope=False)}, *_XDIST)
    assert "1 passed" in child.stdout, f"the planted test did not run as planned\n{output}"
    assert child.returncode == 0, f"the session failed\n{output}"
    assert _REPORT_TITLE not in child.stdout, f"the guard reported one name\n{output}"
    assert _LOST_TITLE not in child.stdout, f"a worker failed to report\n{output}"


# A plain (non-wrapper) sessionfinish that makes worker gw1 drop its findings. xdist's own
# sessionfinish is a hookwrapper that sends `workerfinished` (with `workeroutput`) after its
# yield, so this pop lands first. The plugin also loads in the child's controller, which has
# no `workerinput`, hence the gw1 test.
_WITHHOLD_PLUGIN = """\
from tests.conftest import _WORKER_GUARD_KEY


def pytest_sessionfinish(session):
    workerid = getattr(session.config, "workerinput", {}).get("workerid")
    if workerid == "gw1":
        getattr(session.config, "workeroutput", {}).pop(_WORKER_GUARD_KEY, None)
"""


def test_a_worker_that_sends_no_findings_fails_the_session_and_is_named(tmp_path):
    """SENDS-NOTHING at `-n 2`: the guard fails closed, naming the worker that went quiet.

    The plant is the one-name negative control, which is green on its own, so the child is
    red only through the controller's "finished without reporting" report. The exit code is
    exactly 1 (a test failure, not an internal error), and there is one report, for gw1 alone.
    """
    (tmp_path / "withhold_findings.py").write_text(_WITHHOLD_PLUGIN)
    child, output = _session(
        tmp_path, {"test_plant.py": _plant(
            ("cs2rl.spec.paths", ), at_module_scope=False)}, *_XDIST, "-p", "withhold_findings")
    assert "1 passed" in child.stdout, f"the planted test did not run as planned\n{output}"
    assert child.returncode == 1, f"the session did not fail with exit 1\n{output}"
    assert "Traceback" not in child.stdout + child.stderr, f"the child crashed\n{output}"
    assert _LOST_TITLE in child.stdout, f"the lost worker was not reported\n{output}"
    lost = [line for line in child.stdout.splitlines() if "without reporting" in line]
    assert len(lost) == 1 and "gw1" in lost[0] and "gw0" not in lost[0], (
        f"expected one report, for gw1 alone\n{output}")
    assert "worker gw1 finished without" in lost[0], (
        f"the report does not say the worker finished quietly\n{output}")
    assert _REPORT_TITLE not in child.stdout, f"the guard reported a duplicate\n{output}"


def test_an_xdist_worker_does_not_import_torch_for_its_thread_cap(tmp_path):
    """The cap only touches a torch that a test module already loaded (conftest, pitfall 2)."""
    plant = 'import sys\n\n\ndef test_plant():\n    assert "torch" not in sys.modules\n'
    child, output = _session(tmp_path, {"test_plant.py": plant}, *_XDIST)
    assert "1 passed" in child.stdout, f"a worker imported torch\n{output}"
    assert child.returncode == 0, f"the session failed\n{output}"
    assert _LOST_TITLE not in child.stdout, f"a worker was reported lost\n{output}"
