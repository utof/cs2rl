"""The session guard in tests/conftest.py: no repo file is loaded under two module names.

`files_under_two_module_names` is tested on synthetic module maps. They pin the
prefix rule, which no session in a worktree exercises: a worktree's `.venv` is a
symlink to main's, outside the worktree root, so there `<root>/.venv/bin/pytest`
never resolves under the root at all.

The hook is tested in child pytest sessions that load the real conftest as a
plugin (`-p tests.conftest`, as tests/test_pytest_tmp_isolation.py does) over a
planted test, in a normal session with the imports in the test body and under
--collect-only with the imports at module scope. The plant imports
`src/paths.py`, a light module, under one or two names.
"""
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT, files_under_two_module_names

# The hook's report header, as tests/conftest.py writes it.
_REPORT_TITLE = "repo files loaded under two module names"
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
            "paths": root / "src" / "paths.py",
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
    real = root / "src" / "paths.py"
    real.parent.mkdir(parents=True)
    real.write_text("")
    link = root / "linked_paths.py"
    link.symlink_to(real)
    modules = {
        "paths": _module("paths", real),
        "src.paths": _module("src.paths", real),
        "linked_paths": _module("linked_paths", link),
        "__main__": _module("__main__", venv / "bin" / "pytest"),
        "__mp_main__": _module("__mp_main__", venv / "bin" / "pytest"),
        "builtin_twin": _module("builtin_twin", None),
    }
    assert files_under_two_module_names(modules, root, (venv, base)) == {
        real.resolve(): ["linked_paths", "paths", "src.paths"]
    }


def _plant(spellings: tuple[str, ...], *, at_module_scope: bool) -> str:
    """A test file that imports src/paths.py under each of `spellings`."""
    lines = ["import sys", f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r})", ""]
    imports = [f"import {name}" for name in spellings]
    if at_module_scope:
        lines += [*imports, "", "", "def test_plant():", "    pass"]
    else:
        lines += ["", "def test_plant():", *(f"    {line}" for line in imports)]
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("collect_only", [False, True], ids=["session", "collect-only"])
@pytest.mark.parametrize("spellings", [("paths", "src.paths"), ("paths", )],
                         ids=["two-names", "one-name"])
def test_a_session_that_loads_a_repo_file_under_two_names_fails(tmp_path, collect_only, spellings):
    """The hook, end to end: two names fail the session and name the file; one name passes.

    `two-names` is the positive control and `one-name` the negative one. In the
    `session` rows the planted test itself passes, so exit 1 can only come from
    the guard's `session.testsfailed += 1`; the report line is asserted too, so a
    child that failed for any other reason cannot pass for the guard.
    """
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    plant = tmp_path / "test_plant.py"
    plant.write_text(_plant(spellings, at_module_scope=collect_only))
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    argv = [
        sys.executable, "-m", "pytest",
        str(plant), "-p", "tests.conftest", "-p", "no:cacheprovider", "-q"
    ]
    if collect_only:
        argv.append("--collect-only")
    child = subprocess.run(argv,
                           cwd=tmp_path,
                           env=env,
                           capture_output=True,
                           text=True,
                           timeout=_CHILD_TIMEOUT_S)
    output = (f"exit {child.returncode}\n{child.stdout[-3000:]}\n"
              f"--- stderr ---\n{child.stderr[-2000:]}")
    ran = "1 test collected" if collect_only else "1 passed"
    assert ran in child.stdout, f"the planted test did not run as planned\n{output}"
    report = f"{REPO_ROOT / 'src' / 'paths.py'}: ['paths', 'src.paths']"
    if len(spellings) == 2:
        assert child.returncode == 1, f"the session did not fail\n{output}"
        assert _REPORT_TITLE in child.stdout and report in child.stdout, (
            f"the session failed without naming the file and both names\n{output}")
    else:
        assert child.returncode == 0, f"the session failed\n{output}"
        assert _REPORT_TITLE not in child.stdout, f"the guard reported one name\n{output}"
