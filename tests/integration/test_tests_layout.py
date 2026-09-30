"""tests/ mirrors src/cs2rl/ (#207 part 2): where a test file may live, checked on the real tree.

WHAT. `layout_problems(tests_dir, package_dir)` returns one problem per:
  (a) directory under tests/ that holds a .py file and has no twin: tests/<a>/<b> needs the
      package src/cs2rl/<a>/<b>/__init__.py, unless its top directory is in NO_TWIN;
  (b) NO_TWIN entry that is not a directory of tests/ (a stale entry);
  (c) test file directly in tests/ that imports no flat module of cs2rl (a src/cs2rl/<m>.py
      other than __init__.py). The root mirrors src/cs2rl/ itself, so only the flat modules'
      tests sit there, and a new test that belongs in a package's directory is refused;
  (d) .py file under tests/, other than tests/conftest.py and tests/_helpers/, that builds a
      path from its own `__file__` with `.parent` or `.parents`: a file-relative repo root is
      right only at the depth it was written for, and a move re-arms it. Import `REPO_ROOT`
      from tests.conftest (the conftest never moves). A read of the file itself
      (`Path(__file__).read_text()`) is not a root and passes.
WHY each is a failure and not a convention:
  (a), (c): "where does my new test go" had no answer, so tests went into the nearest
      1,000-line file (#184 F6). A layout nothing checks drifts back to flat.
  (b): NO_TWIN is a hand-kept allow set. Deleting an entry whose directory exists is red by
      (a); an entry for a directory that does not exist is red by (b). Neither can go quiet.
  (d): #207 measured 56 file-relative roots. After a move most fail loudly, but the ones that
      only feed a child's cwd or a skip stay green in the wrong directory (#207 part 2 M8(a)).
PITFALLS.
  * A disk walk, not `git ls-files`: pytest collects an untracked file just the same.
  * The walk's scope is pytest's, read from its config (`python_files`, `norecursedirs`) and
    passed in, never copied here: it prunes `__pycache__` and every directory a
    `norecursedirs` pattern matches, as pytest does, and nothing else. A directory whose name
    is not an identifier (`tests/env-c/`) is walked, because pytest collects from it. pytest's
    other skips (`--ignore`, `collect_ignore`, a virtualenv) are not applied: `--ignore` only
    splits a run, and tests/ has neither of the others.
  * (c) reads imports by AST in every spelling (`import cs2rl.policy`, `from cs2rl import
    policy`, `from cs2rl.policy import X`), module-level or inside a function.
  * The flat modules are read from src/cs2rl/ on disk, never listed here, so a new flat module
    needs no edit.
"""
import ast
import fnmatch
import os
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

# Top-level directories of tests/ with no src/cs2rl twin, each with the reason.
NO_TWIN = {
    "_helpers":
    "shared test code; never collected (tests/integration/test_path_constants_exist.py pins it)",
    "fixtures": "data files the tests read",
    "integration":
    "repo-wide guards, dependency pins and tooling tests: their subject is the checkout",
    "modal":
    "the Modal runner's tests: it lives in scripts/modal_runner/ (#274), outside src/cs2rl",
}


def _matches(pattern: str, path: Path) -> bool:
    """pytest's glob match for `python_files` and `norecursedirs`: a pattern with no `/` matches
    the name, one with a `/` the whole path. A port of `_pytest.pathlib.fnmatch_ex` (POSIX
    branch), which is private."""
    if "/" not in pattern:
        return fnmatch.fnmatch(path.name, pattern)
    if path.is_absolute() and not os.path.isabs(pattern):
        pattern = f"*/{pattern}"
    return fnmatch.fnmatch(str(path), pattern)


def _flat_modules(package_dir: Path) -> set[str]:
    """`<package>.<m>` for every <package>/<m>.py other than __init__.py (`cs2rl.policy`, ...)."""
    return {
        f"{package_dir.name}.{p.stem}"
        for p in package_dir.glob("*.py") if p.stem != "__init__"
    }


def _imported_modules(tree: ast.AST) -> set[str]:
    """Every dotted module an import statement names, with `from a import b` also as `a.b`."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module)
            found |= {f"{node.module}.{alias.name}" for alias in node.names}
    return found


def _file_relative_roots(tree: ast.AST) -> list[int]:
    """Line numbers where `__file__` feeds a `.parent` / `.parents` chain."""
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    lines = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Name) and node.id == "__file__"):
            continue
        up = parents.get(id(node))
        while isinstance(up, (ast.Call, ast.Attribute, ast.Subscript)):
            if isinstance(up, ast.Attribute) and up.attr in ("parent", "parents"):
                lines.append(node.lineno)
                break
            up = parents.get(id(up))
    return lines


def layout_problems(tests_dir: Path, package_dir: Path, python_files: list[str],
                    norecursedirs: list[str]) -> list[str]:
    """(a)-(d) above, for the tree at `tests_dir` against the package at `package_dir`, with
    pytest's `python_files` and `norecursedirs` (the `pytest_scope` fixture reads both)."""
    problems = []
    flat = _flat_modules(package_dir)
    for entry in sorted(NO_TWIN):
        if not (tests_dir / entry).is_dir():
            problems.append(
                f"(b) NO_TWIN names tests/{entry}/, which does not exist: drop the entry")
    for directory, subdirs, files in os.walk(tests_dir):
        here = Path(directory)
        subdirs[:] = sorted(
            d for d in subdirs
            if d != "__pycache__" and not any(_matches(p, here / d) for p in norecursedirs))
        rel = here.relative_to(tests_dir)
        top = rel.parts[0] if rel.parts else ""
        if rel.parts and top not in NO_TWIN and any(f.endswith(".py") for f in files):
            if not (package_dir.joinpath(*rel.parts) / "__init__.py").is_file():
                problems.append(f"(a) tests/{rel.as_posix()}/ has no twin package "
                                f"src/cs2rl/{rel.as_posix()}/: move its files to the directory "
                                "of the package they test, or add the directory to NO_TWIN "
                                "with a reason")
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = here / name
            shown = path.relative_to(tests_dir.parent).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), shown)
            is_test = any(_matches(p, path) for p in python_files)
            if not rel.parts and is_test and not (_imported_modules(tree) & flat):
                problems.append(f"(c) {shown} is at the root of tests/ but imports no flat module "
                                f"of cs2rl ({sorted(flat)}): move it to the directory of the "
                                "package it tests")
            if top != "_helpers" and not (not rel.parts and name == "conftest.py"):
                for line in _file_relative_roots(tree):
                    problems.append(f"(d) {shown}:{line} builds a path from its own __file__: "
                                    "use `from tests.conftest import REPO_ROOT`")
    return problems


@pytest.fixture
def pytest_scope(pytestconfig) -> dict[str, list[str]]:
    """The walk's scope, read from this session's pytest config."""
    return {
        "python_files": pytestconfig.getini("python_files"),
        "norecursedirs": pytestconfig.getini("norecursedirs"),
    }


def test_tests_mirror_the_package(pytest_scope):
    problems = layout_problems(REPO_ROOT / "tests", REPO_ROOT / "src" / "cs2rl", **pytest_scope)
    assert not problems, "tests/ does not mirror src/cs2rl/:\n  " + "\n  ".join(problems)


def _tree(tmp_path: Path, files: dict[str, str]) -> tuple[Path, Path]:
    # The package is `pkg`, not `cs2rl`: the name pin (test_name_strings_resolve.py) reads every
    # string here, and a planted `cs2rl.<missing>` would fail it.
    for rel, text in {
            "src/pkg/__init__.py": "",
            "src/pkg/policy.py": "",
            "src/pkg/env/__init__.py": "",
            "tests/conftest.py": "",
            "tests/_helpers/h.py": "",
            "tests/fixtures/x.json": "",
            "tests/integration/test_i.py": "",
            "tests/modal/test_m.py": "",
            **files
    }.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    return tmp_path / "tests", tmp_path / "src" / "pkg"


def test_a_clean_tree_has_no_problem(tmp_path, pytest_scope):
    """Negative control: every allowed placement at once, including the allowed __file__ reads
    and two directories pytest prunes (`.scratch` by the `.*` pattern, and `__pycache__`)."""
    tests_dir, package_dir = _tree(
        tmp_path, {
            "tests/test_policy_x.py": "from pkg import policy\n",
            "tests/test_policy_y.py": "def test():\n    from pkg.policy import P\n",
            "tests/test_policy_z.py": "import pkg.policy\n",
            "tests/env/test_e.py": "from pathlib import Path\nT = Path(__file__).read_text()\n",
            "tests/_helpers/paths.py": "from pathlib import Path\nR = Path(__file__).parents[2]\n",
            "tests/conftest.py":
            "from pathlib import Path\nR = Path(__file__).resolve().parents[1]\n",
            "tests/.scratch/test_s.py": "",
            "tests/env/__pycache__/test_e.cpython-312.pyc": "",
        })
    assert layout_problems(tests_dir, package_dir, **pytest_scope) == []


@pytest.mark.parametrize("plant, rule", [
    ({
        "tests/viz/test_v.py": ""
    }, "(a) tests/viz/"),
    ({
        "tests/env/c/test_c.py": ""
    }, "(a) tests/env/c/"),
    ({
        "tests/env-c/test_c.py": ""
    }, "(a) tests/env-c/"),
    ({
        "tests/test_new.py": "import pkg.env\n"
    }, "(c) tests/test_new.py"),
    ({
        "tests/test_new.py": "from pkg.policyx import P\n"
    }, "(c) tests/test_new.py"),
    ({
        "tests/env/test_e.py": "from pathlib import Path\nR = Path(__file__).resolve().parents[2]\n"
    }, "(d) tests/env/test_e.py:2"),
    ({
        "tests/env/test_e.py": "import pathlib\nR = pathlib.Path(__file__).parent.parent\n"
    }, "(d) tests/env/test_e.py:2"),
    ({
        "tests/integration/test_i.py": "import os\nR = os.path.dirname(__file__)\n"
    }, None),
])
def test_each_rule_reports_its_plant(tmp_path, pytest_scope, plant, rule):
    """Positive controls: one plant per rule, each reported by the rule that owns it.

    The last row is a KNOWN LIMIT, pinned so it cannot be mistaken for coverage: rule (d)
    reads Path's `.parent`/`.parents`, not `os.path.dirname`, and no test file uses dirname.
    """
    tests_dir, package_dir = _tree(tmp_path, plant)
    problems = layout_problems(tests_dir, package_dir, **pytest_scope)
    if rule is None:
        assert problems == []
    else:
        assert len(problems) == 1 and problems[0].startswith(rule), problems


def test_no_twin_entries_are_watched(tmp_path, monkeypatch, pytest_scope):
    """(b) and (a) together: a stale NO_TWIN entry is red, and so is a deleted live one."""
    tests_dir, package_dir = _tree(tmp_path, {})
    monkeypatch.setitem(NO_TWIN, "gone", "a directory that does not exist")
    assert layout_problems(tests_dir, package_dir, **pytest_scope) == [
        "(b) NO_TWIN names tests/gone/, which does not exist: drop the entry"
    ]
    monkeypatch.delitem(NO_TWIN, "gone")
    monkeypatch.delitem(NO_TWIN, "integration")
    problems = layout_problems(tests_dir, package_dir, **pytest_scope)
    assert len(problems) == 1 and problems[0].startswith("(a) tests/integration/"), problems
