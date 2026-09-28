"""A process imports THIS checkout's cs2rl, or stops: the two checkout guards.

1. The pytest tripwire in tests/conftest.py. Each case is a child pytest session
that loads the real conftest as a plugin (`-p tests.conftest`, as
tests/test_one_module_object_per_file.py does) over one
planted passing test, with a `pytest.ini` in its tmp dir so the repo's own config
stays out of it. Only the child's PYTHONPATH differs between the cases:

  (a) a fake `cs2rl` package first on PYTHONPATH: `cs2rl` no longer resolves
      under this checkout's src/, so the session must stop and name the fix;
  (b) another checkout's src/ (a tmp dir holding pyproject.toml and
      src/stray.py) on PYTHONPATH: `stray` is an importable top-level name
      beside cs2rl, so the session must stop and name it;
  negative control: neither, and the planted test runs and passes.

PITFALL: every child PREPENDS to the inherited PYTHONPATH, never replaces it.
The inherited value is what put this checkout's src/ first (in a worktree that
borrows main's venv), so replacing it would make the negative control fail (a)
for a reason that has nothing to do with the case under test.

2. The import guard in src/cs2rl/__init__.py, for every entry point outside
pytest. Each case is a child `python -c "import cs2rl"` that inherits this
session's PYTHONPATH (so cs2rl is THIS checkout's) and differs only in its cwd:
a tmp checkout (pyproject.toml + src/cs2rl/__init__.py) must fail and name the
fix; this checkout's root, a directory in no checkout, and an installed copy of
the package (the Modal wheel path) must import silently.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

# Generous next to a child's ~2 s runtime; the bound exists so a wedged child
# fails with its output instead of hanging the suite.
_CHILD_TIMEOUT_S = 120
# The prefix pytest_sessionstart's UsageError carries.
_TRIPWIRE = "checkout tripwire (tests/conftest.py)"


def _session(tmp_path: Path, *first_on_path: Path) -> tuple[subprocess.CompletedProcess, str]:
    """Run a child session over one passing test with `first_on_path` ahead of PYTHONPATH."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    plant = tmp_path / "test_plant.py"
    plant.write_text("def test_plant():\n    pass\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    # REPO_ROOT makes `tests.conftest` importable.
    entries = [*map(str, first_on_path), str(REPO_ROOT), os.environ.get("PYTHONPATH")]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, entries))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    child = subprocess.run([
        sys.executable, "-m", "pytest",
        str(plant), "-p", "tests.conftest", "-p", "no:cacheprovider", "-q"
    ],
                           cwd=tmp_path,
                           env=env,
                           capture_output=True,
                           text=True,
                           timeout=_CHILD_TIMEOUT_S)
    output = (f"exit {child.returncode}\n{child.stdout[-3000:]}\n"
              f"--- stderr ---\n{child.stderr[-3000:]}")
    return child, output


def _assert_stopped_before_collection(child: subprocess.CompletedProcess, output: str) -> None:
    """Exit 4 (pytest's usage error) with the tripwire's message, and the planted test never ran."""
    assert child.returncode == pytest.ExitCode.USAGE_ERROR, (
        f"the tripwire did not stop the session\n{output}")
    assert _TRIPWIRE in child.stderr, f"the session stopped, but not by the tripwire\n{output}"
    assert "passed" not in child.stdout, f"the planted test ran: collection was not stopped\n{output}"


def test_a_foreign_cs2rl_first_on_the_path_stops_the_session_and_names_the_fix(tmp_path):
    """(a): `cs2rl` resolves somewhere other than this checkout's src/."""
    fake = tmp_path / "foreign"
    (fake / "cs2rl").mkdir(parents=True)
    (fake / "cs2rl" / "__init__.py").write_text("")
    child, output = _session(tmp_path, fake)
    _assert_stopped_before_collection(child, output)
    assert "(a)" in child.stderr and str(fake / "cs2rl") in child.stderr, (
        f"the message does not say where cs2rl resolved\n{output}")
    assert f"env PYTHONPATH={REPO_ROOT / 'src'} <command>" in child.stderr, (
        f"the message does not name the fix\n{output}")


def test_a_stray_name_in_a_checkouts_src_stops_the_session_and_names_it(tmp_path):
    """(b): a checkout's src/ on sys.path holds an importable name beside cs2rl."""
    other = tmp_path / "other_checkout"
    (other / "src").mkdir(parents=True)
    (other / "pyproject.toml").write_text("")
    (other / "src" / "stray.py").write_text("")
    child, output = _session(tmp_path, other / "src")
    _assert_stopped_before_collection(child, output)
    assert "(b)" in child.stderr and "'stray'" in child.stderr, (
        f"the message does not name the stray module\n{output}")
    assert "(a)" not in child.stderr, f"cs2rl should still resolve to this checkout\n{output}"


def test_negative_control_the_same_session_without_either_runs(tmp_path):
    """Neither condition: the tripwire is silent and the planted test passes."""
    child, output = _session(tmp_path)
    assert child.returncode == 0, f"the session failed\n{output}"
    assert "1 passed" in child.stdout, f"the planted test did not run\n{output}"
    assert _TRIPWIRE not in child.stderr, f"the tripwire fired\n{output}"


def test_b_names_the_pufferlib_origin_of_a_resources_entry(tmp_path):
    """(b)'s message says where a `resources` directory in src/ comes from (verifier V2)."""
    from tests.conftest import checkout_resolution_problems
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "src" / "resources").mkdir(parents=True)
    problems = checkout_resolution_problems(REPO_ROOT / "src", [str(tmp_path / "src")])
    assert any("['resources']" in p for p in problems), problems
    assert any("import pufferlib" in p and "Removing it is safe" in p for p in problems), problems


def _import_cs2rl(cwd: Path, *first_on_path: Path) -> subprocess.CompletedProcess:
    """`python -c "import cs2rl"` in `cwd`, with `first_on_path` PREPENDED to PYTHONPATH."""
    entries = [*map(str, first_on_path), os.environ.get("PYTHONPATH")]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, entries)))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, "-c", "import cs2rl"],
                          cwd=cwd,
                          env=env,
                          capture_output=True,
                          text=True,
                          timeout=_CHILD_TIMEOUT_S)


def _fake_checkout(root: Path) -> Path:
    """What the import guard calls a checkout: pyproject.toml + src/cs2rl/__init__.py."""
    (root / "src" / "cs2rl").mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    (root / "src" / "cs2rl" / "__init__.py").write_text("")
    return root.resolve()


def test_importing_cs2rl_from_inside_another_checkout_fails_and_names_the_fix(tmp_path):
    """cwd in a different checkout than the one cs2rl came from: ImportError, both named."""
    other = _fake_checkout(tmp_path / "other")
    (other / "deeper").mkdir()
    child = _import_cs2rl(other / "deeper")
    output = f"exit {child.returncode}\n--- stderr ---\n{child.stderr[-3000:]}"
    assert child.returncode != 0 and "ImportError" in child.stderr, output
    assert f"another checkout, {other}." in child.stderr, output
    assert f"env PYTHONPATH={other / 'src'} <command>" in child.stderr, output


@pytest.mark.parametrize("where", ["this checkout", "no checkout", "installed copy"])
def test_the_import_guard_is_silent_when_the_checkouts_agree(tmp_path, where):
    """Negative controls: this checkout's root, no checkout at all, and a wheel-style copy.

    `installed copy` is the Modal path: the package outside any src/, run with the
    cwd inside a checkout (there, the extracted archive).
    """
    if where == "this checkout":
        child = _import_cs2rl(REPO_ROOT)
    elif where == "no checkout":
        child = _import_cs2rl(tmp_path)
    else:
        site = tmp_path / "site" / "cs2rl"
        site.mkdir(parents=True)
        (site / "__init__.py").write_text((REPO_ROOT / "src" / "cs2rl" / "__init__.py").read_text())
        child = _import_cs2rl(_fake_checkout(tmp_path / "archive"), site.parent)
    assert child.returncode == 0, f"exit {child.returncode}\n{child.stderr[-3000:]}"
