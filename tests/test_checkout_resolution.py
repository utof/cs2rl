"""The checkout tripwire in tests/conftest.py: a session imports THIS checkout's cs2rl or stops.

Each case is a child pytest session that loads the real conftest as a plugin
(`-p tests.conftest`, as tests/test_one_module_object_per_file.py does) over one
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
    assert f"PYTHONPATH={REPO_ROOT / 'src'}" in child.stderr, (
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
