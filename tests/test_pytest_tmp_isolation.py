"""Every pytest session gets its own temp directory (gh#219).

tests/conftest.py moves pytest's temp ROOT to ~/.cache/cs2rl-pytest for free
space. An earlier revision pinned basetemp itself to ~/.pytest_tmp instead, and
pytest rm_rf's an explicit basetemp the first time a session asks for a temp
dir, so two sessions running at once deleted each other's tmp_path trees
mid-run. The failures landed in whichever unrelated tests happened to be
running at the time (gh#219; possibly gh#195).

Every check here runs in a CHILD pytest session, so how the outer session chose
its own basetemp cannot matter. Each child loads the real tests/conftest.py as a
plugin (`-p tests.conftest` with cwd at the repo root, which `python -m` puts on
sys.path; tests/ is a namespace package, with no __init__.py since #207) and
reports the file it loaded, so a child that ran without the conftest fails
loudly instead of passing vacuously. Every child's HOME, and any
PYTEST_DEBUG_TEMPROOT it is given, point inside the outer tmp_path, and children
write no bytecode, so nothing is written outside it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

REPO = Path(__file__).resolve().parent.parent
CONFTEST = REPO / "tests" / "conftest.py"

# Generous next to a child's ~1 s runtime, and shared by all the children of one
# test: the bound exists so a wedged child fails with its output instead of
# hanging the whole suite.
_CHILD_TIMEOUT_S = 120
# Below _CHILD_TIMEOUT_S, so a peer that never arrives surfaces as the waiting
# child's own assertion, with a message, rather than as a kill.
_BARRIER_TIMEOUT_S = 60
# Stripped from every child's environment. The outer session's conftest may have
# put PYTEST_DEBUG_TEMPROOT into os.environ, and inheriting it would hide the
# child's own default; a PYTEST_ADDOPTS carrying --basetemp would put every
# child on the explicit-basetemp path this file exists to keep them off.
_DROPPED_ENV = ("PYTEST_DEBUG_TEMPROOT", "PYTEST_ADDOPTS")

# The child test. It writes its report BEFORE asserting anything, so the outer
# test can name the basetemp even of a child that then fails. The marker is
# named and filled per child: two children sharing a basetemp can both be handed
# the same `test_...0` directory, and a shared name would let one child's marker
# pass for the other's.
_CHILD_TEST = '''\
import json
import os
import sys
import time
from pathlib import Path


def test_tmp_path_survives_concurrent_sessions(tmp_path, tmp_path_factory):
    me = os.environ["ISOLATION_CHILD_ID"]
    barrier = Path(os.environ["ISOLATION_BARRIER_DIR"])
    conftest = sys.modules.get("tests.conftest")
    Path(os.environ["ISOLATION_REPORT"]).write_text(json.dumps({
        "basetemp": str(tmp_path_factory.getbasetemp()),
        "conftest": getattr(conftest, "__file__", None),
    }))
    marker = tmp_path / f"marker-{me}"
    marker.write_text(me)
    (barrier / f"{me}.ready").touch()

    peers = os.environ["ISOLATION_PEERS"].split(",")
    deadline = time.monotonic() + float(os.environ["ISOLATION_BARRIER_TIMEOUT_S"])
    while not all((barrier / f"{peer}.ready").exists() for peer in peers):
        assert time.monotonic() < deadline, f"barrier: not all of {peers} arrived"
        time.sleep(0.05)

    assert marker.is_file() and marker.read_text() == me, (
        f"child {me}: its tmp_path marker was deleted or replaced while it waited "
        f"for {peers}; tmp_path={tmp_path} exists={tmp_path.exists()}")
'''


class _Child(NamedTuple):
    """What one finished child left behind: exit code, report, output tail."""
    returncode: int
    report: dict[str, Any] | None
    tail: str


def _write_child_test(tmp_path: Path) -> None:
    """Write the child test file beside a bare pytest.ini, plus the barrier dir.

    The pytest.ini pins the child's rootdir and config to its own directory.
    Without it pytest walks up from the test file looking for one, and an ini
    or pyproject.toml in any ancestor of the outer basetemp would leak its
    addopts into the child.
    """
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    (child_dir / "pytest.ini").write_text("[pytest]\n")
    (child_dir / "test_child.py").write_text(_CHILD_TEST)
    (tmp_path / "barrier").mkdir()


def _spawn(tmp_path: Path, child_id: str, peers: list[str],
           temproot: Path | None) -> subprocess.Popen[bytes]:
    """Start one child session on the child test, WITHOUT --basetemp.

    HOME always points inside tmp_path, so the conftest's $HOME-based default
    root (and an old conftest's ~/.pytest_tmp) lands there, never in the real
    home. temproot=None leaves PYTEST_DEBUG_TEMPROOT unset, which is how the
    conftest's own default is exercised. The child waits at the barrier until
    every id in `peers` has written its marker.

    stdout and stderr go to files under tmp_path, not to pipes. A pipe that is
    not being drained blocks a noisy child before it reaches the barrier, and
    its peer then fails with a misleading barrier timeout.
    """
    env = {k: v for k, v in os.environ.items() if k not in _DROPPED_ENV}
    env["HOME"] = str(tmp_path / "home")
    # cwd is the repo, so without this the child would cache tests/conftest.py's
    # bytecode into the repo's tests/__pycache__, outside tmp_path.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if temproot is not None:
        env["PYTEST_DEBUG_TEMPROOT"] = str(temproot)
    env["ISOLATION_CHILD_ID"] = child_id
    env["ISOLATION_PEERS"] = ",".join(peers)
    env["ISOLATION_BARRIER_DIR"] = str(tmp_path / "barrier")
    env["ISOLATION_BARRIER_TIMEOUT_S"] = str(_BARRIER_TIMEOUT_S)
    env["ISOLATION_REPORT"] = str(tmp_path / f"report-{child_id}.json")
    argv = [
        sys.executable, "-m", "pytest",
        str(tmp_path / "child" / "test_child.py"), "-p", "tests.conftest", "-q", "-p",
        "no:cacheprovider"
    ]
    # The child keeps its own copies of these descriptors; closing ours on the
    # way out of the `with` does not cut it off.
    out_path, err_path = tmp_path / f"{child_id}.out", tmp_path / f"{child_id}.err"
    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        return subprocess.Popen(argv, cwd=REPO, env=env, stdout=out, stderr=err)


def _reap(procs: Iterable[subprocess.Popen[bytes]]) -> None:
    """Kill any child still running, so a failed or timed-out wait never leaks one."""
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def _gather(tmp_path: Path, child_id: str, returncode: int) -> _Child:
    """Read back what one finished child left under tmp_path.

    The report is None when the child died before writing it (collection error,
    crashed conftest, a temp-dir error at fixture setup).
    """
    report_path = tmp_path / f"report-{child_id}.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    out = (tmp_path / f"{child_id}.out").read_text(errors="replace")
    err = (tmp_path / f"{child_id}.err").read_text(errors="replace")
    tail = (f"=== child {child_id}: exit {returncode} ===\n{out[-3000:]}\n"
            f"--- stderr ---\n{err[-2000:]}")
    return _Child(returncode, report, tail)


def _run_children(tmp_path: Path, temproots: dict[str, Path | None]) -> tuple[list[_Child], str]:
    """Start one child per id together, wait for all, and return them with every tail.

    `temproots` maps each child id to the PYTEST_DEBUG_TEMPROOT it gets (None:
    unset). Every child waits at the barrier for all the others. Spawning
    happens inside the `try`, so a failed second spawn still reaps the first.
    All children share one _CHILD_TIMEOUT_S deadline; on timeout every child
    still running is killed, and the failure carries every child's output,
    because the child that hung is not necessarily the one whose output
    explains it.
    """
    _write_child_test(tmp_path)
    ids = list(temproots)
    procs: dict[str, subprocess.Popen[bytes]] = {}
    timed_out = False
    try:
        for child_id in ids:
            procs[child_id] = _spawn(tmp_path, child_id, ids, temproots[child_id])
        deadline = time.monotonic() + _CHILD_TIMEOUT_S
        for proc in procs.values():
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        _reap(procs.values())
    children = [_gather(tmp_path, child_id, proc.returncode) for child_id, proc in procs.items()]
    tails = "\n".join(child.tail for child in children)
    assert not timed_out, (
        f"a child did not finish within {_CHILD_TIMEOUT_S}s and was killed\n{tails}")
    return children, tails


def _checked_basetemp(child: _Child, tails: str) -> Path:
    """The basetemp a finished child resolved, once it is known to be meaningful.

    Checks the child ran the REAL tests/conftest.py and that it passed. A child
    that skipped the conftest passes the concurrency and explicit-root checks
    whatever the conftest says -- measured by dropping `-p tests.conftest`.
    `tails` carries every sibling's output, because in a concurrent run the
    child that fails is the victim, not the one whose startup wiped its tree.
    """
    assert child.report is not None, f"a child never wrote its report\n{tails}"
    loaded = child.report["conftest"]
    assert loaded is not None and Path(loaded).resolve() == CONFTEST, (
        f"a child loaded conftest {loaded!r}, not {CONFTEST}, so its result says nothing "
        f"about the conftest under test\n{tails}")
    assert child.returncode == 0, f"a child session failed\n{tails}"
    return Path(child.report["basetemp"])


def test_concurrent_sessions_do_not_delete_each_others_tmp_path(tmp_path):
    """Two sessions started together each keep their own tmp_path (gh#219).

    The positive control. Both children run without --basetemp, write a marker
    into their tmp_path, and wait at a file barrier until both markers exist
    before checking their own. With basetemp pinned to ~/.pytest_tmp, both
    children share it, and each one's first temp-dir request rm_rf's and
    recreates it. Against the pre-fix conftest that was measured in two shapes:
    the later child's rm_rf deleted the earlier one's marker, or the two
    rm_rf+mkdir sequences interleaved, one child errored at fixture setup with
    FileExistsError, and the other then failed at the barrier timeout. The
    basetemp comparison covers the interleaving in which both markers happen to
    survive: two sessions sharing one basetemp is the defect even on a run that
    got lucky.
    """
    root = (tmp_path / "root").resolve()
    root.mkdir()
    children, tails = _run_children(tmp_path, {"a": root, "b": root})

    basetemps = [_checked_basetemp(child, tails) for child in children]
    assert basetemps[0] != basetemps[1], (
        f"both sessions resolved the same basetemp {basetemps[0]}\n{tails}")
    for basetemp in basetemps:
        assert basetemp.is_relative_to(root), (
            f"basetemp {basetemp} ignores PYTEST_DEBUG_TEMPROOT={root}\n{tails}")


def test_default_temp_root_is_under_home_cache(tmp_path):
    """With PYTEST_DEBUG_TEMPROOT unset, basetemp is numbered under ~/.cache/cs2rl-pytest.

    This is the free-space half of the conftest: /tmp sits on the small root
    partition, which is regularly tight. The basetemp must be pytest's own
    `pytest-of-<user>/pytest-<N>` below that root, never one fixed directory
    that every session shares (gh#219). And nothing may land in ~/.pytest_tmp,
    which pre-gh#219 conftests on other checkouts still rm_rf.
    """
    home = tmp_path / "home"
    root = (home / ".cache" / "cs2rl-pytest").resolve()
    children, tails = _run_children(tmp_path, {"solo": None})
    basetemp = _checked_basetemp(children[0], tails)
    assert (basetemp.parent.parent == root and basetemp.parent.name.startswith("pytest-of-")
            and re.fullmatch(r"pytest-\d+", basetemp.name)), (
                f"basetemp {basetemp} is not {root}/pytest-of-<user>/pytest-<N>\n{tails}")
    assert not (home / ".pytest_tmp").exists(), (
        f"the conftest created {home / '.pytest_tmp'}, the directory old conftests wipe\n{tails}")


def test_explicit_temp_root_is_not_overridden(tmp_path):
    """An explicit PYTEST_DEBUG_TEMPROOT wins, and the default root is not even created."""
    custom = (tmp_path / "custom").resolve()
    custom.mkdir()
    children, tails = _run_children(tmp_path, {"solo": custom})
    basetemp = _checked_basetemp(children[0], tails)
    assert basetemp.is_relative_to(custom), (
        f"basetemp {basetemp} is not under PYTEST_DEBUG_TEMPROOT={custom}\n{tails}")
    default_root = tmp_path / "home" / ".cache" / "cs2rl-pytest"
    assert not default_root.exists(), (
        f"the conftest created its default root {default_root} although "
        f"PYTEST_DEBUG_TEMPROOT was set\n{tails}")
