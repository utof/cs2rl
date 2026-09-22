"""Every pytest session gets its own temp directory (gh#219).

tests/conftest.py moves pytest's temp ROOT to ~/.pytest_tmp for free space. An
earlier revision pinned basetemp itself there instead, and pytest rm_rf's an
explicit basetemp at session start, so two sessions running at once deleted
each other's tmp_path trees mid-run. The failures landed in whichever unrelated
tests happened to be running at the time (gh#219; possibly gh#195).

Every check here runs in a CHILD pytest session, so how the outer session chose
its own basetemp cannot matter. Each child loads the real tests/conftest.py as a
plugin (`-p tests.conftest` with cwd at the repo root; tests/ is a package) and
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
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

REPO = Path(__file__).resolve().parent.parent
CONFTEST = REPO / "tests" / "conftest.py"

# Generous next to a child's ~1 s runtime: the bound exists so a wedged child
# fails with its output instead of hanging the whole suite.
_CHILD_TIMEOUT_S = 120
# Below _CHILD_TIMEOUT_S, so a peer that never arrives surfaces as the waiting
# child's own assertion, with a message, rather than as a kill.
_BARRIER_TIMEOUT_S = 60
# Stripped from every child's environment. The outer session's conftest may have
# put PYTEST_DEBUG_TEMPROOT into os.environ, and inheriting it would hide the
# child's own default; a PYTEST_ADDOPTS carrying --basetemp would put every child on the
# explicit-basetemp path this file exists to keep them off.
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
           temproot: Path | None) -> subprocess.Popen[str]:
    """Start one child session on the child test, WITHOUT --basetemp.

    HOME always points inside tmp_path, so a conftest that falls back to
    ~/.pytest_tmp writes there and never into the real home. temproot=None
    leaves PYTEST_DEBUG_TEMPROOT unset, which is how the conftest's own default
    is exercised. The child waits at the barrier until every id in `peers` has
    written its marker; a solo child lists only itself.
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
    return subprocess.Popen(argv,
                            cwd=REPO,
                            env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)


def _collect(tmp_path: Path, child_id: str, proc: subprocess.Popen[str]) -> _Child:
    """Wait, bounded, for one child and gather what it left behind.

    The report is None when the child died before writing it (collection error,
    crashed conftest). On timeout the child is killed and this raises with
    whatever it printed, so a wedged child names itself instead of hanging.
    """
    try:
        out, err = proc.communicate(timeout=_CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired as expired:
        proc.kill()
        out, err = proc.communicate()
        raise AssertionError(f"child {child_id} did not finish within {_CHILD_TIMEOUT_S}s\n"
                             f"{out[-3000:]}\n--- stderr ---\n{err[-2000:]}") from expired
    report_path = tmp_path / f"report-{child_id}.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    tail = (f"=== child {child_id}: exit {proc.returncode} ===\n{out[-3000:]}\n"
            f"--- stderr ---\n{err[-2000:]}")
    return _Child(proc.returncode, report, tail)


def _reap(procs: Iterable[subprocess.Popen[str]]) -> None:
    """Kill any child still running, so a failed or timed-out wait never leaks one."""
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


def _checked_basetemp(child: _Child, tails: str) -> Path:
    """The basetemp a finished child resolved, once it is known to be meaningful.

    Checks the child ran the REAL tests/conftest.py and that it passed. A child
    that skipped the conftest passes the concurrency and explicit-root checks
    whatever the conftest says -- measured by dropping `-p tests.conftest`. `tails`
    carries every sibling's output, because in a concurrent run the child that
    fails is the victim, not the one whose startup wiped its tree.
    """
    assert child.report is not None, f"a child never wrote its report\n{tails}"
    loaded = child.report["conftest"]
    assert loaded is not None and Path(loaded).resolve() == CONFTEST, (
        f"a child loaded conftest {loaded!r}, not {CONFTEST}, so its result says nothing "
        f"about the conftest under test\n{tails}")
    assert child.returncode == 0, f"a child session failed\n{tails}"
    return Path(child.report["basetemp"])


def _run_solo(tmp_path: Path, temproot: Path | None) -> tuple[Path, str]:
    """Run one child to completion and return its checked basetemp and output tail."""
    _write_child_test(tmp_path)
    proc = _spawn(tmp_path, "solo", ["solo"], temproot)
    try:
        child = _collect(tmp_path, "solo", proc)
    finally:
        _reap([proc])
    return _checked_basetemp(child, child.tail), child.tail


def test_concurrent_sessions_do_not_delete_each_others_tmp_path(tmp_path):
    """Two sessions started together each keep their own tmp_path (gh#219).

    The positive control. Both children run without --basetemp, write a marker
    into their tmp_path, and wait at a file barrier until both markers exist
    before checking their own. With basetemp pinned to ~/.pytest_tmp, the later
    session's startup rm_rf deletes the earlier one's tree and a child fails
    here (measured against the pre-fix conftest). The basetemp comparison covers the interleaving in which both markers
    happen to survive: two sessions sharing one basetemp is the defect even on a
    run that got lucky.
    """
    _write_child_test(tmp_path)
    root = (tmp_path / "root").resolve()
    root.mkdir()
    ids = ["a", "b"]
    procs = [_spawn(tmp_path, child_id, ids, root) for child_id in ids]
    try:
        children = [
            _collect(tmp_path, child_id, proc) for child_id, proc in zip(ids, procs, strict=True)
        ]
    finally:
        _reap(procs)
    tails = "\n".join(child.tail for child in children)

    basetemps = [_checked_basetemp(child, tails) for child in children]
    assert basetemps[0] != basetemps[1], (
        f"both sessions resolved the same basetemp {basetemps[0]}\n{tails}")
    for basetemp in basetemps:
        assert basetemp.is_relative_to(root), (
            f"basetemp {basetemp} ignores PYTEST_DEBUG_TEMPROOT={root}\n{tails}")


def test_default_temp_root_is_home_pytest_tmp(tmp_path):
    """With PYTEST_DEBUG_TEMPROOT unset, basetemp is a numbered dir under ~/.pytest_tmp.

    This is the free-space half of the conftest: /tmp sits on the small root
    partition, and test_run_experiment.py's full runs need >= 5 GB free. The
    basetemp must be pytest's own `pytest-of-<user>/pytest-<N>` below that root,
    never ~/.pytest_tmp itself, which is the shared, rm_rf'd directory of gh#219.
    """
    home_tmp = (tmp_path / "home" / ".pytest_tmp").resolve()
    basetemp, tail = _run_solo(tmp_path, temproot=None)
    assert basetemp != home_tmp, (
        f"basetemp is {home_tmp} itself, so every session shares and wipes it\n{tail}")
    assert (basetemp.parent.parent == home_tmp and basetemp.parent.name.startswith("pytest-of-")
            and re.fullmatch(r"pytest-\d+", basetemp.name)), (
                f"basetemp {basetemp} is not {home_tmp}/pytest-of-<user>/pytest-<N>\n{tail}")


def test_explicit_temp_root_is_not_overridden(tmp_path):
    """An explicit PYTEST_DEBUG_TEMPROOT wins over the conftest's ~/.pytest_tmp default."""
    custom = (tmp_path / "custom").resolve()
    custom.mkdir()
    basetemp, tail = _run_solo(tmp_path, temproot=custom)
    assert basetemp.is_relative_to(custom), (
        f"basetemp {basetemp} is not under PYTEST_DEBUG_TEMPROOT={custom}\n{tail}")
