"""gh#254: the vis-cache ProcessPoolExecutor's workers die with their parent.

`NavGraph.build_vis_matrix` forks `os.cpu_count()` workers (~900 MB each on
dust2). Before this fix any parent death that skipped `__exit__` (pytest-timeout's
`os._exit`, an outer `timeout`, SIGKILL, OOM) reparented every worker to PID 1 and
they kept computing; 24 orphans from two killed parents took the 16 GB dev box
to 15.3 GB on 2026-09-25.

Knock-out (`test_pool_workers_die_with_parent`): run the REAL `build_vis_matrix`
in a child process on an 8-area synthetic map (never dust2, never a cold cache),
with `awpy.visibility.VisibilityChecker` replaced by a stub whose `is_visible`
sleeps, so every worker is alive and blocked in a task when we SIGKILL the child.
One second later no worker may be alive. Deleting the `_die_with_parent` call from
`_vis_worker_init` turns this red, because the path under test is the production
initializer, not a copy.

Positive control (`test_pool_workers_survive_without_the_guard`): the same child
with `nav._die_with_parent` replaced by a no-op before the fork. All workers must
still be alive one second after the kill, which proves the assertion above is
live and that the guard, not the kill itself, is what ends them.

PITFALLS:
- The child is started as its own session leader and the whole group is
  SIGKILLed in `finally` (same shape as tests/test_arena_duel.py::_run_group), so
  the control's deliberately orphaned workers never outlive the test. As a second
  fence the stub arms `signal.alarm(30)` in every worker: even if pytest itself
  is SIGKILLed mid-test, nothing survives 30 s.
- `os.cpu_count` is pinned to 3 inside the child so the test forks 3 workers, not
  12; the guard is per-worker, so the count is irrelevant to what is tested.
- Worker pids are collected BEFORE the kill (children of the child); afterwards
  they are reparented, so `ps --ppid` would find nothing either way. Liveness is
  read from /proc/<pid>/stat, and a zombie (`Z`) counts as dead: the kernel has
  already killed it and PID 1 reaps it within milliseconds.
"""
import os
import pathlib
import signal
import subprocess
import sys
import textwrap
import time

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
N_WORKERS = 3
N_AREAS = 8

# Runs the production build_vis_matrix on a synthetic NavGraph. Everything that
# would touch awpy's real data is replaced INSIDE the child before the fork, so
# the workers inherit the stubs: the checker class (sleeps in is_visible), the
# .tri directory (an empty de_dust2.tri, only its existence is checked) and the
# worker count. `nav._die_with_parent` is the guard under test; the control
# replaces it with a no-op via {noguard}.
_CHILD = """
import os, pathlib, signal, sys, time
sys.path.insert(0, {src!r})
import awpy.data as _d
import awpy.visibility as _v
_d.TRIS_DIR = pathlib.Path({tri_dir!r})
class _StubChecker:
    def __init__(self, path=None):
        signal.alarm(30)                     # second fence: no worker outlives 30 s
    def is_visible(self, a, b):
        time.sleep(60)
        return True
_v.VisibilityChecker = _StubChecker
os.cpu_count = lambda: {n_workers}
import nav
if {noguard!r}:
    nav._die_with_parent = lambda *a, **k: None
class _Centroid:
    def __init__(self, x):
        self.x, self.y, self.z = float(x), 0.0, 0.0
class _Area:
    def __init__(self, i):
        self.centroid = _Centroid(i)
g = nav.NavGraph.__new__(nav.NavGraph)
g.areas = {{i: _Area(i) for i in range({n_areas})}}
g.area_ids = list(range({n_areas}))
g.N = {n_areas}
g._nav_path = {nav_path!r}
g._cache_path = {cache_path!r}
g.build_vis_matrix()
"""


def _children_of(pid: int) -> list[int]:
    out = subprocess.run(["ps", "-o", "pid=", "--ppid", str(pid)], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split()]


def _alive(pid: int) -> bool:
    try:
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    state = stat.rsplit(")", 1)[1].split()[0]          # field 3, after the "(comm)"
    return state not in ("Z", "X")


def _pdeathsig():                      # child dies if pytest dies
    import ctypes
    ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)


def _kill_child_and_count_survivors(tmp_path, *, noguard: bool) -> tuple[list[int], list[int]]:
    """Start the child, wait until its N_WORKERS workers exist and have been
    blocked in a task for ~2 s, SIGKILL the child ONLY, and return
    (workers, still-alive-after-1 s). The whole group is SIGKILLed in `finally`,
    which then waits for the reparented workers to vanish (PID 1 reaps them, not
    us) so the test can never hand a live process to the next one."""
    tri_dir = tmp_path / "tris"
    tri_dir.mkdir()
    (tri_dir / "de_dust2.tri").write_bytes(b"")
    nav_path = tmp_path / "fake.json"
    nav_path.write_text("{}")
    script = _CHILD.format(src=str(SRC),
                           tri_dir=str(tri_dir),
                           n_workers=N_WORKERS,
                           noguard=noguard,
                           n_areas=N_AREAS,
                           nav_path=str(nav_path),
                           cache_path=str(tmp_path / "never_written.npy"))
    p = subprocess.Popen([sys.executable, "-c", textwrap.dedent(script)],
                         cwd=REPO_ROOT,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         text=True,
                         start_new_session=True,
                         preexec_fn=_pdeathsig)
    try:
        deadline = time.monotonic() + 30
        workers: list[int] = []
        while time.monotonic() < deadline:
            workers = _children_of(p.pid)
            if len(workers) >= N_WORKERS:
                break
            if p.poll() is not None:
                out, err = p.communicate()
                raise AssertionError(f"child exited early rc={p.returncode}\n{err[-3000:]}")
            time.sleep(0.1)
        assert len(workers) == N_WORKERS, f"expected {N_WORKERS} workers, saw {workers}"
        time.sleep(2.0)                # every worker is now inside is_visible
        assert all(_alive(w) for w in workers), "a worker died before the kill"
        os.kill(p.pid, signal.SIGKILL)
        p.wait(timeout=10)
        time.sleep(1.0)
        survivors = [w for w in workers if _alive(w)]
        return workers, survivors
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for pipe in (p.stdout, p.stderr):
            if pipe is not None:
                pipe.close()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and any(_alive(w) for w in workers):
            time.sleep(0.05)


@pytest.mark.skipif(sys.platform != "linux", reason="prctl + /proc")
@pytest.mark.timeout(90)
def test_pool_workers_die_with_parent(tmp_path):
    workers, survivors = _kill_child_and_count_survivors(tmp_path, noguard=False)
    assert survivors == [], f"orphaned vis-cache workers {survivors} of {workers} survived the parent"


@pytest.mark.skipif(sys.platform != "linux", reason="prctl + /proc")
@pytest.mark.timeout(90)
def test_pool_workers_survive_without_the_guard(tmp_path):
    workers, survivors = _kill_child_and_count_survivors(tmp_path, noguard=True)
    assert sorted(survivors) == sorted(workers), (
        f"positive control: without the guard every worker must survive; {survivors} of {workers} did"
    )


@pytest.mark.skipif(sys.platform != "linux", reason="prctl + /proc")
@pytest.mark.timeout(90)
def test_guard_does_not_kill_a_healthy_build(tmp_path):
    """Regression fence for the prctl's per-THREAD semantics: a build whose parent
    stays alive must complete and write its cache. The stub returns at once here
    (sleep 0) so the full pool round trip runs in well under a second."""
    tri_dir = tmp_path / "tris"
    tri_dir.mkdir()
    (tri_dir / "de_dust2.tri").write_bytes(b"")
    nav_path = tmp_path / "fake.json"
    nav_path.write_text("{}")
    cache = tmp_path / "vis.npy"
    script = _CHILD.format(src=str(SRC),
                           tri_dir=str(tri_dir),
                           n_workers=N_WORKERS,
                           noguard=False,
                           n_areas=N_AREAS,
                           nav_path=str(nav_path),
                           cache_path=str(cache)).replace("time.sleep(60)", "pass")
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(script)],
                       cwd=REPO_ROOT,
                       capture_output=True,
                       text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr[-3000:]
    import numpy as np
    vis = np.load(cache)
    assert vis.shape == (N_AREAS, N_AREAS) and vis.all()               # stub says everything is visible


@pytest.mark.skipif(sys.platform != "linux", reason="prctl")
@pytest.mark.timeout(30)
def test_guard_kills_a_worker_whose_parent_is_already_gone():
    """The prctl arms a FUTURE parent exit only. A worker whose parent died between
    the fork and the prctl call must not survive on the strength of the prctl
    alone: the re-check of os.getppid() against the pid captured in the parent
    has to end it. Simulated with a parent_pid that is not our parent."""
    r = subprocess.run([
        sys.executable, "-c", f"import sys; sys.path.insert(0, {str(SRC)!r}); import nav; "
        "nav._die_with_parent(parent_pid=2**22 - 1); "
        "import time; time.sleep(5); print('survived')"
    ],
                       capture_output=True,
                       text=True,
                       timeout=20)
    assert r.returncode == -signal.SIGKILL and "survived" not in r.stdout, (r.returncode, r.stdout,
                                                                            r.stderr[-1000:])
