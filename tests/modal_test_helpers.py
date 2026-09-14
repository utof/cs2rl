"""Helpers used by BOTH halves of the modal test seam.

NOT a test module: no `test_` prefix, so pytest does not collect it. It exists
because the split is not a clean bisection. Measured, these 7 module-level names
are reached by tests on both sides of the seam (`_aware`: 22 runner tests and 57
client tests). Copying them into both files instead would let two definitions of
the same fixture drift apart with every check in tests/test_modal_packaging.py
green -- which is the failure mode this seam exists to stop.

Membership is computed, not judged: classify_seam() assigns a helper here iff
the set of tests that transitively reach it spans both halves.

WHAT EARNS A NAME A PLACE HERE, and nothing else does: being reached from BOTH
sides of the seam. A one-sided helper belongs in its own half -- however generic
it looks, and however well its name would read in this file. The rule is written
down because a module named for what it IS rather than for what it OWNS becomes
a junk drawer: every future helper looks a little bit shared, and the file
accretes until it is a second monolith. Spec §10 criterion 12 bans a module
named `helpers`; its instrument is §5.1's eight W3 submodules, so this file is
formally out of its scope -- the rule is honoured here anyway, because the
criterion's reason applies and its instrument is what does not reach.

You do not have to apply this by hand, and you should not: classify_seam()
computes it. A one-sided helper hand-written into this file gains a name in
`computed` that the manifest lacks, and the agreement test fires.

The `# --` section headers below travelled with the helper they sit above and
name a runner-half test cycle that is no longer here. Task 5 renames them.
"""
import io
import json
import subprocess
import sys
import threading
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # scripts/ is a namespace package; tests import scripts.modal_runner_lib
    # the same way the later CLIs will. Do not rely on the editable install.
    sys.path.insert(0, str(ROOT))

import scripts.modal_runner_lib as mrl                 # noqa: E402, I001

# ── Task 3 cycle A: clean HEAD / Git object validation ─────────────────────


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_source_repo(tmp_path: Path) -> Path:
    """Tiny real git repo so HEAD/diff checks exercise the actual git CLI."""
    repo = tmp_path / "src-repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@test")
    _git(repo, "config", "user.name", "t")
    (repo / "readme.txt").write_text("hello\n")
    _git(repo, "add", "readme.txt")
    _git(repo, "commit", "-qm", "init")
    return repo


# ── Task 4 cycle B: heartbeat + derived stale ──────────────────────────────


def _aware(hour=12, minute=0, second=0):
    from datetime import datetime

    return datetime(2026, 8, 13, hour, minute, second, tzinfo=UTC)


# ── Task 4 cycle C: completion evidence + transport config + download ───────


def _write_metrics(path: Path, steps: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for epoch, step in enumerate(steps, start=1):
        # Representative live row: pin the live key `step` (src/train.py).
        rows.append(
            json.dumps({
                "run_id": "ok-id",
                "step": step,
                "epoch": epoch,
                "sps": 1.0
            }) + "\n")
    path.write_text("".join(rows))


def _noop_heartbeat(**_kwargs):
    return SimpleNamespace(stop_and_join=lambda: None)


# ── Task 6 cycle C: resume validation + cheap config dump/hash ──────────────


def _write_dumped_config(run_root: Path) -> dict[str, object]:
    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config = {"env": "cs2-dust2", "seed": 2, "data_dir": str(ckpt_dir), "timesteps": 30000000}
    (ckpt_dir / "config.json").write_text(json.dumps(config))
    return config


# ── Task 7 cycle A: process-group start and nontruncating tee ──────────────


class FakeChild:
    """Popen stand-in. BytesIO streams drain like closed pipes."""

    def __init__(self, *, stdout=b"", stderr=b"", returncode=0, pid=4242, hold=False):
        self.pid = pid
        self.returncode = returncode
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.signals: list[int] = []
        self.wait_timeouts: list[float | None] = []
        self._done = threading.Event()
        if not hold:
            self._done.set()

    def poll(self):
        return self.returncode if self._done.is_set() else None

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        # Grace waits must not burn wall-clock time in tests. A held child
        # times out immediately; a released child returns at once.
        effective = timeout
        if timeout is not None and timeout >= mrl.TERM_GRACE_SECONDS:
            effective = 0
        if not self._done.wait(timeout=effective):
            raise subprocess.TimeoutExpired(["fake"], timeout)
        return self.returncode

    def send_signal(self, sig):
        self.signals.append(sig)

    def release(self, returncode=None):
        if returncode is not None:
            self.returncode = returncode
        self._done.set()
