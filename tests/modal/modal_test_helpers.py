"""Helpers reached by tests in two or more files of the modal test seam.

NOT a test module: no `test_` prefix, so pytest does not collect it. The seam
files are the per-module runner test files (tests/modal/test_modal_<module>.py, listed
by `RUNNER_TEST_FILES` in tests/modal/modal_runner_tables.py) and
tests/modal/test_modal_client.py. This file exists because tests in different seam
files share fixtures. Measured at W4's split by module (fdd5ec9), it holds 13
module-level names besides `ROOT`, and tests in two or more seam files reach
each one. The 7 from `_git` to `FakeChild` are reached from both the client
file and runner files (`_aware`: 80 tests, 58 in four runner files and 22 in
the client file); the 6 under the last section header are reached from two or
more runner files only. Copying a helper into each file that uses it instead
would let two definitions of the same fixture drift apart with every check in
tests/modal/test_modal_packaging.py green -- which is the failure mode this seam
exists to stop.

THE 57/22 WAS SHIPPED TRANSPOSED and is corrected here, because the arithmetic
that catches it is worth leaving behind. At W2's split (84622fc), which cut the
runner half from the client half, 79 tests reached `_aware`: 57 in the runner
half and 22 in the client half. The figure arrived verbatim from the task brief
as "22 runner tests and 57 client tests" and no re-derivation was run on it. It
was refutable without measuring anything: the client half then held 54
module-level test functions in total, so no count of client tests could be 57,
and the header of tests/test_modal_client.py stated that 54. A number that its
own sibling file contradicts is the cheapest kind of wrong to find and the
easiest to carry forward untouched.

WHAT EARNS A NAME A PLACE HERE, and nothing else does: being REACHED BY TESTS
IN TWO OR MORE SEAM FILES. A helper that only one file's tests reach belongs in
that file -- however generic it looks, and however well its name would read
here. The rule this replaced at W4, "reached from both halves of the
runner/client seam", is its special case with one runner file. A consumer
outside the seam files (tests/modal/test_modal_patch_bindings.py, say) does not count.
The same rule is written beside the classifier, as THE MEMBERSHIP RULE FOR THE
SHARED FILE in tests/modal/test_modal_packaging.py; the two copies must not drift. It
is written down because a module named for what it IS rather than for what it
OWNS becomes a junk drawer: every future helper looks a little bit shared, and
the file accretes until it is a second monolith. Spec §10 criterion 12 bans a
module named `helpers`; its instrument is §5.1's eight W3 submodules, so this
file is formally out of its scope -- the rule is honoured here anyway, because
the criterion's reason applies and its instrument is what does not reach.

You do not have to apply this by hand, and you should not: classify_seam()
computes it. A single-file helper hand-written into this file is classified to
its one file, not here, so the manifest and `computed` disagree and the
agreement test fires.

Each `# ──` section header but the last names the helpers defined under it;
they once named runner-half test cycles that travelled here with those helpers.
The last, "Shared since W4", holds the 6 names W4's split made shared, which do
not fit one header line, so it states the rule they meet instead.
"""
import io
import json
import subprocess
import threading
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace

from tests.conftest import REPO_ROOT

ROOT = REPO_ROOT

# PITFALL: these runner imports are load-bearing for the seam gate, not only
# for the helpers below. The reach floor (tests/modal/test_modal_packaging.py)
# resolves `mrl.X`, `request.X` and the rest only through aliases bound
# unconditionally at MODULE LEVEL, in a test's file and in the file of every
# helper it reaches, and this file's are the only route by which
# test_manifest_records_authoritative_simple_map_not_legacy_env (in
# tests/modal/test_modal_core.py) reaches `core`: through `_make_manifest`'s
# `mrl.Manifest`. Move the `mrl` import into a function or under an `if`/`try`
# and the floor goes red on that test (the other imports carry reach the same
# way for the tests whose helpers use them). The floor's remedy says to check
# this route before moving the test: moving it would misplace a core test.
import scripts.modal_runner as mrl                                     # noqa: E402, I001
from scripts.modal_runner import checkpoint, request, training         # noqa: E402, I001
from tests.modal.modal_patch_binding_campaign import binding_target    # noqa: E402, I001

# ── _git / _init_source_repo: a real tiny repo for HEAD and diff checks ────


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


# ── _aware: fixed-date timezone-aware timestamps ───────────────────────────


def _aware(hour=12, minute=0, second=0):
    from datetime import datetime

    return datetime(2026, 8, 13, hour, minute, second, tzinfo=UTC)


# ── _write_metrics / _noop_heartbeat: metrics JSONL, inert heartbeat ───────


def _write_metrics(path: Path, steps: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for epoch, step in enumerate(steps, start=1):
        # Representative live row: pin the live key `step` (src/cs2rl/train.py).
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


# ── _write_dumped_config: the checkpoints/config.json a dump produces ──────


def _write_dumped_config(run_root: Path) -> dict[str, object]:
    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config = {"env": "cs2-dust2", "seed": 2, "data_dir": str(ckpt_dir), "timesteps": 30000000}
    (ckpt_dir / "config.json").write_text(json.dumps(config))
    return config


# ── FakeChild: Popen stand-in whose BytesIO streams drain like pipes ───────


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
        if timeout is not None and timeout >= training.TERM_GRACE_SECONDS:
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


# ── Shared since W4: reached by tests in two or more runner files ──────────


def _valid_run_kwargs(**overrides):
    """Minimal valid run fields for `mrl.build_run_request`; tests override the ones they vary."""
    kwargs = {
        "run_id": "140826-b7r-seed2-shared",
        "git_sha": "a" * 40,
        "effective_map": "simple",
        "train_args": "--timesteps 30000000 --seed 2",
    }
    kwargs.update(overrides)
    return kwargs


def _live_batch_size(num_envs: int = 256) -> int:
    """Live compute_batch_dims: num_envs * 10 agents * 64 BPTT horizon."""
    return num_envs * request.AGENTS_PER_ENV * request.BPTT_HORIZON


def _make_manifest(**overrides) -> mrl.Manifest:
    requested = 30_000_000
    batch_size = _live_batch_size()
    payload = {
        "schema_version": 1,
        "run_id": "ok-id",
        "attempt_id": "attempt-a",
        "commit": "a" * 40,
        "tree": "b" * 40,
        "source_archive_sha256": "c" * 64,
        "modal_version": "1.4.3",
        "image_digest": "sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7",
        "effective_map": "simple",
        "gpu": "T4",
        "cpu_request": 8,
        "cpu_soft_limit": 8,
        "memory_request_mib": 16384,
        "memory_hard_limit_mib": 16384,
        "vec_workers": 8,
        "timeout_minutes": 120,
        "training_argv": ["--train", "--timesteps", "30000000"],
        "requested_timesteps": requested,
        "effective_timesteps": (requested // batch_size) * batch_size,
        "batch_size": batch_size,
        "seed": 2,
        "created_at": "2026-08-13T00:00:00+00:00",
        "resume_sha256": None,
        "resume_size": None,
        "resume_source_path": None,
        "runner_commit": "a" * 40,
        "config_hash": "d" * 64,
        "thread_caps": [f"{key}={value}" for key, value in sorted(mrl.THREAD_CAP_ENV.items())],
        "resumed_from_run_id": None,
    }
    payload.update(overrides)
    return mrl.Manifest(**payload)


def _minimal_completed_tree(tmp_path: Path, *, steps: list[int] | None = None):
    import torch

    run_root = tmp_path / "run"
    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "dust2_policy.pt"
    torch.save({"weight": torch.tensor([1.0])}, ckpt)
    batch_size = _live_batch_size(256)
    requested = 30_000_000
    effective = (requested // batch_size) * batch_size
    if steps is None:
        steps = [batch_size, effective]
    _write_metrics(ckpt_dir / "metrics.jsonl", steps)
    config = {
        "env": "cs2-dust2",
        "seed": 2,
        "data_dir": str(ckpt_dir),
        "timesteps": requested,
    }
    normalized = checkpoint.normalize_config_for_transport(config)
    config_hash = mrl.sha256_bytes(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode())
    (ckpt_dir / "config.json").write_text(json.dumps(config))
    manifest = _make_manifest(
        attempt_id="a1",
        requested_timesteps=requested,
        effective_timesteps=effective,
        batch_size=batch_size,
        created_at=_aware().isoformat(),
        config_hash=config_hash,
        training_argv=["--train"],
    )
    return run_root, manifest, effective, ckpt


def _no_torch(monkeypatch, *, prebuilt: str) -> None:
    """Simulate the container runner: no in-process torch, prebuilt venv at `prebuilt`."""

    def raise_import_error():
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(*binding_target("fallback-loader"), raise_import_error)
    monkeypatch.setattr(*binding_target("fallback-python"), prebuilt)
