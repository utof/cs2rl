"""Unit tests for scripts/run_experiment.py — preconditions and failure paths.

These tests use a fake git repo in tmp_path to exercise precondition logic
without touching the real cs2rl repo.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
RUN_EXP = REPO / "scripts" / "run_experiment.py"


def _init_fake_repo(tmp_path: Path) -> Path:
    """Create a minimal fake repo tmp_path is the working dir; returns repo root."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" /
     "train.py").write_text("OBS_DIM = 104\nACTION_HEAD_SIZES = (9, 16, 2, 2, 3, 2, 2)\n")
    (tmp_path / "src" / "c_env").mkdir()
    (tmp_path / "src" / "c_env" / "cs2_rewards.h").write_text(
        "#define INACTION_PENALTY -0.0005f\n"
        "static const float terminal_win_bonus = 1.0f;\n")
    (tmp_path / "src" / "c_env" / "cs2_env.c").write_text("float plant_progress_reward = 0.05f;\n")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "checkpoints").mkdir()
    (tmp_path / "outputs" / "experiments").mkdir()
    (tmp_path / "outputs" / "experiments" / "results.jsonl").touch()
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    return tmp_path


def _run(repo: Path, *args: str, fake_train: bool = True, **kw) -> subprocess.CompletedProcess:
    """Invoke run_experiment.py against `repo` as a fake repo.

    CS2RL_REPO_ROOT redirects all paths (REPO_ROOT, EXPERIMENTS_DIR, etc.) to
    the fake repo so the real cs2rl repo is untouched.
    CS2RL_FAKE_TRAIN short-circuits the training subprocess for precondition tests.
    """
    env = os.environ.copy()
    env["CS2RL_REPO_ROOT"] = str(repo)
    if fake_train:
        env["CS2RL_FAKE_TRAIN"] = "1"
    return subprocess.run([sys.executable, str(RUN_EXP), *args],
                          cwd=repo,
                          capture_output=True,
                          text=True,
                          timeout=30,
                          env=env,
                          **kw)


_BASE_ARGS = ("--tag", "foo", "--hypothesis", "h", "--change-type", "hp")


def test_precondition_not_on_main_fails(tmp_path):
    repo = _init_fake_repo(tmp_path)
    subprocess.run(["git", "checkout", "-b", "other", "-q"], cwd=repo, check=True)
    r = _run(repo, *_BASE_ARGS, "--changed-files", "src/train.py")
    assert r.returncode != 0
    assert "main" in (r.stdout + r.stderr).lower()


def test_precondition_dirty_tree_fails(tmp_path):
    repo = _init_fake_repo(tmp_path)
    (repo / "src" / "train.py").write_text("OBS_DIM = 999\n")          # dirty but not declared
    r = _run(repo, *_BASE_ARGS, "--changed-files", "")
    assert r.returncode != 0
    err = (r.stdout + r.stderr).lower()
    assert "clean" in err or "dirty" in err or "declared" in err


def test_precondition_changed_files_mismatch_fails(tmp_path):
    repo = _init_fake_repo(tmp_path)
    (repo / "src" / "train.py").write_text("OBS_DIM = 999\n")
    r = _run(repo, *_BASE_ARGS, "--changed-files", "src/c_env/cs2_env.c")
    assert r.returncode != 0
    err = (r.stdout + r.stderr).lower()
    assert "mismatch" in err or "declared" in err or "changed-files" in err


def test_precondition_lock_exists_fails(tmp_path):
    repo = _init_fake_repo(tmp_path)
    (repo / "outputs" / "experiments" / ".lock").write_text("1234 2026-01-01T00:00:00Z")
    r = _run(repo, *_BASE_ARGS, "--changed-files", "")
    assert r.returncode != 0
    assert "lock" in (r.stdout + r.stderr).lower()


def test_precondition_passes_clean_main(tmp_path):
    """Clean main + empty changed-files = all preconditions pass; then it should
    proceed (and fail later because CS2RL_FAKE_TRAIN makes it exit before training).
    """
    repo = _init_fake_repo(tmp_path)
    r = _run(repo, *_BASE_ARGS, "--changed-files", "")
    err = (r.stdout + r.stderr).lower()
    assert "preconditions" not in err or r.returncode == 0
