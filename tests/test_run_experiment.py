"""Unit tests for scripts/run_experiment.py — preconditions and failure paths.

These tests use a fake git repo in tmp_path to exercise precondition logic
without touching the real cs2rl repo.
"""

import json
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
     "train.py").write_text("OBS_DIM = 105\nACTION_HEAD_SIZES = (9, 16, 2, 2, 3, 2, 2, 2)\n")
    (tmp_path / "src" / "c_env").mkdir()
    (tmp_path / "src" / "c_env" / "cs2_rewards.h").write_text(
        "#define INACTION_PENALTY -0.0005f\n"
        "static const float terminal_win_bonus = 1.0f;\n")
    (tmp_path / "src" / "c_env" / "cs2_env.c").write_text("float plant_progress_reward = 0.05f;\n")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "checkpoints").mkdir()
    (tmp_path / "outputs" / "experiments").mkdir()
    (tmp_path / "outputs" / "experiments" / "results.jsonl").touch()
    # Gitignore outputs/ so train-produced files (checkpoints, new run dirs,
    # ledger updates) don't show up in git status and trip the post-run
    # clean-tree assertion. Mirrors the real repo's .gitignore layout.
    (tmp_path / ".gitignore").write_text("outputs/\n")
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


def _mock_train_py(repo: Path) -> None:
    """Replace src/train.py with a fake that just writes metrics + checkpoint.
    Simulates the real train.py without pufferlib/torch imports.
    """
    fake = """#!/usr/bin/env python
# env_fingerprint in run_experiment greps the live train.py for these two
# identifiers — keep them present even though this fake doesn't use them.
OBS_DIM = 105
ACTION_HEAD_SIZES = (9, 16, 2, 2, 3, 2, 2, 2)
import argparse, json, sys, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--train", action="store_true")
p.add_argument("--dump-config", dest="dump_config", action="store_true")
p.add_argument("--smoke", action="store_true")
p.add_argument("--timesteps", type=int, default=1_000_000)
p.add_argument("--name", type=str, default="")
p.add_argument("--checkpoint_dir", "--checkpoint-dir", type=str, dest="checkpoint_dir", default="")
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--device", type=str, default="cpu")
p.add_argument("--resume", type=str, default=None)
p.add_argument("--dust2", action="store_true")
args, _ = p.parse_known_args()

ckpt_dir = Path(args.checkpoint_dir)
ckpt_dir.mkdir(parents=True, exist_ok=True)

if args.dump_config:
    cfg = {
        "learning_rate": 3e-4, "gamma": 0.999, "clip_coef": 0.15,
        "batch_size": 163840, "seed": args.seed,
        "data_dir": args.checkpoint_dir,
    }
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, sort_keys=True))
    print("[DumpConfig] fake dumped")
    sys.exit(0)

if args.train:
    mp = ckpt_dir / "metrics.jsonl"
    with mp.open("a") as f:
        for i in range(10):
            f.write(json.dumps({
                "step": i * 163_840,
                "epoch": i,
                "winner_t": 0.5, "winner_ct": 0.5,
                "kills_t": 3.0, "kills_ct": 3.0,
                "bomb_planted": 0.03,
                "approx_kl": 0.02, "clipfrac": 0.3,
                "ret_mean": -0.01, "explained_variance": 0.8,
                "SPS": 200000,
            }) + "\\n")
    (ckpt_dir / "dust2_policy.pt").write_bytes(b"fake weights")
    print(f"[FakeTrain] wrote metrics + policy to {ckpt_dir}")
    sys.exit(0)

sys.exit(0)
"""
    (repo / "src" / "train.py").write_text(fake)
    subprocess.run(["git", "add", "src/train.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "use fake train.py"], cwd=repo, check=True)


def test_full_run_happy_path(tmp_path):
    """End-to-end: preconditions pass → branch created → fake train runs →
    metrics copied → analyzed → ledger appended → STATUS=done."""
    repo = _init_fake_repo(tmp_path)
    _mock_train_py(repo)

    env = os.environ.copy()
    env["CS2RL_REPO_ROOT"] = str(repo)
    env.pop("CS2RL_FAKE_TRAIN", None)
    r = subprocess.run(
        [
            sys.executable,
            str(RUN_EXP),
            "--tag",
            "happy",
            "--hypothesis",
            "does it work",
            "--change-type",
            "hp",
            "--changed-files",
            "",
            "--timesteps",
            "100",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"

    exp_dirs = list((repo / "outputs" / "experiments").glob("*-happy"))
    assert len(exp_dirs) == 1
    run_dir = exp_dirs[0]

    assert (run_dir / "STATUS.txt").read_text().startswith("done")
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["sample_count"] == 10
    assert summary["verdict"] in ("keep", "inconclusive", "bug")
    assert (run_dir / "analysis.md").exists()

    ledger_rows = (repo / "outputs" / "experiments" / "results.jsonl").read_text().splitlines()
    assert len(ledger_rows) == 1
    assert json.loads(ledger_rows[0])["run_id"] == run_dir.name

    assert "run_id" in r.stdout

    branch_r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    assert branch_r.stdout.strip() == "main"
    diff_r = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    assert diff_r.stdout.strip() == ""

    assert not (repo / "outputs" / "experiments" / ".lock").exists()


def test_full_run_with_changed_file(tmp_path):
    """Subagent edits src/train.py; run_experiment commits only that declared file."""
    repo = _init_fake_repo(tmp_path)
    _mock_train_py(repo)

    tp = repo / "src" / "train.py"
    tp.write_text(tp.read_text().replace('"fake weights"', '"edited weights"'))

    env = os.environ.copy()
    env["CS2RL_REPO_ROOT"] = str(repo)
    env.pop("CS2RL_FAKE_TRAIN", None)
    r = subprocess.run(
        [
            sys.executable,
            str(RUN_EXP),
            "--tag",
            "edit",
            "--hypothesis",
            "edit train.py",
            "--change-type",
            "hp",
            "--changed-files",
            "src/train.py",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert r.returncode == 0, f"stdout: {r.stdout}\nstderr: {r.stderr}"

    branches = subprocess.run(
        ["git", "branch", "--list", "exp/*"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "exp/" in branches


def test_reappend_ledger_replaces_row(tmp_path):
    """--reappend-ledger re-reads summary.json and rewrites the ledger row."""
    repo = _init_fake_repo(tmp_path)

    exp_dir = repo / "outputs" / "experiments"
    run_id = "150426-9-rerun"
    rd = exp_dir / run_id
    rd.mkdir(parents=True)
    (rd / "summary.json").write_text(
        json.dumps({
            "run_id": run_id,
            "verdict": "keep",
            "notes": "v1",
        }))

    ledger = exp_dir / "results.jsonl"
    ledger.write_text(json.dumps({"run_id": run_id, "verdict": "discard", "notes": "old"}) + "\n")

    (rd / "summary.json").write_text(
        json.dumps({
            "run_id": run_id,
            "verdict": "keep",
            "notes": "edited",
        }))

    env = os.environ.copy()
    env["CS2RL_REPO_ROOT"] = str(repo)
    r = subprocess.run(
        [sys.executable, str(RUN_EXP), "--reappend-ledger", run_id],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )
    assert r.returncode == 0, r.stderr

    rows = [json.loads(ln) for ln in ledger.read_text().splitlines() if ln.strip()]
    assert len(rows) == 1
    assert rows[0]["notes"] == "edited"
    assert rows[0]["verdict"] == "keep"


def test_reappend_ledger_no_lock(tmp_path):
    """--reappend-ledger works even when .lock exists (bypasses precondition)."""
    repo = _init_fake_repo(tmp_path)
    exp_dir = repo / "outputs" / "experiments"
    (exp_dir / ".lock").write_text("stale")
    rd = exp_dir / "150426-0-x"
    rd.mkdir(parents=True)
    (rd / "summary.json").write_text(json.dumps({"run_id": "150426-0-x", "verdict": "keep"}))

    env = os.environ.copy()
    env["CS2RL_REPO_ROOT"] = str(repo)
    r = subprocess.run(
        [sys.executable, str(RUN_EXP), "--reappend-ledger", "150426-0-x"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )
    assert r.returncode == 0
    assert (exp_dir / ".lock").exists()
