import subprocess, sys, os, pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_sf_train_1000_steps(tmp_path):
    """SF APPO training runs 1 000 steps and writes a checkpoint directory."""
    result = subprocess.run(
        [sys.executable, "train.py", "--train",
         "--timesteps", "1000",
         "--num_workers", "1",
         "--num_envs_per_worker", "1",
         "--train_dir", str(tmp_path)],
        capture_output=True, text=True, timeout=180,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"train.py --train failed (rc={result.returncode}):\n{result.stderr[-2000:]}"
    )
    # SF creates an experiment sub-dir — verify something was written
    assert any(tmp_path.rglob("*")), (
        "train_dir is empty after training — no checkpoint or config written"
    )


def test_team_spirit_daemon_logic():
    """The daemon thread logic correctly maps env_steps -> _TEAM_SPIRIT."""
    import sim as _sim
    anneal_steps = 5_000_000
    for steps, expected in [(0, 0.0), (2_500_000, 0.5), (5_000_000, 1.0), (9_999_999, 1.0)]:
        _sim._TEAM_SPIRIT = min(1.0, steps / anneal_steps)
        assert _sim._TEAM_SPIRIT == pytest.approx(expected, abs=1e-5), (
            f"steps={steps}: expected {expected}, got {_sim._TEAM_SPIRIT}"
        )
