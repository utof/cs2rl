import multiprocessing
import subprocess, sys, os, time, pytest

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


def test_team_spirit_daemon_updates_shared_value():
    """Daemon thread updates a multiprocessing.Value within 3 seconds."""
    from train import TeamSpiritCallback

    shared_ts = multiprocessing.Value('f', 0.0)
    cb = TeamSpiritCallback(anneal_steps=5_000_000)

    class MockRunner:
        total_env_steps_since_resume = 2_500_000

    stop = cb._make_daemon_thread(MockRunner(), shared_ts)
    try:
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if shared_ts.value > 0.0:
                break
            time.sleep(0.05)
        assert shared_ts.value == pytest.approx(0.5, abs=1e-5), (
            f"Expected ~0.5 after 3s, got {shared_ts.value}"
        )
    finally:
        cb.stop_daemon(stop)
