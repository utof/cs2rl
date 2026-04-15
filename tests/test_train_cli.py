import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "src" / "train.py"


def run_train_command(*args, timeout=180):
    return subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_train_help_shows_current_cli():
    result = run_train_command("--help", timeout=30)
    assert result.returncode == 0, f"--help failed:\n{result.stderr}"

    for flag in (
            "--smoke",
            "--train",
            "--record",
            "--eval",
            "--vec-backend",
            "--record-policy",
            "--eval-policy",
    ):
        assert flag in result.stdout, f"{flag} missing from --help output"

    for legacy_flag in ("--num_workers", "--num_envs_per_worker", "--train_dir"):
        assert legacy_flag not in result.stdout, f"{legacy_flag} should not be exposed anymore"


def test_dump_config_writes_json(tmp_path):
    """--dump-config writes <checkpoint_dir>/config.json and exits without training.

    Runs through `uv run python` so the project venv (and its deps) is active.
    Bumped timeout to 60s because uv's cold warm-up can be slow on first call.
    The whole point of --dump-config is zero side-effects: no torch import,
    no map load, no env spin-up — so it must return quickly.
    """
    import json

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(TRAIN_SCRIPT),
            "--dump-config",
            "--checkpoint-dir",
            str(ckpt_dir),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    config_path = ckpt_dir / "config.json"
    assert config_path.exists()

    with config_path.open() as f:
        config = json.load(f)
    for key in ("learning_rate", "gamma", "clip_coef", "batch_size"):
        assert key in config, f"missing key {key}"
    assert isinstance(config["batch_size"], int) and config["batch_size"] > 0


def test_train_smoke_returns_zero():
    result = run_train_command("--smoke")
    assert result.returncode == 0, (
        f"train.py --smoke failed (rc={result.returncode}):\n"
        f"STDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}")
    assert "[Smoke] Completed" in result.stdout, result.stdout
    assert "steps/sec" in result.stdout, result.stdout
