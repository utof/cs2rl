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
            "--no-dead-run-abort",                     # F14: dead-run abort opt-out must stay exposed
            "--warmstart-entropy",
            "--warmstart-grace-steps",
            "--warmstart-ramp-steps",
            "--warmstart-alpha-ceiling",
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


def _dump_config(tmp_path, *extra_args):
    """Run --dump-config through `uv run` (project venv) and return the parsed
    config.json — the four warmstart keys must round-trip through the REAL
    argparse surface, not a hand-built Namespace (which would only exercise
    build_train_config's getattr fallbacks and hide a missing add_argument)."""
    import json

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    r = subprocess.run([
        "uv", "run", "python",
        str(TRAIN_SCRIPT), "--dump-config", "--checkpoint-dir",
        str(ckpt), *extra_args
    ],
                       capture_output=True,
                       text=True,
                       timeout=60,
                       cwd=REPO_ROOT)
    assert r.returncode == 0, f"stderr: {r.stderr}"
    return json.loads((ckpt / "config.json").read_text())


def test_warmstart_entropy_config_keys(tmp_path):
    cfg = _dump_config(tmp_path)
    assert cfg["warmstart_entropy"] is False
    assert cfg["warmstart_grace_steps"] == 5_000_000
    assert cfg["warmstart_ramp_steps"] == 10_000_000
    assert cfg["warmstart_alpha_ceiling"] == 0.0

    # Override ALL four with non-default values. Overriding only some would let
    # a misspelled dest= or a deleted add_argument pass silently: for the
    # untouched flags argparse's default equals build_train_config's getattr
    # fallback, so the dumped config looks correct either way. The 0.25 ceiling
    # is also the only exercise of type=float through the real parser.
    cfg = _dump_config(tmp_path, "--warmstart-entropy", "--warmstart-grace-steps", "1000",
                       "--warmstart-ramp-steps", "2000", "--warmstart-alpha-ceiling", "0.25")
    assert cfg["warmstart_entropy"] is True
    assert cfg["warmstart_grace_steps"] == 1000
    assert cfg["warmstart_ramp_steps"] == 2000
    assert cfg["warmstart_alpha_ceiling"] == 0.25


def test_train_smoke_returns_zero():
    result = run_train_command("--smoke")
    assert result.returncode == 0, (
        f"train.py --smoke failed (rc={result.returncode}):\n"
        f"STDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}")
    assert "[Smoke] Completed" in result.stdout, result.stdout
    assert "steps/sec" in result.stdout, result.stdout
