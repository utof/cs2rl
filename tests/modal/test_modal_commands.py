"""Behavior tests for scripts.modal_runner.commands: argv, child env, command builders.

One of the per-module runner test files (RUNNER_TEST_FILES in
tests/modal_runner_tables.py). Before you add, move or delete a test here, or
add a helper, read THE PLACEMENT RULE FOR RUNNER TESTS in
tests/test_modal_packaging.py: which file a test belongs in, what the change
costs in the seam manifest, and where helpers go.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

ROOT = REPO_ROOT

import scripts.modal_runner as mrl                                     # noqa: E402, I001
from scripts.modal_runner import commands                              # noqa: E402, I001
from tests.modal.modal_test_helpers import _valid_run_kwargs           # noqa: E402

# ── Argv builders: dump-config argv shares the runner-owned flags ──────────


def test_dump_config_argv_uses_same_owned_flags_without_train():
    request = mrl.build_run_request(
        **_valid_run_kwargs(effective_map="dust2", num_envs=64, vec_workers=4))
    argv = commands.build_dump_config_argv(request, None)
    assert argv[0] == "--dump-config"
    assert "--train" not in argv
    assert "--dust2" not in argv
    assert argv.count("--map") == 1 and argv[argv.index("--map") + 1] == "dust2"
    assert argv.count("--num_envs") == 1
    assert argv[argv.index("--num_envs") + 1] == "64"
    assert "--resume" not in argv


# ── Child process setup: environment + exact command builders ──────────────


def test_child_env_preserves_runtime_keys_and_forces_thread_caps():
    parent = {
        "PATH": "/usr/bin",
        "PYTHONPATH": "/opt/extra",
        "LD_LIBRARY_PATH": "/usr/lib/cuda",
        "LIBRARY_PATH": "/usr/lib",
        "CPATH": "/usr/include",
        "CPLUS_INCLUDE_PATH": "/usr/include/c++",
        "CUDA_HOME": "/usr/local/cuda",
        "CUDA_PATH": "/usr/local/cuda",
        "NVIDIA_VISIBLE_DEVICES": "0",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        "HOME": "/home/modal",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "LANGUAGE": "en_US:en",
        "LC_ALL": "C.UTF-8",
        "LC_CTYPE": "C.UTF-8",
        "LC_MESSAGES": "C",
        "SECRET_TOKEN": "drop-me",
        "WANDB_API_KEY": "parent-secret",
        "WANDB_AUTH": "also-secret",
        "OMP_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "32",
        "NUMEXPR_NUM_THREADS": "4",
    }
    env = commands.build_child_env(parent, wandb_enabled=False)
    for key in (
            "PATH",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "LIBRARY_PATH",
            "CPATH",
            "CPLUS_INCLUDE_PATH",
            "CUDA_HOME",
            "CUDA_PATH",
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES",
            "HOME",
            "TMPDIR",
            "LANG",
            "LANGUAGE",
            "LC_ALL",
            "LC_CTYPE",
            "LC_MESSAGES",
    ):
        assert env[key] == parent[key]
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "1"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["NUMEXPR_NUM_THREADS"] == "1"
    assert "SECRET_TOKEN" not in env
    assert not any(key.startswith("WANDB_") for key in env)


def test_child_env_wandb_disabled_strips_every_wandb_credential():
    env = commands.build_child_env(
        {
            "PATH": "/bin",
            "WANDB_API_KEY": "parent-secret",
            "WANDB_API_KEY_FILE": "/secrets/wandb",
            "WANDB_AUTH": "token",
        },
        wandb_enabled=False,
    )
    assert not any(key.startswith("WANDB_") for key in env)


def test_child_env_wandb_enabled_passes_secret_without_serializing():
    secret = "secret-from-modal"
    env = commands.build_child_env(
        {
            "PATH": "/bin",
            "WANDB_API_KEY": "parent-should-not-win",
            "WANDB_AUTH": "drop-this-too",
        },
        wandb_enabled=True,
        wandb_api_key=secret,
    )
    assert env["WANDB_API_KEY"] == secret
    # Parent WANDB_* credentials are not inherited; only the attached Secret.
    assert "WANDB_AUTH" not in env
    # The Secret must not appear in a JSON dump of the rest of the env.
    redacted = json.dumps({key: value for key, value in env.items() if key != "WANDB_API_KEY"})
    assert secret not in redacted
    with pytest.raises(mrl.ValidationError):
        commands.build_child_env({"PATH": "/bin"}, wandb_enabled=True, wandb_api_key=None)


def test_install_and_train_commands_are_exact():
    source_dir = "/tmp/extracted-src"
    assert commands.build_install_command(source_dir) == [
        "/usr/local/bin/uv",
        "pip",
        "install",
        "--python",
        "/opt/cs2rl/.venv/bin/python",
        "--no-deps",
        "--no-build-isolation",
        source_dir,
    ]
    argv = ["--train", "--timesteps", "1"]
    assert commands.build_train_command(argv) == [
        "/opt/cs2rl/.venv/bin/python",
        "-m",
        "cs2rl.train",
        *argv,
    ]


# ── Dump-config command: the exact command the prepare runs ────────────────


def test_dump_config_command_is_exact():
    request = mrl.build_run_request(**_valid_run_kwargs(run_id="ok-id"))
    resume = "/artifacts/inputs/sha256/abc.pt"
    assert commands.build_dump_config_command(request, resume) == [
        "/opt/cs2rl/.venv/bin/python",
        "-m",
        "cs2rl.train",
        *commands.build_dump_config_argv(request, resume),
    ]


# ── CUDA/PufferLib probe: the exact probe command and its verdicts ─────────


def _write_probe_stubs(root: Path, *, advantage_cuda: bool, record_path: Path) -> None:
    """Minimal torch/pufferlib so the probe string can run without a GPU."""
    torch_dir = root / "torch"
    torch_dir.mkdir(parents=True)
    (torch_dir / "__init__.py").write_text(f"""
class _Cuda:
    def is_available(self):
        return True
    def synchronize(self):
        open({str(record_path)!r}, "a").write("synchronize\\n")

class Tensor:
    def __init__(self, shape, device="cpu"):
        self.shape = shape
        self.device = device

def zeros(shape, device="cpu"):
    return Tensor(shape, device)

def ones(shape, device="cpu"):
    return Tensor(shape, device)

cuda = _Cuda()
""")
    puffer_dir = root / "pufferlib"
    puffer_dir.mkdir(parents=True)
    (puffer_dir / "__init__.py").write_text("")
    (puffer_dir / "pufferl.py").write_text(f"""
ADVANTAGE_CUDA = {advantage_cuda!r}

def compute_puff_advantage(values, rewards, terminals, ratio, advantages, *args):
    with open({str(record_path)!r}, "a") as handle:
        handle.write("compute_puff_advantage device=" + str(values.device) + "\\n")
    return advantages
""")


def _run_cuda_probe(tmp_path: Path, *, advantage_cuda: bool):
    stubs = tmp_path / "stubs"
    record_path = tmp_path / "probe.log"
    record_path.write_text("")
    _write_probe_stubs(stubs, advantage_cuda=advantage_cuda, record_path=record_path)
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(stubs)}
    result = subprocess.run(
        [sys.executable, "-c", commands.CUDA_PROBE_SOURCE],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, record_path


def test_cuda_probe_command_is_exact_python_string():
    command = commands.build_cuda_probe_command()
    assert command[0] == "/opt/cs2rl/.venv/bin/python"
    assert command[1] == "-c"
    source = command[2]
    assert source == commands.CUDA_PROBE_SOURCE
    assert "torch.cuda.is_available()" in source
    assert "pufferlib.pufferl" in source
    assert "ADVANTAGE_CUDA" in source
    assert "compute_puff_advantage" in source
    assert "synchronize" in source


def test_cuda_probe_fails_when_advantage_cuda_is_false(tmp_path):
    result, _record_path = _run_cuda_probe(tmp_path, advantage_cuda=False)
    assert result.returncode != 0
    assert "ADVANTAGE_CUDA" in (result.stderr + result.stdout)


def test_cuda_probe_success_invokes_kernel_and_synchronizes(tmp_path):
    result, record_path = _run_cuda_probe(tmp_path, advantage_cuda=True)
    assert result.returncode == 0, result.stderr
    lines = record_path.read_text().splitlines()
    assert "compute_puff_advantage device=cuda" in lines
    assert "synchronize" in lines
