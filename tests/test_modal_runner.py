"""tests/test_modal_runner.py — optional Modal runner (plan tasks 14.1–14.2).

The Modal training runner (scripts/run_modal.py + helpers, later plan tasks) is an
OPTIONAL transport for one training run. Two invariants keep it from leaking into
the default local/scientific environment:

  * `modal` is a NON-DEFAULT dependency group pinned to `modal>=1.4.3,<2`. A plain
    `uv sync` must not pull it (non-default groups are excluded unless explicitly
    requested via `--group modal`), so the locked scientific stack stays
    byte-identical with or without the group present in pyproject.toml.
  * The local entrypoints (`src.train`, `scripts.exp_lib`, `scripts.run_experiment`)
    must be importable WITHOUT modal installed — i.e. nothing in the local stack
    imports modal at module scope. The runner must import modal lazily, inside the
    code paths that actually talk to Modal.

Pitfalls this file is careful about:
  * The import check runs in a FRESH subprocess: the pytest process itself may
    legitimately have `modal` in sys.modules once later runner tests exist, so
    asserting on the parent's sys.modules would give false failures.
  * `cwd=ROOT` makes the `src` / `scripts` namespace packages resolvable in the
    child; the child also prepends `src/` to sys.path because train.py imports its
    generated siblings bare (`from _action_spec import ...`). This keeps the test
    hermetic: it must not depend on the editable install's .pth, which any
    `uv sync --no-install-project` removes from the venv.
"""
import ast
import importlib
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import threading
import time
import tomllib
from datetime import UTC, timedelta
from enum import IntEnum
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # scripts/ is a namespace package; tests import scripts.modal_runner_lib
    # the same way the later CLIs will. Do not rely on the editable install.
    sys.path.insert(0, str(ROOT))

import scripts.modal_runner_lib as mrl                 # noqa: E402, I001


def _valid_run_kwargs(**overrides):
    """Minimal valid run fields for Task 2 cycle A. Later cycles tighten argv."""
    kwargs = {
        "run_id": "140826-b7r-seed2-shared",
        "git_sha": "a" * 40,
        "effective_map": "simple",
        "train_args": "--timesteps 30000000 --seed 2",
    }
    kwargs.update(overrides)
    return kwargs


def test_modal_is_an_explicit_dependency_group():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["dependency-groups"]["modal"] == ["modal>=1.4.3,<2"]


def test_local_entrypoints_do_not_import_modal():
    # sys.path.insert("src"): train.py resolves its generated siblings with bare
    # imports (`from _action_spec import ...`), so the src dir itself must be on
    # the child's path. Relying on the editable install's .pth instead would make
    # this test pass/fail on ambient venv state (any `uv sync --no-install-project`
    # removes it) and could silently import siblings from a DIFFERENT checkout.
    code = """
import sys
sys.path.insert(0, "src")
import src.train
import scripts.exp_lib
import scripts.run_experiment
assert 'modal' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_modal_runner_lib_does_not_import_modal_or_torch():
    # Fresh subprocess: the parent may already have torch (Task 3 checkpoint
    # tests) or modal (later runner tests) in sys.modules.
    code = """
import sys
import scripts.modal_runner_lib
assert 'modal' not in sys.modules
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


# ── Task 2 cycle A: structured run / resource / action fields ──────────────


@pytest.mark.parametrize(
    "run_id",
    ["a", "Z", "0id", "140826-b7r-seed2-shared", "A.B_c-1", "x" * 80],
)
def test_valid_run_ids_are_accepted(run_id):
    assert mrl.validate_run_id(run_id) == run_id
    request = mrl.build_run_request(**_valid_run_kwargs(run_id=run_id))
    assert request.run_id == run_id


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        "-leading-hyphen",
        ".dotstart",
        "has space",
        "has/slash",
        "has..dots",
        "semi;colon",
        "dollar$ign",
        "x" * 81,
        "unicode-ид",
    ],
)
def test_invalid_run_ids_are_rejected(run_id):
    with pytest.raises(mrl.ValidationError):
        mrl.validate_run_id(run_id)
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**_valid_run_kwargs(run_id=run_id))


@pytest.mark.parametrize("name", ["wandb", "W_and-b.1", "s" * 80])
def test_valid_secret_names_are_accepted(name):
    # Coupling (--wandb requires the secret and vice versa) is cycle D.
    assert mrl.validate_secret_name(name) == name


@pytest.mark.parametrize("name", ["", "-bad", "has space", "has/slash", "x" * 81])
def test_invalid_secret_names_are_rejected(name):
    with pytest.raises(mrl.ValidationError):
        mrl.validate_secret_name(name)


@pytest.mark.parametrize("effective_map", ["simple", "dust2"])
def test_required_map_allowlist(effective_map):
    request = mrl.build_run_request(**_valid_run_kwargs(effective_map=effective_map))
    assert request.effective_map == effective_map


@pytest.mark.parametrize("effective_map", [None, "", "cs2-dust2", "DUST2", "dust"])
def test_map_has_no_implicit_default_and_rejects_unknown(effective_map):
    kwargs = _valid_run_kwargs()
    if effective_map is None:
        kwargs.pop("effective_map")
        # The factory must require the map; omitting it is not "simple".
        with pytest.raises(TypeError):
            mrl.build_run_request(**kwargs)
        return
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**_valid_run_kwargs(effective_map=effective_map))


@pytest.mark.parametrize("gpu", ["T4", "L4", "A10"])
def test_allowed_gpus(gpu):
    request = mrl.build_run_request(**_valid_run_kwargs(gpu=gpu))
    assert request.gpu == gpu


@pytest.mark.parametrize("gpu", ["A10G", "A100", "H100", "any", "T4,L4", "t4"])
def test_unknown_gpus_are_rejected(gpu):
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**_valid_run_kwargs(gpu=gpu))


@pytest.mark.parametrize("num_envs", [16, 32, 64, 128, 256])
def test_allowed_num_envs_and_batch_floor(num_envs):
    request = mrl.build_run_request(**_valid_run_kwargs(num_envs=num_envs, vec_workers=1))
    assert request.num_envs == num_envs
    assert request.batch_size == num_envs * 10 * 64
    assert request.batch_size >= 8192


def test_num_envs_defaults_to_256():
    request = mrl.build_run_request(**_valid_run_kwargs())
    assert request.num_envs == mrl.DEFAULT_NUM_ENVS == 256
    assert request.batch_size == 256 * 10 * 64


@pytest.mark.parametrize("num_envs", [0, 8, 15, 257, 512, -16])
def test_num_envs_outside_allowlist_rejected(num_envs):
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**_valid_run_kwargs(num_envs=num_envs, vec_workers=1))


@pytest.mark.parametrize("cpu_cores", [4, 8, 16])
def test_allowed_cpu_request_equals_soft_limit(cpu_cores):
    request = mrl.build_run_request(**_valid_run_kwargs(cpu_cores=cpu_cores, vec_workers=1))
    assert request.cpu_cores == cpu_cores
    assert request.cpu_request_limit == (cpu_cores, cpu_cores)


@pytest.mark.parametrize("cpu_cores", [0, 2, 7, 32])
def test_cpu_outside_allowlist_rejected(cpu_cores):
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**_valid_run_kwargs(cpu_cores=cpu_cores, vec_workers=1))


@pytest.mark.parametrize("memory_mib", [8192, 16384, 32768])
def test_memory_request_equals_hard_limit(memory_mib):
    request = mrl.build_run_request(**_valid_run_kwargs(memory_mib=memory_mib))
    assert request.memory_mib == memory_mib
    assert request.memory_request_limit == (memory_mib, memory_mib)


@pytest.mark.parametrize("memory_mib", [4096, 8191, 32769, 0, -1])
def test_memory_outside_bounds_rejected(memory_mib):
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**_valid_run_kwargs(memory_mib=memory_mib))


@pytest.mark.parametrize(
    ("num_envs", "vec_workers"),
    [(256, 8), (256, 1), (32, 4), (16, 4), (128, 8)],
)
def test_vec_workers_must_divide_envs_and_not_exceed_cpu(num_envs, vec_workers):
    request = mrl.build_run_request(
        **_valid_run_kwargs(num_envs=num_envs, vec_workers=vec_workers, cpu_cores=8))
    assert request.vec_workers == vec_workers
    assert num_envs % vec_workers == 0
    assert vec_workers <= request.cpu_request_limit[1]


@pytest.mark.parametrize(
    ("num_envs", "vec_workers", "cpu_cores"),
    [
        (256, 0, 8),
        (256, -1, 8),
        (256, 7, 8),                                   # does not divide
        (32, 16, 8),                                   # exceeds CPU soft limit
        (16, 32, 16),                                  # exceeds envs and does not divide
    ],
)
def test_invalid_vec_worker_topology_rejected(num_envs, vec_workers, cpu_cores):
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(
            **_valid_run_kwargs(num_envs=num_envs, vec_workers=vec_workers, cpu_cores=cpu_cores))


def test_resource_defaults():
    request = mrl.build_run_request(**_valid_run_kwargs())
    assert request.cpu_cores == mrl.DEFAULT_CPU_CORES == 8
    assert request.memory_mib == mrl.DEFAULT_MEMORY_MIB == 16384
    assert request.vec_workers == mrl.DEFAULT_VEC_WORKERS == 8
    assert request.timeout_minutes == mrl.DEFAULT_TIMEOUT_MINUTES == 120
    assert request.save_every_seconds == mrl.DEFAULT_SAVE_EVERY_SECONDS == 300
    assert request.cpu_request_limit == (8, 8)
    assert request.memory_request_limit == (16384, 16384)


@pytest.mark.parametrize("timeout_minutes", [1, 120, 360])
def test_timeout_bounds_accepted(timeout_minutes):
    request = mrl.build_run_request(**_valid_run_kwargs(timeout_minutes=timeout_minutes))
    assert request.timeout_minutes == timeout_minutes


@pytest.mark.parametrize("timeout_minutes", [0, -5, 361])
def test_timeout_bounds_rejected(timeout_minutes):
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**_valid_run_kwargs(timeout_minutes=timeout_minutes))


@pytest.mark.parametrize("save_every_seconds", [60, 180, 300])
def test_save_cadence_accepted(save_every_seconds):
    request = mrl.build_run_request(**_valid_run_kwargs(save_every_seconds=save_every_seconds))
    assert request.save_every_seconds == save_every_seconds


@pytest.mark.parametrize("save_every_seconds", [59, 301, 0])
def test_save_cadence_rejected(save_every_seconds):
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**_valid_run_kwargs(save_every_seconds=save_every_seconds))


@pytest.mark.parametrize("action", ["status", "download"])
@pytest.mark.parametrize(
    "run_only",
    [
        ["--map", "simple"],
        ["--gpu", "T4"],
        ["--train-args", "--timesteps 1"],
        ["--git-sha", "a" * 40],
        ["--cpu-cores", "8"],
        ["--memory-mib", "8192"],
        ["--num-envs", "256"],
        ["--vec-workers", "8"],
        ["--timeout-minutes", "15"],
        ["--save-every-seconds", "60"],
        ["--resume-local-checkpoint", "x.pt"],
        ["--resume-run-id", "parent"],
        ["--wandb-secret-name", "wandb"],
    ],
)
def test_status_and_download_reject_run_only_options(action, run_only):
    with pytest.raises(mrl.ValidationError):
        mrl.parse_artifact_client_request([action, "--run-id", "ok-id", *run_only])


@pytest.mark.parametrize("action", [mrl.Action.STATUS, mrl.Action.DOWNLOAD])
def test_status_and_download_accept_only_run_id(action):
    request = mrl.parse_artifact_client_request([action.value, "--run-id", "ok-id"])
    assert request.action == action
    assert request.run_id == "ok-id"


def test_mounted_path_translates_client_roots():
    assert mrl.mounted_path(mrl.SOURCES_ROOT /
                            "abc.tar.gz") == mrl.VOLUME_MOUNT / "sources" / "abc.tar.gz"
    assert mrl.mounted_path(mrl.INPUTS_ROOT / "sha256" / "d.pt") == (mrl.VOLUME_MOUNT / "inputs" /
                                                                     "sha256" / "d.pt")
    assert mrl.mounted_path(mrl.RUNS_ROOT / "ok-id" / "STATUS.json") == (mrl.VOLUME_MOUNT / "runs" /
                                                                         "ok-id" / "STATUS.json")


@pytest.mark.parametrize(
    "relative",
    [
        PurePosixPath("/artifacts/sources/x"),
        PurePosixPath("runs/../secrets"),
        PurePosixPath("runs/ok/../../etc/passwd"),
    ],
)
def test_mounted_path_rejects_absolute_and_dotdot(relative):
    with pytest.raises(mrl.ValidationError):
        mrl.mounted_path(relative)


def test_status_enum_splits_terminal_and_nonterminal():
    assert mrl.Status.PREPARING.value == "preparing"
    assert mrl.Status.BUILDING.value == "building"
    assert mrl.Status.TRAINING.value == "training"
    assert {s.value
            for s in mrl.Status} >= {
                "preparing",
                "building",
                "training",
                "completed",
                "failed",
                "interrupted",
                "build_failed",
            }


# ── Task 2 cycle B: exact live-option grammar, no argparse prefixes ────────


def test_parse_train_args_preserves_punctuation_as_data():
    # shlex.split must keep ';', quotes, and paths as argv DATA. Never a shell.
    argv = mrl.parse_train_args("--timesteps 30000000 --wandb-entity 'org/name;rm -rf' --seed 2")
    assert argv == (
        "--timesteps",
        "30000000",
        "--wandb-entity",
        "org/name;rm -rf",
        "--seed",
        "2",
    )
    assert mrl.validate_train_args(argv) == 30_000_000


def test_parse_train_args_unclosed_quote_is_validation_error():
    with pytest.raises(mrl.ValidationError):
        mrl.parse_train_args('--timesteps 1 --wandb-entity "unclosed')
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**_valid_run_kwargs(train_args="--timesteps 1 --name 'oops"))


def test_omitted_train_args_fail_closed():
    kwargs = _valid_run_kwargs()
    del kwargs["train_args"]
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**kwargs)


def test_live_option_mirror_contains_exact_long_names_only():
    assert "--timesteps" in mrl.LIVE_TRAIN_OPTIONS
    assert "--num_envs" in mrl.LIVE_TRAIN_OPTIONS
    assert "--dust2" in mrl.LIVE_TRAIN_OPTIONS
    assert "--checkpoint-dir" in mrl.LIVE_TRAIN_OPTIONS
    assert "--checkpoint_dir" in mrl.LIVE_TRAIN_OPTIONS
    assert "--devi" not in mrl.LIVE_TRAIN_OPTIONS
    assert "--num-envs" not in mrl.LIVE_TRAIN_OPTIONS  # runner spelling, not live


def _live_train_long_options_from_source() -> set[str]:
    """Static train.py long options + hyphenated REWARD_WEIGHT_DEFAULTS keys.

    Reads source (no `import src.train`) so collection cannot pull CUDA.
    Generated `add_argument(f"--{_rw_name...}")` is a JoinedStr and is
    recovered from the defaults dict instead.
    """
    tree = ast.parse((ROOT / "src" / "train.py").read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "REWARD_WEIGHT_DEFAULTS":
                    assert isinstance(node.value, ast.Dict)
                    for key in node.value.keys:
                        assert isinstance(key, ast.Constant) and isinstance(key.value, str)
                        names.add(f"--{key.value.replace('_', '-')}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "add_argument":
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value.startswith("--"):
                        names.add(arg.value)
    return names


def test_live_train_option_mirror_matches_train_py():
    assert _live_train_long_options_from_source() == set(mrl.LIVE_TRAIN_OPTION_ARITY)


@pytest.mark.parametrize(
    "raw",
    [
        "--timesteps 1 leftover",
        "--timesteps 1 positional",
        "30000000",
        "--timesteps 1 -- 2",
    ],
)
def test_unconsumed_positional_tokens_rejected(raw):
    with pytest.raises(mrl.ValidationError):
        mrl.validate_train_args(mrl.parse_train_args(raw))
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**_valid_run_kwargs(train_args=raw))


_FORBIDDEN_FLAGS = [
    "--train",
    "--resume",
    "--name",
    "--checkpoint-dir",
    "--checkpoint_dir",
    "--device",
    "--save_every_sec",
    "--vec-backend",
    "--vec-num-workers",
    "--vec-overwork",
    "--dump-config",
    "--smoke",
    "--record",
    "--eval",
    "--dust2",
    "--num_envs",
    "--num-envs",
]


@pytest.mark.parametrize("flag", _FORBIDDEN_FLAGS)
@pytest.mark.parametrize("form", ["space", "equals"])
def test_forbidden_flags_rejected_in_both_forms(flag, form):
    # store_true flags have no value; arity-1 flags need a dummy value.
    valueless = {
        "--train",
        "--dump-config",
        "--smoke",
        "--record",
        "--eval",
        "--dust2",
        "--vec-overwork",
    }
    if flag in valueless:
        token = flag
    elif form == "equals":
        token = f"{flag}=dummy"
    else:
        token = f"{flag} dummy"
    if flag in valueless and form == "equals":
        token = f"{flag}=true"
    raw = f"--timesteps 30000000 {token}"
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**_valid_run_kwargs(train_args=raw))


@pytest.mark.parametrize(
    "token",
    [
        "--devi cuda",
        "--checkpoint-d /tmp/x",
        "--tim 1",
        "--num 256",
        "--save_every 60",
        "--vec-b multiprocessing",
        "--vec-n 8",
        "--resu /tmp/x.pt",
        "--sma",
        "--rec",
        "--eva",
        "--dum",
        "--tra",
    ],
)
def test_argparse_resolvable_prefixes_rejected(token):
    raw = f"--timesteps 30000000 {token}"
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**_valid_run_kwargs(train_args=raw))


def test_unknown_spelling_rejected_before_ownership():
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**_valid_run_kwargs(train_args="--timesteps 1 --num-env 256"))


@pytest.mark.parametrize(
    "raw",
    [
        "--seed 2",
        "--timesteps",
        "--timesteps 0",
        "--timesteps -1",
        "--timesteps 1 --timesteps 2",
        "--timesteps=0",
        "--timesteps foo",
    ],
)
def test_timesteps_must_be_exactly_one_positive_int(raw):
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**_valid_run_kwargs(train_args=raw))


def test_timesteps_below_one_full_batch_rejected_at_request_build():
    batch_size = _live_batch_size()
    with pytest.raises(mrl.ValidationError, match="full batch"):
        mrl.build_run_request(**_valid_run_kwargs(train_args=f"--timesteps {batch_size - 1}"))
    request = mrl.build_run_request(**_valid_run_kwargs(train_args=f"--timesteps {batch_size}"))
    assert request.timesteps == batch_size
    assert request.timesteps == request.batch_size


def test_exact_allowed_flags_are_kept():
    raw = ("--timesteps 30000000 --seed 2 --warmstart-entropy --no-dead-run-abort "
           "--tag-diagnostic --tag-every 5 --tct-split-heads --reward-win 1.0")
    request = mrl.build_run_request(**_valid_run_kwargs(train_args=raw))
    assert request.timesteps == 30_000_000
    assert request.train_args[0] == "--timesteps"
    assert "--warmstart-entropy" in request.train_args
    assert "--tag-every" in request.train_args


# ── Task 2 cycle C: runner-owned argv injection ────────────────────────────


def test_training_argv_is_exact_for_resume_and_defaults():
    request = mrl.build_run_request(**_valid_run_kwargs())
    run_root = Path("/artifacts/runs/140826-b7r-seed2-shared")
    remote_resume = "/artifacts/inputs/sha256/abc.pt"
    assert request.training_argv(run_root, remote_resume=remote_resume) == [
        "--train",
        "--timesteps",
        "30000000",
        "--seed",
        "2",
        "--num_envs",
        "256",
        "--checkpoint-dir",
        f"{run_root}/checkpoints",
        "--device",
        "cuda",
        "--save_every_sec",
        "300",
        "--vec-backend",
        "multiprocessing",
        "--vec-num-workers",
        "8",
        "--resume",
        remote_resume,
    ]
    assert mrl.build_train_argv(request,
                                remote_resume) == request.training_argv(run_root,
                                                                        remote_resume=remote_resume)


def test_simple_map_omits_dust2_dust2_injects_once():
    run_root = Path("/artifacts/runs/ok-id")
    simple_request = mrl.build_run_request(**_valid_run_kwargs(effective_map="simple"))
    dust2_request = mrl.build_run_request(
        **_valid_run_kwargs(effective_map="dust2", run_id="ok-id"))
    assert simple_request.effective_map == "simple"
    assert "--dust2" not in simple_request.training_argv(run_root)
    assert dust2_request.training_argv(run_root).count("--dust2") == 1


def test_num_envs_injected_exactly_once_with_live_spelling():
    request = mrl.build_run_request(**_valid_run_kwargs(num_envs=32, vec_workers=4))
    argv = request.training_argv(Path("/artifacts/runs/ok-id"))
    assert argv.count("--num_envs") == 1
    assert argv[argv.index("--num_envs") + 1] == "32"
    assert "--num-envs" not in argv


def test_dump_config_argv_uses_same_owned_flags_without_train():
    request = mrl.build_run_request(
        **_valid_run_kwargs(effective_map="dust2", num_envs=64, vec_workers=4))
    argv = mrl.build_dump_config_argv(request, None)
    assert argv[0] == "--dump-config"
    assert "--train" not in argv
    assert argv.count("--dust2") == 1
    assert argv.count("--num_envs") == 1
    assert argv[argv.index("--num_envs") + 1] == "64"
    assert "--resume" not in argv


def test_no_resume_omits_resume_flag():
    request = mrl.build_run_request(**_valid_run_kwargs())
    argv = request.training_argv(Path("/artifacts/runs/ok-id"))
    assert "--resume" not in argv


# ── Task 2 cycle D: resume / W&B coupling ──────────────────────────────────


def test_local_and_prior_resume_are_mutually_exclusive():
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**_valid_run_kwargs(
            resume_local_checkpoint="outputs/checkpoints/bc_warmstart.pt",
            resume_run_id="parent-run",
        ))


def test_each_resume_source_alone_is_accepted():
    local = mrl.build_run_request(**_valid_run_kwargs(
        resume_local_checkpoint="outputs/checkpoints/bc_warmstart.pt"))
    assert local.resume.local_checkpoint == Path("outputs/checkpoints/bc_warmstart.pt")
    assert local.resume.prior_run_id is None
    prior = mrl.build_run_request(**_valid_run_kwargs(resume_run_id="parent-run"))
    assert prior.resume.prior_run_id == "parent-run"
    assert prior.resume.local_checkpoint is None


def test_wandb_requires_named_secret():
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**_valid_run_kwargs(train_args="--timesteps 1 --wandb"))


def test_secret_without_wandb_rejected():
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**_valid_run_kwargs(wandb_secret_name="wandb"))


def test_wandb_with_secret_is_accepted():
    request = mrl.build_run_request(
        **_valid_run_kwargs(train_args="--timesteps 163840 --wandb", wandb_secret_name="wandb"))
    assert request.wandb_secret_name == "wandb"
    assert "--wandb" in request.train_args


def test_run_request_constructor_enforces_allowlists():
    with pytest.raises(mrl.ValidationError):
        mrl.RunRequest(run_id="ok-id", git_sha="a" * 40, effective_map="simple")
    with pytest.raises(mrl.ValidationError):
        mrl.RunRequest(
            run_id="ok-id",
            git_sha="a" * 40,
            effective_map="cs2-dust2",
            train_args=("--timesteps", "1"),
            timesteps=1,
        )
    with pytest.raises(mrl.ValidationError):
        mrl.RunRequest(
            run_id="ok-id",
            git_sha="a" * 40,
            effective_map="simple",
            gpu="H100",
            train_args=("--timesteps", "1"),
            timesteps=1,
        )
    with pytest.raises(mrl.ValidationError):
        mrl.RunRequest(
            run_id="ok-id",
            git_sha="a" * 40,
            effective_map="simple",
            train_args=("--timesteps", "1", "--device", "cuda"),
            timesteps=1,
        )
    with pytest.raises(mrl.ValidationError):
        mrl.RunRequest(
            run_id="ok-id",
            git_sha="a" * 40,
            effective_map="simple",
            train_args=("--timesteps", "1", "--wandb"),
            timesteps=1,
        )


def test_resume_request_constructor_enforces_mutual_exclusion():
    with pytest.raises(mrl.ValidationError):
        mrl.ResumeRequest(
            local_checkpoint=Path("outputs/checkpoints/bc_warmstart.pt"),
            prior_run_id="parent-run",
        )


def test_valid_direct_run_request_still_constructs():
    request = mrl.RunRequest(
        run_id="ok-id",
        git_sha="a" * 40,
        effective_map="simple",
        train_args=("--timesteps", "1"),
        timesteps=1,
    )
    assert request.timesteps == 1
    assert request.effective_map == "simple"


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


def test_validate_clean_head_accepts_matching_clean_commit(tmp_path):
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    assert mrl.validate_clean_head(repo, sha) == sha
    assert mrl.validate_clean_head(repo, sha.upper()) == sha


@pytest.mark.parametrize(
    "bad_sha",
    [
        "not-a-sha",
        "abc",
        "g" * 40,
        "a" * 39,
        "a" * 41,
    ],
)
def test_validate_clean_head_rejects_non_hex_sha(tmp_path, bad_sha):
    repo = _init_source_repo(tmp_path)
    with pytest.raises(mrl.ValidationError):
        mrl.validate_clean_head(repo, bad_sha)


def test_validate_clean_head_rejects_unknown_object(tmp_path):
    repo = _init_source_repo(tmp_path)
    with pytest.raises(mrl.ValidationError):
        mrl.validate_clean_head(repo, "b" * 40)


def test_validate_clean_head_rejects_sha_that_is_not_head(tmp_path):
    repo = _init_source_repo(tmp_path)
    (repo / "readme.txt").write_text("second\n")
    _git(repo, "add", "readme.txt")
    _git(repo, "commit", "-qm", "second")
    parent = _git(repo, "rev-parse", "HEAD^")
    with pytest.raises(mrl.ValidationError):
        mrl.validate_clean_head(repo, parent)


def test_validate_clean_head_rejects_unstaged_tracked_change(tmp_path):
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    (repo / "readme.txt").write_text("dirty\n")
    with pytest.raises(mrl.ValidationError):
        mrl.validate_clean_head(repo, sha)


def test_validate_clean_head_rejects_staged_tracked_change(tmp_path):
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    (repo / "readme.txt").write_text("staged\n")
    _git(repo, "add", "readme.txt")
    with pytest.raises(mrl.ValidationError):
        mrl.validate_clean_head(repo, sha)


# ── Task 3 cycle B: safe archive extraction ────────────────────────────────


def _write_tar(path: Path, info: tarfile.TarInfo, data: bytes = b"") -> None:
    import io

    with tarfile.open(path, "w") as tar:
        payload = io.BytesIO(data) if info.type == tarfile.REGTYPE else None
        if payload is not None:
            info.size = len(data)
        tar.addfile(info, payload)


def test_safe_extract_accepts_git_archive(tmp_path):
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    archive = tmp_path / "src.tar"
    _git(repo, "archive", "--format=tar", f"--output={archive}", sha)
    dest = tmp_path / "out"
    dest.mkdir()
    mrl.safe_extract_git_archive(archive, dest)
    assert (dest / "readme.txt").read_text() == "hello\n"


def test_safe_extract_rejects_unsafe_members(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    cases: list[tarfile.TarInfo] = []
    link = tarfile.TarInfo("link")
    link.type = tarfile.SYMTYPE
    link.linkname = "readme.txt"
    cases.append(link)
    hard = tarfile.TarInfo("hard")
    hard.type = tarfile.LNKTYPE
    hard.linkname = "readme.txt"
    cases.append(hard)
    fifo = tarfile.TarInfo("fifo")
    fifo.type = tarfile.FIFOTYPE
    cases.append(fifo)
    abs_path = tarfile.TarInfo("/etc/passwd")
    abs_path.type = tarfile.REGTYPE
    cases.append(abs_path)
    traversal = tarfile.TarInfo("foo/../../etc/passwd")
    traversal.type = tarfile.REGTYPE
    cases.append(traversal)
    for index, info in enumerate(cases):
        archive = tmp_path / f"bad-{index}.tar"
        _write_tar(archive, info, data=b"x")
        with pytest.raises(mrl.ValidationError):
            mrl.safe_extract_git_archive(archive, dest)


# ── Task 3 cycle C: deterministic archive + provenance sidecar ──────────────


def _source_repo_with_noise(tmp_path: Path) -> Path:
    repo = _init_source_repo(tmp_path)
    script = repo / "tool.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o755)
    _git(repo, "add", "tool.sh")
    _git(repo, "update-index", "--chmod=+x", "tool.sh")
    _git(repo, "commit", "-qm", "add executable")
    (repo / ".env").write_text("SECRET=1\n")
    (repo / "noise.txt").write_text("untracked\n")
    (repo / "outputs").mkdir()
    (repo / "outputs" / "run.log").write_text("nope\n")
    (repo / ".venv").mkdir()
    (repo / ".venv" / "pyvenv.cfg").write_text("x\n")
    docs_git = repo / "docs" / ".git"
    docs_git.mkdir(parents=True)
    (docs_git / "HEAD").write_text("ref: refs/heads/main\n")
    return repo


def _open_bundle_tar(bundle: Path) -> tarfile.TarFile:
    import gzip

    return tarfile.open(fileobj=gzip.open(bundle, "rb"), mode="r:")


def test_source_bundle_excludes_untracked_and_is_deterministic(tmp_path):
    repo = _source_repo_with_noise(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", f"{sha}^{{tree}}")
    first = tmp_path / "a.tar.gz"
    second = tmp_path / "b.tar.gz"
    prov_a = mrl.create_source_bundle(repo, sha, first)
    prov_b = mrl.create_source_bundle(repo, sha, second)
    assert first.read_bytes() == second.read_bytes()
    assert prov_a.archive_sha256 == prov_b.archive_sha256 == mrl.sha256_file(first)
    assert prov_a.commit == sha
    assert prov_a.tree == tree
    with _open_bundle_tar(first) as tar:
        names = set(tar.getnames())
    assert "readme.txt" in names
    assert "tool.sh" in names
    assert ".cs2rl-provenance.json" in names
    assert ".env" not in names
    assert "noise.txt" not in names
    assert "outputs/run.log" not in names
    assert ".venv/pyvenv.cfg" not in names
    assert "docs/.git/HEAD" not in names


def test_source_bundle_hash_changes_for_new_commit(tmp_path):
    repo = _source_repo_with_noise(tmp_path)
    sha1 = _git(repo, "rev-parse", "HEAD")
    first = tmp_path / "old.tar.gz"
    mrl.create_source_bundle(repo, sha1, first)
    (repo / "readme.txt").write_text("changed\n")
    _git(repo, "add", "readme.txt")
    _git(repo, "commit", "-qm", "change")
    sha2 = _git(repo, "rev-parse", "HEAD")
    second = tmp_path / "new.tar.gz"
    mrl.create_source_bundle(repo, sha2, second)
    assert mrl.sha256_file(first) != mrl.sha256_file(second)


def test_source_bundle_normalizes_modes_and_gzip_header(tmp_path):
    repo = _source_repo_with_noise(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    bundle = tmp_path / "src.tar.gz"
    mrl.create_source_bundle(repo, sha, bundle)
    header = bundle.read_bytes()[:10]
    flags = header[3]
    mtime = int.from_bytes(header[4:8], "little")
    assert flags & 0x08 == 0
    assert mtime == 0
    with _open_bundle_tar(bundle) as tar:
        for member in tar.getmembers():
            mode = member.mode & 0o777
            if member.isdir():
                assert mode == 0o755
            elif member.name.endswith("tool.sh"):
                assert mode == 0o755
            else:
                assert mode == 0o644
        sidecar = tar.extractfile(".cs2rl-provenance.json")
        assert sidecar is not None
        import json

        payload = json.loads(sidecar.read().decode())
    assert payload["commit"] == sha
    assert payload["tree"] == _git(repo, "rev-parse", f"{sha}^{{tree}}")


# ── Task 3 cycle D: local checkpoint hash + weights-only load ───────────────


def test_validate_local_checkpoint_hashes_and_maps_paths(tmp_path):
    import torch

    ckpt = tmp_path / "warm.pt"
    torch.save({"weight": torch.tensor([1.0, 2.0])}, ckpt)
    provenance = mrl.validate_local_checkpoint(ckpt)
    digest = mrl.sha256_file(ckpt)
    assert provenance.sha256 == digest
    assert provenance.size == ckpt.stat().st_size
    assert provenance.client_path == mrl.INPUTS_ROOT / "sha256" / f"{digest}.pt"
    assert provenance.mount_path == Path("/artifacts/inputs/sha256") / f"{digest}.pt"


def test_validate_local_checkpoint_rejects_non_checkpoint(tmp_path):
    junk = tmp_path / "nope.txt"
    junk.write_text("not a checkpoint\n")
    with pytest.raises(mrl.ValidationError):
        mrl.validate_local_checkpoint(junk)
    missing = tmp_path / "missing.pt"
    with pytest.raises(mrl.ValidationError):
        mrl.validate_local_checkpoint(missing)


# ── Task 4 cycle A: atomic JSON + status transition table ──────────────────


def _live_batch_size(num_envs: int = 256) -> int:
    """Live compute_batch_dims: num_envs * 10 agents * 64 BPTT horizon."""
    return num_envs * mrl.AGENTS_PER_ENV * mrl.BPTT_HORIZON


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
        "thread_caps": [f"{key}={value}" for key, value in sorted(mrl._THREAD_CAP_ENV.items())],
        "resumed_from_run_id": None,
    }
    payload.update(overrides)
    return mrl.Manifest(**payload)


def test_atomic_write_json_replaces_and_cleans_temp_on_failure(tmp_path):
    path = tmp_path / "STATUS.json"
    mrl.atomic_write_json(path, {"ok": True})
    assert json.loads(path.read_text()) == {"ok": True}

    def boom(src, dst):
        raise OSError("injected replace failure")

    with pytest.raises(OSError, match="injected"):
        mrl.atomic_write_json(path, {"ok": False}, replace=boom)
    assert json.loads(path.read_text()) == {"ok": True}
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "STATUS.json"]
    assert leftovers == []


def test_status_transitions_are_monotonic_and_attempt_owned(tmp_path):
    from datetime import datetime

    run_root = tmp_path / "run"
    run_root.mkdir()
    lock = threading.Lock()
    now = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
    first = mrl.transition_status(run_root,
                                  mrl.Status.PREPARING,
                                  now=now,
                                  attempt_id="attempt-a",
                                  lock=lock)
    assert first.status is mrl.Status.PREPARING
    assert first.attempt_id == "attempt-a"
    mrl.transition_status(run_root, mrl.Status.BUILDING, now=now, attempt_id="attempt-a", lock=lock)
    mrl.transition_status(run_root, mrl.Status.TRAINING, now=now, attempt_id="attempt-a", lock=lock)
    done = mrl.transition_status(run_root,
                                 mrl.Status.COMPLETED,
                                 now=now,
                                 attempt_id="attempt-a",
                                 lock=lock)
    assert done.status is mrl.Status.COMPLETED
    # Idempotent same-terminal write by the original delivery.
    again = mrl.transition_status(run_root,
                                  mrl.Status.COMPLETED,
                                  now=now,
                                  attempt_id="attempt-a",
                                  lock=lock)
    assert again.status is mrl.Status.COMPLETED
    with pytest.raises(mrl.ValidationError):
        mrl.transition_status(run_root,
                              mrl.Status.TRAINING,
                              now=now,
                              attempt_id="attempt-a",
                              lock=lock)
    before = (run_root / "STATUS.json").read_bytes()
    # Redelivered delivery has no authority and must not touch the file.
    denied = mrl.transition_status(run_root,
                                   mrl.Status.FAILED,
                                   now=now,
                                   attempt_id="attempt-b",
                                   lock=lock)
    assert denied is None
    assert (run_root / "STATUS.json").read_bytes() == before


def test_manifest_records_authoritative_simple_map_not_legacy_env():
    manifest = _make_manifest()
    payload = manifest.to_dict()
    assert payload["schema_version"] == 1
    assert payload["attempt_id"] == "attempt-a"
    assert payload["source_archive_sha256"] == "c" * 64
    assert payload["resume_sha256"] is None
    assert payload["resume_size"] is None
    assert payload["resume_source_path"] is None
    assert payload["modal_version"] == "1.4.3"
    assert payload["image_digest"].startswith("sha256:")
    assert payload["gpu"] == "T4"
    assert payload["vec_workers"] == 8
    assert payload["effective_map"] == "simple"
    assert payload["cpu_request"] == payload["cpu_soft_limit"] == 8
    assert payload["memory_request_mib"] == payload["memory_hard_limit_mib"] == 16384
    # Live config.json currently lies; the manifest must not copy that field.
    live_config = {"env": "cs2-dust2", "seed": 2, "data_dir": "/artifacts/runs/ok-id/checkpoints"}
    assert live_config["env"] == "cs2-dust2"
    assert "env" not in payload
    assert payload["effective_map"] == "simple"


def test_run_result_schema_is_explicit():
    result = mrl.RunResult(
        schema_version=1,
        status=mrl.Status.COMPLETED,
        exit_code=0,
        started_at="2026-08-13T12:00:00+00:00",
        finished_at="2026-08-13T12:01:00+00:00",
        artifact_root="/artifacts/runs/ok-id",
        checkpoint_sha256="a" * 64,
        metrics_row_count=2,
        last_step=29_982_720,
    )
    payload = result.to_dict()
    assert payload["schema_version"] == 1
    assert payload["status"] == "completed"
    assert payload["exit_code"] == 0
    assert payload["checkpoint_sha256"] == "a" * 64
    assert payload["last_step"] == 29_982_720


# ── Task 4 cycle B: heartbeat + derived stale ──────────────────────────────


def _aware(hour=12, minute=0, second=0):
    from datetime import datetime

    return datetime(2026, 8, 13, hour, minute, second, tzinfo=UTC)


def _advance_to_training(run_root, attempt_id="a1", *, lock):
    mrl.transition_status(run_root,
                          mrl.Status.PREPARING,
                          now=_aware(),
                          attempt_id=attempt_id,
                          lock=lock)
    mrl.transition_status(run_root,
                          mrl.Status.BUILDING,
                          now=_aware(),
                          attempt_id=attempt_id,
                          lock=lock)
    return mrl.transition_status(run_root,
                                 mrl.Status.TRAINING,
                                 now=_aware(),
                                 attempt_id=attempt_id,
                                 lock=lock)


def test_heartbeat_refreshes_updated_at_under_lock(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    lock = threading.Lock()
    _advance_to_training(run_root, lock=lock)
    # Fake clock: a beat at each 60s mark must refresh updated_at.
    last = None
    for minute in (1, 2):
        now = _aware(minute=minute)
        last = mrl.write_heartbeat(run_root, now=now, attempt_id="a1", lock=lock)
        assert last is not None
        assert last.status is mrl.Status.TRAINING
        assert last.updated_at == now.isoformat()
        persisted = json.loads((run_root / "STATUS.json").read_text())
        assert persisted["updated_at"] == now.isoformat()
    assert last is not None


def test_blocked_heartbeat_cannot_clobber_completed(tmp_path):
    """Hold the lock, queue a beat, write completed, then release.

    The queued heartbeat must observe the terminal write and leave
    STATUS.json as completed. This is the interleaving an unlocked
    transition_status would lose: beat reads training, terminal write
    lands, beat writes training back.
    """
    run_root = tmp_path / "run"
    run_root.mkdir()
    lock = threading.Lock()
    _advance_to_training(run_root, lock=lock)

    lock.acquire()
    started = threading.Event()
    beat_status = []

    def beat():
        started.set()
        beat_status.append(
            mrl.write_heartbeat(run_root, now=_aware(minute=3), attempt_id="a1", lock=lock))

    worker = threading.Thread(target=beat)
    worker.start()
    try:
        assert started.wait(timeout=2.0)
        # started.set() races the acquire; park long enough to be blocked.
        threading.Event().wait(0.05)
        assert worker.is_alive()

        # Critical section is already held; do not re-enter the same Lock.
        written = mrl._transition_status_unlocked(run_root,
                                                  mrl.Status.COMPLETED,
                                                  now=_aware(minute=2),
                                                  attempt_id="a1")
        assert written is not None
        assert written.status is mrl.Status.COMPLETED
    finally:
        lock.release()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert json.loads((run_root / "STATUS.json").read_text())["status"] == "completed"
    assert beat_status and beat_status[0].status is mrl.Status.COMPLETED


def test_late_heartbeat_cannot_replace_terminal(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    lock = threading.Lock()
    stop = threading.Event()
    _advance_to_training(run_root, lock=lock)

    def heartbeat_loop():
        while not stop.is_set():
            mrl.write_heartbeat(run_root, now=_aware(minute=1), attempt_id="a1", lock=lock)
            stop.wait(0.01)

    worker = threading.Thread(target=heartbeat_loop)
    worker.start()
    # Terminal cleanup stops/joins the heartbeat, then transitions while
    # holding the shared lock. A delayed beat after join must no-op.
    stop.set()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    mrl.transition_status(run_root,
                          mrl.Status.COMPLETED,
                          now=_aware(minute=2),
                          attempt_id="a1",
                          lock=lock)
    beat = mrl.write_heartbeat(run_root, now=_aware(minute=3), attempt_id="a1", lock=lock)
    assert beat is not None
    assert beat.status is mrl.Status.COMPLETED
    assert json.loads((run_root / "STATUS.json").read_text())["status"] == "completed"


def test_derive_status_stale_after_five_minutes_does_not_mutate(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    written = _advance_to_training(run_root, lock=threading.Lock())
    before = (run_root / "STATUS.json").read_bytes()
    derived = mrl.derive_status(written, now=_aware(hour=12, minute=5))
    assert derived.stale is True
    assert derived.status is mrl.Status.INTERRUPTED
    assert (run_root / "STATUS.json").read_bytes() == before
    fresh = mrl.derive_status(written, now=_aware(minute=4, second=59))
    assert fresh.stale is False
    assert fresh.status is mrl.Status.TRAINING


def test_reservation_without_status_is_preparing_then_interrupted(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    reserved_at = _aware()
    (run_root / "reservation.json").write_text(
        json.dumps({
            "attempt_id": "a1",
            "created_at": reserved_at.isoformat()
        }))
    early = mrl.derive_run_view(run_root, now=_aware(minute=4))
    assert early.status is mrl.Status.PREPARING
    assert early.reason == "no-heartbeat"
    assert early.stale is False
    late = mrl.derive_run_view(run_root, now=_aware(minute=5))
    assert late.status is mrl.Status.INTERRUPTED
    assert late.reason == "no-heartbeat"
    assert late.stale is True
    assert not (run_root / "STATUS.json").exists()


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
    normalized = mrl.normalize_config_for_transport(config)
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


def test_normalize_config_strips_only_checkpoint_data_dir():
    raw = {"env": "cs2-dust2", "data_dir": "/artifacts/runs/x/checkpoints", "seed": 2}
    normalized = mrl.normalize_config_for_transport(raw)
    assert "data_dir" not in normalized
    assert normalized == {"env": "cs2-dust2", "seed": 2}
    assert mrl.normalize_config_for_transport(normalized) == normalized


def test_validate_completed_run_accepts_representative_metrics(tmp_path):
    train_src = (ROOT / "src" / "train.py").read_text()
    fn = ast.parse(train_src)
    for node in ast.walk(fn):
        if isinstance(node, ast.FunctionDef) and node.name == "compute_batch_dims":
            body = ast.get_source_segment(train_src, node)
            assert body is not None
            assert "agents_per_env = 10" in body
            assert "bptt_horizon = 64" in body
            assert "num_envs * agents_per_env * bptt_horizon" in body
            break
    else:
        raise AssertionError("live compute_batch_dims not found")
    assert mrl.AGENTS_PER_ENV == 10
    assert mrl.BPTT_HORIZON == 64
    run_root, manifest, effective, ckpt = _minimal_completed_tree(tmp_path)
    evidence = mrl.validate_completed_run(run_root, manifest)
    assert evidence.last_step == effective
    assert evidence.checkpoint_sha256 == mrl.sha256_file(ckpt)
    assert evidence.config_hash == manifest.config_hash
    assert manifest.batch_size == _live_batch_size(256)
    assert manifest.effective_timesteps == (30_000_000 // manifest.batch_size) * manifest.batch_size
    assert manifest.effective_timesteps >= manifest.batch_size
    assert json.loads((run_root / "checkpoints" / "config.json").read_text())["env"] == "cs2-dust2"
    assert manifest.effective_map == "simple"


@pytest.mark.parametrize(
    "defect", ["bad_ckpt", "empty", "malformed", "nonmonotonic", "wrong_hash", "short_step"])
def test_validate_completed_run_rejects_bad_evidence(tmp_path, defect):
    run_root, manifest, effective, ckpt = _minimal_completed_tree(tmp_path)
    if defect == "bad_ckpt":
        ckpt.write_text("nope")
    elif defect == "empty":
        (run_root / "checkpoints" / "metrics.jsonl").write_text("")
    elif defect == "malformed":
        (run_root / "checkpoints" / "metrics.jsonl").write_text("{nope\n")
    elif defect == "nonmonotonic":
        _write_metrics(run_root / "checkpoints" / "metrics.jsonl", [100, 50])
    elif defect == "wrong_hash":
        manifest = mrl.Manifest(**{**manifest.to_dict(), "config_hash": "e" * 64})
    elif defect == "short_step":
        _write_metrics(run_root / "checkpoints" / "metrics.jsonl", [effective - 1])
    with pytest.raises(mrl.ValidationError):
        mrl.validate_completed_run(run_root, manifest)


def test_list_run_artifacts_keeps_unknown_trainer_files(tmp_path):
    run_root, manifest, _, _ = _minimal_completed_tree(tmp_path)
    extra = run_root / "checkpoints" / "notes.txt"
    extra.write_text("keep me\n")
    nested = run_root / "checkpoints" / "extra" / "weird.bin"
    nested.parent.mkdir()
    nested.write_bytes(b"\x00\x01")
    listed = {path.relative_to(run_root).as_posix() for path in mrl.list_run_artifacts(run_root)}
    assert "checkpoints/notes.txt" in listed
    assert "checkpoints/extra/weird.bin" in listed
    assert "checkpoints/dust2_policy.pt" in listed


# ── Task 5 cycle A: registry/artifact protocols + durable reservation ──────


class FakeRegistry:
    """In-memory Modal Dict: put_if_absent is the only atomic insert."""

    def __init__(self):
        self._lock = threading.Lock()
        self.data: dict[str, dict[str, object]] = {}
        self.events: list[tuple[object, ...]] = []

    def put_if_absent(self, key: str, value: dict[str, object]) -> bool:
        with self._lock:
            self.events.append(("put_if_absent", key))
            if key in self.data:
                return False
            self.data[key] = dict(value)
            return True

    def get(self, key: str) -> dict[str, object] | None:
        with self._lock:
            stored = self.data.get(key)
            return None if stored is None else dict(stored)

    def set_existing(self, key: str, value: dict[str, object]) -> None:
        with self._lock:
            current = self.data.get(key)
            if current is None or current.get("attempt_id") != value.get("attempt_id"):
                raise mrl.ValidationError(
                    f"registry claim is not owned by {value.get('attempt_id')!r}")
            self.data[key] = dict(value)
            self.events.append(("set_existing", key))

    def expire(self, key: str) -> None:
        """Simulate Modal's seven-day inactivity eviction."""
        with self._lock:
            self.data.pop(key, None)
            self.events.append(("expire", key))


class FakeArtifactIndex:
    """In-memory Volume: client PurePosixPath keys, durable only after commit."""

    def __init__(self):
        self._lock = threading.Lock()
        self.committed: dict[PurePosixPath, bytes] = {}
        self.staged: dict[PurePosixPath, bytes] = {}
        self.events: list[tuple[object, ...]] = []

    def exists(self, path: PurePosixPath) -> bool:
        with self._lock:
            return path in self.committed

    def put_file(self, path: PurePosixPath, data: bytes) -> None:
        if not isinstance(path, PurePosixPath) or path.is_absolute():
            raise AssertionError(f"client Volume path must be relative PurePosixPath, got {path!r}")
        with self._lock:
            self.staged[path] = data
            self.events.append(("put_file", path))

    def commit(self) -> None:
        with self._lock:
            self.committed.update(self.staged)
            self.staged.clear()
            self.events.append(("commit", ))


def test_reserve_run_commits_reservation_immediately_after_dict_claim():
    """Winning claim must persist runs/<id>/reservation.json before any other upload."""
    registry = FakeRegistry()
    artifacts = FakeArtifactIndex()
    now = _aware()
    mrl.reserve_run(registry, artifacts, "ok-id", "attempt-a", now=now)

    reservation_path = mrl.RUNS_ROOT / "ok-id" / mrl.RESERVATION_FILENAME
    assert registry.events[0] == ("put_if_absent", "run:ok-id")
    assert artifacts.events == [("put_file", reservation_path), ("commit", )]
    claim = registry.get("run:ok-id")
    assert claim is not None
    assert claim["attempt_id"] == "attempt-a"
    assert claim["created_at"] == now.isoformat()
    payload = json.loads(artifacts.committed[reservation_path])
    assert payload["attempt_id"] == "attempt-a"
    assert payload["created_at"] == now.isoformat()
    # Reservation is the only Volume write: no source, checkpoint, or Function.
    assert all(event[0] in {"put_file", "commit"} for event in artifacts.events)
    assert artifacts.events[0][1] == reservation_path


# ── Task 5 cycle B: concurrent race + expired-Dict Volume fallback ─────────


def test_concurrent_reserve_run_admits_exactly_one_attempt():
    registry = FakeRegistry()
    artifacts = FakeArtifactIndex()
    results: list[tuple[str, str]] = []
    barrier = threading.Barrier(2)

    def worker(attempt_id: str) -> None:
        barrier.wait()
        try:
            mrl.reserve_run(registry, artifacts, "ok-id", attempt_id, now=_aware())
            results.append(("ok", attempt_id))
        except mrl.ValidationError:
            results.append(("reject", attempt_id))

    threads = [
        threading.Thread(target=worker, args=("attempt-a", )),
        threading.Thread(target=worker, args=("attempt-b", )),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    wins = [attempt for status, attempt in results if status == "ok"]
    losses = [attempt for status, attempt in results if status == "reject"]
    assert len(wins) == 1
    assert len(losses) == 1
    winner = wins[0]
    assert registry.get("run:ok-id")["attempt_id"] == winner
    reservation = json.loads(artifacts.committed[mrl.RUNS_ROOT / "ok-id" /
                                                 mrl.RESERVATION_FILENAME])
    assert reservation["attempt_id"] == winner


def test_dict_miss_with_existing_volume_manifest_rejects_run():
    registry = FakeRegistry()
    artifacts = FakeArtifactIndex()
    artifacts.committed[mrl.RUNS_ROOT / "ok-id" / mrl.MANIFEST_FILENAME] = b"{}\n"
    with pytest.raises(mrl.ValidationError):
        mrl.reserve_run(registry, artifacts, "ok-id", "attempt-b", now=_aware())
    assert registry.get("run:ok-id") is None
    assert artifacts.events == []


def test_upload_failure_after_reservation_keeps_run_id_and_records_failure_code():
    registry = FakeRegistry()
    artifacts = FakeArtifactIndex()
    mrl.reserve_run(registry, artifacts, "ok-id", "attempt-a", now=_aware())

    def boom_upload() -> None:
        raise OSError("could not upload /secrets/key to sources/dead.tar.gz")

    with pytest.raises(OSError, match="could not upload"):
        mrl.finish_reservation(registry, artifacts, "ok-id", "attempt-a", upload=boom_upload)
    claim = registry.get("run:ok-id")
    assert claim is not None
    assert claim["attempt_id"] == "attempt-a"
    assert claim["failure_code"] == "upload_failed"
    assert "secret" not in json.dumps(claim)
    assert "sources/dead.tar.gz" not in json.dumps(claim)
    assert mrl.RUNS_ROOT / "ok-id" / mrl.RESERVATION_FILENAME in artifacts.committed
    with pytest.raises(mrl.ValidationError):
        mrl.reserve_run(registry, artifacts, "ok-id", "attempt-b", now=_aware())


def test_expired_dict_still_rejects_when_volume_reservation_exists():
    registry = FakeRegistry()
    artifacts = FakeArtifactIndex()
    mrl.reserve_run(registry, artifacts, "ok-id", "attempt-a", now=_aware())

    def boom_upload() -> None:
        raise OSError("source upload failed")

    with pytest.raises(OSError):
        mrl.finish_reservation(registry, artifacts, "ok-id", "attempt-a", upload=boom_upload)
    # Seven inactive days evict the Dict lease; the Volume reservation remains.
    registry.expire("run:ok-id")
    assert registry.get("run:ok-id") is None
    with pytest.raises(mrl.ValidationError):
        mrl.reserve_run(registry, artifacts, "ok-id", "attempt-b", now=_aware())
    assert registry.get("run:ok-id") is None


# ── Task 5 cycle C: attempt redelivery / idempotent terminal behavior ──────


def test_first_attempt_claim_owns_canonical_state_writes(tmp_path):
    registry = FakeRegistry()
    artifacts = FakeArtifactIndex()
    mrl.reserve_run(registry, artifacts, "ok-id", "attempt-a", now=_aware())
    run_root = tmp_path / "run"
    run_root.mkdir()
    lock = threading.Lock()

    def train() -> str:
        status = mrl.transition_status(
            run_root,
            mrl.Status.PREPARING,
            now=_aware(),
            attempt_id="attempt-a",
            lock=lock,
        )
        assert status is not None
        (run_root / "result.json").write_text("{}\n")
        (run_root / "train.log").write_text("ok\n")
        (run_root / "checkpoints").mkdir()
        (run_root / "checkpoints" / "dust2_policy.pt").write_bytes(b"ckpt")
        artifacts.put_file(mrl.RUNS_ROOT / "ok-id" / "result.json", b"{}\n")
        artifacts.commit()
        return "trained"

    result = mrl.deliver_attempt(registry, artifacts, attempt_id="attempt-a", train=train)
    assert result == "trained"
    assert mrl.claim_attempt(registry, "attempt-a") is False
    persisted = json.loads((run_root / "STATUS.json").read_text())
    assert persisted["attempt_id"] == "attempt-a"
    assert persisted["status"] == "preparing"
    assert (run_root / "result.json").is_file()
    assert (run_root / "train.log").is_file()
    assert (run_root / "checkpoints" / "dust2_policy.pt").is_file()
    claim = registry.get("attempt:attempt-a")
    assert claim is not None
    assert claim["attempt_id"] == "attempt-a"


def test_same_input_redelivery_returns_without_train_or_writes(tmp_path):
    registry = FakeRegistry()
    artifacts = FakeArtifactIndex()
    mrl.reserve_run(registry, artifacts, "ok-id", "attempt-a", now=_aware())
    run_root = tmp_path / "run"
    run_root.mkdir()
    lock = threading.Lock()

    def train() -> str:
        mrl.transition_status(
            run_root,
            mrl.Status.PREPARING,
            now=_aware(),
            attempt_id="attempt-a",
            lock=lock,
        )
        (run_root / "result.json").write_text('{"status":"completed"}\n')
        (run_root / "train.log").write_text("first\n")
        (run_root / "checkpoints").mkdir()
        (run_root / "checkpoints" / "dust2_policy.pt").write_bytes(b"ckpt")
        artifacts.put_file(mrl.RUNS_ROOT / "ok-id" / "result.json", b"{}\n")
        artifacts.commit()
        return "trained"

    assert mrl.deliver_attempt(registry, artifacts, attempt_id="attempt-a",
                               train=train) == "trained"
    before_status = (run_root / "STATUS.json").read_bytes()
    before_result = (run_root / "result.json").read_bytes()
    before_log = (run_root / "train.log").read_bytes()
    before_ckpt = (run_root / "checkpoints" / "dust2_policy.pt").read_bytes()
    before_events = list(artifacts.events)
    before_committed = dict(artifacts.committed)

    def should_not_run() -> str:
        raise AssertionError("training callback must not run on redelivery")

    assert mrl.deliver_attempt(registry, artifacts, attempt_id="attempt-a",
                               train=should_not_run) == "redelivered"
    assert (run_root / "STATUS.json").read_bytes() == before_status
    assert (run_root / "result.json").read_bytes() == before_result
    assert (run_root / "train.log").read_bytes() == before_log
    assert (run_root / "checkpoints" / "dust2_policy.pt").read_bytes() == before_ckpt
    assert artifacts.events == before_events
    assert artifacts.committed == before_committed


def test_different_attempt_cannot_reach_remote_wrapper():
    registry = FakeRegistry()
    artifacts = FakeArtifactIndex()
    mrl.reserve_run(registry, artifacts, "ok-id", "attempt-a", now=_aware())
    with pytest.raises(mrl.ValidationError):
        mrl.reserve_run(registry, artifacts, "ok-id", "attempt-b", now=_aware())
    assert registry.get("run:ok-id")["attempt_id"] == "attempt-a"
    # Loser never received a Function delivery, so no attempt:<id> claim exists.
    assert registry.get("attempt:attempt-b") is None
    assert registry.get("attempt:attempt-a") is None


# ── Task 6 cycle A: child environment + exact command builders ─────────────


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
    env = mrl.build_child_env(parent, wandb_enabled=False)
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
    env = mrl.build_child_env(
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
    env = mrl.build_child_env(
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
        mrl.build_child_env({"PATH": "/bin"}, wandb_enabled=True, wandb_api_key=None)


def test_install_and_train_commands_are_exact():
    source_dir = "/tmp/extracted-src"
    assert mrl.build_install_command(source_dir) == [
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
    assert mrl.build_train_command(argv) == [
        "/opt/cs2rl/.venv/bin/python",
        "src/train.py",
        *argv,
    ]


# ── Task 6 cycle B: reload / verify / extract / install ────────────────────


class RecordingVolume:
    """Materializes the uploaded archive only on reload, like Volume.reload().

    commit() snapshots run_root. reload() restores that snapshot and drops
    uncommitted STATUS.json — Volume.reload() replaces the mount.
    """

    def __init__(self, src_archive: Path, dest_archive: Path, run_root: Path | None = None):
        self.events: list[str] = []
        self._src = src_archive
        self._dest = dest_archive
        self._run_root = Path(run_root) if run_root is not None else None
        self._committed_run_root: dict[Path, bytes] = {}

    def reload(self) -> None:
        self.events.append("reload")
        if not self._dest.exists():
            self._dest.parent.mkdir(parents=True, exist_ok=True)
            self._dest.write_bytes(self._src.read_bytes())
        self._restore_run_root()

    def commit(self) -> None:
        self.events.append("commit")
        self._snapshot_run_root()

    def _snapshot_run_root(self) -> None:
        if self._run_root is None or not self._run_root.exists():
            return
        snapshot: dict[Path, bytes] = {}
        for path in self._run_root.rglob("*"):
            if path.is_file():
                snapshot[path.relative_to(self._run_root)] = path.read_bytes()
        self._committed_run_root = snapshot

    def _restore_run_root(self) -> None:
        if self._run_root is None:
            return
        if self._run_root.exists():
            for path in self._run_root.rglob("*"):
                if not path.is_file():
                    continue
                if path.relative_to(self._run_root) not in self._committed_run_root:
                    path.unlink()
        for rel, data in self._committed_run_root.items():
            dest = self._run_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)


def _noop_heartbeat(**_kwargs):
    return SimpleNamespace(stop_and_join=lambda: None)


def _source_bundle(tmp_path: Path):
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", f"{sha}^{{tree}}")
    client_archive = tmp_path / "client.tar.gz"
    provenance = mrl.create_source_bundle(repo, sha, client_archive)
    mount_archive = tmp_path / "artifacts" / "sources" / f"{provenance.archive_sha256}.tar.gz"
    return sha, tree, client_archive, mount_archive, provenance


def _preflight_kwargs(tmp_path: Path, **overrides):
    sha, tree, client_archive, mount_archive, provenance = _source_bundle(tmp_path)
    run_root = tmp_path / "run"
    run_root.mkdir()
    kwargs = {
        "volume": RecordingVolume(client_archive, mount_archive, run_root),
        "archive_path": mount_archive,
        "expected_archive_sha256": provenance.archive_sha256,
        "expected_commit": sha,
        "expected_tree": tree,
        "request": mrl.build_run_request(**_valid_run_kwargs(run_id="ok-id")),
        "run_root": run_root,
        "attempt_id": "attempt-a",
        "lock": threading.Lock(),
        "ephemeral_parent": tmp_path / "ephemeral",
        "parent_env": {
            "PATH": "/usr/bin",
            "HOME": "/home/modal",
            "WANDB_API_KEY": "parent-secret"
        },
        "now": lambda: _aware(),
        "start_heartbeat": _noop_heartbeat,
    }
    kwargs.update(overrides)
    return kwargs


def test_recording_volume_reload_restores_committed_run_root(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    src = tmp_path / "src.tar.gz"
    src.write_bytes(b"archive")
    dest = tmp_path / "dest.tar.gz"
    volume = RecordingVolume(src, dest, run_root)
    (run_root / mrl.STATUS_FILENAME).write_text("uncommitted\n")
    volume.reload()
    assert dest.read_bytes() == b"archive"
    assert not (run_root / mrl.STATUS_FILENAME).exists()
    (run_root / mrl.STATUS_FILENAME).write_text("preparing\n")
    (run_root / "keep.txt").write_text("committed\n")
    volume.commit()
    (run_root / mrl.STATUS_FILENAME).write_text("dirty\n")
    (run_root / "extra.txt").write_text("uncommitted\n")
    volume.reload()
    assert (run_root / mrl.STATUS_FILENAME).read_text() == "preparing\n"
    assert (run_root / "keep.txt").read_text() == "committed\n"
    assert not (run_root / "extra.txt").exists()


def test_prepare_reloads_before_status_write_and_commits_before_heartbeat(tmp_path):
    order: list[object] = []

    def fake_run(cmd, **kwargs):
        if "--dump-config" in list(cmd):
            _write_dumped_config(run_root)
        return subprocess.CompletedProcess(cmd, 0)

    def start_heartbeat(**kwargs):
        order.append("heartbeat")
        status = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
        assert status["status"] == "preparing"
        return SimpleNamespace(stop_and_join=lambda: None)

    kwargs = _preflight_kwargs(tmp_path, run=fake_run, start_heartbeat=start_heartbeat)
    volume = kwargs["volume"]
    run_root = kwargs["run_root"]
    orig_reload = volume.reload
    orig_commit = volume.commit

    def tracking_reload():
        order.append(("reload", (run_root / mrl.STATUS_FILENAME).exists()))
        orig_reload()

    def tracking_commit():
        order.append("commit")
        orig_commit()

    volume.reload = tracking_reload
    volume.commit = tracking_commit
    mrl.prepare_remote_source(**kwargs)
    assert order[0] == ("reload", False)
    assert order[1] == "commit"
    assert order[2] == "heartbeat"
    assert Path(mrl.STATUS_FILENAME) in volume._committed_run_root


def test_prepare_reloads_verifies_extracts_then_installs(tmp_path):
    recorded: list[tuple[list[str], dict]] = []
    heartbeat_events: list[str] = []

    def start_heartbeat(**kwargs):
        heartbeat_events.append("start")
        return SimpleNamespace(stop_and_join=lambda: heartbeat_events.append("stopped"))

    def fake_run(cmd, **kwargs):
        assert heartbeat_events == ["start"]
        recorded.append((list(cmd), kwargs))
        if "--dump-config" in list(cmd):
            _write_dumped_config(run_root)
        return subprocess.CompletedProcess(cmd, 0)

    kwargs = _preflight_kwargs(tmp_path, run=fake_run, start_heartbeat=start_heartbeat)
    volume = kwargs["volume"]
    run_root = kwargs["run_root"]
    prepared = mrl.prepare_remote_source(**kwargs)
    assert volume.events[0] == "reload"
    assert recorded, "install command was never invoked"
    install_cmd, install_kwargs = recorded[0]
    assert install_cmd == mrl.build_install_command(prepared.source_dir)
    assert install_kwargs["cwd"] == os.fspath(prepared.source_dir)
    assert install_kwargs["shell"] is False
    assert install_kwargs["env"]["PATH"] == "/usr/bin"
    assert install_kwargs["env"]["OMP_NUM_THREADS"] == "1"
    assert "WANDB_API_KEY" not in install_kwargs["env"]
    assert (prepared.source_dir / "readme.txt").read_text() == "hello\n"
    sidecar = json.loads((prepared.source_dir / mrl.PROVENANCE_NAME).read_text())
    assert sidecar["commit"] == kwargs["expected_commit"]
    assert sidecar["tree"] == kwargs["expected_tree"]
    assert prepared.source_dir.is_relative_to(tmp_path / "ephemeral")
    assert not mount_is_extract_root(prepared.source_dir, kwargs["archive_path"])


def mount_is_extract_root(source_dir: Path, archive_path: Path) -> bool:
    return source_dir == archive_path.parent or archive_path.parent in source_dir.parents


def test_prepare_rejects_archive_hash_mismatch(tmp_path):
    kwargs = _preflight_kwargs(tmp_path, expected_archive_sha256="0" * 64, run=lambda *a, **k: None)
    with pytest.raises(mrl.ValidationError):
        mrl.prepare_remote_source(**kwargs)
    # Hash is checked after reload; the archive must not be trusted blindly.
    assert kwargs["volume"].events[0] == "reload"
    assert not (kwargs["run_root"] / mrl.STATUS_FILENAME).exists()


def test_prepare_rejects_provenance_sidecar_mismatch(tmp_path):
    kwargs = _preflight_kwargs(tmp_path, expected_commit="f" * 40, run=lambda *a, **k: None)
    with pytest.raises(mrl.ValidationError):
        mrl.prepare_remote_source(**kwargs)


def test_prepare_reads_archive_only_after_volume_reload(tmp_path):
    kwargs = _preflight_kwargs(tmp_path, run=lambda *a, **k: None)

    class BlindVolume:
        events: list[str] = []

        def reload(self) -> None:
            self.events.append("reload")

        def commit(self) -> None:
            self.events.append("commit")

    kwargs["volume"] = BlindVolume()
    with pytest.raises((mrl.ValidationError, FileNotFoundError, OSError)):
        mrl.prepare_remote_source(**kwargs)
    assert kwargs["volume"].events[0] == "reload"


# ── Task 6 cycle C: resume validation + cheap config dump/hash ──────────────


def _write_dumped_config(run_root: Path) -> dict[str, object]:
    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config = {"env": "cs2-dust2", "seed": 2, "data_dir": str(ckpt_dir), "timesteps": 30000000}
    (ckpt_dir / "config.json").write_text(json.dumps(config))
    return config


def test_dump_config_command_is_exact():
    request = mrl.build_run_request(**_valid_run_kwargs(run_id="ok-id"))
    resume = "/artifacts/inputs/sha256/abc.pt"
    assert mrl.build_dump_config_command(request, resume) == [
        "/opt/cs2rl/.venv/bin/python",
        "src/train.py",
        *mrl.build_dump_config_argv(request, resume),
    ]


def test_prepare_validates_resume_then_dumps_and_hashes_config(tmp_path):
    import torch

    ckpt = tmp_path / "artifacts" / "inputs" / "sha256" / "warm.pt"
    ckpt.parent.mkdir(parents=True)
    torch.save({"weight": torch.tensor([1.0, 2.0])}, ckpt)
    digest = mrl.sha256_file(ckpt)
    recorded: list[tuple[list[str], dict]] = []
    validated: list[Path] = []

    def fake_run(cmd, **kwargs):
        recorded.append((list(cmd), kwargs))
        cmd_list = list(cmd)
        if "--dump-config" in cmd_list:
            # Resume must already have been accepted before the cheap dump.
            assert validated == [ckpt]
            _write_dumped_config(kwargs_run_root)
        return subprocess.CompletedProcess(cmd, 0)

    orig_validate = mrl.validate_local_checkpoint

    def tracking_validate(path):
        validated.append(Path(path))
        return orig_validate(path)

    kwargs_run_root = tmp_path / "run"
    manifest = _make_manifest(config_hash="0" * 64, run_id="ok-id")
    kwargs = _preflight_kwargs(
        tmp_path,
        run=fake_run,
        start_heartbeat=_noop_heartbeat,
        remote_resume=str(ckpt),
        expected_resume_sha256=digest,
        manifest=manifest,
    )
    kwargs_run_root = kwargs["run_root"]
    monkey_validate = tracking_validate
    mrl.validate_local_checkpoint = monkey_validate
    try:
        prepared = mrl.prepare_remote_source(**kwargs)
    finally:
        mrl.validate_local_checkpoint = orig_validate
    assert validated == [ckpt]
    assert len(recorded) >= 2
    dump_cmd, dump_kwargs = recorded[1]
    request = kwargs["request"]
    assert dump_cmd == mrl.build_dump_config_command(request, str(ckpt))
    assert dump_kwargs["cwd"] == os.fspath(prepared.source_dir)
    assert dump_kwargs["shell"] is False
    assert dump_kwargs["env"]["OMP_NUM_THREADS"] == "1"
    dumped = json.loads((kwargs["run_root"] / "checkpoints" / "config.json").read_text())
    expected_hash = mrl.sha256_bytes(
        json.dumps(mrl.normalize_config_for_transport(dumped),
                   sort_keys=True,
                   separators=(",", ":")).encode())
    payload = json.loads((kwargs["run_root"] / mrl.MANIFEST_FILENAME).read_text())
    assert payload["config_hash"] == expected_hash
    assert prepared.config_hash == expected_hash
    assert "data_dir" not in mrl.normalize_config_for_transport(dumped)


def test_prepare_rejects_resume_hash_mismatch(tmp_path):
    import torch

    ckpt = tmp_path / "warm.pt"
    torch.save({"weight": torch.tensor([1.0])}, ckpt)
    kwargs = _preflight_kwargs(
        tmp_path,
        run=lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0),
        start_heartbeat=_noop_heartbeat,
        remote_resume=str(ckpt),
        expected_resume_sha256="0" * 64,
        manifest=_make_manifest(),
    )
    with pytest.raises(mrl.ValidationError):
        mrl.prepare_remote_source(**kwargs)


def test_prepare_rejects_non_checkpoint_resume(tmp_path):
    ckpt = tmp_path / "warm.pt"
    ckpt.write_text("not a checkpoint\n")
    kwargs = _preflight_kwargs(
        tmp_path,
        run=lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0),
        start_heartbeat=_noop_heartbeat,
        remote_resume=str(ckpt),
        expected_resume_sha256=mrl.sha256_file(ckpt),
        manifest=_make_manifest(),
    )
    with pytest.raises(mrl.ValidationError):
        mrl.prepare_remote_source(**kwargs)


# ── Task 6 cycle D: CUDA/PufferLib probe + preflight heartbeat ─────────────


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
        [sys.executable, "-c", mrl.CUDA_PROBE_SOURCE],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, record_path


def test_cuda_probe_command_is_exact_python_string():
    command = mrl.build_cuda_probe_command()
    assert command[0] == "/opt/cs2rl/.venv/bin/python"
    assert command[1] == "-c"
    source = command[2]
    assert source == mrl.CUDA_PROBE_SOURCE
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


def test_prepare_records_install_dump_probe_then_launch(tmp_path):
    recorded: list[list[str]] = []
    launched: list[object] = []

    def fake_run(cmd, **kwargs):
        recorded.append(list(cmd))
        if "--dump-config" in list(cmd):
            _write_dumped_config(kwargs_run_root)
        return subprocess.CompletedProcess(cmd, 0)

    def on_ready(prepared):
        launched.append(prepared)
        fake_run(
            prepared.train_command,
            cwd=os.fspath(prepared.source_dir),
            shell=False,
            env=prepared.child_env,
        )

    kwargs = _preflight_kwargs(
        tmp_path,
        run=fake_run,
        start_heartbeat=_noop_heartbeat,
        on_ready=on_ready,
        manifest=_make_manifest(run_id="ok-id"),
    )
    kwargs_run_root = kwargs["run_root"]
    prepared = mrl.prepare_remote_source(**kwargs)
    assert launched and launched[0] is prepared
    assert recorded[0] == mrl.build_install_command(prepared.source_dir)
    assert recorded[1] == mrl.build_dump_config_command(kwargs["request"], None)
    assert recorded[2] == mrl.build_cuda_probe_command()
    assert recorded[3] == prepared.train_command
    assert prepared.train_command == mrl.build_train_command(
        mrl.build_train_argv(kwargs["request"], None))
    assert prepared.heartbeat is not None


def test_preflight_failure_stops_heartbeat_then_writes_build_failed(tmp_path):
    order: list[str] = []
    status_at_stop: list[str] = []

    def start_heartbeat(**_kwargs):

        def stop_and_join():
            order.append("stop")
            status_path = kwargs["run_root"] / mrl.STATUS_FILENAME
            status_at_stop.append(json.loads(status_path.read_text())["status"])

        return SimpleNamespace(stop_and_join=stop_and_join)

    def fake_run(cmd, **_kwargs):
        if list(cmd)[:3] == ["/usr/local/bin/uv", "pip", "install"]:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    kwargs = _preflight_kwargs(tmp_path, run=fake_run, start_heartbeat=start_heartbeat)
    with pytest.raises(subprocess.CalledProcessError):
        mrl.prepare_remote_source(**kwargs)
    persisted = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
    assert order == ["stop"]
    assert status_at_stop == ["building"]
    assert persisted["status"] == "build_failed"
    assert persisted["attempt_id"] == "attempt-a"


def test_preflight_failure_keeps_build_failed_when_heartbeat_stop_raises(tmp_path):

    def start_heartbeat(**_kwargs):

        def stop_and_join(timeout: float = 5.0):
            raise RuntimeError("heartbeat worker did not stop")

        return SimpleNamespace(stop_and_join=stop_and_join)

    def fake_run(cmd, **_kwargs):
        if list(cmd)[:3] == ["/usr/local/bin/uv", "pip", "install"]:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    kwargs = _preflight_kwargs(tmp_path, run=fake_run, start_heartbeat=start_heartbeat)
    with pytest.raises(subprocess.CalledProcessError):
        mrl.prepare_remote_source(**kwargs)
    persisted = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "build_failed"
    assert persisted["attempt_id"] == "attempt-a"
    ephemeral = kwargs["ephemeral_parent"]
    assert not ephemeral.exists() or not any(ephemeral.iterdir())


def test_prepare_does_not_persist_wandb_secret(tmp_path):
    secret = "secret-from-modal"
    recorded_envs: list[dict[str, str]] = []

    def fake_run(cmd, **kwargs):
        recorded_envs.append(dict(kwargs["env"]))
        if "--dump-config" in list(cmd):
            _write_dumped_config(kwargs_run_root)
        return subprocess.CompletedProcess(cmd, 0)

    request = mrl.build_run_request(**_valid_run_kwargs(
        run_id="ok-id",
        train_args="--timesteps 163840 --wandb",
        wandb_secret_name="wandb",
    ))
    kwargs = _preflight_kwargs(
        tmp_path,
        run=fake_run,
        start_heartbeat=_noop_heartbeat,
        request=request,
        wandb_api_key=secret,
        manifest=_make_manifest(run_id="ok-id"),
    )
    kwargs_run_root = kwargs["run_root"]
    prepared = mrl.prepare_remote_source(**kwargs)
    assert all(env["WANDB_API_KEY"] == secret for env in recorded_envs)
    assert prepared.child_env["WANDB_API_KEY"] == secret
    assert secret not in repr(prepared)
    for path in kwargs["run_root"].rglob("*"):
        if path.is_file():
            assert secret not in path.read_text(errors="ignore")


def test_heartbeat_commits_throughout_blocked_preflight(tmp_path):

    class Clock:

        def __init__(self):
            self._now = _aware()
            self._lock = threading.Lock()

        def now(self):
            with self._lock:
                return self._now

        def advance(self, seconds: float):
            with self._lock:
                self._now += timedelta(seconds=seconds)
                return self._now

    clock = Clock()
    beat_times: list = []

    def wait(event: threading.Event, seconds: float) -> bool:
        clock.advance(seconds)
        return event.wait(0.01)

    def start_heartbeat(**kwargs):
        return mrl.start_heartbeat_worker(
            run_root=kwargs["run_root"],
            attempt_id=kwargs["attempt_id"],
            lock=kwargs["lock"],
            now=clock.now,
            commit=kwargs["commit"],
            interval=timedelta(seconds=60),
            wait=wait,
        )

    def fake_run(cmd, **kwargs):
        cmd_list = list(cmd)
        if cmd_list[:3] == ["/usr/local/bin/uv", "pip", "install"]:
            started = clock.now()
            deadline = time.monotonic() + 5.0
            while clock.now() - started < timedelta(minutes=5, seconds=1):
                if time.monotonic() > deadline:
                    raise TimeoutError("fake clock did not advance during blocked install")
                time.sleep(0.01)
            return subprocess.CompletedProcess(cmd, 0)
        if "--dump-config" in cmd_list:
            _write_dumped_config(kwargs_run_root)
        return subprocess.CompletedProcess(cmd, 0)

    kwargs = _preflight_kwargs(
        tmp_path,
        run=fake_run,
        start_heartbeat=start_heartbeat,
        now=clock.now,
        manifest=_make_manifest(run_id="ok-id"),
    )
    kwargs_run_root = kwargs["run_root"]
    volume = kwargs["volume"]
    orig_commit = volume.commit

    def recording_commit():
        beat_times.append(clock.now())
        orig_commit()

    volume.commit = recording_commit
    prepared = mrl.prepare_remote_source(**kwargs)
    assert prepared.heartbeat is not None
    assert prepared.heartbeat.thread.is_alive()
    prepared.heartbeat.stop_and_join()
    assert not prepared.heartbeat.thread.is_alive()
    assert len(beat_times) >= 6
    for earlier, later in zip(beat_times, beat_times[1:], strict=False):
        assert later - earlier <= timedelta(seconds=60)
    status = mrl.RunStatus.from_dict(
        json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text()))
    derived = mrl.derive_status(status, now=clock.now())
    assert derived.stale is False


def test_heartbeat_loop_survives_transient_commit_error(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    lock = threading.Lock()
    mrl.transition_status(run_root, mrl.Status.PREPARING, now=_aware(), attempt_id="a1", lock=lock)
    recovered = threading.Event()
    commits = {"n": 0}

    def flaky_commit():
        commits["n"] += 1
        if commits["n"] == 1:
            raise RuntimeError("volume commit blip")
        recovered.set()

    def wait(event: threading.Event, _seconds: float) -> bool:
        return event.wait(0.01)

    worker = mrl.start_heartbeat_worker(
        run_root=run_root,
        attempt_id="a1",
        lock=lock,
        now=_aware,
        commit=flaky_commit,
        interval=timedelta(seconds=60),
        wait=wait,
    )
    try:
        assert recovered.wait(timeout=2.0)
        assert worker.thread.is_alive()
        assert commits["n"] >= 2
    finally:
        worker.stop_and_join()
    assert not worker.thread.is_alive()


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


def _prepared_source(tmp_path: Path, **overrides) -> mrl.PreparedSource:
    source_dir = tmp_path / "src"
    source_dir.mkdir(exist_ok=True)
    prepared = mrl.PreparedSource(
        source_dir=source_dir,
        child_env={
            "PATH": "/usr/bin",
            "OMP_NUM_THREADS": "1"
        },
        train_command=["/opt/cs2rl/.venv/bin/python", "src/train.py", "--train"],
        heartbeat=None,
        config_hash="d" * 64,
    )
    for key, value in overrides.items():
        setattr(prepared, key, value)
    return prepared


def _advance_to_building(run_root, attempt_id="attempt-a", *, lock):
    mrl.transition_status(run_root,
                          mrl.Status.PREPARING,
                          now=_aware(),
                          attempt_id=attempt_id,
                          lock=lock)
    return mrl.transition_status(run_root,
                                 mrl.Status.BUILDING,
                                 now=_aware(),
                                 attempt_id=attempt_id,
                                 lock=lock)


def _training_kwargs(tmp_path: Path, **overrides):
    run_root = tmp_path / "run"
    run_root.mkdir(exist_ok=True)
    lock = threading.Lock()
    _advance_to_building(run_root, lock=lock)
    child = overrides.pop("child", FakeChild(stdout=b"ok\n"))
    launches: list[tuple[tuple, dict]] = []

    def default_factory(*args, **kwargs):
        launches.append((args, kwargs))
        return child

    kwargs = {
        "registry": FakeRegistry(),
        "attempt_id": "attempt-a",
        "run_root": run_root,
        "prepared": _prepared_source(tmp_path),
        "commit": lambda: None,
        "lock": lock,
        "now": lambda: _aware(),
        "process_factory": default_factory,
        "sleep": lambda _seconds: None,
        "log_sink": io.StringIO(),
    }
    kwargs.update(overrides)
    kwargs["_launches"] = launches
    kwargs["_child"] = child
    return kwargs


def test_training_child_starts_in_new_session_without_shell(tmp_path):
    kwargs = _training_kwargs(tmp_path)
    launches = kwargs.pop("_launches")
    kwargs.pop("_child")
    prepared = kwargs["prepared"]
    mrl.execute_training_attempt(**kwargs)
    assert len(launches) == 1
    args, kw = launches[0]
    command = args[0] if args else kw.get("args")
    assert list(command) == prepared.train_command
    assert kw["start_new_session"] is True
    assert kw["shell"] is False
    assert kw["cwd"] == os.fspath(prepared.source_dir)
    assert kw["env"] == prepared.child_env


def test_stdout_stderr_are_teed_to_log_sink_without_truncation(tmp_path):
    payload_out = ("OUT" + ("x" * 200_000) + "END\n").encode()
    payload_err = ("ERR" + ("y" * 200_000) + "FIN\n").encode()
    child = FakeChild(stdout=payload_out, stderr=payload_err)

    class CaptureSink:

        def __init__(self):
            self.parts: list[str] = []

        def write(self, data):
            self.parts.append(data)

        def flush(self):
            return None

        def close(self):
            return None

        def getvalue(self):
            return "".join(self.parts)

    sink = CaptureSink()
    kwargs = _training_kwargs(tmp_path, child=child, log_sink=sink)
    kwargs.pop("_launches")
    kwargs.pop("_child")
    (kwargs["run_root"] / "train.log").write_text("already here\n")
    mrl.execute_training_attempt(**kwargs)
    text = sink.getvalue()
    assert "OUT" in text and "END" in text
    assert "ERR" in text and "FIN" in text
    assert text.count("x") == 200_000
    assert text.count("y") == 200_000
    leftover = (kwargs["run_root"] / "train.log").read_text()
    assert leftover.startswith("already here\n")
    assert leftover.count("x") == 200_000
    assert leftover.endswith("FIN\n") or "FIN\n" in leftover
    assert leftover.count("y") >= 200_000


def test_same_attempt_redelivery_invokes_subprocess_once(tmp_path):
    commits: list[str] = []
    child = FakeChild(stdout=b"first-delivery\n")
    kwargs = _training_kwargs(
        tmp_path,
        child=child,
        commit=lambda: commits.append("commit"),
    )
    launches = kwargs.pop("_launches")
    kwargs.pop("_child")
    first = mrl.execute_training_attempt(**kwargs)
    assert first != mrl.REDELIVERED
    assert len(launches) == 1
    status_after_first = (kwargs["run_root"] / mrl.STATUS_FILENAME).read_bytes()
    commits_after_first = list(commits)

    def must_not_launch(*_args, **_kwargs):
        raise AssertionError("redelivered container must not start training")

    kwargs["process_factory"] = must_not_launch
    second = mrl.execute_training_attempt(**kwargs)
    assert second == mrl.REDELIVERED
    assert len(launches) == 1
    assert (kwargs["run_root"] / mrl.STATUS_FILENAME).read_bytes() == status_after_first
    assert commits == commits_after_first


# ── Task 7 cycle B: heartbeat / checkpoint commits ─────────────────────────


class _FakeClock:

    def __init__(self):
        self._now = _aware()
        self._lock = threading.Lock()

    def now(self):
        with self._lock:
            return self._now

    def advance(self, seconds: float):
        with self._lock:
            self._now += timedelta(seconds=seconds)
            return self._now


def _consume_training_kwargs(kwargs):
    kwargs.pop("_launches", None)
    kwargs.pop("_child", None)
    return kwargs


def _run_attempt_in_thread(kwargs):
    finished = threading.Event()
    boxed: list[object] = []

    def runner():
        try:
            boxed.append(mrl.execute_training_attempt(**kwargs))
        except Exception as err:
            boxed.append(err)
        finally:
            finished.set()

    thread = threading.Thread(target=runner)
    thread.start()
    return thread, finished, boxed


def test_heartbeat_commits_every_60s_while_training(tmp_path):
    clock = _FakeClock()
    child = FakeChild(hold=True)
    beat_times: list = []

    def commit():
        beat_times.append(clock.now())

    def wait(event: threading.Event, seconds: float) -> bool:
        clock.advance(seconds)
        return event.wait(0.01)

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            commit=commit,
            now=clock.now,
            wait=wait,
        ))
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        deadline = time.monotonic() + 5.0
        while len(beat_times) < 6 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(beat_times) >= 6
        for earlier, later in zip(beat_times, beat_times[1:], strict=False):
            assert later - earlier <= timedelta(seconds=60)
    finally:
        child.release()
        assert finished.wait(timeout=2.0)
        thread.join(timeout=2.0)
        assert not thread.is_alive()


def _write_policy_checkpoint(run_root: Path, value: float) -> Path:
    import torch

    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "dust2_policy.pt"
    torch.save({"weight": torch.tensor([value])}, ckpt)
    return ckpt


def test_stable_checkpoint_gets_sidecar_and_joint_commit(tmp_path):
    child = FakeChild(hold=True)
    events: list[tuple] = []
    settle_seen = threading.Event()

    def commit():
        sidecar = kwargs["run_root"] / "checkpoints" / "dust2_policy.pt.meta.json"
        ckpt = kwargs["run_root"] / "checkpoints" / "dust2_policy.pt"
        events.append(("commit", sidecar.is_file(), ckpt.is_file()))

    def fake_sleep(seconds: float) -> None:
        if seconds >= 1.0 and ckpt.is_file() and not settle_seen.is_set():
            settle_seen.set()
            _write_policy_checkpoint(kwargs["run_root"], 2.0)

    kwargs = _consume_training_kwargs(
        _training_kwargs(tmp_path, child=child, commit=commit, sleep=fake_sleep))
    ckpt = _write_policy_checkpoint(kwargs["run_root"], 1.0)
    first_digest = mrl.sha256_file(ckpt)
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    sidecar = kwargs["run_root"] / "checkpoints" / "dust2_policy.pt.meta.json"
    try:
        deadline = time.monotonic() + 5.0
        while not sidecar.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sidecar.is_file()
        meta = json.loads(sidecar.read_text())
        stable_digest = mrl.sha256_file(ckpt)
        assert stable_digest != first_digest
        assert meta["sha256"] == stable_digest
        assert meta["size"] == ckpt.stat().st_size
        assert meta["mtime_ns"] == ckpt.stat().st_mtime_ns
        assert meta["validated_at"]
        assert any(kind == "commit" and has_side and has_ckpt
                   for kind, has_side, has_ckpt in events)
    finally:
        child.release()
        assert finished.wait(timeout=2.0)
        thread.join(timeout=2.0)


def test_torn_checkpoint_does_not_publish_sidecar(tmp_path):
    child = FakeChild(hold=True)
    settle_calls = threading.Event()

    def fake_sleep(seconds: float) -> None:
        if seconds >= 1.0:
            settle_calls.set()

    kwargs = _consume_training_kwargs(_training_kwargs(tmp_path, child=child, sleep=fake_sleep))
    ckpt_dir = kwargs["run_root"] / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "dust2_policy.pt").write_bytes(b"torn-not-a-checkpoint")
    sidecar = ckpt_dir / "dust2_policy.pt.meta.json"
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        assert settle_calls.wait(timeout=2.0)
        time.sleep(0.05)
        assert not sidecar.exists()
    finally:
        child.release()
        assert finished.wait(timeout=2.0)
        thread.join(timeout=2.0)


# ── Task 7 cycle C: SIGINT / KeyboardInterrupt / SIGTERM cleanup ───────────


def _signal_hooks(child, *, release_on=signal.SIGKILL):
    originals = {signal.SIGINT: object(), signal.SIGTERM: object()}
    installed: dict[int, object] = dict(originals)
    kills: list[int] = []
    slept: list[float] = []

    def fake_signal(sig, handler):
        previous = installed.get(sig, originals.get(sig))
        installed[sig] = handler
        return previous

    def fake_getpgid(pid):
        return pid

    def fake_killpg(_pgid, sig):
        kills.append(sig)
        if release_on is not None and sig == release_on:
            child.release()

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    return {
        "originals": originals,
        "installed": installed,
        "kills": kills,
        "slept": slept,
        "signal_signal": fake_signal,
        "getpgid": fake_getpgid,
        "killpg": fake_killpg,
        "sleep": fake_sleep,
    }


def _wait_until_handlers(installed, originals, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        int_handler = installed.get(signal.SIGINT)
        term_handler = installed.get(signal.SIGTERM)
        if (int_handler is not None and int_handler is not originals[signal.SIGINT]
                and term_handler is not None and term_handler is not originals[signal.SIGTERM]):
            return int_handler, term_handler
        time.sleep(0.01)
    raise AssertionError("signal handlers were not installed around the child")


def test_sigint_and_sigterm_share_cleanup_and_restore_handlers(tmp_path):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    order: list[object] = []

    def stop_and_join():
        order.append("heartbeat_stopped")
        order.append(json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())["status"])

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    kwargs["prepared"] = _prepared_source(tmp_path,
                                          heartbeat=SimpleNamespace(stop_and_join=stop_and_join))
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, term_handler = _wait_until_handlers(hooks["installed"], hooks["originals"])
        assert int_handler is term_handler
        int_handler(signal.SIGINT, None)
        term_handler(signal.SIGTERM, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert hooks["installed"][signal.SIGINT] is hooks["originals"][signal.SIGINT]
    assert hooks["installed"][signal.SIGTERM] is hooks["originals"][signal.SIGTERM]
    persisted = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "interrupted"
    assert persisted["attempt_id"] == "attempt-a"
    assert order[0] == "heartbeat_stopped"
    assert order[1] == "training"
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]


def test_keyboard_interrupt_uses_same_cleanup(tmp_path):
    child = FakeChild(hold=True)

    def exploding_wait(timeout=None):
        raise KeyboardInterrupt

    child.wait = exploding_wait
    hooks = _signal_hooks(child)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    result = mrl.execute_training_attempt(**kwargs)
    assert result != mrl.REDELIVERED
    assert hooks["installed"][signal.SIGINT] is hooks["originals"][signal.SIGINT]
    assert hooks["installed"][signal.SIGTERM] is hooks["originals"][signal.SIGTERM]
    persisted = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "interrupted"
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]


def test_child_receives_term_then_kill_after_grace(tmp_path):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child, release_on=signal.SIGKILL)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        _handler, term_handler = _wait_until_handlers(hooks["installed"], hooks["originals"])
        term_handler(signal.SIGTERM, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]
    assert mrl.TERM_GRACE_SECONDS in child.wait_timeouts


def test_cleanup_closes_log_before_final_commit(tmp_path):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    events: list[str] = []

    class RecordingSink(io.StringIO):

        def close(self):
            events.append("log_closed")
            super().close()

    def commit():
        events.append("commit")

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            commit=commit,
            log_sink=RecordingSink(),
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, _term = _wait_until_handlers(hooks["installed"], hooks["originals"])
        int_handler(signal.SIGINT, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert "log_closed" in events
    assert "commit" in events[events.index("log_closed") + 1:]


def test_failed_cleanup_commit_does_not_let_redelivery_write(tmp_path):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    committed: list[dict] = []

    def commit():
        payload = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
        if payload["status"] == "interrupted":
            raise RuntimeError("volume commit failed")
        committed.append(payload)

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            commit=commit,
            now=lambda: _aware(),
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, _term = _wait_until_handlers(hooks["installed"], hooks["originals"])
        int_handler(signal.SIGINT, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert committed
    assert committed[-1]["status"] == "training"
    last = mrl.RunStatus.from_dict(committed[-1])
    derived = mrl.derive_status(last, now=_aware(minute=5))
    assert derived.stale is True
    assert derived.status is mrl.Status.INTERRUPTED
    before = (kwargs["run_root"] / mrl.STATUS_FILENAME).read_bytes()
    commits_before = list(committed)

    def must_not_launch(*_args, **_kwargs):
        raise AssertionError("redelivered container must not start training")

    kwargs["process_factory"] = must_not_launch
    assert mrl.execute_training_attempt(**kwargs) == mrl.REDELIVERED
    assert (kwargs["run_root"] / mrl.STATUS_FILENAME).read_bytes() == before
    assert committed == commits_before


def test_post_spawn_failure_kills_child_and_writes_terminal_status(tmp_path):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    (kwargs["run_root"] / mrl.TRAIN_LOG_NAME).mkdir()
    with pytest.raises(OSError):
        mrl.execute_training_attempt(**kwargs)
    persisted = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] in {"failed", "interrupted"}
    assert persisted["attempt_id"] == "attempt-a"
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]
    assert child.poll() is not None


def test_term_grace_is_deadline_not_mandatory_sleep(tmp_path):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child, release_on=signal.SIGTERM)
    started = time.monotonic()
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        _handler, term_handler = _wait_until_handlers(hooks["installed"], hooks["originals"])
        term_handler(signal.SIGTERM, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert time.monotonic() - started < 5.0
    assert hooks["kills"] == [signal.SIGTERM]
    assert 15.0 not in hooks["slept"]
    persisted = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "interrupted"


def _record_hash_after_terminal(monkeypatch, run_root: Path) -> list[str]:
    hashed: list[str] = []

    def wrapped_validate(path):
        del path
        status = json.loads((run_root / mrl.STATUS_FILENAME).read_text())["status"]
        if status != "training":
            hashed.append("validate")
        raise mrl.ValidationError("test stub: skip torch")

    def wrapped_hash(path):
        del path
        status = json.loads((run_root / mrl.STATUS_FILENAME).read_text())["status"]
        if status != "training":
            hashed.append("hash")
        return "00" * 32

    monkeypatch.setattr(mrl, "validate_local_checkpoint", wrapped_validate)
    monkeypatch.setattr(mrl, "sha256_file", wrapped_hash)
    return hashed


def test_interrupt_uses_sidecar_digest_and_skips_torch_hash(tmp_path, monkeypatch):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    run_root = kwargs["run_root"]
    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir()
    sidecar_digest = "ab" * 32
    (ckpt_dir / mrl.CHECKPOINT_SIDECAR_NAME).write_text(
        json.dumps({
            "sha256": sidecar_digest,
            "size": 13,
            "mtime_ns": 1,
            "validated_at": "2026-08-13T00:00:00+00:00",
        }) + "\n")
    hashed = _record_hash_after_terminal(monkeypatch, run_root)
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, _term = _wait_until_handlers(hooks["installed"], hooks["originals"])
        int_handler(signal.SIGINT, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert hashed == []
    payload = json.loads((run_root / mrl.RESULT_FILENAME).read_text())
    assert payload["status"] == "interrupted"
    assert payload["checkpoint_sha256"] == sidecar_digest


def test_interrupt_without_sidecar_leaves_checkpoint_hash_null(tmp_path, monkeypatch):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    run_root = kwargs["run_root"]
    ckpt_dir = run_root / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "dust2_policy.pt").write_bytes(b"do-not-load-me")
    hashed = _record_hash_after_terminal(monkeypatch, run_root)
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, _term = _wait_until_handlers(hooks["installed"], hooks["originals"])
        int_handler(signal.SIGINT, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert hashed == []
    payload = json.loads((run_root / mrl.RESULT_FILENAME).read_text())
    assert payload["status"] == "interrupted"
    assert payload["checkpoint_sha256"] is None


def test_checkpoint_watcher_stops_before_terminal_status(tmp_path, monkeypatch):
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child, release_on=None)
    watcher_stop: dict[str, threading.Event | None] = {"event": None}
    at_terminal: list[tuple[str, bool]] = []
    real_start = mrl._start_checkpoint_watcher
    real_transition = mrl.transition_status

    def wrapped_start(**kwargs):
        stop, thread = real_start(**kwargs)
        watcher_stop["event"] = stop
        return stop, thread

    def wrapped_transition(run_root, next_status, **kwargs):
        if next_status in mrl.TERMINAL_STATUSES:
            event = watcher_stop["event"]
            at_terminal.append((next_status.value, event is not None and event.is_set()))
        return real_transition(run_root, next_status, **kwargs)

    monkeypatch.setattr(mrl, "_start_checkpoint_watcher", wrapped_start)
    monkeypatch.setattr(mrl, "transition_status", wrapped_transition)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, _term = _wait_until_handlers(hooks["installed"], hooks["originals"])
        int_handler(signal.SIGINT, None)
        child.release()
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert at_terminal == [("interrupted", True)]
    persisted = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "interrupted"


# ── Task 7 cycle D: exit mapping and completion evidence ───────────────────


def test_exit_zero_fails_when_completion_evidence_invalid(tmp_path):
    events: list[str] = []

    class RecordingSink(io.StringIO):

        def close(self):
            events.append("log_closed")
            super().close()

    def commit():
        events.append("commit")

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=FakeChild(returncode=0, stdout=b"done\n"),
            commit=commit,
            log_sink=RecordingSink(),
            manifest=_make_manifest(),
        ))
    result = mrl.execute_training_attempt(**kwargs)
    assert result != mrl.REDELIVERED
    assert result.status is mrl.Status.FAILED
    assert result.reason == mrl.REASON_INVALID_EVIDENCE
    assert result.exit_code == 0
    persisted = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "failed"
    assert "log_closed" in events
    assert "commit" in events[events.index("log_closed") + 1:]


def test_exit_zero_with_valid_evidence_completes(tmp_path):
    run_root, manifest, effective, ckpt = _minimal_completed_tree(tmp_path)
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=FakeChild(returncode=0, stdout=b"done\n"),
            manifest=manifest,
        ))
    result = mrl.execute_training_attempt(**kwargs)
    assert result.status is mrl.Status.COMPLETED
    assert result.reason is None
    assert result.exit_code == 0
    persisted = json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())
    assert persisted["status"] == "completed"
    payload = json.loads((kwargs["run_root"] / "result.json").read_text())
    assert payload["status"] == "completed"
    assert payload["exit_code"] == 0
    assert payload["last_step"] == effective
    assert payload["checkpoint_sha256"] == mrl.sha256_file(ckpt)


def test_dead_run_and_timeout_have_distinct_reasons(tmp_path):
    dead_root = tmp_path / "dead"
    dead_root.mkdir()
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            dead_root,
            child=FakeChild(returncode=3, stdout=b"dead\n"),
            manifest=_make_manifest(),
        ))
    (kwargs["run_root"] / "checkpoints").mkdir(exist_ok=True)
    (kwargs["run_root"] / "checkpoints" / "dust2_policy_dead.pt").write_bytes(b"autopsy")
    dead = mrl.execute_training_attempt(**kwargs)
    assert dead.status is mrl.Status.FAILED
    assert dead.reason == mrl.REASON_DEAD_RUN
    assert dead.exit_code == 3
    assert json.loads((kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())["status"] == "failed"

    clock = _FakeClock()
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    timeout_root = tmp_path / "timeout"
    timeout_root.mkdir()
    kwargs = _consume_training_kwargs(
        _training_kwargs(
            timeout_root,
            child=child,
            now=clock.now,
            timeout=timedelta(minutes=120),
            sleep=hooks["sleep"],
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
            manifest=_make_manifest(),
        ))
    thread, finished, boxed = _run_attempt_in_thread(kwargs)
    try:
        _wait_until_handlers(hooks["installed"], hooks["originals"])
        clock.advance(120 * 60)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    timed_out = boxed[0]
    assert not isinstance(timed_out, Exception), timed_out
    assert timed_out.status is mrl.Status.INTERRUPTED
    assert timed_out.reason == mrl.REASON_TIMEOUT
    assert timed_out.reason != dead.reason
    assert json.loads(
        (kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())["status"] == "interrupted"
    assert hooks["kills"] == [signal.SIGTERM, signal.SIGKILL]


# ── Task 8 cycle A: launch-App import / image / object declarations ────────

PINNED_CUDA_CHILD_DIGEST = (
    "sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7")
PINNED_CUDA_IMAGE = ("nvidia/cuda:12.8.1-devel-ubuntu22.04@" + PINNED_CUDA_CHILD_DIGEST)
PINNED_PUFFERLIB_SDIST = (
    "https://files.pythonhosted.org/packages/7c/e1/5292f9b69c6263707b40ba04a87e6b9bcc177281d31092f77afd90c412f1/"
    "pufferlib-3.0.0.tar.gz#sha256=7df3a3e3f5f894d78d2a1f5374097890aec01473183e748abefe4f3faa10eaa9"
)


class FakeModal:
    """In-process stand-in for the Modal SDK. Importing the app must not call it."""

    def __init__(self):
        self.__version__ = "1.4.3"
        self.apps = []
        self.images = []
        self.volume_creates = []
        self.dict_creates = []
        self.volume_lookups = []
        self.dict_lookups = []
        self.secret_lookups = []
        self.base_remote_calls = []
        self.configured_remote_calls = []
        self.with_options_calls = []
        self.batch_upload_calls = []
        self.read_file_calls = []
        self.iterdir_calls = []
        self.volumes = {}
        self.dicts = {}
        self.known_secrets = set()
        self.invoke_remote = False
        self.Image = FakeImage
        self.Image._fake = self
        self.App = self._app_type()
        self.Volume = self._volume_type()
        self.Dict = self._dict_type()
        self.Secret = self._secret_type()

    def as_module(self) -> SimpleNamespace:
        return SimpleNamespace(
            Image=self.Image,
            App=self.App,
            Volume=self.Volume,
            Dict=self.Dict,
            Secret=self.Secret,
            exception=SimpleNamespace(NotFoundError=FakeNotFoundError),
            NotFoundError=FakeNotFoundError,
            __version__=self.__version__,
        )

    def _app_type(self):
        fake = self

        class App:

            def __init__(self, name: str, include_source=None, **kwargs):
                del kwargs
                self.name = name
                self.include_source = include_source
                self.app_id = "ap-ephemeral-test"
                self.functions: dict[str, object] = {}
                self.entrypoints: dict[str, object] = {}
                fake.apps.append(self)

            def function(self, **kwargs):

                def decorator(fn):
                    bound = FakeFunction(fake, fn, kwargs)
                    self.functions[fn.__name__] = bound
                    return bound

                return decorator

            def local_entrypoint(self, *args, **kwargs):
                del args, kwargs

                def decorator(fn):
                    self.entrypoints[fn.__name__] = fn
                    return fn

                return decorator

        return App

    def _volume_type(self):
        fake = self

        class Volume:

            class objects:

                @staticmethod
                def create(name: str, allow_existing: bool = False, **kwargs):
                    del kwargs
                    fake.volume_creates.append((name, allow_existing))
                    if name in fake.volumes and not allow_existing:
                        raise FileExistsError(name)
                    fake.volumes.setdefault(name, FakeVolume(fake, name))

            @staticmethod
            def from_name(name: str, create_if_missing: bool = False):
                fake.volume_lookups.append((name, create_if_missing))
                if name not in fake.volumes:
                    if create_if_missing:
                        fake.volumes[name] = FakeVolume(fake, name)
                    else:
                        raise FakeNotFoundError(f"Volume {name!r} not found")
                return fake.volumes[name]

        return Volume

    def _dict_type(self):
        fake = self

        class Dict:

            class objects:

                @staticmethod
                def create(name: str, allow_existing: bool = False, **kwargs):
                    del kwargs
                    fake.dict_creates.append((name, allow_existing))
                    if name in fake.dicts and not allow_existing:
                        raise FileExistsError(name)
                    fake.dicts.setdefault(name, FakeDict(fake, name))

            @staticmethod
            def from_name(name: str, create_if_missing: bool = False):
                fake.dict_lookups.append((name, create_if_missing))
                if name not in fake.dicts:
                    if create_if_missing:
                        fake.dicts[name] = FakeDict(fake, name)
                    else:
                        raise FakeNotFoundError(f"Dict {name!r} not found")
                return fake.dicts[name]

        return Dict

    def _secret_type(self):
        fake = self

        class Secret:

            def __init__(self, name: str):
                self.name = name

            def __repr__(self) -> str:
                return "Secret(<redacted>)"

            @staticmethod
            def from_name(name: str, **kwargs):
                del kwargs
                fake.secret_lookups.append(name)
                if name not in fake.known_secrets:
                    raise FakeNotFoundError("requested W&B Secret is missing")
                return Secret(name)

        return Secret


class FakeNotFoundError(Exception):
    """Stand-in for a missing named Modal object."""


class FakeImage:
    _fake: FakeModal | None = None

    def __init__(self):
        self.registry_tag: str | None = None
        self.add_python: str | None = None
        self.apt: list[str] = []
        self.pips: list[str] = []
        self.env_vars: dict[str, str] = {}
        self.local_files: list[tuple[str, str, bool]] = []
        self.commands: list[str] = []

    @classmethod
    def from_registry(cls, tag: str, add_python: str | None = None, **kwargs):
        del kwargs
        image = cls()
        image.registry_tag = tag
        image.add_python = add_python
        if cls._fake is not None:
            cls._fake.images.append(image)
        return image

    def apt_install(self, *packages: str):
        self.apt.extend(packages)
        return self

    def pip_install(self, *packages: str):
        self.pips.extend(packages)
        return self

    def env(self, mapping: dict[str, str]):
        self.env_vars.update(mapping)
        return self

    def add_local_file(self, src: str, dst: str, copy: bool = False):
        self.local_files.append((src, dst, copy))
        return self

    def run_commands(self, *commands: str):
        self.commands.extend(commands)
        return self


class FakeFunction:
    """Decorated Function: base .remote is forbidden; with_options is the only path."""

    def __init__(self, fake: FakeModal, fn, kwargs: dict[str, object]):
        self._fake = fake
        self._fn = fn
        self.kwargs = kwargs
        self.__name__ = fn.__name__

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def remote(self, *args, **kwargs):
        self._fake.base_remote_calls.append((args, kwargs))
        raise AssertionError("base Function must never be called")

    def with_options(self, **kwargs):
        self._fake.with_options_calls.append(dict(kwargs))
        return FakeConfiguredFunction(self._fake, self, kwargs)


class FakeConfiguredFunction:

    def __init__(self, fake: FakeModal, base: FakeFunction, options: dict[str, object]):
        self._fake = fake
        self.base = base
        self.options = options

    def remote(self, *args, **kwargs):
        self._fake.configured_remote_calls.append((self.options, args, kwargs))
        if self._fake.invoke_remote:
            return self.base._fn(*args, **kwargs)
        return {"status": "ok"}


class FakeVolume:

    def __init__(self, fake: FakeModal, name: str):
        self._fake = fake
        self.name = name
        self.files: dict[str, bytes] = {}
        self.pending_creates: dict[str, bytes] = {}
        self.replace_after_read: dict[str, bytes] = {}
        self.commit_count = 0
        self.reject_next_upload = False
        self.iterdir_entries: list[object] | None = None
        self.missing_prefix_exc: type[BaseException] | None = None
        self.fail_prefix: str | None = None

    def _client_path(self, path) -> str:
        text = str(path)
        if text.startswith("/artifacts"):
            raise AssertionError(f"/artifacts leaked to Volume client API: {text}")
        return text

    def batch_upload(self, force: bool = False):
        return FakeBatchUpload(self, force)

    def read_file(self, path):
        key = self._client_path(path)
        self._fake.read_file_calls.append(key)
        if key not in self.files:
            raise FileNotFoundError(key)
        data = self.files[key]
        if key in self.replace_after_read:
            self.files[key] = self.replace_after_read.pop(key)
        yield data

    def iterdir(self, path, *, recursive: bool = True):
        key = self._client_path(path)
        self._fake.iterdir_calls.append((key, recursive))
        if self.iterdir_entries is not None:
            yield from self.iterdir_entries
            return
        prefix = key.rstrip("/")
        matched = False
        for stored in sorted(self.files):
            if prefix == "" or stored == prefix or stored.startswith(prefix + "/"):
                matched = True
                yield SimpleNamespace(path=stored, type="file")
        if not matched and self.missing_prefix_exc is not None:
            raise self.missing_prefix_exc(key)

    def commit(self):
        self.commit_count += 1

    def reload(self):
        return None


class FakeBatchUpload:

    def __init__(self, volume: FakeVolume, force: bool):
        self.volume = volume
        self.force = force
        self.puts: list[tuple[str, str]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def put_file(self, local_path, remote_path):
        key = self.volume._client_path(remote_path)
        self.volume._fake.batch_upload_calls.append((key, self.force, str(local_path)))
        if self.force:
            raise AssertionError("force=True is never used")
        if key in self.volume.pending_creates:
            self.volume.files[key] = self.volume.pending_creates.pop(key)
            raise FileExistsError(key)
        if self.volume.reject_next_upload:
            self.volume.reject_next_upload = False
            raise FileExistsError(key)
        if self.volume.fail_prefix is not None and key.startswith(self.volume.fail_prefix):
            raise OSError(f"could not upload {key}")
        if key in self.volume.files:
            raise FileExistsError(key)
        data = Path(local_path).read_bytes() if not hasattr(local_path,
                                                            "read") else local_path.read()
        self.volume.files[key] = data
        self.puts.append((str(local_path), key))


class FakeDict:

    def __init__(self, fake: FakeModal, name: str):
        self._fake = fake
        self.name = name
        self.data: dict[str, object] = {}

    def put(self, key: str, value, *, skip_if_exists: bool = False) -> bool:
        if skip_if_exists and key in self.data:
            return False
        self.data[key] = value
        return True

    def get(self, key: str):
        if key not in self.data:
            raise KeyError(key)
        return self.data[key]


@pytest.fixture
def fake_modal():
    fake = FakeModal()
    previous = sys.modules.get("modal")
    sys.modules["modal"] = fake.as_module()
    for name in ("scripts.run_modal", "scripts.modal_artifacts"):
        sys.modules.pop(name, None)
    try:
        yield fake
    finally:
        for name in ("scripts.run_modal", "scripts.modal_artifacts"):
            sys.modules.pop(name, None)
        if previous is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = previous


def _import_run_modal():
    return importlib.import_module("scripts.run_modal")


def test_importing_app_creates_no_function_call_or_gpu_work(fake_modal):
    module = _import_run_modal()
    assert fake_modal.base_remote_calls == []
    assert fake_modal.configured_remote_calls == []
    assert fake_modal.with_options_calls == []
    assert fake_modal.volume_creates == []
    assert fake_modal.dict_creates == []
    assert fake_modal.volume_lookups == []
    assert fake_modal.dict_lookups == []
    assert fake_modal.secret_lookups == []
    assert module.app.name == "cs2rl-training"
    assert "gpu" not in module.train_remote.kwargs or module.train_remote.kwargs["gpu"] is None


def test_base_function_has_no_static_named_object_dependency(fake_modal):
    module = _import_run_modal()
    kwargs = module.train_remote.kwargs
    assert kwargs.get("volumes") in (None, {})
    assert "volumes" not in kwargs or not kwargs["volumes"]
    assert kwargs.get("secrets") in (None, [])
    assert kwargs.get("retries") == 0
    assert kwargs.get("single_use_containers") is True
    assert module.app.include_source is False
    assert kwargs.get("include_source") is False
    assert module.app.name == "cs2rl-training"
    assert "main" in module.app.entrypoints
    assert mrl.VOLUME_NAME == "cs2rl-training-artifacts"
    assert mrl.REGISTRY_NAME == "cs2rl-training-run-registry"


def _run_modal_image_reqs(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "modal_image_reqs.py"), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def _requirement_names(text: str) -> list[str]:
    pins = [line for line in text.splitlines() if line.strip()]
    assert pins
    assert all("==" in line for line in pins)
    return [line.split("==", 1)[0] for line in pins]


def test_modal_image_reqs_from_repo_lock_includes_torch_numpy_not_pufferlib_or_modal():
    result = _run_modal_image_reqs(str(ROOT / "uv.lock"))
    names = _requirement_names(result.stdout)
    assert names == sorted(names)
    assert "torch" in names
    assert "numpy" in names
    assert "pufferlib" not in names
    assert "cs2rl" not in names
    assert "modal" not in names


def test_modal_image_reqs_writes_minus_o(tmp_path):
    out = tmp_path / "cs2rl-reqs.txt"
    result = _run_modal_image_reqs(str(ROOT / "uv.lock"), "-o", str(out))
    assert result.stdout == ""
    names = _requirement_names(out.read_text())
    assert names == sorted(names)
    assert "torch" in names
    assert "numpy" in names
    assert "pufferlib" not in names
    assert "modal" not in names


def test_modal_image_reqs_walks_runtime_graph_not_dev_or_modal_groups(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text("""\
version = 1
[[package]]
name = "cs2rl"
version = "0.1.0"
dependencies = [
    { name = "numpy" },
    { name = "pufferlib" },
]

[package.dev-dependencies]
dev = [
    { name = "ruff" },
]
modal = [
    { name = "modal" },
]

[[package]]
name = "numpy"
version = "2.4.3"

[[package]]
name = "pufferlib"
version = "3.0.0"
dependencies = [
    { name = "torch" },
    { name = "shimmy", extra = ["gym-v21"] },
]

[[package]]
name = "torch"
version = "2.10.0"

[[package]]
name = "shimmy"
version = "1.3.0"

[package.optional-dependencies]
gym-v21 = [
    { name = "pyglet" },
]

[[package]]
name = "pyglet"
version = "2.1.0"

[[package]]
name = "modal"
version = "1.5.4"

[[package]]
name = "ruff"
version = "0.11.13"
""")
    result = _run_modal_image_reqs(str(lock))
    assert result.stdout == "numpy==2.4.3\npyglet==2.1.0\nshimmy==1.3.0\ntorch==2.10.0\n"


def test_image_pins_cuda_digest_arch_list_and_hashed_pufferlib_sdist(fake_modal):
    module = _import_run_modal()
    image = module.dependency_image
    assert image.registry_tag == PINNED_CUDA_IMAGE
    assert image.add_python == "3.12"
    assert image.env_vars["TORCH_CUDA_ARCH_LIST"] == "7.5;8.6;8.9"
    assert image.env_vars["NO_OCEAN"] == "1"
    assert "uv==0.11.1" in image.pips
    assert "ziglang==0.14.1" in image.pips
    assert (str(ROOT / "pyproject.toml"), "/opt/cs2rl/pyproject.toml", True) in image.local_files
    assert (str(ROOT / "uv.lock"), "/opt/cs2rl/uv.lock", True) in image.local_files
    assert (str(ROOT / "scripts" / "modal_image_reqs.py"), "/opt/cs2rl/modal_image_reqs.py",
            True) in image.local_files
    commands = "\n".join(image.commands)
    assert "modal_image_reqs.py" in commands
    assert "uv pip install" in commands
    assert "-r" in commands
    locked_dep_installs = [
        part.strip() for command in image.commands for part in command.split("&&")
        if "uv pip install" in part and "-r" in part
    ]
    assert locked_dep_installs
    assert all("--directory /tmp" in cmd for cmd in locked_dep_installs)
    locked_dep_command = " && ".join(locked_dep_installs)
    assert "/tmp/cs2rl-reqs.txt" in locked_dep_command
    assert "--no-deps" in locked_dep_command
    assert "pufferlib" not in locked_dep_command
    assert "uv export" not in locked_dep_command
    assert "uv sync" not in locked_dep_command
    assert "uv export" not in commands
    assert "uv sync" not in commands
    assert "--no-build-isolation" in commands
    assert "--no-deps" in commands
    assert "--no-binary pufferlib" in commands
    assert PINNED_PUFFERLIB_SDIST in commands
    assert "c_extension_paths = []" in commands
    hashed_sdist_installs = [
        part.strip() for command in image.commands for part in command.split("&&")
        if "uv pip install" in part and "--no-binary pufferlib" in part
    ]
    assert hashed_sdist_installs
    assert all("pufferlib" in cmd for cmd in hashed_sdist_installs)
    assert all("/tmp/pufferlib-3.0.0" in cmd for cmd in hashed_sdist_installs)
    assert all("CXX=g++" in cmd for cmd in hashed_sdist_installs)
    assert "Python.h" in commands
    assert "release 12.8" in commands
    assert "pufferlib._C" in commands
    assert "compute_puff_advantage" in commands
    assert "all('sm_'+arch in elf for arch in ('75','86','89'))" in commands
    runner = module.runner_image
    assert (str(ROOT / "scripts" / "modal_runner_lib.py"), "/opt/app/scripts/modal_runner_lib.py",
            True) in runner.local_files
    assert (str(ROOT / "scripts" / "run_modal.py"), "/opt/app/scripts/run_modal.py",
            True) in runner.local_files
    assert runner.env_vars["PYTHONPATH"] == "/opt/app"
    for src, _dst, _copy in (*image.local_files, *runner.local_files):
        assert Path(src).is_absolute()
        assert Path(src).is_relative_to(ROOT)


# ── Task 8 cycle B: run-only parser / omitted sentinels ────────────────────


def _launch_sentinels(**overrides):
    """Every launch option starts as None; callers supply only explicit values."""
    kwargs = {
        "action": None,
        "run_id": None,
        "git_sha": None,
        "map": None,
        "gpu": None,
        "cpu_cores": None,
        "memory_mib": None,
        "num_envs": None,
        "vec_workers": None,
        "timeout_minutes": None,
        "save_every_seconds": None,
        "train_args": None,
        "resume_local_checkpoint": None,
        "resume_run_id": None,
        "wandb_secret_name": None,
    }
    kwargs.update(overrides)
    return kwargs


def _valid_launch_sentinels(**overrides):
    kwargs = _launch_sentinels(
        action="run",
        run_id="140826-b7r-seed2-shared",
        git_sha="a" * 40,
        map="simple",
        train_args="--timesteps 30000000 --seed 2",
    )
    kwargs.update(overrides)
    return kwargs


def test_status_and_download_are_not_app_actions(fake_modal):
    module = _import_run_modal()
    assert list(module.app.entrypoints) == ["main"]
    for action in ("status", "download"):
        with pytest.raises(mrl.ValidationError):
            module.resolve_launch_request(**_valid_launch_sentinels(action=action))
    with pytest.raises(mrl.ValidationError):
        module.resolve_launch_request(**_valid_launch_sentinels(action=None))
    with pytest.raises(mrl.ValidationError):
        module.resolve_launch_request(**_valid_launch_sentinels(action="train"))


def test_omitted_gpu_defaults_to_t4_invalid_explicit_gpu_rejected(fake_modal):
    module = _import_run_modal()
    request = module.resolve_launch_request(**_valid_launch_sentinels(gpu=None))
    assert request.gpu == mrl.DEFAULT_GPU == "T4"
    for gpu in ("T4", "L4", "A10"):
        assert module.resolve_launch_request(**_valid_launch_sentinels(gpu=gpu)).gpu == gpu
    for gpu in ("A10G", "A100", "H100", "any", "T4,L4", "t4", "T4:2", "T4;L4"):
        with pytest.raises(mrl.ValidationError):
            module.resolve_launch_request(**_valid_launch_sentinels(gpu=gpu))


def test_map_has_no_default_and_must_be_allowlisted(fake_modal):
    module = _import_run_modal()
    with pytest.raises(mrl.ValidationError):
        module.resolve_launch_request(**_valid_launch_sentinels(map=None))
    for effective_map in ("simple", "dust2"):
        request = module.resolve_launch_request(**_valid_launch_sentinels(map=effective_map))
        assert request.effective_map == effective_map
    for effective_map in ("", "cs2-dust2", "DUST2", "dust"):
        with pytest.raises(mrl.ValidationError):
            module.resolve_launch_request(**_valid_launch_sentinels(map=effective_map))


def test_omitted_resource_sentinels_apply_defaults_and_smoke_values_pass(fake_modal):
    module = _import_run_modal()
    request = module.resolve_launch_request(**_valid_launch_sentinels())
    assert request.cpu_cores == 8
    assert request.memory_mib == 16384
    assert request.cpu_request_limit == (8, 8)
    assert request.memory_request_limit == (16384, 16384)
    assert request.num_envs == 256
    assert request.vec_workers == 8
    assert request.timeout_minutes == 120
    assert request.save_every_seconds == 300
    smoke = module.resolve_launch_request(**_valid_launch_sentinels(
        cpu_cores=4,
        memory_mib=8192,
        vec_workers=4,
        timeout_minutes=15,
        train_args="--timesteps 163840 --seed 2",
    ))
    assert smoke.cpu_request_limit == (4, 4)
    assert smoke.memory_request_limit == (8192, 8192)
    assert smoke.vec_workers == 4
    assert smoke.timeout_minutes == 15


# ── Task 8 cycle C: Volume namespace + reservation/blob adapters ───────────


def _named_volume(fake_modal, name=mrl.VOLUME_NAME):
    fake_modal.Volume.objects.create(name, allow_existing=True)
    return fake_modal.Volume.from_name(name, create_if_missing=False)


def _named_dict(fake_modal, name=mrl.REGISTRY_NAME):
    fake_modal.Dict.objects.create(name, allow_existing=True)
    return fake_modal.Dict.from_name(name, create_if_missing=False)


def test_volume_adapter_uses_root_relative_client_paths(fake_modal):
    module = _import_run_modal()
    volume = _named_volume(fake_modal)
    artifacts = module.ModalVolumeIndex(volume)
    reservation = mrl.RUNS_ROOT / "ok-id" / mrl.RESERVATION_FILENAME
    artifacts.put_file(reservation, b'{"attempt_id":"a"}\n')
    artifacts.commit()
    assert artifacts.exists(reservation)
    assert reservation.as_posix() in volume.files
    assert all(not path.startswith("/artifacts") for path in volume.files)
    assert all(not remote.startswith("/artifacts")
               for remote, _force, _local in fake_modal.batch_upload_calls)
    source_client = mrl.SOURCES_ROOT / "deadbeef.tar.gz"
    assert mrl.mounted_path(source_client) == Path("/artifacts/sources/deadbeef.tar.gz")
    ckpt_client = mrl.INPUTS_ROOT / "sha256" / "abcd.pt"
    assert mrl.mounted_path(ckpt_client) == Path("/artifacts/inputs/sha256/abcd.pt")


def test_volume_adapter_commit_does_not_call_client_volume_commit(fake_modal):
    module = _import_run_modal()
    volume = _named_volume(fake_modal)
    commit_calls = {"count": 0}

    def raising_commit():
        commit_calls["count"] += 1
        raise RuntimeError("commit() can only be called on a mounted volume inside a container")

    volume.commit = raising_commit
    artifacts = module.ModalVolumeIndex(volume)
    reservation = mrl.RUNS_ROOT / "ok-id" / mrl.RESERVATION_FILENAME
    artifacts.put_file(reservation, b'{"attempt_id":"a"}\n')
    artifacts.commit()
    assert artifacts.exists(reservation)
    assert volume.files[reservation.as_posix()] == b'{"attempt_id":"a"}\n'
    assert artifacts._staged == []
    assert all(force is False for _path, force, _local in fake_modal.batch_upload_calls)
    assert all(not remote.startswith("/artifacts")
               for remote, _force, _local in fake_modal.batch_upload_calls)
    assert commit_calls["count"] == 0


def test_ensure_blob_uploads_missing_and_reuses_after_streamed_verify(fake_modal, tmp_path):
    module = _import_run_modal()
    volume = _named_volume(fake_modal)
    blob = tmp_path / "src.tar.gz"
    blob.write_bytes(b"source-bytes")
    digest = mrl.sha256_file(blob)
    client_path = mrl.SOURCES_ROOT / f"{digest}.tar.gz"
    module.ensure_blob(volume, client_path, blob)
    assert fake_modal.batch_upload_calls == [(client_path.as_posix(), False, str(blob))]
    assert volume.files[client_path.as_posix()] == b"source-bytes"
    assert mrl.mounted_path(client_path) == Path("/artifacts/sources") / f"{digest}.tar.gz"

    fake_modal.batch_upload_calls.clear()
    module.ensure_blob(volume, client_path, blob)
    assert fake_modal.batch_upload_calls == []
    assert fake_modal.read_file_calls[-1] == client_path.as_posix()


def test_ensure_blob_handles_concurrent_create_and_rejects_mismatch(fake_modal, tmp_path):
    module = _import_run_modal()
    volume = _named_volume(fake_modal)
    blob = tmp_path / "warm.pt"
    blob.write_bytes(b"ckpt-bytes")
    digest = mrl.sha256_file(blob)
    client_path = mrl.INPUTS_ROOT / "sha256" / f"{digest}.pt"
    volume.pending_creates[client_path.as_posix()] = b"ckpt-bytes"
    module.ensure_blob(volume, client_path, blob)
    assert mrl.mounted_path(client_path) == Path("/artifacts/inputs/sha256") / f"{digest}.pt"
    volume.files[client_path.as_posix()] = b"other-bytes"
    with pytest.raises(mrl.ValidationError):
        module.ensure_blob(volume, client_path, blob)
    assert all(force is False for _path, force, _local in fake_modal.batch_upload_calls)


def test_reserve_run_through_modal_adapters_stays_in_client_namespace(fake_modal):
    module = _import_run_modal()
    volume = _named_volume(fake_modal)
    registry = module.ModalDictRegistry(_named_dict(fake_modal))
    artifacts = module.ModalVolumeIndex(volume)
    mrl.reserve_run(registry, artifacts, "ok-id", "attempt-a", now=_aware())
    reservation = mrl.RUNS_ROOT / "ok-id" / mrl.RESERVATION_FILENAME
    assert reservation.as_posix() in volume.files
    assert all(not path.startswith("/artifacts") for path in volume.files)
    assert registry.get(mrl.run_registry_key("ok-id"))["attempt_id"] == "attempt-a"
    assert fake_modal.volume_creates == [(mrl.VOLUME_NAME, True)]
    assert fake_modal.dict_creates == [(mrl.REGISTRY_NAME, True)]
    assert fake_modal.volume_lookups == [(mrl.VOLUME_NAME, False)]
    assert fake_modal.dict_lookups == [(mrl.REGISTRY_NAME, False)]


# ── Task 8 cycle D: configured run invocation and W&B gating ───────────────


def _capture_stdout():
    return io.StringIO()


def _write_parent_artifacts(volume, parent_id, *, status, updated_at, ckpt_bytes, sidecar):
    status_path = (mrl.RUNS_ROOT / parent_id / mrl.STATUS_FILENAME).as_posix()
    ckpt_path = (mrl.RUNS_ROOT / parent_id / "checkpoints" / mrl.CHECKPOINT_NAME).as_posix()
    sidecar_path = (mrl.RUNS_ROOT / parent_id / "checkpoints" /
                    mrl.CHECKPOINT_SIDECAR_NAME).as_posix()
    volume.files[status_path] = json.dumps({
        "schema_version": 1,
        "status": status,
        "attempt_id": "parent-attempt",
        "updated_at": updated_at,
    }).encode()
    volume.files[ckpt_path] = ckpt_bytes
    if sidecar is not None:
        volume.files[sidecar_path] = json.dumps(sidecar).encode()
    return ckpt_path, sidecar_path


def test_invalid_inputs_validate_before_claim_or_gpu(fake_modal, tmp_path):
    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    junk = tmp_path / "nope.pt"
    junk.write_text("not-a-checkpoint")
    request = module.resolve_launch_request(
        **_valid_launch_sentinels(git_sha=sha, resume_local_checkpoint=str(junk)))
    with pytest.raises(mrl.ValidationError):
        module.launch_run(request, repo=repo, app_obj=module.app)
    assert fake_modal.volume_creates == []
    assert fake_modal.dict_creates == []
    assert fake_modal.configured_remote_calls == []
    assert fake_modal.base_remote_calls == []


def test_configured_run_uses_with_options_defaults_and_prints_ids(fake_modal, tmp_path):
    import torch

    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    ckpt = tmp_path / "warm.pt"
    torch.save({"weight": torch.tensor([1.0])}, ckpt)
    digest = mrl.sha256_file(ckpt)
    request = module.resolve_launch_request(
        **_valid_launch_sentinels(git_sha=sha, resume_local_checkpoint=str(ckpt)))
    stdout = _capture_stdout()
    result = module.launch_run(request, repo=repo, app_obj=module.app, stdout=stdout)
    assert result["status"] == "ok"
    assert fake_modal.base_remote_calls == []
    assert len(fake_modal.with_options_calls) == 1
    options = fake_modal.with_options_calls[0]
    assert options["gpu"] == "T4"
    assert options["cpu"] == (8, 8)
    assert options["memory"] == (16384, 16384)
    assert options["timeout"] == 120 * 60
    assert list(options["volumes"]) == ["/artifacts"]
    assert "secrets" not in options
    _opts, args, kwargs = fake_modal.configured_remote_calls[0]
    payload = args[0] if args else kwargs["payload"]
    assert payload["run_id"] == request.run_id
    assert payload["resume_mount_path"] == f"/artifacts/inputs/sha256/{digest}.pt"
    assert payload["resume_sha256"] == digest
    assert "wandb_enabled" not in payload
    assert "wandb_secret_name" not in payload
    dumped = json.dumps(payload)
    assert "wandb" not in dumped
    assert all(
        isinstance(value, (str, int, float, bool, list, type(None))) for value in payload.values())
    printed = stdout.getvalue()
    assert module.app.app_id in printed
    assert request.run_id in printed
    source_name = next(path for path in fake_modal.volumes[mrl.VOLUME_NAME].files
                       if path.startswith("sources/"))
    assert source_name.endswith(".tar.gz")
    assert not source_name.startswith("/artifacts")
    assert f"inputs/sha256/{digest}.pt" in fake_modal.volumes[mrl.VOLUME_NAME].files


def test_smoke_resource_tuples_and_cpu_memory_semantics(fake_modal, tmp_path):
    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    request = module.resolve_launch_request(**_valid_launch_sentinels(
        git_sha=sha,
        cpu_cores=4,
        memory_mib=8192,
        vec_workers=4,
        timeout_minutes=15,
        train_args="--timesteps 163840 --seed 2",
    ))
    # CPU tuple is a soft throttling limit; memory tuple is a hard OOM limit.
    assert request.cpu_request_limit == (4, 4)
    assert request.memory_request_limit == (8192, 8192)
    module.launch_run(request, repo=repo, app_obj=module.app, stdout=_capture_stdout())
    options = fake_modal.with_options_calls[0]
    assert options["cpu"] == (4, 4)
    assert options["memory"] == (8192, 8192)
    assert options["timeout"] == 15 * 60


def test_wandb_secret_missing_fails_before_claim_without_leaking_name(fake_modal, tmp_path):
    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    secret_name = "prod-wandb-key"
    request = module.resolve_launch_request(**_valid_launch_sentinels(
        git_sha=sha,
        train_args="--timesteps 163840 --seed 2 --wandb",
        wandb_secret_name=secret_name,
    ))
    with pytest.raises(mrl.ValidationError) as excinfo:
        module.launch_run(request, repo=repo, app_obj=module.app)
    assert secret_name not in str(excinfo.value)
    assert fake_modal.volume_creates == []
    assert fake_modal.dict_creates == []
    assert fake_modal.configured_remote_calls == []
    assert fake_modal.secret_lookups == [secret_name]


def test_wandb_attaches_secret_and_records_enabled_flag_only(fake_modal, tmp_path):
    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    secret_name = "prod-wandb-key"
    fake_modal.known_secrets.add(secret_name)
    request = module.resolve_launch_request(**_valid_launch_sentinels(
        git_sha=sha,
        train_args="--timesteps 163840 --seed 2 --wandb",
        wandb_secret_name=secret_name,
    ))
    stdout = _capture_stdout()
    module.launch_run(request, repo=repo, app_obj=module.app, stdout=stdout)
    options = fake_modal.with_options_calls[0]
    assert "secrets" in options
    assert len(options["secrets"]) == 1
    payload = fake_modal.configured_remote_calls[0][1][0]
    assert payload["wandb_enabled"] is True
    assert "wandb_secret_name" not in payload
    assert secret_name not in json.dumps(payload)
    assert secret_name not in stdout.getvalue()


@pytest.mark.parametrize(
    "case",
    ["active", "missing", "stale", "mismatch", "replaced"],
)
def test_prior_run_resume_fails_closed_without_consuming_new_id(fake_modal, tmp_path, case):
    import torch

    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    ckpt = tmp_path / "parent.pt"
    torch.save({"weight": torch.tensor([3.0])}, ckpt)
    ckpt_bytes = ckpt.read_bytes()
    digest = mrl.sha256_file(ckpt)
    volume = _named_volume(fake_modal)
    sidecar = {
        "sha256": digest,
        "size": len(ckpt_bytes),
        "mtime_ns": 1,
        "validated_at": _aware().isoformat(),
    }
    status = "completed" if case != "active" else "training"
    _ckpt_path, sidecar_path = _write_parent_artifacts(
        volume,
        "parent-run",
        status=status,
        updated_at=_aware().isoformat(),
        ckpt_bytes=ckpt_bytes,
        sidecar=None if case == "missing" else sidecar,
    )
    if case == "stale":
        volume.files[sidecar_path] = json.dumps({
            **sidecar,
            "size": len(ckpt_bytes) + 1,
            "mtime_ns": 99,
        }).encode()
    elif case == "mismatch":
        volume.files[sidecar_path] = json.dumps({**sidecar, "sha256": "0" * 64}).encode()
    elif case == "replaced":
        volume.replace_after_read = {
            sidecar_path: json.dumps({
                **sidecar, "sha256": "1" * 64
            }).encode()
        }
    request = module.resolve_launch_request(
        **_valid_launch_sentinels(git_sha=sha, run_id="child-run", resume_run_id="parent-run"))
    creates_before = list(fake_modal.volume_creates)
    dicts_before = list(fake_modal.dict_creates)
    with pytest.raises(mrl.ValidationError):
        module.launch_run(request, repo=repo, app_obj=module.app, now=_aware())
    assert fake_modal.volume_creates == creates_before
    assert fake_modal.dict_creates == dicts_before
    assert fake_modal.configured_remote_calls == []
    assert fake_modal.dicts.get(mrl.REGISTRY_NAME) is None


def test_prior_run_resume_sends_only_immutable_digest_path(fake_modal, tmp_path):
    import torch

    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    ckpt = tmp_path / "parent.pt"
    torch.save({"weight": torch.tensor([4.0])}, ckpt)
    ckpt_bytes = ckpt.read_bytes()
    digest = mrl.sha256_file(ckpt)
    volume = _named_volume(fake_modal)
    _write_parent_artifacts(
        volume,
        "parent-run",
        status="completed",
        updated_at=_aware().isoformat(),
        ckpt_bytes=ckpt_bytes,
        sidecar={
            "sha256": digest,
            "size": len(ckpt_bytes),
            "mtime_ns": 1,
            "validated_at": _aware().isoformat(),
        },
    )
    request = module.resolve_launch_request(
        **_valid_launch_sentinels(git_sha=sha, run_id="child-run", resume_run_id="parent-run"))
    module.launch_run(request,
                      repo=repo,
                      app_obj=module.app,
                      now=_aware(),
                      stdout=_capture_stdout())
    payload = fake_modal.configured_remote_calls[0][1][0]
    assert payload["resume_mount_path"] == f"/artifacts/inputs/sha256/{digest}.pt"
    assert payload["resume_sha256"] == digest
    assert payload["resumed_from_run_id"] == "parent-run"
    assert module.build_remote_manifest(payload).resumed_from_run_id == "parent-run"
    assert "runs/parent-run" not in payload["resume_mount_path"]
    assert f"inputs/sha256/{digest}.pt" in fake_modal.volumes[mrl.VOLUME_NAME].files
    assert fake_modal.volumes[mrl.VOLUME_NAME].files[f"inputs/sha256/{digest}.pt"] == ckpt_bytes


def _expected_thread_caps():
    return [f"{key}={value}" for key, value in sorted(mrl._THREAD_CAP_ENV.items())]


def test_launch_payload_includes_design_contract_fields(fake_modal, tmp_path):
    import torch

    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    ckpt = tmp_path / "warm.pt"
    torch.save({"weight": torch.tensor([1.0])}, ckpt)
    digest = mrl.sha256_file(ckpt)
    request = module.resolve_launch_request(
        **_valid_launch_sentinels(git_sha=sha, resume_local_checkpoint=str(ckpt)))
    module.launch_run(request,
                      repo=repo,
                      app_obj=module.app,
                      now=_aware(),
                      stdout=_capture_stdout())
    payload = fake_modal.configured_remote_calls[0][1][0]
    run_root = mrl.mounted_path(mrl.RUNS_ROOT / request.run_id)
    resume_mount = f"/artifacts/inputs/sha256/{digest}.pt"
    requested = request.timesteps
    batch_size = request.batch_size
    effective = (requested // batch_size) * batch_size
    assert payload["training_argv"] == request.training_argv(run_root, resume_mount)
    assert payload["requested_timesteps"] == requested
    assert payload["effective_timesteps"] == effective
    assert payload["batch_size"] == batch_size
    assert payload["seed"] == 2
    assert payload["created_at"] == _aware().isoformat()
    assert payload["resume_sha256"] == digest
    assert payload["resume_size"] == ckpt.stat().st_size
    assert payload["resume_source_path"] == resume_mount
    assert payload["runner_commit"] == sha
    assert payload["config_hash"] == "0" * 64
    assert payload["modal_version"] == fake_modal.__version__
    assert payload["image_digest"] == PINNED_CUDA_CHILD_DIGEST
    assert payload["effective_map"] == "simple"
    assert payload["gpu"] == "T4"
    assert payload["cpu_request"] == payload["cpu_soft_limit"] == 8
    assert payload["memory_request_mib"] == payload["memory_hard_limit_mib"] == 16384
    assert payload["vec_workers"] == 8
    assert payload["thread_caps"] == _expected_thread_caps()
    assert payload["resumed_from_run_id"] is None
    assert all(
        isinstance(value, (str, int, float, bool, list, type(None))) for value in payload.values())


def test_build_remote_manifest_records_contract_and_rejects_digest_drift(fake_modal, tmp_path):
    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    request = module.resolve_launch_request(**_valid_launch_sentinels(git_sha=sha))
    module.launch_run(request,
                      repo=repo,
                      app_obj=module.app,
                      now=_aware(),
                      stdout=_capture_stdout())
    payload = fake_modal.configured_remote_calls[0][1][0]
    manifest = module.build_remote_manifest(payload)
    assert manifest.modal_version == fake_modal.__version__
    assert manifest.image_digest == PINNED_CUDA_CHILD_DIGEST
    assert manifest.effective_map == "simple"
    assert manifest.gpu == "T4"
    assert manifest.cpu_request == manifest.cpu_soft_limit == 8
    assert manifest.memory_request_mib == manifest.memory_hard_limit_mib == 16384
    assert manifest.vec_workers == 8
    assert manifest.training_argv == payload["training_argv"]
    assert manifest.requested_timesteps == request.timesteps
    assert manifest.effective_timesteps == (request.timesteps //
                                            request.batch_size) * request.batch_size
    assert manifest.batch_size == request.batch_size
    assert manifest.seed == 2
    assert manifest.created_at == _aware().isoformat()
    assert manifest.resume_sha256 is None
    assert manifest.resume_size is None
    assert manifest.resume_source_path is None
    assert manifest.runner_commit == sha
    assert manifest.config_hash == "0" * 64
    assert manifest.thread_caps == _expected_thread_caps()
    assert manifest.resumed_from_run_id is None
    assert manifest.commit == sha
    drifted = dict(payload)
    drifted["image_digest"] = "sha256:" + "0" * 64
    with pytest.raises(mrl.ValidationError, match="digest"):
        module.build_remote_manifest(drifted)


def test_train_remote_writes_manifest_and_rejects_completed_without_evidence(
        fake_modal, tmp_path, monkeypatch):
    module = _import_run_modal()
    fake_modal.invoke_remote = True
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    request = module.resolve_launch_request(**_valid_launch_sentinels(git_sha=sha))

    def fake_run_root(run_id: str) -> Path:
        path = tmp_path / "runs" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(module, "_remote_run_root", fake_run_root)
    real_execute = mrl.execute_training_attempt
    captured: dict[str, object] = {}

    def fake_prepare(**kwargs):
        manifest = kwargs["manifest"]
        assert isinstance(manifest, mrl.Manifest)
        captured["prepare_manifest"] = manifest
        run_root = Path(kwargs["run_root"])
        run_root.mkdir(parents=True, exist_ok=True)
        lock = kwargs["lock"]
        attempt_id = kwargs["attempt_id"]
        mrl.transition_status(run_root,
                              mrl.Status.PREPARING,
                              now=_aware(),
                              attempt_id=attempt_id,
                              lock=lock)
        mrl.atomic_write_json(run_root / mrl.MANIFEST_FILENAME, manifest.to_dict())
        mrl.transition_status(run_root,
                              mrl.Status.BUILDING,
                              now=_aware(),
                              attempt_id=attempt_id,
                              lock=lock)
        source_dir = tmp_path / "extracted"
        source_dir.mkdir(exist_ok=True)
        return mrl.PreparedSource(
            source_dir=source_dir,
            child_env={
                "PATH": "/usr/bin",
                "OMP_NUM_THREADS": "1"
            },
            train_command=["python", "-c", "pass"],
            heartbeat=_noop_heartbeat(),
            config_hash=manifest.config_hash,
        )

    def fake_execute(**kwargs):
        captured["execute_manifest"] = kwargs["manifest"]
        kwargs["process_factory"] = lambda *a, **k: FakeChild(returncode=0, stdout=b"done\n")
        kwargs["sleep"] = lambda _seconds: None
        kwargs["now"] = lambda: _aware()
        return real_execute(**kwargs)

    monkeypatch.setattr(mrl, "prepare_remote_source", fake_prepare)
    monkeypatch.setattr(mrl, "execute_training_attempt", fake_execute)
    result = module.launch_run(request,
                               repo=repo,
                               app_obj=module.app,
                               now=_aware(),
                               stdout=_capture_stdout())
    run_root = tmp_path / "runs" / request.run_id
    written = json.loads((run_root / mrl.MANIFEST_FILENAME).read_text())
    assert written["image_digest"] == PINNED_CUDA_CHILD_DIGEST
    assert written["modal_version"] == fake_modal.__version__
    assert written["effective_map"] == "simple"
    assert written["gpu"] == "T4"
    assert written["cpu_request"] == written["cpu_soft_limit"] == 8
    assert written["memory_request_mib"] == written["memory_hard_limit_mib"] == 16384
    assert written["vec_workers"] == 8
    assert written["training_argv"][0] == "--train"
    assert written["requested_timesteps"] == request.timesteps
    assert written["effective_timesteps"] == (request.timesteps //
                                              request.batch_size) * request.batch_size
    assert written["batch_size"] == request.batch_size
    assert written["seed"] == 2
    assert written["created_at"] == _aware().isoformat()
    assert written["runner_commit"] == sha
    assert written["resume_sha256"] is None
    assert written["resume_size"] is None
    assert written["resume_source_path"] is None
    assert written["thread_caps"] == _expected_thread_caps()
    assert written["resumed_from_run_id"] is None
    assert captured["execute_manifest"].config_hash == captured["prepare_manifest"].config_hash
    assert captured["execute_manifest"].thread_caps == _expected_thread_caps()
    assert result["status"] == mrl.Status.FAILED.value
    assert result["reason"] == mrl.REASON_INVALID_EVIDENCE
    assert json.loads((run_root / mrl.STATUS_FILENAME).read_text())["status"] == "failed"
    assert json.loads((run_root / mrl.RESULT_FILENAME).read_text())["status"] == "failed"


def test_train_remote_completes_against_post_dump_manifest_hash(fake_modal, tmp_path, monkeypatch):
    import torch

    module = _import_run_modal()
    fake_modal.invoke_remote = True
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    request = module.resolve_launch_request(**_valid_launch_sentinels(git_sha=sha))
    mount = tmp_path / "artifacts"
    mount.mkdir()
    monkeypatch.setattr(mrl, "VOLUME_MOUNT", mount)
    captured: dict[str, object] = {}
    real_prepare = mrl.prepare_remote_source
    real_execute = mrl.execute_training_attempt

    def materialize(self):
        for key, data in self.files.items():
            dest = mrl.VOLUME_MOUNT.joinpath(*PurePosixPath(key).parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)

    monkeypatch.setattr(FakeVolume, "reload", materialize)

    def fake_run(cmd, **kwargs):
        if "--dump-config" in list(cmd):
            _write_dumped_config(Path(captured["run_root"]))
        return subprocess.CompletedProcess(cmd, 0)

    def prepare_with_real_hash_rewrite(**kwargs):
        captured["prepare_in_manifest"] = kwargs["manifest"]
        captured["run_root"] = Path(kwargs["run_root"])
        kwargs["run"] = fake_run
        kwargs["start_heartbeat"] = _noop_heartbeat
        prepared = real_prepare(**kwargs)
        captured["prepared"] = prepared
        return prepared

    def execute_with_valid_evidence(**kwargs):
        captured["execute_manifest"] = kwargs["manifest"]
        run_root = Path(kwargs["run_root"])

        def factory(*_args, **_kwargs):
            ckpt_dir = run_root / "checkpoints"
            torch.save({"weight": torch.tensor([1.0])}, ckpt_dir / "dust2_policy.pt")
            effective = (request.timesteps // request.batch_size) * request.batch_size
            _write_metrics(ckpt_dir / "metrics.jsonl", [request.batch_size, effective])
            return FakeChild(returncode=0, stdout=b"done\n")

        kwargs["process_factory"] = factory
        kwargs["sleep"] = lambda _seconds: None
        kwargs["now"] = lambda: _aware()
        return real_execute(**kwargs)

    monkeypatch.setattr(mrl, "prepare_remote_source", prepare_with_real_hash_rewrite)
    monkeypatch.setattr(mrl, "execute_training_attempt", execute_with_valid_evidence)
    result = module.launch_run(request,
                               repo=repo,
                               app_obj=module.app,
                               now=_aware(),
                               stdout=_capture_stdout())
    run_root = Path(captured["run_root"])
    dumped = json.loads((run_root / "checkpoints" / "config.json").read_text())
    expected_hash = mrl.sha256_bytes(
        json.dumps(mrl.normalize_config_for_transport(dumped),
                   sort_keys=True,
                   separators=(",", ":")).encode())
    on_disk = json.loads((run_root / mrl.MANIFEST_FILENAME).read_text())
    prepared = captured["prepared"]
    execute_manifest = captured["execute_manifest"]
    assert captured["prepare_in_manifest"].config_hash == "0" * 64
    assert prepared.config_hash == expected_hash
    assert on_disk["config_hash"] == expected_hash
    assert execute_manifest.config_hash == expected_hash
    assert execute_manifest.config_hash != "0" * 64
    assert execute_manifest.thread_caps == _expected_thread_caps()
    assert execute_manifest.resumed_from_run_id is None
    assert on_disk["thread_caps"] == _expected_thread_caps()
    assert on_disk["resumed_from_run_id"] is None
    assert result["status"] == mrl.Status.COMPLETED.value
    assert result["reason"] is None
    assert json.loads((run_root / mrl.STATUS_FILENAME).read_text())["status"] == "completed"
    assert json.loads((run_root / mrl.RESULT_FILENAME).read_text())["status"] == "completed"


def test_train_remote_redelivery_claims_before_prepare(fake_modal, tmp_path, monkeypatch):
    module = _import_run_modal()
    fake_modal.invoke_remote = True
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    request = module.resolve_launch_request(**_valid_launch_sentinels(git_sha=sha))
    factory_calls: list[object] = []
    prepare_calls: list[int] = []

    def fake_run_root(run_id: str) -> Path:
        path = tmp_path / "runs" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(module, "_remote_run_root", fake_run_root)
    real_execute = mrl.execute_training_attempt

    def fake_prepare(**kwargs):
        prepare_calls.append(1)
        kwargs["volume"].commit()
        manifest = kwargs["manifest"]
        run_root = Path(kwargs["run_root"])
        run_root.mkdir(parents=True, exist_ok=True)
        lock = kwargs["lock"]
        attempt_id = kwargs["attempt_id"]
        now = _aware(minute=len(prepare_calls))
        mrl.transition_status(run_root,
                              mrl.Status.PREPARING,
                              now=now,
                              attempt_id=attempt_id,
                              lock=lock)
        mrl.atomic_write_json(run_root / mrl.MANIFEST_FILENAME, manifest.to_dict())
        mrl.transition_status(run_root,
                              mrl.Status.BUILDING,
                              now=now,
                              attempt_id=attempt_id,
                              lock=lock)
        source_dir = tmp_path / "extracted"
        source_dir.mkdir(exist_ok=True)
        return mrl.PreparedSource(
            source_dir=source_dir,
            child_env={
                "PATH": "/usr/bin",
                "OMP_NUM_THREADS": "1"
            },
            train_command=["python", "-c", "pass"],
            heartbeat=_noop_heartbeat(),
            config_hash=manifest.config_hash,
        )

    def fake_execute(**kwargs):

        def factory(*args, **factory_kwargs):
            factory_calls.append((args, factory_kwargs))
            return FakeChild(returncode=0, stdout=b"done\n")

        kwargs["process_factory"] = factory
        kwargs["sleep"] = lambda _seconds: None
        kwargs["now"] = lambda: _aware()
        return real_execute(**kwargs)

    monkeypatch.setattr(mrl, "prepare_remote_source", fake_prepare)
    monkeypatch.setattr(mrl, "execute_training_attempt", fake_execute)
    first = module.launch_run(request,
                              repo=repo,
                              app_obj=module.app,
                              now=_aware(),
                              stdout=_capture_stdout())
    assert first["status"] != mrl.REDELIVERED
    assert factory_calls
    payload = fake_modal.configured_remote_calls[0][1][0]
    run_root = tmp_path / "runs" / request.run_id
    status_bytes = (run_root / mrl.STATUS_FILENAME).read_bytes()
    manifest_bytes = (run_root / mrl.MANIFEST_FILENAME).read_bytes()
    volume = fake_modal.volumes[mrl.VOLUME_NAME]
    commits_after_first = volume.commit_count
    factory_count = len(factory_calls)
    prepare_count = len(prepare_calls)
    second = module.train_remote.with_options(
        gpu=request.gpu,
        cpu=request.cpu_request_limit,
        memory=request.memory_request_limit,
        timeout=request.timeout_minutes * 60,
        volumes={
            "/artifacts": volume
        },
    ).remote(payload)
    assert second == {"status": mrl.REDELIVERED, "run_id": request.run_id}
    assert len(prepare_calls) == prepare_count
    assert len(factory_calls) == factory_count
    assert (run_root / mrl.STATUS_FILENAME).read_bytes() == status_bytes
    assert (run_root / mrl.MANIFEST_FILENAME).read_bytes() == manifest_bytes
    assert volume.commit_count == commits_after_first


# ── Task 8 cycle E: client-only status / download ──────────────────────────


def _import_artifacts():
    return importlib.import_module("scripts.modal_artifacts")


def test_artifact_client_never_imports_app_or_creates_objects(fake_modal):
    module = _import_artifacts()
    assert "scripts.run_modal" not in sys.modules
    assert fake_modal.images == []
    assert fake_modal.apps == []
    with pytest.raises(mrl.ValidationError):
        module.collect_status("ok-id")
    assert fake_modal.volume_creates == []
    assert fake_modal.dict_creates == []
    assert fake_modal.volume_lookups == [(mrl.VOLUME_NAME, False)]
    assert fake_modal.configured_remote_calls == []
    assert fake_modal.base_remote_calls == []


def test_status_rejects_launch_only_options_via_client(fake_modal):
    module = _import_artifacts()
    with pytest.raises(mrl.ValidationError):
        module.main(["status", "--run-id", "ok-id", "--gpu", "T4"])
    assert fake_modal.volume_lookups == []
    assert fake_modal.volume_creates == []


@pytest.mark.parametrize("case", ["missing", "stale", "mismatch", "replaced", "ok"])
def test_status_checkpoint_loadable_protocol(fake_modal, tmp_path, case):
    import torch

    module = _import_artifacts()
    ckpt = tmp_path / "dust2_policy.pt"
    torch.save({"weight": torch.tensor([5.0])}, ckpt)
    ckpt_bytes = ckpt.read_bytes()
    digest = mrl.sha256_file(ckpt)
    volume = _named_volume(fake_modal)
    sidecar = {
        "sha256": digest,
        "size": len(ckpt_bytes),
        "mtime_ns": 1,
        "validated_at": _aware().isoformat(),
    }
    _ckpt_path, sidecar_path = _write_parent_artifacts(
        volume,
        "ok-id",
        status="completed",
        updated_at=_aware().isoformat(),
        ckpt_bytes=ckpt_bytes,
        sidecar=None if case == "missing" else sidecar,
    )
    if case == "stale":
        volume.files[sidecar_path] = json.dumps({**sidecar, "size": len(ckpt_bytes) + 8}).encode()
    elif case == "mismatch":
        volume.files[sidecar_path] = json.dumps({**sidecar, "sha256": "0" * 64}).encode()
    elif case == "replaced":
        volume.replace_after_read = {
            sidecar_path: json.dumps({
                **sidecar, "sha256": "1" * 64
            }).encode()
        }
    report = module.collect_status("ok-id", now=_aware())
    assert report["run_id"] == "ok-id"
    assert report["status"] == "completed"
    assert report["checkpoint_loadable"] is (case == "ok")


def test_status_does_not_interrupt_on_mere_file_presence(fake_modal, tmp_path):
    import torch

    module = _import_artifacts()
    ckpt = tmp_path / "dust2_policy.pt"
    torch.save({"weight": torch.tensor([6.0])}, ckpt)
    volume = _named_volume(fake_modal)
    _write_parent_artifacts(
        volume,
        "ok-id",
        status="training",
        updated_at=_aware().isoformat(),
        ckpt_bytes=ckpt.read_bytes(),
        sidecar=None,
    )
    dead = (mrl.RUNS_ROOT / "ok-id" / "checkpoints" / mrl.DEAD_CHECKPOINT_NAME).as_posix()
    volume.files[dead] = b"autopsy"
    report = module.collect_status("ok-id", now=_aware())
    assert report["status"] == "training"
    assert report["stale"] is False
    assert report["checkpoint_loadable"] is False


def test_download_stages_renames_and_refuses_overwrite(fake_modal, tmp_path):
    module = _import_artifacts()
    volume = _named_volume(fake_modal)
    volume.files["runs/ok-id/STATUS.json"] = b'{"status":"completed"}\n'
    volume.files["runs/ok-id/checkpoints/config.json"] = b"{}\n"
    dest_root = tmp_path / "outputs" / "modal"
    dest = module.download_run("ok-id", dest_root=dest_root)
    assert dest == dest_root / "ok-id"
    assert (dest / "STATUS.json").read_bytes() == b'{"status":"completed"}\n'
    assert (dest / "checkpoints" / "config.json").read_bytes() == b"{}\n"
    assert list(dest_root.glob(".ok-id.tmp-*")) == []
    assert fake_modal.iterdir_calls == [("runs/ok-id", True)]
    assert all(not path.startswith("/artifacts") for path, _rec in fake_modal.iterdir_calls)
    assert all(not path.startswith("/artifacts") for path in fake_modal.read_file_calls)
    with pytest.raises(mrl.ValidationError):
        module.download_run("ok-id", dest_root=dest_root)
    escaped = dest_root / "escaped"
    volume.files["runs/ok-id/../../secret"] = b"nope"
    # Existing dest still blocks; use a new id for the escape case.
    volume.files["runs/evil/../../secret"] = b"nope"
    volume.files["runs/evil/STATUS.json"] = b"{}\n"
    with pytest.raises(mrl.ValidationError):
        module.download_run("evil", dest_root=dest_root)
    assert not escaped.exists()
    assert not (dest_root / "evil").exists()
    assert not (dest_root / "secret").exists()


def test_detached_run_can_be_downloaded_later_by_id(fake_modal, tmp_path):
    module = _import_artifacts()
    assert "scripts.run_modal" not in sys.modules
    volume = _named_volume(fake_modal)
    volume.files["runs/detached-1/result.json"] = b'{"status":"completed"}\n'
    dest = module.download_run("detached-1", dest_root=tmp_path / "outputs" / "modal")
    assert (dest / "result.json").read_text() == '{"status":"completed"}\n'
    assert fake_modal.apps == []
    assert fake_modal.images == []
    assert fake_modal.configured_remote_calls == []


def test_detach_is_a_modal_run_cli_flag_not_an_app_option(fake_modal):
    # Modal 1.4.3 places --detach on `modal run` before FUNC_REF:
    # `modal run --detach scripts/run_modal.py --action run ...`
    module = _import_run_modal()
    assert "detach" not in module.main.__code__.co_varnames


# ── Task 8 quality-review: FileEntry types, empty prefixes, reservation ────


class FileEntryType(IntEnum):
    """Stand-in for modal.types.FileEntryType. str() is fileentrytype.directory."""

    FILE = 1
    DIRECTORY = 2
    SYMLINK = 3


def test_download_skips_fileentry_directories_and_refuses_symlinks(fake_modal, tmp_path):
    module = _import_artifacts()
    volume = _named_volume(fake_modal)
    volume.files["runs/ok-id/STATUS.json"] = b'{"status":"completed"}\n'
    volume.files["runs/ok-id/checkpoints/config.json"] = b"{}\n"
    volume.iterdir_entries = [
        SimpleNamespace(path="runs/ok-id/checkpoints", type=FileEntryType.DIRECTORY),
        SimpleNamespace(path="runs/ok-id/STATUS.json", type=FileEntryType.FILE),
        SimpleNamespace(path="runs/ok-id/checkpoints/config.json", type=FileEntryType.FILE),
    ]
    dest_root = tmp_path / "outputs" / "modal"
    dest = module.download_run("ok-id", dest_root=dest_root)
    assert (dest / "STATUS.json").read_bytes() == b'{"status":"completed"}\n'
    assert (dest / "checkpoints" / "config.json").read_bytes() == b"{}\n"
    assert not (dest / "checkpoints").is_file()

    volume.files["runs/link-id/STATUS.json"] = b"{}\n"
    volume.iterdir_entries = [
        SimpleNamespace(path="runs/link-id/outside", type=FileEntryType.SYMLINK),
        SimpleNamespace(path="runs/link-id/STATUS.json", type=FileEntryType.FILE),
    ]
    with pytest.raises(mrl.ValidationError, match="symlink"):
        module.download_run("link-id", dest_root=dest_root)
    assert not (dest_root / "link-id").exists()


def test_iterdir_paths_treats_missing_prefix_not_found_as_empty(fake_modal):
    module = _import_run_modal()
    volume = _named_volume(fake_modal)
    volume.missing_prefix_exc = FakeNotFoundError
    assert module._iterdir_paths(volume, "sources") == []
    assert module._iterdir_paths(volume, "runs") == []
    assert not module._volume_has_client_path(volume, "sources/deadbeef.tar.gz")


def test_first_launch_lists_empty_volume_prefixes(fake_modal, tmp_path):
    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    volume = _named_volume(fake_modal)
    volume.missing_prefix_exc = FakeNotFoundError
    _named_dict(fake_modal)
    request = module.resolve_launch_request(**_valid_launch_sentinels(git_sha=sha))
    module.launch_run(request, repo=repo, app_obj=module.app, stdout=_capture_stdout())
    assert fake_modal.configured_remote_calls
    source_name = next(path for path in volume.files if path.startswith("sources/"))
    assert source_name.endswith(".tar.gz")


def test_launch_upload_failure_records_failure_code_without_freeing_id(fake_modal, tmp_path):
    module = _import_run_modal()
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    volume = _named_volume(fake_modal)
    volume.fail_prefix = "sources/"
    _named_dict(fake_modal)
    request = module.resolve_launch_request(**_valid_launch_sentinels(git_sha=sha))
    with pytest.raises(OSError, match="could not upload"):
        module.launch_run(request, repo=repo, app_obj=module.app, stdout=_capture_stdout())
    claim = fake_modal.dicts[mrl.REGISTRY_NAME].get(mrl.run_registry_key(request.run_id))
    assert claim["failure_code"] == mrl.FAILURE_UPLOAD
    assert claim["attempt_id"]
    assert "secret" not in json.dumps(claim)
    assert (mrl.RUNS_ROOT / request.run_id / mrl.RESERVATION_FILENAME).as_posix() in volume.files
    assert fake_modal.configured_remote_calls == []
    with pytest.raises(mrl.ValidationError):
        module.launch_run(request, repo=repo, app_obj=module.app, stdout=_capture_stdout())


def test_lookup_helpers_chain_unexpected_errors(fake_modal):
    launch = _import_run_modal()
    artifacts = _import_artifacts()

    class BoomFactory:

        @staticmethod
        def from_name(name, create_if_missing=False):
            del name, create_if_missing
            raise RuntimeError("modal backend exploded")

    with pytest.raises(RuntimeError, match="modal backend exploded") as launch_info:
        launch._lookup_named(BoomFactory, mrl.VOLUME_NAME, missing="artifact volume is missing")
    assert launch_info.value.__cause__ is None

    class BoomModal:

        class Volume:

            @staticmethod
            def from_name(name, create_if_missing=False):
                del name, create_if_missing
                raise RuntimeError("volume backend exploded")

    with pytest.raises(RuntimeError, match="volume backend exploded") as artifact_info:
        artifacts._lookup_volume(BoomModal)
    assert artifact_info.value.__cause__ is None


def test_corrupt_volume_json_is_validation_error(fake_modal):
    module = _import_artifacts()
    volume = _named_volume(fake_modal)
    volume.files["runs/ok-id/STATUS.json"] = b"{not-json"
    with pytest.raises(mrl.ValidationError):
        module.collect_status("ok-id", now=_aware())
    del volume.files["runs/ok-id/STATUS.json"]
    volume.files["runs/ok-id/reservation.json"] = b'{"created_at":"not-a-timestamp"}'
    with pytest.raises(mrl.ValidationError):
        module.collect_status("ok-id", now=_aware())
