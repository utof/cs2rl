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
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import threading
import time
from datetime import UTC, timedelta
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # scripts/ is a namespace package; tests import scripts.modal_runner_lib
    # the same way the later CLIs will. Do not rely on the editable install.
    sys.path.insert(0, str(ROOT))

import scripts.modal_runner_lib as mrl                                                 # noqa: E402, I001
from tests.modal_test_helpers import (                                                 # noqa: E402
    FakeChild, _aware, _git, _init_source_repo, _noop_heartbeat, _write_dumped_config,
    _write_metrics)


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


@pytest.mark.parametrize("effective_map", ["simple", "dust2", "arena-duel"])
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
    """Static train.py long options + hyphenated RewardWeights field names.

    Reads source (no `import src.train`) so collection cannot pull CUDA.
    Generated `add_argument(f"--{_rw_name...}")` is a JoinedStr and is
    recovered from the dataclass fields instead.

    ONE FILE PLUS THE DATACLASS (spec 2026-09-03 §2.3): the argparse parser
    still lives in src/train.py (it is built inline under
    `if __name__ == "__main__"`), but the 23 `--reward-*`/`--pbrs-*` flag names
    are now the field names of `env_config.RewardWeights`. Taking only one of
    the two sources silently drops half the option set — train.py alone loses all 23
    reward flags, the dataclass alone loses every other flag — and the
    set-equality assert below would then "fail" against the runner mirror for a
    reason that has nothing to do with the mirror. Neither contribution is
    optional; if a symbol moves again, extend this function.
    """
    names: set[str] = set()
    # The 23 --reward-*/--pbrs-* flags are generated from RewardWeights' fields
    # (spec 2026-09-03 §2.3); env_config is stdlib-only so importing it here
    # keeps collection free of torch/CUDA. train.py's static add_argument
    # calls are still recovered from source below.
    import dataclasses
    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from env_config import RewardWeights
    names.update(f"--{f.name.replace('_', '-')}" for f in dataclasses.fields(RewardWeights))
    for rel in ("src/train.py", ):
        tree = ast.parse((ROOT / rel).read_text())
        for node in ast.walk(tree):
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
    "--map",                           # R0-J: runner owns map choice (effective_map)
    "--run-id",
    "--resume-run",
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


def test_tct_split_trunk_is_allowed():
    """--tct-split-trunk is a live store_true on the 7R scientific argv.

    Mirrors test_exact_allowed_flags_are_kept: the flag must survive
    build_run_request together with the rest of the 7R scientific set
    (seed / warmstart-entropy / TAG / heads / a reward weight). An
    unknown-flag spelling still fails — that pin is
    test_unknown_spelling_rejected_before_ownership.
    """
    raw = ("--timesteps 30000000 --seed 2 --warmstart-entropy --no-dead-run-abort "
           "--tag-diagnostic --tag-every 5 --tct-split-heads --tct-split-trunk "
           "--reward-win 1.0")
    request = mrl.build_run_request(**_valid_run_kwargs(train_args=raw))
    assert request.timesteps == 30_000_000
    assert "--tct-split-trunk" in request.train_args
    assert "--tct-split-heads" in request.train_args


# ── Task 2 cycle C: runner-owned argv injection ────────────────────────────


def test_training_argv_is_exact_for_resume_and_defaults():
    request = mrl.build_run_request(**_valid_run_kwargs())
    run_root = Path("/artifacts/runs/140826-b7r-seed2-shared")
    remote_resume = "/artifacts/inputs/sha256/abc.pt"
    assert request.training_argv(run_root, remote_resume=remote_resume) == [
        "--train",
        "--map",                                                                  # R0-J: runner always emits the map, even the old implicit "simple"
        "simple",
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


def test_runner_emits_map_once_never_dust2():
    """R0-J (Task 14): the runner emits `--map <effective_map>` for EVERY map
    (the `--dust2` alias is gone from the argv — train.py's `--map` wins over
    it anyway, so a stray alias would be silently inert)."""
    run_root = Path("/artifacts/runs/ok-id")
    for name in ("simple", "dust2", "arena-duel"):
        request = mrl.build_run_request(**_valid_run_kwargs(effective_map=name, run_id="ok-id"))
        argv = request.training_argv(run_root)
        assert request.effective_map == name
        assert "--dust2" not in argv
        assert argv.count("--map") == 1
        assert argv[argv.index("--map") + 1] == name


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
    assert "--dust2" not in argv
    assert argv.count("--map") == 1 and argv[argv.index("--map") + 1] == "dust2"
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
    # compute_batch_dims moved train.py -> train_config.py in the post-rung1a
    # refactor (2026-08-31); the runner's AGENTS_PER_ENV/BPTT_HORIZON mirror is
    # pinned against wherever it actually lives, not against train.py by habit.
    train_src = (ROOT / "src" / "train_config.py").read_text()
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
        raise AssertionError("live compute_batch_dims not found in src/train_config.py")
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
        self.replace_after_read: dict[PurePosixPath, bytes | None] = {}

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

    def read_file(self, path: PurePosixPath) -> bytes | None:
        with self._lock:
            if path in self.replace_after_read:
                current = self.committed.get(path)
                replacement = self.replace_after_read.pop(path)
                if replacement is None:
                    self.committed.pop(path, None)
                else:
                    self.committed[path] = replacement
                return current
            return self.committed.get(path)


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


def test_interrupt_publishes_sidecar_after_unstable_live_saves(tmp_path):
    """Live PufferLib rewrites dust2_policy.pt every epoch (~0.5s).

    The 1s settle window never elapses while the child is alive. After SIGINT
    the file is stable and finalize must still publish the sidecar, or resume
    cannot validate the parent.
    """
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    rewrites = {"n": 0}

    def fake_sleep(seconds: float) -> None:
        hooks["sleep"](seconds)
        if seconds >= 1.0 and child.poll() is None:
            rewrites["n"] += 1
            _write_policy_checkpoint(kwargs["run_root"], float(rewrites["n"]))

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            sleep=fake_sleep,
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    ckpt = _write_policy_checkpoint(kwargs["run_root"], 0.0)
    sidecar = kwargs["run_root"] / "checkpoints" / "dust2_policy.pt.meta.json"
    thread, finished, boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, _term = _wait_until_handlers(hooks["installed"], hooks["originals"])
        deadline = time.monotonic() + 2.0
        while rewrites["n"] < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert rewrites["n"] >= 2
        assert not sidecar.exists()
        int_handler(signal.SIGINT, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)
    assert json.loads(
        (kwargs["run_root"] / mrl.STATUS_FILENAME).read_text())["status"] == "interrupted"
    deadline = time.monotonic() + 2.0
    while not sidecar.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sidecar.is_file()
    meta = json.loads(sidecar.read_text())
    assert meta["sha256"] == mrl.sha256_file(ckpt)
    assert meta["size"] == ckpt.stat().st_size


# ── Runner-interpreter checkpoint validation ───────────────────────────────
#
# The Modal runner process and the training child are DIFFERENT interpreters.
# The runner is the image's standalone /usr/local/bin/python (only uv + modal);
# torch lives exclusively in the PREBUILT_PYTHON venv that runs train.py.
# Verified in a live container on 2026-08-14:
#   runner_executable=/usr/local/bin/python  runner_torch=MISSING
# Every test above runs on a laptop where `import torch` succeeds, so none of
# them can see this. These do: they force the torch-less runner condition.


def _no_torch(monkeypatch, *, prebuilt: str) -> None:
    """Simulate the container runner: no in-process torch, prebuilt venv at `prebuilt`."""

    def raise_import_error():
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(mrl, "_import_torch", raise_import_error)
    monkeypatch.setattr(mrl, "PREBUILT_PYTHON", prebuilt)


def _publish(run_root: Path):
    commits: list[int] = []
    outcome = mrl.publish_stable_checkpoint(
        run_root,
        now=_aware,
        commit=lambda: commits.append(1),
        sleep=lambda _seconds: None,
        last_published=None,
    )
    return outcome, commits


def test_publish_validates_via_prebuilt_interpreter_when_runner_lacks_torch(tmp_path, monkeypatch):
    """A valid checkpoint must still publish when the runner cannot import torch."""
    run_root = tmp_path / "run"
    run_root.mkdir()
    ckpt = _write_policy_checkpoint(run_root, 1.0)
    _no_torch(monkeypatch, prebuilt=sys.executable)

    outcome, commits = _publish(run_root)

    sidecar = ckpt.with_name("dust2_policy.pt.meta.json")
    assert sidecar.is_file()
    assert outcome.reason is None
    assert outcome.generation == (ckpt.stat().st_mtime_ns, ckpt.stat().st_size)
    assert len(commits) == 1
    assert json.loads(sidecar.read_text())["sha256"] == mrl.sha256_file(ckpt)


def test_prebuilt_validation_still_rejects_a_torn_checkpoint(tmp_path, monkeypatch):
    """The fallback must not become a rubber stamp: garbage still fails to load."""
    run_root = tmp_path / "run"
    (run_root / "checkpoints").mkdir(parents=True)
    (run_root / "checkpoints" / "dust2_policy.pt").write_bytes(b"torn-not-a-checkpoint")
    _no_torch(monkeypatch, prebuilt=sys.executable)

    outcome, commits = _publish(run_root)

    assert not (run_root / "checkpoints" / "dust2_policy.pt.meta.json").exists()
    assert outcome.generation is None
    assert "not weights-only loadable" in outcome.reason
    assert commits == []


def test_publish_reason_names_the_missing_interpreter(tmp_path, monkeypatch):
    """No torch and no prebuilt venv: skipping is fine, skipping SILENTLY is not."""
    run_root = tmp_path / "run"
    run_root.mkdir()
    _write_policy_checkpoint(run_root, 1.0)
    _no_torch(monkeypatch, prebuilt=str(tmp_path / "nonexistent" / "python"))

    outcome, commits = _publish(run_root)

    assert not (run_root / "checkpoints" / "dust2_policy.pt.meta.json").exists()
    assert "nonexistent" in outcome.reason
    assert commits == []


def test_interrupt_without_publishable_checkpoint_writes_a_reason_file(tmp_path, monkeypatch):
    """finalize must leave evidence on the Volume, before its commit, of WHY there
    is no sidecar. Three T4 runs were burned on a silently swallowed skip."""
    _no_torch(monkeypatch, prebuilt=str(tmp_path / "nonexistent" / "python"))
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    commits: list[bool] = []

    def commit() -> None:
        commits.append(
            (kwargs["run_root"] / "checkpoints" / mrl.CHECKPOINT_PUBLISH_REASON_NAME).is_file())

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            commit=commit,
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    _write_policy_checkpoint(kwargs["run_root"], 1.0)
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, _term = _wait_until_handlers(hooks["installed"], hooks["originals"])
        int_handler(signal.SIGINT, None)
        assert finished.wait(timeout=2.0)
    finally:
        child.release()
        thread.join(timeout=2.0)

    reason_path = kwargs["run_root"] / "checkpoints" / mrl.CHECKPOINT_PUBLISH_REASON_NAME
    assert reason_path.is_file()
    payload = json.loads(reason_path.read_text())
    assert "nonexistent" in payload["reason"]
    assert payload["at"]
    # Written BEFORE a commit, or it never reaches the Volume.
    assert any(commits)


def test_interrupt_commits_status_even_if_prebuilt_load_hangs(tmp_path, monkeypatch):
    """SIGINT finalize must persist STATUS before any hung PREBUILT_PYTHON load.

    Modal preemption grace is ~30s and Function-timeout slack is seconds. A
    120s weights-only load inside finalize can lose both sidecar and STATUS.
    The watcher thread is exempt so its 50ms poll cannot stall this test.
    """
    release = threading.Event()

    def hanging_load(_path):
        if threading.current_thread().name == "cs2rl-checkpoint-watch":
            raise mrl.ValidationError("watcher must not hang the interrupt path")
        if not release.wait(timeout=10.0):
            raise mrl.ValidationError("test timed out waiting to release the hung load")

    monkeypatch.setattr(mrl, "_assert_weights_only_loadable", hanging_load)
    child = FakeChild(hold=True)
    hooks = _signal_hooks(child)
    commits: list[str | None] = []

    def commit() -> None:
        status_path = kwargs["run_root"] / mrl.STATUS_FILENAME
        if not status_path.is_file():
            commits.append(None)
            return
        commits.append(json.loads(status_path.read_text())["status"])

    kwargs = _consume_training_kwargs(
        _training_kwargs(
            tmp_path,
            child=child,
            commit=commit,
            killpg=hooks["killpg"],
            getpgid=hooks["getpgid"],
            signal_signal=hooks["signal_signal"],
        ))
    _write_policy_checkpoint(kwargs["run_root"], 1.0)
    thread, finished, _boxed = _run_attempt_in_thread(kwargs)
    try:
        int_handler, _term = _wait_until_handlers(hooks["installed"], hooks["originals"])
        handler_thread = threading.Thread(target=int_handler,
                                          args=(signal.SIGINT, None),
                                          daemon=True)
        handler_thread.start()
        deadline = time.monotonic() + 2.0
        interrupted = False
        while time.monotonic() < deadline:
            status_path = kwargs["run_root"] / mrl.STATUS_FILENAME
            if (status_path.is_file()
                    and json.loads(status_path.read_text())["status"] == "interrupted"):
                interrupted = True
                break
            time.sleep(0.01)
        assert interrupted, "STATUS must become interrupted while the prebuilt load is still hung"
        assert "interrupted" in commits
    finally:
        release.set()
        child.release()
        assert finished.wait(timeout=2.0)
        thread.join(timeout=2.0)


def test_checkpoint_watcher_threads_generation_into_last_published(tmp_path, monkeypatch):
    """A watcher-only typo on PublishOutcome.generation is swallowed every 50ms.

    Direct publish tests cannot see that: they assert .generation on the
    function return, not on the value the thread feeds back as last_published.
    """
    seen: list[tuple[int, int] | None] = []
    generation = (111, 222)

    def fake_publish(*_args, last_published=None, **_kwargs):
        seen.append(last_published)
        return mrl.PublishOutcome(generation)

    monkeypatch.setattr(mrl, "publish_stable_checkpoint", fake_publish)
    stop, watcher = mrl._start_checkpoint_watcher(
        run_root=tmp_path,
        now=_aware,
        commit=lambda: None,
        sleep=lambda _seconds: None,
    )
    try:
        deadline = time.monotonic() + 2.0
        while len(seen) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(seen) >= 2
        assert seen[0] is None
        assert seen[1] == generation
    finally:
        stop.set()
        watcher.join(timeout=2.0)


def test_completed_run_validates_without_runner_torch(tmp_path, monkeypatch):
    """validate_completed_run torch-loads too: without the fallback every clean
    exit is misfiled as failed/invalid_evidence and no run can ever complete."""
    run_root, manifest, effective, ckpt = _minimal_completed_tree(tmp_path)
    _no_torch(monkeypatch, prebuilt=sys.executable)

    evidence = mrl.validate_completed_run(run_root, manifest)

    assert evidence.last_step == effective
    assert evidence.checkpoint_sha256 == mrl.sha256_file(ckpt)


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
