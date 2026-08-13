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
import json
import subprocess
import sys
import tarfile
import threading
import tomllib
from datetime import UTC
from pathlib import Path, PurePosixPath

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
        **_valid_run_kwargs(train_args="--timesteps 1 --wandb", wandb_secret_name="wandb"))
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
