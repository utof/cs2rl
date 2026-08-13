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
import subprocess
import sys
import tomllib
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


def test_live_option_mirror_contains_exact_long_names_only():
    assert "--timesteps" in mrl.LIVE_TRAIN_OPTIONS
    assert "--num_envs" in mrl.LIVE_TRAIN_OPTIONS
    assert "--dust2" in mrl.LIVE_TRAIN_OPTIONS
    assert "--checkpoint-dir" in mrl.LIVE_TRAIN_OPTIONS
    assert "--checkpoint_dir" in mrl.LIVE_TRAIN_OPTIONS
    assert "--devi" not in mrl.LIVE_TRAIN_OPTIONS
    assert "--num-envs" not in mrl.LIVE_TRAIN_OPTIONS  # runner spelling, not live


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
