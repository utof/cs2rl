"""Behavior tests for scripts.modal_runner.request: run fields, resources, train args.

One of the per-module runner test files (RUNNER_TEST_FILES in
tests/modal/modal_runner_tables.py). Before you add, move or delete a test here, or
add a helper, read THE PLACEMENT RULE FOR RUNNER TESTS in
tests/modal/test_modal_packaging.py: which file a test belongs in, what the change
costs in the seam manifest, and where helpers go.
"""
import ast
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

ROOT = REPO_ROOT

import scripts.modal_runner as mrl                                             # noqa: E402, I001
from scripts.modal_runner import commands, request                             # noqa: E402, I001
from tests.modal.modal_test_helpers import _live_batch_size, _valid_run_kwargs # noqa: E402

# ── Run request: run id, secret, map, resource and action fields ───────────


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
    # Coupling (--wandb requires the secret and vice versa) is checked below.
    assert request.validate_secret_name(name) == name


@pytest.mark.parametrize("name", ["", "-bad", "has space", "has/slash", "x" * 81])
def test_invalid_secret_names_are_rejected(name):
    with pytest.raises(mrl.ValidationError):
        request.validate_secret_name(name)


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


# ── Train args: exact live-option grammar, no argparse prefixes ────────────


def test_parse_train_args_preserves_punctuation_as_data():
    # shlex.split must keep ';', quotes, and paths as argv DATA. Never a shell.
    argv = request.parse_train_args(
        "--timesteps 30000000 --wandb-entity 'org/name;rm -rf' --seed 2")
    assert argv == (
        "--timesteps",
        "30000000",
        "--wandb-entity",
        "org/name;rm -rf",
        "--seed",
        "2",
    )
    assert request.validate_train_args(argv) == 30_000_000


def test_parse_train_args_unclosed_quote_is_validation_error():
    with pytest.raises(mrl.ValidationError):
        request.parse_train_args('--timesteps 1 --wandb-entity "unclosed')
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**_valid_run_kwargs(train_args="--timesteps 1 --name 'oops"))


def test_omitted_train_args_fail_closed():
    kwargs = _valid_run_kwargs()
    del kwargs["train_args"]
    with pytest.raises(mrl.ValidationError):
        mrl.build_run_request(**kwargs)


def test_live_option_mirror_contains_exact_long_names_only():
    assert "--timesteps" in request.LIVE_TRAIN_OPTIONS
    assert "--num_envs" in request.LIVE_TRAIN_OPTIONS
    assert "--dust2" in request.LIVE_TRAIN_OPTIONS
    assert "--checkpoint-dir" in request.LIVE_TRAIN_OPTIONS
    assert "--checkpoint_dir" in request.LIVE_TRAIN_OPTIONS
    assert "--devi" not in request.LIVE_TRAIN_OPTIONS
    assert "--num-envs" not in request.LIVE_TRAIN_OPTIONS              # runner spelling, not live


def _live_train_long_options_from_source() -> set[str]:
    """Static train.py long options + hyphenated RewardWeights field names.

    Reads source (no `import cs2rl.train`) so collection cannot pull CUDA.
    Generated `add_argument(f"--{_rw_name...}")` is a JoinedStr and is
    recovered from the dataclass fields instead.

    ONE FILE PLUS THE DATACLASS (spec 2026-09-03 §2.3): the argparse parser
    still lives in src/cs2rl/train.py (it is built inline under
    `if __name__ == "__main__"`), but the 23 `--reward-*`/`--pbrs-*` flag names
    are now the field names of `env.config.RewardWeights`. Taking only one of
    the two sources silently drops half the option set — train.py alone loses all 23
    reward flags, the dataclass alone loses every other flag — and the
    set-equality assert below would then "fail" against the runner mirror for a
    reason that has nothing to do with the mirror. Neither contribution is
    optional; if a symbol moves again, extend this function.
    """
    names: set[str] = set()
    # The 23 --reward-*/--pbrs-* flags are generated from RewardWeights' fields
    # (spec 2026-09-03 §2.3); env.config is stdlib-only so importing it here
    # keeps collection free of torch/CUDA. train.py's static add_argument
    # calls are still recovered from source below.
    import dataclasses

    from cs2rl.env.config import RewardWeights
    names.update(f"--{f.name.replace('_', '-')}" for f in dataclasses.fields(RewardWeights))
    for rel in ("src/cs2rl/train/__main__.py", ):
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
    assert _live_train_long_options_from_source() == set(request.LIVE_TRAIN_OPTION_ARITY)


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
        request.validate_train_args(request.parse_train_args(raw))
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


# ── Argv builders: runner-owned flag injection into the train argv ─────────


def test_training_argv_is_exact_for_resume_and_defaults():
    request = mrl.build_run_request(**_valid_run_kwargs())
    run_root = Path("/artifacts/runs/140826-b7r-seed2-shared")
    remote_resume = "/artifacts/inputs/sha256/abc.pt"
    assert request.training_argv(run_root, remote_resume=remote_resume) == [
        "--train",
        "--map",                                                                       # R0-J: runner always emits the map, even the old implicit "simple"
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
    assert commands.build_train_argv(request, remote_resume) == request.training_argv(
        run_root, remote_resume=remote_resume)


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


def test_no_resume_omits_resume_flag():
    request = mrl.build_run_request(**_valid_run_kwargs())
    argv = request.training_argv(Path("/artifacts/runs/ok-id"))
    assert "--resume" not in argv


# ── Run preconditions: resume / W&B coupling, request constructors ─────────


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
        request.ResumeRequest(
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
