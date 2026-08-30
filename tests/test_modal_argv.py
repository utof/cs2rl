"""R0-J (Task 14): the Modal runner's train-args mirror vs the Rung 1 launch line.

WHY: `LIVE_TRAIN_OPTION_ARITY` is a hand-kept mirror of train.py's argparse
long options. Every flag the Rung 1 spec (§4) puts on the command line must be
KNOWN to the mirror, or the runner rejects the launch as "unknown option"
after the operator has already paid for the container. This file pins the
whole Rung 1 argv at once so a forgotten mirror entry fails here, not on Modal.

Also pins: the runner owns `--map` (emits `--map <effective_map>` first, never
`--dust2`), `--resume-run` is not a Modal flag (the runner owns paths/ids), and
the small-budget guard compares --timesteps against a RAW batch (conservative).
"""
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from modal_runner_lib import (                                                        # noqa: E402
    ALLOWED_MAPS, LIVE_TRAIN_OPTION_ARITY, RUNNER_OWNED_TRAIN_FLAGS, ValidationError,
    _assemble_train_argv, build_run_request, validate_train_args,
)


def _rung1_train_args():
    # Spec §4 argv minus the runner-owned flags: --num_envs (RunRequest.num_envs),
    # --map (RunRequest.effective_map) and --run-id (RunRequest.run_id — Task 7
    # put it in RUNNER_OWNED_TRAIN_FLAGS; the runner names the run).
    return ("--timesteps 10000000 --n-active-per-team 1 --round-time-ticks 160 "
            "--crouch-enabled 0 --aim-entropy-bonus off --aim-log-std-max -2.9957 "
            "--gamma 0.99 --checkpoint-interval 10 --eval-interval 10 --no-dead-run-abort "
            "--seed 0 --reward-win-t-elimination 1.0 "
            "--reward-win-ct-elimination 1.0 --reward-win-ct-timeout 0 --reward-kill 0.3").split()


def _req(**kw):
    base = dict(run_id="rung1-s0",
                git_sha="0" * 40,
                effective_map="arena-duel",
                num_envs=16,
                cpu_cores=4,
                vec_workers=4,
                train_args=" ".join(_rung1_train_args()))
    base.update(kw)
    return build_run_request(**base)


def test_every_rung1_flag_is_known():
    assert validate_train_args(_rung1_train_args()) == 10_000_000      # returns --timesteps verbatim


def test_pbrs_gamma_is_mirrored():
    assert LIVE_TRAIN_OPTION_ARITY.get("--pbrs-gamma") == 1
    assert LIVE_TRAIN_OPTION_ARITY.get("--gamma") == 1
    assert validate_train_args(["--timesteps", "1000000", "--pbrs-gamma", "0.99"]) == 1_000_000


def test_opponent_is_mirrored_with_arity_one():
    """Rung 1a T3: `--opponent noop` is the whole point of the T4 launch, and it
    takes a VALUE — an arity-0 entry would make the runner treat "noop" as a
    stray positional and reject the launch (test_unconsumed_positional_tokens_
    rejected). The name-set mirror is enforced in tests/test_modal_runner.py."""
    assert LIVE_TRAIN_OPTION_ARITY.get("--opponent") == 1
    assert validate_train_args(["--timesteps", "1000000", "--opponent", "noop",
                                "--no-self-play"]) == 1_000_000


def test_resume_run_is_local_only():
    # Task 7 lists --resume-run as runner-owned (the runner owns paths/ids), so
    # the rejection reads "runner-owned" rather than "unknown"; either way a
    # user --resume-run never reaches the container.
    with pytest.raises(ValidationError, match="runner-owned"):
        validate_train_args(["--timesteps", "1000000", "--resume-run", "x"])
    assert "--resume-run" in RUNNER_OWNED_TRAIN_FLAGS


def test_user_map_is_rejected_and_runner_emits_map():
    assert "--map" in RUNNER_OWNED_TRAIN_FLAGS and "--map" in LIVE_TRAIN_OPTION_ARITY
    with pytest.raises(ValidationError, match="runner-owned"):
        validate_train_args(["--timesteps", "1000000", "--map", "simple"])
    with pytest.raises(ValidationError, match="runner-owned"):
        validate_train_args(["--timesteps", "1000000", "--dust2"])
    assert "arena-duel" in ALLOWED_MAPS


@pytest.mark.parametrize("effective_map", ["arena-duel", "dust2", "simple"])
def test_assemble_emits_map_name(tmp_path, effective_map):
    argv = _assemble_train_argv(_req(effective_map=effective_map),
                                tmp_path,
                                None,
                                dump_config=False)
    assert argv[:3] == ["--train", "--map", effective_map]
    assert "--dust2" not in argv
    assert argv.count("--map") == 1
    argv = _assemble_train_argv(_req(effective_map=effective_map), tmp_path, None, dump_config=True)
    assert argv[:3] == ["--dump-config", "--map", effective_map]


def test_small_budget_guard_is_conservative():
    # timesteps is a PARTICIPATING budget; the guard compares against a raw batch. Pinned.
    with pytest.raises(ValidationError, match="below one full batch"):
        _req(train_args="--timesteps 1000 --no-dead-run-abort")
