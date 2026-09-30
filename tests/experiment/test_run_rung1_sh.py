"""scripts/run_rung1.sh — launcher control flow against a FAKE train.py.

WHAT: drives the real bash script with RUNG1_TRAIN_CMD pointed at a tiny
Python stand-in that records every argv it receives and simulates the
outcomes that matter (clean finish, crash after / before the first checkpoint
set, permanent failure). No sim, no torch.

WHY: the launcher's job is the retry policy (spec §4): a seed whose process
died AFTER writing a full-state checkpoint set must be RESUMED with the same
argv + --resume-run; one that died before is not resumable; a finished seed
(DONE sentinel) is skipped; one dead seed must not abort the sweep. Every one
of those branches is a silent-failure class if wrong (a "finished" seed that
was actually half-run would make the gate judge a partial window).

PITFALLS pinned here:
  - the done marker is `<dir>/DONE`, NOT dust2_policy.pt (written every
    --save_every_sec) — a dir holding dust2_policy.pt + model_*.pt +
    trainer_state.pt and no DONE must be resumed, not skipped;
  - the retry leg must re-emit --map, --seed and every other non-budget flag
    (Task 8/12 rulings: `env`, `seed`, `pin_pitch`, gamma keys are NOT in
    RESUME_CONFIG_ALLOWLIST — a bare `--resume-run` would be refused);
  - RUNG1_EXTRA is appended LAST so its repeated options win in argparse;
  - DONE records the participating --timesteps budget: a re-run with a LARGER
    budget extends the seed (resume), an equal/larger recorded budget skips,
    an empty (legacy) DONE skips;
  - every train.py leg's stdout+stderr is tee'd to <dir>/train.log and a
    non-zero exit still triggers the retry (pipefail / PIPESTATUS handling).
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "run_rung1.sh"

# Fake train.py. Modes (FAKE_MODE env): ok | crash_after_ckpt | crash_no_ckpt | always_fail_after_ckpt.
# Records argv into <checkpoint-dir>/argv.jsonl; "checkpoint set" = the files
# resolve_resume_run keys on (<dir>/<run_id>/trainer_state.pt + model_*.pt)
# plus the periodic dust2_policy.pt that must NOT count as a done marker.
FAKE = textwrap.dedent("""
    import json, os, sys
    from pathlib import Path
    argv = sys.argv[1:]
    d = Path(argv[argv.index("--checkpoint-dir") + 1])
    rid = argv[argv.index("--run-id") + 1]
    d.mkdir(parents=True, exist_ok=True)
    with (d / "argv.jsonl").open("a") as fh:
        fh.write(json.dumps(argv) + "\\n")
    n = sum(1 for _ in (d / "argv.jsonl").open())
    mode = os.environ["FAKE_MODE"]
    print(f"fake-train stdout call={n}", flush=True)           # both streams must land in train.log
    print(f"fake-train stderr call={n}", file=sys.stderr, flush=True)
    resuming = "--resume-run" in argv
    def write_ckpt():
        (d / rid).mkdir(exist_ok=True)
        (d / rid / "model_000010.pt").write_bytes(b"m")
        (d / rid / "trainer_state.pt").write_bytes(b"t")
        (d / rid / "train_state.pt").write_bytes(b"s")
        (d / "dust2_policy.pt").write_bytes(b"p")
    if mode == "ok":
        write_ckpt(); sys.exit(0)
    if mode == "crash_no_ckpt":
        sys.exit(1)
    if mode == "crash_after_ckpt":          # first call dies after its checkpoint, retry succeeds
        write_ckpt(); sys.exit(0 if resuming else 1)
    if mode == "always_fail_after_ckpt":
        write_ckpt(); sys.exit(1)
    raise SystemExit("unknown FAKE_MODE " + mode)
""")


@pytest.fixture
def fake(tmp_path):
    p = tmp_path / "fake_train.py"
    p.write_text(FAKE)
    return p


def run_script(fake, out_root, mode, seeds="0", neg_seeds="", max_retries=5, extra=None, env=None):
    e = dict(os.environ,
             FAKE_MODE=mode,
             RUNG1_TRAIN_CMD=f"{sys.executable} {fake}",
             RUNG1_SEEDS=seeds,
             RUNG1_NEG_SEEDS=neg_seeds)
    if extra is not None:
        e["RUNG1_EXTRA"] = extra
    if env:
        e.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), str(out_root), str(max_retries)],
        capture_output=True,
        text=True,
        timeout=120,
        env=e,
        cwd=tmp_root(out_root))


def tmp_root(out_root):
    return str(Path(out_root).parent)


def argvs(out_root, label):
    p = Path(out_root) / label / "argv.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def _val(argv, flag):
    """LAST value of a repeated option — argparse semantics."""
    return argv[len(argv) - 1 - argv[::-1].index(flag) + 1]


def test_bash_syntax():
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "set -euo pipefail" in SCRIPT.read_text()
    assert os.access(SCRIPT, os.X_OK)


def test_fresh_run_argv_matches_spec_s4(fake, tmp_path):
    out = tmp_path / "root"
    r = run_script(fake, out, "ok", seeds="3")
    assert r.returncode == 0, r.stderr
    (a, ) = argvs(out, "rung1-s3")
    assert a[0] == "--train"
    spec = {
        "--map": "arena-duel",
        "--n-active-per-team": "1",
        "--round-time-ticks": "160",
        "--gamma": "0.99",
        "--seed": "3",
        "--checkpoint-interval": "10",
        "--eval-interval": "10",
        "--run-id": "rung1-s3",
        "--timesteps": "10000000",
        "--num_envs": "256",
        "--aim-entropy-bonus": "off",
        "--aim-log-std-max": "-2.9957",
        "--crouch-enabled": "0",
        "--reward-win-t-elimination": "1.0",
        "--reward-win-ct-elimination": "1.0",
        "--reward-win-ct-timeout": "0",
        "--reward-win-t-detonation": "0",
        "--reward-win-ct-defuse": "0",
        "--reward-kill": "0.3",
        "--reward-death": "0.1",
        "--reward-shot-penalty": "0",
        "--reward-ct-survival": "0",
        "--reward-inaction": "0.0005",
        "--pbrs-hp-weight": "0.002",
        "--pbrs-alive-weight": "0.3",
        "--pbrs-site-weight": "0",
        "--pbrs-bomb-progress-weight": "0",
        "--pbrs-nav-weight-t": "0",
        "--pbrs-nav-weight-ct": "0",
        "--checkpoint-dir": str(out / "rung1-s3")
    }
    for flag, val in spec.items():
        assert flag in a, flag
        assert _val(a, flag) == val, (flag, _val(a, flag))
    assert "--no-dead-run-abort" in a
    assert "--resume-run" not in a and "--no-self-play" not in a       # self-play stays ENABLED (§4)
    assert (out / "rung1-s3" / "DONE").exists()


@pytest.mark.parametrize("inherited", [None, "/a:/b"])
def test_default_train_cmd_puts_this_checkouts_src_first(fake, tmp_path, inherited):
    """RUNG1_TRAIN_CMD unset: the default (production) launch line, which no other test runs.

    A fake `uv` first on PATH records what `env ... uv run python -m cs2rl.train` hands
    it, then runs the fake trainer on the remaining argv.
    """
    (tmp_path / "bin").mkdir()
    uv = tmp_path / "bin" / "uv"
    uv.write_text(f'#!/bin/sh\necho "$UV_NO_SYNC|$PYTHONPATH|$1 $2 $3 $4" >> {tmp_path}/uv.log\n'
                  f'shift 4\nexec {sys.executable} {fake} "$@"\n')
    uv.chmod(0o755)
    e = {k: v for k, v in os.environ.items() if k not in ("RUNG1_TRAIN_CMD", "PYTHONPATH")}
    e.update(FAKE_MODE="ok", RUNG1_SEEDS="0", RUNG1_NEG_SEEDS="", PATH=f"{uv.parent}:{e['PATH']}")
    e.update({"PYTHONPATH": inherited} if inherited else {})
    r = subprocess.run(["bash", str(SCRIPT), str(tmp_path / "root")],
                       env=e,
                       cwd=tmp_path,
                       capture_output=True,
                       text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    pythonpath = ":".join(filter(None, [str(REPO_ROOT / "src"), inherited]))
    assert (tmp_path / "uv.log").read_text() == f"1|{pythonpath}|run python -m cs2rl.train\n"


def test_negative_control_flags(fake, tmp_path):
    out = tmp_path / "root"
    r = run_script(fake, out, "ok", seeds="", neg_seeds="1")
    assert r.returncode == 0, r.stderr
    assert not (out / "rung1-s1").exists()
    (a, ) = argvs(out, "rung1-neg-s1")
    assert _val(a, "--aim-entropy-bonus") == "on" and _val(a, "--aim-log-std-max") == "-0.69315"
    assert _val(a, "--seed") == "1" and _val(a, "--run-id") == "rung1-neg-s1"
    assert "--no-dead-run-abort" in a and _val(a, "--crouch-enabled") == "0"


def test_rung1_extra_is_appended_last_so_it_overrides(fake, tmp_path):
    out = tmp_path / "root"
    r = run_script(fake, out, "ok", extra="--timesteps 10240 --num_envs 16 --device cpu")
    assert r.returncode == 0, r.stderr
    (a, ) = argvs(out, "rung1-s0")
    assert a.count("--timesteps") == 2 and _val(a, "--timesteps") == "10240"
    assert _val(a, "--num_envs") == "16" and _val(a, "--device") == "cpu"


def test_done_marker_skips_seed(fake, tmp_path):
    """Legacy DONE (no budget recorded) skips unconditionally."""
    out = tmp_path / "root"
    (out / "rung1-s0").mkdir(parents=True)
    (out / "rung1-s0" / "DONE").write_text("")
    r = run_script(fake, out, "ok", extra="--timesteps 10240")
    assert r.returncode == 0, r.stderr
    assert argvs(out, "rung1-s0") == []
    assert "already finished" in r.stdout and "no budget recorded" in r.stdout


def test_done_records_budget(fake, tmp_path):
    out = tmp_path / "root"
    r = run_script(fake, out, "ok", extra="--timesteps 10240 --num_envs 16")
    assert r.returncode == 0, r.stderr
    assert (out / "rung1-s0" / "DONE").read_text().strip() == "10240"  # LAST --timesteps wins
    r = run_script(fake, out, "ok")                                    # default 10000000 > 10240
    assert r.returncode == 0, r.stderr
    assert (out / "rung1-s0" / "DONE").read_text().strip() == "10000000"


def test_done_records_budget_given_equals_form(fake, tmp_path):
    """`--timesteps=10240` is valid argparse; DONE must record 10240, not COMMON's 10000000."""
    out = tmp_path / "root"
    r = run_script(fake, out, "ok", extra="--timesteps=10240 --num_envs=16")
    assert r.returncode == 0, r.stderr
    (a, ) = argvs(out, "rung1-s0")
    assert "--timesteps=10240" in a
    assert (out / "rung1-s0" / "DONE").read_text().strip() == "10240"
    r = run_script(fake, out, "ok")    # default 10000000 > 10240 -> extends
    assert r.returncode == 0, r.stderr
    assert "extending --timesteps 10240 -> 10000000" in r.stdout
    assert (out / "rung1-s0" / "DONE").read_text().strip() == "10000000"


def test_done_records_last_of_mixed_flag_forms(fake, tmp_path):
    """Mixed spellings: whichever occurrence is LAST wins, in either direction."""
    out = tmp_path / "root"
    r = run_script(fake, out, "ok", extra="--timesteps 4096 --timesteps=10240")
    assert r.returncode == 0, r.stderr
    assert (out / "rung1-s0" / "DONE").read_text().strip() == "10240"
    out2 = tmp_path / "root2"
    r = run_script(fake, out2, "ok", extra="--timesteps=4096 --timesteps 20480")
    assert r.returncode == 0, r.stderr
    assert (out2 / "rung1-s0" / "DONE").read_text().strip() == "20480"


def _finished_dir(out, label, budget):
    d = out / label
    (d / label).mkdir(parents=True)
    (d / label / "model_000010.pt").write_bytes(b"m")
    (d / label / "trainer_state.pt").write_bytes(b"t")
    (d / "DONE").write_text(f"{budget}\n")
    return d


def test_done_with_smaller_budget_extends_via_resume(fake, tmp_path):
    out = tmp_path / "root"
    d = _finished_dir(out, "rung1-s0", 4096)
    r = run_script(fake, out, "ok", extra="--timesteps 10240")
    assert r.returncode == 0, r.stderr
    (a, ) = argvs(out, "rung1-s0")
    assert _val(a, "--resume-run") == str(d) and _val(a, "--timesteps") == "10240"
    assert "extending --timesteps 4096 -> 10240" in r.stdout
    assert (d / "DONE").read_text().strip() == "10240"


@pytest.mark.parametrize("recorded", [10240, 20480])
def test_done_with_equal_or_larger_budget_skips(fake, tmp_path, recorded):
    out = tmp_path / "root"
    _finished_dir(out, "rung1-s0", recorded)
    r = run_script(fake, out, "ok", extra="--timesteps 10240")
    assert r.returncode == 0, r.stderr
    assert argvs(out, "rung1-s0") == []
    assert f"already finished at --timesteps {recorded}" in r.stdout


def test_done_with_smaller_budget_but_no_checkpoint_set_fails(fake, tmp_path):
    out = tmp_path / "root"
    (out / "rung1-s0").mkdir(parents=True)
    (out / "rung1-s0" / "DONE").write_text("4096")
    r = run_script(fake, out, "ok", extra="--timesteps 10240")
    assert r.returncode == 1
    assert argvs(out, "rung1-s0") == [] and "cannot extend" in r.stderr


def test_train_log_captures_both_streams_and_nonzero_exit_still_retries(fake, tmp_path):
    out = tmp_path / "root"
    r = run_script(fake, out, "crash_after_ckpt")
    assert r.returncode == 0, r.stderr
    log = (out / "rung1-s0" / "train.log").read_text()
    # fresh leg (call=1, exit 1) AND the resume leg (call=2) are both appended, both streams.
    assert "fake-train stdout call=1" in log and "fake-train stderr call=1" in log
    assert "fake-train stdout call=2" in log and "fake-train stderr call=2" in log
    assert "resume attempt 1" in r.stdout and "fake-train stdout call=2" in r.stdout # still streams
    assert len(argvs(out, "rung1-s0")) == 2


def test_prepopulated_checkpoint_set_without_done_is_resumed_not_skipped(fake, tmp_path):
    """Binding ruling: dust2_policy.pt is NOT a done marker."""
    out = tmp_path / "root"
    d = out / "rung1-s0"
    (d / "rung1-s0").mkdir(parents=True)
    (d / "dust2_policy.pt").write_bytes(b"p")
    (d / "rung1-s0" / "model_000010.pt").write_bytes(b"m")
    (d / "rung1-s0" / "trainer_state.pt").write_bytes(b"t")
    r = run_script(fake, out, "ok")
    assert r.returncode == 0, r.stderr
    (a, ) = argvs(out, "rung1-s0")
    assert _val(a, "--resume-run") == str(d)
    assert "resume attempt 1" in r.stdout
    assert (d / "DONE").exists()


def test_crash_after_checkpoint_retries_with_identical_argv_plus_resume(fake, tmp_path):
    out = tmp_path / "root"
    r = run_script(fake, out, "crash_after_ckpt", extra="--timesteps 10240 --num_envs 16")
    assert r.returncode == 0, r.stderr
    first, second = argvs(out, "rung1-s0")
    assert "--resume-run" not in first
    # The retry leg is the ORIGINAL argv (incl. --map, --seed, gammas, RUNG1_EXTRA) + --resume-run.
    assert second == first + ["--resume-run", str(out / "rung1-s0")]
    assert _val(second, "--map") == "arena-duel" and _val(second, "--seed") == "0"
    assert (out / "rung1-s0" / "DONE").exists()


def test_crash_before_checkpoint_is_not_resumable(fake, tmp_path):
    out = tmp_path / "root"
    r = run_script(fake, out, "crash_no_ckpt")
    assert r.returncode == 1
    assert len(argvs(out, "rung1-s0")) == 1
    assert "not resumable" in r.stderr and "FAILED: rung1-s0" in r.stderr
    assert not (out / "rung1-s0" / "DONE").exists()


def test_max_retries_exhausted_does_not_abort_sweep(fake, tmp_path):
    out = tmp_path / "root"
    r = run_script(fake, out, "always_fail_after_ckpt", seeds="0 1", neg_seeds="0", max_retries=2)
    assert r.returncode == 1
    for label in ("rung1-s0", "rung1-s1", "rung1-neg-s0"):
        calls = argvs(out, label)
        assert len(calls) == 3, label  # 1 fresh + 2 resumes
        assert [("--resume-run" in c) for c in calls] == [False, True, True]
        assert not (out / label / "DONE").exists()
    assert "FAILED: rung1-s0 rung1-s1 rung1-neg-s0" in r.stderr


def test_common_argv_is_accepted_by_train_py_dump_config(tmp_path):
    """F19 (final review B6): the ONLY link in the launch chain with no local
    test was COMMON itself against the REAL argparse — every other test here
    drives a fake trainer. Parse `COMMON=(...)` out of the script, drop
    --train, run train.py --dump-config with it (no torch, ~1-2 s for
    `from cs2rl.env import map`) and pin the spec §4 values config.json must carry, incl. the
    participating-budget arithmetic (total = 5x participating at n_active=1).
    PITFALL: a renamed reward key / CLI flag otherwise only dies at argparse on
    a booked GPU box. sys.executable + cwd=REPO_ROOT like tests/test_train_cli.py."""
    import re
    import shlex

    text = SCRIPT.read_text()
    m = re.search(r"^COMMON=\((.*?)\)\s*$", text, re.S | re.M)
    assert m, "COMMON=( ... ) array not found in run_rung1.sh"
    common = shlex.split(m.group(1))
    assert common[0] == "--train"
    common = common[1:]

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    r = subprocess.run([
        sys.executable, "-m", "cs2rl.train", "--dump-config", *common, "--seed", "0",
        "--checkpoint-dir",
        str(ckpt)
    ],
                       capture_output=True,
                       text=True,
                       timeout=300,
                       cwd=REPO_ROOT)
    assert r.returncode == 0, f"stdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-2000:]}"
    cfg = json.loads((ckpt / "config.json").read_text())

    assert cfg["env"] == "cs2-arena-duel"
    assert cfg["seed"] == 0
    assert cfg["n_active_per_team"] == 1
    assert cfg["pin_pitch"] == 1
    assert cfg["crouch_enabled"] == 0
    assert cfg["round_time_ticks"] == 160
    assert cfg["gamma"] == cfg["pbrs_gamma"] == 0.99
    assert cfg["participating_timesteps"] == 10_000_000
    assert cfg["total_timesteps"] == 50_000_000
    assert cfg["reward_kill"] == 0.3
    assert cfg["pbrs_site_weight"] == 0.0
    assert cfg["checkpoint_interval"] == cfg["eval_interval"] == 10


@pytest.mark.parametrize("arm", ["rung1-s", "rung1-neg-s"])
def test_arm_argv_is_accepted_by_train_py_dump_config(tmp_path, arm):
    """Sweep-day regression (2026-08-30): the neg-control leg passed
    --aim-log-std-max -0.6931, which is ABOVE log 0.5 by 5e-5, so train.py
    rejected it in build_policy() — 30 s into every retry, AFTER all 5 treatment
    seeds had run, and invisible to the COMMON-only test above. Parse each
    arm's `run_seed "<arm>$s" "$s" <flags> \\` line out of the script and run
    the REAL train.py --dump-config with COMMON + those flags; the σ-cap check
    now lives above the --dump-config exit, so a bad literal fails here."""
    import re
    import shlex

    text = SCRIPT.read_text()
    common = shlex.split(re.search(r"^COMMON=\((.*?)\)\s*$", text, re.S | re.M).group(1))[1:]
    m = re.search(r'run_seed "' + re.escape(arm) + r'\$s" "\$s" (.*?) \\$', text, re.M)
    assert m, f"run_seed line for {arm} not found"
    arm_flags = shlex.split(m.group(1))
    assert "--aim-log-std-max" in arm_flags

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    r = subprocess.run([
        sys.executable, "-m", "cs2rl.train", "--dump-config", *common, *arm_flags, "--seed", "0",
        "--checkpoint-dir",
        str(ckpt)
    ],
                       capture_output=True,
                       text=True,
                       timeout=300,
                       cwd=REPO_ROOT)
    assert r.returncode == 0, f"stdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-2000:]}"
    cfg = json.loads((ckpt / "config.json").read_text())
    assert cfg["aim_log_std_max"] == float(arm_flags[arm_flags.index("--aim-log-std-max") + 1])


def test_dump_config_rejects_out_of_range_aim_log_std_max(tmp_path):
    """The literal that burned the sweep must now die at --dump-config."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    r = subprocess.run([
        sys.executable, "-m", "cs2rl.train", "--dump-config", "--map", "arena-duel",
        "--aim-log-std-max", "-0.6931", "--seed", "0", "--checkpoint-dir",
        str(ckpt)
    ],
                       capture_output=True,
                       text=True,
                       timeout=300,
                       cwd=REPO_ROOT)
    assert r.returncode != 0
    assert "aim_log_std_max" in r.stderr and not (ckpt / "config.json").exists()
