import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "src" / "train.py"


# gh#95: 600s (not 180s) because the --smoke subprocess competes with a live GPU
# training run on this box — the flake was CPU/GPU contention, not runtime growth.
def run_train_command(*args, timeout=600):
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
            "--reward-ct-survival",                    # A1 arm (spec §5)
            "--reward-win-ct-timeout",                 # A1b arm
            "--pbrs-nav-weight-t",                     # non-`reward_`-prefixed weight
            "--reward-symmetrize",                     # A2 arm
            "--tct-split-heads",                       # Batch 7 heads split (spec 2026-08-13)
            "--tct-split-trunk",                       # T/CT actor-trunk split (spec 2026-08-15)
            "--resume-run",                            # R0-C (Task 7)
            "--run-id",
            "--checkpoint-interval",
            "--seed",                                  # R0-D (Task 8)
            "--pin-pitch",                             # R0-E.2 (Task 9)
            "--crouch-enabled",
            "--jump-enabled",                          # Rung 1a T2b (stance parity)
            "--opponent",                              # Rung 1a T3 (statue opponent)
            "--aim-entropy-bonus",                     # R0-E.3/4 (Task 10)
            "--aim-log-std-max",
            "--n-active-per-team",                     # Rung 0 parking (Task 4)
            "--round-time-ticks",                      # R0-G (Task 11)
            "--laser-range",
            "--max-turn-speed",
            "--map",                                   # R0-H (Task 12)
            "--eval-interval",                         # R0-I (Task 13)
            "--gamma",                                 # R0-J (Task 14)
            "--pbrs-gamma",
    ):
        assert flag in result.stdout, f"{flag} missing from --help output"

    for legacy_flag in ("--num_workers", "--num_envs_per_worker", "--train_dir"):
        assert legacy_flag not in result.stdout, f"{legacy_flag} should not be exposed anymore"


def test_dump_config_without_checkpoint_dir_uses_default(tmp_path):
    """R0-C made --checkpoint-dir default=None; --dump-config must still resolve
    it to CHECKPOINTS_DIR instead of crashing on Path(None). CHECKPOINTS_DIR is
    cwd-relative (src/paths.py: Path("outputs") / "checkpoints"), so running
    with cwd=tmp_path keeps the write out of the repo. sys.executable (not
    `uv run`) because uv would not find the project from a tmp cwd."""
    r = subprocess.run([sys.executable, str(TRAIN_SCRIPT), "--dump-config"],
                       capture_output=True,
                       text=True,
                       timeout=120,
                       cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "outputs" / "checkpoints" / "config.json").exists(), r.stdout


def test_dump_config_writes_json(tmp_path):
    """--dump-config writes <checkpoint_dir>/config.json and exits without training.

    Runs on sys.executable via run_train_command, never `uv run`: a syncing
    `uv run` from a worktree that borrows main's .venv re-pointed the shared
    editable install at the worktree and rebuilt main's binding .so under this
    very pytest process, which then segfaulted (#220).
    The whole point of --dump-config is zero side-effects: no torch import,
    no env spin-up — so it must return quickly (the MapData is built above the
    exit since Task 12: config.json carries the geometry-resolved pin_pitch).
    120s is headroom for gh#95 contention (see _dump_config), not an expected runtime.
    """
    import json

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    result = run_train_command("--dump-config", "--checkpoint-dir", str(ckpt_dir), timeout=120)
    assert result.returncode == 0, f"stderr: {result.stderr}"

    config_path = ckpt_dir / "config.json"
    assert config_path.exists()

    with config_path.open() as f:
        config = json.load(f)
    for key in ("learning_rate", "gamma", "clip_coef", "batch_size"):
        assert key in config, f"missing key {key}"
    assert isinstance(config["batch_size"], int) and config["batch_size"] > 0


def _dump_config(tmp_path, *extra_args):
    """Run --dump-config on sys.executable and return the parsed
    config.json — the four warmstart keys must round-trip through the REAL
    argparse surface, not a hand-built Namespace (which would only exercise
    build_train_config's getattr fallbacks and hide a missing add_argument).

    Not `uv run`, for the #220 reason in test_dump_config_writes_json.
    120s (not 60s) for the same reason as gh#95 on run_train_command: this
    subprocess competes with a live GPU training run on this box, and the
    failure mode was contention, not runtime growth."""
    import json

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    r = run_train_command("--dump-config", "--checkpoint-dir", str(ckpt), *extra_args, timeout=120)
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


def test_reward_weight_config_keys_default_to_make_env_values(tmp_path):
    """Every threaded weight lands in config.json at its make_env default.

    Together with tests/test_env_config.py (which pins the declaration
    itself, on RewardWeights) this is the "unflagged run is identical to
    today" guarantee, verified through the REAL argparse surface: a typo'd
    dest= or a missing add_argument would leave the key at the getattr
    fallback and could not be caught by a hand-built Namespace.
    """
    from env_config import RewardWeights

    cfg = _dump_config(tmp_path)
    for name, default in RewardWeights().as_dict().items():
        assert name in cfg, f"{name} missing from config.json"
        assert cfg[name] == default, f"{name}: {cfg[name]} != {default}"
    assert cfg["reward_symmetrize"] is False


def test_reward_weight_cli_overrides_round_trip(tmp_path):
    """Representative overrides + the symmetrize flag survive CLI → config.json.

    Uses the two weights the A/B actually moves (spec §5) plus one PBRS weight
    (proving the six non-`reward_`-prefixed kwargs are wired too) and one
    per-outcome win magnitude.
    """
    cfg = _dump_config(tmp_path, "--reward-ct-survival", "0.0", "--reward-win-ct-timeout", "3.0",
                       "--pbrs-nav-weight-t", "0.07", "--reward-win-t-detonation", "6.5",
                       "--reward-symmetrize")
    assert cfg["reward_ct_survival"] == 0.0
    assert cfg["reward_win_ct_timeout"] == 3.0
    assert cfg["pbrs_nav_weight_t"] == 0.07
    assert cfg["reward_win_t_detonation"] == 6.5
    assert cfg["reward_symmetrize"] is True
    # untouched neighbours keep their defaults (no accidental global override)
    assert cfg["reward_kill"] == 0.3


FIXTURE_DUMP_CONFIG = REPO_ROOT / "tests" / "fixtures" / "dump_config_pre_165.json"

# The commit the fixture was captured at: main's Phase-A merge, which is also this
# branch's base and therefore the last commit BEFORE #165 Phase B rewrites the code
# that produces config.json. Pinned here so a re-capture cannot pass unnoticed.
PRE_165_CAPTURE_COMMIT = "54d7da01b7df28414dbcda346a2f8c4f6d2b2ee7"


@pytest.mark.parametrize("arm", ["default", "non_default"])
def test_dump_config_matches_the_pre_165_fixture(tmp_path, arm):
    """config.json is byte-identical to the pre-#165 capture (spec Phase B R7).

    This is the provenance half of "not one value moves": build_train_config
    stops restating six keys and starts merging EnvConfig.to_config_dict(), and
    scripts/run_experiment.py hashes this dict as the experiment fingerprint, so
    a single reordered or retyped value silently invalidates every future
    comparison against past runs. The .pt gate cannot see it — that command sets
    no --reward-*/knob flag at all, which is why the non_default arm exists.

    Compared as `json.dumps(..., sort_keys=True, indent=2, default=str)`, the
    exact spelling train.py writes, so a float that became an int fails here.
    PITFALL: `data_dir` is the --checkpoint-dir string verbatim and can never
    match a fixture captured elsewhere; it is asserted to be THIS test's own
    directory and then replaced by the same placeholder the capture stored.
    """
    import json

    fixture = json.loads(FIXTURE_DUMP_CONFIG.read_text())
    assert fixture["_provenance"]["format"] == "cs2rl-dump-config-capture-v1"
    assert fixture["_provenance"]["captured_at_commit"] == PRE_165_CAPTURE_COMMIT, (
        "the fixture was regenerated at a different commit. Re-running "
        "tests/capture_dump_config_pre_165.py --capture to turn a red test green is "
        "exactly how this oracle becomes a mirror: the new capture records "
        "post-migration values, and the comparison below then checks the new code "
        "against itself while staying green. The format tag would not move, so this "
        "assertion is the only thing that catches it. The ONLY legitimate way to "
        "change this fixture is to change PRE_165_CAPTURE_COMMIT deliberately, in a "
        "commit whose message explains which config key moved and why.")
    entry = fixture["arms"][arm]

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    r = subprocess.run(
        [sys.executable,
         str(TRAIN_SCRIPT), *entry["argv"], "--checkpoint-dir",
         str(ckpt)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300)
    assert r.returncode == 0, f"stdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-3000:]}"

    cfg = json.loads((ckpt / "config.json").read_text())
    assert cfg["data_dir"] == str(ckpt), "data_dir is no longer the --checkpoint-dir string"
    cfg["data_dir"] = "<checkpoint_dir>"

    def _dumps(d):
        return json.dumps(d, sort_keys=True, indent=2, default=str)

    assert _dumps(cfg) == _dumps(entry["config"]), (
        f"config.json for the {arm!r} arm changed against tests/fixtures/"
        "dump_config_pre_165.json. Regenerate the fixture ONLY if a key changed on "
        "purpose — otherwise this is the provenance regression it exists to catch.")


def test_tag_diagnostic_config_keys(tmp_path):
    """TAG flags land in config.json (provenance) — spec 2026-08-13 §4.1.

    Both keys overridden together: for untouched flags argparse's default
    equals build_train_config's getattr fallback, so a typo'd dest= would
    pass silently (same rationale as the warmstart key test above).
    """
    cfg = _dump_config(tmp_path)
    assert cfg["tag_diagnostic"] is False
    assert cfg["tag_every"] == 5

    cfg = _dump_config(tmp_path, "--tag-diagnostic", "--tag-every", "2")
    assert cfg["tag_diagnostic"] is True
    assert cfg["tag_every"] == 2


def test_tct_split_heads_config_key(tmp_path):
    """Batch 7 flag lands in config.json (provenance) — spec 2026-08-13 §2.

    NOTE what this key is and is not: it records the FLAG AS PASSED, not the
    architecture the run actually built. A flag-less crash-resume of a split
    run correctly writes false here while running a split policy — which is
    exactly why the TAG analyzer keys off the per-epoch split/active metric
    instead of this file (spec §3.4).
    """
    cfg = _dump_config(tmp_path)
    assert cfg["tct_split_heads"] is False

    cfg = _dump_config(tmp_path, "--tct-split-heads")
    assert cfg["tct_split_heads"] is True


def test_tct_split_trunk_config_key(tmp_path):
    """Trunk-split flag lands in config.json (provenance) — spec 2026-08-15.

    Same contract as test_tct_split_heads_config_key: this key records the
    FLAG AS PASSED, not the architecture the run actually built. A flag-less
    crash-resume of a trunk-split run correctly writes false here while
    running a split-trunk policy — which is exactly why the TAG analyzer
    keys off the per-epoch split/trunk_active metric instead of this file.
    """
    cfg = _dump_config(tmp_path)
    assert cfg["tct_split_trunk"] is False

    cfg = _dump_config(tmp_path, "--tct-split-trunk")
    assert cfg["tct_split_trunk"] is True


def test_opponent_config_key_and_budget(tmp_path):
    """Rung 1a T3 test (iv): --opponent reaches config.json through the REAL
    argparse surface, and the budget it selects travels with it.

    Both halves matter. The key is provenance — months later the run dir alone
    has to say whether the opponent was a statue — and total_timesteps is the
    number PufferLib turns into total_epochs and the cosine-LR T_max, so a
    dumped config that carries `noop` with the `self` horizon would fingerprint
    as the experiment we meant while running half of it.
    """
    cfg = _dump_config(tmp_path)
    assert cfg["opponent"] == "self"
    assert cfg["total_timesteps"] == cfg["participating_timesteps"]    # 5v5 default

    noop = _dump_config(tmp_path, "--opponent", "noop", "--no-self-play", "--n-active-per-team",
                        "1", "--timesteps", "1000000")
    assert noop["opponent"] == "noop"
    assert noop["participating_timesteps"] == 1_000_000
    assert noop["total_timesteps"] == 10_000_000


def test_opponent_noop_without_no_self_play_is_refused_at_startup(tmp_path):
    """Rung 1a T3 test (iii). The guard sits ABOVE the --dump-config exit, so
    the Modal / run_rung1 fingerprint step refuses the launch in milliseconds
    instead of the run discovering at epoch 50 that maybe_switch_teams moved
    the statue to the hero's side of a spawn-asymmetric map. Nothing is
    written: the error precedes even the map build."""
    r = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--dump-config", "--opponent", "noop"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path)
    assert r.returncode != 0, r.stdout[-2000:]
    assert "--no-self-play" in r.stderr, r.stderr[-2000:]
    assert not (tmp_path / "outputs").exists(), "the guard must fire before any side effect"


def test_opponent_flag_declared_with_both_modes():
    """Source-scan pin, same rationale as test_cli_flags_declared_default_none
    in tests/test_env_knobs.py (the parser is built inline under
    `if __name__ == "__main__"` and cannot be imported): the flag must offer
    both modes and default to the historical one.

    TWO FILES since the post-rung1a refactor (2026-08-31): the parser (and so
    the `choices=OPPONENT_MODES` reference) stays in src/train.py, while the
    OPPONENT_MODES tuple itself moved to src/train_config.py. Both halves are
    pinned — a `choices=` naming a vocabulary that no longer holds both modes
    is exactly the silent narrowing this test exists to catch."""
    import re

    src = TRAIN_SCRIPT.read_text()
    m = re.search(r'add_argument\(\s*"--opponent",(.*?)\)\n', src, re.S)
    assert m, "--opponent not declared in train.py"
    body = m.group(1)
    assert "choices=OPPONENT_MODES" in body and 'default="self"' in body, body
    assert 'dest="opponent"' in body, body
    config_src = (REPO_ROOT / "src" / "train_config.py").read_text()
    assert 'OPPONENT_MODES = ("self", "noop")' in config_src


def test_train_smoke_returns_zero():
    result = run_train_command("--smoke")
    assert result.returncode == 0, (
        f"train.py --smoke failed (rc={result.returncode}):\n"
        f"STDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}")
    assert "[Smoke] Completed" in result.stdout, result.stdout
    assert "steps/sec" in result.stdout, result.stdout
