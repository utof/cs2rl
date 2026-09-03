"""Freeze `--dump-config`'s config.json at 54d7da0, before #165 Phase B edits it.

WHY THIS FILE EXISTS. Phase B rewires build_train_config to
`env_config_from_args(args).to_config_dict()` and deletes six literal keys from
its dict (spec 2026-09-03 Phase B §1). The claim being made is "not one value
moves", and the only honest oracle for that is a dump taken BEFORE the rewrite:
a fixture transcribed afterwards would compare the new code to itself. This
script is that dump and `tests/fixtures/dump_config_pre_165.json` is its frozen
output; `tests/test_train_cli.py::test_dump_config_matches_the_pre_165_fixture`
is the assertion.

TWO ARMS, because the gate argv sets no reward or knob flag (spec §5): `default`
is the byte-identity gate's own argv minus `--train` plus `--dump-config`, and
`non_default` adds one reward weight, one knob, one R0-G knob and a broken
gamma pairing — the four channels Phase B rewires. A one-arm fixture would stay
green if env_config_from_args silently ignored every CLI value.

ONE DOCUMENTED SUBSTITUTION (R7). `--dump-config` writes `data_dir` = the
`--checkpoint-dir` string verbatim, so a test running in its own tmp dir can
never be byte-equal to a fixture captured elsewhere. The capture asserts the
key holds its own directory and then stores the placeholder `<checkpoint_dir>`;
the test asserts the same thing about ITS directory and substitutes before
comparing. Everything else in the dict is argv-determined: `env` is
`cs2-simple` (set by the R0-H map block above the --dump-config exit, not by
`--map`'s own default) and `device` is the `cpu` placeholder the dump path sets.

REGENERATING is legitimate only when a config key changes ON PURPOSE, and never
to make a failing migration go green — that deletes the only evidence the
migration preserved the dict.

    UV_NO_SYNC=1 uv run python tests/capture_dump_config_pre_165.py --capture

Deliberately NOT named `test_*`: pytest must not collect it.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "src" / "train.py"
FIXTURE = Path(__file__).parent / "fixtures" / "dump_config_pre_165.json"

# Bumped if the fixture's own shape changes, so a stale file fails on the tag
# rather than on a confusing KeyError inside a comparison.
CAPTURE_FORMAT = "cs2rl-dump-config-capture-v1"
CHECKPOINT_DIR_PLACEHOLDER = "<checkpoint_dir>"

# The byte-identity gate's command shape (tests/test_seed_reproducible.py::_run,
# parent spec §5) with --train dropped and --dump-config added.
GATE_ARGV = ("--dump-config", "--device", "cpu", "--vec-backend", "serial", "--num_envs", "16",
             "--no-self-play", "--no-dead-run-abort", "--seed", "3", "--timesteps", "20480",
             "--save_every_sec", "100000", "--run-id", "rid")
ARMS = {
    "default": (),
    "non_default": ("--reward-ct-survival", "0.0", "--n-active-per-team", "2", "--round-time-ticks",
                    "900", "--pbrs-gamma", "0.99"),
}


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                          text=True).stdout.strip()


def _require_clean_src():
    dirty = _git("status", "--porcelain", "--", "src")
    if dirty:
        raise SystemExit("src/ is dirty; the whole point of this capture is that it records "
                         f"the PRE-migration behaviour:\n{dirty}")


def dump(argv, checkpoint_dir) -> dict:
    """Run train.py --dump-config and return its config.json, placeholder applied."""
    r = subprocess.run(
        [sys.executable,
         str(TRAIN_SCRIPT), *argv, "--checkpoint-dir",
         str(checkpoint_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300)
    if r.returncode != 0:
        raise SystemExit(f"--dump-config failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
    cfg = json.loads((Path(checkpoint_dir) / "config.json").read_text())
    assert cfg["data_dir"] == str(checkpoint_dir), (
        f"data_dir is {cfg['data_dir']!r}, not the --checkpoint-dir string; the one "
        "documented substitution no longer holds and R7 needs revisiting")
    cfg["data_dir"] = CHECKPOINT_DIR_PLACEHOLDER
    return cfg


def capture():
    _require_clean_src()
    out = {
        "_provenance": {
            "format": CAPTURE_FORMAT,
            "captured_at_commit": _git("rev-parse", "HEAD"),
            "why": "pre-#165-Phase-B config.json; a dump taken after the rewrite would "
            "compare build_train_config to itself",
            "substitution": f"data_dir is stored as {CHECKPOINT_DIR_PLACEHOLDER}",
        },
        "arms": {},
    }
    for name, extra in ARMS.items():
        argv = [*GATE_ARGV, *extra]
        with tempfile.TemporaryDirectory() as td:
            out["arms"][name] = {"argv": argv, "config": dump(argv, td)}
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(out, sort_keys=True, indent=2) + "\n")
    print(f"wrote {FIXTURE} ({len(out['arms'])} arms)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--capture", action="store_true", required=True)
    p.parse_args()
    capture()
