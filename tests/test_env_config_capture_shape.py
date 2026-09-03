"""Shape self-check for `tests/fixtures/env_config_pre_165b.json` (spec Phase B §8.2).

WHY THIS FILE EXISTS. That fixture is PR B2's entire oracle: B2 rewrites the six
role builders and `tests/test_env_factory.py` compares the result against this
JSON. B1 captures and commits it and then never reads it, so between the two PRs
a truncated, half-written or hand-edited fixture has NOTHING watching it — B2
would be measured against a broken oracle, and would look green while migrating
the wrong payload.

WHAT IT CHECKS: the file's SHAPE, against a hand-written expectation, plus the
commit it was captured at. Not the values — the values ARE the oracle, and the
only meaningful check on them is a re-capture, which is a B1-only gate (the B1
plan's Task 8). The role/scenario table and the seven `(site, enclosing)` pairs
are written out below rather than imported from
`tests/capture_env_config_pre_165b.py`, on purpose: a check that imports the
producer's own constants compares the fixture to itself and stays green after an
edit that breaks both.

WHAT IT MUST NOT DO: re-run the capture. B2 legitimately changes the call
sources this fixture records, so a re-capture assertion here would fail in B2 by
design and be deleted for the wrong reason.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# I001 is suppressed, not fixed: the import has to follow the sys.path insert
# above, and ruff's isort wants it in the block at the top (same waiver as
# tests/test_env_config.py:21).
from env_config import KNOB_FIELDS, REWARD_FIELDS      # noqa: E402, I001

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "env_config_pre_165b.json"

# The commit the capture ran at: this branch's Task 1 tip, the last commit before
# #165 Phase B touches any role builder. Pinned so a re-capture cannot pass
# unnoticed — see test_format_tag_and_provenance for why that matters.
PRE_165B_CAPTURE_COMMIT = "1e0da96de7c115f48653570141f21daf433d6e6b"

# make_env's six runtime inputs. Every recorded call must carry all six: today's
# shim forwards them unconditionally, so a scenario missing one means the capture
# recorded a call shape nobody makes.
RUNTIME_NAMES = frozenset(
    {"seed", "team_spirit", "auto_reset", "buf", "map_data", "include_step_stats_in_info"})

# role -> its scenarios. Thirteen rows over six roles, hand-written: a role that
# loses a scenario, or gains one nobody decided on, fails here rather than in B2.
EXPECTED_SCENARIOS = {
    "train": {"per_env_seed_wins", "pufferlib_seed_no_knobs", "seed_none_becomes_zero"},
    "eval": {"default_args", "non_default_args", "symmetrize_requested"},
    "harness": {"seed_none_becomes_zero", "explicit_seed_zero_kept"},
    "eval_legacy": {"bare_defaults", "explicit_seed"},
    "smoke": {"fixed_seed"},
    "external": {"both_passed", "wrapper_defaults"},
}

# The seven build_env_for construction sites as (site, enclosing) — the same
# census the capture asserts against the AST. Sites are shared: thirteen
# scenarios over seven distinct pairs.
EXPECTED_SITES = {
    ("src/train.py", "build_env_factory.env_factory"),
    ("src/train.py", "train"),
    ("src/train.py", "load_policy_from_checkpoint"),
    ("src/train.py", "evaluate_checkpoint"),
    ("src/train.py", "smoke_test"),
    ("src/train.py", "make_env"),
    ("src/train_test_harness.py", "_build_trainer_for_test.env_factory"),
}

# The one row whose input_config is deliberately NOT what make_env received: the
# eval env runs raw rewards BY RULE, so a run that asked for --reward-symmetrize
# hands build_env_for True while make_env still sees False.
SYMMETRIZE_ROW = ("eval", "symmetrize_requested")


def _rows():
    """(role, scenario dict) for every captured scenario, in file order."""
    d = json.loads(FIXTURE.read_text())
    return d, [(role, cap) for role, caps in d["roles"].items() for cap in caps]


def test_format_tag_and_provenance():
    """A stale or hand-written file fails on the tag, not on a KeyError deep in B2."""
    d, _ = _rows()
    assert d["_provenance"]["format"] == "cs2rl-env-config-capture-v1"
    assert set(d["_provenance"]["runtime_names"]) == RUNTIME_NAMES
    assert d["_provenance"]["captured_at_commit"] == PRE_165B_CAPTURE_COMMIT, (
        "the fixture was regenerated at a different commit. Re-running "
        "tests/capture_env_config_pre_165b.py --capture to turn a red Phase B test "
        "green is exactly how this oracle becomes a mirror: the new capture records "
        "post-migration payloads, and B2's test_env_factory.py then checks the new "
        "builders against themselves while staying green. The format tag and the "
        "shape would not move, so this assertion is the only thing that catches it. "
        "The ONLY legitimate way to change this fixture is to change "
        "PRE_165B_CAPTURE_COMMIT deliberately, in a commit whose message explains "
        "which construction site changed and why.")


def test_thirteen_scenarios_over_six_roles_and_seven_sites():
    """The census, from a hand-written table.

    Equality both ways: a scenario that vanished (a truncated write) and one
    that appeared without a decision both fail. The site set is the same census
    the capture asserts against the AST, restated here so a fixture written by
    a capture whose own assert was weakened still fails.
    """
    d, rows = _rows()
    assert {
        role: {c["scenario"]
               for c in caps}
        for role, caps in d["roles"].items()
    } == EXPECTED_SCENARIOS
    assert len(rows) == 13
    assert {(c["site"], c["enclosing"]) for _, c in rows} == EXPECTED_SITES


def test_every_scenario_records_all_six_runtime_names():
    """Runtime inputs are what B2 must keep passing positionally-by-name.

    `runtime_kwargs` is the six-name subset and `make_env_kwargs` is the whole
    call; the subset relation is asserted rather than assumed, because a capture
    that filtered with a stale RUNTIME_NAMES would silently record five.
    """
    _, rows = _rows()
    for role, cap in rows:
        where = f"{role}/{cap['scenario']}"
        assert set(cap["runtime_kwargs"]) == RUNTIME_NAMES, where
        assert RUNTIME_NAMES <= set(cap["make_env_kwargs"]), where
        assert cap["call_source"].startswith("build_env_for("), where


def test_both_configs_carry_every_env_config_field():
    """`expected_config` / `input_config` are FULL asdict dumps, not sparse diffs.

    B2 compares them field by field against a live EnvConfig. A fixture holding
    only the fields that happened to be non-default would make every absent
    field unchecked, which is exactly the silent pass this oracle exists to
    prevent — so the key sets are pinned against the dataclass itself.
    """
    _, rows = _rows()
    expected_keys = {"rewards"} | set(KNOB_FIELDS)
    for role, cap in rows:
        for key in ("expected_config", "input_config"):
            cfg = cap[key]
            where = f"{role}/{cap['scenario']} {key}"
            assert set(cfg) == expected_keys, where
            assert set(cfg["rewards"]) == set(REWARD_FIELDS), where


def test_symmetrize_is_the_only_row_where_the_two_configs_differ():
    """One deliberate difference in the whole fixture, and it is the eval rule.

    `input_config` is what B2's test hands `build_env_for`; `expected_config` is
    what make_env must then receive. They are equal everywhere except the eval
    env's raw-reward rule, so pinning "exactly one row, exactly one field" is
    what makes that rule an assertion instead of a comment. A second differing
    row means the capture recorded a translation nobody declared — report it,
    do not relax this test.
    """
    _, rows = _rows()
    differing = {(role, cap["scenario"])
                 for role, cap in rows if cap["expected_config"] != cap["input_config"]}
    assert differing == {SYMMETRIZE_ROW}, differing
    cap = next(c for r, c in rows if (r, c["scenario"]) == SYMMETRIZE_ROW)
    fields = {
        k
        for k in cap["expected_config"] if cap["expected_config"][k] != cap["input_config"][k]
    }
    assert fields == {"reward_symmetrize"}, fields
    assert cap["expected_config"]["reward_symmetrize"] is False
    assert cap["input_config"]["reward_symmetrize"] is True
