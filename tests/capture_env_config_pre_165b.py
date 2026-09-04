"""Record what `make_env` RECEIVES at every build_env_for site, before #165 Phase B.

WHY THIS FILE EXISTS. Phase B PR B2 changes every role builder from legacy
keyword arguments to one `config: EnvConfig` (spec 2026-09-03 Phase B §3). The
oracle for that migration is a per-role comparison of what actually reaches
`make_env`, before vs after, and the "before" half cannot be transcribed once
the builders are typed — that compares the factory to itself. So the capture is
a SEPARATE COMMIT that lands before any migration edit, in PR B1, and PR B2's
`tests/test_env_factory.py` reads it.

FROZEN AS OF PR B2. `capture()` now REFUSES to run — its first statement raises
SystemExit. B2 deleted the two args helpers whose names the recorded `eval`
namespace binds below, so a re-capture would either die on that missing
attribute or — far worse — be "repaired" to bind the migrated resolver instead
and then SUCCEED, overwriting this file's own output with a measurement of the
code under test, which is the one thing the oracle must never be. The script is
kept, not deleted: it is the auditable record of how the fixture was produced,
and `tests/test_env_config_capture_shape.py` points readers at it.
tests/test_env_config_capture_shape.py::test_the_capture_script_refuses_to_run
pins the refusal, because nothing imports this module and pytest collects
nothing from it — deleting the raise would otherwise be invisible.

WHAT IS RECORDED per scenario: the current `build_env_for(...)` call text; the
scenario's args/bindings; the kwargs the stub at `c_env.cs2_env.make_env`
received; the six-name runtime subset; `expected_config`, an `asdict` of the
EnvConfig those non-runtime kwargs describe; and `input_config`, the config the
post-migration test will hand `build_env_for`. For `eval` those two DIFFER when
the scenario sets `--reward-symmetrize`, because the eval env deliberately runs
raw rewards — that difference is the capture-side pin of the
`.replace(reward_symmetrize=False)` rule.

WHY expected_config IS COMPUTED HERE, not in the test: computing it at test time
would have to call `EnvConfig.from_legacy_kwargs`, a method issue #173 deletes,
and the oracle would rot with it. It is built here by direct construction —
splitting the non-runtime names over REWARD_FIELDS / KNOB_FIELDS — and stored as
a plain dict.

HOW EACH SCENARIO IS DRIVEN. The `train` role goes through the REAL
`build_train_env_factory(Namespace(**args), ...)` and its closure, because that
chain (args -> the two helpers -> closure state) is exactly what B1 rewires and
what B2 replaces. Every other site is located by AST, unparsed and `eval`ed with
its free names bound to the scenario's values, because three of them
(`train()`'s eval env and both eval_legacy sites) need a full run or a real
checkpoint to reach and a capture that only covered the cheap sites would freeze
the wrong roles.

SENTINELS. Opaque pass-through objects are distinct JSON-safe strings, so a
builder that routes `map_data` into the `team_spirit` slot — an identity bug
that two real objects would hide — shows up as a plain value mismatch.

REGENERATING was legitimate only while the builders were still pre-migration, and
is now refused outright (see FROZEN above). The invocation is recorded for the
audit trail; it produces a valid capture ONLY from a checkout of the commit named
in `_provenance.captured_at_commit`, never from this branch's tip.

    UV_NO_SYNC=1 uv run python tests/capture_env_config_pre_165b.py --capture

Deliberately NOT named `test_*`: pytest must not collect it.
"""
import argparse
import ast
import dataclasses
import inspect
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

FIXTURE = Path(__file__).parent / "fixtures" / "env_config_pre_165b.json"
CAPTURE_FORMAT = "cs2rl-env-config-capture-v1"

S_SHARED_TS = "<shared_ts>"
S_TEAM_SPIRIT = "<team_spirit>"
S_MAP_DATA = "<map_data>"
S_BUF = "<buf>"

# make_env's runtime inputs (spec §2.2): everything else it receives describes
# the env's dynamics and belongs in the config.
RUNTIME_NAMES = ("seed", "team_spirit", "auto_reset", "buf", "map_data",
                 "include_step_stats_in_info")

# The args the train and eval scenarios are driven with. Every channel Phase B
# rewires is non-default here: two weights, three flag knobs, an R0-G knob, a
# gamma that pbrs_gamma must follow, and the symmetrize flag.
NON_DEFAULT_ARGS = {
    "reward_ct_survival": 0.0,
    "reward_win_ct_timeout": 3.0,
    "n_active_per_team": 3,
    "pin_pitch": 1,
    "crouch_enabled": 0,
    "jump_enabled": 1,
    "gamma": 0.995,
    "round_time_ticks": 900,
    "reward_symmetrize": True,
}

# One entry per scenario: the role, the scenario name, the source file and
# enclosing qualname that locate the call site, how the site is driven, the args
# the scenario derives its knobs from (None = the site takes no args object) and
# the free names its call source needs bound.
#
# The two harness scenarios carry (n_active_per_team, pin_pitch, crouch_enabled,
# jump_enabled) = (5, 0, 1, 1) and (2, 0, 1, 0). Across the pair every one of the
# six knob PAIRS differs in at least one scenario, so a builder that routed any
# two of them into each other's slots shows up as a value mismatch; dropping
# either scenario reopens a swap (spec R8).
#
# These comments sit ABOVE the literal rather than inside it because yapf's
# spaces_before_comment aligner pushes a standalone comment inside a list out to
# column 78, where it is unreadable.
SCENARIOS = [
    dict(role="train",
         scenario="per_env_seed_wins",
         drive="train_closure",
         file="src/train.py",
         enclosing="build_env_factory.env_factory",
         args=NON_DEFAULT_ARGS,
         bindings={
             "buf": S_BUF,
             "seed": 3,
             "_seed": 41
         }),
    dict(role="train",
         scenario="pufferlib_seed_no_knobs",
         drive="train_closure",
         file="src/train.py",
         enclosing="build_env_factory.env_factory",
         args={},
         bindings={
             "buf": S_BUF,
             "seed": 5,
             "_seed": None
         }),
    dict(role="train",
         scenario="seed_none_becomes_zero",
         drive="train_closure",
         file="src/train.py",
         enclosing="build_env_factory.env_factory",
         args=NON_DEFAULT_ARGS,
         bindings={
             "buf": S_BUF,
             "seed": None,
             "_seed": None
         }),
    dict(role="eval",
         scenario="default_args",
         drive="call_site",
         file="src/train.py",
         enclosing="train",
         args={},
         bindings={"_map_data": S_MAP_DATA}),
    dict(role="eval",
         scenario="non_default_args",
         drive="call_site",
         file="src/train.py",
         enclosing="train",
         args={
             k: v
             for k, v in NON_DEFAULT_ARGS.items() if k != "reward_symmetrize"
         },
         bindings={"_map_data": S_MAP_DATA}),
    dict(role="eval",
         scenario="symmetrize_requested",
         drive="call_site",
         file="src/train.py",
         enclosing="train",
         args=NON_DEFAULT_ARGS,
         bindings={"_map_data": S_MAP_DATA}),
    dict(role="harness",
         scenario="seed_none_becomes_zero",
         drive="call_site",
         file="src/train_test_harness.py",
         enclosing="_build_trainer_for_test.env_factory",
         args=None,
         bindings={
             "shared_ts": S_SHARED_TS,
             "buf": S_BUF,
             "seed": None,
             "map_data": S_MAP_DATA,
             "n_active_per_team": 5,
             "pin_pitch": 0,
             "crouch_enabled": 1,
             "jump_enabled": 1
         }),
    dict(role="harness",
         scenario="explicit_seed_zero_kept",
         drive="call_site",
         file="src/train_test_harness.py",
         enclosing="_build_trainer_for_test.env_factory",
         args=None,
         bindings={
             "shared_ts": S_SHARED_TS,
             "buf": S_BUF,
             "seed": 0,
             "map_data": S_MAP_DATA,
             "n_active_per_team": 2,
             "pin_pitch": 0,
             "crouch_enabled": 1,
             "jump_enabled": 0
         }),
    dict(role="eval_legacy",
         scenario="bare_defaults",
         drive="call_site",
         file="src/train.py",
         enclosing="load_policy_from_checkpoint",
         args=None,
         bindings={}),
    dict(role="eval_legacy",
         scenario="explicit_seed",
         drive="call_site",
         file="src/train.py",
         enclosing="evaluate_checkpoint",
         args=None,
         bindings={"seed": 12345}),
    dict(role="smoke",
         scenario="fixed_seed",
         drive="call_site",
         file="src/train.py",
         enclosing="smoke_test",
         args=None,
         bindings={}),
    dict(role="external",
         scenario="both_passed",
         drive="call_site",
         file="src/train.py",
         enclosing="make_env",
         args=None,
         bindings={
             "team_spirit": S_TEAM_SPIRIT,
             "map_data": S_MAP_DATA
         }),
    dict(role="external",
         scenario="wrapper_defaults",
         drive="call_site",
         file="src/train.py",
         enclosing="make_env",
         args=None,
         bindings={
             "team_spirit": None,
             "map_data": None
         }),
]


class _Recorder:
    """Stands in for `c_env.cs2_env.make_env` and records what it was handed.

    Binds against the REAL signature, so the recorded shape is the one the real
    make_env would have seen. That is NOT a name check: make_env has `**legacy`,
    so `bind()` accepts essentially any keyword and only rejects a positional
    overflow or a duplicate — an unknown keyword lands in `legacy`, not in a
    TypeError. What binding buys is the FLATTENING below plus the guarantee that
    positional and keyword arguments end up under the same names. The name check
    is `_config_from_make_env_kwargs`'s `unknown` assert.

    The VAR_KEYWORD parameter is FLATTENED back out: `bind()` nests every legacy
    name under one `legacy` key, and a fixture recording that nesting would be
    unreadable and would not compare against a post-migration call that passes
    the same names typed.
    """

    def __init__(self, sig):
        self._sig = sig
        self._var_kw = next(
            (p.name for p in sig.parameters.values() if p.kind is inspect.Parameter.VAR_KEYWORD),
            None)
        self.calls = []

    def __call__(self, *args, **kwargs):
        bound = self._sig.bind(*args, **kwargs)
        received = dict(bound.arguments)
        if self._var_kw is not None:
            received.update(received.pop(self._var_kw, {}))
        self.calls.append(received)
        return "<env>"


def _qualified_calls(path, func_name):
    """(enclosing qualname, Call node) for every bare `func_name(...)` call in `path`.

    Qualname, not line number: this branch moves code in three PRs and any
    line-anchored lookup is stale before it is read.
    """
    tree = ast.parse((REPO_ROOT / path).read_text())
    out = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(child, f"{prefix}.{child.name}" if prefix else child.name)
                continue
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                    and child.func.id == func_name):
                out.append((prefix, child))
            walk(child, prefix)

    walk(tree, "")
    return out


def _config_from_make_env_kwargs(received):
    """Build the EnvConfig the legacy call describes, by direct construction.

    Deliberately NOT `EnvConfig.from_legacy_kwargs`: #173 deletes that method and
    this fixture must outlive it.

    The `pbrs_gamma is None` drop is DEFENSIVE, not a live rule: it mirrors
    `from_legacy_kwargs`' translation, but at 54d7da0 it cannot fire, because
    make_puffer_env forwards pbrs_gamma to make_env only when it is not None
    (`if pbrs_gamma is not None: kwargs[...]`), so no recorded call carries it.
    Kept so that a builder which starts forwarding None gets the documented
    translation instead of a ValueError from deep inside EnvConfig.
    """
    from env_config import KNOB_FIELDS, REWARD_FIELDS, EnvConfig, RewardWeights

    non_runtime = {k: v for k, v in received.items() if k not in RUNTIME_NAMES and k != "config"}
    unknown = set(non_runtime) - set(REWARD_FIELDS) - set(KNOB_FIELDS)
    assert not unknown, f"capture saw a legacy name that is not a field: {sorted(unknown)}"
    weights = {k: v for k, v in non_runtime.items() if k in REWARD_FIELDS}
    knobs = {k: v for k, v in non_runtime.items() if k in KNOB_FIELDS}
    if knobs.get("pbrs_gamma", 0.0) is None:
        del knobs["pbrs_gamma"]
    return EnvConfig(rewards=RewardWeights(**weights), **knobs)


def _drive(scenario):
    """Run one scenario against the recording stub. Returns the call_source text.

    Takes no recorder: the stub is installed by the caller as a module attribute
    on `c_env.cs2_env`, so every path below reaches it through the real chain
    rather than through an argument this function would have to thread down.
    """
    import train
    from env_factory import build_env_for

    calls = _qualified_calls(scenario["file"], "build_env_for")
    matching = [c for q, c in calls if q == scenario["enclosing"]]
    assert len(matching) == 1, (
        f"{scenario['enclosing']} has {len(matching)} build_env_for calls in "
        f"{scenario['file']}; the capture expects exactly one")
    call_source = ast.unparse(matching[0])

    if scenario["drive"] == "train_closure":
        factory = train.build_train_env_factory(Namespace(**scenario["args"]),
                                                shared_ts=S_SHARED_TS,
                                                map_data=S_MAP_DATA)
        b = scenario["bindings"]
        factory(buf=b["buf"], seed=b["seed"], _seed=b["_seed"])
        return call_source

    ns = {"build_env_for": build_env_for}
    ns.update(scenario["bindings"])
    if scenario["args"] is not None:
        ns["args"] = Namespace(**scenario["args"])
        ns["reward_overrides_from_args"] = train.reward_overrides_from_args
        ns["env_knobs_from_args"] = train.env_knobs_from_args
    # eval, not a hand-transcribed call: the thing measured is the SOURCE at the
    # site, recorded verbatim above, so the fixture is auditable against
    # `git show <commit>:src/train.py` without rerunning anything.
    eval(compile(ast.Expression(matching[0]), "<capture>", "eval"), ns)
    return call_source


def _git(*args):
    # check=True is load-bearing: without it any git failure returns an empty
    # stdout, which the dirty-src guard in capture() reads as "src/ is clean"
    # and which _provenance.captured_at_commit records as an empty string — a
    # guard that fails open and a fixture with no provenance. Fail loudly.
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def capture():
    raise SystemExit(
        "This capture is FROZEN. It records what make_env received BEFORE #165 Phase B PR B2\n"
        "retyped the role builders, and PR B2 deleted the two args helpers the recorded eval\n"
        "call source is evaluated against. Re-running it now would either fail here or — worse —\n"
        "succeed against the MIGRATED builders and overwrite the oracle with a measurement of\n"
        "the code under test. The fixture is checked by tests/test_env_config_capture_shape.py\n"
        "and consumed by tests/test_env_factory.py; to re-capture, check out the commit named in\n"
        "_provenance.captured_at_commit.")

    dirty = _git("status", "--porcelain", "--", "src")
    if dirty:
        raise SystemExit(
            f"src/ is dirty; this capture must record PRE-migration behaviour:\n{dirty}")

    import c_env.cs2_env as cs2_env
    from env_config import EnvConfig

    sig = inspect.signature(cs2_env.make_env)
    real_make_env = cs2_env.make_env

    # Every build_env_for site is expected to exist; a site the capture cannot
    # find is an ERROR, not a missing scenario (spec §4.1). The sweep is EVERY
    # .py file under src/, not the two modules today's thirteen scenarios happen
    # to live in: B2/B3 may add a builder call in a module that does not exist
    # yet, and a census anchored to a literal file list would not see it.
    # What it does NOT cover, so the promise is not read wider than it is: a
    # build_env_for call outside src/ (tests construct their own), and a site
    # spelled as an attribute (`env_factory.build_env_for(...)`), which
    # _qualified_calls does not match and env_factory.py's docstring forbids.
    # Paths are made REPO_ROOT-relative with forward slashes — the spelling
    # SCENARIOS uses — and sorted, so a mismatch here is a real new site and
    # never a path-spelling or iteration-order difference.
    src_files = sorted(
        f.relative_to(REPO_ROOT).as_posix() for f in (REPO_ROOT / "src").rglob("*.py"))
    sites = {(s["file"], s["enclosing"]) for s in SCENARIOS}
    found = {(f, q) for f in src_files for q, _ in _qualified_calls(f, "build_env_for")}
    assert sites == found, (f"build_env_for site census changed.\n  expected: {sorted(sites)}\n"
                            f"  found:    {sorted(found)}")

    out = {
        "_provenance": {
            "format": CAPTURE_FORMAT,
            "captured_at_commit": _git("rev-parse", "HEAD"),
            "why": "pre-#165-Phase-B-PR-B2 per-role oracle; a capture taken after the "
            "builders are typed would compare the factory to itself",
            "runtime_names": list(RUNTIME_NAMES),
        },
        "roles": {},
    }
    for scenario in SCENARIOS:
        rec = _Recorder(sig)
        cs2_env.make_env = rec
        try:
            call_source = _drive(scenario)
        finally:
            cs2_env.make_env = real_make_env
        assert len(rec.calls) == 1, (
            f"{scenario['role']}/{scenario['scenario']} reached make_env {len(rec.calls)}x")
        received = rec.calls[0]
        expected = _config_from_make_env_kwargs(received)
        if scenario["role"] == "eval":
            symmetrize = bool(scenario["args"].get("reward_symmetrize", False))
            input_config = expected.replace(reward_symmetrize=symmetrize)
        elif scenario["role"] in ("train", "harness"):
            input_config = expected
        else:
            input_config = EnvConfig()
        out["roles"].setdefault(scenario["role"], []).append({
            "scenario":
            scenario["scenario"],
            "site":
            scenario["file"],
            "enclosing":
            scenario["enclosing"],
            "call_source":
            call_source,
            "args":
            scenario["args"],
            "bindings":
            scenario["bindings"],
            "make_env_kwargs":
            received,
            "runtime_kwargs": {
                k: v
                for k, v in received.items() if k in RUNTIME_NAMES
            },
            "expected_config":
            dataclasses.asdict(expected),
            "input_config":
            dataclasses.asdict(input_config),
        })
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(out, sort_keys=True, indent=2) + "\n")
    print(f"wrote {FIXTURE} ({sum(len(v) for v in out['roles'].values())} scenarios)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--capture", action="store_true", required=True)
    p.parse_args()
    capture()
