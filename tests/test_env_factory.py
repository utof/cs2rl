"""`env.factory.build_env_for` must construct exactly the env the old sites did.

WHY THIS FILE EXISTS. Spec 2026-08-31 §2 W3 routed every construction through
one role-keyed factory; #165 PR B2 then retyped every role builder to take ONE
frozen `EnvConfig` instead of loose keyword knobs. The failure both migrations
risk is not a crash — it is a value quietly going missing, which produces a
WORKING env built on a default. That failure is invisible to almost every gate
this branch has:

  * the §3 determinism gate sets no `--reward-*` and no R0-G flag, so a factory
    that dropped the reward weights, `reward_symmetrize` or any omittable env
    knob still writes a BYTE-IDENTICAL `dust2_policy.pt`. This is not
    hypothetical: build_env_factory's own docstring records reward keys being
    silently dropped once, which would have made every experiment arm train the
    default weights;
  * `static_data_scalars()` is blind to the six RUNTIME inputs — `auto_reset`,
    `buf`, `include_step_stats_in_info`, `map_data`, `seed`, `team_spirit` —
    and those are precisely what distinguishes the roles from each other;
  * the full suite never passes `crouch_enabled` to the harness builder and its
    default equals the field default, so dropping it from the harness role's
    config mapping is green across the whole suite except for one test written
    for that alone (test_harness_config_carries_the_knobs_no_fixture_row_varies);
  * `--eval-interval` defaults to 0, so the `eval` role's env — the one whose
    `auto_reset=False` eval.baselines raises without, and the one PR B2 makes
    force raw rewards — is never constructed during the §3 run or any train test.

So the oracle is a CAPTURE, taken before the builders were typed
(`tests/fixtures/env_config_pre_165b.json`, recorded by
`tests/capture_env_config_pre_165b.py` one commit earlier; deleted since, it is
at `git show 9878725:tests/capture_env_config_pre_165b.py`). Comparing against a
list transcribed from the typed factory would compare the factory to itself. The
fixture is FROZEN: re-capturing it on the migrated tree would rewrite the oracle
to match whatever was built and turn every red green.

WHAT IS ASSERTED, per role and per recorded scenario, is two things that used to
be one. `make_env` takes a CONFIG plus six runtime inputs, so the comparison is:

  * the CONFIG the role builds, by value, against the config the OLD chain
    resolved (the fixture's `expected_config`, which differs from its
    `input_config` in exactly one row — `eval/symmetrize_requested`, where the
    eval role's raw-reward RULE turns `reward_symmetrize` off);
  * the RUNTIME kwargs, by EFFECTIVE value: both sides are bound against
    `make_env`'s own signature with the VAR_KEYWORD parameter removed and
    defaulted, so a typed builder may omit a runtime kwarg whose value equals
    `make_env`'s default without failing, while any kwarg whose VALUE moved
    fails. Name-set EQUALITY is not available here, because the capture went
    through `make_puffer_env`, which forwarded all six unconditionally.

WHY THE ARGUMENTS ARE SENTINEL STRINGS. `make_env` is replaced by a recording
stub, so nothing validates them, and a DISTINCT sentinel per pass-through
argument turns "routed map_data into the team_spirit slot" into a value mismatch
instead of two equal-looking real objects comparing equal.

KNOCK-OUT COVERAGE. Every role's construction is knocked out — one kwarg deleted
at the recording boundary — and the oracle must then FAIL. The knock-out runs
the SAME comparison the oracle runs (`_oracle_failures`), so it cannot drift into
knocking out a copy of the oracle. The triples where a drop provably cannot be
seen are pinned in `_VACUOUS_KNOCKOUTS`, in both directions. An oracle that
passes against a mutilated factory is measuring nothing, and nothing else in the
suite would tell us.

SCOPE. All six roles' call sites are migrated, and none of them names
`make_puffer_env`; `tests/test_env_construction_enforcement.py` asserts that
permanently. Since PR B2 `build_env_for` imports `env.c.cs2_env.make_env`
directly, so the stub in `_construct` is installed there.

FACTORY vs CALL SITE. Most of this file hands the fixture's bindings to
`build_env_for` by hand, which measures the FACTORY only — a call site that
fills the factory's slots wrongly passes every one of those tests. The tests
under the "MIGRATED CALL SITES" banner close that gap, in two layers, because no
single layer reaches every site:

  * a LIVE drive per site that can be reached without a training run (the two
    closures, smoke_test, make_env, and both eval_legacy sites). This is the only
    layer that can catch a SWAP — two values crossed between slots;
  * an AST comparison of every migrated call against the pre-migration
    `call_source` the fixture recorded, which is the only layer that reaches
    `train()`'s eval site at all. It is a NAME-SET and per-kwarg EXPRESSION
    comparison, so it catches a dropped or re-pointed argument but is blind to a
    swap of two arguments that read the same names — hence the layer above.

Both layers compare against the frozen pre-migration capture, never against the
migrated source, which is what keeps them from agreeing with the bug.
"""
import ast
import copy
import dataclasses
import json
import re
from pathlib import Path

import pytest

from cs2rl.env.config import KNOB_FIELDS, REWARD_FIELDS, EnvConfig, RewardWeights
from cs2rl.env.factory import ROLES, UNSET, build_env_for

FIXTURE = Path(__file__).parent / "fixtures" / "env_config_pre_165b.json"

# Must match CAPTURE_FORMAT in the capture script (`git show
# 9878725:tests/capture_env_config_pre_165b.py`). Duplicated rather than
# imported so a stale fixture fails on the tag here, in the file that consumes
# it, rather than on a KeyError deep inside a comparison.
CAPTURE_FORMAT = "cs2rl-env-config-capture-v1"


def _fixture():
    with FIXTURE.open() as fh:
        data = json.load(fh)
    assert data["_provenance"]["format"] == CAPTURE_FORMAT, (
        f"{FIXTURE.name} was written in format {data['_provenance']['format']!r}, this module "
        f"reads {CAPTURE_FORMAT!r} — and it must NOT be regenerated to fix that: the capture is "
        f"the pre-migration oracle")
    return data


FIXTURE_DATA = _fixture()

# The six inputs `make_env` takes alongside the config, read off the capture
# rather than written down: the capture script recorded them from the real
# signature, so a runtime parameter added to `make_env` cannot leave this file
# silently classifying it as a stray.
RUNTIME_NAMES = tuple(FIXTURE_DATA["_provenance"]["runtime_names"])


class _Recorder:
    """Stands in for `env.c.cs2_env.make_env` and records what the builder passed.

    Records the call as a plain dict of what was PASSED. Signature binding is not
    repeated here: the builders call `make_env` with keywords only, so the kwargs
    dict IS the bound explicit set, and
    test_factory_kwargs_bind_to_the_real_signature checks the whole recorded set
    against the real signature separately.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        return "<env>"


def _config_from(d):
    """Rebuild an EnvConfig from a fixture `asdict`.

    Deliberately NOT `EnvConfig.from_legacy_kwargs`: the test side must not
    depend on a method gh#173 deletes (spec R8), and the fixture stores FIELD
    names, which is what the constructor takes.
    """
    d = dict(d)
    return EnvConfig(rewards=RewardWeights(**d.pop("rewards")), **d)


def _construct(monkeypatch, role, **kwargs):
    """Run `build_env_for(role, ...)` against a recording stub at
    `env.c.cs2_env.make_env` — where build_env_for's function-local import now
    reads from, so this also proves that import is a per-call attribute read."""
    from cs2rl.env.c import cs2_env

    rec = _Recorder()
    monkeypatch.setattr(cs2_env, "make_env", rec)
    build_env_for(role, **kwargs)
    assert len(rec.calls) == 1, f"role {role!r} called make_env {len(rec.calls)} times"
    return rec.calls[0]


def _construct_dropping(monkeypatch, role, dropped, **kwargs):
    """`_construct`, with `dropped` deleted at the recording boundary.

    Simulating the drop here rather than by editing `env/factory.py` is
    indistinguishable — from the oracle's point of view — from a builder that
    never passed it, and it keeps the knock-out reproducible in CI instead of a
    procedure someone has to remember to perform by hand.
    """
    from cs2rl.env.c import cs2_env

    class _Dropping(_Recorder):

        def __call__(self, **kwargs):
            kwargs.pop(dropped, None)
            return super().__call__(**kwargs)

    rec = _Dropping()
    monkeypatch.setattr(cs2_env, "make_env", rec)
    build_env_for(role, **kwargs)
    assert len(rec.calls) == 1, f"role {role!r} called make_env {len(rec.calls)} times"
    return rec.calls[0]


# ── the scenario bindings -> build_env_for kwargs adapters ──────────────────
#
# Each role's pre-migration call site closed over things the factory does not
# take (an argparse Namespace, `make_env`'s own parameters, the harness's four
# plain knob arguments), so every role gets an explicit adapter rather than a
# clever generic one.


def _inputs_for(role, capture):
    b, rt = capture["bindings"], capture["runtime_kwargs"]
    config = _config_from(capture["input_config"])
    if role == "train":
        # shared_ts and map_data were CLOSURE state at capture time, so they are
        # not in `bindings`; they are read back off the runtime kwargs the old
        # chain produced. Both are DISTINCT sentinel strings, so a builder that
        # crossed the two slots still fails the comparison.
        return {
            "shared_ts": rt["team_spirit"],
            "map_data": rt["map_data"],
            "buf": b["buf"],
            "seed": b["seed"],
            "_seed": b["_seed"],
            "config": config
        }
    if role == "harness":
        # The four knobs are now INSIDE the config, so they stop being builder
        # arguments; everything else the closure bound still is one.
        knobs = ("n_active_per_team", "pin_pitch", "crouch_enabled", "jump_enabled")
        return {**{k: v for k, v in b.items() if k not in knobs}, "config": config}
    if role == "eval":
        return {"map_data": b["_map_data"], "config": config}
    if role == "eval_legacy":
        return {"seed": b["seed"]} if "seed" in b else {}
    if role == "smoke":
        return {}
    if role == "external":
        return {"team_spirit": b["team_spirit"], "map_data": b["map_data"]}
    raise AssertionError(f"no adapter for role {role!r}")


def _cases():
    """(role, scenario, capture) for every recorded capture."""
    return [(role, cap["scenario"], cap) for role, caps in FIXTURE_DATA["roles"].items()
            for cap in caps]


def _ids():
    return [f"{role}-{scenario}" for role, scenario, _ in _cases()]


def _role_captures(role):
    """(scenario, capture) for ONE role — the knock-out's slice of _cases()."""
    return [(c["scenario"], c) for c in FIXTURE_DATA["roles"][role]]


def _real_runtime_signature():
    """`make_env`'s signature with the VAR_KEYWORD parameter removed.

    Stripping `**legacy` is what makes an unknown name a TypeError here instead
    of being swallowed: with it in place every misspelling binds happily, which
    is exactly how the pre-B2 version of
    test_factory_kwargs_bind_to_the_real_signature went vacuous.

    PITFALL — WHY THIS IS BOUND ONCE AT IMPORT AND NEVER RE-READ. `_construct`
    monkeypatches `env.c.cs2_env.make_env` with the recording stub, whose own
    signature is `(**kwargs)`. Re-reading the attribute inside a comparison
    would therefore pick up the STUB, strip its VAR_KEYWORD and leave an EMPTY
    parameter list — at which point every runtime name is unbindable and every
    comparison in this file raises TypeError. Binding at import time is what
    makes `RUNTIME_SIG` the real contract rather than whatever a test last
    installed.
    """
    import inspect

    from cs2rl.env.c.cs2_env import make_env

    sig = inspect.signature(make_env)
    return sig.replace(
        parameters=[p for p in sig.parameters.values() if p.kind is not p.VAR_KEYWORD])


RUNTIME_SIG = _real_runtime_signature()

# Cross-check against the capture's own record of the runtime names. Both sides
# are derived — one from today's signature, one from the signature as it stood
# before the migration — so this fires if a runtime parameter was added, removed
# or renamed under the oracle rather than passing silently.
assert set(RUNTIME_NAMES) == {
    n
    for n in RUNTIME_SIG.parameters if n != "config"
}, (f"make_env's runtime parameters are {sorted(RUNTIME_SIG.parameters)}, but the capture "
    f"recorded {sorted(RUNTIME_NAMES)} — the fixture and the signature disagree about what a "
    f"runtime input IS, so every comparison below is comparing the wrong thing")


def _effective(d):
    """A kwarg mapping reduced to the RUNTIME values `make_env` would see.

    Binds against `RUNTIME_SIG` and applies defaults, then drops `config`. One
    definition, shared by `_oracle_failures` and the two call-site tests, so
    "what counts as the same runtime call" cannot mean two different things in
    one file.
    """
    bound = RUNTIME_SIG.bind(**{k: v for k, v in d.items() if k != "config"})
    bound.apply_defaults()
    args = dict(bound.arguments)
    args.pop("config", None)
    return args


def _oracle_failures(got, capture, role, scenario):
    """Every reason `got` is not the captured construction, as a LIST.

    THREE checks, each catching a different failure:

      KEY SET, as a two-sided CONTAINMENT and deliberately not an equality.
      `config` must be there (its absence means the whole payload vanished) and
      nothing outside make_env's six runtime parameters may be. Equality is not
      available: the capture was taken through make_puffer_env, which forwarded
      all six unconditionally, while a typed builder names only what it needs.
      The upper bound is what still catches the failure that matters — a legacy
      name leaking through `_make(...)`, which the migration census cannot see
      because `_make` is a parameter.

      CONFIG, by value, against the config the OLD chain resolved.

      RUNTIME, by EFFECTIVE value: both sides go through `_effective`, so a
      builder may omit a kwarg whose value equals make_env's default without
      failing, while any kwarg whose VALUE moved fails.

    WHY THIS RETURNS FAILURES RATHER THAN RAISING, and why the oracle below is
    two lines over it: test_knockout_dropping_one_kwarg_fails_that_roles_capture
    perturbs the construction and asserts this list is NON-empty, so the
    knock-out perturbs the SAME comparison the oracle runs. A knock-out that
    re-implemented these checks would stop being a knock-out of the oracle the
    moment either copy drifted.

    PITFALL: the KEY SET check GATES the other two, which is why this returns
    early instead of appending four times (one check, two appends). CONFIG and
    RUNTIME are ill-defined without it — `got["config"]` is a KeyError the
    moment the `config` knock-out has dropped it, and the runtime bind is a
    TypeError on a stray legacy name. Written as four independent appends this
    helper RAISES for every `config` knock-out instead of reporting one, and the
    knock-out test errors out for all six roles. The early return is also what
    preserves the assert-by-assert short-circuit the caller would have had.
    """
    import dataclasses

    out = []
    if "config" not in got:
        out.append(f"role {role!r}/{scenario} passed no config — the entire payload. "
                   f"pre-migration call: {capture['call_source']}")
    extra = sorted(set(got) - {"config"} - set(RUNTIME_NAMES))
    if extra:
        out.append(f"role {role!r}/{scenario} passes {extra}, which are neither `config` nor "
                   f"one of make_env's runtime parameters {sorted(RUNTIME_NAMES)}")
    if out:
        return out

    if dataclasses.asdict(got["config"]) != capture["expected_config"]:
        out.append(f"role {role!r}/{scenario}: the CONFIG this role builds changed.\n"
                   f"  pre-migration call was: {capture['call_source']}")

    if _effective(got) != _effective(capture["runtime_kwargs"]):
        out.append(f"role {role!r}/{scenario}: a RUNTIME value changed.\n"
                   f"  pre-migration call was: {capture['call_source']}")
    return out


@pytest.mark.parametrize(("role", "scenario", "capture"), _cases(), ids=_ids())
def test_factory_builds_the_captured_config(monkeypatch, role, scenario, capture):
    """The env this role builds is the env it used to build.

    The three comparisons, and the reason for each, are in `_oracle_failures`
    above — this test and the knock-out below are the only two callers, which
    is the point. Failure here is not "the factory is shaped differently", it
    is "the env this role builds is not the env it used to build": read the
    `call_source` the message carries, which is the pre-migration call
    verbatim, before touching the fixture.

    THE EVAL-ONLY ARG -> CONFIG ASSERTION at the end (spec §4.2) is not part of
    that comparison and no knock-out can perturb it, because it never reads
    `got`. It is here because `train` and `harness` DRIVE their real closures,
    so their `input_config` is PRODUCED and therefore checked, whereas
    `_inputs_for("eval", ...)` hands `_config_from(input_config)` straight back
    from the fixture and the production eval site lives inside `train()` where
    nothing can drive it — the AST layer only pins its SPELLING. Without this,
    nothing in PR B2 checks the eval role's arg -> config link at all, and
    `eval/symmetrize_requested` degrades from a real knock-out into a fixture
    round-trip that proves only that `.replace(...)` ran.

    It is eval-only because only the `eval` and `train` rows carry an `args`
    dict; the other seven record `args: null`, so `Namespace(**capture["args"])`
    would raise there. A guard spelled `"args" in capture` would guard nothing —
    that key is present on all 13 rows.
    """
    got = _construct(monkeypatch, role, **_inputs_for(role, capture))
    failures = _oracle_failures(got, capture, role, scenario)
    assert not failures, "\n".join(failures)

    if role == "eval":
        from argparse import Namespace

        from cs2rl.train.config import env_config_from_args

        assert env_config_from_args(Namespace(**capture["args"])) == _config_from(
            capture["input_config"]), (
                f"eval/{scenario}: env_config_from_args no longer turns the captured ARGS into "
                f"the config the old chain resolved — the oracle above would still pass, because "
                f"it feeds itself _config_from(input_config)")


def test_every_role_in_the_enum_is_covered_by_the_capture():
    """No role may be added without a captured shape, and none may go unchecked.

    Guards both directions of the drift the enum invites: a seventh role added
    with no oracle, and a role whose captures were quietly deleted from the
    fixture to make a failure go away. Without this, an untested role is
    indistinguishable from a role that has no call sites yet.
    """
    assert sorted(FIXTURE_DATA["roles"]) == sorted(ROLES), (
        f"fixture covers {sorted(FIXTURE_DATA['roles'])} but env_factory.ROLES is "
        f"{sorted(ROLES)}")
    for role, caps in FIXTURE_DATA["roles"].items():
        assert caps, f"role {role!r} has no captured scenarios"


def test_unknown_role_is_a_named_error():
    """A typo'd role must not fall through to a default env.

    `build_env_for("evaluation")` silently building the training env would be
    the same class of bug as the dropped reward key: a working run, wrong
    config, no signal.
    """
    with pytest.raises(ValueError, match="unknown env role 'evaluation'"):
        build_env_for("evaluation")


def test_train_closure_still_rejects_stray_kwargs(monkeypatch):
    """`build_env_factory`'s STRICT catch-all survived both rewrites.

    The guard raises on anything pufferlib did not name, and it is the loud form
    of the historical bug: a reward key routed through `_per_env_kwargs` used to
    be silently discarded, so the arm trained the baseline weights. It had ZERO
    test coverage while W3 rewrote the very closure body it guards, and the
    oracle above cannot see it — that oracle only ever observes the happy path,
    i.e. what `make_env` receives when nothing strays.

    Asserted through the real `build_env_factory`, not the factory module: the
    guard belongs to the closure, which is where pufferlib's kwargs arrive.
    `config=` is omitted deliberately — since PR B2 it defaults to `EnvConfig()`,
    resolved above the closure, so this call is legal and the guard is still the
    only thing that can fire.
    """
    from cs2rl.env.c import cs2_env
    from cs2rl.train import envs as train_envs

    monkeypatch.setattr(cs2_env, "make_env", _Recorder())
    factory = train_envs.build_env_factory(shared_ts=None, map_data=None)
    with pytest.raises(TypeError, match="unexpected kwargs.*reward_ct_survival"):
        factory(buf=None, seed=0, reward_ct_survival=0.0)


def test_eval_legacy_absent_seed_is_not_seed_none(monkeypatch):
    """The two eval_legacy sites differ ONLY in whether `seed` is passed.

    `make_env`'s default is `seed=0`, so spelling the absent case as `seed=None`
    would forward None where the bare call forwarded nothing — a real behaviour
    change that `static_data_scalars()` cannot see, because seed is not a
    StaticData field. The oracle above already pins the bare shape; this states
    the sentinel is the mechanism, so a future "simplification" to `seed=None`
    fails here with the reason attached.

    The comparison is against the RAW received kwargs, which is why
    `team_spirit` appears: since PR B2 the builder spells it explicitly rather
    than inheriting `make_puffer_env`'s parameter default.
    """
    assert _construct(monkeypatch, "eval_legacy") == {"config": EnvConfig(), "team_spirit": None}
    assert _construct(monkeypatch, "eval_legacy", seed=UNSET) == {
        "config": EnvConfig(),
        "team_spirit": None
    }
    assert _construct(monkeypatch, "eval_legacy", seed=None) == {
        "config": EnvConfig(),
        "team_spirit": None,
        "seed": None
    }


def test_every_role_builder_parameter_is_required():
    """No role builder may default a knob — `eval_legacy`'s seed sentinel aside.

    A default turns a call site that forgot an argument into a WORKING env built
    on someone else's value, which is the entire failure class W3 removes. Since
    PR B2 the concrete instance is `config`: a builder that defaulted it to
    `EnvConfig()` would give a caller who forgot the run's config an env on the
    baseline weights, silently. The `external` role is the older instance: its
    two parameters used to default to None, mirroring `make_env`'s published
    signature, so a delegate that forwarded only `team_spirit` would have
    quietly produced a dust2 env instead of raising. The defaulting belongs to
    the wrapper, not to the builder.

    `seed=UNSET` is the one exemption, and it is the opposite of a default: the
    sentinel exists precisely BECAUSE `seed=None` would be a silent behaviour
    change (make_env's own default is 0), so it is asserted to be the sentinel
    rather than merely allowed to be anything.
    """
    import inspect

    from cs2rl.env import factory as env_factory

    for role, builder in env_factory._ROLE_BUILDERS.items():
        for name, param in inspect.signature(builder).parameters.items():
            if param.kind is param.POSITIONAL_ONLY:
                continue                                                                          # the injected `_make`
            assert param.kind is param.KEYWORD_ONLY, f"{role}.{name} is not keyword-only"
            if (role, name) == ("eval_legacy", "seed"):
                assert param.default is UNSET, (
                    f"eval_legacy's seed default is {param.default!r}, not the UNSET sentinel — "
                    "see _Unset's docstring for why None is not equivalent")
                continue
            assert param.default is param.empty, (
                f"{role} builder defaults {name}={param.default!r}; a call site that drops it "
                "now builds a working env on that value instead of raising")


def test_factory_kwargs_bind_to_the_real_signature(monkeypatch):
    """Every RUNTIME kwarg every role passes is a real `make_env` parameter.

    WHAT THIS DOES, and it is not what the pre-B2 version did. The recorder
    accepts anything, so the value comparison would happily pass a typo'd name
    through as long as the FIXTURE carried the same typo. Binding closes that —
    but only against a signature with `**legacy` REMOVED. The pre-B2 version
    bound against `make_puffer_env`, which grew a VAR_KEYWORD channel, and from
    that moment every name bound: `total_nonsense=1`, `jmup_enabled=1` and
    `reward_kil=2.0` all returned OK, so the test asserted nothing while its
    docstring still claimed it did.

    `config` is excluded from the bind and checked by
    test_captured_configs_name_only_declared_fields instead — `**legacy` is not
    the config's channel, and a config field name is not a `make_env` parameter.

    In production a stray name would raise TypeError inside a forked vecenv
    worker, far from the mistake.
    """
    sig = RUNTIME_SIG
    for role, scenario, capture in _cases():
        got = _construct(monkeypatch, role, **_inputs_for(role, capture))
        try:
            sig.bind(**{k: v for k, v in got.items() if k != "config"})
        except TypeError as exc:
            pytest.fail(f"role {role!r}/{scenario} passes kwargs make_env rejects: {exc}")


def test_captured_configs_name_only_declared_fields():
    """The other half of the bind: every captured config key is a real field.

    The runtime bind above cannot see inside `config`, and `_config_from` would
    raise on a bogus key — but only for the rows a test actually rebuilds. This
    reads the fixture directly, so a hand-edited `expected_config` carrying a
    misspelled knob (or a weight that no longer exists) fails here rather than
    silently teaching the oracle the wrong shape.

    Both bounds come from `env.config`, never from a list written down here: a
    field added to the dataclass widens them automatically, and a field removed
    narrows them, which is the whole point of deriving them.
    """
    allowed = {"rewards"} | set(KNOB_FIELDS)
    for role, scenario, capture in _cases():
        for which in ("input_config", "expected_config"):
            cfg = capture[which]
            extra = sorted(set(cfg) - allowed)
            assert not extra, f"{role}/{scenario} {which} names non-fields {extra}"
            bad = sorted(set(cfg.get("rewards", {})) - set(REWARD_FIELDS))
            assert not bad, f"{role}/{scenario} {which}.rewards names non-weights {bad}"


# ── the two MIGRATED CALL SITES, driven for real ────────────────────────────
#
# Everything above feeds the fixture's bindings to `build_env_for` directly, so
# it proves the FACTORY reproduces the pre-migration construction and nothing
# about whether the closures hand it those bindings. The gap is not theoretical:
# swapping two knobs at the harness call site's EnvConfig(...) mapping leaves
# every test above green, and the §3 determinism gate never builds a harness env
# at all, so nothing in the repo fires. (The train site has an indirect oracle —
# a `seed=_seed, _seed=seed` swap there moves the gate's checkpoint md5 — but
# only that one gate, and only for that one kwarg pair.)
#
# So the two tests below drive the REAL closures, the one `build_env_factory`
# returns and the one `_build_trainer_for_test` defines, against the same
# pre-migration capture.


def _kwarg_diff(got, expected):
    """Per-key report of how two kwarg dicts differ; empty string when equal.

    Used instead of a bare `assert got == expected` so that a routing bug names
    the slots it crossed — "map_data: got '<shared_ts>'" — instead of printing
    two dicts and leaving the reader to diff them.
    """
    missing = sorted(set(expected) - set(got))
    added = sorted(set(got) - set(expected))
    parts = []
    if missing:
        parts.append(f"  missing: {missing}")
    if added:
        parts.append(f"  added:   {added}")
    parts += [
        f"  {k}: got {got[k]!r}, want {expected[k]!r}" for k in sorted(set(got) & set(expected))
        if got[k] != expected[k]
    ]
    return "\n".join(parts)


def _real_harness_env_factory(monkeypatch, tmp_path, shared_ts, **harness_kwargs):
    """Extract the REAL `env_factory` closure `_build_trainer_for_test` builds.

    Taken from `pufferlib.vector.make`'s first argument, at which point the
    harness build is aborted: everything AFTER that call (build_policy, PuffeRL,
    the self-play patches) costs seconds and constructs nothing this file looks
    at, while everything before it — the team-spirit Value, the mask shm, the
    EnvConfig mapping and the closure itself — is the wiring under test. The
    abort is an exception rather than a stub return value so the harness cannot
    run on against a fake vecenv and fail somewhere confusing.

    `mp` and `tempfile` are replaced on the HARNESS MODULE, not on the stdlib
    modules themselves: `shared_ts` is built inside the function and cannot be
    passed in, so it needs a stub to become an observable sentinel, and the
    scratch dir must not leak from a build that never reaches its own
    `cleanup()`. Module-scoped patches keep both out of every other test.
    """
    import types

    import pufferlib.vector

    from tests._helpers import trainer_harness

    captured = []

    class _Captured(Exception):
        pass

    def _fake_make(env_creators, *_args, **_kwargs):
        captured.append(env_creators[0])
        raise _Captured

    monkeypatch.setattr(pufferlib.vector, "make", _fake_make)
    monkeypatch.setattr(trainer_harness, "mp",
                        types.SimpleNamespace(Value=lambda *_a, **_kw: shared_ts))
    monkeypatch.setattr(trainer_harness, "tempfile",
                        types.SimpleNamespace(mkdtemp=lambda **_kw: str(tmp_path)))
    with pytest.raises(_Captured):
        trainer_harness._build_trainer_for_test(**harness_kwargs)
    assert len(captured) == 1, "pufferlib.vector.make was not reached exactly once"
    return captured[0]


@pytest.mark.parametrize("capture",
                         FIXTURE_DATA["roles"]["train"],
                         ids=[c["scenario"] for c in FIXTURE_DATA["roles"]["train"]])
def test_train_call_site_forwards_the_captured_kwargs(monkeypatch, capture):
    """`build_env_factory`'s closure, run for real, still produces the capture.

    Distinct failure from the factory tests above: those fire when
    `build_env_for("train", ...)` builds the wrong env GIVEN the right
    arguments; this fires when the closure fills the factory's slots wrongly —
    `seed=_seed, _seed=seed` being the cheap example, which every factory-side
    test in this file accepts and only the §3 gate's checkpoint md5 notices.

    Driven from the captured ARGS through `build_train_env_factory`, not from
    the captured config, so the whole args -> EnvConfig -> closure -> builder
    chain is what produces the config being compared. The fixture's own binding
    values are used verbatim, no sentinels needed: `seed` 3 against `_seed` 41,
    plus the distinct `<shared_ts>` / `<map_data>` strings, make every slot in
    this call distinguishable from every other.
    """
    from argparse import Namespace

    from cs2rl.env.c import cs2_env
    from cs2rl.train import envs as train_envs

    b, rt = capture["bindings"], capture["runtime_kwargs"]
    rec = _Recorder()
    monkeypatch.setattr(cs2_env, "make_env", rec)
    factory = train_envs.build_train_env_factory(Namespace(**capture["args"]),
                                                 shared_ts=rt["team_spirit"],
                                                 map_data=rt["map_data"])
    factory(buf=b["buf"], seed=b["seed"], _seed=b["_seed"])
    assert len(rec.calls) == 1, f"the train closure called make_env {len(rec.calls)} times"

    got = rec.calls[0]
    assert got["config"] == _config_from(capture["expected_config"]), (
        f"train CALL SITE / {capture['scenario']}: the CONFIG build_env_factory's closure "
        f"forwards is no longer the one the pre-migration chain resolved.\n"
        f"  pre-migration call was: {capture['call_source']}")
    diff = _kwarg_diff(_effective(got), _effective(rt))
    assert not diff, (f"train CALL SITE / {capture['scenario']}: a RUNTIME value the closure "
                      f"forwards no longer matches the capture.\n{diff}\n"
                      f"  pre-migration call was: {capture['call_source']}")


@pytest.mark.parametrize("capture",
                         FIXTURE_DATA["roles"]["harness"],
                         ids=[c["scenario"] for c in FIXTURE_DATA["roles"]["harness"]])
def test_harness_call_site_forwards_the_captured_kwargs(monkeypatch, tmp_path, capture):
    """`_build_trainer_for_test`'s closure, run for real, still produces the capture.

    This is the site with no other oracle at all: the §3 gate cannot see the
    harness, and no test anywhere passes `crouch_enabled` to
    `_build_trainer_for_test`.

    WHY BOTH SCENARIOS ARE NEEDED, and why dropping either reopens a swap. Since
    PR B2 the four knobs land in ONE EnvConfig, so there are no per-knob slots
    left to route sentinel strings into and swap detection has to be by VALUE.
    Neither captured scenario separates all four on its own:
      seed_none_becomes_zero separates pin_pitch from jump_enabled, and both
        from n_active_per_team;
      explicit_seed_zero_kept separates crouch_enabled from jump_enabled.
    Together every pair of the three flags differs in at least one scenario, and
    each differs from n_active_per_team in both. A future author looking at one
    scenario will see two knobs holding the same value and think it is noise —
    it is not; deleting that scenario reopens a swap.

    WHAT THESE TWO SCENARIOS DO NOT COVER. `pin_pitch` and `crouch_enabled` both
    hold the FIELD DEFAULT in both rows, so this value comparison is blind to
    either being DROPPED from the closure's EnvConfig(...) mapping — swap
    coverage is not drop coverage. The pre-B2 W3 fixture did cover them (its
    harness rows differed in both), and `env_config_pre_165b.json` is FROZEN and
    must NOT be re-captured to fix that, so the coverage is restored by a drive
    instead: test_harness_config_carries_the_knobs_no_fixture_row_varies below.

    `shared_ts` and `buf` remain distinct sentinels, so those two runtime slots
    are still swap-checked by value.
    """
    from cs2rl.env.c import cs2_env

    b, rt = capture["bindings"], capture["runtime_kwargs"]
    rec = _Recorder()
    monkeypatch.setattr(cs2_env, "make_env", rec)
    factory = _real_harness_env_factory(monkeypatch,
                                        tmp_path,
                                        rt["team_spirit"],
                                        num_envs=1,
                                        map_data=b["map_data"],
                                        n_active_per_team=b["n_active_per_team"],
                                        pin_pitch=b["pin_pitch"],
                                        crouch_enabled=b["crouch_enabled"],
                                        jump_enabled=b["jump_enabled"])
    factory(buf=b["buf"], seed=b["seed"])

    assert len(rec.calls) == 1, f"the harness closure called make_env {len(rec.calls)}x"
    got = rec.calls[0]
    assert got["config"] == _config_from(capture["input_config"]), (
        f"harness CALL SITE / {capture['scenario']}: the CONFIG _build_trainer_for_test's "
        f"closure forwards no longer matches the pre-migration capture.\n"
        f"  got      {got['config']}\n  expected {_config_from(capture['input_config'])}\n"
        f"  pre-migration call was: {capture['call_source']}")
    diff = _kwarg_diff(_effective(got), _effective(rt))
    assert not diff, (
        f"harness CALL SITE / {capture['scenario']}: a RUNTIME value _build_trainer_for_test's "
        f"closure forwards no longer matches the pre-migration capture.\n{diff}\n"
        f"  pre-migration call was: {capture['call_source']}")


def test_harness_config_carries_the_knobs_no_fixture_row_varies(monkeypatch, tmp_path):
    """Drive the harness closure OFF-FIXTURE for the two knobs it cannot see.

    Both harness captures hold the field default for pin_pitch and for
    crouch_enabled, so the fixture-driven test above is blind to either being
    dropped from _build_trainer_for_test's EnvConfig(...) mapping. pin_pitch is
    still caught outside this file (test_pitch_pin.py drives the env/policy
    agreement check with a non-default pin); crouch_enabled is caught by
    NOTHING else — no test in the tree passes it to _build_trainer_for_test.
    Before PR B2 it was the harness knock-out for that reason; B2 retires that
    knock-out, so the coverage moves here.

    Non-default on BOTH, in one call, so a mapping that dropped either fails.
    jump_enabled is asserted at its default in the same breath: with
    crouch_enabled non-default and jump_enabled default, a mapping that crossed
    the two fails here too. (pin_pitch and jump_enabled hold the same value in
    this drive; that pair is separated by the seed_none_becomes_zero scenario
    above, which is why both tests are needed.)
    """
    from cs2rl.env.c import cs2_env

    rec = _Recorder()
    monkeypatch.setattr(cs2_env, "make_env", rec)
    factory = _real_harness_env_factory(monkeypatch,
                                        tmp_path,
                                        "<shared_ts>",
                                        num_envs=1,
                                        crouch_enabled=0,
                                        pin_pitch=1)
    factory(buf="<buf>", seed=0)

    assert len(rec.calls) == 1, rec.calls
    cfg = rec.calls[0]["config"]
    assert (cfg.n_active_per_team, cfg.pin_pitch,
            cfg.crouch_enabled, cfg.jump_enabled) == (EnvConfig().n_active_per_team, 1, 0,
                                                      EnvConfig().jump_enabled), cfg


# ── knock-outs: prove the oracle can actually fail ──────────────────────────
#
# One dropped kwarg per role is not enough once `config` carries the whole
# payload: `config` alone would pass even against a builder that stopped
# forwarding every RUNTIME slot. Each tuple is `config` (the payload) plus
# every runtime slot whose loss this role's captures can actually see.
#
#   team_spirit  is in EVERY tuple, and it is the one that generalises: the
#                three defaulting roles spell it explicitly against make_env's
#                own different default, so dropping it is visible even where
#                every other slot equals a default.
#   train        buf / map_data: distinct sentinels on all three scenarios.
#                `seed` is NOT here — the seed_none_becomes_zero scenario
#                resolves to make_env's own seed default, so dropping it is
#                invisible there. Do not "complete" this tuple with it.
#   eval         auto_reset (unreachable by every run-level gate on this
#                branch; eval.baselines raises without it), plus seed and
#                map_data, both distinct from the defaults on all three.
#   harness      include_step_stats_in_info: the flag that makes this role its
#                own role at all, plus buf / map_data.
#   eval_legacy  seed: the only thing separating this role's two call sites.
#   external     map_data: half of this role's entire payload.
#
# NOTE the comments live above the dict, not inside it. Trailing comments in a
# literal get snapped to yapf's comment stops while ruff's isort wants one
# space, and the two then fight forever (gh#97, and the warning in train.py's
# import block).
_KNOCKOUT_KWARG = {
    "train": ("config", "team_spirit", "buf", "map_data"),
    "eval": ("config", "team_spirit", "auto_reset", "map_data", "seed"),
    "eval_legacy": ("config", "team_spirit", "seed"),
    "external": ("config", "team_spirit", "map_data"),
    "harness": ("config", "team_spirit", "buf", "map_data", "include_step_stats_in_info"),
    "smoke": ("config", "team_spirit", "seed"),
}

# (role, scenario, kwarg) triples where the drop provably CANNOT be seen, with
# the reason. Pinned in BOTH directions: a new green triple means a knock-out
# went vacuous, and a triple that starts going red means this row is stale and
# must be deleted. Both entries have the same cause — for these two scenarios
# `team_spirit` is the ONLY load-bearing runtime slot; every other value the
# old chain sent equals make_env's own default, so a builder that stopped
# sending it produces an identical effective mapping.
_VACUOUS_KNOCKOUTS = {
    ("eval_legacy", "bare_defaults", "seed"),
    ("external", "wrapper_defaults", "map_data"),
}


@pytest.mark.parametrize("role", ROLES)
def test_knockout_dropping_one_kwarg_fails_that_roles_capture(monkeypatch, role):
    """Delete one kwarg from a role's construction; the ORACLE must then fail.

    Not a re-implementation of the comparison: this runs `_oracle_failures`, the
    same helper `test_factory_builds_the_captured_config` runs, and asserts the
    list is NON-empty where that test asserts it is empty. A knock-out written
    against a second copy of the three checks would stop being a knock-out of
    the oracle the moment either copy drifted.

    THE VERDICT IS A SET EQUALITY, not "assert it failed". "Is the kwarg
    recorded in the capture?" — the pre-B2 vacuity filter — cannot be asked of
    the typed fixture at all: it has no `explicit_kwargs` key, and `config`
    appears in no capture's `runtime_kwargs`. So the question becomes "does the
    drop change the oracle's VERDICT?", and the triples where it provably does
    not are pinned in `_VACUOUS_KNOCKOUTS` in both directions — a newly green
    triple means a knock-out went vacuous, a newly red one means a pinned row is
    stale and must be deleted.
    """
    green = set()
    for dropped in _KNOCKOUT_KWARG[role]:
        for scenario, capture in _role_captures(role):
            got = _construct_dropping(monkeypatch, role, dropped, **_inputs_for(role, capture))
            if not _oracle_failures(got, capture, role, scenario):
                green.add((role, scenario, dropped))
    assert green == {
        v
        for v in _VACUOUS_KNOCKOUTS if v[0] == role
    }, (f"role {role!r}: the set of knock-outs the oracle CANNOT see moved. "
        f"got {sorted(green)}, pinned {sorted(v for v in _VACUOUS_KNOCKOUTS if v[0] == role)}")


def test_knockout_the_fixture_itself_is_load_bearing(monkeypatch):
    """Corrupt a captured VALUE and the oracle must notice — on both halves.

    Distinct from the kwarg-drop knock-outs above, which perturb the FACTORY.
    This perturbs the EXPECTATION, once per comparison, because the two halves
    fail for different reasons and a comparison that only looked at one of them
    would still pass here if the other were deleted:

      expected_config — the payload half. Caught by the CONFIG comparison only;
        the runtime mapping is untouched by it.
      runtime seed    — the runtime half. Caught by the effective-runtime
        comparison only; the config is untouched by it.
    """
    capture = copy.deepcopy(FIXTURE_DATA["roles"]["train"][0])
    got = _construct(monkeypatch, "train", **_inputs_for("train", capture))
    assert not _oracle_failures(got, capture, "train", "control"), "the unperturbed row must pass"

    bad_config = copy.deepcopy(capture)
    bad_config["expected_config"]["n_active_per_team"] += 1
    assert _oracle_failures(got, bad_config, "train", "bad_config"), (
        "a corrupted expected_config went unnoticed — the CONFIG comparison is not running")

    bad_runtime = copy.deepcopy(capture)
    assert bad_runtime["runtime_kwargs"]["seed"] == 41
    bad_runtime["runtime_kwargs"]["seed"] = 999
    assert _oracle_failures(got, bad_runtime, "train", "bad_runtime"), (
        "a corrupted runtime seed went unnoticed — the RUNTIME comparison is not running")


# ── the four remaining call sites, driven or read ───────────────────────────


class _Captured(Exception):
    """Raised by the recording stand-in once a call site has been observed.

    Used only for sites where nothing has been CONSTRUCTED yet at the moment the
    factory is called, so aborting leaks nothing. (The self-play manager sites
    are not like that — see the spy in tests/test_selfplay_factory.py.)
    """


def _drive(monkeypatch, module, fn, *args, **kwargs):
    """Run `fn(*args, **kwargs)` with `module.build_env_for` recording and aborting.

    Returns the (role, kwargs) the call site asked for. Patching the name on the
    CALLING module rather than on `env.factory` is deliberate: both call sites'
    modules do `from cs2rl.env.factory import build_env_for`, so the module-global is
    the binding production actually reads, and patching the source module would
    not be seen.
    """
    seen = []

    def _record(role, **kw):
        seen.append((role, kw))
        raise _Captured

    monkeypatch.setattr(module, "build_env_for", _record)
    with pytest.raises(_Captured):
        fn(*args, **kwargs)
    assert len(seen) == 1, f"{fn.__name__} reached build_env_for {len(seen)} times"
    return seen[0]


def test_smoke_test_call_site_asks_for_the_smoke_role(monkeypatch):
    """`smoke_test()` builds its env through the factory and passes NO kwargs.

    Seed 42 moved into `_build_smoke`, so the call site passing anything at all
    would mean the role's payload had been duplicated back out of the factory.
    The `env.reset(seed=42)` on the next line is NOT part of construction and
    must stay — the two 42s are a coincidence, not one value.
    """
    from cs2rl.train import envs as train_envs

    role, kwargs = _drive(monkeypatch, train_envs, train_envs.smoke_test)
    assert (role, kwargs) == ("smoke", {})


def test_make_env_delegates_to_the_external_role(monkeypatch):
    """The public wrapper forwards both of its parameters, unswapped.

    Distinct sentinels: `make_env(team_spirit, map_data)` takes two positional
    parameters of the same shape, so crossing them is a one-character edit that
    every value-based comparison in this file would accept.
    """
    from cs2rl.train import envs as train_envs

    seen = []
    monkeypatch.setattr(train_envs, "build_env_for", lambda role, **kw: seen.append((role, kw)))
    train_envs.make_env("<team_spirit>", "<map_data>")
    assert seen == [("external", {"team_spirit": "<team_spirit>", "map_data": "<map_data>"})]

    # The wrapper's own optional defaults stay on the wrapper — `_build_external`
    # requires both, so a forwarding bug is a TypeError rather than a dust2 env.
    seen.clear()
    train_envs.make_env()
    assert seen == [("external", {"team_spirit": None, "map_data": None})]


def test_external_role_returns_under_a_stub(monkeypatch):
    """The `external` role has no recursion trap left, and this is what proves it.

    Before PR B2 the chain was `train.make_env` -> `build_env_for("external")`
    -> `_build_external` -> the function-local `from cs2rl.train import
    make_puffer_env`. Repointing that import at `train.make_env` — the PUBLIC
    wrapper, whose name this module now mentions twice — would have made the
    external role call itself forever. Since B2 the import is
    `env.c.cs2_env.make_env`, the lower-layer constructor, and this test is the
    behavioural statement of that: with the stub installed the call RETURNS.

    A pure-AST guard (test_env_factory_never_names_train_make_env below) states
    the same thing statically; this one would still fail if a future indirection
    reintroduced the cycle by some spelling the AST check does not enumerate.
    """
    from cs2rl.env.c import cs2_env

    rec = _Recorder()
    monkeypatch.setattr(cs2_env, "make_env", rec)
    assert build_env_for("external", team_spirit=None, map_data=None) == "<env>"


def test_load_policy_from_checkpoint_asks_for_bare_eval_legacy(monkeypatch, tmp_path):
    """The bare eval_legacy site: role only, no kwargs, no seed.

    `torch.load` is stubbed because the construction sits AFTER the checkpoint
    read, and this test is about the construction. The stub returns the one key
    the loader inspects, so the function reaches the factory the same way a real
    checkpoint would.
    """
    import types

    import torch

    from cs2rl import policy as policy_mod

    ckpt = tmp_path / "fake.pt"
    ckpt.write_bytes(b"")
    monkeypatch.setattr(
        torch, "load",
        lambda *_a, **_kw: {"encoder.0.weight": types.SimpleNamespace(shape=(64, 105))})

    role, kwargs = _drive(monkeypatch, policy_mod, policy_mod.load_policy_from_checkpoint, ckpt,
                          "cpu")
    assert (role, kwargs) == ("eval_legacy", {}), (
        "load_policy_from_checkpoint must pass NO seed — make_env's own default is 0, and "
        "forwarding None instead would build a different env that scalars cannot see")


def test_evaluate_checkpoint_threads_its_episode_seed(monkeypatch):
    """The other eval_legacy site: same role, but it MUST pass a seed.

    `policy_mode="random"` with no checkpoint skips the policy load entirely, so
    the first loop iteration reaches the construction directly. That the seed is
    the per-episode `start_seed + episode_idx` rather than `start_seed` is not
    observable from episode 0 — the AST wiring test below is what pins the
    expression; this pins that the seed reaches the factory at all, under the
    right role.
    """
    from cs2rl.train import evaluate as train_evaluate

    role, kwargs = _drive(monkeypatch,
                          train_evaluate,
                          train_evaluate.evaluate_checkpoint,
                          checkpoint_path=None,
                          policy_mode="random",
                          start_seed=7717,
                          num_episodes=1)
    assert (role, kwargs) == ("eval_legacy", {"seed": 7717})


# The five attributes check (a) compares, WRITTEN DOWN rather than AST-read out
# of `assert_eval_env_agreement`. That is deliberate and it is the opposite of
# the discipline this file uses elsewhere, so the reason matters: the two checks
# report the SAME input with DIFFERENT messages — (a) says `on <knob>`, (b) says
# `on config.<knob>` — and the parametrization below decides which message to
# demand from this tuple. Read out of the source it would ADAPT to a name
# leaving or joining (a)'s tuple and prove nothing about it; written down, either
# drift flips a message and turns the matching case red. This tuple is therefore
# pinned in BOTH directions by the test itself, which is what makes writing it
# down safe here and would not make it safe anywhere else.
_CHECK_A_KNOBS = ("n_active_per_team", "pin_pitch", "crouch_enabled", "jump_enabled", "round_time")


def _differing_config(base_env, name):
    """An EnvConfig differing from `EnvConfig()` in exactly `name`, and valid.

    DERIVED from EnvConfig()'s own values wherever the type allows it — flip the
    bool, step the int, add to the float — so a changed default can never leave a
    case comparing a value to itself. `n_active_per_team` steps DOWN because the
    validator caps it at TEAM_SIZE.

    `round_time` is read off the BASE ENV, not off the config: its field default
    is the None sentinel and check (a) compares the RESOLVED tick count, so the
    differing value has to be one tick away from whatever Cs2Env resolved None
    to. `laser_range` and `max_turn_speed` are sentinels too but only (b) sees
    them, and (b) compares the field, where any non-None value already differs.

    A FIELD ADDED TO EnvConfig LANDS HERE AS A KeyError, and that is the point:
    the parametrization walks `dataclasses.fields(EnvConfig)`, so a new field
    joins the pin automatically and cannot join it silently unpinned.
    """
    cfg = EnvConfig()
    other = {
        "rewards": cfg.rewards.replace(reward_kill=cfg.rewards.reward_kill + 1.0),
        "pbrs_gamma": cfg.pbrs_gamma / 2,
        "reward_symmetrize": not cfg.reward_symmetrize,
        "recoil": not cfg.recoil,
        "n_active_per_team": cfg.n_active_per_team - 1,
        "pin_pitch": 1 - cfg.pin_pitch,
        "crouch_enabled": 1 - cfg.crouch_enabled,
        "jump_enabled": 1 - cfg.jump_enabled,
        "round_time": base_env.round_time + 1,
        "laser_range": 1.0 + (cfg.laser_range or 0.0),
        "max_turn_speed": 1.0 + (cfg.max_turn_speed or 0.0),
    }[name]
    changed = cfg.replace(**{name: other})
    assert getattr(changed, name) != getattr(
        cfg, name), (f"the differing value for {name} normalised back to the default "
                     f"({getattr(changed, name)!r}); this case would assert nothing")
    return changed


@pytest.mark.parametrize("field_name", [f.name for f in dataclasses.fields(EnvConfig)])
def test_eval_env_agreement_two_directions(simple_map, field_name):
    """Both checks fire, the exclusion holds, and EVERY EnvConfig field is covered.

    WHY IT IS PARAMETRIZED OVER THE DATACLASS and not over a hand-picked few
    (review finding, Important 1). `assert_eval_env_agreement` cannot fire on any
    input reachable today — its own DISCLOSURE paragraph says so — so it exists
    purely to catch a FUTURE divergence, and check (b)'s skip set is the only
    thing that decides what it will then catch. When this test differed two
    fields, widening that skip set from `{reward_symmetrize}` to nine names left
    it green — and, as the review measured, the whole non-slow suite with it. The
    guard was comparing 2 of 11 fields and nothing could see it. The skip set is
    now watched by construction — one case per `dataclasses.fields(EnvConfig)`
    entry, so a field silently added to the skip set (or to the dataclass) has
    nowhere to hide.

    WHAT EACH CASE DEMANDS, and it is a MESSAGE, never just "it raised":

      the five in _CHECK_A_KNOBS — `on <knob>`, (a)'s bare-knob spelling, which
                    (b)'s `on config.<knob>` does not contain. (b) compares all
                    five of these too, so with (a) deleted (b) would raise on the
                    same input; matching plain `<knob>` would pass under both and
                    pin nothing. This is what proves (a) still runs, and it runs
                    FIRST, which is why no case here can prove (b) with one of
                    these names.
      every other field — `on config.<field>`, which ONLY (b) produces, because
                    (a) reads Python attributes and `pbrs_gamma`, `recoil`,
                    `laser_range` and `max_turn_speed` are not attributes of a
                    Cs2Env (measured). `rewards` is the trap: `env.rewards` DOES
                    exist and is the per-agent reward BUFFER, an ndarray with
                    nothing to do with `config.rewards`, so adding `rewards` to
                    (a)'s tuple would compare two buffers and quietly stop
                    covering the 23 weights. These cases are the ones a widened
                    skip set kills.
      reward_symmetrize — must NOT raise. The eval role FORCES it off
                    (`config.replace(reward_symmetrize=False)`) while the driver
                    env takes it from args, so the two configs are MEANT to
                    differ here; a check that compared it would abort every run
                    launched with --reward-symmetrize.

    COST: one base env plus one differing env per field. Measured 2026-09-04 on
    this tree, all 11 cases run in 3.0 s — cheap because `make_env` on the simple
    map is cheap. Do not "optimise" it into one env pair reused across fields:
    reusing a pair is what forced the old two-case shape that produced the
    finding.
    """
    from cs2rl.env.c.cs2_env import make_env
    from cs2rl.train import envs as train_envs

    base = make_env(config=EnvConfig(), map_data=simple_map, seed=0)
    try:
        other = make_env(config=_differing_config(base, field_name), map_data=simple_map, seed=0)
        try:
            if field_name == "reward_symmetrize":
                train_envs.assert_eval_env_agreement(base, other)      # must not raise
                return
            wanted = (f"on {field_name}"
                      if field_name in _CHECK_A_KNOBS else f"on config.{field_name}")
            with pytest.raises(RuntimeError, match=re.escape(wanted)):
                train_envs.assert_eval_env_agreement(base, other)
        finally:
            other.close()
    finally:
        base.close()


def test_train_calls_assert_eval_env_agreement():
    """Pin train()'s call site — the one line no test can execute.

    Same mechanism and same reason as test_train_uses_build_train_env_factory:
    everything else in train() needs a real run to reach, and this check's
    failure mode is silent (the policy is scored on a different sim than it
    trains on, and the numbers just look worse).
    """
    import inspect

    from cs2rl.train import loop as train_loop

    assert "assert_eval_env_agreement(" in inspect.getsource(train_loop.train)


# ── AST: every migrated site still reads what the old site read ─────────────
#
# The layer that reaches `train()`'s eval site, which no test can drive. Each
# check is derived from `capture["call_source"]` — the pre-migration call,
# recorded verbatim one commit before the builders were typed — so it compares
# the migrated site to the OLD site rather than to the builder it now calls.

# The ONLY change PR B2 is allowed to make to what a site READS. Everything
# else about the call — the role literal, the per-kwarg expressions, the
# absence of a splat — must be unchanged, and the three unlisted roles must be
# byte-identical. Derived from the frozen pre-migration capture, never from the
# migrated source, which is what stops this from agreeing with the bug.
_REWIRED = {
    "train": ({"reward_overrides", "reward_symmetrize", "env_knobs"}, {"config"}),
    "eval": ({"reward_overrides_from_args", "env_knobs_from_args"}, {"env_config_from_args"}),
    "harness": ({"n_active_per_team", "pin_pitch", "crouch_enabled", "jump_enabled"}, {"config"}),
}


def _keywords(call_source):
    """(named, splats) for a call: {kwarg -> unparsed expr}, [unparsed expr]."""
    call = ast.parse(call_source, mode="eval").body
    named = {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg is not None}
    splats = [ast.unparse(kw.value) for kw in call.keywords if kw.arg is None]
    return named, splats


def _free_names(call_source):
    """Every bare Name read anywhere in a call's arguments.

    The comparison that survives a kwarg being RENAMED by the migration
    (`team_spirit=shared_ts` became `shared_ts=shared_ts`) and an expression
    being simplified (`0 if seed is None else seed` became `seed`, the remap
    having moved into the builder), while still failing the moment a call site
    stops reading something it used to read — a dropped reward source, or an
    `evaluate_checkpoint` that passed `start_seed` instead of `seed`.

    Set-based, and therefore blind to two arguments SWAPPED between slots. That
    is what the live drives above are for.
    """
    call = ast.parse(call_source, mode="eval").body
    return {n.id for kw in call.keywords for n in ast.walk(kw.value) if isinstance(n, ast.Name)}


def _is_derivation(expr):
    """True for a conditional/boolean expression — a RULE rather than a value.

    Exactly the shapes W3 moved into the builders (both closures' seed remaps,
    `**env_knobs or {}`'s None guard). Anything else — a name, an attribute, a
    helper call — is a value the call site sources, and those must still be
    spelled identically after the migration.
    """
    node = ast.parse(expr, mode="eval").body
    return isinstance(node, (ast.IfExp, ast.BoolOp))


def _call_in(path, qualname, func_name):
    """The one `func_name(...)` call inside `qualname`, unparsed.

    Located by enclosing qualname, never by line number — W1 moved ~56 symbols
    out of train.py and this branch keeps moving more.
    """
    tree = ast.parse(path.read_text())
    found = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(child, prefix + [child.name])
                continue
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                    and child.func.id == func_name and ".".join(prefix) == qualname):
                found.append(child)
            walk(child, prefix)

    walk(tree, [])
    assert len(found) == 1, (f"expected exactly one {func_name}(...) in {path.name}:{qualname}, "
                             f"found {len(found)} — a site was split, duplicated or removed")
    return found[0]


# Enclosing qualnames that MOVED after the capture, old -> current. The fixture is
# frozen, so its `enclosing` locator stays as captured and the move is recorded
# here. gh#168 W1 factored the harness's env_factory closure (and everything else
# built before the trainer constructor) out of `_build_trainer_for_test` into
# `_harness_parts`; the call itself is unchanged, which the test below still
# proves by comparing its source against the captured one.
MOVED_ENCLOSING = {
    "_build_trainer_for_test.env_factory": "_harness_parts.env_factory",
}

# (fixture site path, top-level function) -> the file it lives in now (#205 part 3).
MOVED_SITE_FILE = {
    ("src/train.py", "train"): "src/cs2rl/train/loop.py",
    ("src/train.py", "evaluate_checkpoint"): "src/cs2rl/train/evaluate.py",
    ("src/train.py", "load_policy_from_checkpoint"): "src/cs2rl/policy.py",
    ("src/train.py", "make_env"): "src/cs2rl/train/envs.py",
    ("src/train.py", "smoke_test"): "src/cs2rl/train/envs.py",
    ("src/train.py", "build_env_factory"): "src/cs2rl/train/envs.py",
    ("src/train_test_harness.py", "_harness_parts"): "tests/_helpers/trainer_harness.py",
}


def _migrated_sites():
    """(role, enclosing qualname, source file, one pre-migration call_source).

    Read entirely out of the frozen fixture: it records each capture's `role`,
    its `enclosing` qualname and its `site` as `<relpath>[:<lineno>]`, so the
    whole table — including which file each site lives in — comes from the
    pre-migration snapshot rather than from a list someone transcribed off the
    migrated code. The enclosing qualname is mapped through MOVED_ENCLOSING so
    a site that was factored into another function is still located, and a
    site that was NOT (a stale map entry) still fails `_call_in` loudly.
    """
    captured = {cap["enclosing"] for caps in FIXTURE_DATA["roles"].values() for cap in caps}
    stale = sorted(set(MOVED_ENCLOSING) - captured)
    assert not stale, (f"MOVED_ENCLOSING keys {stale} match no captured `enclosing`; a key the "
                       "fixture never names is a dead map entry that watches nothing")
    sites = {}
    for role, caps in FIXTURE_DATA["roles"].items():
        for cap in caps:
            enclosing = MOVED_ENCLOSING.get(cap["enclosing"], cap["enclosing"])
            key = (role, enclosing, cap["site"].split(":")[0])
            sites.setdefault(key, set()).add(cap["call_source"])
    out = []
    for (role, enclosing, relpath), sources in sorted(sites.items()):
        assert len(sources) == 1, (f"{role}/{enclosing} was captured with {len(sources)} different "
                                   f"call sources: {sources}")
        # The fixture's `site` paths are pre-#199 labels (`src/train.py`), kept byte for
        # byte. #205 part 3 split train.py by owner and moved the harness to tests/, so the
        # live file is looked up by the site's top-level function; an unmapped one fails.
        top = enclosing.split(".")[0]
        assert (relpath, top) in MOVED_SITE_FILE, (
            f"{relpath}::{top} has no live file in MOVED_SITE_FILE; say where #205 part 3 put it")
        live = Path(__file__).resolve().parents[1] / MOVED_SITE_FILE[(relpath, top)]
        out.append((role, enclosing, live, sources.pop()))
    return out


@pytest.mark.parametrize(("role", "enclosing", "path", "call_source"),
                         _migrated_sites(),
                         ids=[f"{r}-{e}" for r, e, _, _ in _migrated_sites()])
def test_migrated_site_still_reads_what_the_old_site_read(role, enclosing, path, call_source):
    """Per site: the role is right, the names moved by exactly the rewiring, the
    constants left, and the three rewired sites read `config` the one way each is
    allowed to.

    Each assertion catches a different way the migration could be wrong:

      ROLE — the site asks for its own role. `smoke_test` asking for
      `eval_legacy` builds a working env with the wrong seed and nothing else in
      the repo notices.

      NAMES — the set of free names the call reads changed by EXACTLY the
      rewiring `_REWIRED` records and nothing else, which for the three unlisted
      roles is the identity. A call site that quietly stopped forwarding the
      run's reward source (the historical bug: every experiment arm trains the
      baseline weights, checkpoint byte-identical) or that forwards `start_seed`
      where it used to forward `seed` fails here. Per-kwarg EXPRESSIONS are
      compared too, for every kwarg both calls name.

      CONSTANTS — every literal the old call passed is GONE from the new one.
      Vacuous against today's captures, which carry no keyword literals, and
      kept so a constant cannot creep back: `seed=42` respelled at the smoke
      site, or `auto_reset=False` at the eval site, would mean the payload was
      duplicated rather than moved, and the two copies can then drift.

      THE `config=` EXPRESSION, for the rewired roles only. `train` and
      `harness` must read a bare local named `config` — anything else means the
      site rebuilt a config of its own instead of forwarding the one resolved
      above its closure — and `eval` must read `env_config_from_args(args)`, the
      SAME resolver build_train_env_factory uses, because that identity is what
      makes assert_eval_env_agreement's disclosure true.
    """
    new_call = _call_in(path, enclosing, "build_env_for")
    new_source = ast.unparse(new_call)

    assert new_call.args and isinstance(new_call.args[0], ast.Constant), (
        f"{enclosing} calls build_env_for without a literal role: {new_source}")
    assert new_call.args[0].value == role, (
        f"{enclosing} asks for role {new_call.args[0].value!r}; the fixture captured it as "
        f"{role!r}")

    old_named, old_splats = _keywords(call_source)
    new_named, new_splats = _keywords(new_source)

    removed, added = _REWIRED.get(role, (set(), set()))
    want_names = (_free_names(call_source) - removed) | added
    assert _free_names(new_source) == want_names, (
        f"{role}/{enclosing}: the names this construction reads changed by something other than "
        f"the rewiring PR B2 declares.\n"
        f"  unexpectedly no longer read: {sorted(want_names - _free_names(new_source))}\n"
        f"  unexpectedly newly read:     {sorted(_free_names(new_source) - want_names)}\n"
        f"  declared rewiring: -{sorted(removed)} +{sorted(added)}\n"
        f"  pre-migration:  {call_source}\n  now:            {new_source}")

    for kwarg in sorted(set(old_named) & set(new_named)):
        old_expr, new_expr = old_named[kwarg], new_named[kwarg]
        if _is_derivation(old_expr):
            # The one shape allowed to change: a DERIVATION the builder now owns.
            # Both closures' seed remaps are this — `_seed if _seed is not None
            # else seed or 0` and `0 if seed is None else seed` moved into
            # _build_train / _build_harness verbatim, so the call site passes the
            # raw inputs instead. The rule stays as tight as it can be without
            # forbidding that: the new expression may read only names the old one
            # read for THIS kwarg, so a site that started sourcing the seed from
            # somewhere else still fails. Which of those inputs the builder then
            # picks is the oracle's job — it covers all three of the train
            # remap's branches.
            assert _free_names(f"f(x={new_expr})") <= _free_names(f"f(x={old_expr})"), (
                f"{role}/{enclosing}: {kwarg} used to be derived from "
                f"{sorted(_free_names(f'f(x={old_expr})'))} and is now {new_expr!r}")
        else:
            assert new_expr == old_expr, (
                f"{role}/{enclosing}: {kwarg} was {old_expr!r}, is now {new_expr!r}")

    constants = {
        k
        for k, v in old_named.items() if isinstance(ast.parse(v, mode="eval").body, ast.Constant)
    }
    assert constants.isdisjoint(new_named), (
        f"{role}/{enclosing}: {sorted(constants & set(new_named))} are still spelled at the call "
        f"site. Constants belong to the builder now — two copies drift.\n"
        f"  pre-migration: {call_source}\n  now:           {new_source}")

    assert not new_splats, (f"{role}/{enclosing} splats into build_env_for ({new_splats}); every "
                            "role builder takes explicit keywords so a typo is a TypeError")
    assert not old_splats, (
        f"{role}/{enclosing}: the captured call splatted {old_splats}, which no capture in this "
        "fixture does — the fixture changed shape, so re-derive this check rather than deleting it")

    if role in _REWIRED:
        expr = new_named.get("config")
        assert expr is not None, f"{role}/{enclosing} passes no config="
        node = ast.parse(expr, mode="eval").body
        if role == "eval":
            assert (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "env_config_from_args"
                    and _free_names(f"f(x={expr})") == {"env_config_from_args", "args"}), (
                        f"{role}/{enclosing}: config= is {expr!r}; it must be "
                        "env_config_from_args(args) — the SAME resolver build_train_env_factory "
                        "uses, which is what makes assert_eval_env_agreement's disclosure true")
        else:
            assert isinstance(node, ast.Name) and node.id == "config", (
                f"{role}/{enclosing}: config= is {expr!r}; it must be the bare local `config` "
                "resolved above the closure, not a config rebuilt at the call site")


def test_the_ast_wiring_check_covers_every_role():
    """All six roles reach the check above — a site it cannot find is not a pass.

    `_migrated_sites()` is built by grouping captures, so a role whose captures
    were deleted from the fixture, or whose `enclosing` no longer resolves, would
    simply produce fewer parametrized cases. Fewer silent cases is exactly the
    shape of a guard that has stopped guarding.
    """
    assert sorted({role for role, _, _, _ in _migrated_sites()}) == sorted(ROLES)
    # eval_legacy is the one role with two distinct sites; the others have one
    # each, so seven sites for six roles.
    assert len(_migrated_sites()) == len(ROLES) + 1


def test_mask_view_attach_stays_out_of_the_factory():
    """The harness's `_attach_mask_view` is post-construction wiring, not the factory's.

    It needs the caller's shm handle and the per-env index pufferlib supplies,
    neither of which the factory has. Pulling it in would force the factory to
    take shared-memory arguments it cannot validate — and would make the train
    closure's own `_attach_cont_action_view` / `_attach_mask_view` pair, which
    has different None-guard semantics, look like it belongs there too.

    Asserted over the AST, not the source text: the module docstring NAMES
    `_attach_mask_view` while explaining that it stays at the call site, so a
    substring check would either fail on the prose or be weakened until it
    stopped checking anything.
    """
    from cs2rl.env import factory as env_factory

    tree = ast.parse(Path(env_factory.__file__).read_text())
    attached = [
        n.func.attr for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in (
            "_attach_mask_view", "_attach_cont_action_view")
    ]
    assert not attached, (f"env/factory.py calls {attached} — shared-memory attach is the "
                          "caller's job; see this test's docstring")


def test_env_factory_never_names_train_make_env():
    """`env.factory` must not reach `train.make_env` — statically, by any spelling.

    Since PR B2 `build_env_for` imports the LOWER-layer `env.c.cs2_env.make_env`.
    Re-pointing that at `train`'s public wrapper would make the `external` role
    call itself forever, and would put the whole training stack back on
    `--dump-config`'s import path. Three shapes are refused: a
    `from cs2rl.train import make_env`, a `train.make_env` attribute access, and
    any module-LEVEL import of `cs2rl.train` at all, in every spelling
    (`import cs2rl.train`, `from cs2rl.train import ...`, `from cs2rl import train`).
    Names are compared in FULL: every first-party module is top-level `cs2rl`, so
    a first-component check could no longer tell `train` from `env.config`.

    Scoped to module-level imports plus the name `make_env`, deliberately:
    `build_selfplay_manager`'s FUNCTION-LOCAL `from cs2rl.train import SelfPlayManager`
    is legitimate — it is why train.py's `__main__` self-alias is still a hard
    prerequisite of this module — and must not be flagged.
    """
    from cs2rl.env import factory as env_factory

    tree = ast.parse(Path(env_factory.__file__).read_text())
    offenders = []
    for node in tree.body:                                                                       # module level ONLY
        if isinstance(node, ast.Import):
            offenders += [
                f"line {node.lineno}: import {a.name}" for a in node.names
                if a.name == "cs2rl.train" or a.name.startswith("cs2rl.train.")
            ]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if (module == "cs2rl.train" or module.startswith("cs2rl.train.")
                    or (module == "cs2rl" and any(a.name == "train" for a in node.names))):
                offenders.append(f"line {node.lineno}: from {module} import ...")
    for node in ast.walk(tree):
        if (isinstance(node, ast.ImportFrom) and node.module == "cs2rl.train"
                and any(a.name == "make_env" for a in node.names)):
            offenders.append(f"line {node.lineno}: from cs2rl.train.envs import make_env")
        if (isinstance(node, ast.Attribute) and node.attr == "make_env"
                and isinstance(node.value, ast.Name) and node.value.id == "train"):
            offenders.append(f"line {node.lineno}: train.make_env")
    assert not offenders, ("env_factory reaches train's env wrapper or imports train at module "
                           f"scope: {offenders}")


# ── reward wiring, relocated from tests/test_reward_weight_wiring.py ────────
#
# That file was retired by PR B2 (it existed to pin the legacy override dict).
# What survives it is everything DOWNSTREAM of the declaration: that the env
# factory actually injects the config it is handed, and that the seam between
# train()'s args and the factory carries it. The three "the last boundary
# rejects a bad key/value" tests went to tests/test_env_config.py, where the
# validator they exercise now lives.


def test_env_factory_injects_reward_overrides():
    """Spec §6.3: the run's weights must arrive through the FACTORY path.

    Deliberately does not construct an env directly — the discard trap
    (env_factory swallows **kwargs) lives in the factory, so a direct
    construction would pass while training ran the baseline.
    Read back through the ctypes overlay: sd is a POINTER, .contents required.
    """
    import multiprocessing as mp

    from cs2rl.train import envs as train_envs

    # PARTIAL RewardWeights on purpose (review fix 5): naming all 23 would set
    # reward_kill to its default explicitly, so the "untouched" assertion below
    # would pass even with the omitted-field fallback broken. Naming three and
    # letting the dataclass supply the rest is what keeps that assertion live.
    factory = train_envs.build_env_factory(
        shared_ts=mp.Value("f", 0.3),
        map_data=None,
        config=EnvConfig(rewards=RewardWeights(
            reward_ct_survival=0.0, reward_win_ct_timeout=3.0, pbrs_nav_weight_t=0.07)))
    env = factory(seed=0)
    try:
        sd = env._c_env.sd.contents
        assert sd.reward_ct_survival == pytest.approx(0.0)             # A1 arm
        assert sd.reward_win_ct_timeout == pytest.approx(3.0)          # A1b arm
        assert sd.pbrs_nav_weight_t == pytest.approx(0.07)             # non-`reward_`-prefixed
        assert sd.reward_kill == pytest.approx(
            RewardWeights().reward_kill), "untouched weight must keep its default"
    finally:
        env.close()


def test_env_factory_without_overrides_keeps_defaults():
    """`config=None` must reproduce today's env exactly."""
    import multiprocessing as mp

    from cs2rl.train import envs as train_envs

    factory = train_envs.build_env_factory(shared_ts=mp.Value("f", 0.3), map_data=None)
    env = factory(seed=0)
    try:
        sd = env._c_env.sd.contents
        for name, default in RewardWeights().as_dict().items():
            assert getattr(sd, name) == pytest.approx(default), name
    finally:
        env.close()


def test_env_factory_rejects_unexpected_kwargs():
    """Review fix 1: the catch-all **kwargs must be fatal, not silent.

    pufferlib only passes buf/seed/env_kwargs[i], all named parameters, so a
    stray kwarg means someone routed reward keys through _per_env_kwargs —
    the discard trap. Crash instead of training the baseline.
    """
    import multiprocessing as mp

    from cs2rl.train import envs as train_envs

    factory = train_envs.build_env_factory(shared_ts=mp.Value("f", 0.3), map_data=None)
    with pytest.raises(TypeError, match="reward_ct_survival"):
        factory(seed=0, reward_ct_survival=0.0)


def _closure_cells(fn):
    """{free variable -> captured value} for a closure, by name.

    `co_freevars` and `__closure__` are positionally aligned; `strict=True` makes
    a length mismatch an error instead of a silently truncated dict.
    """
    return dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__),
                    strict=True))


def test_build_train_env_factory_carries_args_config():
    """Review fix 2: the train() → factory seam, without launching a run.

    TWO closures, because they pin two different lines and the second one used
    to be uncovered (review finding, Important 3).

      the args-built factory — if the wiring ever regresses to `config=None` (or
                    to a config built from something other than args), the
                    non-default weight below stops arriving and this fails.
      the BARE factory — pins `config = EnvConfig() if config is None else config`
                    in build_env_factory, i.e. that the resolution happens ABOVE
                    the closure so the cell holds an EnvConfig and never a None.
                    The first call CANNOT see that line: build_train_env_factory
                    always passes `env_config_from_args(args)`, never None, so
                    the cell holds an EnvConfig whether or not the resolution
                    exists. Deleting the line left the pre-fix version of this
                    test green (re-measured 2026-09-04; the review measured the
                    whole non-slow suite green with it) while this docstring and
                    two `src/` paragraphs — build_env_factory's own and
                    _build_train's in env/factory.py — cited this test as the
                    evidence for it. Only a call that OMITS config reaches the
                    resolution; five in tests/ do, and this is the one that
                    asserts what it resolved to.

    WHY THE PROPERTY IS WORTH A LINE: the cell crosses the fork boundary into
    every vecenv worker. A None there is a None `_build_train` would forward to
    `make_env` — harmless today, because `make_env` resolves None itself, and
    that is precisely what makes the regression silent rather than loud.
    """
    import multiprocessing as mp
    from argparse import Namespace

    from cs2rl.train import envs as train_envs
    from cs2rl.train.config import env_config_from_args

    args = Namespace(reward_ct_survival=0.0)
    factory = train_envs.build_train_env_factory(args, shared_ts=mp.Value("f", 0.3), map_data=None)
    cells = _closure_cells(factory)
    assert cells["config"] == env_config_from_args(args)
    assert cells["config"].rewards.reward_ct_survival == 0.0

    bare = train_envs.build_env_factory(shared_ts=mp.Value("f", 0.3), map_data=None)
    bare_cells = _closure_cells(bare)
    assert bare_cells["config"] == EnvConfig(), (
        f"build_env_factory(config=None) captured {bare_cells['config']!r}; the "
        f"`config=None ⇒ EnvConfig()` resolution must run ABOVE the closure, so that no forked "
        f"worker is ever handed a None to guard against")


def test_train_uses_build_train_env_factory():
    """Pin train()'s call site itself — the one line no test can execute.

    PITFALL: this is a source-text assertion, deliberately. Everything else in
    train() needs a real run to reach, and the failure this guards (dropping
    the run's config) is invisible at runtime: the arm just trains the baseline.
    If you legitimately rename the helper, update this string.
    """
    import inspect

    from cs2rl.train import loop as train_loop

    src = inspect.getsource(train_loop.train)
    assert "build_train_env_factory(" in src, "train() no longer builds envs through the seam"
