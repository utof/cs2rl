"""`env_factory.build_env_for` must construct exactly what the old sites did.

WHY THIS FILE EXISTS. Spec 2026-08-31 §2 W3 routes every `make_puffer_env`
construction through one role-keyed factory. The failure that migration risks is
not a crash — it is a kwarg quietly going missing, which produces a WORKING env
built on a default. That failure is invisible to almost every gate this branch
has:

  * the §3 determinism gate sets no `--reward-*` and no R0-G flag, so a factory
    that dropped `reward_overrides`, `reward_symmetrize` or any omittable env
    knob still writes a BYTE-IDENTICAL `dust2_policy.pt`. This is not
    hypothetical: build_env_factory's own docstring records reward keys being
    silently dropped once, which would have made every experiment arm train the
    default weights;
  * `static_data_scalars()` is blind to 8 of make_env's 39 kwargs — `auto_reset`,
    `buf`, `include_step_stats_in_info`, `map_data`, `recoil`,
    `reward_symmetrize`, `seed`, `team_spirit` — and those 8 are precisely what
    distinguishes the roles from each other;
  * the full suite never passes `crouch_enabled` or `jump_enabled` to the
    harness builder, and their defaults equal `make_puffer_env`'s, so dropping
    either from the harness role is green across all ~1075 tests;
  * `--eval-interval` defaults to 0, so the `eval` role's env — the one whose
    `auto_reset=False` eval_baselines raises without — is never constructed
    during the §3 run or any train test.

So the oracle is a CAPTURE, taken before the factory existed
(`tests/fixtures/env_kwargs_pre_w3.json`, recorded by
`tests/capture_env_kwargs_pre_w3.py` one commit earlier). Comparing against a
list transcribed from the factory would compare the factory to itself.

WHAT IS ASSERTED, per role and per recorded scenario: the kwarg NAME SET and
every VALUE that `make_puffer_env` receives. The name set matters on its own —
a factory that passed `crouch_enabled=1` explicitly where the old site relied on
the default is value-identical today and diverges the day that default moves.

WHY THE ARGUMENTS ARE SENTINEL STRINGS. `make_puffer_env` is replaced by a
recording stub, so nothing validates them, and a DISTINCT sentinel per
pass-through argument turns "routed map_data into the team_spirit slot" into a
value mismatch instead of two equal-looking real objects comparing equal.

KNOCK-OUT COVERAGE. Every role's builder is knocked out — one kwarg deleted from
its construction — and this file asserts that role's captured-kwargs test then
FAILS. A captured-kwargs test that passes against a mutilated factory is
measuring nothing, and nothing else in the suite would tell us.

SCOPE. All six roles' call sites are now migrated — the two closures first, the
remaining four (smoke, both eval_legacy sites, external, eval) after. Zero direct
`make_puffer_env(...)` calls remain in `src/` or `scripts/`, which
`tests/test_env_construction_enforcement.py` asserts permanently.

FACTORY vs CALL SITE. Most of this file hands the fixture's bindings to
`build_env_for` by hand, which measures the FACTORY only — a call site that
fills the factory's slots wrongly passes every one of those tests. The tests
under the "MIGRATED CALL SITES" banner close that gap, in two layers, because no
single layer reaches every site:

  * a LIVE drive per site that can be reached without a training run (the two
    closures, smoke_test, make_env, and both eval_legacy sites). This is the only
    layer that can catch a SWAP — two arguments crossed between slots — which is
    why the harness drive feeds one distinct sentinel per closure variable;
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
import json
from pathlib import Path

import pytest

from env_factory import ROLES, UNSET, build_env_for

FIXTURE = Path(__file__).parent / "fixtures" / "env_kwargs_pre_w3.json"

# Must match capture_env_kwargs_pre_w3.CAPTURE_FORMAT. Duplicated rather than
# imported so a stale fixture fails on the tag here, in the file that consumes
# it, rather than on a KeyError deep inside a comparison.
CAPTURE_FORMAT = "cs2rl-env-kwargs-capture-v1"


def _fixture():
    with FIXTURE.open() as fh:
        data = json.load(fh)
    assert data["_provenance"]["format"] == CAPTURE_FORMAT, (
        f"{FIXTURE.name} was written in format {data['_provenance']['format']!r}, this module "
        f"reads {CAPTURE_FORMAT!r} — regenerate it with --capture")
    return data


FIXTURE_DATA = _fixture()


class _Recorder:
    """Stands in for `make_puffer_env` and records what the factory handed it.

    Records the call as a plain dict of what was PASSED, mirroring the capture
    script's `explicit_kwargs`. Signature binding is not repeated here: the
    builders call `make_puffer_env` with keywords only, so the kwargs dict IS
    the bound explicit set, and test_factory_kwargs_bind_to_the_real_signature
    checks the whole recorded set against the real signature separately.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        return "<env>"


def _construct(monkeypatch, role, **kwargs):
    """Run `build_env_for(role, ...)` against a recording stub; return the kwargs.

    Patches `train.make_puffer_env`, which is where `build_env_for`'s
    function-local import reads from — so this also proves that import is a
    per-call attribute read rather than something cached at module scope.
    """
    import train

    rec = _Recorder()
    monkeypatch.setattr(train, "make_puffer_env", rec)
    build_env_for(role, **kwargs)
    assert len(rec.calls) == 1, f"role {role!r} called make_puffer_env {len(rec.calls)} times"
    return rec.calls[0]


# ── the scenario bindings -> build_env_for kwargs adapters ──────────────────
#
# For train and harness the factory's parameter names are deliberately the SAME
# as the closures' free-variable names, so their adapter is the identity. The
# other four roles' pre-migration call sites closed over things the factory does
# not take (an argparse Namespace, `make_env`'s own parameters), so they get an
# explicit two-line adapter each rather than a clever generic one.


def _inputs_for(role, capture):
    b = capture["bindings"]
    if role in ("train", "harness"):
        return dict(b)
    if role == "eval":
        # The call site derives these from args; the factory takes the derived
        # values. `resolved` is recorded in the fixture BY the capture script,
        # from the real helpers — not re-derived here from the oracle.
        return {
            "map_data": b["_map_data"],
            "reward_overrides": capture["resolved"]["reward_overrides"],
            "env_knobs": capture["resolved"]["env_knobs"],
        }
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


@pytest.mark.parametrize(("role", "scenario", "capture"), _cases(), ids=_ids())
def test_factory_reproduces_the_pre_migration_kwargs(monkeypatch, role, scenario, capture):
    """THE oracle: the factory hands make_puffer_env what the old site handed it.

    Failure here is not "the factory is shaped differently" — it is "the env
    this role builds is not the env it used to build". Read the diff as a
    behaviour change and check `capture["call_source"]`, which is the
    pre-migration call verbatim, before touching the fixture.
    """
    got = _construct(monkeypatch, role, **_inputs_for(role, capture))
    expected = capture["explicit_kwargs"]
    assert sorted(got) == sorted(expected), (f"role {role!r}/{scenario}: kwarg SET changed.\n"
                                             f"  missing: {sorted(set(expected) - set(got))}\n"
                                             f"  added:   {sorted(set(got) - set(expected))}\n"
                                             f"  pre-migration call was: {capture['call_source']}")
    assert got == expected, (f"role {role!r}/{scenario}: kwarg VALUES changed.\n"
                             f"  pre-migration call was: {capture['call_source']}")


@pytest.mark.parametrize(("role", "scenario", "capture"), _cases(), ids=_ids())
def test_effective_env_config_survives_make_puffer_env_default_drift(monkeypatch, role, scenario,
                                                                     capture):
    """The kwargs AFTER defaults are applied still match the pre-W3 capture.

    Distinct failure from the test above, deliberately kept separate so the
    reason is unambiguous when it fires. That one compares what the factory
    PASSES; this compares the env that RESULTS, by filling in
    `make_puffer_env`'s own defaults. A default moving (say `jump_enabled` 1→0)
    leaves every passed-kwarg set identical while silently changing the env for
    every role that relies on the default — which is most of them, and is the
    ENTIRE config of `eval_legacy`, whose documented contract is that "the
    defaults reproduce the pre-Rung-0 env exactly".

    If this fires alone, the factory is fine and a make_puffer_env default
    changed; decide whether that change was meant to reach these roles before
    re-capturing.
    """
    import inspect

    import train

    # Read the signature BEFORE _construct swaps in the recording stub —
    # monkeypatch only reverts at teardown, so afterwards this would be the
    # stub's `**kwargs` and the whole assertion would collapse to comparing an
    # empty default set.
    sig = inspect.signature(train.make_puffer_env)
    got = _construct(monkeypatch, role, **_inputs_for(role, capture))
    bound = sig.bind(**got)
    bound.apply_defaults()
    assert dict(bound.arguments) == capture["effective_kwargs"], (
        f"role {role!r}/{scenario}: the EFFECTIVE env config changed. Passed kwargs and "
        "make_puffer_env's defaults together no longer reproduce the pre-W3 env.")


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
    """`build_env_factory`'s STRICT catch-all survived the W3 rewrite.

    The guard raises on anything pufferlib did not name, and it is the loud form
    of the historical bug: a reward key routed through `_per_env_kwargs` used to
    be silently discarded, so the arm trained the baseline weights. It had ZERO
    test coverage while W3 rewrote the very closure body it guards, and the
    captured-kwargs oracle above cannot see it — that oracle only ever observes
    the happy path, i.e. what `make_puffer_env` receives when nothing strays.

    Asserted through the real `build_env_factory`, not the factory module: the
    guard belongs to the closure, which is where pufferlib's kwargs arrive.
    """
    import train

    monkeypatch.setattr(train, "make_puffer_env", _Recorder())
    factory = train.build_env_factory(shared_ts=None, map_data=None)
    with pytest.raises(TypeError, match="unexpected kwargs.*reward_ct_survival"):
        factory(buf=None, seed=0, reward_ct_survival=0.0)


def test_eval_legacy_absent_seed_is_not_seed_none(monkeypatch):
    """The two eval_legacy sites differ ONLY in whether `seed` is passed.

    `make_puffer_env`'s default is `seed=0`, so spelling the absent case as
    `seed=None` would forward None where the bare call forwarded nothing — a
    real behaviour change that `static_data_scalars()` cannot see, because seed
    is not a StaticData field. The captured-kwargs test above already pins the
    bare shape; this states the sentinel is the mechanism, so a future
    "simplification" to `seed=None` fails here with the reason attached.
    """
    assert _construct(monkeypatch, "eval_legacy") == {}
    assert _construct(monkeypatch, "eval_legacy", seed=UNSET) == {}
    assert _construct(monkeypatch, "eval_legacy", seed=None) == {"seed": None}


def test_every_role_builder_parameter_is_required():
    """No role builder may default a knob — `eval_legacy`'s seed sentinel aside.

    A default turns a call site that forgot an argument into a WORKING env built
    on someone else's value, which is the entire failure class W3 removes. The
    `external` role is the concrete instance: its two parameters used to default
    to None, mirroring `make_env`'s published signature, so a delegate that
    forwarded only `team_spirit` would have silently produced a dust2 env instead
    of raising. The defaulting belongs to the wrapper, not to the builder.

    `seed=UNSET` is the one exemption, and it is the opposite of a default: the
    sentinel exists precisely BECAUSE `seed=None` would be a silent behaviour
    change (make_puffer_env's own default is 0), so it is asserted to be the
    sentinel rather than merely allowed to be anything.
    """
    import inspect

    import env_factory

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
    """Every kwarg every role passes is a real `make_puffer_env` parameter.

    The recorder accepts anything, so the captured-kwargs comparison would
    happily pass a typo'd kwarg name through as long as the FIXTURE carried the
    same typo. Binding against the real signature closes that: the production
    call would have raised TypeError inside a forked vecenv worker, far from the
    mistake.
    """
    import inspect

    import train

    sig = inspect.signature(train.make_puffer_env)
    for role, scenario, capture in _cases():
        got = _construct(monkeypatch, role, **_inputs_for(role, capture))
        try:
            sig.bind(**got)
        except TypeError as exc:
            pytest.fail(f"role {role!r}/{scenario} passes kwargs make_puffer_env rejects: {exc}")


# ── the two MIGRATED CALL SITES, driven for real ────────────────────────────
#
# Everything above feeds the fixture's bindings to `build_env_for` directly, so
# it proves the FACTORY reproduces the pre-migration kwargs and nothing about
# whether the closures T5a migrated hand it those bindings. The gap is not
# theoretical: swapping `crouch_enabled=jump_enabled, jump_enabled=crouch_enabled`
# at the harness call site leaves every test above green, and the §3
# determinism gate never builds a harness env at all, so nothing in the repo
# fired. (The train site has an indirect oracle — a `seed=_seed, _seed=seed`
# swap there moves the gate's checkpoint md5 — but only that one gate, and only
# for that one kwarg pair.)
#
# So the two tests below drive the REAL closures, the one `build_env_factory`
# returns and the one `_build_trainer_for_test` defines, against the same
# pre-migration capture.


def _kwarg_diff(got, expected):
    """Per-key report of how two kwarg dicts differ; empty string when equal.

    Used instead of a bare `assert got == expected` so that a routing bug names
    the slots it crossed — "crouch_enabled: got '<jump_enabled>'" — instead of
    printing two nine-key dicts and leaving the reader to diff them.
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


def _call_source_routing(capture):
    """{make_puffer_env kwarg -> the closure variable the OLD call read for it}.

    Parsed out of `capture["call_source"]`, the pre-migration call recorded
    verbatim one commit before the factory existed. Deriving the routing from
    the post-migration call site (or from the factory) instead would compare the
    migration to itself, which is the whole failure this file exists to avoid.

    Only bare-name arguments are routable. `seed=0 if seed is None else seed`
    and `include_step_stats_in_info=True` are derived/constant, and the caller
    checks those against the capture's recorded VALUE instead.
    """
    import ast

    call = ast.parse(capture["call_source"], mode="eval").body
    return {
        kw.arg: kw.value.id
        for kw in call.keywords if kw.arg is not None and isinstance(kw.value, ast.Name)
    }


def _real_harness_env_factory(monkeypatch, tmp_path, shared_ts, **harness_kwargs):
    """Extract the REAL `env_factory` closure `_build_trainer_for_test` builds.

    Taken from `pufferlib.vector.make`'s first argument, at which point the
    harness build is aborted: everything AFTER that call (build_policy, PuffeRL,
    the self-play patches) costs seconds and constructs nothing this file looks
    at, while everything before it — the team-spirit Value, the mask shm, the
    closure itself — is the wiring under test. The abort is an exception rather
    than a stub return value so the harness cannot run on against a fake vecenv
    and fail somewhere confusing.

    `mp` and `tempfile` are replaced on the HARNESS MODULE, not on the stdlib
    modules themselves: `shared_ts` is built inside the function and cannot be
    passed in, so it needs a stub to become an observable sentinel, and the
    scratch dir must not leak from a build that never reaches its own
    `cleanup()`. Module-scoped patches keep both out of every other test.
    """
    import types

    import pufferlib.vector

    import train_test_harness

    captured = []

    class _Captured(Exception):
        pass

    def _fake_make(env_creators, *_args, **_kwargs):
        captured.append(env_creators[0])
        raise _Captured

    monkeypatch.setattr(pufferlib.vector, "make", _fake_make)
    monkeypatch.setattr(train_test_harness, "mp",
                        types.SimpleNamespace(Value=lambda *_a, **_kw: shared_ts))
    monkeypatch.setattr(train_test_harness, "tempfile",
                        types.SimpleNamespace(mkdtemp=lambda **_kw: str(tmp_path)))
    with pytest.raises(_Captured):
        train_test_harness._build_trainer_for_test(**harness_kwargs)
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

    The fixture's own binding values are used verbatim, no sentinels needed:
    `seed` 3 against `_seed` 41, plus the distinct `<shared_ts>` / `<map_data>`
    strings and the two dissimilar dicts the capture already carries, make every
    slot in this call distinguishable from every other.
    """
    import train

    b = capture["bindings"]
    rec = _Recorder()
    monkeypatch.setattr(train, "make_puffer_env", rec)
    factory = train.build_env_factory(shared_ts=b["shared_ts"],
                                      map_data=b["map_data"],
                                      reward_overrides=b["reward_overrides"],
                                      reward_symmetrize=b["reward_symmetrize"],
                                      env_knobs=b["env_knobs"])
    factory(buf=b["buf"], seed=b["seed"], _seed=b["_seed"])
    assert len(rec.calls) == 1, f"the train closure called make_puffer_env {len(rec.calls)} times"
    diff = _kwarg_diff(rec.calls[0], capture["explicit_kwargs"])
    assert not diff, (f"train CALL SITE / {capture['scenario']}: what build_env_factory's closure "
                      f"forwards no longer matches the pre-W3 capture.\n{diff}\n"
                      f"  pre-migration call was: {capture['call_source']}")


@pytest.mark.parametrize("capture",
                         FIXTURE_DATA["roles"]["harness"],
                         ids=[c["scenario"] for c in FIXTURE_DATA["roles"]["harness"]])
def test_harness_call_site_forwards_the_captured_kwargs(monkeypatch, tmp_path, capture):
    """`_build_trainer_for_test`'s closure, run for real, still produces the capture.

    This is the site with no other oracle at all: the §3 gate cannot see the
    harness, and no test anywhere passes `crouch_enabled` or `jump_enabled` to
    `_build_trainer_for_test`.

    WHY SENTINELS RATHER THAN THE FIXTURE'S OWN BINDING VALUES. The two harness
    captures record crouch/jump as (1, 1) and (0, 0), so a call site that
    swapped the two would reproduce both captures exactly — value comparison is
    structurally blind to it. Feeding one distinct sentinel per closure variable
    makes the SLOT each variable lands in observable, and
    `_call_source_routing` reads the required slot off the pre-migration call
    rather than off the code being tested. `seed` and
    `include_step_stats_in_info` are not routed variables, so those two are
    still compared against the capture's recorded value — which is what
    exercises the `0 if seed is None else seed` remap in both directions.
    """
    import train

    routing = _call_source_routing(capture)
    sentinel = {var: f"<{var}>" for var in routing.values()}
    # Everything routed except `shared_ts`, which _build_trainer_for_test builds
    # itself (hence the mp stub), and `buf`, which pufferlib passes per call.
    injectable = sorted(set(routing.values()) - {"shared_ts", "buf"})

    rec = _Recorder()
    monkeypatch.setattr(train, "make_puffer_env", rec)
    knobs = {name: sentinel[name] for name in injectable}
    factory = _real_harness_env_factory(monkeypatch,
                                        tmp_path,
                                        sentinel["shared_ts"],
                                        num_envs=1,
                                        **knobs)
    factory(buf=sentinel["buf"], seed=capture["bindings"]["seed"])

    assert len(rec.calls) == 1, f"the harness closure called make_puffer_env {len(rec.calls)}x"
    expected = {
        kwarg: sentinel[routing[kwarg]] if kwarg in routing else value
        for kwarg, value in capture["explicit_kwargs"].items()
    }
    diff = _kwarg_diff(rec.calls[0], expected)
    assert not diff, (
        f"harness CALL SITE / {capture['scenario']}: what _build_trainer_for_test's closure "
        f"forwards no longer matches the pre-W3 capture.\n{diff}\n"
        f"  pre-migration call was: {capture['call_source']}")


# ── knock-outs: prove the oracle can actually fail ──────────────────────────
#
# One dropped kwarg per role, applied to the role's builder, asserting the
# captured-kwargs test then fails. Without this the whole file could be green
# against a factory that passed nothing at all — which is exactly what a
# defaults-producing bug looks like.

# One kwarg per role, each chosen as the one whose loss is HARDEST to see
# anywhere else — a knock-out on an easily-noticed kwarg would prove much less.
#
#   train        reward_overrides: the historical bug verbatim. Losing it makes
#                every experiment arm train the baseline weights while the §3
#                checkpoint stays byte-identical.
#   eval         auto_reset: unreachable by every run-level gate on this branch,
#                and eval_baselines raises without it.
#   eval_legacy  seed: the only thing separating this role's two call sites, and
#                not a StaticData field, so scalars cannot see it either.
#   harness      crouch_enabled: no test passes it and its default matches
#                make_puffer_env's, so the whole suite is blind to it.
#   smoke        seed / external map_data: each role's entire payload.
#
# NOTE the comments live above the dict, not inside it. Trailing comments in a
# literal get snapped to yapf's comment stops while ruff's isort wants one
# space, and the two then fight forever (gh#97, and the warning in train.py's
# import block).
_KNOCKOUT_KWARG = {
    "train": "reward_overrides",
    "eval": "auto_reset",
    "eval_legacy": "seed",
    "smoke": "seed",
    "harness": "crouch_enabled",
    "external": "map_data",
}


@pytest.mark.parametrize("role", ROLES)
def test_knockout_dropping_one_kwarg_fails_that_roles_capture(monkeypatch, role):
    """Delete one kwarg from a role's construction; that role's oracle must fail.

    The drop is simulated at the recording boundary rather than by editing
    `env_factory.py`: the stub discards the named kwarg before recording, which
    is indistinguishable — from the oracle's point of view — from a builder that
    never passed it. That keeps the knock-out reproducible in CI instead of a
    procedure someone has to remember to perform by hand.
    """
    import train

    dropped = _KNOCKOUT_KWARG[role]
    captures = [c for r, _, c in _cases() if r == role and dropped in c["explicit_kwargs"]]
    assert captures, (f"knock-out kwarg {dropped!r} appears in no captured scenario for role "
                      f"{role!r} — it cannot demonstrate anything")

    class _Dropping(_Recorder):

        def __call__(self, **kwargs):
            kwargs.pop(dropped, None)
            return super().__call__(**kwargs)

    for capture in captures:
        rec = _Dropping()
        monkeypatch.setattr(train, "make_puffer_env", rec)
        build_env_for(role, **_inputs_for(role, capture))
        got = rec.calls[0]
        assert got != capture["explicit_kwargs"], (
            f"role {role!r}/{capture['scenario']}: dropping {dropped!r} from the construction "
            "still matched the capture — this role's oracle is vacuous")


def test_knockout_the_fixture_itself_is_load_bearing(monkeypatch):
    """Corrupt a captured VALUE and the oracle must notice.

    Distinct from the kwarg-drop knock-outs above, which perturb the factory.
    This perturbs the EXPECTATION, and catches a comparison that only ever looks
    at key sets — `sorted(got) == sorted(expected)` alone would pass here.
    """
    capture = copy.deepcopy(FIXTURE_DATA["roles"]["train"][0])
    assert capture["explicit_kwargs"]["seed"] == 41
    capture["explicit_kwargs"]["seed"] = 999
    got = _construct(monkeypatch, "train", **_inputs_for("train", capture))
    assert sorted(got) == sorted(capture["explicit_kwargs"])
    assert got != capture["explicit_kwargs"]


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
    CALLING module rather than on `env_factory` is deliberate: both call sites'
    modules do `from env_factory import build_env_for`, so the module-global is
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
    import train

    role, kwargs = _drive(monkeypatch, train, train.smoke_test)
    assert (role, kwargs) == ("smoke", {})


def test_make_env_delegates_to_the_external_role(monkeypatch):
    """The public wrapper forwards both of its parameters, unswapped.

    Distinct sentinels: `make_env(team_spirit, map_data)` takes two positional
    parameters of the same shape, so crossing them is a one-character edit that
    every value-based comparison in this file would accept.
    """
    import train

    seen = []
    monkeypatch.setattr(train, "build_env_for", lambda role, **kw: seen.append((role, kw)))
    train.make_env("<team_spirit>", "<map_data>")
    assert seen == [("external", {"team_spirit": "<team_spirit>", "map_data": "<map_data>"})]

    # The wrapper's own optional defaults stay on the wrapper — `_build_external`
    # requires both, so a forwarding bug is a TypeError rather than a dust2 env.
    seen.clear()
    train.make_env()
    assert seen == [("external", {"team_spirit": None, "map_data": None})]


def test_load_policy_from_checkpoint_asks_for_bare_eval_legacy(monkeypatch, tmp_path):
    """The bare eval_legacy site: role only, no kwargs, no seed.

    `torch.load` is stubbed because the construction sits AFTER the checkpoint
    read, and this test is about the construction. The stub returns the one key
    the loader inspects, so the function reaches the factory the same way a real
    checkpoint would.
    """
    import types

    import torch

    import train

    ckpt = tmp_path / "fake.pt"
    ckpt.write_bytes(b"")
    monkeypatch.setattr(
        torch, "load",
        lambda *_a, **_kw: {"encoder.0.weight": types.SimpleNamespace(shape=(64, 105))})

    role, kwargs = _drive(monkeypatch, train, train.load_policy_from_checkpoint, ckpt, "cpu")
    assert (role, kwargs) == ("eval_legacy", {}), (
        "load_policy_from_checkpoint must pass NO seed — make_puffer_env's own default is 0, "
        "and forwarding None instead would build a different env that scalars cannot see")


def test_evaluate_checkpoint_threads_its_episode_seed(monkeypatch):
    """The other eval_legacy site: same role, but it MUST pass a seed.

    `policy_mode="random"` with no checkpoint skips the policy load entirely, so
    the first loop iteration reaches the construction directly. That the seed is
    the per-episode `start_seed + episode_idx` rather than `start_seed` is not
    observable from episode 0 — the AST wiring test below is what pins the
    expression; this pins that the seed reaches the factory at all, under the
    right role.
    """
    import train

    role, kwargs = _drive(monkeypatch,
                          train,
                          train.evaluate_checkpoint,
                          checkpoint_path=None,
                          policy_mode="random",
                          start_seed=7717,
                          num_episodes=1)
    assert (role, kwargs) == ("eval_legacy", {"seed": 7717})


# ── AST: every migrated site still reads what the old site read ─────────────
#
# The layer that reaches `train()`'s eval site, which no test can drive. Each
# check is derived from `capture["call_source"]` — the pre-migration call,
# recorded verbatim one commit before the factory existed — so it compares the
# migrated site to the OLD site rather than to the builder it now calls.

SRC = Path(__file__).resolve().parents[1] / "src"

# The one thing not derivable from the capture: which kwarg a `**splat` became.
# Both splats carry the knob dict; the `or {}` in the train site's
# `**env_knobs or {}` moved INTO the builder, so only the free NAMES are
# comparable there, not the expression.
_SPLAT_BECOMES = {"train": "env_knobs", "eval": "env_knobs"}


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
    stops reading something it used to read — a dropped `reward_overrides`, or an
    `evaluate_checkpoint` that passed `start_seed` instead of `seed`.

    Set-based, and therefore blind to two arguments SWAPPED between slots. That
    is what the live drives above are for.
    """
    call = ast.parse(call_source, mode="eval").body
    return {n.id for kw in call.keywords for n in ast.walk(kw.value) if isinstance(n, ast.Name)}


def _is_derivation(expr):
    """True for a conditional/boolean expression — a RULE rather than a value.

    Exactly the two shapes W3 was supposed to move into the builders (both
    closures' seed remaps, `**env_knobs or {}`'s None guard). Anything else — a
    name, an attribute, a helper call — is a value the call site sources, and
    those must still be spelled identically after the migration.
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


def _migrated_sites():
    """(role, enclosing qualname, source file, one pre-migration call_source).

    Read entirely out of the frozen fixture: it records each capture's `role`,
    its `enclosing` qualname and its `site` as `<relpath>:<lineno>`, so the whole
    table — including which file each site lives in — comes from the
    pre-migration snapshot rather than from a list someone transcribed off the
    migrated code.
    """
    sites = {}
    for role, caps in FIXTURE_DATA["roles"].items():
        for cap in caps:
            key = (role, cap["enclosing"], cap["site"].split(":")[0])
            sites.setdefault(key, set()).add(cap["call_source"])
    out = []
    for (role, enclosing, relpath), sources in sorted(sites.items()):
        assert len(sources) == 1, (f"{role}/{enclosing} was captured with {len(sources)} different "
                                   f"call sources: {sources}")
        out.append((role, enclosing, Path(__file__).resolve().parents[1] / relpath, sources.pop()))
    return out


@pytest.mark.parametrize(("role", "enclosing", "path", "call_source"),
                         _migrated_sites(),
                         ids=[f"{r}-{e}" for r, e, _, _ in _migrated_sites()])
def test_migrated_site_still_reads_what_the_old_site_read(role, enclosing, path, call_source):
    """Per site: the role is right, the names are the same, the constants left.

    Three assertions, each catching a different way the migration could be wrong:

      ROLE — the site asks for its own role. `smoke_test` asking for
      `eval_legacy` builds a working env with the wrong seed and nothing else in
      the repo notices.

      NAMES — the set of free names the call reads is unchanged. A call site that
      quietly stopped forwarding `reward_overrides` (the historical bug: every
      experiment arm trains the baseline weights, checkpoint byte-identical) or
      that forwards `start_seed` where it used to forward `seed` fails here.
      Per-kwarg EXPRESSIONS are compared too, for every kwarg both calls name.

      CONSTANTS — every literal the old call passed is GONE from the new one.
      This is what asserts the migration actually happened: `seed=42` still
      spelled at the smoke site, or `auto_reset=False` still at the eval site,
      means the payload was duplicated rather than moved, and the two copies can
      then drift — which is the entire failure W3 exists to end.
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

    assert _free_names(new_source) == _free_names(call_source), (
        f"{role}/{enclosing}: the names this construction reads changed.\n"
        f"  no longer read: {sorted(_free_names(call_source) - _free_names(new_source))}\n"
        f"  newly read:     {sorted(_free_names(new_source) - _free_names(call_source))}\n"
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
            # picks is the captured-kwargs oracle's job — it covers all three of
            # the train remap's branches.
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
    for splat in old_splats:
        kwarg = _SPLAT_BECOMES[role]
        assert kwarg in new_named, (
            f"{role}/{enclosing} dropped the `**{splat}` the old call splatted; it must now be "
            f"passed as {kwarg}=")
        assert _free_names(f"f(x={splat})") <= _free_names(f"f(x={new_named[kwarg]})"), (
            f"{role}/{enclosing}: {kwarg}={new_named[kwarg]!r} no longer reads what "
            f"`**{splat}` read")


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
    import ast

    import env_factory

    tree = ast.parse(Path(env_factory.__file__).read_text())
    attached = [
        n.func.attr for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in (
            "_attach_mask_view", "_attach_cont_action_view")
    ]
    assert not attached, (f"env_factory.py calls {attached} — shared-memory attach is the "
                          "caller's job; see this test's docstring")
