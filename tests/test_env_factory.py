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

SCOPE. T5a migrates the two CLOSURE sites (train, harness); the other four
roles' builders exist here and are checked against the same capture, but their
CALL SITES still construct directly and are migrated in T5b.
"""
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
