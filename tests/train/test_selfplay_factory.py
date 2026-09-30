"""`env.factory.build_selfplay_manager` must construct what the three old sites did.

WHY THIS FILE EXISTS, separately from `test_env_factory.py`. W3 collapsed THREE
`SelfPlayManager(...)` constructions — `train()` and both branches of
`_build_trainer_for_test` — onto one builder. Nothing else in the repo can see
that go wrong:

  * the §3 determinism gate runs `--no-self-play`, so the manager it builds has
    `p_past=0.0` and an empty pool `should_use_past()` never activates. A builder
    that dropped `aim_log_std_max`, flipped `pin_pitch` or let `opponent_mode`
    fall to its default still writes a BYTE-IDENTICAL `dust2_policy.pt`;
  * the two harness sites are not reachable by ANY gate on this branch;
  * `aim_log_std_max` and `pin_pitch` are R0-E (#131) run properties re-applied
    to every past policy. Losing either does not crash — it makes past opponents
    sample a pitch dim the env ignores, so their stored `logprob_c` carries a
    factor the live policy's does not and self-play `ratio_c` drifts. Silent.

So the oracle is `tests/fixtures/selfplay_kwargs_pre_w3.json`, captured by
`tests/capture_selfplay_kwargs_pre_w3.py` one commit before the builder existed
(deleted since; `git show 9878725:tests/capture_selfplay_kwargs_pre_w3.py`).
Comparing against a list transcribed from the builder would compare the builder
to itself.

WHAT THE COLLAPSE RESTS ON, stated because it is the risky part: the three sites
differed ONLY in `p_past`, and the two values were the same rule twice
(`0.3 if <self-play on> else 0.0` in train(), spelled out as two literals in the
harness). The fixture records all three pre-migration shapes independently, and
this file drives each one through the single builder — so the claim "they were
the same construction" is asserted from the pre-migration source, not assumed.
"""
import ast
import copy
import json

import pytest

from cs2rl.train.selfplay import build_selfplay_manager
from tests.conftest import REPO_ROOT

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "selfplay_kwargs_pre_w3.json"

# Must match CAPTURE_FORMAT in the capture script (`git show
# 9878725:tests/capture_selfplay_kwargs_pre_w3.py`). Duplicated rather
# than imported so a stale fixture fails on the tag here, in the file that reads
# it, rather than on a KeyError deep inside a comparison.
CAPTURE_FORMAT = "cs2rl-selfplay-kwargs-capture-v1"

# The `self_play_enabled` value each captured HARNESS site corresponds to.
# Stated here rather than read out of the capture's own `p_past`, on purpose:
# deriving the input from the expected output would make every comparison below
# tautological (`p_past=0.3` in, `p_past=0.3` out, nothing measured). These two
# constants ARE the migration's claim — "the harness's two branches are the same
# builder called with False and with True" — so they are asserted, not computed.
#
# `train` is absent because it needs no entry: that site already had the flag as
# a free variable, so the capture's own bindings carry it per scenario.
_SELF_PLAY_ENABLED = {
    "harness_no_selfplay": False,
    "harness_selfplay": True,
}


def _fixture():
    with FIXTURE.open() as fh:
        data = json.load(fh)
    assert data["_provenance"]["format"] == CAPTURE_FORMAT, (
        f"{FIXTURE.name} was written in format {data['_provenance']['format']!r}, this module "
        f"reads {CAPTURE_FORMAT!r} — its capture script is only in git history now "
        f"(git show 9878725:tests/capture_selfplay_kwargs_pre_w3.py)")
    return data


FIXTURE_DATA = _fixture()


class _Recorder:
    """Stands in for `SelfPlayManager` and records what the builder handed it."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        return "<manager>"


def _construct(monkeypatch, **kwargs):
    """Run `build_selfplay_manager(...)` against a recording stub; return the kwargs.

    Patches `train.SelfPlayManager`, which is where the builder's function-local
    import reads from — so this also proves that import is a per-call attribute
    read rather than something cached at module scope.
    """
    from cs2rl.train import selfplay as train_selfplay

    rec = _Recorder()
    monkeypatch.setattr(train_selfplay, "SelfPlayManager", rec)
    build_selfplay_manager(**kwargs)
    assert len(rec.calls) == 1, f"build_selfplay_manager called SelfPlayManager {len(rec.calls)}x"
    return rec.calls[0]


def _inputs_for(site, capture):
    """The builder arguments corresponding to one captured scenario.

    Everything except `self_play_enabled` is read out of the capture's own
    BINDINGS — the free names the pre-migration call site closed over — never out
    of its `explicit_kwargs`. That distinction is what keeps these tests from
    comparing the fixture to itself: the bindings are the call site's INPUTS, the
    explicit kwargs are its OUTPUTS, and the builder is what has to get from one
    to the other.
    """
    b = capture["bindings"]
    if site == "train":
        args = b["args"]["__namespace__"]
        return {
                                                                       # `getattr(args, "aim_log_std_max", None)` verbatim — the migrated
                                                                       # call site still spells it that way, and the `absent` scenario is
                                                                       # what measures the fallback.
            "aim_log_std_max": args.get("aim_log_std_max"),
            "pin_pitch": args["pin_pitch"],
            "opponent_mode": b["_opponent_mode"],
            "self_play_enabled": b["self_play_enabled"],
        }
    return {
        "aim_log_std_max": b["aim_log_std_max"],
        "pin_pitch": b["pin_pitch"],
        "opponent_mode": b["opponent"],
        "self_play_enabled": _SELF_PLAY_ENABLED[site],
    }


def _cases():
    """(site, scenario, capture) for every recorded capture."""
    return [(site, cap["scenario"], cap) for site, caps in FIXTURE_DATA["sites"].items()
            for cap in caps]


def _ids():
    return [f"{site}-{scenario}" for site, scenario, _ in _cases()]


@pytest.mark.parametrize(("site", "scenario", "capture"), _cases(), ids=_ids())
def test_builder_reproduces_the_pre_migration_kwargs(monkeypatch, site, scenario, capture):
    """THE oracle: the builder hands SelfPlayManager what the old site handed it.

    Failure here is not "the builder is shaped differently" — it is "the manager
    this site builds is not the manager it used to build". Read the diff as a
    behaviour change and check `capture["call_source"]`, the pre-migration call
    recorded verbatim, before touching the fixture.
    """
    got = _construct(monkeypatch, **_inputs_for(site, capture))
    expected = capture["explicit_kwargs"]
    assert sorted(got) == sorted(expected), (f"{site}/{scenario}: kwarg SET changed.\n"
                                             f"  missing: {sorted(set(expected) - set(got))}\n"
                                             f"  added:   {sorted(set(got) - set(expected))}\n"
                                             f"  pre-migration call was: {capture['call_source']}")
    assert got == expected, (f"{site}/{scenario}: kwarg VALUES changed.\n"
                             f"  pre-migration call was: {capture['call_source']}")


@pytest.mark.parametrize(("site", "scenario", "capture"), _cases(), ids=_ids())
def test_builder_passes_every_parameter_explicitly(monkeypatch, site, scenario, capture):
    """No `SelfPlayManager` default is load-bearing for these three sites.

    All three pre-migration calls passed all eight parameters, so their
    `explicit_kwargs` and `effective_kwargs` are equal in the fixture — which
    makes a "does the effective config still match" test degenerate here, unlike
    its counterpart for `make_puffer_env` (where the roles lean heavily on
    defaults). What is NOT degenerate is the property that produced that
    equality, so that is what this asserts: the builder still names every
    parameter, and no site is silently relying on a class default that someone
    could move.

    When it fires: a NINTH parameter was added to `SelfPlayManager.__init__` with
    a default, and nobody decided whether these sites want it. That decision is
    the point — the failure says "choose", not "the builder is broken".
    """
    import inspect

    from cs2rl.train import selfplay as train_selfplay

    # Read the signature BEFORE _construct swaps in the stub; afterwards it would
    # be the recorder's `**kwargs` and this would collapse to comparing an empty
    # default set against itself.
    sig = inspect.signature(train_selfplay.SelfPlayManager)
    got = _construct(monkeypatch, **_inputs_for(site, capture))
    bound = sig.bind(**got)
    bound.apply_defaults()
    assert dict(bound.arguments) == capture["effective_kwargs"], (
        f"{site}/{scenario}: the EFFECTIVE manager config changed — the builder's kwargs plus "
        "SelfPlayManager's defaults no longer reproduce the pre-W3 manager.")
    assert set(got) == set(bound.arguments), (
        f"{site}/{scenario}: the builder no longer passes every SelfPlayManager parameter; "
        f"{sorted(set(bound.arguments) - set(got))} now come from class defaults. Decide "
        "whether these three sites want that value before pinning it.")


def test_builder_kwargs_bind_to_the_real_signature(monkeypatch):
    """Every kwarg the builder passes is a real `SelfPlayManager` parameter.

    The recorder accepts anything, so the comparison above would happily let a
    typo'd kwarg name through as long as the FIXTURE carried the same typo.
    Binding against the real signature closes that.
    """
    import inspect

    from cs2rl.train import selfplay as train_selfplay

    sig = inspect.signature(train_selfplay.SelfPlayManager)
    for site, scenario, capture in _cases():
        got = _construct(monkeypatch, **_inputs_for(site, capture))
        try:
            sig.bind(**got)
        except TypeError as exc:
            pytest.fail(f"{site}/{scenario} passes kwargs SelfPlayManager rejects: {exc}")


def test_p_past_is_derived_from_the_flag_not_taken_as_a_value(monkeypatch):
    """`self_play_enabled` is the parameter; `p_past` is not accepted at all.

    The three sites' whole difference was `p_past`. If the builder took it as an
    argument, every caller could pick its own mixing probability again and the
    collapse would have bought nothing — the tests above would still be green,
    because they only ever pass the values the fixture recorded.
    """
    with pytest.raises(TypeError, match="p_past"):
        _construct(monkeypatch,
                   self_play_enabled=True,
                   aim_log_std_max=None,
                   pin_pitch=0,
                   opponent_mode="self",
                   p_past=0.9)

    on = _construct(monkeypatch,
                    self_play_enabled=True,
                    aim_log_std_max=None,
                    pin_pitch=0,
                    opponent_mode="self")
    off = _construct(monkeypatch,
                     self_play_enabled=False,
                     aim_log_std_max=None,
                     pin_pitch=0,
                     opponent_mode="self")
    assert (on["p_past"], off["p_past"]) == (0.3, 0.0)
    assert {
        k: v
        for k, v in on.items() if k != "p_past"
    } == {
        k: v
        for k, v in off.items() if k != "p_past"
    }, "self_play_enabled changed something other than p_past"


def test_every_builder_parameter_is_required(monkeypatch):
    """No defaults on the builder — a dropped argument must be a TypeError.

    A default would make each of these silently un-settable: `aim_log_std_max`
    defaulting to None would un-pin the aim head's log-std cap on past policies
    (R0-E, #131), `pin_pitch` defaulting to False would let a pinned run mix in
    unpinned opponents, `opponent_mode` defaulting to "self" would turn a `noop`
    run's statue opponent back into a live one. None of the three crashes; all
    three change training.
    """
    import inspect

    complete = dict(self_play_enabled=True, aim_log_std_max=None, pin_pitch=0, opponent_mode="self")
    params = inspect.signature(build_selfplay_manager).parameters
    assert sorted(params) == sorted(complete), (
        f"build_selfplay_manager's parameters changed to {sorted(params)}; this test and "
        "the fixture adapters below it both need updating")
    for p in params.values():
        assert p.kind is p.KEYWORD_ONLY, f"{p.name} is not keyword-only"
        assert p.default is p.empty, (
            f"{p.name} acquired the default {p.default!r} — a dropped argument at a call site "
            "is now silent; see this test's docstring for what each one silently changes")

    for missing in complete:
        with pytest.raises(TypeError, match=missing):
            _construct(monkeypatch, **{k: v for k, v in complete.items() if k != missing})


# ── knock-outs: prove the oracle can actually fail ──────────────────────────

# One kwarg per site, each chosen as the one whose loss is hardest to see
# anywhere else:
#   train                aim_log_std_max — a run property, invisible to the §3
#                        gate (which passes no --aim-log-std-max) and to every
#                        no-self-play path, since it only reaches PAST policies.
#   harness_no_selfplay  opponent_mode — defaults to "self", so losing it turns a
#                        `noop` harness run's statue opponent live with no error.
#   harness_selfplay     pin_pitch — R0-E; a wrong value drifts ratio_c silently.
#
# NOTE the comments live above the dict, not inside it: trailing comments in a
# literal get snapped to yapf's comment stops while ruff's isort wants one space,
# and the two then fight forever (gh#97).
_KNOCKOUT_KWARG = {
    "train": "aim_log_std_max",
    "harness_no_selfplay": "opponent_mode",
    "harness_selfplay": "pin_pitch",
}


@pytest.mark.parametrize("site", sorted(_KNOCKOUT_KWARG))
def test_knockout_dropping_one_kwarg_fails_that_sites_capture(monkeypatch, site):
    """Delete one kwarg from the construction; that site's oracle must fail.

    The drop is simulated at the recording boundary rather than by editing
    `env/factory.py`: the stub discards the named kwarg before recording, which
    is — from the oracle's point of view — indistinguishable from a builder that
    never passed it. That keeps the knock-out reproducible in CI instead of a
    procedure someone has to remember to perform by hand.
    """
    from cs2rl.train import selfplay as train_selfplay

    dropped = _KNOCKOUT_KWARG[site]
    captures = [c for s, _, c in _cases() if s == site and dropped in c["explicit_kwargs"]]
    assert captures, (f"knock-out kwarg {dropped!r} appears in no captured scenario for {site!r} "
                      "— it cannot demonstrate anything")

    class _Dropping(_Recorder):

        def __call__(self, **kwargs):
            kwargs.pop(dropped, None)
            return super().__call__(**kwargs)

    for capture in captures:
        rec = _Dropping()
        monkeypatch.setattr(train_selfplay, "SelfPlayManager", rec)
        build_selfplay_manager(**_inputs_for(site, capture))
        assert rec.calls[0] != capture["explicit_kwargs"], (
            f"{site}/{capture['scenario']}: dropping {dropped!r} from the construction still "
            "matched the capture — this site's oracle is vacuous")


def test_knockout_the_fixture_itself_is_load_bearing(monkeypatch):
    """Corrupt a captured VALUE and the oracle must notice.

    Distinct from the kwarg-drop knock-outs, which perturb the builder. This
    perturbs the EXPECTATION, and catches a comparison that only ever looks at
    key sets — `sorted(got) == sorted(expected)` alone would pass here.
    """
    capture = copy.deepcopy(FIXTURE_DATA["sites"]["train"][0])
    assert capture["explicit_kwargs"]["pool_size"] == 15
    capture["explicit_kwargs"]["pool_size"] = 999
    got = _construct(monkeypatch, **_inputs_for("train", capture))
    assert sorted(got) == sorted(capture["explicit_kwargs"])
    assert got != capture["explicit_kwargs"]


# ── the three MIGRATED CALL SITES ───────────────────────────────────────────
#
# Everything above feeds the fixture's bindings to `build_selfplay_manager` by
# hand, which measures the BUILDER only: a call site that fills its slots wrongly
# passes every one of those tests. `pin_pitch=aim_log_std_max` swapped at the
# harness site would be green above and green across the whole suite.
#
# The harness site is driven for real below. train()'s site cannot be — reaching
# it needs a full training run — so it is checked against its pre-migration
# `call_source` over the AST, which is what the fixture recorded and therefore
# the one comparison that is not against the new code itself.


def _routing(call_source):
    """{kwarg -> the source text the call passed for it}, from an unparsed Call.

    `ast.unparse` of each argument rather than a bare Name id: all THREE captured
    SPM sites pass a wrapped `pin_pitch` — train()'s passes `bool(args.pin_pitch)`
    and the two `_build_trainer_for_test` branches (which the migration collapsed
    into one call) pass `bool(pin_pitch)` — and train()'s wraps its cap on top of
    that, as `getattr(args, 'aim_log_std_max', None)`, where those two branches
    pass a bare `aim_log_std_max`. A map keyed on `kw.value.id` would have no
    entry AT ALL for a wrapped argument, so it would route only the kwargs that
    happen to be bare names; train()'s site — the only one this function is
    applied to — is where both wrappings land. The env oracle's AST layer
    unparses per kwarg for
    the same reason (`test_env_factory._keywords`); its `_free_names` companion
    reads bare Name ids, but it unions them across ALL arguments instead of
    keying by kwarg, so it is a different comparison rather than a cheaper
    spelling of this one.
    """
    call = ast.parse(call_source, mode="eval").body
    return {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg is not None}


def _factory_call_in(path, qualname, func_name):
    """The one `func_name(...)` call inside `qualname`, as an ast.Call.

    Located by enclosing qualname, never by line number — the same discipline the
    capture script uses, and for the same reason.
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
                             f"found {len(found)}")
    return found[0]


PACKAGE = REPO_ROOT / "src" / "cs2rl"


def test_train_call_site_forwards_the_same_expressions_it_used_to():
    """train()'s migrated site passes the pre-migration argument EXPRESSIONS.

    WHY AST AND NOT A LIVE CALL: this site sits ~200 lines into `train()`, after
    the vecenv, the policy and the PuffeRL trainer are built; nothing short of a
    real run reaches it, and the §3 gate that does run it is `--no-self-play`
    and structurally blind to every kwarg here but `p_past`.

    WHAT IS COMPARED: the three pass-through arguments, as SOURCE TEXT, against
    `capture["call_source"]` — the pre-migration call recorded verbatim one
    commit before the builder existed. So this checks the migrated site against
    the OLD site, not against the builder it now calls.
    `pool_size`/`save_every_epochs`/`win_threshold`/`phase_length` moved INTO the
    builder by design and are covered by the captured-kwargs oracle above;
    `p_past` became the `self_play_enabled` flag, asserted separately here.
    """
    capture = FIXTURE_DATA["sites"]["train"][0]
    old = _routing(capture["call_source"])
    new = _routing(
        ast.unparse(
            _factory_call_in(PACKAGE / "train" / "loop.py", "train", "build_selfplay_manager")))

    for kwarg in ("aim_log_std_max", "opponent_mode"):
        assert new.get(kwarg) == old[kwarg], (
            f"train() now passes {kwarg}={new.get(kwarg)!r}; the pre-migration site passed "
            f"{old[kwarg]!r}. Pre-migration call: {capture['call_source']}")

    # pin_pitch's `bool(...)` moved into the builder ON PURPOSE (so the kwarg the
    # captured oracle compares is still the coerced one), hence the argument
    # itself must now be the BARE expression the old call wrapped.
    assert old["pin_pitch"] == "bool(args.pin_pitch)"
    assert new["pin_pitch"] == "args.pin_pitch", (
        f"train() passes pin_pitch={new.get('pin_pitch')!r}; expected the unwrapped "
        "`args.pin_pitch` — build_selfplay_manager applies the bool()")

    assert old["p_past"] == "0.3 if self_play_enabled else 0.0"
    assert new["self_play_enabled"] == "self_play_enabled", (
        f"train() passes self_play_enabled={new.get('self_play_enabled')!r}; the flag the old "
        f"p_past expression branched on was `self_play_enabled`")
    assert "p_past" not in new, "train() is passing p_past again; the rule belongs to the builder"


@pytest.mark.parametrize("with_selfplay", [False, True])
def test_harness_call_site_builds_the_captured_manager(monkeypatch, with_selfplay):
    """`_build_trainer_for_test`, run for real, still constructs the captured manager.

    This is the site with NO other oracle: the §3 gate cannot see the harness and
    nothing else on this branch reaches it. It is also the site the migration
    changed most — two constructions under `if not with_selfplay:` became one
    call taking the flag — so a live drive is worth its ~7 s.

    A SPY, NOT A STUB — and the difference is not stylistic. Aborting at the
    construction (recording, then raising) leaves the vecenv and the PuffeRL
    trainer built and unclosed, because `_build_trainer_for_test` hands its
    `cleanup()` back only on the way out; the interpreter then never exits.
    Measured: `pytest` and a bare `python -c` both hang indefinitely that way.
    So the spy records and DELEGATES to the real builder, the harness finishes,
    and the test closes it properly. Cost is the full ~7 s build per case.

    WHY THE SENTINELS ARE NOT STRINGS here, unlike the fixture's. These arguments
    are consumed by `build_train_config` on the way to the site
    (`float(aim_log_std_max)`, `assert_opponent_self_play_compatible(opponent)`),
    so an opaque `"<aim_log_std_max>"` raises long before the construction. The
    substitute is TYPE distinctness: a float cap, a string mode, an int
    `pin_pitch` and a bool flag, so a call site that crossed any two slots
    produces a value of the wrong type rather than an equal-looking one.

    `opponent="noop"` in the no-self-play case is deliberate — it is the one
    combination the guard allows, and it is the only way to make `opponent_mode`
    NON-default at this site, so a builder that dropped it (falling back to
    SelfPlayManager's own "self") is caught here rather than nowhere.
    """
    from cs2rl.train import selfplay as train_selfplay
    from tests._helpers import trainer_harness

    site = "harness_selfplay" if with_selfplay else "harness_no_selfplay"
    capture = FIXTURE_DATA["sites"][site][0]
    assert capture["scenario"] == "pin_pitch_truthy"

    # Distinctive but valid: build_train_config floats it, and "noop" cannot be
    # combined with self-play (assert_opponent_self_play_compatible).
    aim_cap = -1.2345
    opponent = "self" if with_selfplay else "noop"

    recorded = []
    real_builder = train_selfplay.build_selfplay_manager

    def _spy(**kwargs):
        recorded.append(kwargs)
        return real_builder(**kwargs)

    monkeypatch.setattr(trainer_harness, "build_selfplay_manager", _spy)
    _, cleanup = trainer_harness._build_trainer_for_test(num_envs=2,
                                                         with_selfplay=with_selfplay,
                                                         opponent=opponent,
                                                         aim_log_std_max=aim_cap,
                                                         pin_pitch=capture["bindings"]["pin_pitch"])
    cleanup()

    assert len(recorded) == 1
    got = recorded[0]
    assert got["self_play_enabled"] is with_selfplay, (
        f"the harness passed self_play_enabled={got['self_play_enabled']!r} for "
        f"with_selfplay={with_selfplay!r}; the two branches it replaced differed only in the "
        f"p_past those flags select ({capture['call_source']})")
    assert got["aim_log_std_max"] == aim_cap, (
        f"aim_log_std_max landed as {got['aim_log_std_max']!r} — routed into the wrong slot")
    assert got["pin_pitch"] == capture["bindings"]["pin_pitch"]
    assert got["opponent_mode"] == opponent, (
        f"opponent_mode landed as {got['opponent_mode']!r}, not the harness's {opponent!r}")
    assert sorted(got) == ["aim_log_std_max", "opponent_mode", "pin_pitch", "self_play_enabled"
                           ], f"the harness call site's kwarg set is {sorted(got)}"


def test_harness_given_manager_skips_the_builder(monkeypatch):
    """With ``self_play_mgr=`` given, the harness calls ``build_selfplay_manager`` 0 times.

    gh#168 W1.5 added the parameter so a test can seed a manager BEFORE construction the
    way a resume does (tests/train/test_resume_state.py, tests/train/test_pitch_pin.py). The
    companion case above pins the 1-call path; without this one, a harness that built a
    second manager and silently discarded the given one would keep every test green
    (the trainer would carry the builder's manager, not the caller's). Same spy shape as
    the companion, for the same reason: a stub that raised would leave the trainer
    unclosed. The manager passed must agree with the harness knobs (the harness asserts
    that), so it is built with the values the harness would pass: ``opponent_mode="self"``
    and ``pin_pitch=bool(_ENV_DEFAULTS.pin_pitch)`` (spelled, not left to
    SelfPlayManager's own default coinciding with the env default), with ``p_past=0.0``
    (no self-play).
    """
    from cs2rl.train import selfplay as train_selfplay
    from cs2rl.train.selfplay import SelfPlayManager
    from tests._helpers import trainer_harness

    recorded = []
    real_builder = train_selfplay.build_selfplay_manager

    def _spy(**kwargs):
        recorded.append(kwargs)
        return real_builder(**kwargs)

    monkeypatch.setattr(trainer_harness, "build_selfplay_manager", _spy)
    mgr = SelfPlayManager(p_past=0.0,
                          opponent_mode="self",
                          pin_pitch=bool(trainer_harness._ENV_DEFAULTS.pin_pitch))
    _, cleanup = trainer_harness._build_trainer_for_test(num_envs=2, self_play_mgr=mgr)
    try:
        assert recorded == [], (
            f"the harness called build_selfplay_manager {len(recorded)}x although a manager "
            "was given; the caller's pre-seeded manager would be discarded")
    finally:
        cleanup()


def test_harness_no_longer_branches_on_with_selfplay_to_construct():
    """The two harness constructions really did collapse to one call site.

    Without this, the migration could have left the `if not with_selfplay:` in
    place with a `build_selfplay_manager(...)` in each arm — every other test in
    this file would stay green, and the next knob would go into one arm only,
    which is the drift W3 exists to end.
    """
    tree = ast.parse((PACKAGE.parents[1] / "tests" / "_helpers" / "trainer_harness.py").read_text())
    calls = [
        n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "build_selfplay_manager"
    ]
    assert len(calls) == 1, (
        f"trainer_harness.py has {len(calls)} build_selfplay_manager call sites; the two "
        "pre-migration branches were identical apart from p_past and must stay collapsed")
