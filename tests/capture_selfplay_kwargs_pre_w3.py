"""Record the PRE-W3 `SelfPlayManager(...)` call shape at all three sites.

WHY A SECOND CAPTURE FILE. `tests/fixtures/env_kwargs_pre_w3.json` froze the six
`make_puffer_env` roles one commit before `src/env_factory.py` existed, and it is
IMMUTABLE — regenerating it on the migrated tree would compare the factory to
itself, which is the single failure the capture-before-migration rule exists to
prevent. The three `SelfPlayManager` sites were deliberately left standing in
their pre-migration shape by that task (see the "NOT HERE YET" note at the end of
`env_factory`'s module docstring) precisely so this capture would still be
possible. This file is that capture; `tests/fixtures/selfplay_kwargs_pre_w3.json`
is its frozen output, and it lands in its own commit BEFORE any migration edit.

WHY THE SITES NEED AN ORACLE AT ALL. The §3 determinism gate runs
`--no-self-play`, so it constructs a manager with `p_past=0.0` and an empty pool
that `should_use_past()` never activates: a migration that dropped
`aim_log_std_max`, flipped `pin_pitch` or defaulted `opponent_mode` writes a
BYTE-IDENTICAL `dust2_policy.pt`. The two harness sites are worse — no gate on
this branch reaches them at all. So the captured kwargs are the only oracle these
three sites have, and they have to be recorded before the builder is written.

WHAT `p_past` MAKES DISTINCT, and why one builder covers three sites. The three
call sites are already identical except for `p_past`, and even that is the same
expression twice: `train()` writes `0.3 if self_play_enabled else 0.0` inline,
while the harness spells the two branches out as separate `p_past=0.0` /
`p_past=0.3` constructions under `if not with_selfplay:`. Capturing all three
shapes separately is what lets the migration collapse them onto one builder
WITHOUT that collapse being an unchecked assertion: the fixture states the three
pre-migration shapes, and the post-migration factory has to reproduce each.

HOW THE CAPTURE WORKS — identical protocol to `capture_env_kwargs_pre_w3.py`:
locate the Call node by ENCLOSING FUNCTION QUALNAME (never by line number; every
workstream in this branch invalidates those), `ast.unparse` it back to source and
record that text verbatim, then `eval` that exact string in a namespace where
`SelfPlayManager` is a recording stub bound to the real signature and every free
name is a scenario binding.

WHY `pin_pitch` SCENARIOS USE 0/1 RATHER THAN A STRING SENTINEL. Both call sites
wrap it — `bool(args.pin_pitch)` / `bool(pin_pitch)` — so a string sentinel would
be recorded as `True` and the falsy branch would never be measured. `0` and `1`
scenarios pin both outcomes of that `bool()`, which is what tells a migration
that dropped the wrapper (or moved it) from one that kept it.
`aim_log_std_max` and `opponent_mode` ARE opaque pass-throughs and do get
distinct string sentinels, so a routed-into-the-wrong-slot bug is a value
mismatch rather than two equal-looking reals.

REGENERATING is legitimate when a construction site changes ON PURPOSE, and at no
other time — never to make a failing migration go green.

    UV_NO_SYNC=1 uv run python tests/capture_selfplay_kwargs_pre_w3.py --capture

Deliberately NOT named `test_*`: pytest must not collect it. Its assertions live
in `tests/test_selfplay_factory.py`, which reads the fixture.
"""
import argparse
import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

FIXTURE = Path(__file__).parent / "fixtures" / "selfplay_kwargs_pre_w3.json"

# Bumped if the fixture's own shape changes, so a stale file fails on the tag
# rather than on a confusing KeyError deep inside a comparison.
CAPTURE_FORMAT = "cs2rl-selfplay-kwargs-capture-v1"

# Opaque pass-throughs. Distinct on purpose — see the module docstring.
S_AIM = "<aim_log_std_max>"
S_OPPONENT = "<opponent_mode>"


class _Recorder:
    """Stands in for `SelfPlayManager` and records what it was handed.

    Binds against the REAL signature, so a call the class itself would have
    rejected is a TypeError here too — the stub cannot record a shape that would
    not have constructed.
    """

    def __init__(self, sig):
        self._sig = sig
        self.calls = []

    def __call__(self, *args, **kwargs):
        bound = self._sig.bind(*args, **kwargs)
        explicit = dict(bound.arguments)
        with_defaults = self._sig.bind(*args, **kwargs)
        with_defaults.apply_defaults()
        self.calls.append((explicit, dict(with_defaults.arguments)))
        return "<manager>"


def _qualified_calls(path, name):
    """(enclosing qualname, Call node) for every bare `name(...)` call.

    Qualname, not line number, for the same reason the env capture uses it: W1
    moved ~56 symbols out of train.py and later workstreams move more, so a
    line-anchored lookup in this branch is stale before it is read.

    Only `Call(func=Name)`. An ATTRIBUTE-spelled construction
    (`train.SelfPlayManager(...)`) would be missed rather than mis-recorded, so
    `_capture` separately asserts that no such site exists in src/ — the same
    matcher `tests/test_env_construction_enforcement.py` runs permanently.
    """
    tree = ast.parse(path.read_text())
    found = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(child, prefix + [child.name])
                continue
            if isinstance(child, ast.Call) and isinstance(child.func,
                                                          ast.Name) and child.func.id == name:
                found.append((".".join(prefix), child))
            walk(child, prefix)

    walk(tree, [])
    return found


def _attribute_calls(path, name):
    """Line numbers of `<anything>.name(...)` — the spelling `_qualified_calls` misses.

    NOT `<name>.<other>()`: that is CLASSMETHOD ACCESS (`SelfPlayManager.
    initial_hero_team()`), which is not a construction and must stay uncounted.
    Matching on the ATTRIBUTE position rather than the value position is exactly
    what keeps the two apart.
    """
    tree = ast.parse(path.read_text())
    return [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == name
    ]


def _site(path, qualname, index, expect_in_source):
    """The `index`-th `SelfPlayManager` call inside `qualname`, as (relpath:lineno, source).

    `index` exists only because `_build_trainer_for_test` holds TWO constructions
    (the `if not with_selfplay:` branches). `expect_in_source` is the anti-swap
    guard: an index that silently pointed at the other branch would capture the
    wrong `p_past` and freeze a shape no site ever had, so each index states the
    substring that identifies its branch.
    """
    matches = [(q, n) for q, n in _qualified_calls(path, "SelfPlayManager") if q == qualname]
    assert len(matches) > index, (
        f"{path.name}:{qualname} has {len(matches)} SelfPlayManager calls, no index {index} — "
        "the site moved or split; re-read the source before touching this script")
    node = matches[index][1]
    source = ast.unparse(node)
    assert expect_in_source in source, (
        f"{path.name}:{qualname}[{index}] does not contain {expect_in_source!r} — the two "
        f"branches swapped order. Got: {source}")
    return f"{path.relative_to(REPO_ROOT)}:{node.lineno}", source


def _jsonable(value):
    """JSON view of a scenario binding; Namespaces become their attribute dict."""
    if isinstance(value, argparse.Namespace):
        return {"__namespace__": {k: v for k, v in sorted(vars(value).items())}}
    return value


def _record(recorder, path, qualname, index, expect_in_source, scenario, bindings):
    """Eval one site's call source under `bindings` and return a fixture entry."""
    site, source = _site(path, qualname, index, expect_in_source)
    ns = dict(bindings)
    ns["SelfPlayManager"] = recorder
    before = len(recorder.calls)
    # eval of a string is the point: `source` is this repo's own file, unparsed
    # from its AST a few lines above, so the capture is a READING of the source
    # rather than a transcription of it.
    eval(compile(source, f"<{site}>", "eval"), ns)
    assert len(recorder.calls) == before + 1, f"{site} did not call SelfPlayManager once"
    explicit, effective = recorder.calls[-1]
    return {
        "scenario": scenario,
        "site": site,
        "enclosing": qualname,
        "call_source": source,
        "bindings": {
            k: _jsonable(v)
            for k, v in sorted(bindings.items())
        },
        "explicit_kwargs": {
            k: explicit[k]
            for k in sorted(explicit)
        },
        "effective_kwargs": {
            k: effective[k]
            for k in sorted(effective)
        },
    }


def _capture():
    import train

    train_py = REPO_ROOT / "src" / "train.py"
    harness_py = REPO_ROOT / "src" / "train_test_harness.py"
    rec = _Recorder(inspect.signature(train.SelfPlayManager))

    # Anti-omission guard: spec §2 W3's census is one site in train() and two in
    # the harness. A fourth site, or one spelled as an attribute call, would make
    # the per-site list below cover less than the files do.
    assert len(_qualified_calls(train_py, "SelfPlayManager")) == 1, _qualified_calls(
        train_py, "SelfPlayManager")
    assert len(_qualified_calls(harness_py, "SelfPlayManager")) == 2, _qualified_calls(
        harness_py, "SelfPlayManager")
    for p in sorted((REPO_ROOT / "src").rglob("*.py")):
        assert not _attribute_calls(p, "SelfPlayManager"), (
            f"{p} has an attribute-spelled SelfPlayManager construction at "
            f"{_attribute_calls(p, 'SelfPlayManager')} — this capture would miss it")

    sites = {}

    # ── train() ─────────────────────────────────────────────────────────────
    # Three scenarios because the call site carries two conditionals, not one:
    #   self_play_on / self_play_off   the `0.3 if self_play_enabled else 0.0`
    #                                  p_past branch AND, via pin_pitch 1/0, both
    #                                  outcomes of `bool(args.pin_pitch)`.
    #   aim_log_std_max_absent         `getattr(args, "aim_log_std_max", None)`
    #                                  — args WITHOUT the attribute. Nothing else
    #                                  measures that fallback, and a migration
    #                                  that passed `args.aim_log_std_max`
    #                                  directly would raise AttributeError only
    #                                  on the argument-less code paths.
    sites["train"] = [
        _record(
            rec, train_py, "train", 0, "p_past=0.3 if self_play_enabled else 0.0", "self_play_on", {
                "self_play_enabled": True,
                "args": argparse.Namespace(aim_log_std_max=S_AIM, pin_pitch=1),
                "_opponent_mode": S_OPPONENT,
            }),
        _record(
            rec, train_py, "train", 0, "p_past=0.3 if self_play_enabled else 0.0", "self_play_off",
            {
                "self_play_enabled": False,
                "args": argparse.Namespace(aim_log_std_max=S_AIM, pin_pitch=0),
                "_opponent_mode": "noop",
            }),
        _record(
            rec, train_py, "train", 0, "p_past=0.3 if self_play_enabled else 0.0",
            "aim_log_std_max_absent", {
                "self_play_enabled": True,
                "args": argparse.Namespace(pin_pitch=1),
                "_opponent_mode": S_OPPONENT,
            }),
    ]

    # ── _build_trainer_for_test, both branches ──────────────────────────────
    # Index 0 is `if not with_selfplay:` (p_past=0.0), index 1 the `else`
    # (p_past=0.3); `_site` refuses to record either unless the branch's own
    # p_past literal is in the unparsed source.
    for role, index, marker, p_past_note in (("harness_no_selfplay", 0, "p_past=0.0", "0.0"),
                                             ("harness_selfplay", 1, "p_past=0.3", "0.3")):
        sites[role] = [
            _record(rec, harness_py, "_build_trainer_for_test", index, marker, "pin_pitch_truthy", {
                "aim_log_std_max": S_AIM,
                "pin_pitch": 1,
                "opponent": S_OPPONENT,
            }),
            _record(rec, harness_py, "_build_trainer_for_test", index, marker, "pin_pitch_falsy", {
                "aim_log_std_max": None,
                "pin_pitch": 0,
                "opponent": "self",
            }),
        ]
        assert all(c["explicit_kwargs"]["p_past"] == float(p_past_note) for c in sites[role])

    head = subprocess.run(["git", "rev-parse", "HEAD"],
                          cwd=REPO_ROOT,
                          capture_output=True,
                          text=True,
                          check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "src/"],
                           cwd=REPO_ROOT,
                           capture_output=True,
                           text=True,
                           check=True).stdout.strip()
    assert not dirty, (f"src/ is dirty ({dirty!r}) — a pre-migration capture taken on a "
                       "modified tree records something that is not in any commit")

    fixture = {
        "_provenance": {
            "format":
            CAPTURE_FORMAT,
            "captured_at_commit":
            head,
            "why": ("pre-W3 baseline for build_selfplay_manager (spec 2026-08-31 §2 W3); a "
                    "snapshot taken after the migration would compare the factory to itself"),
            "how": ("each site's call expression is located by enclosing-function qualname "
                    "(plus a branch index guarded by its own p_past literal), unparsed from the "
                    "AST and evaluated against a recording stub bound to SelfPlayManager's real "
                    "signature"),
            "sentinels": ("aim_log_std_max / opponent_mode are distinct string sentinels; "
                          "pin_pitch is 0/1 instead, because both call sites wrap it in bool()"),
            "regenerate": ("UV_NO_SYNC=1 uv run python "
                           "tests/capture_selfplay_kwargs_pre_w3.py --capture"),
        },
        "sites": sites,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE.open("w") as fh:
        json.dump(fixture, fh, indent=1, sort_keys=False)
        fh.write("\n")
    n = sum(len(v) for v in sites.values())
    print(f"wrote {FIXTURE.relative_to(REPO_ROOT)}: {len(sites)} sites, {n} captures, at {head}")


if __name__ == "__main__":
    if sys.argv[1:] != ["--capture"]:
        raise SystemExit(f"usage: {sys.argv[0]} --capture")
    _capture()
