"""Record the PRE-W3 `make_puffer_env` call shape at every construction site.

WHY THIS FILE EXISTS. Spec 2026-08-31 §2 W3 routes seven `make_puffer_env`
construction sites through one factory (`src/env_factory.py`). The verification
oracle for that migration is a per-role comparison of the kwargs
`make_puffer_env` ACTUALLY RECEIVES, before vs after. The "before" half cannot be
transcribed by hand from the factory once the factory exists — that compares the
factory to itself and asserts nothing — so the spec makes the capture a SEPARATE
COMMIT that lands BEFORE any migration edit. This script is that capture, and
`tests/fixtures/env_kwargs_pre_w3.json` is its frozen output.

WHY NOT `static_data_scalars()`. It is structurally blind to 8 of make_env's 39
kwargs (`auto_reset, buf, include_step_stats_in_info, map_data, recoil,
reward_symmetrize, seed, team_spirit` have no scalars key; seed/team_spirit are
not StaticData fields at all) — and those 8 are exactly what distinguishes the
roles from one another. It is kept elsewhere as a corroborating second
measurement, never as the role oracle.

HOW THE CAPTURE WORKS. Not by driving each code path: three of the seven sites
(`train()`'s eval env, and both `eval_legacy` sites) need a full training run or
a real checkpoint to reach, and a capture that only covers the cheap sites would
freeze the wrong four. Instead, for each site this script

  1. parses the source file with `ast` and locates the `make_puffer_env` Call
     node by ENCLOSING FUNCTION QUALNAME (not by line number, which every
     workstream in this branch invalidates);
  2. `ast.unparse`s that node back to source and records the text verbatim in
     the fixture, so the thing that was measured is auditable against
     `git show <commit>:src/train.py` without rerunning anything;
  3. `eval`s that exact string in a namespace where `make_puffer_env` is a
     recording stub and every free name is bound to a scenario value;
  4. records `inspect.signature(make_puffer_env).bind(...)`'s arguments both
     WITHOUT defaults (`explicit_kwargs` — the call's SHAPE, which is what tells
     the roles apart) and WITH them (`effective_kwargs` — the env that actually
     results).

Step 3 is the reason the scenario values are JSON-safe sentinel STRINGS
(`"<map_data>"`, `"<shared_ts>"`, `"<buf>"`) rather than real objects: the stub
replaces `make_puffer_env`, so nothing validates them, and using a DISTINCT
sentinel per pass-through argument turns "the factory routed map_data into the
team_spirit slot" — an identity bug that equal-looking real objects would hide —
into a plain value mismatch.

MULTIPLE SCENARIOS PER SITE where the call site contains a conditional: the
train closure's `_seed if _seed is not None else (seed or 0)` has three branches
and the harness closure's `0 if seed is None else seed` has two. One scenario
each would freeze one branch and let the migration silently rewrite the others.

REGENERATING is legitimate when a construction site changes ON PURPOSE, and at
no other time. Regenerating it to make a failing migration go green deletes the
only evidence that pre- and post-migration construction agree, which is the
entire reason the fixture is committed.

    UV_NO_SYNC=1 uv run python tests/capture_env_kwargs_pre_w3.py --capture

This module is deliberately NOT named `test_*`: pytest must not collect it. Its
assertions live in `tests/test_env_factory.py`, which reads the fixture.
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

FIXTURE = Path(__file__).parent / "fixtures" / "env_kwargs_pre_w3.json"

# Bumped if the fixture's own shape changes, so a stale file fails on the tag
# rather than on a confusing KeyError deep inside a comparison.
CAPTURE_FORMAT = "cs2rl-env-kwargs-capture-v1"

# Sentinels for the opaque objects the call sites pass straight through. Each is
# distinct on purpose — see the module docstring.
S_SHARED_TS = "<shared_ts>"
S_TEAM_SPIRIT = "<team_spirit>"
S_MAP_DATA = "<map_data>"
S_BUF = "<buf>"


class _Recorder:
    """Stands in for `make_puffer_env` and records what it was handed.

    Binds against the REAL signature, so a call that would have been a TypeError
    against `make_puffer_env` is a TypeError here too — the stub cannot record a
    call shape that the real function would have rejected.
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
        return "<env>"


def _qualified_calls(path):
    """(enclosing qualname, Call node) for every bare `make_puffer_env(...)` call.

    Qualname, not line number: W1 already moved ~56 symbols out of train.py and
    W3/W4/W5 move more, so any line-anchored lookup in this branch is stale
    before it is read. `build_env_factory.env_factory` stays true across all of
    it.

    Only `Call(func=Name)` — `train_test_harness` does `from train import
    make_puffer_env` function-locally and every other site is a module global,
    so no construction is spelled as an attribute today. An attribute-spelled
    site would be MISSED rather than mis-recorded, and `_capture` asserts the
    exact expected site count to catch that.
    """
    tree = ast.parse(path.read_text())
    found = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(child, prefix + [child.name])
                continue
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                    and child.func.id == "make_puffer_env"):
                found.append((".".join(prefix), child))
            walk(child, prefix)

    walk(tree, [])
    return found


def _site(path, qualname):
    """The one `make_puffer_env` call inside `qualname`, as (relpath:lineno, source)."""
    matches = [(q, n) for q, n in _qualified_calls(path) if q == qualname]
    assert len(matches) == 1, (
        f"expected exactly one make_puffer_env call in {path.name}:{qualname}, "
        f"found {len(matches)} — the site moved or split; re-read the source before "
        "touching this script")
    node = matches[0][1]
    return f"{path.relative_to(REPO_ROOT)}:{node.lineno}", ast.unparse(node)


def _jsonable(value):
    """JSON view of a scenario binding.

    Scenario values are chosen to be JSON-safe already; argparse Namespaces are
    the one exception and are recorded as their attribute dict so the fixture
    states which flags produced the derived reward/knob dicts.
    """
    if isinstance(value, argparse.Namespace):
        return {"__namespace__": {k: v for k, v in sorted(vars(value).items())}}
    if callable(value):
        return f"<{value.__module__}.{value.__qualname__}>"
    return value


def _record(recorder, path, qualname, scenario, bindings, extra=None):
    """Eval one site's call source under `bindings` and return a fixture entry."""
    site, source = _site(path, qualname)
    ns = dict(bindings)
    ns["make_puffer_env"] = recorder
    before = len(recorder.calls)
    # eval of a string is the point: `source` is this repo's own file, unparsed
    # from its AST a few lines above, and evaluating THAT is what makes the
    # capture a reading of the source rather than a transcription of it.
    eval(compile(source, f"<{site}>", "eval"), ns)
    assert len(recorder.calls) == before + 1, f"{site} did not call make_puffer_env once"
    explicit, effective = recorder.calls[-1]
    entry = {
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
    if extra:
        entry.update(extra)
    return entry


def _capture():
    import train
    from train_config import env_knobs_from_args
    from train_shared import reward_overrides_from_args

    train_py = REPO_ROOT / "src" / "train.py"
    harness_py = REPO_ROOT / "src" / "train_test_harness.py"
    rec = _Recorder(inspect.signature(train.make_puffer_env))

    # Anti-omission guard: the census in spec §2 W3 says six sites in train.py
    # and one in the harness. If a site were spelled as an attribute call, or a
    # new one appeared, the per-role list below would silently cover less than
    # the file does.
    assert len(_qualified_calls(train_py)) == 6, _qualified_calls(train_py)
    assert len(_qualified_calls(harness_py)) == 1, _qualified_calls(harness_py)

    roles = {}

    # ── train ───────────────────────────────────────────────────────────────
    # Non-default reward_overrides / reward_symmetrize / env_knobs on purpose:
    # the §3 determinism gate sets no --reward-* and no R0-G flag, so a factory
    # that DROPPED any of them still produces a byte-identical checkpoint. That
    # is the historical bug this role's oracle exists to catch (build_env_factory's
    # own docstring records reward keys being silently dropped), and defaults
    # here would reproduce it.
    knobs = {
        "n_active_per_team": 3,
        "pin_pitch": 1,
        "crouch_enabled": 0,
        "jump_enabled": 1,
        "pbrs_gamma": 0.995,
        "round_time": 900,
    }
    overrides = {"reward_ct_survival": 0.0, "reward_t_kill": 2.5}
    roles["train"] = [
                                                                                         # R0-D: env_kwargs["_seed"] wins over the seed pufferlib passes.
        _record(
            rec, train_py, "build_env_factory.env_factory", "per_env_seed_wins", {
                "shared_ts": S_SHARED_TS,
                "buf": S_BUF,
                "seed": 3,
                "_seed": 41,
                "map_data": S_MAP_DATA,
                "reward_overrides": overrides,
                "reward_symmetrize": True,
                "env_knobs": knobs,
            }),
                                                                                         # Legacy path: no _seed, so pufferlib's own seed is used verbatim. Also
                                                                                         # pins that env_knobs=None splats nothing rather than passing None.
        _record(
            rec, train_py, "build_env_factory.env_factory", "pufferlib_seed_no_knobs", {
                "shared_ts": S_SHARED_TS,
                "buf": S_BUF,
                "seed": 5,
                "_seed": None,
                "map_data": S_MAP_DATA,
                "reward_overrides": None,
                "reward_symmetrize": False,
                "env_knobs": None,
            }),
                                                                                         # `seed or 0` — pufferlib passes seed=None for some backends.
        _record(
            rec, train_py, "build_env_factory.env_factory", "seed_none_becomes_zero", {
                "shared_ts": S_SHARED_TS,
                "buf": S_BUF,
                "seed": None,
                "_seed": None,
                "map_data": S_MAP_DATA,
                "reward_overrides": overrides,
                "reward_symmetrize": False,
                "env_knobs": knobs,
            }),
    ]

    # ── harness ─────────────────────────────────────────────────────────────
    # crouch_enabled / jump_enabled are passed non-default here because NO test
    # passes either to _build_trainer_for_test and their defaults equal
    # make_puffer_env's — so dropping them in the migration is invisible to the
    # whole suite. Only this capture can see it.
    roles["harness"] = [
        _record(
            rec, harness_py, "_build_trainer_for_test.env_factory", "seed_none_becomes_zero", {
                "shared_ts": S_SHARED_TS,
                "buf": S_BUF,
                "seed": None,
                "map_data": S_MAP_DATA,
                "n_active_per_team": 5,
                "pin_pitch": 0,
                "crouch_enabled": 1,
                "jump_enabled": 1,
            }),
        _record(
            rec, harness_py, "_build_trainer_for_test.env_factory", "explicit_seed_zero_kept", {
                "shared_ts": S_SHARED_TS,
                "buf": S_BUF,
                "seed": 0,
                "map_data": S_MAP_DATA,
                "n_active_per_team": 2,
                "pin_pitch": 1,
                "crouch_enabled": 0,
                "jump_enabled": 0,
            }),
    ]

    # ── eval ────────────────────────────────────────────────────────────────
    # auto_reset=False is load-bearing and unreachable by every run-level gate
    # on this branch: eval_baselines raises without it, and --eval-interval
    # defaults to 0 so this env is never built during the §3 run or the suite's
    # train runs. This capture is the only thing that sees it.
    # The two helper functions are bound to the REAL implementations, so the
    # recorded overrides/knobs are the values a run would actually derive.
    eval_args = argparse.Namespace(reward_ct_survival=0.0,
                                   n_active_per_team=3,
                                   pin_pitch=1,
                                   crouch_enabled=0,
                                   jump_enabled=1,
                                   round_time_ticks=900,
                                   gamma=0.995)
    roles["eval"] = []
    for name, a in (("default_args", argparse.Namespace()), ("non_default_args", eval_args)):
        roles["eval"].append(
            _record(
                rec,
                train_py,
                "train",
                name,
                {
                    "_map_data": S_MAP_DATA,
                    "args": a,
                    "reward_overrides_from_args": reward_overrides_from_args,
                    "env_knobs_from_args": env_knobs_from_args,
                },
                extra={
                                                                              # The derived values spelled out, so the post-migration
                                                                              # test can feed the factory the same inputs without
                                                                              # re-deriving them from the oracle it is checking.
                    "resolved": {
                        "reward_overrides": reward_overrides_from_args(a),
                        "env_knobs": env_knobs_from_args(a),
                    }
                }))

    # ── eval_legacy (TWO sites, differing only in seed) ─────────────────────
    # A bare call and a seed= call. Scalars cannot see the difference — seed is
    # not a StaticData field — so each site's shape is frozen separately and the
    # factory has to thread evaluate_checkpoint's seed through.
    roles["eval_legacy"] = [
        _record(rec, train_py, "load_policy_from_checkpoint", "bare_defaults", {}),
        _record(rec, train_py, "evaluate_checkpoint", "explicit_seed", {"seed": 12345}),
    ]

    # ── smoke ───────────────────────────────────────────────────────────────
    roles["smoke"] = [_record(rec, train_py, "smoke_test", "fixed_seed_42", {})]

    # ── external (`make_env`, the public wrapper; one test caller) ───────────
    roles["external"] = [
        _record(rec, train_py, "make_env", "both_passed", {
            "team_spirit": S_TEAM_SPIRIT,
            "map_data": S_MAP_DATA,
        }),
        _record(rec, train_py, "make_env", "wrapper_defaults", {
            "team_spirit": None,
            "map_data": None,
        }),
    ]

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
            "why": ("pre-W3 baseline for the make_puffer_env construction factory "
                    "(spec 2026-08-31 §2 W3); a snapshot taken after the migration would "
                    "compare the factory to itself"),
            "how": ("each site's call expression is located by enclosing-function qualname, "
                    "unparsed from the AST (recorded verbatim as call_source) and evaluated "
                    "against a recording stub bound to make_puffer_env's real signature"),
            "sentinels": ("opaque pass-through arguments are distinct sentinel strings so a "
                          "routed-into-the-wrong-slot bug shows up as a value mismatch"),
            "regenerate": ("UV_NO_SYNC=1 uv run python tests/capture_env_kwargs_pre_w3.py "
                           "--capture"),
        },
        "roles": roles,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE.open("w") as fh:
        json.dump(fixture, fh, indent=1, sort_keys=False)
        fh.write("\n")
    n = sum(len(v) for v in roles.values())
    print(f"wrote {FIXTURE.relative_to(REPO_ROOT)}: {len(roles)} roles, {n} captures, at {head}")


if __name__ == "__main__":
    if sys.argv[1:] != ["--capture"]:
        raise SystemExit(f"usage: {sys.argv[0]} --capture")
    _capture()
