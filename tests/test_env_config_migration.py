"""tests/test_env_config_migration.py — the legacy-call census (#165, spec §3).

WHAT: an AST census of every `make_env(...)`, `make_puffer_env(...)` and
`Cs2Env(...)` call under `src/`, `scripts/` and `tests/`, split TYPED vs LEGACY
by the three clauses of spec 2026-09-03 §3:

  clause 1  a keyword whose name is in neither the function's RUNTIME set nor
            `{config}` — i.e. a pre-#165 weight or knob name;
  clause 2  any `**splat` keyword. In the AST a splat is an `ast.keyword` with
            `arg=None`: it carries no name, so clause 1 is structurally blind
            to it, and a typed call never needs one. Not hypothetical —
            `scripts/sim_fingerprint.py` passed `n_active_per_team` through
            exactly this channel until PR B3;
  clause 3  any POSITIONAL argument. The first positional slot changes type on
            all three constructors (`seed` / `team_spirit` -> `config`), and
            the house style passes `config` by keyword, so a typed call has
            none.

WHY IT EXISTS: `make_env` and `make_puffer_env` keep a `**legacy` channel so
#165 did not have to migrate ~45 files in the branch that moved the seam. This
census is the ratchet that stops the channel growing. gh#173 drives the `tests`
literals to zero and then deletes the channel.

WHAT A GREEN HERE DOES *NOT* MEAN. `src`=0 and `scripts`=0 are RATCHETS: they
are already zero, so on real source they cannot fail today. What makes them
worth having is the planted-tree controls below, which run the SAME predicate
against a scratch tree where each clause does fire. A census that finds zero is
also what a broken census finds; read `test_each_clause_is_found_in_a_planted_tree`
and `test_the_production_pin_actually_fails_on_a_planted_site` before quoting a
green run as evidence of anything.

THE LIMIT, stated so nobody reads past it: resolution is STATIC. A call through
`getattr`, `exec`, or a function PARAMETER is invisible — `env_factory`'s role
builders take the constructor as the parameter `_make` and call `_make(...)`,
which this census does not see and never will. That channel is covered instead
by the per-role typed oracle in `tests/test_env_factory.py` (spec R8), whose
key-set assertion pins exactly what reaches `make_env` there.
"""
import ast
import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# I001 suppressed, not fixed: these imports must follow the sys.path insert
# above (same waiver as tests/test_no_restated_env_defaults.py:63).
import train                                           # noqa: E402, I001
from c_env.cs2_env import Cs2Env, make_env             # noqa: E402

# Explicit roots, never a walk from the repo root — a walk would visit
# `.worktrees/`, which holds checkouts of this same repo with their own copy of
# every call site, and report findings that belong to another branch.
ROOTS = ("src", "scripts", "tests")

# symbol -> the module that DEFINES it. This mapping is what makes
# `train.make_env(...)` not a census subject. `src/train.py:1191` defines its OWN
# `make_env(team_spirit=None, map_data=None)`, a delegate to
# `build_env_for("external")` — a DIFFERENT function that neither re-exports nor
# calls `c_env.cs2_env.make_env`. `make_env` here means the one `c_env.cs2_env`
# defines, so an attribute call on the `train` module does not resolve.
# test_the_public_train_wrapper_is_not_a_census_subject pins that, with a
# POSITIONAL call, which would be clause 3 the moment it resolved at all.
DEFINING_MODULE = {
    "make_env": "c_env.cs2_env",
    "Cs2Env": "c_env.cs2_env",
    "make_puffer_env": "train",
}
_MODULES = frozenset(DEFINING_MODULE.values())

# symbol -> the repo-relative FILE its defining module lives in. The in-module
# clause in `_bindings` is scoped by this and not by the name alone, because
# `src/train.py` defines a `make_env` that is NOT this one: keyed on the name, a
# bare `make_env(...)` inside train.py would be attributed to `c_env.cs2_env` and
# a positional call to train's own wrapper would be reported as clause 3.
# test_an_in_module_def_binds_only_inside_its_own_defining_file plants both
# halves.
DEFINING_FILE = {
    sym: "src/" + module.replace(".", "/") + ".py"
    for sym, module in DEFINING_MODULE.items()
}

# Parent spec §2.2. Asserted against the live signatures below in BOTH
# spellings, so a runtime parameter added to `make_env` but not to `Cs2Env`
# fails here rather than quietly reclassifying a typed call as legacy.
RUNTIME_LITERALS = {"make_env": 6, "make_puffer_env": 8, "Cs2Env": 7}


def _runtime_params(fn, drop_self=False):
    """Every named parameter except `config`; VAR_KEYWORD excluded.

    `drop_self` is spelled out rather than relying on `inspect.signature(Cs2Env)`
    to hide it: `Cs2Env` subclasses `pufferlib.PufferEnv`, and a base class that
    grew a `__signature__` or a custom `__new__` would silently change what the
    shorter spelling returns. Reading `__init__` and naming `self` cannot drift.
    """
    params = inspect.signature(fn).parameters
    return frozenset(
        name for name, p in params.items()
        if name != "config" and p.kind is not p.VAR_KEYWORD and not (drop_self and name == "self"))


RUNTIME = {
    "make_env": _runtime_params(make_env),
    "make_puffer_env": _runtime_params(train.make_puffer_env),
    "Cs2Env": _runtime_params(Cs2Env.__init__, drop_self=True),
}

# Measured at the end of PR B3. gh#173 LOWERS these as it migrates; a rise
# fails, printing file:line per offender. CALL-site counts, not file counts:
# the 54 live in 17 files, the 8 in 4.
PINNED_TESTS_MAKE_ENV = 54
PINNED_TESTS_MAKE_PUFFER_ENV = 8

# The roots the zero-pins run over, DERIVED from ROOTS rather than restated, so a
# rename or reorder there cannot leave a pin scanning a root that does not exist
# (`legacy_in` filters on a string; `legacy_in("scriptz", ...)` is silently []).
# The literal is asserted in test_runtime_parameter_sets_match_the_spec_literals.
PRODUCTION_ROOTS = ROOTS[:2]

# symbol -> the file whose legacy calls exist BY DESIGN and therefore supply the
# floor under that symbol's count. Named per symbol rather than shared, because
# the two floors are different files: `tests/test_make_env_shim.py` holds ZERO
# legacy `make_puffer_env` calls (measured), so it cannot underwrite that half.
# Both named files hold `pytest.raises` calls that exercise the legacy channel's
# own error paths — gh#173 cannot migrate them to `config=` without deleting the
# behaviour under test, which is exactly the property a floor needs.
FLOOR_FILES = {
    "make_env": "tests/test_make_env_shim.py",
    "make_puffer_env": "tests/test_env_config.py",
}


def _bindings(tree, rel):
    """(names, modules) for one parsed module. `rel` is its repo-relative path.

    names    local name -> canonical symbol, for the spellings spec §4.1 allows:
             `from c_env.cs2_env import make_env|Cs2Env [as X]`,
             `from train import make_puffer_env [as X]`, and a `ClassDef` /
             `FunctionDef` of that name IN ITS OWN DEFINING FILE.
    modules  local name -> dotted module, for `import train`,
             `import c_env.cs2_env as m`, and `from c_env import cs2_env [as X]`.

    THE IN-MODULE CLAUSE IS NOT BOOKKEEPING. `make_env`'s own
    `return Cs2Env(config=config, ...)` binds `Cs2Env` by `ClassDef` in the very
    same file, and parent §3 lists that site as a census subject. Without the
    clause the `Cs2Env` `src`=0 pin has no site it could ever fail on, and a
    revert of `make_env` to legacy forwarding would leave it green.

    IT IS SCOPED BY `DEFINING_FILE`, NOT BY THE NAME. `src/train.py:1191` defines
    a `make_env` of its own — a delegate to `build_env_for("external")`, not this
    census's subject. Keyed on the name, every bare `make_env(...)` inside
    train.py would be attributed to `c_env.cs2_env.make_env`, and a positional
    call to train's own wrapper would be reported as clause 3: a FALSE legacy
    site in the `src`=0 ratchet, contradicting this module's own rule about
    `train.make_env`. The `make_puffer_env` half of the same clause is unaffected
    — train.py IS its defining file — and is what catches an in-module
    `make_puffer_env(...)` call there.

    LIMITS, all four measured absent from the tree today, all four silent if they
    appear:
      * a bare `import c_env.cs2_env` (dotted, no `as`) binds the PACKAGE name
        `c_env`; `c_env.cs2_env.make_env(...)` is an Attribute over an Attribute,
        not over a Name, so `_target` does not reach it. Binding `c_env` to the
        submodule would make an unrelated `c_env.make_env(...)` a false positive.
      * a RELATIVE import (`from .cs2_env import make_env`): `node.module` is
        `cs2_env`, which is not the dotted name in DEFINING_MODULE.
      * an assignment alias (`mk = make_env`), which needs dataflow, not binding.
      * a SUBCLASS (`class Sub(Cs2Env)`), whose constructor call names `Sub`.
    `tests/test_env_construction_enforcement.py`'s docstring discloses the last
    two for its own scan; they are repeated here because this census has them
    too. If any appears, extend `_target`/`_bindings` rather than loosening a pin.
    """
    names, modules = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if DEFINING_MODULE.get(alias.name) == node.module:
                    names[alias.asname or alias.name] = alias.name
                elif node.module and f"{node.module}.{alias.name}" in _MODULES:
                    # `from c_env import cs2_env [as X]` — house style here
                    # (five sites at 6b3bf29). The resulting `cs2_env.make_env(...)`
                    # is an Attribute over a Name, which `_target` already handles.
                    modules[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _MODULES:
                    continue
                if alias.asname:
                    modules[alias.asname] = alias.name
                elif "." not in alias.name:
                    modules[alias.name] = alias.name
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if DEFINING_FILE.get(node.name) == rel:
                names[node.name] = node.name
    return names, modules


def _target(call, names, modules):
    """The canonical symbol this Call constructs, or None."""
    func = call.func
    if isinstance(func, ast.Name):
        return names.get(func.id)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        module = modules.get(func.value.id)
        if module is not None and DEFINING_MODULE.get(func.attr) == module:
            return func.attr
    return None


def _legacy_reasons(call, target):
    """[] for a typed call; one string per clause that fires otherwise."""
    reasons = []
    if call.args:
        reasons.append(f"clause 3: {len(call.args)} positional arg(s)")
    for kw in call.keywords:
        if kw.arg is None:
            reasons.append("clause 2: **splat")
        elif kw.arg != "config" and kw.arg not in RUNTIME[target]:
            reasons.append(f"clause 1: legacy keyword {kw.arg!r}")
    return reasons


def census(roots=ROOTS, repo_root=REPO_ROOT):
    """(scanned paths, [(symbol, relpath, lineno, reasons)]).

    The file list comes back too, because "which files did you actually read" is
    the one question a vacuous green cannot answer: `Path.rglob` returns [] in
    silence for a root that does not exist.

    Parse errors are NOT caught. A file the census cannot parse is a file it
    cannot see, and swallowing that turns a hole into a pass.
    """
    repo_root = Path(repo_root)
    scanned, found = [], []
    for root in roots:
        for path in sorted((repo_root / root).rglob("*.py")):
            scanned.append(path)
            rel = path.relative_to(repo_root).as_posix()
            tree = ast.parse(path.read_text(), filename=str(path))
            names, modules = _bindings(tree, rel)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                target = _target(node, names, modules)
                if target is None:
                    continue
                found.append((target, rel, node.lineno, _legacy_reasons(node, target)))
    return scanned, found


SCANNED, FOUND = census()


def legacy_in(root, symbol, found=None):
    """[(relpath, lineno, reasons)] — the legacy sites for one root and symbol.

    `found` lets a planted-tree control re-use the exact predicate the pins use,
    which is the only reason those pins' zeros mean anything.
    """
    return [(rel, lineno, why) for sym, rel, lineno, why in (FOUND if found is None else found)
            if sym == symbol and why and rel.split("/", 1)[0] == root]


def _report(sites):
    return "\n".join(f"  {rel}:{lineno}  {'; '.join(why)}" for rel, lineno, why in sites)


# ── guard: the scan read a real, populated tree ─────────────────────────────

ANCHORS = ("src/c_env/cs2_env.py", "scripts/sim_fingerprint.py", FLOOR_FILES["make_env"],
           FLOOR_FILES["make_puffer_env"])


def test_the_scan_roots_resolve_to_real_populated_directories():
    """`rglob` on a missing or renamed root returns [] in silence, so every pin
    below would pass on an empty file list. This is the assertion that makes a
    zero mean something: each root contributed files, and three named files —
    one per root, each of which really does construct an env — are among them.
    """
    for root in ROOTS:
        paths = [p for p in SCANNED if p.relative_to(REPO_ROOT).as_posix().split("/", 1)[0] == root]
        assert paths, (
            f"root {root!r} contributed no .py files. rglob returns [] for a root that does "
            f"not exist, so this is what a rename or a typo in ROOTS looks like — repoint "
            f"ROOTS, never delete this assert.")
    scanned_rel = {p.relative_to(REPO_ROOT).as_posix() for p in SCANNED}
    missing = [a for a in ANCHORS if a not in scanned_rel]
    assert not missing, f"anchors missing from the scan: {missing} ({len(SCANNED)} files read)"


def test_runtime_parameter_sets_match_the_spec_literals():
    """Parent §2.2's 6 / 8 / 7. Both spellings on purpose: a runtime parameter
    added to `make_env` but not to `Cs2Env` (or the reverse) silently
    reclassifies a typed call as legacy at every site that passes it, and the
    census would then report a migration regression that is really a signature
    drift.
    """
    measured = {name: len(params) for name, params in RUNTIME.items()}
    assert measured == RUNTIME_LITERALS, (
        f"runtime sets moved: {measured} != {RUNTIME_LITERALS}\n" +
        "\n".join(f"  {k}: {sorted(v)}" for k, v in RUNTIME.items()))
    assert "config" not in RUNTIME["make_env"]
    assert "self" not in RUNTIME["Cs2Env"]
    assert PRODUCTION_ROOTS == ("src", "scripts"), (
        f"PRODUCTION_ROOTS is {PRODUCTION_ROOTS}: it is ROOTS[:2], so reordering ROOTS silently "
        f"repoints the zero-pins. Fix ROOTS or this literal, never the slice.")
    assert set(DEFINING_FILE) == set(DEFINING_MODULE)


# ── the pins ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("symbol", sorted(DEFINING_MODULE))
@pytest.mark.parametrize("root", PRODUCTION_ROOTS)
def test_production_roots_hold_no_legacy_construction(root, symbol):
    """A RATCHET, not a driver (parent §3). `src` and `scripts` reached zero in
    #165 PR B3; this fails when someone reintroduces a legacy site. It cannot
    fail on today's tree, which is exactly why
    test_the_production_pin_actually_fails_on_a_planted_site exists below.
    """
    sites = legacy_in(root, symbol)
    assert not sites, f"legacy {symbol} call(s) reintroduced under {root}/:\n{_report(sites)}"


def test_cs2env_has_no_legacy_callers_in_tests():
    """`Cs2Env` took no legacy channel at all (parent §3): it is the L1
    constructor, `config` has no default and there is no `**legacy`. Its 18 test
    call sites migrated in Phase A. A legacy `Cs2Env(...)` is a TypeError at
    runtime, so this pin is a static early warning, not a duplicate of one.
    """
    sites = legacy_in("tests", "Cs2Env")
    assert not sites, f"legacy Cs2Env call(s) under tests/:\n{_report(sites)}"


@pytest.mark.parametrize("symbol,pin", [("make_env", PINNED_TESTS_MAKE_ENV),
                                        ("make_puffer_env", PINNED_TESTS_MAKE_PUFFER_ENV)])
def test_the_tests_root_legacy_count_only_ever_falls(symbol, pin):
    """gh#173 lowers these; nothing may raise them.

    The `> 0` half is not decoration. While the `**legacy` channel exists these
    counts CANNOT legitimately be zero — `tests/test_make_env_shim.py` calls it
    on purpose — so a zero here means the census stopped resolving, not that the
    migration finished. That is the failure this test is really watching for;
    the ceiling is the easy half.
    """
    sites = legacy_in("tests", symbol)
    assert len(sites) <= pin, (
        f"legacy {symbol} calls under tests/ rose to {len(sites)} (pinned {pin}). "
        f"gh#173 lowers this literal; nothing else may raise it. Sites:\n{_report(sites)}")
    assert sites, (
        f"ZERO legacy {symbol} calls found under tests/ while `**legacy` still exists on the "
        f"shim. {FLOOR_FILES[symbol]} calls it deliberately, so this is a broken census, not a "
        f"finished migration. If gh#173 really did delete the channel, delete this file "
        f"rather than editing the literal.")


@pytest.mark.parametrize("symbol,pin", [("make_env", PINNED_TESTS_MAKE_ENV),
                                        ("make_puffer_env", PINNED_TESTS_MAKE_PUFFER_ENV)])
def test_the_named_floor_file_holds_calls_gh173_cannot_migrate(symbol, pin):
    """Names WHICH sites are legitimate, so the floor above cannot be satisfied
    by an accidental legacy call elsewhere while the deliberate ones vanish.

    PER SYMBOL, because the two floors are different files and the obvious shared
    answer is wrong: `tests/test_make_env_shim.py` holds zero legacy
    `make_puffer_env` calls, so a shared floor would leave that half of
    test_the_tests_root_legacy_count_only_ever_falls telling a correct migration
    it is "a broken census".

    The second assertion is the one with teeth: it forbids gh#173 lowering the
    ceiling BELOW the floor, which would leave every later run red with no legal
    way to get green short of deleting the channel.
    """
    floor = [s for s in legacy_in("tests", symbol) if s[0] == FLOOR_FILES[symbol]]
    assert floor, (
        f"{FLOOR_FILES[symbol]} holds no legacy {symbol} call — it is named here because it "
        f"exists to hold them. Either the census stopped resolving that file, or the floor "
        f"moved and FLOOR_FILES must name where it moved to.")
    assert len(floor) <= pin, (
        f"the pinned ceiling for {symbol} ({pin}) is below its immovable floor ({len(floor)} "
        f"calls in {FLOOR_FILES[symbol]}): no migration can ever satisfy it.")


# ── positive controls (mandatory, parent §3) ────────────────────────────────

_PLANTS = {
                                                                                                     # name -> (path in the scratch tree, source). The PATH is part of the
                                                                                                     # control: `_bindings`'s in-module clause is scoped by `DEFINING_FILE`, so
                                                                                                     # the two in-module plants only resolve when planted at the file that really
                                                                                                     # defines the symbol.
    "clause1-name": ("src/p.py", "from c_env.cs2_env import make_env\nmake_env(reward_kill=2.0)\n"),
    "clause1-alias":
    ("src/p.py", "from c_env.cs2_env import make_env as mk\nmk(reward_kill=2.0)\n"),
    "clause1-attr": ("src/p.py", "import c_env.cs2_env as m\nm.make_env(reward_kill=2.0)\n"),
    "clause1-submodule":
    ("src/p.py", "from c_env import cs2_env\ncs2_env.make_env(reward_kill=2.0)\n"),
    "clause2-name": ("src/p.py", "from c_env.cs2_env import make_env\nmake_env(**kw)\n"),
    "clause2-attr": ("src/p.py", "import c_env.cs2_env as m\nm.make_env(**kw)\n"),
    "clause3-name": ("src/p.py", "from c_env.cs2_env import make_env\nmake_env(0)\n"),
    "clause3-attr": ("src/p.py", "import c_env.cs2_env as m\nm.make_env(0)\n"),
    "puffer-name":
    ("src/p.py", "from train import make_puffer_env\nmake_puffer_env(reward_kill=2.0)\n"),
    "puffer-alias": ("src/p.py", "from train import make_puffer_env as mpe\nmpe(**kw)\n"),
    "puffer-attr": ("src/p.py", "import train\ntrain.make_puffer_env(0)\n"),
    "puffer-inmodule":
    ("src/train.py",
     "def make_puffer_env(**kw):\n    pass\n\n\nmake_puffer_env(reward_kill=2.0)\n"),
    "classdef-inmodule": ("src/c_env/cs2_env.py",
                          "class Cs2Env:\n    pass\n\n\nCs2Env(n_active_per_team=1)\n"),
}

# Asserted as a literal below. Without this the set the docstring calls "THE
# control for every zero" is itself unwatched: deleting both clause2 entries
# removes the only control for the `**splat` clause and nothing goes red.
_PLANT_NAMES = frozenset({
    "clause1-name",
    "clause1-alias",
    "clause1-attr",
    "clause1-submodule",
    "clause2-name",
    "clause2-attr",
    "clause3-name",
    "clause3-attr",
    "puffer-name",
    "puffer-alias",
    "puffer-attr",
    "puffer-inmodule",
    "classdef-inmodule",
})


def _plant(tmp_path, relpath, source):
    """Write one file at `relpath` in a scratch tree and census THAT tree.

    The path is load-bearing, not decoration: `_bindings`'s in-module clause is
    scoped by `DEFINING_FILE`, so `class Cs2Env` resolves at
    `src/c_env/cs2_env.py` and nowhere else. The root censused is taken from
    `relpath`, so a `scripts/` plant exercises the `scripts/` filter path.
    """
    path = tmp_path / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return census(roots=(relpath.split("/", 1)[0], ), repo_root=tmp_path)


def test_the_plant_set_is_pinned_and_covers_every_clause_in_both_spellings():
    """`_PLANTS` is the control for every zero in this file, which makes it a
    guard set — and a guard set nothing watches is the failure class this repo
    keeps hitting. The literal catches a deletion; the loop states the invariant
    the literal has to satisfy, so an edit that keeps the count while dropping a
    clause fails too.
    """
    assert set(_PLANTS) == set(_PLANT_NAMES), (
        f"_PLANTS composition changed: added {sorted(set(_PLANTS) - _PLANT_NAMES)}, "
        f"removed {sorted(_PLANT_NAMES - set(_PLANTS))}. Update _PLANT_NAMES deliberately; "
        f"do not delete this assertion.")
    for clause in ("clause1", "clause2", "clause3"):
        for spelling in ("name", "attr"):
            assert f"{clause}-{spelling}" in _PLANTS, (
                f"no {spelling}-spelled control for {clause}. The splat clause in particular is "
                f"not hypothetical: scripts/sim_fingerprint.py used it until #165 PR B3.")


@pytest.mark.parametrize("name", sorted(_PLANTS))
def test_each_clause_is_found_in_a_planted_tree(tmp_path, name):
    """THE control for every zero above. One planted file per clause per
    spelling, each of which MUST come back legacy.

    `classdef-inmodule` and `puffer-inmodule` are the controls for `_bindings`'s
    in-module clause, each planted AT its own defining file: the first is the
    shape `make_env`'s own `return Cs2Env(...)` has in `src/c_env/cs2_env.py`,
    the second the shape a bare `make_puffer_env(...)` would have in
    `src/train.py`. Without that clause the `Cs2Env` `src`=0 pin has no site it
    could ever fail on.
    """
    _, found = _plant(tmp_path, *_PLANTS[name])
    legacy = [f for f in found if f[3]]
    assert legacy, f"planted {name} was NOT found: census returned {found}"


def test_an_in_module_def_binds_only_inside_its_own_defining_file(tmp_path):
    """The other half of the in-module clause, and the one that is easy to get
    wrong: `src/train.py:1191` defines a `make_env` that is NOT this census's
    subject. Planted at `src/train.py` the def must NOT bind; the identical
    source planted at `src/c_env/cs2_env.py` must. Bound by name alone, the
    first half resolves and reports clause 3 — a false legacy site in the
    `src`=0 ratchet, and a direct contradiction of this module's own rule about
    `train.make_env`.
    """
    wrapper = ("def make_env(team_spirit=None, map_data=None):\n"
               "    return build_env_for('external', team_spirit, map_data)\n"
               "\n\nmake_env(0, '<map>')\n")
    _, elsewhere = _plant(tmp_path / "a", "src/train.py", wrapper)
    assert elsewhere == [], (
        f"train.py's own make_env resolved as a census subject: {elsewhere}. The in-module "
        f"clause is keyed on the name again, not on DEFINING_FILE.")
    _, at_home = _plant(tmp_path / "b", "src/c_env/cs2_env.py", wrapper)
    assert at_home and at_home[0][3], (
        f"the SAME source at c_env/cs2_env.py did not resolve: {at_home}. The clause is now "
        f"scoped so tightly it cannot see the site it exists for.")


@pytest.mark.parametrize("spelling,source", [
    ("name", "from c_env.cs2_env import make_env\nfrom env_config import EnvConfig\n"
     "make_env(config=EnvConfig(), map_data=md)\n"),
    ("attr", "import c_env.cs2_env as m\nfrom env_config import EnvConfig\n"
     "m.make_env(config=EnvConfig(), map_data=md, seed=3)\n"),
    ("submodule", "from c_env import cs2_env\nfrom env_config import EnvConfig\n"
     "cs2_env.make_env(config=EnvConfig(), map_data=md)\n"),
])
def test_a_typed_call_is_resolved_but_not_flagged(tmp_path, spelling, source):
    """The other direction, and it needs both halves: the census must RESOLVE a
    typed call (so the target matcher is doing its job) and then report no
    clause. A matcher that resolved nothing would also "not flag" it.
    """
    _, found = _plant(tmp_path, f"src/typed_{spelling}.py", source)
    assert found, "typed call was not resolved at all — the matcher, not the clauses, is broken"
    assert all(not f[3] for f in found), f"typed call flagged legacy: {found}"


def test_the_public_train_wrapper_is_not_a_census_subject(tmp_path):
    """`train.make_env` is train.py's OWN function — a delegate to
    `build_env_for("external")`, not a re-export of and not a call to
    `c_env.cs2_env.make_env` — so it is NOT a census subject. The call is planted
    POSITIONALLY, which would be clause 3 the moment it resolved, so this fails
    loudly if the module check is ever dropped from `_target`.
    """
    _, found = _plant(tmp_path, "src/public_wrapper.py",
                      "import train\ntrain.make_env(0, '<map>')\n")
    assert found == [], f"train.make_env resolved as a census subject: {found}"


@pytest.mark.parametrize("root", PRODUCTION_ROOTS)
def test_the_production_pin_actually_fails_on_a_planted_site(tmp_path, root):
    """`src`=0 and `scripts`=0 cannot fail on real source, so they are proved
    here instead: the SAME predicate the pins use (`legacy_in`), over a scratch
    tree holding one legacy site, must report exactly that site. Parametrized
    over BOTH production roots, because `legacy_in`'s root filter is a string
    compare — proving it for `src` proves nothing about `scripts`, whose arm is
    otherwise a zero that never looked.
    """
    _, found = _plant(tmp_path, f"{root}/ratchet.py",
                      "from c_env.cs2_env import make_env\nmake_env(recoil=True)\n")
    sites = legacy_in(root, "make_env", found=found)
    assert len(sites) == 1, sites
    assert "clause 1" in sites[0][2][0], sites
