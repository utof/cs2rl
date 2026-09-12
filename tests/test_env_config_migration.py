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
#165 could move the seam without migrating every caller in the same branch. HOW
MANY callers that is has no single answer, and this census is the instrument
that answers it, so the number is stated here with the question attached.
Measured by this module, on this tree, at the tip of PR B3: **19 files under
`tests/` hold at least one call these three clauses call legacy** — 17 for
`make_env`, 4 for `make_puffer_env`, two files in both — and `src/` and
`scripts/` hold none. Two neighbouring counts answer DIFFERENT questions and
must not be quoted for this one: 32 files under `tests/` (41 across all three
roots) hold at least one call the census RESOLVES, typed or legacy; and
`src/c_env/cs2_env.py`'s own `**legacy` docstring carries a third figure for the
pre-migration population — 38, which names the right TREE but a DIFFERENT
INSTRUMENT, namely `git grep -lE 'make_env|make_puffer_env' 139a3a3 --
'tests/*.py'`: a count of files that MENTION either name in text at the pre-#165
baseline, not of calls any clause here calls legacy. Repaired elsewhere in this
PR (spec §8.9 item 4).
Anyone re-quoting a number from here states the tree, the commit and the
reading, because this workstream has already burned a day on three "different
answers" that were three different questions. This census is the ratchet that
stops the channel growing. gh#173 drives the `tests` literals to zero and then
deletes the channel.

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
PINNED_TESTS_MAKE_ENV = 44
PINNED_TESTS_MAKE_PUFFER_ENV = 5

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

    LIMITS, all five measured silent if they appear; no call site for any of them
    exists in the tree today, but the fifth already has a live ENABLER:
      * a bare `import c_env.cs2_env` (dotted, no `as`) binds the PACKAGE name
        `c_env`; `c_env.cs2_env.make_env(...)` is an Attribute over an Attribute,
        not over a Name, so `_target` does not reach it. Binding `c_env` to the
        submodule would make an unrelated `c_env.make_env(...)` a false positive.
      * a RELATIVE import (`from .cs2_env import make_env`): `node.module` is
        `cs2_env`, which is not the dotted name in DEFINING_MODULE.
      * an assignment alias (`mk = make_env`), which needs dataflow, not binding.
      * a SUBCLASS (`class Sub(Cs2Env)`), whose constructor call names `Sub`.
      * the symbol reached through an import ALIAS in any position but a bare
        local Name. Both arms of the matcher key on the CANONICAL name —
        `_target` compares `DEFINING_MODULE.get(func.attr)`, and the ImportFrom
        arm above compares `alias.name` — so all three of
        `train.make_c_env(...)`, `profile_step.make_c_env(...)` and
        `from profile_step import make_c_env` came back SILENT when planted,
        while the canonical `m.make_env(...)` resolved in the same harness, which
        is what proves the spelling and not the plant is at fault. This one has a
        LIVE ENABLER: `src/profile_step.py:36` is a MODULE-LEVEL
        `from c_env.cs2_env import make_env as make_c_env`, so
        `profile_step.make_c_env` is a real module attribute today and only the
        qualified CALL is missing. Nor is `make_c_env` an exotic spelling —
        `tests/test_env_construction_enforcement.py`'s LOWER_LAYER comment calls
        it house style "at the three files that ALIAS it" — a MINORITY spelling,
        not what every importer writes: that comment's own AST census reads 9
        importers to 3 aliasers over `src/` + `scripts/`, and 35 to the same 3
        over all three roots. (It said "wherever it is imported into a module
        that also has a `make_*_env` of its own" until PR B3, then briefly "at
        all three files that import it", which was the same conflation this
        sentence is warning about.) It resolves here today at six CALL
        SITES in THREE files — `src/profile_step.py` ×3, `src/train.py` ×2,
        `src/train_bc.py` ×1 — because a bare `make_c_env(...)` binds through
        `alias.asname`. Say which of the three quantities you mean, always: the
        aliased IMPORT STATEMENTS number FOUR over those same three files
        (train.py has two, both function-local), and conflating statements with
        files is what put a wrong "four files" in this very sentence in the round
        that was fixing wrong counts. Both instruments agree on the calls: the
        sibling's own scan reports the same 6 alias-spelled findings in the same
        3 files. That sibling scans the Name position for aliases too, and
        states why it stops there: resolving an alias in the ATTRIBUTE position
        would flag any unrelated method sharing the local's name. So this limit
        is a deliberate boundary in both files, not an oversight in one — but it
        is a boundary, and a legacy call written on the far side of it is
        invisible to every pin below.
    `tests/test_env_construction_enforcement.py`'s docstring discloses the third
    and fourth for its own scan; they are repeated here because this census has
    them too. If any appears, extend `_target`/`_bindings` rather than loosening a
    pin — widening the resolver moves this file's own pinned counts, so it is a
    change with a measurement attached, not a one-line fix.
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

# Asserted as a literal in test_the_anchor_set_is_pinned_and_each_anchor_is_a_
# resolved_call_site, for the same reason `_PLANT_NAMES` exists further down:
# ANCHORS is a guard set, and a guard set nothing watches is the failure class
# this repo keeps hitting. Measured before this pin existed: deleting an entry
# from ANCHORS left all 34 cases of the pre-fix file green.
# The two `tests/` entries are spelled out rather than read from FLOOR_FILES ON
# PURPOSE, even though ANCHORS derives them from it: a floor that legitimately
# moves has to be re-declared here, which is the whole point of a composition
# pin. The failure message says so.
_ANCHOR_NAMES = frozenset({
    "src/c_env/cs2_env.py",
    "scripts/sim_fingerprint.py",
    "tests/test_make_env_shim.py",
    "tests/test_env_config.py",
})

# Sanity floor for the "the root is real" guard, and deliberately the SAME value
# and shape the sibling scanner uses: `tests/test_env_construction_enforcement.py`
# pins MIN_FILES_PER_ROOT = 8 "deliberately far below the current counts so
# ordinary churn never touches them". (Its comment cites 24 and 19 files; run
# today its scan reads 26 and 19 — the same numbers this census reads for `src`
# and `scripts`, because both walk those two roots the same way.) Measured here
# at the tip of PR B3: 26 src / 19 scripts / 82 tests, so 8 leaves 11 files of
# shrinkage in the smallest root and 74 in the largest — a floor no legitimate
# deletion trips. It is loose ON PURPOSE, because a floor is the wrong instrument
# for the failure this file actually measured: a skip inside `census` that
# dropped 12 of the 17 test files holding legacy `make_env` calls still leaves 70
# files under `tests/`. That case is caught by the set-equality pin below
# instead, which needs no literal at all. This constant catches only what
# equality cannot see — both walks agreeing on a root that resolved to the wrong,
# or a nearly empty, directory.
MIN_FILES_PER_ROOT = 8


def test_the_scan_roots_resolve_to_real_populated_directories():
    """`rglob` on a missing or renamed root returns [] in silence, so every pin
    below would pass on an empty file list. This is the assertion that makes a
    zero mean something.

    WHAT THE ANCHORS ACTUALLY ARE, measured, because the sentence that used to
    stand here got all three of its claims wrong. There are FOUR of them, not
    three, and they are NOT one per root: `src/c_env/cs2_env.py` (1 resolved
    call), `scripts/sim_fingerprint.py` (1) and BOTH per-symbol floor files under
    `tests/` (5 and 5). Nor does each of them construct an env — all five
    resolved calls in `tests/test_env_config.py` sit inside `pytest.raises`, and
    that file's own section comment says "every one of them raises before any env
    is built", so it reaches construction on no code path at all. What every
    anchor really does is hold at least one call this census RESOLVES, which is
    the property the zeros below need and is asserted in
    test_the_anchor_set_is_pinned_and_each_anchor_is_a_resolved_call_site.
    """
    for root in ROOTS:
        paths = [p for p in SCANNED if p.relative_to(REPO_ROOT).as_posix().split("/", 1)[0] == root]
        assert paths, (
            f"root {root!r} contributed no .py files. rglob returns [] for a root that does "
            f"not exist, so this is what a rename or a typo in ROOTS looks like — repoint "
            f"ROOTS, never delete this assert.")
        assert len(paths) >= MIN_FILES_PER_ROOT, (
            f"root {root!r} yielded only {len(paths)} .py files (floor {MIN_FILES_PER_ROOT}). "
            f"The root exists but cannot be the directory this file is about — a repoint to a "
            f"subdirectory looks exactly like this, and every pin under it would pass over the "
            f"wrong file list.")
    scanned_rel = {p.relative_to(REPO_ROOT).as_posix() for p in SCANNED}
    missing = [a for a in ANCHORS if a not in scanned_rel]
    assert not missing, f"anchors missing from the scan: {missing} ({len(SCANNED)} files read)"


def test_the_scan_reads_every_py_file_under_the_roots():
    """THE DENOMINATOR PIN. Nothing else in this file asserts how many files the
    walk read, and every pin here is a property of a file list.

    MEASURED, which is why it exists: a skip added to `census`'s loop that
    dropped 12 of the 17 `tests/` files holding legacy `make_env` calls took that
    count from 54 to 12 and left all 34 cases of the pre-fix file GREEN. The
    ceiling pin only ever objects to a RISE, the floor pin is satisfied by the
    two sites left in `tests/test_make_env_shim.py`, and a file that is never
    read contributes no findings — so a census that reads less looks exactly like
    a migration that finished. Re-measured with this pin in place: the same
    mutation fails this test and nothing else, and names all 12 files.

    A floor on `len(SCANNED)` cannot see that (82 − 12 = 70), so this asserts SET
    EQUALITY against an independent re-walk of the same roots: no literal to go
    stale, no objection to a file being ADDED, and it fires on a skip of any
    size. What it cannot see is a wrong `ROOTS` or `REPO_ROOT`, because both
    walks read those — that half belongs to
    test_the_scan_roots_resolve_to_real_populated_directories above, which is why
    both tests exist.
    """
    expected = {p for root in ROOTS for p in (REPO_ROOT / root).rglob("*.py")}
    scanned = set(SCANNED)
    assert scanned == expected, (
        f"the census did not read every .py file under {ROOTS}.\n"
        f"  never read: {sorted(str(p.relative_to(REPO_ROOT)) for p in expected - scanned)}\n"
        f"  read but not re-walked: "
        f"{sorted(str(p.relative_to(REPO_ROOT)) for p in scanned - expected)}\n"
        f"A skipped file contributes no findings, and no findings is what every pin here wants "
        f"to see. Repair `census`; never repair this assertion.")
    assert len(SCANNED) == len(expected), (
        f"`census` returned {len(SCANNED)} paths for {len(expected)} distinct files — it read "
        f"something twice, so any count derived from SCANNED is inflated.")


def test_the_anchor_set_is_pinned_and_each_anchor_is_a_resolved_call_site():
    """`ANCHORS` is what stops the zeros above being zeros over the wrong tree,
    which makes it a guard set — and the pin above only checks that its members
    were SCANNED. Two holes follow, and this closes both.

    First, composition: measured, deleting an entry from `ANCHORS` left all 34
    cases of the pre-fix file green, so the set that underwrites every zero could
    shrink to nothing unwatched. With this pin, that same deletion fails here and
    nowhere else. Same remedy and same reason as `_PLANT_NAMES`.

    Second, and the stronger half: being scanned is not the same as being
    UNDERSTOOD. The two `tests/` anchors are also the floor files, so
    test_the_named_floor_file_holds_calls_gh173_cannot_migrate already proves the
    resolver works on them — but nothing proved it still resolves anything in
    `src/` or `scripts/`, where the pins are ratchets that read zero on a healthy
    tree. A resolver that silently stopped matching production spellings would
    leave those pins green. Requiring every anchor to contribute a resolved call
    (typed or legacy: 1 / 1 / 5 / 5 at the tip of PR B3) makes them a positive
    control over real source, not just a file-existence check.
    """
    assert set(ANCHORS) == set(_ANCHOR_NAMES), (
        f"ANCHORS composition changed: added {sorted(set(ANCHORS) - _ANCHOR_NAMES)}, "
        f"removed {sorted(_ANCHOR_NAMES - set(ANCHORS))}. Update _ANCHOR_NAMES deliberately; "
        f"do not delete this assertion.")
    assert len(ANCHORS) == len(_ANCHOR_NAMES), f"duplicate entry in ANCHORS: {ANCHORS}"
    resolved = {rel for _sym, rel, _lineno, _why in FOUND}
    silent = [a for a in ANCHORS if a not in resolved]
    assert not silent, (
        f"anchors the census resolved NO call in: {silent}. Each anchor is named because it "
        f"holds a call this census must recognise, so this is either a resolver that stopped "
        f"matching a spelling — the failure the `src`/`scripts` ratchets cannot show you — or "
        f"an anchor that genuinely lost its call, in which case repoint ANCHORS and "
        f"_ANCHOR_NAMES together at a file that still has one.")


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
    constructor, `config` has no default and there is no `**legacy`. A legacy
    `Cs2Env(...)` is a TypeError at runtime, so this pin is a static early
    warning, not a duplicate of one.

    TWO POPULATIONS, AND THIS DOCSTRING USED TO NAME ONLY THE SMALLER ONE ("its
    18 test call sites migrated in Phase A"), which is true of what Phase A
    MIGRATED and is not the population this pin covers. Re-measured 2026-09-11
    with this module's own `census` over a `git archive` of `tests/` at three
    trees, so only the tree varies: 18 resolved `Cs2Env` call sites in 3 files at
    `139a3a3`, the pre-#165 baseline; 19 in 4 files at the Phase-A tip
    `54d7da0`; 19 in 4 at this branch's base `60a2e66`. The extra file is
    `tests/test_make_env_shim.py`, which does not exist at `139a3a3` and which
    `git log --diff-filter=A` attributes to `ae478f8` — Phase A itself — so its
    one site was born typed rather than migrated. Neither figure is asserted
    anywhere; the assertion below is a zero.
    """
    sites = legacy_in("tests", "Cs2Env")
    assert not sites, f"legacy Cs2Env call(s) under tests/:\n{_report(sites)}"


@pytest.mark.parametrize("symbol,pin", [("make_env", PINNED_TESTS_MAKE_ENV),
                                        ("make_puffer_env", PINNED_TESTS_MAKE_PUFFER_ENV)])
def test_the_tests_root_legacy_count_only_ever_falls(symbol, pin):
    """gh#173 lowers these; nothing may raise them.

    The `> 0` half is not decoration. While the `**legacy` channel exists neither
    count can legitimately be zero, because for EACH symbol the file
    `FLOOR_FILES` names calls that symbol legacy on purpose — so a zero here
    means the census stopped resolving, not that the migration finished. That is
    the failure this test is really watching for; the ceiling is the easy half.

    THE FLOOR IS PER SYMBOL, and this docstring says so rather than naming one
    file, because the measured matrix is anti-diagonal: at the tip of PR B3
    `tests/test_make_env_shim.py` holds 2 legacy `make_env` calls and ZERO legacy
    `make_puffer_env` calls, while `tests/test_env_config.py` holds 5 legacy
    `make_puffer_env` calls and ZERO legacy `make_env` calls. Naming either file
    as "the" reason the count cannot be zero is therefore false for one of this
    test's two parametrizations — which is exactly the shared-floor answer that
    `FLOOR_FILES` was keyed per symbol to refute, and that the next test's
    docstring argues against. The assertion message interpolates
    `FLOOR_FILES[symbol]`, so it has always named the right file; only this
    docstring did not.
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

# name -> (path in the scratch tree, source, the ONE clause number it must report).
#
# The PATH is part of the control: `_bindings`'s in-module clause is scoped by
# `DEFINING_FILE`, so the two in-module plants only resolve when planted at the
# file that really defines the symbol.
#
# The CLAUSE is the other half of the control, and PR B3 is where it arrived.
# Until then the plant test asserted only that a plant came back LEGACY, never
# which clause said so — and the three clauses are not independent, because
# clause 1's `elif` sees the splat keyword too: an `ast.keyword` with
# `arg=None` satisfies `kw.arg != "config" and kw.arg not in RUNTIME[target]`.
# Measured 2026-09-11 by disabling `if kw.arg is None` in `_legacy_reasons`:
# `uv run pytest -q -p no:randomly tests/test_env_config_migration.py
# tests/test_env_construction_enforcement.py` stayed at 56 passed, with all
# three splat plants reporting `['clause 1: legacy keyword None']` instead —
# 56 being the two-file total AT `60a2e66`, before this fix; re-run at HEAD the
# same mutation gives 3 failed of 57. The
# same command with clause 1 disabled gives 12 failed and with clause 3 disabled
# 4 failed, both still true at HEAD, so clause 2 was the one uncovered arm, not
# a general weakness. Every
# plant below reports exactly one clause (measured), which is why the assertion
# is set equality against `{clause}` and not a membership test.
_PLANTS = {
    "clause1-name":
    ("src/p.py", "from c_env.cs2_env import make_env\nmake_env(reward_kill=2.0)\n", 1),
    "clause1-alias":
    ("src/p.py", "from c_env.cs2_env import make_env as mk\nmk(reward_kill=2.0)\n", 1),
    "clause1-attr": ("src/p.py", "import c_env.cs2_env as m\nm.make_env(reward_kill=2.0)\n", 1),
    "clause1-submodule":
    ("src/p.py", "from c_env import cs2_env\ncs2_env.make_env(reward_kill=2.0)\n", 1),
    "clause2-name": ("src/p.py", "from c_env.cs2_env import make_env\nmake_env(**kw)\n", 2),
    "clause2-attr": ("src/p.py", "import c_env.cs2_env as m\nm.make_env(**kw)\n", 2),
    "clause3-name": ("src/p.py", "from c_env.cs2_env import make_env\nmake_env(0)\n", 3),
    "clause3-attr": ("src/p.py", "import c_env.cs2_env as m\nm.make_env(0)\n", 3),
    "puffer-name": ("src/p.py",
                    "from train import make_puffer_env\nmake_puffer_env(reward_kill=2.0)\n", 1),
    "puffer-alias": ("src/p.py", "from train import make_puffer_env as mpe\nmpe(**kw)\n", 2),
    "puffer-attr": ("src/p.py", "import train\ntrain.make_puffer_env(0)\n", 3),
    "puffer-inmodule":
    ("src/train.py", "def make_puffer_env(**kw):\n    pass\n\n\nmake_puffer_env(reward_kill=2.0)\n",
     1),
    "classdef-inmodule": ("src/c_env/cs2_env.py",
                          "class Cs2Env:\n    pass\n\n\nCs2Env(n_active_per_team=1)\n", 1),
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
    # The declared clause is now data, so it needs the same treatment the names
    # get. Both directions: every clause must be represented at all, and a name
    # that STATES a clause must declare that one — the shape of "the census
    # broke, so the expectation was edited down until it passed again".
    declared = {clause for _path, _src, clause in _PLANTS.values()}
    assert declared == {
        1, 2, 3
    }, (f"the plants declare clauses {sorted(declared)}; every clause in `_legacy_reasons` needs "
        f"at least one plant that must report it, or that clause is unwatched again.")
    for name, (_path, _src, clause) in _PLANTS.items():
        if name.startswith("clause"):
            assert clause == int(name[len("clause")]), (
                f"plant {name!r} declares clause {clause}. The name and the expectation disagree, "
                f"which is what lowering an expectation to match a broken census looks like.")


@pytest.mark.parametrize("name", sorted(_PLANTS))
def test_each_clause_is_found_in_a_planted_tree(tmp_path, name):
    """THE control for every zero above. One planted file per clause per
    spelling, each of which must come back legacy FOR THE CLAUSE IT WAS WRITTEN
    FOR.

    THAT LAST PART IS THE WHOLE FIX, and PR B3 is where it arrived. "Came back
    legacy" is not the same claim as "clause 2 works": clause 1's `elif` in
    `_legacy_reasons` matches a splat keyword too, because `arg=None` is neither
    `"config"` nor a member of `RUNTIME[target]`. Measured 2026-09-11 with
    `if kw.arg is None` disabled, the whole splat clause dead: the two-file run
    `uv run pytest -q -p no:randomly tests/test_env_config_migration.py
    tests/test_env_construction_enforcement.py` stayed at 56 passed, and the
    three splat plants came back as `['clause 1: legacy keyword None']` — legacy,
    for a reason that is a misreport. So this parametrization was, for clause 2
    alone, a control that could not tell working from broken. Clauses 1 and 3
    were never in that position: the same command with clause 1 disabled fails 12
    and with clause 3 disabled fails 4.

    THE 56 IS THE PRE-FIX EPOCH, and the 12 and the 4 are not. 56 is the
    two-file total at `60a2e66`, before this fix and before the sibling's new
    composition pin took it to 57; re-run the clause-2 mutation at HEAD and it
    fails 3 of 57 rather than passing all of them, which is the whole point of
    the fix. The 12 and the 4 were re-measured at HEAD and are unchanged, so
    only this one figure needed its tree named.

    `classdef-inmodule` and `puffer-inmodule` are the controls for `_bindings`'s
    in-module clause, each planted AT its own defining file: the first is the
    shape `make_env`'s own `return Cs2Env(...)` has in `src/c_env/cs2_env.py`,
    the second the shape a bare `make_puffer_env(...)` would have in
    `src/train.py`. Without that clause the `Cs2Env` `src`=0 pin has no site it
    could ever fail on. Neither name states a clause, so their expectation is
    only readable in `_PLANTS`: both pass a legacy KEYWORD, so both are clause 1.
    """
    path, source, clause = _PLANTS[name]
    _, found = _plant(tmp_path, path, source)
    legacy = [f for f in found if f[3]]
    assert legacy, f"planted {name} was NOT found: census returned {found}"
    fired = {int(why.split(":")[0].removeprefix("clause ")) for f in legacy for why in f[3]}
    assert fired == {
        clause
    }, (f"planted {name} came back legacy for clause(s) {sorted(fired)}, not for clause {clause}: "
        f"{[f[3] for f in legacy]}. A plant that fires the WRONG clause is still 'legacy', which "
        f"is how clause 2 stayed uncovered until PR B3 — clause 1's `elif` reports the splat "
        f"keyword as `legacy keyword None` the moment `if kw.arg is None` stops matching first. "
        f"Repair the clause, never this expectation.")


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


def test_a_file_the_census_cannot_parse_is_an_error_and_never_a_skip(tmp_path):
    """`census`'s docstring states the policy — "Parse errors are NOT caught" —
    and this is the control that turns the policy into an invariant.

    MEASURED twice. On the 34-case file that shipped without this control,
    adding `except SyntaxError: continue` to `census`'s loop left every case
    GREEN; with the control in place the same mutation fails exactly one of the
    37, this one, with DID NOT RAISE. It has to be silent otherwise, by the shape
    of the pins: an unreadable file yields no findings, and no findings is what a
    ratchet, a ceiling and a floor of two all want to see. So the policy holding
    today is a property of `census`'s current text and of nothing else, and a
    stated invariant with no control is the failure class this branch keeps
    closing.

    The `filename` half has teeth of its own: `ast.parse` is called with
    `filename=`, so the raise names the file that broke. A swallow-and-log
    rewrite would still satisfy a bare `pytest.raises`; it would not satisfy an
    assertion that the exception carries the path.
    """
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "unparseable.py").write_text("def f(:\n")
    with pytest.raises(SyntaxError) as exc:
        census(roots=("src", ), repo_root=tmp_path)
    assert "unparseable.py" in (exc.value.filename or ""), (
        f"the parse error did not name the file it came from: {exc.value.filename!r}. "
        f"`ast.parse` is called with filename= precisely so an unreadable file is "
        f"attributable rather than a bare traceback.")


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
