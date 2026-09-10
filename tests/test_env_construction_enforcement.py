"""No `make_puffer_env(...)` or `SelfPlayManager(...)` call outside `src/env_factory.py`.

READ THE SCOPE LINE ABOVE LITERALLY. This file enforces exactly two SYMBOLS in
two ROOTS, and nothing wider. It is not "no env is built outside the factory" —
a lower layer of constructors (`make_env` / `Cs2Env`) is called directly at a
dozen live sites inside those same roots, on purpose, and this scan does not look
at them. The census and the reasoning are in LOWER_LAYER_SITES below; read it
before quoting this file as evidence that all env construction is centralised.

WHAT THIS ENFORCES (spec 2026-08-31 §2 W3, #154). W3 exists because
`make_puffer_env` had seven independently-drifting call sites and
`SelfPlayManager` three: a knob added to one silently skipped the rest, and
`--smoke --reward-ct-survival 0.0` running the DEFAULT weights is the documented
instance. Routing them through one factory only helps while they STAY routed —
the eighth site someone adds next year is the failure, and it looks like working
code. So the roots `<repo>/src` and `<repo>/scripts` are scanned, and the only
file allowed to construct either symbol is the factory itself.

WHY THIS FILE IS UNUSUALLY DEFENSIVE ABOUT ITSELF. A green result here is "zero
constructions found", and that is ALSO what every broken version of this scan
reports: a matcher with the wrong attribute name, a root that resolves to an
empty or missing directory, a walk that skips subdirectories, an over-eager
carve-out. None of those raise — they return an empty list and pass forever. The
`scripts/` root makes it concrete: it legitimately contains zero constructions
TODAY, so half of this scan is already indistinguishable from a scan that never
looked at it. Hence five separate guards ahead of the enforcement assertion:

  1. the roots exist, contain a plausible number of .py files, and contain named
     anchor files — kills "root resolved to nothing";
  2. the matcher finds all four real SPELLINGS in this repo's own source
     (`make_puffer_env(...)`, `train.make_puffer_env(...)`, `SelfPlayManager(...)`,
     `train.SelfPlayManager(...)`), which live in `tests/` and in the factory —
     kills "matcher names the wrong thing";
  3. a construction PLANTED in a scratch tree is found, in both spellings, in a
     nested subdirectory — kills "the walk or the parse is broken", end to end;
  4. the classmethod carve-out is checked in both directions, against real
     source — kills "the carve-out was loosened until it swallowed the coverage";
  5. an IMPORT-ALIASED construction is planted and found, and a call to the same
     local name WITHOUT the aliasing import is left alone — kills "a one-line
     rename walks past the matcher", which it did until the final review's I-4.

ALIAS RESOLUTION, and why it is not optional. `from train import make_puffer_env
as _mpe` followed by `_mpe(...)` is one line and defeats a matcher that keys on
the literal spelling of the call. That is not a hypothetical spelling in this
repo: `from c_env.cs2_env import make_env as make_c_env` is house style at three
of the FILES in the LOWER_LAYER_SITES census below — four import STATEMENTS,
because `src/train.py` writes it twice, function-locally in two functions — so
the alias form is what this codebase actually writes. `import_aliases` therefore
binds `asname -> symbol` per FILE and the matcher resolves through it, reporting
the canonical symbol with the local name in the spelling column so a failure
names the alias it resolved.
Only `ImportFrom` is walked: `import X as Y` can bind only a MODULE, and
`Y.make_puffer_env(...)` is already the attribute spelling. Assignment aliases
(`f = make_puffer_env`), `getattr`, subclassing, `exec` and `subprocess -c`
remain OUT of scope — see M-6 in the final review; this scan is a guard against
drift by ordinary editing, not against a determined bypass.

THE CARVE-OUT, precisely. A construction is `Call(func=Name(X))` or
`Call(func=Attribute(attr=X))` for X in {make_puffer_env, SelfPlayManager}.
`Call(func=Attribute(value=Name(X), attr=<other>))` is CLASSMETHOD ACCESS and is
NOT a construction: `SelfPlayManager.initial_hero_team()` is a live, legitimate
call in both `train()` and `_build_trainer_for_test`. Matching the ATTRIBUTE
position (`attr`) rather than the value position is the whole mechanism — a
naive value-position match flags those two sites, and the tempting fix is to
loosen the matcher until the real coverage dies, with nothing to notice.

WHY `src/env_factory.py` SHOWS ONE CONSTRUCTION AND NOT TWO, AND HAS TWO PINS
FOR IT. It builds the manager directly (`SelfPlayManager(...)`), so the scan finds
that. It contains no CONSTRUCTOR call at all: `build_env_for` imports the
constructor function-locally and passes it to the role builder as a VALUE
(`src/env_factory.py:372` reads `return builder(make_env, **kwargs)`; the six role
builders receive it positional-only as `_make`, `def _build_train(_make, /, ...)`),
so no call node in that file names it. Both halves are pinned, because they are
different regressions: the banned `make_puffer_env`, which the enforcement
assertion at the bottom of this file excludes the factory from (`if p != FACTORY`),
and the lower-layer `make_env`/`Cs2Env` that B2 made `build_env_for` import, one
inlined line away. The absent `make_puffer_env(...)` call is also why guard 2 takes
its `make_puffer_env` evidence from `tests/`, where ~40 direct constructions
legitimately live, rather than from the factory.

THE UNSCANNED LOWER LAYER — the honest limit of this file (final review I-1).
`make_puffer_env` is a wrapper: it applies the reward-override validation and the
Rung-0 knobs and then calls `make_env` (aliased `make_c_env`), which loads the map
and calls `Cs2Env`. The spec banned the two TOP-layer symbols only, because that
is the layer whose per-site knob drift caused #154; a caller that wants a bare
default env with no training knobs at all — the profiler, the recorder, the BC
demo generator, the fingerprint script — is not drifting from anything, it is
using a lower constructor on purpose. Those calls are real, they are inside the
scanned roots, and a green result here says NOTHING about them. LOWER_LAYER_SITES
below is their census, pinned by a test so this paragraph cannot rot into a claim
about a population that has since doubled. Note what they no longer all share:
#165 gave five of the twelve resolved calls an `EnvConfig` — the wrapper chain's
own two (`Cs2Env` inside `make_env`, `make_c_env` inside `make_puffer_env`) plus
`src/play.py`, `scripts/oracle_statue_check.py` and `scripts/sim_fingerprint.py` —
while the other seven still take `make_env`'s defaults and so see no W5 stance flag
and no Rung-0 knob from config: `scripts/gen_bc_demos.py`,
`scripts/measure_budget.py`, all three in `src/profile_step.py`, `record_episode` in
`src/train.py`, and `src/train_bc.py` (measured 2026-09-10; `LOWER_LAYER_SITES` pins
calls per FILE, not which of them are configured, so this split is prose, not a
pin). Extending the ban (or adding factory roles) to that layer is a separate
decision, tracked as future work, not something this file quietly did.
"""
import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The two symbols whose construction W3 centralises.
CONSTRUCTED = ("make_puffer_env", "SelfPlayManager")

# Explicit roots, per spec §2 W3 — never a walk from the repo root, which would
# visit `.worktrees/` (checkouts of this same repo, with their own copies of
# every site) and report findings that belong to another branch.
ROOTS = (REPO_ROOT / "src", REPO_ROOT / "scripts")

# The one file allowed to construct. Everything else routes through it.
FACTORY = REPO_ROOT / "src" / "env_factory.py"

# Sanity floors for the "the root is real" guard. Deliberately far below the
# current counts (24 and 19 files) so ordinary churn never touches them; they
# exist to catch a root that resolved to nothing, not to pin a file count.
MIN_FILES_PER_ROOT = 8

# Anchors that must be among the scanned files. A root can exist, contain .py
# files and still be the WRONG directory; naming the files whose contents this
# test is actually about removes that.
ANCHORS = ("src/train.py", "src/train_test_harness.py", "src/env_factory.py")

# ── The lower layer this file deliberately does NOT ban ─────────────────────
#
# `make_env` (spelled `make_c_env` wherever it is imported into a module that
# also has a `make_*_env` of its own) and the `Cs2Env` class it returns. See the
# module docstring's LOWER LAYER paragraph for why they are out of the ban.
LOWER_LAYER = ("make_env", "Cs2Env")

# Every call to one of those, per file, as of the final review of
# refactor/post-rung1a. Two of the twelve are the constructors' own definitions
# (the `return make_c_env(...)` inside `make_puffer_env`, and the
# `return Cs2Env(...)` inside `make_env`) — i.e. the wrapper chain itself, not
# extra callers; the other ten are the direct callers named in review finding
# I-1. This is a DISCLOSURE list, not a ban: a new entry is allowed, it just has
# to be written down here so the docstring above keeps telling the truth.
LOWER_LAYER_SITES = {
    "src/c_env/cs2_env.py": 1,                         # `make_env`'s own `return Cs2Env(...)`
    "src/play.py": 1,                                  # the interactive viewer
    "src/profile_step.py": 3,                          # three step-timing harnesses
    "src/train.py": 2,                                 # `make_puffer_env`'s own call + `record_episode`
    "src/train_bc.py": 1,                              # BC demo replay env
    "scripts/gen_bc_demos.py": 1,
    "scripts/measure_budget.py": 1,
    "scripts/oracle_statue_check.py": 1,
    "scripts/sim_fingerprint.py": 1,
}

# The alias-spelled SUBSET of LOWER_LAYER_SITES: `{file: calls}` for the calls
# reachable only through `from c_env.cs2_env import make_env as make_c_env`.
# Three UNITS meet here and they differ, which is how a `>= 4` ended up under a
# docstring claiming six: six CALLS, in three FILES, bound by four import
# STATEMENTS (`src/profile_step.py:36`, module-level and covering all three of
# its calls; `src/train.py:756` and `:1021` and `src/train_bc.py:467`, each
# function-local and covering one).
#
# WHY EXACT EQUALITY RATHER THAN A FLOOR, and where the headroom went. This dict
# is the oracle of the live positive control below, whose entire value IS the
# count: a floor lets the population shrink under a green run, and at `>= 4`
# either `src/train_bc.py` (1 call) or `src/train.py` (2) could stop resolving,
# or stop being written with the alias, with the docstring above false and
# nothing red. Zero headroom in the total costs a one-line edit here — in the
# commit that already has to fix the two docstrings naming these numbers — and it
# is the policy LOWER_LAYER_SITES already gets, twenty lines up, for this same
# population. The headroom that DOES belong here is per FILE, and this shape has
# it: alias resolution binds per file, so a real regression drops a whole entry
# and the failure message names which, while a legitimate de-alias of one file
# changes that entry and nothing else.
ALIASED_LOWER_LAYER_SITES = {
    "src/profile_step.py": 3,
    "src/train.py": 2,
    "src/train_bc.py": 1,
}


def import_aliases(tree, symbols):
    """`{local name: symbol}` for every `from <mod> import <symbol> as <local>`.

    WHY (final review I-4): the matcher below keys on the literal spelling of the
    call, so `from train import make_puffer_env as _mpe` + `_mpe(...)` walked
    straight past it — a one-line bypass. And the alias form is not hypothetical
    here: `from c_env.cs2_env import make_env as make_c_env` is house style at
    three FILES — four import STATEMENTS, since `src/train.py` writes it twice —
    so it is a spelling this repo genuinely writes.

    ONLY `ImportFrom`. `import X as Y` can bind a MODULE and nothing else, and
    `Y.make_puffer_env(...)` is already the attribute spelling.
    """
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in symbols and alias.asname:
                    out[alias.asname] = alias.name
    return out


def constructions(source, label="<source>", symbols=CONSTRUCTED):
    """[(symbol, lineno, spelling)] for every construction call in `source`.

    `spelling` is "name" for `X(...)`, "attribute" for `<anything>.X(...)`, and
    "alias:<local>" for a call through an `import ... as` binding. All three
    count: `train.make_puffer_env(...)` and `_mpe(...)` build exactly the same
    env as `make_puffer_env(...)`, and a scan that only looked for bare names
    would be bypassed by one import line at the top of the new file.

    `symbols` is a parameter so the same matcher can take the census of the
    LOWER_LAYER constructors it does not ban — which is also the strongest
    available positive control for alias resolution, since it has to resolve the
    four real `make_env as make_c_env` imports to report those sites at all.
    """
    tree = ast.parse(source, filename=label)
    aliases = import_aliases(tree, symbols)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in symbols:
            out.append((func.id, node.lineno, "name"))
        elif isinstance(func, ast.Name) and func.id in aliases:
            # Report the CANONICAL symbol, and name the local in the spelling so
            # a failure message points at the line that has to change. Scoped to
            # the Name position on purpose: resolving aliases in the attribute
            # position would flag any unrelated method that happens to share the
            # local's name.
            out.append((aliases[func.id], node.lineno, f"alias:{func.id}"))
        elif isinstance(func, ast.Attribute) and func.attr in symbols:
            # ATTRIBUTE position only. `SelfPlayManager.initial_hero_team()` has
            # attr="initial_hero_team" and is therefore not matched here — see
            # the module docstring, and test_classmethod_access_is_not_a_
            # construction, which pins that in both directions.
            out.append((func.attr, node.lineno, "attribute"))
    return out


def scan(roots, symbols=CONSTRUCTED):
    """(scanned files, findings) over every .py under `roots`, recursively.

    A root may be a directory or a single .py file; the guards below point it at
    both. Returns the file list as well as the findings, because "which files did
    you actually read" is the question a vacuous green cannot answer.
    """
    files, findings = [], []
    for root in roots:
        root = Path(root)
        for path in ([root] if root.is_file() else sorted(root.rglob("*.py"))):
            files.append(path)
            for symbol, lineno, spelling in constructions(path.read_text(), str(path), symbols):
                findings.append((path, symbol, lineno, spelling))
    return files, findings


# ── guard 1: the roots are real ─────────────────────────────────────────────


def test_the_scan_roots_resolve_to_real_populated_directories():
    """Each root exists, holds .py files, and the anchors are among them.

    Without this, renaming `src/` (or a typo in ROOTS) turns the enforcement
    assertion below into a permanent pass over an empty file list. `scripts/`
    makes the risk concrete: it contains zero constructions today, so nothing
    about its RESULT could tell you whether it was read.
    """
    scanned, _ = scan(ROOTS)
    for root in ROOTS:
        assert root.is_dir(), f"scan root {root} does not exist"
        in_root = [p for p in scanned if root in p.parents or p.parent == root]
        assert len(in_root) >= MIN_FILES_PER_ROOT, (
            f"scan root {root} yielded only {len(in_root)} .py files — the walk is not reaching "
            "its contents, and every assertion below it would pass vacuously")

    relative = {str(p.relative_to(REPO_ROOT)) for p in scanned}
    missing = [a for a in ANCHORS if a not in relative]
    assert not missing, f"the scan never visited {missing}; it is reading the wrong tree"


# ── guard 2: the matcher finds real constructions in this repo ──────────────


def test_the_matcher_finds_every_spelling_in_this_repos_own_source():
    """All four (symbol, spelling) combinations are found where they really live.

    THE POSITIVE CONTROL. `tests/` constructs envs and managers directly by
    design — it is testing them — so it is a standing, unmigrated population of
    exactly what this matcher must recognise, and `src/env_factory.py` supplies
    the fourth combination. A matcher that named `make_env` instead of
    `make_puffer_env`, or that only looked at bare names, reports zero findings
    in `src/` either way; here it fails loudly.

    If a future task migrates the test suite onto the factory too, this will
    fail — correctly, because it will mean the positive control has evaporated
    and needs a new source, not that the scan is fine.
    """
    _, found = scan([REPO_ROOT / "tests", FACTORY])
    combos = {(symbol, spelling) for _, symbol, _, spelling in found}
    expected = {(s, sp) for s in CONSTRUCTED for sp in ("name", "attribute")}
    assert combos >= expected, (
        f"the matcher found only {sorted(combos)}; it must recognise all of {sorted(expected)}. "
        "Missing spellings mean the enforcement assertion is blind to them in src/ too.")

    envs = [f for f in found if f[1] == "make_puffer_env"]
    assert len(envs) >= 20, (f"only {len(envs)} make_puffer_env constructions found across tests/ "
                             "and the factory; the matcher has stopped matching")


def test_the_factory_itself_is_where_the_construction_lives():
    """`src/env_factory.py` constructs — the exemption is not covering an empty file.

    An exemption for a file that constructs nothing is the same vacuous-pass
    shape as an empty root: it would mean the constructions moved somewhere the
    scan cannot see rather than into the factory.
    """
    _, found = scan([FACTORY])
    symbols = {symbol for _, symbol, _, _ in found}
    assert "SelfPlayManager" in symbols, (
        "env_factory.py no longer constructs a SelfPlayManager; either build_selfplay_manager "
        "moved out, or it stopped calling the class directly")
    # And the documented asymmetry, pinned so the docstrings above stay true:
    # build_env_for passes make_puffer_env as a VALUE, so no call node names it.
    assert "make_puffer_env" not in symbols, (
        "env_factory.py now contains a direct make_puffer_env(...) call. That is allowed by the "
        "exemption, but this file's module docstring and env_factory's both explain that it does "
        "NOT — update both rather than deleting this assertion.")


def test_the_factory_does_not_call_the_lower_layer_constructor_directly():
    """The same documented asymmetry as the assertion above, pointed at the
    symbol #165 PR B2 put at risk.

    `build_env_for` imports `c_env.cs2_env.make_env` function-locally and hands it
    to the role builder as a VALUE (`src/env_factory.py:370` and `:372`; the six
    builders receive it positional-only as `_make`), so no CALL node in that file
    names it. B2 is what made this the pin worth adding: it swapped the
    function-local `from train import make_puffer_env` for
    `from c_env.cs2_env import make_env`, so `make_env` is now the only
    constructor that file references in CODE at all. Measured 2026-09-10:
    `make_puffer_env` has ZERO code references there and ten textual ones, all on
    docstring lines; `make_env` has two, the import and the value pass. So the
    regression the assertion above guards needs someone to import a name that file
    no longer mentions in code, while the regression this one guards needs one
    line inlined into a builder.

    This is an ADDITION, not a replacement. The `make_puffer_env` assertion above
    is a live negative, not a vacuous one: plant a real `make_puffer_env(...)` call
    into a copy of `src/env_factory.py` and the same scan reports
    `{'SelfPlayManager', 'make_puffer_env'}`. And
    `test_no_make_puffer_env_or_selfplaymanager_call_outside_the_factory` filters
    the factory out of its own scan (`if p != FACTORY`), so deleting that assertion
    would leave a direct `make_puffer_env(...)` call inside the factory forbidden
    nowhere in the suite.

    If a later branch inlines `_make` into the builders this goes red: update
    `env_factory`'s docstring and `LOWER_LAYER_SITES` (which gains ONE entry,
    `"src/env_factory.py"`, with a count — the dict is keyed by file, not by
    builder), do not delete the assertion.
    """
    _, found = scan([FACTORY], symbols=LOWER_LAYER)
    symbols = {symbol for _, symbol, _, _ in found}
    assert "make_env" not in symbols, (
        "src/env_factory.py now contains a direct make_env(...) call. Both this file's "
        "module docstring and env_factory's own explain that it does NOT — update both, and "
        "LOWER_LAYER_SITES, rather than deleting this assertion.")
    assert "Cs2Env" not in symbols, (
        "src/env_factory.py now constructs a Cs2Env directly, bypassing make_env's map "
        "loading entirely.")


# symbol -> the body inlined into a planted copy of `src/env_factory.py`. One
# entry per assertion in the pin above; a pin with two negatives needs two
# knock-outs, not one. STRING LITERALS on purpose: this file is inside
# tests/test_env_config_migration.py's scan root, and a real `make_env(...)`
# call node here would raise that census's pinned tests/ count.
_INLINED_CONSTRUCTION = {
    "make_env": ("    from c_env.cs2_env import make_env\n"
                 "    return make_env(config=None, map_data=md)\n"),
    "Cs2Env": ("    from c_env.cs2_env import Cs2Env\n"
               "    return Cs2Env(config=None, map_data=md)\n"),
}


@pytest.mark.parametrize("symbol", sorted(_INLINED_CONSTRUCTION))
def test_knockout_the_repointed_asymmetry_pin_fails_on_a_planted_call(tmp_path, symbol):
    """The pin above is a NEGATIVE over one real file, so on today's tree it is
    green no matter what the matcher does. Plant `src/env_factory.py`'s own source
    PLUS one direct call and require the same scan to report it.

    Parametrized over BOTH of the pin's assertions: they are two separate
    negatives, and proving the matcher for `make_env` proves nothing about `Cs2Env`
    that the reader can check without reading `scan`.

    What this does NOT prove: `constructions` matches `Call(func=Name(...))`
    directly and consults `import_aliases` only for `as`-aliases, so the planted
    `from ... import ...` line is inert here — measured, removing it leaves this
    test green. Name-position matching is what is being proved; the import line is
    present so the plant reads like the real file, not because it is exercised.
    """
    planted = tmp_path / "env_factory.py"
    planted.write_text(FACTORY.read_text() + "\n\ndef _inlined(md):\n" +
                       _INLINED_CONSTRUCTION[symbol])
    _, found = scan([planted], symbols=LOWER_LAYER)
    symbols = {s for _, s, _, _ in found}
    assert symbol in symbols, (
        f"the LOWER_LAYER scan of a factory WITH a direct {symbol} call reported {symbols} — "
        f"the pin above would pass straight through a real regression")


def test_alias_resolution_is_exercised_by_real_source_not_only_by_a_plant():
    """The six live `make_env as make_c_env` calls are reachable ONLY through aliases.

    THE POSITIVE CONTROL FOR ALIAS RESOLUTION, and the reason it is taken from the
    LOWER_LAYER census rather than from a plant: `train.py`, `train_bc.py` and
    `profile_step.py` all import the lower constructor under a different local
    name, so a matcher that keys on literal spellings reports those six sites as
    zero — the exact I-4 failure, in real source, on every run. If the repo ever
    stops writing the alias form this fails, which is correct: it means the live
    control is gone and the plant below is all that is left.

    IT ASSERTS THE POPULATION, NOT A FLOOR. The sentence above claims six calls in
    three files, so any threshold under six is an assertion weaker than its own
    claim — which is what `>= 4` was, a number taken from the four import
    STATEMENTS and applied to a count of CALLS. Under it the two smallest entries
    could vanish with this control green and its message silent.
    ALIASED_LOWER_LAYER_SITES carries the units and the headroom argument.

    Read a failure the way `test_the_unbanned_lower_layer_census_is_accurate` asks
    its own to be read — by checking WHICH reading moved. An entry that dropped to
    zero means alias resolution stopped binding in that file; an entry that is
    simply gone means the file stopped writing the alias form and the prose above
    needs a new count. Neither is a reason to lower a number until it passes.
    """
    _, found = scan(ROOTS, LOWER_LAYER)
    aliased = {}
    for path, _, lineno, spelling in found:
        if spelling.startswith("alias:"):
            aliased.setdefault(str(path.relative_to(REPO_ROOT)), []).append(lineno)
    counts = {rel: len(linenos) for rel, linenos in aliased.items()}
    assert counts == ALIASED_LOWER_LAYER_SITES, (
        f"the import-aliased lower-layer population has changed. resolved {counts} "
        f"(lines {aliased}), expected {ALIASED_LOWER_LAYER_SITES}\n"
        f"  no longer resolving at all: {sorted(set(ALIASED_LOWER_LAYER_SITES) - set(counts))}\n"
        "Either alias resolution stopped working, or the repo stopped writing "
        "`make_env as make_c_env`. Check WHICH before touching this dict, and update the "
        "ALIAS RESOLUTION paragraph and import_aliases' docstring with it.")


def test_the_unbanned_lower_layer_census_is_accurate():
    """LOWER_LAYER_SITES matches the tree, per file, exactly.

    This is a DISCLOSURE test, not an enforcement one — read a failure as "the
    module docstring's LOWER LAYER paragraph just went stale", not as "route this
    through the factory". The paragraph tells readers that a dozen direct
    `make_env`/`Cs2Env` calls live inside the scanned roots and are NOT covered by
    the enforcement assertion below; an un-pinned prose count is exactly the kind
    of claim that is true on the day it is written and quietly false a year later,
    which is the whole class of defect (final review I-1) this test closes.

    When it fails: update LOWER_LAYER_SITES, and take the moment to ask whether
    the new site wants a `build_env_for` role instead. That question is the point
    of pinning the count; the answer is allowed to be no.
    """
    _, found = scan(ROOTS, LOWER_LAYER)
    counts = {}
    for path, _, _, _ in found:
        rel = str(path.relative_to(REPO_ROOT))
        counts[rel] = counts.get(rel, 0) + 1
    assert counts == LOWER_LAYER_SITES, (
        "the census of unbanned lower-layer env constructions has changed.\n"
        f"  new/changed: {sorted(set(counts.items()) - set(LOWER_LAYER_SITES.items()))}\n"
        f"  gone/changed: {sorted(set(LOWER_LAYER_SITES.items()) - set(counts.items()))}\n"
        "Update LOWER_LAYER_SITES and the module docstring's LOWER LAYER paragraph.")


# ── guard 3: the knock-out — a planted construction must be found ───────────


@pytest.mark.parametrize("symbol", CONSTRUCTED)
@pytest.mark.parametrize("spelling", ["name", "attribute"])
def test_knockout_a_planted_construction_in_a_scratch_tree_is_found(tmp_path, symbol, spelling):
    """Plant a construction in a nested directory; the scan must report it.

    THE KNOCK-OUT, and the reason it plants into a scratch tree rather than into
    `src/`: this way it runs on every CI run instead of being a procedure someone
    performs by hand once and never again. It exercises the whole pipeline —
    directory walk, recursion into a subdirectory, file read, parse, match — so a
    break anywhere in it fails here rather than turning the enforcement assertion
    quietly green.

    (The by-hand version was also performed once, against a real `src/` file, to
    confirm the enforcement test itself fails: recorded in the task report.)
    """
    nested = tmp_path / "fakesrc" / "deeper"
    nested.mkdir(parents=True)
    (tmp_path / "fakesrc" / "innocent.py").write_text("x = 1\n")
    call = f"{symbol}(seed=1)" if spelling == "name" else f"train.{symbol}(seed=1)"
    (nested / "offender.py").write_text(f"def f():\n    return {call}\n")

    files, found = scan([tmp_path])
    assert len(files) == 2, f"the walk visited {files}, not both planted files"
    assert [(f[1], f[3]) for f in found] == [
        (symbol, spelling)
    ], (f"the scan did not report the planted {spelling}-spelled {symbol} construction: {found}")


def test_knockout_the_enforcement_assertion_fails_on_a_planted_offender(tmp_path):
    """The assertion itself — not just the scan — must reject a real offender.

    Distinct from the knock-out above, which proves the scan FINDS things. This
    proves the pass/fail rule around it is wired up: a finding outside the
    exempt file has to make the check fail. A scan that found everything and then
    filtered it all away as "exempt" would satisfy the previous test and still
    enforce nothing.
    """
    (tmp_path / "offender.py").write_text("env = make_puffer_env(seed=1)\n")
    _, found = scan([tmp_path])
    offenders = [f for f in found if f[0] != FACTORY]
    assert offenders, "the exemption filter swallowed a finding in a file that is not the factory"


# ── guard 5: import aliases, in both directions ─────────────────────────────


@pytest.mark.parametrize("symbol", CONSTRUCTED)
def test_knockout_an_import_aliased_construction_is_found(tmp_path, symbol):
    """`from train import <symbol> as _x` + `_x(...)` must be reported as <symbol>.

    THE I-4 KNOCK-OUT. Before alias resolution this planted file scanned clean —
    one import line was the entire bypass — so this is the test that fails if
    `import_aliases` is deleted or narrowed. It asserts the CANONICAL symbol name
    is what gets reported (not the local), because that is what makes the
    enforcement failure message readable, and that the local name survives in the
    spelling so the message points at the line to change.
    """
    (tmp_path / "offender.py").write_text(f"from train import {symbol} as _aliased\n"
                                          "def f():\n"
                                          "    return _aliased(seed=1)\n")
    _, found = scan([tmp_path])
    assert [(f[1], f[3]) for f in found] == [
        (symbol, "alias:_aliased")
    ], f"the import-aliased {symbol} construction was not reported: {found}"


def test_an_unimported_local_of_the_same_name_is_not_a_construction():
    """The other direction: aliases bind per FILE, from a real import, or not at all.

    Without this, "resolve aliases" could degrade into "flag any call whose name
    ever appears as an alias somewhere", which would put false positives into a
    scan whose whole value is that a red result means something. `_aliased` here
    is an ordinary local function; the file has no aliasing import, so nothing is
    flagged. The `as` form of an UNRELATED symbol must not bind either.
    """
    source = ("from train import build_env_factory as _aliased\n"
              "from train import make_puffer_env\n"
              "a = _aliased(seed=1)\n"
              "b = make_puffer_env(seed=1)\n")
    assert constructions(source) == [
        ("make_puffer_env", 4, "name")
    ], ("alias resolution is binding names it was never told about: "
        f"{constructions(source)}")


# ── guard 4: the carve-out, in both directions ──────────────────────────────


def test_classmethod_access_is_not_a_construction():
    """`SelfPlayManager.initial_hero_team()` is not flagged; constructions are.

    Both directions in one test, because each is worthless alone: a matcher that
    flags nothing passes the first half, and one that flags everything passes the
    second.
    """
    source = ("a = SelfPlayManager.initial_hero_team()\n"
              "b = SelfPlayManager.INITIAL_OPPONENT_TEAM\n"
              "c = train.SelfPlayManager.initial_hero_team()\n"
              "d = mgr.make_puffer_env_helper()\n"
              "e = SelfPlayManager(pool_size=15)\n"
              "f = train.SelfPlayManager(pool_size=15)\n"
              "g = make_puffer_env(seed=0)\n"
              "h = train.make_puffer_env(seed=0)\n")
    found = constructions(source)
    assert [(lineno, symbol, spelling) for symbol, lineno, spelling in found] == [
        (5, "SelfPlayManager", "name"),
        (6, "SelfPlayManager", "attribute"),
        (7, "make_puffer_env", "name"),
        (8, "make_puffer_env", "attribute"),
    ], f"carve-out is wrong: {found}"


def test_the_real_classmethod_call_sites_still_exist_and_are_not_flagged():
    """The carve-out is exercised by LIVE source, not only by the synthetic case.

    `train()` and `_build_trainer_for_test` both call
    `SelfPlayManager.initial_hero_team()` to build the participation vector
    before any manager exists. If those two calls were ever removed, the carve-out
    would still be "correct" and would be protecting nothing — and the synthetic
    test above would keep passing. This pins that the real sites are there and
    that the scan leaves them alone.
    """
    live = []
    for path in (REPO_ROOT / "src" / "train.py", REPO_ROOT / "src" / "train_test_harness.py"):
        tree = ast.parse(path.read_text())
        live += [(path.name, n.lineno) for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == "SelfPlayManager"]
    assert len(live) >= 2, (
        f"expected the two SelfPlayManager.<classmethod>() call sites, found {live}; the "
        "carve-out below is no longer protecting anything real")

    _, found = scan(ROOTS)
    flagged = {(p.name, lineno) for p, _, lineno, _ in found}
    assert not (flagged & set(live)), (
        f"the scan flagged classmethod access as a construction: {sorted(flagged & set(live))}")


# ── the enforcement assertion ───────────────────────────────────────────────


def test_no_make_puffer_env_or_selfplaymanager_call_outside_the_factory():
    """Zero `make_puffer_env(...)` / `SelfPlayManager(...)` in src/ or scripts/.

    THE TWO BANNED SYMBOLS AND NOTHING ELSE. Passing does not mean no env is
    built outside the factory: the twelve LOWER_LAYER_SITES calls to `make_env` /
    `Cs2Env` are in these same roots and are out of scope by design. Read the
    module docstring's LOWER LAYER paragraph before citing this test.

    Read a failure here as "a new construction site appeared", not as "this test
    is in the way". The fix is a role on `build_env_for` (or the manager
    builder), not an exemption — every exemption added here is a site that can
    drift again, which is the whole thing W3 removed.

    Pre-migration this scan reported EIGHT findings (five `make_puffer_env` in
    train.py, three `SelfPlayManager` across train.py and the harness) and zero
    attribute-spelled ones; that number is what calibrated the guards above.
    """
    _, found = scan(ROOTS)
    offenders = [(str(p.relative_to(REPO_ROOT)), symbol, lineno, spelling)
                 for p, symbol, lineno, spelling in found if p != FACTORY]
    assert not offenders, ("constructions found outside src/env_factory.py:\n" +
                           "\n".join(f"  {p}:{ln} {sym} ({sp})" for p, sym, ln, sp in offenders) +
                           "\nRoute them through build_env_for / build_selfplay_manager.")
