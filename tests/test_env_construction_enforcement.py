"""No `Cs2Env` or `SelfPlayManager` is constructed outside `src/env_factory.py`.

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
looked at it. Hence four separate guards ahead of the enforcement assertion:

  1. the roots exist, contain a plausible number of .py files, and contain named
     anchor files — kills "root resolved to nothing";
  2. the matcher finds all four real SPELLINGS in this repo's own source
     (`make_puffer_env(...)`, `train.make_puffer_env(...)`, `SelfPlayManager(...)`,
     `train.SelfPlayManager(...)`), which live in `tests/` and in the factory —
     kills "matcher names the wrong thing";
  3. a construction PLANTED in a scratch tree is found, in both spellings, in a
     nested subdirectory — kills "the walk or the parse is broken", end to end;
  4. the classmethod carve-out is checked in both directions, against real
     source — kills "the carve-out was loosened until it swallowed the coverage".

THE CARVE-OUT, precisely. A construction is `Call(func=Name(X))` or
`Call(func=Attribute(attr=X))` for X in {make_puffer_env, SelfPlayManager}.
`Call(func=Attribute(value=Name(X), attr=<other>))` is CLASSMETHOD ACCESS and is
NOT a construction: `SelfPlayManager.initial_hero_team()` is a live, legitimate
call in both `train()` and `_build_trainer_for_test`. Matching the ATTRIBUTE
position (`attr`) rather than the value position is the whole mechanism — a
naive value-position match flags those two sites, and the tempting fix is to
loosen the matcher until the real coverage dies, with nothing to notice.

WHY `src/env_factory.py` SHOWS ONE CONSTRUCTION AND NOT TWO. It builds the
manager directly (`SelfPlayManager(...)`), so the scan finds that. It does NOT
contain a `make_puffer_env(...)` call: `build_env_for` imports the function and
passes it to the role builder as a VALUE (`builder(_make, **kwargs)`), so no call
node in that file names it. That is why guard 2 takes its `make_puffer_env`
evidence from `tests/`, where ~40 direct constructions legitimately live, rather
than from the factory.
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


def constructions(source, label="<source>"):
    """[(symbol, lineno, spelling)] for every construction call in `source`.

    `spelling` is "name" for `X(...)` and "attribute" for `<anything>.X(...)`.
    Both count: `train.make_puffer_env(...)` builds exactly the same env as
    `make_puffer_env(...)`, and a scan that only looked for bare names would be
    trivially bypassed by an `import train` at the top of the new file.
    """
    out = []
    for node in ast.walk(ast.parse(source, filename=label)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in CONSTRUCTED:
            out.append((func.id, node.lineno, "name"))
        elif isinstance(func, ast.Attribute) and func.attr in CONSTRUCTED:
            # ATTRIBUTE position only. `SelfPlayManager.initial_hero_team()` has
            # attr="initial_hero_team" and is therefore not matched here — see
            # the module docstring, and test_classmethod_access_is_not_a_
            # construction, which pins that in both directions.
            out.append((func.attr, node.lineno, "attribute"))
    return out


def scan(roots):
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
            for symbol, lineno, spelling in constructions(path.read_text(), str(path)):
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


def test_no_direct_construction_outside_the_factory():
    """Zero `make_puffer_env(...)` / `SelfPlayManager(...)` in src/ or scripts/.

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
