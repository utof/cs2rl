"""Guards on HOW the Modal runner is imported and packaged, not on what it does.

WHY: `scripts/` and the repo root are BOTH on sys.path in every FULL-SUITE
pytest process. The unit matters and the spec is careful about it (design doc
:147, "Every full-suite process therefore keeps both roots on `sys.path`"):
measured, collecting THIS FILE alone leaves `scripts/` off sys.path entirely --
only `src/` is there, inserted by `tests/conftest.py`. Of the 9 files that do put
it there, 8 insert at MODULE scope, so collecting any one of those is enough; the
9th (`tests/test_resume_state.py:352`) inserts inside a test body, so it lands
only when that test RUNS. A guard that only fires in a full-suite process is
still the right guard; a reader who thinks it fires everywhere is not.
Measured by AST census over every tracked .py, counting `sys.path.insert/append`
whose argument names `scripts` directly OR through a variable assigned from such
a path: **11** test files at 6c937ca, **9** after this commit converts two of
them. (Both numbers, because a docstring that states only the before-state is
wrong from the moment it lands. Two of the 11 -- test_modal_argv.py's
SCRIPTS_DIR and test_oracle_statue.py's -- are invisible to a census that only
looks for the literal string in the call.)
So `modal_runner_lib` and `scripts.modal_runner_lib` both resolve, and Python
caches them as TWO DISTINCT module objects. Measured: their `ValidationError`
classes are not identical, so `except mrl.ValidationError` does NOT catch a
ValidationError raised through the other spelling -- the exact class #166 spent
twelve commits routing error handling through.

PITFALL -- the one that decides whether this file works: the scan MUST walk the
whole tree (`ast.walk`), not `tree.body`. Until THIS COMMIT converted it,
`tests/test_eval_baselines.py` imported the bare spelling INSIDE a test body;
that real in-body violation is the case this guard was built around. A
`tree.body` scan reported that file clean while the defect was live, and still
passed a module-scope positive control. Past tense on purpose -- the file is
converted, so the live demonstration now lives in
`test_guard_detects_a_planted_bare_import`, which plants at three AST depths
(1, 2 and 3 nodes below `Module`, measured) across two scopes (module and
function body), in three shapes -- including the `importlib.import_module` shape
this repo's Modal tests use to reach their `scripts/` siblings.

HOW MEASUREMENTS ARE WRITTEN DOWN HERE, because getting this wrong has cost this
branch four correction commits: every mutation result below names the TEST THAT
OBJECTS, never a `1 failed, N passed` pair. The passed half is just this file's
collection minus the failures -- it says nothing the objector's name does not,
and it goes stale the moment a test is added, which Task 2 will do. So no
collection total appears anywhere in this file and nothing asserts on one. If
you need the number, `pytest --collect-only -q` has it and cannot be wrong.
"""
import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BARE = "modal_runner_lib"
# The one legal spelling. `BARE` is what must never appear; this is what must
# appear instead, and it is what the runtime probe expects to find alone in
# `sys.modules`.
PACKAGED = "scripts." + BARE

# The file set the RUNTIME gate below executes: every TEST file that imports the
# runner. Not a hand-picked sample -- `test_the_runtime_probe_runs_every_test_
# file_that_imports_the_runner` asserts this list EQUALS the set discovered by
# AST census, in both directions, so it cannot silently drift.
#
# Two of these were W1 conversions -- `test_modal_argv.py` (bare, module scope)
# and `test_eval_baselines.py` (bare, inside a test body), which between them
# cover both historic spellings and both depths. The other two were already on
# the packaged spelling at 6c937ca and are here because they import the runner,
# which is the only membership rule: `test_modal_runner.py` (the bulk of the
# runtime) and `test_modal_protocol.py`. Saying "the files W1 converted" would
# be wrong by two.
#
# Tracked NON-test importers, which are deliberately out of scope here because
# this list feeds a pytest session: `scripts/modal_artifacts.py:27`,
# `scripts/modal_backfill_sidecar.py:43`, `scripts/run_modal.py:33`. So SEVEN
# tracked files import the runner, four of them tests. Recorded for W3, which
# deletes `modal_runner_lib` and has to find every one of them.
#
# The BOUND that survives the scope control, stated rather than left to be
# discovered: the runtime gate only sees imports the session it spawns actually
# EXECUTES, and the census that feeds it is static. A dynamic import whose
# argument is a variable, sitting in a file that imports the runner no other
# way, is invisible to both halves. Neither guard closes that; together they
# make it the only remaining hole.
_IMPORTERS = [
    "tests/test_modal_argv.py",
    "tests/test_modal_protocol.py",
    "tests/test_modal_runner.py",
    "tests/test_eval_baselines.py",
]

# The test that carries the HARD case: the bare import used to live inside this
# body, second-to-last statement, behind four assertions (it is now the
# converted `import scripts.modal_runner_lib as mrl` at
# tests/test_eval_baselines.py:323, still in-body). If this test does not reach
# its end, the module census below is measuring a session that never executed
# the line the gate exists for.
_INBODY_CARRIER = ("tests/test_eval_baselines.py::test_eval_interval_cli_config_and_modal_mirror")


def _repo_python_files():
    """Every TRACKED .py file: `git ls-files --cached "*.py"`, and nothing else.

    Tracked-only is the scope the spec sets (W1: "Add an AST guard over every
    tracked `.py`"), and it is the right one. An earlier version added
    `--others --exclude-standard` to catch untracked files too; measured, that
    pulled in 7 extra .py living in `.ua/.trash-<n>/tmp/` -- a TRASH directory.
    A census that reads the trash makes the guard's verdict depend on what
    somebody happened to delete: drop a file containing a bare import into the
    trash and the guard reddens over code that is not in the repo. Tracked-only
    has no such coupling, and `.venv/` and `outputs/` fall out for free because
    gitignored files are never tracked -- that is what stops this reading
    thousands of vendored files.

    The accepted trade, stated so nobody mistakes it for coverage: a brand-new
    offender is invisible to this census until it is staged. That is survivable
    precisely because staging is not optional -- an unstaged file cannot be
    committed, so by the time a bare import is IN the repo it is tracked and
    this census sees it.

    Counts, with the unit, because a bare number invites the wrong comparison:
    tracked `.py` at this commit = 133 (the spec's 131 at 6c937ca, plus this
    file, plus Task 2's `tests/_modal_import_probe.py`). NOTHING asserts on
    that total and nothing should -- it moves with
    every added .py. `test_the_census_scans_the_whole_repo` pins the structural
    property instead, which does not move.

    `cwd=ROOT` is load-bearing: `git ls-files "*.py"` is CWD-RELATIVE, so
    running it from `tests/` returns 84 tracked paths instead of 133 and the
    guard silently stops watching `scripts/` and `src/`. Pinning the CWD is what
    makes the scope independent of where pytest was invoked from.
    `test_the_census_scans_the_whole_repo` covers a WRONG cwd, not a MISSING
    one, and the difference is measurable: repointing it at `tests/` turns that
    test red on assert 1 (and takes `test_no_file_imports_...` down with it, on
    FileNotFoundError), but DELETING `cwd=ROOT` leaves EVERY test in this file
    green when pytest runs from the repo root, because the subprocess then
    inherits a CWD that happens to be the right one. That deletion only bites
    once something runs pytest from elsewhere -- and then the same test does go
    red. Stated rather than left to read as full coverage.
    """
    out = subprocess.run(["git", "ls-files", "--cached", "*.py"],
                         cwd=ROOT,
                         capture_output=True,
                         text=True,
                         check=True)
    return [ROOT / line for line in sorted(set(out.stdout.splitlines())) if line]


def _bare_name(dotted, target=BARE):
    """True if `dotted` is `target` itself or a submodule of it.

    `target` defaults to `BARE`, so every pre-existing call site means exactly
    what it meant before this parameter existed.
    """
    return dotted == target or dotted.startswith(target + ".")


def bare_spelling_imports(source, target=BARE):
    """Every import of `target` in `source`, at ANY nesting depth.

    Returns a list of (lineno, kind). `ast.walk`, never `tree.body` -- see the
    module docstring.

    `target` defaults to `BARE`, which is the reason this function is
    parameterised AT ALL rather than copied: `test_the_runtime_probe_runs_every
    _test_file_that_imports_the_runner` needs the same three-shape,
    any-depth walk pointed at `PACKAGED` instead. A second walker is how the two
    halves drift apart -- one learns a new import shape and the other does not
    -- and this one already documents three shapes it must keep straight. The
    default keeps every existing caller and the whole mutation table below
    byte-identical in meaning.

    Covers three shapes, because this repo's Modal tests write all three:
    `import X`, `from X import ...`, and `importlib.import_module("X")` /
    `__import__("X")` with a STRING LITERAL argument -- positional OR `name=`.
    The keyword spelling is a real way back into the trap, not a theoretical
    one: measured on 3.12, `importlib.import_module(name="json")` AND
    `__import__(name="json")` both import successfully.

    Relative imports are NOT hits (`node.level` must be 0). `from
    .modal_runner_lib import X` names a DIFFERENT module -- a sibling of the
    importing package -- and `tests/` is a real package, so treating it as a hit
    would redden the guard over a legal import. Measured before the level check
    existed: both `from .modal_runner_lib import ValidationError` and the
    two-dot form returned `[(1, 'from-import')]`.

    PITFALL -- known blind spot, stated rather than hidden: a dynamic import
    whose argument is a variable (`importlib.import_module(M)`) is invisible to
    any static scan. `test_modal_runner_lib_resolves_to_exactly_one_module_object`
    is the runtime gate that covers it; that is why this repo has both and not
    just this one.
    """
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _bare_name(alias.name, target):
                    hits.append((node.lineno, "import"))
        elif isinstance(node, ast.ImportFrom):
            # level == 0 means absolute; see the relative-import note above.
            if node.level == 0 and _bare_name(node.module or "", target):
                hits.append((node.lineno, "from-import"))
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            # Both callables take the module name first positionally or as
            # `name=`; `arg` stays None for a no-argument call, and the
            # isinstance below rejects None without a separate branch.
            arg = node.args[0] if node.args else next(
                (kw.value for kw in node.keywords if kw.arg == "name"), None)
            if (name in ("import_module", "__import__") and isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str) and _bare_name(arg.value, target)):
                hits.append((node.lineno, "dynamic-import"))
    return hits


def _test_files_importing(target):
    """Tracked `tests/*.py` that import `target`, as repo-relative posix paths.

    Same census and same walker as the static guard -- `_repo_python_files()`
    and `bare_spelling_imports`, just pointed at a different module name. That
    reuse is the point: this function decides WHICH FILES the runtime probe
    runs, so if it and the guard disagreed about what "imports" means, the probe
    would be scoped by a rule nobody else in this file enforces.

    Scoped to `tests/` because the result feeds a pytest session. The three
    tracked non-test importers are listed at `_IMPORTERS`.

    FileNotFoundError is deliberately NOT caught, for the reason spelled out in
    `test_no_file_imports_the_bare_modal_runner_lib_spelling`: `--cached`
    enumerates the INDEX, so a tracked .py deleted without `git rm` lands here
    and raises. Loud and opaque beats a silent narrowing of the census.
    """
    found = set()
    for path in _repo_python_files():
        rel = path.relative_to(ROOT).as_posix()
        if not rel.startswith("tests/"):
            continue
        try:
            hits = bare_spelling_imports(path.read_text(encoding="utf-8"), target)
        except SyntaxError:
            # A .py that does not parse cannot import anything either; see the
            # identical branch in the static guard.
            continue
        if hits:
            found.add(rel)
    return found


def test_no_file_imports_the_bare_modal_runner_lib_spelling():
    """One spelling repo-wide, so one module object exists at run time."""
    offenders = {}
    for path in _repo_python_files():
        # `--cached` enumerates the INDEX, so a tracked .py deleted from the
        # working tree without `git rm` is still handed to us and `read_text`
        # raises FileNotFoundError. Measured (deleting `scripts/exp_lib.py`):
        # THIS test is the one that goes red, and it dies on a traceback out of
        # pathlib rather than on its own assertion message -- LOUD but opaque.
        # Nothing else in the file notices. Deliberately not caught: it can never
        # produce a false PASS, and a `try` here would let the census silently
        # stop covering a real file, which is the failure that matters. W2 and
        # Tasks 3-5 move files, which is when to expect it; `git mv` / `git rm`
        # keep the index consistent and it does not fire.
        try:
            hits = bare_spelling_imports(path.read_text(encoding="utf-8"))
        except SyntaxError:
            # A .py that does not parse cannot import anything either, so
            # skipping it is sound for THIS guard -- it is a scope statement,
            # not an excuse. Measured: 0 of the 133 tracked .py hit this branch,
            # so it is dead today; it is kept because a tracked .py that stops
            # parsing (a bad merge, a py313-only syntax) would otherwise redden
            # THIS guard for a reason that has nothing to do with imports.
            continue
        if hits:
            offenders[path.relative_to(ROOT).as_posix()] = hits
    assert offenders == {}, ("import `scripts.modal_runner_lib`, not `modal_runner_lib` -- both "
                             "spellings in one process create two module objects whose exception "
                             f"classes are not identical. Offenders: {offenders}")


@pytest.mark.parametrize("planted, shape", [
    ("import modal_runner_lib as mrl\n", "module scope"),
    ("from modal_runner_lib import ValidationError\n", "module-scope from-import"),
    ("def f():\n    import modal_runner_lib as mrl\n", "function body"),
    ("def f():\n    from modal_runner_lib import ValidationError\n", "function-body from-import"),
    ('import importlib\nm = importlib.import_module("modal_runner_lib")\n',
     "importlib.import_module literal"),
    ('def f():\n    return __import__("modal_runner_lib")\n', "__import__ literal in a body"),
    ('import importlib\nm = importlib.import_module(name="modal_runner_lib")\n',
     "importlib.import_module keyword"),
    ("import modal_runner_lib.state\n", "submodule"),
])
def test_guard_detects_a_planted_bare_import(planted, shape):
    """POSITIVE CONTROL at three AST depths, two scopes and three shapes.

    Each number with its unit, because "depth" on its own has two defensible
    readings and the wrong one oversells the control. Measured over these eight
    rows: the planted import sits 1, 2 or 3 nodes below `Module` (three DEPTHS)
    and in one of two SCOPES -- five at module scope (plain, from-import, both
    `importlib` rows, submodule), three in a function body -- in three SHAPES
    (`import`, `from-import`, `dynamic-import`).

    A `tree.body` implementation fails every row whose import sits deeper than
    ONE node below `Module`, and two of those are at MODULE scope: both
    `importlib` rows hide their `Call` inside an `Assign`, so `tree.body` never
    reaches them. Measured -- the failing set is the two function-body imports,
    the `__import__` row and both module-scope `importlib` rows; the survivors
    are exactly the depth-1 rows. So `tree.body` is blind to DEPTH, not to
    scope, which is what the old wording got wrong.

    A shape-blind implementation instead passes every depth and fails the
    dynamic-import rows -- the shape this repo's Modal tests use to reach their
    `scripts/` siblings (three sites at the time of writing; W2 splits the file
    that holds them, so the SHAPE is the durable citation and the filename is
    not). One control is not enough here; that is the whole point.
    """
    assert bare_spelling_imports(planted), f"guard is blind to {shape}"


@pytest.mark.parametrize("legal, why", [
    ("import scripts.modal_runner_lib as mrl\n", "the CORRECT spelling this task converts TO"),
    ("from .modal_runner_lib import ValidationError\n", "a relative import -- a different module"),
    ("import modal_runner_lib_extra\n", "a name that merely starts with the bare one"),
])
def test_the_guard_does_not_fire_on_legal_lookalikes(legal, why):
    """NEGATIVE CONTROL. A guard that reddens on legal code gets switched off.

    Each row was a live false positive or a near miss, not a hypothetical:

    - the relative form returned `[(1, 'from-import')]` until `node.level == 0`
      was added, and `tests/` is a real package where such an import is legal;
    - the `startswith` in `_bare_name` is `BARE + "."`, not `BARE`; drop the dot
      and `modal_runner_lib_extra` becomes a hit. This row is what objects;
    - the correct spelling must stay silent, or the guard fails the whole repo
      the moment W1's conversions land.
    """
    assert bare_spelling_imports(legal) == [], f"false positive on {why}"


def test_the_census_scans_the_whole_repo():
    """POSITIVE CONTROL for the guard's OWN SCOPE, which the row above cannot
    reach: every parametrized row calls `bare_spelling_imports` on a string and
    never touches `_repo_python_files`, the half that decides WHAT is read.

    WHY this exists: narrowing the pattern from `*.py` to `scripts/*.py` -- one
    token -- leaves every row above green, and at the time this landed both real
    violations lived in `tests/`. This repo's named #1 defect class is a guard
    blind to its own scope, in the file this plan calls its durable deliverable.

    Assert 1 names FILES, and it names all three W1 files rather than just one,
    because the one-name version was measured blind to the case that matters:
    filter the census to drop `tests/test_eval_baselines.py` -- the in-body
    violation this entire guard exists for -- and NOTHING objected: every test
    in the file passed, this one included. A census
    that has stopped enumerating the files W1 was written for has stopped doing
    its job, whatever else it still reaches.

    The asserts are not padding. Each is the FIRST objector to a different
    narrowing, measured by mutation against this exact argv -- so deleting any
    one of them silently retires a distinct check:

        *.py -> scripts/*.py       -> assert 1 (19 paths; all 3 names missing)
        cwd=ROOT -> cwd=ROOT/tests -> assert 1 (84 paths; ls-files is relative)
        census drops a W1 file     -> assert 1 (names it)
        *.py -> tests/*.py         -> assert 3 (asserts 1 and 2 both PASS)
        census drops THIS file     -> assert 2 (asserts 1 and 3 both PASS)

    Asserts 2 and 3 earn their place on the last two rows: a narrowing that
    keeps `tests/` sails past assert 1, and one that drops only this file sails
    past both 1 and 3.

    RE-MEASURED at Task 2, after it parameterised `bare_spelling_imports` --
    the first change to this file's behaviour since Task 1's final review proved
    that commit AST-identical. All five rows still map to the same assert, read
    off the actual `E AssertionError:` line rather than off marker strings
    (those also appear in pytest's traceback source listing, which makes rows 4
    and 5 look like they fire three asserts and two; they fire one each, the
    last one shown).

    What DID change is the objector SET on the first three rows, so the old
    parenthetical "nothing else fires" on row 3 is gone -- it is now false.
    `test_the_runtime_probe_runs_every_test_file_that_imports_the_runner` reads
    the same census, so any narrowing that hides a test file also shrinks the
    set it discovers and it objects too. Measured objectors: row 1 = this test +
    the probe-scope test; row 2 = those two plus
    `test_no_file_imports_the_bare_modal_runner_lib_spelling` (FileNotFoundError,
    as documented in `_repo_python_files`); row 3 = this test + the probe-scope
    test; rows 4 and 5 = this test alone. Two guards over one census is
    redundancy, not a defect -- but a docstring claiming exclusivity it no
    longer has is.

    MEASURED BOUND, in the same register as the `cwd=ROOT` gap documented in
    `_repo_python_files` -- a stated limit, not coverage. A filter that drops
    some OTHER single tracked file is still invisible: measured, dropping
    `tests/test_oracle_statue.py` and dropping `src/train.py` each leave the
    whole file green. Nothing closes that without re-deriving the census from the
    census, which would prove nothing. Per-file coverage stops at the three
    names below; the rest of the tree is covered at DIRECTORY granularity, by
    assert 3.
    """
    scanned = _repo_python_files()
    rel = {p.relative_to(ROOT).as_posix() for p in scanned}
    required = {
        "tests/test_modal_runner.py",
        "tests/test_eval_baselines.py",
        "tests/test_modal_argv.py",
    }
    missing = required - rel
    assert not missing, ("the census no longer reaches the files this guard was written for: "
                         f"{sorted(missing)}; the enumeration has been narrowed. "
                         f"Scanned {len(scanned)} paths.")
    assert Path(__file__).resolve() in scanned, (
        "the census cannot see its own file, so it cannot police itself; the "
        "enumeration has been narrowed.")
    tops = {r.split("/")[0] for r in rel}
    assert {
        "scripts", "src", "tests"
    } <= tops, (f"the census must reach scripts/, src/ and tests/; it reached {sorted(tops)}")


def test_the_runtime_probe_runs_every_test_file_that_imports_the_runner():
    """POSITIVE CONTROL for `_IMPORTERS`' OWN SCOPE -- the list that decides
    what the runtime gate below can see at all.

    WHY this exists, and it is this repo's named #1 defect class: `_IMPORTERS`
    was a hand-maintained list with nothing watching it. The gate below can only
    catch an import in a file it RUNS, so a name quietly dropped from this list
    retires coverage without reddening anything. One narrowing was already
    caught by accident -- empty the list of real importers and the census reads
    `[]`, which the gate's last assertion rejects -- but a PARTIAL narrowing was
    not: drop one of four and the census still reads exactly
    `["scripts.modal_runner_lib"]` from the survivors, and every assertion in
    the gate passes.

    Set EQUALITY, not `<=`, because the two directions fail differently and both
    are real:

    - a name MISSING from `_IMPORTERS` is lost coverage. Measured by dropping
      `tests/test_modal_protocol.py`: THIS test is the only objector -- the gate
      below stays green, which is the whole argument for this test existing.
    - a name in `_IMPORTERS` that imports NOTHING is a session paying for a file
      with no reason to be there, and usually a typo'd path that has silently
      stopped matching a real file. Measured by adding `tests/test_demo_format.py`:
      again THIS test is the only objector.

    CROSS-TASK WARNING, for whoever hits this in Task 4: W2 splits
    `tests/test_modal_runner.py` into a runner half and a client half. When the
    `scripts.modal_runner_lib` import travels to the new file, THIS TEST GOES
    RED, naming the new file as missing from `_IMPORTERS`. That is correct and
    deliberate -- it is the control doing its job. Add the new file to
    `_IMPORTERS`. Do NOT "fix" it by loosening this to a subset check or by
    deleting the moved name; either one hands W2 a silently narrowed probe,
    which is the exact failure this test was added to prevent.
    """
    discovered = _test_files_importing(PACKAGED)
    listed = set(_IMPORTERS)
    assert listed == discovered, (
        "`_IMPORTERS` no longer matches the test files that import "
        f"{PACKAGED}, so the runtime gate below is scoped to the wrong set. "
        f"Missing from _IMPORTERS (imports the runner but is never run, so a "
        f"dynamic bare import there is invisible): {sorted(discovered - listed)}. "
        f"Listed but imports nothing (dead entry or a path that stopped matching "
        f"a real file): {sorted(listed - discovered)}.")


def test_modal_runner_lib_resolves_to_exactly_one_module_object():
    """RUNTIME gate for the one-spelling rule.

    Runs a real pytest session over `_IMPORTERS` and reads `sys.modules` at
    session finish. The static guard above proves no file CONTAINS a bare import
    in a shape it knows; this proves none HAPPENS -- including through
    `importlib.import_module(<variable>)`, which no static census can see. Read
    `_IMPORTERS` for the matching bound: this covers the files it runs, not the
    repo, and `test_the_runtime_probe_runs_every_test_file_that_imports_the_runner`
    is what keeps that set honest.

    PITFALL 1: the second module object is created when test_eval_baselines'
    in-body import EXECUTES, so this must run the session, not import it.

    PITFALL 2 -- the one that made the first draft of this gate decorative:
    "the probe file was written" is not "the import ran". The in-body import is
    the second-to-last statement of its test; anything that fails that test
    earlier leaves the census clean and the defect live. So this asserts the
    inner session's EXIT CODE and that the carrier test itself reported passed,
    before it looks at the module names at all. Order matters: a bare module
    list is the least informative of the three failures. All three orderings
    were demonstrated red against a tree with the in-body bare import live --
    see the Task 2 report; the third control (carrier silenced with
    `@pytest.mark.skip`) is why the probe filters on `when == "call"`.

    IF YOU ARE HERE BECAUSE THIS TEST WENT RED, READ THIS FIRST. Assertion 1
    demands the nested session exit 0, and that session runs all 424 tests in
    the four `_IMPORTERS` files. So this test inherits the flakiness of every
    one of them, and reports it as "the probe session did not finish clean" --
    a headline pointing at the import machinery when the fault is very likely
    somewhere else entirely. The nested session's tail, including its `FAILED`
    line, is embedded in the assertion message: READ THAT LINE before you
    suspect anything about module objects or spellings. A known instance is
    gh#211 (`test_interrupt_without_publishable_checkpoint_writes_a_reason_file`,
    seen once, not reproducible on demand). Trading this away means dropping
    assertion 1, and control (a) in the Task 2 report shows exactly what that
    costs: the census reads clean while the bare import sits unreached.

    COST, because this is a planning fact and not a rounding error: this spawns
    a nested pytest session running all four `_IMPORTERS` files end to end, 424
    tests. Measured by `pytest --durations`, this test's `call` phase is **37s**
    against **0.78s** for the next slowest test in this file, so it is ~96% of
    the file's 38.06s. Run-to-run spread on this machine is 33-41s, wider than
    the 0.7s that adding `test_modal_protocol.py` cost -- so treat the figure as
    "half a minute", not as a number precise enough to regress against.

    Suite-wide, so nobody oversells it: it is the SECOND slowest test in the
    repo, behind
    `test_resume_state.py::test_subprocess_resume_run_flat_map_without_pin_pitch_flag`,
    and roughly 9% of the suite's wall clock. Measured by `--durations` on a full
    run, not estimated. Adding a file to `_IMPORTERS` adds that file's whole
    runtime here -- which is the price of the coverage, and the scope control
    above means the list grows only when a file genuinely starts importing the
    runner.
    """
    # Fork-bomb guard, and it is not theoretical now that a sibling test
    # mutates `_IMPORTERS` for its own controls: this file in `_IMPORTERS` makes
    # the nested session collect THIS test, which spawns another nested session,
    # forever -- one process per level, no natural bottom. The scope control
    # above rejects that entry too (this file imports no runner), but it cannot
    # save you here, because pytest runs tests in definition order within a file
    # and a `-k` selecting only this one skips it entirely. Cheap, local, first.
    assert Path(__file__).relative_to(ROOT).as_posix() not in _IMPORTERS, (
        "this file is in `_IMPORTERS`; the nested session would collect this "
        "test and recurse without bound")

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "probe.json")
        env = {**os.environ, "MODAL_IMPORT_PROBE_OUT": out}
        # --basetemp is NOT cosmetic. tests/conftest.py redirects basetemp to the
        # machine-global ~/.pytest_tmp unless --basetemp was passed explicitly,
        # and pytest rm_rf's the given basetemp before recreating it. Without
        # this flag the nested session deletes the OUTER session's temp tree
        # mid-run, and two pytest sessions anywhere on this box collide at
        # fixture setup with `FileExistsError: /home/<user>/.pytest_tmp`.
        #
        # `-o addopts=` and `-p no:randomly` are forward insurance, not load
        # bearing today: measured at this commit there is no
        # `[tool.pytest.ini_options]` in pyproject.toml, no pytest.ini / tox.ini
        # / setup.cfg, and pytest-randomly is not installed. Both are no-ops
        # now; they keep the nested session deterministic if either arrives.
        argv = [
            sys.executable, "-m", "pytest", *_IMPORTERS, "-q", "-o", "addopts=", "-p",
            "no:randomly", "-p", "tests._modal_import_probe", "--basetemp",
            os.path.join(tmp, "pt")
        ]
        result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
        # stderr is in every message below on purpose: when the nested session
        # dies before sessionfinish (bad plugin, collection crash) stdout is
        # empty and the traceback is on stderr, which is the one case where the
        # failure message is all a reader gets.
        tail = f"{result.stdout[-3000:]}\n--- stderr ---\n{result.stderr[-2000:]}"
        assert os.path.exists(out), (f"probe never ran; pytest exit={result.returncode}\n{tail}")
        payload = json.loads(Path(out).read_text(encoding="utf-8"))

    assert result.returncode == 0 and payload["exitstatus"] == 0, (
        "the probe session did not finish clean, so its module census describes "
        f"a run that may never have reached the import. exit={result.returncode} "
        f"sessionfinish={payload['exitstatus']}\n{tail}")
    assert _INBODY_CARRIER in payload["passed"], (
        f"{_INBODY_CARRIER} did not report passed, so the in-body import -- the "
        "only shape a static census cannot see -- was not executed. The module "
        f"census below is vacuous. passed={len(payload['passed'])} node ids")
    assert payload["modules"] == [
        "scripts.modal_runner_lib"
    ], ("modal_runner_lib resolved under more than one name, so two module "
        "objects exist and their exception classes are not identical: "
        f"{payload['modules']}")
