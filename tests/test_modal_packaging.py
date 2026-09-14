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
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BARE = "modal_runner_lib"
# The one legal spelling. `BARE` is what must never appear; this is what must
# appear instead, and it is what the runtime probe expects to find alone in
# `sys.modules`.
PACKAGED = "scripts." + BARE

# The file set the RUNTIME gate below executes: every file UNDER `tests/` that
# imports the runner. Not a hand-picked sample -- `test_the_runtime_probe_runs_
# every_test_file_that_imports_the_runner` asserts this list EQUALS the set
# discovered by AST census, in both directions, so it cannot silently drift.
#
# "Under tests/" and not "every TEST file", which is what this sentence used to
# say: `_test_files_importing` scopes by DIRECTORY (`rel.startswith("tests/")`),
# and W2's shared-helper module is the case where the two readings diverge.
#
# Two of these were W1 conversions -- `test_modal_argv.py` (bare, module scope)
# and `test_eval_baselines.py` (bare, inside a test body), which between them
# cover both historic spellings and both depths. The other two were already on
# the packaged spelling at 6c937ca and are here because they import the runner,
# which is the only membership rule: `test_modal_runner.py` (the bulk of the
# runtime) and `test_modal_protocol.py`. Saying "the files W1 converted" would
# be wrong by two.
#
# `test_modal_client.py` and `modal_test_helpers.py` arrived with W2's split of
# `test_modal_runner.py`: the packaged import travelled into BOTH halves plus the
# shared module, so all three import the runner and the census discovers all
# three. `test_the_runtime_probe_runs_every_test_file_that_imports_the_runner`
# pre-registered this in its own docstring and told Task 4 to add the new file
# rather than loosen the equality to a subset. That is what happened here; the
# assertion was not touched.
#
# `modal_test_helpers.py` is NOT a test module -- no `test_` prefix, so pytest
# collects nothing from it -- and it belongs here anyway, because membership is
# "imports the runner AND lives under tests/", not "collects tests".
#
# WHAT THE ENTRY ACTUALLY BUYS, measured three ways, because the obvious answer
# is wrong and this comment asserted it for one draft. Handed that file ALONE, a
# session leaves `tests.modal_test_helpers` in `sys.modules` and the probe
# reports `{"modules": ["scripts.modal_runner_lib"], "passed": [],
# "exitstatus": 5}` -- so pytest does import a .py named explicitly on its
# command line even when the name does not match `python_files`, and the gate
# CAN see into this file. But run the real six-file session WITHOUT this entry
# and the module is in `sys.modules` regardless, because both halves carry
# `from tests.modal_test_helpers import (...)`. So today the entry extends the
# probe's reach by NOTHING. The honest reasons to list it are that the
# set-equality assertion requires it, and that it keeps the coverage if either
# half ever stops importing the shared module -- a forward guarantee, not a
# present-day gain. Cost is nil: 421 passed either way, 32.31s with against
# 33.17s without, inside the run-to-run spread.
#
# The `exitstatus: 5` above is NO_TESTS_COLLECTED and an artefact of running the
# file alone; alongside the five collecting entries the session exits 0. A list
# that ever consisted only of non-collecting modules would red assertion 1 of
# the gate for that reason and not for a spelling reason.
#
# THE PITFALL THAT LET THIS LIST GO STALE THROUGH A FULLY GREEN TASK, and it
# recurs for any task that ADDS a tracked file: `_repo_python_files` is
# `git ls-files --cached`, so the census cannot see a file until it is TRACKED.
# Every W2 gate -- the split's byte-identity census, its four red controls, the
# file-alone runs, and two full-suite runs at `1390 passed, 3 skipped` -- ran
# while both new files were still untracked, and the objecting test passed in
# every one of them because the files did not exist as far as it could look. It
# went red on the first run after the split commit, having been green minutes
# earlier on identical bytes. A green suite BEFORE a commit that adds files is
# not evidence about any tracked-only guard. Re-run them after.
#
# Tracked importers OUTSIDE `tests/`, which the directory scope above excludes
# because this list feeds a pytest session: `scripts/modal_artifacts.py:27`,
# `scripts/modal_backfill_sidecar.py:43`, `scripts/run_modal.py:33`. So NINE
# tracked files import the runner, six of them under `tests/` (was seven and
# four before W2's split). Recorded for W3, which deletes `modal_runner_lib` and
# has to find every one of them.
#
# BOUNDS that survive the scope control, stated rather than left to be
# discovered. The runtime gate only sees imports the session it spawns actually
# EXECUTES, and the census that feeds it is static, so a file can reach the
# runner in a way neither half sees. Two such ways are known and NAMED; this is
# not a claim that the list is complete, and an earlier revision of this comment
# calling the first one "the only remaining hole" was wrong on its own terms:
#
#   1. a dynamic import whose argument is a VARIABLE
#      (`importlib.import_module(M)`), in a file that imports the runner no
#      other way. Invisible to the static walker by construction.
#   2. an import written inside a subprocess CODE STRING. Invisible to the
#      walker, which sees a string, and to the runtime probe, which reads
#      `sys.modules` in the parent and never sees a child's. This repo writes
#      that shape in `test_local_entrypoints_do_not_import_modal` and
#      `test_modal_runner_lib_does_not_import_modal_or_torch` -- cited as
#      `tests/test_modal_runner.py:93` until W2's split moved both into THIS
#      file, which is a citation that rotted inside the commit that moved it.
#      Named rather than numbered now, because a line number citing the file it
#      lives in rots on the next edit. The bound is unchanged and if anything
#      firmer: this file is barred from `_IMPORTERS` by the fork-bomb guard, so
#      the probe never runs those two at all. Parsing string literals is
#      deliberately not attempted -- see `bare_spelling_imports`.
#
# A third was closed rather than stated: `from scripts import modal_runner_lib`
# used to return `[]`, which would have let Task 4's moved import escape the
# scope control entirely. It is now the `from-parent-import` shape.
_IMPORTERS = [
    "tests/test_modal_argv.py",
    "tests/test_modal_protocol.py",
    "tests/test_modal_runner.py",
    "tests/test_modal_client.py",
    "tests/modal_test_helpers.py",
    "tests/test_eval_baselines.py",
]

# The test that carries the HARD case: the bare import used to live inside this
# body, second-to-last statement, behind four assertions (it is now the
# converted `import scripts.modal_runner_lib as mrl` at
# tests/test_eval_baselines.py:323, still in-body). If this test does not reach
# its end, the module census below is measuring a session that never executed
# the line the gate exists for.
_INBODY_CARRIER = ("tests/test_eval_baselines.py::test_eval_interval_cli_config_and_modal_mirror")

# Wall-clock ceiling for the nested session, in seconds. Named rather than
# inlined so the value and the message that quotes it cannot drift apart -- a
# literal in both places is one edit away from a message that lies about its own
# threshold. Chosen from this test's measured spread, not picked round: 10 clean
# sessions ran 33.30-44.78s, so this is ~13x the slowest observed.
_NESTED_TIMEOUT_S = 600


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

    Covers FOUR shapes, because this repo's Modal tests write the first three
    and Task 4 is free to write the fourth: `import X`, `from X import ...`,
    `importlib.import_module("X")` / `__import__("X")` with a STRING LITERAL
    argument -- positional OR `name=` -- and `from <parent> import <leaf>`.
    The keyword spelling is a real way back into the trap, not a theoretical
    one: measured on 3.12, `importlib.import_module(name="json")` AND
    `__import__(name="json")` both import successfully.

    The FOURTH shape, `from scripts import modal_runner_lib`, only exists when
    `target` is DOTTED, and it is the one this walker was missing. The
    `from-import` branch above matches on the module PATH, which is right for a
    top-level target and incomplete for a dotted one, because the leaf can be
    imported from its parent. Measured against the walker before this branch
    existed, `from scripts import modal_runner_lib` returned `[]` under
    `PACKAGED` -- an ordinary idiom, invisible. That mattered because
    `test_the_runtime_probe_runs_every_test_file_that_imports_the_runner`
    promises Task 4 by name that it reddens when the import moves; had Task 4
    written this spelling, it would have stayed green and the runtime gate would
    have silently stopped covering the new file. W3 makes the idiom MORE likely
    by turning `modal_runner_lib` into a package.

    The branch is unreachable for a top-level target: `BARE.rpartition(".")`
    yields an empty parent, and `if parent` gates it. Proven rather than
    asserted -- an instrumented copy that raises on entry never fired across the
    whole tracked census under `BARE`, and did fire under `PACKAGED`.

    Relative imports are NOT hits (`node.level` must be 0). `from
    .modal_runner_lib import X` names a DIFFERENT module -- a sibling of the
    importing package -- and `tests/` is a real package, so treating it as a hit
    would redden the guard over a legal import. Measured before the level check
    existed: both `from .modal_runner_lib import ValidationError` and the
    two-dot form returned `[(1, 'from-import')]`.

    PITFALL -- known blind spots, named rather than hidden, and NOT claimed to
    be exhaustive:

    1. a dynamic import whose argument is a variable
       (`importlib.import_module(M)`) is invisible to any static scan.
       `test_modal_runner_lib_resolves_to_exactly_one_module_object` is the
       runtime gate that covers it; that is why this repo has both and not just
       this one.
    2. an import written inside a subprocess CODE STRING is invisible to this
       walker -- it reads the string as a string -- and to the runtime probe,
       which reads `sys.modules` in the parent process and never sees a child's.
       This repo writes that shape: `tests/test_modal_runner.py:93` holds
       `import scripts.modal_runner_lib` inside a code string, and the census
       correctly reports only line 51 for that file. A code string importing
       BOTH spellings would rebuild the two-module-object trap with neither
       guard objecting. Parsing string literals is deliberately NOT attempted:
       it is a false-positive generator, since a string that looks like code is
       not necessarily executed as code.
    """
    # Split once, outside the walk. For a top-level target `parent` is "", which
    # is what makes the `from <parent> import <leaf>` branch unreachable.
    parent, _, leaf = target.rpartition(".")
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
            elif node.level == 0 and parent and node.module == parent:
                # `from scripts import modal_runner_lib [as mrl]`. The alias is
                # irrelevant -- `alias.name` is the imported name and
                # `alias.asname` is only what it is bound to locally, so the
                # `as` spelling needs no separate case. Exact equality on the
                # leaf, never startswith: `from scripts import
                # modal_runner_lib_extra` is a different module.
                for alias in node.names:
                    if alias.name == leaf:
                        hits.append((node.lineno, "from-parent-import"))
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
    tracked importers OUTSIDE that scope -- all under `scripts/` -- are listed
    in the comment at `_IMPORTERS`, which documents them rather than holding
    them.

    THE COUNT IS UNCHANGED AND WAS NEVER WRONG. Re-measured at this commit:
    nine tracked files import the runner, six under `tests/` and three under
    `scripts/`. Only the wording moved, and each change is here because a real
    reader tripped on it. "listed at `_IMPORTERS`" was read as "a member of
    `_IMPORTERS`" and a finding was filed on the count before the re-measurement
    retracted it -- "documented at that site" versus "a member of that
    collection" is an ambiguity worth spending four words on. And W2 made
    `modal_test_helpers.py` a non-test file that imports the runner AND sits in
    the list, so the old phrase "non-test importers" stopped naming the set it
    meant; a reader counting them now gets four. Naming the scope beats naming
    the file kind, which is what survives the next module that is neither.

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
    ("from scripts import modal_runner_lib\n", "the CORRECT spelling, imported from its parent"),
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
      the moment W1's conversions land -- in BOTH its spellings, which is why
      `from scripts import modal_runner_lib` is here. That row is the BARE half
      of the fourth shape: it must be a HIT under `PACKAGED` (the row of the
      same name in `test_the_walker_finds_the_packaged_spelling_in_every_shape`)
      and SILENT under `BARE`, and one direction without the other is what let
      the shape go missing in the first place.
    """
    assert bare_spelling_imports(legal) == [], f"false positive on {why}"


@pytest.mark.parametrize("planted, shape", [
    ("import scripts.modal_runner_lib\n", "module scope"),
    ("import scripts.modal_runner_lib as mrl\n", "module scope, aliased"),
    ("from scripts.modal_runner_lib import ValidationError\n", "from-import on the full path"),
    ("def f():\n    import scripts.modal_runner_lib as mrl\n", "function body"),
    ('import importlib\nm = importlib.import_module("scripts.modal_runner_lib")\n',
     "importlib.import_module literal"),
    ("import scripts.modal_runner_lib.state\n", "submodule"),
    ("from scripts import modal_runner_lib\n", "from-parent-import -- THE SHAPE THAT WAS MISSING"),
    ("from scripts import modal_runner_lib as mrl\n", "from-parent-import, aliased"),
    ("from scripts import modal_artifacts, modal_runner_lib\n",
     "from-parent-import, one of several names"),
    ("def f():\n    from scripts import modal_runner_lib\n", "from-parent-import in a body"),
])
def test_the_walker_finds_the_packaged_spelling_in_every_shape(planted, shape):
    """POSITIVE CONTROL for the walker aimed at `PACKAGED` -- the half that
    decides which files reach `_IMPORTERS`, and which had NO shape-level control
    before this round.

    Every row above `test_guard_detects_a_planted_bare_import` covers the `BARE`
    target only. `_test_files_importing(PACKAGED)` calls the same walker with a
    DOTTED target, where one shape behaves differently: the leaf can be imported
    from its parent. Review found `from scripts import modal_runner_lib`
    returning `[]`, so the file that Task 4 is warned about could have moved its
    import into that idiom and never entered `_IMPORTERS`.

    Each row also asserts SILENCE under `BARE`. That is not padding -- it is the
    inertness half. A fix that made the new branch fire for a top-level target
    would redden `test_no_file_imports_the_bare_modal_runner_lib_spelling` over
    the legal packaged spelling, i.e. over the entire repo after W1.
    """
    assert bare_spelling_imports(planted, PACKAGED), f"walker is blind to {shape} under PACKAGED"
    bare_hits = bare_spelling_imports(planted, BARE)
    assert bare_hits == [], (
        f"{shape} matched the BARE target; the packaged spelling is LEGAL and this "
        "would redden the repo-wide guard over correct code")


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
    demands the nested session exit 0, and that session runs every test in the
    four `_IMPORTERS` files. So this test inherits the flakiness of every one
    of them, and reports it as "the probe session did not finish clean" --
    a headline pointing at the import machinery when the fault is very likely
    somewhere else entirely. The nested session's tail, including its `FAILED`
    line, is embedded in the assertion message: READ THAT LINE before you
    suspect anything about module objects or spellings. gh#211 records what is
    known: 2 spurious failures across ~20 nested sessions, one node id captured
    (`test_interrupt_without_publishable_checkpoint_writes_a_reason_file`),
    cause NOT VERIFIED and not reproducible on demand.

    AND YET ASSERTION 1 MUST STAY, so before you delete it, reproduce this. Let
    an unrelated test in an `_IMPORTERS` file fail while the carrier still
    passes -- exactly the gh#211 shape -- and neuter assertion 1. Measured, the
    gate goes GREEN: assertions 2 and 3 both pass, because the carrier DID run
    and the census IS clean. Assertion 1 is the sole objector to a session that
    failed for an unrelated reason, and without it this test cannot tell a
    healthy run from a broken one.

    Do NOT reach for control (a) here, which an earlier revision of this
    paragraph cited and which does not show this. In control (a) the CARRIER
    is what fails, so it never reaches `_PASSED` and assertion 2 catches the
    case on its own -- measured, with assertion 1 neutered, control (a) still
    fails on assertion 2. A reader following that citation would watch
    assertion 2 cover for assertion 1 and conclude assertion 1 is redundant,
    which is the deletion this paragraph exists to prevent. Control (a) shows
    what a CENSUS-ONLY gate costs, which is a different and also real result.

    COST, because this is a planning fact and not a rounding error: this spawns
    a nested pytest session running all four `_IMPORTERS` files end to end.
    Measured by `pytest --durations`, this test's `call` phase is **37s**
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
    # Entries are resolved before comparing, not string-matched: `./tests/x.py`,
    # `tests//x.py` and an absolute path all name this file while comparing
    # unequal as strings, and `ROOT / <absolute>` yields the absolute path
    # unchanged, so one expression covers relative and absolute alike. The scope
    # control above would reject any of them anyway; this guard exists for the
    # `-k`-selects-only-this-test case, where it is the only thing standing
    # between a typo and an unbounded fork.
    _self = Path(__file__).resolve()
    assert not any((ROOT / entry).resolve() == _self for entry in _IMPORTERS), (
        "this file is in `_IMPORTERS`; the nested session would collect this "
        f"test and recurse without bound. _IMPORTERS={_IMPORTERS}")

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
        # timeout= so a hung child cannot take the whole outer suite down with no
        # diagnostic; the value and its rationale live at `_NESTED_TIMEOUT_S`.
        # In-repo precedent is `tests/test_resume_state.py`, which passes
        # timeout=1500/600 on its subprocess runs.
        try:
            result = subprocess.run(argv,
                                    cwd=ROOT,
                                    env=env,
                                    capture_output=True,
                                    text=True,
                                    timeout=_NESTED_TIMEOUT_S)
        except subprocess.TimeoutExpired as expired:
            # TimeoutExpired carries whatever was captured before the kill.
            # BOTH streams, and stderr is the one that matters here: a hang
            # caused by a bad plugin or a collection crash writes to stderr and
            # nothing to stdout, which is precisely the case this branch exists
            # to report. An earlier revision printed only `.stdout` and would
            # have shown an empty tail for it -- the same gap the `tail =` line
            # below was written to close, reproduced in the handler four lines
            # above it. Each stream is repr'd separately because `text=True`
            # does not guarantee str here; `or b""` covers the None case and the
            # reprs read correctly either way.
            partial_out = (expired.stdout or b"")[-3000:]
            partial_err = (expired.stderr or b"")[-2000:]
            raise AssertionError(
                f"the probe session did not finish within {_NESTED_TIMEOUT_S}s, so it hung "
                "rather than failed. This is not a module-spelling problem; look at what "
                f"the nested session was doing.\n{partial_out!r}\n"
                f"--- stderr ---\n{partial_err!r}") from expired
        # stderr is in every message below on purpose: when the nested session
        # dies before sessionfinish (bad plugin, collection crash) stdout is
        # empty and the traceback is on stderr, which is the one case where the
        # failure message is all a reader gets.
        tail = f"{result.stdout[-3000:]}\n--- stderr ---\n{result.stderr[-2000:]}"
        assert os.path.exists(out), (f"probe never ran; pytest exit={result.returncode}\n{tail}")
        # A session killed mid-write leaves a file that passes the exists check
        # and then blows up in the decoder. Caught so the tail -- the only thing
        # that says WHY -- survives; an unguarded JSONDecodeError throws it away.
        raw = Path(out).read_text(encoding="utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as bad:
            raise AssertionError(
                "the probe wrote its output file but it is not valid JSON, so the session "
                f"was killed mid-write. exit={result.returncode} bytes={len(raw)} "
                f"{bad}\nfile starts: {raw[:200]!r}\n{tail}") from bad

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


# ── The modal test seam: concern, recomputed from source ──────────────────────
#
# WHY this section exists at all. `tests/test_modal_runner.py` held two suites.
# Splitting it needs an answer to "which half does this name belong to?" that a
# checked-in test can RE-DERIVE, because the alternative -- freeze a manifest,
# then check the files agree with the manifest -- is green on a maximally wrong
# split: review shuffled every name by even/odd index, split the file to match
# its own shuffled manifest, and the consistency check passed. A manifest is a
# record, not evidence. `classify_seam` is the evidence.

MANIFEST = ROOT / "tests" / "fixtures" / "modal_test_seam_manifest.json"

RUNNER_FILE = "tests/test_modal_runner.py"
CLIENT_FILE = "tests/test_modal_client.py"
PACKAGING_FILE = "tests/test_modal_packaging.py"

# THE MEMBERSHIP RULE FOR THE SHARED FILE. Task 4 creates
# `tests/modal_test_helpers.py` and must carry these words into that module's own
# docstring; until it exists, this is the only place the rule can live.
#
# A name earns a place in the shared file by being REACHED FROM BOTH SIDES of the
# seam. Nothing else earns it. A helper only runner tests reach belongs in the
# runner file, a helper only client tests reach belongs in the client file --
# however generic the helper looks, and however well its name would read here.
# `classify_seam` computes exactly this rule and will not send a one-sided helper
# to this file, so the rule is enforced rather than merely stated.
#
# WHY write down a rule the classifier already computes. Spec §10 criterion 12
# bans a module named `utils` / `helpers` / `common` / `misc`. Its instrument is
# the per-module segment sums of §5.1 -- the eight `scripts/modal_runner/`
# submodules of W3 -- so a test module is outside its scope and there is no
# conflict here. But the criterion exists because a module named for what it IS
# rather than for what it OWNS becomes a junk drawer, and that failure mode does
# not care which directory it happens in. A one-line membership test is what
# keeps this file a seam artefact instead of a drawer: measured, it holds exactly
# 7 names, and every one of them is reached from both halves.
SHARED_FILE = "tests/modal_test_helpers.py"

# The three module-level guards that lived above line 100 of the monolith. They
# are about packaging and dependencies, so they belong to neither half of the
# seam; they are named rather than computed because "is a packaging guard" is a
# judgement about subject matter that no reference graph encodes. Measured: the
# monolith's only three top-level `test_` functions defined above line 100 are
# exactly these, which is the spec's §2.3 "three tests ... belong to neither
# half" recomputed rather than copied.
SEAM_GUARDS = frozenset({
    "test_modal_is_an_explicit_dependency_group",
    "test_local_entrypoints_do_not_import_modal",
    "test_modal_runner_lib_does_not_import_modal_or_torch",
})

# Names the seam does not govern, with the reason.
#
# `ROOT` is `Path(__file__).resolve().parents[1]` -- module-header boilerplate
# that every destination file defines for itself by construction. Measured at
# this commit, by AST over module-level bindings of every collected
# `tests/test_*.py`: 5 files define a `ROOT` of their own (`test_modal_argv.py`,
# `test_modal_packaging.py`, `test_modal_protocol.py`, `test_modal_runner.py`,
# `test_train_loop_timing.py`), and `ROOT` is the ONLY one of the monolith's 261
# module-level names that any other collected test file also defines -- so this
# exemption list is one name long because the collision set is one name long,
# not because the rest were not looked for. Treating `ROOT` as a shared helper
# would make the seam's own scan collide with those unrelated files.
#
# STATED RATHER THAN HIDDEN: an exemption nothing checks is a hole. Task 4's
# placement gate is what closes it in both directions, by asserting `ROOT` IS
# defined in every destination file, so "not governed" cannot quietly become
# "lost". Until that gate lands, this is an unguarded exemption.
SEAM_HEADER_NAMES = frozenset({"ROOT"})

# A name whose own body reaches the client modules. These two sets are the
# spec's own §2.3 instrument, member for member.
#
# `module` is here because the monolith's client tests bind
# `module = _import_run_modal()` and then talk to `module`: measured, 13 of the
# 54 client tests carry that signal and NO other in their own body. Deleting it
# still moves 0 names, because the transitive closure covers the same 13 -- see
# `classify_seam`'s stage-1 note. Every member of both sets has a dedicated case
# in `_SEAM_CLASSIFIER_PROBE`, because before those cases existed, deleting
# `"module"` was measured to leave the whole file green.
_CLIENT_BINDINGS = frozenset({"_import_run_modal", "_run_modal_image_reqs", "module"})
_CLIENT_MODULES = ("scripts.run_modal", "scripts.modal_artifacts", "scripts.modal_backfill_sidecar")

_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _module_level_names(tree):
    """Every module-level binding in `tree`, as {name: defining node}.

    Covers `Assign`/`AnnAssign` targets as well as defs and classes. Dropping
    constants is not a simplification: measured, the monolith has 6 module-level
    assignments, and knocking the `Assign` branch out leaves 5 of those names
    with NO destination at all -- `PINNED_CUDA_CHILD_DIGEST`, `PINNED_CUDA_IMAGE`,
    `PINNED_PUFFERLIB_SDIST` and `PROTOCOL_TOKENS`, all four of which classify to
    the client half, plus `_FORBIDDEN_FLAGS`, which classifies to the runner
    half. (The 6th is `ROOT`, which `SEAM_HEADER_NAMES` excludes on purpose.) A
    census that walks defs alone leaves those five unassigned, and a pinned CUDA
    digest copied into both files can then drift apart with every check in this
    file green.
    """
    found = {}
    for node in tree.body:
        if isinstance(node, _DEFS):
            found[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        found[sub.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found[node.target.id] = node
    return found


def _referenced_module_names(node, own, universe):
    """Module-level names `node` references, decorators included.

    An `ast.Attribute` is resolved to its ROOT name (`FakeModal.spec` ->
    `FakeModal`), which is what makes attribute-spelled reaches visible. Local
    shadowing is deliberately NOT modelled: over-reporting an edge can only
    merge two names into one destination, while missing one strands a helper --
    the failure that is a NameError at run time.
    """
    out = set()
    for sub in ast.walk(node):
        target = None
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            target = sub.id
        elif isinstance(sub, ast.Attribute):
            root = sub
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                target = root.id
        if target and target != own and target in universe:
            out.add(target)
    return out


def _reaches_client_directly(node, own):
    """Seed test: does this node's OWN body name a client module?"""
    if own in _CLIENT_BINDINGS:
        return True
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in _CLIENT_BINDINGS:
            return True
        if isinstance(sub, ast.Attribute):
            root = sub
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in _CLIENT_BINDINGS:
                return True
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if any(m in sub.value for m in _CLIENT_MODULES):
                return True
    return False


def classify_seam(sources):
    """Compute each module-level name's destination file from the code alone.

    `sources` is {relative path: source text}. Returns
    `(destinations, defined_in)`: {name: destination path} and
    {name: [files that define it]}. Works on the monolith (one entry) and on the
    split (several), because it treats the union of the files as one namespace
    -- which is exactly why it can be re-run after the split and still be
    evidence rather than a tautology.

    TWO STAGES, and the split between them is the whole design:

    1. TESTS are classified by transitive closure. A name is CLIENT if its own
       body names a client module, or if it references -- transitively -- a name
       that does. Measured against the seam the spec measured four ways, 0 of
       194 tests land on the wrong side.

       WHICH PART OF STAGE 1 EARNS THAT ZERO, because a one-at-a-time census
       gets this backwards. Knocked out singly, on the monolith: the closure
       moves 0 names, `"module"` moves 0, the `_CLIENT_MODULES` string seed
       moves 0, and attribute-root resolution moves 0. Knock out the closure
       AND `"module"` together and 30 names move, 13 of them tests -- the two
       cover the same 13 client tests, so each looks like dead weight until the
       other is gone. `_run_modal_image_reqs` is the one seed that is
       load-bearing alone (5 names). The string seed and attribute-root
       resolution move 0 even jointly with the closure disabled: they are
       over-coverage, kept because an extra edge can only merge two names into
       one destination while a missing one strands a helper.

    2. HELPERS follow the tests that reach them, because a helper has no concern
       of its own -- it has its consumers'. `FakeModal` names nothing
       client-specific; it is client because only client tests reach it. This
       stage is also the only one that can return a THIRD answer, and it does:
       measured, 7 names are reached from BOTH halves and go to a shared module.
       A classifier forced to pick a half for those 7 would strand them, which
       is a NameError at run time in whichever file lost.

    HONESTY NOTE, because the bar was known before stage 2 was written: stage 1
    alone puts 21 names on the wrong side -- all of them non-test helpers, 0 of
    them tests -- and stage 2 was added afterwards, with the target already
    known. Tuning a classifier until it matches a number you already have
    certifies it against the answer rather than against the source. The reasons
    to believe stage 2 anyway are that it is a different KIND of rule rather
    than a longer marker list, and that it produced a finding nobody had -- the
    7 shared names -- which a second instrument (a cross-seam reference census
    that uses LINE POSITION, not this closure, as each test's concern)
    reproduces exactly. Bound on that word "second": it shares
    `_module_level_names` and `_referenced_module_names` with this function, so
    it is independent of the seed and the closure but NOT of the reference
    graph. A bug in edge extraction would be invisible to both.

    KNOWN LIMIT, stated rather than hidden: a name reached by no test at all is
    unclassifiable by stage 2 and raises. Measured on the monolith: 0 such
    names. If one appears it is either dead code or a new entry point, and both
    deserve a human, not a default.

    SECOND KNOWN LIMIT: this reads the AST, so a reference written inside a
    subprocess `code = \"\"\"...\"\"\"` string is invisible to it, exactly as it is to
    `bare_spelling_imports` above. Measured, `tests/test_modal_runner.py` has
    two such blocks (`test_local_entrypoints_do_not_import_modal` and
    `test_modal_runner_lib_does_not_import_modal_or_torch`), both of them inside
    `SEAM_GUARDS` tests that are assigned by fiat anyway, and both naming only
    modules outside this file. Measured the other way too: of the monolith's
    module-level names, exactly 3 appear as a word inside any multiline string
    literal in the file -- `ROOT`, which is ungoverned, and two test names cited
    by a docstring at line 574, which is prose and not a reach. So this limit
    changes no destination today. A future subprocess block could.
    """
    trees = {rel: ast.parse(text) for rel, text in sources.items()}
    defined_in, nodes = {}, {}
    for rel, tree in trees.items():
        for name, node in _module_level_names(tree).items():
            defined_in.setdefault(name, []).append(rel)
            nodes[name] = node

    universe = set(nodes)
    edges = {n: _referenced_module_names(node, n, universe) for n, node in nodes.items()}
    tests = {
        n
        for n, node in nodes.items()
        if n.startswith("test_") and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    client = {n for n, node in nodes.items() if _reaches_client_directly(node, n)}
    changed = True
    while changed:                     # stage 1: reach-a-seed, to a fixed point
        changed = False
        for name, targets in edges.items():
            if name not in client and (targets & client):
                client.add(name)
                changed = True

    reached_by = {n: set() for n in nodes}
    for test in tests:                 # stage 2: which tests reach each helper
        seen, stack = set(), [test]
        while stack:
            for nxt in edges[stack.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        for name in seen:
            reached_by[name].add(test)

    destinations = {name: PACKAGING_FILE for name in SEAM_GUARDS}
    for name in nodes:
        if name in SEAM_HEADER_NAMES or name in SEAM_GUARDS:
            continue
        if name in tests:
            destinations[name] = CLIENT_FILE if name in client else RUNNER_FILE
        else:
            consumers = reached_by[name]
            if not consumers:
                raise ValueError(f"{name!r} is reached by no test, so stage 2 cannot place it; "
                                 "it is dead code or a new entry point and needs a human")
            if consumers & client and consumers - client:
                destinations[name] = SHARED_FILE
            else:
                destinations[name] = CLIENT_FILE if consumers & client else RUNNER_FILE
    return destinations, defined_in


def _seam_sources():
    """The declared destination files that exist right now, as {rel: text}.

    `tests/test_modal_packaging.py` is deliberately NOT in the list even though
    it is a destination. It holds this classifier, whose own body names the
    client modules in a string constant, so feeding the file to the graph seeds
    the classifier itself as a client test. Measured at this commit: fed in,
    `classify_seam` returns 296 destinations rather than 260, and of the 36
    names this file contributes, 11 go to `tests/test_modal_client.py` --
    `classify_seam` and `_reaches_client_directly` among them -- while 18 go to
    the runner half and 7 to the shared module. Every one of those is nonsense.
    A census of how test files import the runner has no side of this seam.

    SCOPE OF THE DAMAGE, measured, because it is smaller than it sounds and this
    exclusion should not be over-trusted: feeding this file in changes the
    destination of **0** monolith names. The self-poisoning is confined to this
    file's own names. The three names this file legitimately receives are
    declared in `SEAM_GUARDS` and assigned unconditionally, so nothing is lost
    by leaving it out.

    A CLAIM RETIRED HERE RATHER THAN REWRITTEN, because matching prose to code is
    only right when the code is right. A previous revision of this paragraph said
    that feeding this file in RAISES -- that `_names_defined_under_tests` was
    reached by no test, so the known-limit raise fired on it. It no longer does,
    and the reason matters more than the sentence did: that raise was never a
    property of this exclusion or of the classifier. Measured -- hand `c1f7a7a`'s
    copy of this file to TODAY's classifier and it still raises, on that same
    name. The trip-wire was only ever the fact that `_names_defined_under_tests`
    had no caller, which was itself the defect review filed; giving it one
    removed the trip-wire as a side effect. Nothing load-bearing went with it.
    The orphan raise is untouched and still reachable -- see `classify_seam`'s
    KNOWN LIMIT and the `pytest.raises` closing
    `test_the_seam_classifier_places_a_planted_name_by_its_reference_graph`. What
    protects the seam here is the `rels` list this function ends with, which omits
    `PACKAGING_FILE` unconditionally and never depended on the raise at all.

    Existence-tolerant on purpose: `tests/test_modal_client.py` and
    `tests/modal_test_helpers.py` do not exist until the split lands, and
    `classify_seam` computes from CONTENT, not from where content lives, so the
    manifest it produces over the unsplit monolith is the same manifest it
    produces over the split files. That is what lets the agreement test below be
    green on both sides of the split instead of shipping red for one task.
    """
    rels = [RUNNER_FILE, CLIENT_FILE, SHARED_FILE]
    return {rel: (ROOT / rel).read_text(encoding="utf-8") for rel in rels if (ROOT / rel).exists()}


def _names_defined_under_tests():
    """{name: [files]} for every module-level name in the files the seam governs
    plus every OTHER collected modal test file.

    The glob is the point. Review moved one test into a brand-new
    `tests/test_modal_stray.py` and the first draft of this gate stayed green,
    because it only ever looked at files the manifest itself names. A
    destination the manifest does not know about has to be reachable, or the
    check is asking the suspect for the list of places to search.

    CALLED BY `test_no_governed_name_is_defined_outside_the_seams_own_files`, and
    the reason that matters is a second review finding: for one round this
    function shipped with no caller in the committed suite at all, so the same
    stray-file plant still left the file green and the only certification was a
    gitignored script. Task 4's placement gate is the other caller. If you are
    about to remove the last caller, delete the function with it.

    Scope is `tests/test_*.py` plus the shared helper module: a stray file that
    pytest never collects is not the threat, and measured, widening past
    `test_*.py` pulls in `tests/capture_dump_config_pre_165.py` and
    `tests/capture_env_config_pre_165b.py`, which each define a `_git` of their
    own and would make a gate built on this red for an unrelated reason.

    CONTRACT for the caller: the return value is a SUPERSET of the seam. It
    covers every `tests/test_*.py` in the repo, not just the four destination
    files, which is exactly what makes a fourth file visible -- and exactly why
    a caller comparing it against the manifest must scope the disk-to-manifest
    direction to names it actually governs rather than flagging every unrelated
    test file's helpers.
    """
    found = {}
    paths = sorted(set((ROOT / "tests").glob("test_*.py")) | {ROOT / SHARED_FILE})
    for path in paths:
        if not path.exists():
            continue
        rel = path.relative_to(ROOT).as_posix()
        for name in _module_level_names(ast.parse(path.read_text(encoding="utf-8"))):
            found.setdefault(name, []).append(rel)
    return found


# A synthetic module that exercises all four answers `classify_seam` can give,
# with the answers known by construction rather than measured off the monolith.
# WHY a synthetic probe and not just the real file: the real file certifies the
# classifier against a seam we already knew, which is the weakest kind of
# evidence there is. This certifies the RULE. It is also the only part of this
# section that survives Task 4 unchanged -- the "0 names on the wrong side of
# line 3638" check that certified the classifier against the monolith is a
# one-off by construction, because after the split there is no line 3638 to
# measure against. That check's successor is Task 4's placement gate, which
# swaps position for which-file as the ground truth.
#
# `_import_run_modal` and `_import_backfill` are deliberately NOT defined here:
# a seed does not have to be a module-level binding to seed, and leaving them
# undefined keeps them out of `universe`, so the only signal reaching the
# classifier is the one each probe test is named for.
#
# THERE IS ONE PROBE TEST PER MEMBER of `_CLIENT_BINDINGS` and `_CLIENT_MODULES`,
# and that is not thoroughness for its own sake. A guard whose own allow-set is
# unwatched is this repo's most-repeated defect, and it bit here: the first
# version of this probe named `_import_run_modal` and `scripts.modal_artifacts`
# and nothing else, and deleting `"module"` from `_CLIENT_BINDINGS` was measured
# to leave BOTH tests in this section green.
#
# AND ONE PER MODULE-LEVEL NODE KIND, for the same reason one level down. Review
# measured that this probe parsed to `['FunctionDef']` and nothing else -- no
# `ClassDef`, no `Assign`/`AnnAssign`, no `SEAM_HEADER_NAMES` name -- and ran
# four mutations across three of `classify_seam`'s decision surfaces that the
# probe therefore could not see: dropping `ast.ClassDef` from `_DEFS` (14 classes
# lose a destination), dropping the `Assign` census (5 names lose one), emptying
# `SEAM_HEADER_NAMES`, and adding a governed name to it. Each was caught by the
# MANIFEST AGREEMENT test ALONE -- and that test's own docstring concedes it is
# green on a coordinated edit that regenerates the manifest in the same commit.
# So those surfaces were guarded only by the instrument such an edit rides in on.
# Two more were added here for the same reason review found the first three:
# `AnnAssign` is a separate branch from `Assign`, and `SEAM_GUARDS` had an
# assertion that could not fail.
_SEAM_CLASSIFIER_PROBE = '''
ROOT = "module-header boilerplate: SEAM_HEADER_NAMES, governed by nobody"

_PROBE_PINNED_DIGEST = "sha256:0000"

_PROBE_TIMEOUT_S: int = 5


class _ProbeSharedDouble:
    """A module-level CLASS, reached from both halves."""


def _shared_helper():
    return 1

def _runner_only_helper():
    return _shared_helper(), _PROBE_TIMEOUT_S

def _client_only_helper():
    module = _import_run_modal()
    return module, _PROBE_PINNED_DIGEST

def test_probe_runner_via_helper():
    return _runner_only_helper()

def test_probe_runner_via_shared():
    return _shared_helper(), _ProbeSharedDouble(), ROOT

def test_probe_client_direct():
    return _import_run_modal()

def test_probe_client_by_image_reqs():
    return _run_modal_image_reqs("lock")

def test_probe_client_by_module_binding():
    module = _import_backfill()
    return module.App

def test_probe_client_transitive():
    return _client_only_helper()

def test_probe_client_via_shared():
    return _client_only_helper(), _shared_helper(), _ProbeSharedDouble()

def test_probe_client_by_artifacts_string():
    return "scripts.modal_artifacts"

def test_probe_client_by_run_modal_string():
    return "import scripts.run_modal as rm"

def test_probe_client_by_sidecar_string():
    return "scripts.modal_backfill_sidecar"

def test_modal_is_an_explicit_dependency_group():
    return _import_run_modal()
'''


def test_the_seam_classifier_places_a_planted_name_by_its_reference_graph():
    """Positive control for `classify_seam`, on a module whose answer is known.

    Each of the four destinations is exercised by a name that can only land
    there for the stated reason:

    - six SINGLE-SIGNAL cases -- `_direct`, `_by_image_reqs`,
      `_by_module_binding` and the three `_by_*_string` -- carry exactly one
      client signal each, one per member of `_CLIENT_BINDINGS` and
      `_CLIENT_MODULES`. Deleting any member of either set turns one of them red
      by name; all six deletions were run and each has an objector.
    - `test_probe_client_transitive` and `test_probe_client_via_shared` carry no
      signal at all and are client purely because they call something that is.
      That is the whole reason this is a closure and not a marker list, and the
      closure is what covers `"module"` on the monolith: measured, deleting
      `"module"` from `_CLIENT_BINDINGS` moves 0 names and disabling the closure
      moves 0 names, but doing BOTH moves 30, 13 of them tests. Two mechanisms
      covering the same 13 tests each score dead in a one-at-a-time census.
      Neither is.
    - `_runner_only_helper` carries no marker at all -- no helper does -- and is
      runner because only runner tests reach it.
    - `_shared_helper` is reached from BOTH halves, so it goes to the shared
      module. A classifier forced to pick a half would strand it in whichever
      file lost, which is a NameError at run time and not a collection error, so
      neither `--collect-only` nor a name-set comparison would see it.

    AND ONE CASE PER MODULE-LEVEL NODE KIND, which is the second half of this
    test and the more easily lost one. Review measured that the probe parsed to
    `['FunctionDef']` and nothing else, so four of `classify_seam`'s decision
    surfaces had no objector here and were caught by the manifest agreement test
    ALONE -- the one instrument whose own docstring concedes it is green on a
    coordinated edit that regenerates the manifest in the same commit. Each of
    the four now turns THIS test red, measured by running the mutation:

      `_DEFS` drops `ast.ClassDef`          -> `_ProbeSharedDouble` unplaced
      `_module_level_names` drops `Assign`  -> `_PROBE_PINNED_DIGEST` unplaced
      ... drops `AnnAssign`                 -> `_PROBE_TIMEOUT_S` unplaced
      `SEAM_HEADER_NAMES` emptied           -> `ROOT` gains a destination
      ... gains a governed name             -> the pinned set differs

    `AnnAssign` gets its own case because it is its own branch: removing only it
    leaves the `Assign` control green, and that mutation is the one where this
    test is the SOLE objector -- the agreement test does not fire, because the
    monolith has no module-level `AnnAssign` for it to lose.

    PITFALL this control exists for: `classify_seam` assigns `SEAM_GUARDS`
    unconditionally, from the constant and before it reads any source. So the
    guard names appear in the result for ANY input, including a source that
    defines none of them. A reader who assumes those were computed will misread
    every other result in this file -- and an earlier revision of this test
    enshrined the confusion, asserting `{destinations[n] for n in SEAM_GUARDS} ==
    {PACKAGING_FILE}`, which iterates the same constant the dict was built from
    and therefore cannot fail. The replacement plants one guard name in the probe
    as a test whose own body would classify it CLIENT, so the OVERRIDE is what is
    measured, and pins the set so that adding or removing a member fires.
    """
    destinations, defined_in = classify_seam({"probe.py": _SEAM_CLASSIFIER_PROBE})

    # One per member of _CLIENT_BINDINGS, then one per member of _CLIENT_MODULES.
    assert destinations["test_probe_client_direct"] == CLIENT_FILE
    assert destinations["test_probe_client_by_image_reqs"] == CLIENT_FILE
    assert destinations["test_probe_client_by_module_binding"] == CLIENT_FILE, (
        "a test whose only client signal is binding and dereferencing `module` "
        "was classified as runner. On the monolith 13 client tests carry that "
        "and no other signal in their own body.")
    assert destinations["test_probe_client_by_artifacts_string"] == CLIENT_FILE
    assert destinations["test_probe_client_by_run_modal_string"] == CLIENT_FILE
    assert destinations["test_probe_client_by_sidecar_string"] == CLIENT_FILE

    assert destinations["test_probe_client_transitive"] == CLIENT_FILE, (
        "a test that reaches the client only THROUGH a helper was classified as "
        "runner, so the transitive closure is not running and the classifier has "
        "degenerated into the marker list it was built to replace")
    assert destinations["test_probe_client_via_shared"] == CLIENT_FILE
    assert destinations["_client_only_helper"] == CLIENT_FILE

    assert destinations["test_probe_runner_via_helper"] == RUNNER_FILE
    assert destinations["test_probe_runner_via_shared"] == RUNNER_FILE
    assert destinations["_runner_only_helper"] == RUNNER_FILE, (
        "a helper reached only by runner tests was not sent to the runner half; "
        "stage 2 follows consumers and this one has only runner consumers")

    assert destinations["_shared_helper"] == SHARED_FILE, (
        "a helper reached from BOTH halves was forced into one of them. That is "
        "the stranding this third destination exists to prevent.")

    # ── the module-level node kinds the census must cover ────────────────────
    # `.get` rather than `[...]`: the failure these three exist for is the name
    # being ABSENT from the census, and a KeyError says that far less clearly
    # than the message does.
    assert destinations.get("_ProbeSharedDouble") == SHARED_FILE, (
        "a module-level CLASS reached from both halves was not placed where its "
        "consumers say. If it is missing entirely, `_DEFS` has stopped covering "
        "`ast.ClassDef` and the monolith's 14 classes have no destination at "
        f"all. got={destinations.get('_ProbeSharedDouble')!r}")
    assert destinations.get("_PROBE_PINNED_DIGEST") == CLIENT_FILE, (
        "a module-level ASSIGN reached only by client tests was not placed. If "
        "it is missing entirely, `_module_level_names` has stopped walking "
        "`ast.Assign` -- the branch whose whole point is that a pinned CUDA "
        "digest must not be copied into both files and left to drift. "
        f"got={destinations.get('_PROBE_PINNED_DIGEST')!r}")
    assert destinations.get("_PROBE_TIMEOUT_S") == RUNNER_FILE, (
        "a module-level ANNASSIGN reached only by runner tests was not placed. "
        "`_module_level_names` handles AnnAssign in a SEPARATE branch from "
        "Assign, so it needs its own control; the Assign control above stays "
        f"green when only this branch is removed. got={destinations.get('_PROBE_TIMEOUT_S')!r}")

    # ── SEAM_HEADER_NAMES: ungoverned must not mean invisible ────────────────
    assert "ROOT" not in destinations, (
        "`ROOT` was given a destination, so `SEAM_HEADER_NAMES` has stopped "
        "exempting it. Measured, 5 collected test files define a `ROOT` of "
        "their own, so governing it makes the seam's scan collide with four "
        "files that have nothing to do with the seam.")
    assert defined_in.get("ROOT") == [
        "probe.py"
    ], ("`ROOT` fell out of `defined_in` as well as out of `destinations`. "
        "Ungoverned must not mean invisible -- Task 4's placement gate reads "
        "this to prove `ROOT` survives into every destination file, so an "
        f"exemption that also hides it is how `ROOT` gets lost. got={defined_in.get('ROOT')!r}")
    assert SEAM_HEADER_NAMES == frozenset(
        {"ROOT"}), ("the exemption set changed. Every name in it is silently ungoverned, so "
                    "an ADDITION here deletes a name from the seam with nothing else "
                    f"objecting -- which is why the set is pinned. got={sorted(SEAM_HEADER_NAMES)}")

    # ── SEAM_GUARDS overrides computed concern, and that IS testable ─────────
    # The previous revision asserted `{destinations[n] for n in SEAM_GUARDS} ==
    # {PACKAGING_FILE}` and called it a check. It cannot fail: `classify_seam`
    # writes that dict from the constant before it reads any source, and the set
    # comprehension iterates the same constant. An assertion that cannot fail is
    # not an assertion. The probe now DEFINES one guard, as a test whose own body
    # would classify it CLIENT, so the override is measured instead of restated.
    planted = "test_modal_is_an_explicit_dependency_group"
    assert planted in SEAM_GUARDS and defined_in.get(planted) == ["probe.py"], (
        "the planted guard is not both in SEAM_GUARDS and defined by the probe, "
        "so the override assertion below is measuring the wrong thing")
    assert destinations[planted] == PACKAGING_FILE, (
        f"{planted} names a client binding in its own body, so concern alone "
        "would send it to the client half. It is PACKAGING only because "
        "SEAM_GUARDS overrides the computation, and that override is now gone. "
        f"got={destinations[planted]!r}")
    assert SEAM_GUARDS == frozenset({
        "test_modal_is_an_explicit_dependency_group",
        "test_local_entrypoints_do_not_import_modal",
        "test_modal_runner_lib_does_not_import_modal_or_torch",
    }), ("the guard set changed. Removing a member hands that test back to the "
         "computed seam and adding one takes a test out of it; neither shows up "
         f"anywhere else in this test. got={sorted(SEAM_GUARDS)}")
    # The other two are assigned by fiat, from the constant, with nothing in the
    # source defining them. Asserted so nobody reads the override result above as
    # evidence about names the probe never planted.
    unplanted = sorted(SEAM_GUARDS - {planted})
    assert all(name not in defined_in for name in unplanted), (
        f"the probe defines {unplanted}, so their destinations are no longer the "
        "assigned-by-fiat case this asserts")
    assert {destinations[name] for name in unplanted} == {PACKAGING_FILE}

    # A name no test reaches has no consumers to follow, so stage 2 cannot place
    # it. It raises rather than defaulting, because the two things it can be --
    # dead code, or a new entry point -- want opposite answers.
    with pytest.raises(ValueError, match="reached by no test"):
        classify_seam({"orphan.py": "def _reached_by_nothing():\n    return 1\n"})


def test_seam_manifest_agrees_with_the_classifier():
    """The frozen manifest still says what the code says.

    WHY the manifest is frozen at all, given the classifier can recompute it:
    the manifest is what makes a reclassification VISIBLE IN A DIFF. A change to
    `classify_seam` that quietly moves eleven names is a two-line diff with no
    other trace; the same change with this test in place moves eleven lines of
    JSON as well, in the same commit, where a reviewer reads them. That is also
    the answer to "what stops someone narrowing `_CLIENT_BINDINGS`": nothing
    stops it, but it cannot happen silently.

    This test is green before the split and after it, because `classify_seam`
    reads content and not location. That is deliberate and it is a fix: the
    previous draft's Task 3 test was a POST-split assertion committed PRE-split,
    so this task would have committed a red suite and could not have been
    reviewed independently of the split that follows it.

    WHAT THIS DOES NOT DO, so nobody reads it as more than it is: it compares
    the manifest against the classifier, not against where the names actually
    live on disk. Nothing here would notice a name that the split dropped on the
    floor or duplicated into two files. That is Task 4's placement gate.
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    computed, _ = classify_seam(_seam_sources())
    only_manifest = {n: manifest[n] for n in sorted(set(manifest) - set(computed))}
    only_computed = {n: computed[n] for n in sorted(set(computed) - set(manifest))}
    assert not only_manifest and not only_computed, (
        "the manifest and the classifier disagree about WHICH names exist. "
        f"manifest-only={only_manifest} classifier-only={only_computed}")
    disagree = {n: (manifest[n], computed[n]) for n in manifest if manifest[n] != computed[n]}
    assert not disagree, ("the manifest is stale: (frozen, recomputed) for each name that "
                          f"moved: {disagree}")


def test_no_governed_name_is_defined_outside_the_seams_own_files():
    """A name the manifest governs may not be defined under `tests/` elsewhere.

    WHY THIS EXISTS, and it is not the obvious reason. `_names_defined_under_tests`
    scans by GLOB over the test directory rather than by the manifest's own list
    of destination files, because a check that asks the manifest where to look
    cannot see a file the manifest does not know about. Review demonstrated the
    hole on the previous revision: a governed name (`_git`) moved into a
    brand-new `tests/test_modal_stray.py` left this file green, because nothing
    committed here called the scanner at all. It is called now.

    WHAT IT CATCHES TODAY, pre-split: every governed name lives in
    `tests/test_modal_runner.py`, so any governed name appearing in a fourth
    file is either a copy or a migration nothing declared. WHAT IT BECOMES after
    Task 4: the same sentence, with four legal homes instead of one. The
    assertion does not change across the split, which is why it is written here
    rather than left for the task that needs it most.

    WHAT IT DELIBERATELY DOES NOT CATCH: a BRAND-NEW name in a stray file. That
    name is not governed, and an unrelated new test module is legal. Only
    `governed` is in scope -- which is also why the scan's 1,400-odd-name
    superset does not turn this red for every helper in the repo.

    PITFALL, and it is the one this repo keeps paying for: a scan that reached
    nothing would pass this silently. So the scope controls come FIRST and are
    not decoration. `outside` proves the glob reaches past the seam's own files,
    without which nothing could ever be found straying; `unseen` proves the scan
    actually sees the governed names rather than passing on an empty
    intersection. Both are stated as failures of THIS TEST's instrument, not of
    the tree, because that is what they would mean.
    """
    governed = set(json.loads(MANIFEST.read_text(encoding="utf-8")))
    seam_files = {RUNNER_FILE, CLIENT_FILE, SHARED_FILE, PACKAGING_FILE}
    found = _names_defined_under_tests()

    outside = {f for files in found.values() for f in files} - seam_files
    assert outside, ("the scan reached no file outside the seam's own four, so it could not "
                     "report a strayed name even if one existed. `_names_defined_under_tests` "
                     "globs `tests/test_*.py`; if that returned only seam files the glob is "
                     "broken, not the tree.")
    unseen = governed - set(found)
    assert not unseen, ("names the manifest governs are defined nowhere the scan can see, so the "
                        "check below would pass by looking at nothing. Either the scan lost a "
                        "file it should cover, or these names were deleted from the tree without "
                        f"being deleted from the manifest: {sorted(unseen)}")

    strayed = {
        name: sorted(set(files) - seam_files)
        for name, files in found.items() if name in governed and set(files) - seam_files
    }
    assert not strayed, ("these names are governed by the seam manifest but are ALSO defined in a "
                         "file the seam does not own, so a reader has no way to tell which "
                         "definition the suite runs and Task 4's split would silently pick one. "
                         f"{strayed}")


def test_the_modal_test_split_matches_concern_recomputed_from_source():
    """Every governed name lives in the file its RECOMPUTED concern says.

    THE PITFALL THIS EXISTS FOR, measured: a check that compares the split
    against a frozen manifest is green on a maximally wrong split. Review
    reassigned all 250 non-guard, non-shared names runner/client by even/odd
    source index, split the
    file to match its own shuffled manifest, and got `1 passed`. A manifest is a
    record of a decision; it is not evidence the decision was right. So this
    test re-runs `classify_seam` over the files ON DISK and compares the answer
    to where each name actually sits. The reference graph does not change when
    you shuffle names between two files -- which is exactly why the shuffle
    cannot hide from it.

    Three more holes the first draft had, all demonstrated, all closed here:

    * DELETION was invisible: iterating disk->manifest never examines a manifest
      name with no definition anywhere. Deleting `test_allowed_gpus` outright
      gave `1 passed`. Hence the manifest->disk direction below.
    * A FOURTH FILE was invisible: the scan set was the manifest's own value
      set. Moving a test into a new `tests/test_modal_stray.py` gave
      `1 passed`. Hence `_names_defined_under_tests`'s glob.
    * The HEADER EXCLUSION could hide a loss: `ROOT` is ungoverned, so dropping
      it from a destination would be silent. Hence the last assertion.
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    computed, _ = classify_seam(_seam_sources())
    on_disk = _names_defined_under_tests()

    missing = sorted(n for n in manifest if n not in on_disk)
    assert not missing, (
        "the manifest names definitions that exist nowhere under tests/. Either "
        f"they were deleted or they were moved out of reach of this scan: {missing}")

    strayed = {n: on_disk[n] for n in manifest if n not in computed}
    assert not strayed, ("a governed name is defined under tests/ but NOT in any file the seam "
                         "governs, so no concern can be recomputed for it. A new destination is "
                         f"not a place to put split output: {strayed}")

    duplicated = {n: on_disk[n] for n in manifest if len(on_disk[n]) > 1}
    assert not duplicated, (
        "a governed name is defined in two files. Two copies of a pinned digest "
        f"or a fixture drift apart and every other check here stays green: {duplicated}")

    misplaced = {
        n: {
            "on disk": on_disk[n][0],
            "concern says": computed[n],
            "manifest says": manifest[n]
        }
        for n in manifest if on_disk[n][0] != computed[n]
    }
    assert not misplaced, (
        f"{len(misplaced)} of {len(manifest)} names are not in the file their recomputed "
        "concern assigns them to. Note that `manifest says` agreeing with `on disk` proves "
        "nothing -- a wrong split and a manifest written to match it agree perfectly, which "
        f"is why this compares against `concern says`. First 10: "
        f"{dict(list(misplaced.items())[:10])}")

    for rel in sorted(set(manifest.values())):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        defined = set(_module_level_names(tree))
        absent = sorted(SEAM_HEADER_NAMES - defined)
        assert not absent, (
            f"{rel} does not define {absent}, which the seam treats as module-header "
            "boilerplate rather than governing. Ungoverned must not mean lost.")


def test_modal_is_an_explicit_dependency_group():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["dependency-groups"]["modal"] == ["modal>=1.4.3,<2"]


def test_local_entrypoints_do_not_import_modal():
    # sys.path.insert("src"): train.py resolves its generated siblings with bare
    # imports (`from _action_spec import ...`), so the src dir itself must be on
    # the child's path. Relying on the editable install's .pth instead would make
    # this test pass/fail on ambient venv state (any `uv sync --no-install-project`
    # removes it) and could silently import siblings from a DIFFERENT checkout.
    code = """
import sys
sys.path.insert(0, "src")
import src.train
import scripts.exp_lib
import scripts.run_experiment
assert 'modal' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_modal_runner_lib_does_not_import_modal_or_torch():
    # Fresh subprocess: the parent may already have torch (Task 3 checkpoint
    # tests) or modal (later runner tests) in sys.modules.
    code = """
import sys
import scripts.modal_runner_lib
assert 'modal' not in sys.modules
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
