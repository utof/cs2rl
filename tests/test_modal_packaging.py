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
    tracked `.py` at this commit = 135 (the spec's 131 at 6c937ca, plus this
    file, plus Task 2's `tests/_modal_import_probe.py`, plus W2's
    `tests/test_modal_client.py` and `tests/modal_test_helpers.py`). NOTHING
    asserts on that total and nothing should -- it moves with every added .py,
    and this sentence read 133 while W2 added two of them, which is the decay
    a stated total invites and the reason nothing may depend on one.
    `test_the_census_scans_the_whole_repo` pins the structural property
    instead, which does not move.

    `cwd=ROOT` is load-bearing: `git ls-files "*.py"` is CWD-RELATIVE, so
    running it from `tests/` returns 86 tracked paths instead of 135 and the
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
       This repo writes that shape twice in THIS file, over which the census
       reports no hit at all: `test_local_entrypoints_do_not_import_modal`,
       whose code string imports `src.train`, `scripts.exp_lib` and
       `scripts.run_experiment` but never the runner lib, and
       `test_modal_runner_lib_does_not_import_modal_or_torch`, whose code
       string holds `import scripts.modal_runner_lib` itself. Named rather
       than numbered because this cite read `tests/test_modal_runner.py:93`
       until W2's split moved both tests here, and a line number aimed at the
       file it lives in rots on the next edit. The `_IMPORTERS` comment fixed
       that rot at its own copy of this citation and left this one, which is
       the shape a diff-scoped reader cannot catch: the correction and the
       staleness are 150 lines apart and only one of them is in the diff. A
       code string importing
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
    """Every tracked `.py` under `tests/` that imports `target`, at any depth, as
    repo-relative posix paths.

    Same census and same walker as the static guard -- `_repo_python_files()`
    and `bare_spelling_imports`, just pointed at a different module name. That
    reuse is the point: this function decides WHICH FILES the runtime probe
    runs, so if it and the guard disagreed about what "imports" means, the probe
    would be scoped by a rule nobody else in this file enforces.

    Scoped to `tests/` because the result feeds a pytest session, and scoped by
    `rel.startswith("tests/")` rather than by a top-level glob: a runner-importing
    test in a future subdirectory belongs in the probe, and narrowing the code to
    match a `tests/*.py` reading would silently drop it, which is the failure this
    section exists to prevent. So when the docstring and the code disagreed about
    the scope, the PROSE is what changed. Measured, all 86 tracked `.py` under
    `tests/` are top-level today -- the only subdirectory is `tests/fixtures`,
    which holds no `.py` -- so the two readings agree by layout rather than by
    rule, which is exactly the kind of agreement that stops holding without
    warning.

    Said as a scope and not as a file kind because W2 made
    `modal_test_helpers.py` a non-test file that imports the runner AND a member
    of `_IMPORTERS`, so "non-test importers" stopped naming the `scripts/` three.
    Those three are listed in the comment at `_IMPORTERS`, which documents them
    rather than holding them.

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
            # not an excuse. Measured: 0 of the 135 tracked .py hit this branch,
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
        cwd=ROOT -> cwd=ROOT/tests -> assert 1 (86 paths; ls-files is relative)
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
    six `_IMPORTERS` files -- 421 of them, contributed by five, because
    `tests/modal_test_helpers.py` defines no tests and is on the argv so the
    probe can see it imported. So this test inherits the flakiness of every one
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
    a nested pytest session running all six `_IMPORTERS` files end to end.
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


# ── The modal test seam: concern, recomputed from source ───────────────────
#
# WHY this section exists at all. `tests/test_modal_runner.py` held two suites.
# Splitting it needs an answer to "which half does this name belong to?" that a
# checked-in test can RE-DERIVE, because the alternative -- freeze a manifest,
# then check the files agree with the manifest -- is green on a maximally wrong
# split: review shuffled every name by even/odd index, split the file to match
# its own shuffled manifest, and the consistency check passed. A manifest is a
# record, not evidence. `classify_seam` is the evidence.

MANIFEST = ROOT / "tests" / "fixtures" / "modal_test_seam_manifest.json"

# How many names the seam governs, pinned so that SHRINKING it costs a diff line.
#
# THE HOLE THIS FILLS, demonstrated rather than argued. Every assertion in the
# placement gate iterates the manifest, and the agreement test compares
# `set(manifest)` against `set(computed)`. A name deleted from the tree AND from
# the manifest is therefore examined by nothing: review excised
# `test_allowed_gpus` from both and the seam reported `3 passed`. That is not an
# exotic mutation -- it is exactly what a real deleting commit looks like, because
# deleting the test alone reddens the agreement test and whoever did it fixes
# that before pushing. Regenerating the manifest is the natural fix and it is
# also what hides the loss.
#
# Pinning the size does not stop a deletion; nothing here can, and it should not.
# It makes one visible, which is the same argument the manifest itself rests on:
# a reclassification that moves eleven names moves eleven lines of JSON where a
# reviewer reads them. This moves one number. Change it deliberately, in the
# commit that changes the seam, and say why.
#
# RE-DERIVED ONCE FOR THE WHOLE W3a GATES BRANCH, not per task, because every
# module-level name any task adds to a `_seam_sources()` file moves it and a
# per-task re-pin would be re-derived three times and wrong twice. 261 -> 273:
# the twelve new names are the gate (a) and gate (d) helpers and tests added to
# `tests/test_modal_client.py`. Measured, not inferred -- `len(classify_seam(
# _seam_sources())[0])`, noting that `classify_seam` returns a PAIR, so the
# obvious `len(classify_seam(...))` is 2. Destinations: 168 runner, 95 client,
# 7 shared, 3 packaging guards.
#
# A GREEN `assert len(manifest) == GOVERNED_NAME_COUNT` IS NOT EVIDENCE THIS
# NUMBER IS RIGHT. It compares two frozen artifacts -- a checked-in manifest
# against a checked-in constant -- so adding names to a governed file changes
# neither and it stays green with a stale count and N ungoverned names on disk.
# Its own message says so: it is a DELETION detector. The addition detector is
# `test_seam_manifest_agrees_with_the_classifier`, which recomputes.
GOVERNED_NAME_COUNT = 273

RUNNER_FILE = "tests/test_modal_runner.py"
CLIENT_FILE = "tests/test_modal_client.py"
PACKAGING_FILE = "tests/test_modal_packaging.py"

# THE MEMBERSHIP RULE FOR THE SHARED FILE. `tests/modal_test_helpers.py` exists
# as of W2's split and carries these words in its own docstring, which is where a
# reader opening that file will look for them. This copy is the one the
# classifier sits next to; they must not drift.
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
# `tests/test_*.py`: 6 files define a `ROOT` of their own (`test_modal_argv.py`,
# `test_modal_client.py`, `test_modal_packaging.py`, `test_modal_protocol.py`,
# `test_modal_runner.py`, `test_train_loop_timing.py`). `tests/modal_test_helpers.py`
# defines one too and is correctly absent from that list: it is outside the
# `test_*.py` glob this sentence scopes by.
#
# WAS 5 BEFORE W2, AND THE SIXTH IS THIS SEAM'S OWN CLIENT FILE -- which is why
# the second half of this paragraph had to be rewritten rather than renumbered.
# It used to read "`ROOT` is the ONLY one of the monolith's 261 module-level
# names that any other collected test file also defines". After the split that is
# false by 86: the client half is itself a collected test file and defines 84 of
# those names, and the three relocated `SEAM_GUARDS` are defined here. The claim
# the sentence was making survives once the seam's own four files are excluded
# from "any other", which is what it always meant -- measured that way, `ROOT` is
# still the ONLY collision, and the files it collides with are
# `test_modal_argv.py`, `test_modal_protocol.py` and `test_train_loop_timing.py`,
# 3 of them. Measured at this commit.
#
# THE 261 IN THIS PARAGRAPH IS THE MONOLITH'S OWN NAME COUNT AND IS HISTORICAL.
# It used to be the live governed count as well, and saying "261 is unchanged"
# here stopped being true on the W3a gates branch, which took the seam to 273
# (`GOVERNED_NAME_COUNT`). What this paragraph actually measures did NOT move:
# re-measured at this commit, the collision set is still `ROOT` alone and the
# same 3 files, and the client half still defines 83 of the monolith's governed
# names plus `ROOT` -- which is the 84, and 83 + the 3 relocated `SEAM_GUARDS`
# is the 86. The branch's twelve new client-half names are not monolith names,
# so none of them enters any figure in this paragraph.
#
# So this exemption list is one name long because the collision set is one name
# long, not because the rest were not looked for. Treating `ROOT` as a shared
# helper would make the seam's own scan collide with those unrelated files.
#
# STATED RATHER THAN HIDDEN: an exemption nothing checks is a hole. It is closed
# in both directions as of W2 by `test_the_modal_test_split_matches_concern_
# recomputed_from_source` below, whose last assertion requires `ROOT` to BE
# defined in every destination file, so "not governed" cannot quietly become
# "lost".
SEAM_HEADER_NAMES = frozenset({"ROOT"})

# A name whose own body reaches the client modules. These two sets are the
# spec's own §2.3 instrument, member for member.
#
# `module` is here because the monolith's client tests bind
# `module = _import_run_modal()` and then talk to `module`: measured, 13 of the
# 63 client tests carry that signal and NO other in their own body. THE
# DENOMINATOR MOVED ON THE W3a GATES BRANCH AND THE NUMERATOR DID NOT -- 54 -> 63
# as the four gates were added, while 13 stayed 13. The mechanism, measured
# rather than assumed: all 9 new client-half gate tests carry the signal set
# `{_import_run_modal, module}`, because GC5c's placement fix gives each one a
# `module = _import_run_modal()` positive control, and this sentence counts
# tests whose ONLY own-body signal is `module`. Bumping the 13 alongside the 54
# would have put a false figure into the file that holds the classifier; it is
# re-derived here, not inferred. Deleting it
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


def _module_level_binding_counts(tree):
    """{name: how many module-level statements bind it} -- the LIST-shaped census.

    WHY THIS EXISTS NEXT TO `_module_level_names`, which looks like it already
    answers the question: that function returns a DICT, so two module-level
    definitions of one name inside ONE file collapse to a single entry, and
    everything built on it inherits the blindness. The placement gate's
    `duplicated` check counts FILES and therefore cannot see the case at all.
    Measured by review at `84622fc`: appending a second `PINNED_CUDA_IMAGE` to
    tests/test_modal_client.py, shadowing the real digest with
    `...@sha256:deadbeef`, left the whole seam at `3 passed`.

    THE PITFALL, and it is the failure the seam was built to stop rather than a
    new one: two copies of a pinned digest that drift apart. Python does not let
    you have two LIVE definitions of a name across two files -- one import wins
    and the other is dead -- but it lets you have them inside one file, where the
    second silently shadows the first and both are in the source a reader greps.
    So the intra-file shape is the only one the failure can actually take at run
    time, and it was the one shape nothing watched.

    Mirrors `_module_level_names`' branches exactly: defs and classes via
    `_DEFS`, `Assign` targets walked so tuple unpacking counts each name, and
    `AnnAssign`. It counts where that function assigns, so the two cannot come to
    disagree about what a module-level binding is -- which matters, because a
    census that disagreed with the one the gate uses everywhere else would report
    duplicates nobody else believes in.
    """
    counts = {}
    for node in tree.body:
        names = []
        if isinstance(node, _DEFS):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [s.id for t in node.targets for s in ast.walk(t) if isinstance(s, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        for name in names:
            counts[name] = counts.get(name, 0) + 1
    return counts


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
       194 tests land on the wrong side. That 194 is a MONOLITH figure and is
       kept as one deliberately, because the measurement it reports was taken
       against the spec's line-3638 seam, which no longer exists. Do not try to
       re-derive it from this function's inputs: `_seam_sources()` yields 201
       tests today (138 runner + 63 client), and the missing 3 are the
       `SEAM_GUARDS`, which moved into a file `_seam_sources()` excludes. That
       201 was 192 (138 + 54) at the W3a branch point; the nine gate tests the
       branch added to the client half are the whole difference, and the runner
       half did not move. These are LIVE counts of module-level `test_`
       functions, re-derivable from either file's AST -- unlike the 194 above.

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
    alone puts 28 names on the wrong side -- all of them non-test helpers, 0 of
    them tests -- or 21 once the 7 SHARED names are set aside, which stage 1
    structurally cannot produce and which the next sentences are about. Both
    numbers are one measurement under two scopes, so the scope is stated rather
    than left to the reader. Stage 2 was added afterwards, with the target
    already
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
    `bare_spelling_imports` above. Re-measured at this commit over the files
    `_seam_sources()` actually yields: **zero** such blocks remain in any of
    them. Both of the two that existed --
    `test_local_entrypoints_do_not_import_modal` and
    `test_modal_runner_lib_does_not_import_modal_or_torch` -- moved into THIS
    file with the `SEAM_GUARDS` in W2, and this file is excluded from
    `_seam_sources()`. They were assigned by fiat anyway and named only modules
    outside the seam, so the limit's consequence is unchanged; its location is
    not. Until this commit the paragraph said `tests/test_modal_runner.py` has
    two such blocks, which the same commit made false.

    Measured the other way too: 8 of the monolith's 261 module-level names now
    appear as a word inside a multiline string literal somewhere in the seam, and
    none of them is a reach. Five are prose in the rewritten runner header
    (`ROOT`, `_run_cuda_probe`, and the three `SEAM_GUARDS` names it says moved
    out); `_aware` is prose in the shared module's docstring; and
    `test_exact_allowed_flags_are_kept` and
    `test_unknown_spelling_rejected_before_ownership` are cited by
    `test_tct_split_trunk_is_allowed`'s docstring.

    THAT 261 IS THE MONOLITH'S OWN NAME COUNT AND IS HISTORICAL -- the same label
    the `SEAM_HEADER_NAMES` comment block above puts on its identical phrasing,
    and this one needs it more, because the live governed count in this file is
    now 273 (`GOVERNED_NAME_COUNT`). AND THE CANDIDATE SET IS THE 247 NON-CLASS
    NAMES AMONG THOSE 261, which is the only reading that reproduces the 8.
    Naming it is not pedantry: over all 261 the same instrument returns 9 at the
    branch point and 10 here, so a re-deriver who takes the obvious reading finds
    a number that looks stale and "corrects" one that is right. The two names
    separating the readings are both CLASSES -- `FakeChild`, pre-existing, and
    `FakeImage`, added by this branch. Over the 247 it returns 8 at BOTH ends,
    and that 8 is the paragraph above's enumeration, member for member.

    EXPECT THAT COUNT TO MOVE, and do not read a change in it as a finding by
    itself. It rises whenever any docstring names any governed test, which is
    what good docstrings do -- it was 3 before W2 and this commit's own header
    rewrite took it to 8. What must not move is the CONSEQUENCE, and that is
    checked directly rather than inferred from the count: feeding this file in
    changes the destination of 0 monolith names, and every one of these mentions
    sits in prose rather than in a `code = \"\"\"...\"\"\"` block a subprocess
    executes.

    Named rather than cited by line number throughout: this paragraph used to end
    "a docstring at line 574", which was accurate in the monolith and points at
    argv-list content after the split -- the second bare line number in this file
    to rot inside the commit that moved the code it named. So this limit changes
    no destination today. A future subprocess block could.
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
    the classifier itself as a client test. Re-measured at this commit: fed in,
    `classify_seam` returns 321 destinations rather than 273, and of the 51 names
    this file contributes to that graph, 14 go to `tests/test_modal_client.py` --
    `classify_seam` and `_reaches_client_directly` among them -- while 26 go to
    the runner half, 8 to the shared module and 3 to this file. Every one of the
    first three groups is nonsense. A census of how test files import the
    runner has no side of this seam. (The 3 are the `SEAM_GUARDS`, assigned by
    fiat and the only ones this file legitimately receives; they are counted here
    because they are among its module-level names, which is what "contributes"
    means. 14 + 26 + 8 + 3 = 51, and 321 - 273 = 48 because the guards already
    have destinations without this file being fed in.)

    "51 NAMES", NOT "51 GOVERNED NAMES", and the distinction is this file's own
    vocabulary rather than pedantry: `governed` here means `in the manifest` --
    it is what `GOVERNED_NAME_COUNT` counts and what
    `test_no_governed_name_is_defined_outside_the_seams_own_files` iterates. Of
    these 51, exactly 3 are governed, and they are the `SEAM_GUARDS`. The file
    has 52 module-level names; the 52nd is `ROOT`, which `SEAM_HEADER_NAMES`
    excludes from classification. Writing "51 governed" tells a reader this file
    contributes 51 manifest entries, which is wrong by 48.

    THOSE FIVE NUMBERS WERE 296 / 36 / 11 / 18 / 7 BEFORE W2 AND
    300 / 42 / 14 / 17 / 8 AFTER IT, AND THE FIRST TRANSITION MOVED ALL FIVE
    WHILE THE SECOND MOVED ONLY THREE (300 -> 321, 42 -> 51, 17 -> 26; the 14
    and the 8 stayed), which is worth more than the numbers are. "Measured at this
    commit" is what makes this sentence a live re-derivable claim rather than
    a historical one, and a live claim in a file the same commit edits has to
    be re-run in that commit. It was not, and the review caught it. Note how
    cheaply it moves: any module-level name added here shifts every figure.
    The split added four (the placement gate and the three relocated guards),
    the review fixes added two more (`GOVERNED_NAME_COUNT` and
    `_module_level_binding_counts`), and the W3a gates branch added nine --
    gate (e)'s helper and two tests, gate (g)'s helper and five tests -- which
    is why the corrected figures are not the ones the review reported either;
    they were correct when it measured them.

    AND TWO OF THE FIVE DID NOT MOVE, WHICH IS THE REASON THESE ARE RE-DERIVED
    AND NOT ARITHMETIC. The W3a gates branch added 9 names here and ALL NINE
    landed on the runner half: 17 -> 26, while 14 / 8 / 3 stayed exactly where
    they were. A reviewer applying "+9 spread across the partition" -- or
    applying it to the 14 because the 14 is the group the paragraph names
    first -- writes a false figure that sums correctly and is therefore
    invisible to the arithmetic check three lines up. Every number in this
    paragraph was measured by feeding this file into `classify_seam` at this
    commit.

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

    Existence-tolerant on purpose, and it stays that way now that the split has
    landed. `tests/test_modal_client.py` and `tests/modal_test_helpers.py` did
    not exist until W2 created them; all three files are present today, so the
    `exists()` filter is a no-op at this commit. It is kept because
    `classify_seam` computes from CONTENT, not from where content lives, so the
    manifest it produces over the unsplit monolith is the same one it produces
    over the split files -- which is what let the agreement test below stay green
    on BOTH sides of the split instead of shipping red for one task, and is what
    would let the seam be re-derived from a pre-split checkout.
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
        "exempting it. Measured, 6 collected test files define a `ROOT` of "
        "their own -- the comment above `SEAM_HEADER_NAMES` names them -- so "
        "governing it makes the seam's scan collide with the 3 of those that "
        "have nothing to do with the seam.")
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

    WHAT IT CATCHES: every governed name lives in one of the seam's four legal
    homes, so a governed name appearing in a FIFTH file is either a copy or a
    migration nothing declared. Written pre-split, when the sentence read "every
    governed name lives in `tests/test_modal_runner.py` ... one legal home" and
    "WHAT IT BECOMES after Task 4" was the four-home version; W2 made the second
    half the present tense and 105 of the governed names no longer live in the
    runner file. That figure was 93 at W2 and moved on the W3a gates branch, which
    added twelve governed names and put all twelve in the client half; the runner
    half's 168 did not move. Re-derived here, not carried forward -- it is a
    present-tense count over the manifest this commit regenerates, and the same
    commit that regenerates the manifest is the one that has to re-run it. The
    ASSERTION itself did not change across the split -- it reads
    the manifest's own value set -- which is why it was written here rather than
    left for the task that needed it most, and is why only this prose moved.

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
    source index, split the file to match its own shuffled manifest, and got
    `1 passed`. A manifest is a record of a decision; it is not evidence the
    decision was right. So this test re-runs `classify_seam` over the files ON
    DISK and compares the answer to where each name actually sits. The reference
    graph does not change when you shuffle names between two files -- which is
    exactly why the shuffle cannot hide from it.

    Three more holes the first draft had, all demonstrated, all closed here:

    * DELETION, in the shape that leaves the manifest naming the deleted test:
      iterating disk->manifest never examines a manifest name with no definition
      anywhere. Deleting `test_allowed_gpus` outright gave `1 passed`. Hence the
      manifest->disk direction below.
    * A FOURTH FILE was invisible: the scan set was the manifest's own value
      set. Moving a test into a new `tests/test_modal_stray.py` gave
      `1 passed`. Hence `_names_defined_under_tests`'s glob.
    * The HEADER EXCLUSION could hide a loss: `ROOT` is ungoverned, so dropping
      it from a destination would be silent. Hence the last assertion.

    TWO MORE SHAPES THIS GATE SHIPPED GREEN ON, found by review at `84622fc` by
    running them rather than reading them, and closed here. Both are the same
    defect class as each other and as the reason this file exists: a bullet above
    claimed a hole was closed when only ONE SHAPE of that hole was.

    * DELETION THAT ALSO EDITS THE MANIFEST -- which is the shape a real deleting
      commit produces, because leaving the manifest stale reddens
      `test_seam_manifest_agrees_with_the_classifier` and the author fixes that
      before pushing. Excising `test_allowed_gpus` from the file AND removing its
      manifest key gave `3 passed`. Every assertion here iterates `manifest`, and
      the agreement test compares `set(manifest)` against `set(computed)`, so a
      name absent from BOTH is examined by nothing at all. Hence
      `GOVERNED_NAME_COUNT`: the seam's size is pinned, so a deletion can no
      longer be absorbed by regenerating the manifest -- it has to move a number
      a reviewer reads in the diff.
    * AN INTRA-FILE DUPLICATE. `duplicated` counts FILES, and the `on_disk` lists
      come from `_module_level_names`, which is a DICT -- so two module-level
      definitions of one name inside ONE file collapse to a single entry and the
      list length stays 1. Appending a second `PINNED_CUDA_IMAGE` to
      tests/test_modal_client.py gave `3 passed`. That is precisely the failure
      `duplicated`'s own message describes, in the only arrangement Python
      actually permits: you cannot have two live definitions across two files,
      but you can inside one, where the second silently shadows the first. Step
      6's control (d) covered only the cross-file shape. Hence
      `_module_level_binding_counts` and `redefined` below.
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    computed, _ = classify_seam(_seam_sources())
    on_disk = _names_defined_under_tests()
    trees = {
        rel: ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for rel in sorted(set(manifest.values()))
    }

    assert len(manifest) == GOVERNED_NAME_COUNT, (
        f"the seam governs {len(manifest)} names, not {GOVERNED_NAME_COUNT}. If you deleted a test "
        "and regenerated the manifest to match, every other check in this file stays green and the "
        "loss is invisible -- that is the shape this pin exists for. If the change is intended, "
        "move the number and say why in the commit; if it is not, you have lost a test.")

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

    redefined = {}
    for rel, tree in trees.items():
        for name, times in sorted(_module_level_binding_counts(tree).items()):
            if name in manifest and times > 1:
                redefined[name] = {"file": rel, "module-level definitions": times}
    assert not redefined, (
        "a governed name is defined more than once at module level INSIDE one file, so the second "
        "definition silently shadows the first. `duplicated` above counts files and cannot see "
        "this; it is the same drift, in the only arrangement Python permits. Two copies of a "
        f"pinned digest is the case that motivated the seam: {redefined}")

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

    for rel, tree in trees.items():
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
    # Fresh subprocess: the parent may already have torch (the resume-input
    # checkpoint tests) or modal (later runner tests) in sys.modules.
    code = """
import sys
import scripts.modal_runner_lib
assert 'modal' not in sys.modules
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def _nonstdlib_module_scope_imports(repo_root, top_level_only=False):
    """Gate (e), criterion 5. Every MODULE-SCOPE import in the optional runner's
    library must name a STANDARD-LIBRARY module. Returns `(scanned, violations)`.

    `scanned` lists the repo-relative posix path of every file actually READ.
    `violations` is a sorted list of `(rel, lineno, root_module)` triples, one
    per offending import NAME (so `import a, b` can contribute two).

    WHY A PAIR AND NOT JUST THE VIOLATION LIST, which is the whole reason the
    signature looks like this: a guard that enumerates NOTHING returns no
    violations, and an empty violation list from an empty file set is
    byte-identical to one from a clean tree. Measured -- point the candidate
    list at a path that does not exist and this returns `([], [])`, vacuously
    green forever. GC1 permits evidence to be a pair, and `scanned` is the half
    that makes the gate's OWN SCOPE assertable;
    `test_gate_e_criterion_5_the_runner_library_imports_only_the_standard_library`
    asserts it by name, so "the gate stopped reading the file" is a red rather
    than a silent green. This repo's recurring gate defect is a guard blind to
    its own scope; a bare violation list has nowhere to put that evidence.

    `repo_root` is a PARAMETER and never module-level `ROOT` (GC1), because the
    knock-out below runs against a `tmp_path` copy. A helper that reached for
    `ROOT` would scan the real file no matter what it was handed, which is a
    mutant nothing could kill.

    THE TARGET SET IS ENUMERATED FROM THE FILESYSTEM (`is_file()`), not asserted
    to exist. Pre-split that set has exactly one literal member, and that is a
    stated compromise rather than a hidden one: a `scripts/*.py` glob also picks
    up `scripts/run_modal.py`, whose module scope imports `modal` (`:31`) and
    `scripts.modal_runner_lib` (`:33`) -- measured, 2 violations -- and reddens
    this gate over a file that is SUPPOSED to import Modal. W3b replaces the
    literal with enumeration over the split package's directory, and that swap
    is a B12 obligation, not an optional tidy-up.

    THE INSTRUMENT, and the PRINCIPLE that generates it rather than a list to
    memorise: **the gate checks exactly what EXECUTES at import time.** Every
    skip and every exemption below is that one question asked of one construct,
    and nothing is in the walk because it looked like "module scope".

      * `FunctionDef` / `AsyncFunctionDef` bodies do NOT run on import -- skipped.
        That is why `torch` at `:810` and `:1498` is the design rather than a
        violation, and it is the whole reason bare `ast.walk` cannot be used.
      * `ClassDef` bodies DO run on import -- WALKED. `class C:` + `import modal`
        really does import Modal. (Methods inside are `FunctionDef`s, so the walk
        reaches them and then skips them, which is correct for both reasons at
        once.)
      * `if TYPE_CHECKING:` bodies NEVER run -- exempted, and the `else:` branch
        is still walked because it is exactly the branch that DOES run. Both
        spellings are recognised, bare `TYPE_CHECKING` and `typing.TYPE_CHECKING`.
      * every other block -- `try`, `if`, `with`, `for` -- runs on import and is
        walked, which is what the decoy's `try:`-wrapped `import modal` proves.

    "Do not descend into `FunctionDef` / `AsyncFunctionDef` / `ClassDef`" was the
    plan's original wording and it got two of those three wrong; the principle
    above is the correction, applied in fix round 1. The `TYPE_CHECKING`
    exemption is NOT a hole in the same sense: importing this module does not
    import what such a block names, so reddening on one would be a false positive
    over correct code, and a guard that reddens on correct code gets switched off.

    INTERACTION WITH GATE (g), stated because a later reader will otherwise think
    one of the two is wrong: criterion 13 governs the SHAPE of a `TYPE_CHECKING`
    block (at most one, imports only); criterion 5 declines to look INSIDE one at
    all. They are complementary -- (g) polices the construct, (e) polices what
    running the module costs -- and neither subsumes the other.

    Measured on today's unmodified module:

        instrument         | import nodes | non-stdlib roots reported
        tree.body          |      23      | none
        import-time walk   |      23      | none
        bare ast.walk      |      25      | `torch` -- `:810` in `_import_torch`,
                           |              | `:1498` in `_load_checkpoint_weights`

    So a bare-`ast.walk` gate is RED ON THE CORRECT, UNMODIFIED MODULE, and the
    obvious repair is to fall back to `tree.body`, which is blind to everything
    below depth 1. This SUPERSEDES the spec's own bold instruction to use
    `ast.walk`; the override is deliberate and measured.

    `top_level_only` exists for exactly ONE caller: the knock-out, which must
    ASSERT that the `tree.body` instrument is green on a plant this one catches.
    The two instruments return the identical set on the unmodified module
    (23 == 23, both clean), so the plant is the ONLY observation that separates
    them, and a described difference is not an asserted one.

    EVERY BRANCH ABOVE IS INERT ON THIS SUBJECT and is therefore held by a plant
    instead: 0 async defs, 0 `TYPE_CHECKING` blocks, and 23 classes none of which
    contains an import. An exemption the subject never exercises is
    indistinguishable from a gate that never sees one -- so the knock-out ships a
    row for each, including a GREEN `TYPE_CHECKING` row importing something
    NON-stdlib, since a stdlib import there would pass for the wrong reason.

    THE RESOLUTION RULE, spelled out because the obvious wrong one is red on the
    correct module: `alias.name.split(".")[0]` for `Import`,
    `node.module.split(".")[0]` for `ImportFrom` with `node.level == 0`.

      * Treating `ImportFrom` ALIASES as module names -- the natural mistake --
        reports 13 false non-stdlib names on today's module (`Callable`,
        `Mapping`, `Path`, `Protocol`, `PurePosixPath`, `Sequence`, `UTC`,
        `annotations`, `asdict`, `dataclass`, `field`, `replace`, `timedelta`).
      * Skipping `.split(".")[0]` reports `collections.abc`, which is not a key
        in `sys.stdlib_module_names` -- that set holds TOP-LEVEL names only. On
        the `ImportFrom` side today's module objects; on the `Import` side it
        does NOT, because it has no dotted plain `import` at module scope, and
        fault seeding is what found that. The knock-out's
        `import xml.etree.ElementTree` row is the observation that covers it.
      * `alias.asname` instead of `alias.name` is silent on today's module and
        on a plain `import modal`; the `import modal as ...` row in the knock-out
        is what objects.
      * `node.level == 0` skips relative imports, which are intra-package and
        not third-party dependencies. Inert today (all 23 imports are level 0)
        and load-bearing the moment W3b's package modules import each other.

    `node.module` is additionally tested for truth because `from . import x`
    parses to `module=None`; that can only happen with `level > 0`, so the test
    is unreachable under the rule above and is kept so the helper stays total if
    the level check is ever loosened.

    A `SyntaxError` out of `ast.parse` is deliberately NOT caught: an unparsable
    runner library is a red for a real reason, and a `try` here would let the
    gate silently stop covering its only target.
    """
    root = Path(repo_root)
    # Enumerated from the filesystem, not asserted into existence -- a missing
    # target must show up as an empty `scanned`, which the green test reads.
    candidates = [root / "scripts" / "modal_runner_lib.py"]
    scanned = []
    violations = []
    for path in candidates:
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        scanned.append(rel)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # Annotated `ast.AST` rather than inferred `ast.stmt`, because
        # `ast.iter_child_nodes` yields `AST` (expressions and `alias` nodes
        # included) and the pre-commit pyrefly gate reds on the unannotated
        # `stack.extend`. Nothing below reads an attribute that only `stmt` has:
        # `node.lineno` is reached solely inside the `Import` / `ImportFrom`
        # narrowing.
        nodes: list[ast.AST]
        if top_level_only:
            nodes = list(tree.body)
        else:
            nodes = []
            stack: list[ast.AST] = list(tree.body)
            while stack:
                node = stack.pop()
                nodes.append(node)
                # A function body does not run on import. A CLASS body does, so
                # it is walked; the methods inside it are `FunctionDef`s and get
                # skipped here on the next iteration, which is the right answer
                # for the same reason.
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                # `if TYPE_CHECKING:` never runs, so its body costs nothing at
                # import and is exempt. `node.orelse` IS still walked -- that is
                # precisely the branch that does run -- and skipping the whole
                # `If` node would be a real blind spot rather than an exemption.
                # Inlined rather than extracted because GC3 pins this file to
                # three new module-level names and a predicate would be a fourth.
                if isinstance(node, ast.If):
                    test = node.test
                    if ((isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
                            or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")):
                        stack.extend(node.orelse)
                        continue
                stack.extend(ast.iter_child_nodes(node))
        for node in nodes:
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = [node.module.split(".")[0]]
            else:
                continue
            for name in roots:
                if name not in sys.stdlib_module_names:
                    violations.append((rel, node.lineno, name))
    return scanned, sorted(violations)


def test_gate_e_criterion_5_the_runner_library_imports_only_the_standard_library():
    """GREEN ON THE REAL SUBJECT. `scripts/modal_runner_lib.py` must be importable
    without pulling in Modal, Torch or anything else off the standard library.

    WHY THIS IS THE POINT OF THE WHOLE FILE: the module's own docstring promises
    "status/download and all validation must work without importing or hydrating
    a Modal App", and `test_modal_runner_lib_does_not_import_modal_or_torch`
    already proves it at RUN TIME in a fresh subprocess. That runtime probe is
    not a substitute for this one and vice versa. The probe asserts `'modal' not
    in sys.modules`, so a `try: import modal / except ImportError: pass` at
    module scope passes it in any environment where Modal is NOT installed --
    which is every environment that does not opt into the `modal` dependency
    group -- and then reddens for the first developer who does opt in. This gate
    reads the SOURCE, so it is independent of what happens to be installed.

    THE SCOPE ASSERTION COMES FIRST and is not decoration. `scanned` proves the
    gate read its target; without it, a gate that enumerated nothing would pass
    this by looking at nothing, which is the failure this repo keeps paying for.

    MEASURED, and this is why the knock-out below is mandatory rather than
    optional: the import-time walk and `tree.body` see the same 23 import nodes
    and report the same empty violation set on this file, so on the unmodified
    subject the correct instrument and the blind one are INDISTINGUISHABLE.
    Bare `ast.walk` sees 25 and reports `torch` (`:810`, `:1498`) -- it is red on
    the correct module, which is why the spec's `ast.walk` clause is overridden.

    INTERACTION WITH GATE (g), criterion 13. Gate (g)'s `find_spec` header is
    green under THIS gate through the WALK, not through the `TYPE_CHECKING`
    exemption -- it is an ordinary `If` and gate (e) does read the imports inside
    it. The two gates disagree about the `TYPE_CHECKING` BODY on purpose:
    criterion 13 governs its SHAPE (at most one block, imports only), criterion 5
    exempts it from purity because it never executes. If a future edit moves an
    `import` inside the `find_spec` header, gate (e) reddens CORRECTLY -- that
    import runs, and this gate's whole question is what running the module costs.
    """
    scanned, violations = _nonstdlib_module_scope_imports(ROOT)
    assert scanned == [
        "scripts/modal_runner_lib.py"
    ], ("gate (e) did not read its target, so an empty violation list below would mean "
        "nothing. Either the file moved (W3b moves it, and the enumeration must move with "
        f"it) or the enumeration stopped pointing at it. Scanned: {scanned}")
    assert violations == [], (
        "these module-scope imports are not in the standard library, so importing "
        "`scripts.modal_runner_lib` now costs whatever they cost -- a CUDA image, in the "
        f"case the module was written to avoid. Import them lazily instead: {violations}")

    _, blind = _nonstdlib_module_scope_imports(ROOT, top_level_only=True)
    assert blind == violations, (
        "the two instruments disagree on the UNMODIFIED module. They are supposed to be "
        "indistinguishable here -- that equality is the reason the knock-out below is the "
        f"only thing that separates them. import-time: {violations}, tree.body: {blind}")


def test_gate_e_criterion_5_reddens_on_plants_the_tree_body_instrument_misses(tmp_path):
    """KNOCK-OUT. Plants live in a `tmp_path` copy; `scripts/modal_runner_lib.py`
    is not edited, not one byte (GC2).

    THE DELIVERABLE IS THE SECOND ASSERTION of the first block. A version of this
    gate built on `tree.body` PASSES the decoy -- that is the exact blindness the
    decoy exists to prevent -- so the blind instrument's green is ASSERTED here,
    not described in prose. Prose cannot fail.

    Every plant goes AFTER LINE 19 (GC2). Line 19 is
    `from __future__ import annotations`, the first statement after the
    docstring, and a `__future__` import must precede all other code: a plant
    above it raises `SyntaxError: from __future__ imports must occur at the
    beginning of the file`, which is a red for the wrong reason. The plant site
    is asserted below so that an edit which moves that line is loud.

    THE TABLE IS ONE ROW PER INDEPENDENT CLAUSE of the gate, because a gate whose
    headline clause has a knock-out and whose secondary clauses have none is the
    defect this branch has shipped three times. Nothing here is a variation on
    the decoy for its own sake -- each row is the only observation that kills one
    mutant:

      * `import modal as ...` is the ONLY row that objects to resolving names
        through `alias.asname`; that mutant is silent on the real module AND on
        the decoy, because a plain `import modal` has no `asname`.
      * `from modal.functions import ...` objects to dropping `ImportFrom`
        handling entirely (silent on the real module, whose from-imports are all
        stdlib) and to keeping the full dotted path instead of the root.
      * `import xml.etree.ElementTree` is the same root-resolution clause for
        `Import`, and it was MISSING until fault seeding found it: the module has
        no dotted plain `import` at module scope, and neither did any other row
        here, so `alias.name` without `.split(".")[0]` survived everything. It is
        a FALSE-RED mutant rather than a false-green one -- the gate would have
        reported `os.path` as non-stdlib -- and a gate that reddens on legal code
        gets switched off, which is the failure mode this file is built around.
      * `import numpy` objects to a DENY-LIST implementation. A gate that looks
        for `modal` and `torch` by name passes the decoy and every other row
        here; `sys.stdlib_module_names` is an ALLOW-LIST, and the polarity is the
        reason `ruff` TID253 was measured GREEN on the decoy and rejected.
      * the two-violation row is the only place either "report just the first
        offender" or "return the walk's own order" is visible; every other row
        has at most one violation, where both mutants are identity functions.
      * the missing-target block at the end objects to dropping the `is_file()`
        filter, and it is what makes the `scanned` assertions mean anything: an
        unread file has to stay OUT of `scanned` for their green to be evidence.
      * the `if`-guarded row objects to adding `If` to the skip set, and it is
        the shape gate (g) plants (`if importlib.util.find_spec(...)`), so the
        two gates' readings of the same construct stay pinned together. Its test
        is a `Call`, so it also stops the `TYPE_CHECKING` exemption from widening
        into "any `If`" -- but a `Call` is neither a `Name` nor an `Attribute`,
        which leaves the widenings that keep the node-class test and drop the
        NAME test invisible to it. THE EXEMPTION HAS TWO CLAUSES AND NEEDS THREE
        ROWS; this one holds only the outermost. The next two hold the rest, and
        they exist because "the headline clause has a knock-out, the secondary
        clause has none" is this branch's named defect class in its B2 form --
        the guard's own allow-set going unwatched.
      * the `nametestguard` row (`_DEBUG = False` + `if _DEBUG:`) objects to
        matching any bare `ast.Name` test. Drop `test.id == "TYPE_CHECKING"` and
        every flag-guarded module-scope import in the file becomes exempt: that
        is a SILENT GREEN ON A REAL IMPORT, not a missed edge case. No other row
        here has a bare-`Name` `If` test THAT IS NOT `TYPE_CHECKING` -- the
        `typechecking` and `typecheckingelse` rows have one each, and both stay
        green under that mutant because the exemption is exactly what they
        exercise -- so nothing else can see it.
      * the `attrtestguard` row (`import os` + `if os.name:`) is the same clause
        on the attribute spelling. `typing.TYPE_CHECKING` is exempt because of
        `test.attr`, NOT because it is an `ast.Attribute`; without this row,
        `isinstance(test, ast.Attribute)` on its own passes every other
        observation in the file.
      * the CLASS-BODY row objects to putting `ClassDef` back in the skip set.
        A class body executes at import time, so this is a real violation; the
        module's own 23 classes contain no import, so nothing else can see it.
      * the class-METHOD row is the other half of that change. Once the walk
        enters class bodies it reaches the methods inside them, and a method is
        a `FunctionDef` whose body does not run on import -- this row is what
        says the walk stops there instead of flagging every lazy import in a
        class.
      * the NESTED-CONTAINER row (`class` > `try` > `class` > `import modal`)
        is the only row whose offending import sits inside a NESTED container
        -- three containers deep, not one. IN CONTAINERS, NOT DEPTHS, and the
        distinction is the claim: under this file's own unit (nodes below
        `Module`, top-level = 1) six other rows put their offending import
        below depth 1 as well, all of them at depth 2. What makes this row the
        only one is that every other container in this table -- the decoy's
        `try`, the `if`-guarded header, the class body -- is TOP-LEVEL, and a
        top-level node enters the walk from `tree.body` rather than by being
        descended into, so a mutant that expands depth 1 and then stops passes
        every one of them while going blind to everything nested inside. The
        nesting runs `ClassDef` > `Try` > `ClassDef` > `Import` ON PURPOSE. It
        is one row, but it objects to dropping EITHER container type from the
        descent as well as to dropping both -- MEASURED: the obvious two-deep
        shapes each hold only half of that, a `try:` inside a class body is
        blind to "stop at a nested `ClassDef`" and a class inside a `try:` is
        blind to "stop at a nested `Try`", while this one kills all three
        mutants. The instrument is a stack, not a single expansion, and this
        is the only observation in the file that says so.
      * the two `TYPE_CHECKING` rows are POSITIVE CONTROLS FOR AN EXEMPTION,
        which is the shape that has gone wrong here before: the subject has 0
        such blocks, so an unexercised exemption is indistinguishable from a
        gate that never meets one. Both spellings are covered (bare
        `TYPE_CHECKING`, `typing.TYPE_CHECKING`), and each imports something
        NON-stdlib on purpose -- `import decimal` inside the block would go green
        whether or not the exemption existed and would certify nothing.
      * the `TYPE_CHECKING`-with-`else` row objects to exempting the whole `If`
        node rather than just its body. The `else:` branch is exactly the branch
        that DOES execute, so skipping it would be a blind spot wearing an
        exemption's clothes.
      * the function-body `import torch` row is the NEGATIVE control that
        separates this instrument from bare `ast.walk`. It is the shape the real
        module uses twice, so `test_..._imports_only_the_standard_library`
        already objects to descending -- this row says so where a reader of the
        instrument is standing.
      * the `async def` row is the only thing that objects to dropping
        `AsyncFunctionDef` from the skip set. The module has 0 async defs, so
        every other observation here is blind to that entry.
      * the relative-import row is the only thing that objects to dropping
        `node.level == 0`. Silent on today's module, which has no relative
        imports; W3b's package modules will have them, and without this row the
        gate would redden over legal intra-package imports on the day of the
        split.

    EVERY ROW BELOW IS INERT ON THE UNMODIFIED SUBJECT, by construction: the
    module has no class-body import, no async def, no `TYPE_CHECKING` block, no
    relative import, no dotted plain `import`, no conditional module-scope import
    of any spelling, and nothing imported inside a container that RUNS at import
    time. Stated that way rather than as "nothing below depth 1", which is false:
    the module has two imports below depth 1, both `torch`, in `_import_torch`
    and `_load_checkpoint_weights` -- function bodies, which is the `lazy` row's
    subject and does not execute on import. That is the point. Each one is the
    ONLY observation in the suite that holds its clause, which is why they are
    rows in a knock-out rather than sentences in a docstring.

    INTERACTION WITH GATE (g), criterion 13 -- stated here and in the green test
    because the two `TYPE_CHECKING` rows and the `if`-guarded row look, from the
    outside, like the same construct getting two different verdicts. They are not
    the same construct:

      * gate (g)'s `find_spec` header is green under gate (e) THROUGH THE WALK,
        not through the exemption. It is an ordinary `If`, and gate (e) does read
        the imports inside it -- which is exactly what the `ifguard` row asserts.
      * the two gates disagree about the `TYPE_CHECKING` BODY on purpose.
        Criterion 13 governs its SHAPE (at most one block, imports only);
        criterion 5 exempts it from the purity check because it never executes.
        Complementary, not contradictory, and neither subsumes the other.
      * so if a future edit moves an `import` INSIDE the `find_spec` header,
        gate (e) reddens -- CORRECTLY. That import runs.
    """
    source = (ROOT / "scripts" / "modal_runner_lib.py").read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    assert lines[18].rstrip("\n") == "from __future__ import annotations", (
        "GC2 pins plants to line 19 because a `__future__` import must come first. That "
        f"line now reads {lines[18]!r}, so every plant below would land in the wrong place "
        "-- re-measure the plant site before trusting anything in this test.")

    def plant(text, slug):
        """Write a tmp_path repo whose `scripts/modal_runner_lib.py` is the real
        file with `text` spliced in after line 19, and return its repo root.

        Nothing here is ever EXECUTED -- the gate parses. So a plant may name
        `os` or `modal` without either being importable.
        """
        target = tmp_path / slug / "scripts" / "modal_runner_lib.py"
        target.parent.mkdir(parents=True)
        target.write_text("".join(lines[:19]) + text + "".join(lines[19:]), encoding="utf-8")
        return target.parents[1]

    decoy = plant("try:\n    import modal\nexcept ImportError:\n    modal = None\n", "decoy")
    scanned, violations = _nonstdlib_module_scope_imports(decoy)
    assert scanned == [
        "scripts/modal_runner_lib.py"
    ], (f"the knock-out did not reach the planted copy, so its red below would be "
        f"unrelated to the plant. Scanned: {scanned}")
    assert violations == [
        ("scripts/modal_runner_lib.py", 21, "modal")
    ], ("gate (e) missed a try-wrapped module-scope `import modal` -- the exact shape an "
        "optional dependency gets written in, and the one that makes importing the runner "
        f"library cost a Modal import. Reported: {violations}")

    _, blind = _nonstdlib_module_scope_imports(decoy, top_level_only=True)
    assert blind == [], (
        "`tree.body` was supposed to be BLIND to this plant, and it is the blindness this "
        "gate exists to rule out. If it now sees the plant, the two instruments no longer "
        "differ anywhere measurable and this knock-out has stopped certifying that gate "
        f"(e) walks past depth 1. tree.body reported: {blind}")

    for slug, text, expected, clause in [
        ("asname", "import modal as _modal_shim\n", [("scripts/modal_runner_lib.py", 20, "modal")],
         "resolve `Import` through `alias.name`, never `alias.asname`"),
        ("fromimport", "from modal.functions import FunctionCall\n", [
            ("scripts/modal_runner_lib.py", 20, "modal")
        ], "`ImportFrom` counts, and its module resolves to the ROOT, not the dotted path"),
        ("denylist", "import numpy\n", [("scripts/modal_runner_lib.py", 20, "numpy")],
         "the check is an allow-list over `sys.stdlib_module_names`, not a deny-list"),
        ("two", "import numpy\nimport modal\n", [("scripts/modal_runner_lib.py", 20, "numpy"),
                                                 ("scripts/modal_runner_lib.py", 21, "modal")],
         "EVERY offending import is reported, sorted. The walk pops its stack from the "
         "end, so an unsorted return hands these back in reverse source order and a "
         "report-only-the-first implementation drops the second -- and this is the only "
         "row with two violations in one file, so nothing else can see either mutant"),
        ("ifguard", 'if os.environ.get("CS2RL_MODAL"):\n    import modal\n', [
            ("scripts/modal_runner_lib.py", 21, "modal")
        ], "the walk descends through `If` -- gate (g) plants exactly this header"),
        ("classbody", "class _C:\n    import modal\n", [
            ("scripts/modal_runner_lib.py", 21, "modal")
        ], "a CLASS BODY EXECUTES at import time, so it is walked, not skipped"),
        ("classmethod", "class _C:\n    def m(self):\n        import torch\n        return torch\n",
         [], "walking class bodies must not reach METHOD bodies -- a method is a `FunctionDef` "
         "and does not run on import"),
        ("nestedcontainers", "class _C3:\n    try:\n        class _C4:\n            import modal\n"
         "    except ImportError:\n        pass\n", [("scripts/modal_runner_lib.py", 23, "modal")],
         "descent is a STACK and not a single expansion -- an import THREE containers "
         "deep still executes at import time. Every other container in this table is "
         "top-level, so a walk that expands depth 1 and then stops passes all of them"),
        ("typechecking", "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import modal\n",
         [], "`if TYPE_CHECKING:` NEVER runs, so it costs nothing at import and is exempt. The "
         "import inside is deliberately NON-stdlib: `import decimal` would pass for the wrong "
         "reason and certify nothing"),
        ("typecheckingattr", "import typing\nif typing.TYPE_CHECKING:\n    import modal\n", [],
         "the `typing.TYPE_CHECKING` spelling is the same exemption; matching only the bare "
         "`Name` leaves half the idiom reddening"),
        ("typecheckingelse",
         "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import modal\n"
         "else:\n    import numpy\n", [
             ("scripts/modal_runner_lib.py", 24, "numpy")
         ], "the `else:` branch of a `TYPE_CHECKING` block is exactly the branch that DOES run, so "
         "exempting the whole `If` node instead of just its body is a blind spot, not an "
         "exemption"),
        ("dotted", "import xml.etree.ElementTree\n", [],
         "a DOTTED `Import` resolves to its ROOT -- `sys.stdlib_module_names` holds "
         "top-level names only, so keeping the dotted path reddens the gate over stdlib"),
        ("lazy", "def _lazy():\n    import torch\n    return torch\n", [],
         "a function-body import is LEGAL; descending into it is what makes bare "
         "`ast.walk` red on the correct module"),
        ("asynclazy", "async def _lazy_async():\n    import torch\n    return torch\n", [],
         "`AsyncFunctionDef` is in the skip set for the same reason `FunctionDef` is; "
         "the module has 0 async defs today, so nothing else objects to dropping it"),
        ("relative", "from .paths import RUN_ROOT\n", [],
         "a relative import is intra-package, not a third-party dependency -- "
         "`node.level == 0`"),
        ("nametestguard", "_DEBUG = False\nif _DEBUG:\n    import modal\n", [
            ("scripts/modal_runner_lib.py", 22, "modal")
        ], "the `TYPE_CHECKING` exemption matches the NAME, not merely the node class -- a "
         "bare `Name` test that is not `TYPE_CHECKING` still runs, so the import under it "
         "still costs what it costs"),
        ("attrtestguard", "import os\nif os.name:\n    import modal\n", [
            ("scripts/modal_runner_lib.py", 22, "modal")
        ], "same clause on the attribute spelling: `x.TYPE_CHECKING` is exempt, any other "
         "attribute test is not -- the exemption reads `test.attr`, not `isinstance(test, "
         "ast.Attribute)`"),
    ]:
        scanned, violations = _nonstdlib_module_scope_imports(plant(text, slug))
        assert scanned == [
            "scripts/modal_runner_lib.py"
        ], (f"the {slug} plant was not read at all, so its verdict is vacuous: {scanned}")
        assert violations == expected, f"gate (e) is wrong about: {clause}. Got {violations}"

    # A root with no target at all. The evidence must say "I read nothing"
    # rather than raise, because the `scanned` half is the ONLY thing standing
    # between an empty violation list and a vacuous green -- and the assertion
    # above, which reads `scanned`, is meaningless unless an unread file really
    # does stay out of it. Dropping the helper's `is_file()` filter turns this
    # into a `FileNotFoundError`, which is the mutant this pins.
    empty = tmp_path / "noscripts"
    empty.mkdir()
    assert _nonstdlib_module_scope_imports(empty) == ([], []), (
        "gate (e) must report an unreadable target as an empty SCOPE, not as an empty "
        "verdict on a file it never opened.")


def _module_scope_shape_violations(tree):
    """Gate (g), criterion 13. `tree.body` may hold ONLY the module docstring,
    imports, at most one imports-only `if TYPE_CHECKING:` block, and declared
    symbols. Returns `(examined, violations)`.

    `examined` is how many `tree.body` statements the gate actually reached a
    verdict on. `violations` is a list of `(kind, lineno, reason)` triples in
    source order -- `kind` is the offending node's class name, so a red names the
    construct and the line rather than saying "the shape is wrong".

    WHY THIS GATE EXISTS AT ALL, and it is not "the module looks tidy today".
    It is the only criterion that constrains the shape of a module header. §5.1
    already measured this module as conforming, so building it here makes it a
    DETECTOR rather than a description: criterion 2's byte-identity constrains
    only RELOCATED segments, and W3b's eight `scripts/modal_runner/` module
    headers will be 100% new code. After the split this is the only thing
    standing between those headers and a module-scope `if
    importlib.util.find_spec('modal') is not None:` -- a header that is GREEN on
    criterion 3 (it never imports Modal, so the container-equivalent import
    succeeds), GREEN on criterion 5 (its body assigns, it does not import), and
    invisible to criterion 4 (which never reads module source at all).

    THE CENSUS IS THIS ALLOW-LIST'S DERIVATION, NOT THE ALLOW-LIST, and the
    difference is measurable rather than stylistic. Today's `tree.body` is 195
    nodes:

        FunctionDef 77 | Assign 69 | ClassDef 23 | Import 17 | ImportFrom 6
        AnnAssign 2 | Expr 1        module-scope `If` nodes: []

    A gate whose rule is "allow every kind in that table" allows `Expr`, because
    the docstring is one. Measured: plant `logging.basicConfig(level=logging.INFO)`
    after line 19 and the census becomes `Import 18 | Expr 2 | ...` with **no `If`
    at all** -- so a kind-only gate is GREEN on a bare module-scope call, which is
    a far likelier mistake in a brand-new W3b header than the `find_spec` probe
    is. Hence the two shape branches:

      * `Expr` is legal at `body[0]` and only if its value is a `Constant`
        holding a `str` -- i.e. the module docstring, and nothing else, ever. An
        f-string in that position is a `JoinedStr` and is NOT a docstring, so it
        reddens, correctly.
      * `If` is legal only as `if TYPE_CHECKING:`, only with an imports-only
        body, and only once per module.

    Measured again, and this is why the kind census cannot be the instrument:
    `if TYPE_CHECKING: import decimal` and `if TYPE_CHECKING: import decimal;
    _X = 1` produce **byte-identical censuses** (`ImportFrom 7 | If 1 | ...`).
    One is legal and one is not. Nothing at the kind level can tell them apart.

    HOW A `TYPE_CHECKING` BLOCK IS RECOGNISED, and it is the same rule gate (e)
    uses on purpose: a bare `Name` spelled `TYPE_CHECKING` or ANY `Attribute`
    whose `attr` is `TYPE_CHECKING`. Both spellings of the real idiom are live
    (`from typing import TYPE_CHECKING` and `import typing`), and resolving the
    binding properly would need an import table. The two gates MUST agree about
    which construct this is -- if (g) recognised a narrower set, the same block
    would be "exempt" to (e) and "not a `TYPE_CHECKING` block" to (g), and a
    reader meeting the red would have no way to attribute it. Ruling 29 records
    the accepted limit: `if some_module.TYPE_CHECKING:` is recognised too. The
    other direction fails CLOSED -- `if TYPE_CHECKING and X:` is a `BoolOp`, is
    not recognised, and reddens here.

    THREE LIMITS, STATED RATHER THAN HIDDEN, because an unstated limit reads as a
    bug to whoever next meets it:

      1. The `else:` branch of a `TYPE_CHECKING` block is NOT examined, AND
         NOTHING ELSE EXAMINES IT EITHER. Criterion 13's text governs "a
         `TYPE_CHECKING` block whose body is imports only", and `else:` is not
         that body; the common `else: modal = None` fallback is a declaration and
         legal. Gate (e) walks that branch -- its `typecheckingelse` row is the
         assertion -- but FOR IMPORTS ONLY, which is the whole of what criterion 5
         asks. Anything else there is caught by no gate on this branch. MEASURED,
         on a plant spliced after line 19 of the real module:

             import logging
             from typing import TYPE_CHECKING
             if TYPE_CHECKING:
                 import decimal
             else:
                 logging.basicConfig(level=logging.INFO)

             gate (g): examined == len(tree.body), violations == []
             gate (e): scanned == ['scripts/modal_runner_lib.py'], violations == []

         Gate (e)'s green is not vacuous -- `scanned` names the planted file, so
         it read it and had nothing to say. And the POSITIVE CONTROL is the same
         call one level up: as a bare module-scope `Expr` it is
         `('Expr', 21, ...)` RED here, so what the `else:` buys is the
         indentation, not an inert gate. That is knock-out 2 row 1's exact defect
         -- a module-scope call that runs on every import -- surviving one
         indentation level down. Declining an
         `else:` clause HERE is still right: the plan's instrument table says
         "every statement in its body", and an unexercised rule risks false reds
         on W3b's own headers. So read this as a GAP, not as a delegation. W3b /
         B12 candidate; a sentence that sends the reader to criterion 5 sends them
         somewhere that does not have it.
      2. The final `elif`'s declaration tuple rejects every other statement kind,
         but only ONE of them ships an observation: a module-scope `try:`. That is
         the deliberate choice, not an oversight -- `try: import X / except
         ImportError:` is the shape an optional dependency actually gets written
         in, and it is the only non-`If`, non-`Expr` block plausible in a module
         header. `With` / `For` / `While` / `Match` at module scope have no row,
         so a mutant that adds one of THOSE to that tuple survives this file.
         Named so a later reader adds a row rather than assuming one exists.
      3. THE GATE NEVER ENTERS AN EXPRESSION POSITION, and this is the largest of
         the three. It classifies `tree.body` statements by KIND; it does not look
         inside an `Assign` / `AnnAssign` value, a decorator, or a parameter
         default. MEASURED, and it is the damning shape rather than a corner case:

             _ON_CONTAINER = importlib.util.find_spec('modal') is not None
                 gate (g): examined == len(tree.body), violations == []
                 gate (e): scanned == ['scripts/modal_runner_lib.py'],
                           violations == []

         That is the one-line ASSIGNMENT form of the very header this docstring
         names above as the gate's entire reason to exist. As an `Assign` it is a
         declared symbol, so it is legal here; criterion 5 is measured green above
         and criterion 3 is green for the reason the `if` form is (it never
         imports Modal, so a container-equivalent import succeeds), while
         criterion 4 never reads module source at all. So this is a module-scope
         failure all four gates on this branch miss TOGETHER.
         WHY IT IS DEFERRED RATHER THAN MISSED: spec §10 criterion 13 reads "no
         assignment that is not a declared symbol", with the instrument named as
         `tree.body` PLUS §5.1's manifest. The manifest half is what would decide
         whether `_ON_CONTAINER` is DECLARED, and §5.1's manifest does not exist
         pre-split -- there is no right-hand side to check a name against, the
         same reason Ruling 6 split criterion 4 and Ruling 8 moved criterion 14.
         The deferral is correct. Silence about it is not: do not read "criterion
         13 exists" as "criterion 13 is done". W3b / B12 owes this half.

    GC1: the subject is a PARAMETER and the return is evidence. The helper never
    reads `ROOT`, never parses a path of its own and never asserts -- every
    knock-out below hands it a tree built from a `tmp_path` splice, and a helper
    that reached for the real module would be green on all of them no matter what
    it did.

    WHY `examined` RIDES ALONG (GC1/Ruling 28: evidence may be richer where the
    extra half is load-bearing). Every clause of this gate is INERT on the
    unmodified subject -- it has zero module-scope `If`, zero non-docstring
    `Expr` and zero disallowed kinds -- so the green test on the real module is
    almost entirely a statement about what the gate did NOT find. `examined`
    is the one half of that green with teeth: a gate that inspected
    `tree.body[:20]` and stopped returns `[]` from a 195-node module and is
    indistinguishable from a correct one, because every plant GC2 permits lands
    at index 2 (right after the docstring and `from __future__ import
    annotations`) and would be caught by such a gate anyway. `examined ==
    len(tree.body)` is what the green test asserts, and it does not rot the way a
    literal 195 would.

    THE LIMIT OF THAT, said plainly: `examined` is counted INSIDE the loop, so it
    catches a loop that stops early. It cannot catch a mutant that returns
    `len(tree.body)` while iterating something else. No evidence shape defends
    against a helper that lies about its own scope; `scanned` in gate (e) has the
    identical hole.
    """
    examined = 0
    violations = []
    type_checking_blocks = 0
    for index, node in enumerate(tree.body):
        examined += 1
        kind = type(node).__name__
        if isinstance(node, ast.Expr):
            # The docstring, and only the docstring. `index == 0` is the whole
            # rule: a string literal anywhere else at module scope is either
            # dead or a comment somebody wrote wrong, and a non-string `Expr`
            # is a bare CALL, which is the case the kind census hides.
            if index == 0 and isinstance(node.value, ast.Constant) and isinstance(
                    node.value.value, str):
                continue
            violations.append(
                (kind, node.lineno, "only the module docstring may be a bare expression at module "
                 "scope"))
        elif isinstance(node, ast.If):
            test = node.test
            if not ((isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or
                    (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")):
                violations.append(
                    (kind, node.lineno, "the only conditional allowed at module scope is "
                     "`if TYPE_CHECKING:`"))
                continue
            type_checking_blocks += 1
            if type_checking_blocks > 1:
                violations.append(
                    (kind, node.lineno, "a module may hold at most one `if TYPE_CHECKING:` block"))
            # The offending BODY statement is what gets named, not the `If`.
            # Whoever meets this red has to be told which line to delete, and
            # "the block is wrong" does not say that.
            for inner in node.body:
                if not isinstance(inner, (ast.Import, ast.ImportFrom)):
                    violations.append((type(inner).__name__, inner.lineno,
                                       "an `if TYPE_CHECKING:` body may hold imports only"))
        # Everything else at module scope is an import or a declaration, or it
        # is a violation. The tuple is INLINED rather than hoisted to a
        # module-level constant because GC3 pins this task to six module-level
        # names and a seventh would break the pre-registered budget -- the same
        # trade gate (e) made with its `TYPE_CHECKING` predicate. `Expr` and
        # `If` are deliberately absent: both are legal in exactly one shape, so
        # they are decided by the branches above. Putting either here is the
        # blind kind-only allow-list this whole gate exists to rule out.
        elif not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                   ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            violations.append(
                (kind, node.lineno, "only imports and declarations may appear at module scope"))
    return examined, violations


def test_gate_g_criterion_13_the_runner_library_declares_and_does_nothing_else():
    """GREEN ON THE REAL SUBJECT. Nothing at `scripts/modal_runner_lib.py`'s
    module scope may do anything except declare.

    §5.1 measured this module as conforming before the gate existed, which is
    exactly why the four knock-outs below are not optional: on this subject the
    correct instrument and a blind kind-only allow-list return the identical
    `(195, [])`. The plants are the ONLY observations that separate them.

    THE SCOPE ASSERTION COMES FIRST and is not decoration. Every clause of this
    gate is inert here -- 0 module-scope `If`, 0 non-docstring `Expr`, 0
    disallowed kinds -- so `violations == []` on its own is a statement about
    absence, and absence is what a gate that looked at nothing also reports.
    `examined == len(tree.body)` is the half that says the gate read the module.

    THE FLOOR IS A FLOOR, DELIBERATELY NOT A PIN. `examined > 50` catches the
    vacuous cases (an empty or truncated subject) without pinning today's 195,
    which would redden every time somebody legitimately adds a function to the
    runner library -- and a gate that reddens on legal code gets switched off,
    which is the failure mode this whole file is built around.

    W3B OBLIGATION, the same one gate (e) carries: this test names
    `scripts/modal_runner_lib.py` as a literal. The split replaces that single
    module with eight `scripts/modal_runner/` submodules, and this test must
    become an enumeration over them -- each of whose headers is 100% new code
    and is constrained by nothing else in the spec. Swapping the literal is a
    B12 deliverable, not an optional tidy-up.
    """
    tree = ast.parse((ROOT / "scripts" / "modal_runner_lib.py").read_text(encoding="utf-8"))
    examined, violations = _module_scope_shape_violations(tree)
    assert examined == len(tree.body), (
        "gate (g) did not reach a verdict on every module-scope statement, so the empty "
        f"violation list below is a verdict on a prefix. examined={examined}, "
        f"len(tree.body)={len(tree.body)}")
    assert examined > 50, (
        "gate (g) examined a module-scope body far smaller than the runner library's, so it "
        "is almost certainly looking at the wrong file or a truncated one. This is a FLOOR, "
        f"not a pin -- do not turn it into today's exact count. examined={examined}")
    assert violations == [], (
        "`scripts/modal_runner_lib.py` now does something at module scope other than declare. "
        "Everything here runs on every import of the module, and criterion 13 is the only "
        f"criterion that constrains a module header's shape: {violations}")


def test_criterion_13_reddens_on_the_conditional_modal_probe(tmp_path):
    """KNOCK-OUT 1, and the criterion-13 half of a two-file pair.

    THE PLANT IS THE HEADER EVERY OTHER GATE IN THIS PLAN IS GREEN ON:

        import importlib.util
        if importlib.util.find_spec('modal') is not None:
            _ON_CONTAINER = True
        else:
            _ON_CONTAINER = False

    It never imports Modal, so a container-equivalent import of the package
    succeeds and criterion 3 stays green. Its body assigns rather than imports,
    so criterion 5 stays green -- and note WHY, because it is not the obvious
    reason: this is an ORDINARY `If`, so gate (e) DESCENDS THROUGH IT and does
    read the imports inside. It is green because there are none, not because of
    the `TYPE_CHECKING` exemption (Ruling 30). Move an `import` inside this
    header and gate (e) reddens CORRECTLY -- that import runs -- and the red is a
    plant-design error here, not a gate (e) defect.

    BOTH HALVES OF THE CONTRAST ARE ASSERTED, NOT DESCRIBED. Criterion 5's half
    is the second block below. Criterion 3's half is a shipped test in the other
    file:

        tests/test_modal_client.py::test_gate_a_criterion_3_stays_green_on_a_conditional_modal_probe

    which names this test by node id in return. It lives there rather than here
    because `tests/test_modal_packaging.py` imports zero `tests.*` modules and
    that exclusion is load-bearing -- `_seam_sources()` omits this file because
    self-feeding the classifier returns a poisoned destination count -- so
    duplicating that gate's subprocess helper into this file would put two copies
    of one gate in the tree, which is the defect class this branch exists to
    close. Neither half is optional: together they establish that criterion 13 is
    the SOLE objector to this header.

    AND THE THIRD BLOCK HOLDS THE AGREEMENT THE TWO HELPERS' DOCSTRINGS DEMAND.
    Both recognise a `TYPE_CHECKING` block with the SAME rule, written out twice
    -- once in gate (e)'s descent, once in the `If` branch of gate (g) -- and gate
    (g)'s docstring says the two "MUST agree", which until now nothing enforced.
    Measured: narrow gate (e)'s attribute clause to require `typing.*` and the two
    disagree on `if os.TYPE_CHECKING:` while all 7 tests that call either helper
    still pass. Extracting one shared predicate would cost a module-level name
    against GC3's pinned budget -- the trade both helpers already state -- so the
    agreement is ASSERTED over the shapes that separate the readings instead.
    `os.TYPE_CHECKING` is the SOLE OBJECTOR and it holds BOTH directions: seeding
    the narrowing into gate (e) makes that row disagree, seeding the same
    narrowing into gate (g) makes it disagree the other way, and the other four
    rows are blind to both. Delete that row and either mutant survives.

    CRITERION 4 IS DELIBERATELY NOT IN THE CONTRAST. Gate (d) reads
    `runner_image.local_files` and the git index and never reads module source at
    all, so "criterion 4 is green on this plant" would be green for a reason
    unrelated to the plant. A control that cannot see the subject is not evidence
    about it -- that is precisely the class this branch closes, and asserting it
    would dress a vacuous green as a contrast.
    """
    lines = (ROOT / "scripts" /
             "modal_runner_lib.py").read_text(encoding="utf-8").splitlines(keepends=True)
    assert lines[18].rstrip("\n") == "from __future__ import annotations", (
        "GC2 pins plants to line 19 because a `__future__` import must come first. That line "
        f"now reads {lines[18]!r}, so the plant below lands somewhere else and every line "
        "number in this test is about a different statement.")
    planted = ("".join(lines[:19]) + "import importlib.util\n"
               "if importlib.util.find_spec('modal') is not None:\n"
               "    _ON_CONTAINER = True\n"
               "else:\n"
               "    _ON_CONTAINER = False\n" + "".join(lines[19:]))

    tree = ast.parse(planted)
    examined, violations = _module_scope_shape_violations(tree)
    assert examined == len(tree.body), (
        f"gate (g) stopped short of the whole module scope: {examined} of {len(tree.body)}")
    assert violations == [
        ("If", 21, "the only conditional allowed at module scope is `if TYPE_CHECKING:`")
    ], ("criterion 13 missed the `find_spec` header -- the one module-scope construct that is "
        "green on criteria 3, 4 and 5, and which W3b's eight brand-new module headers have "
        f"nothing else stopping them from acquiring. Reported: {violations}")

    # The criterion-5 half of the contrast, on the SAME planted source. Gate (e)
    # takes a repo root, so the plant is materialised rather than parsed.
    target = tmp_path / "scripts" / "modal_runner_lib.py"
    target.parent.mkdir(parents=True)
    target.write_text(planted, encoding="utf-8")
    scanned, impure = _nonstdlib_module_scope_imports(tmp_path)
    assert scanned == [
        "scripts/modal_runner_lib.py"
    ], (f"gate (e) never read the planted copy, so its green below is vacuous: {scanned}")
    assert impure == [], (
        "gate (e) was expected to stay GREEN on this plant -- that is the whole contrast. A red "
        "here means the plant acquired an import that executes, which would be a correct gate "
        f"(e) verdict and a plant-design error on this side (Ruling 30). Reported: {impure}")

    # The two gates' `TYPE_CHECKING` predicates must agree, and this is what says
    # so. Each row plants `if <shape>:` holding a NON-stdlib import, then asks
    # both helpers the same question: gate (e) EXEMPTS the body iff it recognises
    # the block, gate (g) reports no conditional violation iff it recognises the
    # block. The verdicts are read off the helpers, never hardcoded, so the row
    # objects no matter WHICH gate is the one that moved. `recognised` is asserted
    # as well as the agreement, because two gates that both stopped recognising
    # anything would agree vacuously.
    for shape, recognised, why in [
        ("TYPE_CHECKING", True, "the bare `Name` spelling"),
        ("typing.TYPE_CHECKING", True, "the `typing.` attribute spelling"),
        ("os.TYPE_CHECKING", True, "ANY attribute whose `attr` is `TYPE_CHECKING` -- Ruling 29's "
         "accepted limit, and the ONLY row here that separates the two readings"),
        ("_probe()", False, "a `Call` test is not recognised by either gate"),
        ("TYPE_CHECKING is True", False, "a `Compare` test is not recognised by either gate"),
    ]:
        slug = "agree_" + shape.replace(".", "_").replace("(", "").replace(")", "").replace(
            " ", "_")
        block = "".join(lines[:19]) + f"if {shape}:\n    import modal\n" + "".join(lines[19:])
        agree_root = tmp_path / slug / "scripts"
        agree_root.mkdir(parents=True)
        (agree_root / "modal_runner_lib.py").write_text(block, encoding="utf-8")
        agree_scanned, agree_impure = _nonstdlib_module_scope_imports(agree_root.parent)
        assert agree_scanned == [
            "scripts/modal_runner_lib.py"
        ], (f"the {slug} plant was not read, so gate (e)'s verdict on it is vacuous: "
            f"{agree_scanned}")
        _, agree_shape = _module_scope_shape_violations(ast.parse(block))
        e_exempts = agree_impure == []
        g_recognises = agree_shape == []
        assert e_exempts == recognised and g_recognises == recognised, (
            f"the two gates no longer read `if {shape}:` the same way, or no longer read it as "
            f"{recognised}. This is {why}. Gate (g)'s docstring requires them to agree: a block "
            "one gate exempts and the other calls illegal leaves whoever meets the red with no "
            f"way to attribute it. gate (e) exempted: {e_exempts} ({agree_impure}); gate (g) "
            f"recognised: {g_recognises} ({agree_shape})")


def test_gate_g_criterion_13_reddens_on_module_scope_work_the_kind_census_hides():
    """KNOCK-OUT 2. The two shapes a kind-only allow-list is green on.

    ROW 1 -- A BARE MODULE-SCOPE CALL. `logging.basicConfig(level=logging.INFO)`
    adds NO `If` node; measured, the census goes to `Import 18 | Expr 2 | ...`
    and every other kind is unchanged. A gate that allows the kind `Expr` because
    `body[0]` is one passes this, passes the real module, and passes every other
    check in this plan. Of the four defect shapes this task covers, it is the
    likeliest to appear in a real W3b header -- somebody configures logging, or
    calls `os.environ.setdefault`, at the top of a new module -- and it is the
    one nothing else in the branch can see.

    ROW 2 -- A MODULE-SCOPE `try:`. This is the only observation holding the
    helper's final `elif` -- its declaration tuple -- at all: drop that branch, or
    add `ast.Try` to the tuple, and nothing else in this file objects. `try: import
    X / except ImportError:` is the shape an optional dependency actually gets
    written in, which is why this kind and not `With` / `For` / `While` gets the
    row. The plant's body ASSIGNS rather than imports on purpose -- an `import
    modal` in there would make it gate (e)'s decoy, and then a red could be
    either gate's and the row would certify neither.

    Note both rows land at a DIFFERENT line (21 and 20) because row 1 needs a
    preceding `import logging`. Asserting the line is what makes these knock-outs
    say "this statement", and a plant whose line number was not re-derived is the
    common way that stops being true.
    """
    lines = (ROOT / "scripts" /
             "modal_runner_lib.py").read_text(encoding="utf-8").splitlines(keepends=True)
    assert lines[18].rstrip("\n") == "from __future__ import annotations", (
        f"GC2 pins plants to line 19; that line now reads {lines[18]!r}.")

    def plant(text):
        """Parse the real module with `text` spliced in after line 19. Repeated
        in each knock-out rather than hoisted to a module-level helper because
        GC3 pins this task to six module-level names and a seventh would break
        the pre-registered budget; nothing here is ever executed, only parsed."""
        return ast.parse("".join(lines[:19]) + text + "".join(lines[19:]))

    for text, expected, clause in [
        ("import logging\nlogging.basicConfig(level=logging.INFO)\n", [
            ("Expr", 21, "only the module docstring may be a bare expression at module scope")
        ], "a bare CALL at module scope adds no `If` and one more `Expr`, so a gate that "
         "allows the kind `Expr` because the docstring is one is green on it"),
        ("try:\n    _X = 1\nexcept Exception:\n    _X = 2\n", [
            ("Try", 20, "only imports and declarations may appear at module scope")
        ], "a module-scope `try:` is neither an import nor a declaration, and it is the "
         "only row holding the helper's final `elif` and its declaration tuple"),
    ]:
        tree = plant(text)
        examined, violations = _module_scope_shape_violations(tree)
        assert examined == len(tree.body), (
            f"gate (g) stopped short of the whole module scope: {examined} of {len(tree.body)}")
        assert violations == expected, f"criterion 13 is wrong about: {clause}. Got {violations}"


def test_gate_g_criterion_13_reddens_on_a_type_checking_body_that_is_not_imports_only():
    """KNOCK-OUT 3. A `TYPE_CHECKING` block is exempt from criterion 5, not from
    criterion 13 -- and the kind census cannot tell a legal one from this.

    MEASURED, and it is the sharpest evidence in this task that a kind-level
    instrument is the wrong one:

        if TYPE_CHECKING:            ->  ImportFrom 7 | If 1 | Import 17 | ...
            import decimal

        if TYPE_CHECKING:            ->  ImportFrom 7 | If 1 | Import 17 | ...
            import decimal
            _X = 1

    Byte-identical censuses. One is legal and one is not, and the difference is
    one statement INSIDE the block, which no count of module-scope kinds reaches.
    The knock-out's counterpart -- the legal block going green -- is
    `test_gate_g_criterion_13_allows_exactly_one_type_checking_block_recognised_by_name`;
    without that pair a gate could reject the construct outright and pass this.

    THE VIOLATION NAMES THE INNER STATEMENT (`Assign`, line 23), not the `If` at
    line 21. Whoever meets this red has to be told which line to move out of the
    block; "the block is malformed" does not say that.

    WHY THIS SHAPE IS WORTH A GATE: `if TYPE_CHECKING:` bodies never execute, so
    a non-import in one is dead code that a reader nonetheless takes for a live
    declaration -- and gate (e) will never object, because its whole rule is that
    the block costs nothing at import. Criterion 13 polices the construct;
    criterion 5 declines to look inside it. Complementary, and neither subsumes
    the other (Ruling 33).
    """
    lines = (ROOT / "scripts" /
             "modal_runner_lib.py").read_text(encoding="utf-8").splitlines(keepends=True)
    assert lines[18].rstrip("\n") == "from __future__ import annotations", (
        f"GC2 pins plants to line 19; that line now reads {lines[18]!r}.")
    tree = ast.parse("".join(lines[:19]) + "from typing import TYPE_CHECKING\n"
                     "if TYPE_CHECKING:\n"
                     "    import decimal\n"
                     "    _X = 1\n" + "".join(lines[19:]))

    examined, violations = _module_scope_shape_violations(tree)
    assert examined == len(tree.body), (
        f"gate (g) stopped short of the whole module scope: {examined} of {len(tree.body)}")
    assert violations == [
        ("Assign", 23, "an `if TYPE_CHECKING:` body may hold imports only")
    ], ("criterion 13 accepted a `TYPE_CHECKING` block carrying a statement that is not an "
        "import. It must name the OFFENDING BODY STATEMENT -- `Assign` at line 23 -- not the "
        f"`If` at line 21, or the red does not say which line to move. Reported: {violations}")


def test_gate_g_criterion_13_allows_exactly_one_type_checking_block_recognised_by_name():
    """KNOCK-OUT 4 plus the positive control the other three depend on.

    ROW 1 IS THE GREEN HALF THE WHOLE TASK RESTS ON. Today's module has ZERO
    module-scope `If` nodes, so a gate that simply rejects every one of them is
    observationally identical to a correct gate on the real subject and on all
    three red knock-outs -- and would false-redden on exactly the
    `TYPE_CHECKING` blocks W3b's eight new module headers will want, at which
    point somebody deletes the gate. This row is the only thing in the suite
    that says the construct is ALLOWED.

    ROWS 2 AND 3 HOLD THE `at most one` CLAUSE AND THE NAME CLAUSE, which are
    secondary clauses of the same branch -- and "the headline clause has a
    knock-out, the secondary clause has none" is this branch's named defect class
    in its B2 form: the guard's own allow-set going unwatched. Ruling 31 measured
    it on gate (e), where three mutants that kept the node-class test and dropped
    the NAME test survived the entire file. The same mutants live here, and
    knock-out 1's plant cannot see them -- its `If` test is a `Compare`
    (`find_spec(...) is not None`), which is neither a `Name` nor an `Attribute`,
    so `isinstance(test, (ast.Name, ast.Attribute))` passes it. Rows 3 and 4 are
    the bare-`Name` and plain-`Attribute` cases that do object.

    ROW 5 IS THE OTHER HALF OF THE NAME CLAUSE, and it is a GREEN one:
    `if typing.TYPE_CHECKING:` is the same exemption gate (e) grants, matched on
    `test.attr` rather than on the node being an `Attribute`. Both gates must
    agree about which construct this IS -- if criterion 13 recognised a narrower
    set than criterion 5 exempts, the same block would be "exempt" to one and
    "not a `TYPE_CHECKING` block" to the other, and whoever met the red could not
    attribute it. Ruling 29 records the accepted limit that comes with matching
    by name: `if some_module.TYPE_CHECKING:` is recognised too. It fails CLOSED
    in the other direction -- `if TYPE_CHECKING and X:` is a `BoolOp` and
    reddens here, which is row 6.

    WHAT NONE OF THESE ROWS COVERS, said so a later reader adds one rather than
    assuming: the `else:` branch of a `TYPE_CHECKING` block is not examined by
    this gate at all. `else: modal = None` is a declaration and legal; what
    RUNNING that branch costs is criterion 5's question, and gate (e)'s
    `typecheckingelse` row is the assertion for it.
    """
    lines = (ROOT / "scripts" /
             "modal_runner_lib.py").read_text(encoding="utf-8").splitlines(keepends=True)
    assert lines[18].rstrip("\n") == "from __future__ import annotations", (
        f"GC2 pins plants to line 19; that line now reads {lines[18]!r}.")

    def plant(text):
        """Parse the real module with `text` spliced in after line 19."""
        return ast.parse("".join(lines[:19]) + text + "".join(lines[19:]))

    for text, expected, clause in [
        ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import decimal\n", [],
         "a single imports-only `if TYPE_CHECKING:` block is LEGAL -- a gate that rejects "
         "every module-scope `If` passes every other observation in this task and "
         "false-reddens on the headers W3b is about to write"),
        ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import decimal\n"
         "if TYPE_CHECKING:\n    import fractions\n", [
             ("If", 23, "a module may hold at most one `if TYPE_CHECKING:` block")
         ], "at most ONE such block per module; the second is what this row holds, and "
         "nothing else in the file has two"),
        ("_DEBUG = False\nif _DEBUG:\n    import decimal\n", [
            ("If", 21, "the only conditional allowed at module scope is `if TYPE_CHECKING:`")
        ], "the recognition matches the NAME, not merely the node class -- drop "
         "`test.id == \"TYPE_CHECKING\"` and every flag-guarded module-scope block becomes "
         "legal shape"),
        ("import os\nif os.name:\n    import decimal\n", [
            ("If", 21, "the only conditional allowed at module scope is `if TYPE_CHECKING:`")
        ], "same clause on the attribute spelling: the rule reads `test.attr`, not "
         "`isinstance(test, ast.Attribute)`"),
        ("import typing\nif typing.TYPE_CHECKING:\n    import decimal\n", [],
         "`typing.TYPE_CHECKING` is the same construct and the same exemption gate (e) "
         "grants; recognising only the bare `Name` leaves half the real idiom reddening"),
        ("from typing import TYPE_CHECKING\nif TYPE_CHECKING and os.name:\n"
         "    import decimal\n", [
             ("If", 21, "the only conditional allowed at module scope is `if TYPE_CHECKING:`")
         ], "`if TYPE_CHECKING and X:` is a `BoolOp`, not a recognised `TYPE_CHECKING` test. "
         "The name-based rule fails CLOSED in this direction, which is the direction it "
         "should fail in"),
    ]:
        tree = plant(text)
        examined, violations = _module_scope_shape_violations(tree)
        assert examined == len(tree.body), (
            f"gate (g) stopped short of the whole module scope: {examined} of {len(tree.body)}")
        assert violations == expected, f"criterion 13 is wrong about: {clause}. Got {violations}"
