"""Guards for runner import spelling, runtime identity and package structure.

The static walker descends into function bodies and recognizes literal dynamic
imports. Runtime evidence observes executed imports, including variable-driven
ones; it does not observe imports in child processes. The carrier pass check
prevents an unreached in-body import from producing a vacuous clean census.
"""
import ast
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

# BINDING_SITES is a data dict, and the campaign module imports only importlib
# and typing at module scope (it reaches the runner package only inside
# `binding_target`, from a formatted string). So this import pulls in no runner
# module and keeps this file out of `_IMPORTERS`. The reach floor reads it to
# resolve `binding_target("key")` to the key's owning module.
from tests.modal_patch_binding_campaign import BINDING_SITES

# `MANIFEST` is renamed on import because this file's own `MANIFEST` is the seam
# manifest's path; the tables' one is {"<module>.py": [the names it owns]}.
from tests.modal_runner_tables import MANIFEST as RUNNER_OWNERS
from tests.modal_runner_tables import (
    RUNNER_MODULES,
    RUNNER_PATHS,
    RUNNER_TEST_FILES,
    TABLES,
)

ROOT = Path(__file__).resolve().parents[1]
BARE = "modal_runner"
# The one legal spelling. `BARE` is what must never appear; this is what must
# appear instead. The runtime probe requires the package and every submodule
# in `sys.modules` under this spelling and no other.
PACKAGED = "scripts." + BARE

# Explicit importer population, independently checked against the tracked-file
# AST census below. Shared helpers belong even when they collect no tests.
# New files must be staged before the tracked census can observe them.
_IMPORTERS = [
    "tests/test_modal_argv.py",
    "tests/test_modal_protocol.py",
    "tests/test_modal_runner.py",
    "tests/test_modal_client.py",
    "tests/modal_test_helpers.py",
    "tests/test_eval_baselines.py",
    "tests/test_modal_patch_bindings.py",
]

# The test that carries the HARD case: the bare import used to live inside this
# body as its second-to-last statement (W1 converted it; it is now
# `from scripts.modal_runner import request`, still in-body and still
# second-to-last). If this test does not reach its end, the module census below
# is measuring a session that never executed the line the gate exists for.
_INBODY_CARRIER = ("tests/test_eval_baselines.py::test_eval_interval_cli_config_and_modal_mirror")

# Tests the nested session skips. Each runs its instrument in a CHILD process
# (the binding campaign's driver runs 70 pytest child sessions; each rejection
# case runs one Python child), whose `sys.modules` the probe cannot see, so they
# add wall time and no identity evidence. Their file stays on the argv:
# collecting it still imports the runner at its module scope, which the probe
# does see. pytest's `--deselect` is a node-id PREFIX match
# (`nodeid.startswith`), so the first entry already covers the second and would
# silently cover any future `test_patch_binding_campaign*` sibling too; and a
# `--deselect` that matches nothing is ignored. The gate therefore checks, for
# each entry, that it is still defined AND that every module-level `def` or
# `async def` it prefix-matches is listed here, so no such function leaves the
# nested session unnamed. A prefix-matching test bound by assignment or import,
# or defined under a module-level `if` or `try`, is not seen.
_NESTED_DESELECT = (
    "tests/test_modal_patch_bindings.py::test_patch_binding_campaign",
    "tests/test_modal_patch_bindings.py::test_patch_binding_campaign_rejects_invalid_evidence",
)

# Wall-clock ceiling for the nested session, in seconds. Named rather than
# inlined so the value and the message that quotes it cannot drift apart -- a
# literal in both places is one edit away from a message that lies about its own
# threshold. Chosen from measured spread, not picked round: on 2026-09-23, with
# the runner split into scripts/modal_runner/ and `_NESTED_DESELECT` applied,
# three nested sessions reported 49.34-50.17s ("467 passed, 24 deselected") and
# this test's call phase took 51.06-51.90s, so 600 is ~11.5x the slowest observed.
# The figures move with every test added to an `_IMPORTERS` file; re-measure
# before trusting the margin after `_IMPORTERS` or `_NESTED_DESELECT` changes.
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

    Counts, with the unit, because a bare number invites the wrong comparison
    (each is `git ls-files --cached "*.py" | wc -l` at the tree named): 131 at
    6c937ca (the spec's figure); 135 at cc228fa (plus this file, Task 2's
    `tests/_modal_import_probe.py`, and W2's `tests/test_modal_client.py` and
    `tests/modal_test_helpers.py`); 138 at 2bb32ac; 149 on the W3b split tree
    (measured 2026-09-23). NOTHING asserts on that total and nothing should --
    it moves with every added .py, and this sentence once read 133 while W2
    added two of them, which is the decay a stated total invites and the
    reason nothing may depend on one.
    `test_the_census_scans_the_whole_repo` pins the structural property
    instead, which does not move.

    `cwd=ROOT` is load-bearing: `git ls-files "*.py"` is CWD-RELATIVE, so
    running it from `tests/` returns only the tests (91 of 149 tracked paths on
    the W3b split tree, 2026-09-23; 86 of 135 at cc228fa) and the guard
    silently stops watching `scripts/` and `src/`. Pinning the CWD is what
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

    The FOURTH shape, `from scripts import modal_runner`, only exists when
    `target` is DOTTED, and it is the one this walker was missing. The
    `from-import` branch above matches on the module PATH, which is right for a
    top-level target and incomplete for a dotted one, because the leaf can be
    imported from its parent. Measured against the walker before this branch
    existed, when the runner was still `modal_runner_lib`: `from scripts
    import modal_runner_lib` returned `[]` under `PACKAGED` -- an ordinary
    idiom, invisible. That mattered because
    `test_the_runtime_probe_runs_every_test_file_that_imports_the_runner`
    promised Task 4 by name that it reddens when the import moves; had Task 4
    written this spelling, it would have stayed green and the runtime gate would
    have silently stopped covering the new file. W3 makes the idiom MORE likely
    by turning `modal_runner` into a package.

    The branch is unreachable for a top-level target: `BARE.rpartition(".")`
    yields an empty parent, and `if parent` gates it. Proven rather than
    asserted -- an instrumented copy that raises on entry never fired across the
    whole tracked census under `BARE`, and did fire under `PACKAGED`.

    Relative imports are NOT hits (`node.level` must be 0). `from
    .modal_runner import X` names a DIFFERENT module -- a sibling of the
    importing package -- and `tests/` is a real package, so treating it as a hit
    would redden the guard over a legal import. Measured before the level check
    existed, on the pre-W3b name: both `from .modal_runner_lib import
    ValidationError` and the two-dot form returned `[(1, 'from-import')]`.

    PITFALL -- known blind spots, named rather than hidden, and NOT claimed to
    be exhaustive:

    1. a dynamic import whose argument is a variable
       (`importlib.import_module(M)`) is invisible to any static scan.
       `test_modal_runner_resolves_to_exactly_one_module_object` is the
       runtime gate that covers it; that is why this repo has both and not just
       this one.
    2. an import written inside a subprocess CODE STRING is invisible to this
       walker -- it reads the string as a string -- and to the runtime probe,
       which reads `sys.modules` in the parent process and never sees a child's.
       This repo writes that shape twice in THIS file, over which the census
       reports no hit at all: `test_local_entrypoints_do_not_import_modal`,
       whose code string imports `src.train`, `scripts.exp_lib` and
       `scripts.run_experiment` but never the runner lib, and
       `test_modal_runner_does_not_import_modal_or_torch`, whose code
       string holds `import scripts.modal_runner` itself. Named rather
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
                # `from scripts import modal_runner [as mrl]`. The alias is
                # irrelevant -- `alias.name` is the imported name and
                # `alias.asname` is only what it is bound to locally, so the
                # `as` spelling needs no separate case. Exact equality on the
                # leaf, never startswith: `from scripts import
                # modal_runner_extra` is a different module.
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
    the scope, the PROSE is what changed. Measured on the W3b split tree
    (2026-09-23), all 91 tracked `.py` under `tests/` are top-level -- the only
    subdirectory is `tests/fixtures`, which holds no `.py` -- so the two
    readings agree by layout rather than by
    rule, which is exactly the kind of agreement that stops holding without
    warning.

    Said as a scope and not as a file kind because W2 made
    `modal_test_helpers.py` a non-test file that imports the runner AND a member
    of `_IMPORTERS`, so "non-test importers" stopped naming the `scripts/` three.
    Those three are listed in the comment at `_IMPORTERS`, which documents them
    rather than holding them.

    FileNotFoundError is deliberately NOT caught, for the reason spelled out in
    `test_no_file_imports_the_bare_modal_runner_spelling`: `--cached`
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


def test_no_file_imports_the_bare_modal_runner_spelling():
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
            # not an excuse. Measured: 0 of the 149 tracked .py on the W3b split
            # tree (2026-09-23) hit this branch, so it is dead; it is kept
            # because a tracked .py that stops parsing (a bad merge, a
            # py313-only syntax) would otherwise redden THIS guard for a reason
            # that has nothing to do with imports.
            continue
        if hits:
            offenders[path.relative_to(ROOT).as_posix()] = hits
    assert offenders == {}, ("import `scripts.modal_runner`, not `modal_runner` -- both "
                             "spellings in one process create two module objects whose exception "
                             f"classes are not identical. Offenders: {offenders}")


@pytest.mark.parametrize("planted, shape", [
    ("import modal_runner as mrl\n", "module scope"),
    ("from modal_runner import ValidationError\n", "module-scope from-import"),
    ("def f():\n    import modal_runner as mrl\n", "function body"),
    ("def f():\n    from modal_runner import ValidationError\n", "function-body from-import"),
    ('import importlib\nm = importlib.import_module("modal_runner")\n',
     "importlib.import_module literal"),
    ('def f():\n    return __import__("modal_runner")\n', "__import__ literal in a body"),
    ('import importlib\nm = importlib.import_module(name="modal_runner")\n',
     "importlib.import_module keyword"),
    ("import modal_runner.state\n", "submodule"),
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
    ("import scripts.modal_runner as mrl\n", "the CORRECT spelling this task converts TO"),
    ("from scripts import modal_runner\n", "the CORRECT spelling, imported from its parent"),
    ("from .modal_runner import ValidationError\n", "a relative import -- a different module"),
    ("import modal_runner_extra\n", "a name that merely starts with the bare one"),
])
def test_the_guard_does_not_fire_on_legal_lookalikes(legal, why):
    """NEGATIVE CONTROL. A guard that reddens on legal code gets switched off.

    Each row was a live false positive or a near miss, not a hypothetical:

    - the relative form returned `[(1, 'from-import')]` until `node.level == 0`
      was added, and `tests/` is a real package where such an import is legal;
    - the `startswith` in `_bare_name` is `BARE + "."`, not `BARE`; drop the dot
      and `modal_runner_extra` becomes a hit. This row is what objects;
    - the correct spelling must stay silent, or the guard fails the whole repo
      the moment W1's conversions land -- in BOTH its spellings, which is why
      `from scripts import modal_runner` is here. That row is the BARE half
      of the fourth shape: it must be a HIT under `PACKAGED` (the row of the
      same name in `test_the_walker_finds_the_packaged_spelling_in_every_shape`)
      and SILENT under `BARE`, and one direction without the other is what let
      the shape go missing in the first place.
    """
    assert bare_spelling_imports(legal) == [], f"false positive on {why}"


@pytest.mark.parametrize("planted, shape", [
    ("import scripts.modal_runner\n", "module scope"),
    ("import scripts.modal_runner as mrl\n", "module scope, aliased"),
    ("from scripts.modal_runner import ValidationError\n", "from-import on the full path"),
    ("def f():\n    import scripts.modal_runner as mrl\n", "function body"),
    ('import importlib\nm = importlib.import_module("scripts.modal_runner")\n',
     "importlib.import_module literal"),
    ("import scripts.modal_runner.state\n", "submodule"),
    ("from scripts import modal_runner\n", "from-parent-import -- THE SHAPE THAT WAS MISSING"),
    ("from scripts import modal_runner as mrl\n", "from-parent-import, aliased"),
    ("from scripts import modal_artifacts, modal_runner\n",
     "from-parent-import, one of several names"),
    ("def f():\n    from scripts import modal_runner\n", "from-parent-import in a body"),
])
def test_the_walker_finds_the_packaged_spelling_in_every_shape(planted, shape):
    """POSITIVE CONTROL for the walker aimed at `PACKAGED` -- the half that
    decides which files reach `_IMPORTERS`, and which had NO shape-level control
    before this round.

    Every row above `test_guard_detects_a_planted_bare_import` covers the `BARE`
    target only. `_test_files_importing(PACKAGED)` calls the same walker with a
    DOTTED target, where one shape behaves differently: the leaf can be imported
    from its parent. Review found `from scripts import modal_runner` (then
    `modal_runner_lib`) returning `[]`, so the client half W2's split created,
    tests/test_modal_client.py, could have moved its import into that idiom and
    never entered `_IMPORTERS`.

    Each row also asserts SILENCE under `BARE`. That is not padding -- it is the
    inertness half. A fix that made the new branch fire for a top-level target
    would redden `test_no_file_imports_the_bare_modal_runner_spelling` over
    the legal packaged spelling, i.e. over the entire repo after W1.
    """
    assert bare_spelling_imports(planted, PACKAGED), f"walker is blind to {shape} under PACKAGED"
    bare_hits = bare_spelling_imports(planted, BARE)
    assert bare_hits == [], (
        f"{shape} matched the BARE target; the packaged spelling is LEGAL and this "
        "would redden the repo-wide guard over correct code")


def test_import_guard_plant_populations_are_complete():
    populations = {
        "test_guard_detects_a_planted_bare_import": 8,
        "test_the_guard_does_not_fire_on_legal_lookalikes": 4,
        "test_the_walker_finds_the_packaged_spelling_in_every_shape": 10,
    }
    for name, expected in populations.items():
        marks = [mark for mark in globals()[name].pytestmark if mark.name == "parametrize"]
        assert len(marks) == 1, f"{name} must carry exactly one parametrize mark, not {len(marks)}"
        rows = len(marks[0].args[1])
        assert rows == expected, (
            f"{name} has {rows} rows, expected {expected}; if you added or removed a plant on "
            f"purpose, update populations[{name!r}] in this test "
            "(test_import_guard_plant_populations_are_complete)")


def test_runtime_identity_rejects_competing_module_objects(monkeypatch):
    """A second object must fail identity even with the same canonical names."""
    import importlib
    import types

    from tests._modal_import_probe import assert_module_identity, module_census

    package = importlib.import_module(PACKAGED)
    for _ in range(2):
        baseline = module_census()
        assert_module_identity(baseline)
        with monkeypatch.context() as patch:
            twin = types.ModuleType("modal_runner")
            twin.__file__ = package.__file__
            patch.setitem(sys.modules, "modal_runner", twin)
            planted = module_census()
            assert planted["modules"] == baseline["modules"]
            assert len({row["object_id"] for row in planted["identities"][PACKAGED]}) == 2
            with pytest.raises(AssertionError, match="module object identity"):
                assert_module_identity(planted)
        assert_module_identity(module_census())


def test_runtime_identity_requires_every_governed_module(monkeypatch, tmp_path):
    """The probe governs exactly the declared modules, and a missing one fails clause 1.

    The probe derives `_EXPECTED` from the package directory
    (`_governed_modules`), so this pins it to the modules MANIFEST
    declares: an empty set, which would pass every identity clause over
    nothing, or a module file the tables do not declare, reddens here. The
    planted directory shows the derivation governs a new module the moment
    its file exists and ignores anything whose name does not match `*.py`
    directly in the package directory: `sub/nested.py` pins the non-recursive glob,
    which an `rglob` would break.
    """
    import importlib

    from tests._modal_import_probe import (
        _EXPECTED,
        _PACKAGE_DIR,
        _governed_modules,
        assert_module_identity,
        module_census,
    )

    declared = {PACKAGED} | {f"{PACKAGED}.{name}" for name in RUNNER_MODULES}
    undeclared, missing = sorted(_EXPECTED - declared), sorted(declared - _EXPECTED)
    remedies = []
    if undeclared:
        remedies.append(f"On disk but undeclared: {undeclared}; declare each in {TABLES}, or "
                        "delete it.")
    if missing:
        remedies.append(f"Declared but not on disk: {missing}; restore the file, or remove the "
                        f"module from {TABLES}. If every declared module is missing, "
                        "`_PACKAGE_DIR` no longer points at scripts/modal_runner/.")
    assert not remedies, (
        f"tests/_modal_import_probe.py governs every entry matching *.py in {_PACKAGE_DIR}, "
        "and that must match the modules MANIFEST declares. " + " ".join(remedies))
    planted = tmp_path / "modal_runner"
    (planted / "__pycache__").mkdir(parents=True)
    (planted / "sub").mkdir()
    for name in ("__init__.py", "core.py", "ledger.py", "notes.txt", "__pycache__/core.pyc",
                 "sub/nested.py"):
        (planted / name).write_text("", encoding="utf-8")
    assert _governed_modules(planted) == {PACKAGED, f"{PACKAGED}.core", f"{PACKAGED}.ledger"}
    importlib.import_module(PACKAGED)
    with monkeypatch.context() as patch:
        patch.delitem(sys.modules, PACKAGED + ".training")
        with pytest.raises(AssertionError,
                           match=r"Never reached: \['scripts\.modal_runner\.training'\]"):
            assert_module_identity(module_census())
    assert_module_identity(module_census())


def test_runtime_identity_rejects_bare_alias_of_same_object(monkeypatch):
    """A bare spelling fails even when it aliases the canonical object."""
    import importlib

    from tests._modal_import_probe import assert_module_identity, module_census

    package = importlib.import_module(PACKAGED)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "modal_runner", package)
        with pytest.raises(AssertionError, match="forbidden bare"):
            assert_module_identity(module_census())
    assert_module_identity(module_census())


# The three controls below plant under NON-bare names, so the forbidden-bare
# clause cannot be what rejects them; each asserts that before it asserts the
# clause it pins. Like the controls above, they reach the runner through
# `importlib.import_module(PACKAGED)`: a variable argument, which the static
# walker cannot see. That is what keeps this file out of
# `_test_files_importing(PACKAGED)` -- writing `import scripts.modal_runner`
# here would make the scope test demand this file in `_IMPORTERS`, and the
# fork-bomb guard in the runtime gate forbids that entry.


def test_runtime_identity_groups_a_second_object_by_its_source_file(monkeypatch):
    """A second object under an unrelated name is grouped by its `__file__`.

    `core` is not a bare spelling, so only `module_census`'s origin route can
    file it under `scripts.modal_runner.core`; without that route the census
    ignores it and the payload passes.
    """
    import importlib
    import types

    from tests._modal_import_probe import assert_module_identity, module_census

    importlib.import_module(PACKAGED)
    canonical = PACKAGED + ".core"
    with monkeypatch.context() as patch:
        twin = types.ModuleType("core")
        twin.__file__ = sys.modules[canonical].__file__
        patch.setitem(sys.modules, "core", twin)
        planted = module_census()
        assert planted["forbidden_bare"] == []
        spellings = [row["name"] for row in planted["identities"][canonical]]
        assert spellings == ["core", canonical]
        with pytest.raises(AssertionError, match="module object identity"):
            assert_module_identity(planted)
    assert_module_identity(module_census())


def test_runtime_identity_rejects_one_object_under_two_governed_names(monkeypatch):
    """Two governed names bound to ONE object fail the distinct-objects clause.

    Each name still holds a single object under its own spelling, so the
    per-module identity clause and the spelling clauses all pass; only the
    count of distinct objects across the governed names sees the collapse.
    """
    import importlib

    from tests._modal_import_probe import assert_module_identity, module_census

    importlib.import_module(PACKAGED)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, PACKAGED + ".core", sys.modules[PACKAGED + ".state"])
        planted = module_census()
        assert planted["forbidden_bare"] == []
        assert planted["identities"][PACKAGED + ".core"][0]["object_id"] == id(
            sys.modules[PACKAGED + ".state"])
        with pytest.raises(AssertionError, match="distinct governed module object identities"):
            assert_module_identity(planted)
    assert_module_identity(module_census())


def test_runtime_identity_rejects_a_non_bare_alias_of_the_same_object(monkeypatch):
    """A second NAME for the canonical object fails the competing-spellings clause.

    One object, so the identity clauses pass; not bare, so the forbidden-bare
    clause passes. This is the only clause that objects to a second spelling
    which does not start with `modal_runner`.
    """
    import importlib

    from tests._modal_import_probe import assert_module_identity, module_census

    importlib.import_module(PACKAGED)
    canonical = PACKAGED + ".core"
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "core_alias", sys.modules[canonical])
        planted = module_census()
        assert planted["forbidden_bare"] == []
        spellings = [row["name"] for row in planted["identities"][canonical]]
        assert spellings == ["core_alias", canonical]
        with pytest.raises(AssertionError, match="competing import spellings"):
            assert_module_identity(planted)
    assert_module_identity(module_census())


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
    `test_no_file_imports_the_bare_modal_runner_spelling` (FileNotFoundError,
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
    retires coverage without reddening anything. Measured at W1, on the
    single-module census the per-module probe replaced: emptying the list of
    real importers was caught by accident -- the census read `[]`, which the
    gate's last assertion rejected -- but a PARTIAL narrowing was not: drop one
    of four and the census still read exactly `["scripts.modal_runner_lib"]`
    from the survivors, and every assertion in the gate passed.

    Set EQUALITY, not `<=`, because the two directions fail differently and both
    are real:

    - a name MISSING from `_IMPORTERS` is lost coverage. Measured by dropping
      `tests/test_modal_protocol.py`: THIS test is the only objector -- the gate
      below stays green, which is the whole argument for this test existing.
    - a name in `_IMPORTERS` that imports NOTHING is a session paying for a file
      with no reason to be there, and usually a typo'd path that has silently
      stopped matching a real file. Measured by adding `tests/test_demo_format.py`:
      again THIS test is the only objector.
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


def test_modal_runner_resolves_to_exactly_one_module_object(tmp_path):
    """RUNTIME gate for the one-spelling rule.

    Runs a real pytest session over `_IMPORTERS` with `tests._modal_import_probe`
    loaded, and requires the in-body carrier to have passed and every
    governed module to resolve to one object, under one spelling. The
    static guard above proves no file CONTAINS a bare import in a shape it
    knows; this proves none HAPPENS -- including through
    `importlib.import_module(<variable>)`, which no static census can see. It
    covers the files it runs, not the repo:
    `test_the_runtime_probe_runs_every_test_file_that_imports_the_runner` keeps
    that set honest, and imports made in child processes are invisible to it
    (hence `_NESTED_DESELECT`).

    The nested session's whole stdout/stderr is written to a log under this
    test's `tmp_path`, and every failure message after the run names it.

    ORDER OF THE ASSERTIONS AFTER THE RUN, and why each is where it is:

    1. the probe wrote its output file. Without it there is no payload, and
       the exit status and the output tails are all a reader gets.
    2. that file is valid JSON. A session killed mid-write leaves a truncated
       file; the decode error is reported with the tails rather than escaping
       as a bare JSONDecodeError.
    3. identity failed in an otherwise clean session -> reported as itself,
       with the probe's `identity_error`. The probe turns that session's exit 0
       into 1, so without this clause the 1 surfaces below as "did not finish
       clean" and points the reader at flaky tests instead of module identity.
    4. the session exited 0 (EXIT-STATUS assertion).
    5. the carrier passed its call phase.
    6. `assert_module_identity` on the payload, re-checked here so the gate
       does not depend on the probe having checked it.

    IF THIS TEST WENT RED ON THE EXIT-STATUS ASSERTION, READ THIS FIRST. The
    nested session runs every non-deselected test in the `_IMPORTERS` files, so
    this test inherits the flakiness of every one of them, and reports it as
    "the probe session did not finish clean" -- a headline pointing at the
    import machinery when the fault is very likely elsewhere. Read the nested
    `FAILED` line (in the message tail, or in the log it names) before you
    suspect anything about module objects. gh#211 records what is known: 2
    spurious failures across ~20 nested sessions, one node id captured
    (`test_interrupt_without_publishable_checkpoint_writes_a_reason_file`),
    cause NOT VERIFIED and not reproducible on demand.

    AND YET THE EXIT-STATUS ASSERTION MUST STAY. Let an unrelated test in an
    `_IMPORTERS` file fail while the carrier still passes -- exactly the gh#211
    shape -- and neuter the exit-status assertion: the gate goes GREEN, because
    the carrier DID run and the census IS clean. It is the sole objector to a
    session that failed for an unrelated reason. (Measured on the single-module
    probe the per-module probe replaced; the argument does not depend on the
    census's shape.)

    PITFALL: an in-body import such as test_eval_baselines' only runs when its
    test EXECUTES, so this must run the session, not import the test modules;
    and "the probe file was written" is not "the import ran", which is why the
    carrier assertion exists.
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

    # pytest silently ignores a `--deselect` that matches nothing, so a renamed
    # campaign test would rejoin the nested session with nothing objecting. And
    # `--deselect` matches node-id PREFIXES, so an entry also drops every
    # function whose name starts with its own: each such function must be
    # listed, or a new sibling test would leave the nested session unnoticed.
    # The check reads module-level `def` and `async def` statements only (see
    # `_NESTED_DESELECT`); pytest collects an `async def test_*` as an item too.
    for node_id in _NESTED_DESELECT:
        rel, _, name = node_id.partition("::")
        defined = {
            node.name
            for node in ast.parse((ROOT / rel).read_text(encoding="utf-8")).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert name in defined, (
            f"`_NESTED_DESELECT` names {node_id}, which {rel} no longer defines, so the "
            "deselect matches nothing and that test runs inside the nested session again. "
            "Update `_NESTED_DESELECT` to the test's current name.")
        listed = {
            entry.partition("::")[2]
            for entry in _NESTED_DESELECT if entry.startswith(rel + "::")
        }
        swept = {function for function in defined if function.startswith(name)}
        assert swept <= listed, (
            f"`_NESTED_DESELECT` entry {node_id} is a prefix, so it also deselects "
            f"{sorted(swept - listed)} from the nested session without naming them. List "
            "each one in `_NESTED_DESELECT` if it belongs out of the session, or rename it "
            "so the prefix stops matching.")

    out = tmp_path / "probe.json"
    log = tmp_path / "nested-session.log"
    env = {**os.environ, "MODAL_IMPORT_PROBE_OUT": str(out)}
    # --basetemp keeps the nested session's temp tree inside this test's own
    # `tmp_path`, next to its log and probe output. Isolation from the OUTER
    # session does not depend on it: without the flag pytest would give the
    # nested session its own numbered pytest-<N> dir under the temp root
    # (tests/conftest.py points that root at ~/.cache/cs2rl-pytest unless
    # PYTEST_DEBUG_TEMPROOT is set, gh#219), which would outlive this test
    # until pytest's rotation of old sessions cleared it.
    #
    # `-o addopts=` and `-p no:randomly` are forward insurance, not load
    # bearing today: checked 2026-09-22 on the W3b tree, there is no
    # `[tool.pytest.ini_options]` in pyproject.toml, no pytest.ini / tox.ini
    # / setup.cfg, and pytest-randomly is not installed. Both are no-ops
    # now; they keep the nested session deterministic if either arrives.
    argv = [
        sys.executable, "-m", "pytest", *_IMPORTERS,
        *(f"--deselect={node_id}" for node_id in _NESTED_DESELECT), "-q", "-o", "addopts=", "-p",
        "no:randomly", "-p", "tests._modal_import_probe", "--basetemp",
        str(tmp_path / "pt")
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
        # to report. `text=True` does not guarantee str here, so each stream
        # is normalised to bytes for the log and repr'd for the message;
        # `or b""` covers the None case.
        stdout = expired.stdout or b""
        stderr = expired.stderr or b""
        log.write_bytes((stdout.encode() if isinstance(stdout, str) else stdout) +
                        b"\n--- stderr ---\n" +
                        (stderr.encode() if isinstance(stderr, str) else stderr))
        raise AssertionError(
            f"the probe session did not finish within {_NESTED_TIMEOUT_S}s, so it hung "
            "rather than failed. This is not a module-spelling problem; look at what "
            f"the nested session was doing. log: {log}\n{stdout[-3000:]!r}\n"
            f"--- stderr ---\n{stderr[-2000:]!r}") from expired
    log.write_text(result.stdout + "\n--- stderr ---\n" + result.stderr, encoding="utf-8")
    # stderr is in every message below on purpose: when the nested session
    # dies before sessionfinish (bad plugin, collection crash) stdout is
    # empty and the traceback is on stderr, which is the one case where the
    # failure message is all a reader gets.
    tail = f"log: {log}\n{result.stdout[-3000:]}\n--- stderr ---\n{result.stderr[-2000:]}"
    assert out.exists(), f"probe never ran; pytest exit={result.returncode}\n{tail}"
    # A session killed mid-write leaves a file that passes the exists check
    # and then blows up in the decoder. Caught so the tail -- the only thing
    # that says WHY -- survives; an unguarded JSONDecodeError throws it away.
    raw = out.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as bad:
        raise AssertionError(
            "the probe wrote its output file but it is not valid JSON, so the session "
            f"was killed mid-write. exit={result.returncode} bytes={len(raw)} "
            f"{bad}\nfile starts: {raw[:200]!r}\n{tail}") from bad

    identity_error = payload["identity_error"]
    assert not (identity_error and payload["exitstatus"] == 0), (
        "module identity failed in an otherwise clean nested session: "
        f"{identity_error}\n{tail}")
    assert result.returncode == 0 and payload["exitstatus"] == 0, (
        "the probe session did not finish clean, so its module census describes "
        f"a run that may never have reached the import. exit={result.returncode} "
        f"sessionfinish={payload['exitstatus']} identity_error={identity_error}\n{tail}")
    assert _INBODY_CARRIER in payload["passed"], (
        f"{_INBODY_CARRIER} did not report passed, so the in-body import -- the "
        "only shape a static census cannot see -- was not executed. The module "
        f"census below is vacuous. passed={len(payload['passed'])} node ids. log: {log}")
    from tests._modal_import_probe import assert_module_identity
    try:
        assert_module_identity(payload)
    except AssertionError as error:
        raise AssertionError(f"{error}\nlog: {log}") from error


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
# 279 is `len(classify_seam(_seam_sources(), RUNNER_FILES)[0])` after the §2a
# safety commit of gh#163's close-out. Against the W3b split tree's 278: one
# runner-side test added (`test_signal_process_group_refuses_groups_a_live_child_cannot_have`),
# nothing removed, no destination moves. W3b's own move, against 2bb32ac's 273:
# five client-side helper/test names added, one `SEAM_GUARDS` name renamed, no
# destination moves -- the manifest's own diff lists the names. Do not infer
# this value from additions in the packaging file, which the classifier
# excludes.
#
# A GREEN `assert len(manifest) == GOVERNED_NAME_COUNT` IS NOT EVIDENCE THIS
# NUMBER IS RIGHT. It compares two frozen artifacts -- a checked-in manifest
# against a checked-in constant -- so adding names to a governed file changes
# neither and it stays green with a stale count and N ungoverned names on disk.
# Its own message says so: it is a DELETION detector. The addition detector is
# `test_seam_manifest_agrees_with_the_classifier`, which recomputes.
GOVERNED_NAME_COUNT = 279

# THE DECLARED RUNNER SET: the runner-half test files the seam reads, as a set,
# so the classifier and the placement gate are written for the eight per-module
# files before those files exist. It still holds the single unsplit runner test
# file, and that entry is the one transitional literal the seam keeps. The
# relocation commit that splits the runner tests by module replaces this whole
# tuple with `RUNNER_TEST_FILES` from tests/modal_runner_tables.py; a surviving
# copy of the literal is a hit for that commit's repo-wide grep. PITFALL: every
# gate in this section takes the runner files from here (or as a parameter);
# a second, retyped list anywhere is how the gate and the tree drift apart.
RUNNER_FILES = ("tests/test_modal_runner.py", )
CLIENT_FILE = "tests/test_modal_client.py"
PACKAGING_FILE = "tests/test_modal_packaging.py"

# THE MEMBERSHIP RULE FOR THE SHARED FILE. `tests/modal_test_helpers.py` exists
# as of W2's split and carries these words in its own docstring, which is where a
# reader opening that file will look for them. This copy is the one the
# classifier sits next to; they must not drift.
#
# A name earns a place in the shared file by being REACHED BY TESTS IN TWO OR
# MORE SEAM FILES (the declared runner files and the client file). Nothing else
# earns it. A helper that only one file's tests reach belongs in that file --
# however generic the helper looks, and however well its name would read here.
# The rule this replaced, "reached from both halves of the runner/client seam",
# is its special case with one runner file. A consumer outside the seam files
# does not count: the classifier never reads one. `classify_seam` computes
# exactly this rule and will not send a single-file helper to this file, so the
# rule is enforced rather than merely stated.
#
# WHY write down a rule the classifier already computes. Spec §10 criterion 12
# bans a module named `utils` / `helpers` / `common` / `misc`. Its instrument,
# the `forbidden-name` clause in tests/test_modal_runner_package_shape.py, reads
# only the `scripts/modal_runner/` submodules, so a test module is outside its
# scope and there is no conflict here. But the criterion exists because a module
# named for what it IS rather than for what it OWNS becomes a junk drawer, and
# that failure mode does not care which directory it happens in. A one-line
# membership test is what keeps this file a seam artefact instead of a drawer:
# measured, it holds exactly 7 names, and every one of them is reached from both
# halves.
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
    "test_modal_runner_does_not_import_modal_or_torch",
})

# Names the seam does not govern, with the reason.
# Unless a sentence names another tree, the figures in this comment were
# measured at 2bb32ac, by AST over the module-level bindings of every collected
# `tests/test_*.py`. They are that tree's values, not current ones: the file
# counts rise whenever a new test file defines its own `ROOT`, which W3b's
# `tests/test_modal_runner_package_shape.py` does.
#
# `ROOT` is `Path(__file__).resolve().parents[1]` -- module-header boilerplate
# that every destination file defines for itself by construction. At 2bb32ac,
# 6 files define a `ROOT` of their own (`test_modal_argv.py`,
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
# 3 of them.
#
# THE 261 IN THIS PARAGRAPH IS THE MONOLITH'S OWN NAME COUNT AND IS HISTORICAL.
# It used to be the live governed count as well, and saying "261 is unchanged"
# here stopped being true on the W3a gates branch, which took the seam to 273
# (then `GOVERNED_NAME_COUNT`). What this paragraph actually measures did NOT
# move: re-measured at 2bb32ac, the collision set is still `ROOT` alone and the
# same 3 files, and the client half still defines 83 of the monolith's governed
# names plus `ROOT` -- which is the 84, and 83 + the 3 relocated `SEAM_GUARDS`
# is the 86. The W3a branch's twelve new client-half names are not monolith
# names, so none of them enters any figure in this paragraph.
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

# The seed-count figures below describe 2bb32ac, measured by the own-body signal
# census this comment describes. They move as client tests are added, so
# re-derive them with that census rather than advancing the prose.
#
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


def _closure(start, edges):
    """Every name reachable from the names in `start` along `edges` ({name: names it references}).

    Shared by `classify_seam`'s stage 2 (which tests reach each helper) and the
    reach floor (which helpers a test reaches), so the two cannot come to
    disagree about what "reaches" means. `start` is a name's own out-edges, not
    the name itself: the name is in the result only if a cycle leads back to it.
    """
    seen, stack = set(start), list(start)
    while stack:
        for nxt in edges[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def classify_seam(sources, runner_files):
    """Compute each module-level name's destination file from the code alone.

    `sources` is {relative path: source text}; `runner_files` is the declared
    runner set, the files a runner-half TEST may live in. It is a required
    parameter, not a default, so every caller states the set it classifies
    against: the seam gate passes `RUNNER_FILES`, and the synthetic probes (and
    the relocation's helper map) pass `RUNNER_TEST_FILES`, whose files need not
    exist for a pure-data call. Returns `(destinations, defined_in)`:
    {name: destination path} and {name: [files that define it]}. Works on the
    unsplit file (one entry) and on the split (several), because it treats the
    union of the files as one namespace -- which is exactly why it can be re-run
    after the split and still be evidence rather than a tautology.

    TWO STAGES, and the split between them is the whole design:

    1. TESTS are classified by transitive closure. A name is CLIENT if its own
       body names a client module, or if it references -- transitively -- a name
       that does. Measured against the seam the spec measured four ways, 0 of
       194 tests land on the wrong side. That 194 is a MONOLITH figure and is
       kept as one deliberately, because the measurement it reports was taken
       against the spec's line-3638 seam, which no longer exists. Do not try to
       re-derive it from this function's inputs, which have grown since: the
       live population is the `test_` names
       `classify_seam(_seam_sources(), RUNNER_FILES)[0]` sends to a runner file
       or to `CLIENT_FILE` (the `SEAM_GUARDS` sit in a file `_seam_sources()`
       excludes). That population moves with every test either half adds, so
       no live count is written here; derive it when you need it.

       A RUNNER-HALF TEST'S DESTINATION IS THE FILE OF `runner_files` THAT
       DEFINES IT. Which runner file a test belongs in is the placement
       judgement the seam manifest records, and the reach floor
       (`reach_floor_violations`) and the core rule check it from below; this
       function does not re-derive it. So for tests the placement gate's
       `misplaced` check compares on-disk against on-disk, and only the
       relocation's one-shot declared-placement check sees a test in the wrong
       runner file. A runner-half test defined OUTSIDE `runner_files` (the
       client file, the shared file, an undeclared file) has no legal
       destination and raises `ValueError`, naming every such test at once.

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
       client-specific; it is client because only client tests reach it. A
       helper whose consumers all sit in ONE seam file goes to that file (the
       client file, or the one runner file its runner consumers share); a helper
       whose consumers span TWO OR MORE seam files goes to `SHARED_FILE`. With
       one runner file that is the old "reached from both halves" rule, and
       measured on that tree, 7 names take the shared answer. A classifier
       forced to pick one file for those would strand them, which is a
       NameError at run time in whichever file lost.

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
    string of code that a child process runs is invisible to it, exactly as it
    is to `bare_spelling_imports` above. The seam's files do hold such strings
    -- `_container_equivalent_import` in the client half builds one for
    `python -c`, and `_write_probe_stubs` in the runner half writes stub
    modules a child imports -- so the limit is live. Checked 2026-09-22 on the
    W3b tree, by word-matching the string constants of every module-level seam
    function that passes `-c` or writes multi-line code to a file against the
    seam's module-level names: no hits, so the limit changes no destination
    today. A code string that did name one would be a reach this function
    misses, and nothing in this file would object. The monolith's two
    `code = \"\"\"...\"\"\"` blocks are not among them: both moved into THIS file
    with the `SEAM_GUARDS` in W2, and this file is excluded from
    `_seam_sources()`.

    Names mentioned in PROSE -- docstrings that cite a test or helper -- are not
    reaches either: `_referenced_module_names` ignores string constants, and
    `_reaches_client_directly` reads them only for `_CLIENT_MODULES`
    substrings. How many monolith names such prose mentions is a count that
    moves with every docstring that cites one; this docstring used to carry it
    as a figure and it went stale twice, the second time inside the wave that
    rewrote the headers it counted, so it is retired rather than advanced.
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
        for name in _closure(edges[test], edges):
            reached_by[name].add(test)

    governed = [n for n in nodes if n not in SEAM_HEADER_NAMES and n not in SEAM_GUARDS]
    destinations = {name: PACKAGING_FILE for name in SEAM_GUARDS}
    destinations.update(
        _test_destinations([n for n in governed if n in tests], client, defined_in, runner_files))
    for name in governed:
        if name not in tests:
            destinations[name] = _helper_destination(name, reached_by[name], destinations)
    return destinations, defined_in


def _test_destinations(tests, client, defined_in, runner_files):
    """{test: destination} for `classify_seam`'s governed tests; raises on a stray runner test.

    A client test goes to `CLIENT_FILE` wherever it sits (the placement gate
    then compares that with disk). A runner-half test goes to the file of
    `runner_files` that defines it. PITFALL, and the reason this raises rather
    than returning some destination: a runner-half test defined anywhere else
    has no right answer here. Sending it to "the" runner file stopped meaning
    anything once the runner set can hold eight files, and sending it to the
    file it sits in would make an undeclared file a legal home. Every such test
    is collected first and named in one error, so one run lists them all.
    """
    out, stray = {}, {}
    for name in tests:
        if name in client:
            out[name] = CLIENT_FILE
            continue
        outside = [rel for rel in defined_in[name] if rel not in runner_files]
        if outside:
            stray[name] = outside
        else:
            out[name] = defined_in[name][0]
    if stray:
        raise ValueError(
            f"runner-half tests defined outside the declared runner files {sorted(runner_files)}: "
            f"{stray}. A runner-half test (one whose reference graph reaches no client signal) "
            "must live in the runner test file of the module it tests. Move it there and set its "
            "value in the seam manifest to that file; a move changes no key, so "
            "GOVERNED_NAME_COUNT stays.")
    return out


def _helper_destination(name, consumers, destinations):
    """Stage 2 for one helper: the single seam file its consumers sit in, else `SHARED_FILE`.

    `consumers` are the tests that reach `name` transitively; each one's file is
    its destination from stage 1 and `_test_destinations`. A consumer whose
    destination is not a seam file (a `SEAM_GUARDS` name, sent to the packaging
    file by fiat) does not count, because the rule is about the seam's own
    files. PITFALL: a helper reached by no counting consumer raises instead of
    defaulting, because the two things it can be -- dead code, or a new entry
    point -- want opposite answers.
    """
    files = {destinations[test] for test in consumers} - {PACKAGING_FILE}
    if not files:
        raise ValueError(f"{name!r} is reached by no test, so stage 2 cannot place it; "
                         "it is dead code or a new entry point and needs a human")
    if len(files) > 1:
        return SHARED_FILE
    return files.pop()


def _seam_sources(root=ROOT, runner_files=RUNNER_FILES):
    """Return {path: source} for every declared runner file, the client file and the shared file.

    Exclude this classifier's own file: its source mentions client bindings,
    so self-feeding assigns the instruments to the concerns they inspect. The
    three legitimate packaging destinations are supplied by SEAM_GUARDS.

    Derive the current population from
    ``classify_seam(_seam_sources(), RUNNER_FILES)[0]``. A self-fed histogram
    also depends on which client bindings and client-module paths this file
    happens to mention, docstrings included; it is a diagnostic measurement,
    not an invariant worth another test or prose pin.

    A DECLARED FILE THAT DOES NOT EXIST RAISES `FileNotFoundError`; it is never
    skipped. The error names every missing path and no other declared path. An
    existence filter here is the hole this closes: a declared runner file that
    drops out of the classifier's input silently shrinks every gate built on
    it, and a repo-wide grep for the file's name cannot find a filter that
    holds no file name. `test_the_seam_source_reader_fails_on_a_missing_declared_file`
    pins it. `root` and `runner_files` are parameters (with the live values as
    defaults) so that check can hand this reader a declared set with one bogus
    path while every other declared file really exists; with an empty `root`,
    every file is missing and a reader that raises only when NOTHING is left
    would pass it.

    Live deletion of a NAME is guarded separately:
    `test_the_modal_test_split_matches_concern_recomputed_from_source` pins
    `GOVERNED_NAME_COUNT` and requires every manifest name to be defined on
    disk, and `test_seam_manifest_agrees_with_the_classifier` compares the
    manifest with the classifier.
    """
    rels = [*runner_files, CLIENT_FILE, SHARED_FILE]
    missing = [rel for rel in rels if not (root / rel).is_file()]
    if missing:
        raise FileNotFoundError(
            f"declared seam file(s) missing: {missing}. The seam never skips a declared file: "
            "restore it, or remove it from the declaration in the same commit that removes it.")
    return {rel: (root / rel).read_text(encoding="utf-8") for rel in rels}


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


# ── The reach floor: a test in tests/test_modal_<m>.py reaches module m ─────
#
# `classify_seam` decides runner versus client versus shared. It cannot decide
# WHICH runner file a runner test belongs in: that is a judgement about what
# the test tests, and the seam manifest records it. The reach floor and the
# core rule (`reach_floor_violations`) are a floor under that judgement, not a
# proof of it; the function's docstring states the measured residual.

# Declared exemptions from the reach floor: {(file, test): reason}. Keyed by
# FILE as well as test, so a test moved to another file loses its exemption
# there, and its old entry, which now names a file that does not define it,
# fails. An entry must also still be needed: an exempt test that does reach its
# file's module fails (minimality), so this set cannot go stale unnoticed.
#
# EMPTY UNTIL THE RELOCATION COMMIT. Its one entry,
# ("tests/test_modal_preflight.py",
#  "test_recording_volume_reload_restores_committed_run_root"),
# lands with the eight per-module files, with its reason: that test tests the
# `RecordingVolume` test double, which lives with its only consumers, the
# preflight tests, and it reaches `core` only through `mrl.STATUS_FILENAME`.
# PITFALL: landing it before that file exists reds the floor, because an entry
# whose file does not define its test fails minimality; and the natural
# "repair", skipping entries whose file is not among the sources, is an
# existence filter over this set -- the class of hole `_seam_sources` refuses.
# Until then, minimality and the (file, test) keying are exercised only by the
# `test_reach_floor_*` synthetic probes, which pass their own exemptions.
_REACH_EXEMPTIONS: dict[tuple[str, str], str] = {}

# What to do about each kind of floor violation, keyed like the violations'
# first field. The manifest edit and the `GOVERNED_NAME_COUNT` rule are named
# because they are the part a first-time reader cannot guess.
_FLOOR_REMEDIES = {
    "reach-floor":
    ("move the test to the runner test file of a module it reaches (RUNNER_TEST_FILES) "
     "and set its value in the seam manifest to that file -- a move changes no key, so "
     "GOVERNED_NAME_COUNT stays; or, if it tests a test double rather than a module, add a "
     "(file, test) entry with its reason to _REACH_EXEMPTIONS"),
    "core-rule":
    ("a test in the core file may name only `core` in its own body: move it to the file of "
     "the module it names and set its seam manifest value to that file (GOVERNED_NAME_COUNT "
     "stays); the core rule has no exemptions"),
    "exemption-not-needed":
    "the test reaches its file's module now: delete its _REACH_EXEMPTIONS entry",
    "exemption-names-no-such-test":
    ("an exemption does not follow its test: re-key the entry to the (file, test) that "
     "exists, or delete it"),
    "exemption-without-reason":
    "an exemption is legal only with a reason: write it as the entry's value",
}


def _bind_runner_alias(dotted, local, facade, subs):
    """Record `local` as a facade alias or a submodule alias if `dotted` names one.

    `dotted` is the full module path the import binds `local` to, so
    `import scripts.modal_runner as mrl`, `from scripts import modal_runner as
    mrl`, `from scripts.modal_runner import core` and `import
    scripts.modal_runner.core as core` all come through one rule. A name
    imported FROM a submodule (`from scripts.modal_runner.core import X`) is
    not a module alias and is not recorded.
    """
    prefix = PACKAGED + "."
    if dotted == PACKAGED:
        facade.add(local)
    elif dotted.startswith(prefix) and dotted.removeprefix(prefix) in RUNNER_MODULES:
        subs[local] = dotted.removeprefix(prefix)


def _runner_aliases(tree):
    """(facade names, {alias: module}) bound by one file's MODULE-LEVEL runner imports.

    Aliases are per file: a helper in the shared file resolves `training.X`
    through the shared file's own imports, not through the importing test's.
    PITFALL: only `tree.body` is read, so an import inside a function body, or
    under a module-level `if`/`try`, binds no alias here, and a reference
    through it resolves to nothing, which makes the floor stricter and the
    core rule looser (see `_own_module_refs`). A bare `import
    scripts.modal_runner` binds `scripts`, and `scripts.modal_runner.X` is
    likewise not resolved.
    """
    facade, subs = set(), {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    _bind_runner_alias(alias.name, alias.asname, facade, subs)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                _bind_runner_alias(f"{node.module}.{alias.name}", alias.asname or alias.name,
                                   facade, subs)
    return facade, subs


def _locally_bound(node):
    """Names `node` binds itself: its parameters, and every name stored anywhere inside it.

    A module alias rebound locally is not the module. The case is live: a
    runner test binds `request = mrl.build_run_request(...)` over the `request`
    submodule alias and then reads `request.run_id`, and a test that takes
    pytest's `request` fixture shadows it the same way. PITFALL: this is
    deliberately coarse -- a store ANYWHERE in the body (a nested def's, a
    comprehension's) shadows the alias for the whole body. That can only drop a
    reference, which makes the floor stricter and the core rule looser; the
    census this floor was specified against measured with the same rule.
    """
    bound = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        params = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
        bound |= {param.arg for param in params if param is not None}
    bound |= {
        s.id
        for s in ast.walk(node) if isinstance(s, ast.Name) and isinstance(s.ctx, ast.Store)
    }
    return bound


def _binding_target_key(call):
    """The literal site key of a `binding_target("key")` call, else None.

    Only the bare-name spelling counts, as in the binding census
    (tests/test_modal_patch_binding_census.py matches `func.id` the same way),
    so the two instruments agree on what a `binding_target` call is.
    """
    if getattr(call.func, "id", None) != "binding_target" or not call.args:
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def _own_module_refs(node, aliases, owners):
    """[(module, spelling)] for each reference in `node`'s OWN body that resolves to a runner module.

    The three resolution rules, and nothing else:
      * `<facade>.X` (`mrl.X`) resolves to X's owner in the tables' MANIFEST;
      * `<sub>.X` resolves to `sub`, where `<sub>` is a module-level alias of
        `scripts.modal_runner.<sub>` in the file that defines `node`;
      * `binding_target("key")` resolves to `BINDING_SITES[key]`'s owner.
    Decorators are part of the body (`ast.walk` visits them), so a
    parametrize list that names `mrl.X` counts. A locally rebound alias does
    not (`_locally_bound`).

    PITFALL: these rules are narrower than "anything that touches the module".
    A bare alias passed as a value (`monkeypatch.setattr(training, "x", f)`),
    `mrl.<submodule>.X`, a facade name the tables do not own, and a
    non-literal or unknown site key all resolve to nothing. That makes the
    floor stricter but the core rule LOOSER: a core-file test whose only
    non-core reference takes one of these shapes passes the core rule.
    """
    facade, subs = aliases
    shadowed = _locally_bound(node)
    refs = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
            root = sub.value.id
            if root in shadowed:
                continue
            module = owners.get(sub.attr) if root in facade else subs.get(root)
            if module:
                refs.append((module, f"{root}.{sub.attr}"))
        elif isinstance(sub, ast.Call):
            key = _binding_target_key(sub)
            if key in BINDING_SITES:
                refs.append((BINDING_SITES[key][0], f'binding_target("{key}")'))
    return refs


def _floor_reach(sources, manifest):
    """{(file, test): (own refs, modules reached)} for every test defined in `sources`.

    THE CLOSURE'S NAMESPACE IS THE UNION of every file in `sources` -- the
    same one `classify_seam` builds -- so a helper that sits in the shared file
    still carries its reach to a test in any runner file. A per-file closure
    fails tests whose only route to their module is a shared helper (measured
    on the declared placement: the core test that reaches `core` only through
    `_make_manifest`, once that helper moves to the shared file), and the
    natural "fix" for that, a second exemption, would pass minimality.

    A test's reach is its own references plus the own references of every
    module-level name it reaches transitively along `_referenced_module_names`
    edges. Each name's references resolve through the aliases of the file that
    defines it. `manifest` is the tables' {"<module>.py": [names it owns]},
    passed in so the probes and measurement scripts can supply their own.
    """
    owners = {name: f.removesuffix(".py") for f, names in manifest.items() for name in names}
    trees = {rel: ast.parse(text) for rel, text in sources.items()}
    aliases = {rel: _runner_aliases(tree) for rel, tree in trees.items()}
    nodes, own = {}, {}
    for rel, tree in trees.items():
        for name, node in _module_level_names(tree).items():
            nodes[name] = node
            own[name] = _own_module_refs(node, aliases[rel], owners)
    universe = set(nodes)
    edges = {n: _referenced_module_names(node, n, universe) for n, node in nodes.items()}
    reach = {}
    for rel, tree in trees.items():
        for name, node in _module_level_names(tree).items():
            if not (name.startswith("test_")
                    and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
                continue
            refs = _own_module_refs(node, aliases[rel], owners)
            reached = {module for module, _ in refs}
            for helper in _closure(_referenced_module_names(node, name, universe), edges):
                reached |= {module for module, _ in own[helper]}
            reach[(rel, name)] = (refs, reached)
    return reach


def _exemption_violations(pair, reason, reach, file_module):
    """The minimality violations of one `_REACH_EXEMPTIONS`-shaped entry, as a list."""
    rel, test = pair
    if not str(reason).strip():
        return [("exemption-without-reason", rel, test, "its reason is empty")]
    if pair not in reach or rel not in file_module:
        return [("exemption-names-no-such-test", rel, test,
                 f"no runner test file {rel} that defines {test} is among the sources")]
    if file_module[rel] in reach[pair][1]:
        return [("exemption-not-needed", rel, test, f"it reaches {file_module[rel]!r}")]
    return []


def reach_floor_violations(sources, manifest, exemptions):
    """Check the reach floor and the core rule; return `(violations, examined)`.

    `sources` is {path: source text} -- the union namespace, so pass every seam
    file, not just the runner files. `manifest` is the tables' owner map
    (`RUNNER_OWNERS`); `exemptions` is {(file, test): reason}. Pure: it reads
    no file, so the probes can call it on synthetic maps whose paths need not
    exist.

    WHAT IS EXAMINED: every module-level `test_` function in a file of
    `RUNNER_TEST_FILES` (tests/test_modal_<m>.py for m in RUNNER_MODULES). No
    other file is: not the client file, not the shared file, and not the
    unsplit runner test file, which names no module. `examined` is that set of
    (file, test) pairs, returned so the gate can assert its scope -- a floor
    that iterates a partial file list otherwise passes every other check.

    THE RULES. Violations are (rule, file, test, detail) tuples, sorted within
    each pass, and `_FLOOR_REMEDIES` says what to do about each rule:
      * "reach-floor": the test does not reach its file's module through its
        own body or any helper it reaches, and is not exempt;
      * "core-rule": in the core file only, the test's OWN body names a module
        other than `core`. `core` is reached by most runner tests, so the floor
        alone does not police that file. The core rule has no exemptions.
      * minimality, per exemption entry (the dict is iterated, not the examined
        pairs): "exemption-not-needed" if the test reaches its file's module;
        "exemption-names-no-such-test" if that file is not a runner test file
        among the sources or does not define that test; "exemption-without-
        reason" if the reason is blank.

    RESIDUAL: A FLOOR, NOT A PLACEMENT CHECK. A test that reaches two modules
    can sit in either file with both rules green; choosing between them is the
    judgement the manifest diff records. These are UNSPLIT-FILE figures: the
    review's AST reach instrument measured them on the unsplit file's reference
    graph (138 tests, before the process-group guard test was added), and this
    module's `_floor_reach` reproduced every one on that tree and on the tree
    with the guard test (139; it adds no alternative home). They are to be
    re-measured with this function once the eight per-module files exist:
      * 48 of the 138 tests have at least one other file where the floor and
        the core rule both stay green;
      * 13 training tests name no non-core module in their own body, so they
        could sit in the core file green (`test_heartbeat_commits_every_60s_
        while_training` is one), and so could the one exempt preflight test;
      * 23 training tests reach `state`, so they could sit in the state file
        green (`test_failed_cleanup_commit_does_not_let_redelivery_write` is
        one);
      * the core rule rejects most, not all, of the maximally wrong split:
        moving every core-reaching test into the core file (95 moves) fails 81
        of them, where the floor alone fails none;
      * the placement the replaced rule produces (each test in the file of the
        non-core module it references most) fails only 3 tests;
      * 0 of the 49 request tests reach training, so a request test placed in
        the training file is rejected.
    So after the relocation, a misplaced new test is caught only if it breaks
    the floor or the core rule.
    """
    file_module = dict(zip(RUNNER_TEST_FILES, RUNNER_MODULES, strict=True))
    reach = _floor_reach(sources, manifest)
    examined = {pair for pair in reach if pair[0] in file_module}
    violations = []
    for rel, test in sorted(examined):
        refs, reached = reach[(rel, test)]
        module = file_module[rel]
        if module not in reached and (rel, test) not in exemptions:
            violations.append(
                ("reach-floor", rel, test, f"reaches {sorted(reached)}, not {module!r}"))
        foreign = sorted({spelling for owner, spelling in refs if owner != "core"})
        if module == "core" and foreign:
            violations.append(("core-rule", rel, test, f"its own body names {foreign}"))
    for pair, reason in sorted(exemptions.items()):
        violations.extend(_exemption_violations(pair, reason, reach, file_module))
    return violations, examined


def _manifest_edits(*, add=None, delete=None, move=None):
    """The exact seam-manifest edits a gate failure calls for, with the `GOVERNED_NAME_COUNT` rule.

    `add` and `move` are {name: file}; `delete` is an iterable of names. The
    count rule is spelled out because it is the step people miss: adding or
    deleting a key moves `GOVERNED_NAME_COUNT` by the same amount, in the same
    commit, with the reason in the commit message; a move changes no key and
    leaves it alone. The file is written with
    `json.dumps(d, indent=2, sort_keys=True) + "\\n"`.
    """
    add, delete, move = add or {}, sorted(delete or ()), move or {}
    rel = MANIFEST.relative_to(ROOT).as_posix()
    lines = [f'add "{name}": "{dest}"' for name, dest in sorted(add.items())]
    lines += [f'delete "{name}"' for name in delete]
    lines += [f'set "{name}": "{dest}"' for name, dest in sorted(move.items())]
    if not lines:
        return f"{rel} needs no edit and GOVERNED_NAME_COUNT stays."
    delta = len(add) - len(delete)
    count = ("GOVERNED_NAME_COUNT stays: these edits add as many keys as they delete (a move "
             "changes no key)" if delta == 0 else
             f"change GOVERNED_NAME_COUNT by {delta:+d} in the same commit and say why")
    return f"in {rel}: {'; '.join(lines)}. Then {count}."


def _describe_floor_violations(violations):
    """One line per violation, with its rule's remedy from `_FLOOR_REMEDIES`."""
    return "\n".join(f"  [{rule}] {rel}::{test}: {detail}. Remedy: {_FLOOR_REMEDIES[rule]}"
                     for rule, rel, test, detail in violations)


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
    # The probe is keyed as a declared runner file: under `classify_seam`'s
    # rejection rule a runner-half test in an undeclared file raises, so a
    # made-up key would reject the probe's own runner tests.
    probe_file = RUNNER_TEST_FILES[0]
    destinations, defined_in = classify_seam({probe_file: _SEAM_CLASSIFIER_PROBE},
                                             runner_files=RUNNER_TEST_FILES)

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

    assert destinations["test_probe_runner_via_helper"] == probe_file
    assert destinations["test_probe_runner_via_shared"] == probe_file
    assert destinations["_runner_only_helper"] == probe_file, (
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
    assert destinations.get("_PROBE_TIMEOUT_S") == probe_file, (
        "a module-level ANNASSIGN reached only by runner tests was not placed. "
        "`_module_level_names` handles AnnAssign in a SEPARATE branch from "
        "Assign, so it needs its own control; the Assign control above stays "
        f"green when only this branch is removed. got={destinations.get('_PROBE_TIMEOUT_S')!r}")

    # ── SEAM_HEADER_NAMES: ungoverned must not mean invisible ────────────────
    # The unrelated `ROOT` definers are read off disk inside the message, so it
    # names today's files instead of a count that grows with every new test file.
    seam_files = {*RUNNER_FILES, CLIENT_FILE, SHARED_FILE, PACKAGING_FILE}
    assert "ROOT" not in destinations, (
        "`ROOT` was given a destination, so `SEAM_HEADER_NAMES` has stopped "
        "exempting it. Governing it makes the seam's scan collide with every "
        "unrelated test file that defines its own `ROOT`; today those are "
        f"{sorted(set(_names_defined_under_tests().get('ROOT', [])) - seam_files)}. "
        "The comment above `SEAM_HEADER_NAMES` has the history.")
    assert defined_in.get("ROOT") == [
        probe_file
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
    assert planted in SEAM_GUARDS and defined_in.get(planted) == [probe_file], (
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
        "test_modal_runner_does_not_import_modal_or_torch",
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
        classify_seam({"orphan.py": "def _reached_by_nothing():\n    return 1\n"},
                      runner_files=RUNNER_TEST_FILES)


def _probe_test_names(sources):
    """The module-level `test_` function names defined in a {path: source} map, in order."""
    return [
        name for text in sources.values()
        for name, node in _module_level_names(ast.parse(text)).items()
        if name.startswith("test_") and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def test_the_seam_classifier_places_runner_tests_by_their_declared_file():
    """Positive control for `classify_seam` over a declared set of SEVERAL runner files.

    The single-file probe above cannot tell "the runner file" from "the runner
    file that defines this test", nor "reached from both halves" from "reached
    from two seam files", because with one runner file they coincide. This
    source set has three runner files and the client file, all
    `RUNNER_TEST_FILES` paths or seam constants (they need not exist on disk):

    - each runner test's destination is the declared file that defines it;
    - a helper reached from TWO RUNNER FILES, and no client test, goes to the
      shared file -- the generalised rule, which the old two-halves rule would
      have sent to "the" runner file;
    - a helper reached from one runner file only goes to that file;
    - a runner-half test in a file outside the declared set -- here the shared
      file, and an undeclared per-module-looking path -- is REJECTED, with every
      such test named in one error.

    The last pair has its positive half: the same undeclared file, once
    declared, is accepted. So the rejection is about the declared set the
    caller passes, not about the file's name, which is what lets the gate pass
    its transitional declared set and the relocation pass `RUNNER_TEST_FILES`.
    """
    first, second, third = RUNNER_TEST_FILES[:3]
    sources = {
        first: ("def _two_runner_files_helper():\n    return 1\n\n"
                "def _first_file_only_helper():\n    return 2\n\n"
                "def test_in_the_first_file():\n"
                "    return _two_runner_files_helper(), _first_file_only_helper()\n"),
        second: ("def test_in_the_second_file():\n    return _two_runner_files_helper()\n"),
        third: ("def test_in_the_third_file():\n    return 3\n"),
        CLIENT_FILE: ("def test_on_the_client_side():\n    return _import_run_modal()\n"),
    }
    destinations, _ = classify_seam(sources, runner_files=RUNNER_TEST_FILES)
    placed = {name: destinations[name] for name in _probe_test_names(sources)}
    assert placed == {
        "test_in_the_first_file": first,
        "test_in_the_second_file": second,
        "test_in_the_third_file": third,
        "test_on_the_client_side": CLIENT_FILE,
    }, f"a test was not sent to the declared file that defines it: {placed}"
    assert destinations["_two_runner_files_helper"] == SHARED_FILE, (
        "a helper reached from two runner files was not sent to the shared file, so the shared "
        "rule is still 'both halves of the runner/client seam', which after the split strands "
        f"it in one runner file. got={destinations['_two_runner_files_helper']!r}")
    assert destinations["_first_file_only_helper"] == first, (
        "a helper reached from one runner file only did not follow its tests to that file. "
        f"got={destinations['_first_file_only_helper']!r}")

    undeclared = "tests/test_modal_undeclared_probe.py"
    stray = {
        **sources,
        SHARED_FILE: "def test_runner_side_in_the_shared_file():\n    return 4\n",
        undeclared: "def test_runner_side_in_an_undeclared_file():\n    return 5\n",
    }
    with pytest.raises(ValueError, match="outside the declared runner files") as rejected:
        classify_seam(stray, runner_files=RUNNER_TEST_FILES)
    for name in ("test_runner_side_in_the_shared_file", "test_runner_side_in_an_undeclared_file"):
        assert name in str(rejected.value), (
            f"the rejection did not name {name}, so one run no longer lists every stray: "
            f"{rejected.value}")
    del stray[SHARED_FILE]
    accepted, _ = classify_seam(stray, runner_files=(*RUNNER_TEST_FILES, undeclared))
    assert accepted["test_runner_side_in_an_undeclared_file"] == undeclared, (
        "the undeclared file's test was not accepted once its file was declared, so the "
        "rejection keys on something other than the declared set")


def test_the_seam_source_reader_fails_on_a_missing_declared_file():
    """`_seam_sources` raises on a missing declared file; it never skips one.

    THE HOLE THIS PINS. An existence filter over the declared seam files drops
    a declared runner file from the classifier's input without a word, and
    every gate built on that input then passes over a smaller seam. A
    repo-wide grep for the old file's name cannot find such a filter, because
    the filter holds no file name. So this is checked by behaviour.

    WHY THE REAL ROOT PLUS ONE BOGUS PATH, and not an empty `tmp_path` root:
    with an empty root every declared file is missing, so a reader that
    filters by existence and raises only when NOTHING is left also raises, and
    this check would be green on exactly the reader it exists to reject. Here
    every other declared file exists, so only a reader that refuses the one
    missing file raises. The message must name that path and no other
    declared path, so a reader that raises for some other reason, or blames
    the wrong file, is red too. The last assertion is the positive half: the
    real declared set is read whole.
    """
    bogus = "tests/test_modal_no_such_declared_file.py"
    declared = [*RUNNER_FILES, CLIENT_FILE, SHARED_FILE]
    assert not (ROOT / bogus).exists(), f"{bogus} exists, so it cannot stand for a missing file"
    assert all((ROOT / rel).is_file() for rel in declared), (
        "a declared seam file is missing from the tree, so this check cannot isolate the bogus "
        f"path: {[rel for rel in declared if not (ROOT / rel).is_file()]}")
    with pytest.raises(FileNotFoundError) as missing:
        _seam_sources(root=ROOT, runner_files=(*RUNNER_FILES, bogus))
    message = str(missing.value)
    assert bogus in message, f"the error does not name the missing file: {message}"
    assert [rel for rel in declared if rel in message] == [], (
        f"the error names declared files that exist, so it does not say which one is missing: "
        f"{message}")
    assert sorted(_seam_sources()) == sorted(declared), (
        "the source reader did not return every declared seam file for the real tree")


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
    computed, _ = classify_seam(_seam_sources(), RUNNER_FILES)
    only_manifest = {n: manifest[n] for n in sorted(set(manifest) - set(computed))}
    only_computed = {n: computed[n] for n in sorted(set(computed) - set(manifest))}
    assert not only_manifest and not only_computed, (
        "the manifest and the classifier disagree about WHICH names exist. "
        f"manifest-only={only_manifest} classifier-only={only_computed}. "
        "If the tree is right: " + _manifest_edits(add=only_computed, delete=only_manifest))
    disagree = {n: (manifest[n], computed[n]) for n in manifest if manifest[n] != computed[n]}
    assert not disagree, (
        f"the manifest is stale: (frozen, recomputed) for each name that moved: {disagree}. "
        "If the recomputed file is right: " + _manifest_edits(move={
            n: new
            for n, (_, new) in disagree.items()
        }))


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
    half the live rule. Current destination counts are derived from the manifest,
    rather than duplicated in prose:
    `test_seam_manifest_agrees_with_the_classifier` is what verifies that
    manifest against the source.

    WHAT IT DELIBERATELY DOES NOT CATCH: a BRAND-NEW name in a stray file. That
    name is not governed, and an unrelated new test module is legal. Only
    `governed` is in scope -- which is also why the scan's superset (every
    `tests/test_*.py` plus the shared helper module) does not turn this red for
    every helper in those files.

    PITFALL, and it is the one this repo keeps paying for: a scan that reached
    nothing would pass this silently. So the scope controls come FIRST and are
    not decoration. `outside` proves the glob reaches past the seam's own files,
    without which nothing could ever be found straying; `unseen` proves the scan
    actually sees the governed names rather than passing on an empty
    intersection. Both are stated as failures of THIS TEST's instrument, not of
    the tree, because that is what they would mean.
    """
    governed = set(json.loads(MANIFEST.read_text(encoding="utf-8")))
    seam_files = {*RUNNER_FILES, CLIENT_FILE, SHARED_FILE, PACKAGING_FILE}
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

    AND THE REACH FLOOR, over the same union namespace (`reach_floor_violations`,
    whose docstring states what it cannot catch, with figures). For a runner
    test, `concern says` is simply the runner file that defines it, so the
    `misplaced` check below compares on-disk with on-disk for tests and cannot
    see a test in the wrong runner file; the floor and the core rule are what
    object to one. While the declared runner set is the one unsplit file, which
    names no module, the floor examines nothing, and that is asserted rather
    than left to look like a pass.
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = _seam_sources()
    computed, _ = classify_seam(sources, RUNNER_FILES)
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
        f"they were deleted or they were moved out of reach of this scan: {missing}. "
        "Restore them, or, if the deletion is intended: " + _manifest_edits(delete=missing))

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
        f"{dict(list(misplaced.items())[:10])}. Move each definition to `concern says`, then " +
        _manifest_edits(
            move={
                n: where["concern says"]
                for n, where in misplaced.items() if where["manifest says"] != where["concern says"]
            }))

    for rel, tree in trees.items():
        defined = set(_module_level_names(tree))
        absent = sorted(SEAM_HEADER_NAMES - defined)
        assert not absent, (
            f"{rel} does not define {absent}, which the seam treats as module-header "
            "boilerplate rather than governing. Ungoverned must not mean lost.")

    violations, examined = reach_floor_violations(sources, RUNNER_OWNERS, _REACH_EXEMPTIONS)
    assert not violations, (f"{len(violations)} reach-floor / core-rule / exemption violation(s):\n"
                            f"{_describe_floor_violations(violations)}")
    # While the declared runner set is the one unsplit file, no source is a
    # per-module test file, so the floor examines nothing. The relocation commit
    # replaces this with the floor's scope assertion: `examined` equals the
    # manifest's `test_` entries valued at a file of RUNNER_TEST_FILES, and each
    # of those files contributes at least one.
    assert examined == set(), (
        "the reach floor examined tests although the declared runner set holds no per-module "
        f"test file: {sorted(examined)[:10]}. If RUNNER_FILES now names the per-module files, "
        "replace this assertion with the floor's scope assertion in the same commit.")


# ── The reach floor's synthetic probes: one per rule, each with both halves ──
#
# WHY THEY EXIST. On the real tree the floor has, by design, no failing case
# except its one exempt test, so a change that makes it more permissive -- every
# reference resolving to every module, a closure that over-reaches, a
# minimality check that does nothing, exemptions keyed by test name alone -- is
# green on the tree and invisible after merge. Each probe below is a pure-data
# call of `reach_floor_violations` on a synthetic {path: source} map, parsed and
# never executed; the paths are `RUNNER_TEST_FILES` entries and need not exist.
# Names that `mrl.X` resolves through are read from the tables rather than
# typed here (`_owned`), so moving a production symbol never breaks a probe.


def _floor_file(module):
    """The per-module test file path for `module`, from `RUNNER_TEST_FILES`."""
    return RUNNER_TEST_FILES[RUNNER_MODULES.index(module)]


def _owned(module):
    """One top-level name the tables say `module` owns, for a probe's `mrl.<name>` reference."""
    return RUNNER_OWNERS[f"{module}.py"][0]


def _floor_rules(sources, exemptions=None):
    """`reach_floor_violations` on a probe map with the real owner table; (rule, file, test) only."""
    violations, _ = reach_floor_violations(sources, RUNNER_OWNERS, exemptions or {})
    return [violation[:3] for violation in violations]


def test_reach_floor_resolves_mrl_names_to_owner():
    """`<facade>.X` resolves to X's owner in the tables' MANIFEST, and only there.

    Positive: a request-file test naming a request-owned name through the
    facade passes, under `mrl` and under a second facade spelling (`from
    scripts import modal_runner as facade`), because the facade alias is read
    from the file's imports, not assumed. Negative: a request-file test naming
    only a core-owned name fails the floor. It also pins what is EXAMINED: the
    client file's test is not, because the client file names no module.
    """
    req = _floor_file("request")
    header = "import scripts.modal_runner as mrl\nfrom scripts import modal_runner as facade\n\n"
    tests = (f"def test_names_its_module():\n    return mrl.{_owned('request')}\n\n"
             f"def test_names_it_via_another_spelling():\n    return facade.{_owned('request')}\n\n"
             f"def test_names_another_module():\n    return mrl.{_owned('core')}\n")
    sources = {req: header + tests, CLIENT_FILE: "def test_on_the_client_side():\n    return 1\n"}
    violations, examined = reach_floor_violations(sources, RUNNER_OWNERS, {})
    names = ("test_names_its_module", "test_names_it_via_another_spelling",
             "test_names_another_module")
    pairs = {(req, name) for name in names}
    assert examined == pairs, f"the floor examined the wrong (file, test) pairs: {sorted(examined)}"
    got = [violation[:3] for violation in violations]
    expected = [("reach-floor", req, "test_names_another_module")]
    assert got == expected, (
        "`mrl.X` did not resolve to X's owner alone: a test naming its own module's name must "
        f"pass and one naming only another module's name must fail. got={violations}")


def test_reach_floor_resolves_submodule_aliases():
    """`<sub>.X` resolves to `sub` through the file's module-level alias, unless rebound locally.

    Positive: `training.X` (bound by `from scripts.modal_runner import
    training`) passes in the training file; `st.X` (bound by `import
    scripts.modal_runner.state as st`) passes in the state file. Negative: the
    same `training.X` placed in the state file fails, and a training-file test
    that rebinds `training` locally before `training.X` fails too -- the local
    rebinding is not the module (the unsplit file has a live case of this
    shape: `request = mrl.build_run_request(...)`, then `request.run_id`).
    """
    header = ("from scripts.modal_runner import training\n"
              "import scripts.modal_runner.state as st\n\n")
    training_file, state_file = _floor_file("training"), _floor_file("state")
    sources = {
        training_file:
        header + ("def test_training_alias():\n    return training.anything\n\n"
                  "def test_training_alias_rebound():\n"
                  "    training = object()\n    return training.anything\n"),
        state_file:
        header + ("def test_state_alias():\n    return st.anything\n\n"
                  "def test_training_alias_in_state():\n"
                  "    return training.anything\n"),
    }
    assert _floor_rules(sources) == [
        ("reach-floor", state_file, "test_training_alias_in_state"),
        ("reach-floor", training_file, "test_training_alias_rebound"),
    ], f"submodule aliases resolved wrongly: {_floor_rules(sources)}"


def test_reach_floor_resolves_binding_target_through_binding_sites():
    """`binding_target("key")` resolves to `BINDING_SITES[key]`'s owner.

    Positive: the call passes in its owner's file. Negative: the same call in
    another (non-core) module's file fails, and an unknown key resolves to
    nothing, so a test whose only reach is a misspelt key fails even in the
    owner's file. The key's owner is read from `BINDING_SITES`, not typed here.
    """
    key = "attempt-watcher"
    owner = BINDING_SITES[key][0]
    other = next(module for module in RUNNER_MODULES if module not in (owner, "core"))
    header = "from tests.modal_patch_binding_campaign import binding_target\n\n"
    sources = {
        _floor_file(owner):
        header + (f'def test_installs_the_site():\n    return binding_target("{key}")\n\n'
                  'def test_installs_an_unknown_site():\n'
                  '    return binding_target("no-such-site")\n'),
        _floor_file(other):
        header + f'def test_installs_it_elsewhere():\n    return binding_target("{key}")\n',
    }
    expected = sorted([("reach-floor", _floor_file(owner), "test_installs_an_unknown_site"),
                       ("reach-floor", _floor_file(other), "test_installs_it_elsewhere")])
    assert _floor_rules(sources) == expected, (
        f"binding_target resolved wrongly: {_floor_rules(sources)}")


def test_reach_floor_follows_module_level_helpers():
    """A test that reaches its module only through helpers passes; without the helper it fails.

    The reach is two hops deep (`test -> _outer -> _inner -> mrl.X`), so the
    closure is transitive, not one level. Removing `_inner` leaves the test's
    call to `_outer` intact and the floor red, which is the negative half.
    """
    req = _floor_file("request")
    inner = f"def _inner():\n    return mrl.{_owned('request')}\n\n"
    rest = ("def _outer():\n    return _inner()\n\n"
            "def test_through_two_helpers():\n    return _outer()\n")
    header = "import scripts.modal_runner as mrl\n\n"
    assert _floor_rules({req: header + inner + rest}) == [], (
        "a test that reaches its module through two helpers failed the floor, so the closure "
        "is not transitive")
    without = _floor_rules({req: header + rest})
    expected = [("reach-floor", req, "test_through_two_helpers")]
    assert without == expected, (
        "a test whose helper chain no longer reaches its module still passed the floor, so the "
        f"floor is crediting reach it cannot see. got={without}")


def test_reach_floor_follows_helpers_in_the_shared_file():
    """The closure's namespace is the union: a shared-file helper carries its reach to a runner file.

    Positive: a state-file test whose only reach is a helper defined in the
    SHARED file passes. The helper's `state.X` resolves through the shared
    file's own import; the test's file imports nothing, so a floor that
    resolved every name through the test's file, or that closed over the
    test's file alone, is red here. Negative: remove the helper from the
    shared file and the same test fails.
    """
    state_file = _floor_file("state")
    test_source = "def test_through_the_shared_file():\n    return _shared_state_helper()\n"
    shared_header = "from scripts.modal_runner import state\n\n"
    helper = "def _shared_state_helper():\n    return state.anything\n"
    with_helper = _floor_rules({state_file: test_source, SHARED_FILE: shared_header + helper})
    assert with_helper == [], (
        "a test that reaches its module only through a shared-file helper failed the floor, so "
        "the closure is per-file, or the helper's reference was resolved through the wrong "
        f"file's imports. got={with_helper}")
    without = _floor_rules({state_file: test_source, SHARED_FILE: shared_header})
    expected = [("reach-floor", state_file, "test_through_the_shared_file")]
    assert without == expected, f"the test still passed with the shared helper gone. got={without}"


def test_reach_floor_applies_the_core_rule_to_the_core_file():
    """In the core file, a test's OWN body may name only `core`; elsewhere the rule does not apply.

    Negative: a core-file test naming a request-owned name in its own body
    fails the core rule (it reaches core, so the floor alone passes it).
    Positive: a core-file test naming only core passes; so does one that
    reaches request only through a helper, because the rule is about the own
    body; and a request-file test naming core and request in its own body is
    not subject to it.
    """
    core_file, req = _floor_file("core"), _floor_file("request")
    header = "import scripts.modal_runner as mrl\n\n"
    core_name, request_name = _owned("core"), _owned("request")
    sources = {
        core_file:
        header + (f"def _request_helper():\n    return mrl.{request_name}\n\n"
                  f"def test_core_only():\n    return mrl.{core_name}\n\n"
                  f"def test_core_and_request_in_own_body():\n"
                  f"    return mrl.{core_name}, mrl.{request_name}\n\n"
                  f"def test_request_only_through_a_helper():\n"
                  f"    return mrl.{core_name}, _request_helper()\n"),
        req:
        header + (f"def test_request_file_names_core_too():\n"
                  f"    return mrl.{request_name}, mrl.{core_name}\n"),
    }
    got = _floor_rules(sources)
    expected = [("core-rule", core_file, "test_core_and_request_in_own_body")]
    assert got == expected, f"the core rule fired on the wrong tests: {got}"


def test_reach_floor_exemption_must_still_be_needed():
    """Minimality: an exemption for a test that reaches its file's module fails; a needed one holds.

    Positive: a request-file test that reaches only core fails the floor with
    no exemption, and passes -- with no minimality objection -- once exempt
    with a reason. Negative: an exemption for a test that DOES reach request
    fails as not needed, and an exemption with a blank reason fails.
    """
    req = _floor_file("request")
    sources = {
        req: ("import scripts.modal_runner as mrl\n\n"
              f"def test_reaches_its_module():\n    return mrl.{_owned('request')}\n\n"
              f"def test_reaches_only_core():\n    return mrl.{_owned('core')}\n"),
    }
    needed = {(req, "test_reaches_only_core"): "tests a test double, not a module"}
    assert _floor_rules(sources) == [("reach-floor", req, "test_reaches_only_core")]
    assert _floor_rules(sources, needed) == [], "a needed, reasoned exemption was not honoured"
    stale = {**needed, (req, "test_reaches_its_module"): "no longer true"}
    got = _floor_rules(sources, stale)
    expected = [("exemption-not-needed", req, "test_reaches_its_module")]
    assert got == expected, (
        "an exemption for a test that reaches its file's module passed minimality, so the "
        f"allow-set can go stale unnoticed. got={got}")
    got = _floor_rules(sources, {(req, "test_reaches_only_core"): "  "})
    expected = [("exemption-without-reason", req, "test_reaches_only_core")]
    assert got == expected, f"an exemption with a blank reason was accepted. got={got}"


def test_reach_floor_exemptions_are_keyed_by_file():
    """An exemption keyed to another file does not exempt the test, and fails itself.

    Positive: keyed to the (file, test) that defines the failing test, the
    exemption holds. Negative, three shapes: keyed to a runner file that is
    among the sources but does not define the test; keyed to a runner file not
    among the sources at all (the shape a relocated exemption takes before its
    file exists, which must fail rather than be skipped); and keyed to a
    non-runner file that does define a test of that name. Each leaves the
    test's own floor violation standing, and each entry fails.
    """
    req, state_file = _floor_file("request"), _floor_file("state")
    absent = _floor_file("preflight")
    header = "import scripts.modal_runner as mrl\n\n"
    sources = {
        req: header + f"def test_moved_here():\n    return mrl.{_owned('core')}\n",
        state_file: header + f"def test_stays():\n    return mrl.{_owned('state')}\n",
        CLIENT_FILE: "def test_moved_here():\n    return 1\n",
    }
    reason = "tests a test double, not a module"
    assert _floor_rules(sources, {(req, "test_moved_here"): reason}) == []
    for wrong_file in (state_file, absent, CLIENT_FILE):
        got = _floor_rules(sources, {(wrong_file, "test_moved_here"): reason})
        expected = [("reach-floor", req, "test_moved_here"),
                    ("exemption-names-no-such-test", wrong_file, "test_moved_here")]
        assert got == expected, (
            f"an exemption keyed to {wrong_file} was honoured for the test in {req}, or was not "
            f"itself reported: {got}")


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


def test_modal_runner_does_not_import_modal_or_torch():
    # Fresh subprocess: the parent may already have torch (the resume-input
    # checkpoint tests) or modal (later runner tests) in sys.modules.
    code = """
import sys
import scripts.modal_runner
assert 'modal' not in sys.modules
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def _runner_module_population(repo_root):
    """The package's submodule files as found ON DISK, checked against RUNNER_MODULES.

    Returns `(candidates, undeclared)` as repo-relative posix paths.
    `candidates` is every declared module in RUNNER_MODULES order, followed by
    every undeclared `*.py` in `scripts/modal_runner/`, sorted; `undeclared` is
    that tail alone.

    WHY the directory and not the constant: a scan whose population is its own
    declared list cannot see a file nobody declared. Measured by the W3b
    structural review (2026-09-22): a real `scripts/modal_runner/utils.py`
    holding `import modal` and a module-scope call passed every structural gate
    while the population was RUNNER_MODULES. The glob is the one
    `scripts/run_modal.py`'s mount loop uses, so every file the image ships
    other than the facade is a file these gates read.

    PITFALLS: declared modules stay in `candidates` even when absent, so a
    reader raises with the missing member's path instead of silently scanning
    seven. `__init__.py` is excluded: it is the facade and declares nothing.
    Its module scope has a stricter rule of its own (docstring, one-dot
    relative imports, a literal `__all__`), `_facade_violations` in
    tests/test_modal_runner_package_shape.py; the surface gates in
    tests/test_modal_client.py pin the names it exports.
    pathlib's `*` also matches dotfiles, and `*.py` matches a directory so
    named. The mount loop globs both too: a regular dotfile ships, while a
    dangling one (an Emacs `.#core.py` lock is a dangling symlink) and the
    directory fail the image build (FileNotFoundError / IsADirectoryError;
    Modal does not check the path when the image is defined). So both are in
    the population here, and a gate reading either raises.
    """
    root = Path(repo_root)
    declared = list(RUNNER_PATHS)
    found = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "scripts" / "modal_runner").glob("*.py") if path.name != "__init__.py")
    undeclared = [rel for rel in found if rel not in declared]
    return declared + undeclared, undeclared


def _is_type_checking_test(test):
    """True iff an `if` statement's test is `TYPE_CHECKING` or `<anything>.TYPE_CHECKING`.

    The one definition of a `TYPE_CHECKING` block for every gate that reads
    one: gate (e) exempts its body from the purity check, gate (g) allows one
    imports-only block at module scope, and the package-shape dependency walk
    (tests/test_modal_runner_package_shape.py) files its imports under
    ANNOTATION_DEPENDENCIES. They used to hold three copies of this test, and a
    copy that drifted would make one gate exempt a block another calls illegal.

    Matching by name has an accepted limit (Ruling 29): `if os.TYPE_CHECKING:`
    is recognised too. It fails closed the other way: `if TYPE_CHECKING and X:`
    is a `BoolOp` and is not recognised. A bare `TYPE_CHECKING` is only
    typing's while nothing rebinds it (`from os import environ as
    TYPE_CHECKING` makes the block run); inside scripts/modal_runner/ the
    package-shape `trusted-binding` clause rejects any such binding.
    """
    return ((isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
            or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"))


# The synthetic module text the gate (e) and gate (g) knock-outs plant into.
_MONOLITH_STUB = ROOT / "tests" / "fixtures" / "modal_runner_monolith_stub.txt"


def _monolith_stub_source():
    """The knock-outs' fixture text, after checking each property they rely on.

    WHY A CHECKED-IN STUB: five knock-outs of gates (e) and (g) were written
    against the pre-split monolith and read it with
    `git show 2bb32ac:scripts/modal_runner_lib.py`, so they failed in a
    shallow clone or a source tarball (gh#223). None of the live package
    modules can stand in: they change with every edit, all but core.py hold
    relative imports, and core.py's `__future__` import is not its line 19.

    The knock-outs splice each plant in after line 19 (`_splice_into_stub`),
    assert violation line numbers counted from there, and argue that every
    row is inert on the unplanted text. Each property those arguments rest on
    is asserted here, so an edit to the stub that breaks one fails naming it.
    """
    text = _MONOLITH_STUB.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    nodes = list(ast.walk(tree))
    body_kinds = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.FunctionDef,
                  ast.ClassDef)
    plain = [
        alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names
    ]
    froms = [node.module for node in tree.body if isinstance(node, ast.ImportFrom)]
    lazy = [
        alias.name for function in nodes if isinstance(function, ast.FunctionDef)
        for node in ast.walk(function) if isinstance(node, ast.Import) for alias in node.names
    ]
    claims = [
        ("line 19 is `from __future__ import annotations`", len(lines) > 18
         and lines[18] == "from __future__ import annotations\n"),
        ("module scope holds only the docstring, imports and declarations (so no `if`, "
         "`try` or `TYPE_CHECKING` block)", ast.get_docstring(tree) is not None
         and all(isinstance(node, body_kinds) for node in tree.body[1:])),
        ("every module-scope import is stdlib, and no plain import is dotted",
         all("." not in name and name in sys.stdlib_module_names for name in plain)
         and all(module and module.split(".")[0] in sys.stdlib_module_names for module in froms)),
        ("no relative import", not any(isinstance(n, ast.ImportFrom) and n.level for n in nodes)),
        ("no async def", not any(isinstance(n, ast.AsyncFunctionDef) for n in nodes)),
        ("no import in a class body", not any(
            isinstance(child, (ast.Import, ast.ImportFrom))
            for n in nodes if isinstance(n, ast.ClassDef) for child in n.body)),
        ("torch is imported in exactly two function bodies", lazy == ["torch", "torch"]),
    ]
    broken = [claim for claim, holds in claims if not holds]
    assert not broken, (
        f"{_MONOLITH_STUB.relative_to(ROOT)} no longer satisfies: {broken}. The gate (e) and "
        "gate (g) knock-outs' line numbers and inertness arguments rest on each; restore it.")
    return text


def _splice_into_stub(text):
    """The stub with `text` inserted after its line 19, the `__future__` import.

    A plant above that line would raise `SyntaxError: from __future__ imports
    must occur at the beginning of the file`, a red for the wrong reason.
    Nothing here is executed, only parsed, so a plant may name `os` or `modal`
    without either being importable.
    """
    lines = _monolith_stub_source().splitlines(keepends=True)
    return "".join(lines[:19]) + text + "".join(lines[19:])


def _nonstdlib_module_scope_imports(repo_root, top_level_only=False, *, paths=None):
    """Criterion 5: non-stdlib imports that execute when a package module is imported.

    Returns `(scanned, violations)`: the repo-relative paths actually read, and
    sorted `(path, lineno, root module)` tuples. The default population is
    `_runner_module_population`'s, so an undeclared module on disk is scanned
    too and shows up in `scanned`; `paths` replaces it for controls that plant
    into a single file.

    Function bodies and recognized TYPE_CHECKING bodies are exempt; class bodies,
    other containers, and TYPE_CHECKING else branches execute and are inspected.
    Missing/unreadable files raise with their path rather than pretending to scan.
    """
    root = Path(repo_root)
    candidates = [
        root / rel for rel in (paths if paths is not None else _runner_module_population(root)[0])
    ]
    scanned = []
    violations = []
    for path in candidates:
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        scanned.append(rel)
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
                if isinstance(node, ast.If) and _is_type_checking_test(node.test):
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


def test_gate_e_criterion_5_reddens_on_plants_the_tree_body_instrument_misses(tmp_path):
    """KNOCK-OUT. Plants live in a `tmp_path` file named
    `scripts/modal_runner/core.py` whose text is the stub from
    `_monolith_stub_source`, not the live package; nothing live is edited (GC2).

    THE DELIVERABLE IS THE SECOND ASSERTION of the first block. A version of this
    gate built on `tree.body` PASSES the decoy -- that is the exact blindness the
    decoy exists to prevent -- so the blind instrument's green is ASSERTED here,
    not described in prose. Prose cannot fail.

    Every plant goes AFTER LINE 19 (GC2). Line 19 is
    `from __future__ import annotations`, the first statement after the
    docstring, and a `__future__` import must precede all other code: a plant
    above it raises `SyntaxError: from __future__ imports must occur at the
    beginning of the file`, which is a red for the wrong reason.
    `_monolith_stub_source` asserts the plant site, so an edit that moves that
    line is loud.

    THE TABLE IS ONE ROW PER INDEPENDENT CLAUSE of the gate, because a gate whose
    headline clause has a knock-out and whose secondary clauses have none is the
    defect this branch has shipped three times. Nothing here is a variation on
    the decoy for its own sake -- each row is the only observation that kills one
    mutant:

      * `import modal as ...` is the ONLY row that objects to resolving names
        through `alias.asname`; that mutant is silent on the stub AND on
        the decoy, because a plain `import modal` has no `asname`.
      * `from modal.functions import ...` objects to dropping `ImportFrom`
        handling entirely (silent on the stub, whose from-imports are all
        stdlib) and to keeping the full dotted path instead of the root.
      * `import xml.etree.ElementTree` is the same root-resolution clause for
        `Import`, and it was MISSING until fault seeding found it: the stub has
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
      * the missing-target block at the end objects to skipping an unreadable
        target, and it is what makes the `scanned` assertions mean anything: a
        file the gate could not read must raise, never drop out of `scanned`
        and leave an empty violation list behind.
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
        stub's class holds no import, so nothing else can see it.
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
        separates this instrument from bare `ast.walk`. It is the shape the stub
        uses twice, so tests/test_modal_runner_package_shape.py's
        `test_package_import_purity_scans_every_module` already objects to
        descending -- this row says so where a reader of the
        instrument is standing.
      * the `async def` row is the only thing that objects to dropping
        `AsyncFunctionDef` from the skip set. The stub has no async def, so
        every other observation here is blind to that entry.
      * the relative-import row is the only thing that objects to dropping
        `node.level == 0`. Silent on the stub, which has no relative
        import; the package modules all have them, and without this row the
        gate would redden over legal intra-package imports.

    EVERY ROW BELOW IS INERT ON THE UNMODIFIED SUBJECT, by construction, and
    `_monolith_stub_source` asserts each property: the stub has no class-body
    import, no async def, no `TYPE_CHECKING` block, no
    relative import, no dotted plain `import`, no conditional module-scope import
    of any spelling, and nothing imported inside a container that RUNS at import
    time. Stated that way rather than as "nothing below depth 1", which is false:
    the stub has two imports below depth 1, both `torch`, in `_import_torch`
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

    def plant(text, slug):
        """Write a tmp_path repo whose `scripts/modal_runner/core.py` is the
        stub with `text` spliced in after line 19; return its root."""
        target = tmp_path / slug / "scripts" / "modal_runner" / "core.py"
        target.parent.mkdir(parents=True)
        target.write_text(_splice_into_stub(text), encoding="utf-8")
        return target.parents[2]

    decoy = plant("try:\n    import modal\nexcept ImportError:\n    modal = None\n", "decoy")
    scanned, violations = _nonstdlib_module_scope_imports(decoy,
                                                          paths=["scripts/modal_runner/core.py"])
    assert scanned == [
        "scripts/modal_runner/core.py"
    ], (f"the knock-out did not reach the planted copy, so its red below would be "
        f"unrelated to the plant. Scanned: {scanned}")
    assert violations == [
        ("scripts/modal_runner/core.py", 21, "modal")
    ], ("gate (e) missed a try-wrapped module-scope `import modal` -- the exact shape an "
        "optional dependency gets written in, and the one that makes importing the runner "
        f"library cost a Modal import. Reported: {violations}")

    _, blind = _nonstdlib_module_scope_imports(decoy,
                                               top_level_only=True,
                                               paths=["scripts/modal_runner/core.py"])
    assert blind == [], (
        "`tree.body` was supposed to be BLIND to this plant, and it is the blindness this "
        "gate exists to rule out. If it now sees the plant, the two instruments no longer "
        "differ anywhere measurable and this knock-out has stopped certifying that gate "
        f"(e) walks past depth 1. tree.body reported: {blind}")

    for slug, text, expected, clause in [
        ("asname", "import modal as _modal_shim\n", [("scripts/modal_runner/core.py", 20, "modal")],
         "resolve `Import` through `alias.name`, never `alias.asname`"),
        ("fromimport", "from modal.functions import FunctionCall\n", [
            ("scripts/modal_runner/core.py", 20, "modal")
        ], "`ImportFrom` counts, and its module resolves to the ROOT, not the dotted path"),
        ("denylist", "import numpy\n", [("scripts/modal_runner/core.py", 20, "numpy")],
         "the check is an allow-list over `sys.stdlib_module_names`, not a deny-list"),
        ("two", "import numpy\nimport modal\n", [("scripts/modal_runner/core.py", 20, "numpy"),
                                                 ("scripts/modal_runner/core.py", 21, "modal")],
         "EVERY offending import is reported, sorted. The walk pops its stack from the "
         "end, so an unsorted return hands these back in reverse source order and a "
         "report-only-the-first implementation drops the second -- and this is the only "
         "row with two violations in one file, so nothing else can see either mutant"),
        ("ifguard", 'if os.environ.get("CS2RL_MODAL"):\n    import modal\n', [
            ("scripts/modal_runner/core.py", 21, "modal")
        ], "the walk descends through `If` -- gate (g) plants exactly this header"),
        ("classbody", "class _C:\n    import modal\n", [
            ("scripts/modal_runner/core.py", 21, "modal")
        ], "a CLASS BODY EXECUTES at import time, so it is walked, not skipped"),
        ("classmethod", "class _C:\n    def m(self):\n        import torch\n        return torch\n",
         [], "walking class bodies must not reach METHOD bodies -- a method is a `FunctionDef` "
         "and does not run on import"),
        ("nestedcontainers", "class _C3:\n    try:\n        class _C4:\n            import modal\n"
         "    except ImportError:\n        pass\n", [("scripts/modal_runner/core.py", 23, "modal")],
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
             ("scripts/modal_runner/core.py", 24, "numpy")
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
            ("scripts/modal_runner/core.py", 22, "modal")
        ], "the `TYPE_CHECKING` exemption matches the NAME, not merely the node class -- a "
         "bare `Name` test that is not `TYPE_CHECKING` still runs, so the import under it "
         "still costs what it costs"),
        ("attrtestguard", "import os\nif os.name:\n    import modal\n", [
            ("scripts/modal_runner/core.py", 22, "modal")
        ], "same clause on the attribute spelling: `x.TYPE_CHECKING` is exempt, any other "
         "attribute test is not -- the exemption reads `test.attr`, not `isinstance(test, "
         "ast.Attribute)`"),
    ]:
        scanned, violations = _nonstdlib_module_scope_imports(
            plant(text, slug), paths=["scripts/modal_runner/core.py"])
        assert scanned == [
            "scripts/modal_runner/core.py"
        ], (f"the {slug} plant was not read at all, so its verdict is vacuous: {scanned}")
        assert violations == expected, f"gate (e) is wrong about: {clause}. Got {violations}"

    # A root with no target at all. The helper must raise, naming the file,
    # because an unread target that silently dropped out of `scanned` would
    # leave an empty violation list -- a vacuous green. Filtering candidates
    # through `is_file()` (the W3a behaviour) is the mutant this pins.
    empty = tmp_path / "noscripts"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="core.py"):
        _nonstdlib_module_scope_imports(empty, paths=["scripts/modal_runner/core.py"])


def _module_scope_shape_violations(tree):
    """Classify every module-scope statement; what a declaration contains is not checked.

    A block is recognised by `_is_type_checking_test`, the predicate gate (e)
    uses. A single imports-only block is legal, without else.
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
            if not _is_type_checking_test(node.test):
                violations.append(
                    (kind, node.lineno, "the only conditional allowed at module scope is "
                     "`if TYPE_CHECKING:`"))
                continue
            type_checking_blocks += 1
            if node.orelse:
                violations.append(
                    ("If", node.lineno, "TYPE_CHECKING blocks must have no else branch"))
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
        # is a violation. `Expr` and `If` are deliberately absent: both are legal in exactly one shape, so
        # they are decided by the branches above. Putting either here is the
        # blind kind-only allow-list this whole gate exists to rule out.
        elif not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                   ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            violations.append(
                (kind, node.lineno, "only imports and declarations may appear at module scope"))
    return examined, violations


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
    because `tests/test_modal_packaging.py` imports no test module at module
    scope but the data-only tests/modal_runner_tables.py (the runtime-identity
    tests import `tests._modal_import_probe` inside their bodies), and that exclusion is
    load-bearing -- `_seam_sources()` omits this file because
    self-feeding the classifier returns a poisoned destination count -- so
    duplicating that gate's subprocess helper into this file would put two copies
    of one gate in the tree, which is the defect class this branch exists to
    close. Neither half is optional: together they establish that criterion 13 is
    the SOLE objector to this header.

    AND THE THIRD BLOCK HOLDS THE TWO GATES TO ONE READING OF `TYPE_CHECKING`.
    Both call `_is_type_checking_test`, so they agree by construction; the
    block reads each shape's verdict off both gates, so a gate that stops
    calling the shared predicate, or a predicate that narrows or widens, turns
    a row red. Before the predicate was shared, gate (e) and gate (g) held two
    copies of it, and this block was the only thing that kept them in step: a
    narrowing seeded into either copy made the `os.TYPE_CHECKING` row, and no
    other test, go red.

    CRITERION 4 IS DELIBERATELY NOT IN THE CONTRAST. Gate (d) reads
    `runner_image.local_files` and the git index and never reads module source at
    all, so "criterion 4 is green on this plant" would be green for a reason
    unrelated to the plant. A control that cannot see the subject is not evidence
    about it -- that is precisely the class this branch closes, and asserting it
    would dress a vacuous green as a contrast.
    """
    planted = _splice_into_stub("import importlib.util\n"
                                "if importlib.util.find_spec('modal') is not None:\n"
                                "    _ON_CONTAINER = True\n"
                                "else:\n"
                                "    _ON_CONTAINER = False\n")

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
    target = tmp_path / "scripts" / "modal_runner" / "core.py"
    target.parent.mkdir(parents=True)
    target.write_text(planted, encoding="utf-8")
    scanned, impure = _nonstdlib_module_scope_imports(tmp_path,
                                                      paths=["scripts/modal_runner/core.py"])
    assert scanned == [
        "scripts/modal_runner/core.py"
    ], (f"gate (e) never read the planted copy, so its green below is vacuous: {scanned}")
    assert impure == [], (
        "gate (e) was expected to stay GREEN on this plant -- that is the whole contrast. A red "
        "here means the plant acquired an import that executes, which would be a correct gate "
        f"(e) verdict and a plant-design error on this side (Ruling 30). Reported: {impure}")

    # The two gates must read `TYPE_CHECKING` the same way. Each row plants `if <shape>:` holding a NON-stdlib import, then asks
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
         "accepted limit"),
        ("_probe()", False, "a `Call` test is not recognised by either gate"),
        ("TYPE_CHECKING is True", False, "a `Compare` test is not recognised by either gate"),
    ]:
        slug = "agree_" + shape.replace(".", "_").replace("(", "").replace(")", "").replace(
            " ", "_")
        block = _splice_into_stub(f"if {shape}:\n    import modal\n")
        agree_root = tmp_path / slug / "scripts"
        (agree_root / "modal_runner").mkdir(parents=True)
        (agree_root / "modal_runner" / "core.py").write_text(block, encoding="utf-8")
        agree_scanned, agree_impure = _nonstdlib_module_scope_imports(
            agree_root.parent, paths=["scripts/modal_runner/core.py"])
        assert agree_scanned == [
            "scripts/modal_runner/core.py"
        ], (f"the {slug} plant was not read, so gate (e)'s verdict on it is vacuous: "
            f"{agree_scanned}")
        _, agree_shape = _module_scope_shape_violations(ast.parse(block))
        e_exempts = agree_impure == []
        g_recognises = agree_shape == []
        assert e_exempts == recognised and g_recognises == recognised, (
            f"the two gates no longer read `if {shape}:` the same way, or no longer read it as "
            f"{recognised}. This is {why}. Both must call `_is_type_checking_test`: a block "
            "one gate exempts and the other calls illegal leaves whoever meets the red with no "
            f"way to attribute it. gate (e) exempted: {e_exempts} ({agree_impure}); gate (g) "
            f"recognised: {g_recognises} ({agree_shape})")


def test_gate_g_criterion_13_reddens_on_module_scope_work_the_kind_census_hides():
    """KNOCK-OUT 2. The two shapes a kind-only allow-list is green on.

    ROW 1 -- A BARE MODULE-SCOPE CALL. `logging.basicConfig(level=logging.INFO)`
    adds NO `If` node; measured on the stub, the census goes from
    `Import 2 | Expr 1` to `Import 3 | Expr 2` and every other kind is
    unchanged. A gate that allows the kind `Expr` because
    `body[0]` is one passes this, passes the stub, and passes every other
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
        tree = ast.parse(_splice_into_stub(text))
        examined, violations = _module_scope_shape_violations(tree)
        assert examined == len(tree.body), (
            f"gate (g) stopped short of the whole module scope: {examined} of {len(tree.body)}")
        assert violations == expected, f"criterion 13 is wrong about: {clause}. Got {violations}"


def test_gate_g_criterion_13_reddens_on_a_type_checking_body_that_is_not_imports_only():
    """KNOCK-OUT 3. A `TYPE_CHECKING` block is exempt from criterion 5, not from
    criterion 13 -- and the kind census cannot tell a legal one from this.

    MEASURED, and it is the sharpest evidence in this task that a kind-level
    instrument is the wrong one:

        if TYPE_CHECKING:            ->  ImportFrom 4 | If 1 | Import 2 | ...
            import decimal

        if TYPE_CHECKING:            ->  ImportFrom 4 | If 1 | Import 2 | ...
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
    tree = ast.parse(
        _splice_into_stub("from typing import TYPE_CHECKING\n"
                          "if TYPE_CHECKING:\n"
                          "    import decimal\n"
                          "    _X = 1\n"))

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

    ROW 1 IS THE GREEN HALF THE WHOLE TASK RESTS ON. The stub these plants
    splice into has no module-scope `If`, so on this substrate a
    gate that simply rejects every one of them is observationally identical to
    a correct gate on all three red knock-outs. Row 1 says the construct is
    ALLOWED. On the live package, commands.py and preflight.py each hold one
    such block, so tests/test_modal_runner_package_shape.py's
    `test_package_structure_contract` and
    `test_package_header_accepts_import_only_type_checking` say so as well.

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

    The helper's no-else clause has no row here. Its control is
    tests/test_modal_runner_package_shape.py's
    `test_package_ownership_and_header_controls[else]`, which plants into an
    in-memory copy of the live package's sources and runs the combined
    instrument over it; criterion 5
    independently checks the imports in that branch.
    """
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
        tree = ast.parse(_splice_into_stub(text))
        examined, violations = _module_scope_shape_violations(tree)
        assert examined == len(tree.body), (
            f"gate (g) stopped short of the whole module scope: {examined} of {len(tree.body)}")
        assert violations == expected, f"criterion 13 is wrong about: {clause}. Got {violations}"
