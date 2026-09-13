"""Guards on HOW the Modal runner is imported and packaged, not on what it does.

WHY: `scripts/` and the repo root are BOTH on sys.path in every pytest process.
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
whole tree (`ast.walk`), not `tree.body`. `tests/test_eval_baselines.py` imports
the bare spelling INSIDE a test body. A `tree.body` scan reports that file clean
while the defect is live, and still passes a module-scope positive control. That
is why `test_guard_detects_a_planted_bare_import` plants at four depths and in
the `importlib.import_module` shape this file's own siblings actually use.
"""
import ast
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BARE = "modal_runner_lib"


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
    tracked `.py` at this commit = 132 (the spec's 131 at 6c937ca plus this
    file). NOTHING asserts on that total and nothing should -- it moves with
    every added .py. `test_the_census_scans_the_whole_repo` pins the structural
    property instead, which does not move.

    `cwd=ROOT` is load-bearing: `git ls-files "*.py"` is CWD-RELATIVE, so
    running it from `tests/` returns 83 tracked paths instead of 132 and the
    guard silently stops watching `scripts/` and `src/`.
    `test_the_census_scans_the_whole_repo` is the control for exactly that.
    """
    out = subprocess.run(["git", "ls-files", "--cached", "*.py"],
                         cwd=ROOT,
                         capture_output=True,
                         text=True,
                         check=True)
    return [ROOT / line for line in sorted(set(out.stdout.splitlines())) if line]


def _bare_name(dotted):
    """True if `dotted` is the bare module or a submodule of it."""
    return dotted == BARE or dotted.startswith(BARE + ".")


def bare_spelling_imports(source):
    """Every bare-`modal_runner_lib` import in `source`, at ANY nesting depth.

    Returns a list of (lineno, kind). `ast.walk`, never `tree.body` -- see the
    module docstring.

    Covers three shapes, because this file's siblings write all three:
    `import X`, `from X import ...`, and `importlib.import_module("X")` /
    `__import__("X")` with a STRING LITERAL argument.

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
                if _bare_name(alias.name):
                    hits.append((node.lineno, "import"))
        elif isinstance(node, ast.ImportFrom):
            if _bare_name(node.module or ""):
                hits.append((node.lineno, "from-import"))
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            arg = node.args[0]
            if (name in ("import_module", "__import__") and isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str) and _bare_name(arg.value)):
                hits.append((node.lineno, "dynamic-import"))
    return hits


def test_no_file_imports_the_bare_modal_runner_lib_spelling():
    """One spelling repo-wide, so one module object exists at run time."""
    offenders = {}
    for path in _repo_python_files():
        try:
            hits = bare_spelling_imports(path.read_text(encoding="utf-8"))
        except SyntaxError:
            # A .py that does not parse cannot import anything either, so
            # skipping it is sound for THIS guard -- it is a scope statement,
            # not an excuse. Measured: 0 of the 132 tracked .py hit this branch,
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
    ("import modal_runner_lib.state\n", "submodule"),
])
def test_guard_detects_a_planted_bare_import(planted, shape):
    """POSITIVE CONTROL at four depths and in three shapes.

    A `tree.body` implementation passes the module-scope rows and fails the
    function-body rows -- and the function-body shape is the one
    `test_eval_baselines.py` actually used. A shape-blind implementation passes
    both and fails the `import_module` rows -- and THAT is the shape
    `tests/test_modal_runner.py` uses at three sites to reach its siblings. One
    control is not enough here; that is the whole point.
    """
    assert bare_spelling_imports(planted), f"guard is blind to {shape}"


def test_the_census_scans_the_whole_repo():
    """POSITIVE CONTROL for the guard's OWN SCOPE, which the row above cannot
    reach: every parametrized row calls `bare_spelling_imports` on a string and
    never touches `_repo_python_files`, the half that decides WHAT is read.

    WHY this exists: narrowing the pattern from `*.py` to `scripts/*.py` -- one
    token -- leaves every row above green, and at the time this landed both real
    violations lived in `tests/`. This repo's named #1 defect class is a guard
    blind to its own scope, in the file this plan calls its durable deliverable.

    The three asserts are not padding. Each is the FIRST objector to a different
    narrowing, measured by mutation against this exact argv -- so deleting any
    one of them silently retires a distinct check:

        *.py -> scripts/*.py       -> assert 1 (19 paths scanned)
        cwd=ROOT -> cwd=ROOT/tests -> assert 1 (83 paths; ls-files is relative)
        *.py -> tests/*.py         -> assert 3 (asserts 1 and 2 both PASS)
        census drops this file     -> assert 2 (asserts 1 and 3 both PASS)

    The last two are the reason asserts 2 and 3 exist at all: a narrowing that
    keeps `tests/` sails past assert 1, and one that drops only this file sails
    past both 1 and 3. If you add a filter to `_repo_python_files`, expect
    assert 2 to be what tells you it excluded more than you meant.
    """
    scanned = _repo_python_files()
    rel = {p.relative_to(ROOT).as_posix() for p in scanned}
    assert "tests/test_modal_runner.py" in rel, (
        "the census does not reach the file this guard was written for; the "
        f"enumeration has been narrowed. Scanned {len(scanned)} paths.")
    assert Path(__file__).resolve() in scanned, (
        "the census cannot see its own file, so it cannot police itself; the "
        "enumeration has been narrowed.")
    tops = {r.split("/")[0] for r in rel}
    assert {
        "scripts", "src", "tests"
    } <= tops, (f"the census must reach scripts/, src/ and tests/; it reached {sorted(tops)}")
