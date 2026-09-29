"""The package's layering: pyproject.toml's import-linter contracts, and the two holes they leave.

WHAT runs here:
  1. `lint-imports` over this checkout: every contract in [tool.importlinter] holds.
  2. The checkout check: the linter's child resolved `cs2rl` to THIS checkout.
  3. The coverage check: every tracked `src/cs2rl/*.py` is a module in grimp's graph.
  4. The scope pin: every site of every ignore_imports pair sits inside a def.
  5. Positive controls on a tmp copy of the package, one plant each, so every check
     above has been seen to reject something.
  6. The one layer placement a test elsewhere relies on: eval (metrics_schema) above train.

WHY 2: import-linter and grimp find `cs2rl` with importlib.util.find_spec, which
never executes src/cs2rl/__init__.py, so the package's foreign-checkout guard cannot
fire. In a worktree whose PYTHONPATH does not put its own src/ first they check
MAIN's tree and pass. PYTHONPATH wins over the editable install's .pth. Every graph
child (this checkout's and each control's) asserts it resolved the tree whose files
the checks then read, so no check can map one tree's lines onto another's AST.

WHY 3: grimp skips a directory without an __init__.py, silently. A module in such
a directory is in no layer and in no cycle, whatever it imports, and `exhaustive`
never hears of it. The expected set comes from `git ls-files`, never from grimp
(comparing grimp with itself is a tautology), so a new file must be tracked to be
covered. An untracked module fails here as "extra in grimp" only where grimp can see
its directory; an untracked one in a directory without __init__.py is on neither
side and passes until it is tracked (or `git add -N`).

WHY 4: grimp counts a function-local import as an edge, and an ignore_imports entry
names a module PAIR, never a line or a scope. So an ignored pair hides every future
site of that pair, a module-scope one included, and adding a second site does not
trip import-linter's unmatched-ignore alert (only deleting the last one does). The
pin reads the pairs from the config, so a new entry is pinned the day it lands.

HOW to run the linter by hand: `lint-imports --no-cache --no-logo` from the repo root,
with PYTHONPATH=<checkout>/src. `--no-cache` because a bare run writes
.import_linter_cache/ into cwd.

PITFALLS:
  - The CLI is run as `[sys.executable, <venv>/bin/lint-imports]`, never by its
    shebang: the shebang names an absolute venv path, which a drive remount or a
    worktree-path install breaks silently. There is no `python -m importlinter`
    (2.15 ships no __main__).
  - Never in-process: `importlinter.cli.lint_imports()` inserts cwd on sys.path for
    good, and grimp's find_spec would then return the cs2rl this session already
    imported. Every tree is checked in a child. (`importlinter.api.read_configuration`
    has no such side effect; test 6 calls it in-process.)
  - Every lint passes `--config <this checkout>/pyproject.toml`, this checkout's
    included: see `_lint`.
  - COLUMNS=200 in every child: rich wraps at 80 columns when stdout is not a TTY,
    which splits the tokens the controls look for.
  - Nothing here skips. A venv without the dev group fails with the remedy.
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

PYPROJECT = REPO_ROOT / "pyproject.toml"
# Beside the interpreter pytest runs under: the venv's own console script.
LINT_IMPORTS = Path(sys.executable).with_name("lint-imports")
_MISSING_TOOLS = "dev group not installed: import-linter is missing from this venv"
_CHILD_TIMEOUT_S = 120

# One grimp graph per tree, built in a child with the tree's src/ first on
# PYTHONPATH. It reports where `cs2rl` resolved, the module set, and the line of
# every site of every ignore_imports pair of every contract (grimp resolves every
# spelling: `from . import x`, `import cs2rl.x`, aliases). The entries are raw
# "a -> b" strings; a wildcard entry matches no module, so it reports no sites and
# fails the pin loudly until someone expands it (grimp's find_matching_direct_imports).
_GRAPH_CHILD = """
import importlib.util, json, sys
import grimp
from importlinter import api

config = api.read_configuration(sys.argv[1])
pairs = sorted({tuple(side.strip() for side in entry.split("->"))
                for contract in config["contracts_options"]
                for entry in contract.get("ignore_imports", [])})
graph = grimp.build_graph("cs2rl", cache_dir=None)
print(json.dumps({
    "origin": importlib.util.find_spec("cs2rl").origin,
    "modules": sorted(graph.modules),
    "sites": [[importer, imported,
               [d["line_number"] for d in graph.get_import_details(importer=importer,
                                                                   imported=imported)]]
              for importer, imported in pairs],
}))
"""


def _child_env(src: Path) -> dict[str, str]:
    """This session's environment with `src` PREPENDED to PYTHONPATH, and COLUMNS=200."""
    pythonpath = os.pathsep.join(filter(None, [str(src), os.environ.get("PYTHONPATH")]))
    return dict(os.environ, PYTHONPATH=pythonpath, COLUMNS="200")


def _lint(tree: Path) -> subprocess.CompletedProcess:
    """`lint-imports --no-cache --no-logo --config <PYPROJECT>` over `tree`/src, cwd = `tree`.

    `--config` is passed for this checkout too, not only for the tmp copies (which
    have no config of their own). Without it import-linter 2.15 discovers its config
    from cwd and tries the INI files, setup.cfg and .importlinter, BEFORE
    pyproject.toml (adapters/user_options.py; application/use_cases.py
    `read_user_options`), so a stray `.importlinter` at the repo root would silently
    replace the contracts under test.
    """
    assert LINT_IMPORTS.is_file(), f"{_MISSING_TOOLS} (no {LINT_IMPORTS})"
    argv = [sys.executable, str(LINT_IMPORTS), "--no-cache", "--no-logo"]
    return subprocess.run([*argv, "--config", str(PYPROJECT)],
                          cwd=tree,
                          env=_child_env(tree / "src"),
                          capture_output=True,
                          text=True,
                          timeout=_CHILD_TIMEOUT_S)


def _graph_facts(tree: Path) -> dict:
    """The graph child's report for `tree`: {"origin", "modules", "sites"}.

    Asserts the child resolved `cs2rl` to `tree` itself. Every consumer maps grimp's
    line numbers onto `tree`'s files, so facts from any other tree (main's, through
    the editable .pth, when `tree`'s own package is broken) must never reach them.
    """
    r = subprocess.run(
        [sys.executable, "-c", _GRAPH_CHILD, str(PYPROJECT)],
        cwd=tree,
        env=_child_env(tree / "src"),
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S)
    assert r.returncode == 0, (f"the grimp child failed (if it cannot import grimp or "
                               f"importlinter: {_MISSING_TOOLS}).\nSTDERR:\n{r.stderr}")
    facts = json.loads(r.stdout)
    # samefile, not a string compare: the drive is mounted under two names.
    own = tree / "src" / "cs2rl" / "__init__.py"
    origin = facts["origin"]
    assert origin is not None and own.is_file() and os.path.samefile(origin, own), (
        f"the graph child resolved cs2rl to {origin}, not {own}: it is checking another "
        f"tree. Put that tree's src/ first: env PYTHONPATH={tree / 'src'} <command>")
    return facts


def _dotted(relative: str) -> str:
    """'src/cs2rl/c_env/__init__.py' -> 'cs2rl.c_env'; 'src/cs2rl/env/nav.py' -> 'cs2rl.env.nav'."""
    parts = Path(relative).with_suffix("").parts[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _module_file(tree: Path, module: str) -> Path:
    path = tree.joinpath("src", *module.split("."))
    return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")


def coverage_failures(facts: dict, expected_files: Iterable[str]) -> list[str]:
    """Every expected file is a module in grimp's graph, and grimp has no other.

    `expected_files` is a PARAMETER, relative to the tree root: `git ls-files` for
    this checkout, the copied list plus any plant for a tmp tree. Never `git
    ls-files` inside a tmp tree (not a repo: the set would be empty) and never this
    checkout's list for a planted tree (the plant would be missing from both sides).
    """
    expected = {_dotted(f) for f in expected_files}
    got = set(facts["modules"])
    if expected == got:
        return []
    return [
        f"COVERAGE: missing from grimp's graph: {sorted(expected - got)}; extra in grimp's "
        f"graph: {sorted(got - expected)}. A module is missing when its directory has no "
        "__init__.py (grimp skips it silently, so no contract sees it): add one. An extra "
        "module is on disk but not tracked: `git add` it."
    ]


def scope_pin_failures(tree: Path, facts: dict) -> list[str]:
    """Every site of every ignored pair is inside a FunctionDef or AsyncFunctionDef.

    A method counts (a FunctionDef inside a ClassDef); a class body, an `if` at
    module level (`if TYPE_CHECKING:` included), or a `try:` at module level does not:
    control (e) plants one site of each shape. grimp reports a multi-line import at
    its first line, which is inside the def whenever the statement is.

    BLIND SPOT, stated: this checks where an import STATEMENT sits, not when it
    RUNS. `def f(): import cs2rl.viz` followed by a module-level `f()` passes the
    pin and both contracts. For train -> viz, the one pair nothing else would
    notice, tests/test_w1_modules.py lists `rerun` in HEAVY, so `from cs2rl import
    train` loading viz fails there.
    """
    failures = []
    for importer, imported, lines in facts["sites"]:
        path = _module_file(tree, importer)
        if not lines:
            failures.append(
                f"SCOPE PIN: {importer} -> {imported}: grimp finds no import site in {path}. "
                "Either the last site is gone (delete the ignore_imports entry) or the entry is "
                "a wildcard (expand it to explicit pairs).")
            continue
        spans = [(node.lineno, node.end_lineno)
                 for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for line in lines:
            if not any(lo <= line <= hi for lo, hi in spans):
                failures.append(
                    f"SCOPE PIN: {importer} -> {imported}: {path}:{line} is not inside a def. "
                    "The pair is in ignore_imports, so import-linter accepts this site at any "
                    "scope; only function-local sites are allowed. Move it into the function "
                    "that needs it.")
    return failures


def _tracked_package_files() -> list[str]:
    r = subprocess.run(["git", "ls-files", "src/cs2rl/*.py"],
                       cwd=REPO_ROOT,
                       capture_output=True,
                       text=True,
                       check=True)
    files = r.stdout.split()
    assert files, "git ls-files found no src/cs2rl/*.py: the expected module set is empty"
    return files


# ── this checkout ─────────────────────────────────────────────────────────────


def test_contracts_hold_on_this_checkout():
    r = _lint(REPO_ROOT)
    assert r.returncode == 0, f"lint-imports rc={r.returncode}\n{r.stdout}\n{r.stderr}"


def test_the_linter_checks_this_checkout():
    """The child that builds the graph resolved `cs2rl` to this checkout's src/.

    The assertion lives in `_graph_facts`, so every other check on this checkout, and
    every control on its tmp copy, makes it too; this test is the one that names it.
    The lint child runs with the same env and cwd, so it resolves the same way.
    """
    _graph_facts(REPO_ROOT)


def test_every_tracked_module_is_in_the_graph():
    failures = coverage_failures(_graph_facts(REPO_ROOT), _tracked_package_files())
    assert not failures, "\n".join(failures)


def test_every_ignored_import_stays_function_local():
    failures = scope_pin_failures(REPO_ROOT, _graph_facts(REPO_ROOT))
    assert not failures, "\n".join(failures)


def test_metrics_schema_sits_above_train():
    """The `cs2rl layers` contract puts eval, metrics_schema's package, above train.

    tests/test_w1_modules.py's NOT_IMPORTED_BY_TRAIN says a train.py import of
    metrics_schema is an upward edge lint-imports rejects. That is true only while this
    holds: moving eval down into train's layer keeps every other check green and makes
    that entry silently false.

    PITFALL: the contract is selected by NAME. pyproject.toml holds more than one layers
    contract (`cs2rl.env layers` too), so picking "the" layers contract by type either
    fails to unpack or reads the wrong one.
    """
    try:
        from importlinter import api
    except ImportError as e:
        raise AssertionError(_MISSING_TOOLS) from e
    [layers] = [
        c["layers"] for c in api.read_configuration(str(PYPROJECT))["contracts_options"]
        if c["name"] == "cs2rl layers"
    ]
    # Highest layer first; `|` and `:` both separate the members of one layer.
    rank = {
        member.strip(): index
        for index, layer in enumerate(layers)
        for member in re.split(r"[|:]", layer)
    }
    assert rank["eval"] < rank["train"], (
        f"eval (metrics_schema's package) is in layer {layers[rank['eval']]!r}, not above "
        f"train's {layers[rank['train']]!r}: tests/test_w1_modules.py's NOT_IMPORTED_BY_TRAIN "
        "entry for it is no longer true. Move it back up, or rewrite that entry.")


# ── positive controls: a tmp copy of the package, one plant each ─────────────────


def _copy_package(tmp_path: Path) -> tuple[Path, list[str]]:
    """Copy the tracked `src/cs2rl/*.py` into `tmp_path`; return it and the copied list."""
    files = _tracked_package_files()
    for relative in files:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, target)
    return tmp_path, files


def _append(tree: Path, relative: str, text: str) -> int:
    """Append `text` at column 0 after the last statement, i.e. at module scope.

    Returns the line number `text` starts on (after the one blank separator line).
    """
    path = tree / relative
    before = path.read_text(encoding="utf-8")
    path.write_text(before + f"\n{text}\n", encoding="utf-8")
    return before.count("\n") + 2


def test_control_upward_import_breaks_the_layers(tmp_path):
    """(a) env.nav (L1) importing train (L2) at module scope."""
    tree, _ = _copy_package(tmp_path)
    _append(tree, "src/cs2rl/env/nav.py", "from cs2rl import train")
    r = _lint(tree)
    assert r.returncode == 1 and "cs2rl.env.nav -> cs2rl.train" in r.stdout, r.stdout + r.stderr


def test_control_a_module_without_a_layer_is_rejected(tmp_path):
    """(b) `exhaustive`: a new module in no layer."""
    tree, _ = _copy_package(tmp_path)
    (tree / "src" / "cs2rl" / "newmod.py").write_text('"""Planted."""\n')
    r = _lint(tree)
    assert r.returncode == 1 and "- cs2rl.newmod" in r.stdout, r.stdout + r.stderr


def test_control_the_unplanted_copy_passes_every_check(tmp_path):
    """(c) The copy is complete: every check that the controls turn red passes on it.

    `_graph_facts` also asserts the child resolved the copy, not this checkout.
    """
    tree, files = _copy_package(tmp_path)
    r = _lint(tree)
    assert r.returncode == 0, r.stdout + r.stderr
    facts = _graph_facts(tree)
    failures = coverage_failures(facts, files) + scope_pin_failures(tree, facts)
    assert not failures, "\n".join(failures)


def test_control_a_cycle_inside_one_layer_is_rejected(tmp_path):
    """(d) env.nav <-> env.map, one layer of `cs2rl.env layers`: only the acyclic contract sees it.

    It checks siblings at every depth, so a cycle inside env/ is caught, not only one
    among cs2rl's own children.
    """
    tree, _ = _copy_package(tmp_path)
    _append(tree, "src/cs2rl/env/nav.py", "from cs2rl.env import map")
    r = _lint(tree)
    assert r.returncode == 1 and ".nav -> .map" in r.stdout, r.stdout + r.stderr


def test_control_env_config_importing_nav_breaks_the_env_layers(tmp_path):
    """(g) env.config importing env.nav: only the `cs2rl.env layers` contract sees it.

    config, nav and map are all the one member `env` of `cs2rl layers`, and the edge
    closes no cycle, so the other two contracts pass it. The assertion names the
    contract and the edge, not only rc 1: another contract's failure would satisfy rc 1.
    """
    tree, _ = _copy_package(tmp_path)
    _append(tree, "src/cs2rl/env/config.py", "from cs2rl.env import nav")
    r = _lint(tree)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "cs2rl.env layers BROKEN" in r.stdout, r.stdout + r.stderr
    assert "cs2rl.env.config is not allowed to import cs2rl.env.nav" in r.stdout, r.stdout
    assert "cs2rl layers KEPT" in r.stdout and "cs2rl acyclic siblings KEPT" in r.stdout, r.stdout


# One module-scope `import cs2rl.viz` site per construct that is not a def. The pin's
# docstring says none of them counts as function-local; a pin widened to exempt one
# (say, `if TYPE_CHECKING:` blocks) fails the matching case.
_MODULE_SCOPE_SHAPES = {
    "bare": "import cs2rl.viz",
    "if_type_checking": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import cs2rl.viz",
    "class_body": "class _Plant:\n    import cs2rl.viz",
    "try_except": "try:\n    import cs2rl.viz\nexcept ImportError:\n    pass",
}


@pytest.mark.parametrize("shape", sorted(_MODULE_SCOPE_SHAPES))
def test_control_a_module_scope_site_of_an_ignored_pair(tmp_path, shape):
    """(e) The blind spot: import-linter accepts it, the scope pin rejects the planted line.

    The assertion names the planted LINE, not only the pair: the pin's "no import
    site" failure carries the pair too, and would satisfy a pair-only check.
    """
    tree, _ = _copy_package(tmp_path)
    text = _MODULE_SCOPE_SHAPES[shape]
    start = _append(tree, "src/cs2rl/train.py", text)
    planted = start + next(i for i, line in enumerate(text.split("\n")) if "cs2rl.viz" in line)
    r = _lint(tree)
    assert r.returncode == 0, ("import-linter now rejects a module-scope site of an ignored "
                               "pair; the scope pin's premise changed.\n" + r.stdout + r.stderr)
    failures = scope_pin_failures(tree, _graph_facts(tree))
    assert any(f"train.py:{planted} is not inside a def" in f for f in failures), failures


def test_control_a_directory_without_init_is_covered(tmp_path):
    """(f) grimp skips noinit/ (no __init__.py), so only the coverage check sees it.

    PITFALL: the plant directory must not exist in the package. It was spec/ until #205
    part 2a made spec/ a real package.
    """
    tree, files = _copy_package(tmp_path)
    plant = "src/cs2rl/noinit/action.py"
    (tree / plant).parent.mkdir()
    (tree / plant).write_text("from cs2rl import train\n")
    r = _lint(tree)
    assert r.returncode == 0, ("import-linter now sees a directory without __init__.py; the "
                               "coverage check's premise changed.\n" + r.stdout + r.stderr)
    failures = coverage_failures(_graph_facts(tree), [*files, plant])
    assert any("cs2rl.noinit.action" in f for f in failures), failures
