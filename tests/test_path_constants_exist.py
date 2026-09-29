"""Every path constant whose consumer TOLERATES a missing path names a tracked path.

WHAT: one case per constant. Each one must match at least one tracked file,
checked with `git ls-files --error-unmatch`, which also works for a directory
(it matches the files under it). The constants:
  - train_bc.DEMO_RELEVANT_PATHS;
  - run_experiment's TRAIN_PY / REWARDS_H / ENV_C / C_ENV_DIR;
  - sync_action_spec's OUTPUT / OBS_OUTPUT, the generated spec modules (#205
    moved them into spec/; the generator would write a stray file at a stale
    path and leave the real one unregenerated). Both are tracked, so a row
    cannot tell them apart: test_sync_action_spec_outputs_are_not_swapped does;
  - tests/_helpers/metrics_census.py's SRC, which must be a DIRECTORY (the
    pathspec's trailing `/`): its sweeps walk `SRC.rglob("*.py")`, which yields
    nothing on a wrong SRC, so they would pass vacuously;
  - every MANIFEST.in include;
  - setup.py's `source_dir`.

The ROOTS those constants hang off are pinned too, against the conftest's
REPO_ROOT and never against their own: a case built relative to a module's own
REPO_ROOT cannot see that root being wrong (#207 found the run_experiment rows
blind to it: its REPO_ROOT one level off still left them green). So every
module constant above is taken relative to the conftest's REPO_ROOT.

WHY a standing test and not a one-shot check (#199 verifier finding V3): none of
these consumers fails on a wrong path, they go quiet.
  - `git diff --quiet <sha> -- <path>` is quiet for a path on neither side, so
    check_demo_sha stops watching it (#199's knock-out, from nav's path before
    #205: `src/cs2rl/nav.py` -> `src/nav.py` left the whole BC suite green).
  - cs2rl.experiment.lib.env_fingerprint skips a missing file, and path_last_commit_sha
    returns "".
  - A MANIFEST.in line that matches nothing only warns during the build.
A rename that forgets one of them is then invisible. Here it is a red test
naming the constant.

PITFALL: setup.py is read with ast, never imported (importing it runs
setuptools on pytest's argv).

scripts/run_experiment.py is a CLI, not a library (#204), so its constants are
read with `runpy.run_path(..., run_name=...)`: the module-level code is only path
constants and imports, and a run_name other than "__main__" skips main().
scripts/sync_action_spec.py is read the same way; #205 moved its side effects
into main() so that reading it writes nothing, and
test_sync_action_spec_module_body_runs_nothing keeps them there.

Also here: nav.CACHE_PATH, the vis cache, is pinned to its one location
(test_vis_cache_path_is_the_package_root_file). Its file is untracked, so it has
no row above.

Also here, because it is a rule about where test code may live: tests/_helpers/
holds no collectable file (test_helpers_hold_no_collectable_file).
"""
import ast
import runpy
import subprocess
from pathlib import Path

import pytest

from cs2rl.train_bc import DEMO_RELEVANT_PATHS
from tests._helpers import metrics_census
from tests.conftest import REPO_ROOT

run_experiment = runpy.run_path(str(REPO_ROOT / "scripts" / "run_experiment.py"),
                                run_name="run_experiment_constants")
sync_action_spec = runpy.run_path(str(REPO_ROOT / "scripts" / "sync_action_spec.py"),
                                  run_name="sync_action_spec_constants")


def _manifest_paths():
    """`include` paths, and `recursive-include DIR PATTERN` as the git pathspec DIR/PATTERN."""
    for line in (REPO_ROOT / "MANIFEST.in").read_text().splitlines():
        words = line.split()
        if words[:1] == ["include"]:
            yield from words[1:]
        elif words[:1] == ["recursive-include"]:
            # A git pathspec `*` also crosses `/`, so DIR/PATTERN is recursive.
            yield from (f"{words[1]}/{pattern}" for pattern in words[2:])


def _setup_source_dirs():
    """Every `source_dir=` string literal passed to a call in setup.py."""
    tree = ast.parse((REPO_ROOT / "setup.py").read_text())
    return [
        ast.literal_eval(kw.value) for node in ast.walk(tree) if isinstance(node, ast.Call)
        for kw in node.keywords if kw.arg == "source_dir"
    ]


def _in_this_checkout(path: Path) -> str:
    """`path` as a pathspec relative to the conftest's REPO_ROOT.

    A path outside this checkout stays absolute, so git rejects it and the case
    fails naming it, instead of relative_to() raising at collection.
    """
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return str(resolved)


_RUN_EXPERIMENT = ("TRAIN_PY", "REWARDS_H", "ENV_C", "C_ENV_DIR")
CASES = [
    *[(f"DEMO_RELEVANT_PATHS:{p}", p) for p in DEMO_RELEVANT_PATHS],
    *[(f"run_experiment.{n}", _in_this_checkout(run_experiment[n])) for n in _RUN_EXPERIMENT],
    *[(f"sync_action_spec.{n}", _in_this_checkout(sync_action_spec[n]))
      for n in ("OUTPUT", "OBS_OUTPUT")],
    ("metrics_census.SRC", f"{_in_this_checkout(metrics_census.SRC)}/"),
    *[(f"MANIFEST.in:{p}", p) for p in _manifest_paths()],
    *[(f"setup.py:source_dir={p}", p) for p in _setup_source_dirs()],
]

# The roots the module constants above hang off, each read from its module.
ROOTS = {
    "run_experiment.REPO_ROOT": run_experiment["REPO_ROOT"],
    "metrics_census.REPO_ROOT": metrics_census.REPO_ROOT,
}


def test_every_source_is_represented():
    """Each consumer contributes cases, and each module its root, so neither passes vacuously.

    PITFALL: ROOTS is a dict, and deleting its row deletes the only case that
    would object: the root parametrize below just runs one case fewer.
    """
    labels = [label for label, _ in CASES]
    for prefix in ("DEMO_RELEVANT_PATHS:", "run_experiment.", "sync_action_spec.",
                   "metrics_census.", "MANIFEST.in:", "setup.py:"):
        assert any(label.startswith(prefix) for label in labels), f"no case from {prefix}"
    for module in ("run_experiment", "metrics_census"):
        assert f"{module}.REPO_ROOT" in ROOTS, f"{module}'s root is not pinned in ROOTS"


@pytest.mark.parametrize("name", sorted(ROOTS))
def test_root_constant_is_this_checkouts_root(name):
    """A module's REPO_ROOT is the conftest's, the one root no module can move.

    PITFALL: never compare a root with anything derived from itself. After #207
    moved metrics_census one level down, a stale `parents[1]` would still build
    a self-consistent tree under tests/, and its sweep of SRC would pass empty.
    """
    root = ROOTS[name]
    assert root.resolve() == REPO_ROOT, (
        f"{name} is {root}, not this checkout's root {REPO_ROOT}: every path built on it "
        "points into the wrong tree. Fix its `parents[N]`.")


def test_vis_cache_path_is_the_package_root_file():
    """nav.CACHE_PATH is src/cs2rl/vis_cache.npy, the one vis cache every worktree provisions.

    WHY: the cache is gitignored and takes minutes to rebuild. #205 moved nav.py
    one level down, and a path built from nav's own directory would silently look
    in src/cs2rl/env/, miss, and rebuild (then write a second cache there).
    PITFALL: compare paths, never `samefile`: the file is absent on a cold
    checkout and on Modal, where this must still pass. nav is imported here, not
    at module level, so collecting this file stays light.
    """
    from cs2rl.env import nav

    assert Path(nav.CACHE_PATH).resolve() == REPO_ROOT / "src" / "cs2rl" / "vis_cache.npy", (
        f"nav.CACHE_PATH is {nav.CACHE_PATH}: anchor it on the cs2rl package root")


def test_sync_action_spec_outputs_are_not_swapped():
    """OUTPUT is spec/action.py and OBS_OUTPUT is spec/obs.py.

    WHY: the tracked-path rows pass them swapped, since both files are tracked. The
    generator would then write the obs layout into action.py and the action layout
    into obs.py, and the next import of either would break far from the cause.
    """
    assert Path(sync_action_spec["OUTPUT"]).name == "action.py", sync_action_spec["OUTPUT"]
    assert Path(sync_action_spec["OBS_OUTPUT"]).name == "obs.py", sync_action_spec["OBS_OUTPUT"]


def test_sync_action_spec_module_body_runs_nothing():
    """scripts/sync_action_spec.py's module body is its docstring, imports, assignments,
    defs and the `if __name__ == "__main__":` guard, and no assignment calls one of
    its own defs.

    WHY: this file loads the script with runpy.run_path at COLLECTION, under a
    run_name that skips main(). Any other top-level statement runs on every
    collection, and a stray `main()` would regenerate spec/action.py and spec/obs.py.
    PITFALL: an assignment is a statement kind this allows, so the call check is what
    stops `X = main()`; a call through anything but a bare name of a top-level def
    (an attribute, an alias) is not seen.
    """
    tree = ast.parse((REPO_ROOT / "scripts" / "sync_action_spec.py").read_text(encoding="utf-8"))
    defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    own = {node.name for node in tree.body if isinstance(node, defs)}
    bad = []
    for i, node in enumerate(tree.body):
        docstring = (i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                     and isinstance(node.value.value, str))
        main_guard = (isinstance(node, ast.If) and not node.orelse
                      and ast.unparse(node.test) == "__name__ == '__main__'")
        if docstring or main_guard or isinstance(node, defs):
            continue
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            bad.append(f":{node.lineno} {type(node).__name__}")
        bad += [
            f":{node.lineno} calls {call.func.id}()" for call in ast.walk(node) if
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in own
        ]
    assert not bad, (f"scripts/sync_action_spec.py runs code when loaded: {bad}. Move it into "
                     "main(); test collection loads this script.")


def _collectable_files_under(directory: Path) -> list[str]:
    """Each file under `directory`, at any depth, that pytest's default patterns collect.

    Relative to `directory`, sorted. PITFALL: rglob, never glob: pytest collects
    a test file one directory down just the same.
    """
    return sorted(
        p.relative_to(directory).as_posix() for p in directory.rglob("*.py")
        if p.name.startswith("test_") or p.name.endswith("_test.py"))


def test_helpers_hold_no_collectable_file():
    """tests/_helpers/ holds no test file (#207).

    pytest would collect one there, but the gates that scan test files with a
    flat, non-recursive `(ROOT / "tests").glob(...)` (tests/test_modal_packaging.py,
    tests/test_modal_preflight.py, tests/test_modal_training.py) would miss it.
    """
    found = _collectable_files_under(REPO_ROOT / "tests" / "_helpers")
    assert not found, f"move these test files out of tests/_helpers/: {found}"


def test_the_helpers_scan_finds_a_test_file_at_any_depth(tmp_path):
    """The scan above on a tmp tree: both patterns, flat and nested; near-misses skipped.

    The real tests/_helpers/ holds no test file, so the check above is green
    whether or not its scan works; this is its positive control. PITFALL
    guarded: a flat glob() passes a test file one directory down (the #207
    verifier's surviving V19 mutant).
    """
    for name in ("test_flat.py", "flat_test.py", "sub/test_x.py", "sub/deeper/y_test.py",
                 "helper.py", "sub/testing.py", "sub/test_data.txt"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("")
    assert _collectable_files_under(tmp_path) == [
        "flat_test.py", "sub/deeper/y_test.py", "sub/test_x.py", "test_flat.py"
    ]


@pytest.mark.parametrize("path", [p for _, p in CASES], ids=[label for label, _ in CASES])
def test_path_constant_names_a_tracked_path(path):
    r = subprocess.run(["git", "ls-files", "--error-unmatch", "--", path],
                       cwd=REPO_ROOT,
                       capture_output=True,
                       text=True)
    assert r.returncode == 0 and r.stdout.strip(), (
        f"{path!r} matches no tracked file. Its consumer tolerates that silently; see "
        f"this file's docstring.\n{r.stderr}")
