"""Every path constant whose consumer TOLERATES a missing path names a tracked path.

WHAT: one case per constant. Each one must match at least one tracked file,
checked with `git ls-files --error-unmatch`, which also works for a directory
(it matches the files under it). The constants:
  - train_bc.DEMO_RELEVANT_PATHS;
  - run_experiment's TRAIN_PY / REWARDS_H / ENV_C / C_ENV_DIR;
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
    check_demo_sha stops watching it (knock-out: `src/cs2rl/nav.py` ->
    `src/nav.py` left the whole BC suite green).
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
    """Each consumer contributes cases, so an emptied source cannot pass vacuously."""
    labels = [label for label, _ in CASES]
    for prefix in ("DEMO_RELEVANT_PATHS:", "run_experiment.", "metrics_census.", "MANIFEST.in:",
                   "setup.py:"):
        assert any(label.startswith(prefix) for label in labels), f"no case from {prefix}"


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


def test_helpers_hold_no_collectable_file():
    """tests/_helpers/ holds no test file (#207).

    pytest would collect one there, but the gates that scan test files with a
    flat, non-recursive `(ROOT / "tests").glob(...)` (tests/test_modal_packaging.py,
    tests/test_modal_preflight.py, tests/test_modal_training.py) would miss it.
    """
    helpers = REPO_ROOT / "tests" / "_helpers"
    found = sorted(
        p.relative_to(REPO_ROOT).as_posix() for p in helpers.rglob("*.py")
        if p.name.startswith("test_") or p.name.endswith("_test.py"))
    assert not found, f"move these test files out of tests/_helpers/: {found}"


@pytest.mark.parametrize("path", [p for _, p in CASES], ids=[label for label, _ in CASES])
def test_path_constant_names_a_tracked_path(path):
    r = subprocess.run(["git", "ls-files", "--error-unmatch", "--", path],
                       cwd=REPO_ROOT,
                       capture_output=True,
                       text=True)
    assert r.returncode == 0 and r.stdout.strip(), (
        f"{path!r} matches no tracked file. Its consumer tolerates that silently; see "
        f"this file's docstring.\n{r.stderr}")
