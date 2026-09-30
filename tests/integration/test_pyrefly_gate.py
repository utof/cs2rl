"""Tests for scripts/pyrefly_gate.py -- the pyrefly type gate.

Each test here names a failure the gate claims to close. A claim with no test is
the one that regresses, and four review rounds of this design each shipped tests
that passed against the very gate they were written to reject. So:

**Every test asserts on named content** -- a specific path, a specific error
entry, a removal -- never on the exit code alone. An exit-code-only assertion is
satisfied by a gate that aborted for an unrelated reason, which is exactly how
the earlier versions went wrong.

**Four of these tests are mode-distinguishing** and are the reason the gate is
built the way it is: 2 (code tree), 3 (snapshot), 11 (config) and 13 (cwd
binding). Deleting any of them leaves the rest of the suite green against a gate
that reads the working tree -- the index and worktree error sets are otherwise
byte-identical.

The companion knock-out record lives at
.superpowers/sdd/2026-09-13-modal-split-163/pyrefly-B-knockout.md: each of those
properties was verified by breaking the gate and observing the named test fail.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

REPO = REPO_ROOT
GATE = REPO / "scripts" / "pyrefly_gate.py"
REAL_PYTHON = REPO / ".venv" / "bin" / "python"
REAL_PYREFLY = REPO / ".venv" / "bin" / "pyrefly"

WIDE_CONFIG = """\
preset = "default"
project-includes = ["src", "scripts"]
search-path = ["src", "scripts", "."]
"""


def run_gate(project: Path, *args: str) -> subprocess.CompletedProcess:
    """Invoke the gate out-of-process, always from the repo root.

    cwd stays at REPO on purpose: `uv run` and anything that resolves a project
    must never be launched from /tmp, where uv tries to build the directory as a
    project. The gate itself never chdirs either.
    """
    return subprocess.run(
        [str(REAL_PYTHON), str(GATE), "--project",
         str(project), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def git(project: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(project), *args], capture_output=True, text=True)
    assert out.returncode == 0, f"git {args} failed: {out.stderr}"
    return out.stdout


def make_project(root: Path,
                 files: dict[str, str],
                 *,
                 config: str = WIDE_CONFIG,
                 interpreter: str = "real") -> Path:
    """Build a git-initialised synthetic project the gate can actually check.

    Every synthetic project needs FOUR things, not one. Missing any of them makes
    the gate abort before it reaches the behaviour under test, which turns a
    real test into a vacuous one:

      1. a git repo, because the gate materialises an index;
      2. the plant STAGED, because a working-tree-only file is invisible to it;
      3. a committed pyrefly.toml, because a missing config is a hard abort;
      4. a .venv/bin/ holding python and pyrefly, because the gate resolves both
         from <project>/.venv.

    `interpreter` selects the shape of <project>/.venv/bin/python, for test 10:
    "real", "missing", "not_executable", or "bad_shebang".
    """
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "t")

    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    (root / "pyrefly.toml").write_text(config)

    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "pyrefly").symlink_to(REAL_PYREFLY)
    if interpreter == "real":
        (venv_bin / "python").symlink_to(REAL_PYTHON)
    elif interpreter == "not_executable":
        (venv_bin / "python").write_text("#!/bin/sh\nexit 0\n")
        (venv_bin / "python").chmod(0o644)
    elif interpreter == "bad_shebang":
        (venv_bin / "python").write_text("#!/nonexistent/interpreter\n")
        (venv_bin / "python").chmod(0o755)
    elif interpreter != "missing":
        raise ValueError(interpreter)

    git(root, "add", "pyrefly.toml", *files.keys())
    return root


def freeze(project: Path) -> None:
    """Snapshot the project's current staged state and stage the snapshot."""
    res = run_gate(project, "--update")
    assert res.returncode == 0, res.stderr
    git(project, "add", "pyrefly-snapshot.json")


# --------------------------------------------------------------------------
# 1 -- the happy path
# --------------------------------------------------------------------------


def test_the_snapshot_matches_the_committed_index():
    """The real repo's committed snapshot matches its committed code.

    Hermetic BECAUSE the gate reads the index: a developer with an in-progress
    type error in their working tree no longer reds this unrelated test.
    """
    res = subprocess.run([str(REAL_PYTHON), str(GATE)], capture_output=True, text=True, cwd=REPO)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    assert "ADDED" not in res.stdout
    assert "REMOVED" not in res.stdout


# --------------------------------------------------------------------------
# 2 -- MODE-DISTINGUISHING: the code tree
# --------------------------------------------------------------------------


def test_a_broken_staged_version_reds_even_when_the_working_tree_is_clean(tmp_path):
    """Stage a type error, then fix the working copy. The gate must still red.

    This is the test that separates an index-reading gate from a worktree-reading
    one, and it is the reason `git checkout-index` is in the design. Every other
    test in this file passes under either mode, because the two error sets are
    otherwise identical down to line and column. If you are deleting this test,
    that is the failure -- not the test.
    """
    proj = make_project(tmp_path / "p", {"src/m.py": "def f() -> int:\n    return 1\n"})
    freeze(proj)

    (proj / "src" / "m.py").write_text('def f() -> int:\n    return "broken"\n')
    git(proj, "add", "src/m.py")
    (proj / "src" / "m.py").write_text("def f() -> int:\n    return 1\n") # worktree CLEAN

    res = run_gate(proj)
    assert res.returncode == 1
    assert "ADDED 1" in res.stdout, res.stdout
    assert "src/m.py" in res.stdout
    assert "bad-return" in res.stdout
    # The gate reached pyrefly rather than aborting on a guard -- otherwise a
    # broken gate that aborts early would satisfy the assertions above.
    assert "materialised" in res.stderr


# --------------------------------------------------------------------------
# 3 -- MODE-DISTINGUISHING: the snapshot
# --------------------------------------------------------------------------


def test_a_gutted_staged_snapshot_reds_even_when_the_working_tree_snapshot_is_intact(tmp_path):
    """Stage an emptied snapshot while the working-tree copy stays correct.

    The symmetric argument to test 2, one level down. A gate that reads the
    snapshot from the working tree compares 752 against an intact 752, exits 0,
    and lets someone break the freeze for everyone who commits next -- while
    showing them green.
    """
    proj = make_project(tmp_path / "p", {"src/m.py": 'def f() -> int:\n    return "x"\n'})
    freeze(proj)
    intact = (proj / "pyrefly-snapshot.json").read_text()
    assert intact.strip(), "fixture must freeze at least one error"

    (proj / "pyrefly-snapshot.json").write_text("")                    # gutted, staged
    git(proj, "add", "pyrefly-snapshot.json")
    (proj / "pyrefly-snapshot.json").write_text(intact)                # worktree EXACTLY correct

    res = run_gate(proj)
    assert res.returncode == 1
    assert "ADDED 1" in res.stdout, res.stdout
    assert "src/m.py" in res.stdout


# --------------------------------------------------------------------------
# 4 -- the column-blindness control, with its own positive control
# --------------------------------------------------------------------------


def test_a_new_error_at_a_column_the_baseline_would_blind_is_caught(tmp_path):
    """A second error at the same start column: --baseline green, this gate red.

    Without the --baseline arm this test discriminates nothing -- column is not
    in the gate's key, so any key at all would catch the plant. The two arms are
    what earn the name and what keeps the 64.9% measurement honest.
    """
    before = "class A:\n    pass\n\na = A()\nx = a.foo\n"
    after = "class A:\n    pass\n\na = A()\nx = a.foo\ny = a.bar\n"
    proj = make_project(tmp_path / "p", {"src/m.py": before})

    # --- arm 1: pyrefly's own --baseline, which is a TWO-step workflow.
    # --baseline <FILE> takes a required argument and does not create the file;
    # --update-baseline is what writes it.
    baseline = tmp_path / "baseline.json"
    env = {**os.environ, "VIRTUAL_ENV": ""}
    common = [
        str(REAL_PYREFLY), "check", "-c",
        str(proj / "pyrefly.toml"), "--python-interpreter-path",
        str(REAL_PYTHON)
    ]
    subprocess.run(
        [*common, "--baseline", str(baseline), "--update-baseline"],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=env)
    (proj / "src" / "m.py").write_text(after)
    arm = subprocess.run([*common, "--baseline", str(baseline)],
                         capture_output=True,
                         text=True,
                         cwd=REPO,
                         env=env)
    assert arm.returncode == 0, (
        "--baseline was expected to SWALLOW the same-column error. If this now "
        f"fails, pyrefly's suppression key changed.\n{arm.stdout}\n{arm.stderr}")

    # --- arm 2: this gate, on the same fixture
    (proj / "src" / "m.py").write_text(before)
    git(proj, "add", "src/m.py")
    freeze(proj)
    (proj / "src" / "m.py").write_text(after)
    git(proj, "add", "src/m.py")

    res = run_gate(proj)
    assert res.returncode == 1
    assert "ADDED 1" in res.stdout, res.stdout
    assert "bar" in res.stdout


# --------------------------------------------------------------------------
# 5 -- THE KEY CONTROL: a count-preserving substitution
# --------------------------------------------------------------------------


def test_a_count_preserving_substitution_is_caught(tmp_path):
    """Delete one error and add a different one with the same message.

    Tests that merely ADD an error are tautologies here: under a multiset
    compared by exact equality the multiset grows under any key at all --
    (path,), (), even a bare len(). Only a count-preserving substitution can tell
    whether `source_line` is in the key, which makes this the one test that
    controls the key's design.
    """
    before = "def f(n: None) -> None:\n    a = n[0]\n    b = n[1]\n"
    # Same path, same name, same concise_description; DIFFERENT source line.
    after = "def f(n: None) -> None:\n    a = n[0]\n    c = n[2]\n"
    proj = make_project(tmp_path / "p", {"src/m.py": before})
    freeze(proj)

    (proj / "src" / "m.py").write_text(after)
    git(proj, "add", "src/m.py")

    res = run_gate(proj)
    assert res.returncode == 1, (
        "count-preserving substitution went undetected -- source_line is not in "
        f"the key.\n{res.stdout}")
    assert "ADDED 1" in res.stdout, res.stdout
    assert "REMOVED 1" in res.stdout, res.stdout


# --------------------------------------------------------------------------
# 6 -- the remediation loop, end to end
# --------------------------------------------------------------------------


def test_the_remediation_loop_actually_recovers(tmp_path):
    """Red -> --update -> git add -> green, and the snapshot is deterministic.

    Asserting that the banner CONTAINS "--update" would hold by construction,
    since the implementer types that string. What matters is that following it
    works: --update must write the WORKING TREE (not the temp dir it reads), or
    the second attempt reds identically and the user concludes the gate is broken.
    """
    proj = make_project(tmp_path / "p", {"src/m.py": 'def f() -> int:\n    return "x"\n'})
    freeze(proj)

    (proj / "src" / "m.py").write_text("def f() -> int:\n    return 1\n")
    git(proj, "add", "src/m.py")

    red = run_gate(proj)
    assert red.returncode == 1
    assert "REMOVED 1" in red.stdout, red.stdout
    assert "--update" in red.stdout
    assert "git add pyrefly-snapshot.json" in red.stdout

    upd = run_gate(proj, "--update")
    assert upd.returncode == 0, upd.stderr
    first = (proj / "pyrefly-snapshot.json").read_bytes()
    git(proj, "add", "pyrefly-snapshot.json")

    green = run_gate(proj)
    assert green.returncode == 0, f"{green.stdout}\n{green.stderr}"

    # Determinism, so that "merges textually" is a property somebody checks.
    run_gate(proj, "--update")
    assert (proj / "pyrefly-snapshot.json").read_bytes() == first
    text = (proj / "pyrefly-snapshot.json").read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert all(json.loads(ln) for ln in lines), "one JSON object per line"


# --------------------------------------------------------------------------
# 7 -- coverage, measured rather than proxied
# --------------------------------------------------------------------------


def test_every_tracked_python_file_is_covered_by_the_config(tmp_path):
    """Every tracked .py is in pyrefly's scope, and nothing untracked is.

    Run against a MATERIALISED INDEX, not the working tree. Against the working
    tree, one untracked scratch .py under src/ or scripts/ takes the covered
    count above the tracked count and reds this test for a developer who did
    nothing wrong.
    """
    # NOT tmp_path: pytest's basetemp here lives under a dot-directory, and
    # pyrefly silently skips every project-includes pattern whose absolute path
    # has a hidden ancestor -- measured, the same tree yields 136 covered files
    # under /tmp/x and 26 under /tmp/.x. The gate is unaffected because its own
    # mkdtemp lands in /tmp, but this test materialises its own tree and would
    # otherwise compare 26 against 135 and blame the config.
    tree = Path(tempfile.mkdtemp(prefix="pyrefly-cov-"))
    try:
        git(REPO, "checkout-index", "-a", f"--prefix={tree}/")

        out = subprocess.run(
            [
                str(REAL_PYREFLY), "dump-config", "-c",
                str(tree / "pyrefly.toml"), "--max-files", "all"
            ],
            capture_output=True,
            text=True,
            cwd=REPO,
        ).stdout
        covered = {
            str(Path(ln.strip()).resolve().relative_to(tree.resolve()))
            for ln in out.splitlines() if ln.strip().endswith(".py") and str(tree) in ln
        }
    finally:
        shutil.rmtree(tree, ignore_errors=True)
    tracked = {p for p in git(REPO, "ls-files", "*.py").split() if p}

    assert tracked - covered == set(), f"tracked but unchecked: {sorted(tracked - covered)}"
    assert covered - tracked == set(), f"checked but untracked: {sorted(covered - tracked)}"


# --------------------------------------------------------------------------
# 8 -- the highest-severity silent green
# --------------------------------------------------------------------------


def test_empty_pyrefly_stdout_aborts_and_is_never_swallowed(tmp_path):
    """pyrefly matching no files must abort, not read as "zero differences".

    Measured: 0 bytes on stdout, exit 1, message on stderr only. A gate that
    catches JSONDecodeError and carries on turns the most complete failure the
    tool has into a green commit.

    The fixture's snapshot is deliberately EMPTY, so that a gate which swallowed
    the error and substituted [] would compare [] against [] and exit 0 -- making
    this test fail on the exit code, which is the signal we want load-bearing.
    """
    proj = make_project(
        tmp_path / "p",
        {"src/m.py": "x = 1\n"},
        config='preset = "default"\nproject-includes = ["nonexistent"]\n',
    )
    (proj / "pyrefly-snapshot.json").write_text("")
    git(proj, "add", "pyrefly-snapshot.json")

    res = run_gate(proj)
    assert res.returncode == 1, f"empty stdout was swallowed:\n{res.stdout}"
    assert "matched no files" in res.stderr, res.stderr
    assert "ADDED" not in res.stdout and "REMOVED" not in res.stdout


# --------------------------------------------------------------------------
# 9 -- the silently-truncated tree
# --------------------------------------------------------------------------


def test_an_unmerged_index_aborts_by_name(tmp_path):
    """An unmerged index must abort naming the path, before anything else.

    `git checkout-index -a` exits 0 and writes NOTHING for a file at stages 2/3
    -- no error, no warning. git commit refuses an unmerged index, so the hook
    path is protected, but a manual run is not.
    """
    proj = make_project(tmp_path / "p", {"src/m.py": "x = 1\n"})
    freeze(proj)
    git(proj, "commit", "-q", "-m", "base")

    git(proj, "checkout", "-q", "-b", "other")
    (proj / "src" / "m.py").write_text("x = 2\n")
    git(proj, "commit", "-qam", "two")
    git(proj, "checkout", "-q", "master" if "master" in git(proj, "branch") else "main")
    (proj / "src" / "m.py").write_text("x = 3\n")
    git(proj, "commit", "-qam", "three")
    subprocess.run(["git", "-C", str(proj), "merge", "other"], capture_output=True)

    res = run_gate(proj)
    assert res.returncode == 1
    assert "unmerged" in res.stderr, res.stderr
    assert "src/m.py" in res.stderr
    # It must abort BEFORE running pyrefly, not check a tree missing that file.
    assert "ADDED" not in res.stdout and "REMOVED" not in res.stdout


# --------------------------------------------------------------------------
# 10 -- the interpreter guard, probed at its BOUNDARY
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["missing", "not_executable", "bad_shebang"])
def test_a_bad_interpreter_aborts_instead_of_reporting_328_additions(tmp_path, shape):
    """All three shapes of "bad interpreter" must abort.

    pyrefly does NOT fail on a bad --python-interpreter-path: it prints a WARN on
    stderr, falls back to the default environment, and returns a full,
    plausible-looking result set -- 826 errors here against a frozen 752, i.e.
    ADDED 328 / REMOVED 254, 270 of the additions bare missing-import.

    Only "missing" is caught by an existence check. The other two return True
    from exists() and reach the fallback anyway, which is why the gate probes the
    interpreter by running it. A fixture drawn only from inside the guard's own
    trigger set can never find the boundary.
    """
    proj = make_project(tmp_path / "p", {"src/m.py": "x = 1\n"}, interpreter=shape)
    (proj / "pyrefly-snapshot.json").write_text("")
    git(proj, "add", "pyrefly-snapshot.json")

    res = run_gate(proj)
    assert res.returncode == 1
    assert "not a working interpreter" in res.stderr, res.stderr
    assert "ADDED" not in res.stdout, "the gate produced a diff instead of aborting"


# --------------------------------------------------------------------------
# 11 -- MODE-DISTINGUISHING: the config
# --------------------------------------------------------------------------


def test_a_narrowed_staged_config_reds_even_when_the_working_tree_config_is_intact(tmp_path):
    """Stage a narrowed project-includes; the working-tree config stays wide.

    The third sourcing property, after the code tree and the snapshot. This is
    also the only test that exercises the REMOVED half of exact-equality
    polarity against a real narrowing -- an additions-only gate reads a config
    that drops half the repo as success.
    """
    files = {
        "src/a.py": 'def f() -> int:\n    return "x"\n',
        "scripts/b.py": 'def g() -> int:\n    return "y"\n'
    }
    proj = make_project(tmp_path / "p", files)
    freeze(proj)
    assert "scripts/b.py" in (proj / "pyrefly-snapshot.json").read_text()

    narrow = 'preset = "default"\nproject-includes = ["src"]\nsearch-path = ["src", "scripts", "."]\n'
    (proj / "pyrefly.toml").write_text(narrow)
    git(proj, "add", "pyrefly.toml")
    (proj / "pyrefly.toml").write_text(WIDE_CONFIG)    # worktree still WIDE

    res = run_gate(proj)
    assert res.returncode == 1, ("a narrowed staged config went undetected -- the gate read the "
                                 f"working-tree config.\n{res.stdout}")
    assert "REMOVED 1" in res.stdout, res.stdout
    assert "scripts/b.py" in res.stdout


# --------------------------------------------------------------------------
# 12 -- a file that does not parse
# --------------------------------------------------------------------------


def test_a_conflicted_file_prints_the_path_not_the_wall(tmp_path):
    """Unresolved conflict markers give the path and the cause, not 18 entries.

    pyrefly recovers from a syntax error rather than skipping the file, so one
    unmerged 6-line file measured 18 errors. Printing that as a diff buries the
    one fact the user needs.
    """
    proj = make_project(tmp_path / "p", {"src/m.py": "x = 1\n"})
    freeze(proj)

    (proj / "src" / "m.py").write_text("<<<<<<< HEAD\nx = 1\n=======\nx = 2\n>>>>>>> other\n")
    git(proj, "add", "src/m.py")

    res = run_gate(proj)
    assert res.returncode == 1
    assert "does not parse" in res.stderr, res.stderr
    assert "src/m.py" in res.stderr
    assert "ADDED" not in res.stdout, "printed the wall instead of the cause"


# --------------------------------------------------------------------------
# 13 -- MODE-DISTINGUISHING: the cwd binding
# --------------------------------------------------------------------------


def test_project_from_a_subdirectory_does_not_truncate_the_tree(tmp_path):
    """--project from a subdirectory must still see the whole index.

    The gate never chdirs, so a bare `git` call would be scoped to the caller's
    cwd: ls-files -u reports 0 on an unmerged index, checkout-index materialises
    only the subtree with exit 0 and no warning, and the parity check then
    compares a truncated tree against an identically truncated ls-files -- 1 == 1,
    green. Every git call is bound with -C <project> to prevent that.

    The parity counts are read from stderr because the temp tree is removed in a
    finally; that emission is this assertion's only observation channel.
    """
    files = {"src/a.py": "x = 1\n", "scripts/b.py": "y = 2\n", "scripts/deep/c.py": "z = 3\n"}
    proj = make_project(tmp_path / "p", files)
    freeze(proj)

    res = subprocess.run(
        [str(REAL_PYTHON), str(GATE), "--project",
         str(proj)],
        capture_output=True,
        text=True,
        cwd=proj / "scripts",
    )
    assert res.returncode == 0, f"{res.stdout}\n{res.stderr}"
    # 3 .py tracked; a truncated run from scripts/ would report 2.
    assert "materialised 3 .py, index has 3" in res.stderr, res.stderr
