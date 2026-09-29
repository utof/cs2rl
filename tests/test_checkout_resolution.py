"""A process imports THIS checkout's cs2rl, or stops: the two checkout guards.

1. The pytest tripwire in tests/conftest.py. Each case is a child pytest session
that loads the real conftest as a plugin (`-p tests.conftest`, as
tests/test_one_module_object_per_file.py does) over one
planted passing test, with a `pytest.ini` in its tmp dir so the repo's own config
stays out of it. Only the child's PYTHONPATH differs between the cases:

  (a) a fake `cs2rl` package first on PYTHONPATH: `cs2rl` no longer resolves
      under this checkout's src/, so the session must stop and name the fix;
  (b) another checkout's src/ (a tmp dir holding pyproject.toml and
      src/stray.py) on PYTHONPATH: `stray` is an importable top-level name
      beside cs2rl, so the session must stop and name it;
  negative control: neither, and the planted test runs and passes.

PITFALL: every child PREPENDS to the inherited PYTHONPATH, never replaces it.
The inherited value is what put this checkout's src/ first (in a worktree that
borrows main's venv), so replacing it would make the negative control fail (a)
for a reason that has nothing to do with the case under test.

2. The import guard in src/cs2rl/__init__.py. Each case is a child interpreter
that inherits this session's PYTHONPATH, so the cs2rl it imports is THIS
checkout's; the cases differ in the child's cwd and in what it runs.
  - Refused: `python -c "import cs2rl"` with the cwd inside a tmp checkout
    (pyproject.toml + src/cs2rl/__init__.py); a tmp checkout's script run by
    path, from a cwd in no checkout and from this checkout's root (verifier
    finding N1: the script's checkout decides, not the cwd).
  - Silent: the cwd in this checkout, in no checkout, deleted, or under a bare
    pyproject.toml; an installed copy (the Modal wheel path); no sys.argv at
    all; this checkout's script run by path from inside another checkout (N2);
    a console script or a site-packages __main__.py that sits inside another
    checkout (the shapes of `.venv/bin/pytest` and `python -m pytest`, whose
    files resolve into main's .venv), which must go by the cwd.

3. The leftover-directory check, clause (e) of the tripwire: a directory under
src/cs2rl/ that Python would import as a namespace package (an importable file,
no tracked __init__.py) stops the session. Walks on tmp layouts (positive,
negatives, an unstaged __init__.py), the fail-closed git failure, and
pytest_sessionstart itself for the wiring and the production root.
"""
import importlib.machinery
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from tests.conftest import REPO_ROOT

# Generous next to a child's ~2 s runtime; the bound exists so a wedged child
# fails with its output instead of hanging the suite.
_CHILD_TIMEOUT_S = 120
# The prefix pytest_sessionstart's UsageError carries.
_TRIPWIRE = "checkout tripwire (tests/conftest.py)"


def _session(tmp_path: Path, *first_on_path: Path) -> tuple[subprocess.CompletedProcess, str]:
    """Run a child session over one passing test with `first_on_path` ahead of PYTHONPATH."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    plant = tmp_path / "test_plant.py"
    plant.write_text("def test_plant():\n    pass\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    # REPO_ROOT makes `tests.conftest` importable.
    entries = [*map(str, first_on_path), str(REPO_ROOT), os.environ.get("PYTHONPATH")]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, entries))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    child = subprocess.run([
        sys.executable, "-m", "pytest",
        str(plant), "-p", "tests.conftest", "-p", "no:cacheprovider", "-q"
    ],
                           cwd=tmp_path,
                           env=env,
                           capture_output=True,
                           text=True,
                           timeout=_CHILD_TIMEOUT_S)
    output = (f"exit {child.returncode}\n{child.stdout[-3000:]}\n"
              f"--- stderr ---\n{child.stderr[-3000:]}")
    return child, output


def _assert_stopped_before_collection(child: subprocess.CompletedProcess, output: str) -> None:
    """Exit 4 (pytest's usage error) with the tripwire's message, and the planted test never ran."""
    assert child.returncode == pytest.ExitCode.USAGE_ERROR, (
        f"the tripwire did not stop the session\n{output}")
    assert _TRIPWIRE in child.stderr, f"the session stopped, but not by the tripwire\n{output}"
    assert "passed" not in child.stdout, f"the planted test ran: collection was not stopped\n{output}"


def test_a_foreign_cs2rl_first_on_the_path_stops_the_session_and_names_the_fix(tmp_path):
    """(a): `cs2rl` resolves somewhere other than this checkout's src/."""
    fake = tmp_path / "foreign"
    (fake / "cs2rl").mkdir(parents=True)
    (fake / "cs2rl" / "__init__.py").write_text("")
    child, output = _session(tmp_path, fake)
    _assert_stopped_before_collection(child, output)
    assert "(a)" in child.stderr and str(fake / "cs2rl") in child.stderr, (
        f"the message does not say where cs2rl resolved\n{output}")
    assert f"env PYTHONPATH={REPO_ROOT / 'src'} <command>" in child.stderr, (
        f"the message does not name the fix\n{output}")


def test_a_stray_name_in_a_checkouts_src_stops_the_session_and_names_it(tmp_path):
    """(b): a checkout's src/ on sys.path holds an importable name beside cs2rl."""
    other = tmp_path / "other_checkout"
    (other / "src").mkdir(parents=True)
    (other / "pyproject.toml").write_text("")
    (other / "src" / "stray.py").write_text("")
    child, output = _session(tmp_path, other / "src")
    _assert_stopped_before_collection(child, output)
    assert "(b)" in child.stderr and "'stray'" in child.stderr, (
        f"the message does not name the stray module\n{output}")
    assert "(a)" not in child.stderr, f"cs2rl should still resolve to this checkout\n{output}"


def test_negative_control_the_same_session_without_either_runs(tmp_path):
    """Neither condition: the tripwire is silent and the planted test passes."""
    child, output = _session(tmp_path)
    assert child.returncode == 0, f"the session failed\n{output}"
    assert "1 passed" in child.stdout, f"the planted test did not run\n{output}"
    assert _TRIPWIRE not in child.stderr, f"the tripwire fired\n{output}"


def test_b_names_the_pufferlib_origin_of_a_resources_entry(tmp_path):
    """(b)'s message says where a `resources` directory in src/ comes from (verifier V2)."""
    from tests.conftest import checkout_resolution_problems
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "src" / "resources").mkdir(parents=True)
    problems = checkout_resolution_problems(REPO_ROOT / "src", [str(tmp_path / "src")])
    assert any("['resources']" in p for p in problems), problems
    assert any("import pufferlib" in p and "Removing it is safe" in p for p in problems), problems


def _python(cwd: Path, args: list[str], *first_on_path: Path) -> subprocess.CompletedProcess:
    """`python <args>` in `cwd`, with `first_on_path` PREPENDED to PYTHONPATH."""
    entries = [*map(str, first_on_path), os.environ.get("PYTHONPATH")]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, entries)))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, *args],
                          cwd=cwd,
                          env=env,
                          capture_output=True,
                          text=True,
                          timeout=_CHILD_TIMEOUT_S)


def _import_cs2rl(cwd: Path, *first_on_path: Path) -> subprocess.CompletedProcess:
    """`python -c "import cs2rl"` in `cwd`: the cwd decides (sys.argv[0] is '-c')."""
    return _python(cwd, ["-c", "import cs2rl"], *first_on_path)


def _fake_checkout(root: Path) -> Path:
    """What the import guard calls a checkout: pyproject.toml + src/cs2rl/__init__.py."""
    (root / "src" / "cs2rl").mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    (root / "src" / "cs2rl" / "__init__.py").write_text("")
    return root.resolve()


def test_importing_cs2rl_from_inside_another_checkout_fails_and_names_the_fix(tmp_path):
    """cwd in a different checkout than the one cs2rl came from: ImportError, both named."""
    other = _fake_checkout(tmp_path / "other")
    (other / "deeper").mkdir()
    child = _import_cs2rl(other / "deeper")
    output = f"exit {child.returncode}\n--- stderr ---\n{child.stderr[-3000:]}"
    assert child.returncode != 0 and "ImportError" in child.stderr, output
    assert f"the working directory is inside another checkout, {other}." in child.stderr, output
    assert f"env PYTHONPATH={other / 'src'} <command>" in child.stderr, output
    assert f"To run {REPO_ROOT}'s code, cd {REPO_ROOT} first" in child.stderr, output


@pytest.mark.parametrize("cwd", ["no checkout", "this checkout"])
def test_a_script_of_another_checkout_run_by_path_is_refused_from_any_cwd(tmp_path, cwd):
    """N1: `python <other>/scripts/run.py`. From /tmp or from main's root the cwd
    names no checkout or cs2rl's own, so only the script's path can tell."""
    other = _fake_checkout(tmp_path / "other")
    script = other / "scripts" / "run.py"
    script.parent.mkdir()
    script.write_text("import cs2rl\n")
    child = _python(tmp_path if cwd == "no checkout" else REPO_ROOT, [str(script)])
    output = f"exit {child.returncode}\n--- stderr ---\n{child.stderr[-3000:]}"
    assert child.returncode != 0 and "ImportError" in child.stderr, output
    assert f"the script {script} is inside another checkout, {other}." in child.stderr, output
    assert f"env PYTHONPATH={other / 'src'} <command>" in child.stderr, output
    assert (f"run {REPO_ROOT}'s copy of the script instead: {REPO_ROOT / 'scripts' / 'run.py'}"
            in child.stderr), output


def test_this_checkouts_script_run_by_path_from_inside_another_checkout_is_silent(tmp_path):
    """N2: cs2rl and the script are both this checkout's; only the cwd is elsewhere."""
    child = _python(_fake_checkout(tmp_path / "other"),
                    [str(REPO_ROOT / "scripts" / "sim_fingerprint.py"), "--help"])
    assert child.returncode == 0, f"exit {child.returncode}\n{child.stderr[-3000:]}"
    assert "usage: sim_fingerprint.py" in child.stdout, child.stdout[-3000:]


@pytest.mark.parametrize("entry", ["console script", "site-packages __main__.py"])
def test_launchers_inside_another_checkout_go_by_the_cwd(tmp_path, entry):
    """`.venv/bin/pytest` and pytest's __main__.py resolve into main's .venv, i.e. into
    another checkout once main is one. They are not the user's script: the cwd decides."""
    other = _fake_checkout(tmp_path / "other")
    launcher = other / ("bin/pytest"
                        if entry == "console script" else "lib/site-packages/pytest/__main__.py")
    launcher.parent.mkdir(parents=True)
    launcher.write_text("import cs2rl\n")
    child = _python(REPO_ROOT, [str(launcher)])
    assert child.returncode == 0, f"exit {child.returncode}\n{child.stderr[-3000:]}"


@pytest.mark.parametrize("where", ["this checkout", "no checkout", "installed copy"])
def test_the_import_guard_is_silent_when_the_checkouts_agree(tmp_path, where):
    """Negative controls: this checkout's root, no checkout at all, and a wheel-style copy.

    `installed copy` is the Modal path: the package outside any src/, run with the
    cwd inside a checkout (there, the extracted archive).
    """
    if where == "this checkout":
        child = _import_cs2rl(REPO_ROOT)
    elif where == "no checkout":
        child = _import_cs2rl(tmp_path)
    else:
        site = tmp_path / "site" / "cs2rl"
        site.mkdir(parents=True)
        (site / "__init__.py").write_text((REPO_ROOT / "src" / "cs2rl" / "__init__.py").read_text())
        child = _import_cs2rl(_fake_checkout(tmp_path / "archive"), site.parent)
    assert child.returncode == 0, f"exit {child.returncode}\n{child.stderr[-3000:]}"


@pytest.mark.parametrize("case", ["deleted cwd", "pyproject.toml alone", "no sys.argv"])
def test_the_import_guard_skips_what_it_cannot_judge(tmp_path, case):
    """N3: an OSError (here a deleted cwd) skips the check; a pyproject.toml without
    src/cs2rl/__init__.py is not a checkout; an embedded interpreter may lack sys.argv."""
    if case == "deleted cwd":
        code = ("import os, tempfile\nd = tempfile.mkdtemp()\nos.chdir(d)\nos.rmdir(d)\n"
                "import cs2rl\n")
        child = _python(tmp_path, ["-c", code])
    elif case == "pyproject.toml alone":
        (tmp_path / "pyproject.toml").write_text("")
        child = _import_cs2rl(tmp_path)
    else:
        child = _python(tmp_path, ["-c", "import sys\ndel sys.argv\nimport cs2rl\n"])
    assert child.returncode == 0, f"exit {child.returncode}\n{child.stderr[-3000:]}"


# ── 3. The leftover-directory check in tests/conftest.py, clause (e) (#205 part 2b) ──
#
# A directory under src/cs2rl/ with an importable file and no tracked __init__.py is a
# namespace package: after a package move, the old directory's untracked .so imports
# silently. The real tree has none, so the session passes whether or not the check
# works; these are its controls. Each walk runs on a tmp layout with its own `tracked`
# set; the wiring and the production root run pytest_sessionstart itself.

# pytest_sessionstart never reads its session; this stands in for one.
_NO_SESSION = cast(pytest.Session, None)


def _layout(root: Path, *files: str) -> None:
    """Create each of `files` (relative to `root`) empty, with its parent directories."""
    for name in files:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("")


def test_e_flags_an_untracked_extension_left_in_a_package_directory(tmp_path):
    """Positive: c_env/ holding only a built binding .so (a pulled move's leftover) is flagged;
    a tracked package beside it, and a directory without an importable file, are not."""
    from tests.conftest import leftover_package_dirs
    so = f"binding{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    _layout(tmp_path, f"c_env/{so}", "c_env/nav_data.h", "env/__init__.py", "env/nav.py",
            "c_env_assets/beep.wav")
    tracked = {tmp_path / "env" / "__init__.py", tmp_path / "env" / "nav.py"}
    assert leftover_package_dirs(tmp_path, tracked) == [tmp_path / "c_env"]


def test_e_does_not_flag_build_outputs_bytecode_or_a_tracked_package(tmp_path):
    """Negatives: zig-out/lib/libbinding.so (not an identifier, never walked),
    __pycache__/x.pyc (pruned by name; .pyc is not a counted suffix), and a package
    whose __init__.py is tracked."""
    from tests.conftest import leftover_package_dirs
    _layout(tmp_path, "c/__init__.py", "c/zig-out/lib/libbinding.so", "c/__pycache__/x.pyc",
            "c/.zig-cache/o/binding.so", "__pycache__/y.pyc")
    assert leftover_package_dirs(tmp_path, {tmp_path / "c" / "__init__.py"}) == []


def test_e_an_unstaged_init_still_stops_the_session(tmp_path):
    """A package whose __init__.py exists on disk but is not tracked is flagged: the
    index decides, not the file system."""
    from tests.conftest import leftover_package_dirs
    _layout(tmp_path, "viz/__init__.py", "viz/render.py")
    assert leftover_package_dirs(tmp_path, set()) == [tmp_path / "viz"]


@pytest.mark.parametrize("failure", ["git fails", "no directory"])
def test_e_fails_closed_when_git_ls_files_cannot_run(tmp_path, monkeypatch, failure):
    """A `git ls-files` that fails, or cannot start, is a problem naming the failure,
    never a silent pass."""
    from tests.conftest import leftover_package_problems
    if failure == "git fails":
        monkeypatch.setenv("GIT_DIR", str(tmp_path / "no-such-git-dir"))
        problems = leftover_package_problems(tmp_path)
        assert len(problems) == 1 and "`git ls-files` failed" in problems[0], problems
    else:
        problems = leftover_package_problems(tmp_path / "no-such-directory")
        assert len(problems) == 1 and "could not run `git ls-files`" in problems[0], problems


def test_e_is_wired_into_the_session_start(monkeypatch):
    """pytest_sessionstart raises the tripwire's UsageError for a leftover problem."""
    import tests.conftest as conftest
    monkeypatch.setattr(conftest, "leftover_package_problems", lambda root: ["(e) planted"])
    with pytest.raises(pytest.UsageError) as stopped:
        conftest.pytest_sessionstart(_NO_SESSION)
    assert str(stopped.value).startswith(_TRIPWIRE) and "(e) planted" in str(stopped.value)


def test_e_walks_the_production_package_root(monkeypatch):
    """The session start checks REPO_ROOT/src/cs2rl, a directory, and its walk reaches
    the known subpackages.

    PITFALL: the real tree has no leftover, so a wrong root (src/, or one
    subpackage) would pass the session silently; this spies on the root the session
    passes, then records the directories the real check walks from it.
    """
    import tests.conftest as conftest
    real_check, real_walk = conftest.leftover_package_problems, os.walk
    roots, walked = [], []
    monkeypatch.setattr(conftest, "leftover_package_problems",
                        lambda root: roots.append(root) or real_check(root))
    conftest.pytest_sessionstart(_NO_SESSION)
    assert roots == [REPO_ROOT / "src" / "cs2rl"] and roots[0].is_dir(), roots

    def recording_walk(top, *args, **kwargs):
        for entry in real_walk(top, *args, **kwargs):
            walked.append(Path(entry[0]).name)
            yield entry

    monkeypatch.setattr(os, "walk", recording_walk)
    assert real_check(roots[0]) == []
    assert {"env", "spec", "eval"} <= set(walked), walked
