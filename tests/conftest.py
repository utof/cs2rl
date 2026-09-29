import importlib.machinery
import importlib.util
import os
import subprocess
import sys
from collections.abc import Collection, Generator, Iterable, Mapping
from pathlib import Path

import pytest

# The repo root, from this file's own location. Never `config.rootpath`, which
# narrows to a subdirectory when pytest is started from one.
REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Checkout tripwire: this session imports THIS checkout's cs2rl (#199) ──
#
# The code is one package, `cs2rl`, under src/, found through sys.path: the
# editable install's .pth, or PYTHONPATH. Nothing here inserts src/ itself. The
# .pth belongs to the shared .venv and names ONE checkout's src/ (main's), so a
# worktree borrowing that venv would silently test main's code unless its own
# src/ comes first. The session stops before collection when either holds:
#   (a) `cs2rl` does not resolve under this conftest's own src/. Fix: put
#       <this checkout>/src first on PYTHONPATH.
#   (b) a checkout's src/ on sys.path (basename `src`, parent holds
#       pyproject.toml) holds an importable top-level name other than cs2rl: a
#       leftover flat module, a stale binding*.so or bytecode, or an old package
#       or namespace directory such as c_env/. Such a name imports silently
#       where it should fail, so a bare `import train` left anywhere would load
#       it instead of raising.
# WHY NOT pytest's `pythonpath = ["src"]` ini option: it edits only this
# process's sys.path. Child interpreters (the many tests that launch
# `-m cs2rl.train`, or code strings) do not inherit it and would import through
# the .pth, i.e. possibly another checkout's code. PYTHONPATH reaches them.
# tests/test_checkout_resolution.py pins (a), (b) and a negative control. At import,
# src/cs2rl/__init__.py's guard also refuses a script run by path, or else a cwd,
# that sits in a checkout other than the one cs2rl came from.

# Longest first, so `binding.cpython-312-x86_64-linux-gnu.so` strips the whole
# ABI tag (name `binding`) before the bare `.so` suffix could leave a non-name.
_IMPORTABLE_SUFFIXES = sorted(importlib.machinery.SOURCE_SUFFIXES +
                              importlib.machinery.BYTECODE_SUFFIXES +
                              importlib.machinery.EXTENSION_SUFFIXES,
                              key=len,
                              reverse=True)


def checkout_resolution_problems(own_src: Path, path_entries: Iterable[str]) -> list[str]:
    """Why this process would not import `cs2rl` from `own_src` alone; empty when it would.

    `path_entries` is sys.path. PITFALL: compared with os.path.samefile, not by
    string: this drive is mounted under two names, and a string compare would
    call the same directory two different checkouts.
    """
    problems = []
    spec = importlib.util.find_spec("cs2rl")
    origin = spec.origin if spec is not None else None
    if origin is None or not os.path.samefile(Path(origin).parent.parent, own_src):
        # `origin` is None for a namespace package (no __init__.py): name its dirs.
        if spec is None:
            where = "nowhere"
        elif origin is None:
            where = list(spec.submodule_search_locations or [])
        else:
            where = Path(origin).parent
        # `env VAR=value cmd` parses in bash and fish alike (the owner's shell is fish).
        problems.append(f"(a) cs2rl resolves to {where}, not to {own_src / 'cs2rl'}. Put this "
                        f"checkout's src/ first: env PYTHONPATH={own_src} <command>")
    for entry in path_entries:
        src = Path(entry or ".").resolve()
        if src.name != "src" or not src.is_dir() or not (src.parent / "pyproject.toml").is_file():
            continue
        stray = set()
        # A file counts only with an importable suffix (so vis_cache*.npy and
        # *.egg-info files do not); a directory counts by its name alone.
        for child in src.iterdir():
            suffix = "" if child.is_dir() else next(
                (s for s in _IMPORTABLE_SUFFIXES if child.name.endswith(s)), None)
            if suffix is None:
                continue
            name = child.name[:len(child.name) - len(suffix)]
            if name.isidentifier() and name not in ("cs2rl", "__pycache__"):
                stray.add(name)
        if stray:
            problems.append(f"(b) {src} is on sys.path and holds importable names other than "
                            f"cs2rl: {sorted(stray)}. Move them out, or take that src/ off "
                            "sys.path.")
        if "resources" in stray:
            # A resolving symlink is a directory, so it counts; a dangling one does not.
            problems.append(f"    `resources` is the symlink `import pufferlib` plants in its "
                            f"working directory (pufferlib/__init__.py runs os.symlink(<pufferlib>/"
                            f"resources, 'resources')), left by a process started with cwd = "
                            f"{src}. Removing it is safe.")
    return problems


# ── Leftover package directories under src/cs2rl (#205 part 2b) ──
#
# (e) The session stops if a directory under src/cs2rl/ meets all three:
#   - it is reachable through identifier-named directories only (so Python can
#     name it; zig-out/, .zig-cache/ and zig-pkg/ cannot be, and are skipped);
#   - it holds a `.py` file or a file with an importlib.machinery.EXTENSION_SUFFIXES
#     suffix (a built .so);
#   - it has no TRACKED __init__.py (`git ls-files`, i.e. the index).
# WHY: such a directory is a PEP 420 namespace package. After a package move (c_env
# -> env/c), a checkout that pulls the move keeps the untracked build outputs in the
# old directory, and `import cs2rl.c_env.binding` then SUCCEEDS on the stale .so
# (measured, #205 part 2b). Neither the checkout tripwire above (it looks only at
# the direct children of src/) nor import-linter (grimp sees no package without an
# __init__.py) nor any test noticed.
# The message names the directory: `git add` its __init__.py, or remove it. The
# session also stops while a new package's __init__.py is written but not staged,
# which is the point: an unstaged __init__.py is one `git commit -a` from missing.
# PITFALLS.
#   * __pycache__ is an identifier, so it is pruned by name. The suffixes are `.py`
#     plus EXTENSION_SUFFIXES, never importlib.machinery.all_suffixes(): that
#     includes `.pyc`, and on main it flagged 7 __pycache__ directories.
#   * `git ls-files` is the session's first git subprocess. If it fails (no git, not
#     a checkout, a broken index), the check fails CLOSED with a message naming the
#     failure; it never passes because it could not look.
# LIMITS.
#   1. A sourceless legacy `.pyc` directly in a leftover directory is importable,
#      and not flagged.
#   2. A snapshot at session start: a directory that appears later is not seen.
# tests/test_checkout_resolution.py pins the walk, the git failure and the wiring.
_LEFTOVER_SUFFIXES = (".py", *importlib.machinery.EXTENSION_SUFFIXES)


def leftover_package_dirs(package_root: Path, tracked: Collection[Path]) -> list[Path]:
    """Directories under `package_root` Python would import, as a namespace package, that git
    does not track as a package.

    `tracked` holds the tracked files, each as `package_root / <path git printed>`, so
    both sides are spelled from the same root (no resolve: the drive is mounted under
    two names). The walk never enters a directory whose name is not an identifier, or
    __pycache__, and never reports `package_root` itself.
    """
    leftovers = []
    for directory, subdirs, files in os.walk(package_root):
        subdirs[:] = sorted(d for d in subdirs if d.isidentifier() and d != "__pycache__")
        here = Path(directory)
        if here == package_root:
            continue
        importable = any(f.endswith(_LEFTOVER_SUFFIXES) for f in files)
        if importable and here / "__init__.py" not in tracked:
            leftovers.append(here)
    return leftovers


def leftover_package_problems(package_root: Path) -> list[str]:
    """(e): one problem per leftover directory under `package_root`, or one naming a git failure.

    Lists the tracked files with `git ls-files -z -- .` run IN `package_root`, so
    git prints paths relative to it.
    """
    try:
        listing = subprocess.run(["git", "ls-files", "-z", "--", "."],
                                 cwd=package_root,
                                 capture_output=True,
                                 text=True,
                                 timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return [
            f"(e) could not run `git ls-files` in {package_root} ({e!r}), so the check for "
            "leftover package directories cannot run. Fix git, or run from a checkout."
        ]
    if listing.returncode != 0:
        return [
            f"(e) `git ls-files` failed in {package_root} (rc {listing.returncode}: "
            f"{listing.stderr.strip()}), so the check for leftover package directories "
            "cannot run. Fix git, or run from a checkout."
        ]
    tracked = {package_root / p for p in listing.stdout.split("\0") if p}
    return [
        f"(e) {d} is importable as a namespace package (it holds a .py or extension file, and no "
        "tracked __init__.py), so a stale module there imports silently. `git add` its "
        "`__init__.py`, or remove the directory (a package move leaves its build outputs "
        "behind)." for d in leftover_package_dirs(package_root, tracked)
    ]


# ── Namespace guard: `tests` and `scripts` are this checkout's own (#207) ──
#
# Neither tests/ nor scripts/ has an __init__.py (tests/ since #207; scripts/
# never had one), so each is a PEP 420 namespace package. CPython builds a
# namespace package's __path__ from EVERY same-named directory without an
# __init__.py on sys.path, and recomputes it whenever sys.path changes or
# importlib.invalidate_caches() runs. (A same-named directory WITH an
# __init__.py, anywhere on sys.path, is a regular package: it wins outright over
# namespace portions for an import that has not run yet, replacing ours.)
# So a second `tests/` anywhere on sys.path (another checkout's root, e.g. main's
# while a worktree is tested) becomes part of this checkout's `tests`, and a
# `tests.X` that this checkout lacks loads SILENTLY from the other tree instead
# of raising ModuleNotFoundError. Neither guard beside this one sees it: the
# tripwire above looks only at `cs2rl` and src/ directories, and the
# one-module-object guard below only at files under REPO_ROOT. Two halves:
#   (c) at session start, reported with the tripwire: every sys.path entry other
#       than this checkout's root that holds a `tests` or `scripts` directory or
#       module stops the session, naming the entry and the name.
#   (d) at session end, in the pytest_runtestloop wrapper below: every
#       `tests`, `tests.*`, `scripts` or `scripts.*` module in sys.modules whose
#       __file__, or a __path__ entry, is not where its dotted name puts it in
#       ITS OWN package directory (`tests.a.b` is REPO_ROOT/tests/a/b: that
#       directory, its __init__ file, or a/b.<importable suffix>) fails the
#       session, naming each such module and path. It catches a tree that
#       reached sys.path after (c) ran, wherever that tree sits, inside tests/
#       or scripts/ included, as long as a module it served is still loaded at
#       the end (LIMIT 1).
# tests/test_namespace_guard.py pins both halves, with positive controls
# (including the nested layout) and a negative control.
#
# PITFALLS.
#   * Never compare against a directory prefix. Worktrees live in
#     <main>/.worktrees/, so another checkout's tests/ can sit UNDER this root:
#     "under REPO_ROOT" does not mean "ours", and neither does "under
#     REPO_ROOT/tests" (a tree at REPO_ROOT/tests/x/tests merges just the same).
#     (c) skips only the root itself, by os.path.samefile (the drive is mounted
#     under two names), and (d) compares each place with the exact location the
#     module's own name implies.
#   * `''` on sys.path is the cwd (`python -c`; `python -m pytest` puts the
#     absolute cwd there instead), so (c) takes every entry against the cwd.
#     Measured at #207, the only entry holding either name is the root itself
#     (pyproject.toml's `pythonpath = ["."]`, and sys.path[0] under `python -m
#     pytest` from the root): the stdlib, site-packages and the src/ entries hold
#     none. A future dependency that ships a top-level `tests` or `scripts` stops
#     the session at (c), which is right: its modules would merge into ours, or
#     (a regular package, with __init__.py) replace them.
#
# LIMITS.
#   1. (d) inspects a snapshot of sys.modules at session end. A foreign module
#      evicted before then leaves no trace, and `monkeypatch.syspath_prepend`
#      restores sys.path at teardown, so `tests.__path__` recalculates back as
#      well: measured, `syspath_prepend` + import + `del sys.modules[...]`
#      passes. So (d) does NOT catch every sys.path change made during the
#      session, only those whose modules are still loaded at its end.
#   2. (c) reads sys.path once, before collection, and never looks inside a zip
#      or other non-directory entry.
#   3. Imports made in a child process are invisible to both halves, as to the
#      one-module-object guard: a child is checked only if it loads this conftest.
#   4. (d) judges a module by its places alone. One with neither a string
#      __file__ nor a __path__ (a stub put in sys.modules by hand, or the empty
#      parent pytest's importlib mode inserts for a package it cannot import) is
#      never reported.
#   Under `-x` with a failure, or with collection errors, pytest stops before
#   (d) runs; that session fails anyway, and only the report is lost.
_NAMESPACE_PACKAGES = ("tests", "scripts")


def namespace_entry_problems(root: Path, path_entries: Iterable[str]) -> list[str]:
    """(c): each entry of `path_entries` (sys.path), other than `root`, that holds a namespace name.

    One problem per entry and name, naming both and the file or directory found.
    `''` and any relative entry are taken against the cwd, which the message then
    names. An entry that does not exist holds nothing and is skipped. A
    directory counts by its name alone: a namespace portion needs no
    __init__.py, and one WITH an __init__.py is a regular package that replaces
    ours outright. A file counts with an importable suffix.
    """
    problems = []
    for entry in path_entries:
        # '' is the cwd: os.path.abspath('') == os.getcwd().
        where = Path(os.path.abspath(entry))
        try:
            if os.path.samefile(where, root):
                continue
        except OSError:
            continue
        shown = repr(entry) if entry == str(where) else f"{entry!r} ({where})"
        for name in _NAMESPACE_PACKAGES:
            found = where / name
            if not found.is_dir():
                modules = (where / f"{name}{suffix}" for suffix in _IMPORTABLE_SUFFIXES)
                found = next((module for module in modules if module.is_file()), None)
            if found is not None:
                problems.append(f"(c) sys.path entry {shown} holds `{name}` ({found}). `{name}` is "
                                "a namespace package here, so that tree's modules would load as "
                                "this checkout's. Take the entry off sys.path (PYTHONPATH, or the "
                                "cwd under `python -m pytest`).")
    return problems


def namespace_modules_outside_their_package(modules: Mapping[str, object],
                                            root: Path) -> dict[str, list[str]]:
    """(d): map each `tests`/`scripts` module with a place other than its name's to those places.

    A module's places are its string `__file__` and its `__path__` entries. Its
    home is where its dotted name puts it under `root / <top-level name>`:
    `tests.a.b` is `<root>/tests/a/b` as a package directory, a `b/__init__` file
    or an `a/b` module file, each with any importable suffix (source, bytecode,
    extension). Every place that is not one of those is reported. Both sides are
    resolved, so a symlink or the drive's second mount name is no false alarm.

    PITFALL: never a prefix compare, against `root` OR against its own package
    directory: a checkout nested under the root (<main>/.worktrees/x) passes the
    first, and a tree nested inside tests/ itself (<root>/tests/x/tests) passes
    the second.
    """
    own = {name: (root / name).resolve() for name in _NAMESPACE_PACKAGES}
    init_names = {f"__init__{suffix}" for suffix in _IMPORTABLE_SUFFIXES}
    outside: dict[str, list[str]] = {}
    # A copy: sys.modules can change size mid-iteration when another thread imports.
    for name, module in list(modules.items()):
        top, _, rest = name.partition(".")
        package = own.get(top)
        if package is None:
            continue
        file = getattr(module, "__file__", None)
        places = [file] if isinstance(file, str) else []
        # A namespace package's __path__ is recomputed from sys.path as it is read.
        places += [p for p in getattr(module, "__path__", None) or () if isinstance(p, str)]
        # `tests.a.b` -> <root>/tests/a/b; `tests` itself when `rest` is "". Each is
        # resolved on its own: a symlinked package directory must not move the parent.
        implied = package.joinpath(*rest.split("."))
        home, parent = implied.resolve(), implied.parent.resolve()
        module_names = {f"{implied.name}{suffix}" for suffix in _IMPORTABLE_SUFFIXES}
        foreign = []
        for place in places:
            where = Path(place).resolve()
            if not (where == home or (where.parent == home and where.name in init_names) or
                    (where.parent == parent and where.name in module_names)):
                foreign.append(place)
        if foreign:
            outside[name] = foreign
    return outside


def pytest_sessionstart(session: pytest.Session) -> None:
    """Stop the session, before collection, if it would import another checkout's code, or a
    leftover directory's."""
    problems = checkout_resolution_problems(REPO_ROOT / "src", sys.path)
    problems += leftover_package_problems(REPO_ROOT / "src" / "cs2rl")
    problems += namespace_entry_problems(REPO_ROOT, sys.path)
    if problems:
        raise pytest.UsageError("checkout tripwire (tests/conftest.py):\n  " +
                                "\n  ".join(problems))


@pytest.fixture(scope="session")
def make_map():
    from cs2rl.env.map import make_simple_map
    return make_simple_map()


# Alias fixture used by test_map.py verticality tests (spec §3.10).
# Returns the same session-scoped instance as make_map.
@pytest.fixture(scope="session")
def simple_map(make_map):
    """Shared simple-map fixture (session-scoped, READ-ONLY).

    Mutating arrays on this fixture (adjacency, vis_matrix, centroids_z, is_ramp,
    ...) corrupts shared state across the entire test session because of session
    scope. If your test needs to mutate map data, build a fresh one with
    make_simple_map() inside the test instead of using this fixture.
    """
    return make_map


# ── The ProcessControl tripwire (gh#163; the kill seam's third safety layer) ──
#
# `execute_training_attempt(process=None)` resolves None to
# `ProcessControl.system()`, the REAL spawn/getpgid/killpg/signal functions.
# Production never passes `process`, and the client tests' wrappers forward
# production's keywords, so a test that forgets its own control would get the
# real ones without ever naming `system`. On 2026-09-23 a real
# `killpg(getpgid(1), SIGTERM)` (= `kill(-1, SIGTERM)`) ended the user's
# desktop session. Under pytest this fixture makes `system()` return a control
# whose every field raises, so that mistake fails loudly instead of spawning or
# signalling. `test_process_control_tripwire_poisons_system` and
# `test_process_control_tripwire_guards_the_resolution_path`
# (tests/test_modal_training.py) pin it.
#
# It covers every OTHER test only because it is `autouse`: dropped, or moved
# into a narrower conftest or a test file, the fixture still reaches a test
# that names it, and nothing else. So `autouse=True` is pinned twice: the
# resolution-path control does NOT name the fixture, and its first statement
# asserts the fixture is active in it anyway; and clause (viii) of
# test_kill_seam_static_safety requires, by AST, exactly one definition, here,
# decorated exactly `@pytest.fixture(autouse=True)` (no wider scope), that
# builds the poison itself.
#
# PITFALLS.
#   * The module is LOOKED UP in sys.modules, never imported. That keeps a
#     session that never loads the runner free of it. Importing it here would
#     close RESIDUAL 1 below, at the cost of loading the runner into every
#     pytest session.
#   * The patch goes through the fixture's OWN `pytest.MonkeyPatch.context()`,
#     never the test's `monkeypatch`: a test body that calls
#     `monkeypatch.undo()` would otherwise restore the real `system` for the rest
#     of that test (tests/test_no_restated_env_defaults.py calls it).
#   * RESIDUAL: three windows are not poisoned, because the fixture is
#     function-scoped and needs the module already loaded.
#       1. A test whose own body is the first thing in the process to import the
#          package: the fixture found nothing to patch. Every test file that
#          drives the attempt imports the package at module scope, so collection
#          has loaded it before any fixture runs. Not every modal test file
#          does: three tests/test_modal_*.py files import it only inside test
#          bodies or not at all, and none of those reaches the attempt.
#       2. Code that runs at collection: module level, parametrize arguments.
#       3. Module-, class- and session-scoped fixtures, setup and teardown.
#     The only cover for all three is static, and partial: clause (iii) of
#     test_kill_seam_static_safety bans any `.system` read (on any receiver)
#     under tests/ outside the two tripwire tests. It does not see an attempt
#     driven with `process` forgotten from one of these windows; today no modal
#     test file has a higher-scoped fixture or drives the attempt at collection.
#   * The poison raises a RuntimeError subclass on purpose. The attempt swallows
#     a ValueError from the handler install and a ProcessLookupError from
#     getpgid/killpg, so a poison of either type would be silent exactly there.
#   * Build the poisoned control with all four fields as keywords: the static
#     safety test rejects a `*`/`**` splat in any ProcessControl(...) under tests/.

# The runner module both fixtures below look up. The tripwire's two tests fail
# if this name stops resolving (the tripwire then patches nothing), so the
# precondition fixture, which reads the same name, cannot go quietly blind.
_TRAINING_MODULE = "scripts.modal_runner.training"


class ProcessControlTripwire(RuntimeError):
    """Raised by every field of the poisoned ProcessControl the tripwire installs."""


def _process_control_poison(field):
    """A stand-in for ProcessControl.<field>: raises ProcessControlTripwire on any call."""

    def poisoned(*_args, **_kwargs):
        raise ProcessControlTripwire(
            f"ProcessControl.{field} was called under pytest: execute_training_attempt was "
            "called without process=... (or something else reached ProcessControl.system()). "
            "Pass a ProcessControl whose spawn, getpgid and killpg are fakes; the training test "
            "builder does.")

    return poisoned


@pytest.fixture(autouse=True)
def _process_control_tripwire():
    """Make `ProcessControl.system()` return a poisoned control; yield that control.

    Yields None, and patches nothing, when the training module is not loaded.
    The patch is undone when the test ends, by the fixture's own MonkeyPatch,
    which nothing the test does to its `monkeypatch` reaches. A knock-out that
    deletes the one `setattr` line (to see the two tripwire tests go red) may be
    run only on those two test nodes, because it switches the backstop off for
    the whole session.
    """
    training = sys.modules.get(_TRAINING_MODULE)
    if training is None:
        yield None
        return
    poisoned = training.ProcessControl(spawn=_process_control_poison("spawn"),
                                       getpgid=_process_control_poison("getpgid"),
                                       killpg=_process_control_poison("killpg"),
                                       install_signal=_process_control_poison("install_signal"))
    with pytest.MonkeyPatch.context() as tripwire_patch:
        tripwire_patch.setattr(training.ProcessControl, "system", staticmethod(lambda: poisoned))
        yield poisoned


@pytest.fixture
def process_control_tripwire_error():
    """The tripwire's exception type, so a test names it without importing conftest."""
    return ProcessControlTripwire


# ── The kill-path tests' own-group precondition ──
#
# The process-group guard in `_signal_process_group` refuses to signal the
# runner's OWN process group. Every modal test that hands the attempt a
# recording killpg -- the training test builder's default, `_signal_hooks`
# (tests/test_modal_training.py), the client tests' execute wrappers, the
# binding campaign -- pairs it with an identity getpgid and a FakeChild whose
# pid is FakeChild's default. In a session whose process group is that pid,
# the guard would refuse every one of their kills: a test asserting a kill
# would fail for a reason unrelated to its subject, and a test asserting that
# nothing was killed (`kills == []`) would pass without testing anything. So
# the session stops here, at the cause, before any test runs. The
# process-group guard test builds its own pids, apart from the session's group
# by construction, and `_signal_hooks` repeats the check for its own child.
#
# PITFALLS.
#   * FakeChild's default pid is stated here AND read from FakeChild whenever
#     tests/modal_test_helpers.py is loaded: a changed default fails the
#     comparison instead of leaving this check stale. The helpers are LOOKED
#     UP in sys.modules, like the runner above: importing them would import the
#     runner.
#   * It runs once per session, at the first test, and checks only if the
#     runner is loaded by then, keyed on `_TRAINING_MODULE` like the tripwire.
#     Collection has finished by then, and every test file that drives the
#     attempt imports the runner at module scope (the tripwire's RESIDUAL 1), so
#     a session that can reach a recording killpg is always checked.
_FAKE_CHILD_PID = 4242


@pytest.fixture(scope="session", autouse=True)
def _kill_path_own_group_precondition():
    """Fail every test of a session whose process group is FakeChild's default pid."""
    if _TRAINING_MODULE not in sys.modules:
        return
    helpers = sys.modules.get("tests.modal_test_helpers")
    if helpers is not None:
        default = helpers.FakeChild.__init__.__kwdefaults__["pid"]
        assert default == _FAKE_CHILD_PID, (
            f"FakeChild's default pid is {default}, but tests/conftest.py checks the session's "
            f"process group against {_FAKE_CHILD_PID}: update `_FAKE_CHILD_PID`")
    assert os.getpgrp() != _FAKE_CHILD_PID, (
        f"this session's process group is {os.getpgrp()}, FakeChild's default pid, so the "
        "process-group guard in `_signal_process_group` refuses every kill the modal kill-path "
        "tests record (their `kills == []` assertions would pass without testing anything). "
        "Run pytest from another process group, e.g. from a new shell.")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "performance: performance-sensitive tests excluded from default pytest runs",
    )
    # Unregistered markers are an error under --strict-markers.
    config.addinivalue_line(
        "markers",
        "slow: over ~15 s of wall. Deselecting it DROPS coverage (the patch-binding campaign, "
        "the seed positive control, the fast-math builds), so `-m 'not slow'` is only for "
        "intermediate per-commit checks and never goes in addopts. A test over ~15 s that "
        "stays unmarked carries an `always-on: <why>` comment.",
    )

    # The default pytest tmp_path lives under /tmp on the system root
    # partition, which on small devices is regularly tight. So unless the user
    # passed --basetemp or set PYTEST_DEBUG_TEMPROOT, the temp ROOT moves to
    # $HOME/.cache/cs2rl-pytest:
    # $HOME is the user's primary partition, with persistent free space. Keep
    # it on $HOME, not in the repo: on the main dev machine the repo drive is
    # fuseblk/NTFS, where chmod is a no-op, and chmod-based tests (e.g.
    # tests/test_pyrefly_gate.py) break there.
    #
    # Move the ROOT, never basetemp itself. Setting config.option.basetemp puts
    # pytest on its explicit-basetemp path, which rm_rf's that exact directory
    # the first time the session asks for a temp dir (in
    # TempPathFactory.getbasetemp, reached by the first tmp_path, tmpdir or
    # tmp_path_factory use), so two concurrent sessions deleted each other's
    # tmp_path trees mid-run (gh#219). With PYTEST_DEBUG_TEMPROOT pytest keeps
    # its default layout under the new root instead: each session gets its own
    # numbered <root>/pytest-of-<user>/pytest-<N>/, lock-protected while the
    # session runs, and only the oldest beyond the newest 3 are rotated away.
    # pytest reads the variable lazily, in getbasetemp, so setting it here is
    # early enough. Nested pytest sessions that tests spawn without --basetemp
    # inherit the variable through the environment, so they too get their own
    # numbered dir under the same root instead of wiping the outer session's.
    #
    # Why not ~/.pytest_tmp, the old pinned basetemp: conftests from before
    # gh#219, still present on other branches and checkouts, rm_rf that exact
    # directory whenever they run without --basetemp and a test asks for a
    # temp dir, and rm_rf ignores pytest's locks, so live sessions under a
    # root there would be deleted.
    # ~/.cache is the XDG default cache directory, and no old conftest touches
    # cs2rl-pytest. Nothing is created when PYTEST_DEBUG_TEMPROOT is already
    # set, so an explicitly redirected run leaves $HOME alone.
    # tests/test_pytest_tmp_isolation.py pins the per-session basetemp, the
    # default root and the explicit override.
    if not config.option.basetemp and "PYTEST_DEBUG_TEMPROOT" not in os.environ:
        temproot = Path(os.path.expanduser("~/.cache/cs2rl-pytest"))
        temproot.mkdir(parents=True, exist_ok=True)
        os.environ["PYTEST_DEBUG_TEMPROOT"] = str(temproot)


def pytest_collection_modifyitems(config, items):
    explicit_targets = {Path(str(arg)).as_posix() for arg in config.invocation_params.args}
    run_performance = any(
        target.endswith("tests/smoke_test.py") or target.endswith("smoke_test.py")
        for target in explicit_targets)
    if run_performance:
        return

    skip_performance = pytest.mark.skip(
        reason="performance test; run explicitly with `uv run pytest tests/smoke_test.py -q -s`")
    for item in items:
        if "performance" in item.keywords:
            item.add_marker(skip_performance)


# ── Running the main session on pytest-xdist workers (#285) ──
#
# `pytest -n 2 --dist loadgroup tests ...` is the main session's command (CONTRIBUTING.md).
# `-n`, `--dist` and `-m` stay OUT of addopts: a later `-n 0` or `-m ""` would have to undo
# them (`-p no:xdist` would fail on the `-n`), and `-m 'not slow'` in addopts would
# silently drop the slow tests from every full run.
# The guards' side of this is the block after the "One module object per file" one below.


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config) -> int:
    """`-n auto` means 2 workers here, or $PYTEST_XDIST_AUTO_NUM_WORKERS when that is set.

    WHY: the default is one worker per core, 12 on the dev box. Two workers already take
    2.9 GB more than a serial session on a 15 GB machine that also runs the owner's work
    (#285), so a mistyped `-n auto` must not start twelve.
    LIMIT: xdist asks for this before it loads any conftest but the initial ones, so the
    hook applies only when tests/conftest.py is one: the args name `tests` or a path under it.
    """
    return int(os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS", 2))


def pytest_collection_finish(session: pytest.Session) -> None:
    """In an xdist worker, give torch an equal share of the cores, if a test module loaded it.

    torch defaults to one intra-op thread per core, so N workers oversubscribe the machine
    N-fold. At `-n 2` the cap equals torch's own default on the 12-thread dev box (6), so it
    only acts from `-n 3` on; it exists for `-n 4`. In-process
    only: child interpreters keep their own default (train.py pins OMP_NUM_THREADS=1).
    PITFALL 1: detect a worker by `config.workerinput`, never by `PYTEST_XDIST_WORKER`: xdist
    exports that variable into os.environ, so every child pytest a test spawns would look like
    a worker (measured: the patch-binding campaign went from 32 s to 138 s).
    PITFALL 2: never import torch here. The `-n 2` guard tests start nested xdist sessions
    whose workers also have `workerinput`, and an eager import would cost each of them the
    import for a plant that never touches torch. Only a torch that is already loaded is capped.
    """
    workerinput = getattr(session.config, "workerinput", None)
    torch = sys.modules.get("torch")
    if workerinput is not None and torch is not None:
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // int(workerinput["workercount"])))


def assert_child_had_xdist(child: subprocess.CompletedProcess) -> None:
    """Fail a child-session row with the remedy when the child's pytest lacks pytest-xdist.

    The `-n 2` guard rows run their child unconditionally, so a missing xdist is a red row
    with a fix, never a skip that someone can leave in place (the #288 lesson). The dev group
    provides xdist; a checkout whose shared `.venv` predates that needs a sync in MAIN.
    """
    assert "unrecognized arguments: -n" not in child.stderr, (
        "the child's pytest does not know `-n`: pytest-xdist is not installed in this "
        "environment. Run `uv sync --all-groups --inexact` in the MAIN checkout (never in a "
        f"worktree), then re-run.\n{child.stderr[-1000:]}")


# ── One module object per file ──
#
# A file imported under two names (`cs2rl.spec.paths` and `src.cs2rl.spec.paths`) is two module
# objects with two copies of the module's state, so a monkeypatch or an
# `except` aimed at one misses the other. After the last test (or after
# collection, under --collect-only) every session checks sys.modules for a repo
# file held under two names and fails if it finds one, the way pytest-cov fails
# a session on coverage: `session.testsfailed += 1` in a `pytest_runtestloop`
# wrapper. The static half is ruff's TID251 banned-api table in pyproject.toml.
# tests/test_one_module_object_per_file.py pins both the function and the hook.
#
# LIMITS.
#   (a) ruff cannot see a literal `importlib.import_module("...")` or
#       `__import__("...")`, and this guard sees one only when it executes in a
#       session that also loads the other spelling.
#   (b) the ban is enforced by the pre-commit hook, at commit time and on staged
#       files only, so `--no-verify`, merges, rebases and cherry-picks skip it.
#   (c) this is a snapshot of sys.modules at session end, so a second copy
#       is invisible if it is evicted before then, registered only while it
#       runs (an in-process `runpy.run_path`), or never registered at all
#       (`importlib.util.module_from_spec` + `exec_module`). `fake_modal` in
#       tests/test_modal_client.py pops `scripts.run_modal`,
#       `scripts.modal_artifacts` and `scripts.modal_backfill_sidecar`, so for
#       those three the ruff ban on their bare names is the only check.
#   (d) imports made in a child process are invisible to both: ruff reads a
#       code string as a string, and this guard reads only its own process's
#       sys.modules. A code string that imports two spellings rebuilds the trap
#       in the child with nothing objecting.
#   Under `-x` with a failure, or with collection errors, pytest stops before
#   the check runs; that session fails anyway, and only the report is lost.
def files_under_two_module_names(modules: Mapping[str, object], root: Path,
                                 ignored_prefixes: Iterable[Path]) -> dict[Path, list[str]]:
    """Map every file under `root` that `modules` holds under two or more names to those names.

    Files are compared by resolved real path, so a symlink and its target are one
    file. A module without a string `__file__` (builtins, namespace packages) is
    skipped, and so is every file under one of `ignored_prefixes`. There are no
    name-based exclusions.

    PITFALL: the session passes the interpreter's `sys.prefix` and
    `sys.base_prefix` as `ignored_prefixes`, and both are needed. The stdlib and
    the venv hold legitimate aliases (`os.path` is `posixpath`), and
    `multiprocessing` registers `__main__` a second time as `__mp_main__`: under
    `.venv/bin/pytest` (what `uv run pytest` runs) that file is
    `<repo>/.venv/bin/pytest`, under the repo root and outside any
    site-packages directory, so only the prefix rule excludes it.
    """
    return _files_under_two_names(_named_files(modules), root, ignored_prefixes)


def _named_files(modules: Mapping[str, object]) -> list[tuple[str, str]]:
    """(name, `__file__`) of every module in `modules` whose `__file__` is a string."""
    # A copy: sys.modules can change size mid-iteration when another thread imports.
    return [(name, file) for name, module in list(modules.items())
            if isinstance(file := getattr(module, "__file__", None), str)]


def _files_under_two_names(pairs: Iterable[tuple[str, str]], root: Path,
                           ignored_prefixes: Iterable[Path]) -> dict[Path, list[str]]:
    """The duplicate rule of `files_under_two_module_names`, on (name, `__file__`) pairs.

    Pairs, not a module map, so the xdist controller can apply it to the UNION of every
    process's pairs: the same (name, file) pair coming from two workers counts once.
    """
    root = root.resolve()
    ignored = [prefix.resolve() for prefix in ignored_prefixes]
    names: dict[Path, set[str]] = {}
    for name, file in pairs:
        path = Path(file).resolve()
        if path.is_relative_to(root) and not any(path.is_relative_to(p) for p in ignored):
            names.setdefault(path, set()).add(name)
    return {path: sorted(found) for path, found in names.items() if len(found) > 1}


# ── The session guards under pytest-xdist (#285) ──
#
# LIMIT xdist puts on the two checks below: the test modules are imported in the
# WORKERS, so the controller's own sys.modules holds none of them, and a worker's
# `session.testsfailed` and terminal output never reach the controller (xdist's
# DSession.worker_workerfinished reads only exitstatus 2, shouldfail and shouldstop
# from a worker). Left alone, both guards pass every `-n` session. So each worker
# runs no verdict, only collects the facts and ships them in `config.workeroutput`
# (xdist sends that dict with its `workerfinished` event); the controller collects
# them per node in `pytest_testnodedown` and judges the UNION with its own facts, so
# a file imported as `a` in one worker and `b` in another is still caught, as it is
# serially. This is pytest-cov's own pattern for its per-worker data.
#
# Fail closed: a node that went down with an error, or finished without the key, is
# itself a red finding, since its modules were never checked.
# tests/test_one_module_object_per_file.py and tests/test_namespace_guard.py pin this
# with `-n 2` child sessions; their other child sessions are serial and cannot see it.
_WORKER_GUARD_KEY = "cs2rl_session_guards"
_worker_findings = pytest.StashKey[list]()


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error) -> None:
    """Controller side: keep what each xdist worker's guard found (or that it could not look)."""
    found = node.config.stash.setdefault(_worker_findings, [])
    workerid = node.gateway.id
    output = getattr(node, "workeroutput", None) or {}
    if error is not None or _WORKER_GUARD_KEY not in output:
        how = "finished" if error is None else f"went down ({error!r})"
        found.append((workerid, None, f"worker {workerid} {how} without reporting "
                      "the session guards' findings, so its modules were never checked"))
        return
    found.append((workerid, output[_WORKER_GUARD_KEY], None))


@pytest.hookimpl(wrapper=True)
def pytest_runtestloop(session: pytest.Session) -> Generator[None, object, object]:
    """Fail the session, with a red report, if a repo file is loaded under two module names,
    or a `tests`/`scripts` module came from another tree (the namespace guard's (d)).

    Serially the process judges its own sys.modules. Under xdist a worker only hands its
    facts to the controller (see the block comment above) and the controller judges.
    """
    result = yield
    named_files = _named_files(sys.modules)
    outside = namespace_modules_outside_their_package(sys.modules, REPO_ROOT)
    workeroutput = getattr(session.config, "workeroutput", None)
    if workeroutput is not None:
        workeroutput[_WORKER_GUARD_KEY] = {"named_files": named_files, "outside": outside}
        return result
    lost = []
    for _workerid, findings, problem in session.config.stash.get(_worker_findings, []):
        if problem is not None:
            lost.append(problem)
            continue
        named_files += [(name, file) for name, file in findings["named_files"]]
        for name, places in findings["outside"].items():
            outside[name] = sorted(set(outside.get(name, [])) | set(places))
    duplicates = _files_under_two_names(named_files, REPO_ROOT,
                                        (Path(sys.prefix), Path(sys.base_prefix)))
    if lost:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_sep("=",
                               "session guards could not check an xdist worker",
                               red=True,
                               bold=True)
            for line in sorted(lost):
                reporter.line(line, red=True)
        session.testsfailed += 1
    if duplicates:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_sep("=", "repo files loaded under two module names", red=True, bold=True)
            for path, names in sorted(duplicates.items()):
                reporter.line(f"{path}: {names}", red=True)
            reporter.line(
                "Import each module under one name: `tests.X` for a module under tests/ (the "
                "name pytest collects it under); pyproject.toml's banned-api table names the "
                "spelling for src/, the libraries #204 moved out of scripts/ and deploy/, and "
                "the Modal scripts.",
                red=True)
        session.testsfailed += 1
    if outside:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_sep("=",
                               "tests/scripts modules loaded from another tree",
                               red=True,
                               bold=True)
            for name, places in sorted(outside.items()):
                reporter.line(f"{name}: {places}", red=True)
            reporter.line(
                "Each `tests`/`scripts` module must load from where its dotted name puts it in "
                "this checkout. Neither package has an __init__.py, so a directory of either name "
                "elsewhere on sys.path merges into it, or, holding an __init__.py, replaces it. "
                "Take the other tree off sys.path; see the namespace guard in tests/conftest.py.",
                red=True)
        session.testsfailed += 1
    return result
