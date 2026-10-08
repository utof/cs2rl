"""Tests for the pyrefly git hooks (.githooks/pre-commit, .githooks/pre-merge-commit).

**Every test here asserts on the GATE's own output** -- a named error entry, a
named path, an ADDED/REMOVED count -- and never on the hook's marker line or on a
nonzero exit alone.

That rule exists because of a measured failure. An earlier version of these tests
asserted "the pyrefly step ran and the commit was rejected", where "ran" meant the
marker text appeared. Both assertions were satisfied by a hook whose gate never
executed at all: the marker is an `echo` on the line BEFORE the gate call, so a
hook that dies for any unrelated reason prints it and rejects the commit. That
moves the blind spot one line later instead of closing it.

Each rejection test is paired with a clean-commit negative control, for the same
reason: a hook that rejects everything passes a rejection test.

The exceptions are the #220 tests at the end: two read the hooks' source for
their uv spelling, where there is no gate output to assert on, and one asserts
the `uv sync` NOTE a dependency commit prints, alongside the gate's own line.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from tests.conftest import REPO_ROOT

REPO = REPO_ROOT
REAL_PYTHON = REPO / ".venv" / "bin" / "python"
REAL_PYREFLY = REPO / ".venv" / "bin" / "pyrefly"
HOOK = REPO / ".githooks" / "pre-commit"
MERGE_HOOK = REPO / ".githooks" / "pre-merge-commit"

CONFIG = 'preset = "default"\nproject-includes = ["src"]\nsearch-path = ["src", "."]\n'

# The only uv spelling the hooks may use (#220), and the prefix install() rewrites.
NO_SYNC = "uv run --no-sync "
# `uv` as a word, quoted (`"uv" run`) or not. `(?!\.)` keeps the `uv.lock` trigger
# pattern out; `\b` keeps `uvx` (which never touches the project env) and UV_PYTHON
# out. BLIND SPOT: a call through a variable (`"$UV" run`) has no `uv` word on its
# line. It is caught only by an assignment that spells one (`UV=uv` is flagged); a
# variable taken from the environment is invisible to every line matcher here, and
# the exact count in test_every_hook_uv_call_is_no_sync notices it only when it
# replaces a counted call.
UV_CALL = re.compile(r"\buv\b(?!\.)")
# A uv word NOT followed by the NO_SYNC spelling. Per occurrence, not per line, so
# a bare call cannot hide behind a no-sync one on the same line: chained after it
# (`... && uv run yapf`) or behind a trailing `# was: uv run --no-sync ...`.
BARE_UV = re.compile(r"\buv\b(?!\.)(?! run --no-sync )")
# An echo whose one argument is a double-quoted literal with no `$`, backtick or
# backslash runs nothing but echo, so a uv it names is a message, not a call.
PURE_ECHO = re.compile(r'^\s*echo\s+"[^"$`\\]*"\s*$')


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def uv_call_lines(hook_text: str) -> list[str]:
    """The hook's lines that can run uv: every line naming it, less comments and pure echoes.

    Comment lines are skipped: the hooks' prose names `uv run` freely, and a
    commented-out command is inert until someone uncomments it -- then it counts.
    A PURE_ECHO line prints uv and cannot run it (the pre-commit `uv sync` NOTE).
    """
    return [
        line for line in hook_text.splitlines()
        if not line.lstrip().startswith("#") and not PURE_ECHO.match(line) and UV_CALL.search(line)
    ]


def bare_uv_lines(hook_text: str) -> list[str]:
    """uv_call_lines that hold at least one uv call not spelled NO_SYNC."""
    return [line for line in uv_call_lines(hook_text) if BARE_UV.search(line)]


def install(root: Path, src: Path, name: str) -> None:
    """Copy a hook into the throwaway, rewriting it to use the real repo's tools.

    Two rewrites, and the second one is the part that is easy to miss:

      1. `uv run --no-sync ` -> `<REPO>/.venv/bin/`, so no `uv` invocation ever
         happens from /tmp. uv would try to build the throwaway as a project and
         download gigabytes. This also removes any need for UV_OFFLINE/UV_NO_SYNC.
         Asserted, not assumed: a spelling the replace does not match (a bare
         `uv run`, say) would otherwise run real uv from /tmp and the tests below
         would fail for a reason that has nothing to do with the gate.
      2. the RELATIVE `scripts/pyrefly_gate.py` -> an absolute path into the real
         repo, plus `--project .`. The hook cd's to the throwaway first, so
         rewriting only the interpreter leaves it looking for a gate script that
         does not exist there -- which produces a rejected commit with the marker
         printed and the gate never run, i.e. exactly the false pass these tests
         are written to prevent.
    """
    tools = f"{REPO}/.venv/bin/"
    text = src.read_text().replace(NO_SYNC, tools)
    # The tool prefix is blanked before looking: REPO is this checkout's path, and a
    # worktree directory named, say, `uv-fix` would read as a uv call.
    leftover = uv_call_lines(text.replace(tools, ""))
    assert not leftover, f"{src.name}: uv calls the {NO_SYNC!r} rewrite missed: {leftover}"
    text = text.replace("python scripts/pyrefly_gate.py",
                        f"python {REPO}/scripts/pyrefly_gate.py --project .")
    dest = root / ".git" / "hooks" / name
    dest.write_text(text)
    dest.chmod(0o755)


def make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "r"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "t")

    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    (root / "pyrefly.toml").write_text(CONFIG)

    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(REAL_PYTHON)
    (venv_bin / "pyrefly").symlink_to(REAL_PYREFLY)

    git(root, "add", "pyrefly.toml", *files.keys())
    res = subprocess.run([
        str(REAL_PYTHON),
        str(REPO / "scripts" / "pyrefly_gate.py"), "--project",
        str(root), "--update"
    ],
                         capture_output=True,
                         text=True,
                         cwd=REPO)
    assert res.returncode == 0, res.stderr
    git(root, "add", "pyrefly-snapshot.json")
    git(root, "commit", "-q", "-m", "seed")

    install(root, HOOK, "pre-commit")
    install(root, MERGE_HOOK, "pre-merge-commit")
    return root


def commit(root: Path, msg: str) -> subprocess.CompletedProcess:
    return git(root, "commit", "-m", msg)


def head_count(root: Path) -> int:
    return len(git(root, "log", "--oneline").stdout.splitlines())


# --------------------------------------------------------------------------


def test_a_clean_commit_passes_the_hook(tmp_path):
    """The negative control every rejection test below depends on.

    A hook that rejects everything would pass all of them.
    """
    root = make_repo(tmp_path, {"src/m.py": "def f() -> int:\n    return 1\n"})
    before = head_count(root)

    (root / "src" / "ok.py").write_text("def g() -> int:\n    return 2\n")
    git(root, "add", "src/ok.py")
    res = commit(root, "clean")

    out = res.stdout + res.stderr
    assert head_count(root) == before + 1, f"clean commit was rejected:\n{out}"
    assert "materialised" in out, f"the gate did not actually run:\n{out}"


def test_the_hook_rejects_a_commit_carrying_a_new_type_error(tmp_path):
    """A staged type error is rejected, and the GATE is what rejects it."""
    root = make_repo(tmp_path, {"src/m.py": "def f() -> int:\n    return 1\n"})
    before = head_count(root)

    (root / "src" / "m.py").write_text('def f() -> int:\n    return "broken"\n')
    git(root, "add", "src/m.py")
    res = commit(root, "broken")

    out = res.stdout + res.stderr
    assert head_count(root) == before, "the commit was created anyway"
    # Gate-specific, not the marker: the diff must name the error.
    assert "ADDED 1" in out, out
    assert "src/m.py" in out
    assert "bad-return" in out


def test_a_deletion_only_commit_is_gated(tmp_path):
    """`git rm` of a module its dependents import must be caught.

    The bug this guards: staged_files is ACMR-filtered, so a pure deletion leaves
    it EMPTY. A gate appended to the Python-files branch never runs, and the
    early exit keyed off the same filtered list exits 0 first. The gate reads the
    unfiltered list for exactly this commit shape.
    """
    root = make_repo(tmp_path, {
        "src/m.py": "def f() -> int:\n    return 1\n",
        "src/user.py": "import m\n\nx: int = m.f()\n",
    })
    before = head_count(root)

    git(root, "rm", "-q", "src/m.py")
    assert git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR").stdout.strip() == "", \
        "fixture is not deletion-only; the test would not exercise the bug"
    res = commit(root, "delete m")

    out = res.stdout + res.stderr
    assert head_count(root) == before, "a deletion-only commit slipped through"
    assert "ADDED" in out, out
    assert "src/user.py" in out, "the gate did not name the dependent"


def test_a_config_narrowing_commit_is_gated(tmp_path):
    """A commit whose only change is pyrefly.toml still invokes the gate.

    This is the commit that DISABLES the gate, and a `*.py`-only trigger never
    runs it. Exact-equality polarity does not help -- it only decides what
    happens if the gate runs at all.
    """
    root = make_repo(
        tmp_path, {
            "src/m.py": 'def f() -> int:\n    return "x"\n',
            "src/other.py": 'def g() -> int:\n    return "y"\n',
        })
    before = head_count(root)

    (root / "pyrefly.toml").write_text(
        'preset = "default"\nproject-includes = ["nothing"]\nsearch-path = ["src", "."]\n')
    git(root, "add", "pyrefly.toml")
    res = commit(root, "narrow the config")

    out = res.stdout + res.stderr
    assert head_count(root) == before, "a config narrowing was committed"
    assert "REMOVED" in out or "matched no files" in out, out


def test_a_commit_touching_no_trigger_path_skips_the_gate(tmp_path):
    """The trigger is a filter, not a formality: unrelated commits stay fast.

    Without this, "the gate runs" is untested in the negative direction and the
    trigger list could be `*` without any test objecting.
    """
    root = make_repo(tmp_path, {"src/m.py": "def f() -> int:\n    return 1\n"})
    before = head_count(root)

    (root / "notes.txt").write_text("hello\n")
    git(root, "add", "notes.txt")
    res = commit(root, "docs")

    out = res.stdout + res.stderr
    assert head_count(root) == before + 1, f"unrelated commit rejected:\n{out}"
    assert "pyrefly type check" not in out, f"the gate ran for a .txt-only commit:\n{out}"


def test_pre_merge_commit_gates_a_clean_merge(tmp_path):
    """A clean `git merge --no-ff` runs pre-merge-commit, not pre-commit.

    Measured: a clean merge never invokes pre-commit, so before this hook existed
    every merge in this repo's workflow was an ungated commit -- including the
    merge that would land the gate itself.
    """
    root = make_repo(tmp_path, {"src/m.py": "def f() -> int:\n    return 1\n"})
    before = head_count(root)

    git(root, "checkout", "-q", "-b", "feat")
    (root / "src" / "bad.py").write_text('def h() -> int:\n    return "nope"\n')
    git(root, "add", "src/bad.py")
    git(root, "commit", "-q", "--no-verify", "-m", "sneak it in past pre-commit")
    git(root, "checkout", "-q", "-")

    res = git(root, "merge", "--no-ff", "-m", "merge feat", "feat")
    out = res.stdout + res.stderr

    assert "[pre-merge-commit]" in out, f"pre-merge-commit did not fire:\n{out}"
    assert "ADDED 1" in out, out
    assert "src/bad.py" in out
    assert head_count(root) == before, "the merge commit was created despite the error"
    assert (root / ".git" / "MERGE_HEAD").exists(), \
        "a failed pre-merge-commit should leave the merge incomplete"


SYNC_NOTE = "run 'uv sync' in the main checkout first"


def test_a_dependency_commit_prints_the_sync_note(tmp_path):
    """Staging pyproject.toml or uv.lock prints the `uv sync` NOTE; a .py commit does not.

    #220: the hooks run uv with --no-sync, so after a dependency change the gate
    checks the .venv as it is, and this NOTE is the lock-bumper's only reminder.
    Both names are staged, one per commit, so dropping either from the hook's
    pattern is red. The NOTE must not block: every commit lands, and the gate
    still runs for each dependency commit.
    """
    root = make_repo(tmp_path, {"src/m.py": "def f() -> int:\n    return 1\n"})
    before = head_count(root)

    commits = [
        ("src/ok.py", "def g() -> int:\n    return 2\n"),
        ("uv.lock", "version = 1\n"),
        ("pyproject.toml", '[project]\nname = "t"\nversion = "0"\n'),
    ]
    outs = {}
    for rel, text in commits:
        (root / rel).write_text(text)
        git(root, "add", rel)
        res = commit(root, f"stage {rel}")
        outs[rel] = res.stdout + res.stderr

    assert head_count(root) == before + 3, f"a commit was rejected: {outs}"
    assert SYNC_NOTE not in outs["src/ok.py"], outs["src/ok.py"]
    for rel in ("uv.lock", "pyproject.toml"):
        assert SYNC_NOTE in outs[rel], f"no NOTE for {rel}:\n{outs[rel]}"
        assert "materialised" in outs[rel], f"the gate did not run for {rel}:\n{outs[rel]}"


def test_every_hook_uv_call_is_no_sync():
    """Every uv call the hooks execute is `uv run --no-sync`, and there are
    exactly as many as the hooks are known to make.

    Why --no-sync: a git worktree here borrows main's .venv, and a syncing
    `uv run` from it re-points the shared editable install at the worktree and
    can rebuild main's binding .so in place (#220). The hooks run on every
    commit, so one bare `uv run` turns every worktree commit into that trigger.

    Why the exact count: without it this test passes vacuously if uv_call_lines
    stops matching anything -- and install()'s leftover check, which uses the
    same matcher, would go blind with it. The count is that matcher's positive
    control, and it also makes a new or dropped uv call a deliberate edit here.

    BLIND SPOT: a call through a variable (`"$UV" run ...`) names no `uv` word on
    its line. `UV=uv` in the hook is itself flagged, but a variable the hook takes
    from the environment is not: a NEW call through it passes this test and
    install()'s check, and one that REPLACES a counted call fails only the count.
    """
    expected = {HOOK: 6, MERGE_HOOK: 1}
    for hook, n in expected.items():
        text = hook.read_text()
        bare = bare_uv_lines(text)
        assert not bare, f"{hook.name}: uv call not spelled {NO_SYNC!r}: {bare}"
        calls = uv_call_lines(text)
        assert len(calls) == n, (
            f"{hook.name}: expected {n} uv calls, found {len(calls)}: {calls}. If intended, "
            "update `expected` in test_every_hook_uv_call_is_no_sync")


def test_the_uv_line_matchers_see_each_spelling():
    """One line per matcher clause: each shape must be flagged, or must not be.

    The real hooks hold none of these shapes, so without this test a clause
    dropped from UV_CALL, BARE_UV or PURE_ECHO would leave
    test_every_hook_uv_call_is_no_sync green and blind.
    """
    flagged = {
        "the #220 spelling": 'uv run clang-format -i "${c_files[@]}"',
        "--no-sync only in a trailing comment":
        'uv run clang-format -i x  # was: uv run --no-sync clang-format',
        "a bare call chained after a no-sync one": 'uv run --no-sync ruff check x && uv run yapf x',
        "a quoted command word": '"uv" run python -c pass',
        "another syncing subcommand": 'uv sync --quiet',
        "an off-spelling install() would not rewrite": 'uv  run --no-sync ruff',
        "safe, but not the one spelling": 'UV_NO_SYNC=1 uv run ruff',
        "an echo, but not only an echo": 'echo "note" && uv run yapf x',
        "command substitution runs uv": 'echo "$(uv run python -V)"',
    }
    ignored = {
        "the one spelling": 'uv run --no-sync ruff check x',
        "a file name": '        *.py|pyproject.toml|uv.lock|\\',
        "an env var": 'export UV_PYTHON=python3.12',
        "uvx never touches the project env": 'uvx ruff --version',
        "a comment": '    # uv run clang-format -i x',
        "a message": '    echo "[pre-commit] NOTE: run \'uv sync\' first."',
    }
    assert [why for why, line in flagged.items() if not bare_uv_lines(line)] == []
    assert [why for why, line in ignored.items() if bare_uv_lines(line)] == []
