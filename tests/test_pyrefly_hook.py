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
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REAL_PYTHON = REPO / ".venv" / "bin" / "python"
REAL_PYREFLY = REPO / ".venv" / "bin" / "pyrefly"
HOOK = REPO / ".githooks" / "pre-commit"
MERGE_HOOK = REPO / ".githooks" / "pre-merge-commit"

CONFIG = 'preset = "default"\nproject-includes = ["src"]\nsearch-path = ["src", "."]\n'


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def install(root: Path, src: Path, name: str) -> None:
    """Copy a hook into the throwaway, rewriting it to use the real repo's tools.

    Two rewrites, and the second one is the part that is easy to miss:

      1. `uv run ` -> `<REPO>/.venv/bin/`, so no `uv` invocation ever happens from
         /tmp. uv would try to build the throwaway as a project and download
         gigabytes. This also removes any need for UV_OFFLINE/UV_NO_SYNC.
      2. the RELATIVE `scripts/pyrefly_gate.py` -> an absolute path into the real
         repo, plus `--project .`. The hook cd's to the throwaway first, so
         rewriting only the interpreter leaves it looking for a gate script that
         does not exist there -- which produces a rejected commit with the marker
         printed and the gate never run, i.e. exactly the false pass these tests
         are written to prevent.
    """
    text = src.read_text()
    text = text.replace("uv run ", f"{REPO}/.venv/bin/")
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
