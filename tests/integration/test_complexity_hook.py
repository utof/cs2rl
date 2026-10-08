"""Tests for the complexity ratchet in .githooks/pre-commit (#216).

The hook runs complexipy with `--diff HEAD --staged --max-complexity-allowed 40`:
a staged function fails the commit when it is NEW and over 40, or when it is over
40 and got more complex than at HEAD. Like the pyrefly tests, each rejection asserts
the gate's own output (the `REGRESSED` / `NEW` row naming the function), and is
paired with controls that a hook rejecting everything would fail: a function under
40, and an edit that leaves an already-over-40 function exactly as complex.

The throwaway repo is the pyrefly tests' (`make_repo`), whose hook install rewrites
`uv run --no-sync ` to the real checkout's .venv/bin, so ruff and complexipy are the
installed ones. The complexity step runs before pyrefly, so a rejection here never
reaches it.
"""

from __future__ import annotations

from tests.integration.test_pyrefly_hook import commit, git, head_count, make_repo


def _fn(name: str, branches: int) -> str:
    """A typed function whose cognitive complexity is 2 * branches + 1.

    The `for` costs 1 and each `if` nested in it costs 2.
    """
    lines = [f"def {name}(xs: list[int]) -> int:", "    t = 0", "    for x in xs:"]
    for i in range(branches):
        lines += [f"        if x == {i}:", f"            t += {i}"]
    return "\n".join(lines + ["    return t", ""])


OVER = 21                              # complexity 43
UNDER = 19                             # complexity 39
SEED = _fn("old", 20)                  # complexity 41: an existing offender


def test_a_new_function_over_the_threshold_is_rejected(tmp_path):
    root = make_repo(tmp_path, {"src/m.py": SEED})
    before = head_count(root)

    (root / "src" / "sibling.py").write_text(_fn("fresh", OVER))
    git(root, "add", "src/sibling.py")
    res = commit(root, "new offender")

    out = res.stdout + res.stderr
    assert head_count(root) == before, "the commit was created anyway"
    assert "NEW" in out and "src/sibling.py::fresh" in out, out


def test_a_new_function_under_the_threshold_is_accepted(tmp_path):
    """The negative control: the ratchet does not reject every new function."""
    root = make_repo(tmp_path, {"src/m.py": SEED})
    before = head_count(root)

    (root / "src" / "sibling.py").write_text(_fn("fresh", UNDER))
    git(root, "add", "src/sibling.py")
    res = commit(root, "new but fine")

    assert head_count(root) == before + 1, res.stdout + res.stderr


def test_an_existing_function_pushed_over_the_threshold_is_rejected(tmp_path):
    root = make_repo(tmp_path, {"src/m.py": _fn("old", UNDER)})
    before = head_count(root)

    (root / "src" / "m.py").write_text(_fn("old", OVER))
    git(root, "add", "src/m.py")
    res = commit(root, "push over")

    out = res.stdout + res.stderr
    assert head_count(root) == before, "the commit was created anyway"
    assert "REGRESSED" in out and "src/m.py::old" in out, out


def test_an_existing_offender_made_worse_is_rejected(tmp_path):
    root = make_repo(tmp_path, {"src/m.py": SEED})
    before = head_count(root)

    (root / "src" / "m.py").write_text(_fn("old", 21))
    git(root, "add", "src/m.py")
    res = commit(root, "worse")

    out = res.stdout + res.stderr
    assert head_count(root) == before, "the commit was created anyway"
    assert "REGRESSED" in out and "src/m.py::old" in out, out


def test_an_existing_offender_edited_without_getting_worse_is_accepted(tmp_path):
    """The ratchet's whole point: the 41 stays allowed while it does not grow."""
    root = make_repo(tmp_path, {"src/m.py": SEED})
    before = head_count(root)

    (root / "src" / "m.py").write_text(SEED.replace("t += 0", "t += 100"))
    git(root, "add", "src/m.py")
    res = commit(root, "same complexity")

    assert head_count(root) == before + 1, res.stdout + res.stderr
