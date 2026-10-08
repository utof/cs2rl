"""Tests for the complexity ratchet in .githooks/pre-commit (#216).

The hook runs complexipy with `--diff HEAD --staged --max-complexity-allowed 40`:
a staged function fails the commit when it is NEW and over 40, or when it is over
40 and got more complex than at HEAD. Like the pyrefly tests, each rejection asserts
the gate's own output (the `REGRESSED` / `NEW` row naming the function), and is
paired with controls that a hook rejecting everything would fail: a function under
40, and an edit that leaves an already-over-40 function exactly as complex.

Two more pins: the ruff C901 step (a function that is over 40 by McCabe but not by
cognitive complexity, so only ruff can reject it) and pre-merge-commit (a clean merge
whose two sides each pass but whose result is over 40). The last test checks that
every tool the hooks run through `uv run --no-sync` is declared in the dev group.

The throwaway repo is the pyrefly tests' (`make_repo`), whose hook install rewrites
`uv run --no-sync ` to the real checkout's .venv/bin, so ruff and complexipy are the
installed ones. The complexity step runs before pyrefly, so a rejection here never
reaches it.
"""

from __future__ import annotations

import re
import tomllib

from tests.conftest import REPO_ROOT
from tests.integration.test_pyrefly_hook import (
    HOOK,
    MERGE_HOOK,
    NO_SYNC,
    commit,
    git,
    head_count,
    make_repo,
    uv_call_lines,
)


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


def _flat(name: str, ifs: int) -> str:
    """`ifs` flat `if`s: cognitive complexity `ifs`, McCabe complexity `ifs + 1`."""
    lines = [f"def {name}(x: int) -> int:", "    t = 0"]
    for i in range(ifs):
        lines += [f"    if x == {i}:", f"        t += {i}"]
    return "\n".join(lines + ["    return t", ""])


def test_ruff_rejects_a_function_over_40_by_mccabe_only(tmp_path):
    """Cognitive 40 passes complexipy (not over 40); McCabe 41 is over ruff's 40."""
    root = make_repo(tmp_path, {"src/m.py": SEED})
    before = head_count(root)

    (root / "src" / "wide.py").write_text(_flat("wide", 40))
    git(root, "add", "src/wide.py")
    res = commit(root, "mccabe only")

    out = res.stdout + res.stderr
    assert head_count(root) == before, "the commit was created anyway"
    assert "C901" in out and "`wide`" in out, out
    assert "Cognitive complexity ratchet" not in out, f"ruff should stop it first:\n{out}"


def _merge_sides(root, feat_edit, main_edit):
    """Commit `feat_edit` on a branch and `main_edit` on the base, then merge."""
    git(root, "checkout", "-q", "-b", "feat")
    feat_edit()
    git(root, "commit", "-q", "-am", "feat side")
    git(root, "checkout", "-q", "-")
    main_edit()
    git(root, "commit", "-q", "-am", "main side")
    return git(root, "merge", "--no-ff", "-m", "merge feat", "feat")


def _add_branch(root, marker: str, at_top: bool):
    """Add one nested `if` (cognitive +2) at the top or the bottom of `near`."""
    path = root / "src" / "m.py"
    lines = path.read_text().split("\n")
    block = [f"        if x == {marker}:", "            t += 1"]
    at = 3 if at_top else lines.index("    return t")
    lines[at:at] = block
    path.write_text("\n".join(lines))


def test_pre_merge_commit_rejects_a_clean_merge_that_crosses_40(tmp_path):
    """Each side adds one branch to a 37 and passes at 39; the merged result is 41."""
    root = make_repo(tmp_path, {"src/m.py": _fn("near", 18)})
    before = head_count(root)

    res = _merge_sides(root, lambda: _add_branch(root, "100", True),
                       lambda: _add_branch(root, "200", False))
    out = res.stdout + res.stderr

    # Both side commits went through pre-commit at 39; only the merge is over 40.
    assert head_count(root) == before + 1, f"a side commit was rejected:\n{out}"
    assert "[pre-merge-commit] Cognitive complexity ratchet" in out, out
    assert "REGRESSED" in out and "src/m.py::near" in out, out
    assert (root / ".git" / "MERGE_HEAD").exists(), "the failed merge should stay incomplete"


def test_pre_merge_commit_accepts_a_clean_merge_that_stays_under_40(tmp_path):
    """The control: the same shape with only one side editing `near` merges."""
    root = make_repo(tmp_path, {"src/m.py": _fn("near", 18), "src/o.py": _fn("other", 1)})

    res = _merge_sides(root, lambda: _add_branch(root, "100", True), lambda:
                       (root / "src" / "o.py").write_text(_fn("other", 2)))
    out = res.stdout + res.stderr

    assert not (root / ".git" / "MERGE_HEAD").exists(), f"the merge was rejected:\n{out}"
    assert "[pre-merge-commit] Cognitive complexity ratchet" in out, out


def test_every_hook_tool_is_declared_in_the_dev_group():
    """A tool the hooks run but pyproject does not declare vanishes on `uv sync`.

    complexipy once ran from the shared .venv without being declared (#216).
    `python` is the interpreter, not a package.
    """
    dev = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["dependency-groups"]["dev"]
    declared = {re.split(r"[=<>!~\[ ]", d, maxsplit=1)[0] for d in dev}
    tools = set()
    for hook in (HOOK, MERGE_HOOK):
        for line in uv_call_lines(hook.read_text()):
            tools.add(line.split(NO_SYNC, 1)[1].split()[0])
    tools.discard("python")
    assert {"ruff", "yapf", "complexipy", "clang-format"} <= tools, tools
    assert tools <= declared, f"hook tools missing from the dev group: {sorted(tools - declared)}"
