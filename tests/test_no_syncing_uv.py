"""No test may shell out to a syncing `uv` (#220).

WHY: a git worktree here borrows main's .venv. A syncing `uv run` from it (one
without --no-sync) re-installs the editable cs2rl pointing at the worktree and
rebuilds main's binding .so in place, under the very pytest process that
spawned it, which then segfaulted (#220). The defect has recurred: the W3b
binding campaign, then tests/test_train_cli.py's --dump-config calls. Run
sys.executable instead (or `uv run --no-sync`).

WHAT IS FLAGGED, by parsing each tests/**/*.py with `ast` (nothing is run):
  * an argv literal: a list or tuple whose first element is the literal "uv"
    and which has no "--no-sync" element (`["uv", "run", "python", ...]`,
    `("uv", "sync")`);
  * a command string with `uv run` at a shell command position and no
    --no-sync after `run` nor UV_NO_SYNC=1 before `uv`, where the string is
    handed to something that runs or splits it: a positional argument of a
    callee named in _RUNNERS (subprocess.run/call/check_call/check_output/Popen/
    getoutput/getstatusoutput, os.system, shlex.split), the receiver of
    `.split()`, or the element after "-c" in an argv literal. f-strings count,
    with each field read as `{}`.
Docstrings, assert messages and `regenerate` hints are none of those, so they
may name `uv run` freely.

BLIND SPOTS (the measured scope, not a proof):
  * argv or strings reached through a variable (`[UV, "run"]`, a command
    assigned first and passed later);
  * argv whose first element is not the literal "uv" (`["env", "FOO=1", "uv",
    ...]`, `[shutil.which("uv"), ...]`);
  * scripts/ that tests execute. scripts/run_rung1.sh runs `uv run python` and is
    safe under test only because it prefixes `UV_NO_SYNC=1`.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Callees whose positional string arguments are run, or split into an argv, as a command line.
_RUNNERS = frozenset({
    "run", "call", "check_call", "check_output", "Popen", "getoutput", "getstatusoutput", "system",
    "split"
})
# `uv run` where a shell command starts (string start, a new line, or after `;`, `&`,
# `|`, `(`), with optional `env` and VAR=value prefixes captured in group 1, and not
# followed by --no-sync.
_UV_RUN = re.compile(r"(?:^|[\n;&|(])\s*((?:env\s+)?(?:\w+=\S*\s+)*)uv\s+run\b(?!\s+--no-sync\b)")


def _text(node: ast.AST) -> str | None:
    """A string literal's text (an f-string's, with each field as `{}`), else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}"
                       for v in node.values)
    return None


def _is_syncing_argv(node: ast.List | ast.Tuple) -> bool:
    """A list/tuple literal that starts with "uv" and has no "--no-sync" element."""
    if not node.elts:
        return False
    first = node.elts[0]
    if not (isinstance(first, ast.Constant) and first.value == "uv"):
        return False
    return not any(isinstance(e, ast.Constant) and e.value == "--no-sync" for e in node.elts)


def _command_strings(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """(line, text) of every string literal the module hands to a runner or splitter."""
    for node in ast.walk(tree):
        candidates: list[ast.expr] = []
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in _RUNNERS:
                candidates.extend(node.args)
            if isinstance(func, ast.Attribute) and func.attr == "split":
                candidates.append(func.value)
        elif isinstance(node, (ast.List, ast.Tuple)):
            candidates.extend(arg for flag, arg in pairwise(node.elts)
                              if isinstance(flag, ast.Constant) and flag.value == "-c")
        for candidate in candidates:
            text = _text(candidate)
            if text is not None:
                yield candidate.lineno, text


def syncing_uv_sites(src: str, name: str) -> list[int]:
    """Line numbers in `src` of an argv literal or command string that runs a syncing uv."""
    tree = ast.parse(src, name)
    hits = {
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.List, ast.Tuple)) and _is_syncing_argv(node)
    }
    for line, text in _command_strings(tree):
        if any("UV_NO_SYNC=1" not in m.group(1) for m in _UV_RUN.finditer(text)):
            hits.add(line)
    return sorted(hits)


def test_the_checker_sees_each_spelling():
    """One plant per clause, beside the safe spellings it must leave alone.

    tests/ holds none of the syncing shapes today, so without these a clause
    that stopped matching would narrow the scan below and leave it green.
    """
    syncing = {
        "argv (the #220 spelling)": 'subprocess.run(["uv", "run", "python", "train.py"])',
        "argv, sync": 'subprocess.run(("uv", "sync"))',
        "shell string": 'subprocess.run("uv run python x", shell=True)',
        "after a separator": 'subprocess.run("cd src && uv run python x", shell=True)',
        "env prefix, no UV_NO_SYNC": 'os.system("env FOO=1 uv run python x")',
        "bash -c": 'subprocess.run(["bash", "-c", "uv run pytest"])',
        "shlex.split": 'subprocess.run(shlex.split("uv run python x"))',
        ".split()": 'subprocess.run("uv run python x".split())',
        "f-string": 'subprocess.run(f"uv run python {script}", shell=True)',
    }
    safe = {
        "argv --no-sync": 'subprocess.run(["uv", "run", "--no-sync", "python", "x"])',
        "string --no-sync": 'subprocess.run("uv run --no-sync python x", shell=True)',
        "UV_NO_SYNC=1 prefix": 'os.system("UV_NO_SYNC=1 uv run python x")',
        "sys.executable": 'subprocess.run([sys.executable, "train.py"])',
        "a failure message": 'pytest.fail("regenerate: uv run python x")',
        "a docstring": 'def f():\n    """uv run python x"""',
        "an assert message": 'assert ok, "run: uv run python setup.py build_ext"',
    }
    assert [label for label, src in syncing.items() if not syncing_uv_sites(src, label)] == []
    assert [label for label, src in safe.items() if syncing_uv_sites(src, label)] == []


def test_no_test_shells_out_to_a_syncing_uv():
    """No file under tests/ runs a syncing uv (#220); see the module docstring for scope."""
    paths = sorted((REPO / "tests").rglob("*.py"))
    # Population: the scan must read the file that carried the #220 defect.
    assert REPO / "tests" / "test_train_cli.py" in paths, paths[:5]
    hits = [
        f"{path.relative_to(REPO)}:{line}" for path in paths
        for line in syncing_uv_sites(path.read_text(encoding="utf-8"), str(path))
    ]
    assert hits == [], f"run sys.executable (or `uv run --no-sync`), not a syncing uv (#220): {hits}"
