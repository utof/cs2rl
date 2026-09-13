"""Pin collect/update timing at train()'s evaluate/train call sites."""
from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "src" / "train.py"


def _stmt_lists(fn: ast.FunctionDef) -> list[list[ast.stmt]]:
    """Every statement LIST reachable in `fn`, so neighbours can be inspected.

    The pin needs sibling order (is the statement before this one a timer
    start?), which `ast.walk` destroys — hence lists, not nodes. Nested
    def/class bodies are skipped: a helper defined inside train() has its own
    scope and its `trainer.train()` would be a false positive.
    """
    lists: list[list[ast.stmt]] = []
    stack: list[list[ast.stmt]] = [list(fn.body)]
    while stack:
        body = stack.pop()
        lists.append(body)
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for child in ast.iter_child_nodes(stmt):
                if isinstance(child, ast.ExceptHandler):
                    stack.append(list(child.body))
                elif isinstance(child, ast.stmt):
                    continue
            for attr in ("body", "orelse", "finalbody"):
                block = getattr(stmt, attr, None)
                if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                    if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        stack.append(block)
    return lists


def _is_perf_counter(call: ast.Call) -> bool:
    """True for any `<anything>.perf_counter()` call.

    Matched on the attribute name alone so `time.perf_counter()` and an
    aliased `t.perf_counter()` both count.
    """
    func = call.func
    return isinstance(func, ast.Attribute) and func.attr == "perf_counter"


def _is_helper_call(call: ast.Call, helper_names: set[str]) -> bool:
    """True if this is a direct call to one of train.py's `_time*` helpers.

    Lets the spec's optional `_time_ms(fn)` refactor satisfy the pin without
    literal `perf_counter` at the call site.
    """
    func = call.func
    return isinstance(func, ast.Name) and func.id in helper_names


def _stmt_times(stmt: ast.AST, helper_names: set[str]) -> bool:
    """True if `stmt` anywhere starts a clock (perf_counter or a `_time*` helper)."""
    for node in ast.walk(stmt):
        if isinstance(node, ast.Call) and (_is_perf_counter(node)
                                           or _is_helper_call(node, helper_names)):
            return True
    return False


def _stmt_writes_key(stmt: ast.AST, key: str) -> bool:
    """True if `stmt` subscripts anything with the literal `key`.

    Deliberately blind to WHICH object is subscripted: `logs[...]` and
    `trainer._timing[...]` both count, so the pin does not freeze the storage
    choice — only that the measurement is written.
    """
    for node in ast.walk(stmt):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if node.slice.value == key:
                return True
    return False


def _trainer_method_call(node: ast.AST, method: str) -> bool:
    """True for a literal `trainer.<method>()` call node."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Attribute) and func.attr == method
            and isinstance(func.value, ast.Name) and func.value.id == "trainer")


def _helper_arg_is_trainer_method(call: ast.Call, method: str) -> bool:
    """True if `call`'s first arg is `trainer.<method>` (bound or already called).

    Covers `_time_ms(trainer.train)` — the helper form, where the method is
    passed rather than called at the site.
    """
    if not call.args:
        return False
    arg = call.args[0]
    if isinstance(arg, ast.Attribute) and arg.attr == method:
        return isinstance(arg.value, ast.Name) and arg.value.id == "trainer"
    if _trainer_method_call(arg, method):
        return True
    return False


def _expr_nodes(stmt: ast.stmt) -> list[ast.AST]:
    """This statement's expressions only — not nested body lists (While/If)."""
    roots: list[ast.AST | None] = []
    if isinstance(stmt, ast.Expr):
        roots.append(stmt.value)
    elif isinstance(stmt, ast.Assign):
        roots.extend(stmt.targets)
        roots.append(stmt.value)
    elif isinstance(stmt, ast.AnnAssign):
        roots.append(stmt.value)
    elif isinstance(stmt, ast.AugAssign):
        roots.append(stmt.value)
    elif isinstance(stmt, ast.If):
        roots.append(stmt.test)
    elif isinstance(stmt, ast.While):
        roots.append(stmt.test)
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        roots.append(stmt.target)
        roots.append(stmt.iter)
    nodes: list[ast.AST] = []
    for root in roots:
        if root is not None:
            nodes.extend(ast.walk(root))
    return nodes


def test_train_evaluate_and_train_calls_are_timed():
    """The real gate: both Call nodes in module-level `train()` must be timed.

    Fails if `train()` still names `_patch_trainer_with_timing` (the monkey-patch
    is back) or if either call is neither preceded by a clock start, nor timed
    inline, nor followed by a `timing/*_ms` write. Pitfall: this reads the file
    as text — it can never catch a timing bug at RUNTIME, only the shape of the
    call sites. That is the point; the runtime path needs a GPU.
    """
    tree = ast.parse(TRAIN_PY.read_text())
    train_fn = None
    helper_names = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "train":
            train_fn = node
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_time"):
            helper_names.add(node.name)
    assert train_fn is not None
    for node in ast.walk(train_fn):
        if isinstance(node, ast.Name) and node.id == "_patch_trainer_with_timing":
            pytest.fail("train() still names _patch_trainer_with_timing")

    timed_eval = 0
    timed_train = 0
    untimed = []
    for body in _stmt_lists(train_fn):
        for i, stmt in enumerate(body):
            nested_defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            if isinstance(stmt, nested_defs):
                continue
            for node in _expr_nodes(stmt):
                if isinstance(node, nested_defs):
                    continue
                if _trainer_method_call(
                        node, "evaluate") or (isinstance(node, ast.Call)
                                              and _is_helper_call(node, helper_names)
                                              and _helper_arg_is_trainer_method(node, "evaluate")):
                    prev_ok = i > 0 and _stmt_times(body[i - 1], helper_names)
                    here_ok = _stmt_times(stmt, helper_names)
                    write_ok = _stmt_writes_key(stmt, "timing/collect_ms")
                    next_ok = (i + 1 < len(body)
                               and _stmt_writes_key(body[i + 1], "timing/collect_ms"))
                    if prev_ok or here_ok or write_ok or next_ok:
                        timed_eval += 1
                    else:
                        untimed.append(("evaluate", ast.dump(stmt)))
                if _trainer_method_call(
                        node, "train") or (isinstance(node, ast.Call) and _is_helper_call(
                            node, helper_names) and _helper_arg_is_trainer_method(node, "train")):
                    prev_ok = i > 0 and _stmt_times(body[i - 1], helper_names)
                    here_ok = _stmt_times(stmt, helper_names)
                    write_ok = _stmt_writes_key(stmt, "timing/update_ms")
                    next_ok = (i + 1 < len(body)
                               and _stmt_writes_key(body[i + 1], "timing/update_ms"))
                    if prev_ok or here_ok or write_ok or next_ok:
                        timed_train += 1
                    else:
                        untimed.append(("train", ast.dump(stmt)))
    assert untimed == []
    assert timed_eval >= 1
    assert timed_train >= 1


def test_time_ms_helper_is_not_the_gate():
    """Positive elapsed ms on a dummy callable. Not sufficient without the AST pin."""

    def _time_ms(fn):
        t0 = time.perf_counter()
        result = fn()
        return result, (time.perf_counter() - t0) * 1000.0

    def work():
        time.sleep(0.01)
        return {"SPS": 1}

    logs, elapsed = _time_ms(work)
    assert logs["SPS"] == 1
    assert elapsed > 0
