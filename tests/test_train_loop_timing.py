"""Pin collect/update timing at train()'s evaluate/train call sites."""
from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "src" / "cs2rl" / "train.py"

# The ONE helper name allowed to stand in for an inline perf_counter at a call
# site. Deliberately an exact name rather than a `_time*` prefix: `_timestamp_dir`
# and `_timesteps_from_args` are entirely plausible names in a training script,
# and under a prefix rule the day one of them happens to be called next to
# `trainer.evaluate()` this pin would silently start accepting an untimed site.
HELPER_NAME = "_time_ms"

# (trainer method, suffix of the metric key its elapsed time must be written to).
# Driving both sites off one table instead of two hand-synced copy-paste branches.
SITES = (("evaluate", "collect_ms"), ("train", "update_ms"))


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
    """True if this is a direct call to train.py's `_time_ms` helper.

    `helper_names` is empty unless a module-level `def _time_ms` actually
    exists, so this cannot fire on a name that was never defined.
    """
    func = call.func
    return isinstance(func, ast.Name) and func.id in helper_names


def _stmt_times(stmt: ast.AST, helper_names: set[str]) -> bool:
    """True if `stmt` anywhere starts a clock (perf_counter or the `_time_ms` helper).

    Necessary but NOT sufficient — see `_clock_names`. On its own this accepts
    `print(time.perf_counter())`, which measures nothing.
    """
    for node in ast.walk(stmt):
        if isinstance(node, ast.Call) and (_is_perf_counter(node)
                                           or _is_helper_call(node, helper_names)):
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


def _writes_key_suffix(stmt: ast.stmt, suffix: str) -> bool:
    """True if `stmt` ITSELF subscripts a constant key whose name ends in `suffix`.

    Suffix, not equality: the loop stores `trainer._timing["collect_ms"]` while
    the logs dict uses `logs["timing/collect_ms"]`. The pin's first version
    compared against the literal `"timing/collect_ms"`, which never appears in
    the statement next to the call — so that half of the gate was dead code.

    Uses `_expr_nodes`, NOT `ast.walk`. Walking an `if isinstance(logs, dict):`
    would descend into its body and find the downstream `logs["timing/update_ms"]`
    write, letting the train site take credit for a statement that is not the
    elapsed computation at all.
    """
    for node in _expr_nodes(stmt):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            value = node.slice.value
            if isinstance(value, str) and value.endswith(suffix):
                return True
    return False


def _names_assigned(stmt: ast.stmt) -> set[str]:
    """Names this statement binds — its OWN bindings only.

    `_expr_nodes` keeps this out of nested bodies, so an `if`/`while` binds
    nothing here even though its body does. That is what makes the forward scan
    in `_elapsed_write_on_chain` stop at an unrelated block instead of absorbing
    the block's variables into the timing chain.
    """
    return {
        n.id
        for n in _expr_nodes(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }


def _names_read(stmt: ast.stmt) -> set[str]:
    """Names this statement reads — its own expressions only (see `_names_assigned`)."""
    return {
        n.id
        for n in _expr_nodes(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _clock_names(body: list[ast.stmt], i: int, via_helper: bool,
                 helper_names: set[str]) -> set[str]:
    """Names holding the clock reading that the elapsed write must consume.

    Two accepted shapes:
      A. the PRECEDING statement binds a `perf_counter()` reading (`t0 = ...`);
      B. the call site itself is `_time_ms(trainer.X)`, which times internally,
         so whatever that statement binds IS the elapsed value.

    Returns an empty set when nothing is bound, and emptiness is the whole
    point: `print(time.perf_counter())` on the line above mentions a clock but
    binds nothing, so no later statement can consume it and the site stays
    untimed. Requiring a BINDING is what makes this stricter than `_stmt_times`.
    """
    if via_helper:
        return _names_assigned(body[i])
    if i > 0 and _stmt_times(body[i - 1], helper_names):
        return _names_assigned(body[i - 1])
    return set()


def _elapsed_write_on_chain(body: list[ast.stmt],
                            i: int,
                            clock_names: set[str],
                            suffix: str,
                            lookahead: int = 3) -> bool:
    """True if the elapsed time is written near the call AND consumes that clock.

    Conjunctive on purpose. The first version of this pin OR-ed four hatches,
    three of which were dead against the real code shape, leaving the surviving
    property as merely "the previous line mentions perf_counter" — so deleting
    both elapsed writes kept the pin green while `collect_ms` shipped 0.0 to
    W&B and metrics.jsonl forever. Requiring BOTH a bound clock and a write that
    reads it is what closes that.

    Walks forward at most `lookahead` statements from the call. Every statement
    crossed must stay on the timing chain (read a name the clock tainted) and
    may then extend it, so the two-step spelling

        collect_ms = (time.perf_counter() - t0) * 1000.0
        trainer._timing["collect_ms"] = collect_ms

    still passes. An unrelated statement wedged between the call and the write
    breaks the chain and fails the pin — its cost would otherwise be billed to
    collect_ms with nothing objecting.
    """
    tainted = set(clock_names)
    for offset, stmt in enumerate(body[i:i + lookahead]):
        # The call statement itself is on the chain by definition.
        on_chain = offset == 0 or bool(_names_read(stmt) & tainted)
        if on_chain and _writes_key_suffix(stmt, suffix):
            return True
        if not on_chain:
            return False
        tainted |= _names_assigned(stmt)
    return False


def test_train_evaluate_and_train_calls_are_timed():
    """The real gate: both Call nodes in module-level `train()` must be timed.

    A site counts as timed only when a clock reading is BOUND next to it (or by
    `_time_ms` at it) and an elapsed value derived from that binding is written
    to a `collect_ms` / `update_ms` key within the next couple of statements.

    Pitfall: this reads the file as text — it can never catch a timing bug at
    RUNTIME, only the shape of the call sites. That is the point; the runtime
    path needs a GPU. The companion `test_time_ms_helper_is_not_the_gate` is a
    dummy-callable smoke test and is explicitly NOT coverage of these sites.
    """
    tree = ast.parse(TRAIN_PY.read_text())
    train_fn = None
    helper_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "train":
            train_fn = node
        if isinstance(node, ast.FunctionDef) and node.name == HELPER_NAME:
            helper_names.add(node.name)
    assert train_fn is not None
    for node in ast.walk(train_fn):
        if isinstance(node, ast.Name) and node.id == "_patch_trainer_with_timing":
            pytest.fail("train() still names _patch_trainer_with_timing")

    nested_defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    timed = {method: 0 for method, _ in SITES}
    untimed = []
    for body in _stmt_lists(train_fn):
        for i, stmt in enumerate(body):
            if isinstance(stmt, nested_defs):
                continue
            for node in _expr_nodes(stmt):
                if not isinstance(node, ast.Call):
                    continue
                # A plain loop over SITES, NOT @pytest.mark.parametrize: a
                # parametrize's own data table is itself unwatched, and deleting
                # a row silently deletes the case that would have objected.
                for method, suffix in SITES:
                    via_helper = (_is_helper_call(node, helper_names)
                                  and _helper_arg_is_trainer_method(node, method))
                    if not (_trainer_method_call(node, method) or via_helper):
                        continue
                    clock = _clock_names(body, i, via_helper, helper_names)
                    if clock and _elapsed_write_on_chain(body, i, clock, suffix):
                        timed[method] += 1
                    else:
                        untimed.append((method, ast.dump(stmt)))
    assert untimed == []
    assert timed["evaluate"] >= 1
    assert timed["train"] >= 1


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
