"""Constructor composition and moved-method guards for Cs2PuffeRL.

The stock surface is measured at the end of PuffeRL.__init__. The remaining
constructor surface is compared against self stores in Cs2PuffeRL.__init__,
its called _init_* methods, including hybrid-aim state. Source
scanning includes inactive branches, so an undeclared runtime attribute or
an unexecuted declaration turns the test red.

The independent anchor reads moved method bodies and checkpoint helpers to
check that required attributes exist. Method identity tests pin train,
evaluate and save_checkpoint to Cs2PuffeRL, with no instance bindings.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "cs2rl"

# (file, qualname) of each moved body and checkpoint helper whose self/trainer
# reads must be backed by constructor state. The save body moved to the class
# in W2c; all three public bodies now resolve from trainer.py.
ANCHOR_FUNCTIONS = (
    ("train/trainer.py", "Cs2PuffeRL.train"),
    ("train/trainer.py", "Cs2PuffeRL.evaluate"),
    ("train/trainer.py", "Cs2PuffeRL.save_checkpoint"),
    ("train/resume.py", "collect_train_state"),
    ("train/resume.py", "restore_train_state"),
)
# 22 required data names: 19 after W2a plus three self-play fields after W2b.
# W2c uses the already-anchored `_self_play_mgr`, so this count stays 22.
EXPECTED_ANCHOR_COUNT = 22


def _bindings_in_scope(scope, name):
    """Statements in `scope`'s OWN body that bind `name` (def/class, Name target, import).

    Recurses through if/for/while/try/with/match blocks (same scope) but never into a
    nested def/class body, which is its own scope. Same rule as ast_oracle._scope_bindings
    in the SDD folder (the `match` arms were the W2a review's nit: `case` bodies are
    neither `body` nor `orelse`, so a def bound inside one was invisible here).
    """
    hits: list[ast.stmt] = []

    def walk(stmts):
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if s.name == name:
                    hits.append(s)
                continue
            if isinstance(s, (ast.Import, ast.ImportFrom)):
                if any((a.asname or a.name.split(".")[0]) == name for a in s.names):
                    hits.append(s)
            elif isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = s.targets if isinstance(s, ast.Assign) else [s.target]
                if any(
                        isinstance(n, ast.Name) and n.id == name for t in targets
                        for n in ast.walk(t)):
                    hits.append(s)
            for field in ("body", "orelse", "finalbody"):
                walk(getattr(s, field, []) or [])
            for h in getattr(s, "handlers", []) or []:
                walk(h.body)
            for c in getattr(s, "cases", []) or []:
                walk(c.body)

    walk(scope.body)
    return hits


def test_bindings_in_scope_sees_a_def_inside_a_match_case():
    """Pin on the `cases` clause of _bindings_in_scope (W2b fold of the W2a review nit).

    A `match` arm's body is neither `body` nor `orelse` nor a handler, so before the
    clause a def bound inside one was invisible: `find_def` would report the name as
    unbound and the anchor derivation would raise on a body that actually exists.
    Measured: deleting the two `cases` lines makes this test red (0 hits) and nothing
    else in the file notices.
    """
    mod = ast.parse("match x:\n    case 1:\n        def f(): ...\n")
    hits = _bindings_in_scope(mod, "f")
    assert len(hits) == 1 and isinstance(hits[0], ast.FunctionDef), hits


def find_def(module, qualname):
    """The def at a dotted qualname; each part must be bound EXACTLY once in its scope.

    Python keeps the LAST binding, so a first-match lookup would anchor on a dead def
    while a later duplicate runs (spec §2.1, re-review Major 2). Raising on zero or two
    bindings is what makes a rename or a stale copy fail loudly instead of vacuously.
    A part may be a ClassDef (W2a re-points to `Cs2PuffeRL.train`), exactly as
    ast_oracle.find_def accepts; the LAST part must be a def, since the caller reads its
    arguments and body.
    """
    node = module
    for part in qualname.split("."):
        hits = _bindings_in_scope(node, part)
        defs = [
            h for h in hits if isinstance(h, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if len(hits) != 1 or not defs:
            raise AssertionError(f"{qualname!r}: {part!r} bound {len(hits)} times "
                                 f"(lines {[h.lineno for h in hits]}); expected exactly one "
                                 "def/class")
        node = defs[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise AssertionError(f"{qualname!r} names a class, not a function")
    return node


def _attribute_reads_and_stores(fn):
    """({read}, {stored}) attribute names accessed on the trainer instance.

    The receiver is ``self`` OR ``trainer``: the closure bodies take ``self`` but also read
    the enclosing patch function's ``trainer`` (the same object; ``_batch1_max_entropy`` is
    read ONLY that way in the train body), and the resume_state helpers take ``trainer``.
    Reads: ``recv.<name>`` in Load context, and ``getattr(recv, "<name>")`` with exactly two
    arguments (a 3-argument getattr has a default and so does not require the attribute).
    Stores: ``recv.<name>`` in Store/Del context. Reads inside nested defs count too: a
    nested helper moves with its body.
    """
    recv = {"self", "trainer"}
    reads, stores = set(), set()
    for node in ast.walk(fn):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in recv):
            (reads if isinstance(node.ctx, ast.Load) else stores).add(node.attr)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "getattr" and len(node.args) == 2
              and isinstance(node.args[0], ast.Name) and node.args[0].id in recv
              and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)):
            reads.add(node.args[1].value)
    return reads, stores


def derive_anchor():
    """Names read and never assigned across ANCHOR_FUNCTIONS (class attrs filtered later)."""
    reads, stores = set(), set()
    for rel, qualname in ANCHOR_FUNCTIONS:
        fn = find_def(ast.parse((PACKAGE / rel).read_text()), qualname)
        r, s = _attribute_reads_and_stores(fn)
        reads |= r
        stores |= s
    return reads - stores


@pytest.fixture
def composed(monkeypatch):
    """(trainer, stock_surface): the trainer ``_build_trainer_for_test`` returns.

    ``PuffeRL.__init__`` is wrapped so the stock surface is measured on THIS instance, not
    copied from a list that could drift with a pufferlib bump. W1 built the trainer here
    from ``_harness_parts``; since W1.5 the harness constructs ``Cs2PuffeRL`` itself, so
    going through it is what proves the production class is the one every harness test
    gets. ``_action_mask_shm`` (a GC pin the harness and train() both set after the
    constructor) is removed from the measured surface so (2) compares construction only.
    """
    from pufferlib.pufferl import PuffeRL

    from tests._helpers.trainer_harness import _build_trainer_for_test

    stock = {}
    orig_init = PuffeRL.__init__

    def recording_init(self, *a, **k):
        orig_init(self, *a, **k)
        stock["names"] = set(vars(self))

    monkeypatch.setattr(PuffeRL, "__init__", recording_init)
    trainer, cleanup = _build_trainer_for_test(num_envs=16)
    try:
        # Pin the pin: folding a name into the stock set that the harness no
        # longer sets would hide its disappearance from (2), so check first.
        assert "_action_mask_shm" in vars(trainer), (
            "the harness no longer pins the mask RawArray on the trainer after the "
            "constructor; (2) would silently absorb the missing name")
        yield trainer, stock["names"] | {"_action_mask_shm"}
    finally:
        cleanup()


def test_harness_returns_a_direct_pufferl_subclass(composed):
    from pufferlib.pufferl import PuffeRL

    from cs2rl.train.trainer import Cs2PuffeRL
    trainer, stock = composed
    assert type(trainer) is Cs2PuffeRL, (
        f"_build_trainer_for_test returned a {type(trainer).__name__}; since gh#168 W1.5 the "
        "harness must construct the production class (gh#169)")
    assert type(trainer).__mro__[1] is PuffeRL
    assert stock, "PuffeRL.__init__ did not run through the recording wrapper"


@pytest.mark.parametrize(
    ("source", "receiver", "expected_stores", "expected_init_calls"),
    [
        (
            "def patch(trainer):\n"
            "    trainer.constructed = 1\n"
            "    def callback():\n"
            "        trainer.call_time_only = 2\n"
            "    class Deferred:\n"
            "        trainer.class_time_only = 3\n",
            "trainer",
            {"constructed"},
            set(),
        ),
        (
            "def init(self):\n"
            "    del self.removed\n"
            "    def callback():\n"
            "        self._init_deferred()\n",
            "self",
            set(),
            set(),
        ),
        (
            "def init(self):\n"
            "    if False:\n"
            "        self.inactive = 1\n"
            "        self._init_inactive()\n"
            "    try:\n"
            "        for value in ():\n"
            "            self.loop_store = value\n"
            "    except Exception:\n"
            "        self.error_store = 2\n",
            "self",
            {"inactive", "loop_store", "error_store"},
            {"_init_inactive"},
        ),
    ],
)
def test_constructor_scope_scan_ignores_deferred_stores_and_deletes(source, receiver,
                                                                    expected_stores,
                                                                    expected_init_calls):
    """Only own-scope stores declare constructed state, even in inactive arms."""
    fn = ast.parse(source).body[0]
    assert _constructor_scope(fn, receiver) == (expected_stores, expected_init_calls)


def _constructor_scope(fn, receiver):
    """Own-scope Store targets and called initializers, including inactive branches.

    Nested defs/classes/lambdas run later or in a different scope; their stores
    cannot declare construction-time state. Del targets do not create state.
    The independent body-read anchor deliberately uses a deeper scan above.
    """
    stores, init_calls = set(), set()

    class Scan(ast.NodeVisitor):

        def visit_FunctionDef(self, node):
            return

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef

        def visit_Attribute(self, node):
            if (isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name)
                    and node.value.id == receiver):
                stores.add(node.attr)
            self.generic_visit(node)

        def visit_Call(self, node):
            if (receiver == "self" and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
                    and node.func.attr.startswith("_init_")):
                init_calls.add(node.func.attr)
            self.generic_visit(node)

    scan = Scan()
    for statement in fn.body:
        scan.visit(statement)
    return stores, init_calls


def derive_constructor_surface():
    """All names the subclass declares during construction, including inactive branches.

    The constructor and called initializers own every added runtime field.
    """
    trainer_tree = ast.parse((PACKAGE / "train" / "trainer.py").read_text())
    cls = next(n for n in trainer_tree.body
               if isinstance(n, ast.ClassDef) and n.name == "Cs2PuffeRL")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    names = set()
    pending = ["__init__"]
    scanned = set()
    while pending:
        method_name = pending.pop()
        if method_name in scanned:
            continue
        scanned.add(method_name)
        stores, init_calls = _constructor_scope(methods[method_name], "self")
        names |= stores
        pending.extend(init_calls - scanned)
    return names


def test_constructed_surface_equals_declared_constructor_surface(composed):
    """A dead-branch self store is a declaration and must be detected as extra."""
    trainer, stock = composed
    composed_surface = set(vars(trainer)) - stock
    declared = derive_constructor_surface()
    missing = declared - composed_surface
    extra = composed_surface - declared
    assert not missing and not extra, (
        f"constructor declaration differs from runtime: missing {sorted(missing)}, "
        f"unexpected {sorted(extra)}")


def test_every_attribute_the_bodies_read_is_declared(composed):
    from cs2rl.train.resume import _WARMSTART_ATTRS
    trainer, stock = composed
    # Class attributes are excluded through the INSTANCE's class, not PuffeRL: at W1 the
    # two agree (the 18-name count below asserts it), and from W2a on Cs2PuffeRL's own
    # methods (`_normalize_returns`, `train`, ...) must be excluded too.
    cls = type(trainer)
    anchor = {n for n in derive_anchor() if n not in stock and not hasattr(cls, n)}
    assert len(anchor) == EXPECTED_ANCHOR_COUNT, (
        f"derived anchor is {sorted(anchor)} ({len(anchor)} names); the derivation changed "
        f"or a body gained/lost a read. Update EXPECTED_ANCHOR_COUNT only with the W that "
        "moved the body.")
    absent = anchor - set(vars(trainer))
    assert not absent, f"read by a moved body but never set by __init__: {sorted(absent)}"
    warm_absent = set(_WARMSTART_ATTRS) - set(vars(trainer))
    assert not warm_absent, f"_WARMSTART_ATTRS not on the instance: {sorted(warm_absent)}"


def test_save_checkpoint_is_the_class_method(composed):
    """The full-state checkpoint body belongs to the subclass and has no instance binding."""
    from pufferlib.pufferl import PuffeRL

    from cs2rl.train.trainer import Cs2PuffeRL
    trainer, _ = composed
    cls = type(trainer)
    assert cls.save_checkpoint is Cs2PuffeRL.save_checkpoint
    assert cls.save_checkpoint is not PuffeRL.save_checkpoint
    assert Cs2PuffeRL.save_checkpoint.__qualname__ == ANCHOR_FUNCTIONS[2][1]
    assert "save_checkpoint" not in vars(trainer)
    assert trainer.save_checkpoint.__func__ is Cs2PuffeRL.save_checkpoint


def test_train_is_the_class_method(composed):
    """(4) for `train` after gh#168 W2a: the class defines it, and nothing re-binds it.

    Two halves, both needed. (a) `Cs2PuffeRL.train` is the def ANCHOR_FUNCTIONS[0] names
    and is NOT `PuffeRL.train`: a `def train` deleted from the class would fall through
    to the stock body, which knows nothing of return normalisation, and every harness
    test would run it. (b) `train` is not in `vars(trainer)`: a leftover MethodType
    binding (a re-introduced patcher, or a test that pokes one in) would shadow the
    class method and (a) alone would stay green. Read through `type(trainer)` so a
    subclass used by the harness would be caught too.
    """
    from pufferlib.pufferl import PuffeRL

    from cs2rl.train.trainer import Cs2PuffeRL
    trainer, _ = composed
    cls = type(trainer)
    assert cls.train is Cs2PuffeRL.train and cls.train is not PuffeRL.train, (
        "Cs2PuffeRL no longer defines train(); the stock PuffeRL body would run")
    assert Cs2PuffeRL.train.__qualname__ == ANCHOR_FUNCTIONS[0][1], (
        f"Cs2PuffeRL.train is {Cs2PuffeRL.train.__qualname__!r}; ANCHOR_FUNCTIONS[0] names "
        f"{ANCHOR_FUNCTIONS[0][1]!r}")
    assert "train" not in vars(trainer), (
        "train is an INSTANCE attribute again: something re-bound it after construction "
        "and shadows Cs2PuffeRL.train")
    assert trainer.train.__func__ is Cs2PuffeRL.train


def test_evaluate_is_the_class_method(composed):
    """(4) for `evaluate` after gh#168 W2b: the class defines it, and nothing re-binds it.

    Mirrors test_train_is_the_class_method. (a) `Cs2PuffeRL.evaluate` is the def
    ANCHOR_FUNCTIONS[1] names and is NOT `PuffeRL.evaluate`: a `def evaluate` deleted
    from the class would fall through to the stock rollout, which knows nothing of the
    self-play opponent override, the hybrid aim head or the batch-1 reward processing.
    (b) `evaluate` is not in `vars(trainer)`: a leftover MethodType binding (a
    re-introduced patcher, or a test that pokes one in) would shadow the class method
    and (a) alone would stay green. Read through `type(trainer)`.
    """
    from pufferlib.pufferl import PuffeRL

    from cs2rl.train.trainer import Cs2PuffeRL
    trainer, _ = composed
    cls = type(trainer)
    assert cls.evaluate is Cs2PuffeRL.evaluate and cls.evaluate is not PuffeRL.evaluate, (
        "Cs2PuffeRL no longer defines evaluate(); the stock PuffeRL rollout would run")
    assert Cs2PuffeRL.evaluate.__qualname__ == ANCHOR_FUNCTIONS[1][1], (
        f"Cs2PuffeRL.evaluate is {Cs2PuffeRL.evaluate.__qualname__!r}; ANCHOR_FUNCTIONS[1] "
        f"names {ANCHOR_FUNCTIONS[1][1]!r}")
    assert "evaluate" not in vars(trainer), (
        "evaluate is an INSTANCE attribute again: something re-bound it after construction "
        "and shadows Cs2PuffeRL.evaluate")
    assert trainer.evaluate.__func__ is Cs2PuffeRL.evaluate


def test_a_raise_inside_init_stops_the_utilization_thread(monkeypatch, tmp_path):
    """A setup failure inside ``Cs2PuffeRL.__init__`` must not hang the interpreter.

    ``PuffeRL.__init__`` starts ``Utilization``, a NON-daemon thread whose loop ends only
    when its ``stop()`` is called, and the only production caller of that is
    ``PuffeRL.close()``. A raise after ``super().__init__`` leaves an instance nobody can
    close, so before the fix (PR #259 review, MAJOR-1) pytest printed its summary and
    then sat forever on the thread. The harness's own wrapper cannot help: it never gets
    the half-built trainer. So ``Cs2PuffeRL.__init__`` stops the thread itself on any
    raise, and this test drives that path in-process (no subprocess: a hang would be a
    timeout, not a diagnosis) by making ``_init_selfplay`` raise, after the
    hybrid-aim patch has run.

    Two assertions, both against the interpreter's state rather than the code:
    - every ``Utilization`` thread that exists afterwards has ``stopped`` set (the fix's
      one line), and each one actually ends within its own ``delay`` (1 s) plus slack;
    - the harness scratch dir is gone (``_harness_parts`` and ``_build_trainer_for_test``
      both rmtree on a raise; ``tempfile.tempdir`` is pointed at ``tmp_path`` so the
      check is exact and cannot see another process's scratch dirs).
    The threads are enumerated by TYPE: pufferlib names them "Thread-N", not
    "Utilization". A mutant that deletes the ``stop()`` is red on the first assertion
    immediately (``stopped`` is False) and on the second after the join times out.
    """
    import tempfile
    import threading

    from pufferlib.pufferl import Utilization

    from cs2rl.train import trainer as trainer_mod
    from tests._helpers.trainer_harness import _build_trainer_for_test

    def _boom(self, self_play_mgr):
        raise RuntimeError("simulated patch-time failure")

    monkeypatch.setattr(trainer_mod.Cs2PuffeRL, "_init_selfplay", _boom)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    before = {t for t in threading.enumerate() if isinstance(t, Utilization)}

    with pytest.raises(RuntimeError, match="simulated patch-time failure"):
        _build_trainer_for_test(num_envs=16)

    started = [t for t in threading.enumerate() if isinstance(t, Utilization)]
    started = [t for t in started if t not in before]
    try:
        assert started, (
            "PuffeRL.__init__ did not start a Utilization thread; the test drives nothing")
        assert all(t.stopped for t in started), (
            "Cs2PuffeRL.__init__ raised without stopping the Utilization thread; the "
            "interpreter would hang at exit")
        for t in started:
            t.join(timeout=5.0)
        still_alive = [t.name for t in started if t.is_alive()]
        assert not still_alive, (
            f"Utilization threads still alive after stop()+join: {still_alive}")
        leaked = sorted(p.name for p in tmp_path.glob("cs2rl-harness-*"))
        assert leaked == [], f"harness scratch dirs leaked on the raise path: {leaked}"
    finally:
        # PR #259 review nit: when the assertion under test FAILS (the stop() mutant), the
        # non-daemon thread it complains about would otherwise outlive the test and hang
        # the whole pytest process at exit, turning a red test into a timeout. Stop it
        # here, after the assertions have already recorded the failure.
        for t in started:
            t.stopped = True
