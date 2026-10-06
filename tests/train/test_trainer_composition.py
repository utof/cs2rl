"""Constructor composition and moved-method guards for Cs2PuffeRL.

The stock surface is measured at the end of PuffeRL.__init__. The remaining
constructor surface is compared against self stores in Cs2PuffeRL.__init__,
its called _init_* methods, including hybrid-aim state. Source
scanning includes inactive branches, so an undeclared runtime attribute or
an unexecuted declaration turns the test red.

The independent anchor reads every post-construction Cs2PuffeRL method and the
checkpoint helpers to check that required attributes exist. It cannot require a name
those bodies also overwrite, so the three such names the constructor declares are
pinned by name (DECLARED_AND_OVERWRITTEN). Method identity tests
pin train, evaluate and save_checkpoint to Cs2PuffeRL, with no instance bindings,
and one real rollout checks that evaluate() still feeds every info to the collector.
"""
from __future__ import annotations

import ast

import pytest

from tests.conftest import REPO_ROOT

PACKAGE = REPO_ROOT / "src" / "cs2rl"

# (file, qualname) of the checkpoint helpers whose `trainer.` reads must be backed by
# constructor state; anchor_functions() adds every Cs2PuffeRL method that runs after
# construction.
CHECKPOINT_HELPERS = (
    ("train/resume.py", "collect_train_state"),
    ("train/resume.py", "restore_train_state"),
)
# Required data names over anchor_functions(). gh#92 widened the anchor from five named
# bodies (22 names; the split train/evaluate alone would have left 8) to every
# post-construction method: +_max_entropy and +_ret_device (methods the fixed list
# never read), +_action_mask_view_main (a 3-argument getattr became a direct read),
# -_warmstart_phase (now also stored by an anchored method, _prepare_entropy_update;
# the _WARMSTART_ATTRS check below still requires it on the instance).
EXPECTED_ANCHOR_COUNT = 24
# Read AND overwritten by anchored bodies, so derive_anchor() (reads minus stores) never
# requires them. They were created late and read through getattr defaults until gh#92
# part 3 declared them in the constructor; this list pins those declarations.
DECLARED_AND_OVERWRITTEN = ("_last_nan_warn_t", "_selfplay_used_past", "_tag_metrics")


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

    The receiver is ``self`` in trainer methods or ``trainer`` in checkpoint
    helpers; both names refer to the trainer instance.
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


def anchor_functions():
    """(file, qualname) of every body whose self/trainer reads need constructor state.

    Every def in the Cs2PuffeRL class body except constructor_methods() (``__init__``
    and the ``_init_*`` methods it calls: they DECLARE the state), plus
    CHECKPOINT_HELPERS. The exclusion is by role, not by name prefix: an ``_init_*``
    method ``__init__`` does not call is anchored, and
    test_constructor_methods_run_only_during_construction fails if anything calls an
    excluded method after construction. Derived from the class body, so a new method
    is anchored when it is added.
    """
    tree = ast.parse((PACKAGE / "train" / "trainer.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Cs2PuffeRL")
    skip = constructor_methods()
    methods = tuple(
        ("train/trainer.py", f"Cs2PuffeRL.{n.name}") for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name not in skip)
    return methods + CHECKPOINT_HELPERS


def _anchor_reads_and_stores():
    """({read}, {stored}) trainer attribute names across anchor_functions()."""
    reads, stores = set(), set()
    for rel, qualname in anchor_functions():
        fn = find_def(ast.parse((PACKAGE / rel).read_text()), qualname)
        r, s = _attribute_reads_and_stores(fn)
        reads |= r
        stores |= s
    return reads, stores


def derive_anchor():
    """Names read and never assigned across anchor_functions() (class attrs filtered later)."""
    reads, stores = _anchor_reads_and_stores()
    return reads - stores


@pytest.fixture
def composed(monkeypatch):
    """(trainer, stock_surface): the trainer ``_build_trainer_for_test`` returns.

    ``PuffeRL.__init__`` is wrapped so the stock surface is measured on THIS instance, not
    copied from a list that could drift with a pufferlib bump. The harness builds through
    ``cs2rl.train.compose.build_trainer``, as train() does, so going through it is what
    proves the production class is the one every harness test gets. ``_action_mask_shm``
    (a GC pin build_trainer sets after the constructor) is removed from the measured
    surface so (2) compares construction only.
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
        # Pin the pin: folding a name into the stock set that build_trainer no
        # longer sets would hide its disappearance from (2), so check first.
        assert "_action_mask_shm" in vars(trainer), (
            "build_trainer no longer pins the mask RawArray on the trainer after the "
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


def _constructor_walk():
    """(declared names, constructor methods): ``__init__`` and the ``_init_*`` calls in
    each scanned method's own scope, followed transitively (see _constructor_scope)."""
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
    return names, scanned


def derive_constructor_surface():
    """All names the subclass declares during construction, including inactive branches.

    The constructor and called initializers own every added runtime field.
    """
    return _constructor_walk()[0]


def constructor_methods():
    """Names of the Cs2PuffeRL methods that run during construction (anchor_functions()
    skips them): ``__init__`` and the ``_init_*`` methods it calls, transitively."""
    return _constructor_walk()[1]


def _late_constructor_method_references():
    """(file, line, name) of each reference to a constructor method that is not a
    ``self.<name>(...)`` call in the own scope of a constructor method.

    Scans every ``*.py`` under src/cs2rl and scripts/. A call inside a nested def,
    lambda or class runs later, so it counts; so does a bare reference such as
    ``callback = self._init_x``, since the bound method can be called at any time.
    ``__init__`` itself is left out: every class has one.
    """
    ctor = constructor_methods()
    names = ctor - {"__init__"}
    paths = sorted(PACKAGE.rglob("*.py")) + sorted((REPO_ROOT / "scripts").rglob("*.py"))
    late = []
    for path in paths:
        tree = ast.parse(path.read_text())
        allowed = set()
        if path == PACKAGE / "train" / "trainer.py":
            cls = next(n for n in tree.body
                       if isinstance(n, ast.ClassDef) and n.name == "Cs2PuffeRL")
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name in ctor:
                    allowed |= _own_scope_self_call_ids(fn, names)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in names and id(node) not in allowed:
                late.append((path.relative_to(REPO_ROOT).as_posix(), node.lineno, node.attr))
    return late


def _own_scope_self_call_ids(fn, names):
    """ids of the ``self.<name>`` callee nodes of calls in ``fn``'s own scope."""
    ids = set()

    class Scan(ast.NodeVisitor):

        def visit_FunctionDef(self, node):
            return

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef

        def visit_Call(self, node):
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr in names
                    and isinstance(func.value, ast.Name) and func.value.id == "self"):
                ids.add(id(func))
            self.generic_visit(node)

    scan = Scan()
    for statement in fn.body:
        scan.visit(statement)
    return ids


def test_constructor_methods_run_only_during_construction():
    """anchor_functions() skips the constructor methods because their stores DECLARE the
    state; their reads are not checked. That is sound only while they run during
    construction alone, i.e. while their only callers are own-scope calls from other
    constructor methods. A later call (from train(), a deferred callback, or another
    module) would run their reads against a constructed trainer unanchored.
    """
    ctor = constructor_methods()
    # Known count: __init__ calls _init_return_norm, _init_hybrid_aim and _init_selfplay.
    # A walk that found nothing would make the scan below vacuously green.
    expected = {"__init__", "_init_return_norm", "_init_hybrid_aim", "_init_selfplay"}
    assert ctor == expected, (
        f"constructor methods are now {sorted(ctor)}; check the walk still finds the "
        "initializers, then update this set with the change that moved them")
    late = _late_constructor_method_references()
    assert not late, (
        f"constructor methods referenced outside construction: {late}. anchor_functions() "
        "skips them, so their reads would go unchecked; do the later work in a method "
        "__init__ does not call")


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
        f"or a body gained/lost a read. Update EXPECTED_ANCHOR_COUNT only with the change "
        "that moved the body.")
    absent = anchor - set(vars(trainer))
    assert not absent, f"read by a moved body but never set by __init__: {sorted(absent)}"
    warm_absent = set(_WARMSTART_ATTRS) - set(vars(trainer))
    assert not warm_absent, f"_WARMSTART_ATTRS not on the instance: {sorted(warm_absent)}"


def test_state_the_bodies_read_and_overwrite_is_declared():
    """Every DECLARED_AND_OVERWRITTEN name is constructor state, and is still read and
    overwritten by an anchored body.

    The anchor subtracts every name an anchored body stores, so deleting one of these
    declarations would leave the anchor test green while the body's read raises at run
    time. The first check keeps the list itself honest: a name no anchored body both
    reads and stores any more is stale. Static, so the default tier runs it;
    test_constructed_surface_equals_declared_constructor_surface ties the declared
    surface to the instance.
    """
    names = set(DECLARED_AND_OVERWRITTEN)
    reads, stores = _anchor_reads_and_stores()
    stale = names - (reads & stores)
    assert not stale, f"no anchored body reads and stores {sorted(stale)}; drop them here"
    undeclared = names - derive_constructor_surface()
    assert not undeclared, (
        f"{sorted(undeclared)} are read by a post-construction body but no longer declared "
        "by the constructor")


def test_save_checkpoint_is_the_class_method(composed):
    """The full-state checkpoint body belongs to the subclass and has no instance binding."""
    from pufferlib.pufferl import PuffeRL

    from cs2rl.train.trainer import Cs2PuffeRL
    trainer, _ = composed
    cls = type(trainer)
    assert cls.save_checkpoint is Cs2PuffeRL.save_checkpoint
    assert cls.save_checkpoint is not PuffeRL.save_checkpoint
    assert ("train/trainer.py", Cs2PuffeRL.save_checkpoint.__qualname__) in anchor_functions()
    assert "save_checkpoint" not in vars(trainer)
    assert trainer.save_checkpoint.__func__ is Cs2PuffeRL.save_checkpoint


def test_train_is_the_class_method(composed):
    """(4) for `train` after gh#168 W2a: the class defines it, and nothing re-binds it.

    Two halves, both needed. (a) `Cs2PuffeRL.train` is a def anchor_functions() reads
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
    assert ("train/trainer.py", Cs2PuffeRL.train.__qualname__) in anchor_functions(), (
        f"Cs2PuffeRL.train is {Cs2PuffeRL.train.__qualname__!r}, a def anchor_functions() "
        "does not read")
    assert "train" not in vars(trainer), (
        "train is an INSTANCE attribute again: something re-bound it after construction "
        "and shadows Cs2PuffeRL.train")
    assert trainer.train.__func__ is Cs2PuffeRL.train


def test_evaluate_is_the_class_method(composed):
    """(4) for `evaluate` after gh#168 W2b: the class defines it, and nothing re-binds it.

    Mirrors test_train_is_the_class_method. (a) `Cs2PuffeRL.evaluate` is a def
    anchor_functions() reads and is NOT `PuffeRL.evaluate`: a `def evaluate` deleted
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
    assert ("train/trainer.py", Cs2PuffeRL.evaluate.__qualname__) in anchor_functions(), (
        f"Cs2PuffeRL.evaluate is {Cs2PuffeRL.evaluate.__qualname__!r}, a def "
        "anchor_functions() does not read")
    assert "evaluate" not in vars(trainer), (
        "evaluate is an INSTANCE attribute again: something re-bound it after construction "
        "and shadows Cs2PuffeRL.evaluate")
    assert trainer.evaluate.__func__ is Cs2PuffeRL.evaluate


@pytest.mark.training
def test_evaluate_collects_every_info_into_stats():
    """A real rollout: every info value the env returns lands in ``trainer.stats``.

    ``evaluate()`` hands each chunk's infos to ``_collect_infos``, which appends each
    value to ``stats[key]``; ``mean_and_log`` then means each list, and every
    ``environment/*`` window mean in eval/metrics_schema.py rests on that.
    tests/eval/test_metrics_schema.py pins the collector's shape from its source; this
    test pins that ``evaluate()`` still calls it. The expected counts come from what
    ``vecenv.recv`` returned. In the harness every info carries one ``step_stats`` view
    object (include_step_stats_in_info is on), so each occurrence adds one list entry:
    a dropped call leaves ``stats`` empty, and an assignment in place of the append
    stores the view itself, which has no ``len``.
    """
    import collections

    import pufferlib

    from tests._helpers.trainer_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=4)
    try:
        seen = collections.Counter()
        recv = trainer.vecenv.recv

        def recording_recv():
            out = recv()
            for entry in out[4]:
                seen.update(k for k, _ in pufferlib.unroll_nested_dict(entry))
            return out

        trainer.vecenv.recv = recording_recv
        stats = trainer.evaluate()
        assert seen, "the harness env returned no info; the comparison below would be vacuous"
        counts = {k: len(v) for k, v in stats.items()}
        assert counts == dict(seen), (
            "evaluate() no longer feeds every info value to trainer.stats one entry per "
            "occurrence; every environment/* window mean would be wrong or missing")
    finally:
        cleanup()


def test_collect_infos_extends_lists_appends_scalars_and_drops_arrays():
    """``_collect_infos`` on its own: the default-tier half of the rollout test above.

    PufferLib 3.0's rule, which every ``environment/*`` window mean rests on: a nested
    dict flattens to ``outer/inner`` keys, a list or tuple extends, an ndarray is
    dropped, and any other value appends one entry.
    """
    import collections

    import numpy as np

    from cs2rl.train.trainer import Cs2PuffeRL

    # An instance without __init__: the method reads only `stats`.
    trainer = Cs2PuffeRL.__new__(Cs2PuffeRL)
    trainer.stats = collections.defaultdict(list)
    first = {"kills": 1.0, "hits": [2, 3], "grid": np.zeros(2), "team": {"t": 4}}
    second = {"kills": 5.0, "hits": (6, )}
    trainer._collect_infos([first, second])
    assert dict(trainer.stats) == {"kills": [1.0, 5.0], "hits": [2, 3, 6], "team/t": [4]}


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
    - the harness scratch dir is gone (``_build_trainer_for_test`` rmtrees it on a raise;
      ``tempfile.tempdir`` is pointed at ``tmp_path`` so the check is exact and cannot
      see another process's scratch dirs).
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
