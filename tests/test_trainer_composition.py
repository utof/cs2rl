"""gh#168 W1: ``trainer.Cs2PuffeRL`` composes exactly the trainer ``train()`` used to build.

Four assertions; the first three in the spec's order
(.superpowers/sdd/2026-09-24-168-trainer-subclass/spec.md §W1 test), the fourth from the
review of PR #257:

1. The class is a direct ``PuffeRL`` subclass, and (W1.5) it is what the harness's
   ``_build_trainer_for_test`` returns: ``type(trainer) is Cs2PuffeRL``, so no harness
   test can run stock ``PuffeRL.train`` or skip full checkpointing (gh#169). The STOCK
   attribute surface is recorded by wrapping ``PuffeRL.__init__`` and snapshotting
   ``vars(self)`` at its end.
2. The attributes the constructor adds ON TOP of stock equal a FROZEN list (spec §1.1: the
   35 names the four patch functions plus ``_timing`` set at 2a3573f). This is the
   declaration-versus-runtime pin: a patch call dropped from ``__init__`` (knock-out W1-K1)
   makes this red naming the missing buffers. The list changes only when a W says so
   (W2a: -train -_normalize_returns +_entropy_floor +_ret_device; W2b: -evaluate
   +_self_play_mgr +_past_lstm_h +_past_lstm_c; W2c: -save_checkpoint; after W2c the list
   becomes a derivation from ``__init__``'s ``self.<name>`` store targets).
3. An EXTERNAL anchor, derived rather than listed: every attribute the moved bodies and the
   checkpoint helpers READ through ``self.``/``trainer.`` (or a 2-argument ``getattr``) and
   never assign must be present on the instance. The functions are located by qualname
   with an exactly-once binding rule, so a renamed or duplicated def raises instead of
   silently anchoring on nothing. WHAT (3) COVERS, measured at W1: the 18 DATA attributes
   the moved bodies read; 19 since W2a, when the closure local ``entropy_floor`` became
   the attribute ``_entropy_floor`` that ``Cs2PuffeRL.train`` reads (knock-out W2a-K1:
   deleting its store in ``_init_return_norm`` is red here by name, in (2), and in the
   warm-start tests). Deleting one of those from BOTH the frozen list and ``__init__``
   keeps (2) green and (3) catches it. It does NOT cover the other 16 names: the two
   remaining method aliases (``evaluate``/``save_checkpoint`` are class attributes of
   PuffeRL, filtered out; ``train`` and ``_normalize_returns`` are Cs2PuffeRL's own
   methods since W2a and are filtered the same way), and every read-before-write or
   never-read name (``_timing``, ``_cont_action_view_main``, ``_action_mask_view_main``,
   ``_ret_device`` (read by ``_update_return_stats``, not an anchor function) and the
   ``_batch1_*`` names the bodies assign somewhere; some, like ``_batch1_reward_scratch``
   and ``_batch1_log_alpha_reset_done``, are read first). Those are pinned by (2) only,
   plus (4) for the aliases. Measured on PR #257's first commit: dropping
   ``_install_full_checkpointing`` from ``__init__`` AND ``save_checkpoint`` from the list
   left all three green; so did dropping ``self._timing`` AND ``_timing``. With (4) the
   first mutant is red naming ``save_checkpoint``; the second still passes, by design:
   ``_timing`` has no reader among the moved bodies (train()'s loop reads it), so the
   frozen list is its only pin here.
   ``_WARMSTART_ATTRS`` is read by ``collect_train_state`` through a getattr over a tuple,
   which the AST walk cannot see, so it is asserted as a subset separately.
4. Method identity, independent of the frozen list: each of ``train``, ``evaluate``,
   ``save_checkpoint`` is an INSTANCE attribute whose bound function is the closure body
   named in ANCHOR_FUNCTIONS. A dropped patch call (or a stock method left in place) is
   red here by name even after the list is edited to match. W2a (gh#168) turned ``train``
   into a class method and re-pointed its pin (test_train_is_the_class_method: the class
   defines it, it is not ``PuffeRL.train``, and no instance binding shadows it); W2b/W2c
   do the same for ``evaluate`` and ``save_checkpoint``.

PITFALLS
- The anchor is PRESENCE-only: ``self._ret_count = None`` in ``__init__`` passes (3). Values
  are the byte gates' and construct_snapshot.py's job, not this file's.
- The anchor count is asserted (18 at W1, 19 at W2a) as a positive control on the derivation itself,
  not as a second frozen list: a walk that silently found zero reads would otherwise pass
  (3) vacuously. Each W that moves a body re-derives the count and updates it here.
"""
from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

# Spec §1.1: set(vars(trainer)) after the four patches and _timing, minus set(vars(trainer))
# at the end of PuffeRL.__init__, measured on main at 2a3573f, then edited per W. W2a (gh#168):
# `train` and `_normalize_returns` are class methods now (no instance binding), and the two
# closure locals train() reads became attributes: `_entropy_floor` (the 0.3·max_entropy
# warm-start floor) and `_ret_device` (the device the Welford tensors live on). Re-derive
# with construct_snapshot.py; its --expect-only-a/-b flags name exactly this delta.
FROZEN_COMPOSED_SURFACE = frozenset({
    "_action_mask_view_main",
    "_alpha_optimizer",
    "_batch1_current_segment_has_event",
    "_batch1_current_target_entropy",
    "_batch1_effective_alpha",
    "_batch1_event_mask",
    "_batch1_grad_norm",
    "_batch1_last_entropy_mean",
    "_batch1_log_alpha_reset_done",
    "_batch1_max_entropy",
    "_batch1_reward_scratch",
    "_batch1_warmstart_h0",
    "_batch1_warmstart_h_anchor",
    "_batch1_warmstart_phase",
    "_batch1_warmstart_warn_epoch",
    "_batch1_welford_combat",
    "_batch1_welford_objective",
    "_batch1_welford_positional",
    "_cont_action_view_main",
    "_entropy_floor",
    "_log_alpha_tensor",
    "_participating_rows",
    "_participating_rows_np",
    "_ret_count",
    "_ret_device",
    "_ret_mean",
    "_ret_var",
    "_timing",
    "action_masks",
    "cont_actions",
    "evaluate",
    "logprobs_c",
    "logprobs_d",
    "participating",
    "save_checkpoint",
})
assert len(FROZEN_COMPOSED_SURFACE) == 35

# (file, qualname) of every function that reads trainer state the constructor must have
# declared. W2a re-pointed the train body to trainer.py::Cs2PuffeRL.train (gh#168); W2b
# re-points the evaluate body, W2c the save body; the two resume_state helpers stay.
ANCHOR_FUNCTIONS = (
    ("trainer.py", "Cs2PuffeRL.train"),
    ("train.py", "_patch_trainer_with_selfplay._evaluate_with_selfplay"),
    ("resume_state.py", "_install_full_checkpointing._save_checkpoint"),
    ("resume_state.py", "collect_train_state"),
    ("resume_state.py", "restore_train_state"),
)
# 18 at W1. W2a: 19 — `_entropy_floor` was a closure local and is now read as
# `self._entropy_floor`; `_ret_device` is read only by `_update_return_stats`, which is
# not an anchor function, so it is pinned by the frozen list and construct_snapshot only.
EXPECTED_ANCHOR_COUNT = 19

# Assertion (4): instance method alias -> the closure body it must be bound to (the same
# qualnames as ANCHOR_FUNCTIONS, so the two cannot drift apart). Python spells a nested
# def's __qualname__ as `outer.<locals>.inner`; the `<locals>` is dropped before comparing.
# W2a (gh#168) took `train` out: it is a method of the class now, pinned by
# test_train_is_the_class_method below (the same ANCHOR_FUNCTIONS[0] qualname, checked on
# the class rather than in vars(trainer)). W2b/W2c will move the other two the same way.
METHOD_ALIASES = {
    "evaluate": ANCHOR_FUNCTIONS[1][1],
    "save_checkpoint": ANCHOR_FUNCTIONS[2][1],
}


def _bindings_in_scope(scope, name):
    """Statements in `scope`'s OWN body that bind `name` (def/class, Name target, import).

    Recurses through if/for/while/try/with blocks (same scope) but never into a nested
    def/class body, which is its own scope. Same rule as ast_oracle._scope_bindings in
    the SDD folder.
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

    walk(scope.body)
    return hits


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
        fn = find_def(ast.parse((SRC / rel).read_text()), qualname)
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

    from train_test_harness import _build_trainer_for_test

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

    from trainer import Cs2PuffeRL
    trainer, stock = composed
    assert type(trainer) is Cs2PuffeRL, (
        f"_build_trainer_for_test returned a {type(trainer).__name__}; since gh#168 W1.5 the "
        "harness must construct the production class (gh#169)")
    assert type(trainer).__mro__[1] is PuffeRL
    assert stock, "PuffeRL.__init__ did not run through the recording wrapper"


def test_constructed_surface_equals_the_frozen_list(composed):
    trainer, stock = composed
    composed_surface = set(vars(trainer)) - stock
    missing = FROZEN_COMPOSED_SURFACE - composed_surface
    extra = composed_surface - FROZEN_COMPOSED_SURFACE
    assert not missing and not extra, (
        f"Cs2PuffeRL.__init__ no longer composes the 2a3573f surface: "
        f"missing {sorted(missing)}, unexpected {sorted(extra)}")


def test_every_attribute_the_bodies_read_is_declared(composed):
    from train_shared import _WARMSTART_ATTRS
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


def test_method_aliases_are_bound_to_the_closure_bodies(composed):
    """(4): the still-patched methods are instance attributes bound to the named closures.

    Independent of FROZEN_COMPOSED_SURFACE on purpose: a patch call dropped from __init__
    together with its name from the list keeps (2) green and, for these names, (3) too
    (they are PuffeRL class attributes, which the anchor excludes). Here the name must be
    in `vars(trainer)` (stock `PuffeRL.evaluate` is a class attribute, so a missing patch
    shows as "not an instance attribute") and its `__func__.__qualname__` must be the
    closure body ANCHOR_FUNCTIONS names. `train` left this loop at W2a; see
    test_train_is_the_class_method.
    """
    trainer, _ = composed
    for name, qualname in METHOD_ALIASES.items():
        assert name in vars(trainer), (
            f"{name!r} is not an instance attribute: the patch that replaces it was not applied, "
            f"so stock PuffeRL.{name} would run")
        bound = vars(trainer)[name]
        assert isinstance(bound, types.MethodType), (
            f"{name!r} is a {type(bound).__name__}, not a bound method: the patch binds the "
            "closure with types.MethodType")
        got = bound.__func__.__qualname__.replace(".<locals>.", ".")
        assert got == qualname, f"{name!r} is bound to {got!r}, expected {qualname!r}"


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

    from trainer import Cs2PuffeRL
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


def test_a_raise_inside_init_stops_the_utilization_thread(monkeypatch, tmp_path):
    """A patch that raises inside ``Cs2PuffeRL.__init__`` must not hang the interpreter.

    ``PuffeRL.__init__`` starts ``Utilization``, a NON-daemon thread whose loop ends only
    when its ``stop()`` is called, and the only production caller of that is
    ``PuffeRL.close()``. A raise after ``super().__init__`` leaves an instance nobody can
    close, so before the fix (PR #259 review, MAJOR-1) pytest printed its summary and
    then sat forever on the thread. The harness's own wrapper cannot help: it never gets
    the half-built trainer. So ``Cs2PuffeRL.__init__`` stops the thread itself on any
    raise, and this test drives that path in-process (no subprocess: a hang would be a
    timeout, not a diagnosis) by making the LAST patch raise, so every earlier patch has
    run and the thread has been alive the longest.

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

    import trainer as trainer_mod
    from train_test_harness import _build_trainer_for_test

    def _boom(self, self_play_mgr):
        raise RuntimeError("simulated patch-time failure")

    monkeypatch.setattr(trainer_mod, "_install_full_checkpointing", _boom)
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
