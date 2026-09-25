"""gh#168 W1: ``trainer.Cs2PuffeRL`` composes exactly the trainer ``train()`` used to build.

Three assertions, in the spec's order (.superpowers/sdd/2026-09-24-168-trainer-subclass/spec.md
§W1 test):

1. The class is a direct ``PuffeRL`` subclass, constructed from the harness's parts
   (``_harness_parts``), with the STOCK attribute surface recorded by wrapping
   ``PuffeRL.__init__`` and snapshotting ``vars(self)`` at its end.
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
   silently anchoring on nothing. Deleting a name from BOTH the frozen list and
   ``__init__`` keeps (2) green; (3) catches it. ``_WARMSTART_ATTRS`` is read by
   ``collect_train_state`` through a getattr over a tuple, which the AST walk cannot see,
   so it is asserted as a subset separately.

PITFALL: the anchor count is asserted (18 at W1) as a positive control on the derivation
itself, not as a second frozen list: a walk that silently found zero reads would otherwise
pass (3) vacuously. Each W that moves a body updates the expected count (W2a: 19, W2b: 22).
"""
from __future__ import annotations

import ast
import shutil
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

# Spec §1.1: set(vars(trainer)) after the four patches and _timing, minus set(vars(trainer))
# at the end of PuffeRL.__init__, measured on main. Re-derive with construct_snapshot.py.
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
    "_log_alpha_tensor",
    "_normalize_returns",
    "_participating_rows",
    "_participating_rows_np",
    "_ret_count",
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
    "train",
})
assert len(FROZEN_COMPOSED_SURFACE) == 35

# (file, qualname) of every function that reads trainer state the constructor must have
# declared. W2a re-points the train body to trainer.py::Cs2PuffeRL.train, W2b the evaluate
# body, W2c the save body; the two resume_state helpers stay.
ANCHOR_FUNCTIONS = (
    ("train_update.py", "_patch_trainer_with_return_norm._train_with_return_norm"),
    ("train.py", "_patch_trainer_with_selfplay._evaluate_with_selfplay"),
    ("resume_state.py", "_install_full_checkpointing._save_checkpoint"),
    ("resume_state.py", "collect_train_state"),
    ("resume_state.py", "restore_train_state"),
)
EXPECTED_ANCHOR_COUNT = 18


def _bindings_in_scope(scope, name):
    """Statements in `scope`'s OWN body that bind `name` (def/class, Name target, import).

    Recurses through if/for/while/try/with blocks (same scope) but never into a nested
    def/class body, which is its own scope. Mirrors ast_oracle.find_def in the SDD folder.
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
    """
    node = module
    for part in qualname.split("."):
        hits = _bindings_in_scope(node, part)
        defs = [h for h in hits if isinstance(h, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(hits) != 1 or not defs:
            raise AssertionError(f"{qualname!r}: {part!r} bound {len(hits)} times "
                                 f"(lines {[h.lineno for h in hits]}); expected exactly one def")
        node = defs[0]
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
    """(trainer, stock_surface): a Cs2PuffeRL built from the harness parts.

    ``PuffeRL.__init__`` is wrapped so the stock surface is measured on THIS instance, not
    copied from a list that could drift with a pufferlib bump.
    """
    from pufferlib.pufferl import PuffeRL

    from train_test_harness import _harness_parts
    from trainer import Cs2PuffeRL

    stock = {}
    orig_init = PuffeRL.__init__

    def recording_init(self, *a, **k):
        orig_init(self, *a, **k)
        stock["names"] = set(vars(self))

    monkeypatch.setattr(PuffeRL, "__init__", recording_init)
    parts, pins = _harness_parts(num_envs=16)
    trainer = None
    try:
        trainer = Cs2PuffeRL(**parts)
        yield trainer, stock["names"]
    finally:
        for obj in (trainer, parts["vecenv"]):
            try:
                if obj is not None:
                    obj.close()
            except Exception:          # noqa: BLE001 — best-effort teardown
                pass
        shutil.rmtree(pins["tmp_checkpoint_dir"], ignore_errors=True)
        del pins                       # the mask RawArray outlives the envs until here


def test_direct_pufferl_subclass_built_from_harness_parts(composed):
    from pufferlib.pufferl import PuffeRL
    trainer, stock = composed
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
    from pufferlib.pufferl import PuffeRL

    from train_shared import _WARMSTART_ATTRS
    trainer, stock = composed
    anchor = {n for n in derive_anchor() if n not in stock and not hasattr(PuffeRL, n)}
    assert len(anchor) == EXPECTED_ANCHOR_COUNT, (
        f"derived anchor is {sorted(anchor)} ({len(anchor)} names); the derivation changed "
        f"or a body gained/lost a read. Update EXPECTED_ANCHOR_COUNT only with the W that "
        "moved the body.")
    absent = anchor - set(vars(trainer))
    assert not absent, f"read by a moved body but never set by __init__: {sorted(absent)}"
    warm_absent = set(_WARMSTART_ATTRS) - set(vars(trainer))
    assert not warm_absent, f"_WARMSTART_ATTRS not on the instance: {sorted(warm_absent)}"
