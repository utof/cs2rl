"""The cs2rl trainer class: ``PuffeRL`` plus the behaviours ``train()`` used to bolt on.

WHAT: ``Cs2PuffeRL`` is the trainer ``train()`` builds. Before gh#168 W1, ``train()``
constructed a stock ``PuffeRL`` and then mutated the INSTANCE four times: replace
``train`` (return-norm), extend the rollout buffers + wrap ``vecenv.send`` (hybrid aim),
replace ``evaluate`` (self-play), replace ``save_checkpoint`` (full checkpointing), plus a
``_timing`` dict. ADR 0002 (docs/adr/0002-subclass-pufferl-do-not-mutate.md, a LOCAL file:
docs/ is under .git/info/exclude and is in no clone; the decision is restated in gh#168 and
in the spec at .superpowers/sdd/2026-09-24-168-trainer-subclass/spec.md) says that
composition belongs in a subclass. W1 moves ONLY the composition here: ``__init__`` still
calls the same four patch functions, in ``train()``'s order, so an instance is
attribute-for-attribute the trainer ``train()`` built before (the byte gates and the
construction snapshot in .superpowers/sdd/2026-09-24-168-trainer-subclass/ pin that).
The bodies move in W2a-c and the vecenv plumbing in W3.

WHY a module of its own and not a class inside train.py: this module subclasses
``PuffeRL``, so it imports torch and pufferlib at module scope and is HEAVY by
construction. ``import train`` must stay torch-free (tests/test_w1_modules.py: it is what
keeps ``--dump-config`` at ~1 s), so train.py imports this module function-locally, inside
``train()``. train_test_harness.py (gh#168 W1.5) imports it the same way, function-locally
inside ``_build_trainer_for_test``, so ``import train_test_harness`` stays as light as
``import train``; tests/test_trainer_composition.py imports it inside a fixture. Never add
``from trainer import ...`` at train.py's module level (knock-out W1-K3 in the spec:
test_import_train_stays_light_and_really_imports_the_shims goes red: with the import next
to the other module-level imports it is a circular-import ImportError, after all defs it
is the guard naming torch).

IMPORT DIRECTION: this module imports ``train`` at module scope; ``train`` imports this
module only inside ``train()``. That is acyclic at import time: by the time ``train()``
runs, ``train`` is fully initialised. When src/train.py runs as a script, its main block
aliases ``sys.modules["train"]`` to ``__main__`` as its first statement, before ``train()``
is called, so ``from train import ...`` here does not re-execute train.py.
"""

from __future__ import annotations

# Heavy by construction (module docstring): pufferl imports torch at ITS module scope.
from pufferlib.pufferl import PuffeRL

from resume_state import _install_full_checkpointing
from train import _patch_trainer_with_hybrid_aim, _patch_trainer_with_selfplay
from train_update import _patch_trainer_with_return_norm


class Cs2PuffeRL(PuffeRL):
    """The cs2rl trainer: PuffeRL plus the behaviours train() used to bolt on after construction.

    W1 (gh#168) moves the COMPOSITION here and nothing else: __init__ calls the four existing patch
    functions in train()'s order, so an instance is attribute-for-attribute the trainer train()
    built at 2a3573f. Bodies move in W2a-c, vecenv plumbing in W3 (ADR 0002).

    Parameters beyond PuffeRL's ``(config, vecenv, policy, logger=None)`` are exactly what the
    patch functions took at train()'s call sites:

    cont_action_view_main : np.ndarray or None
        Main-process view of the continuous-action shared array. train() allocates it and
        passes it on BOTH backends (src/train.py builds `_cont_action_view_main` before the
        vecenv, unconditionally); only the harness passes None, and the hybrid-aim patcher
        takes None to mean "nothing to forward".
    mask_view_main : np.ndarray
        Main-process view of the action-mask shared array (F8).
    participating_rows : np.ndarray
        Static per-run participation vector (Rung 0 §2.2), built by the caller with
        ``build_participating_rows`` BEFORE construction, exactly as train() does.
    self_play_mgr : SelfPlayManager
        Built by the caller (``build_selfplay_manager``) and, on a resume, pre-seeded BEFORE
        construction. The pool is read only at evaluate() time, so seeding after construction
        would also work today; the order is kept to match train() byte for byte.

    PITFALLS
    - The order of the four patch calls is the order train() applied them. The only
      load-bearing constraint is "all four before the first evaluate()/train() call"
      (spec §1); knock-out W1-K2 measured that swapping selfplay and checkpointing leaves
      both byte gates identical.
    - ``_timing`` is created here because train()'s loop assigns INTO it and the [Timing]
      print reads it; it is not a patch (#166 deleted the timing patch).
    - Everything a caller used to set on the instance AFTER construction (``logger.run_id``,
      ``weight_decay``, the aim-σ param group, the shm GC pins) still happens in train(),
      after this constructor returns: none of it is read at patch time (spec §W1 table).
    """

    def __init__(self,
                 config,
                 vecenv,
                 policy,
                 *,
                 cont_action_view_main,
                 mask_view_main,
                 participating_rows,
                 self_play_mgr,
                 logger=None):
        super().__init__(config, vecenv, policy, logger=logger)
        # WHAT: everything after super().__init__ runs under one try that stops the
        # Utilization thread on ANY raise, then re-raises.
        # WHY: PuffeRL.__init__ starts ``self.utilization = Utilization()``, a
        # NON-daemon threading.Thread whose loop only ends when its stop() is called,
        # and the only production caller of that is PuffeRL.close(). A raise from a
        # patch function (an assert on a bad view shape, a config the patcher
        # refuses) leaves a half-built instance nobody can close(), so the thread
        # keeps the interpreter alive at exit: pytest prints its summary and then
        # hangs (gh#168 W1.5 review, MAJOR-1; pinned by
        # tests/test_trainer_composition.py::test_a_raise_inside_init_stops_the_utilization_thread).
        # PITFALLS: this is the ONE line of PuffeRL.close() that must run on the
        # failure path; vecenv.close() is the caller's (the vecenv was theirs before
        # this constructor), and save_checkpoint() would write a half-built trainer.
        # ``except BaseException`` so KeyboardInterrupt mid-construction does not
        # hang either. Never move the try above super().__init__: before it returns
        # there is no ``self.utilization`` to stop (AttributeError would replace the
        # real error). Keep the patch order below unchanged (the class docstring).
        try:
            _patch_trainer_with_return_norm(self)
            _patch_trainer_with_hybrid_aim(self,
                                           cont_action_view_main=cont_action_view_main,
                                           mask_view_main=mask_view_main,
                                           participating_rows=participating_rows)
            _patch_trainer_with_selfplay(self, self_play_mgr)
            # Per-epoch wall-clock, measured at the evaluate()/train() call sites in
            # train()'s loop (#166 replaced a monkey-patch that wrapped both methods; the
            # patch had to be installed LAST so selfplay could not shadow it, which made
            # patch order load-bearing for a measurement). The dict is created here
            # because the loop assigns INTO it and the [Timing] print reads it, so it
            # has to exist before the first epoch.
            self._timing = {"collect_ms": 0.0, "update_ms": 0.0}
            # R0-C (#134): full-state checkpointing. Last here because train() applied it
            # last at 2a3573f, not because anything depends on it: the installer reads no
            # trainer attribute at patch time, and knock-out W1-K2 (installed BEFORE the
            # selfplay patch) reproduced every hash of both byte gates.
            _install_full_checkpointing(self, self_play_mgr)
        except BaseException:
            # Exactly what PuffeRL.close() does to the thread; nothing else of close().
            self.utilization.stop()
            raise
