"""The cs2rl trainer class: ``PuffeRL`` plus the behaviours ``train()`` used to bolt on.

WHAT: ``Cs2PuffeRL`` is the trainer ``train()`` builds. Before gh#168 W1, ``train()``
constructed a stock ``PuffeRL`` and then mutated the INSTANCE four times: replace
``train`` (return-norm), extend the rollout buffers + wrap ``vecenv.send`` (hybrid aim),
replace ``evaluate`` (self-play), replace ``save_checkpoint`` (full checkpointing), plus a
``_timing`` dict. ADR 0002 (docs/adr/0002-subclass-pufferl-do-not-mutate.md, a LOCAL file:
docs/ is under .git/info/exclude and is in no clone; the decision is restated in gh#168 and
in the spec at .superpowers/sdd/2026-09-24-168-trainer-subclass/spec.md) says that
composition belongs in a subclass. W1 moved ONLY the composition here: ``__init__`` called
the same four patch functions, in ``train()``'s order, so an instance is
attribute-for-attribute the trainer ``train()`` built before (the byte gates and the
construction snapshot in .superpowers/sdd/2026-09-24-168-trainer-subclass/ pin that).
W2a (gh#168) folded the first of the four in: ``_init_return_norm`` seeds the return-norm
state and ``train`` / ``_normalize_returns`` / ``_update_return_stats`` are methods, so
W2b folded the self-play patcher: ``_init_selfplay`` stores the manager, the past
policy's LSTM state and the ``_batch1_*`` reward state, and ``evaluate`` is a method, so
``__init__`` calls ``_init_return_norm``, ``_init_hybrid_aim``, and
``_init_selfplay``; callers install ``HybridAimVecEnv`` first. The checkpoint body is
``save_checkpoint`` on the class. Construction keeps the state setup order.

WHY a module of its own and not a class inside train.py: this module subclasses
``PuffeRL``, so it imports torch and pufferlib at module scope and is HEAVY by
construction. ``from cs2rl import train`` must stay torch-free (tests/test_w1_modules.py: it is what
keeps ``--dump-config`` at ~1 s), so train.py imports this module function-locally, inside
``train()``. train_test_harness.py (gh#168 W1.5) imports it the same way, function-locally
inside ``_build_trainer_for_test``, so ``from cs2rl import train_test_harness`` stays as light
as ``from cs2rl import train``; tests/test_trainer_composition.py imports it inside a fixture. Never add
``from cs2rl.trainer import ...`` at train.py's module level (knock-out W1-K3 in the spec:
test_import_train_stays_light_and_really_imports_the_shims goes red: with the import next
to the other module-level imports it is a circular-import ImportError, after all defs it
is the guard naming torch).

IMPORT DIRECTION: this module imports ``train`` at module scope; ``train`` imports this
module only inside ``train()``. That is acyclic at import time: by the time ``train()``
runs, ``train`` is fully initialised. When train.py runs as ``python -m cs2rl.train``, its
main block aliases ``sys.modules["cs2rl.train"]`` to ``__main__`` as its first statement,
before ``train()`` is called, so ``from cs2rl.train import ...`` here does not re-execute
train.py.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

# Heavy by construction (module docstring): pufferl imports torch at ITS module scope.
# `pufferlib` is read by the self-play evaluate body (gh#168 W2b) at exactly one site,
# `pufferlib.unroll_nested_dict`. `import pufferlib.pytorch` has NO use in this module:
# the old closure imported both forms and never touched `.pytorch`, and ast_oracle.py
# check S2 passes only if this module binds `pufferlib` through every import form the
# old closure used (extra forms are fine) or re-imports the name from train.py. Measured
# in the #262 fold: without this line O1 fails S2 on `pufferlib`. Carried for that check
# only; it can go once `--moves evaluate` is no longer run as a gate.
import pufferlib
import pufferlib.pytorch
import torch
from pufferlib.pufferl import PuffeRL, compute_puff_advantage

from cs2rl._action_spec import ACTION_HEAD_NAMES, ACTION_HEAD_SIZES, ACTION_MASK_DIM, AIM_DIM
from cs2rl.resume_state import collect_train_state
from cs2rl.train import _hybrid_sample_logits

# W2a (gh#168): everything train() reads that used to be a function-local import of
# the patcher, or a global of train_update.py, is a module-level import HERE, of the
# same object from its defining module. ast_oracle.py check S2 fails on a shadowing
# definition or a missing import; tests/test_tag_trainer.py patches tag_grad_cossim
# on THIS module because the body resolves it through these globals. W2b added
# `WelfordStd` / `process_step_rewards` (the self-play state and evaluate body) and
# `_hybrid_sample_logits` above, under the same rule.
from cs2rl.train_helpers_batch1 import (
    WS_GRACE,
    WS_OFF,
    WelfordStd,
    process_step_rewards,
    warmstart_entropy_state,
)
from cs2rl.train_shared import LOG_STD_MAX, _atomic_save_state_dict
from cs2rl.train_update import (
    _hybrid_ppo_loss,
    _scheduled_target_entropy,
    masked_explained_variance,
    masked_mean,
    tag_grad_cossim,
)


class HybridAimVecEnv:
    """Carry continuous aim beside discrete actions through Serial or shared MP views.

    PuffeRL sees the vector interface through delegation. Serial calls each env's
    step with its matching agent rows; MP workers consume the shared view written
    before the backend is sent its discrete actions.
    """

    def __init__(self, backend, cont_action_view_main=None):
        self._backend = backend
        self._cont_action_view_main = cont_action_view_main
        self._cont_action_buf = None
        if hasattr(backend, "envs"):
            agents_per_env = backend.driver_env.num_agents
            for env_idx, env in enumerate(backend.envs):
                row_start = env_idx * agents_per_env
                row_end = row_start + agents_per_env
                orig_step = env.step

                def hybrid_step(actions, _orig=orig_step, _start=row_start, _end=row_end):
                    """Forward only this env's action rows to the original step."""
                    cont = self._cont_action_buf
                    if cont is not None:
                        cont = cont[_start:_end]
                    return _orig(actions, continuous_actions=cont)

                env.step = hybrid_step

    def __getattr__(self, name):
        """Expose the backend's vector properties and methods to PuffeRL."""
        return getattr(self._backend, name)

    def send(self, action_pair):
        """Mirror float aim before dispatch; a bare action clears stale aim."""
        if isinstance(action_pair, tuple):
            action, cont_action = action_pair
        else:
            action, cont_action = action_pair, None
        if cont_action is not None and hasattr(cont_action, "cpu"):
            cont_action = cont_action.cpu().numpy().astype(np.float32, copy=False)
        self._cont_action_buf = cont_action
        view = self._cont_action_view_main
        if view is not None:
            if cont_action is None:
                view.fill(0.0)
            else:
                assert cont_action.size == view.size, (
                    f"cont_action.size={cont_action.size} but "
                    f"view.size={view.size} (view.shape={view.shape})")
                view[:] = cont_action.reshape(view.shape)
        return self._backend.send(action)


class Cs2PuffeRL(PuffeRL):
    """The cs2rl trainer: PuffeRL plus the behaviours train() used to bolt on after construction.

    W1 (gh#168) moved the COMPOSITION here: __init__ runs the same steps train() used to, in
    train()'s order, so an instance is attribute-for-attribute the trainer train() built at
    2a3573f. W2a moved the return-norm step in as well: ``_init_return_norm`` (state) and
    ``train`` / ``_normalize_returns`` / ``_update_return_stats`` (bodies) are methods now, and
    the return-norm patcher is gone from train_update.py. The self-play evaluate body (W2b), the
    checkpoint override is also a class method (W2c); hybrid action transport
    lives in the vecenv wrapper (W3).

    Parameters beyond PuffeRL's ``(config, vecenv, policy, logger=None)`` are exactly what the
    patch functions took at train()'s call sites:

    cont_action_view_main : np.ndarray or None
        Main-process view of the continuous-action shared array. train() allocates it and
        passes it on BOTH backends (src/cs2rl/train.py builds `_cont_action_view_main` before the
        vecenv, unconditionally); only the harness passes None.
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
    - The order of state setup matches train()'s former patch order. All state
      must exist before the first evaluate()/train()/save_checkpoint() call.
    - ``_timing`` is created here because train()'s loop assigns INTO it and the [Timing]
      print reads it; it is not a patch (#166 deleted the timing patch).
    - Everything a caller used to set on the instance AFTER construction (``logger.run_id``,
      ``weight_decay``, the aim-σ param group, the shm GC pins) still happens in train(),
      after this constructor returns: none of it is read at patch time (spec §W1 table).
    """

    # PuffeRL owns the stock rollout buffers and recurrent-state maps; their
    # indexed writes in evaluate use integer indices narrowed at that boundary.
    actions: torch.Tensor
    logprobs: torch.Tensor
    observations: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    values: torch.Tensor
    lstm_h: dict[int, torch.Tensor]
    lstm_c: dict[int, torch.Tensor]
    # PuffeRL may replace the concrete policy with torch.compile; its base
    # attribute inference is FunctionType, so retain the dynamic policy boundary.
    policy: Any
    # Hybrid tensors are allocated by _init_hybrid_aim; annotations do not
    # initialize state or change the constructor's attribute surface.
    action_masks: torch.Tensor
    cont_actions: torch.Tensor
    logprobs_c: torch.Tensor
    logprobs_d: torch.Tensor
    participating: torch.Tensor
    _tag_metrics: dict | None

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
        # step below (the gh#85 segments assert, a bad view shape, a config the
        # patcher refuses) leaves a half-built instance nobody can close(), so the thread
        # keeps the interpreter alive at exit: pytest prints its summary and then
        # hangs (gh#168 W1.5 review, MAJOR-1; pinned by
        # tests/test_trainer_composition.py::test_a_raise_inside_init_stops_the_utilization_thread).
        # PITFALLS: this is the ONE line of PuffeRL.close() that must run on the
        # failure path; vecenv.close() is the caller's (the vecenv was theirs before
        # this constructor), and save_checkpoint() would write a half-built trainer.
        # ``except BaseException`` so KeyboardInterrupt mid-construction does not
        # hang either. Never move the try above super().__init__: before it returns
        # there is no ``self.utilization`` to stop (AttributeError would replace the
        # real error). Keep the step order below unchanged (the class docstring).
        try:
            self._init_return_norm()
            self._init_hybrid_aim(cont_action_view_main, mask_view_main, participating_rows)
            self._init_selfplay(self_play_mgr)
            # Per-epoch wall-clock, measured at the evaluate()/train() call sites in
            # train()'s loop (#166 replaced a monkey-patch that wrapped both methods; the
            # patch had to be installed LAST so selfplay could not shadow it, which made
            # patch order load-bearing for a measurement). The dict is created here
            # because the loop assigns INTO it and the [Timing] print reads it, so it
            # has to exist before the first epoch.
            self._timing = {"collect_ms": 0.0, "update_ms": 0.0}
        except BaseException:
            # Exactly what PuffeRL.close() does to the thread; nothing else of close().
            self.utilization.stop()
            raise

    def _init_hybrid_aim(self, cont_action_view_main, mask_view_main, participating_rows):
        """Allocate the nine hybrid rollout fields after stock PuffeRL buffers exist.

        The view aliases the shared array used by the vecenv wrapper. This setup
        keeps construction state on the class; action transport belongs to the
        wrapper, before the trainer is constructed.
        """
        self._cont_action_view_main = cont_action_view_main
        # F8: main-process numpy view over the mask shm (env→trainer direction;
        # see Cs2Env._attach_mask_view). Cs2PuffeRL.evaluate reads rows for
        # the recv'd env_id slice right after recv() — the workers finished their
        # step by then, so the bytes are the masks for the obs batch in hand.
        # None ⇒ rollout runs unmasked (legacy callers without shm plumbing) and
        # action_masks stays all-ones, which makes the update path a no-op mask.
        self._action_mask_view_main = mask_view_main

        # ── Rollout buffer extension (step 5.4) ──
        # self.actions has shape (segments, bptt_horizon, ACTION_DIM=7) int32 —
        # we mirror with AIM_DIM trailing dim, float32. self.logprobs is
        # (segments, bptt_horizon) float32; we add per-factor halves with the
        # same shape so the caller can fetch self.logprobs_d[idx] etc. without
        # any reshaping.
        self.cont_actions = torch.zeros(
            (*self.actions.shape[:-1], AIM_DIM),
            dtype=torch.float32,
            device=self.actions.device,
        )
        self.logprobs_d = torch.zeros_like(self.logprobs)
        self.logprobs_c = torch.zeros_like(self.logprobs)
        # F8: per-step action masks, parallel to actions but ACTION_MASK_DIM wide.
        # Initialised to ONES (= everything valid): rows never written (mask shm
        # absent, or rollout rounds that don't fill every segment) degrade to the
        # exact pre-F8 unmasked behaviour instead of masking everything to the
        # no-op. bool keeps the buffer small (segments × 64 × 22 bytes).
        self.action_masks = torch.ones(
            (*self.actions.shape[:-1], ACTION_MASK_DIM),
            dtype=torch.bool,
            device=self.actions.device,
        )

        # ── Rung 0 §2.2: per-row participation ───────────────────────────────
        # participating_rows: numpy bool (total_agents,) — STATIC per run, derived
        # from args.n_active_per_team in train(). None ⇒ all rows participate
        # (harness default, exact identity with pre-Rung-0 behaviour).
        # self.participating is the BUFFER-LAYOUT flag [segments, bptt],
        # scattered by evaluate() via ep_indices exactly like action_masks.
        # ZERO-initialised (unlike action_masks, which defaults to all-ones): a
        # skipped write must mask EVERYTHING and trip the per-epoch
        # `participating.any()` assert in train(), never silently train on parked
        # rows. _participating_rows_np is kept beside the torch copy because
        # evaluate()'s global_step accounting works on the numpy `mask` recv()
        # returns; converting per-recv would allocate on every rollout tick.
        n_rows = self.total_agents
        if participating_rows is None:
            participating_rows = np.ones(n_rows, dtype=bool)
        participating_rows = np.asarray(participating_rows, dtype=bool).reshape(-1)
        assert participating_rows.shape == (n_rows, ), (participating_rows.shape, n_rows)
        assert participating_rows.any(), "no participating rows — n_active_per_team=0?"
        self._participating_rows_np = participating_rows
        self._participating_rows = torch.as_tensor(participating_rows, device=self.actions.device)
        self.participating = torch.zeros(self.actions.shape[:-1],
                                         dtype=torch.bool,
                                         device=self.actions.device)
        print("[Train] Hybrid-aim trainer patch enabled "
              f"(cont_actions buffer={self.cont_actions.shape}, "
              f"vecenv_kind={type(self.vecenv._backend).__name__}).")

    def _init_return_norm(self):
        """Return-normalisation + adaptive-entropy state; was the patch-time body of
        the return-norm patcher (src/cs2rl/train_update.py) until gh#168 W2a.

        WHAT: asserts the BPTT segment invariant (gh#85), creates the Welford running
        stats (``_ret_mean/_ret_var/_ret_count``), the SAC-style ``_log_alpha_tensor`` +
        ``_alpha_optimizer``, the entropy ceiling/floor and the Task 9/warm-start
        bookkeeping attributes that ``train()`` reads and writes.

        WHY a method and not inline in ``__init__``: it is ~130 lines of state setup with
        its own history; ``__init__`` stays the readable composition order. The values
        it produces under every config arm are pinned by
        .superpowers/sdd/2026-09-24-168-trainer-subclass/construct_snapshot.py (O4).

        PITFALLS: two names that were closure variables before W2a are now attributes,
        ``_entropy_floor`` and ``_ret_device``; ``tests/test_trainer_composition.py``'s
        derived constructor-surface test and the O4 snapshot both expect them. Everything else is set on
        ``self`` exactly as the patcher set it on ``trainer`` (relocation, not a rewrite).
        """
        # gh #85: BPTT zero-init exactness (Dust2Policy.forward/_lstm_bptt) is only
        # EXACT when each agent row fills exactly one buffer segment per evaluate(),
        # i.e. segments == total_agents. Upstream PuffeRL only enforces
        # total_agents <= segments (pufferl.py:83-86); our equality holds by
        # construction in compute_batch_dims but nothing asserted it — one
        # batch_size/bptt_horizon config edit away from silently-biased importance
        # ratios. Fail loudly at construction instead.
        assert self.segments == self.total_agents, (
            f"segments ({self.segments}) != total_agents ({self.total_agents}): "
            "BPTT zero-init exactness broken — revisit batch_size/bptt_horizon "
            "(compute_batch_dims) or the _lstm_bptt initial-state design. See gh #85.")

        # Running stats for return normalization (Welford-style, torch tensors).
        # W2a (gh#168): `device` used to be a closure variable of _update_return_stats;
        # it is kept on the instance as `_ret_device` so that method can build its
        # batch_count tensor on the same device the stats live on.
        device = self.config["device"]
        self._ret_device = device
        _ret_mean = torch.zeros(1, device=device)
        _ret_var = torch.ones(1, device=device)
        _ret_count = torch.zeros(1, device=device)

        # ── Batch 1 Task 9a: force-reset return-norm stats + expose on trainer ──
        # WHAT: zero _ret_mean/_ret_count and set _ret_var=1 in-place at patch
        #   apply time, then attach the tensors to the trainer instance.
        # WHY: Task 6c symlog-compresses rewards before they enter the rollout
        #   buffer, so mb_returns = advantages + values lives in symlog space.
        #   The return-norm running stats must therefore start fresh — carrying
        #   stale raw-scale stats from a pre-Batch-1 checkpoint would contaminate
        #   the symlog-space computation throughout warmup.
        # PITFALL: use in-place .zero_()/.fill_() rather than reassigning the
        #   names. `_update_return_stats` mutates these tensors IN PLACE
        #   (.copy_()), so a test reading self._ret_var sees the live value, not a
        #   stale snapshot; rebinding the attribute would break that contract and
        #   the resume path (`restore_train_state` copies into them in place).
        _ret_mean.zero_()
        _ret_var.fill_(1.0)
        _ret_count.zero_()
        self._ret_mean = _ret_mean
        self._ret_var = _ret_var
        self._ret_count = _ret_count
        # ──────────────────────────────────────────────────────────────────────

        # ── ADAPTIVE ENTROPY (Lagrangian / SAC-style alpha) ────────────────────
        # Batch 3: max_entropy = sum of discrete max entropies + closed-form
        # Gaussian entropy at σ = exp(LOG_STD_MAX). Used for entropy-coefficient
        # annealing schedules (target_entropy ramp in Task 9A) and the SAC-α
        # dual loop's bookkeeping. Discrete heads contribute log(N_i) each;
        # the Gaussian contributes 0.5·log(2πe·σ²) per AIM_DIM — using σ_max
        # is the conservative ceiling, since the policy's actual σ is clamped
        # ≤ exp(LOG_STD_MAX) in every forward call.
        # R0-E (#131): the ceiling follows the run — σ cap (policy.aim_log_std_max),
        # number of live aim dims (aim_dim_mask.sum(): 1 when pitch is pinned) and
        # the entropy-bonus switch (0 continuous entropy when the Gaussian is
        # excluded from the objective, so the target/floor track the discrete
        # heads only). getattr defaults keep pre-R0-E policies/configs working.
        max_entropy_discrete = sum(np.log(n) for n in ACTION_HEAD_SIZES)
        _cap = float(getattr(self.policy, "aim_log_std_max", LOG_STD_MAX))
        _n_aim = float(getattr(self.policy, "aim_dim_mask", torch.ones(AIM_DIM)).sum())
        _bonus = bool(self.config.get("aim_entropy_bonus", True))
        max_entropy_continuous = (_n_aim * 0.5 * np.log(2 * np.pi * np.e * np.exp(_cap)**2))\
            if _bonus else 0.0
        max_entropy = max_entropy_discrete + max_entropy_continuous
        # Task 9A: target_entropy is no longer a static scalar — it's recomputed
        # each train() call from a linear ramp 0.7→0.5*max_entropy across
        # [0, 10_000_000] global steps (see target_entropy_schedule). The live
        # value lives in the local _t9_target_entropy in train() and on
        # self._batch1_current_target_entropy.
        entropy_floor = 0.3 * max_entropy              # collapse threshold
                                                       # W2a (gh#168): the floor used to be a closure variable of the train body;
                                                       # tests/test_warmstart_entropy_trainer.py sets it directly to force the
                                                       # clamp arm.
        self._entropy_floor = entropy_floor

        log_alpha = torch.tensor([math.log(0.1)], requires_grad=True, device=device)
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=1e-4)
        # R0-C (#134): the train_state.pt sidecar reads both from these attributes.
        # Restore MUST be in place (`_log_alpha_tensor.data.copy_`) — rebinding the
        # attribute would leave alpha_optimizer's param list on the old tensor.
        self._log_alpha_tensor = log_alpha
        self._alpha_optimizer = alpha_optimizer

        # Task 9A/9B: trainer-level state for target_entropy schedule + log_alpha
        # reset. Instance attributes (not locals of train()) so:
        #   - tests can inspect/pin _batch1_max_entropy and current_target_entropy
        #   - the wandb log layer can read _batch1_current_target_entropy without
        #     reaching into train().
        # _batch1_log_alpha_reset_done is the idempotency flag for Task 9B —
        # the first train() call after construction resets log_alpha to
        # log(ent_coef); every later train() call leaves log_alpha alone so the
        # SAC dual-gradient loop can do its job.
        self._batch1_max_entropy = float(max_entropy)
        self._batch1_log_alpha_reset_done = False
        # Seed from the schedule at the CURRENT global_step (not a hardcoded
        # warmup constant) so checkpoint-resumed trainers start consistent;
        # the per-train()-call recompute below overwrites it every call anyway.
        self._batch1_current_target_entropy = _scheduled_target_entropy(
            self.config, self.global_step, float(max_entropy))
        # Pre-init effective_alpha + grad_norm metrics (utof/cs2rl#16). The
        # post-loop reads in train() refresh these, but if the
        # target_kl early-break trips on mb=0 OR no accumulation boundary fires,
        # the local names are never bound — leaving the trainer attrs
        # AttributeError on first read. Seeding them with sane defaults here
        # turns those edge cases into "stale-from-previous-call" instead of a
        # crash, and the post-loop refresh overwrites whenever the loop runs
        # all the way through.
        self._batch1_effective_alpha = float(self.config["ent_coef"])
        self._batch1_grad_norm = 0.0

        # ── Warm-start entropy mode state (spec 2026-08-01) ────────────────────
        # h_anchor: mean policy entropy captured at grace end (None until then);
        # last_entropy_mean: previous update's post-divisor losses["entropy"] —
        # the ONLY valid anchor source (there is no entropy EMA in this codebase,
        # and trainer.losses is refreshed only inside the throttled log-flush
        # block, so it can be several updates stale — spec finding 4).
        # h0: first update's mean entropy, denominator of warmstart_h_over_h0
        # (grace collapse watch — with the floor disabled AND alpha~0 the run
        # has no anti-collapse guard, spec finding 6).
        self._batch1_warmstart_h_anchor = None
        self._batch1_warmstart_h0 = None
        self._batch1_warmstart_phase = WS_OFF
        self._batch1_last_entropy_mean = None
        self._batch1_warmstart_warn_epoch = -10**9
        print("[Train] Value target normalization enabled (running mean/std of returns).")

    def _update_return_stats(self, returns_flat):
        """Welford-merge one flat batch of returns into ``_ret_mean/_ret_var/_ret_count``.

        In place (``.copy_()``): the three tensors are also what ``collect_train_state``
        saves and ``restore_train_state`` restores, so their identity must not change.
        Was a closure over the same three tensors until gh#168 W2a.
        """
        with torch.no_grad():
            n = returns_flat.numel()
            if n == 0:
                return
            batch_mean = returns_flat.mean()
            batch_var = returns_flat.var(unbiased=False)
            batch_count = torch.tensor(float(n), device=self._ret_device)

            delta = batch_mean - self._ret_mean
            tot = self._ret_count + batch_count
            new_mean = self._ret_mean + delta * batch_count / tot
            m_a = self._ret_var * self._ret_count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta.pow(2) * self._ret_count * batch_count / tot
            new_var = m2 / tot

            self._ret_mean.copy_(new_mean)
            self._ret_var.copy_(new_var)
            self._ret_count.copy_(tot)

    def _normalize_returns(self, mb_returns, mb_part=None):
        """Return normalized copy of mb_returns; update running stats first.

        mb_part (Rung 0 §2.2): BOOL [S, T] participation mask. The running
        mean/var are updated from the PARTICIPATING rows ONLY — parked rows
        carry reward 0 and value 0, so feeding them in would drag the return
        scale toward 0 by a factor of n_active/TEAM_SIZE and shrink every
        normalized value target. None ⇒ all rows (pre-Rung-0 behaviour,
        bit-identical).
        PITFALL: the whole tensor is still normalized and returned — masking
        applies to the STATISTICS, not the output. The parked rows' value loss
        is dropped later, by the masked_mean over the same mask.
        """
        sel = mb_returns.detach()
        if mb_part is not None:
            sel = sel[mb_part]
        self._update_return_stats(sel.flatten())
        std = (self._ret_var + 1e-8).sqrt()
        return (mb_returns - self._ret_mean) / std

    def _init_selfplay(self, self_play_mgr):
        """Self-play state for evaluate() (the patch-time half of the former self-play
        patcher in train.py; W2b of gh#168). Called from __init__ right after
        hybrid-aim buffer setup and before ``_timing``, where the patch call stood.

        Stores the manager, the past policy's own LSTM state (the same dict structure as
        ``self.lstm_h``: keyed by agent-batch start ``i*n``, one ``(agents_per_batch,
        hidden_size)`` tensor per chunk, so ``evaluate`` can index it by ``env_id.start``),
        and the six ``_batch1_*`` reward-processing attributes. ``process_step_rewards``
        updates the three ``WelfordStd`` and reads their ``std()``; ``train`` copies the
        three ``std()`` values into ``_batch1_std_*`` at the end of each call.
        ``process_step_rewards`` writes each env's channel sum into
        ``_batch1_reward_scratch`` when step_stats exists, or a 0.0 placeholder
        when it does not, then reads the scratch buffer for one host-to-device copy;
        ``evaluate`` replaces the buffer with a longer one when the info list (at most one
        entry per env) is longer than it. ``process_step_rewards`` only sets rows of
        ``_batch1_current_segment_has_event`` to True (bomb planted this tick); ``evaluate``
        reads those rows into ``_batch1_event_mask`` and clears them at the segment
        boundary. Apart from the zero fill here, ``_batch1_event_mask`` is written only by
        ``evaluate`` at that boundary, and ``train`` reads it.
        ``save_checkpoint`` reads the stored manager when writing the sidecar.
        """
        self._self_play_mgr = self_play_mgr
        self._past_lstm_h = {k: torch.zeros_like(v) for k, v in self.lstm_h.items()}
        self._past_lstm_c = {k: torch.zeros_like(v) for k, v in self.lstm_h.items()}
        # Batch 1 reward processing state (per-channel Welford, per-segment event
        # masks, and the numpy scratch buffer process_step_rewards writes into; evaluate
        # replaces it with a longer one when the info list, at most one entry per env, is
        # longer than the buffer).
        self._batch1_welford_combat = WelfordStd(prior_std=1.0, min_count=1000)
        self._batch1_welford_objective = WelfordStd(prior_std=1.0, min_count=1000)
        self._batch1_welford_positional = WelfordStd(prior_std=1.0, min_count=1000)
        _dev = self.config["device"]
        self._batch1_event_mask = torch.zeros(self.segments, dtype=torch.bool, device=_dev)
        self._batch1_current_segment_has_event = torch.zeros(self.total_agents,
                                                             dtype=torch.bool,
                                                             device=_dev)
        self._batch1_reward_scratch = np.empty(0, dtype=np.float32)
        print("[Train] Self-play evaluate patch enabled.")

    def save_checkpoint(self):
        """Write the full three-file checkpoint set and return its model path.

        Unlike stock PuffeRL, never return early when the model path exists:
        a resumed run can re-save an epoch after loading, and a crash between
        writes must not freeze older state files. Every write is atomic, and
        ``train_state.pt`` holds the extra Cs2PuffeRL state. PuffeRL.close()
        copies the returned model path to ``<data_dir>/<run_id>.pt``.

        WRITE ORDER is load-bearing: model → train_state → trainer_state. The
        LAST file names the model and carries the epoch checked against the
        sidecar, so a crash leaves a set that resume accepts whole or refuses.
        """
        run_id = self.logger.run_id
        path = Path(self.config["data_dir"]) / run_id
        path.mkdir(parents=True, exist_ok=True)
        model_name = f"model_{self.epoch:06d}.pt"
        model_path = path / model_name
        _atomic_save_state_dict(self.uncompiled_policy.state_dict(), model_path)
        _atomic_save_state_dict(collect_train_state(self, self._self_play_mgr),
                                path / "train_state.pt")
        _atomic_save_state_dict(
            {
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "agent_step": self.global_step,
                "update": self.epoch,
                "model_name": model_name,
                "run_id": run_id,
            }, path / "trainer_state.pt")
        return str(model_path)

    def evaluate(self):
        """Rollout collection with the self-play opponent override (the body of the
        former self-play patcher's closure in train.py; W2b of gh#168).

        For each evaluation epoch ``SelfPlayManager.should_use_past()`` decides (once)
        whether to activate self-play. When active, a random past checkpoint is loaded
        and its actions+logprobs replace the current-policy outputs for the
        opponent-team slots in the rollout buffer. The current policy's LSTM state is
        updated normally; the past policy has its own independent LSTM state tensors
        (``self._past_lstm_h`` / ``self._past_lstm_c``, allocated by
        ``_init_selfplay``). ``train()`` sees the overridden actions as if they came
        from the current policy at collection time; the importance ratio
        (pi_new / pi_old) is well-defined because the *past* policy's logprobs are
        stored as pi_old.

        Rung 1a T3: a SECOND, unconditional opponent override for
        ``self._self_play_mgr.opponent_mode == "noop"`` (the stationary statue). It is
        independent of the past-policy branch (dead at p_past = 0, the only
        configuration noop allows) and pairs with the hero-team-only participation
        vector from ``build_participating_rows``.

        Every module global the body reads (``pufferlib``, ``torch``, ``np``,
        ``process_step_rewards``, ``_hybrid_sample_logits``) is imported at the top
        of this module from its defining module; ast_oracle.py check S2 pins that.
        """
        profile = self.profile
        epoch = self.epoch
        profile("eval", epoch)
        profile("eval_misc", epoch, nest=True)

        cfg = self.config
        dev = cfg["device"]

        if cfg["use_rnn"]:
            for k in self.lstm_h:
                self.lstm_h[k].zero_()
                self.lstm_c[k].zero_()

        # ── Decide self-play for this epoch ────────────────────────────────
        use_past = self._self_play_mgr.should_use_past()
        past_policy = None
        if use_past:
            past_policy = self._self_play_mgr.load_past_policy(dev, self.vecenv)
            use_past = past_policy is not None

        # TAG (spec 2026-08-13 §4.2): expose whether THIS epoch's rollout
        # used a past-policy opponent — on those epochs one team's rows are
        # off-policy and cross-team cos-sim measures on-vs-off-policy
        # asymmetry, not T/CT conflict; the analyzer drops them.
        self._selfplay_used_past = bool(use_past)

        if use_past:
            for k in self._past_lstm_h:
                self._past_lstm_h[k].zero_()
                self._past_lstm_c[k].zero_()
        # ───────────────────────────────────────────────────────────────────

        self.full_rows = 0
        while self.full_rows < self.segments:
            profile("env", epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile("eval_misc", epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)
            # Rung 0 §2.2: global_step counts PARTICIPATING agent-steps, so
            # --timesteps means the same thing at any n_active_per_team.
            # `mask` is the recv() chunk's live-agent mask (all-True for this
            # env, vector.py:188); env_id is the agent-row slice bound just
            # above, so the static row flags line up element-for-element.
            self.global_step += int(
                (np.asarray(mask, dtype=bool) & self._participating_rows_np[env_id]).sum())

            profile("eval_copy", epoch)
            o = torch.as_tensor(o)
            o_device = o.to(dev)
            r = torch.as_tensor(r).to(dev)
            d = torch.as_tensor(d).to(dev)

            # F8: pull the C-computed action masks for exactly this batch of
            # agent rows. env_id indexes agent rows, matching the shm layout
            # (num_envs*N_AGENTS, ACTION_MASK_DIM). `!= 0` both converts to
            # bool AND copies — the shm bytes get overwritten by the next
            # worker step, so we must not keep a view. None ⇒ unmasked
            # (legacy trainer built without the mask shm).
            mask_view = getattr(self, "_action_mask_view_main", None)
            action_mask = None
            if mask_view is not None:
                action_mask = torch.as_tensor(mask_view[env_id]).to(dev) != 0

            profile("eval_forward", epoch)
            with torch.no_grad(), self.amp_context:
                state = dict(reward=r, done=d, env_id=env_id, mask=mask)
                if cfg["use_rnn"]:
                    state["lstm_h"] = self.lstm_h[env_id.start]
                    state["lstm_c"] = self.lstm_c[env_id.start]

                # Batch 3 (T5) + Fix #1: hybrid rollout. Policy returns 4-tuple
                # (logits, mu_aim, log_std, value). _hybrid_sample_logits now
                # returns the per-factor log-prob halves directly (6-tuple),
                # eliminating the previous double-construction of 7 Categorical
                # + 1 Normal at the rollout site (was +437 ms/epoch on the
                # smoke benchmark per perf investigation post-PR #28).
                logits, mu_aim, log_std_aim, value = self.policy.forward_eval(o_device, state)
                action, cont_action, logprob_d, logprob_c, _, _ = _hybrid_sample_logits(
                    (logits, mu_aim, log_std_aim, value),
                    max_turn_speed=self.policy.max_turn_speed.item(),
                    mask=action_mask,
                    aim_dim_mask=getattr(self.policy, "aim_dim_mask", None),
                )
                # Joint log-prob for self.logprobs (back-compat slot read by
                # PufferLib's diagnostics + the KL/clipfrac path). Per-factor
                # halves go to self.logprobs_d / self.logprobs_c for the
                # H-PPO clip in _hybrid_ppo_loss.
                logprob = logprob_d + logprob_c

                # ── Task 6c: per-channel reward norm + symlog (replaces the
                # old hard-clip of r to [-1, 1]). Pipeline:
                #   step_stats (Task 6a info payload)
                #     → split_into_channels (Task 4)
                #     → WelfordStd.update + normalize per channel (Task 5)
                #     → sum channels → symlog (Task 4) → r written to buffer
                #
                # Info shape: PufferLib's Serial/Multiprocessing backend
                # collects info with list-extend semantics (pufferlib/vector.py
                # ~L149-153). Cs2Env returns `[{"step_stats": view}]` per tick
                # so `len(info)` is the number of envs in this batch, while
                # r.shape[0] == len(info) * agents_per_env (10 for Cs2Env).
                # All 10 agents in an env share the same step_stats because
                # step_stats aggregates team-level reward fields; we update
                # Welford ONCE per env (not per agent — that would over-count
                # by 10x) and apply the same symlog'd channel sum to every
                # agent row in that env.
                #
                # Fallback: if info[e] lacks step_stats (flag off OR an older
                # info entry that predates Task 6a), pass raw r through for
                # that env's rows unchanged — the minimal-disruption path if
                # the flag gets toggled or an upstream change sneaks through.
                #
                # Task 7: process_step_rewards also ORs the per-tick
                # bomb_planted flag into _batch1_current_segment_has_event for
                # every agent row in the env. The C side sets ss->bomb_planted
                # only on the transition tick (process_bomb, cs2_bomb.h — guarded by
                # `if g->bomb_plant_ticks >= sd->bomb_plant_time`) and StepStats
                # is cleared every step via clear_stats(ss) at the top of
                # env_step (cs2_env.h), so the field is already a per-tick delta (1 only
                # on the plant tick) — NO edge-trigger needed. All 10 agent rows
                # in an env share the event state; it is flushed into
                # _batch1_event_mask at the segment boundary below.
                #
                # PERF: the per-env loop lives in process_step_rewards() and
                # builds the tick's rewards in a host float32 scratch buffer, so
                # the device sees one H2D copy + one symlog per tick instead of
                # three single-scalar torch.tensor() constructions per env
                # (~213k launch-bound CUDA ops/epoch at 256 envs x 64 ticks,
                # inside this timed eval_forward region). The helper's docstring
                # carries the bit-exactness invariants — read it before touching
                # the arithmetic.
                agents_per_env_local = self.vecenv.driver_env.num_agents
                if self._batch1_reward_scratch.shape[0] < len(info):
                    self._batch1_reward_scratch = np.empty(len(info), dtype=np.float32)
                r = process_step_rewards(
                    info,
                    r,
                    agents_per_env_local,
                    self._batch1_welford_combat,
                    self._batch1_welford_objective,
                    self._batch1_welford_positional,
                    self._batch1_reward_scratch,
                    current_segment_has_event=self._batch1_current_segment_has_event,
                )

                # ── SELF-PLAY: override opponent-team actions ───────────────
                if use_past:
                    batch_n = o_device.shape[0]
                    opp_mask = self._self_play_mgr.get_opponent_mask(batch_n, dev)
                    opp_idx = torch.where(opp_mask)[0]

                    past_state = {
                        "done": d[opp_mask],
                        "lstm_h": self._past_lstm_h[env_id.start][opp_mask],
                        "lstm_c": self._past_lstm_c[env_id.start][opp_mask],
                    }
                    # Batch 3 (T5) + Fix #1: past policy is a HybridPolicy too;
                    # same 4-tuple contract. _hybrid_sample_logits now surfaces
                    # the per-factor log-prob halves directly (6-tuple), so we
                    # no longer reconstruct 7 Categorical + 1 Normal here. The
                    # rollout buffer entries stored at this opponent slot stay
                    # consistent with the current-policy branch (PPO update
                    # treats them indistinguishably).
                    opp_logits, opp_mu, opp_log_std, _opp_value = past_policy.forward_eval(
                        o_device[opp_mask], past_state)
                    (opp_action, opp_cont_action, opp_logprob_d, opp_logprob_c, _,
                     _) = _hybrid_sample_logits(
                         (opp_logits, opp_mu, opp_log_std, None),
                         max_turn_speed=past_policy.max_turn_speed.item(),
                         mask=action_mask[opp_mask] if action_mask is not None else None,
                         aim_dim_mask=getattr(past_policy, "aim_dim_mask", None),
                     )
                    opp_logprob = opp_logprob_d + opp_logprob_c

                    # Write back updated past-policy LSTM states (cast from fp16 if needed)
                    self._past_lstm_h[env_id.start][opp_mask] = past_state["lstm_h"].to(
                        self._past_lstm_h[env_id.start].dtype)
                    self._past_lstm_c[env_id.start][opp_mask] = past_state["lstm_c"].to(
                        self._past_lstm_c[env_id.start].dtype)

                    # Replace opponent slots in action & logprob buffers.
                    # Cast to destination dtype (amp_context may yield fp16).
                    # Continuous action and per-factor logprobs are also
                    # spliced in so train()'s _hybrid_ppo_loss sees consistent
                    # mb_cont_actions / mb_old_logp_{d,c} for opponent rows.
                    action[opp_idx] = opp_action.to(action.dtype)
                    logprob[opp_idx] = opp_logprob.to(logprob.dtype)
                    cont_action[opp_idx] = opp_cont_action.to(cont_action.dtype)
                    logprob_d[opp_idx] = opp_logprob_d.to(logprob_d.dtype)
                    logprob_c[opp_idx] = opp_logprob_c.to(logprob_c.dtype)
                # ──────────────────────────────────────────────────────────

                # ── STATUE OPPONENT (--opponent noop, Rung 1a T3) ──────────
                # UNCONDITIONAL branch, deliberately NOT nested in the
                # `if use_past:` splice above: that branch never runs at
                # p_past = 0, which is exactly the configuration noop demands
                # (assert_opponent_self_play_compatible). It sits AFTER the
                # splice so the statue would win if both were ever live, and
                # BEFORE both the buffer scatter and vecenv.send below — the
                # same tensors feed the rollout buffer and the env.
                #
                # Bin 0 on every discrete head is the no-op action by
                # construction (cs2_env.h:51-63; move_dir == 0 is genuinely
                # stationary, valid_dir in process_movement) and is never masked out by
                # the C-side action mask, so this cannot sample an illegal
                # action. cont_action = 0 means zero Δyaw/Δpitch: the statue
                # keeps its spawn orientation.
                #
                # The stored logprobs go to 0 for the same reason the past-
                # policy splice rewrites them: they are the π_old the PPO
                # update would divide by. Under noop these rows are
                # non-participating, so every loss masks them out anyway —
                # this keeps the buffer self-consistent rather than carrying
                # log-probs of actions that were never sampled.
                if self._self_play_mgr.opponent_mode == "noop":
                    opp_idx = torch.where(
                        self._self_play_mgr.get_opponent_mask(o_device.shape[0], dev))[0]
                    action[opp_idx] = 0
                    cont_action[opp_idx] = 0
                    logprob[opp_idx] = 0
                    logprob_d[opp_idx] = 0
                    logprob_c[opp_idx] = 0
                    # Redundant with the participation scatter below (which
                    # multiplies values by the row flag) and kept anyway: the
                    # statue's critic output must never bootstrap GAE, no
                    # matter which of the two masks a future edit touches.
                    value[opp_idx] = 0
                # ──────────────────────────────────────────────────────────

            profile("eval_copy", epoch)
            with torch.no_grad():
                if cfg["use_rnn"]:
                    self.lstm_h[env_id.start] = cast(torch.Tensor, state["lstm_h"])
                    self.lstm_c[env_id.start] = cast(torch.Tensor, state["lstm_c"])

                # These rollout counters are integer tensors. The installed
                # Tensor.item() stub returns a wider scalar union than runtime.
                seq_pos = cast(int, self.ep_lengths[env_id.start].item())
                batch_rows = slice(
                    cast(int, self.ep_indices[env_id.start].item()),
                    1 + cast(int, self.ep_indices[env_id.stop - 1].item()),
                )

                if cfg["cpu_offload"]:
                    self.observations[batch_rows, seq_pos] = o
                else:
                    self.observations[batch_rows, seq_pos] = o_device

                self.actions[batch_rows, seq_pos] = action
                self.logprobs[batch_rows, seq_pos] = logprob
                # Batch 3 (T5): parallel writes for the new buffers allocated
                # by _init_hybrid_aim. The PPO update (`Cs2PuffeRL.train`
                # in src/cs2rl/trainer.py, gh#168 W2a:
                # `mb_cont_actions = self.cont_actions[idx]` and the two logprob
                # reads beside it) reads these by the same idx; missing this write would
                # silently feed zeros to _hybrid_ppo_loss → ratio_c always
                # equals exp(new_logp_c - 0), which would diverge.
                self.cont_actions[batch_rows, seq_pos] = cont_action
                self.logprobs_d[batch_rows, seq_pos] = logprob_d
                self.logprobs_c[batch_rows, seq_pos] = logprob_c
                # F8: persist the masks the sampler just used so the PPO
                # update (mb_masks in _hybrid_ppo_loss) recomputes logprobs
                # over the identical masked distribution. Skipped when
                # unmasked — the buffer's all-ones default is the no-op mask.
                if action_mask is not None:
                    self.action_masks[batch_rows, seq_pos] = action_mask
                self.rewards[batch_rows, seq_pos] = r
                self.terminals[batch_rows, seq_pos] = d.float()
                # Rung 0 §2.2: scatter the static row flag into buffer layout,
                # and zero the critic output on parked rows (defence in depth —
                # the masked reductions in train() are what make it correct;
                # this just keeps GAE from propagating a bootstrap value
                # through rows whose reward is identically 0).
                # OUTSIDE the `if action_mask is not None:` guard above ON
                # PURPOSE: inside it, `participating` would stay all-zero for a
                # mask-less run and the per-epoch any() assert would fire.
                _part_rows = self._participating_rows[env_id]
                self.participating[batch_rows, seq_pos] = _part_rows
                self.values[batch_rows, seq_pos] = value.flatten() * _part_rows.to(value.dtype)

                self.ep_lengths[env_id] += 1
                if seq_pos + 1 >= cfg["bptt_horizon"]:
                    num_full = env_id.stop - env_id.start
                    # Task 7: flush the live event accumulator → segment mask
                    # BEFORE overwriting ep_indices. Each agent row's current
                    # segment index lives in self.ep_indices[env_id]; once we
                    # reassign ep_indices to (free_idx + arange(num_full)) a
                    # few lines down, the old segment index is lost. Clone
                    # first, write to _batch1_event_mask at those OLD slots,
                    # then reset the live accumulator so the next segment
                    # starts clean. Pitfall: writing AFTER the re-index would
                    # clobber freshly-allocated future segments (off-by-one
                    # bug that would silently mark the wrong rollout rows).
                    old_seg_indices = self.ep_indices[env_id].clone().long()
                    self._batch1_event_mask[old_seg_indices] = (
                        self._batch1_current_segment_has_event[env_id])
                    self._batch1_current_segment_has_event[env_id] = False
                    self.ep_indices[env_id] = (self.free_idx +
                                               torch.arange(num_full, device=dev).int())
                    self.ep_lengths[env_id] = 0
                    self.free_idx += num_full
                    self.full_rows += num_full

                action = action.cpu().numpy()
                if isinstance(logits, torch.distributions.Normal):
                    import numpy as _np

                    lo, hi = self.vecenv.action_space.low, self.vecenv.action_space.high
                    action = _np.clip(action, lo, hi)

            profile("eval_misc", epoch)
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.stats[k].extend(v)
                    else:
                        self.stats[k].append(v)

            profile("env", epoch)
            # Batch 3 (T5/T5b): HybridAimVecEnv.send accepts an
            # (action, cont_action)
            # tuple. Discrete action is the numpy int32 buffer that the C
            # env still receives positionally. For the Serial backend
            # cont_action is forwarded to env.step's continuous_actions
            # kwarg via the per-env step wrapper. For the Multiprocessing
            # backend cont_action is mirrored into a multiprocessing.RawArray
            # shm view by HybridAimVecEnv.send before backend.send runs, so workers
            # see the same Δyaw on their next Cs2Env.step via the per-env
            # numpy view installed by _attach_cont_action_view (see the
            # HybridAimVecEnv docstring for the shm pattern).
            self.vecenv.send((action, cont_action))

        profile("eval_misc", epoch)
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=dev, dtype=torch.int32)
        self.ep_lengths.zero_()
        profile.end()
        return self.stats

    def train(self):
        """One PPO update over the rollout buffer; the return-norm patcher's inner train()
        replacement (src/cs2rl/train_update.py) until gh#168 W2a, moved verbatim.

        WHAT: return-normalised value targets, the hybrid discrete+continuous PPO loss
        (``_hybrid_ppo_loss``), the SAC-style α dual loop, the warm-start entropy mode,
        the TAG gradient-cosine diagnostic and every ``losses[...]``/``_batch1_*`` metric
        the log layer and the metrics census read.

        WHY this replaces ``PuffeRL.train`` outright (it never calls ``super().train``):
        the stock loop cannot unpack the policy's 4-tuple output (see the F11 note in
        src/cs2rl/train.py::train). The AST oracle in
        .superpowers/sdd/2026-09-24-168-trainer-subclass/ast_oracle.py pins this body
        to the pre-move closure node for node (N1-N5); the byte gates pin the arms the
        2-epoch run executes.

        PITFALLS: names that were closure variables are now attributes
        (``_ret_mean/_ret_var``, ``_log_alpha_tensor``, ``_alpha_optimizer``,
        ``_entropy_floor``, ``_normalize_returns``); ``self`` here is what the closure
        called both ``self`` and ``trainer``. The stock ``@record`` decorator is not
        re-applied (nothing runs under torchrun; spec §1).
        """
        profile = self.profile
        epoch = self.epoch
        profile("train", epoch)
        losses = defaultdict(float)
        # Rung 0 §2.2 diagnostics. Local ints (not losses[...] entries) because
        # they are ABSOLUTE counts: anything accumulated into `losses` inside
        # the loop gets divided by _mb_run afterwards (gh#90). They are written
        # onto `losses` after that divisor.
        _floor_fires = 0               # minibatches where the entropy-floor clamp bound
        _empty_mb = 0                  # minibatches with zero participating rows (skipped)
        config = self.config
        device = config["device"]

        b0 = config["prio_beta0"]
        a = config["prio_alpha"]
        clip_coef = config["clip_coef"]
        vf_clip = config["vf_clip_coef"]
        anneal_beta = b0 + (1 - b0) * a * self.epoch / self.total_epochs
        self.ratio[:] = 1

        # Task 9A: recompute target_entropy from the linear ramp once per
        # train() call. WHY here (not inside the minibatch loop): the schedule
        # is keyed on global_step which is fixed for the duration of a single
        # train() call, so recomputing per-minibatch would burn cycles for no
        # signal. We mirror the value onto self._batch1_current_target_entropy
        # so the wandb log layer can read it without touching this method.
        # Fracs/warmup_steps come from config via _scheduled_target_entropy
        # (finding 4 residual — previously hardcoded 0.7→0.5).
        # PITFALL: read self._batch1_max_entropy, never a copy taken at
        # construction: the value follows the run (σ cap, pinned pitch, bonus
        # switch) and a resume restores the instance attribute.
        _t9_target_entropy = _scheduled_target_entropy(config, self.global_step,
                                                       self._batch1_max_entropy)
        self._batch1_current_target_entropy = float(_t9_target_entropy)

        # Task 9B: one-shot log_alpha reset on the first train() call after
        # construction. WHY: the entropy schedule + log_alpha are coupled — the
        # outer training loop can leave log_alpha at a stale value from a
        # previous run / re-init, and we need a deterministic starting point
        # of log(ent_coef) so the SAC dual-gradient loop converges from a
        # known floor. The flag is an instance attribute so a checkpoint-restored
        # trainer still resets exactly once.
        if not self._batch1_log_alpha_reset_done:
            with torch.no_grad():
                self._log_alpha_tensor.fill_(math.log(config["ent_coef"]))
            self._batch1_log_alpha_reset_done = True

        # ── Warm-start entropy mode: resolve phase once per train() call ───
        # (global_step only advances in evaluate(), so it is constant here —
        # same reasoning as the Task 9A recompute above; all transitions land
        # on update boundaries.) Ordering vs Task 9B: 9B runs FIRST and sets
        # log_alpha to log(ent_coef) — exactly the operating point warm-start
        # wants (continuity comes from target==h_anchor at release, never
        # from moving log_alpha: Adam(lr=1e-4) travels ~1e-4/minibatch, so a
        # parked log_alpha is stranded — spec finding 1).
        # PITFALL: keep grace+ramp >= entropy_target_warmup_steps. OFF falls
        # through to the Task 9A schedule (see the override condition below),
        # and 9A ramps DOWNWARD — warmup_high_frac*max (0.5) at step 0 to
        # base_frac*max (0.35) at entropy_target_warmup_steps — so during
        # warmup it reads strictly ABOVE the base_frac*max the warm-start ramp
        # lands on. With defaults (grace 5M + ramp 10M = 15M >= 10M warmup) 9A
        # has already flattened at base_frac*max and the handoff is exactly
        # continuous. But e.g. grace=2M+ramp=3M puts ramp_end at 5M, where 9A
        # still reads 0.425*max: the target jumps UPWARD 0.35*max -> 0.425*max,
        # i.e. 2.87 -> 3.49 nats (+0.62, at max_entropy=8.21), at the exact
        # boundary the spec promises is clean.
        _ws_enabled = bool(config.get("warmstart_entropy", False))
        _ws_floor_active = True
        if _ws_enabled:
            _ws_grace = int(config.get("warmstart_grace_steps", 5_000_000))
            if (self._batch1_warmstart_h_anchor is None and self.global_step >= _ws_grace
                    and self._batch1_last_entropy_mean is not None):
                # one-shot anchor capture (idempotent: guarded on None).
                # Finite-check (Task 1 review): a NaN/inf entropy mean latched
                # here would poison target and alpha_loss for the whole ramp —
                # skip the capture (stay GRACE) and shout instead.
                if math.isfinite(self._batch1_last_entropy_mean):
                    self._batch1_warmstart_h_anchor = float(self._batch1_last_entropy_mean)
                else:
                    print(f"[Train] WARN warm-start: non-finite entropy mean "
                          f"{self._batch1_last_entropy_mean} at grace end — "
                          f"anchor capture skipped, staying in GRACE.")
            _ws = warmstart_entropy_state(
                self.global_step,
                grace_steps=_ws_grace,
                ramp_steps=int(config.get("warmstart_ramp_steps", 10_000_000)),
                h_anchor=self._batch1_warmstart_h_anchor,
                base_target=(config.get("entropy_target_base_frac", 0.35) *
                             self._batch1_max_entropy))
            self._batch1_warmstart_phase = _ws.phase
            _ws_floor_active = _ws.floor_active
            if _ws.phase != WS_OFF and _ws.target is not None:
                # Override the Task 9A schedule during the ramp AND mirror it,
                # or the wandb target trace plots the unmodified base schedule
                # (spec finding 9). Effectively RAMP-only: GRACE carries
                # target=None (no target is consumed while alpha is ceilinged).
                # PITFALL: the WS_OFF guard is load-bearing — the helper returns
                # target=base_target (NOT None) once OFF, so testing target
                # alone would pin the target at base_frac*max for the rest of
                # the run and silently flatten the tail of the 9A warmup ramp
                # whenever grace+ramp < entropy_target_warmup_steps. Falling
                # through here is what makes OFF byte-for-byte pre-feature
                # behavior at ANY config, which is what the spec promises.
                _t9_target_entropy = _ws.target
                self._batch1_current_target_entropy = float(_ws.target)
        else:
            # Config can be toggled off in-process (tests do this; production
            # builds the config once). Re-seed the phase so a stale GRACE can
            # never keep the alpha optimizer frozen after the mode is disabled.
            self._batch1_warmstart_phase = WS_OFF

        # Task 8: raw event-segment fraction (mask mean) — computed once per
        # train() call because _batch1_event_mask doesn't change inside the
        # minibatch loop. Persisted onto losses["event_oversample_fraction"]
        # AFTER the gh#90 divisor loop (per-call scalar, like ret_mean).
        _t8_event_mask = getattr(self, "_batch1_event_mask", None)
        # Masked over participating segments (participating[:, 0] is the
        # per-segment flag) so parked rows don't dilute the fraction at n<5.
        self._batch1_event_oversample_fraction = (float(
            masked_mean(_t8_event_mask.float(), self.participating[:, 0].float()))
                                                  if _t8_event_mask is not None else 0.0)

        # ── gh#90: KL early-stop bookkeeping ───────────────────────────────
        # WHAT: the target_kl early-stop is (a) gated to update-epoch
        #   boundaries and (b) decoupled from the losses/* divisor.
        # WHY (root-caused 2026-08-01, run checkpoints-20260801-022606):
        #   the old inline `break` sat before the logging block while every
        #   losses/* metric divided by the PLANNED self.total_minibatches —
        #   a truncated update silently scaled all logged losses by k/N
        #   ("importance=0.0167" was really ratio=1.0 with k=1). And because
        #   this flattened loop collapses all update_epochs into one range,
        #   one KL spike aborted passes over data never visited — harsher
        #   than standard PPO, which finishes the current epoch first.
        # HOW: losses accumulate RAW sums inside the loop and are divided by
        #   the EXECUTED count (_mb_run) after it; a KL trip sets _kl_stop
        #   and the loop exits at the next epoch boundary, so epoch 0 always
        #   completes (⇒ _mb_run >= 1, and the effective_alpha / advantages
        #   post-loop reads can no longer see an mb=0 abort).
        # PITFALL: total_minibatches need not divide update_epochs evenly
        #   (harness: 7 mbs / 3 epochs) — the boundary stride uses floor
        #   division with a >=1 clamp, never a modulo of zero.
        target_kl = config.get("target_kl", None)
        _mbs_per_epoch = max(1,
                             self.total_minibatches // max(1, int(config.get("update_epochs", 1))))
        _kl_stop = False
        _mb_run = 0

        for mb in range(self.total_minibatches):
            if _kl_stop and mb % _mbs_per_epoch == 0:
                break                  # epoch boundary: honor the KL trip
            profile("train_misc", epoch, nest=True)
            self.amp_context.__enter__()

            shape = self.values.shape
            advantages = torch.zeros(shape, device=device)
            advantages = compute_puff_advantage(
                self.values,
                self.rewards,
                self.terminals,
                self.ratio,
                advantages,
                config["gamma"],
                config["gae_lambda"],
                config["vtrace_rho_clip"],
                config["vtrace_c_clip"],
            )

            profile("train_copy", epoch)
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)

            # ── Batch 1 Task 8: event-biased prio_probs oversampling ──────
            # WHAT: when the segment-level event mask is populated and at
            #   least one segment contains a bomb-plant event, multiply the
            #   prio_probs of those segments by OVERSAMPLE_FACTOR before
            #   renormalising. The downstream torch.multinomial call then
            #   draws biased samples without any further changes — and the
            #   importance-sampling correction (mb_prio, consumed inside
            #   _hybrid_ppo_loss) uses the BOOSTED prio_probs[idx], so the
            #   gradient stays unbiased.
            # WHY: bomb-plant events are sparse in early training (the exact
            #   fraction is itself a Task 9 metric, reported via
            #   _batch1_event_oversample_fraction). Uniform prio sampling
            #   under-replays them; oversampling accelerates value-function
            #   fit on the rare-but-decisive transitions. Plan §Task 8
            #   target: event-mask hit-rate among sampled segments >= 25%.
            # PITFALLS:
            #   * mask absent / all-False → skip the boost so pre-Batch-1
            #     training paths and the warm-up pass before any plant
            #     happens still work (no division by zero, no NaN).
            #   * Boost the prob, not the weight — boosting `prio_weights`
            #     and re-running the (w+1e-6)/(sum+1e-6) renorm would alter
            #     the abs-advantage prior shape; multiplying prio_probs and
            #     dividing by sum keeps the prior intact on non-event rows.
            #   * Cloning before the in-place mul protects callers that
            #     might still hold a reference to the original prio_probs.
            #   * The exposed metric is the RAW event fraction (mask mean),
            #     NOT the post-boost sampled fraction. Written onto
            #     losses["event_oversample_fraction"] after the divisor
            #     loop — do not accumulate it inside this minibatch loop.
            OVERSAMPLE_FACTOR = 4.0
            if _t8_event_mask is not None and _t8_event_mask.any():
                boosted = prio_probs.clone()
                boosted[_t8_event_mask] *= OVERSAMPLE_FACTOR
                prio_probs = boosted / boosted.sum()
            # ──────────────────────────────────────────────────────────────

            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments * prio_probs[idx, None])**-anneal_beta
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            # (mb_rewards pull removed with the dead per-minibatch
            # compute_puff_advantage recompute — see finding-1 note below)
            mb_terminals = self.terminals[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]
            # Batch 3 (T5): pull continuous actions + per-factor old logprobs
            # from the parallel buffers allocated by _init_hybrid_aim.
            # mb_logprobs (the SUM) stays the canonical "logp from rollout" for
            # KL/clipfrac diagnostics below; the per-factor halves drive the
            # per-factor PPO clip in _hybrid_ppo_loss.
            mb_cont_actions = self.cont_actions[idx]
            mb_old_logp_d = self.logprobs_d[idx]
            mb_old_logp_c = self.logprobs_c[idx]
            # F8: rollout-stored action masks (all-ones = unmasked fallback).
            # getattr for trainers built before hybrid buffer setup
            # ran (shouldn't happen in prod; keeps direct-call tests working).
            _masks_buf = getattr(self, "action_masks", None)
            mb_masks = _masks_buf[idx] if _masks_buf is not None else None

            # ── Rung 0 §2.2: participating mask for this minibatch ───────────
            # Two dtypes on purpose (see the masked_* helpers' contract):
            # mb_part is BOOL for indexing, mb_part_f/flat_part are the FLOAT
            # weights the reductions take. flat_part is for the flat (S*T,)
            # tensors (entropy, ratio_d/ratio_c, per-head entropies);
            # mb_part_f for the [S, T]-shaped ones (v_loss, value writeback).
            # Shapes: mb_part / mb_part_f are [S, T]; flat_part is (S*T,).
            mb_part = self.participating[idx]
            mb_part_f = mb_part.to(torch.float32)
            n_part = mb_part_f.sum()
            # Empty-minibatch tripwire: only reachable with non-uniform segment
            # sampling (prio_alpha != 0 or a marked event segment) that happens
            # to draw an all-parked minibatch — see spec §2.2. Skipping is the
            # right call (every reduction below would be 0/0), but it must be
            # VISIBLE, so it is counted into losses/empty_minibatches rather
            # than silently swallowed. The `continue` sits before _mb_run += 1,
            # so the gh#90 divisor keeps counting only executed minibatches.
            if n_part.item() == 0:
                _empty_mb += 1
                # No zero_grad here: accumulate_minibatches is always 1 today,
                # so no partial gradient can be pending. Revisit if that changes.
                continue
            flat_part = mb_part_f.reshape(-1)

            # ── VALUE TARGET NORMALISATION ─────────────────────────────────
            # Normalize returns before value regression.  The value head learns
            # to predict normalized targets; advantages are unaffected.
            mb_returns_norm = self._normalize_returns(mb_returns, mb_part)
            # Also normalize the stored baseline values so clipping stays valid
            mb_values_norm = (mb_values - self._ret_mean) / (self._ret_var + 1e-8).sqrt()
            # ──────────────────────────────────────────────────────────────

            profile("train_forward", epoch)
            if not config["use_rnn"]:
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            # LSTM-BPTT fix: lstm_h/lstm_c None → zero initial state, which
            # is exact (each stored segment began at evaluate()'s zeroed
            # state — see Dust2Policy.forward doc). terminals drives the
            # mid-segment done-reset inside _lstm_bptt so the training
            # forward replicates the rollout's (1-done)*state masking.
            state = dict(
                action=mb_actions,
                lstm_h=None,
                lstm_c=None,
                terminals=mb_terminals,
            )

            # Batch 3 (T5): hybrid PPO update — per-factor clipped loss
            # (Fan et al. IJCAI 2019). The helper does the policy forward
            # pass (returning mu_aim/log_std + value) and assembles the
            # clipped policy loss with INDEPENDENT discrete and continuous
            # ratios. F16 (2026-07-06 adversarial review): it now also
            # returns the logits it computed, killing the redundant no-grad
            # diagnostic forward that used to run here per minibatch
            # (halves update-forward cost). Post-F8 the returned logits are
            # MASKED, so the per-head entropy diagnostics below report the
            # true sampled distribution.
            (pg_loss, entropy, newvalue, newlogprob, ratio_d, ratio_c, logits) = _hybrid_ppo_loss(
                self.policy,
                mb_obs,
                mb_actions,
                mb_cont_actions,
                mb_old_logp_d,
                mb_old_logp_c,
                mb_advantages,
                clip_coef,
                state,
                mb_prio=mb_prio,
                mb_masks=mb_masks,
                mb_part=mb_part_f,
                aim_dim_mask=getattr(self.policy, "aim_dim_mask", None),
                aim_entropy_bonus=bool(config.get("aim_entropy_bonus", True)),
            )
            # NOTE: pre-Batch-3 the inline `actions = ...` from sample_logits
            # was used by downstream diagnostics; T5 dropped that consumer
            # (mb_actions is the canonical stored discrete action). No
            # rebinding here — the variable is unused after this point.
            # (F16: the former no-grad diagnostic re-forward that lived here
            # is gone — `logits` now comes straight from _hybrid_ppo_loss.)

            profile("train_misc", epoch)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            # Batch 3: keep the joint ratio for KL/clipfrac diagnostics so the
            # existing log surface (approx_kl, clipfrac, importance) is
            # backwards-compatible. ratio_d is what gets stored in self.ratio
            # because compute_puff_advantage was tuned for the discrete-head
            # importance ratio in pre-Batch-3 runs; substituting ratio_d here
            # preserves vtrace's behaviour.
            ratio = logratio.exp()
            # Batch 3 (T5): _hybrid_ppo_loss returns flat (B*T,) ratios.
            # self.ratio is (segments, bptt_horizon); reshape ratio_d to
            # match so the indexed-write writes the right shape. ratio
            # (joint) is already reshaped by mb_logprobs.shape on the
            # previous line.
            self.ratio[idx] = ratio_d.detach().reshape(mb_logprobs.shape)

            with torch.no_grad():
                # Rung 0 §2.2: every diagnostic below is a mean over rows, so
                # every one of them is masked. _pm is mb_part_f reshaped to the
                # joint ratio's [S, T] layout; ratio_d/ratio_c are flat, hence
                # flat_part. An unmasked KL here would be diluted 5× at
                # n_active=1 and the target_kl early-stop would never fire.
                _pm = mb_part_f.reshape(logratio.shape)
                old_approx_kl = masked_mean(-logratio, _pm)
                approx_kl = masked_mean((ratio - 1) - logratio, _pm)
                clipfrac = masked_mean(((ratio - 1.0).abs() > config["clip_coef"]).float(), _pm)
                # Observe-only (spec 2026-08-15 §3.4): same formula as the
                # joint `clipfrac` above, split by the per-factor ratios
                # `_hybrid_ppo_loss` already returns. `.item()` into the
                # logging dict only — NEVER add these to the `loss` tensor
                # (they are diagnostics, not a training signal). Last-
                # minibatch-only is forbidden; they accumulate like
                # `clipfrac` and ride the existing `_mb_run` divisor.
                clipfrac_d = masked_mean(((ratio_d - 1.0).abs() > config["clip_coef"]).float(),
                                         flat_part)
                clipfrac_c = masked_mean(((ratio_c - 1.0).abs() > config["clip_coef"]).float(),
                                         flat_part)

            # Early stopping (gh#90): a KL trip finishes the CURRENT epoch
            # (this minibatch included — matches standard PPO's post-epoch
            # check) and stops at the next epoch boundary via the loop-top
            # gate, instead of the old immediate mid-pass break.
            if target_kl is not None and approx_kl.item() > target_kl:
                _kl_stop = True

            # Batch 3 (T5): pg_loss already computed by _hybrid_ppo_loss above
            # via per-factor clipping (the pre-Batch-3 single-ratio block
            # would over-clip — spec L8 decision). Advantage normalization +
            # the mb_prio importance weight now live INSIDE _hybrid_ppo_loss
            # (finding 1, 2026-07-06 adversarial review); the orphaned
            # normalization stub and the discarded per-minibatch
            # compute_puff_advantage recompute that used to sit here were
            # dead compute and have been removed.

            newvalue = newvalue.view(mb_returns_norm.shape)
            v_loss_unclipped = (newvalue - mb_returns_norm)**2
            if vf_clip is not None:
                v_clipped = mb_values_norm + torch.clamp(newvalue - mb_values_norm, -vf_clip,
                                                         vf_clip)
                v_loss_clipped = (v_clipped - mb_returns_norm)**2
                v_loss = 0.5 * masked_mean(torch.max(v_loss_unclipped, v_loss_clipped), mb_part_f)
            else:
                v_loss = 0.5 * masked_mean(v_loss_unclipped, mb_part_f)

            # Rung 0 §2.2: the entropy the SAC-α dual loop and the collapse
            # floor react to must be the participating rows' entropy. Parked
            # rows are noop-masked (exactly one valid bin per head ⇒ discrete
            # entropy 0), so an unmasked mean at n_active=1 reads ~1/5 of the
            # truth and would peg alpha at the floor forever.
            current_entropy = masked_mean(entropy, flat_part)
            entropy_unmasked = entropy.mean()          # diagnostic only (losses/entropy_unmasked)

            # ── ADAPTIVE ALPHA (SAC-style Lagrangian entropy tuning) ───────
            alpha = self._log_alpha_tensor.exp()
            # Task 9A: use the scheduled target_entropy (recomputed at top of
            # this train() call) instead of the static fallback. _t9_target_entropy
            # is a Python float; .detach() on a tensor minus a float is fine —
            # autograd treats the float as a constant.
            # alpha_loss is computed UNCONDITIONALLY (the logging block below
            # accumulates it every minibatch — spec finding 7); during the
            # warm-start GRACE phase only the optimizer step is skipped, so
            # log_alpha stays at its operating point (see phase-resolution
            # comment above for why that matters).
            alpha_loss = (self._log_alpha_tensor *
                          (current_entropy - _t9_target_entropy).detach()).mean()
            if self._batch1_warmstart_phase != WS_GRACE:
                self._alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self._alpha_optimizer.step()

            effective_alpha = alpha.detach()
            if self._batch1_warmstart_phase == WS_GRACE:
                # grace: entropy pressure ceilinged (default 0.0 — pure
                # PPO+reward; the knob exists for a nonzero-alpha rerun if
                # the collapse watch fires)
                effective_alpha = torch.clamp(effective_alpha,
                                              max=float(config.get("warmstart_alpha_ceiling", 0.0)))
            # Entropy floor: prevent collapse. Gated off for the ENTIRE
            # warm-start window (grace+ramp): the BC policy lives below the
            # floor by design, and re-arming mid-ramp would jump effective
            # alpha ~1e-3 -> 0.5 in one minibatch (spec finding 2). It re-arms
            # at ramp_end — a plotted boundary.
            if _ws_floor_active and current_entropy.item() < self._entropy_floor:
                effective_alpha = torch.clamp(effective_alpha, min=0.5)
                _floor_fires += 1

            entropy_loss = -effective_alpha * current_entropy
            # ──────────────────────────────────────────────────────────────

            loss = pg_loss + config["vf_coef"] * v_loss + entropy_loss
            self.amp_context.__enter__()

            # Denormalize before writing back so advantage computation stays in raw scale
            std = (self._ret_var + 1e-8).sqrt()
            # Rung 0 §2.2: keep parked rows at exactly 0 in the value buffer —
            # the rollout wrote 0 there and the next epoch's GAE reads it.
            self.values[idx] = (newvalue.detach().float() * std + self._ret_mean) * mb_part_f

            # ── PER-HEAD ENTROPY ──────────────────────────────────────────
            with torch.no_grad():
                _dists = [torch.distributions.Categorical(logits=lgt) for lgt in logits]
                # Batch 3: head names sourced from _action_spec.ACTION_HEAD_NAMES
                # (auto-gen from cs2_types.h). Pre-Batch-3 hardcoded "aim" here;
                # now removed since aim is a continuous head emitted on a separate
                # path. zip(strict=True) catches any future drift between
                # _action_spec and the policy logits list.
                _head_names = list(ACTION_HEAD_NAMES)
                for _hi, (_hn, _hd) in enumerate(zip(_head_names, _dists, strict=True)):
                    # flat_part: _hd.entropy() is flat (S*T,), like `entropy`.
                    losses[f"entropy/{_hn}"] += masked_mean(_hd.entropy(), flat_part).item()
            losses["entropy/total"] += current_entropy.item()
            # ──────────────────────────────────────────────────────────────

            # Logging
            profile("train_misc", epoch)
            losses["policy_loss"] += pg_loss.item()
            losses["value_loss"] += v_loss.item()
            losses["entropy"] += current_entropy.item()
            # Rung 0 §2.2: the same mean WITHOUT the mask. Diagnostic only —
            # its ratio to losses/entropy is the live check that the mask is
            # actually doing something: the expected ratio is the PARTICIPATING
            # ROW FRACTION, which is ≈ n_active/TEAM_SIZE under --opponent self
            # (both teams contribute n_active rows) but ≈ n_active/(2*TEAM_SIZE)
            # under --opponent noop, where only the hero team participates.
            # Reading the self-mode number on a noop run looks like a mask that
            # is masking twice as much as it should.
            losses["entropy_unmasked"] += entropy_unmasked.item()
            losses["alpha"] += alpha.detach().item()
            losses["alpha_loss"] += alpha_loss.item()
            losses["old_approx_kl"] += old_approx_kl.item()
            losses["approx_kl"] += approx_kl.item()
            losses["clipfrac"] += clipfrac.item()
            losses["clipfrac_d"] += clipfrac_d.item()
            losses["clipfrac_c"] += clipfrac_c.item()
            losses["importance"] += masked_mean(ratio, _pm).item()
            # gh#90: count EXECUTED minibatches — the divisor for every
            # accumulated losses/* above and the per-head entropy block.
            # Incremented here (with the stats) so a future early-`continue`
            # placed above the logging block can't desync count from sums.
            _mb_run += 1

            # Learn on accumulated minibatches
            profile("learn", epoch)
            # ── Batch 3 (T5) NaN guard ─────────────────────────────────────
            # The continuous Gaussian aim head can emit non-finite μ / log_std
            # during pathological early training (e.g. an obs that drives the
            # tanh into hard saturation while σ explores LOG_STD_MAX — the
            # log-prob of a far-tail sample under near-zero σ blows up).
            # Skip optimizer.step() with a throttled stdout warning and
            # zero out grads so the next minibatch starts from a clean slate.
            # Do NOT raise — one bad minibatch shouldn't kill a run. `continue`
            # is correct here: the enclosing `for mb in range(...)` is the
            # PPO update loop. There is no nested loop between this check and
            # that for-statement (verified before landing T5).
            if not torch.isfinite(loss).all():
                _now = time.time()
                _last = getattr(self, '_last_nan_warn_t', 0.0)
                if _now - _last > 60.0:
                    print(f"[hybrid_aim NaN guard] non-finite loss "
                          f"({float(loss.detach())}); skipping optimizer step")
                    self._last_nan_warn_t = _now
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # ── TAG diagnostic hook (spec 2026-08-13 §4.2/§4.3) ────────────
            # mb0 = pre-update on-policy regime. mbL = last EXECUTED
            # minibatch: total_minibatches-1 normally, or the final mb of
            # the epoch the KL gate tripped on (the loop-top gate exits at
            # the next epoch boundary) — conditioning mbL on the gate NOT
            # tripping would select against the late-update regime it
            # exists to observe. Placed AFTER the NaN guard (never measure
            # a batch the update skips) and BEFORE loss.backward() (.grad
            # still untouched). Results stash on the TRAINER — see
            # _inject_tag_metrics for the two routing constraints.
            if config.get("tag_diagnostic", False)\
                    and epoch % max(1, int(config.get("tag_every", 5))) == 0:
                _tag_mb0 = (mb == 0)
                _tag_mbL = (mb == self.total_minibatches - 1
                            or (_kl_stop and (mb + 1) % _mbs_per_epoch == 0))
                if _tag_mb0 or _tag_mbL:
                    _tag = tag_grad_cossim(
                        self.policy,
                        mb_obs=mb_obs,
                        mb_actions=mb_actions,
                        mb_cont_actions=mb_cont_actions,
                        mb_old_logp_d=mb_old_logp_d,
                        mb_old_logp_c=mb_old_logp_c,
                        mb_advantages=mb_advantages,
                        clip_coef=clip_coef,
                        state=state,
                        mb_prio=mb_prio,
                        mb_masks=mb_masks,
                        mb_returns_norm=mb_returns_norm,
                        idx=idx,
                        mb_label="mb0" if _tag_mb0 else "mbL",
                        mb_part=mb_part_f,
                        aim_dim_mask=getattr(self.policy, "aim_dim_mask", None),
                        aim_entropy_bonus=bool(config.get("aim_entropy_bonus", True)),
                    )
                    tag_metrics = getattr(self, "_tag_metrics", None)
                    if tag_metrics is None:
                        tag_metrics = {}
                    self._tag_metrics = tag_metrics
                    self._tag_metrics.update(_tag)
                    if not _tag_mb0:
                        self._tag_metrics["tag/mbL_index"] = float(mb)
                    self._tag_metrics["tag/selfplay_active"] = float(
                        getattr(self, "_selfplay_used_past", False))
            # ──────────────────────────────────────────────────────────────
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                # Task 9C: capture pre-clip grad norm. clip_grad_norm_ returns
                # the total norm computed BEFORE clipping (PyTorch contract,
                # see torch.nn.utils.clip_grad_norm_ docs). Storing it on the
                # trainer makes it available to the log layer; the .item()
                # call forces a host sync which is fine here because the
                # caller already syncs via .item() on losses below.
                _t9_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                                               config["max_grad_norm"])
                self._batch1_grad_norm = float(_t9_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        # gh#90: normalize the accumulated losses/* sums by the EXECUTED
        # minibatch count. Must run BEFORE the scalar (non-accumulated) keys
        # below (explained_variance, ret_mean, ...) are inserted — dividing
        # those would corrupt them. minibatches_run itself is added after
        # the division for the same reason. max(_mb_run, 1) is pure belt-and-
        # braces: the epoch-boundary gate guarantees epoch 0 completes.
        for _lk in list(losses):
            losses[_lk] /= max(_mb_run, 1)
        losses["minibatches_run"] = _mb_run

        # Rung 0 §2.2 counters/absolutes — inserted AFTER the divisor (gh#90
        # trap: anything written before it is silently scaled by 1/_mb_run).
        losses["entropy_floor_fires"] = float(_floor_fires)
        losses["empty_minibatches"] = float(_empty_mb)
        losses["participating_rows"] = float(self.participating.sum().item())

        # Warm-start metrics are ABSOLUTE values — inserted after the gh#90
        # divisor loop above, alongside minibatches_run, or they'd be divided
        # by the executed-minibatch count (the exact bug class gh#90 fixed).
        if config.get("warmstart_entropy", False):
            losses["warmstart_phase"] = self._batch1_warmstart_phase
            # h0 is captured on the mode's first update, when entropy is
            # healthy (BC policy ~1.8 nats) — the >1e-9 guard exists because
            # total entropy (discrete + Gaussian differential) CAN go
            # non-positive in the collapse regime, and a non-positive
            # denominator would flip the watch's sign. If capture is ever
            # skipped, say so once instead of silently disabling the watch.
            if self._batch1_warmstart_h0 is None:
                if losses["entropy"] > 1e-9:
                    self._batch1_warmstart_h0 = float(losses["entropy"])
                else:
                    print(f"[Train] WARN warm-start: first-update entropy "
                          f"{losses['entropy']:.3f} <= 0 — h_over_h0 collapse "
                          f"watch cannot arm (will retry next update).")
            if self._batch1_warmstart_h0:
                losses["warmstart_h_over_h0"] = losses["entropy"] / self._batch1_warmstart_h0
                # collapse watch (spec finding 6): grace disables BOTH
                # anti-collapse guards (floor clamp + alpha), so shout —
                # throttled to every 20 epochs — if H halves.
                if (self._batch1_warmstart_phase == WS_GRACE and losses["warmstart_h_over_h0"] < 0.5
                        and self.epoch - self._batch1_warmstart_warn_epoch >= 20):
                    self._batch1_warmstart_warn_epoch = self.epoch
                    print(f"[Train] WARN warm-start grace: entropy at "
                          f"{losses['warmstart_h_over_h0']:.2f} of its start value "
                          f"({losses['entropy']:.3f} nats) with alpha ceilinged and the "
                          f"entropy floor disabled — collapse watch (spec finding 6).")
        # Anchor source: maintained EVERY update, unconditionally (mode may be
        # enabled on a later resume of this process in tests; cost is one float).
        self._batch1_last_entropy_mean = float(losses["entropy"])

        # Reprioritize experience
        profile("train_misc", epoch)
        if config["anneal_lr"]:
            self.scheduler.step()

        # Rung 0 §2.2: whole-buffer EV over PARTICIPATING rows only. Parked
        # rows have value 0 and advantage 0, i.e. a perfectly-predicted
        # constant — leaving them in would inflate EV toward 1 by exactly the
        # parked fraction and make the critic look healthy at n_active=1
        # regardless of what it learned.
        losses["explained_variance"] = masked_explained_variance(
            self.values.flatten(),
            advantages.flatten() + self.values.flatten(), self.participating.flatten())
        losses["ret_mean"] = self._ret_mean.item()
        losses["ret_std"] = (self._ret_var + 1e-8).sqrt().item()
        # Observe-only persist (spec 2026-08-15 §3.4). Task 8 already
        # computes `_batch1_event_oversample_fraction` once per train()
        # call; older comments that say the wandb/log layer already
        # reports it were stale. MUST sit after the gh#90 divisor loop —
        # this is a per-call scalar like ret_mean, not a minibatch sum.
        # Expected ~0.0 while #100 keeps include_step_stats_in_info=False
        # (no event mask); that zero is the production signal. The
        # KL-break harness turns the flag on, so its test must not
        # assert == 0.0.
        losses["event_oversample_fraction"] = float(
            getattr(self, "_batch1_event_oversample_fraction", 0.0))
        losses["log_alpha"] = self._log_alpha_tensor.item()

        # Task 9C: expose per-train()-call metrics on the trainer for the
        # wandb log layer. Captured here (not inside the minibatch loop)
        # because the log layer reports one value per train() call, not
        # per-minibatch — and effective_alpha / log_alpha are last-write-
        # wins after the inner loop anyway.
        # PITFALL: effective_alpha is bound inside the minibatch loop;
        # Python keeps the last bound value visible at this scope so
        # reading it here works in the happy path. Since gh#90 the
        # target_kl early-stop can only exit at an epoch boundary (epoch 0
        # always completes), so the old "break on mb=0 leaves
        # effective_alpha unbound" NameError is structurally impossible —
        # the try/except below stays as defense-in-depth only. The NaN
        # guard's `continue` can still skip the optimizer-step site, so
        # _batch1_grad_norm keeps its pre-seeded default in that edge.
        self._batch1_log_alpha = float(self._log_alpha_tensor.item())
        # effective_alpha may be unbound this call if target_kl early-broke
        # on mb=0 — leave the pre-initialised attr (set in _init_return_norm
        # block above) intact in that case rather than crashing.
        try:
            self._batch1_effective_alpha = float(effective_alpha.detach().item())
        except (NameError, UnboundLocalError):
            pass
        # Rung 0 §2.2: log the alpha the loss ACTUALLY used (post ceiling /
        # post floor clamp), not just the raw log_alpha.exp() above — with the
        # floor now counted by entropy_floor_fires, the pair says whether the
        # collapse guard is holding the run up. Absolute value, so it sits here
        # after the gh#90 divisor, and it reads the trainer attr rather than
        # the loop-local so the KL-early-break edge cannot NameError.
        losses["effective_alpha"] = float(self._batch1_effective_alpha)
        # Welford std exposure: guard with getattr+fallback so a trainer whose
        # _init_selfplay (where these get attached) did not run keeps the
        # no-selfplay code path (a legacy guard from the monkey-patch era).
        _w_combat = getattr(self, "_batch1_welford_combat", None)
        self._batch1_std_combat = (float(_w_combat.std()) if _w_combat is not None else 1.0)
        _w_obj = getattr(self, "_batch1_welford_objective", None)
        self._batch1_std_objective = (float(_w_obj.std()) if _w_obj is not None else 1.0)
        _w_pos = getattr(self, "_batch1_welford_positional", None)
        self._batch1_std_positional = (float(_w_pos.std()) if _w_pos is not None else 1.0)

        profile.end()
        logs = None
        self.epoch += 1
        # Rung 0 §2.2: global_step is in PARTICIPATING units, so it must be
        # compared against participating_timesteps (= the user's --timesteps),
        # not against total_timesteps (the TEAM_SIZE/n_active-scaled raw-row
        # budget PufferLib's epoch cap needs). .get() keeps trainers built from
        # a pre-Rung-0 config.json working. The epoch clause is load-bearing,
        # not belt-and-braces: total_epochs floor-divides, so at the defaults
        # (--timesteps 10M, --num_envs 256 ⇒ batch 163840, n_active=1 ⇒ raw
        # budget 50M) the 305 epochs PufferLib allows collect only
        # 305 × 163840 / 5 = 9,994,240 participating steps — the first clause
        # would never fire and the last epoch would never checkpoint.
        done_training = (self.global_step >= config.get("participating_timesteps",
                                                        config["total_timesteps"])
                         or self.epoch >= self.total_epochs)
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            # R0-A: episode count for the window, written INTO self.stats so
            # mean_and_log (pufferl.py mean_and_log) emits environment/episodes
            # through the logger too. DECLARED DEVIATION from spec §3 R0-A
            # ("inject after mean_and_log returns"): the logger call is inside
            # mean_and_log, so the post-hoc form would reach metrics.jsonl
            # only. `.get` — a defaultdict(list) `[]` read would insert an
            # empty list and np.mean([]) → NaN. Unit: terminal infos (rounds),
            # one per env per round regardless of n_active_per_team.
            self.stats["episodes"] = [float(len(self.stats.get("kills_t", ())))]
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % config["checkpoint_interval"] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f"Checkpoint saved at update {self.epoch}"

        return logs
