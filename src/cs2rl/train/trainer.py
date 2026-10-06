"""The cs2rl trainer: ``Cs2PuffeRL``, a ``PuffeRL`` subclass, and ``HybridAimVecEnv``.

WHAT: ``Cs2PuffeRL.__init__`` runs PufferLib's constructor, then ``_init_return_norm``,
``_init_hybrid_aim`` and ``_init_selfplay`` in that order, then creates ``_timing``. The
class overrides ``evaluate`` (the hybrid-aim, self-play rollout), ``train`` (the
return-normalised hybrid PPO update) and ``save_checkpoint`` (full-state checkpoints).
``evaluate`` and ``train`` are sequences of phase methods; their docstrings list the
phases in order. Callers wrap the vector env in ``HybridAimVecEnv`` before constructing
the trainer. ADR 0002 (docs/adr/0002-subclass-pufferl-do-not-mutate.md) is why this is
a subclass and nothing mutates a trainer instance.

WHY a module of its own and not a class inside cs2rl.train.loop: this module subclasses
``PuffeRL``, so it imports torch and pufferlib at module scope and is HEAVY by
construction. The CLI module's scope (``cs2rl.train.__main__``, which imports
``cs2rl.train.loop`` and through it ``cs2rl.train.compose`` at module level) must stay
torch-free (tests/train/test_w1_modules.py::test_cli_module_scope_stays_light: it is
what keeps ``--dump-config`` at ~1 s), so this module's one importer in src/,
``cs2rl.train.compose.build_trainer``, imports it function-locally. The test harness
builds through compose, so ``from tests._helpers import trainer_harness`` stays as light
as the CLI module (tests/train/test_w1_modules.py::test_import_train_test_harness_stays_light).
Never add ``from cs2rl.train.trainer import ...`` at the module level of ``compose`` or
``loop`` (knock-out W1-K3 in the spec: test_cli_module_scope_stays_light goes red naming
torch).

IMPORT DIRECTION: ``cs2rl.train.compose`` imports this module, inside ``build_trainer``;
this module imports nothing from ``compose``, ``loop`` or ``__main__``, at any scope.
pyproject.toml's ``cs2rl.train layers`` contract enforces that (``__main__``, ``loop``
and ``compose`` sit above ``trainer``), so there is no import cycle to order and no
``__main__`` aliasing to get right.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pufferlib
import torch
from pufferlib.pufferl import PuffeRL, compute_puff_advantage

from cs2rl.policy import LOG_STD_MAX, _hybrid_sample_logits
from cs2rl.spec.action import ACTION_HEAD_NAMES, ACTION_HEAD_SIZES, ACTION_MASK_DIM, AIM_DIM
from cs2rl.train.entropy import WS_GRACE, WS_OFF, warmstart_entropy_state
from cs2rl.train.resume import _atomic_save_state_dict, collect_train_state
from cs2rl.train.rewards import WelfordStd, process_step_rewards
from cs2rl.train.update import (
    _hybrid_ppo_loss,
    _scheduled_target_entropy,
    masked_explained_variance,
    masked_mean,
    masked_value_loss,
    tag_grad_cossim,
)

# Replay-probability multiplier for segments that hold a bomb-plant event (Task 8);
# see Cs2PuffeRL._segment_probs.
EVENT_OVERSAMPLE_FACTOR = 4.0


@dataclass(frozen=True)
class _RolloutStep:
    """One chunk's sampled actions; the opponent overrides edit these tensors in place."""
    state: dict                        # the forward's state dict; holds the new LSTM state
    action: torch.Tensor
    cont_action: torch.Tensor
    logprob: torch.Tensor              # joint: logprob_d + logprob_c
    logprob_d: torch.Tensor
    logprob_c: torch.Tensor
    value: torch.Tensor


@dataclass
class _UpdateState:
    """What one ``train()`` call carries across its minibatches.

    ``sums`` holds per-minibatch metric SUMS. Only ``_accumulate_minibatch`` writes
    it, and it counts ``minibatches_run`` in the same place; ``_finish_update``
    divides the sums by that count and writes every per-call absolute into the
    resulting dict, never into ``sums``.
    """
    target_entropy: float
    floor_active: bool
    anneal_beta: float
    target_kl: float | None
    mbs_per_epoch: int
    kl_stop: bool = False
    sums: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    minibatches_run: int = 0
    empty_minibatches: int = 0
    floor_fires: int = 0
    # The latest minibatch's advantages; the explained-variance metric reads them.
    advantages: torch.Tensor | None = None
    # The latest non-empty minibatch's effective alpha.
    effective_alpha: torch.Tensor | None = None


@dataclass(frozen=True)
class _Minibatch:
    """One sampled minibatch: rollout rows gathered at ``idx`` and its participation.

    ``part`` is the BOOL [S, T] mask, for indexing. ``part_f`` [S, T] and
    ``flat_part`` (S*T,) are the FLOAT weights the masked reductions take: pass
    ``flat_part`` for flat tensors, or the [S, T] weight silently broadcasts.
    """
    idx: torch.Tensor
    prio: torch.Tensor                 # importance weight from the (boosted) replay probability
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor             # joint rollout logprob, for the diagnostics
    terminals: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    cont_actions: torch.Tensor
    old_logp_d: torch.Tensor
    old_logp_c: torch.Tensor
    masks: torch.Tensor
    part: torch.Tensor
    part_f: torch.Tensor
    flat_part: torch.Tensor


@dataclass(frozen=True)
class _RatioStats:
    """One minibatch's joint importance ratio and its masked diagnostics."""
    ratio: torch.Tensor
    part: torch.Tensor                 # participation weight in the joint ratio's [S, T] shape
    old_approx_kl: torch.Tensor
    approx_kl: torch.Tensor
    clipfrac: torch.Tensor
    clipfrac_d: torch.Tensor
    clipfrac_c: torch.Tensor


@dataclass(frozen=True)
class _MinibatchStep:
    """One minibatch's loss and the terms the later phases of ``train()`` read."""
    batch: _Minibatch
    obs: torch.Tensor                  # batch.obs as the policy saw it
    state: dict                        # the forward's state dict; TAG re-runs with it
    returns_norm: torch.Tensor
    loss: torch.Tensor
    pg_loss: torch.Tensor
    v_loss: torch.Tensor
    newvalue: torch.Tensor
    logits: list
    entropy: torch.Tensor              # mean over participating rows
    entropy_unmasked: torch.Tensor
    alpha: torch.Tensor                # raw exp(log_alpha), before this minibatch's alpha step
    alpha_loss: torch.Tensor
    ratio: _RatioStats


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
    """The cs2rl trainer: PuffeRL with return-normalised hybrid PPO, self-play, full checkpoints.

    Construction composes the state in a fixed order (``__init__``); ``evaluate`` and
    ``train`` replace PufferLib's loops and ``save_checkpoint`` its checkpoint. The
    hybrid (discrete + continuous) action transport lives in ``HybridAimVecEnv``.

    Parameters beyond PuffeRL's ``(config, vecenv, policy, logger=None)``:

    The caller is ``cs2rl.train.compose.build_trainer``, for a CLI run and for a test
    trainer alike; it builds every argument below.

    cont_action_view_main : np.ndarray or None
        Main-process view of the continuous-action shared array. A CLI run passes it on
        both backends; a test trainer (``env_role="harness"``) passes None, and the
        Serial per-env step wrapper in ``HybridAimVecEnv`` carries the aim.
    mask_view_main : np.ndarray or None
        Main-process view of the action-mask shared array (F8). None: the rollout samples
        unmasked and ``action_masks`` keeps its all-ones default.
    participating_rows : np.ndarray or None
        Static per-run participation vector (Rung 0 §2.2; None: every row participates),
        from ``build_participating_rows``.
    self_play_mgr : SelfPlayManager
        From ``build_selfplay_manager``, or given by the caller. The pool is read only at
        evaluate() time, so a CLI warm start seeds it after construction.

    PITFALLS
    - State a phase method reads is created here, before the first
      evaluate()/train()/save_checkpoint() call. Three attributes are the exception:
      created later and read through getattr fallbacks, ``_selfplay_used_past`` (set by
      evaluate), ``_tag_metrics`` (reset by the epoch loop, filled by TAG minibatches)
      and ``_last_nan_warn_t`` (set by the first NaN warning).
    - ``_timing`` is created here because the epoch loop (cs2rl.train.loop._run_epochs)
      assigns INTO it and the [Timing] print reads it.
    - After this constructor returns, build_trainer pins the shared-memory owners
      (``_action_mask_shm``, and ``_cont_action_shm`` when it allocated one), and a CLI
      run's train() sets ``logger.run_id``, the weight decay and the aim-σ param group.
      The constructor reads none of them.
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
    # cs2rl.train.compose.build_trainer pins these shared-memory owners after
    # construction (the continuous one only for a CLI run); the trainer reads their
    # numpy views.
    _action_mask_shm: object
    _cont_action_shm: object

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
        # PufferLib can raise AFTER starting Utilization (e.g. dashboard output
        # hits a closed pipe), so protect the base constructor too. A failed
        # constructor never takes vecenv ownership from its caller and must not
        # checkpoint a partially initialized trainer.
        try:
            super().__init__(config, vecenv, policy, logger=logger)
            self._init_return_norm()
            self._init_hybrid_aim(cont_action_view_main, mask_view_main, participating_rows)
            self._init_selfplay(self_play_mgr)
            # Per-epoch wall-clock, measured around the evaluate()/train() calls in
            # cs2rl.train.loop._run_epochs. The dict is created here because the loop
            # assigns INTO it and the [Timing] print reads it, so it has to exist
            # before the first epoch.
            self._timing = {"collect_ms": 0.0, "update_ms": 0.0}
        except BaseException:
            # Early base-constructor failures may precede thread creation.
            utilization = getattr(self, "utilization", None)
            if utilization is not None:
                utilization.stop()
            raise

    def close_resources(self):
        """Release vector/thread without publishing an unaccepted setup/resume state.

        PuffeRL.close also publishes a checkpoint. Before setup is accepted, its
        resource operations are safe; stop Utilization even if vector close fails.
        """
        try:
            self.vecenv.close()
        finally:
            if not self.utilization.stopped:
                self.utilization.stop()

    def close(self):
        """Keep PufferLib's shutdown/save contract and stop its thread on failure.

        Upstream closes the vector before stopping Utilization. A vector-close
        failure must not strand that non-daemon thread. Reuse upstream's close
        and only supply the missing stop when it has not already happened.
        """
        try:
            return super().close()
        finally:
            if not self.utilization.stopped:
                self.utilization.stop()

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
        """Return-normalisation and adaptive-entropy state for ``train()``.

        Asserts the BPTT segment invariant (gh#85), then creates the running return
        stats (``_ret_mean/_ret_var/_ret_count``), the SAC-style ``_log_alpha_tensor``
        and its ``_alpha_optimizer``, the entropy ceiling and floor, and the
        entropy-schedule and warm-start attributes the update reads and writes.
        ``tests/train/test_trainer_composition.py`` derives the constructor surface
        from these stores.
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

        # Running stats of the returns the value head regresses. Those returns are
        # built from normalised, symlog'd rewards when the infos carry step_stats
        # (the test harness) and from raw rewards otherwise (production until #100;
        # see _process_rewards).
        # `_update_return_stats` builds its batch-count tensor on `_ret_device`.
        # PITFALL: these three tensors are updated IN PLACE (.copy_()), as
        # restore_train_state restores them, so a reference taken to one stays live.
        device = self.config["device"]
        self._ret_device = device
        self._ret_mean = torch.zeros(1, device=device)
        self._ret_var = torch.ones(1, device=device)
        self._ret_count = torch.zeros(1, device=device)

        # ── ADAPTIVE ENTROPY (Lagrangian / SAC-style alpha) ────────────────────
        # max_entropy: the sum of the discrete heads' maximum entropies (log N_i each)
        # plus the Gaussian's 0.5·log(2πe·σ²) per live aim dim at the σ cap — a
        # ceiling, since every forward clamps σ to it. It follows the run (R0-E,
        # #131): the σ cap (policy.aim_log_std_max), the live aim dims
        # (aim_dim_mask.sum(): 1 when pitch is pinned) and the entropy-bonus switch
        # (0 continuous entropy when the Gaussian is excluded from the objective, so
        # the target and floor track the discrete heads only). getattr defaults keep
        # pre-R0-E policies/configs working.
        max_entropy_discrete = sum(np.log(n) for n in ACTION_HEAD_SIZES)
        _cap = float(getattr(self.policy, "aim_log_std_max", LOG_STD_MAX))
        _n_aim = float(getattr(self.policy, "aim_dim_mask", torch.ones(AIM_DIM)).sum())
        _bonus = bool(self.config.get("aim_entropy_bonus", True))
        max_entropy_continuous = (_n_aim * 0.5 * np.log(2 * np.pi * np.e * np.exp(_cap)**2))\
            if _bonus else 0.0
        max_entropy = max_entropy_discrete + max_entropy_continuous
        # Collapse threshold. tests/train/test_warmstart_entropy_trainer.py sets it
        # directly to force the floor-clamp arm.
        self._entropy_floor = 0.3 * max_entropy

        log_alpha = torch.tensor([math.log(0.1)], requires_grad=True, device=device)
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=1e-4)
        # R0-C (#134): the train_state.pt sidecar reads both from these attributes.
        # Restore MUST be in place (`_log_alpha_tensor.data.copy_`) — rebinding the
        # attribute would leave alpha_optimizer's param list on the old tensor.
        self._log_alpha_tensor = log_alpha
        self._alpha_optimizer = alpha_optimizer

        # The entropy target follows target_entropy_schedule and is recomputed at the
        # start of every update (_prepare_entropy_update). This seed runs at
        # construction, where global_step is still 0; a resumed trainer then takes
        # the saved value from the train_state.pt sidecar (resume._WARMSTART_ATTRS).
        # _log_alpha_reset_done makes the one-time reset of log_alpha to
        # log(ent_coef) (first update after construction) idempotent; later updates
        # leave log_alpha to the SAC dual loop.
        self._max_entropy = float(max_entropy)
        self._log_alpha_reset_done = False
        self._current_target_entropy = _scheduled_target_entropy(self.config, self.global_step,
                                                                 float(max_entropy))
        # Defaults for the trainer attributes the update refreshes (utof/cs2rl#16):
        # _effective_alpha keeps its value through an update whose minibatches were
        # all empty, _grad_norm through one in which no optimizer step ran.
        self._effective_alpha = float(self.config["ent_coef"])
        self._grad_norm = 0.0

        # ── Warm-start entropy mode state (spec 2026-08-01) ────────────────────
        # h_anchor: mean policy entropy captured at grace end (None until then);
        # last_entropy_mean: previous update's losses["entropy"] — the ONLY valid
        # anchor source (there is no entropy EMA in this codebase, and
        # trainer.losses is refreshed only inside the throttled log-flush block, so
        # it can be several updates stale — spec finding 4).
        # h0: first update's mean entropy, denominator of warmstart_h_over_h0
        # (grace collapse watch — with the floor disabled AND alpha~0 the run
        # has no anti-collapse guard, spec finding 6).
        self._warmstart_h_anchor = None
        self._warmstart_h0 = None
        self._warmstart_phase = WS_OFF
        self._last_entropy_mean = None
        self._warmstart_warn_epoch = -10**9
        print("[Train] Value target normalization enabled (running mean/std of returns).")

    def _update_return_stats(self, returns_flat):
        """Welford-merge one flat batch of returns into ``_ret_mean/_ret_var/_ret_count``.

        In place (``.copy_()``): the three tensors are also what ``collect_train_state``
        saves and ``restore_train_state`` restores, so their identity must not change.
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
        mean/var are updated from the PARTICIPATING rows ONLY: parked rows are
        never value targets (their value loss is masked), so their returns must
        not set the scale of the normalized value targets. None ⇒ all rows
        (pre-Rung-0 behaviour, bit-identical).
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
        """Self-play state for evaluate(). Called from __init__ after ``_init_hybrid_aim``
        and before ``_timing``.

        Stores the manager, the past policy's own LSTM state (the same dict structure as
        ``self.lstm_h``: keyed by agent-batch start ``i*n``, one ``(agents_per_batch,
        hidden_size)`` tensor per chunk, so ``evaluate`` can index it by ``env_id.start``),
        and the reward-channel statistics, event flags and scratch buffer.
        ``process_step_rewards``
        updates the three ``WelfordStd`` and reads their ``std()``; ``train`` copies the
        three ``std()`` values into ``_std_*`` at the end of each call.
        ``process_step_rewards`` writes each env's channel sum into
        ``_reward_scratch`` when step_stats exists, or a 0.0 placeholder
        when it does not, then reads the scratch buffer for one host-to-device copy;
        ``evaluate`` replaces the buffer with a longer one when the info list (at most one
        entry per env) is longer than it. ``process_step_rewards`` only sets rows of
        ``_current_segment_has_event`` to True (bomb planted this tick); ``evaluate``
        reads those rows into ``_event_mask`` and clears them at the segment
        boundary. Apart from the zero fill here, ``_event_mask`` is written only by
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
        self._welford_combat = WelfordStd(prior_std=1.0, min_count=1000)
        self._welford_objective = WelfordStd(prior_std=1.0, min_count=1000)
        self._welford_positional = WelfordStd(prior_std=1.0, min_count=1000)
        _dev = self.config["device"]
        self._event_mask = torch.zeros(self.segments, dtype=torch.bool, device=_dev)
        self._current_segment_has_event = torch.zeros(self.total_agents,
                                                      dtype=torch.bool,
                                                      device=_dev)
        self._reward_scratch = np.empty(0, dtype=np.float32)
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
        """Collect one rollout of ``segments`` buffer rows; PufferLib's loop plus cs2rl's.

        Per chunk of agent rows the vector env returns: count participating steps,
        ``_sample_actions`` (policy forward and hybrid sample), ``_process_rewards``,
        the opponent overrides (``_play_past_opponent`` when this epoch drew a past
        policy, then ``_freeze_statue_opponents`` under ``--opponent noop``),
        ``_store_step``, ``_collect_infos``, and send the (discrete, continuous)
        action pair to ``HybridAimVecEnv``.

        The past policy's actions and logprobs replace the current policy's on the
        opponent rows, so those rows' pi_old in the PPO update is the past policy's
        and the importance ratio stays well defined.
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

        past_policy = self._draw_past_policy()

        self.full_rows = 0
        while self.full_rows < self.segments:
            profile("env", epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile("eval_misc", epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)
            # global_step counts PARTICIPATING agent-steps, so --timesteps means the
            # same at any n_active_per_team. `mask` is the chunk's live-agent mask.
            self.global_step += int(
                (np.asarray(mask, dtype=bool) & self._participating_rows_np[env_id]).sum())

            profile("eval_copy", epoch)
            o = torch.as_tensor(o)
            o_device = o.to(dev)
            r = torch.as_tensor(r).to(dev)
            d = torch.as_tensor(d).to(dev)

            # This chunk's C-computed action masks (env_id indexes agent rows, as the
            # shm does). `!= 0` converts AND copies: the next worker step overwrites
            # the shm. None: a trainer without the mask view samples unmasked.
            mask_view = self._action_mask_view_main
            action_mask = None
            if mask_view is not None:
                action_mask = torch.as_tensor(mask_view[env_id]).to(dev) != 0

            profile("eval_forward", epoch)
            with torch.no_grad(), self.amp_context:
                step = self._sample_actions(o_device, r, d, env_id, mask, action_mask)
                r = self._process_rewards(info, r)
                if past_policy is not None:
                    self._play_past_opponent(step, past_policy, o_device, d, env_id, action_mask)
                if self._self_play_mgr.opponent_mode == "noop":
                    self._freeze_statue_opponents(step, o_device.shape[0])

            profile("eval_copy", epoch)
            with torch.no_grad():
                self._store_step(step, o, o_device, r, d, env_id, action_mask)
                action = step.action.cpu().numpy()

            profile("eval_misc", epoch)
            self._collect_infos(info)

            profile("env", epoch)
            self.vecenv.send((action, step.cont_action))

        profile("eval_misc", epoch)
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=dev, dtype=torch.int32)
        self.ep_lengths.zero_()
        profile.end()
        return self.stats

    def _draw_past_policy(self):
        """This epoch's past-policy opponent, or None; decided once per ``evaluate()``.

        Sets ``_selfplay_used_past`` (the self_play/used_past metric and the TAG
        diagnostic read it: on those epochs one team's rows are off-policy) and zeroes
        the past policy's own LSTM state.
        """
        past_policy = None
        if self._self_play_mgr.should_use_past():
            past_policy = self._self_play_mgr.load_past_policy(self.config["device"], self.vecenv)
        self._selfplay_used_past = past_policy is not None
        if past_policy is not None:
            for k in self._past_lstm_h:
                self._past_lstm_h[k].zero_()
                self._past_lstm_c[k].zero_()
        return past_policy

    def _sample_actions(self, o_device, r, d, env_id, mask, action_mask) -> _RolloutStep:
        """Current-policy forward and hybrid sample for one chunk (no grad, under autocast).

        The stored joint logprob is ``logprob_d + logprob_c``: the KL/clipfrac
        diagnostics read it, and the halves feed the per-factor clip in
        ``_hybrid_ppo_loss``. ``state`` carries the LSTM state the forward wrote back.
        """
        state = dict(reward=r, done=d, env_id=env_id, mask=mask)
        if self.config["use_rnn"]:
            state["lstm_h"] = self.lstm_h[env_id.start]
            state["lstm_c"] = self.lstm_c[env_id.start]
        logits, mu_aim, log_std_aim, value = self.policy.forward_eval(o_device, state)
        action, cont_action, logprob_d, logprob_c, _, _ = _hybrid_sample_logits(
            (logits, mu_aim, log_std_aim, value),
            max_turn_speed=self.policy.max_turn_speed.item(),
            mask=action_mask,
            aim_dim_mask=getattr(self.policy, "aim_dim_mask", None),
        )
        # A sum over the heads, so the stub types it int | Tensor; it is a Tensor.
        logprob_d = cast(torch.Tensor, logprob_d)
        return _RolloutStep(state=state,
                            action=action,
                            cont_action=cont_action,
                            logprob=logprob_d + logprob_c,
                            logprob_d=logprob_d,
                            logprob_c=logprob_c,
                            value=value)

    def _process_rewards(self, info, r):
        """Per-channel reward normalisation and symlog for one chunk (``process_step_rewards``).

        Only infos that carry ``step_stats`` are processed. With
        include_step_stats_in_info on (the test harness), Cs2Env returns exactly one
        such info per env per tick, so info ``e`` is env ``e`` of the chunk and all its
        agent rows get that env's reward; process_step_rewards also ORs each env's
        bomb-plant tick into ``_current_segment_has_event``. With it off (production
        until #100), no info carries step_stats and every raw reward passes through.
        The scratch buffer grows to the longest info list seen. Read
        ``process_step_rewards``'s docstring before touching the arithmetic.
        """
        agents_per_env = self.vecenv.driver_env.num_agents
        if self._reward_scratch.shape[0] < len(info):
            self._reward_scratch = np.empty(len(info), dtype=np.float32)
        return process_step_rewards(
            info,
            r,
            agents_per_env,
            self._welford_combat,
            self._welford_objective,
            self._welford_positional,
            self._reward_scratch,
            current_segment_has_event=self._current_segment_has_event,
        )

    def _play_past_opponent(self, step, past_policy, o_device, d, env_id, action_mask):
        """Replace the opponent rows of ``step`` with the past policy's actions and logprobs.

        The past policy runs on its own LSTM state (``_past_lstm_h`` / ``_past_lstm_c``,
        keyed like ``lstm_h``). The continuous action and both logprob halves are
        replaced too, so ``_hybrid_ppo_loss`` sees consistent rows. Writes cast to the
        destination dtype: autocast may produce fp16.
        """
        opp_mask = self._self_play_mgr.get_opponent_mask(o_device.shape[0], self.config["device"])
        opp_idx = torch.where(opp_mask)[0]
        past_h = self._past_lstm_h[env_id.start]
        past_c = self._past_lstm_c[env_id.start]
        past_state = {
            "done": d[opp_mask],
            "lstm_h": past_h[opp_mask],
            "lstm_c": past_c[opp_mask],
        }
        opp_logits, opp_mu, opp_log_std, _opp_value = past_policy.forward_eval(
            o_device[opp_mask], past_state)
        (opp_action, opp_cont_action, opp_logprob_d, opp_logprob_c, _, _) = _hybrid_sample_logits(
            (opp_logits, opp_mu, opp_log_std, None),
            max_turn_speed=past_policy.max_turn_speed.item(),
            mask=action_mask[opp_mask] if action_mask is not None else None,
            aim_dim_mask=getattr(past_policy, "aim_dim_mask", None),
        )
        opp_logprob_d = cast(torch.Tensor, opp_logprob_d)              # as in _sample_actions
        opp_logprob = opp_logprob_d + opp_logprob_c

        past_h[opp_mask] = past_state["lstm_h"].to(past_h.dtype)
        past_c[opp_mask] = past_state["lstm_c"].to(past_c.dtype)

        step.action[opp_idx] = opp_action.to(step.action.dtype)
        step.logprob[opp_idx] = opp_logprob.to(step.logprob.dtype)
        step.cont_action[opp_idx] = opp_cont_action.to(step.cont_action.dtype)
        step.logprob_d[opp_idx] = opp_logprob_d.to(step.logprob_d.dtype)
        step.logprob_c[opp_idx] = opp_logprob_c.to(step.logprob_c.dtype)

    def _freeze_statue_opponents(self, step, batch_n):
        """``--opponent noop``: the opponent rows of ``step`` stand still (Rung 1a T3).

        Bin 0 of every discrete head is the no-op (``move_dir == 0`` is stationary) and
        the C action mask never masks it out, so this cannot pick an illegal action; a
        zero continuous action keeps the spawn orientation. The stored logprobs (the
        update's pi_old) become 0 rather than log-probs of actions never taken; every
        loss masks these non-participating rows anyway. The value becomes 0 so the
        statue's critic output never bootstraps GAE; the participation scatter in
        ``_store_step`` zeroes it as well, and either alone suffices.

        ``evaluate()`` calls this whenever the mode is noop, after the past-policy
        override (so the statue would win if both were live; noop requires
        p_past = 0) and before the rows are stored and sent.
        """
        opp_idx = torch.where(self._self_play_mgr.get_opponent_mask(batch_n,
                                                                    self.config["device"]))[0]
        step.action[opp_idx] = 0
        step.cont_action[opp_idx] = 0
        step.logprob[opp_idx] = 0
        step.logprob_d[opp_idx] = 0
        step.logprob_c[opp_idx] = 0
        step.value[opp_idx] = 0

    def _store_step(self, step, o, o_device, r, d, env_id, action_mask):
        """Write one chunk into the rollout buffers at each row's current segment slot."""
        cfg = self.config
        if cfg["use_rnn"]:
            self.lstm_h[env_id.start] = cast(torch.Tensor, step.state["lstm_h"])
            self.lstm_c[env_id.start] = cast(torch.Tensor, step.state["lstm_c"])

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

        self.actions[batch_rows, seq_pos] = step.action
        self.logprobs[batch_rows, seq_pos] = step.logprob
        # The PPO update reads these at the same rows; a missed write would feed
        # zeros to _hybrid_ppo_loss as the continuous action and old logprob halves.
        self.cont_actions[batch_rows, seq_pos] = step.cont_action
        self.logprobs_d[batch_rows, seq_pos] = step.logprob_d
        self.logprobs_c[batch_rows, seq_pos] = step.logprob_c
        # The update recomputes logprobs under the masks the sampler used. Unmasked
        # runs keep the buffer's all-ones default, which masks nothing.
        if action_mask is not None:
            self.action_masks[batch_rows, seq_pos] = action_mask
        self.rewards[batch_rows, seq_pos] = r
        self.terminals[batch_rows, seq_pos] = d.float()
        # Participation is written whether or not the run is masked: the buffer is
        # zero-initialised, and the training loop asserts participating.any() after
        # every rollout. Parked rows' values are zeroed so GAE never bootstraps
        # through a parked row's value (the masked reductions in the update are
        # what make training correct; this is defence in depth).
        part_rows = self._participating_rows[env_id]
        self.participating[batch_rows, seq_pos] = part_rows
        self.values[batch_rows, seq_pos] = step.value.flatten() * part_rows.to(step.value.dtype)

        self._advance_segments(env_id, seq_pos)

    def _advance_segments(self, env_id, seq_pos):
        """Advance the chunk's rows one step; rows that filled a segment move to fresh ones.

        PITFALL: the live event flags are flushed into ``_event_mask`` at the rows' OLD
        segment indices, so the flush must run before ``ep_indices`` is reassigned;
        after it, the write would mark the freshly allocated segments instead.
        """
        self.ep_lengths[env_id] += 1
        if seq_pos + 1 >= self.config["bptt_horizon"]:
            num_full = env_id.stop - env_id.start
            old_seg_indices = self.ep_indices[env_id].clone().long()
            self._event_mask[old_seg_indices] = (self._current_segment_has_event[env_id])
            self._current_segment_has_event[env_id] = False
            self.ep_indices[env_id] = (self.free_idx +
                                       torch.arange(num_full, device=self.config["device"]).int())
            self.ep_lengths[env_id] = 0
            self.free_idx += num_full
            self.full_rows += num_full

    def _collect_infos(self, info):
        """Append every info value to ``self.stats[key]``; ``mean_and_log`` means each list.

        Same rule as PufferLib 3.0's ``evaluate``: a list or tuple extends, any other
        value appends, and an ndarray value is dropped (upstream converts it with
        ``tolist()`` and then stores nothing).
        """
        for i in info:
            for k, v in pufferlib.unroll_nested_dict(i):
                if isinstance(v, np.ndarray):
                    continue
                if isinstance(v, (list, tuple)):
                    self.stats[k].extend(v)
                else:
                    self.stats[k].append(v)

    def _prepare_entropy_update(self) -> tuple[float, bool]:
        """Resolve this update's entropy target and whether its floor is active.

        Called once per update, by ``_begin_update``: global_step is constant during
        train(). Keep the checkpointed latches and optimizer tensor on the trainer; a
        restored reset flag prevents repeating the initial in-place alpha reset.
        Schedule math stays in the existing pure helpers. The minibatch alpha terms
        and step are ``_entropy_terms`` / ``_step_alpha``; the entropy history and
        warm-start metrics are written by ``_finish_update``.
        """
        config = self.config
        # Read the live entropy bound (including this run's aim settings), then
        # mirror the scheduled target even when GRACE consumes no alpha target.
        target_entropy = _scheduled_target_entropy(config, self.global_step, self._max_entropy)
        self._current_target_entropy = float(target_entropy)

        # Reset before phase resolution, retaining Adam's original parameter.
        # Warm-start continuity comes from target == anchor at release, not
        # from parking log_alpha far below its operating point.
        if not self._log_alpha_reset_done:
            with torch.no_grad():
                self._log_alpha_tensor.fill_(math.log(config["ent_coef"]))
            self._log_alpha_reset_done = True

        _ws_enabled = bool(config.get("warmstart_entropy", False))
        _ws_floor_active = True
        if _ws_enabled:
            _ws_grace = int(config.get("warmstart_grace_steps", 5_000_000))
            if (self._warmstart_h_anchor is None and self.global_step >= _ws_grace
                    and self._last_entropy_mean is not None):
                # Latch only a finite completed-update mean; retry next update
                # rather than poison every target/alpha loss through the ramp.
                if math.isfinite(self._last_entropy_mean):
                    self._warmstart_h_anchor = float(self._last_entropy_mean)
                else:
                    print(f"[Train] WARN warm-start: non-finite entropy mean "
                          f"{self._last_entropy_mean} at grace end — "
                          f"anchor capture skipped, staying in GRACE.")
            _ws = warmstart_entropy_state(
                self.global_step,
                grace_steps=_ws_grace,
                ramp_steps=int(config.get("warmstart_ramp_steps", 10_000_000)),
                h_anchor=self._warmstart_h_anchor,
                base_target=(config.get("entropy_target_base_frac", 0.35) * self._max_entropy))
            self._warmstart_phase = _ws.phase
            _ws_floor_active = _ws.floor_active
            if _ws.phase != WS_OFF and _ws.target is not None:
                # Override the consumed target AND its logging mirror. OFF also
                # returns a target, but must hand back to the normal schedule:
                # when grace+ramp ends before normal warmup, that target can
                # still be above base_target. Retaining that handoff is deliberate.
                target_entropy = _ws.target
                self._current_target_entropy = float(_ws.target)
        else:
            # Disabling the mode in-process must release a previously frozen
            # alpha optimizer, even if the last update was still in GRACE.
            self._warmstart_phase = WS_OFF
        return target_entropy, _ws_floor_active

    def train(self):
        """One PPO update over the rollout buffer, then PufferLib's log/checkpoint tail.

        Replaces ``PuffeRL.train`` and never calls it: the stock loop cannot unpack
        the policy's 4-tuple output. Phases, in order:

        1. ``_begin_update``: per-call constants, the entropy schedule, the KL gate.
        2. Per minibatch: ``_minibatch_loss`` (sample, normalise returns, forward and
           loss, all inside the lexical autocast scope); ``_step_alpha``, before the
           policy backward; ``_write_back_values``; ``_accumulate_minibatch``; the NaN
           guard; ``_record_tag``; ``_step_policy``.
        3. ``_finish_update``: means over the executed minibatches plus the per-call
           absolutes, and the per-update trainer attributes.
        4. ``_log_and_checkpoint``: the throttled log flush and the checkpoint.

        The stock ``@record`` decorator is not re-applied: nothing runs under torchrun.
        """
        self.profile("train", self.epoch)
        update = self._begin_update()
        for mb in range(self.total_minibatches):
            if update.kl_stop and mb % update.mbs_per_epoch == 0:
                break                  # a KL trip ends the update at an epoch boundary (gh#90)
            self.profile("train_misc", self.epoch, nest=True)
            step = self._minibatch_loss(update)
            if step is None:
                continue               # no participating row: counted, nothing to train on
            self._step_alpha(step)
            self._write_back_values(step)
            self._accumulate_minibatch(update, step)
            self.profile("learn", self.epoch)
            if not torch.isfinite(step.loss).all():
                self._skip_nonfinite_step(step.loss)
                continue
            self._record_tag(update, mb, step)
            self._step_policy(mb, step.loss)
        losses = self._finish_update(update)
        self.profile.end()
        return self._log_and_checkpoint(losses)

    def _begin_update(self) -> _UpdateState:
        """Per-call constants: the replay-priority beta, the entropy target, the KL gate.

        The KL early stop (gh#90) is gated to update-epoch boundaries: a trip finishes
        the current epoch, as standard PPO does, so epoch 0 always runs whole. The
        stride floor-divides with a >= 1 clamp because ``total_minibatches`` need not
        divide ``update_epochs`` evenly (7 minibatches over 3 epochs).
        """
        config = self.config
        b0 = config["prio_beta0"]
        anneal_beta = b0 + (1 - b0) * config["prio_alpha"] * self.epoch / self.total_epochs
        self.ratio[:] = 1

        target_entropy, floor_active = self._prepare_entropy_update()

        # The raw event-segment fraction, a per-call metric. Masked to participating
        # segments (participating[:, 0] is the per-segment flag) so parked rows do not
        # dilute it at n_active < 5.
        self._event_oversample_fraction = float(
            masked_mean(self._event_mask.float(), self.participating[:, 0].float()))

        update_epochs = max(1, int(config.get("update_epochs", 1)))
        return _UpdateState(
            target_entropy=target_entropy,
            floor_active=floor_active,
            anneal_beta=anneal_beta,
            target_kl=config.get("target_kl", None),
            mbs_per_epoch=max(1, self.total_minibatches // update_epochs),
        )

    def _minibatch_loss(self, update: _UpdateState) -> _MinibatchStep | None:
        """Sample one minibatch and build its loss inside the lexical autocast scope.

        Returning closes the scope, so the backward passes ``train()`` runs next, an
        empty-minibatch skip and an exception all restore the caller's autocast state.
        Returns None, counted in ``empty_minibatches``, when no sampled row participates.
        """
        with self.amp_context:
            update.advantages = self._compute_advantages()
            self.profile("train_copy", self.epoch)
            batch = self._sample_minibatch(update.advantages, update.anneal_beta)
            if batch is None:
                update.empty_minibatches += 1
                return None
            return self._loss_terms(update, batch)

    def _compute_advantages(self) -> torch.Tensor:
        """GAE / V-trace advantages over the whole buffer, from the current values and ratios.

        Recomputed every minibatch, as upstream does: earlier minibatches of this update
        have rewritten ``values`` and ``ratio`` at their rows.
        """
        config = self.config
        advantages = torch.zeros(self.values.shape, device=config["device"])
        return compute_puff_advantage(
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

    def _segment_probs(self, advantages) -> torch.Tensor:
        """Replay probability per segment: advantage priority, event segments boosted.

        Segments that hold a bomb-plant event (``_event_mask``) get
        ``EVENT_OVERSAMPLE_FACTOR`` times their probability, so rare decisive
        transitions are replayed more. The importance weight
        (``_Minibatch.prio``, ``(segments * p) ** -beta``) is computed from these
        boosted probabilities, so it corrects for the boost as it does for the
        advantage priority (fully at beta = 1, the prio_beta0 default). The boost
        multiplies the smoothed probabilities, 1e-6 floor included, then renormalises,
        so the non-event segments keep their relative probabilities. No segment is
        marked in production while include_step_stats_in_info is off.

        KNOWN LIMIT: ``adv`` also sums non-participating segments, whose advantages are
        non-zero when the infos carry step_stats or, under ``--opponent noop``, on the
        statue rows. At the ``prio_alpha`` default 0.0 every segment weighs the same;
        mask them out of ``adv`` before prioritised replay (``prio_alpha > 0``) is ever
        enabled.
        """
        adv = advantages.abs().sum(axis=1)
        prio_weights = torch.nan_to_num(adv**self.config["prio_alpha"], 0, 0, 0)
        prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
        event_mask = self._event_mask
        if event_mask.any():
            boosted = prio_probs.clone()
            boosted[event_mask] *= EVENT_OVERSAMPLE_FACTOR
            prio_probs = boosted / boosted.sum()
        return prio_probs

    def _sample_minibatch(self, advantages, anneal_beta) -> _Minibatch | None:
        """Draw ``minibatch_segments`` segments by replay priority and gather their rows.

        Returns None when no drawn row participates (every drawn segment parked):
        every reduction would be 0/0.
        """
        prio_probs = self._segment_probs(advantages)
        idx = torch.multinomial(prio_probs, self.minibatch_segments)
        part = self.participating[idx]
        part_f = part.to(torch.float32)
        if part_f.sum().item() == 0:
            return None
        values = self.values[idx]
        return _Minibatch(
            idx=idx,
            prio=(self.segments * prio_probs[idx, None])**-anneal_beta,
            obs=self.observations[idx],
            actions=self.actions[idx],
            logprobs=self.logprobs[idx],
            terminals=self.terminals[idx],
            values=values,
            returns=advantages[idx] + values,
            advantages=advantages[idx],
            cont_actions=self.cont_actions[idx],
            old_logp_d=self.logprobs_d[idx],
            old_logp_c=self.logprobs_c[idx],
            masks=self.action_masks[idx],
            part=part,
            part_f=part_f,
            flat_part=part_f.reshape(-1),
        )

    def _loss_terms(self, update: _UpdateState, batch: _Minibatch) -> _MinibatchStep:
        """Normalise the value targets, run the hybrid PPO forward, assemble the loss.

        The value head regresses normalised returns (advantages are not normalised
        here; ``_hybrid_ppo_loss`` does that). The stored values are normalised with
        the same, just-updated statistics so value clipping compares like with like.
        """
        config = self.config
        returns_norm = self._normalize_returns(batch.returns, batch.part)
        values_norm = (batch.values - self._ret_mean) / (self._ret_var + 1e-8).sqrt()

        self.profile("train_forward", self.epoch)
        mb_obs = batch.obs
        if not config["use_rnn"]:
            mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

        # lstm_h/lstm_c None: the zero initial state is exact, because every stored
        # segment began at evaluate()'s zeroed state; terminals drive the mid-segment
        # resets inside the policy's BPTT so the forward replays the rollout's masking.
        state = dict(
            action=batch.actions,
            lstm_h=None,
            lstm_c=None,
            terminals=batch.terminals,
        )
        # Per-factor clipped loss with independent discrete and continuous ratios
        # (Fan et al. IJCAI 2019). The returned logits are masked, so the per-head
        # entropy metrics report the distribution the rollout sampled from.
        (pg_loss, entropy, newvalue, newlogprob, ratio_d, ratio_c, logits) = _hybrid_ppo_loss(
            self.policy,
            mb_obs,
            batch.actions,
            batch.cont_actions,
            batch.old_logp_d,
            batch.old_logp_c,
            batch.advantages,
            config["clip_coef"],
            state,
            mb_prio=batch.prio,
            mb_masks=batch.masks,
            mb_part=batch.part_f,
            aim_dim_mask=getattr(self.policy, "aim_dim_mask", None),
            aim_entropy_bonus=bool(config.get("aim_entropy_bonus", True)),
        )

        self.profile("train_misc", self.epoch)
        ratio = self._ratio_stats(update, batch, newlogprob, ratio_d, ratio_c)
        newvalue = newvalue.view(returns_norm.shape)
        v_loss = masked_value_loss(newvalue, returns_norm, values_norm, config["vf_clip_coef"],
                                   batch.part_f)
        current_entropy, entropy_unmasked, alpha, alpha_loss, entropy_loss = self._entropy_terms(
            update, entropy, batch.flat_part)
        loss = pg_loss + config["vf_coef"] * v_loss + entropy_loss
        return _MinibatchStep(
            batch=batch,
            obs=mb_obs,
            state=state,
            returns_norm=returns_norm,
            loss=loss,
            pg_loss=pg_loss,
            v_loss=v_loss,
            newvalue=newvalue,
            logits=logits,
            entropy=current_entropy,
            entropy_unmasked=entropy_unmasked,
            alpha=alpha,
            alpha_loss=alpha_loss,
            ratio=ratio,
        )

    def _ratio_stats(self, update: _UpdateState, batch: _Minibatch, newlogprob, ratio_d,
                     ratio_c) -> _RatioStats:
        """Joint ratio and its KL/clip diagnostics; store ratio_d; trip the KL gate.

        ``self.ratio`` gets the DISCRETE ratio, the one V-trace in
        ``compute_puff_advantage`` was tuned on. The joint ratio feeds the diagnostics
        only. Every diagnostic is a mean over participating rows: an unmasked KL is
        diluted by the parked fraction, and target_kl would never fire at n_active=1.
        The clip fractions per factor are observe-only. A KL trip sets ``kl_stop``;
        ``train()`` stops at the next epoch boundary.
        """
        clip_coef = self.config["clip_coef"]
        newlogprob = newlogprob.reshape(batch.logprobs.shape)
        logratio = newlogprob - batch.logprobs
        ratio = logratio.exp()
        # _hybrid_ppo_loss returns flat (B*T,) ratios; self.ratio is [segments, bptt].
        self.ratio[batch.idx] = ratio_d.detach().reshape(batch.logprobs.shape)

        with torch.no_grad():
            part = batch.part_f.reshape(logratio.shape)
            old_approx_kl = masked_mean(-logratio, part)
            approx_kl = masked_mean((ratio - 1) - logratio, part)
            clipfrac = masked_mean(((ratio - 1.0).abs() > clip_coef).float(), part)
            clipfrac_d = masked_mean(((ratio_d - 1.0).abs() > clip_coef).float(), batch.flat_part)
            clipfrac_c = masked_mean(((ratio_c - 1.0).abs() > clip_coef).float(), batch.flat_part)

        if update.target_kl is not None and approx_kl.item() > update.target_kl:
            update.kl_stop = True
        return _RatioStats(ratio=ratio,
                           part=part,
                           old_approx_kl=old_approx_kl,
                           approx_kl=approx_kl,
                           clipfrac=clipfrac,
                           clipfrac_d=clipfrac_d,
                           clipfrac_c=clipfrac_c)

    def _entropy_terms(self, update: _UpdateState, entropy, flat_part):
        """The SAC-style alpha loss and the entropy bonus at this minibatch's effective alpha.

        Returns ``(entropy, entropy_unmasked, alpha, alpha_loss, entropy_loss)``.

        The controller and the collapse floor read the participating rows' entropy.
        The other rows would misstate it. An n_active-parked row has one valid bin per
        discrete head (discrete entropy 0), but with aim_entropy_bonus on (the
        default) its entropy also includes the aim Gaussian's, which is negative for
        σ below 1/sqrt(2πe) ≈ 0.24. A statue row under ``--opponent noop`` is a
        spawned agent with the env's ordinary masks. ``alpha_loss`` is built every
        minibatch because it is logged; ``_step_alpha`` skips only the step in GRACE.
        Effective alpha: raw alpha, ceilinged during warm-start GRACE, then clamped up
        to 0.5 when entropy is below the collapse floor, unless the warm-start window
        has the floor off (re-arming it mid-ramp would jump alpha from ~1e-3 to 0.5
        in one minibatch).
        """
        current_entropy = masked_mean(entropy, flat_part)
        entropy_unmasked = entropy.mean()              # diagnostic only (losses/entropy_unmasked)

        alpha = self._log_alpha_tensor.exp()
        alpha_loss = (self._log_alpha_tensor *
                      (current_entropy - update.target_entropy).detach()).mean()

        effective_alpha = alpha.detach()
        if self._warmstart_phase == WS_GRACE:
            effective_alpha = torch.clamp(effective_alpha,
                                          max=float(self.config.get("warmstart_alpha_ceiling",
                                                                    0.0)))
        if update.floor_active and current_entropy.item() < self._entropy_floor:
            effective_alpha = torch.clamp(effective_alpha, min=0.5)
            update.floor_fires += 1
        update.effective_alpha = effective_alpha
        return (current_entropy, entropy_unmasked, alpha, alpha_loss,
                -effective_alpha * current_entropy)

    def _step_alpha(self, step: _MinibatchStep):
        """Step log_alpha on this minibatch's alpha loss, except in warm-start GRACE.

        Runs before the policy backward. The policy loss was built from the pre-step
        alpha, so this step cannot reach this minibatch's policy gradient; GRACE
        keeps log_alpha at its operating point for the ramp's handover.
        """
        if self._warmstart_phase != WS_GRACE:
            self._alpha_optimizer.zero_grad()
            step.alpha_loss.backward()
            self._alpha_optimizer.step()

    def _write_back_values(self, step: _MinibatchStep):
        """Store this minibatch's value predictions, denormalised, for the next GAE.

        Parked rows stay exactly 0: the rollout wrote 0 there and GAE reads it.
        """
        std = (self._ret_var + 1e-8).sqrt()
        batch = step.batch
        self.values[batch.idx] = ((step.newvalue.detach().float() * std + self._ret_mean) *
                                  batch.part_f)

    def _accumulate_minibatch(self, update: _UpdateState, step: _MinibatchStep):
        """Add this minibatch's metrics to ``update.sums`` and count the minibatch as run.

        ``update.sums`` holds SUMS that ``_finish_update`` divides by
        ``minibatches_run``, so every key written here is logged as a mean over the
        executed minibatches. The count sits here, beside the sums, so the two cannot
        drift. An absolute per-call value belongs in ``_finish_update`` instead.
        """
        sums = update.sums
        batch = step.batch
        with torch.no_grad():
            dists = [torch.distributions.Categorical(logits=lgt) for lgt in step.logits]
            # strict=True fails on drift between the action spec and the policy's heads.
            for name, dist in zip(ACTION_HEAD_NAMES, dists, strict=True):
                sums[f"entropy/{name}"] += masked_mean(dist.entropy(), batch.flat_part).item()
        sums["entropy/total"] += step.entropy.item()

        self.profile("train_misc", self.epoch)
        ratio = step.ratio
        sums["policy_loss"] += step.pg_loss.item()
        sums["value_loss"] += step.v_loss.item()
        sums["entropy"] += step.entropy.item()
        # The same mean without the mask, a diagnostic. Its ratio to losses/entropy is
        # about the participating row fraction only when the other rows' entropy is 0
        # and segments are drawn uniformly: --opponent self, aim_entropy_bonus off,
        # prio_alpha 0, no event segment (test_parked_rows_masked's twenty-update
        # test pins that case).
        sums["entropy_unmasked"] += step.entropy_unmasked.item()
        sums["alpha"] += step.alpha.detach().item()
        sums["alpha_loss"] += step.alpha_loss.item()
        sums["old_approx_kl"] += ratio.old_approx_kl.item()
        sums["approx_kl"] += ratio.approx_kl.item()
        sums["clipfrac"] += ratio.clipfrac.item()
        sums["clipfrac_d"] += ratio.clipfrac_d.item()
        sums["clipfrac_c"] += ratio.clipfrac_c.item()
        sums["importance"] += masked_mean(ratio.ratio, ratio.part).item()
        update.minibatches_run += 1

    def _skip_nonfinite_step(self, loss):
        """Drop a minibatch whose loss is non-finite: zero the gradients, warn once a minute.

        The continuous aim head can emit non-finite mu/log_std in pathological early
        training; one bad minibatch must not end the run. Its metrics are already in
        the sums (it ran; it only does not step the policy).
        """
        now = time.time()
        if now - getattr(self, "_last_nan_warn_t", 0.0) > 60.0:
            print(f"[hybrid_aim NaN guard] non-finite loss "
                  f"({float(loss.detach())}); skipping optimizer step")
            self._last_nan_warn_t = now
        self.optimizer.zero_grad(set_to_none=True)

    def _record_tag(self, update: _UpdateState, mb: int, step: _MinibatchStep):
        """TAG gradient-cosine diagnostic at the update's first and last executed minibatch.

        mb0 is the pre-update on-policy regime. mbL is ``total_minibatches - 1``, or
        the last minibatch of the epoch a KL trip ends the update on: requiring no
        trip would select against the late-update regime mbL exists to observe.
        ``train()`` calls this after the NaN guard (a skipped batch is never measured)
        and before ``loss.backward()`` (the gradients are still untouched). Results
        go on ``self._tag_metrics``; ``_inject_tag_metrics`` moves them into the row.
        """
        config = self.config
        if not (config.get("tag_diagnostic", False)
                and self.epoch % max(1, int(config.get("tag_every", 5))) == 0):
            return
        is_mb0 = mb == 0
        is_mbl = (mb == self.total_minibatches - 1
                  or (update.kl_stop and (mb + 1) % update.mbs_per_epoch == 0))
        if not (is_mb0 or is_mbl):
            return
        batch = step.batch
        tag = tag_grad_cossim(
            self.policy,
            mb_obs=step.obs,
            mb_actions=batch.actions,
            mb_cont_actions=batch.cont_actions,
            mb_old_logp_d=batch.old_logp_d,
            mb_old_logp_c=batch.old_logp_c,
            mb_advantages=batch.advantages,
            clip_coef=config["clip_coef"],
            state=step.state,
            mb_prio=batch.prio,
            mb_masks=batch.masks,
            mb_returns_norm=step.returns_norm,
            idx=batch.idx,
            mb_label="mb0" if is_mb0 else "mbL",
            mb_part=batch.part_f,
            aim_dim_mask=getattr(self.policy, "aim_dim_mask", None),
            aim_entropy_bonus=bool(config.get("aim_entropy_bonus", True)),
        )
        tag_metrics = getattr(self, "_tag_metrics", None)
        if tag_metrics is None:
            tag_metrics = {}
        self._tag_metrics = tag_metrics
        self._tag_metrics.update(tag)
        if not is_mb0:
            self._tag_metrics["tag/mbL_index"] = float(mb)
        self._tag_metrics["tag/selfplay_active"] = float(getattr(self, "_selfplay_used_past",
                                                                 False))

    def _step_policy(self, mb: int, loss):
        """Backward the policy loss; clip and step every ``accumulate_minibatches``.

        ``_grad_norm`` is the pre-clip total norm ``clip_grad_norm_`` returns. A
        minibatch the NaN guard skipped leaves the previous value in place.
        """
        loss.backward()
        if (mb + 1) % self.accumulate_minibatches == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                                       self.config["max_grad_norm"])
            self._grad_norm = float(grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _finish_update(self, update: _UpdateState):
        """The update's ``losses`` dict: minibatch means plus per-call absolutes.

        Means divide by the EXECUTED minibatch count (gh#90: a KL-truncated update
        must not scale its losses by k/N). Every other key is written here, after the
        division, into the dict the division produced. Also refreshes the
        per-update trainer attributes: ``_last_entropy_mean`` (the warm-start
        anchor's source), ``_log_alpha``, ``_effective_alpha`` and the ``_std_*``.
        """
        n = update.minibatches_run
        losses = defaultdict(float, {key: total / n for key, total in update.sums.items()})
        losses["minibatches_run"] = n
        losses["entropy_floor_fires"] = float(update.floor_fires)
        losses["empty_minibatches"] = float(update.empty_minibatches)
        losses["participating_rows"] = float(self.participating.sum().item())

        if self.config.get("warmstart_entropy", False):
            self._warmstart_metrics(losses)
        # The warm-start anchor's source, kept every update whether or not the mode is
        # on (a later resume may enable it). KNOWN LIMIT: an update whose minibatches
        # were all empty has no entropy mean; this defaultdict read then stores 0.0
        # (and adds losses/entropy = 0.0 to its row).
        self._last_entropy_mean = float(losses["entropy"])

        self.profile("train_misc", self.epoch)
        if self.config["anneal_lr"]:
            self.scheduler.step()

        # Explained variance over PARTICIPATING rows: parked rows are not value
        # targets (their stored value is 0), so including them would distort EV.
        advantages = update.advantages
        assert advantages is not None, "train() ran no minibatch (total_minibatches == 0)"
        losses["explained_variance"] = masked_explained_variance(
            self.values.flatten(),
            advantages.flatten() + self.values.flatten(), self.participating.flatten())
        losses["ret_mean"] = self._ret_mean.item()
        losses["ret_std"] = (self._ret_var + 1e-8).sqrt().item()
        # ~0.0 in production while include_step_stats_in_info is off (#100): no event
        # is ever marked.
        losses["event_oversample_fraction"] = float(self._event_oversample_fraction)
        losses["log_alpha"] = self._log_alpha_tensor.item()

        self._log_alpha = float(self._log_alpha_tensor.item())
        # None when every minibatch was empty: the previous value stays.
        if update.effective_alpha is not None:
            self._effective_alpha = float(update.effective_alpha.item())
        # The alpha the loss used (after the GRACE ceiling and the floor clamp);
        # losses/alpha is the raw exp(log_alpha).
        losses["effective_alpha"] = float(self._effective_alpha)
        self._std_combat = float(self._welford_combat.std())
        self._std_objective = float(self._welford_objective.std())
        self._std_positional = float(self._welford_positional.std())
        return losses

    def _warmstart_metrics(self, losses):
        """Warm-start phase and the GRACE collapse watch, as absolutes on ``losses``.

        h0 is the mode's first-update mean entropy (a BC policy starts near 1.8 nats).
        It must be positive: total entropy (discrete plus Gaussian differential) can
        go non-positive in collapse and would flip the watch's sign, so a
        non-positive first value is reported and retried next update. GRACE disables
        both anti-collapse guards (the floor and alpha), so entropy at half of h0 or
        less warns, at most every 20 epochs.
        """
        losses["warmstart_phase"] = self._warmstart_phase
        if self._warmstart_h0 is None:
            if losses["entropy"] > 1e-9:
                self._warmstart_h0 = float(losses["entropy"])
            else:
                print(f"[Train] WARN warm-start: first-update entropy "
                      f"{losses['entropy']:.3f} <= 0 — h_over_h0 collapse "
                      f"watch cannot arm (will retry next update).")
        if self._warmstart_h0:
            losses["warmstart_h_over_h0"] = losses["entropy"] / self._warmstart_h0
            if (self._warmstart_phase == WS_GRACE and losses["warmstart_h_over_h0"] < 0.5
                    and self.epoch - self._warmstart_warn_epoch >= 20):
                self._warmstart_warn_epoch = self.epoch
                print(f"[Train] WARN warm-start grace: entropy at "
                      f"{losses['warmstart_h_over_h0']:.2f} of its start value "
                      f"({losses['entropy']:.3f} nats) with alpha ceilinged and the "
                      f"entropy floor disabled — collapse watch (spec finding 6).")

    def _log_and_checkpoint(self, losses):
        """PufferLib's update tail: advance the epoch, flush logs (throttled), checkpoint.

        As upstream, ``self.losses`` is set AFTER ``mean_and_log``, so a logged row
        carries the previous flushed update's losses.

        ``done_training`` compares participating steps with ``participating_timesteps``
        (the user's --timesteps); ``total_timesteps`` is the raw-row budget
        PufferLib's epoch cap needs. The epoch clause is load-bearing: total_epochs
        floor-divides, so at the defaults (--timesteps 10M, --num_envs 256, batch
        163840, n_active=1, raw budget 50M) the 305 allowed epochs collect only
        305 × 163840 / 5 = 9,994,240 participating steps, and without it the last
        epoch would never checkpoint. ``.get`` keeps configs without the key working.
        """
        logs = None
        self.epoch += 1
        config = self.config
        done_training = (self.global_step >= config.get("participating_timesteps",
                                                        config["total_timesteps"])
                         or self.epoch >= self.total_epochs)
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            # The window's episode count (terminal infos, one per env per round at any
            # n_active_per_team), written INTO self.stats so mean_and_log logs it as
            # environment/episodes. A one-element list: np.mean of it is the count.
            # `.get`: a defaultdict(list) read would insert [] and np.mean([]) is NaN.
            self.stats["episodes"] = [float(len(self.stats.get("kills_t", ())))]
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            self.profile.clear()

        if self.epoch % config["checkpoint_interval"] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f"Checkpoint saved at update {self.epoch}"

        return logs
