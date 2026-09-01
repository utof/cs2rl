"""PPO update path (#140), split out of train.py.

WHAT: the whole trainer-update surface — the masked reductions every trainer
statistic goes through, the hybrid discrete+continuous PPO loss, the TAG
gradient-cosine diagnostic and its parameter partition, the entropy-target
schedule, and ``_patch_trainer_with_return_norm``, the monkey-patch that
replaces PuffeRL's ``train()`` wholesale. Moved here VERBATIM by the 2026-08-31
post-rung1a refactor: no renames, no signature changes, no behaviour change.
``train.py`` re-exports every name below (see its ``__all__``), so existing
``from train import X`` call sites keep working unchanged.

WHY its own module: ``_patch_trainer_with_return_norm`` is a 911-line patcher
whose inner ``_train_with_return_norm`` is a 713-line closure over 15 freevars —
it CANNOT be split from its enclosing function without changing behaviour, and
the loss/reduction helpers it calls are only meaningful next to it. Ten test
files import that patcher; giving it a file makes the update path reviewable
without paging through the entry point.

PITFALL (runtime rebinding): a test that wants to intercept ``tag_grad_cossim``
must patch it on THIS module, not on ``train`` — the call site inside
``_train_with_return_norm`` resolves through this module's globals, so a patch
on ``train`` is silently unreachable and the assertion becomes vacuous. See
tests/test_tag_trainer.py.

IMPORT-LIGHTNESS INVARIANT: module scope stays torch/nav/c_env-free, for the
reason spelled out in train_shared.py's header. Every torch, pufferlib and
train_helpers_batch1 import below is function-local ON PURPOSE.
"""
import numpy as np

from _action_spec import ACTION_HEAD_NAMES, ACTION_HEAD_SIZES, AIM_DIM
from train_shared import _LOG_2PI, LOG_STD_MAX, _apply_action_masks

# ── Masked reductions over participating rows (Rung 0, spec 2026-08-29 §2.2) ──
# WHY these are free functions and not methods on the trainer: the trainer is a
# monkey-patched PuffeRL instance (pufferl.py is a site-package and is never
# edited), so every reduction the update path needs has to live here where a
# unit test can call it without building a trainer.
# DTYPE CONTRACT used by every caller below: the *bool* [S,T] mask is for
# INDEXING (`sel[mb_part]`); the *float* copy (`mb_part.to(torch.float32)`) is
# the weight `w` these helpers take. `sel[mb_part_f]` is an IndexError and
# `masked_mean(x, bool_mask)` would work only by accident — the helpers
# `.to(x.dtype)` their weight, so pass whichever, but do not swap the two roles.
# SHAPE PITFALL: `w` is multiplied (not indexed) against `x`, so it must be
# broadcast-compatible with `x`. Reducing a FLAT (S*T,) tensor (entropy,
# per-head entropies, ratio_d/ratio_c) with an [S,T] weight silently
# broadcasts to [S, S*T] — pass the flattened weight for flat tensors.


def masked_mean(x, w):
    """Mean of x over rows where w == 1. w broadcasts to x; w.sum() == 0 ⇒ 0.

    Rung 0 §2.2: parked agent rows (noop-masked, entropy exactly 0, zero
    reward) must not enter any trainer statistic, or every all-row mean at
    n_active=1 is diluted 5×. Masked mean = (x·w).sum() / max(w.sum(), 1).
    """
    w = w.to(x.dtype)
    return (x * w).sum() / w.sum().clamp(min=1.0)


def masked_std_unbiased(x, w, mean):
    """Unbiased (n−1) std over the w == 1 rows, matching torch .std() on the subset.

    `mean` is the caller's already-computed masked_mean — passed in rather than
    recomputed so the two-pass (mean, then deviation) reduction is done once.
    PITFALL: the (n−1) clamp means a single participating row yields std 0, not
    NaN; masked_normalize_adv's +1e-8 then makes that row's normalised
    advantage 0 rather than inf.
    """
    w = w.to(x.dtype)
    n = w.sum()
    return (((x - mean)**2 * w).sum() / (n - 1.0).clamp(min=1.0)).sqrt()


def masked_normalize_adv(flat_adv, w):
    """(adv − mean) / (std + 1e-8) over participating rows; parked rows → 0.

    Uses the same unbiased std as the unmasked path so the two agree exactly
    when w is all-ones. Zeroing parked rows makes their pg contribution 0
    regardless of ratio, which is what the masked pg mean then divides out.
    """
    w = w.to(flat_adv.dtype).reshape(-1)
    m = masked_mean(flat_adv, w)
    s = masked_std_unbiased(flat_adv, w, m)
    return (flat_adv - m) / (s + 1e-8) * w


def masked_explained_variance(y_pred, y_true, part):
    """explained_variance over part == True rows (whole-buffer, Rung 0 §2.2).

    Returns nan when the participating y_true has zero variance, mirroring
    PufferLib's `torch.nan if var_y == 0` convention. `part` is the BOOL mask
    here (this one indexes rather than weights — the variance of a weighted
    tensor is not the variance of the subset).
    """
    import torch

    part = part.to(torch.bool)
    yt = y_true[part]
    yp = y_pred[part]
    if yt.numel() < 2:
        return float("nan")
    var_y = yt.var()
    if var_y == 0:
        return float("nan")
    return float(1 - (yt - yp).var() / var_y)


def _scheduled_target_entropy(config, global_step: int, max_entropy: float) -> float:
    """Config-driven entropy target for the SAC-style α controller.

    Single source for both the patch-time seed and the per-train()-call
    recompute in _patch_trainer_with_return_norm — keeping them identical
    means a checkpoint-resumed trainer seeds at its true scheduled value
    instead of a hardcoded warmup constant. `config` is anything with
    .get() (PuffeRL config or a plain dict); missing keys fall back to the
    build_train_config defaults so harness/older-checkpoint configs keep
    working.
    """
    from train_helpers_batch1 import target_entropy_schedule
    return target_entropy_schedule(
        global_step,
        max_entropy,
        warmup_end=config.get("entropy_target_warmup_steps", 10_000_000),
        warmup_high_frac=config.get("entropy_target_warmup_frac", 0.5),
        base_frac=config.get("entropy_target_base_frac", 0.35),
    )


# ── SECTION: Value target normalization ───────────────────────────────────


def _patch_trainer_with_return_norm(trainer):
    """Monkey-patch a PuffeRL instance to normalize value regression targets.

    MAPPO's strongest recommendation: normalize the returns (advantages + values)
    that the value function regresses against.  This stabilizes value learning
    especially with high gamma (0.999) where return variance is large.

    Implementation: maintain a running mean/std of returns on the training
    device; before the value loss, normalize mb_returns to zero-mean unit-std.
    The value head learns to predict normalized returns; no denormalization is
    needed (unlike PopArt) because we don't use the raw value for anything
    outside the loss.
    """
    # gh #85: BPTT zero-init exactness (Dust2Policy.forward/_lstm_bptt) is only
    # EXACT when each agent row fills exactly one buffer segment per evaluate(),
    # i.e. segments == total_agents. Upstream PuffeRL only enforces
    # total_agents <= segments (pufferl.py:83-86); our equality holds by
    # construction in compute_batch_dims but nothing asserted it — one
    # batch_size/bptt_horizon config edit away from silently-biased importance
    # ratios. Fail loudly at patch time instead.
    assert trainer.segments == trainer.total_agents, (
        f"segments ({trainer.segments}) != total_agents ({trainer.total_agents}): "
        "BPTT zero-init exactness broken — revisit batch_size/bptt_horizon "
        "(compute_batch_dims) or the _lstm_bptt initial-state design. See gh #85.")

    import time
    import types
    from collections import defaultdict

    import torch
    from pufferlib.pufferl import compute_puff_advantage

    # Warm-start entropy mode (spec 2026-08-01). Function-local like every
    # other import here — train.py has no module-level train_helpers_batch1
    # import. _train_with_return_norm is nested inside this function, so it
    # picks these up as closure freevars. WS_RAMP is not needed here.
    from train_helpers_batch1 import WS_GRACE, WS_OFF, warmstart_entropy_state

    # Running stats for return normalization (Welford-style, torch tensors)
    device = trainer.config["device"]
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
    #   names. The trainer attribute below is meant to be the same tensor
    #   reference the closure mutates, so `_update_return_stats` writes are
    #   visible via trainer._ret_var (and conversely tests reading the attr
    #   see the live value, not a stale snapshot).
    _ret_mean.zero_()
    _ret_var.fill_(1.0)
    _ret_count.zero_()
    trainer._ret_mean = _ret_mean
    trainer._ret_var = _ret_var
    trainer._ret_count = _ret_count
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
    _cap = float(getattr(trainer.policy, "aim_log_std_max", LOG_STD_MAX))
    _n_aim = float(getattr(trainer.policy, "aim_dim_mask", torch.ones(AIM_DIM)).sum())
    _bonus = bool(trainer.config.get("aim_entropy_bonus", True))
    max_entropy_continuous = (_n_aim * 0.5 * np.log(2 * np.pi * np.e * np.exp(_cap)**2)) \
        if _bonus else 0.0
    max_entropy = max_entropy_discrete + max_entropy_continuous
    # Task 9A: target_entropy is no longer a static scalar — it's recomputed
    # each train() call from a linear ramp 0.7→0.5*max_entropy across
    # [0, 10_000_000] global steps (see target_entropy_schedule). The live
    # value lives in the closure-local _t9_target_entropy in
    # _train_with_return_norm and on trainer._batch1_current_target_entropy.
    entropy_floor = 0.3 * max_entropy  # collapse threshold
    import math

    log_alpha = torch.tensor([math.log(0.1)], requires_grad=True, device=device)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=1e-4)
    # R0-C (#134): the train_state.pt sidecar needs both; they are closure-
    # locals, so alias them here. Restore MUST be in place
    # (`log_alpha.data.copy_`) — rebinding the attribute would leave the
    # closure (and alpha_optimizer's param list) on the old tensor.
    trainer._log_alpha_tensor = log_alpha
    trainer._alpha_optimizer = alpha_optimizer

    # Task 9A/9B: trainer-level state for target_entropy schedule + log_alpha
    # reset. Attached to the trainer (not closure-local) so:
    #   - tests can inspect/pin _batch1_max_entropy and current_target_entropy
    #   - the wandb log layer can read _batch1_current_target_entropy without
    #     reaching into the closure of _train_with_return_norm.
    # _batch1_log_alpha_reset_done is the idempotency flag for Task 9B —
    # the first train() call after this patch is applied resets log_alpha to
    # log(ent_coef); every later train() call leaves log_alpha alone so the
    # SAC dual-gradient loop can do its job.
    trainer._batch1_max_entropy = float(max_entropy)
    trainer._batch1_log_alpha_reset_done = False
    # Seed from the schedule at the CURRENT global_step (not a hardcoded
    # warmup constant) so checkpoint-resumed trainers start consistent;
    # the per-train()-call recompute below overwrites it every call anyway.
    trainer._batch1_current_target_entropy = _scheduled_target_entropy(
        trainer.config, trainer.global_step, float(max_entropy))
    # Pre-init effective_alpha + grad_norm metrics (utof/cs2rl#16). The
    # post-loop reads in _train_with_return_norm refresh these, but if the
    # target_kl early-break trips on mb=0 OR no accumulation boundary fires,
    # the local names are never bound — leaving the trainer attrs
    # AttributeError on first read. Seeding them with sane defaults here
    # turns those edge cases into "stale-from-previous-call" instead of a
    # crash, and the post-loop refresh overwrites whenever the loop runs
    # all the way through.
    trainer._batch1_effective_alpha = float(trainer.config["ent_coef"])
    trainer._batch1_grad_norm = 0.0

    # ── Warm-start entropy mode state (spec 2026-08-01) ────────────────────
    # h_anchor: mean policy entropy captured at grace end (None until then);
    # last_entropy_mean: previous update's post-divisor losses["entropy"] —
    # the ONLY valid anchor source (there is no entropy EMA in this codebase,
    # and trainer.losses is refreshed only inside the throttled log-flush
    # block, so it can be several updates stale — spec finding 4).
    # h0: first update's mean entropy, denominator of warmstart_h_over_h0
    # (grace collapse watch — with the floor disabled AND alpha~0 the run
    # has no anti-collapse guard, spec finding 6).
    trainer._batch1_warmstart_h_anchor = None
    trainer._batch1_warmstart_h0 = None
    trainer._batch1_warmstart_phase = WS_OFF
    trainer._batch1_last_entropy_mean = None
    trainer._batch1_warmstart_warn_epoch = -10**9

    # ──────────────────────────────────────────────────────────────────────

    def _update_return_stats(returns_flat):
        nonlocal _ret_mean, _ret_var, _ret_count
        with torch.no_grad():
            n = returns_flat.numel()
            if n == 0:
                return
            batch_mean = returns_flat.mean()
            batch_var = returns_flat.var(unbiased=False)
            batch_count = torch.tensor(float(n), device=device)

            delta = batch_mean - _ret_mean
            tot = _ret_count + batch_count
            new_mean = _ret_mean + delta * batch_count / tot
            m_a = _ret_var * _ret_count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta.pow(2) * _ret_count * batch_count / tot
            new_var = m2 / tot

            _ret_mean.copy_(new_mean)
            _ret_var.copy_(new_var)
            _ret_count.copy_(tot)

    def _normalize_returns(mb_returns, mb_part=None):
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
        _update_return_stats(sel.flatten())
        std = (_ret_var + 1e-8).sqrt()
        return (mb_returns - _ret_mean) / std

    # Test hook (tests/test_parked_rows_masked.py): the closure is otherwise
    # unreachable, and the participating-rows-only stats update is exactly the
    # part worth pinning.
    trainer._normalize_returns = _normalize_returns

    def _train_with_return_norm(self):
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
        # signal. We mirror the value onto trainer._batch1_current_target_entropy
        # so the wandb log layer can read it without touching this closure.
        # Fracs/warmup_steps come from config via _scheduled_target_entropy
        # (finding 4 residual — previously hardcoded 0.7→0.5).
        # PITFALL: do NOT capture max_entropy from the outer closure here —
        # use trainer._batch1_max_entropy. Closure capture would silently break
        # if the patch were re-applied on the same trainer instance.
        _t9_target_entropy = _scheduled_target_entropy(config, self.global_step,
                                                       trainer._batch1_max_entropy)
        trainer._batch1_current_target_entropy = float(_t9_target_entropy)

        # Task 9B: one-shot log_alpha reset on the first train() call after
        # this patch. WHY: the entropy schedule + log_alpha are coupled — the
        # outer training loop can leave log_alpha at a stale value from a
        # previous run / re-init, and we need a deterministic starting point
        # of log(ent_coef) so the SAC dual-gradient loop converges from a
        # known floor. The flag is on the trainer (not the closure) so a
        # checkpoint-restored trainer that gets re-patched still resets once.
        if not trainer._batch1_log_alpha_reset_done:
            with torch.no_grad():
                log_alpha.fill_(math.log(config["ent_coef"]))
            trainer._batch1_log_alpha_reset_done = True

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
            if (trainer._batch1_warmstart_h_anchor is None and self.global_step >= _ws_grace
                    and trainer._batch1_last_entropy_mean is not None):
                # one-shot anchor capture (idempotent: guarded on None).
                # Finite-check (Task 1 review): a NaN/inf entropy mean latched
                # here would poison target and alpha_loss for the whole ramp —
                # skip the capture (stay GRACE) and shout instead.
                if math.isfinite(trainer._batch1_last_entropy_mean):
                    trainer._batch1_warmstart_h_anchor = float(trainer._batch1_last_entropy_mean)
                else:
                    print(f"[Train] WARN warm-start: non-finite entropy mean "
                          f"{trainer._batch1_last_entropy_mean} at grace end — "
                          f"anchor capture skipped, staying in GRACE.")
            _ws = warmstart_entropy_state(
                self.global_step,
                grace_steps=_ws_grace,
                ramp_steps=int(config.get("warmstart_ramp_steps", 10_000_000)),
                h_anchor=trainer._batch1_warmstart_h_anchor,
                base_target=(config.get("entropy_target_base_frac", 0.35) *
                             trainer._batch1_max_entropy))
            trainer._batch1_warmstart_phase = _ws.phase
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
                trainer._batch1_current_target_entropy = float(_ws.target)
        else:
            # Config can be toggled off in-process (tests do this; production
            # builds the config once). Re-seed the phase so a stale GRACE can
            # never keep the alpha optimizer frozen after the mode is disabled.
            trainer._batch1_warmstart_phase = WS_OFF

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
            # from the parallel buffers added by _patch_trainer_with_hybrid_aim.
            # mb_logprobs (the SUM) stays the canonical "logp from rollout" for
            # KL/clipfrac diagnostics below; the per-factor halves drive the
            # per-factor PPO clip in _hybrid_ppo_loss.
            mb_cont_actions = self.cont_actions[idx]
            mb_old_logp_d = self.logprobs_d[idx]
            mb_old_logp_c = self.logprobs_c[idx]
            # F8: rollout-stored action masks (all-ones = unmasked fallback).
            # getattr for trainers built before _patch_trainer_with_hybrid_aim
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
            mb_returns_norm = _normalize_returns(mb_returns, mb_part)
            # Also normalize the stored baseline values so clipping stays valid
            mb_values_norm = (mb_values - _ret_mean) / (_ret_var + 1e-8).sqrt()
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
            alpha = log_alpha.exp()
            # Task 9A: use the scheduled target_entropy (recomputed at top of
            # this train() call) instead of the static fallback. _t9_target_entropy
            # is a Python float; .detach() on a tensor minus a float is fine —
            # autograd treats the float as a constant.
            # alpha_loss is computed UNCONDITIONALLY (the logging block below
            # accumulates it every minibatch — spec finding 7); during the
            # warm-start GRACE phase only the optimizer step is skipped, so
            # log_alpha stays at its operating point (see phase-resolution
            # comment above for why that matters).
            alpha_loss = (log_alpha * (current_entropy - _t9_target_entropy).detach()).mean()
            if trainer._batch1_warmstart_phase != WS_GRACE:
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

            effective_alpha = alpha.detach()
            if trainer._batch1_warmstart_phase == WS_GRACE:
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
            if _ws_floor_active and current_entropy.item() < entropy_floor:
                effective_alpha = torch.clamp(effective_alpha, min=0.5)
                _floor_fires += 1

            entropy_loss = -effective_alpha * current_entropy
            # ──────────────────────────────────────────────────────────────

            loss = pg_loss + config["vf_coef"] * v_loss + entropy_loss
            self.amp_context.__enter__()

            # Denormalize before writing back so advantage computation stays in raw scale
            std = (_ret_var + 1e-8).sqrt()
            # Rung 0 §2.2: keep parked rows at exactly 0 in the value buffer —
            # the rollout wrote 0 there and the next epoch's GAE reads it.
            self.values[idx] = (newvalue.detach().float() * std + _ret_mean) * mb_part_f

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
            if config.get("tag_diagnostic", False) \
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
                    if getattr(trainer, "_tag_metrics", None) is None:
                        trainer._tag_metrics = {}
                    trainer._tag_metrics.update(_tag)
                    if not _tag_mb0:
                        trainer._tag_metrics["tag/mbL_index"] = float(mb)
                    trainer._tag_metrics["tag/selfplay_active"] = float(
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
                trainer._batch1_grad_norm = float(_t9_grad_norm)
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
            losses["warmstart_phase"] = trainer._batch1_warmstart_phase
            # h0 is captured on the mode's first update, when entropy is
            # healthy (BC policy ~1.8 nats) — the >1e-9 guard exists because
            # total entropy (discrete + Gaussian differential) CAN go
            # non-positive in the collapse regime, and a non-positive
            # denominator would flip the watch's sign. If capture is ever
            # skipped, say so once instead of silently disabling the watch.
            if trainer._batch1_warmstart_h0 is None:
                if losses["entropy"] > 1e-9:
                    trainer._batch1_warmstart_h0 = float(losses["entropy"])
                else:
                    print(f"[Train] WARN warm-start: first-update entropy "
                          f"{losses['entropy']:.3f} <= 0 — h_over_h0 collapse "
                          f"watch cannot arm (will retry next update).")
            if trainer._batch1_warmstart_h0:
                losses["warmstart_h_over_h0"] = losses["entropy"] / trainer._batch1_warmstart_h0
                # collapse watch (spec finding 6): grace disables BOTH
                # anti-collapse guards (floor clamp + alpha), so shout —
                # throttled to every 20 epochs — if H halves.
                if (trainer._batch1_warmstart_phase == WS_GRACE
                        and losses["warmstart_h_over_h0"] < 0.5
                        and self.epoch - trainer._batch1_warmstart_warn_epoch >= 20):
                    trainer._batch1_warmstart_warn_epoch = self.epoch
                    print(f"[Train] WARN warm-start grace: entropy at "
                          f"{losses['warmstart_h_over_h0']:.2f} of its start value "
                          f"({losses['entropy']:.3f} nats) with alpha ceilinged and the "
                          f"entropy floor disabled — collapse watch (spec finding 6).")
        # Anchor source: maintained EVERY update, unconditionally (mode may be
        # enabled on a later resume of this process in tests; cost is one float).
        trainer._batch1_last_entropy_mean = float(losses["entropy"])

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
        losses["ret_mean"] = _ret_mean.item()
        losses["ret_std"] = (_ret_var + 1e-8).sqrt().item()
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
        losses["log_alpha"] = log_alpha.item()

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
        trainer._batch1_log_alpha = float(log_alpha.item())
        # effective_alpha may be unbound this call if target_kl early-broke
        # on mb=0 — leave the pre-initialised trainer attr (set in the patch
        # block above) intact in that case rather than crashing.
        try:
            trainer._batch1_effective_alpha = float(effective_alpha.detach().item())
        except (NameError, UnboundLocalError):
            pass
        # Rung 0 §2.2: log the alpha the loss ACTUALLY used (post ceiling /
        # post floor clamp), not just the raw log_alpha.exp() above — with the
        # floor now counted by entropy_floor_fires, the pair says whether the
        # collapse guard is holding the run up. Absolute value, so it sits here
        # after the gh#90 divisor, and it reads the trainer attr rather than
        # the loop-local so the KL-early-break edge cannot NameError.
        losses["effective_alpha"] = float(trainer._batch1_effective_alpha)
        # Welford std exposure: guard with getattr+fallback because
        # _patch_trainer_with_selfplay (Task 6c, where these get attached)
        # may not have been applied — preserves the no-selfplay code path.
        _w_combat = getattr(trainer, "_batch1_welford_combat", None)
        trainer._batch1_std_combat = (float(_w_combat.std()) if _w_combat is not None else 1.0)
        _w_obj = getattr(trainer, "_batch1_welford_objective", None)
        trainer._batch1_std_objective = (float(_w_obj.std()) if _w_obj is not None else 1.0)
        _w_pos = getattr(trainer, "_batch1_welford_positional", None)
        trainer._batch1_std_positional = (float(_w_pos.std()) if _w_pos is not None else 1.0)

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

    # Bind the patched method to the specific trainer instance
    trainer.train = types.MethodType(_train_with_return_norm, trainer)
    print("[Train] Value target normalization enabled (running mean/std of returns).")
    return trainer


def _aim_dim_weight(aim_dim_mask, mu_aim):
    """(AIM_DIM,) weight for the per-dim Gaussian terms (R0-E.2, #131).

    WHAT: ``aim_dim_mask`` moved to mu_aim's device/dtype, or all-ones when
    None. Shared by _hybrid_sample_logits and _hybrid_ppo_loss so the rollout
    and the update can never disagree on which dims are live — that
    disagreement would be an importance-ratio bug no single-site test sees.
    PITFALL: returns ones (not None) on the None path so callers can multiply
    unconditionally; the multiply by ones is exact in fp32.
    """
    import torch

    if aim_dim_mask is None:
        return torch.ones(mu_aim.shape[-1], device=mu_aim.device, dtype=mu_aim.dtype)
    return aim_dim_mask.to(device=mu_aim.device, dtype=mu_aim.dtype)


def _hybrid_ppo_loss(policy,
                     mb_obs,
                     mb_actions,
                     mb_cont_actions,
                     mb_old_logp_d,
                     mb_old_logp_c,
                     mb_advantages,
                     clip_coef,
                     state,
                     mb_prio=None,
                     mb_masks=None,
                     return_pg_rows=False,
                     mb_part=None,
                     aim_dim_mask=None,
                     aim_entropy_bonus=True):
    """Per-factor PPO clipped loss (H-PPO, Fan et al. IJCAI 2019).

    aim_dim_mask (R0-E.2): same per-dim weight the rollout sampler used
    (policy.aim_dim_mask) — MUST match, or ratio_c ≠ 1 for an unchanged
    policy. None ⇒ all-ones (pre-R0-E behaviour).
    aim_entropy_bonus (R0-E.4, #131): False ⇒ the returned ``entropy`` is the
    DISCRETE entropy only, so the entropy objective (and its α dual loop)
    stops pushing aim σ to the cap. The Gaussian log-prob still enters
    ratio_c either way — only the bonus is switched off.

    THE CORE OF T5. Re-runs the policy on mb_obs with the stored
    mb_actions / mb_cont_actions, computes new log-probs split into
    discrete and continuous halves, and applies the PPO clip
    INDEPENDENTLY to each half before summing. This is the spec L8
    decision: discrete head saturating doesn't have to drag the
    continuous head's gradient through clipping (and vice versa).

    Returns
    -------
    pg_loss : scalar — sum of clipped discrete + clipped continuous losses.
    entropy : (B,) — joint factorised entropy (same shape as old logprobs).
    new_value : (B, 1) — fresh value estimate for the value-loss path.
    new_logp_total : (B,) — sum of new discrete + new continuous log-probs;
        used for KL/clipfrac diagnostics in the caller.
    ratio_d, ratio_c : (B,) — exposed so the caller can attribute clipfrac
        per factor and substitute ratio_d into the existing self.ratio
        slot for vtrace advantages (spec carry-forward: keep diagnostics
        backwards-compatible by using the discrete ratio there).
    logits_list : list of 7 (B, head_size) tensors — the per-head logits
        this loss was computed from (F16: post-mask when mb_masks is given,
        so per-head entropy diagnostics reflect the TRUE sampled
        distribution). Returned so the caller doesn't need a second full
        forward pass for diagnostics — that redundant no-grad forward used
        to double the update-forward cost. Autograd-attached; consume under
        torch.no_grad() and don't hold past backward() if memory matters.

    LSTM-BPTT fix: `state` must carry `terminals` (the minibatch's
    (segments, bptt_horizon) done flags) so the policy forward runs
    done-masked BPTT over the time dimension — without it the recomputed
    logprobs at post-done ticks silently diverge from the rollout-stored
    ones (state["lstm_h"]/["lstm_c"]=None means zero init, which is exact;
    see Dust2Policy.forward). Test-path callers passing flat 2D tensors may
    omit terminals: T=1 has no through-time state to reset.

    mb_masks (F8): the rollout-stored action masks for this minibatch,
    (segments, bptt_horizon, ACTION_MASK_DIM) bool (or flat (B, MASK_DIM) on
    the test path). MUST be the same masks the rollout sampler used —
    masking here but not there (or vice versa) silently skews the PPO
    ratios for any agent-step where a mask bit was 0. None = unmasked
    (pre-F8 callers / BC paths).

    return_pg_rows (TAG diagnostic, spec 2026-08-13 §4.3): when True the
    return tuple gains an 8th element — the per-row pg loss vector
    max(pg_d_un, pg_d_cl) + max(pg_c_un, pg_c_cl), flat (B*T,), graph-
    attached, advantage-normalized + prio-weighted exactly like pg_loss
    (whose value is the mean of the two factor vectors separately; the sum
    vector's .mean() equals it up to fp reduction order). TAG forms subset
    losses as weighted means over this vector so every subset gradient is a
    true restriction of the real gradient from ONE forward pass. False (the
    default, all production update paths) returns the existing 7-tuple
    bitwise-identically — pinned by
    test_return_pg_rows_default_is_bitwise_identical_7_tuple.

    mb_part (Rung 0 §2.2): FLOAT participation weights, broadcastable to
    mb_advantages (the trainer passes [S, T]). When given, BOTH reductions in
    this function switch to their masked forms — advantage normalisation over
    participating rows only, and a masked mean over the per-row pg terms.
    Doing only one of the two would be wrong in a way no test at
    n_active=TEAM_SIZE can see: unmasked normalisation shifts the parked rows'
    advantage off 0, and that offset survives into pg through their (arbitrary)
    ratio. None ⇒ the pre-Rung-0 lines run verbatim; that path is what
    test_return_pg_rows_default_is_bitwise_identical_7_tuple pins, so
    production at n_active=5 takes the MASKED branch with an all-ones weight
    and is identical to the old numbers only to fp tolerance, not bitwise.
    """
    import torch
    import torch.nn.functional as F

    logits_list, mu_aim, log_std_aim, new_value = policy(mb_obs, state)

    # ── Shape harmonisation ──
    # PufferLib's PPO update path passes mb_obs with shape (segments,
    # bptt_horizon, OBS_DIM); HybridPolicy.forward flattens to (B*T, ...)
    # before the heads, so logits/mu_aim/log_std/new_value come back at the
    # FLAT batch dim while mb_actions / mb_cont_actions / mb_advantages /
    # mb_old_logp_{d,c} retain their original (segments, bptt_horizon, …)
    # shape. Flatten the latter to match logits' batch dim. If mb_actions
    # is already 2D (test path passing flat tensors directly), .view keeps
    # it 2D — a no-op.
    flat_actions = mb_actions.reshape(-1, mb_actions.shape[-1])
    flat_cont = mb_cont_actions.reshape(-1, mb_cont_actions.shape[-1])
    flat_old_d = mb_old_logp_d.reshape(-1)
    flat_old_c = mb_old_logp_c.reshape(-1)
    flat_adv = mb_advantages.reshape(-1)

    # ── Advantage normalization + prio-IS weight (finding 1, 2026-07-06
    # adversarial review) ──
    # Stock PufferLib normalizes per-minibatch and applies the prioritized-
    # replay importance weight BEFORE the pg term:
    #     adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)
    # The T5 refactor moved the pg term into this function but fed it RAW
    # advantages, orphaning the normalization at the call site. Consequence
    # (verified at 98e3f32): with sparse rewards the pg gradient scale was
    # ~0, so the entropy objective faced no counter-pressure and the 30M
    # run drifted to an exactly-uniform discrete policy. Normalizing HERE
    # (not at the call site) makes the contract self-contained and lets the
    # test assert scale-invariance of pg_loss directly.
    # PITFALLS:
    #   * Normalize the RAW adv first, then multiply by mb_prio — reversing
    #     the order changes the statistics (matches stock).
    #   * mb_prio arrives as (segments, 1) from the trainer (broadcast over
    #     bptt_horizon) or (B,) from tests; expand_as handles both. None ⇒
    #     uniform replay (weight 1), e.g. BC/eval callers.
    #   * A constant-adv minibatch has std 0 ⇒ normalized adv is exactly 0
    #     (0/1e-8); pg_loss 0, no NaN.
    #   * mb_part (Rung 0 §2.2): stats over participating rows only, and
    #     parked rows are zeroed so they contribute nothing to pg regardless
    #     of their ratio — which is what the masked pg mean below then
    #     divides out. The mb_prio multiply stays AFTER, unchanged.
    if mb_part is None:
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
    else:
        flat_adv = masked_normalize_adv(flat_adv, mb_part.reshape(-1))
    if mb_prio is not None:
        flat_adv = mb_prio.expand_as(mb_advantages).reshape(-1) * flat_adv

    # ── Re-evaluate discrete and continuous halves under the new policy ──
    # Fix #3 (perf): replaces 7× torch.distributions.Categorical(logits=lg) +
    # 1× torch.distributions.Normal(mu, sigma) construction per minibatch
    # with the same hand-rolled forms used in _hybrid_sample_logits (Fix #2).
    # Microbench measured 8.6 ms/MB savings; PPO update calls this 10×4 = 40
    # times per epoch → ~345 ms/epoch saved on heavy-update epochs. Same
    # numerical contract as before: |Δ| ≤ ~2e-6 vs torch.distributions
    # reference (different softmax reduction order; well within fp32 noise).
    # F8: apply the rollout's action masks to the fresh logits so new_logp /
    # entropy are computed over the SAME masked distribution the sampler drew
    # from — otherwise ratios drift wherever a mask bit was 0.
    if mb_masks is not None:
        flat_masks = mb_masks.reshape(-1, mb_masks.shape[-1])
        logits_list = _apply_action_masks(logits_list, flat_masks)
    log_probs_per_head = [F.log_softmax(lg, dim=-1) for lg in logits_list]
    new_logp_d = sum(
        lp.gather(-1, flat_actions[..., i:i + 1]).squeeze(-1)
        for i, lp in enumerate(log_probs_per_head))
    entropy_d = sum(-(lp.exp() * lp).sum(-1) for lp in log_probs_per_head)

    sigma = torch.exp(log_std_aim).expand_as(mu_aim)
    log_std_b = log_std_aim.expand_as(mu_aim)
    diff = (flat_cont - mu_aim) / sigma
    w_aim = _aim_dim_weight(aim_dim_mask, mu_aim)
    new_logp_c = ((-0.5 * diff * diff - log_std_b - 0.5 * _LOG_2PI) * w_aim).sum(-1)
    entropy_c = ((0.5 + 0.5 * _LOG_2PI + log_std_b) * w_aim).sum(-1)
    entropy = entropy_d + entropy_c if aim_entropy_bonus else entropy_d

    # ── Per-factor PPO ratios + clipped loss ──
    # max(unclipped, clipped) is taken element-wise per factor; the per-
    # element scalars are then meaned. Summing the two means matches the
    # H-PPO Eq. 8 in Fan et al. — equal weighting of the two heads. If a
    # future variant wants weighted heads (e.g. up-weight continuous early
    # in training), introduce per-factor coefficients HERE, not by scaling
    # ratios.
    ratio_d = torch.exp(new_logp_d - flat_old_d)
    ratio_c = torch.exp(new_logp_c - flat_old_c)
    pg_d_un = -flat_adv * ratio_d
    pg_d_cl = -flat_adv * torch.clamp(ratio_d, 1 - clip_coef, 1 + clip_coef)
    pg_c_un = -flat_adv * ratio_c
    pg_c_cl = -flat_adv * torch.clamp(ratio_c, 1 - clip_coef, 1 + clip_coef)
    pg_d_rows = torch.max(pg_d_un, pg_d_cl)
    pg_c_rows = torch.max(pg_c_un, pg_c_cl)
    if mb_part is None:
        pg_loss = pg_d_rows.mean() + pg_c_rows.mean()
    else:
        # Rung 0 §2.2: parked rows are already exactly 0 in these vectors
        # (their normalised advantage is 0), so the mask only fixes the
        # DENOMINATOR — without it the gradient is scaled by n_active/5.
        _w = mb_part.reshape(-1).to(pg_d_rows.dtype)
        pg_loss = masked_mean(pg_d_rows, _w) + masked_mean(pg_c_rows, _w)

    if return_pg_rows:
        return (pg_loss, entropy, new_value, new_logp_d + new_logp_c, ratio_d, ratio_c, logits_list,
                pg_d_rows + pg_c_rows)
    return pg_loss, entropy, new_value, new_logp_d + new_logp_c, ratio_d, ratio_c, logits_list


def _tag_param_groups(policy):
    """Partition policy params into the TAG groups (spec 2026-08-13 §4.2).

    trunk        — encoder.* + lstm.* (620,544 of 626,971 trainable params,
                   99.0%, LSTM alone 526,336; this is why no 'total' group
                   exists — it would replicate trunk while reading as
                   independent signal). Trunk-split (spec 2026-08-15 §3.4):
                   encoder_t./encoder_ct./lstm_t./lstm_ct. map into this
                   SAME group, doubling it. The union is what makes the
                   T-vs-CT trunk cross cos-sim exactly 0.0 — each team's
                   gradient is zero on the other's copy — which the
                   analyzer labels structural via split/trunk_active.
    policy_heads — action_heads.* + aim_mu.* + the aim_log_std parameter
                   (6,170 params). Batch 7: under --tct-split-heads BOTH team
                   copies (action_heads_t/_ct, aim_mu_t/_ct, aim_log_std_t/_ct)
                   map to this ONE group, doubling it to 12,340. The union is
                   deliberate — it is what makes the T-vs-CT cross cos-sim
                   exactly 0.0 (each team's gradient is zero on the other's
                   copy), which the analyzer labels as a structural artifact
                   rather than a conflict (spec §3.4). Splitting the group per
                   team instead would produce a within-copy number that
                   answers a different question than the trunk cells.
    value_head   — value_head.* (257 params; used ONLY for the vf control —
                   pg metrics skip it, the pg graph never touches it).

    Uses named_parameters() filtered to requires_grad — NOT state_dict(),
    which would sweep in non-trainable buffers (e.g. max_turn_speed).
    PITFALL: an unmapped parameter RAISES. Silent fallthrough would let a
    renamed module drop out of every group and the metric would quietly
    measure a subset of the network.
    """
    groups = {"trunk": [], "policy_heads": [], "value_head": []}
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(
            ("encoder.", "encoder_t.", "encoder_ct.", "lstm.", "lstm_t.", "lstm_ct.")):
            groups["trunk"].append(p)
        elif name.startswith(("action_heads.", "action_heads_t.", "action_heads_ct.",
                              "aim_mu.", "aim_mu_t.", "aim_mu_ct.")) \
                or name in ("aim_log_std", "aim_log_std_t", "aim_log_std_ct"):
            groups["policy_heads"].append(p)
        elif name.startswith("value_head."):
            groups["value_head"].append(p)
        else:
            raise AssertionError(
                f"TAG: unmapped policy parameter {name!r} — update _tag_param_groups")
    return groups


def tag_grad_cossim(policy,
                    *,
                    mb_obs,
                    mb_actions,
                    mb_cont_actions,
                    mb_old_logp_d,
                    mb_old_logp_c,
                    mb_advantages,
                    clip_coef,
                    state,
                    mb_prio,
                    mb_masks,
                    mb_returns_norm,
                    idx,
                    mb_label,
                    mb_part=None,
                    aim_dim_mask=None,
                    aim_entropy_bonus=True):
    """T-vs-CT gradient cosine-similarity measurement (spec 2026-08-13 §4.2).

    aim_dim_mask / aim_entropy_bonus (R0-E): forwarded verbatim to the inner
    _hybrid_ppo_loss so its ratio_c matches the real update's — otherwise the
    TAG subset gradients would include a pinned pitch factor and stop being
    restrictions of the actual gradient.

    WHAT: ONE extra forward via _hybrid_ppo_loss(return_pg_rows=True), then
    six subset losses as weighted means over the per-row pg vector — T, CT,
    and the env-parity halves T_a/T_b, CT_a/CT_b — each differentiated with
    torch.autograd.grad against the SAME graph (retain_graph=True because
    successive grad calls need it; the real update's graph is a separate,
    untouched object). Advantage normalization is shared over the full
    minibatch (it lives inside _hybrid_ppo_loss, before the per-row max),
    so every subset gradient is a true restriction of the real gradient.
    vf-only T/CT losses on the same forward's new_value feed the
    known-anticorrelated tag/cossim_vf control over value_head params.

    Cost per measured minibatch: 1 forward + 8 backwards (spec §4.3 budget:
    ≤5% of epoch time at --tag-every 5; raise tag_every if exceeded, don't
    optimize).

    WHY cross_half exists: the criterion statistic is cos(g_Ta, g_CTa) —
    size-matched to the within-team null (all arms at n/2 rows). The
    full-size cos(g_T, g_CT) is reported as the lower-noise descriptive
    number but has a LARGER expected same-distribution cosine than any n/2
    statistic, which would bias within − cross toward "no conflict"
    (plan-review finding 2).

    WHY the entropy term is absent: the pg vector contains no entropy —
    deliberate (spec §4.2): entropy is team-agnostic (pushes cos-sim toward
    +1 mechanically) and its effective_alpha is warmstart-phase-dependent.

    ROW IDENTITY: segment index ≡ global agent index (env-major, 10/env, T
    at slots 0-4 — the trainer asserts segments == total_agents, gh#85), so
    team T rows are (idx % 10) < 5 and env parity is (idx // 10) % 2, where
    idx is the minibatch's multinomial segment gather.

    PITFALLS:
    * Never touches .grad, self.ratio, KL bookkeeping, or the Welford
      return-norm state — mb_returns_norm arrives already normalized.
      Training with the flag on is bitwise-identical (pinned by
      tests/test_tag_trainer.py).
    * Zero-norm subsets (subset advantage exactly 0 after shared
      normalization) yield a DELIBERATE NaN cos-sim (0/0) and gnorm 0 —
      analysis drops them; do not "fix" with an epsilon. These NaNs are
      also why the outer-loop injection must stay after
      dead_run_detector.check (see _inject_tag_metrics).
    * pg metrics cover trunk + policy_heads only — the pg graph never
      touches value_head, so those keys would be dead NaN/0 noise.
    * The loss path evaluates stored actions and samples nothing, so there
      is no RNG interaction.
    * mb_part (Rung 0 §2.2) is forwarded verbatim to _hybrid_ppo_loss so the
      shared advantage normalisation is the SAME one the real update used —
      that shared normalisation is what makes each subset gradient a true
      restriction of the real gradient. Parked rows land in whichever team
      subset their slot belongs to, but their pg_rows entries are exactly 0,
      so they only inflate the subset means' denominators (w.sum() here
      counts rows, not participation) — a uniform rescale that cosine
      similarity is invariant to. The reported gnorms ARE scaled by it.
      The vf control is weaker: mb_returns_norm on a parked row is
      -_ret_mean/std, not 0, so parked rows add a common-mode residual to
      BOTH team value gradients and bias tag/cossim_vf upward at
      n_active < TEAM_SIZE. Read that control with n_active in mind.
    """
    import torch

    groups = _tag_param_groups(policy)
    pg_group_names = ("trunk", "policy_heads")
    pg_params = [p for g in pg_group_names for p in groups[g]]
    sizes = [len(groups[g]) for g in pg_group_names]
    bounds = [sum(sizes[:i]) for i in range(len(sizes) + 1)]

    team_t = (idx % 10) < 5
    env_even = ((idx // 10) % 2) == 0                  # env-parity split (exchangeable)
    subsets = {
        "T": team_t,
        "CT": ~team_t,
        "T_a": team_t & env_even,
        "T_b": team_t & ~env_even,
        "CT_a": (~team_t) & env_even,
        "CT_b": (~team_t) & ~env_even,
    }

    def _row_weights(mask):
        # segment mask → flat per-row weights, matching pg_rows' layout
        # ((segments, bptt).reshape(-1) segment-major; flat test path is 1:1)
        w = mask.to(mb_advantages.dtype)
        if mb_advantages.dim() > 1:
            w = w.unsqueeze(1)
        return w.expand_as(mb_advantages).reshape(-1)

    def _flat(grads):
        return torch.cat([g.reshape(-1) for g in grads])

    def _cos(a, b):
        # 0-norm ⇒ 0/0 ⇒ NaN, deliberately (see docstring)
        return float((a @ b) / (a.norm() * b.norm()))

    (_, _, newvalue, *_rest, pg_rows) = _hybrid_ppo_loss(policy,
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
                                                         return_pg_rows=True,
                                                         mb_part=mb_part,
                                                         aim_dim_mask=aim_dim_mask,
                                                         aim_entropy_bonus=aim_entropy_bonus)

    pg_grads = {}                                                      # subset -> {group: flat grad}
    vf_grads = {}                                                      # 'T'/'CT' -> flat value_head grad
    for name, mask in subsets.items():
        w = _row_weights(mask)
        loss_s = (pg_rows * w).sum() / w.sum()
        gs = torch.autograd.grad(loss_s,
                                 pg_params,
                                 retain_graph=True,
                                 allow_unused=True,
                                 materialize_grads=True)
        pg_grads[name] = {
            g: _flat(gs[bounds[i]:bounds[i + 1]]).detach()
            for i, g in enumerate(pg_group_names)
        }
        if name in ("T", "CT"):
                                                                       # vf-only control: unclipped value loss restricted to the subset
            newv = newvalue.view(mb_returns_norm.shape)
            w_full = w.reshape(mb_returns_norm.shape)
            vf_s = 0.5 * (((newv - mb_returns_norm)**2) * w_full).sum() / w_full.sum()
            vgs = torch.autograd.grad(vf_s,
                                      groups["value_head"],
                                      retain_graph=True,
                                      allow_unused=True,
                                      materialize_grads=True)
            vf_grads[name] = _flat(vgs).detach()

    out = {}
    for g in pg_group_names:
        out[f"tag/cossim_cross/{g}/{mb_label}"] = _cos(pg_grads["T"][g], pg_grads["CT"][g])
        out[f"tag/cossim_cross_half/{g}/{mb_label}"] = _cos(pg_grads["T_a"][g], pg_grads["CT_a"][g])
        out[f"tag/cossim_within_t/{g}/{mb_label}"] = _cos(pg_grads["T_a"][g], pg_grads["T_b"][g])
        out[f"tag/cossim_within_ct/{g}/{mb_label}"] = _cos(pg_grads["CT_a"][g], pg_grads["CT_b"][g])
        out[f"tag/gnorm_t/{g}/{mb_label}"] = float(pg_grads["T"][g].norm())
        out[f"tag/gnorm_ct/{g}/{mb_label}"] = float(pg_grads["CT"][g].norm())
    out[f"tag/cossim_vf/{mb_label}"] = _cos(vf_grads["T"], vf_grads["CT"])
    return out
