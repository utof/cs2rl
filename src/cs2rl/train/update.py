"""PPO update helpers (#140), split out of the flat train.py.

WHAT: the loss/reduction surface of the trainer update — the masked reductions
every trainer statistic goes through, the hybrid discrete+continuous PPO loss,
the TAG gradient-cosine diagnostic and its parameter partition, and the
entropy-target schedule. Moved here VERBATIM by the 2026-08-31 post-rung1a
refactor; the flat ``train.py`` re-exported every name below (see its ``__all__``)
until #205 part 3 removed the re-exports: ``cs2rl.train`` exports nothing now, so
import from this module.

The update itself is ``Cs2PuffeRL.train`` in ``cs2rl.train.trainer``, whose module
imports the helpers below at module scope; its phase methods call them. Nothing here
touches a trainer instance.

PITFALL (runtime rebinding): a test that wants to intercept ``tag_grad_cossim``
or ``_hybrid_ppo_loss`` at its call site must patch it on ``cs2rl.train.trainer``,
NOT on this module — the call sites (``Cs2PuffeRL._record_tag``,
``Cs2PuffeRL._loss_terms``) resolve the names through cs2rl.train.trainer's
globals, so a patch here is silently unreachable and the assertion becomes
vacuous. See tests/train/test_tag_trainer.py, whose positive control pins the
reachable module.

IMPORT-LIGHTNESS INVARIANT: module scope stays torch/nav/env.c-free, for the
reason spelled out in tests/train/test_w1_modules.py's docstring (WHY property 3 is
load-bearing). Every runtime torch, pufferlib and cs2rl.train.entropy import below is
function-local ON PURPOSE; the module-scope torch import sits under
``if TYPE_CHECKING:``, for annotations only.
"""

from typing import TYPE_CHECKING, Literal, overload

from cs2rl.policy import _LOG_2PI, _aim_dim_weight, _apply_action_masks

if TYPE_CHECKING:                      # annotations only; never at runtime
    import torch

# ── Masked reductions over participating rows (Rung 0, spec 2026-08-29 §2.2) ──
# WHY these are free functions and not methods on the trainer: a unit test can
# call them without building a trainer (a PuffeRL subclass that needs envs).
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

    Rung 0 §2.2: non-participating rows (parked agents, and the statue team
    under ``--opponent noop``) must not enter any trainer statistic. At
    n_active=1 they are four of every five rows (nine of ten under noop), and
    their entropy and reward are not 0 in general. Masked mean =
    (x·w).sum() / max(w.sum(), 1).
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


def masked_value_loss(newvalue, returns, old_values, vf_clip, part):
    """0.5 * masked mean of the squared value error, PPO-clipped when ``vf_clip`` is set.

    ``returns`` and ``old_values`` are in the value head's (normalised) scale. With
    ``vf_clip`` the prediction may move at most ``vf_clip`` from ``old_values`` and the
    larger of the clipped and unclipped errors counts; None disables clipping.
    ``part`` is the FLOAT [S, T] participation weight.
    """
    import torch

    v_loss_unclipped = (newvalue - returns)**2
    if vf_clip is None:
        return 0.5 * masked_mean(v_loss_unclipped, part)
    v_clipped = old_values + torch.clamp(newvalue - old_values, -vf_clip, vf_clip)
    v_loss_clipped = (v_clipped - returns)**2
    return 0.5 * masked_mean(torch.max(v_loss_unclipped, v_loss_clipped), part)


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

    Single source for both the construction-time seed (Cs2PuffeRL._init_return_norm)
    and the per-update recompute (Cs2PuffeRL._prepare_entropy_update).
    `config` is anything with .get() (PuffeRL config or a plain dict); missing
    keys fall back to the build_train_config defaults so harness/older-checkpoint
    configs keep working.
    """
    from cs2rl.train.entropy import target_entropy_schedule
    return target_entropy_schedule(
        global_step,
        max_entropy,
        warmup_end=config.get("entropy_target_warmup_steps", 10_000_000),
        warmup_high_frac=config.get("entropy_target_warmup_frac", 0.5),
        base_frac=config.get("entropy_target_base_frac", 0.35),
    )


# The return length follows return_pg_rows, so a type checker needs one overload per
# value to check the callers' 7- and 8-name unpacks. return_pg_rows is keyword-only
# in the True overload (a non-default parameter cannot follow defaulted ones); no
# caller passes it positionally. Typing only: the def below is the one that runs.
@overload
def _hybrid_ppo_loss(
    policy,
    mb_obs,
    mb_actions,
    mb_cont_actions,
    mb_old_logp_d,
    mb_old_logp_c,
    mb_advantages,
    clip_coef,
    state,
    mb_prio=...,
    mb_masks=...,
    return_pg_rows: Literal[False] = ...,
    mb_part=...,
    aim_dim_mask=...,
    aim_entropy_bonus=...
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]":
    ...


@overload
def _hybrid_ppo_loss(
    policy,
    mb_obs,
    mb_actions,
    mb_cont_actions,
    mb_old_logp_d,
    mb_old_logp_c,
    mb_advantages,
    clip_coef,
    state,
    mb_prio=...,
    mb_masks=...,
    *,
    return_pg_rows: Literal[True],
    mb_part=...,
    aim_dim_mask=...,
    aim_entropy_bonus=...
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor]":
    ...


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
    # bptt_horizon, OBS_DIM); Dust2Policy.forward flattens to (B*T, ...)
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
      tests/train/test_tag_trainer.py).
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
