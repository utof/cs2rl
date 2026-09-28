"""TAG diagnostic — direct-call tests (spec 2026-08-13 §5, tests 2/5/6).

These tests exercise _hybrid_ppo_loss(return_pg_rows=True) and
tag_grad_cossim on a real Dust2Policy via the bare make_puffer_env path (no
trainer) — same fixture pattern as test_hybrid_loss_clip_applies_per_factor.
Trainer-level contracts (inertness, row mask, no-perturbation) live in
tests/test_tag_trainer.py.

PITFALL: _flat_batch builds old logprobs with the SAME hand-rolled
F.log_softmax / analytic-Normal forms _hybrid_ppo_loss uses — NOT
torch.distributions, which differs by ~2e-6 (see the loss's Fix #3
docstring) and would make the "ratios are exactly 1" premise false and the
analytic assertions tolerance-fragile (plan-review finding 10).
"""
import pytest
import torch
import torch.nn.functional as F

from cs2rl import train
from cs2rl._action_spec import AIM_DIM
from cs2rl.c_env.cs2_env import make_env


@pytest.fixture()
def env_policy():
    env = make_env(seed=0)
    try:
        yield env, train.build_policy(env, device="cpu")
    finally:
        env.close()


def _flat_batch(policy, B, adv, row_map=None):
    """Build a flat minibatch whose ratios are BITWISE 1 (old logp := new
    logp, computed with the loss's own hand-rolled forms).

    With ratio == 1 the clipped and unclipped pg terms coincide, so
    pg_rows_i = -2 * normalized_adv_i (discrete + continuous factors) — an
    analytic target the subset-mean test can hit at tight tolerance.

    row_map (LongTensor of length B) re-indexes obs/actions/cont AFTER
    generation, so the sign-check tests can tie row identities together:
    row_map=zeros → every row identical; a T↔CT mirror map → each CT row is
    an exact copy of its T counterpart. Identity control is what makes the
    ±1 cosine targets EXACT rather than approximate.
    """
    torch.manual_seed(7)
    mb_obs = torch.randn(B, train.OBS_DIM) * 0.5
    mb_actions = torch.randint(0, 2, (B, 7), dtype=torch.int64)
    mb_cont = (torch.rand(B, AIM_DIM) - 0.5) * 0.4
    if row_map is not None:                            # tie row identities together
        mb_obs = mb_obs[row_map]
        mb_actions = mb_actions[row_map]
        mb_cont = mb_cont[row_map]
    with torch.no_grad():
        logits_list, mu_aim, log_std_aim, _ = policy(mb_obs, state={})
        lps = [F.log_softmax(lg, dim=-1) for lg in logits_list]
        lp_d = sum(lp.gather(-1, mb_actions[..., i:i + 1]).squeeze(-1) for i, lp in enumerate(lps))
        sigma = torch.exp(log_std_aim).expand_as(mu_aim)
        log_std_b = log_std_aim.expand_as(mu_aim)
        diff = (mb_cont - mu_aim) / sigma
        lp_c = (-0.5 * diff * diff - log_std_b - 0.5 * train._LOG_2PI).sum(-1)
    return dict(mb_obs=mb_obs,
                mb_actions=mb_actions,
                mb_cont_actions=mb_cont,
                mb_old_logp_d=lp_d,
                mb_old_logp_c=lp_c,
                mb_advantages=adv,
                clip_coef=0.2,
                state={})


def test_return_pg_rows_default_is_bitwise_identical_7_tuple(env_policy):
    """Spec §5 test 5: the default call keeps the exact 7-tuple contract and
    pg_loss is bitwise equal across both modes — the refactor only NAMES the
    max(...) intermediates, it must not reorder the arithmetic.
    """
    _, policy = env_policy
    B = 16
    kw = _flat_batch(policy, B, torch.randn(B))
    ret_default = train._hybrid_ppo_loss(policy, **kw)
    assert len(ret_default) == 7
    ret_rows = train._hybrid_ppo_loss(policy, **kw, return_pg_rows=True)
    assert len(ret_rows) == 8
    assert torch.equal(ret_default[0], ret_rows[0])
    pg_rows = ret_rows[7]
    assert pg_rows.shape == (B, )
    assert pg_rows.requires_grad, "pg_rows must stay graph-attached"
    # mean-of-sums vs sum-of-means: fp reduction order only
    assert torch.allclose(pg_rows.mean(), ret_rows[0], atol=1e-7)


def test_pg_rows_subset_mean_hits_analytic_target(env_policy):
    """Spec §4.2/§5: the subset weighted mean (pg_rows·w)/Σw with a 0/1 mask
    must equal -2 * mean(normalized_adv[mask]) where normalization
    statistics come from the FULL minibatch — a per-subset re-normalization
    (the bug the spec review flagged) lands on a different number and fails.
    Ratios are bitwise 1 (see _flat_batch), so the target is analytic.
    """
    _, policy = env_policy
    B = 16
    adv = torch.arange(B, dtype=torch.float32)                         # asymmetric on purpose
    kw = _flat_batch(policy, B, adv)
    *_, pg_rows = train._hybrid_ppo_loss(policy, **kw, return_pg_rows=True)
    mask = (torch.arange(B) < 4).float()                               # subset mean != full mean
    subset_loss = (pg_rows * mask).sum() / mask.sum()
    adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
    expected = -2.0 * adv_norm[:4].mean()
    assert torch.allclose(subset_loss, expected, atol=1e-6), \
        f"{subset_loss} != {expected}"


def test_tag_param_groups_partition_the_policy(env_policy):
    """Spec §4.2: trunk = encoder+lstm (620,544 of 626,971 params — 99%),
    policy_heads = action_heads+aim_mu+aim_log_std (6,170), value_head
    (257). The three groups must PARTITION named_parameters (requires_grad
    only) — an unmapped param must raise, not silently vanish from every
    group (a renamed module would otherwise drop out of the metric
    unnoticed).
    """
    _, policy = env_policy
    groups = train._tag_param_groups(policy)
    assert set(groups) == {"trunk", "policy_heads", "value_head"}
    n_grouped = sum(len(ps) for ps in groups.values())
    n_policy = sum(1 for _, p in policy.named_parameters() if p.requires_grad)
    assert n_grouped == n_policy
    assert all(len(ps) > 0 for ps in groups.values())
    numel = {g: sum(p.numel() for p in ps) for g, ps in groups.items()}
    assert numel["trunk"] > 100 * numel["policy_heads"]                # 99% dominance


def _tag_call(policy, kw, idx):
    B = kw["mb_advantages"].shape[0]
    return train.tag_grad_cossim(
        policy,
        idx=idx,
        mb_returns_norm=torch.zeros(B),
        mb_prio=None,
        mb_masks=None,
        mb_label="mb0",
        **kw,
    )


def test_synthetic_sign_check(env_policy):
    """Spec §5 test 2 — both directions, with EXACT cosine targets.

    Opposed: every row is the same transition (row_map = all-zeros) with
    adv +1 on T rows and -1 on CT rows. Each subset loss is then a scalar
    multiple of the single row's pg term with opposite signs ⇒ cross and
    cross_half exactly -1; within-team halves see identical rows ⇒ exactly
    +1.

    Aligned: each CT row is a bitwise COPY of its T counterpart (mirror
    row_map, mirrored advantages), so loss_T ≡ loss_CT as functions of the
    parameters ⇒ identical gradients ⇒ cross exactly +1.

    NOTE the shared-normalization trap this construction dodges: two
    team-constant advantage levels can never co-align (mean subtraction
    puts them on opposite sides of zero), so the aligned case must mirror
    ROWS, not just choose friendly constants. A sign or mask bug cannot
    pass both branches.
    """
    _, policy = env_policy
    B = 20                             # 2 envs × 10 slots
    idx = torch.arange(B)              # (idx % 10) < 5 → rows 0-4, 10-14 are T
    team_t = (idx % 10) < 5

    adv = torch.where(team_t, torch.tensor(1.0), torch.tensor(-1.0))
    kw = _flat_batch(policy, B, adv, row_map=torch.zeros(B, dtype=torch.long))
    m = _tag_call(policy, kw, idx)
    for g in ("trunk", "policy_heads"):
        assert m[f"tag/cossim_cross/{g}/mb0"] < -0.99
        assert m[f"tag/cossim_cross_half/{g}/mb0"] < -0.99
        assert m[f"tag/cossim_within_t/{g}/mb0"] > 0.99
        assert m[f"tag/cossim_within_ct/{g}/mb0"] > 0.99
        assert m[f"tag/gnorm_t/{g}/mb0"] > 0
        assert m[f"tag/gnorm_ct/{g}/mb0"] > 0

    mirror = idx.clone()
    mirror[~team_t] -= 5                               # CT slot j ← T slot j-5
    torch.manual_seed(11)
    adv_aligned = torch.randn(B)[mirror]               # advantages mirrored too
    kw2 = _flat_batch(policy, B, adv_aligned, row_map=mirror)
    m2 = _tag_call(policy, kw2, idx)
    for g in ("trunk", "policy_heads"):
        assert m2[f"tag/cossim_cross/{g}/mb0"] > 0.99


def test_within_halves_split_by_env_parity(env_policy):
    """Spec §4.2 (plan-review finding 2): halves are split by ENV parity
    (idx // 10) % 2 — exchangeable, balanced — NOT by segment parity, which
    would pin T_a to agent slots {0,2,4} forever. Construction: 2 envs;
    env 0's T rows get adv +1, env 1's T rows get -1 (CT rows vary to keep
    the batch non-degenerate). Under env parity, within_t compares env 0
    vs env 1 ⇒ opposed ⇒ within_t < 0. Under slot parity both halves would
    mix the envs and within_t would sit near 0/positive — the assertion
    separates the two implementations.
    """
    _, policy = env_policy
    B = 20
    idx = torch.arange(B)
    row_map = torch.cat([torch.arange(10), torch.arange(10)])                 # env1 copies env0's rows
    adv = torch.zeros(B)
    adv[0:5] = 1.0                                                            # env 0 T rows
    adv[10:15] = -1.0                                                         # env 1 T rows (identical transitions)
    adv[5:10] = 0.5                                                           # CT rows keep full-batch stats sane
    adv[15:20] = 0.5
    kw = _flat_batch(policy, B, adv, row_map=row_map)
    m = _tag_call(policy, kw, idx)
    assert m["tag/cossim_within_t/trunk/mb0"] < -0.9, (
        "within_t should compare env 0 vs env 1 (opposed by construction) — "
        "a slot-parity split mixes the envs and lands elsewhere")


def test_zero_norm_subset_yields_nan_others_finite(env_policy):
    """Spec §5 test 6: a subset whose normalized advantages are exactly zero
    (CT rows pinned at the full-batch mean) produces a deliberate NaN in
    every CT-involving cos-sim and gnorm_ct == 0, while the T-side keys stay
    finite. Analysis drops NaNs; this pins that they ARE NaN rather than a
    fake 0 or a crash.
    """
    import math
    _, policy = env_policy
    B = 20
    idx = torch.arange(B)
    team_t = (idx % 10) < 5
    # T rows: ±1 alternating (mean 0); CT rows: exactly 0 == full-batch mean
    adv = torch.zeros(B)
    t_pos = team_t.nonzero().squeeze(1)
    adv[t_pos] = torch.where(
        torch.arange(len(t_pos)) % 2 == 0, torch.tensor(1.0), torch.tensor(-1.0))
    m = _tag_call(policy, _flat_batch(policy, B, adv), idx)
    for g in ("trunk", "policy_heads"):
        assert math.isnan(m[f"tag/cossim_cross/{g}/mb0"])
        assert math.isnan(m[f"tag/cossim_cross_half/{g}/mb0"])
        assert math.isnan(m[f"tag/cossim_within_ct/{g}/mb0"])
        assert m[f"tag/gnorm_ct/{g}/mb0"] == 0.0
        assert math.isfinite(m[f"tag/cossim_within_t/{g}/mb0"])
        assert m[f"tag/gnorm_t/{g}/mb0"] > 0


def test_vf_control_present_and_no_value_head_pg_keys(env_policy):
    """Spec §4.2 (plan-review finding 14): tag/cossim_vf/<mb> is the
    known-anticorrelated control (vf-only subset losses over value_head
    params) and must be finite on a generic batch. The pg loss never
    touches value_head, so pg keys for that group must NOT exist at all —
    dead NaN/0 keys in metrics.jsonl read as signal to nobody's benefit.
    """
    import math
    _, policy = env_policy
    B = 20
    idx = torch.arange(B)
    torch.manual_seed(3)
    m = _tag_call(policy, _flat_batch(policy, B, torch.randn(B)), idx)
    assert math.isfinite(m["tag/cossim_vf/mb0"])
    assert not any("value_head" in k for k in m), ("pg metrics must not be emitted for value_head")


def test_tag_param_groups_partition_a_split_policy():
    """Spec §5 test 5a: _tag_param_groups RAISES on unmapped parameters, so
    the split names must map — both team copies of action_heads, aim_mu and
    aim_log_std belong to the existing policy_heads group (the union group is
    what makes the cross cos-sim structurally 0; see spec §3.4).
    """
    env = make_env(seed=0)
    try:
        policy = train.build_policy(env, device="cpu", tct_split_heads=True)
    finally:
        env.close()
    groups = train._tag_param_groups(policy)
    assert set(groups) == {"trunk", "policy_heads", "value_head"}
    n_grouped = sum(len(ps) for ps in groups.values())
    assert n_grouped == sum(1 for _, p in policy.named_parameters() if p.requires_grad)
    heads_ids = {id(p) for p in groups["policy_heads"]}
    for name, p in policy.named_parameters():
        if "action_heads_" in name or "aim_mu_" in name or "aim_log_std_" in name:
            assert id(p) in heads_ids, f"{name} did not land in policy_heads"
    # Both copies present: the heads group is EXACTLY twice the legacy 6,170.
    # value_head (257) stays its own group under both architectures — it is
    # shared, and Batch 8 is where the critic split gets its turn.
    assert sum(p.numel() for p in groups["policy_heads"]) == 2 * 6170
    assert sum(p.numel() for p in groups["value_head"]) == 257
    assert sum(p.numel() for p in groups["trunk"]) == 620544


def test_tag_param_groups_partition_a_both_flags_policy():
    """Spec §5 test 5a (trunk): _tag_param_groups RAISES on unmapped names.

    WHAT: a both-flags policy (heads + trunk split) still partitions every
    trainable param. Both encoder/lstm copies land in `trunk` (count > the
    shared 620,544); both head copies stay in `policy_heads`.

    WHY: the union trunk group is what makes T-vs-CT cross cos-sim exactly
    0.0 (each team's gradient is zero on the other copy). Unmapped
    encoder_t/lstm_t names would raise at the first TAG epoch of the
    experiment arm.

    PITFALL: this is NOT a rewrite of the heads-only partition test above —
    that path stays the Batch 7 pin. A prefix matcher that used startswith
    ("encoder") without the trailing dot would also swallow encoder_t, but
    the live prefixes keep the dots; this test pins the mapping, not the
    string form.
    """
    env = make_env(seed=0)
    try:
        policy = train.build_policy(env, device="cpu", tct_split_heads=True, tct_split_trunk=True)
    finally:
        env.close()
    groups = train._tag_param_groups(policy)
    assert set(groups) == {"trunk", "policy_heads", "value_head"}
    n_grouped = sum(len(ps) for ps in groups.values())
    assert n_grouped == sum(1 for _, p in policy.named_parameters() if p.requires_grad)
    trunk_ids = {id(p) for p in groups["trunk"]}
    heads_ids = {id(p) for p in groups["policy_heads"]}
    for name, p in policy.named_parameters():
        if name.startswith(("encoder_t.", "encoder_ct.", "lstm_t.", "lstm_ct.")):
            assert id(p) in trunk_ids, f"{name} did not land in trunk"
        if "action_heads_" in name or "aim_mu_" in name or "aim_log_std_" in name:
            assert id(p) in heads_ids, f"{name} did not land in policy_heads"
    # Both trunk copies: strictly larger than the shared-trunk 620,544.
    assert sum(p.numel() for p in groups["trunk"]) > 620544
    assert sum(p.numel() for p in groups["policy_heads"]) == 2 * 6170
    assert sum(p.numel() for p in groups["value_head"]) == 257
