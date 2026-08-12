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

import train
from _action_spec import AIM_DIM


@pytest.fixture()
def env_policy():
    env = train.make_puffer_env(seed=0)
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
