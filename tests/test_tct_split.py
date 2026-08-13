"""Batch 7 T/CT policy-heads split — policy-level contracts.

Spec: docs/superpowers/specs/2026-08-13-batch7-tct-heads-split-design.md
(tests 1, 2, 3, 4a, 6, 7, 9, 10 land in this file; the trainer-level
invariant re-pin is test 4b in tests/test_tag_trainer.py, the TAG param-group
partition is test 5a in tests/test_tag_diagnostic.py, the analyzer labeling is
test 5b in tests/test_analyze_tag.py, and the CLI round-trip is test 8 in
tests/test_train_cli.py).

Fixture pattern mirrors tests/test_tag_diagnostic.py: one real
make_puffer_env, policies built off it on CPU. The env is module-scoped
because make_puffer_env loads the nav graph / visibility matrix (~seconds)
and every test here only needs its observation space + static data.
"""

import pytest
import torch

import train
from _action_spec import AIM_DIM


@pytest.fixture(scope="module")
def env():
    e = train.make_puffer_env(seed=0)
    try:
        yield e
    finally:
        e.close()


# The legacy parameter-name set, SPELLED OUT (spec §5 test 1). Hardcoded on
# purpose: diffing the flag-off constructor against itself would be
# tautological, so the pin is against this literal list. A construction change
# — renamed module, extra layer, split leaking into the flag-off path — fails
# here loudly instead of silently changing what every checkpoint contains.
LEGACY_PARAM_NAMES = {
    "encoder.0.weight",
    "encoder.0.bias",
    "encoder.2.weight",
    "encoder.2.bias",
    "lstm.weight_ih_l0",
    "lstm.weight_hh_l0",
    "lstm.bias_ih_l0",
    "lstm.bias_hh_l0",
    "action_heads.0.weight",
    "action_heads.0.bias",
    "action_heads.1.weight",
    "action_heads.1.bias",
    "action_heads.2.weight",
    "action_heads.2.bias",
    "action_heads.3.weight",
    "action_heads.3.bias",
    "action_heads.4.weight",
    "action_heads.4.bias",
    "action_heads.5.weight",
    "action_heads.5.bias",
    "action_heads.6.weight",
    "action_heads.6.bias",
    "value_head.weight",
    "value_head.bias",
    "aim_mu.weight",
    "aim_mu.bias",
    "aim_log_std",
}


def _obs(n_t, n_ct, seed=0):
    """(n_t + n_ct, OBS_DIM) batch: first n_t rows are T (obs[24] == 1)."""
    torch.manual_seed(seed)
    x = torch.randn(n_t + n_ct, train.OBS_DIM) * 0.5
    x[:n_t, 24] = 1.0
    x[n_t:, 24] = 0.0
    return x


def test_flag_off_builds_exactly_the_legacy_modules(env):
    """Spec §5 test 1 (structural half): the default constructor's parameter
    name set is EXACTLY the legacy list, no _t/_ct module exists, and the
    split marker is False. Behavioral coverage of the legacy path comes from
    the whole existing suite, which exercises it heavily.
    """
    p = train.build_policy(env, device="cpu")
    assert {n for n, _ in p.named_parameters()} == LEGACY_PARAM_NAMES
    assert p.tct_split_heads is False
    assert not any(
        n.endswith(("_t", "_ct")) or "_t." in n or "_ct." in n for n, _ in p.named_parameters())
    assert not hasattr(p, "aim_log_std_t")


def test_flag_off_forward_equals_direct_legacy_head_application(env):
    """Spec §5 test 1 (behavioral half): with the flag off, forward's outputs
    equal the heads applied directly to the trunk output — proving the blend
    path is not executed at all (a blend with a degenerate all-ones mask would
    also match numerically, so this asserts against the SAME module objects
    and is paired with the structural assertion above).
    """
    p = train.build_policy(env, device="cpu")
    x = _obs(4, 4)
    logits, mu, log_std, value = p(x, state={})
    # Replicate forward's trunk with the IDENTICAL op sequence (reshape →
    # seq-first → zero-state LSTM → flatten back) so the comparison is
    # bitwise, not merely close — a re-derivation via _forward_core would
    # differ in reduction order and force a tolerance that hides real bugs.
    B, TT = x.shape[0], 1
    h = p.encoder(x.reshape(B * TT, x.shape[-1]).float())
    h = h.reshape(B, TT, p.hidden_size).transpose(0, 1)
    hc = (h.new_zeros(1, B, p.hidden_size), h.new_zeros(1, B, p.hidden_size))
    h, _ = p.lstm(h, hc)
    h = h.transpose(0, 1).reshape(B * TT, p.hidden_size)
    for i, lg in enumerate(logits):
        assert torch.equal(lg, p.action_heads[i](h))
    assert torch.equal(value, p.value_head(h))
    assert torch.equal(mu, torch.tanh(p.aim_mu(h)) * p.max_turn_speed)


def test_split_constructor_shapes_and_names(env):
    """The split policy carries BOTH head copies and no shared copy — the
    shared trunk + value head are untouched (spec §3.1).
    """
    p = train.build_policy(env, device="cpu", tct_split_heads=True)
    names = {n for n, _ in p.named_parameters()}
    assert p.tct_split_heads is True
    for stem in ("action_heads_t.0.weight", "action_heads_ct.0.weight", "aim_mu_t.weight",
                 "aim_mu_ct.weight", "aim_log_std_t", "aim_log_std_ct"):
        assert stem in names, stem
    assert "aim_log_std" not in names
    assert not any(n.startswith(("action_heads.", "aim_mu.")) for n in names)
    assert {"encoder.0.weight", "lstm.weight_ih_l0", "value_head.weight"} <= names
    assert p.aim_log_std_t.shape == (AIM_DIM, )
    assert "tct_split_heads" not in p.state_dict(), \
        "the split marker must be a plain attribute, never a state_dict entry"


def test_pure_team_batch_leaves_other_copy_gradient_exactly_zero(env):
    """Spec §5 test 2: a batch of pure-T rows must leave EVERY CT-copy
    parameter's gradient exactly zero (and vice versa); a mixed batch makes
    both nonzero. This is the routing correctness proof — the blend weight is
    0 on the other team's copy, so autograd contributes literally nothing.
    """
    p = train.build_policy(env, device="cpu", tct_split_heads=True)

    def _grads(x):
        p.zero_grad(set_to_none=True)
        logits, mu, _log_std, _v = p(x, state={})
        (sum(lg.sum() for lg in logits) + mu.sum()).backward()
        out = {}
        for name, param in p.named_parameters():
            out[name] = 0.0 if param.grad is None else param.grad.abs().sum().item()
        return out

    g_t = _grads(_obs(8, 0))
    assert all(v == 0.0 for n, v in g_t.items() if "_ct" in n), \
        "pure-T batch leaked gradient into a CT copy"
    assert any(v > 0.0 for n, v in g_t.items() if "_t" in n)

    g_ct = _grads(_obs(0, 8))
    assert all(v == 0.0 for n, v in g_ct.items() if "_t" in n), \
        "pure-CT batch leaked gradient into a T copy"
    assert any(v > 0.0 for n, v in g_ct.items() if "_ct" in n)

    g_mix = _grads(_obs(4, 4))
    assert any(v > 0.0 for n, v in g_mix.items() if "_t" in n)
    assert any(v > 0.0 for n, v in g_mix.items() if "_ct" in n)


def test_obs_bit_selects_the_serving_copy_in_every_forward_path(env):
    """Spec §5 test 4a: flipping obs[24] on a row flips which copy serves it,
    in forward (2D and 3D) and forward_eval. Constructed by zeroing one copy's
    aim_mu bias and setting the other's to a marker value, so the served
    copy is readable straight off mu_aim's sign.

    PITFALL under test (spec §3.2): on a 3D (B, T, obs) input the mask must be
    x[..., 24].reshape(B*TT, 1). Writing x[:, 24] there selects TIMESTEP 24 —
    the exact silent bug this test exists to catch, which is why the 3D case
    uses T > 1 with a per-row (not per-timestep) team assignment.
    """
    p = train.build_policy(env, device="cpu", tct_split_heads=True)
    with torch.no_grad():
        for m in (p.aim_mu_t, p.aim_mu_ct):
            m.weight.zero_()
        p.aim_mu_t.bias.fill_(5.0)     # tanh(+5) ≈ +1 → mu > 0 means "T copy served"
        p.aim_mu_ct.bias.fill_(-5.0)   # tanh(-5) ≈ -1 → mu < 0 means "CT copy served"

    x2 = _obs(3, 3)
    for path in (lambda z: p(z, state={}), lambda z: p.forward_eval(z, state={})):
        _lg, mu, _ls, _v = path(x2)
        assert (mu[:3] > 0).all(), "T rows must be served by the _t copy"
        assert (mu[3:] < 0).all(), "CT rows must be served by the _ct copy"

    # 3D training path: 4 segments × 6 timesteps; segments 0/1 are T, 2/3 CT.
    x3 = torch.randn(4, 6, train.OBS_DIM) * 0.5
    x3[:2, :, 24] = 1.0
    x3[2:, :, 24] = 0.0
    _lg, mu3, _ls, _v = p(x3, state={})
    mu3 = mu3.reshape(4, 6, AIM_DIM)
    assert (mu3[:2] > 0).all(), "3D path: T segments must hit the _t copy"
    assert (mu3[2:] < 0).all(), "3D path: CT segments must hit the _ct copy"


def test_log_std_is_clamped_per_copy_then_blended(env):
    """Spec §3.2: clamp each copy, THEN blend. With one copy driven far above
    LOG_STD_MAX the served value must be the CLAMP, not a blend of raw
    parameters — same result for a 0/1 mask either way, but this pins the
    order §3.6's per-team σ logs depend on.
    """
    p = train.build_policy(env, device="cpu", tct_split_heads=True)
    with torch.no_grad():
        p.aim_log_std_t.fill_(10.0)    # far above LOG_STD_MAX
        p.aim_log_std_ct.fill_(train.LOG_STD_MIN)
    _lg, _mu, log_std, _v = p(_obs(2, 2), state={})
    assert torch.allclose(log_std[:2], torch.full_like(log_std[:2], train.LOG_STD_MAX))
    assert torch.allclose(log_std[2:], torch.full_like(log_std[2:], train.LOG_STD_MIN))


def test_get_action_and_value_routes_by_team(env):
    """Spec §3.1: get_action_and_value is split for consistency even though no
    production path calls it (its only in-tree caller is
    tests/test_train_env.py:1181; train_bc.py uses forward_eval). Same marker
    trick as the forward test, read off the returned value/continuous action.
    """
    p = train.build_policy(env, device="cpu", tct_split_heads=True)
    with torch.no_grad():
        for m in (p.aim_mu_t, p.aim_mu_ct):
            m.weight.zero_()
        p.aim_mu_t.bias.fill_(5.0)
        p.aim_mu_ct.bias.fill_(-5.0)
        p.aim_log_std_t.fill_(train.LOG_STD_MIN)       # σ ≈ 0.01: sample ≈ μ
        p.aim_log_std_ct.fill_(train.LOG_STD_MIN)
    torch.manual_seed(0)
    _a, cont, _lp, _ent, _v, _st = p.get_action_and_value(_obs(3, 3))
    assert (cont[:3] > 0).all()
    assert (cont[3:] < 0).all()
