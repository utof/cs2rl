"""T/CT actor-trunk split — trunk L2 metric (spec 2026-08-15 Task 6).

Policy/routing contracts live in tests/test_tct_split.py (~800 lines).
This file holds the trunk-divergence readout so that file stays under
the plan's ~800-line cap.
"""

import pytest
import torch

import train


@pytest.fixture(scope="module")
def env():
    e = train.make_puffer_env(seed=0)
    try:
        yield e
    finally:
        e.close()


def _obs(n_t, n_ct, seed=0):
    """(n_t + n_ct, OBS_DIM) batch: first n_t rows are T (obs[24] == 1)."""
    torch.manual_seed(seed)
    x = torch.randn(n_t + n_ct, train.OBS_DIM) * 0.5
    x[:n_t, 24] = 1.0
    x[n_t:, 24] = 0.0
    return x


def test_trunk_l2_rel_emitted_and_grows_with_asymmetric_step(env):
    """≈0 after a warm split (identical T/CT copies); encoder ratio grows
    after one T-only Adam step.

    Fresh construction draws encoder_ct/lstm_ct under fork_rng, so the
    copies are *not* identical at init — same as heads. The ≈0 clause is
    the warm-split-identical condition (convert_shared_trunk_to_split),
    matching test_head_divergence_zero_at_warm_split_and_keys_present.
    """
    p = train.build_policy(env, "cpu", tct_split_heads=True, tct_split_trunk=True)
    heads_only = train.build_policy(env, "cpu", tct_split_heads=True, tct_split_trunk=False)
    p.load_state_dict(train.convert_shared_trunk_to_split(heads_only.state_dict()))
    d0 = train.compute_trunk_divergence(p)
    assert set(d0) == {"split/trunk_l2_rel/encoder", "split/trunk_l2_rel/lstm"}
    assert d0["split/trunk_l2_rel/encoder"] < 1e-6
    assert d0["split/trunk_l2_rel/lstm"] < 1e-6
    opt = torch.optim.Adam(p.parameters(), lr=1e-2, weight_decay=1e-4)
    x = _obs(4, 0)
    logits, mu, ls, v = p.forward(x, {})
    (sum(t.sum() for t in logits) + mu.sum() + v.sum()).backward()
    opt.step()
    d1 = train.compute_trunk_divergence(p)
    assert d1["split/trunk_l2_rel/encoder"] > d0["split/trunk_l2_rel/encoder"]
    assert train.compute_trunk_divergence(train.build_policy(env, "cpu")) == {}
