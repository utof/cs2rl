"""T/CT actor-trunk split — remaining spec tests (2026-08-15).

Policy/routing contracts live in tests/test_tct_split.py (~800 lines).
This file holds the trunk-divergence readout (Task 6) plus spec §5
tests 6 and 10 so that file stays under the plan's ~800-line cap.
"""

import pytest
import torch

from cs2rl import policy as policy_mod
from cs2rl.env.c.cs2_env import make_env
from cs2rl.spec import obs as spec_obs
from cs2rl.train import metrics as train_metrics
from cs2rl.train import resume as train_resume
from cs2rl.train import selfplay as train_selfplay


@pytest.fixture(scope="module")
def env():
    e = make_env(seed=0)
    try:
        yield e
    finally:
        e.close()


def _obs(n_t, n_ct, seed=0):
    """(n_t + n_ct, OBS_DIM) batch: first n_t rows are T (obs[24] == 1)."""
    torch.manual_seed(seed)
    x = torch.randn(n_t + n_ct, spec_obs.OBS_DIM) * 0.5
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
    p = policy_mod.build_policy(env, "cpu", tct_split_heads=True, tct_split_trunk=True)
    heads_only = policy_mod.build_policy(env, "cpu", tct_split_heads=True, tct_split_trunk=False)
    p.load_state_dict(train_resume.convert_shared_trunk_to_split(heads_only.state_dict()))
    d0 = train_metrics.compute_trunk_divergence(p)
    assert set(d0) == {"split/trunk_l2_rel/encoder", "split/trunk_l2_rel/lstm"}
    assert d0["split/trunk_l2_rel/encoder"] < 1e-6
    assert d0["split/trunk_l2_rel/lstm"] < 1e-6
    opt = torch.optim.Adam(p.parameters(), lr=1e-2, weight_decay=1e-4)
    x = _obs(4, 0)
    logits, mu, ls, v = p.forward(x, {})
    (sum(t.sum() for t in logits) + mu.sum() + v.sum()).backward()
    opt.step()
    d1 = train_metrics.compute_trunk_divergence(p)
    assert d1["split/trunk_l2_rel/encoder"] > d0["split/trunk_l2_rel/encoder"]
    assert train_metrics.compute_trunk_divergence(policy_mod.build_policy(env, "cpu")) == {}


def test_self_play_loads_three_checkpoint_vintages(env, tmp_path):
    """Spec §5.6: legacy, heads-only, and heads+trunk snapshots each
    SelfPlayManager.load_past_policy into a both-flags run without error.

    WHAT: three vintages in a tiny pool dir; load_past_policy infers both
    architecture bits from keys (it receives no config) and returns an
    eval-mode policy. A both-flags current policy and the loaded past
    both forward the same obs.

    WHY: self-play activates on ~30% of epochs (p_past=0.3). Under a
    flag-only design a both-flags run would crash hours in when the pool
    still holds a pre-trunk snapshot — or when a heads+trunk file has no
    encoder.0.weight. The Batch 7 pin only covers legacy + heads-only.

    PITFALL: do not require the past policy to match the run's bits —
    inference is the point. A trunk-only vintage is already covered by
    test_loaders_infer_both_bits_and_warm_split_trunk_only.
    """
    pool = tmp_path / "pool"
    pool.mkdir()
    vintages = (
        (pool / "past_legacy.pt", False, False, policy_mod.build_policy(env, device="cpu")),
        (pool / "past_heads.pt", True, False,
         policy_mod.build_policy(env, device="cpu", tct_split_heads=True)),
        (pool / "past_both.pt", True, True,
         policy_mod.build_policy(env, device="cpu", tct_split_heads=True, tct_split_trunk=True)),
    )
    for path, _heads, _trunk, src in vintages:
        torch.save(src.state_dict(), path)

    current = policy_mod.build_policy(env, "cpu", tct_split_heads=True, tct_split_trunk=True)
    x = _obs(2, 2)
    with torch.no_grad():
        current.forward(x, {})
    for path, expect_heads, expect_trunk, _src in vintages:
        mgr = train_selfplay.SelfPlayManager()
        mgr.pool = [path]
        past = mgr.load_past_policy("cpu", env)
        assert past is not None, path
        assert past.tct_split_heads is expect_heads, path
        assert past.tct_split_trunk is expect_trunk, path
        assert not past.training, "past policies must be in eval mode"
        with torch.no_grad():
            past.forward(x, {})


def test_trunk_split_lstm_bptt_resets_on_terminal(env):
    """Spec §5.10: trunk-split _lstm_bptt still zero-masks at done ticks
    per team LSTM.

    WHAT: mixed T/CT batch, terminals reset tick 2 for half the rows
    (one T, one CT). After the reset, that suffix matches a fresh
    zero-state forward of the same suffix. No-done rows match an
    unmasked forward. _lstm_bptt returns the output sequence only.

    WHY: forgetting terminals= on lstm_ct (or lstm_t) leaks that copy
    across episode boundaries — biased PPO ratios and cross-episode
    grads on one copy. The flag-off pin is
    test_policy_forward_bptt_resets_on_terminal.

    PITFALL: do not change _lstm_bptt to return (h,c). A T-only batch
    would miss a CT leftover; mixed rows cover both copies.
    """
    p = policy_mod.build_policy(env, "cpu", tct_split_heads=True, tct_split_trunk=True)
    p.eval()
    torch.manual_seed(2)
    B, T, k = 4, 6, 2
    x_seq = torch.randn(B, T, spec_obs.OBS_DIM)
    # rows 0,1 T (obs[24]==1); rows 2,3 CT (obs[24]==0)
    x_seq[:2, :, 24] = 1.0
    x_seq[2:, :, 24] = 0.0
    terminals = torch.zeros(B, T)
    terminals[0, k] = 1.0              # T row resets at tick 2
    terminals[2, k] = 1.0              # CT row resets at tick 2

    with torch.no_grad():
        _, _, _, value_masked = p.forward(x_seq, state={"terminals": terminals})
        _, _, _, value_suffix_t = p.forward(x_seq[0:1, k:, :], state={})
        _, _, _, value_suffix_ct = p.forward(x_seq[2:3, k:, :], state={})
        _, _, _, value_plain = p.forward(x_seq, state={})

        H = p.hidden_size
        x_flat = x_seq.reshape(B * T, -1).float()
        h_t = p.encoder_t(x_flat).reshape(B, T, H).transpose(0, 1)
        h_ct = p.encoder_ct(x_flat).reshape(B, T, H).transpose(0, 1)
        hc_t = (h_t.new_zeros(1, B, H), h_t.new_zeros(1, B, H))
        hc_ct = (h_ct.new_zeros(1, B, H), h_ct.new_zeros(1, B, H))
        y_t = p._lstm_bptt(p.lstm_t, h_t, hc_t, terminals)
        y_ct = p._lstm_bptt(p.lstm_ct, h_ct, hc_ct, terminals)

    assert isinstance(y_t, torch.Tensor) and y_t.shape == (T, B, H)
    assert isinstance(y_ct, torch.Tensor) and y_ct.shape == (T, B, H)

    for t in range(k, T):
        assert torch.allclose(value_masked[0 * T + t], value_suffix_t[t - k],
                              atol=1e-5), (f"T row tick {t}: masked output != fresh-suffix — "
                                           f"lstm_t did not reset at terminal")
        assert torch.allclose(value_masked[2 * T + t], value_suffix_ct[t - k],
                              atol=1e-5), (f"CT row tick {t}: masked output != fresh-suffix — "
                                           f"lstm_ct did not reset at terminal")
    for t in range(T):
        assert torch.allclose(value_masked[1 * T + t], value_plain[1 * T + t],
                              atol=1e-5), f"T no-done row tick {t} perturbed by the masking path"
        assert torch.allclose(value_masked[3 * T + t], value_plain[3 * T + t],
                              atol=1e-5), f"CT no-done row tick {t} perturbed by the masking path"

    # Direct helper: each team's reset-row suffix == a fresh zero-state
    # unroll of that team's encoder activations (same pin as forward(),
    # without the shared value_head blend).
    with torch.no_grad():
        y_t_suf = p._lstm_bptt(p.lstm_t, h_t[k:, 0:1],
                               (h_t.new_zeros(1, 1, H), h_t.new_zeros(1, 1, H)), None)
        y_ct_suf = p._lstm_bptt(p.lstm_ct, h_ct[k:, 2:3],
                                (h_ct.new_zeros(1, 1, H), h_ct.new_zeros(1, 1, H)), None)
    assert torch.allclose(y_t[k:, 0:1], y_t_suf, atol=1e-5)
    assert torch.allclose(y_ct[k:, 2:3], y_ct_suf, atol=1e-5)
