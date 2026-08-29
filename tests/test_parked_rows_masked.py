"""Spec 2026-08-29 §2.2 (i): every trainer reduction, computed with the
participating mask, equals the same reduction over the participating
sub-tensor alone. Plus the harness-level wiring check at n_active_per_team=1.
§2.2 (ii), the 20-update ratio identity, is APPENDED TO THIS FILE in Task 9
(it needs the harness's `aim_entropy_bonus=False` kwarg, which Task 9 adds).
"""
import numpy as np
import pytest
import torch

from _action_spec import ACTION_HEAD_SIZES
from train import masked_explained_variance, masked_mean, masked_normalize_adv, masked_std_unbiased


@pytest.fixture
def fixed():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(40, 8, generator=g)
    part = torch.zeros(40, 8, dtype=torch.bool)
    part[::5, :] = True                # rows 0,5,10,... participate (1-in-5, like 1v1)
    return x, part


def test_masked_mean_equals_subset_mean(fixed):
    x, part = fixed
    assert masked_mean(x, part.float()).item() == pytest.approx(x[part].mean().item(), abs=1e-6)


def test_masked_std_equals_subset_std(fixed):
    x, part = fixed
    w = part.float()
    m = masked_mean(x, w)
    assert masked_std_unbiased(x, w, m).item() == pytest.approx(x[part].std().item(), abs=1e-6)


def test_masked_normalize_adv_matches_subset_normalisation(fixed):
    x, part = fixed
    out = masked_normalize_adv(x.reshape(-1), part.float().reshape(-1))
    sub = x[part]
    ref = (sub - sub.mean()) / (sub.std() + 1e-8)
    assert torch.allclose(out.reshape(40, 8)[part], ref, atol=1e-6)
    assert torch.all(out.reshape(40, 8)[~part] == 0.0)


def test_masked_explained_variance_matches_subset(fixed):
    x, part = fixed
    y_true = x
    y_pred = x + 0.1 * torch.randn_like(x)
    ev = masked_explained_variance(y_pred.flatten(), y_true.flatten(), part.flatten())
    yt, yp = y_true[part], y_pred[part]
    ref = 1 - (yt - yp).var() / yt.var()
    assert ev == pytest.approx(ref.item(), abs=1e-6)


def test_masked_mean_all_ones_is_plain_mean(fixed):
    x, _ = fixed
    assert masked_mean(x, torch.ones_like(x)).item() == pytest.approx(x.mean().item(), abs=1e-6)


def test_pg_loss_masked_equals_participating_subtensor():
    """Spec §2.2 (i) for pg_loss: the COMPOSITE (masked adv-norm → per-row PPO
    terms → masked mean) must equal _hybrid_ppo_loss run on the participating
    rows alone with mb_part=None. This is the non-obvious claim; helper-level
    tests do not cover it."""
    from train import _hybrid_ppo_loss
    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=4, n_active_per_team=1)
    try:
        torch.manual_seed(0)
        pol = trainer.policy
        S, T = 10, 8                                                   # 10 segments (2 teams×5 rows), horizon 8
        obs = torch.randn(S, T, trainer.observations.shape[-1])
        act = torch.stack([torch.randint(0, n, (S, T)) for n in ACTION_HEAD_SIZES], -1)
        cont = torch.randn(S, T, 2) * 0.1
        old_d, old_c = torch.randn(S, T), torch.randn(S, T)
        adv = torch.randn(S, T)
        part = torch.zeros(S, dtype=torch.bool)
        part[0] = part[5] = True                                       # n_active=1
        part_st = part[:, None].expand(S, T)
        state = dict(action=None, lstm_h=None, lstm_c=None, terminals=torch.zeros(S, T))
        full = _hybrid_ppo_loss(pol,
                                obs,
                                act,
                                cont,
                                old_d,
                                old_c,
                                adv,
                                0.2,
                                state,
                                mb_part=part_st.to(torch.float32))
        sub = _hybrid_ppo_loss(pol,
                               obs[part],
                               act[part],
                               cont[part],
                               old_d[part],
                               old_c[part],
                               adv[part],
                               0.2,
                               dict(action=None,
                                    lstm_h=None,
                                    lstm_c=None,
                                    terminals=torch.zeros(2, T)),
                               mb_part=None)
                                                                       # full[0] / sub[0] = pg_loss
        assert full[0].item() == pytest.approx(sub[0].item(), abs=1e-6)
                                                                       # full[1] / sub[1] = entropy, returned PER ROW (the caller reduces it
                                                                       # via masked_mean) — an .item() on the 80-element tensor would raise.
                                                                       # Reduce both sides the way the caller does.
        full_entropy = masked_mean(full[1], part_st.reshape(-1).float()).item()
        assert full_entropy == pytest.approx(sub[1].mean().item(), abs=1e-6)
    finally:
        cleanup()


def test_return_stats_update_on_participating_rows_only():
    """Spec §2.2 (i) for _ret_mean/_ret_var: one _normalize_returns call from the
    zero-count state must leave the running stats equal to the participating
    sub-tensor's mean / population variance (Welford's first update)."""
    from train import _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=4, n_active_per_team=1)
    try:
        _patch_trainer_with_return_norm(trainer)
        torch.manual_seed(1)
        x = torch.randn(10, 8) * 3 + 1
        part = torch.zeros(10, 8, dtype=torch.bool)
        part[0] = part[5] = True
        trainer._normalize_returns(x, part)            # alias installed in Step 6
        sel = x[part]
        assert trainer._ret_mean.item() == pytest.approx(sel.mean().item(), abs=1e-6)
        assert trainer._ret_var.item() == pytest.approx(sel.var(unbiased=False).item(), abs=1e-6)
        assert trainer._ret_count.item() == pytest.approx(sel.numel())
    finally:
        cleanup()


def test_harness_n_active_1_masks_four_fifths_of_rows():
    from train import _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=16, n_active_per_team=1)
    try:
        _patch_trainer_with_return_norm(trainer)
        bs = trainer.config["batch_size"]
        assert trainer.config["participating_timesteps"] * 5 == trainer.config["total_timesteps"]
        trainer.evaluate()
        assert trainer.global_step == bs // 5
        n_part = trainer.participating.sum().item()
        assert n_part == trainer.segments * trainer.config["bptt_horizon"] // 5
        # parked rows carry zero critic output in the buffer
        assert torch.all(trainer.values[~trainer.participating] == 0.0)
        # force the throttled block that assigns trainer.losses
        trainer.last_log_time = 0.0
        trainer.train()
        losses = trainer.losses
        assert losses["participating_rows"] == n_part
        assert losses["empty_minibatches"] == 0
        assert 0 < trainer._ret_count.item() <= trainer.config["update_epochs"] * n_part
        assert np.isfinite(losses["entropy"]) and np.isfinite(losses["entropy_unmasked"])
        # parked rows are noop-masked ⇒ 0 discrete entropy, so the unmasked
        # mean is dragged down by the 4/5 of rows the masked mean drops
        assert losses["entropy"] > losses["entropy_unmasked"]
    finally:
        cleanup()
