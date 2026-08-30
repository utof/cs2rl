"""R0-E.3/4 (#131): per-run aim σ cap, entropy-bonus switch, §2.2(ii) parked-row integration."""
import math

import numpy as np
import pytest
import torch


def test_cap_applied_in_forward_and_metrics(simple_map):
    from train import LOG_STD_MAX, log_aim_log_std
    from train_test_harness import _build_trainer_for_test
    cap = math.log(0.05)
    trainer, cleanup = _build_trainer_for_test(num_envs=4, map_data=simple_map, aim_log_std_max=cap)
    try:
        pol = trainer.policy
        assert pol.aim_log_std_max == pytest.approx(cap)
        with torch.no_grad():
            pol.aim_log_std.fill_(LOG_STD_MAX)
        obs = torch.zeros(4, trainer.vecenv.single_observation_space.shape[0])
        _, _, log_std, _ = pol.forward_eval(obs, {
            "lstm_h": None,
            "lstm_c": None,
            "done": torch.zeros(4, dtype=torch.bool)
        })
        assert float(log_std.max()) <= cap + 1e-6
        # the BPTT training forward and the nn.Module sampler take the same cap
        _, _, log_std_tr, _ = pol(obs, {})
        assert float(log_std_tr.max()) <= cap + 1e-6
        with torch.no_grad():
            _a, c, *_ = pol.get_action_and_value(obs)
        assert c.shape == (4, 2)
        logs = {}
        log_aim_log_std(pol, logs)
        assert logs["policy/aim_log_std_yaw"] == pytest.approx(cap)
        assert logs["policy/aim_log_std_pitch"] == pytest.approx(cap)
    finally:
        cleanup()


def test_cap_outside_band_is_refused(simple_map):
    from train import LOG_STD_MAX, LOG_STD_MIN
    from train_test_harness import _build_trainer_for_test
    for bad in (LOG_STD_MIN, LOG_STD_MAX + 0.1, LOG_STD_MIN - 1.0):
        with pytest.raises(ValueError):
            trainer, cleanup = _build_trainer_for_test(num_envs=4,
                                                       map_data=simple_map,
                                                       aim_log_std_max=bad)
            cleanup()


def test_reinit_frozen_respects_cap():
    from train import AIM_LOG_STD_RESUME_INIT, LOG_STD_INIT, reinit_frozen_aim_log_std
    sd = {"aim_log_std": torch.full((2, ), LOG_STD_INIT)}
    assert reinit_frozen_aim_log_std(sd, cap=math.log(0.05))
    assert torch.allclose(sd["aim_log_std"], torch.full((2, ), math.log(0.05)))
    sd = {"aim_log_std": torch.full((2, ), LOG_STD_INIT)}
    assert reinit_frozen_aim_log_std(sd)
    assert torch.allclose(sd["aim_log_std"], torch.full((2, ), AIM_LOG_STD_RESUME_INIT))
    # a cap ABOVE the resume init leaves the resume init in charge
    sd = {"aim_log_std": torch.full((2, ), LOG_STD_INIT)}
    assert reinit_frozen_aim_log_std(sd, cap=math.log(0.4))
    assert torch.allclose(sd["aim_log_std"], torch.full((2, ), AIM_LOG_STD_RESUME_INIT))


def test_max_entropy_reflects_cap_pin_and_bonus(simple_map):
    from train import ACTION_HEAD_SIZES, _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test
    disc = sum(math.log(n) for n in ACTION_HEAD_SIZES)
    cap = math.log(0.05)
    for pin, bonus, n_dims in ((0, True, 2), (1, True, 1), (0, False, 0)):
        trainer, cleanup = _build_trainer_for_test(num_envs=4,
                                                   map_data=simple_map,
                                                   aim_log_std_max=cap,
                                                   pin_pitch=pin,
                                                   aim_entropy_bonus=bonus)
        try:
            assert trainer.config["aim_entropy_bonus"] is bonus
            assert trainer.config["aim_log_std_max"] == pytest.approx(cap)
            assert trainer.config["pin_pitch"] == pin
            _patch_trainer_with_return_norm(trainer)
            cont = n_dims * 0.5 * math.log(2 * math.pi * math.e * math.exp(cap)**2)
            assert trainer._batch1_max_entropy == pytest.approx(disc + cont)
        finally:
            cleanup()


def test_ppo_loss_entropy_bonus_switch():
    """aim_entropy_bonus=False ⇒ the entropy the loss returns is the DISCRETE
    entropy only; True (default) ⇒ discrete + Gaussian. Pure-function check on
    a fake 2D-input policy so it needs no env."""
    from _action_spec import ACTION_HEAD_SIZES, AIM_DIM
    from train import _LOG_2PI, _hybrid_ppo_loss
    B = 5

    class _Pol:

        def __call__(self, obs, state):
            logits = [torch.zeros(B, n) for n in ACTION_HEAD_SIZES]
            return logits, torch.zeros(B, AIM_DIM), torch.full((B, AIM_DIM), math.log(0.1)), \
                torch.zeros(B, 1)

    obs = torch.zeros(B, 3)
    acts = torch.zeros(B, len(ACTION_HEAD_SIZES), dtype=torch.int64)
    cont = torch.zeros(B, AIM_DIM)
    z = torch.zeros(B)
    ent_on = _hybrid_ppo_loss(_Pol(), obs, acts, cont, z, z, torch.randn(B), 0.2, {})[1]
    ent_off = _hybrid_ppo_loss(_Pol(),
                               obs,
                               acts,
                               cont,
                               z,
                               z,
                               torch.randn(B),
                               0.2, {},
                               aim_entropy_bonus=False)[1]
    ent_pin = _hybrid_ppo_loss(_Pol(),
                               obs,
                               acts,
                               cont,
                               z,
                               z,
                               torch.randn(B),
                               0.2, {},
                               aim_dim_mask=torch.tensor([1.0, 0.0]))[1]
    disc = sum(math.log(n) for n in ACTION_HEAD_SIZES)
    gauss = 0.5 + 0.5 * _LOG_2PI + math.log(0.1)
    assert torch.allclose(ent_on, torch.full((B, ), disc + 2 * gauss), atol=1e-5)
    assert torch.allclose(ent_off, torch.full((B, ), disc), atol=1e-5)
    assert torch.allclose(ent_pin, torch.full((B, ), disc + gauss), atol=1e-5)


def test_parked_rows_do_not_move_objective(simple_map):
    """Spec §2.2(ii): with parked rows, adv-norm/objective ignore them.
    Setup: n_active=1, entropy bonus off; run one train(); then perturb the
    parked rows' stored advantages/logprobs by a huge amount, re-run the
    minibatch reductions through a second identical trainer and compare
    losses/policy_loss, losses/entropy, losses/approx_kl."""
    from train import _patch_trainer_with_return_norm
    from train_test_harness import _build_trainer_for_test

    def _one(perturb):
        torch.manual_seed(0)
        np.random.seed(0)
        trainer, cleanup = _build_trainer_for_test(num_envs=16,
                                                   map_data=simple_map,
                                                   n_active_per_team=1,
                                                   aim_entropy_bonus=False)
        try:
            _patch_trainer_with_return_norm(trainer)
            trainer.evaluate()
            if perturb:
                parked = ~trainer.participating
                trainer.logprobs_d[parked] += 50.0
                trainer.logprobs_c[parked] -= 50.0
                trainer.rewards[parked] += 1e3
            # (do NOT set trainer.config["update_epochs"] here — total_minibatches is
            # fixed in PuffeRL.__init__; the mutation would only disable the KL
            # early-abort boundary check)
            trainer.last_log_time = 0.0
            trainer.train()
            # trainer.losses, not train()'s return: mean_and_log() runs BEFORE
            # self.losses is assigned, so the returned logs carry the PREVIOUS
            # update's losses/* (documented one-epoch lag).
            return {k: trainer.losses[k] for k in ("policy_loss", "entropy", "approx_kl")}
        finally:
            cleanup()

    a, b = _one(False), _one(True)
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-4, abs=1e-6), (k, a[k], b[k])
