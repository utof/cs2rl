"""R0-E.2 (#131): pin_pitch — C ignores cont[:,1]; trainer log_prob_c excludes it.

Also the crouch gate (StaticData.crouch_enabled → compute_masks) and the
arena hit test the preflight ruled mandatory: pinning pitch in C but
forgetting the crouch mask leaves |rz| = 24 > HIT_HALF_WIDTH on every
stand-vs-crouch shot — a miss the policy cannot observe — and every OTHER
test in this file would still pass.
"""
import math

import numpy as np
import pytest
import torch

from _action_spec import ACTION_HEAD_SIZES

N_AGENTS, ACTION_DIM, AIM_DIM = 10, 7, 2
H_SHOOT = 1
HEAD_CROUCH = 5                        # cs2_types.h enum (HEAD_JUMP = 6)


def _zero(env_cont=None):
    return (np.zeros((N_AGENTS, ACTION_DIM), np.int32), np.zeros((N_AGENTS, AIM_DIM), np.float32))


def test_c_ignores_pitch_when_pinned(simple_map):
    from c_env.cs2_env import make_env
    env = make_env(map_data=simple_map, pin_pitch=1, seed=1)
    try:
        env.reset()
        act, cont = _zero()
        cont[0, 1] = 0.4
        env.step(act, cont)
        assert env._c_env.game.agents[0].pitch == 0.0
        assert env._c_env.episode_stats.aim_delta_pitch_count == 0
    finally:
        env.close()


def test_c_applies_pitch_when_unpinned(simple_map):
    """Control for the test above: the default path must still consume cont[:,1]
    (otherwise a broken gate that pins EVERYONE would pass the pinned test)."""
    from c_env.cs2_env import make_env
    env = make_env(map_data=simple_map, seed=1)
    try:
        env.reset()
        act, cont = _zero()
        cont[0, 1] = 0.4
        env.step(act, cont)
        assert env._c_env.game.agents[0].pitch == pytest.approx(0.4, abs=1e-6)
        assert env._c_env.episode_stats.aim_delta_pitch_count == N_AGENTS
    finally:
        env.close()


def test_crouch_masked_when_disabled(simple_map):
    from c_env.cs2_env import make_env
    moff = np.concatenate([[0], np.cumsum(ACTION_HEAD_SIZES)[:-1]])
    for flag, expect in ((1, 1), (0, 0)):
        env = make_env(map_data=simple_map, crouch_enabled=flag, seed=1)
        try:
            env.reset()
            assert int(env._masks_view[0, moff[HEAD_CROUCH] + 1]) == expect
            assert int(env._masks_view[0, moff[HEAD_CROUCH] + 0]) == 1
        finally:
            env.close()


def _place_duel(env):
    """Agent 0 (T) and agent 5 (CT) 40u apart in agent 0's spawn room, agent 0
    facing agent 5 dead-on (same pattern as tests/test_stepstats_export.py)."""
    env.reset()
    ag = env._c_env.game.agents
    a0, a5 = ag[0], ag[5]
    a5.x, a5.y, a5.z = a0.x + 40.0, a0.y, a0.z
    a5.area_idx = a0.area_idx
    a5.facing = math.pi
    a0.facing = 0.0
    act, cont = _zero()
    act[0, H_SHOOT] = 1
    return act, cont


def test_arena_stance_parity_hit_and_stance_blocked(simple_map):
    """Preflight ruling: pin 1 / crouch 0, two standing agents, on-target shot
    ⇒ shots_hit == 1. Then crouch_enabled=1 with a CROUCHED target ⇒ the same
    shot is stance-blocked (|rz| = 24 > 16) — proving the crouch gate is what
    makes the pinned-pitch duel winnable, not the pitch pin alone.
    auto_reset=False: at n_active=1 a head-roll hit can end the round, and
    the auto-reset would clear episode_stats before the asserts read it."""
    from c_env.cs2_env import make_env
    env = make_env(map_data=simple_map,
                   n_active_per_team=1,
                   pin_pitch=1,
                   crouch_enabled=0,
                   seed=1,
                   auto_reset=False)
    try:
        act, cont = _place_duel(env)
        cont[0, 1] = 0.7               # ignored: pitch is pinned
        obs, *_ = env.step(act, cont)
        assert obs[0][56 + 3] == 1.0, "agent 0 cannot see agent 5 — _place_duel geometry"
        es = env._c_env.episode_stats
        assert es.shots_fired == 1
        assert es.shots_on_target == 1
        assert es.shots_hit == 1
        assert es.shots_stance_blocked == 0
        assert env._c_env.game.agents[0].pitch == 0.0
    finally:
        env.close()

    env = make_env(map_data=simple_map,
                   n_active_per_team=1,
                   pin_pitch=1,
                   crouch_enabled=1,
                   seed=1,
                   auto_reset=False)
    try:
        act, cont = _place_duel(env)
        act[5, HEAD_CROUCH] = 1        # process_movement sets is_crouching before combat
        obs, *_ = env.step(act, cont)
        assert obs[0][56 + 3] == 1.0
        es = env._c_env.episode_stats
        assert env._c_env.game.agents[5].is_crouching == 1
        assert es.shots_fired == 1
        assert es.shots_on_target == 1
        assert es.shots_stance_blocked == 1
        assert es.shots_hit == 0
    finally:
        env.close()


def test_aim_dim_mask_shapes_logprob_and_entropy():
    from train import _hybrid_sample_logits
    B = 4
    logits = [torch.zeros(B, n) for n in ACTION_HEAD_SIZES]
    mu = torch.zeros(B, AIM_DIM)
    ls = torch.full((B, AIM_DIM), math.log(0.1))                                  # (B, AIM_DIM) as the rollout passes it
    cont = torch.tensor([[0.1, 0.3]] * B)
    _, _, _, lp_full, _, ent_full = _hybrid_sample_logits((logits, mu, ls, None),
                                                          continuous_action=cont)
    _, _, _, lp_pin, _, ent_pin = _hybrid_sample_logits((logits, mu, ls, None),
                                                        continuous_action=cont,
                                                        aim_dim_mask=torch.tensor([1.0, 0.0]))
    per_dim = -0.5 * (cont / 0.1)**2 - math.log(0.1) - 0.5 * math.log(2 * math.pi)
    assert torch.allclose(lp_full, per_dim.sum(-1), atol=1e-5)
    assert torch.allclose(lp_pin, per_dim[:, 0], atol=1e-5)
    assert torch.allclose(ent_full, 2 * ent_pin, atol=1e-5)


def test_policy_get_action_and_value_respects_aim_dim_mask(simple_map):
    """The nn.Module path (get_action_and_value) must apply the same mask as
    the functional sampler — it is what BC/eval callers and the ONNX-free
    action path use."""
    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=4, map_data=simple_map, pin_pitch=1)
    try:
        pol = trainer.policy
        assert pol.aim_dim_mask.tolist() == [1.0, 0.0]
        obs = torch.zeros(3, trainer.vecenv.single_observation_space.shape[0])
        with torch.no_grad():
            a, c, lp, ent, _v, _s = pol.get_action_and_value(obs)
            pol.aim_dim_mask.fill_(1.0)
            _a, _c, lp2, ent2, _v2, _s2 = pol.get_action_and_value(obs,
                                                                   action=a,
                                                                   continuous_action=c)
            pol.aim_dim_mask.copy_(torch.tensor([1.0, 0.0]))
            _a, _c, lp3, ent3, _v3, _s3 = pol.get_action_and_value(obs,
                                                                   action=a,
                                                                   continuous_action=c)
        assert torch.allclose(lp, lp3, atol=1e-6)
        assert torch.allclose(ent, ent3, atol=1e-6)
        assert not torch.allclose(lp, lp2, atol=1e-4)
        assert not torch.allclose(ent, ent2, atol=1e-4)
    finally:
        cleanup()


def _ratio_c_after_rollout(pin_pitch, map_data, num_envs=8):
    """Unchanged policy ⇒ ratio_c == 1 on every row, with self-play p_past=1.0
    and the pool seeded with a bit-identical snapshot of the live policy."""
    import tempfile
    from pathlib import Path

    from train import SelfPlayManager, _hybrid_ppo_loss, _patch_trainer_with_selfplay
    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=num_envs,
                                               map_data=map_data,
                                               pin_pitch=pin_pitch)
    try:
        d = Path(tempfile.mkdtemp())
        snap = d / "sp_000000.pt"
        torch.save(trainer.policy.state_dict(), snap)
        mgr = SelfPlayManager(p_past=1.0, pin_pitch=bool(pin_pitch))
        mgr._add_to_pool(snap)
        _patch_trainer_with_selfplay(trainer, mgr)
        trainer.evaluate()
        assert trainer._selfplay_used_past
        past = mgr.load_past_policy(trainer.config["device"], trainer.vecenv)
        assert torch.equal(past.aim_dim_mask, trainer.policy.aim_dim_mask)
        assert past.aim_log_std_max == trainer.policy.aim_log_std_max
        idx = torch.arange(trainer.segments)
        state = dict(action=trainer.actions[idx],
                     lstm_h=None,
                     lstm_c=None,
                     terminals=trainer.terminals[idx])
        with torch.no_grad():
            out = _hybrid_ppo_loss(trainer.policy,
                                   trainer.observations[idx],
                                   trainer.actions[idx],
                                   trainer.cont_actions[idx],
                                   trainer.logprobs_d[idx],
                                   trainer.logprobs_c[idx],
                                   torch.zeros_like(trainer.logprobs[idx]),
                                   0.2,
                                   state,
                                   mb_masks=trainer.action_masks[idx],
                                   aim_dim_mask=trainer.policy.aim_dim_mask)
        ratio_c = out[5]
        assert torch.allclose(ratio_c, torch.ones_like(ratio_c), atol=1e-4), \
            (ratio_c.min().item(), ratio_c.max().item())
        return trainer.policy.aim_dim_mask.tolist()
    finally:
        cleanup()


def test_ratio_c_identity_simple_map_unpinned(simple_map):
    assert _ratio_c_after_rollout(0, simple_map) == [1.0, 1.0]


def test_ratio_c_identity_arena_pinned(simple_map):
    # Task 12 switches this to make_arena_duel_map(); until then simple_map + pin_pitch=1.
    assert _ratio_c_after_rollout(1, simple_map) == [1.0, 0.0]


def test_env_trainer_pin_agreement_raises(simple_map):
    from train import assert_pin_pitch_agreement
    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=4, map_data=simple_map, pin_pitch=1)
    try:
        assert_pin_pitch_agreement(trainer.vecenv, trainer.policy)
        trainer.policy.aim_dim_mask.fill_(1.0)
        with pytest.raises(RuntimeError):
            assert_pin_pitch_agreement(trainer.vecenv, trainer.policy)
    finally:
        cleanup()


def test_pin_agreement_rejects_non_c_env():
    from train import assert_pin_pitch_agreement

    class _Pol:
        aim_dim_mask = torch.tensor([1.0, 1.0])

    with pytest.raises(RuntimeError):
        assert_pin_pitch_agreement(object(), _Pol())
