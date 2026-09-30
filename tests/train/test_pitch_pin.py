"""R0-E.2 (#131): pin_pitch — C ignores cont[:,1]; trainer log_prob_c excludes it.

Also the crouch gate (StaticData.crouch_enabled → compute_masks) and the
arena hit test the preflight ruled mandatory: pinning pitch in C but
(pre-v1c) forgetting the crouch mask left |rz| = 24 > HIT_HALF_WIDTH on every
stand-vs-crouch shot — a miss the policy cannot observe — and every OTHER
test in this file would still pass.
"""
import math

import numpy as np
import pytest
import torch

from cs2rl.spec.action import ACTION_HEAD_SIZES

N_AGENTS, ACTION_DIM, AIM_DIM = 10, 7, 2
H_SHOOT = 1
HEAD_CROUCH = 5                        # cs2_types.h enum
HEAD_JUMP = 6


def _zero():
    return (np.zeros((N_AGENTS, ACTION_DIM), np.int32), np.zeros((N_AGENTS, AIM_DIM), np.float32))


def test_c_ignores_pitch_when_pinned(simple_map):
    from cs2rl.env.c.cs2_env import make_env
    from cs2rl.env.config import EnvConfig
    env = make_env(map_data=simple_map, config=EnvConfig(pin_pitch=1), seed=1)
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
    from cs2rl.env.c.cs2_env import make_env
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
    from cs2rl.env.c.cs2_env import make_env
    from cs2rl.env.config import EnvConfig
    moff = np.concatenate([[0], np.cumsum(ACTION_HEAD_SIZES)[:-1]])
    for flag, expect in ((1, 1), (0, 0)):
        env = make_env(map_data=simple_map, config=EnvConfig(crouch_enabled=flag), seed=1)
        try:
            env.reset()
            assert int(env._masks_view[0, moff[HEAD_CROUCH] + 1]) == expect
            assert int(env._masks_view[0, moff[HEAD_CROUCH] + 0]) == 1
        finally:
            env.close()


def test_jump_masked_when_disabled(simple_map):
    """jump_enabled (Rung 1a T2a) gates HEAD_JUMP bin 1, exactly as
    crouch_enabled gates HEAD_CROUCH bin 1.

    The default (1) case is the load-bearing half: the C gate ORs the new flag
    into a condition that ALREADY masks bin 1 while airborne / on cooldown /
    crouching (cs2_env.h compute_masks), so a jump_enabled read that is
    accidentally inverted — or a field packed under the wrong name, reading 0 —
    would still look "correctly masked" if only flag=0 were checked.
    Right after reset every agent is grounded with jump_cd 0, so bin 1 must be
    OPEN unless the knob closed it. Bin 0 (no jump) stays valid either way: the
    per-head no-op invariant the masked softmax depends on.
    """
    from cs2rl.env.c.cs2_env import make_env
    from cs2rl.env.config import EnvConfig
    moff = np.concatenate([[0], np.cumsum(ACTION_HEAD_SIZES)[:-1]])
    for flag, expect in ((1, 1), (0, 0)):
        env = make_env(map_data=simple_map, config=EnvConfig(jump_enabled=flag), seed=1)
        try:
            env.reset()
            assert int(env._masks_view[0, moff[HEAD_JUMP] + 1]) == expect
            assert int(env._masks_view[0, moff[HEAD_JUMP] + 0]) == 1
        finally:
            env.close()


def _place_duel(env):
    """Agent 0 (T) and agent 5 (CT) 40u apart in agent 0's spawn room, agent 0
    facing agent 5 dead-on (same pattern as tests/env/c/test_stepstats_export.py)."""
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


def test_arena_stance_parity_hit_and_stance_blocked():
    """On ARENA_DUEL_V1 (R0-H), the map Rung 1 trains on.
    Preflight ruling: pin 1 / crouch 0, two standing agents, on-target shot
    ⇒ shots_hit == 1. Then crouch_enabled=1 with a CROUCHED target: under the
    v1b 16u sphere the same shot was stance-blocked (|rz| = 24 > 16); under
    the v1c ellipsoid (gh #150) a crouched target is 54u tall, |rz| = 24 <
    HIT_HALF_HEIGHT_CROUCH = 27, so the horizontal shot still CONNECTS and is
    not stance-blocked. Third and fourth cases: airborne agents, once with the
    TARGET off the ground and once with the SHOOTER off it (gh #150 describes
    both). At the 57u apex |rz| > 36 ⇒ stance-blocked, miss (the pitch-pin
    residual); at z = 20 it is hittable either way — the geometry is symmetric
    in the sign of rz. auto_reset=False: at n_active=1 a head-roll hit can end
    the round, and the auto-reset would clear episode_stats before the asserts.

    Every case states shots_hit and shots_stance_blocked SEPARATELY on purpose:
    they answer different questions (see the counters' comment in
    cs2_combat.h), so a coupled `blocked == 1 - hit` assert would pass if both
    ever flipped together."""
    from cs2rl.env.c.cs2_env import make_env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_arena_duel_map
    arena = make_arena_duel_map()
    env = make_env(map_data=arena,
                   config=EnvConfig(n_active_per_team=1, pin_pitch=1, crouch_enabled=0),
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

    env = make_env(map_data=arena,
                   config=EnvConfig(n_active_per_team=1, pin_pitch=1, crouch_enabled=1),
                   seed=1,
                   auto_reset=False)
    try:
        act, cont = _place_duel(env)
        act[5, HEAD_CROUCH] = 1                        # process_movement sets is_crouching before combat
        obs, *_ = env.step(act, cont)
        assert obs[0][56 + 3] == 1.0
        es = env._c_env.episode_stats
        assert env._c_env.game.agents[5].is_crouching == 1
        assert es.shots_fired == 1
        assert es.shots_on_target == 1
        assert es.shots_stance_blocked == 0            # v1c: 24u < 27u crouched semi-axis
        assert es.shots_hit == 1
    finally:
        env.close()

    # (which agent leaves the ground, jump height, expected hit, expected block).
    # |rz| ends ~1.56u under z_off because the leapfrog integrator applies one
    # tick of gravity (g=800, dt=1/16) before combat — 55.4 and 18.4, so both
    # sit well clear of the 36u standing semi-axis on the correct side.
    for airborne_idx, z_off, expect_hit, expect_blocked in ((5, 57.0, 0, 1), (5, 20.0, 1, 0),
                                                            (0, 57.0, 0, 1), (0, 20.0, 1, 0)):
        env = make_env(map_data=arena,
                       config=EnvConfig(n_active_per_team=1, pin_pitch=1, crouch_enabled=0),
                       seed=1,
                       auto_reset=False)
        try:
            act, cont = _place_duel(env)
            ag = env._c_env.game.agents[airborne_idx]
            # mid-jump: is_airborne=1 keeps process_movement from snapping z back
            # to the ground (cs2_movement.h), so the offset survives into combat.
            ag.z, ag.vz, ag.is_airborne = ag.z + z_off, 0.0, 1
            obs, *_ = env.step(act, cont)
            assert obs[0][56 + 3] == 1.0
            es = env._c_env.episode_stats
            case = (airborne_idx, z_off)
            assert es.shots_fired == 1 and es.shots_on_target == 1
            assert es.shots_hit == expect_hit, case
            assert es.shots_stance_blocked == expect_blocked, case
        finally:
            env.close()


def test_aim_dim_mask_shapes_logprob_and_entropy():
    from cs2rl.policy import _hybrid_sample_logits
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
    from tests._helpers.trainer_harness import _build_trainer_for_test
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

    from cs2rl.train.selfplay import SelfPlayManager
    from cs2rl.train.update import _hybrid_ppo_loss
    from tests._helpers.trainer_harness import _build_trainer_for_test
    # gh#168 W1.5: the p_past=1.0 manager is built first and handed to the
    # harness, which constructs Cs2PuffeRL around it (the harness's own manager
    # would have p_past=0.0). The pool is seeded AFTER construction: the
    # constructor never reads it, only evaluate() does, and the snapshot has to
    # be of the policy the constructor built.
    mgr = SelfPlayManager(p_past=1.0, pin_pitch=bool(pin_pitch))
    trainer, cleanup = _build_trainer_for_test(num_envs=num_envs,
                                               map_data=map_data,
                                               pin_pitch=pin_pitch,
                                               self_play_mgr=mgr)
    try:
        d = Path(tempfile.mkdtemp())
        snap = d / "sp_000000.pt"
        torch.save(trainer.policy.state_dict(), snap)
        mgr._add_to_pool(snap)
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


def test_ratio_c_identity_arena_pinned():
    from cs2rl.env.map import make_arena_duel_map
    assert _ratio_c_after_rollout(1, make_arena_duel_map()) == [1.0, 0.0]


def test_env_trainer_pin_agreement_raises(simple_map):
    from cs2rl.train.envs import assert_pin_pitch_agreement
    from tests._helpers.trainer_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=4, map_data=simple_map, pin_pitch=1)
    try:
        assert_pin_pitch_agreement(trainer.vecenv, trainer.policy)
        trainer.policy.aim_dim_mask.fill_(1.0)
        with pytest.raises(RuntimeError):
            assert_pin_pitch_agreement(trainer.vecenv, trainer.policy)
    finally:
        cleanup()


def test_pin_agreement_rejects_non_c_env():
    from cs2rl.train.envs import assert_pin_pitch_agreement

    class _Pol:
        aim_dim_mask = torch.tensor([1.0, 1.0])

    with pytest.raises(RuntimeError):
        assert_pin_pitch_agreement(object(), _Pol())


# ── Fix round 1: Critical #1 — None must resolve the LOADED map, not the sentinel ──


def _flat_copy_of(md):
    """A synthetic FLAT MapData: simple_map with centroids_z zeroed."""
    import dataclasses
    return dataclasses.replace(md, centroids_z=np.zeros_like(md.centroids_z))


def test_pin_pitch_for_map_geometry(simple_map):
    """simple_map spans z 0..128 ⇒ 0; the same map flattened ⇒ 1. Pins that the
    decision is the z-span of the map passed in, nothing else."""
    from cs2rl.train.envs import pin_pitch_for_map
    assert float(simple_map.centroids_z.max() - simple_map.centroids_z.min()) == 128.0
    assert pin_pitch_for_map(simple_map) == 0
    assert pin_pitch_for_map(_flat_copy_of(simple_map)) == 1


def test_pin_pitch_for_map_none_loads_dust2():
    """None ⇒ the cs2 nav map make_env(map_data=None) loads (same _ENV_CACHE
    key). The expected value is computed from THAT MapData's centroids_z, so
    this test keeps holding when dust2 verticality lands (today make_cs2_map
    zero-fills z ⇒ 1, matching plan §R0-E.2 "true for dust2")."""
    from cs2rl.env import nav
    from cs2rl.env.c.cs2_env import _ENV_CACHE
    from cs2rl.env.map import make_cs2_map
    from cs2rl.train.envs import pin_pitch_for_map
    md = make_cs2_map(nav.NAV_PATH, nav.CACHE_PATH)
    expect = int(float(md.centroids_z.max() - md.centroids_z.min()) == 0.0)
    assert pin_pitch_for_map(None) == expect
    assert expect == 1                                                 # documents today's in-sim dust2 (env/map.py zero-fill)
    assert (nav.NAV_PATH, nav.CACHE_PATH) in _ENV_CACHE                # cached for make_env


def test_pin_pitch_build_vis_false_never_builds_vis_nor_caches(monkeypatch):
    """gh#251: the --dump-config path resolves dust2's pin_pitch WITHOUT the
    vis-matrix build (whose cold-cache ProcessPoolExecutor forked cpu_count()
    workers that a killed dump orphaned, ~900 MB each) and WITHOUT caching the
    vis-less MapData (a later make_env would get vis_matrix=None).

    Caches are emptied first so the cold path is exercised even when an earlier
    test warmed them; build_vis_matrix raising proves it is never called."""
    import argparse

    from cs2rl.env import map as map_mod
    from cs2rl.env import nav
    from cs2rl.env.c import cs2_env
    from cs2rl.train.envs import pin_pitch_for_map, resolve_pin_pitch

    def _boom(self):
        raise AssertionError("build_vis_matrix called on the build_vis=False path")

    monkeypatch.setattr(nav.NavGraph, "build_vis_matrix", _boom)
    monkeypatch.setattr(map_mod, "_CS2_MAP_CACHE", {})
    monkeypatch.setattr(cs2_env, "_ENV_CACHE", {})
    assert pin_pitch_for_map(None, build_vis=False) == 1               # same zero-fill answer as the full load
    a = argparse.Namespace(map_data=None, pin_pitch=None)
    assert resolve_pin_pitch(a, build_vis=False) == 1 and a.pin_pitch == 1
    assert map_mod._CS2_MAP_CACHE == {} and cs2_env._ENV_CACHE == {}   # vis-less MapData never cached
    with pytest.raises(AssertionError, match="build_vis_matrix called"):
        pin_pitch_for_map(None)                                        # positive control: default still builds vis


def test_resolve_pin_pitch_dust2_and_simple(simple_map, capsys):
    """The train() path with the CLI's `--dust2` args (map_data=None,
    pin_pitch=None): resolves from the loaded map; an explicit value equal to
    it is accepted, the other one refused. simple_map: None ⇒ 0, 1 refused."""
    import argparse

    from cs2rl.train.envs import pin_pitch_for_map, resolve_pin_pitch
    dust2 = pin_pitch_for_map(None)

    a = argparse.Namespace(map_data=None, pin_pitch=None)
    assert resolve_pin_pitch(a) == dust2 and a.pin_pitch == dust2
    assert f"pin_pitch={dust2}" in capsys.readouterr().out
    a = argparse.Namespace(map_data=None, pin_pitch=dust2)
    assert resolve_pin_pitch(a) == dust2
    with pytest.raises(ValueError):
        resolve_pin_pitch(argparse.Namespace(map_data=None, pin_pitch=1 - dust2))

    a = argparse.Namespace(map_data=simple_map, pin_pitch=None)
    assert resolve_pin_pitch(a) == 0 and a.pin_pitch == 0
    assert resolve_pin_pitch(argparse.Namespace(map_data=simple_map, pin_pitch=0)) == 0
    with pytest.raises(ValueError):
        resolve_pin_pitch(argparse.Namespace(map_data=simple_map, pin_pitch=1))


def test_log_aim_log_std_skips_pitch_when_pinned(simple_map):
    """Minor: a pinned policy must not report a σ for the dead pitch dim."""
    from cs2rl.train.metrics import log_aim_log_std
    from tests._helpers.trainer_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=4, map_data=simple_map, pin_pitch=1)
    try:
        logs = {}
        log_aim_log_std(trainer.policy, logs)
        assert "policy/aim_log_std_yaw" in logs
        assert not any(k.startswith("policy/aim_log_std_pitch") for k in logs)
        trainer.policy.aim_dim_mask.fill_(1.0)
        logs = {}
        log_aim_log_std(trainer.policy, logs)
        assert "policy/aim_log_std_pitch" in logs
    finally:
        cleanup()
