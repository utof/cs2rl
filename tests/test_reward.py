# tests/test_reward.py
import numpy as np
import pytest

import sim as sim_module
from c_env.wrapper import make_env
from train import TeamSpiritCallback


def test_team_spirit_module_default_zero():
    """sim._TEAM_SPIRIT must be 0.0 at module load so existing training is unchanged."""
    assert sim_module._TEAM_SPIRIT == 0.0


def test_pbrs_rewards_are_finite():
    """PBRS must not produce NaN or inf over a full episode."""
    env = make_env()
    env.reset()
    rng = np.random.default_rng(7)
    for step_n in range(500):
        actions = rng.integers([9, 2, 2, 2], size=(10, 4)).astype(np.int64)
        _, rewards, terms, _, _ = env.step(actions)
        for i, r in enumerate(rewards):
            assert np.isfinite(r), f"Non-finite reward at step {step_n} agent {i}: {r}"
        if terms.all():
            break
    env.close()


def test_pbrs_shaping_positive_on_kill():
    """Killing an enemy produces a positive reward for the shooter."""
    import math

    env = make_env(auto_reset=False)
    env.reset()
    id2idx = {int(aid): i for i, aid in enumerate(env.map_data.area_ids)}

    # Find two visible areas within shooting range
    nav = env.nav_graph
    pair = None
    for i, area_i in enumerate(nav.area_ids[:400]):
        for area_j in nav.area_ids[i + 1 : i + 200]:
            if not env.map_data.vis_matrix[id2idx[area_i], id2idx[area_j]]:
                continue
            dx = nav.centroids[area_j][0] - nav.centroids[area_i][0]
            dy = nav.centroids[area_j][1] - nav.centroids[area_i][1]
            if 50 < float((dx * dx + dy * dy) ** 0.5) < 1500:
                pair = (area_i, area_j)
                break
        if pair is not None:
            break
    assert pair is not None, "no visible test pair found"

    area_t, area_ct = pair
    for i in range(10):
        env._c_env.game.agents[i].alive = 0
        env._c_env.game.agents[i].hp = 0

    t_c = nav.centroids[area_t]
    ct_c = nav.centroids[area_ct]
    t = env._c_env.game.agents[0]
    ct = env._c_env.game.agents[5]

    t.alive = 1
    t.hp = 100
    t.area_idx = id2idx[area_t]
    t.x, t.y, t.z = float(t_c[0]), float(t_c[1]), 0.0

    ct.alive = 1
    ct.hp = 100
    ct.area_idx = id2idx[area_ct]
    ct.x, ct.y, ct.z = float(ct_c[0]), float(ct_c[1]), 0.0

    t.facing = math.atan2(ct.y - t.y, ct.x - t.x)
    ct.facing = math.atan2(t.y - ct.y, t.x - ct.x)

    actions = np.zeros((10, 4), dtype=np.int64)
    actions[0, 1] = 1  # t0 shoots
    _, rewards, _, _, _ = env.step(actions)

    assert rewards[0] > 0, f"Killing CT gives non-positive reward: {rewards[0]:.4f}"
    env.close()


def test_team_spirit_zero_unchanged():
    """At team_spirit=0.0, two identical-seed envs produce identical rewards."""
    env1 = make_env(seed=42)
    env1.reset()
    actions = np.zeros((10, 4), dtype=np.int64)
    _, rewards1, _, _, _ = env1.step(actions)
    env1.close()

    env2 = make_env(seed=42)
    env2.reset()
    _, rewards2, _, _, _ = env2.step(actions)
    env2.close()

    np.testing.assert_allclose(
        rewards1,
        rewards2,
        atol=1e-6,
        err_msg="Reward mismatch at team_spirit=0 between identical-seed envs",
    )


def test_team_spirit_one_equalizes_alive_team():
    """At team_spirit=1.0, all alive agents on the same team get equal rewards."""
    env = make_env(seed=11, team_spirit=1.0)
    env.reset()
    actions = np.zeros((10, 4), dtype=np.int64)
    _, rewards, _, _, _ = env.step(actions)
    env.close()

    t_alive = [i for i in range(5) if env._c_env.game.agents[i].alive]
    ct_alive = [i for i in range(5, 10) if env._c_env.game.agents[i].alive]

    if len(t_alive) > 1:
        t_rewards = [float(rewards[i]) for i in t_alive]
        assert all(abs(r - t_rewards[0]) < 1e-5 for r in t_rewards), (
            f"T alive rewards not equal at team_spirit=1: {t_rewards}"
        )

    if len(ct_alive) > 1:
        ct_rewards = [float(rewards[i]) for i in ct_alive]
        assert all(abs(r - ct_rewards[0]) < 1e-5 for r in ct_rewards), (
            f"CT alive rewards not equal at team_spirit=1: {ct_rewards}"
        )


def test_team_spirit_callback_anneals():
    """TeamSpiritCallback linearly anneals sim._TEAM_SPIRIT from 0→1."""
    sim_module._TEAM_SPIRIT = 0.0
    cb = TeamSpiritCallback(anneal_steps=1_000_000)

    cb.num_timesteps = 0
    cb._on_step()
    assert sim_module._TEAM_SPIRIT == pytest.approx(0.0, abs=1e-6)

    cb.num_timesteps = 500_000
    cb._on_step()
    assert sim_module._TEAM_SPIRIT == pytest.approx(0.5, abs=1e-3)

    cb.num_timesteps = 1_000_000
    cb._on_step()
    assert sim_module._TEAM_SPIRIT == pytest.approx(1.0, abs=1e-3)

    cb.num_timesteps = 2_000_000
    cb._on_step()
    assert sim_module._TEAM_SPIRIT == pytest.approx(1.0, abs=1e-3), (
        "team_spirit must not exceed 1.0 after anneal_steps"
    )

    sim_module._TEAM_SPIRIT = 0.0  # restore
