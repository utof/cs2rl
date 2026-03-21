# tests/test_reward.py
import numpy as np
import pytest

from c_env.wrapper import make_env
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


# ── Phase 4.4 reward unit tests ───────────────────────────────────────────


def test_idle_penalty():
    """Idle action (move=0) for a live agent should incur -0.0005 penalty."""
    env = make_env(seed=0, auto_reset=False)
    env.reset()

    # All agents idle (move action = 0)
    actions = np.zeros((10, 4), dtype=np.int64)
    _, rewards, _, _, _ = env.step(actions)

    # Every alive agent should have received the -0.0005 idle penalty.
    # The reward may also contain PBRS terms, so we only check the sign / range.
    for i in range(10):
        if env._c_env.game.agents[i].alive:
            assert rewards[i] <= 0, f"Agent {i} idled but got non-negative reward: {rewards[i]:.6f}"
            # The idle penalty alone is -0.0005; PBRS shaping should be small.
            # Verify the penalty is at most -0.0005 (PBRS can add to it).
            assert rewards[i] <= -0.0004, (
                f"Agent {i} idle penalty smaller than expected: {rewards[i]:.6f}"
            )
    env.close()


def test_win_terminal_reward():
    """Winning team receives +1.0 terminal reward; losing team -1.0."""
    env = make_env(seed=0, auto_reset=False)
    env.reset()

    # Kill all CT agents — T team wins
    for i in range(5, 10):
        env._c_env.game.agents[i].alive = 0
        env._c_env.game.agents[i].hp = 0

    actions = np.zeros((10, 4), dtype=np.int64)
    _, rewards, terms, _, _ = env.step(actions)

    # T agents (indices 0-4) are alive and should receive +1 terminal bonus.
    # The total reward may include small PBRS shaping terms so allow down to 0.9.
    for i in range(5):
        if env._c_env.game.agents[i].alive or terms[i]:
            assert rewards[i] >= 0.9, (
                f"T agent {i} should get +1 win bonus (>=0.9 after PBRS), got {rewards[i]:.4f}"
            )
    env.close()


def test_bomb_entry_bonus():
    """T bomb carrier entering bombsite for the first time gets +0.3 bonus."""
    env = make_env(seed=0, auto_reset=False)
    env.reset()
    nav_graph = env.nav_graph
    map_data = env.map_data

    # Find first bombsite area index
    site_idx = None
    site_centroid = None
    for idx, is_site in enumerate(map_data.bombsite_by_idx):
        if is_site:
            site_idx = idx
            site_centroid = nav_graph.centroids[map_data.area_ids[idx]]
            break
    assert site_idx is not None, "No bombsite found in map"

    # Assign bomb to agent 0 and teleport them to bombsite
    for i in range(10):
        env._c_env.game.agents[i].has_bomb = 0
    bomber = env._c_env.game.agents[0]
    bomber.has_bomb = 1
    env._c_env.game.bomb_carrier_id = 0

    # Ensure bombsite_entered flag is clear for bomber
    env._c_env.game.bombsite_entered[0] = 0

    # Teleport bomber to bombsite centroid
    bomber.area_idx = site_idx
    bomber.x = float(site_centroid[0])
    bomber.y = float(site_centroid[1])
    bomber.z = 0.0

    # Step with use=1 — the C env checks use action to trigger entry bonus
    actions = np.zeros((10, 4), dtype=np.int64)
    actions[0, 2] = 1  # use action required to trigger bombsite_entered check
    _, rewards, _, _, _ = env.step(actions)

    # Reward for agent 0 must include the +0.3 bombsite entry bonus
    assert rewards[0] >= 0.29, f"Bombsite entry bonus not found: agent 0 reward = {rewards[0]:.4f}"
    env.close()


def test_plant_progress_reward():
    """Each tick of active bomb planting yields +0.05 to the planting agent."""
    env = make_env(seed=0, auto_reset=False)
    env.reset()
    nav_graph = env.nav_graph
    map_data = env.map_data

    # Find first bombsite area
    site_idx = None
    site_centroid = None
    for idx, is_site in enumerate(map_data.bombsite_by_idx):
        if is_site:
            site_idx = idx
            site_centroid = nav_graph.centroids[map_data.area_ids[idx]]
            break
    assert site_idx is not None, "No bombsite found in map"

    # Set up bomber at bombsite — mark entry as already done so no entry bonus
    for i in range(10):
        env._c_env.game.agents[i].has_bomb = 0
    bomber_idx = 0
    bomber = env._c_env.game.agents[bomber_idx]
    bomber.has_bomb = 1
    env._c_env.game.bomb_carrier_id = bomber_idx
    env._c_env.game.bombsite_entered[bomber_idx] = 1  # suppress entry bonus

    bomber.area_idx = site_idx
    bomber.x = float(site_centroid[0])
    bomber.y = float(site_centroid[1])
    bomber.z = 0.0

    # Start planting: set bomb_being_planted_by to bomber_idx and advance ticks
    env._c_env.game.bomb_being_planted_by = bomber_idx
    env._c_env.game.bomb_plant_ticks = 1  # already started (not tick 0)

    # use=1 to continue planting
    actions = np.zeros((10, 4), dtype=np.int64)
    actions[bomber_idx, 2] = 1
    _, rewards, _, _, _ = env.step(actions)

    # The per-tick plant progress reward is +0.05
    assert rewards[bomber_idx] >= 0.04, (
        f"Plant progress reward missing: agent {bomber_idx} reward = {rewards[bomber_idx]:.4f}"
    )
