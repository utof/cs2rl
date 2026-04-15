# tests/test_reward.py
import math

import numpy as np

from c_env.cs2_env import make_env
from nav import ACTION_DIM

_ACTION_HEAD_SIZES = [9, 16, 2, 2, 3, 2, 2, 2]


def _facing_to_aim(angle):
    normalized = angle % (2 * math.pi)
    if normalized < 0:
        normalized += 2 * math.pi
    return int(normalized * 16 / (2 * math.pi)) % 16


def test_pbrs_rewards_are_finite():
    """PBRS must not produce NaN or inf over a full episode."""
    env = make_env()
    env.reset()
    rng = np.random.default_rng(7)
    for step_n in range(500):
        actions = rng.integers(_ACTION_HEAD_SIZES, size=(10, ACTION_DIM)).astype(np.int64)
        _, rewards, terms, _, _ = env.step(actions)
        for i, r in enumerate(rewards):
            assert np.isfinite(r), f"Non-finite reward at step {step_n} agent {i}: {r}"
        if terms.all():
            break
    env.close()


def test_pbrs_shaping_positive_on_kill():
    """Killing an enemy produces a positive reward for the shooter."""
    env = make_env(auto_reset=False)
    env.reset()
    id2idx = {int(aid): i for i, aid in enumerate(env.map_data.area_ids)}

    # Find two visible areas within shooting range
    nav = env.nav_graph
    pair = None
    for i, area_i in enumerate(nav.area_ids[:400]):
        for area_j in nav.area_ids[i + 1:i + 200]:
            if not env.map_data.vis_matrix[id2idx[area_i], id2idx[area_j]]:
                continue
            dx = nav.centroids[area_j][0] - nav.centroids[area_i][0]
            dy = nav.centroids[area_j][1] - nav.centroids[area_i][1]
            if 50 < float((dx * dx + dy * dy)**0.5) < 1500:
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
    ct.hp = 1                          # low HP so any hit kills
    ct.armor = 0
    ct.area_idx = id2idx[area_ct]
    ct.x, ct.y, ct.z = float(ct_c[0]), float(ct_c[1]), 0.0

    t_facing = math.atan2(ct.y - t.y, ct.x - t.x)

    actions = np.zeros((10, ACTION_DIM), dtype=np.int64)
    actions[0, 1] = _facing_to_aim(t_facing)           # aim at CT (head index 1)
    actions[0, 2] = 1                                  # t0 shoots (shoot is head index 2)
    _, rewards, _, _, _ = env.step(actions)

    assert rewards[0] > 0, f"Killing CT gives non-positive reward: {rewards[0]:.4f}"
    env.close()


def test_team_spirit_zero_unchanged():
    """At team_spirit=0.0, two identical-seed envs produce identical rewards."""
    env1 = make_env(seed=42)
    env1.reset()
    actions = np.zeros((10, ACTION_DIM), dtype=np.int64)
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
    actions = np.zeros((10, ACTION_DIM), dtype=np.int64)
    _, rewards, _, _, _ = env.step(actions)
    env.close()

    t_alive = [i for i in range(5) if env._c_env.game.agents[i].alive]
    ct_alive = [i for i in range(5, 10) if env._c_env.game.agents[i].alive]

    if len(t_alive) > 1:
        t_rewards = [float(rewards[i]) for i in t_alive]
        assert all(abs(r - t_rewards[0]) < 1e-5
                   for r in t_rewards), (f"T alive rewards not equal at team_spirit=1: {t_rewards}")

    if len(ct_alive) > 1:
        ct_rewards = [float(rewards[i]) for i in ct_alive]
        assert all(
            abs(r - ct_rewards[0]) < 1e-5
            for r in ct_rewards), (f"CT alive rewards not equal at team_spirit=1: {ct_rewards}")


# ── Phase 4.4 reward unit tests ───────────────────────────────────────────


def test_idle_penalty():
    """Idle action (move=0) for a live agent should incur -0.0005 penalty."""
    env = make_env(seed=0, auto_reset=False)
    env.reset()

    # All agents idle (move action = 0)
    actions = np.zeros((10, ACTION_DIM), dtype=np.int64)
    _, rewards, _, _, _ = env.step(actions)

    # Every alive agent should have received the -0.0005 idle penalty.
    # The reward may also contain PBRS terms, so we only check the sign / range.
    for i in range(10):
        if env._c_env.game.agents[i].alive:
            assert rewards[i] <= 0, f"Agent {i} idled but got non-negative reward: {rewards[i]:.6f}"
            # The idle penalty alone is -0.0005; PBRS shaping should be small.
            # Verify the penalty is at most -0.0005 (PBRS can add to it).
            assert rewards[i] <= -0.0004, (
                f"Agent {i} idle penalty smaller than expected: {rewards[i]:.6f}")
    env.close()


def test_win_terminal_reward():
    """Winning team receives +1.0 terminal reward; losing team -1.0."""
    env = make_env(seed=0, auto_reset=False)
    env.reset()

    # Kill all CT agents — T team wins
    for i in range(5, 10):
        env._c_env.game.agents[i].alive = 0
        env._c_env.game.agents[i].hp = 0

    actions = np.zeros((10, ACTION_DIM), dtype=np.int64)
    _, rewards, terms, _, _ = env.step(actions)

    # T agents (indices 0-4) are alive and should receive +1 terminal bonus.
    # The total reward may include small PBRS shaping terms so allow down to 0.9.
    for i in range(5):
        if env._c_env.game.agents[i].alive or terms[i]:
            assert rewards[i] >= 0.9, (
                f"T agent {i} should get +1 win bonus (>=0.9 after PBRS), got {rewards[i]:.4f}")
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
    actions = np.zeros((10, ACTION_DIM), dtype=np.int64)
    actions[0, 5] = 1                  # use action (head index 5) required to trigger bombsite_entered check
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
    env._c_env.game.bombsite_entered[bomber_idx] = 1   # suppress entry bonus

    bomber.area_idx = site_idx
    bomber.x = float(site_centroid[0])
    bomber.y = float(site_centroid[1])
    bomber.z = 0.0

    # Start planting: set bomb_being_planted_by to bomber_idx and advance ticks
    env._c_env.game.bomb_being_planted_by = bomber_idx
    env._c_env.game.bomb_plant_ticks = 1               # already started (not tick 0)

    # use=1 to continue planting (head index 5)
    actions = np.zeros((10, ACTION_DIM), dtype=np.int64)
    actions[bomber_idx, 5] = 1
    _, rewards, _, _, _ = env.step(actions)

    # The per-tick plant progress reward is +0.05
    assert rewards[bomber_idx] >= 0.04, (
        f"Plant progress reward missing: agent {bomber_idx} reward = {rewards[bomber_idx]:.4f}")


# ── Phase 5 reward-externalization tests ──────────────────────────────────────


def test_kill_reward_weight_is_configurable():
    """make_env(reward_kill=X) scales the kill reward; zero-out all other weights
    so the kill reward is the only non-zero contribution."""
    env = make_env(
        reward_kill=0.9,
        reward_death=0.0,
        reward_win=0.0,
        reward_bombsite_entry=0.0,
        reward_plant_bonus=0.0,
        reward_plant_base=0.0,
        reward_plant_progress_scale=0.0,
        reward_plant_interrupted=0.0,
        reward_defuse=0.0,
        reward_shot_penalty=0.0,
        reward_ct_survival=0.0,
        reward_inaction=0.0,
        pbrs_alive_weight=0.0,
        pbrs_hp_weight=0.0,
        pbrs_site_weight=0.0,
        pbrs_bomb_progress_weight=0.0,
        pbrs_nav_weight_t=0.0,
        pbrs_nav_weight_ct=0.0,
        auto_reset=False,
    )
    env.reset()
    id2idx = {int(aid): i for i, aid in enumerate(env.map_data.area_ids)}
    nav = env.nav_graph

    pair = None
    for i, area_i in enumerate(nav.area_ids[:400]):
        for area_j in nav.area_ids[i + 1:i + 200]:
            if not env.map_data.vis_matrix[id2idx[area_i], id2idx[area_j]]:
                continue
            dx = nav.centroids[area_j][0] - nav.centroids[area_i][0]
            dy = nav.centroids[area_j][1] - nav.centroids[area_i][1]
            if 50 < float((dx * dx + dy * dy)**0.5) < 1500:
                pair = (area_i, area_j)
                break
        if pair is not None:
            break
    assert pair is not None

    area_t, area_ct = pair
    for i in range(10):
        env._c_env.game.agents[i].alive = 0
        env._c_env.game.agents[i].hp = 0

    t = env._c_env.game.agents[0]
    ct = env._c_env.game.agents[5]
    tc = nav.centroids[area_t]
    ctc = nav.centroids[area_ct]

    t.alive = 1
    t.hp = 100
    t.armor = 0
    t.area_idx = id2idx[area_t]
    t.x = float(tc[0])
    t.y = float(tc[1])
    t.z = 0.0
    ct.alive = 1
    ct.hp = 1
    ct.armor = 0
    ct.area_idx = id2idx[area_ct]
    ct.x = float(ctc[0])
    ct.y = float(ctc[1])
    ct.z = 0.0

    t_facing = math.atan2(ct.y - t.y, ct.x - t.x)
    actions = np.zeros((10, ACTION_DIM), dtype=np.int64)
    actions[0, 1] = _facing_to_aim(t_facing)
    actions[0, 2] = 1
    _, rewards, _, _, _ = env.step(actions)

    # With all weights zeroed except reward_kill=0.9, killer reward must be ≈0.9
    assert abs(rewards[0] - 0.9) < 0.05, f"Expected kill reward ≈0.9, got {rewards[0]:.4f}"
    env.close()


def test_reward_components_logged_in_terminal_info():
    """_build_terminal_info must include the 8 reward component keys."""
    EXPECTED_KEYS = {
        "reward_win",
        "reward_kills",
        "reward_deaths",
        "reward_bomb",
        "reward_pbrs",
        "reward_shots",
        "reward_survival",
        "reward_inaction",
    }
    env = make_env(seed=0, auto_reset=False)
    env.reset()

    # Run until episode ends
    rng = np.random.default_rng(1)
    info = {}
    for _ in range(1000):
        actions = rng.integers([9, 16, 2, 2, 3, 2, 2, 2], size=(10, ACTION_DIM)).astype(np.int64)
        _, _, terms, _, infos = env.step(actions)
        for d in infos:
            if d:
                info = d
                break
        if terms.all():
            break

    assert info, "Episode did not terminate within 1000 steps; no terminal info captured"
    missing = EXPECTED_KEYS - set(info.keys())
    assert not missing, f"Missing reward component keys in terminal info: {missing}"
    env.close()
