# tests/test_reward.py
import math

import numpy as np
import pytest

from c_env.cs2_env import make_env
from nav import ACTION_DIM
from train import ACTION_HEAD_SIZES


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
        actions = rng.integers(ACTION_HEAD_SIZES, size=(10, ACTION_DIM)).astype(np.int64)
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
    # use action (head 5) triggers bombsite_entered check
    actions[0, 5] = 1
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
    so the kill reward is the only non-zero contribution.

    Batch 1 (RL overhaul): must also zero the per-mechanism win-reward fields,
    otherwise killing the last enemy triggers a T-elimination reward on top of
    the kill reward and corrupts the assertion. These fields did not exist before
    Task 3 so they were not in the original zero-out list.
    """
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
        reward_win_t_detonation=0.0,
        reward_win_t_elimination=0.0,
        reward_win_ct_defuse=0.0,
        reward_win_ct_timeout=0.0,
        reward_win_ct_elimination=0.0,
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
        actions = rng.integers(list(ACTION_HEAD_SIZES), size=(10, ACTION_DIM)).astype(np.int64)
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


# ── Batch 1 Task 3: differential win-reward magnitude tests ──────────────────
#
# Strategy: direct-stimulus white-box approach.
# We set game state (winner, bomb_planted, bomb_ticks_left, round_over, alive
# agents) directly via ctypes, then call env.step() with all-zero actions.
# All other reward weights (kill, death, pbrs, survival, shot, inaction) are
# zeroed so step_stats.reward_win reflects only the win-magnitude path.
#
# We sum step_stats.reward_win over alive winners to get per-agent magnitude,
# then assert it matches the expected differential value.
#
# Direct-stimulus is preferred here because:
# 1. compute_rewards is not exposed via ctypes as a standalone callable.
# 2. Driving a full round to a specific outcome (detonation/defuse/etc.) would
#    require scripting agent actions and is brittle.
# 3. The white-box approach gives a clean FAIL before the C change and a crisp
#    PASS after — matching the TDD contract.
#
# Isolation: setting round_over=1 and winner=0/1 before step() causes
# compute_rewards to fire the round-over block exactly once per step call.


def _make_zeroed_env():
    """Return a make_env with all shaping weights zeroed; only win-reward matters.

    Pitfall: reward_win (the old symmetric weight) must also be zero so the
    existing code path does not pollute the result before Task 3 replaces it.
    The new per-mechanism fields default to the desired magnitudes.
    """
    return make_env(
        seed=0,
        auto_reset=False,
        reward_win=0.0,                                # silence old symmetric path (pre-Task-3)
        reward_kill=0.0,
        reward_death=0.0,
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
                                                       # New per-mechanism defaults (Task 3):
        reward_win_t_detonation=5.0,
        reward_win_t_elimination=3.0,
        reward_win_ct_defuse=5.0,
        reward_win_ct_timeout=4.0,
        reward_win_ct_elimination=3.0,
    )


def _setup_round_end(env, winner, bomb_planted, bomb_ticks_left, round_ticks_left, alive_teams):
    """Configure game state for a deterministic round-end scenario.

    Sets one agent alive per team (agent 0 = T, agent 5 = CT) and marks the
    round over with the specified winner/bomb conditions. All other agents dead.

    Use this helper for scenarios where round_over is already set before step()
    (detonation, elimination, timeout, and post-plant elimination edge-cases).
    For ct_defuse, see the standalone `test_natural_defuse` — pre-setting
    round_over blocks the defuse branch in process_bomb (cs2_bomb.h:27), so
    bomb_just_defused would never fire and the spec-compliant classifier in
    compute_rewards (which requires bomb_just_defused=1 for defuse) would
    misclassify as elimination.

    Args:
        winner:          0=T wins, 1=CT wins, -1=timeout
        bomb_planted:    1 if bomb is planted
        bomb_ticks_left: remaining bomb timer (<=0 means detonated)
        round_ticks_left: remaining round timer
        alive_teams:     set of teams that have survivors ({0}, {1}, or {0,1})
    """
    g = env._c_env.game
    # Kill all agents first
    for i in range(10):
        g.agents[i].alive = 0
        g.agents[i].hp = 0
    # Revive one agent per alive team
    if 0 in alive_teams:
        g.agents[0].alive = 1
        g.agents[0].hp = 100
        g.agents[0].team = 0
    if 1 in alive_teams:
        g.agents[5].alive = 1
        g.agents[5].hp = 100
        g.agents[5].team = 1
    # Set round-end state
    g.winner = winner
    g.bomb_planted = bomb_planted
    g.bomb_ticks_left = bomb_ticks_left
    g.round_ticks_left = round_ticks_left
    g.round_over = 1


@pytest.mark.parametrize(
    "scenario,winner,bomb_planted,bomb_ticks_left,round_ticks_left,"
    "alive_teams,expected_mag", [
        ("t_detonation", 0, 1, -1, 100, {0}, 5.0),
        ("t_elimination", 0, 0, 0, 50, {0}, 3.0),
        ("ct_elimination", 1, 0, 0, 50, {1}, 3.0),
        ("ct_elimination_postplant", 1, 1, 50, 100, {1}, 3.0),
        ("ct_timeout", -1, 0, 0, 0, {1}, 4.0),
    ])
def test_differential_win_magnitudes(scenario, winner, bomb_planted, bomb_ticks_left,
                                     round_ticks_left, alive_teams, expected_mag):
    """Each round-end outcome must yield exactly its specified win magnitude.

    Direct-stimulus: we set game state via ctypes, call step(), and read
    step_stats.reward_win. With all other weights zeroed, the only contribution
    to reward_win is the per-mechanism win-reward block in compute_rewards.

    All five scenarios pre-set round_over=1 via _setup_round_end; step() then
    runs compute_rewards exactly once on the terminal state. This faithfully
    represents live-play states where round_over is decided upstream
    (detonation, elimination, timeout, post-plant elimination).

    Classification logic (mirrors compute_rewards round-over block):
      T win   (winner == 0): detonation if bomb_planted && ticks<=0, else elimination
      CT win  (winner == 1): defuse if bomb_just_defused, else elimination
      Timeout (winner == -1): timed_out flag set; CT gets ct_timeout reward

    Scenarios covered:
      t_detonation — bomb_planted, ticks<=0 → exploded (5.0).
      t_elimination — no plant, T killed all CT (3.0).
      ct_elimination (preplant) — !bomb_planted, T dead before plant.
        bomb_just_defused=0 → elimination branch → 3.0.
      ct_elimination_postplant — LIVE-PLAY EDGE. CT killed last T with bomb
        planted but not defused. cs2_env.h:146-152 sets winner=1/round_over=1
        when !t_alive, then process_bomb's defuse branch is skipped
        (round_over guard), so bomb_just_defused stays 0. Must classify as
        elimination (3.0), NOT defuse (5.0).
      ct_timeout — round timer expired, no plant → CT tactical win (4.0). C
        code sets winner=-1 and timed_out=1; we reward CT survivors.

    ct_defuse is excluded from this table because it requires a structurally
    different setup (live T agents, round_over=0, process_bomb driving defuse
    naturally). See test_natural_defuse for that case.
    """
    import numpy as np
    env = _make_zeroed_env()
    env.reset()

    _setup_round_end(env, winner, bomb_planted, bomb_ticks_left, round_ticks_left, alive_teams)
    actions = np.zeros((10, 8), dtype=np.int64)

    _, _, _, _, _ = env.step(actions)

    ss = env._c_env.step_stats
    # reward_win accumulates ±mag for every alive agent on winning/losing side.
    # With one alive winner and no losers alive, reward_win == +expected_mag.
    actual = float(ss.reward_win)
    assert abs(actual - expected_mag) < 0.01, (
        f"Scenario '{scenario}': expected reward_win≈{expected_mag}, got {actual:.4f}")
    env.close()


def test_natural_defuse():
    """CT-defuse outcome, driven end-to-end through process_bomb.

    Unlike the parametrized magnitude tests (which pre-set round_over=1 and
    exercise ONLY the classification branch of compute_rewards), this test
    drives the full bomb-defuse code path:

      process_combat (no kills) → elimination check (t_alive > 0, skipped)
      → process_bomb defuse branch fires → bomb_just_defused=1, round_over=1,
      winner=1 → compute_rewards sees ct_won && bomb_just_defused → defuse.

    Why a dedicated test (not a parametrize row):
      - ct_defuse is the only scenario where round_over is NOT pre-set; the
        test must arrange alive T agents so cs2_env.h:146's elimination check
        is skipped, then step once to let process_bomb complete the defuse.
        That requires structurally different setup from the other cases.
      - We assert multiple invariants (flag values, per-agent rewards for
        winner and loser) that wouldn't fit cleanly in a parametrize row.

    Pitfall avoided:
      An earlier attempt killed all T agents in the defuse setup. That made
      t_alive=0, triggering cs2_env.h:146's unconditional elimination path
      (round_over=1, winner=1) BEFORE process_bomb could fire. The defuse
      branch then got skipped (cs2_bomb.h:27 guard), bomb_just_defused stayed
      0, and the scenario mis-classified as elimination (3.0). We keep at
      least one T agent alive, placed at a non-bomb area so process_combat
      does nothing (actions are zeroed → no shoot), to let process_bomb reach
      the defuse gate.
    """
    import numpy as np

    env = _make_zeroed_env()
    env.reset()

    g = env._c_env.game
    sd = env._c_env.sd.contents

    # Kill every agent first to zero the slate.
    for i in range(10):
        g.agents[i].alive = 0
        g.agents[i].hp = 0

    # Alive CT defuser at the bomb area (agent 5).
    ct = g.agents[5]
    ct.alive = 1
    ct.hp = 100
    ct.team = 1
    ct.has_kit = 0                     # use no-kit defuse_time
    ct.area_idx = 0                    # arbitrary valid area

    # Alive T at a different area so process_combat doesn't kill them
    # (zeroed actions → no shoot → no combat resolution). Having a T alive
    # is REQUIRED to avoid the elimination check in cs2_env.h:146 firing
    # before process_bomb runs.
    t = g.agents[0]
    t.alive = 1
    t.hp = 100
    t.team = 0
    t.area_idx = 1                     # anywhere != ct.area_idx

    # Bomb state: planted, live, co-located with the CT defuser, round open.
    g.bomb_planted = 1
    g.bomb_area_idx = ct.area_idx
    g.bomb_ticks_left = 50             # plenty of bomb-timer headroom
    g.round_ticks_left = 100           # round timer well above zero
    g.round_over = 0                   # CRITICAL: leave the round open
    g.winner = -1                      # ongoing

    # Short-circuit the defuse timer so ONE step completes the defuse.
    defuse_time = int(sd.bomb_defuse_time)
    g.bomb_being_defused_by = 5
    g.bomb_defuse_ticks = defuse_time - 1

    # HEAD_USE (index 5) = 1 keeps the CT defusing this tick.
    actions = np.zeros((10, ACTION_DIM), dtype=np.int64)
    actions[5, 5] = 1

    env.step(actions)

    # Pin the load-bearing precondition: the decoy T must survive this step
    # so process_combat's t_alive count stays positive and cs2_env.h:146's
    # elimination guard does NOT fire before process_bomb. If a future
    # process_combat change (passive chip damage, AoE, long-range hit) kills
    # this T mid-step, the defuse branch gets skipped silently and the
    # downstream reward assertions flip to the elimination magnitudes —
    # this assertion points the failure at the real cause instead.
    assert int(
        g.agents[0].alive) == 1, ("decoy T agent must survive the step for natural defuse to fire; "
                                  "if this trips, process_combat has grown side effects that break "
                                  "test_natural_defuse's setup assumption")

    ss = env._c_env.step_stats

    # Classification flags: defuse fired, detonation did not.
    assert int(ss.win_by_defuse) == 1, "win_by_defuse must be 1 after natural defuse"
    assert int(ss.win_by_detonation) == 0, "win_by_detonation must be 0 after defuse"

    # Per-agent rewards: CT gets +ct_defuse magnitude, T gets the penalty.
    # _make_zeroed_env sets reward_win_ct_defuse=5.0.
    ct_reward = float(env._c_env.rewards[5])
    t_reward = float(env._c_env.rewards[0])
    assert abs(ct_reward - 5.0) < 0.01, (f"CT defuser reward expected +5.0, got {ct_reward:.4f}")
    assert abs(t_reward - (-5.0)) < 0.01, (f"T loser penalty expected -5.0, got {t_reward:.4f}")

    # reward_win accumulator nets to zero (equal +mag and -mag with one of
    # each team alive) — documents the accounting, doesn't gate correctness.
    assert abs(float(ss.reward_win) - 0.0) < 0.01

    env.close()


def test_detonation_beats_elimination():
    """T win-by-detonation (5.0) must exceed T win-by-elimination (3.0).

    High-level smoke test; parametrized test above covers exact values.
    """
    import numpy as np

    def _run(bomb_planted, bomb_ticks_left):
        env = _make_zeroed_env()
        env.reset()
        _setup_round_end(env,
                         winner=0,
                         bomb_planted=bomb_planted,
                         bomb_ticks_left=bomb_ticks_left,
                         round_ticks_left=50,
                         alive_teams={0})
        actions = np.zeros((10, 8), dtype=np.int64)
        env.step(actions)
        val = float(env._c_env.step_stats.reward_win)
        env.close()
        return val

    r_deton = _run(bomb_planted=1, bomb_ticks_left=-1)                 # detonation
    r_elim = _run(bomb_planted=0, bomb_ticks_left=0)                   # elimination

    assert r_deton > r_elim, (
        f"Detonation reward ({r_deton}) should exceed elimination reward ({r_elim})")
    assert abs(r_deton - 5.0) < 0.01, f"Detonation expected 5.0, got {r_deton:.4f}"
    assert abs(r_elim - 3.0) < 0.01, f"Elimination expected 3.0, got {r_elim:.4f}"


# ── Batch 1 Task 2: win-type flag lifecycle tests ─────────────────────────────


def test_win_flags_cleared_on_round_reset():
    """White-box: stuffing win_by_detonation/win_by_defuse to 1, then calling
    env.reset(), must yield 0 for both fields.

    Rationale for white-box approach (vs. driving to a real round end):
    Task 3 (not yet implemented) is what sets these flags organically during
    play. Testing against a live round-end would give a trivial pass (flags
    never get set, so they stay 0) rather than a true FAIL→PASS cycle. By
    force-setting the fields and asserting they are cleared, we get a
    deterministic FAIL here (if the reset path were broken) and a PASS once
    we confirm the existing memset in clear_stats() covers the new fields.

    Implementation note: step_stats is a single StepStatsC struct on
    Dust2EnvC (not per-team array). clear_stats() calls
    memset(stats, 0, sizeof(StepStats)), which already zeroes every field
    including the Batch-1-added win_by_detonation / win_by_defuse. No C
    change is needed — this test confirms the existing bulk-zero is sufficient.
    """
    env = make_env(seed=0, auto_reset=False)
    env.reset()
    ss = env._c_env.step_stats         # ctypes StepStatsC — single struct, not array-of-two

    # Force-set both win-type flags to non-zero to simulate a previous round
    # that ended by detonation or defuse.
    ss.win_by_detonation = 1
    ss.win_by_defuse = 1

    # Round reset path: env.reset() calls clear_stats(&env->step_stats) which
    # does memset(..., 0, sizeof(StepStats)) — must zero the new fields.
    env.reset()

    assert int(ss.win_by_detonation) == 0, (
        f"win_by_detonation not cleared on round reset: {ss.win_by_detonation}")
    assert int(
        ss.win_by_defuse) == 0, (f"win_by_defuse not cleared on round reset: {ss.win_by_defuse}")
    env.close()


def test_step_stats_in_info_flag_default_off():
    """With include_step_stats_in_info=False (default), non-terminal step() ticks
    must return info == [] (byte-identical to pre-Task-6a behavior)."""
    import numpy as np
    env = make_env()                   # default flag off
    env.reset()
    actions = np.zeros((10, 8), dtype=np.int64)
    _, _, _, _, info = env.step(actions)
    assert info == [], f"flag-off should preserve empty info, got {info!r}"
    env.close()


def test_step_stats_in_info_flag_on_populates_view():
    """With the flag on, every step() returns info[0]['step_stats'] exposing the
    fields consumed by split_into_channels (reward_*, win_by_detonation/defuse).

    Uses make_env to construct the env; no trainer involved — this is a pure
    env-level unit test per Task 6a scope (no trainer harness yet; utof/cs2rl#8).
    """
    import numpy as np
    env = make_env(include_step_stats_in_info=True)
    env.reset()
    actions = np.zeros((10, 8), dtype=np.int64)
    _, _, _, _, info = env.step(actions)
    assert len(info) == 1, f"flag-on expects one info dict per step, got {len(info)}"
    assert "step_stats" in info[0]
    ss = info[0]["step_stats"]
    # Field surface required by split_into_channels (see train_helpers_batch1.py).
    # After one step() the PBRS delta may be non-zero, so we assert readability + type
    # rather than a specific value. Do NOT use `hasattr(ss, "_ss")` here: it is always
    # True (StepStatsView.__slots__ guarantees _ss exists post-init), so an `or`-chained
    # assertion would short-circuit and never touch the field list.
    for f in ("reward_win", "reward_kills", "reward_deaths", "reward_bomb", "reward_pbrs",
              "reward_shots", "reward_survival", "reward_inaction", "win_by_detonation",
              "win_by_defuse"):
        val = ss[f]                    # raises AttributeError via __getitem__ if field is missing
        assert isinstance(val, (int, float)), f"expected numeric for {f}, got {type(val).__name__}"
                                       # ndim==0 so split_into_channels's ndim==1 squeeze does not trigger.
    assert ss.ndim == 0
                                       # get() works with default.
    assert ss.get("nonexistent_field", "sentinel") == "sentinel"
    env.close()


def test_step_stats_in_info_flag_on_merges_with_terminal_summary():
    """On round_over ticks, info[0] must contain BOTH the terminal summary
    (winner_t/winner_ct/bomb_planted/...) AND step_stats — not one or the other."""
    import numpy as np
    env = make_env(include_step_stats_in_info=True)
    env.reset()
    # Force round_over via direct state manipulation (same pattern as
    # _setup_round_end in test_reward.py). Minimal: kill all agents of one team.
    g = env._c_env.game
    for i in range(10):
        g.agents[i].alive = 0
        g.agents[i].hp = 0
    g.agents[5].alive = 1
    g.agents[5].hp = 100
    g.agents[5].team = 1
    g.winner = 1
    g.round_over = 1
    actions = np.zeros((10, 8), dtype=np.int64)
    _, _, _, _, info = env.step(actions)
    assert len(info) == 1
    summary = info[0]
    # Terminal summary fields (from _build_terminal_info) still present.
    assert "winner_ct" in summary or "winner_t" in summary
    # step_stats also present.
    assert "step_stats" in summary
    env.close()
