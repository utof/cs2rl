"""R0-A: combat StepStats → _build_terminal_info → compute_game_metrics."""
import math

import numpy as np
import pytest

from cs2rl.env.c.cs2_env import make_env
from cs2rl.env.config import EnvConfig, RewardWeights
from cs2rl.train import compute_game_metrics
from cs2rl.train_helpers_batch1 import split_into_channels

N_AGENTS, AIM_DIM, ACTION_DIM = 10, 2, 7
H_SHOOT = 1


def _place_duel(env, facing0):
    """Agent 0 (T) and agent 5 (CT) 40u apart in agent 0's spawn room, both standing."""
    env.reset()
    ag = env._c_env.game.agents
    a0, a5 = ag[0], ag[5]
    a5.x, a5.y, a5.z = a0.x + 40.0, a0.y, a0.z
    a5.area_idx = a0.area_idx
    a5.facing = math.pi                # looks back at agent 0
    a0.facing = facing0
    act = np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32)
    cont = np.zeros((N_AGENTS, AIM_DIM), dtype=np.float32)
    act[0, H_SHOOT] = 1
    return act, cont


def test_scripted_hit_tick(simple_map):
    # auto_reset=False: with n_active=1 a single hit can be a kill (head roll ×4
    # = 138 > 100 hp), which ends the round; the auto-reset branch
    # (cs2_env.py step -> binding.reset -> clear_stats) would memset
    # episode_stats before the asserts below read it. ~10 % flake otherwise.
    env = make_env(map_data=simple_map,
                   config=EnvConfig(n_active_per_team=1),
                   seed=1,
                   auto_reset=False)
    try:
        act, cont = _place_duel(env, facing0=0.0)
        obs, *_ = env.step(act, cont)
        # Geometry guard first: both agents must share LoS (same room, 40u apart).
        # If this fails the placement straddles a wall — fix _place_duel, not R0-A.
        # obs layout: enemy block starts at 56, 3 = can_see flag of the nearest enemy slot
        assert obs[0][56 + 3] == 1.0, "agent 0 cannot see agent 5 — _place_duel geometry"
        es = env._c_env.episode_stats
        assert es.shots_fired == 1
        assert es.shots_with_enemy_in_los == 1
        assert es.shots_facing_enemy == 1
        assert es.shots_on_target == 1
        assert es.shots_hit == 1
        assert es.shots_stance_blocked == 0
        assert es.damage_dealt > 0.0
        assert es.mutual_vis_pair_ticks == 1
        assert es.agent_ticks_with_visible_enemy == 2
        assert es.min_enemy_distance == pytest.approx(40.0)
    finally:
        env.close()


def test_scripted_shot_facing_90_off(simple_map):
    env = make_env(map_data=simple_map,
                   config=EnvConfig(n_active_per_team=1),
                   seed=1,
                   auto_reset=False)                                   # see above
    try:
        act, cont = _place_duel(env, facing0=math.pi / 2)
        env.step(act, cont)
        es = env._c_env.episode_stats
        assert es.shots_fired == 1
        assert es.shots_with_enemy_in_los == 1
        assert es.shots_facing_enemy == 0
        assert es.shots_on_target == 0
        assert es.shots_hit == 0
    finally:
        env.close()


def test_terminal_info_exports_new_keys_and_sentinel(simple_map):
    env = make_env(map_data=simple_map, config=EnvConfig(n_active_per_team=1), seed=1)
    try:
        env.reset()
        env._c_env.episode_stats.min_enemy_distance = 1e30                                           # clear_stats sentinel
        info = env._build_terminal_info()
        for k in ("shots_fired", "shots_with_enemy_in_los", "shots_facing_enemy", "shots_on_target",
                  "shots_hit", "shots_stance_blocked", "damage_dealt", "mutual_vis_pair_ticks",
                  "agent_ticks_with_visible_enemy", "reward_win_t", "reward_win_ct"):
            assert k in info, k
        assert info["min_enemy_distance_sum"] == 0.0
        assert info["min_enemy_distance_valid"] == 0
        env._c_env.episode_stats.min_enemy_distance = 123.0
        info = env._build_terminal_info()
        assert info["min_enemy_distance_sum"] == pytest.approx(123.0)
        assert info["min_enemy_distance_valid"] == 1
    finally:
        env.close()


def test_win_t_ct_one_sided_and_equal_to_terminal_rewards(simple_map):
    env = make_env(
        map_data=simple_map,
        seed=1,
        team_spirit=0.0,
        config=EnvConfig(
            n_active_per_team=1,
            rewards=RewardWeights(
                reward_win_ct_elimination=3.0,
                reward_win_t_elimination=3.0,
                pbrs_alive_weight=0.0,
                pbrs_hp_weight=0.0,
                reward_kill=0.0,
                reward_death=0.0,
                reward_inaction=0.0,
            ),
        ),
    )
    try:
        act, cont = _place_duel(env, facing0=0.0)
        env._c_env.game.agents[5].hp = 1               # one hit kills
        _, rew, term, _, info = env.step(act, cont)
        assert term[0]
        es_info = info[0]
        assert es_info["winner_t"] == 1
        assert es_info["reward_win_t"] == pytest.approx(3.0)
        assert es_info["reward_win_ct"] == pytest.approx(-3.0)
        assert es_info["reward_win"] == pytest.approx(0.0)
    finally:
        env.close()


def test_split_into_channels_ignores_win_t_ct():
    dt = np.dtype([("reward_win", "f4"), ("reward_kills", "f4"), ("reward_deaths", "f4"),
                   ("reward_bomb", "f4"), ("reward_pbrs", "f4"), ("reward_shots", "f4"),
                   ("reward_survival", "f4"), ("reward_inaction", "f4"),
                   ("win_by_detonation", "i1"), ("win_by_defuse", "i1"), ("reward_win_t", "f4"),
                   ("reward_win_ct", "f4")])
    rec = np.zeros(1, dtype=dt)
    rec["reward_kills"] = 0.3
    base = split_into_channels(rec)
    rec["reward_win_t"] = 5.0
    rec["reward_win_ct"] = -5.0
    assert split_into_channels(rec) == base


def test_compute_game_metrics_new_keys():
    logs = {
        "environment/kills_t": 1.0,
        "environment/kills_ct": 0.0,
        "environment/shots_fired": 10.0,
        "environment/shots_hit": 4.0,
        "environment/shots_facing_enemy": 8.0,
        "environment/shots_on_target": 5.0,
        "environment/shots_with_enemy_in_los": 10.0,
        "environment/shots_stance_blocked": 0.0,
        "environment/damage_dealt": 120.0,
        "environment/mutual_vis_pair_ticks": 50.0,
        "environment/agent_ticks_with_visible_enemy": 100.0,
        "environment/min_enemy_distance_sum": 100.0,
        "environment/min_enemy_distance_valid": 0.5,
        "environment/reward_win_t": 3.0,
        "environment/reward_win_ct": -3.0,
        "environment/reward_win": 0.0,
    }
    gm = compute_game_metrics(logs)
    assert "game/reward/win" not in gm
    assert gm["game/shots_fired"] == 10.0
    assert gm["game/shots_hit"] == 4.0
    assert gm["game/min_enemy_distance"] == pytest.approx(200.0)       # sum/valid = ratio of window means
    assert gm["game/min_enemy_distance_valid_frac"] == 0.5
    assert gm["game/reward/win_t"] == 3.0
    assert gm["game/reward/win_ct"] == -3.0
    gm0 = compute_game_metrics({
        "environment/min_enemy_distance_sum": 0.0,
        "environment/min_enemy_distance_valid": 0.0
    })
    assert gm0["game/min_enemy_distance"] == 0.0


def test_environment_episodes_counts_terminal_infos():
    """One evaluate() is far shorter than a round, so no real terminal info
    arrives; inject a synthetic window and assert the count is forwarded
    THROUGH mean_and_log. Deliberate deviation from spec §3 R0-A's wording
    ("inject after mean_and_log returns"): pufferl.py logs INSIDE
    mean_and_log, so a post-hoc write reaches metrics.jsonl but not the
    logger; writing self.stats first satisfies both. Same observable result."""
    from cs2rl.train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=16)
    try:
        trainer.evaluate()
        trainer.stats["kills_t"] = [0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0] # 7 synthetic episodes
        trainer.last_log_time = 0.0                                    # force the throttled mean_and_log to run
        logs = trainer.train()
        assert isinstance(logs, dict)
        assert logs["environment/episodes"] == 7.0
    finally:
        cleanup()
