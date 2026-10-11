"""process_combat resolves a tick's shots simultaneously (#152 L2).

Every shot of a tick is resolved against the agents alive BEFORE combat (cs2_combat.h): a
shooter killed earlier in the loop still fires, a later ray still stops on an agent this tick
has already killed, and that agent's death is recorded once. A round in which the last T and
the last CT die on one tick is a draw (WINNER_DRAW, cs2_types.h): no outcome reward and
neither winner_t nor winner_ct.
WHY: with deaths applied inside the loop, the T rows (0..4) fired first, so in a same-tick
exchange the T lived and the CT never fired. Two identical scripted duel bots then won 646 T
rounds of 1000 (lane E, 2026-10-11).

Each test pins one hunk of the fix:
  * the shooter test (a CT killed by agent 0 this tick still fires): the exchange is a double
    kill and a draw, and both shooters pay the shot penalty (cs2_rewards.h charges every agent
    that fired this tick, dead or alive);
  * the enemy test (a killed agent still blocks a later ray): the agent behind it is unhurt;
  * the kill record (a body hit again is not killed again): one kill, credited once;
  * the round end (both teams dead is a draw, not a CT win).
A lone shooter is the control: its shot is the same in either order.

SETUP: the arena duel map, the cone off (exact rays), a round that never times out, rows on one
lane facing each other; the target's hp is 1, so any hit kills.
"""
import math

import pytest

from cs2rl.env.c.cs2_env import make_env
from cs2rl.env.c.hit_model import hit_geometry
from cs2rl.env.config import EnvConfig
from cs2rl.env.map import make_arena_duel_map
from cs2rl.spec.action import ACTION_HEAD_NAMES
from tests._helpers.scenario import place_agent, ready_to_fire, zero_actions

HEAD_SHOOT = ACTION_HEAD_NAMES.index("shoot")
WINNER_DRAW = -2                       # MIRROR of cs2_types.h WINNER_DRAW
LANE_Y = 200.0


def _env(n_active):
    env = make_env(config=EnvConfig(n_active_per_team=n_active,
                                    pin_pitch=0,
                                    jump_enabled=0,
                                    round_time=1_000_000,
                                    accuracy=False),
                   map_data=make_arena_duel_map(),
                   auto_reset=False,
                   seed=0)
    env.reset()
    return env


def _aim_at(env, shooter, target, cont):
    """Face `shooter` at `target` and write the pitch onto the target's head into `cont`."""
    g = env._c_env.game
    a, t = g.agents[shooter], g.agents[target]
    geo = hit_geometry(env.config.hitboxes)
    dx, dy = t.x - a.x, t.y - a.y
    a.facing = math.atan2(dy, dx)
    cont[shooter,
         1] = math.atan2(geo.target_z(t.z, False) - geo.eye_z(a.z, False), math.hypot(dx, dy))


def _fire(env, shooters, targets):
    """Each shooter (ready, aimed at its target) presses shoot on one step; returns the rewards."""
    acts, cont = zero_actions()
    for s, t in zip(shooters, targets, strict=True):
        ready_to_fire(env, s)
        _aim_at(env, s, t, cont)
        acts[s, HEAD_SHOOT] = 1
    _, rew, term, _, _ = env.step(acts, cont)
    return rew, term


def test_a_same_tick_exchange_is_a_double_kill_and_a_draw():
    env = _env(1)
    try:
        g = env._c_env.game
        place_agent(env, 0, 150.0, LANE_Y)
        place_agent(env, 5, 350.0, LANE_Y)
        g.agents[0].hp = g.agents[5].hp = 1
        rew, term = _fire(env, [0, 5], [5, 0])
        es = env._c_env.episode_stats
        assert (g.agents[0].alive, g.agents[5].alive) == (0, 0)
        assert (es.kills_t, es.kills_ct, es.shots_hit) == (1, 1, 2)
        assert (g.round_over, g.winner) == (1, WINNER_DRAW)
        assert (es.winner, es.winner_t, es.winner_ct, es.timed_out) == (WINNER_DRAW, 0, 0, 0)
        assert (es.reward_win_t, es.reward_win_ct) == (0.0, 0.0)
        # Both shooters died on their firing tick and both pay the shot penalty (cs2_rewards.h).
        sd = env._c_env.sd.contents
        assert sd.reward_shot_penalty > 0
        assert es.reward_shots == pytest.approx(-2 * sd.reward_shot_penalty)
        assert term.all()
        assert rew[0] == rew[5]
    finally:
        env.close()


def test_a_lone_shooter_wins_the_round():
    """The control: one shooter's kill is an elimination win, as before the fix."""
    env = _env(1)
    try:
        g = env._c_env.game
        place_agent(env, 0, 150.0, LANE_Y)
        place_agent(env, 5, 350.0, LANE_Y)
        g.agents[5].hp = 1
        _fire(env, [0], [5])
        es = env._c_env.episode_stats
        assert (g.agents[0].alive, g.agents[5].alive) == (1, 0)
        assert (es.kills_t, es.kills_ct) == (1, 0)
        assert (g.round_over, g.winner, es.winner_t, es.winner_ct) == (1, 0, 1, 0)
    finally:
        env.close()


def _two_on_a_lane():
    """T rows 0 (x 150) and 1 (x 100) on one lane, CT 5 at 250 (hp 1) in front of CT 6 at 350:
    both T rays reach CT 5 first. Agent 0 shoots first in the loop and kills CT 5."""
    env = _env(2)
    g = env._c_env.game
    place_agent(env, 0, 150.0, LANE_Y)
    place_agent(env, 1, 100.0, LANE_Y)
    place_agent(env, 5, 250.0, LANE_Y)
    place_agent(env, 6, 350.0, LANE_Y)
    g.agents[5].hp = 1
    rew, _ = _fire(env, [0, 1], [5, 5])
    return env, rew


def test_a_body_killed_this_tick_still_blocks_a_later_ray():
    env, _ = _two_on_a_lane()
    try:
        g = env._c_env.game
        es = env._c_env.episode_stats
        assert g.agents[5].alive == 0
        assert es.shots_hit == 2       # agent 1's shot stopped on the body
        assert (g.agents[6].alive, g.agents[6].hp) == (1, 100)
    finally:
        env.close()


def test_a_body_hit_again_is_killed_once_and_credited_once():
    env, rew = _two_on_a_lane()
    try:
        es = env._c_env.episode_stats
        sd = env._c_env.sd.contents
        assert (es.kills_t, es.kills_ct) == (1, 0)
        assert es.reward_kills == pytest.approx(sd.reward_kill)
        # The kill is agent 0's, the lowest index whose shot took hp to 0 (PITFALL in cs2_combat.h).
        assert rew[0] - rew[1] == pytest.approx(sd.reward_kill)
        assert env._c_env.game.round_over == 0         # CT 6 lives
    finally:
        env.close()
