"""Pin the exported bomb contract, including deliberately stale fields and tick order.

The short clocks only accelerate transitions. Placement uses the map's real area
centres and the existing scripted carrier setup; actions still drive the C step.
"""
import numpy as np
import pytest

from cs2rl.env.c.cs2_env import make_env
from cs2rl.env.map import make_simple_map
from cs2rl.eval.scripted_expert import setup_bomb_carrier
from cs2rl.spec.action import ACTION_DIM, ACTION_HEAD_NAMES, ACTION_HEAD_SIZES
from cs2rl.spec.obs import OBS_BLOCKS

OBS_GLOBAL_BASE = OBS_BLOCKS['global'][0]


def _place(env, index, area_idx):
    """Place a stationary agent at a real area centre, including its floor height."""
    a = env._c_env.game.agents[index]
    a.x, a.y = map(float, env.map_data.centroids[area_idx])
    a.z = float(env.map_data.centroids_z[area_idx])
    a.area_idx = area_idx
    a.vx = a.vy = a.vz = 0


@pytest.fixture
def bomb_env():
    """An active round with a carrier on site and short, distinct bomb clocks."""
    map_data = make_simple_map()
    env = make_env(map_data=map_data, auto_reset=False, seed=7)
    try:
        env.reset()
        setup_bomb_carrier(env, 0)
        site = next(i for i, flag in enumerate(map_data.bombsite_by_idx) if flag)
        _place(env, 0, site)
        sd = env._c_env.sd.contents
        sd.bomb_plant_time, sd.bomb_timer = 3, 7
        sd.bomb_defuse_kit, sd.bomb_defuse_time = 2, 4
        yield env
    finally:
        env.close()


def _step(env, *users):
    """Use is the only action; inspect behavior without combat/movement noise."""
    actions = np.zeros((10, ACTION_DIM), dtype=np.int32)
    actions[list(users), ACTION_HEAD_NAMES.index('use')] = 1
    return env.step(actions)


def _use_mask(env, index):
    """Read the actual policy-facing USE press bin, derived from head widths."""
    offset = sum(ACTION_HEAD_SIZES[:ACTION_HEAD_NAMES.index('use')])
    return env._masks_view[index, offset + 1]


def _plant(env):
    """Complete planting through USE, retaining native progress and stale identity."""
    for _ in range(env._c_env.sd.contents.bomb_plant_time):
        _step(env, 0)
    assert env._c_env.game.bomb_planted == 1


def test_plant_completion_retains_raw_state_and_ticks_timer(bomb_env):
    """A completion pays once, leaves legacy carrier/progress, then ticks the timer."""
    env = bomb_env
    g, sd = env._c_env.game, env._c_env.sd.contents
    _plant(env)
    assert (g.bomb_carrier_id, g.agents[0].has_bomb) == (0, 0)
    assert (g.bomb_being_planted_by, g.bomb_plant_ticks) == (-1, 3)
    assert g.bomb_ticks_left == sd.bomb_timer - 1
    assert env._c_env.step_stats.plant_tick == g.tick == 3
    assert env._c_env.episode_stats.bomb_planted == 1
    assert _use_mask(env, 0) == 0
    np.testing.assert_array_equal(env.observations[0, OBS_GLOBAL_BASE + 1:OBS_GLOBAL_BASE + 5],
                                  [0, 0, 0, 1])
    _step(env)
    assert env._c_env.step_stats.bomb_planted == 0
    assert env._c_env.episode_stats.plant_tick == 3


def test_plant_release_pauses_but_leaving_site_with_use_interrupts(bomb_env):
    """Release keeps progress even off-site; a subsequent USE interrupts and pays."""
    env = bomb_env
    g, sd = env._c_env.game, env._c_env.sd.contents
    _step(env, 0)
    _step(env)
    assert (g.bomb_being_planted_by, g.bomb_plant_ticks) == (0, 1)
    away = next(i for i, flag in enumerate(env.map_data.bombsite_by_idx) if not flag)
    _place(env, 0, away)
    _step(env)
    assert (g.bomb_being_planted_by, g.bomb_plant_ticks) == (0, 1)
    assert _use_mask(env, 0) == 0
    _step(env, 0)
    assert (g.bomb_being_planted_by, g.bomb_plant_ticks) == (-1, 0)
    assert env._c_env.step_stats.reward_bomb == -sd.reward_plant_interrupted


@pytest.mark.parametrize('loss', ['death', 'possession'])
def test_planter_loss_resets_progress_without_interruption_penalty(bomb_env, loss):
    """Death/loss frees the plant lock; a new carrier earns its own progress."""
    env = bomb_env
    g = env._c_env.game
    _step(env, 0)
    if loss == 'death':
        g.agents[0].alive = g.agents[0].hp = 0
    else:
        g.agents[0].has_bomb = 0
    _step(env)
    assert (g.bomb_being_planted_by, g.bomb_plant_ticks) == (-1, 0)
    assert env._c_env.step_stats.reward_bomb == 0
    _place(env, 1, g.agents[0].area_idx)
    _step(env)                         # death drops before pickup; pickup cannot plant on this tick
    if loss == 'possession':
        setup_bomb_carrier(env, 1)
    _step(env, 1)
    assert (g.bomb_being_planted_by, g.bomb_plant_ticks) == (1, 1)


@pytest.mark.parametrize('distance,holder', [(32.0, 2), (32.25, -1)])
def test_drop_pickup_radius_tie_and_next_tick_use(bomb_env, distance, holder):
    """Pickup is inclusive, ties choose the later T, and USE waits until next tick."""
    env = bomb_env
    g = env._c_env.game
    site = g.agents[0].area_idx
    for i in (1, 2):
        _place(env, i, site)
        g.agents[i].x += distance
    g.agents[0].alive = g.agents[0].hp = 0
    _step(env, 1, 2)
    assert g.bomb_carrier_id == holder
    assert g.bomb_is_dropped == (holder == -1)
    assert g.round_designated_carrier_id == 0
    assert g.agents[0].has_bomb == 0
    assert g.bomb_area_idx == -1       # drop stores coordinates only
    assert g.bomb_plant_ticks == 0
    if holder >= 0:
        assert _use_mask(env, holder) == 1
        assert env.observations[holder, OBS_GLOBAL_BASE + 1] == 1
        _step(env, holder)
        assert (g.bomb_being_planted_by, g.bomb_plant_ticks) == (holder, 1)


@pytest.mark.parametrize('losing_team', [0, 1])
def test_elimination_still_drops_but_does_not_invalidate_or_pick_up(bomb_env, losing_team):
    """Round end precedes drop; round-gated cleanup/pickup must remain skipped."""
    env = bomb_env
    g = env._c_env.game
    _step(env, 0)
    _place(env, 1, g.agents[0].area_idx)
    g.agents[0].alive = g.agents[0].hp = 0
    for i in range(losing_team * 5, losing_team * 5 + 5):
        g.agents[i].alive = g.agents[i].hp = 0
    _, _, terminal, _, info = _step(env)
    assert terminal.all() and info
    assert (g.round_over, g.winner, g.bomb_is_dropped, g.bomb_carrier_id) == (1, 1 - losing_team, 1,
                                                                              -1)
    assert g.agents[0].has_bomb == 0
    assert g.agents[1].has_bomb == 0   # the live T in the CT-elimination case cannot pick up
    assert (g.bomb_being_planted_by, g.bomb_plant_ticks) == (0, 1)
    assert env._c_env.step_stats.win_by_defuse == 0


@pytest.mark.parametrize('kit', [0, 1])
def test_defuse_completion_wins_before_timer_and_routes_stats(bomb_env, kit):
    """Kit timing and terminal flags come from the real completion event."""
    env = bomb_env
    g, sd = env._c_env.game, env._c_env.sd.contents
    _plant(env)
    _place(env, 5, g.bomb_area_idx)
    g.agents[5].has_kit = kit
    ticks = sd.bomb_defuse_kit if kit else sd.bomb_defuse_time
    g.bomb_ticks_left = ticks
    for _ in range(ticks - 1):
        _step(env, 5)
        assert not g.round_over
    _, _, terminal, _, _ = _step(env, 5)
    assert terminal.all() and g.winner == 1
    assert (g.bomb_ticks_left, g.bomb_defuse_ticks, g.bomb_being_defused_by) == (1, ticks, 5)
    ss, es = env._c_env.step_stats, env._c_env.episode_stats
    assert (ss.bomb_defused, es.bomb_defused, ss.win_by_defuse, ss.win_by_detonation) == (1, 1, 1,
                                                                                          0)
    assert ss.reward_bomb == sd.reward_defuse


@pytest.mark.parametrize('cancel', ['release', 'leave', 'death'])
def test_defuse_cancellation_resets_progress(bomb_env, cancel):
    """Defusing resets on release, leaving, or death, unlike planting's pause."""
    env = bomb_env
    g = env._c_env.game
    _plant(env)
    _place(env, 5, g.bomb_area_idx)
    _step(env, 5)
    assert (g.bomb_being_defused_by, g.bomb_defuse_ticks) == (5, 1)
    if cancel == 'leave':
        _place(env, 5, next(i for i, flag in enumerate(env.map_data.bombsite_by_idx) if not flag))
    elif cancel == 'death':
        g.agents[5].alive = g.agents[5].hp = 0
    _step(env, *(() if cancel == 'release' else (5, )))
    assert (g.bomb_being_defused_by, g.bomb_defuse_ticks) == (-1, 0)
    assert env._c_env.step_stats.bomb_defused == 0


@pytest.mark.parametrize('ending', ['detonation', 'timeout', 'elimination'])
def test_end_reason_precedence(bomb_env, ending):
    """Elimination beats bomb work; planting beats round timeout; timer routes win."""
    env = bomb_env
    g = env._c_env.game
    if ending != 'timeout':
        _plant(env)
        g.bomb_ticks_left = 1
    g.round_ticks_left = 1
    if ending == 'elimination':
        for i in range(5):
            g.agents[i].alive = g.agents[i].hp = 0
    _step(env)
    ss = env._c_env.step_stats
    assert g.round_over
    assert (g.winner, ss.win_by_detonation, ss.timed_out) == {
        'detonation': (0, 1, 0),
        'timeout': (-1, 0, 1),
        'elimination': (1, 0, 0)
    }[ending]
    if ending == 'elimination':
        assert g.bomb_ticks_left == 1 and not g.bomb_is_dropped


def test_overlapping_flags_keep_distinct_observation_and_use_precedence(bomb_env):
    """Safe legacy raw assembly: dropped one-hot, planted timer and CT eligibility."""
    env = bomb_env
    g = env._c_env.game
    _plant(env)
    g.bomb_is_dropped = 1
    _place(env, 5, g.bomb_area_idx)
    _step(env)
    np.testing.assert_array_equal(env.observations[5, OBS_GLOBAL_BASE + 1:OBS_GLOBAL_BASE + 5],
                                  [0, 0, 1, 0])
    assert env.observations[5, OBS_GLOBAL_BASE + 8] == np.float32(g.bomb_ticks_left / 7)
    assert (_use_mask(env, 0), _use_mask(env, 5)) == (0, 1)
    assert (g.bomb_planted, g.bomb_is_dropped, g.bomb_carrier_id) == (1, 1, 0)


@pytest.mark.parametrize('resolution', ['defuse', 'detonate', 'plant'])
def test_plant_on_round_deadline_and_same_tick_resolution(bomb_env, resolution):
    """Plant runs before CT USE, then bomb timer, then the unplanted round clock."""
    env = bomb_env
    g, sd = env._c_env.game, env._c_env.sd.contents
    sd.bomb_plant_time = sd.bomb_defuse_kit = 1
    sd.bomb_timer = 7 if resolution == 'plant' else 1
    g.round_ticks_left = 1
    _place(env, 5, g.agents[0].area_idx)
    g.agents[5].has_kit = 1
    _step(env, *((0, 5) if resolution == 'defuse' else (0, )))
    ss = env._c_env.step_stats
    assert ss.bomb_planted == 1 and ss.timed_out == 0
    assert (g.round_over, g.winner, ss.win_by_defuse, ss.win_by_detonation) == {
        'defuse': (1, 1, 1, 0),
        'detonate': (1, 0, 0, 1),
        'plant': (0, -1, 0, 0)
    }[resolution]
    assert g.bomb_ticks_left == {'defuse': 1, 'detonate': 0, 'plant': 6}[resolution]
