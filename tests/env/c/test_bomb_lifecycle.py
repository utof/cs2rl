"""Pin the canonical bomb lifecycle (#164): one phase state, its transitions and tick order.

GameState.bomb (BombStateC) is the only bomb state. These tests drive it through
actions and the sanctioned setter (Cs2Env.give_bomb), pin the deliberate changes
from the pre-#164 raw fields (each marked "#164 decision"), and pin that
hand-assembled impossible states are rejected: a removed field name raises
AttributeError, and binding.step raises ValueError on a bomb state outside the
BombState table in cs2_types.h.

The short clocks only accelerate transitions. Placement uses the map's real area
centres and the existing scripted carrier setup; actions still drive the C step.
"""
import numpy as np
import pytest

from cs2rl.env.c.cs2_env import BombPhase, make_env
from cs2rl.env.map import make_simple_map
from cs2rl.eval.scripted_expert import setup_bomb_carrier
from cs2rl.spec.action import ACTION_DIM, ACTION_HEAD_NAMES, ACTION_HEAD_SIZES
from cs2rl.spec.obs import OBS_BLOCKS
from tests._helpers.scenario import area_centroid, kill_agent, place_agent, place_in_area

OBS_GLOBAL_BASE = OBS_BLOCKS['global'][0]


@pytest.fixture
def bomb_env():
    """An active round with a carrier on site and short, distinct bomb clocks."""
    map_data = make_simple_map()
    env = make_env(map_data=map_data, auto_reset=False, seed=7)
    try:
        env.reset()
        setup_bomb_carrier(env, 0)
        site = next(i for i, flag in enumerate(map_data.bombsite_by_idx) if flag)
        place_in_area(env, 0, site)
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


def _bomb(env):
    """(phase, agent, progress): the part of the bomb state that names who and how far."""
    b = env._c_env.game.bomb
    return (BombPhase(b.phase), b.agent, b.progress)


def _plant(env):
    """Complete planting through USE."""
    for _ in range(env._c_env.sd.contents.bomb_plant_time):
        _step(env, 0)
    assert env._c_env.game.bomb_planted == 1


def test_plant_completion_clears_carrier_and_progress_and_ticks_timer(bomb_env):
    """A completion pays once, leaves no carrier or plant progress, then ticks the timer."""
    env = bomb_env
    g, sd = env._c_env.game, env._c_env.sd.contents
    _plant(env)
    # #164 decision: before #164 the raw fields kept bomb_carrier_id 0 and
    # bomb_plant_ticks 3 here (stale); no observation, reward or mask read them.
    assert _bomb(env) == (BombPhase.PLANTED, -1, 0)
    assert g.bomb_carrier == -1
    planter = g.agents[0]
    assert g.bomb.area_idx == planter.area_idx
    assert (g.bomb.x, g.bomb.y, g.bomb.z) == (planter.x, planter.y, planter.z)
    assert g.bomb.ticks_left == sd.bomb_timer - 1
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
    sd = env._c_env.sd.contents
    _step(env, 0)
    _step(env)
    assert _bomb(env) == (BombPhase.PLANTING, 0, 1)
    away = next(i for i, flag in enumerate(env.map_data.bombsite_by_idx) if not flag)
    place_in_area(env, 0, away)
    _step(env)
    assert _bomb(env) == (BombPhase.PLANTING, 0, 1)
    assert _use_mask(env, 0) == 0
    _step(env, 0)
    assert _bomb(env) == (BombPhase.CARRIED, 0, 0)
    assert env._c_env.step_stats.reward_bomb == -sd.reward_plant_interrupted


@pytest.mark.parametrize('loss', ['death', 'handover'])
def test_planter_loss_resets_progress_without_interruption_penalty(bomb_env, loss):
    """Death or a hand-over ends the plant; the next carrier earns its own progress."""
    env = bomb_env
    g = env._c_env.game
    _step(env, 0)
    if loss == 'death':
        kill_agent(env, 0)
    else:
        env.give_bomb(1)               # the sanctioned setter moves possession without a death
    _step(env)
    assert _bomb(env)[0] != BombPhase.PLANTING and g.bomb.progress == 0
    assert env._c_env.step_stats.reward_bomb == 0
    place_in_area(env, 1, g.agents[0].area_idx)
    _step(env)                         # death drops before pickup; pickup cannot plant on this tick
    assert g.bomb_carrier == 1
    _step(env, 1)
    assert _bomb(env) == (BombPhase.PLANTING, 1, 1)


@pytest.mark.parametrize('distance,holder', [(32.0, 2), (32.25, -1)])
def test_drop_pickup_radius_tie_and_next_tick_use(bomb_env, distance, holder):
    """Pickup is inclusive, ties choose the later T, and USE waits until next tick."""
    env = bomb_env
    g = env._c_env.game
    site = g.agents[0].area_idx
    x, y = area_centroid(env, site)
    for i in (1, 2):
        place_agent(env, i, x + distance, y)
    dead_x, dead_y = g.agents[0].x, g.agents[0].y
    kill_agent(env, 0)
    _step(env, 1, 2)
    assert g.bomb_carrier == holder
    assert (g.bomb.phase == BombPhase.DROPPED) == (holder == -1)
    assert g.round_designated_carrier_id == 0
    # A drop stores coordinates only.
    assert g.bomb.area_idx == -1
    assert g.bomb.progress == 0
    if holder >= 0:
        # #164 decision: a carried bomb has no ground position. Before #164
        # bomb_x/y/z kept the old drop point after a pickup; obs and render
        # read them only while dropped or planted.
        assert (g.bomb.x, g.bomb.y, g.bomb.z) == (0.0, 0.0, 0.0)
        assert _use_mask(env, holder) == 1
        assert env.observations[holder, OBS_GLOBAL_BASE + 1] == 1
        _step(env, holder)
        assert _bomb(env) == (BombPhase.PLANTING, holder, 1)
    else:
        assert (g.bomb.x, g.bomb.y) == (dead_x, dead_y)


@pytest.mark.parametrize('losing_team', [0, 1])
def test_elimination_still_drops_but_does_not_pick_up(bomb_env, losing_team):
    """Round end precedes drop; round-gated pickup must remain skipped."""
    env = bomb_env
    g = env._c_env.game
    _step(env, 0)
    place_in_area(env, 1, g.agents[0].area_idx)
    kill_agent(env, 0)
    # Agent 0 is already dead when the T side loses.
    for i in range(losing_team * 5, losing_team * 5 + 5):
        if g.agents[i].alive:
            kill_agent(env, i)
    _, _, terminal, _, info = _step(env)
    assert terminal.all() and info
    # The live T in the CT-elimination case cannot pick up: carrier stays -1.
    assert (g.round_over, g.winner, _bomb(env)[0], g.bomb_carrier) == (1, 1 - losing_team,
                                                                       BombPhase.DROPPED, -1)
    # #164 decision: the drop discards the plant on the round-over tick too.
    # Before #164 bomb_being_planted_by/bomb_plant_ticks stayed (0, 1) here, so
    # this terminal observation's plant-progress slot read 1/3. With auto_reset
    # the terminal observation is replaced by the reset one.
    assert g.bomb.progress == 0
    assert env.observations[:, OBS_GLOBAL_BASE + 9].max() == 0
    assert env._c_env.step_stats.win_by_defuse == 0


@pytest.mark.parametrize('kit', [0, 1])
def test_defuse_completion_wins_before_timer_and_routes_stats(bomb_env, kit):
    """Kit timing and terminal flags come from the real completion event."""
    env = bomb_env
    g, sd = env._c_env.game, env._c_env.sd.contents
    _plant(env)
    place_in_area(env, 5, g.bomb.area_idx)
    g.agents[5].has_kit = kit
    ticks = sd.bomb_defuse_kit if kit else sd.bomb_defuse_time
    # A shorter countdown: still a reachable PLANTED state.
    g.bomb.ticks_left = ticks
    for _ in range(ticks - 1):
        _step(env, 5)
        assert not g.round_over
    _, _, terminal, _, _ = _step(env, 5)
    assert terminal.all() and g.winner == 1
    assert _bomb(env) == (BombPhase.DEFUSED, 5, ticks)
    assert g.bomb.ticks_left == 1
    assert env.observations[5, OBS_GLOBAL_BASE + 10] == 1.0
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
    place_in_area(env, 5, g.bomb.area_idx)
    _step(env, 5)
    assert _bomb(env) == (BombPhase.DEFUSING, 5, 1)
    if cancel == 'leave':
        place_in_area(env, 5,
                      next(i for i, flag in enumerate(env.map_data.bombsite_by_idx) if not flag))
    elif cancel == 'death':
        kill_agent(env, 5)
    _step(env, *(() if cancel == 'release' else (5, )))
    assert _bomb(env) == (BombPhase.PLANTED, -1, 0)
    assert env._c_env.step_stats.bomb_defused == 0


def test_detonation_during_defuse_discards_defuse_progress(bomb_env):
    """The countdown keeps running while defusing; detonation ends the defuse."""
    env = bomb_env
    g = env._c_env.game
    _plant(env)
    place_in_area(env, 5, g.bomb.area_idx)
    # Without a kit defuse_time is 4: two ticks cannot finish it.
    g.agents[5].has_kit = 0
    g.bomb.ticks_left = 2
    _step(env, 5)
    assert _bomb(env) == (BombPhase.DEFUSING, 5, 1) and g.bomb.ticks_left == 1
    _, _, terminal, _, _ = _step(env, 5)
    assert terminal.all() and g.winner == 0
    assert env._c_env.step_stats.win_by_detonation == 1
    # #164 decision: before #164 the defuse lock and its ticks stayed (5, 2)
    # after detonation, so this terminal observation's defuse-progress slot
    # read 2/4. DETONATED names no agent and no progress.
    assert _bomb(env) == (BombPhase.DETONATED, -1, 0) and g.bomb.ticks_left == 0
    assert env.observations[5, OBS_GLOBAL_BASE + 10] == 0
    np.testing.assert_array_equal(env.observations[5, OBS_GLOBAL_BASE + 1:OBS_GLOBAL_BASE + 5],
                                  [0, 0, 0, 1])
    # Defuse eligibility outlives the resolution, as before #164.
    assert _use_mask(env, 5) == 1


@pytest.mark.parametrize('ending', ['detonation', 'timeout', 'elimination'])
def test_end_reason_precedence(bomb_env, ending):
    """Elimination beats bomb work; planting beats round timeout; timer routes win."""
    env = bomb_env
    g = env._c_env.game
    if ending != 'timeout':
        _plant(env)
        g.bomb.ticks_left = 1
    g.round_ticks_left = 1
    if ending == 'elimination':
        for i in range(5):
            kill_agent(env, i)
    _step(env)
    ss = env._c_env.step_stats
    assert g.round_over
    assert (g.winner, ss.win_by_detonation, ss.timed_out) == {
        'detonation': (0, 1, 0),
        'timeout': (-1, 0, 1),
        'elimination': (1, 0, 0)
    }[ending]
    assert _bomb(env)[0] == {
        'detonation': BombPhase.DETONATED,
        'timeout': BombPhase.CARRIED,
        'elimination': BombPhase.PLANTED
    }[ending]
    if ending == 'elimination':
        assert g.bomb.ticks_left == 1


@pytest.mark.parametrize('resolution', ['defuse', 'detonate', 'plant'])
def test_plant_on_round_deadline_and_same_tick_resolution(bomb_env, resolution):
    """Plant runs before CT USE, then bomb timer, then the unplanted round clock."""
    env = bomb_env
    g, sd = env._c_env.game, env._c_env.sd.contents
    sd.bomb_plant_time = sd.bomb_defuse_kit = 1
    sd.bomb_timer = 7 if resolution == 'plant' else 1
    g.round_ticks_left = 1
    place_in_area(env, 5, g.agents[0].area_idx)
    g.agents[5].has_kit = 1
    _step(env, *((0, 5) if resolution == 'defuse' else (0, )))
    ss = env._c_env.step_stats
    assert ss.bomb_planted == 1 and ss.timed_out == 0
    assert (g.round_over, g.winner, ss.win_by_defuse, ss.win_by_detonation) == {
        'defuse': (1, 1, 1, 0),
        'detonate': (1, 0, 0, 1),
        'plant': (0, -1, 0, 0)
    }[resolution]
    assert g.bomb.ticks_left == {'defuse': 1, 'detonate': 0, 'plant': 6}[resolution]
    assert _bomb(env)[0] == {
        'defuse': BombPhase.DEFUSED,
        'detonate': BombPhase.DETONATED,
        'plant': BombPhase.PLANTED
    }[resolution]


# ── Rejection of hand-assembled states ──────────────────────────────────────
# These tests write raw fields ON PURPOSE (#170): the states are deliberately impossible,
# or (valid_planted) the positive control of the check that rejects the others.


@pytest.mark.parametrize('name', [
    'bomb_planted', 'bomb_is_dropped', 'bomb_carrier_id', 'bomb_carrier', 'bomb_area_idx', 'bomb_x',
    'bomb_ticks_left', 'bomb_being_planted_by', 'bomb_plant_ticks', 'bomb_being_defused_by',
    'bomb_defuse_ticks'
])
def test_removed_or_derived_game_fields_cannot_be_written(bomb_env, name):
    """A write to a pre-#164 field name raises instead of setting a dead attribute.

    bomb_planted and bomb_carrier are read-only derived properties; the rest no
    longer exist. Without GameStateC's empty __slots__ ctypes would accept the
    write as a plain instance attribute the sim never reads.
    """
    with pytest.raises(AttributeError):
        setattr(bomb_env._c_env.game, name, 1)


def test_removed_agent_has_bomb_cannot_be_written(bomb_env):
    """has_bomb left AgentState: possession is GameState.bomb (read bomb_carrier)."""
    with pytest.raises(AttributeError):
        bomb_env._c_env.game.agents[0].has_bomb = 1


def _set_valid_planted(g, site):
    b = g.bomb
    b.phase, b.agent, b.progress, b.ticks_left, b.area_idx = BombPhase.PLANTED, -1, 0, 5, site


@pytest.mark.parametrize('case', [
    'valid_planted', 'planted_with_carrier', 'carried_by_ct', 'dropped_with_progress',
    'planted_off_site', 'detonated_round_live', 'unknown_phase'
])
def test_step_rejects_a_bomb_state_outside_the_table(bomb_env, case):
    """binding.step runs bomb_state_error first: impossible states raise, valid ones step."""
    env = bomb_env
    g = env._c_env.game
    site = g.agents[0].area_idx
    off_site = next(i for i, flag in enumerate(env.map_data.bombsite_by_idx) if not flag)
    b = g.bomb
    if case == 'valid_planted':
        _set_valid_planted(g, site)
    elif case == 'planted_with_carrier':
        _set_valid_planted(g, site)
        b.agent = 0                    # planted AND carried: the pre-#164 stale-carrier state
    elif case == 'carried_by_ct':
        b.agent = 5
    elif case == 'dropped_with_progress':
        b.phase, b.agent, b.progress = BombPhase.DROPPED, -1, 2
    elif case == 'planted_off_site':
        _set_valid_planted(g, off_site)
    elif case == 'detonated_round_live':
        _set_valid_planted(g, site)
        b.phase, b.ticks_left = BombPhase.DETONATED, 0
    else:
        b.phase = 99
    if case == 'valid_planted':
        _step(env)
        assert _bomb(env) == (BombPhase.PLANTED, -1, 0) and g.bomb.ticks_left == 4
    else:
        tick = g.tick
        with pytest.raises(ValueError, match='invalid GameState.bomb before step'):
            _step(env)
        assert g.tick == tick          # rejected before env_step ran


@pytest.mark.parametrize('case', ['live_t', 'ct', 'out_of_range', 'dead_t', 'planted'])
def test_give_bomb_hands_over_only_to_a_live_t_before_the_plant(bomb_env, case):
    """Cs2Env.give_bomb is the one sanctioned carrier setter, with bomb_give_error's checks."""
    env = bomb_env
    g = env._c_env.game
    target = {'live_t': 2, 'ct': 5, 'out_of_range': 10, 'dead_t': 3, 'planted': 2}[case]
    if case == 'dead_t':
        kill_agent(env, 3)
    if case == 'planted':
        _plant(env)
    if case == 'live_t':
        env.give_bomb(target)
        assert _bomb(env) == (BombPhase.CARRIED, 2, 0) and g.bomb_carrier == 2
        return
    before = (g.bomb.phase, g.bomb.agent, g.bomb.progress)
    with pytest.raises(ValueError, match=f'give_bomb\\({target}\\)'):
        env.give_bomb(target)
    assert (g.bomb.phase, g.bomb.agent, g.bomb.progress) == before
