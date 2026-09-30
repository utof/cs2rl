"""R0-B (#129): compute_masks runs at the END of env_step, one decrement
before the next tick's tick_weapon, so it read fire_cd/reload_ticks/switch_ticks
one tick stale — the policy saw "cannot fire" one tick longer than
process_combat actually enforces, and a legal first shot was masked out.
Fixed by evaluating max(c-1, 0) in compute_masks.

Tick arithmetic (all against tick_weapon in cs2_weapons.h):
- rifle cycle_ticks=2 ⇒ 1 shot / 2 ticks when spamming through the mask.
- empty-mag reload (reload_ticks=39): mag is refilled INSIDE tick_weapon at the
  zero-crossing, so the mask stays closed one extra tick on has_ammo ⇒ T+40.
- partial-mag reload: has_ammo already true ⇒ mask opens at T+39.
- weapon switch (WEAPON_SWITCH_TICKS=8) ⇒ T+8.
Each test drives agent 0's shoot head straight from env._masks_view, so a
stale mask shows up as a delayed first shot (fired_this_tick).
"""
import numpy as np

from cs2rl.env.c.cs2_env import make_env
from cs2rl.env.config import EnvConfig
from cs2rl.spec.action import ACTION_HEAD_SIZES

N_AGENTS, ACTION_DIM, AIM_DIM = 10, 7, 2
H_SHOOT, H_RELOAD, H_WEAPON = 1, 2, 3
MOFF = np.concatenate([[0], np.cumsum(ACTION_HEAD_SIZES)[:-1]])


def _shoot_mask(env, i=0):
    return int(env._masks_view[i, MOFF[H_SHOOT] + 1])


def _step(env, act):
    return env.step(act, np.zeros((N_AGENTS, AIM_DIM), dtype=np.float32))


def _spam_shoot_ticks(env, n):
    """Return the tick indices (0-based) at which agent 0 actually fired,
    pressing shoot exactly when the mask says it is allowed."""
    fired = []
    for t in range(n):
        act = np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32)
        act[0, H_SHOOT] = _shoot_mask(env)
        _step(env, act)
        if env._c_env.game.agents[0].fired_this_tick:
            fired.append(t)
    return fired


def test_rifle_one_shot_per_two_ticks(simple_map):
    env = make_env(map_data=simple_map, config=EnvConfig(n_active_per_team=1), seed=1)
    try:
        env.reset()
        fired = _spam_shoot_ticks(env, 10)
        assert fired == [0, 2, 4, 6, 8], fired
    finally:
        env.close()


def test_empty_mag_reload_first_shot_at_T_plus_40(simple_map):
    env = make_env(map_data=simple_map, config=EnvConfig(n_active_per_team=1), seed=1)
    try:
        env.reset()
        a = env._c_env.game.agents[0]
        a.ammo_clip[0] = 0
        act = np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32)
        act[0, H_RELOAD] = 1
        _step(env, act)                # T: reload starts (reload_ticks=39)
        fired = _spam_shoot_ticks(env, 45)
        assert fired[0] == 39, fired   # T+40 overall (T + 1 + 39)
    finally:
        env.close()


def test_partial_mag_reload_first_shot_at_T_plus_39(simple_map):
    env = make_env(map_data=simple_map, config=EnvConfig(n_active_per_team=1), seed=1)
    try:
        env.reset()
        a = env._c_env.game.agents[0]
        a.ammo_clip[0] = 5
        act = np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32)
        act[0, H_RELOAD] = 1
        _step(env, act)
        fired = _spam_shoot_ticks(env, 45)
        assert fired[0] == 38, fired   # T+39: mask opens one tick earlier than empty-mag
    finally:
        env.close()


def test_weapon_switch_first_shot_at_T_plus_8(simple_map):
    env = make_env(map_data=simple_map, config=EnvConfig(n_active_per_team=1), seed=1)
    try:
        env.reset()
        act = np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32)
        act[0, H_WEAPON] = 2           # switch to pistol
        _step(env, act)                # T: switch_ticks=8
        fired = _spam_shoot_ticks(env, 12)
        assert fired[0] == 7, fired    # T+8
    finally:
        env.close()


# --- Direct reads of the reload / weapon heads (fix round 1) -----------------
# The tests above only exercise reload_next/switch_next through the SHOOT head.
# These pin the side effects compute_masks must predict on the last tick:
# clip refill (reload_ticks 1->0) and slot flip (switch_ticks 1->0).


def _reload_mask(env, i=0):
    return int(env._masks_view[i, MOFF[H_RELOAD] + 1])


def _weapon_mask(env, opt, i=0):
    """opt 1 = switch-to-primary(0), opt 2 = switch-to-secondary(1)."""
    return int(env._masks_view[i, MOFF[H_WEAPON] + opt])


def _run_until(env, pred, limit=60):
    """Step no-op actions until pred(agent0) holds; return ticks stepped."""
    for t in range(limit):
        if pred(env._c_env.game.agents[0]):
            return t
        _step(env, np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32))
    raise AssertionError("predicate never held")


def test_reload_mask_closed_on_refill_tick(simple_map):
    """reload_ticks==1 at mask time => next tick_weapon refills the clip to
    mag_size and takes 1 from reserve before try_start_reload runs, which then
    rejects (clip full). The mask must say 0 there — and pressing reload on
    that tick must be a no-op in the sim."""
    env = make_env(map_data=simple_map, config=EnvConfig(n_active_per_team=1), seed=1)
    try:
        env.reset()
        a = env._c_env.game.agents[0]
        mag, clip0, res0 = a.ammo_clip[0], 5, a.ammo_reserve[0]
        a.ammo_clip[0] = clip0
        act = np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32)
        act[0, H_RELOAD] = 1
        _step(env, act)
        assert a.reload_ticks > 1 and _reload_mask(env) == 0           # mid-reload: closed
        _run_until(env, lambda s: s.reload_ticks == 1)
        assert a.ammo_clip[0] == clip0                                 # not refilled yet
        assert _reload_mask(env) == 0, "refill tick must not offer reload"
        _step(env, act)                                                # press reload anyway
        assert a.reload_ticks == 0                                     # sim rejected: clip full
        assert a.ammo_clip[0] == mag and a.ammo_reserve[0] == res0 - 1
        assert _reload_mask(env) == 0                                  # still full
        a.ammo_clip[0] = mag - 1
        _step(env, np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32))
        assert _reload_mask(env) == 1                                  # partial + idle: open
    finally:
        env.close()


def test_weapon_head_describes_target_slot_on_flip_tick(simple_map):
    """switch_ticks==1 at mask time => env_step flips weapon_slot to the target
    before the weapon head is parsed, so the mask must show the TARGET as the
    already-held option and the shoot head must use the TARGET's clip."""
    env = make_env(map_data=simple_map, config=EnvConfig(n_active_per_team=1), seed=1)
    try:
        env.reset()
        a = env._c_env.game.agents[0]
        a.ammo_clip[1] = 0                                             # pistol empty, rifle full
        assert _weapon_mask(env, 1) == 0 and _weapon_mask(env, 2) == 1
        act = np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32)
        act[0, H_WEAPON] = 2                                           # switch to pistol
        _step(env, act)
        assert a.switch_ticks > 1
        assert _weapon_mask(env, 1) == 0 and _weapon_mask(env, 2) == 0 # in progress
        _run_until(env, lambda s: s.switch_ticks == 1)
        assert a.weapon_slot == 0                                      # flip has not happened yet
        assert _weapon_mask(env, 2) == 0, "target slot must read as held"
        assert _weapon_mask(env, 1) == 1, "switch back to rifle is legal next tick"
        assert _shoot_mask(env) == 0, "shoot head must use the target (empty) clip"
        _step(env, np.zeros((N_AGENTS, ACTION_DIM), dtype=np.int32))
        assert a.weapon_slot == 1 and a.switch_ticks == 0
        assert _weapon_mask(env, 2) == 0 and _weapon_mask(env, 1) == 1
    finally:
        env.close()
