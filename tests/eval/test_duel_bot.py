"""cs2rl.eval.duel_bot: the scripted duel bot (#152 L2).

WHAT IS PINNED
* The movement mirror: ``_ground_velocity`` predicts the C state's velocity after
  env.step, to 1e-3 u/s, on every tick of a seeded run of random move bins and turns
  that stays WALL_MARGIN inside the map's extent (a wall clips the velocity; 68 of the
  160 ticks at seed 3, among them ticks at the speed cap and at rest).
  The stop and adad styles' "fire only when slow" rule and every style's aim from the
  post-move position rest on it.
* The CS2 profiles at 16 Hz: every template's dials in ticks (ReGameDLL's half-up
  rounding), and the shipped focus values.
* The dials' validation: each out-of-range dial raises ValueError.
* reaction_mode: with k ticks of reaction, neither mode engages before tick k; then
  "recognition" aims at the enemy as seen now and "percept" at the enemy as seen k
  ticks ago, and "recognition" drops an enemy it no longer sees while "percept" keeps
  shooting at the old percept.
* Rows are independent: one DuelBot over two rows with different dials (and a reset
  of one row) equals two one-row bots fed the same rows. That is what a trainer
  needs (per-row state, per-row dials through ``configure``).
* DuelBotActor fills only its own rows.
* Importing the module pulls none of the HEAVY modules of tests/train/test_w1_modules.py.

The duels themselves (symmetry, obs-only = ground truth, accounting, floors) are
pinned in tests/experiment/test_oracle_duel.py.
"""
import numpy as np
import pytest

from cs2rl.eval import duel_bot as B
from cs2rl.eval.duel_bot import PROFILES, DuelBot, DuelBotActor, DuelBotParams
from cs2rl.spec.action import ACTION_DIM, ACTION_MASK_DIM, AIM_DIM
from cs2rl.spec.obs import OBS_DIM


def _arena():
    from cs2rl.eval.baselines import BaselineEvaluator
    from cs2rl.experiment import oracle_statue as L0
    env = L0.build_env(0)
    return env, BaselineEvaluator(env, episodes=2, seed=0)


def test_ground_velocity_predicts_the_sim_velocity_after_a_step():
    from cs2rl.experiment import oracle_statue as L0
    env, ev = _arena()
    rng = np.random.default_rng(3)
    vmax = 220.0
    try:
        env.reset()
        st = ev.reader.read().snapshot()
        sd = env._c_env.sd.contents
        hx, hy = 1.0 / sd.inv_x_range, 1.0 / sd.inv_y_range
        cx, cy = sd.x_offset * hx, sd.y_offset * hy
        checked, fastest, stopped, move = 0, 0.0, 0, 0
        for t in range(160):
            if t % 8 == 0:
                move = int(rng.integers(0, 9))
            act = np.zeros((len(st["x"]), ACTION_DIM), dtype=np.int32)
            cont = np.zeros((len(st["x"]), AIM_DIM), dtype=np.float32)
            act[L0.HERO, B.H_MOVE] = move
            cont[L0.HERO, 0] = rng.uniform(-0.3, 0.3)
            pred = B._ground_velocity(st["vx"][L0.HERO], st["vy"][L0.HERO], move,
                                      st["facing"][L0.HERO], vmax)
            x0, y0 = st["x"][L0.HERO], st["y"][L0.HERO]
            env.step(act, cont)
            st = ev.reader.read().snapshot()
            got = np.array([st["vx"][L0.HERO], st["vy"][L0.HERO]])
            # A wall clips the move and zeroes a velocity component (cs2_movement.h's axis
            # split): compare only ticks that start and end WALL_MARGIN inside the extent.
            near = [
                abs(v - c) > h - B.WALL_MARGIN
                for v, c, h in ((x0, cx, hx), (y0, cy, hy), (st["x"][L0.HERO], cx, hx),
                                (st["y"][L0.HERO], cy, hy))
            ]
            if any(near):
                continue
            assert np.allclose(pred, got, atol=1e-3), (t, move, pred, got)
            checked += 1
            fastest = max(fastest, float(np.hypot(*got)))
            stopped += bool(move == 0 and not got.any())
    finally:
        env.close()
    # The run reaches the speed cap (the accelerate clamp) and comes to rest (friction to 0).
    assert checked >= 60 and fastest > 215.0 and stopped > 0, (checked, fastest, stopped)


def test_the_profiles_are_the_cs2_templates_at_16_hz():
    ticks = {
        k: (p.reaction_ticks, p.attack_delay_ticks, p.focus_interval_ticks)
        for k, p in PROFILES.items()
    }
    assert ticks == {
        "elite": (1, 0, 1),
        "expert": (3, 0, 3),
        "veryhard": (5, 0, 5),
        "hard": (6, 0, 6),
        "tough": (8, 11, 8),
        "normal": (10, 13, 10),
        "fair": (10, 14, 11),
        "easy": (10, 11, 11)
    }
    focus = {
        k: (p.focus_initial_deg, p.focus_decay, p.focus_offset_scale)
        for k, p in PROFILES.items()
    }
    assert focus["hard"] == (10.0, 0.4, 0.2) and focus["elite"] == (0.5, 0.1, 0.05)
    assert focus["easy"] == (20.0, 0.7, 0.6)
    assert all(p.style == "plant" and p.lead and p.reaction_mode == "recognition"
               for p in PROFILES.values())


@pytest.mark.parametrize("kw", [
    dict(reaction_ticks=-1),
    dict(reaction_ticks=B.MAX_REACTION_TICKS + 1),
    dict(attack_delay_ticks=-1),
    dict(focus_interval_ticks=0),
    dict(strafe_ticks=0),
    dict(focus_initial_deg=-1.0),
    dict(focus_decay=0.0),
    dict(focus_decay=1.5),
    dict(focus_offset_scale=1.0),
    dict(style="crouch"),
    dict(reaction_mode="late")
])
def test_an_out_of_range_dial_raises(kw):
    with pytest.raises(ValueError):
        DuelBotParams(**kw)


def _state(target_y, seen=True):
    """A two-agent C snapshot: the bot at the origin facing +x, an enemy at (100, y)."""
    st = {
        "x": np.array([0.0, 100.0]),
        "y": np.array([0.0, target_y]),
        "z": np.zeros(2),
        "team": np.array([0, 1]),
        "alive": np.ones(2),
        "vx": np.zeros(2),
        "vy": np.array([0.0, 160.0]),
        "facing": np.zeros(2),
        "fire_cd": np.zeros(2),
        "reload_ticks": np.zeros(2),
        "switch_ticks": np.zeros(2),
        "ammo_clip": np.full(2, 30),
        "ammo_reserve": np.full(2, 90)
    }
    return st, np.array([[False, seen], [seen, False]])


@pytest.mark.parametrize("mode", B.REACTION_MODES)
def test_reaction_mode_says_what_is_late(mode):
    k = 3
    bot = DuelBot(1,
                  np.random.default_rng(0),
                  DuelBotParams(reaction_ticks=k, reaction_mode=mode, lead=False),
                  max_turn_speed=1.0,
                  half_x=500.0,
                  half_y=500.0,
                  aim_dz=0.0,
                  move_speed=220.0)
    ys = [10.0 * t for t in range(8)]
    for t, y in enumerate(ys):
        st, seen = _state(y)
        act, cont = bot.step_state(st, [0], seen)
        if t < k:
            assert act[0, B.H_SHOOT] == 0 and cont[0, 0] == 0.0, t
            continue
        aim_y = y if mode == "recognition" else ys[t - k]
        assert act[0, B.H_SHOOT] == 1 and cont[0, 0] == pytest.approx(np.arctan2(aim_y, 100.0)), t
    # The enemy drops out of sight: recognition stops at once; percept fires at the old percept.
    st, seen = _state(80.0, seen=False)
    act, _ = bot.step_state(st, [0], seen)
    assert act[0, B.H_SHOOT] == (mode == "percept")


def _random_rows(rng, n):
    obs = rng.uniform(-1.0, 1.0, (n, OBS_DIM))
    mask = rng.integers(0, 2, (n, ACTION_MASK_DIM))
    return obs, mask


def test_rows_are_independent_and_reset_per_row():
    a = DuelBotParams(style="stop", reaction_ticks=3)
    b = DuelBotParams(style="adad",
                      reaction_ticks=5,
                      reaction_mode="percept",
                      lead=False,
                      strafe_ticks=3)
    geo = dict(max_turn_speed=0.4, half_x=200.0, half_y=150.0, aim_dz=2.4, move_speed=220.0)
    both = DuelBot(2, np.random.default_rng(0), [a, b], **geo)
    one = [DuelBot(1, np.random.default_rng(0), p, **geo) for p in (a, b)]
    rng = np.random.default_rng(1)
    for t in range(40):
        if t == 17:
            both.reset([1])
            one[1].reset()
        obs, mask = _random_rows(rng, 2)
        act, cont = both.step_obs(obs, mask)
        for r in (0, 1):
            a1, c1 = one[r].step_obs(obs[r:r + 1], mask[r:r + 1])
            assert np.array_equal(act[r], a1[0]) and np.array_equal(cont[r], c1[0]), (t, r)


def test_the_actor_fills_only_its_rows():
    from cs2rl.experiment import oracle_statue as L0
    env, _ = _arena()
    try:
        obs, _ = env.reset()
        actor = DuelBotActor([L0.STATUE], DuelBotParams(style="run"), np.random.default_rng(0))
        moved = False
        for _ in range(12):
            act, cont = actor.act(obs, None, None, env)
            others = np.arange(len(obs)) != L0.STATUE
            assert not act[others].any() and not cont[others].any()
            moved |= bool(act[L0.STATUE].any())
            obs, *_ = env.step(act, cont)
        assert moved
    finally:
        env.close()


def test_importing_the_module_pulls_no_heavy_module():
    from tests.train.test_w1_modules import HEAVY, _run_child
    r = _run_child(f"""
import cs2rl.eval.duel_bot
heavy = [m for m in {HEAVY!r} if m in sys.modules]
assert not heavy, f"cs2rl.eval.duel_bot's module scope pulled {{heavy}}"
""")
    assert r.returncode == 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
