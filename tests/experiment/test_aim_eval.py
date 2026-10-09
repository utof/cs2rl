"""cs2rl.experiment.aim_eval: a policy checkpoint against scripted opponents, misses decomposed.

WHAT IS PINNED HERE
-------------------
* ``HIT_HALF_WIDTH`` equals cs2_combat.h's.
* The decomposition's sign on hand-built geometry with a known answer: a crosshair that
  trails the target reads along > 0 whichever way the target moves, one that leads it reads
  along < 0, and a vertical error against horizontal motion is all across.
* The recorder: errors in half-window units, the hit flag from the opponent's hp, and the lag's
  tracking ticks starting at each round's first on-target tick.
* The summary on hand-set fires: hit/fired up to and after tick ``bptt_horizon``, the
  off-window share, the |e| quantiles and the along/across means; and in every played cell the
  gap to the oracle row of the same opponent, found by name, with the policy-worse sign.
* End to end on the walker: ObsOracleActor (aims where the target was) reads 1 tick of lag
  in every |w| quintile and a positive along; OracleActor (one-tick lead) reads under 0.3. The
  recounts equal the C counters, the killing shot included (``on_step`` runs on the terminal
  tick). The lag's error bar resamples whole episodes, and the lag curve splits by |w| quintile.
* The opponents are eval.walker's Rung 1b families (TRAIN_MIX and HELD_OUT) under names that
  say which: the plain ones draw as WalkerActor does (walker-4-16 is oracle_tracker's walker),
  and only the stop-and-go one stands still.
* ``PolicyActor(mean_aim=True)`` changes only the continuous aim: on the same inputs and
  seed it returns the sampled mode's discrete actions tick after tick, and cont = mu.
* ``LstmZeroedEvery`` zeroes before forward ``period``, ``2 * period``, ... counted across
  rounds, as ``Cs2PuffeRL.evaluate`` zeroes at every rollout start.
* ``env_mismatches`` flags a run trained on another map, laser range or turn speed.
* The CLI on the Rung 1a checkpoint, 5 episodes per cell. It needs the checkpoint, which only
  the owner's checkouts have (outputs/ is not in git), so it skips without one.
"""
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import REPO_ROOT

RUNG1A_CKPT = "outputs/checkpoints/rung1a/s0/rung1a-s0.pt"


def _state(xy, facing=0.0, z=(0.0, 0.0), hp=(100, 100), fired=0):
    """A two-agent snapshot (row 0 = hero, row 1 = opponent) with the keys the recorder reads."""
    return {
        "x": np.array([0.0, xy[0]]),
        "y": np.array([0.0, xy[1]]),
        "z": np.array(z, dtype=float),
        "facing": np.array([facing, 0.0]),
        "pitch": np.zeros(2),
        "is_crouching": np.zeros(2, int),
        "alive": np.array([1, int(hp[1] > 0)]),
        "hp": np.array(hp),
        "fired_this_tick": np.array([fired, 0]),
    }


def test_hit_half_width_mirrors_cs2_combat_h():
    from cs2rl.experiment.aim_eval import HIT_HALF_WIDTH
    header = (REPO_ROOT / "src" / "cs2rl" / "env" / "c" / "cs2_combat.h").read_text()
    found = re.findall(r"static const float\s+HIT_HALF_WIDTH\s*=\s*([0-9.]+)f", header)
    assert found == [str(HIT_HALF_WIDTH)], found


@pytest.mark.parametrize("dy, facing, along_sign", [(15.0, 0.0, +1), (-15.0, 0.0, +1),
                                                    (15.0, 0.1, -1)])
def test_along_is_positive_when_the_crosshair_trails(dy, facing, along_sign):
    from cs2rl.experiment.aim_eval import aim_geometry, decompose
    e, w, d = aim_geometry(_state((300.0, 0.0), facing), _state((300.0, dy), facing), 0, 1)
    assert w[0] == pytest.approx(math.atan2(dy, 300.0)) and d == pytest.approx(math.hypot(300, dy))
    split = decompose(e, w)
    assert split is not None
    along, across = split
    assert along == pytest.approx(along_sign * abs(math.atan2(dy, 300.0) - facing))
    assert across == pytest.approx(0.0, abs=1e-12)


def test_a_vertical_error_against_horizontal_motion_is_across():
    from cs2rl.experiment.aim_eval import aim_geometry, decompose
    on_target = math.atan2(15.0, 300.0)
    e, w, _ = aim_geometry(_state((300.0, 0.0), on_target, z=(0.0, 30.0)),
                           _state((300.0, 15.0), on_target, z=(0.0, 30.0)), 0, 1)
    split = decompose(e, w)
    assert split is not None
    along, across = split
    assert abs(along) < 1e-3 and across == pytest.approx(math.atan2(30.0, math.hypot(300, 15)),
                                                         rel=1e-3)
    assert decompose(e, (0.0, 0.0)) is None


def test_the_recorder_scales_by_the_half_window_and_tracks_from_acquisition():
    from cs2rl.experiment.aim_eval import MissRecorder
    rec = MissRecorder(0, 1)
    # Tick 1: the opening turn, 0.5 rad off; recorded as a fire, kept out of the lag ticks.
    rec(1, _state((300.0, 0.0), 0.5), _state((300.0, 5.0), 0.5, fired=1))
    # Ticks 2..12: the crosshair stays where the target was a tick ago (1 tick of lag).
    for k in range(2, 13):
        rec(k, _state((300.0, 5.0 * (k - 1)), math.atan2(5.0 * (k - 2), 300.0)),
            _state((300.0, 5.0 * k), math.atan2(5.0 * (k - 1), 300.0), hp=(100, 90), fired=k == 12))
    # A second round opens 0.5 rad off target: acquisition restarts, so none of it is tracked.
    for k in (1, 2):
        rec(k, _state((300.0, 5.0 * (k - 1)), 0.5), _state((300.0, 5.0 * k), 0.5))
    s = rec.summary(64)
    assert s["track_ticks"] == 11 and s["lag_ticks"] == pytest.approx(1.0, abs=0.02)
    assert s["fires"] == 2 and s["hit_recount"] == 1
    half = math.asin(16.0 / math.hypot(300.0, 5.0))
    assert rec.fires[0][1] == pytest.approx((0.5 - math.atan2(5.0, 300.0)) / half)


def test_the_summary_splits_fires_at_the_horizon_and_by_the_window():
    from cs2rl.experiment.aim_eval import MissRecorder
    rec = MissRecorder(0, 1)
    # (episode tick, |e_yaw|, along, across, hit) in half-windows; tick 64 is still early at 64.
    rec.fires = [(10, 0.5, 0.5, 0.0, True), (64, 2.0, 2.0, 0.0, False), (65, 0.2, -0.2, 0.0, True)]
    s = rec.summary(64)
    assert (s["hit_per_fired_early"], s["hit_per_fired_late"], s["late_fires"]) == (0.5, 1.0, 1)
    assert s["off_window_frac"] == pytest.approx(1 / 3) and s["on_target_recount"] == 2
    assert (s["abs_err_median"], s["abs_err_p90"]) == pytest.approx((0.5, 1.7))
    assert s["along_mean"] == pytest.approx(2.3 / 3) and s["across_mean"] == 0.0


def test_lag_reads_one_tick_without_lead_and_zero_with_it():
    from cs2rl.experiment import aim_eval
    from cs2rl.experiment.oracle_statue import run_check
    out = {}
    for hero, obs_only in aim_eval.REFERENCE_HEROES:
        rec = aim_eval.MissRecorder()
        res = run_check(20,
                        0,
                        obs_only=obs_only,
                        opponent=aim_eval.OPPONENTS["walker-4-16"](0),
                        on_step=rec)
        s = rec.summary(64)
        assert (s["fires"], s["on_target_recount"], s["hit_recount"]) == \
            (res["shots_fired"], res["shots_on_target"], res["shots_hit"]), hero
        out[hero] = s
    assert out["obs-oracle"]["lag_ticks"] == pytest.approx(1.0, abs=0.01), out["obs-oracle"]
    assert [r for _, r, _ in out["obs-oracle"]["lag_by_speed"]] == pytest.approx([1.0] * 5,
                                                                                 abs=0.01)
    assert out["obs-oracle"]["along_mean"] > 0.2
    assert abs(out["oracle"]["lag_ticks"]) < 0.3, out["oracle"]


def test_the_lag_error_bar_resamples_whole_episodes():
    from cs2rl.experiment.aim_eval import _episode_bootstrap_se
    # One long round tracking at 1 tick of lag, one single-tick round at 3. Resampling ticks
    # would almost never move the median off 1; resampling rounds moves it a quarter of the time.
    ratio = np.array([1.0] * 100 + [3.0])
    episode = np.array([0] * 100 + [1])
    assert _episode_bootstrap_se(ratio, episode) > 0.5


def test_the_lag_curve_splits_by_angular_speed_quintile():
    from cs2rl.experiment.aim_eval import _by_speed_quintile
    # A lag that grows with |w|: each quintile holds two ticks and reads its own median.
    speed = np.arange(1.0, 11.0)
    assert _by_speed_quintile(speed, speed) == [[k + 0.5, k + 0.5, 2] for k in (1, 3, 5, 7, 9)]


def test_mean_aim_changes_only_the_continuous_aim():
    import torch

    from cs2rl.eval.baselines import ACTION_DIM, AIM_DIM, N_AGENTS, PolicyActor
    from cs2rl.experiment.oracle_statue import build_env
    from cs2rl.policy import build_policy, init_policy_state
    env = build_env(0)
    try:
        torch.manual_seed(0)
        policy = build_policy(env, "cpu", aim_log_std_max=-2.9957, pin_pitch=True)
        obs_seq = [env.reset()[0]]
        for _ in range(11):
            idle = (np.zeros((N_AGENTS, ACTION_DIM),
                             np.int32), np.zeros((N_AGENTS, AIM_DIM), np.float32))
            obs_seq.append(env.step(*idle)[0])
        played = {}
        for mean_aim in (False, True):
            actor = PolicyActor(policy, "cpu", mean_aim=mean_aim)
            actor.reset()
            torch.manual_seed(5)
            played[mean_aim] = [actor.act(o, None, None, env) for o in obs_seq]
        state, mus = init_policy_state(policy, "cpu"), []
        with torch.no_grad():
            for o in obs_seq:
                mus.append(policy.forward_eval(torch.as_tensor(o), state)[1].numpy())
    finally:
        env.close()
    for (a_s, c_s), (a_m, c_m), mu in zip(played[False], played[True], mus, strict=True):
        assert (a_s == a_m).all()
        assert np.allclose(c_m, mu) and not np.allclose(c_s, mu)


def test_the_lstm_is_zeroed_every_period_forwards_across_rounds():
    from cs2rl.experiment.aim_eval import LstmZeroedEvery

    class Counting:

        def __init__(self):
            self.acts, self.resets = 0, []

        def reset(self):
            self.resets.append(self.acts)

        def act(self, *_):
            self.acts += 1

    inner = Counting()
    hero = LstmZeroedEvery(inner, 64)
    for length in (50, 30, 100):
        hero.reset()
        for _ in range(length):
            hero.act(None, None, None, None)
    # Round starts at forwards 0, 50, 80; zeroing before forwards 64 and 128.
    assert inner.resets == [0, 50, 64, 80, 128]


def test_run_check_plays_the_given_hero():
    from cs2rl.eval.baselines import IdleActor
    from cs2rl.experiment.oracle_statue import run_check
    assert run_check(5, 0)["kills"] == 5
    res = run_check(5, 0, hero=IdleActor())
    assert (res["kills"], res["shots_fired"]) == (0, 0)


def test_the_opponents_are_the_rung1b_walker_families():
    from cs2rl.eval.walker import HELD_OUT, STATUE, TRAIN_MIX
    from cs2rl.experiment.aim_eval import OPPONENTS
    # Each cell's name says its family: a retuned or reordered TRAIN_MIX must rename the cell.
    assert list(OPPONENTS) == ["statue", "walker-4-16", "walker-2-8-stop", "heldout-24-48"]
    assert TRAIN_MIX[0] == STATUE and STATUE.p_stop == (1.0, 1.0)
    assert (TRAIN_MIX[1].hold, TRAIN_MIX[1].p_stop, TRAIN_MIX[1].duty) == ((4, 16), (0, 0), (1, 1))
    assert TRAIN_MIX[2].hold == (2, 8) and TRAIN_MIX[2].p_stop[0] > 0
    assert (HELD_OUT.hold, HELD_OUT.p_stop, HELD_OUT.duty) == ((24, 48), (0, 0), (1, 1))


def _moves(actor, ticks=600, round_len=150):
    """The CT row's move bin per tick, a reset every ``round_len``; every other entry is 0."""
    from cs2rl.eval.walker import H_MOVE
    obs = np.zeros((10, 1), dtype=np.float32)
    out = []
    for t in range(ticks):
        if t % round_len == 0:
            actor.reset()
        act, cont = actor.act(obs, None, None, None)
        assert not act[[i for i in range(10) if i != 5]].any() and not cont.any()
        out.append(int(act[5, H_MOVE]))
    return np.array(out)


def test_the_walker_families_play_as_drawn():
    from cs2rl.eval.walker import WalkerActor
    from cs2rl.experiment.aim_eval import OPPONENTS
    from cs2rl.experiment.oracle_statue import STATUE
    from cs2rl.experiment.oracle_tracker import build_walker
    assert STATUE == 5
    # The plain families are WalkerActor's draws: walker-4-16 is oracle_tracker's walker.
    assert (_moves(OPPONENTS["walker-4-16"](3)) == _moves(build_walker(3))).all()
    held = WalkerActor(np.random.default_rng(3), [5], 24, 48)
    assert (_moves(OPPONENTS["heldout-24-48"](3)) == _moves(held)).all()
    # The stop-and-go family stands still on about a quarter of its holds, and only it does.
    stop = _moves(OPPONENTS["walker-2-8-stop"](3), ticks=4000)
    assert 0.15 < (stop == 0).mean() < 0.35, (stop == 0).mean()
    assert (_moves(OPPONENTS["walker-4-16"](3)) != 0).all()


def test_each_mode_builds_its_hero():
    import torch

    from cs2rl.eval.baselines import PolicyActor
    from cs2rl.experiment.aim_eval import MODES, LstmZeroedEvery, build_policy_hero

    class Policy:
        max_turn_speed = torch.tensor(0.785)

    for aim, lstm in MODES:
        hero = build_policy_hero(Policy(), aim, lstm, 64)
        if lstm == "zeroed":
            assert isinstance(hero, LstmZeroedEvery) and hero.period == 64
            hero = hero.actor
        assert isinstance(hero, PolicyActor) and hero.mean_aim == (aim == "mean")
    with pytest.raises(ValueError):
        build_policy_hero(Policy(), "argmax", "carried", 64)


def test_cells_reproduce_in_any_order():
    import torch

    from cs2rl.experiment import aim_eval
    from cs2rl.experiment.oracle_statue import build_env
    from cs2rl.policy import build_policy
    env = build_env(0)
    try:
        torch.manual_seed(0)
        policy = build_policy(env, "cpu", aim_log_std_max=-2.9957, pin_pitch=True)
    finally:
        env.close()
    walker = {"walker-4-16": aim_eval.OPPONENTS["walker-4-16"]}
    modes = (("sample", "carried"), ("mean", "zeroed"))

    def cells(m, caller_seed):
        torch.manual_seed(caller_seed)                 # the caller's stream must not matter either
        out = aim_eval.run_eval(policy, 64, episodes=2, seed=3, opponents=walker, modes=m)

        # Every gap is nonzero, else a flipped sign reads the same. KNOWN LIMIT: on these 2
        # rounds the oracle and obs-oracle rows read the same, so a run_eval that took the
        # obs-oracle row as its oracle would pass this check too.
        for c in out["cells"]:
            oracle = next(r for r in out["references"]
                          if (r["opponent"], r["hero"]) == (c["opponent"], "oracle"))
            gap = c["gap_to_oracle"]
            assert all(v != 0 for v in gap.values()), gap
            assert gap == {
                "kill_rate": oracle["kill_rate"] - c["kill_rate"],
                "ttk_median": c["ttk_median"] - oracle["ttk_median"],
                "hit_per_fired": oracle["hit_per_fired"] - c["hit_per_fired"],
            }
        return {(c["aim"], c["lstm"]): c["per_episode"] for c in out["cells"]}

    assert cells(modes, 1) == cells(modes[::-1], 2)


def test_a_run_on_another_map_or_turn_speed_is_flagged():
    from cs2rl.experiment.aim_eval import env_mismatches
    flagged = env_mismatches({"env": "cs2-dust2", "laser_range": 1500.0, "max_turn_speed": 0.5})
    assert {
        "env: run cs2-dust2 vs harness cs2-arena-duel", "laser_range: run 1500.0 vs harness None",
        "max_turn_speed: run 0.5 vs harness None"
    } <= set(flagged), flagged


def _rung1a_checkpoint():
    """This checkout's Rung 1a checkpoint, else the main checkout's (a worktree has no outputs/)."""
    common = subprocess.run(
        ["git", "-C",
         str(REPO_ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False)
    roots = [REPO_ROOT] + ([Path(common.stdout.strip()).parent] if common.returncode == 0 else [])
    for root in roots:
        if (root / RUNG1A_CKPT).is_file():
            return root / RUNG1A_CKPT
    pytest.skip(f"no {RUNG1A_CKPT} under {[str(r) for r in roots]}")


@pytest.mark.slow
def test_the_cli_plays_every_cell_on_the_rung1a_checkpoint(tmp_path, capsys):
    from cs2rl.experiment import aim_eval
    ckpt = _rung1a_checkpoint()
    out_path = tmp_path / "aim_eval.json"
    assert aim_eval.main([str(ckpt), "--episodes", "5", "--json", str(out_path)]) == 0
    out = json.loads(out_path.read_text())
    assert out["env_mismatches"] == [] and out["bptt_horizon"] == 64
    assert len(out["cells"]) == len(aim_eval.OPPONENTS) * len(aim_eval.MODES)
    assert len(out["references"]) == len(aim_eval.OPPONENTS) * len(aim_eval.REFERENCE_HEROES)
    for r in out["cells"] + out["references"]:
        assert r["episodes"] == 5 and len(r["per_episode"]) == 5
        assert (r["on_target_recount"], r["hit_recount"]) == (r["shots_on_target"],
                                                              r["shots_hit"]), r
    assert "WARNING" not in capsys.readouterr().out
