"""cs2rl.experiment.aim_eval: a policy checkpoint against scripted opponents, misses decomposed.

WHAT IS PINNED HERE
-------------------
* ``HIT_HALF_WIDTH`` equals cs2_combat.h's.
* The decomposition's sign on hand-built geometry with a known answer: a crosshair that
  trails the target reads along > 0 whichever way the target moves, one that leads it reads
  along < 0, and a vertical error against horizontal motion is all across.
* The recorder: errors in half-window units, the hit flag from the opponent's hp, and the lag
  fit starting at the round's first on-target tick.
* End to end on the walker: ObsOracleActor (aims where the target was) reads 1 tick of lag
  and a positive along; OracleActor (one-tick lead) reads about 0. The recounts equal the C
  counters, the killing shot included (``on_step`` runs on the terminal tick).
* ``PolicyActor(mean_aim=True)`` changes only the continuous aim: on the same inputs and
  seed it returns the sampled mode's discrete actions tick after tick, and cont = mu.
* ``LstmZeroedEvery`` zeroes before forward ``period``, ``2 * period``, ... counted across
  rounds, as ``Cs2PuffeRL.evaluate`` zeroes at every rollout start.
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


def test_the_recorder_scales_by_the_half_window_and_fits_from_acquisition():
    from cs2rl.experiment.aim_eval import MissRecorder
    rec = MissRecorder(0, 1)
    # Tick 1: the opening turn, 0.5 rad off; recorded as a fire, kept out of the lag fit.
    rec(1, _state((300.0, 0.0), 0.5), _state((300.0, 5.0), 0.5, fired=1))
    # Ticks 2..12: the crosshair stays where the target was a tick ago (1 tick of lag).
    for k in range(2, 13):
        rec(k, _state((300.0, 5.0 * (k - 1)), math.atan2(5.0 * (k - 2), 300.0)),
            _state((300.0, 5.0 * k), math.atan2(5.0 * (k - 1), 300.0), hp=(100, 90), fired=k == 12))
    s = rec.summary(64)
    assert s["track_ticks"] == 11 and s["lag_ticks"] == pytest.approx(1.0, abs=0.02)
    assert s["fires"] == 2 and s["hit_recount"] == 1
    half = math.asin(16.0 / math.hypot(300.0, 5.0))
    assert rec.fires[0][1] == pytest.approx((0.5 - math.atan2(5.0, 300.0)) / half)


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
    assert out["obs-oracle"]["lag_ticks"] == pytest.approx(1.0, abs=0.05), out["obs-oracle"]
    assert out["obs-oracle"]["along_mean"] > 0.2
    assert abs(out["oracle"]["lag_ticks"]) < 0.15, out["oracle"]


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
        return {(c["aim"], c["lstm"]): c["per_episode"] for c in out["cells"]}

    assert cells(modes, 1) == cells(modes[::-1], 2)


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
