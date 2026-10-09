"""#152 L1: the oracle against the random walker (cs2rl.experiment.oracle_tracker).

WHAT IS PINNED HERE
-------------------
* The floor, at the observed rate: the ground-truth oracle kills the walker in all
  200 episodes of seed 0 (2026-10-09). A miss fails with its episode and spawn, so it
  can be replayed (same seed, ``--episodes k+1``) and then explained or filed. The
  TTK is pinned at its observed median 10 / p90 13 too, so a sim change that slows
  the seed-0 kill past either value fails here.
* The motion checks have teeth. A statue (``IdleActor``) and a walker whose move bins
  are forced to 0 both FAIL the L1 verdict while passing every L0 check, because kills
  cannot tell a moving target from a still one. A walker that redraws its direction
  every tick fails the displacement floor.
* ``--obs-only`` matches ground truth episode for episode on the walker (TTK, shots
  fired, shots hit), with no blind tick.
* The premise of the docstring's "does not certify lead": switching OracleActor's lead
  off changes no episode. If a faster walker ever makes the lead matter, this goes red
  and that paragraph has to be rewritten.
* The per-episode table logs seed, episode and spawn, and a rerun reproduces it.
* ``main``'s exit code follows the verdict.
"""
import numpy as np
import pytest

L1_MOTION_CHECKS = ["walker moving_frac >= 0.90", "walker net_disp median >= 50 u"]


def _failing(checks):
    return [name for name, ok, _ in checks if not ok]


def test_the_oracle_kills_the_walker_in_every_episode():
    from cs2rl.experiment.oracle_tracker import run_tracker, verdict
    res = run_tracker(episodes=200, seed=0)
    missed = [p for p in res["per_episode"] if p["ttk"] is None]
    assert not missed, f"seed 0 misses (replay with --episodes ep+1): {missed}"
    passed, checks = verdict(res)
    assert passed, checks
    assert res["ttk_median"] < 120 and res["unmatched_vis_slots"] == 0
    # Observed 2026-10-09 at seed 0: median 10, p90 13 ticks. Re-pin on a deliberate sim change.
    assert res["ttk_median"] <= 10 and res["ttk_p90"] <= 13, (res["ttk_median"], res["ttk_p90"])


@pytest.mark.parametrize("opponent", ["statue", "move bins forced to 0"])
def test_a_target_that_does_not_move_fails_l1_and_passes_l0(opponent, monkeypatch):
    import cs2rl.eval.walker as walker
    from cs2rl.eval.baselines import IdleActor
    from cs2rl.experiment import oracle_statue
    from cs2rl.experiment.oracle_tracker import build_walker, run_tracker, verdict
    if opponent == "statue":
        target = IdleActor()
    else:
        monkeypatch.setattr(walker, "MOVE_BINS", np.array([0], dtype=np.int32))
        target = build_walker(0)
    res = run_tracker(episodes=20, seed=0, opponent=target)
    assert res["kills"] == 20
    assert oracle_statue.verdict(res)[0]
    assert (res["opp_moving_frac"], res["opp_net_disp_median"]) == (0.0, 0.0)
    passed, checks = verdict(res)
    assert not passed and _failing(checks) == L1_MOTION_CHECKS


def test_a_walker_that_redraws_every_tick_fails_the_displacement_floor():
    from cs2rl.eval.walker import WalkerActor
    from cs2rl.experiment.oracle_statue import STATUE
    from cs2rl.experiment.oracle_tracker import run_tracker, verdict
    jitter = WalkerActor(np.random.default_rng(0), [STATUE], hold_min=1, hold_max=1)
    res = run_tracker(episodes=50, seed=0, opponent=jitter)
    assert res["opp_moving_frac"] >= 0.90
    assert _failing(verdict(res)[1]) == [L1_MOTION_CHECKS[1]]


def test_obs_only_matches_ground_truth_in_every_episode():
    from cs2rl.experiment.oracle_tracker import run_tracker, verdict
    truth = run_tracker(episodes=200, seed=0)
    obs = run_tracker(episodes=200, seed=0, obs_only=True)
    assert obs["obs_blind_ticks"] == 0
    keys = ("ttk", "shots_fired", "shots_hit")
    assert [[p[k] for k in keys] for p in obs["per_episode"]] == \
        [[p[k] for k in keys] for p in truth["per_episode"]]
    assert verdict(obs)[0], verdict(obs)[1]


def test_switching_the_oracle_lead_off_changes_no_episode(monkeypatch):
    import cs2rl.eval.baselines as baselines
    from cs2rl.experiment.oracle_tracker import run_tracker
    led = run_tracker(episodes=200, seed=0)
    monkeypatch.setattr(baselines, "TICK_DT", 0.0)
    unled = run_tracker(episodes=200, seed=0)
    assert unled["per_episode"] == led["per_episode"]


def test_the_table_logs_seed_episode_and_spawn_and_a_rerun_reproduces_it():
    from cs2rl.experiment.oracle_tracker import episode_rows, format_report, run_tracker
    res = run_tracker(episodes=5, seed=3)
    rows = episode_rows(res)
    assert rows[0].split()[:3] == ["seed", "ep", "hero"]
    assert len(rows) == 6
    for ep, row in enumerate(rows[1:]):
        assert row.split()[:2] == ["3", str(ep)]
        p = res["per_episode"][ep]
        assert f"{p['spawn_dist']:6.1f}" in row and f"{p['statue_xy'][0]:5.0f}" in row
    assert episode_rows(run_tracker(episodes=5, seed=3)) == rows
    assert "\n".join(rows) in format_report(res)


def test_main_exit_code_follows_the_verdict(capsys, monkeypatch):
    from cs2rl.eval.baselines import IdleActor
    from cs2rl.experiment import oracle_tracker
    assert oracle_tracker.main(["--episodes", "20"]) == 0
    assert capsys.readouterr().out.rstrip().splitlines()[-1].startswith("PASS")
    monkeypatch.setattr(oracle_tracker, "build_walker", lambda seed: IdleActor())
    assert oracle_tracker.main(["--episodes", "20", "--obs-only"]) == 1
    out = capsys.readouterr().out
    assert "OBS-ONLY hero" in out and out.rstrip().splitlines()[-1].startswith("FAIL")
