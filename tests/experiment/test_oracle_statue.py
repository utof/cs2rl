"""Rung 1a Task 0: the oracle-vs-statue solvability precondition, in-process.

WHAT IS PINNED HERE
-------------------
* The scripted oracle kills a stationary CT on the exact Rung 1a env
  (arena-duel, n_active=1, round_time=160, pin_pitch=1, crouch=0, jump=0) in a small
  fraction of the round. If this ever goes red, no Rung 1a §3 learning verdict
  measured on that env means anything — read the sim, not the policy.
* The ``--statue-z`` hold is LIVE. This needs saying because at an offset of
  24 u every printed number is bit-identical to the grounded run: the shots all
  connect either way and the RNG stream is unchanged, so "kill rate 1.0 at
  statue_z=24" is on its own indistinguishable from the offset being silently
  dropped. Two things give it teeth — the measured ``rz_min``/``rz_max`` (the
  |rz| cs2_combat.h actually computed), and the negative control at 57 u, which
  is outside ``HIT_HALF_HEIGHT_STAND`` and must produce zero hits.
* The gate arithmetic in ``verdict`` and its wiring to the process exit code.
* ``--obs-only``: the same kill, driven off the OBSERVATION VECTOR instead of
  the C state. The ground-truth oracle would pass this gate 200/200 with the
  enemy block of the obs encoded wrongly, because it never reads it — that bug
  class makes a solvable env unlearnable, and this mode is what sees it. The
  two modes must agree exactly: both see the obs ``env.reset`` returns (real
  since #157), and against a statue the ground-truth lead is zero.
* The FAIL report's diagnostics: censored TTK printed as "n/a (no kills)"
  rather than "None", and the failing episodes' spawn geometry.

WHY the thresholds here are looser than the script's (0.8 vs 0.90 kill rate):
this is the fast regression tripwire at N=20, not the gate. The gate is
``UV_NO_SYNC=1 uv run python -m cs2rl.experiment.oracle_statue`` at N>=200.
"""
import pytest

# Offset that reproduces a crouched target's |rz| against a STANDING hitbox
# (TORSO_OFFSET_CROUCH 24 vs EYE_HEIGHT_STAND 48), and one that clears the
# standing vertical semi-axis (HIT_HALF_HEIGHT_STAND = 36) outright.
CROUCH_HEIGHT_OFFSET = 24.0
ABOVE_SEMI_AXIS_OFFSET = 57.0
# cs2_movement.h leapfrog: one half gravity step lands between the hold and
# process_combat, so the ray sees `offset - 1.5625`.
GRAVITY_SAG = 0.5 * 800.0 * (1.0 / 16.0)**2


def test_build_env_runs_the_smoke_knobs():
    """The harness env is the Rung 1a smoke's env, read back from the live C StaticData.

    jump_enabled is the knob this preset once inherited from EnvConfig's default
    (1) while the smoke ran 0. The scripted actors never jump, so no kill or TTK
    number shows the drift; only a policy evaluated here does.
    """
    from cs2rl.experiment.oracle_statue import ROUND_TIME, build_env
    env = build_env(0)
    try:
        sd = env._c_env.sd.contents
        assert (sd.n_active_per_team, sd.pin_pitch, sd.crouch_enabled, sd.jump_enabled,
                sd.round_time) == (1, 1, 0, 0, ROUND_TIME)
        assert env._auto_reset is False
    finally:
        env.close()


def test_oracle_kills_grounded_statue():
    """The precondition itself: perfect aim vs a target that never moves."""
    from cs2rl.experiment.oracle_statue import run_check, verdict
    res = run_check(episodes=20, seed=0)
    assert res["kill_rate"] >= 0.8, res
    assert res["ttk_min"] is not None and res["ttk_min"] < 160, res
    # A blind oracle would still be able to win a 100 %-LoS arena by walking into
    # the enemy, so the win alone does not prove the vis_prev thread survived;
    # an unmatched slot means vis_from_obs could not tie a visible obs slot to an
    # agent, which is that thread breaking.
    assert res["unmatched_vis_slots"] == 0, res
    assert res["shots_fired"] > 0 and res["shots_stance_blocked"] == 0, res
    assert verdict(res)[0], res
    # Ground-truth mode never reads an obs slot, so it must not report obs
    # diagnostics at all — "0 blind ticks" here would be a claim it cannot make.
    assert res["obs_only"] is False and res["obs_blind_ticks"] is None, res
    assert res["obs_inconsistent_slots"] is None, res


def test_oracle_kills_statue_held_at_crouch_height_offset():
    """v1c ellipsoid, in-env: a standing target displaced ~22 u vertically.

    Asserts exactly two things and no more: the hold really placed the target at
    that |rz| (``rz_min``/``rz_max``, measured off the live C state), and the
    oracle still kills it with ``shots_stance_blocked == 0``. It does NOT assert
    anything about ``HIT_HALF_HEIGHT_CROUCH``: the statue is standing, so the
    gate used HH = 36. crouch_enabled=0 masks the crouch head, so a genuinely
    crouched target is not reachable through the Rung 1a action space.
    """
    from cs2rl.experiment.oracle_statue import run_check, verdict
    res = run_check(episodes=20, seed=0, statue_z=CROUCH_HEIGHT_OFFSET)
    expect_rz = CROUCH_HEIGHT_OFFSET - GRAVITY_SAG
    assert res["rz_min"] == pytest.approx(expect_rz, abs=0.05), res
    assert res["rz_max"] == pytest.approx(expect_rz, abs=0.05), res
    assert res["kill_rate"] >= 0.8, res
    assert res["shots_stance_blocked"] == 0, res
    assert verdict(res)[0], res


def test_statue_above_the_standing_semi_axis_is_unkillable():
    """Negative control — the teeth for the test above.

    At 57 u the target centre sits ~55 u above the shooter's eye, past
    HIT_HALF_HEIGHT_STAND = 36, so the ellipsoid gate rejects every shot no
    matter how perfect the yaw. Every shot must be stance-blocked AND a miss,
    and every shot but one kind must be on target (the yaw window is a 2D
    quantity, unaffected by z). If the hold were a no-op this test would look
    exactly like the grounded run and fail on the first assert, which is what
    makes the 24 u case meaningful.

    The one kind of off-target shot is the opening shot on a spawn row whose
    yaw error is wider than max_turn_speed (56.3 deg vs 45 on the arena).
    ``vis_prev`` comes from the real reset obs (#157), so the oracle fires on
    tick 1, and its fire rule does not wait for the turn to land. Exactly one
    such shot per wide-spawn episode, measured.

    Few episodes on purpose: nothing dies, so each one runs the full 160 ticks.
    """
    from cs2rl.env.nav import MAX_TURN_SPEED_RAD
    from cs2rl.experiment.oracle_statue import run_check, verdict
    res = run_check(episodes=4, seed=0, statue_z=ABOVE_SEMI_AXIS_OFFSET)
    assert res["rz_min"] == pytest.approx(ABOVE_SEMI_AXIS_OFFSET - GRAVITY_SAG, abs=0.05), res
    assert res["kills"] == 0 and res["ttk_min"] is None, res
    assert res["shots_fired"] > 0, res
    # Nothing died, so every episode is in `failures`, with its spawn yaw error.
    wide = sum(abs(f["yaw_err"]) > MAX_TURN_SPEED_RAD for f in res["failures"])
    assert len(res["failures"]) == res["episodes"], res
    assert res["shots_fired"] - res["shots_on_target"] == wide, (wide, res)
    assert res["shots_stance_blocked"] == res["shots_fired"], res
    assert res["shots_hit"] == 0, res
    assert not verdict(res)[0], res


def test_obs_only_oracle_kills_the_statue_it_can_only_see_in_the_obs():
    """The obs-encoding half of the precondition (``--obs-only``).

    The actor's ONLY source of enemy geometry is the hero's observation row, so
    this failing while the ground-truth mode passes means the enemy block of
    ``cs2_observations.h`` is wrong (rotated by the wrong sign, normalised by
    the wrong constant) — a bug that leaves the env solvable by actions and
    unlearnable by a policy. NOT the visibility gate: ``can_see`` and ``alive``
    are equal on every tick of this env, so reading the wrong one of the two is
    undetectable here (see the script's "DOES NOT CERTIFY" list).

    ``obs_blind_ticks == 0``: ``env.reset`` returns the spawn-state obs (#157)
    and both agents are in permanent 2D LoS in the arena, so the hero sees the
    statue on every tick. A blind tick means visibility dropped mid-round (the
    kill numbers below would then be a statement about LoS, not encoding) or
    the reset obs went back to zero (before #157 this pinned one per episode).
    """
    from cs2rl.experiment.oracle_statue import run_check, verdict
    res = run_check(episodes=20, seed=0, obs_only=True)
    assert res["kill_rate"] >= 0.8, res
    assert res["ttk_min"] is not None and res["ttk_min"] < 160, res
    assert res["shots_fired"] > 0 and res["shots_stance_blocked"] == 0, res
    assert res["obs_blind_ticks"] == 0, res
    # The slot encodes the relative position twice ((rx, ry) vs bearing +
    # distance); disagreement is an encoding bug the kill rate cannot see,
    # because the actor steers by only one of the two.
    assert res["obs_inconsistent_slots"] == 0, res
    assert verdict(res)[0], res


def test_obs_only_decodes_the_enemy_z_delta_the_c_state_reports():
    """Slot +2, checked against the truth — because no kill count can check it.

    ``pin_pitch=1`` makes the env ignore the pitch this feeds, and the range
    test never binds at laser_range 3000 on a 360 u arena, so the obs-only run
    would kill 200/200 with the enemy z-delta scaled by the wrong constant or
    dropped entirely. The elevated statue puts a known, non-zero offset in that
    slot (24 u minus one half gravity step) and asserts the actor read it back.

    The low end is 0: the hold runs before each step, so the reset obs (which
    the actor reads on tick 1, #157) shows the statue still on the ground. The
    C side of the comparison is the rz of the same observed states, so it
    carries that 0 too; the realised rz at combat never does.
    """
    from cs2rl.experiment.oracle_statue import OBS_RZ_TOL, _rz_disagrees, run_check
    res = run_check(episodes=4, seed=0, statue_z=CROUCH_HEIGHT_OFFSET, obs_only=True)
    expect_rz = CROUCH_HEIGHT_OFFSET - GRAVITY_SAG
    assert res["obs_rz_max"] == pytest.approx(expect_rz, abs=0.05), res
    assert res["obs_rz_min"] == pytest.approx(0.0, abs=0.05), res
    assert res["observed_rz_min"] == pytest.approx(0.0, abs=0.05), res
    assert res["observed_rz_max"] == pytest.approx(expect_rz, abs=0.05), res
    assert res["rz_min"] == pytest.approx(expect_rz, abs=0.05), res
    # ... and agrees with the same quantity read off the live C state.
    assert not _rz_disagrees(res), res
    # The agreement check must have teeth: a decode off by more than the
    # tolerance has to trip it.
    assert _rz_disagrees({**res, "obs_rz_min": res["obs_rz_min"] + 2 * OBS_RZ_TOL})


def test_obs_only_and_ground_truth_modes_agree():
    """Cross-mode tie: the obs must say the same thing the C state says.

    Both modes run the same env, seed, statue and loop, so a divergence is the
    encoding — that is the whole design. No tolerance: both see the obs
    ``env.reset`` returns (#157, with ``vis_prev`` threaded from it), and the
    ground-truth lead is zero against a statue, so the two modes take the same
    action on every tick (measured over 200 episodes: same TTK and same shots
    in every one). Before #157 the obs actor lost tick 1 to an all-zero reset
    obs, and this test had to allow that gap.
    """
    from cs2rl.experiment.oracle_statue import run_check
    truth = run_check(episodes=20, seed=0)
    obs = run_check(episodes=20, seed=0, obs_only=True)
    for k in ("kills", "ttk_median", "ttk_p90", "ttk_min", "shots_fired", "shots_hit",
              "shots_on_target"):
        assert obs[k] == truth[k], (k, truth, obs)
    # Both modes fire only at an enemy the sim says is visible and in range, so
    # neither may waste a shot with no line of sight.
    assert obs["shots_with_enemy_in_los"] == obs["shots_fired"], obs


def test_verdict_is_the_conjunction_of_all_three_checks():
    """Pure-function gate arithmetic: each check must be able to fail alone."""
    from cs2rl.experiment.oracle_statue import PASS_MAX_MEDIAN_TTK, PASS_MIN_KILL_RATE, verdict
    ok = {"kill_rate": 1.0, "ttk_median": 10.0, "shots_stance_blocked": 0}
    passed, checks = verdict(ok)
    assert passed and len(checks) == 3
    for bad in ({
            "kill_rate": PASS_MIN_KILL_RATE - 0.01
    }, {
            "ttk_median": PASS_MAX_MEDIAN_TTK
    }, {
            "shots_stance_blocked": 1
    }):
        assert not verdict({**ok, **bad})[0], bad
    # Boundaries are inclusive/exclusive exactly as the brief words them.
    assert verdict({**ok, "kill_rate": PASS_MIN_KILL_RATE})[0]
    assert verdict({**ok, "ttk_median": PASS_MAX_MEDIAN_TTK - 0.5})[0]


def test_obs_only_verdict_gates_on_each_obs_tripwire():
    """r1-I1: in ``--obs-only`` the three obs diagnostics ARE gate checks.

    They used to be report-only ``<-- WARNING`` markers, so the script printed
    the bug and still exited 0 — and the documented CLI contract is "exit 0 =
    PASS", which is what a controller scripts against. Pure-function here; the
    two mutations that motivated the fix are driven end to end below.
    """
    from cs2rl.experiment.oracle_statue import OBS_RZ_TOL, verdict
    ok = {
        "kill_rate": 1.0,
        "ttk_median": 11.0,
        "shots_stance_blocked": 0,
        "obs_only": True,
        "episodes": 20,
        "obs_inconsistent_slots": 0,
        "obs_blind_ticks": 0,
        "obs_rz_min": 22.44,
        "obs_rz_max": 22.44,
        "observed_rz_min": 22.44,
        "observed_rz_max": 22.44,
    }
    passed, checks = verdict(ok)
    assert passed and len(checks) == 6, checks
    # EN_DIST pointed one slot over: kills stay 20/20, only this check moves.
    assert not verdict({**ok, "obs_inconsistent_slots": 131})[0]
    # OBS_Z_SCALE halved: the decoded rz is half the measured one.
    assert not verdict({**ok, "obs_rz_min": 11.22, "obs_rz_max": 11.22})[0]
    assert not verdict({**ok, "obs_rz_max": 22.44 + 2 * OBS_RZ_TOL})[0]
    # ...but the tolerance is a real band, not a float-equality test.
    assert verdict({**ok, "obs_rz_min": 22.44 + 0.9 * OBS_RZ_TOL})[0]
    # Any blind tick fails: the reset obs is real (#157), so there is no
    # structural one any more. 1 is sight lost once; 20 is the pre-#157 zero
    # reset obs coming back (one blind tick per episode).
    assert not verdict({**ok, "obs_blind_ticks": 1})[0]
    assert not verdict({**ok, "obs_blind_ticks": 20})[0]
    # The C side of the rz check is the OBSERVED states, not the combat rz.
    assert not verdict({**ok, "observed_rz_min": 0.0})[0]
    assert verdict({**ok, "rz_min": 0.0})[0]


def test_ground_truth_verdict_ignores_the_obs_diagnostics():
    """The obs checks must not leak into the default mode's verdict.

    Both dict shapes have to work: the partial one a unit test hands the pure
    function (no obs keys at all), and the real ground-truth summary, which
    carries every obs key as None because that actor never read a slot and so
    cannot claim "0 inconsistent". Neither may grow a fourth check.
    """
    from cs2rl.experiment.oracle_statue import verdict
    ok = {"kill_rate": 1.0, "ttk_median": 10.0, "shots_stance_blocked": 0}
    passed, checks = verdict(ok)
    assert passed and len(checks) == 3, checks
    full = {
        **ok,
        "obs_only": False,
        "episodes": 20,
        "obs_inconsistent_slots": None,
        "obs_blind_ticks": None,
        "obs_rz_min": None,
        "obs_rz_max": None,
        "rz_min": 0.0,
        "rz_max": 0.0,
    }
    passed, checks = verdict(full)
    assert passed and len(checks) == 3, checks
    # Values that would fail the gate in obs-only mode are inert here.
    poisoned = {
        **full,
        "obs_inconsistent_slots": 999,
        "obs_blind_ticks": 10**6,
        "obs_rz_min": -50.0,
        "obs_rz_max": -50.0,
    }
    assert verdict(poisoned)[0], poisoned


def test_a_mutated_enemy_distance_slot_fails_the_obs_only_exit_code(monkeypatch):
    """End-to-end teeth for r1-I1, mutation 1: ``EN_DIST`` pointed at slot +2.

    Reading the z-delta as the distance is invisible to every kill-based check —
    ``laser_range`` 3000 never binds on a 360 u arena, so the range test passes
    on any garbage distance and the run still kills 2/2. Before the fix this
    printed "obs slot inconsistency 20  <-- WARNING" and exited 0.
    """
    from cs2rl.experiment import oracle_statue as mod
    argv = ["--episodes", "2", "--seed", "0", "--obs-only"]
    assert mod.main(argv) == 0
    monkeypatch.setattr(mod, "EN_DIST", mod.EN_DZ)
    res = mod.run_check(episodes=2, seed=0, obs_only=True)
    # The mutation must stay invisible to the kill checks, or this test would
    # pass for the wrong reason.
    assert res["kill_rate"] == 1.0 and res["obs_inconsistent_slots"] > 0, res
    passed, checks = mod.verdict(res)
    failed = [n for n, ok, _ in checks if not ok]
    assert not passed and failed == ["obs_inconsistent_slots == 0"], checks
    assert mod.main(argv) == 1


def test_a_mutated_z_normaliser_fails_the_obs_only_exit_code(monkeypatch):
    """End-to-end teeth for r1-I1, mutation 2: ``OBS_Z_SCALE`` halved.

    ``pin_pitch=1`` makes the decoded height inert in the action path, so this
    one does not even move the slot-consistency counter (which never touches
    +2): the rz cross-check against the C state is the only thing in the run
    that sees it. Needs the elevated statue — at rz 0 both scalings decode 0.
    """
    from cs2rl.experiment import oracle_statue as mod
    argv = ["--episodes", "2", "--seed", "0", "--statue-z", str(CROUCH_HEIGHT_OFFSET), "--obs-only"]
    assert mod.main(argv) == 0
    monkeypatch.setattr(mod, "OBS_Z_SCALE", mod.OBS_Z_SCALE / 2)
    res = mod.run_check(episodes=2, seed=0, statue_z=CROUCH_HEIGHT_OFFSET, obs_only=True)
    expect_rz = CROUCH_HEIGHT_OFFSET - GRAVITY_SAG
    assert res["kill_rate"] == 1.0 and res["obs_inconsistent_slots"] == 0, res
    assert res["observed_rz_max"] == pytest.approx(expect_rz, abs=0.05), res
    assert res["obs_rz_max"] == pytest.approx(expect_rz / 2, abs=0.05), res
    passed, checks = mod.verdict(res)
    failed = [n for n, ok, _ in checks if not ok]
    assert not passed and failed == ["|obs_rz - C rz| <= 0.5"], checks
    assert mod.main(argv) == 1


def test_a_zeroed_reset_obs_fails_the_obs_only_exit_code(monkeypatch):
    """End-to-end teeth for the #157 bound: a reset obs that is all zero again.

    Before #157 env_reset left the obs buffer zeroed, and the obs-only hero was
    blind on tick 1 of every round. Kills stay 2/2 that way (it is a one-tick
    delay), so only ``obs_blind_ticks == 0`` can see the regression. Simulated
    here by zeroing what ``Cs2Env.reset`` returns.
    """
    import numpy as np

    from cs2rl.env.c import cs2_env
    from cs2rl.experiment import oracle_statue as mod
    argv = ["--episodes", "2", "--seed", "0", "--obs-only"]
    assert mod.main(argv) == 0
    real_reset = cs2_env.Cs2Env.reset

    def zeroed_reset(self, seed=None):
        obs, infos = real_reset(self, seed)
        return np.zeros_like(obs), infos

    monkeypatch.setattr(cs2_env.Cs2Env, "reset", zeroed_reset)
    res = mod.run_check(episodes=2, seed=0, obs_only=True)
    assert res["kill_rate"] == 1.0 and res["obs_blind_ticks"] == 2, res
    passed, checks = mod.verdict(res)
    failed = [n for n, ok, _ in checks if not ok]
    assert not passed and failed == ["obs_blind_ticks == 0"], checks
    assert mod.main(argv) == 1


def test_main_exit_code_follows_the_verdict(capsys):
    """The CLI contract the controller scripts against: 0 on PASS, 1 on FAIL."""
    from cs2rl.experiment.oracle_statue import main
    assert main(["--episodes", "4", "--seed", "0"]) == 0
    assert "PASS" in capsys.readouterr().out
    assert main(["--episodes", "2", "--seed", "0", "--statue-z", str(ABOVE_SEMI_AXIS_OFFSET)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_obs_only_is_reachable_from_the_cli(capsys):
    """``--obs-only`` is wired to the actor swap, and the report says so.

    Cheap on purpose (2 episodes): the numbers are pinned by the run_check
    tests above; what is pinned HERE is that the flag is not silently ignored,
    which would make an obs-encoding bug read as a pass.
    """
    from cs2rl.experiment.oracle_statue import main
    assert main(["--episodes", "2", "--seed", "0", "--obs-only"]) == 0
    out = capsys.readouterr().out
    assert "OBSERVATION VECTOR" in out and "obs blind ticks" in out
    assert "PASS" in out


def test_fail_report_names_the_facing_denominator_and_the_censored_ttk():
    """The three FAIL-path reporting fixes, on the run that produces them.

    * shots_facing_enemy is PRINTED (M1). It is collected from the C stats and
      is the denominator of the Rung 1 §5 aim criterion hit/facing > 0.45; a
      report that omits it cannot be used to read that criterion.
    * a zero-kill run says "n/a (no kills)", not "None" (M3) — a bare None
      reads as a crashed statistic rather than an empty one.
    * the failing episodes' indices and spawn geometry are printed (M2-lite),
      which is what separates "one bad spawn row" from "nothing can die".
    """
    from cs2rl.experiment.oracle_statue import format_summary, run_check
    res = run_check(episodes=1, seed=0, statue_z=ABOVE_SEMI_AXIS_OFFSET)
    out = format_summary(res)
    assert res["kills"] == 0 and res["shots_facing_enemy"] > 0, res
    assert f"shots_facing_enemy       {res['shots_facing_enemy']}" in out, out
    assert "n/a (no kills)" in out, out
    assert "None" not in out, out
    assert "failing episodes (no kill)  1 of 1" in out, out
    assert "ep    0" in out and "yaw err" in out, out
    # hit/facing is the §5 ratio, so it must divide by FACING, not by fired.
    doctored = {**res, "shots_facing_enemy": 100, "shots_hit": 45, "shots_fired": 1000}
    assert "hit/facing 0.450" in format_summary(doctored), doctored


def test_failure_detail_is_capped_but_the_spread_is_always_reported():
    """A total wipeout must not bury the verdict under one line per episode.

    Pure formatting, no env: the point is the cap and the aggregate quantiles
    that survive it — with 200 failures the individual rows stop being the
    useful read, but "did they all share one spawn distance?" still is.
    """
    from cs2rl.experiment.oracle_statue import FAIL_DETAIL_LIMIT, _failure_lines
    n = FAIL_DETAIL_LIMIT + 15
    failures = [{
        "episode": i,
        "hero_xy": (150.0, 250.0),
        "statue_xy": (350.0, 250.0),
        "spawn_dist": 200.0 + i,
        "yaw_err": 0.1 * i,
    } for i in range(n)]
    lines = _failure_lines({"episodes": n, "failures": failures})
    assert sum(1 for ln in lines if ln.strip().startswith("ep ")) == FAIL_DETAIL_LIMIT, lines
    assert f"  ... {n - FAIL_DETAIL_LIMIT} more" in lines, lines
    assert any("spawn distance 2D (u)    min 200.0" in ln for ln in lines), lines
    assert any(f"max {200.0 + n - 1:.1f}" in ln for ln in lines), lines
    # No failures at all (a FAIL on median TTK alone) prints nothing.
    assert _failure_lines({"episodes": 4, "failures": []}) == []
