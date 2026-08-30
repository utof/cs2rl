"""Rung 1a Task 0: the oracle-vs-statue solvability precondition, in-process.

WHAT IS PINNED HERE
-------------------
* The scripted oracle kills a stationary CT on the exact Rung 1a env
  (arena-duel, n_active=1, round_time=160, pin_pitch=1, crouch=0) in a small
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

WHY the thresholds here are looser than the script's (0.8 vs 0.90 kill rate):
this is the fast regression tripwire at N=20, not the gate. The gate is
``UV_NO_SYNC=1 uv run python scripts/oracle_statue_check.py`` at N>=200.
"""
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Offset that reproduces a crouched target's |rz| against a STANDING hitbox
# (TORSO_OFFSET_CROUCH 24 vs EYE_HEIGHT_STAND 48), and one that clears the
# standing vertical semi-axis (HIT_HALF_HEIGHT_STAND = 36) outright.
CROUCH_HEIGHT_OFFSET = 24.0
ABOVE_SEMI_AXIS_OFFSET = 57.0
# cs2_movement.h leapfrog: one half gravity step lands between the hold and
# process_combat, so the ray sees `offset - 1.5625`.
GRAVITY_SAG = 0.5 * 800.0 * (1.0 / 16.0)**2


def test_oracle_kills_grounded_statue():
    """The precondition itself: perfect aim vs a target that never moves."""
    from oracle_statue_check import run_check, verdict
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


def test_oracle_kills_statue_held_at_crouch_height_offset():
    """v1c ellipsoid, in-env: a standing target displaced ~22 u vertically.

    Asserts exactly two things and no more: the hold really placed the target at
    that |rz| (``rz_min``/``rz_max``, measured off the live C state), and the
    oracle still kills it with ``shots_stance_blocked == 0``. It does NOT assert
    anything about ``HIT_HALF_HEIGHT_CROUCH``: the statue is standing, so the
    gate used HH = 36. crouch_enabled=0 masks the crouch head, so a genuinely
    crouched target is not reachable through the Rung 1a action space.
    """
    from oracle_statue_check import run_check, verdict
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
    matter how perfect the yaw. Every shot must be on target (the yaw window is
    a 2D quantity, unaffected by z) AND stance-blocked AND a miss. If the hold
    were a no-op this test would look exactly like the grounded run and fail on
    the first assert, which is what makes the 24 u case meaningful.

    Few episodes on purpose: nothing dies, so each one runs the full 160 ticks.
    """
    from oracle_statue_check import run_check, verdict
    res = run_check(episodes=4, seed=0, statue_z=ABOVE_SEMI_AXIS_OFFSET)
    assert res["rz_min"] == pytest.approx(ABOVE_SEMI_AXIS_OFFSET - GRAVITY_SAG, abs=0.05), res
    assert res["kills"] == 0 and res["ttk_min"] is None, res
    assert res["shots_fired"] > 0, res
    assert res["shots_on_target"] == res["shots_fired"], res
    assert res["shots_stance_blocked"] == res["shots_fired"], res
    assert res["shots_hit"] == 0, res
    assert not verdict(res)[0], res


def test_verdict_is_the_conjunction_of_all_three_checks():
    """Pure-function gate arithmetic: each check must be able to fail alone."""
    from oracle_statue_check import PASS_MAX_MEDIAN_TTK, PASS_MIN_KILL_RATE, verdict
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


def test_main_exit_code_follows_the_verdict(capsys):
    """The CLI contract the controller scripts against: 0 on PASS, 1 on FAIL."""
    from oracle_statue_check import main
    assert main(["--episodes", "4", "--seed", "0"]) == 0
    assert "PASS" in capsys.readouterr().out
    assert main(["--episodes", "2", "--seed", "0", "--statue-z", str(ABOVE_SEMI_AXIS_OFFSET)]) == 1
    assert "FAIL" in capsys.readouterr().out
