"""Warm-start entropy mode — pure schedule tests (spec 2026-08-01 §2/§4).

Phase semantics pinned here:
  GRACE (0): entropy pressure off (alpha ceilinged, floor clamp disabled,
             alpha optimizer paused). Active while step < grace_steps OR the
             entropy anchor has not been captured yet (h_anchor=None covers
             the grace_steps=0 first-update case where no previous update's
             mean entropy exists to anchor to).
  RAMP  (1): target ramps linearly h_anchor -> base_target over ramp_steps;
             floor clamp STAYS disabled (re-arming it mid-ramp would be a
             ~500x alpha discontinuity at an unplotted step — spec finding 2).
  OFF   (2): steady state, identical to no-warmstart behavior; floor active.
"""
from train_helpers_batch1 import WS_GRACE, WS_OFF, WS_RAMP, warmstart_entropy_state


def test_grace_phase_before_grace_end():
    s = warmstart_entropy_state(0,
                                grace_steps=5_000_000,
                                ramp_steps=10_000_000,
                                h_anchor=None,
                                base_target=2.874)
    assert s.phase == WS_GRACE
    assert s.target is None
    assert s.floor_active is False
    s = warmstart_entropy_state(4_999_999,
                                grace_steps=5_000_000,
                                ramp_steps=10_000_000,
                                h_anchor=None,
                                base_target=2.874)
    assert s.phase == WS_GRACE


def test_no_anchor_yet_stays_grace_even_past_boundary():
    # grace_steps=0 with no anchor captured yet: grace semantics until the
    # caller has a previous update's entropy mean to anchor to.
    s = warmstart_entropy_state(100,
                                grace_steps=0,
                                ramp_steps=10_000_000,
                                h_anchor=None,
                                base_target=2.874)
    assert s.phase == WS_GRACE
    assert s.floor_active is False


def test_grace_phase_with_anchor_already_set_but_step_before_grace_end():
    # Mutation guard: deleting the `step < grace_steps` guard entirely would
    # still pass every other test in this module as long as h_anchor is None
    # in those cases. Pin the guard explicitly with a non-None h_anchor and
    # step < grace_steps: must still be GRACE with target None.
    s = warmstart_entropy_state(1_000_000,
                                grace_steps=5_000_000,
                                ramp_steps=10_000_000,
                                h_anchor=1.8,
                                base_target=2.874)
    assert s.phase == WS_GRACE
    assert s.target is None
    assert s.floor_active is False


def test_ramp_endpoints_and_midpoint():
    kw = dict(grace_steps=5_000_000, ramp_steps=10_000_000, h_anchor=1.8, base_target=2.874)
    at_start = warmstart_entropy_state(5_000_000, **kw)
    assert at_start.phase == WS_RAMP
    assert abs(at_start.target - 1.8) < 1e-9
    assert at_start.floor_active is False
    mid = warmstart_entropy_state(10_000_000, **kw)
    assert abs(mid.target - (1.8 + 2.874) / 2) < 1e-9
    end = warmstart_entropy_state(15_000_000, **kw)
    assert end.phase == WS_OFF
    assert abs(end.target - 2.874) < 1e-9
    assert end.floor_active is True


def test_ramp_steps_zero_goes_straight_to_off():
    s = warmstart_entropy_state(5_000_000,
                                grace_steps=5_000_000,
                                ramp_steps=0,
                                h_anchor=1.8,
                                base_target=2.874)
    assert s.phase == WS_OFF
    assert s.floor_active is True
    assert abs(s.target - 2.874) < 1e-9


def test_grace_steps_zero_with_anchor_starts_ramp_immediately():
    s = warmstart_entropy_state(0,
                                grace_steps=0,
                                ramp_steps=10_000_000,
                                h_anchor=1.8,
                                base_target=2.874)
    assert s.phase == WS_RAMP
    assert abs(s.target - 1.8) < 1e-9
    assert s.floor_active is False


def test_far_future_step_is_off_at_base_target():
    s = warmstart_entropy_state(10**12,
                                grace_steps=5_000_000,
                                ramp_steps=10_000_000,
                                h_anchor=1.8,
                                base_target=2.874)
    assert s.phase == WS_OFF
    assert abs(s.target - 2.874) < 1e-9
    assert s.floor_active is True


def test_phase_constants_are_the_wire_contract():
    # These ints are written into the losses/* metrics dict, so their values
    # are a wire contract for downstream plotting/consumers — pin them.
    assert WS_GRACE == 0
    assert WS_RAMP == 1
    assert WS_OFF == 2


def test_downward_ramp_when_anchor_above_base():
    # h_anchor can exceed base_target (policy hotter than steady-state target);
    # the ramp must interpolate downward, not clamp.
    s = warmstart_entropy_state(7_500_000,
                                grace_steps=5_000_000,
                                ramp_steps=5_000_000,
                                h_anchor=4.0,
                                base_target=2.874)
    assert abs(s.target - (4.0 + 2.874) / 2) < 1e-9
