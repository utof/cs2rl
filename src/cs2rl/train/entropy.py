"""The entropy targets: how much policy entropy the PPO update steers toward, and when.

Owns two pure schedules and nothing else:
  target_entropy_schedule      -- the normal linear ramp from `warmup_high_frac * max_entropy`
                                  down to `base_frac * max_entropy`, held after `warmup_end`;
  warmstart_entropy_state      -- the override for BC-warm-started runs (GRACE -> RAMP -> OFF),
                                  its frozen result `WarmstartEntropyState` and the WS_GRACE /
                                  WS_RAMP / WS_OFF phase constants.

Both are stateless and torch-free (stdlib only): the trainer decides when to call them and
what to latch. `cs2rl.train.update._scheduled_target_entropy` feeds the first the config's
values, and `cs2rl.train.trainer` reads the second's phase and `floor_active`.
"""

from __future__ import annotations

import dataclasses


def target_entropy_schedule(
    step: int,
    max_entropy: float,
    warmup_end: int = 10_000_000,
    warmup_high_frac: float = 0.5,
    base_frac: float = 0.35,
) -> float:
    """Linear ramp from `warmup_high_frac * max_entropy` at step 0 down to
    `base_frac * max_entropy` at step `warmup_end`; held constant after.

    Defaults lowered 0.7→0.5 / 0.5→0.35 (finding 4 residual, 2026-07-06
    adversarial review): the old targets were high enough to hold the
    policy near-uniform indefinitely. These defaults are FALLBACK mirrors
    of build_train_config's entropy_target_{warmup,base}_frac /
    entropy_target_warmup_steps — production threads the config values
    through train._scheduled_target_entropy, so tune there, not here.
    Keep base_frac > 0.3: the trainer's hard entropy floor (clamp α ≥ 0.5
    when H < 0.3·max) must stay strictly below the scheduled target."""
    # Single branch: clamp t to [0, 1] so the linear formula yields
    # base_frac * max_entropy at step >= warmup_end without a discontinuity.
    t = min(step / warmup_end, 1.0)
    frac = warmup_high_frac + (base_frac - warmup_high_frac) * t
    return frac * max_entropy


# Warm-start entropy mode phases (spec docs/superpowers/specs/
# 2026-08-01-warmstart-entropy-control-design.md). Ints (not Enum) so the
# value can go straight into the losses/* dict as a plottable metric.
WS_GRACE, WS_RAMP, WS_OFF = 0, 1, 2


@dataclasses.dataclass(frozen=True)
class WarmstartEntropyState:
    """One evaluation of the warm-start entropy schedule.

    target is None during GRACE (alpha is ceilinged so no target is consumed);
    floor_active gates the trainer's hard alpha>=0.5 entropy-floor clamp —
    False for the ENTIRE warm-start window (grace + ramp), True at/after
    ramp_end. PITFALL (spec finding 2): do NOT derive floor_active from
    "target < floor" — the ramp crosses the floor mid-window and re-arming
    there is a ~500x effective-alpha discontinuity at an unplotted step.
    """
    phase: int
    target: float | None
    floor_active: bool


def warmstart_entropy_state(step: int, *, grace_steps: int, ramp_steps: int, h_anchor: float | None,
                            base_target: float) -> WarmstartEntropyState:
    """Pure warm-start entropy schedule: GRACE -> RAMP -> OFF.

    WHAT: while step < grace_steps (or no h_anchor captured yet), entropy
    pressure is off (GRACE). Then the target ramps linearly from h_anchor
    (the policy's measured mean entropy at grace end) to base_target over
    ramp_steps (RAMP), after which behavior is identical to the normal
    schedule (OFF). ramp_steps <= 0 (including negative values, treated the
    same as 0) skips RAMP entirely: the state jumps straight to OFF at
    step == grace_steps.

    h_anchor LATCHING CONTRACT: the caller must capture h_anchor exactly
    ONCE, at the grace->ramp transition, from that moment's previous-update
    mean entropy — then pass that SAME frozen float on every subsequent call
    for the rest of the ramp. This function does not latch anything itself;
    it is pure and re-evaluates its inputs every call. Passing a live,
    per-update entropy value instead of the frozen one degenerates the
    linear ramp into a contraction of the *current* entropy toward
    base_target (a different, unintended schedule) rather than a fixed
    interpolation from the grace-end anchor.

    WHY h_anchor=None => GRACE even past the boundary: the anchor is read
    from the PREVIOUS update's mean entropy (trainer._batch1_last_entropy_mean
    — added by the trainer wiring task, not yet present in this module),
    which doesn't exist on the very first update of a grace_steps=0 run —
    grace semantics until the caller can anchor keeps that case well-defined.

    PITFALL: continuity at grace end comes from target==h_anchor (alpha's
    Lagrangian gradient ~0), NOT from moving log_alpha — Adam(lr=1e-4) on a
    scalar moves log-alpha at most ~1e-4/minibatch (~0.9 over a 30M run), so
    a parked log_alpha can never climb back (spec §1 finding-1 bound).

    PITFALL: there is no "mode disabled" representation here. A caller with
    the warm-start feature disabled must NOT call this function at all — a
    permanently-None h_anchor makes every call return GRACE with
    floor_active=False forever, silently disarming the trainer's entropy
    floor for the whole run. The trainer wiring task gates all calls behind
    the `warmstart_entropy` config flag; this module has no such flag.

    Args:
        step: trainer.global_step — agent steps, the same counter
            target_entropy_schedule's `step` argument uses.
        grace_steps: length of the GRACE window, in the same agent-step
            unit as `step`.
        ramp_steps: length of the RAMP window (after grace_steps), in the
            same agent-step unit as `step`.
        h_anchor: the frozen mean-entropy anchor captured at the grace->ramp
            transition (see LATCHING CONTRACT above), or None before it has
            been captured.
        base_target: the steady-state (OFF-phase) entropy target, in the
            same units as h_anchor.

    Returns:
        WarmstartEntropyState with phase in {WS_GRACE, WS_RAMP, WS_OFF},
        target (None only during GRACE), and floor_active (True only once
        OFF is reached).
    """
    if step < grace_steps or h_anchor is None:
        return WarmstartEntropyState(WS_GRACE, None, False)
    if ramp_steps <= 0 or step >= grace_steps + ramp_steps:
        return WarmstartEntropyState(WS_OFF, base_target, True)
    t = (step - grace_steps) / ramp_steps
    return WarmstartEntropyState(WS_RAMP, h_anchor + (base_target - h_anchor) * t, False)
