"""Pure helpers for Batch 1 reward-architecture changes.

Separate module so unit tests can import without pulling in the full
PufferLib trainer. These functions are imported by the monkey-patches in
src/train.py.

Functions:
    symlog(x):           sign-preserving log compression; bounds scale without hard cutoff
    symexp(x):           inverse of symlog
    target_entropy_schedule(step, max_entropy, warmup_end=10_000_000,
                             warmup_high_frac=0.5, base_frac=0.35):
                         linear ramp from warmup_high_frac*max_entropy to base_frac*max_entropy
                         over warmup_end steps; held constant after.
    split_into_channels(step_stats_view):
                         route StepStats reward fields into combat/objective/positional channels
                         using win_by_detonation / win_by_defuse flags
    warmstart_entropy_state(step, grace_steps, ramp_steps, h_anchor, base_target):
                         two-phase warm-start entropy schedule: GRACE (alpha ceilinged,
                         floor disabled) -> RAMP (target h_anchor -> base_target,
                         floor still disabled) -> OFF (steady state, floor active).

Classes:
    WelfordStd:          scalar online-std estimator with prior_std warmup fallback;
                         Task 6 instantiates three (one per reward channel).
    WarmstartEntropyState: frozen dataclass result of warmstart_entropy_state
                         (phase, target, floor_active).

All free functions are stateless and deterministic. WelfordStd carries
per-instance running statistics — each channel needs its own instance.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch


def symlog(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * ln(|x| + 1). Preserves sign; bounds scale without hard cutoff.

    Note: torch.log1p(|x|) is used for numerical stability near zero.
    Do NOT substitute torch.log(|x| + 1) — different rounding behavior for small |x|.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


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
    schedule (OFF).

    WHY h_anchor=None => GRACE even past the boundary: the anchor is read
    from the PREVIOUS update's mean entropy (trainer._batch1_last_entropy_mean),
    which doesn't exist on the very first update of a grace_steps=0 run —
    grace semantics until the caller can anchor keeps that case well-defined.

    PITFALL: continuity at grace end comes from target==h_anchor (alpha's
    Lagrangian gradient ~0), NOT from moving log_alpha — Adam(lr=1e-4) on a
    scalar moves log-alpha at most ~1e-4/minibatch (~0.9 over a 30M run), so
    a parked log_alpha can never climb back (spec §1 finding-1 bound).
    """
    if step < grace_steps or h_anchor is None:
        return WarmstartEntropyState(WS_GRACE, None, False)
    if ramp_steps <= 0 or step >= grace_steps + ramp_steps:
        return WarmstartEntropyState(WS_OFF, base_target, True)
    t = (step - grace_steps) / ramp_steps
    return WarmstartEntropyState(WS_RAMP, h_anchor + (base_target - h_anchor) * t, False)


# Channel assignment from spec §Channel assignment
_COMBAT_FIELDS = ("reward_kills", "reward_deaths", "reward_shots")
_OBJECTIVE_FIELDS = ("reward_bomb", )
_POSITIONAL_FIELDS = ("reward_pbrs", "reward_survival", "reward_inaction")


def split_into_channels(step_stats: np.ndarray) -> dict[str, float]:
    """Route StepStats reward fields into combat/objective/positional channels.

    Input: structured numpy array (0-d record OR shape (1,) length-1) with
    float fields reward_{win,kills,deaths,bomb,pbrs,shots,survival,inaction}
    and int8 flags win_by_{detonation,defuse}. The length-1 form arises
    because test fixtures construct records via np.zeros(1, dtype=...); the
    0-d form arises when the trainer indexes a per-env slice. Both are
    accepted and produce identical output.

    reward_win is routed by mechanism:
      - win_by_detonation or win_by_defuse → objective
      - otherwise (elimination/timeout) → combat

    NOTE: only the 8 reward_* fields listed above are routed. If StepStats
    gains a new reward_* field in a future batch, this helper will silently
    ignore it (see utof/cs2rl#3 for a routing-drift guard).
    """
    # Normalise a length-1 structured array (as produced by the spec's own
    # test fixtures, e.g. `ss = np.zeros(1, dtype=[...])`) to a 0-d record
    # so field indexing returns a scalar rather than a length-1 array.
    if step_stats.ndim == 1:
        step_stats = step_stats[0]
    combat = sum(float(step_stats[f]) for f in _COMBAT_FIELDS)
    objective = sum(float(step_stats[f]) for f in _OBJECTIVE_FIELDS)
    positional = sum(float(step_stats[f]) for f in _POSITIONAL_FIELDS)

    win = float(step_stats["reward_win"])
    if int(step_stats["win_by_detonation"]) or int(step_stats["win_by_defuse"]):
        objective += win
    else:
        combat += win

    return {"combat": combat, "objective": objective, "positional": positional}


class WelfordStd:
    """Welford's online std estimator, per-channel.

    Prior seed: std = prior_std (default 1.0) until count reaches min_count;
    this prevents a rare-event channel (few non-zero samples) from being
    systematically down-weighted by a spuriously small std during warmup.

    This is a scalar-only version; each channel gets its own instance.

    Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self, prior_std: float = 1.0, min_count: int = 1000):
        self.prior_std = prior_std
        self.min_count = min_count
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0                  # sum of squared diffs from current mean

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def variance(self) -> float:
        if self.count < 2:
            return self.prior_std * self.prior_std
        return self.m2 / (self.count - 1)

    def std(self) -> float:
        if self.count < self.min_count:
            return self.prior_std
        v = self.variance()
        return float(v**0.5) if v > 0 else self.prior_std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Return x / max(std, 1e-8). No mean subtraction (rewards are sign-meaningful)."""
        s = self.std()
        return x / max(s, 1e-8)
