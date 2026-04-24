"""Pure helpers for Batch 1 reward-architecture changes.

Separate module so unit tests can import without pulling in the full
PufferLib trainer. These functions are imported by the monkey-patches in
src/train.py.

Functions:
    symlog(x):           sign-preserving log compression; bounds scale without hard cutoff
    symexp(x):           inverse of symlog
    target_entropy_schedule(step, max_entropy, warmup_end=10_000_000):
                         linear ramp from 0.7*max_entropy to 0.5*max_entropy over warmup_end steps
    split_into_channels(step_stats_view):
                         route StepStats reward fields into combat/objective/positional channels
                         using win_by_detonation / win_by_defuse flags

All functions are stateless and deterministic. Welford running-std is
implemented separately (see train.py inline class).
"""
from __future__ import annotations

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
    warmup_high_frac: float = 0.7,
    base_frac: float = 0.5,
) -> float:
    """Linear ramp from `warmup_high_frac * max_entropy` at step 0 down to
    `base_frac * max_entropy` at step `warmup_end`; held constant after."""
    if step >= warmup_end:
        return base_frac * max_entropy
    t = step / warmup_end
    frac = warmup_high_frac + (base_frac - warmup_high_frac) * t
    return frac * max_entropy


# Channel assignment from spec §Channel assignment
_COMBAT_FIELDS = ("reward_kills", "reward_deaths", "reward_shots")
_OBJECTIVE_FIELDS = ("reward_bomb", )
_POSITIONAL_FIELDS = ("reward_pbrs", "reward_survival", "reward_inaction")


def split_into_channels(step_stats: np.ndarray) -> dict[str, float]:
    """Route StepStats reward fields into combat/objective/positional channels.

    reward_win is routed by mechanism:
      - win_by_detonation or win_by_defuse → objective
      - otherwise (elimination/timeout) → combat

    Accepts either a 0-d structured array (single record) or a length-1
    structured array — both arise in practice depending on the caller.
    """
    # Normalise to a 0-d record so field indexing always returns a scalar.
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
