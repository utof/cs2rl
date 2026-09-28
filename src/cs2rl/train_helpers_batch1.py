"""Pure helpers for Batch 1 reward-architecture changes.

Separate module so unit tests can import without pulling in the full
PufferLib trainer. These functions are imported by the monkey-patches in
src/cs2rl/train.py.

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


def _scale_cuda(x: float, std: float) -> np.float32:
    """x / max(std,1e-8) exactly as CUDA torch does it: multiply by the double
    reciprocal rounded to float32 (torch lowers div-by-Scalar to mul)."""
    return np.float32(x) * np.float32(1.0 / max(std, 1e-8))


def _scale_cpu(x: float, std: float) -> np.float32:
    """x / max(std,1e-8) exactly as CPU torch does it: true float32 division
    against the float32-rounded divisor."""
    return np.float32(x) / np.float32(max(std, 1e-8))


def process_step_rewards(
    info,
    r: torch.Tensor,
    agents_per_env: int,
    welford_combat: WelfordStd,
    welford_objective: WelfordStd,
    welford_positional: WelfordStd,
    scratch: np.ndarray,
    current_segment_has_event: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-channel reward normalisation + symlog for one vecenv tick, batched.

    WHAT
      Consumes the per-env `info` list produced by Cs2Env (each entry either a
      dict carrying a "step_stats" view, or something without it), advances the
      three per-channel WelfordStd estimators once per env, and returns a new
      reward tensor shaped like `r` (one row per *agent*) where every agent row
      of env `e` holds symlog(sum of that env's normalised channels).

    WHY (perf)
      The previous inline version built three single-scalar CUDA tensors per env
      per tick (torch.tensor(python_float, device=cuda) x3) plus ~10 more tiny
      device ops. At 256 envs x 64 ticks that is ~213k launch-bound CUDA ops per
      epoch inside the timed eval_forward region, with GPU utilisation at 33%.
      Here the per-env arithmetic happens in numpy float32 on the host, and the
      device sees ONE host-to-device copy, ONE repeat_interleave and ONE symlog
      per tick.

    PITFALLS — both of these silently change training values if broken:
      1. The per-env Python loop MUST stay. WelfordStd state is global and
         mutated per env, so env e's normalisation has to observe the state as
         of *after* env e's own update(): it is the update/normalize
         INTERLEAVING that matters, not merely the update order. Only tensor
         construction was hoisted out; nothing was reordered.
      2. The channel arithmetic is float32 end-to-end, and it has to reproduce
         torch's `float32_tensor / python_float` lowering *per device* — those
         two lowerings do not agree with each other to the last bit:
           - CUDA: torch turns division by a Scalar into a multiply by its
             reciprocal, with the reciprocal taken in double and then rounded to
             float32 → np.float32(x) * np.float32(1.0 / s).
           - CPU: a genuine float32 division against the float32-rounded
             divisor → np.float32(x) / np.float32(s).
         Measured over thousands of samples each lowering matches its device
         exactly and the other one differs on ~20-25% of inputs by 1 ULP. Plain
         Python-float (float64) math matches neither. The association order
         (combat + objective) + positional is likewise preserved.
         If a future torch release changes this lowering,
         tests/test_reward_loop_equivalence.py fails on the affected device —
         that is the intended tripwire, so fix the formula rather than the test.
      3. symlog is applied once on device to the assembled vector — never on the
         host scalars — so the log1p rounding matches the old path bit for bit.

    Envs whose info entry has no "step_stats" (flag off, or an older info entry)
    fall back to passing their raw `r` rows through UNCHANGED and are NOT
    symlog'd. A short info list (len(info) * agents_per_env < r.shape[0]) leaves
    the trailing raw rows untouched as well.

    `scratch` is a caller-owned np.float32 buffer of length >= len(info); it is
    reused across ticks to keep the loop allocation-free. Exact bit-for-bit
    equivalence with the old per-scalar torch path is guaranteed for
    r.dtype == torch.float32 (what the env actually produces); other float
    dtypes are supported by a final cast but not ULP-audited.

    `current_segment_has_event`, when given, is the live per-agent-row event
    accumulator: rows of an env whose step_stats reports bomb_planted this tick
    are set True (per-tick delta from the C side — no edge trigger needed).
    """
    n_env = len(info)
    r_new = torch.empty_like(r)
    buf = scratch[:n_env]
    # raw_envs: indices of envs with no step_stats — their raw r rows pass through.
    raw_envs = []
    # Device-specific mirror of WelfordStd.normalize() — see PITFALL 2. Bound
    # once per tick, not per env.
    _scale = _scale_cuda if r.is_cuda else _scale_cpu

    for e in range(n_env):
        entry = info[e]
        ss = entry.get("step_stats", None) if isinstance(entry, dict) else None
        if ss is None:
            raw_envs.append(e)
            # Placeholder; this env's rows get overwritten with raw r below.
            buf[e] = 0.0
            continue
        channels = split_into_channels(ss)
        # Welford.update takes scalar floats (one observation per env/tick).
        welford_combat.update(channels["combat"])
        welford_objective.update(channels["objective"])
        welford_positional.update(channels["positional"])
        # bool(int(...)) is deliberate: stubs or numpy scalars may not
        # truthy-coerce cleanly; int() normalises first. Do not strip the cast.
        if current_segment_has_event is not None and bool(int(ss.get("bomb_planted", 0))):
            current_segment_has_event[e * agents_per_env:(e + 1) * agents_per_env] = True
        c = _scale(channels["combat"], welford_combat.std())
        o = _scale(channels["objective"], welford_objective.std())
        p = _scale(channels["positional"], welford_positional.std())
        buf[e] = (c + o) + p

    # min(): if info is somehow LONGER than the batch, the extra envs still get
    # their Welford update (as in the old per-env-slice-assign path) but their
    # rewards have nowhere to go — clamp instead of raising on shape mismatch.
    used = min(n_env * agents_per_env, r.shape[0])
    if used:
        vec = torch.from_numpy(buf).to(r.device)
        if vec.dtype != r.dtype:
            vec = vec.to(r.dtype)
        r_new[:used] = symlog(vec).repeat_interleave(agents_per_env)[:used]
    for e in raw_envs:
        row_start = e * agents_per_env
        r_new[row_start:row_start + agents_per_env] = r[row_start:row_start + agents_per_env]
    if used < r.shape[0]:
        r_new[used:] = r[used:]
    return r_new
