"""The env configuration contract: one typed object owns every env knob and reward default.

WHAT: two frozen dataclasses. `RewardWeights` holds the 23 reward/PBRS
coefficients; `EnvConfig` holds a `RewardWeights` plus the ten sim knobs. Both
validate in `__post_init__`. `Cs2Env` reads its scalars off an `EnvConfig`;
`make_env` / `make_puffer_env` translate legacy keyword names through
`from_legacy_kwargs`; `build_train_config` writes `to_config_dict()`.

WHY (gh#165, ADR 0003): before this module the same 23 defaults were declared in
`make_env`, `Cs2Env.__init__` and `train_shared.REWARD_WEIGHT_DEFAULTS`, agreeing
only because a test compared them. One declaration, here, is the fix.

IMPORT BUDGET: stdlib ONLY. `c_env.cs2_env` (layer L1) imports this module, and
`train.py --dump-config` must stay free of torch/nav/c_env, so nothing heavier
than `dataclasses` may ever be imported here. `TEAM_SIZE` is a literal for the
same reason `train_shared.py` carries one: `nav` costs awpy/polars to read a 5.
tests/test_train_env.py cross-checks it against nav.TEAM_SIZE.

PITFALL: the values below ARE the trained baseline. An unflagged run must stay
byte-identical to the pre-#165 env; do not "tidy" a number here.

PITFALL: `None` on round_time / laser_range / max_turn_speed means "the nav.py
constant" and is resolved (and validated) inside Cs2Env.__init__, never here —
resolving it here would need `nav`. Only non-None values are validated here.
"""
from __future__ import annotations

import dataclasses
import difflib
import math
import numbers
from dataclasses import dataclass, field, fields

TEAM_SIZE = 5                          # cross-checked against nav.TEAM_SIZE by tests/test_train_env.py


class _Unset:
    """Sentinel for from_legacy_kwargs: "the caller did not pass this name"."""
    __slots__ = ()

    def __repr__(self):
        return "<unset>"


UNSET = _Unset()


def _knob_number(name, value, cast):
    """Coerce a knob, re-raising the conversion failure as a ValueError that NAMES it.

    WHY: `float(None)` and `int("x")` raise messages that describe the builtin,
    not the knob, and these values arrive from an args namespace with ten
    candidates. ValueError (not the underlying TypeError) so every bad-knob path
    has one class, matching the reward-weight rule above.
    """
    try:
        return cast(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}={value!r} is not a number") from None


def _reject_unset(name, value):
    """UNSET is the "caller did not pass this" sentinel, never a value.

    It is truthy and float-less, so an UNSET that reaches a field would
    normalise to True/1 (flags) or explode far from the caller. Rejected in both
    __post_init__s, BEFORE coercion, naming the field.
    """
    if value is UNSET:
        raise TypeError(f"{name}=UNSET: the sentinel means 'not given', not a value; "
                        "omit the argument instead")


def _check_weight(name, value):
    """Today's make_puffer_env rule (the last boundary before C): real, non-bool, finite."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"reward weight {name}={value!r} is not a real number "
                         f"(got {type(value).__name__})")
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"reward weight {name}={f} is not finite; pass a real number "
                         "(this would poison the loss silently)")
    return f


@dataclass(frozen=True)
class RewardWeights:
    """The 23 reward/PBRS coefficients. Field order == the historical
    REWARD_WEIGHT_DEFAULTS order == the CLI flag order (`--reward-*`, `--pbrs-*`).

    Non-potential terms first (hackable — sweep with care), then the six PBRS
    potential weights (optimum-safe per Ng et al. 1999). Six of the 23 do NOT
    start with `reward_`; never discover weights by prefix.
    """
    reward_win: float = 1.0
    reward_kill: float = 0.3
    reward_death: float = 0.1
    reward_bombsite_entry: float = 0.3
    reward_plant_bonus: float = 3.0
    reward_plant_base: float = 0.2
    reward_plant_progress_scale: float = 0.05
    reward_plant_interrupted: float = 0.1
    reward_defuse: float = 0.2
    reward_shot_penalty: float = 0.005
    reward_ct_survival: float = 0.001                  # the CT stall drip — A1 arm sets this to 0.0
    reward_inaction: float = 0.0005
    reward_win_t_detonation: float = 5.0
    reward_win_t_elimination: float = 3.0
    reward_win_ct_defuse: float = 5.0
    reward_win_ct_timeout: float = 4.0                 # exceeds ct_elimination on purpose — A1b arm
    reward_win_ct_elimination: float = 3.0
    pbrs_alive_weight: float = 0.3
    pbrs_hp_weight: float = 0.002
    pbrs_site_weight: float = 0.2
    pbrs_bomb_progress_weight: float = 0.3
    pbrs_nav_weight_t: float = 0.04
    pbrs_nav_weight_ct: float = 0.15

    def __post_init__(self):
        # Frozen dataclass: coercion has to go through object.__setattr__.
        for f in fields(self):
            v = getattr(self, f.name)
            _reject_unset(f.name, v)
            object.__setattr__(self, f.name, _check_weight(f.name, v))

    def as_dict(self) -> dict[str, float]:
        """field -> value in declaration order (the CLI loop and pin tests read this)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def replace(self, **changes) -> RewardWeights:
        """dataclasses.replace with validation re-run; unknown field -> TypeError."""
        return dataclasses.replace(self, **changes)


REWARD_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(RewardWeights))


@dataclass(frozen=True)
class EnvConfig:
    """Every env knob plus the reward weights. Runtime inputs (seed, team_spirit,
    buf, map_data, auto_reset, include_step_stats_in_info) are NOT fields: if
    config.json must record it to reproduce the run it is a field; if it differs
    per env instance or per role without changing the dynamics it is a runtime
    input (spec 2026-09-03 §2.2).

    pin_pitch, crouch_enabled and jump_enabled are flags: __post_init__ maps them
    through int(bool(...)), so any truthy value normalises rather than being
    rejected — the rule Cs2Env applied inline before #165.
    """
    rewards: RewardWeights = field(default_factory=RewardWeights)
    pbrs_gamma: float = 0.999                          # MUST equal the training gamma (PBRS policy-invariance)
    reward_symmetrize: bool = False
    recoil: bool = False
    n_active_per_team: int = TEAM_SIZE
    pin_pitch: int = 0
    crouch_enabled: int = 1
    jump_enabled: int = 1
    round_time: int | None = None                      # None ⇒ nav.ROUND_TIME, resolved in Cs2Env
    laser_range: float | None = None                   # None ⇒ nav.LASER_RANGE
    max_turn_speed: float | None = None                # None ⇒ nav.MAX_TURN_SPEED_RAD

    def __post_init__(self):
        s = object.__setattr__
        for f in fields(self):
            _reject_unset(f.name, getattr(self, f.name))
        if not isinstance(self.rewards, RewardWeights):
            raise TypeError(f"rewards must be a RewardWeights, got {type(self.rewards).__name__}")
        g = _knob_number("pbrs_gamma", self.pbrs_gamma, float)
        if not math.isfinite(g):
            raise ValueError(f"pbrs_gamma={self.pbrs_gamma!r} is not finite")
        s(self, "pbrs_gamma", g)
        s(self, "reward_symmetrize", bool(self.reward_symmetrize))
        s(self, "recoil", bool(self.recoil))
        n = self.n_active_per_team
        # reject, never truncate (int(2.9) parks a different roster)
        if _knob_number("n_active_per_team", n, int) != n:
            raise ValueError(f"n_active_per_team must be an integer, got {n!r}")
        n = int(n)
        if not 1 <= n <= TEAM_SIZE:
            raise ValueError(f"n_active_per_team must be in 1..{TEAM_SIZE}, got {n}")
        s(self, "n_active_per_team", n)
        for name in ("pin_pitch", "crouch_enabled", "jump_enabled"):
            s(self, name, int(bool(getattr(self, name))))
        rt = self.round_time
        if rt is not None:
            if _knob_number("round_time", rt, int) != rt:
                raise ValueError(f"round_time must be an integer tick count, got {rt!r}")
            if int(rt) <= 0:
                raise ValueError(f"round_time must be > 0, got {rt}")
            s(self, "round_time", int(rt))
        for name in ("laser_range", "max_turn_speed"):
            v = getattr(self, name)
            if v is not None:
                v = _knob_number(name, v, float)
                if not v > 0.0:        # `not >` also rejects NaN
                    raise ValueError(f"{name} must be > 0, got {v}")
                s(self, name, v)

    def replace(self, **changes) -> EnvConfig:
        return dataclasses.replace(self, **changes)

    def to_config_dict(self) -> dict:
        """The flat provenance keys build_train_config writes: the 23 weights
        verbatim plus six knobs. NOT the R0-G trio (recorded under CLI names by
        build_train_config) and NOT recoil (no CLI flag)."""
        d = self.rewards.as_dict()
        d.update(pbrs_gamma=self.pbrs_gamma,
                 reward_symmetrize=self.reward_symmetrize,
                 n_active_per_team=self.n_active_per_team,
                 pin_pitch=self.pin_pitch,
                 crouch_enabled=self.crouch_enabled,
                 jump_enabled=self.jump_enabled)
        return d

    @classmethod
    def from_legacy_kwargs(
        cls,
        *,
        reward_win=UNSET,
        reward_kill=UNSET,
        reward_death=UNSET,
        reward_bombsite_entry=UNSET,
        reward_plant_bonus=UNSET,
        reward_plant_base=UNSET,
        reward_plant_progress_scale=UNSET,
        reward_plant_interrupted=UNSET,
        reward_defuse=UNSET,
        reward_shot_penalty=UNSET,
        reward_ct_survival=UNSET,
        reward_inaction=UNSET,
        reward_win_t_detonation=UNSET,
        reward_win_t_elimination=UNSET,
        reward_win_ct_defuse=UNSET,
        reward_win_ct_timeout=UNSET,
        reward_win_ct_elimination=UNSET,
        pbrs_alive_weight=UNSET,
        pbrs_hp_weight=UNSET,
        pbrs_site_weight=UNSET,
        pbrs_bomb_progress_weight=UNSET,
        pbrs_nav_weight_t=UNSET,
        pbrs_nav_weight_ct=UNSET,
        pbrs_gamma=UNSET,
        reward_symmetrize=UNSET,
        recoil=UNSET,
        n_active_per_team=UNSET,
        pin_pitch=UNSET,
        crouch_enabled=UNSET,
        jump_enabled=UNSET,
        round_time=UNSET,
        laser_range=UNSET,
        max_turn_speed=UNSET,
        reward_overrides=UNSET,
    ) -> EnvConfig:
        """Translate the pre-#165 make_env / make_puffer_env keyword names.

        34 KEYWORD-ONLY parameters and deliberately NO **kwargs: an unknown
        name must die against a real parameter list (Python's own TypeError),
        and tests bind captured kwargs against inspect.signature(this). A
        parameter left UNSET takes the field default. `pbrs_gamma=None` means
        the default (make_puffer_env's documented spelling). `reward_overrides`
        is the make_puffer_env dict channel: unknown key -> ValueError with a
        did-you-mean hint, bad value -> ValueError (today's rules, moved here).
        """
        local = locals()
        weights = {n: local[n] for n in REWARD_FIELDS if local[n] is not UNSET}
        if reward_overrides is not UNSET and reward_overrides is not None:
            unknown = set(reward_overrides) - set(REWARD_FIELDS)
            if unknown:
                hints = []
                for key in sorted(unknown):
                    near = difflib.get_close_matches(key, REWARD_FIELDS, n=1)
                    if near:
                        hints.append(f"{key!r} — did you mean --{near[0].replace('_', '-')}?")
                raise ValueError(f"unknown reward override keys: {sorted(unknown)}. " +
                                 (" ".join(hints) + " " if hints else "") +
                                 f"Valid keys: {sorted(REWARD_FIELDS)}. Non-weight env knobs "
                                 "(pbrs_gamma, reward_symmetrize) are NOT overrides.")
            weights.update(reward_overrides)
        knobs = {n: local[n] for n in KNOB_FIELDS if local[n] is not UNSET}
        if knobs.get("pbrs_gamma", 0.0) is None:
            del knobs["pbrs_gamma"]
        return cls(rewards=RewardWeights(**weights), **knobs)


# DERIVED, never hand-listed, and therefore defined below the class: every
# EnvConfig field except the nested `rewards` object, in declaration order.
# from_legacy_kwargs (above) reads this at CALL time, so the forward reference is
# fine. A hand-written tuple would let EnvConfig grow an eleventh knob that
# from_legacy_kwargs accepts and then silently discards, leaving the env on the
# default with every test green; tests/test_struct_sizes.py builds its sentinel
# partition from this tuple and would inherit the same blindness.
KNOB_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(EnvConfig) if f.name != "rewards")
