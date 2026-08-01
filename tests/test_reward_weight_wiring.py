"""Reward-weight wiring — the §6.1 default-equality pin (spec 2026-08-01).

WHY this test exists: train.REWARD_WEIGHT_DEFAULTS duplicates make_env's
defaults by hand. It has to — train.py imports c_env lazily (inside
functions) so that --dump-config stays free of the map/binding/torch import
cost, and a module-level `inspect.signature(make_env)` would break that
guarantee. This test is the price of the duplication: it fails the moment
either side drifts, in EITHER direction (a weight added to make_env, a
default changed, a key dropped from the tuple).

PITFALL: _NON_WEIGHT_PARAMS lists every make_env parameter that is
deliberately NOT threaded, each for cause (spec §4.1). If you add a genuinely
new reward weight to make_env, add it to REWARD_WEIGHT_DEFAULTS — do NOT
"fix" this test by widening the exclusion set.
"""
import inspect

import pytest

# Params of make_env that are NOT reward weights, each excluded for cause:
#   seed/team_spirit/auto_reset/buf/map_data — not weights at all;
#   pbrs_gamma — must equal training gamma or PBRS loses policy invariance;
#     make_puffer_env already has a dedicated guarded parameter for it
#     (test_pbrs_gamma_matches_training_gamma);
#   include_step_stats_in_info — the issue #100 decision, out of scope;
#   reward_symmetrize — a bool knob, not a weight (Task 3; listed here from
#   the start so this test does not break when Task 3 lands).
_NON_WEIGHT_PARAMS = {
    "seed",
    "team_spirit",
    "auto_reset",
    "buf",
    "map_data",
    "pbrs_gamma",
    "include_step_stats_in_info",
    "reward_symmetrize",
}


def _make_env_signature():
    from c_env.cs2_env import make_env
    return inspect.signature(make_env)


def test_reward_weight_keys_cover_exactly_the_weight_params():
    params = set(_make_env_signature().parameters) - _NON_WEIGHT_PARAMS
    from train import REWARD_WEIGHT_KEYS
    assert set(REWARD_WEIGHT_KEYS) == params, (
        "REWARD_WEIGHT_KEYS drifted from make_env's signature; "
        f"missing={params - set(REWARD_WEIGHT_KEYS)} "
        f"extra={set(REWARD_WEIGHT_KEYS) - params}")
    # NOTE: the two count assertions below are INFORMATIONAL redundancy over the
    # signature set-equality (one edit → three failures is intended as a loud signal,
    # not a prohibition): when legitimately adding a 24th weight, bump them alongside
    # REWARD_WEIGHT_DEFAULTS (review finding 6).
    assert len(REWARD_WEIGHT_KEYS) == 23, (
        f"spec §4.1 pins 23 threaded weights, got {len(REWARD_WEIGHT_KEYS)}")
    # Six of them do NOT start with reward_ — a prefix scan is wrong by
    # construction (spec §4.2). Pin that so nobody "simplifies" the tuple away.
    assert sum(1 for k in REWARD_WEIGHT_KEYS if not k.startswith("reward_")) == 6


def test_reward_weight_defaults_equal_make_env_defaults():
    sig = _make_env_signature()
    from train import REWARD_WEIGHT_DEFAULTS
    for name, default in REWARD_WEIGHT_DEFAULTS.items():
        assert sig.parameters[name].default == pytest.approx(default), (
            f"{name}: train default {default} != make_env default "
            f"{sig.parameters[name].default} — an unflagged run would no "
            "longer be byte-identical to the pre-wiring env")


def test_reward_weight_keys_are_tuple_of_defaults_dict():
    """Single derivation point (spec §4.2): the tuple IS the dict's keys."""
    from train import REWARD_WEIGHT_DEFAULTS, REWARD_WEIGHT_KEYS
    assert REWARD_WEIGHT_KEYS == tuple(REWARD_WEIGHT_DEFAULTS)
