"""Unit tests for src/cs2rl/env/config.py — the env configuration contract (spec 2026-09-03 §2.1).

This file is the ONE place the default numbers are restated (spec §6 "Added"):
the literals below were copied from train_shared.REWARD_WEIGHT_DEFAULTS at
main @ 139a3a3. If a default changes on purpose, change it in env/config.py AND
here; if this test fails and you did not mean to change a default, the module
drifted.
"""
import dataclasses
import pickle
import sys

import pytest

from cs2rl.env import config as env_config
from cs2rl.env.config import TEAM_SIZE, UNSET, EnvConfig, RewardWeights

DEFAULTS_AT_139a3a3 = {
    "reward_win": 1.0,
    "reward_kill": 0.3,
    "reward_death": 0.1,
    "reward_bombsite_entry": 0.3,
    "reward_plant_bonus": 3.0,
    "reward_plant_base": 0.2,
    "reward_plant_progress_scale": 0.05,
    "reward_plant_interrupted": 0.1,
    "reward_defuse": 0.2,
    "reward_shot_penalty": 0.005,
    "reward_ct_survival": 0.001,
    "reward_inaction": 0.0005,
    "reward_win_t_detonation": 5.0,
    "reward_win_t_elimination": 3.0,
    "reward_win_ct_defuse": 5.0,
    "reward_win_ct_timeout": 4.0,
    "reward_win_ct_elimination": 3.0,
    "pbrs_alive_weight": 0.3,
    "pbrs_hp_weight": 0.002,
    "pbrs_site_weight": 0.2,
    "pbrs_bomb_progress_weight": 0.3,
    "pbrs_nav_weight_t": 0.04,
    "pbrs_nav_weight_ct": 0.15,
}
KNOB_DEFAULTS = {
    "pbrs_gamma": 0.999,
    "reward_symmetrize": False,
    "recoil": False,
    "n_active_per_team": 5,
    "pin_pitch": 0,
    "crouch_enabled": 1,
    "jump_enabled": 1,
    "round_time": None,
    "laser_range": None,
    "max_turn_speed": None,
}
CONFIG_DICT_KEYS = sorted(DEFAULTS_AT_139a3a3) + sorted([
    "pbrs_gamma", "reward_symmetrize", "n_active_per_team", "pin_pitch", "crouch_enabled",
    "jump_enabled"
])


def test_reward_field_census_is_23_with_6_pbrs():
    names = [f.name for f in dataclasses.fields(RewardWeights)]
    assert names == list(DEFAULTS_AT_139a3a3), "order and membership are the contract"
    assert sum(1 for n in names if n.startswith("pbrs_")) == 6


def test_defaults_equal_the_139a3a3_values():
    assert RewardWeights().as_dict() == pytest.approx(DEFAULTS_AT_139a3a3)
    cfg = EnvConfig()
    for k, v in KNOB_DEFAULTS.items():
        assert getattr(cfg, k) == v, k
    assert cfg.n_active_per_team == TEAM_SIZE == 5


def test_knob_field_order_is_the_documented_ten():
    """Both sides are spellings of the ten knobs; KNOB_FIELDS is derived from
    EnvConfig's fields, so this pins the ORDER and the names against a literal.
    """
    assert env_config.KNOB_FIELDS == tuple(KNOB_DEFAULTS)


def test_weights_are_coerced_to_float_and_pickle_round_trips():
    rw = RewardWeights(reward_kill=1)  # int on purpose
    assert type(rw.reward_kill) is float and rw.reward_kill == 1.0
    cfg = EnvConfig(rewards=rw, n_active_per_team=3)
    assert pickle.loads(pickle.dumps(cfg)) == cfg


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.3", None, True])
def test_bad_weight_values_raise_value_error_naming_the_key(bad):
    with pytest.raises(ValueError, match="reward_kill"):
        RewardWeights(reward_kill=bad)


@pytest.mark.parametrize(("field", "bad"), [
    ("n_active_per_team", 0),
    ("n_active_per_team", 6),
    ("n_active_per_team", 2.9),
    ("round_time", 0),
    ("round_time", 2.5),
    ("laser_range", 0.0),
    ("laser_range", float("nan")),
    ("max_turn_speed", -1.0),
    ("pbrs_gamma", float("inf")),
])
def test_bad_knob_values_raise_value_error(field, bad):
    with pytest.raises(ValueError, match=field):
        EnvConfig(**{field: bad})


@pytest.mark.parametrize(("field", "bad"), [
    ("pbrs_gamma", None),
    ("n_active_per_team", None),
    ("laser_range", "not a number"),
    ("max_turn_speed", "fast"),
])
def test_unconvertible_knob_raises_value_error_naming_the_field(field, bad):
    """A knob that will not convert must name itself, like the weights do.

    These values arrive from a CLI namespace with ten knob candidates, and
    `float(None)`'s own message ("float() argument must be a string or a real
    number") says nothing about WHICH knob was wrong. ValueError, not the bare
    TypeError the conversion raises, so callers have one class to catch — the
    same rule parent §2.1's error table sets for the weights.

    `max_turn_speed` is exercised with a STRING, not with None: None is that
    field's documented sentinel ("use the env/nav.py constant"), `EnvConfig()` sets
    it, and `test_none_r0g_knobs_survive_untouched` pins that it survives — so a None
    param here could only ever pass by breaking `EnvConfig()` itself.
    """
    with pytest.raises(ValueError, match=field):
        EnvConfig(**{field: bad})


def test_unset_is_rejected_by_both_dataclasses():
    """The sentinel means "not given"; as a VALUE it is silently wrong.

    UNSET is truthy, so before this guard `EnvConfig(reward_symmetrize=UNSET)`
    normalised to True and `EnvConfig(pin_pitch=UNSET)` to 1 — an env running a
    knob nobody set, with every test green. `replace()` is the live path: Phase B
    adds its first caller.
    """
    with pytest.raises(TypeError, match="reward_symmetrize"):
        EnvConfig(reward_symmetrize=UNSET)
    with pytest.raises(TypeError, match="reward_kill"):
        RewardWeights(reward_kill=UNSET)
    with pytest.raises(TypeError, match="jump_enabled"):
        EnvConfig().replace(jump_enabled=UNSET)
    with pytest.raises(TypeError, match="rewards"):
        EnvConfig(rewards=UNSET)


def test_unset_repr_is_the_documented_spelling():
    """`<unset>` is the sentinel's public spelling; changing it must be deliberate.

    NOT because users see it in an error message — `from_legacy_kwargs` filters
    UNSET out (`if local[n] is not UNSET`) before every message it raises, so no
    error text can contain it. It is asserted because the repr is what shows up
    in a debugger, a failed-assert dump and this suite's own output, and
    `_Unset.__repr__` is three lines nobody would otherwise notice editing.
    """
    assert repr(UNSET) == "<unset>"


def test_flag_knobs_normalise_to_int_bool():
    cfg = EnvConfig(pin_pitch=2, crouch_enabled=True, jump_enabled=0.0)
    assert (cfg.pin_pitch, cfg.crouch_enabled, cfg.jump_enabled) == (1, 1, 0)
    assert all(type(v) is int for v in (cfg.pin_pitch, cfg.crouch_enabled, cfg.jump_enabled))


def test_none_r0g_knobs_survive_untouched():
    cfg = EnvConfig()
    assert cfg.round_time is None and cfg.laser_range is None and cfg.max_turn_speed is None


def test_frozen_and_misspelled_field_is_a_type_error():
    cfg = EnvConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.pin_pitch = 1              # type: ignore[misc]
    with pytest.raises(TypeError, match="rewrad_kill"):
        EnvConfig(rewrad_kill=1)       # type: ignore[call-arg]
    with pytest.raises(TypeError, match="rewrad_kill"):
        RewardWeights(rewrad_kill=1)   # type: ignore[call-arg]


def test_replace_revalidates():
    cfg = EnvConfig()
    assert cfg.replace(jump_enabled=0).jump_enabled == 0
    with pytest.raises(ValueError, match="n_active_per_team"):
        cfg.replace(n_active_per_team=9)
    with pytest.raises(TypeError):
        cfg.rewards.replace(nope=1.0)


def test_to_config_dict_key_set_matches_the_pinned_literal():
    """The key set equals CONFIG_DICT_KEYS above — a literal, not today's output.

    Renamed from "..._is_exactly_todays", which overclaimed: this compares the
    dict to a hand-written list in this file, so it cannot see a key that a real
    `config.json` needs and neither side has. That check is
    tests/train/test_train_cli.py::test_dump_config_matches_the_pre_165_fixture, which
    compares against a config.json captured from the real CLI.
    """
    d = EnvConfig().to_config_dict()
    assert sorted(d) == sorted(CONFIG_DICT_KEYS)
    assert d["reward_kill"] == 0.3 and d["pbrs_gamma"] == 0.999 and d["reward_symmetrize"] is False
    for k in ("round_time", "laser_range", "max_turn_speed", "recoil"):
        assert k not in d


def test_module_is_stdlib_only():
    """Whitelist gate: env_config may import NOTHING outside the stdlib.

    Why a whitelist and not a blacklist of known-bad names: the stdlib-only
    import budget is this module's single hardest global constraint, and a
    blacklist of six names stays GREEN the day someone adds `import polars`,
    `import yaml` or `from cs2rl.env import factory`. That is exactly the silent
    failure this test exists to prevent. So we diff sys.modules across the import
    and require every newly-added name to be exactly `cs2rl`, `cs2rl.env` or
    `cs2rl.env.config` (the packages and this module), or to have a TOP-LEVEL name
    in sys.stdlib_module_names (Python 3.10+). First-party names are compared in
    FULL: every one of them is top-level `cs2rl`, so a top-level allowance for
    it would let `cs2rl.env.factory` and `cs2rl.env.c.cs2_env` through.

    Pitfall: this MUST stay in a subprocess. The parent pytest process has
    already imported numpy, torch and the whole src tree, so an in-process
    check on sys.modules cannot distinguish "env_config imported it" from
    "someone else did" and would mask every violation.
    """
    import subprocess
    child = ("import sys\n"
             "before = set(sys.modules)\n"
             "from cs2rl.env import config as env_config\n"
             "added = set(sys.modules) - before\n"
             "bad = sorted(m for m in added\n"
             "             if m not in ('cs2rl', 'cs2rl.env', 'cs2rl.env.config')\n"
             "             and m.split('.')[0] not in sys.stdlib_module_names)\n"
             "print('NON_STDLIB=' + ','.join(bad))\n"
             "sys.exit(1 if bad else 0)\n")
    r = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True)
    assert r.returncode == 0, (f"env_config must import stdlib only; it pulled in "
                               f"{r.stdout.strip()}\n{r.stderr}")
