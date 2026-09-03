"""Unit tests for src/env_config.py — the env configuration contract (spec 2026-09-03 §2.1).

This file is the ONE place the default numbers are restated (spec §6 "Added"):
the literals below were copied from train_shared.REWARD_WEIGHT_DEFAULTS at
main @ 139a3a3. If a default changes on purpose, change it in env_config.py AND
here; if this test fails and you did not mean to change a default, the module
drifted.
"""
import dataclasses
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# I001 is suppressed, not fixed: yapf snaps these trailing `noqa` comments to its
# spaces_before_comment stops while ruff's isort wants one space, and the two then
# fight forever (gh#97; same waiver as tests/test_modal_runner.py:51).
import env_config                                                      # noqa: E402, I001
from env_config import TEAM_SIZE, UNSET, EnvConfig, RewardWeights      # noqa: E402

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
    It cannot see a field added to EnvConfig with no legacy parameter —
    test_legacy_signature_covers_exactly_the_field_names below is that check."""
    assert env_config.KNOB_FIELDS == tuple(KNOB_DEFAULTS)


def test_legacy_signature_covers_exactly_the_field_names():
    """The two-way drift guard between the dataclass and the legacy shim.

    from_legacy_kwargs builds `knobs` by iterating KNOB_FIELDS, so a knob that is
    a field AND a legacy parameter but missing from KNOB_FIELDS would be accepted
    and then silently dropped — the env would run on the default. Deriving
    KNOB_FIELDS from `fields(EnvConfig)` closes that direction; this closes the
    other two: a field with no legacy parameter, and a legacy parameter with no
    field. test_struct_sizes.py's sentinel partition inherits both guards."""
    import inspect

    from env_config import KNOB_FIELDS, REWARD_FIELDS
    params = set(inspect.signature(EnvConfig.from_legacy_kwargs).parameters)
    assert params == set(REWARD_FIELDS) | set(KNOB_FIELDS) | {"reward_overrides"}


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


def test_from_legacy_kwargs_round_trips_all_33_names_and_overrides():
    legacy = {
        **{
            k: v * 2
            for k, v in DEFAULTS_AT_139a3a3.items()
        }, "pbrs_gamma": 0.99,
        "reward_symmetrize": True,
        "recoil": True,
        "n_active_per_team": 2,
        "pin_pitch": 1,
        "crouch_enabled": 0,
        "jump_enabled": 0,
        "round_time": 900,
        "laser_range": 1234.0,
        "max_turn_speed": 0.05
    }
    cfg = EnvConfig.from_legacy_kwargs(**legacy)
    assert cfg.rewards.as_dict() == pytest.approx({
        k: v * 2
        for k, v in DEFAULTS_AT_139a3a3.items()
    })
    for k in KNOB_DEFAULTS:
        assert getattr(cfg, k) == legacy[k], k
    over = EnvConfig.from_legacy_kwargs(reward_overrides={"reward_ct_survival": 0.0})
    assert over.rewards.reward_ct_survival == 0.0 and over.rewards.reward_kill == 0.3
    assert EnvConfig.from_legacy_kwargs() == EnvConfig()
    assert EnvConfig.from_legacy_kwargs(pbrs_gamma=None, reward_overrides=None) == EnvConfig()


def test_from_legacy_kwargs_signature_has_34_keyword_only_params_and_no_var_keyword():
    import inspect
    sig = inspect.signature(EnvConfig.from_legacy_kwargs)
    kinds = {p.kind for p in sig.parameters.values()}
    assert kinds == {inspect.Parameter.KEYWORD_ONLY}
    assert len(sig.parameters) == 34
    assert all(p.default is UNSET for p in sig.parameters.values())


def test_from_legacy_kwargs_error_contract():
    with pytest.raises(TypeError, match="rewrad_kill"):
        EnvConfig.from_legacy_kwargs(rewrad_kill=1.0)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="did you mean --reward-kill"):
        EnvConfig.from_legacy_kwargs(reward_overrides={"rewrad_kill": 1.0})
    with pytest.raises(ValueError, match="reward_kill"):
        EnvConfig.from_legacy_kwargs(reward_overrides={"reward_kill": float("nan")})


def test_to_config_dict_key_set_is_exactly_todays():
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
    `import yaml` or `import env_factory`. That is exactly the silent failure
    this test exists to prevent. So we diff sys.modules across the import and
    require every newly-added TOP-LEVEL name to be in sys.stdlib_module_names
    (Python 3.10+). Top-level, so `import cs2_env` and `import c_env.cs2_env`
    are both caught.

    Pitfall: this MUST stay in a subprocess. The parent pytest process has
    already imported numpy, torch and the whole src tree, so an in-process
    check on sys.modules cannot distinguish "env_config imported it" from
    "someone else did" and would mask every violation.
    """
    import subprocess
    src = Path(__file__).resolve().parents[1] / "src"
    child = ("import sys\n"
             f"sys.path.insert(0, {str(src)!r})\n"
             "before = set(sys.modules)\n"
             "import env_config\n"
             "added = {m.split('.')[0] for m in set(sys.modules) - before}\n"
             "bad = sorted(m for m in added\n"
             "             if m != 'env_config' and m not in sys.stdlib_module_names)\n"
             "print('NON_STDLIB=' + ','.join(bad))\n"
             "sys.exit(1 if bad else 0)\n")
    r = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True)
    assert r.returncode == 0, (f"env_config must import stdlib only; it pulled in "
                               f"{r.stdout.strip()}\n{r.stderr}")
