"""The legacy make_env shim's contract (spec 2026-09-03 §2.3, §8 criteria 2-3).

WHAT: make_env(config=…) is the typed path; make_env(**legacy) is the
translation path; the two are mutually exclusive. Cs2Env itself has NO legacy
channel and rejects a non-EnvConfig config.

WHY a separate file: this file is OWNED BY THE SHIM. When the follow-up issue
filed at the end of Phase A deletes `**legacy`, `from_legacy_kwargs` and
`make_puffer_env`, this file is deleted with them. tests/test_env_config.py
tests the module and survives that deletion.

PITFALL: the calls below are LEGACY BY DESIGN and Phase B's migration census
(tests/test_env_config_migration.py) counts them. They are the shim's own tests,
so a census that drops to zero while this file exists is measuring the wrong
thing; the census pins a count, and this file's contribution to it goes away
only when the shim does.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from c_env.cs2_env import Cs2Env, make_env             # noqa: E402, I001
from env_config import EnvConfig                       # noqa: E402, I001


def test_config_plus_legacy_kwargs_is_a_type_error(simple_map):
    """Spec §2.3: `config` given AND legacy non-empty -> TypeError. Silently
    letting one win is how a swept weight gets ignored for a whole run."""
    with pytest.raises(TypeError, match="reward_kill"):
        make_env(config=EnvConfig(), reward_kill=1.0, map_data=simple_map)


def test_bare_call_builds_the_default_config(simple_map):
    env = make_env(map_data=simple_map)
    try:
        assert env.config == EnvConfig()
    finally:
        env.close()


def test_the_config_object_is_stored_not_copied(simple_map):
    """Spec §2.3 stores it as `self.config`; Phase B's eval/driver agreement loop
    reads `.config` off two envs, so identity has to survive make_env."""
    cfg = EnvConfig(n_active_per_team=3)
    env = make_env(config=cfg, map_data=simple_map)
    try:
        assert env.config is cfg
        assert env.n_active_per_team == 3              # and it was APPLIED, not just stored
    finally:
        env.close()


def test_misspelled_legacy_name_dies_before_binding_init(simple_map, monkeypatch):
    """Spec §8 criterion 3, both halves. TypeError naming the key, AND raised
    in-process before any C allocation — monkeypatching binding.init to explode
    is the knock-out: without the explicit parameter list on from_legacy_kwargs,
    a **kwargs shim would carry the typo all the way to the C boundary."""
    from c_env import cs2_env

    def _boom(*a, **k):
        raise AssertionError("binding.init must not be reached")

    monkeypatch.setattr(cs2_env.binding, "init", _boom)
    with pytest.raises(TypeError, match="rewrad_kill"):
        make_env(rewrad_kill=1, map_data=simple_map)


def test_cs2env_rejects_a_non_envconfig_config(simple_map):
    """Cs2Env is the L1 constructor and takes no legacy channel (spec §3): a dict
    that happens to have the right keys must not be accepted as a config."""
    with pytest.raises(TypeError, match="EnvConfig"):
        Cs2Env(config={}, map_data=simple_map)
