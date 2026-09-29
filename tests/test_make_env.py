"""The typed make_env path and Cs2Env's config contract.

WHAT: make_env(config=…) is the only configuration path; Cs2Env itself
rejects a non-EnvConfig config. A bare make_env() builds EnvConfig().
Unknown keywords are a TypeError naming the key; the signature has no
VAR_KEYWORD.

The three original tests of the typed path keep their names. The two
criterion-4 tests pin Python's unexpected-keyword contract after **legacy
was deleted.
"""
import inspect

import pytest

from cs2rl.env.c.cs2_env import Cs2Env, make_env
from cs2rl.env.config import EnvConfig


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


def test_cs2env_rejects_a_non_envconfig_config(simple_map):
    """Cs2Env is the L1 constructor and takes no legacy channel (spec §3): a dict
    that happens to have the right keys must not be accepted as a config."""
    with pytest.raises(TypeError, match="EnvConfig"):
        Cs2Env(config={}, map_data=simple_map)


def test_unknown_keyword_is_a_type_error_naming_the_key(simple_map):
    with pytest.raises(TypeError, match="rewrad_kill"):
        make_env(rewrad_kill=1, map_data=simple_map)


def test_make_env_signature_has_no_var_keyword():
    params = inspect.signature(make_env).parameters
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
