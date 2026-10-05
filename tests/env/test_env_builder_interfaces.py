"""Explicit builders reject incomplete calls and resolve the constructor afresh."""

import pytest

from cs2rl.env import factory
from cs2rl.env.config import EnvConfig


@pytest.mark.parametrize(('name', 'kwargs'), [
    ('build_train_env',
     dict(shared_ts=None, buf=None, seed=None, _seed=17, map_data=None, config=EnvConfig())),
    ('build_eval_env', dict(map_data=None, config=EnvConfig(reward_symmetrize=True))),
    ('build_legacy_eval_env', {}),
    ('build_smoke_env', {}),
    ('build_harness_env',
     dict(shared_ts=None, buf=None, seed=None, map_data=None, config=EnvConfig())),
    ('build_external_env', dict(team_spirit=None, map_data=None)),
])
def test_builder_rebinds_constructor_and_rejects_stray_keywords(monkeypatch, name, kwargs):
    """Caching the constructor would use an obsolete binding on the second call."""
    from cs2rl.env.c import cs2_env

    builder = getattr(factory, name)
    first, second = object(), object()
    monkeypatch.setattr(cs2_env, 'make_env', lambda **kw: first)
    assert builder(**kwargs) is first
    monkeypatch.setattr(cs2_env, 'make_env', lambda **kw: second)
    assert builder(**kwargs) is second
    with pytest.raises(TypeError, match='unexpected keyword'):
        builder(**kwargs, misspelled_config=EnvConfig())


@pytest.mark.parametrize(('name', 'kwargs', 'missing'), [
    ('build_train_env', dict(shared_ts=None, buf=None, seed=0, _seed=None,
                             map_data=None), 'config'),
    ('build_eval_env', dict(map_data=None), 'config'),
    ('build_harness_env', dict(shared_ts=None, buf=None, seed=0, map_data=None), 'config'),
    ('build_external_env', dict(team_spirit=None), 'map_data'),
])
def test_incomplete_builder_call_cannot_silently_construct_defaults(monkeypatch, name, kwargs,
                                                                    missing):
    """A dropped config/map must fail before allocating an environment."""
    from cs2rl.env.c import cs2_env

    def unexpected_constructor(**kw):
        pytest.fail('incomplete call reached the constructor')

    monkeypatch.setattr(cs2_env, 'make_env', unexpected_constructor)
    with pytest.raises(TypeError, match=missing):
        getattr(factory, name)(**kwargs)
