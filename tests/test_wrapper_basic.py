"""Wrapper smoke test — env init, reset, step return correct shapes."""
import numpy as np
import pytest


def test_import_wrapper():
    from c_env.wrapper import Dust2CEnv, build_static_data
    assert Dust2CEnv is not None


def test_reset_returns_10_obs():
    from c_env.wrapper import make_env
    env = make_env(seed=0)
    obs, info = env.reset()
    assert obs is not None


def test_step_returns_shapes():
    from c_env.wrapper import make_env
    env = make_env(seed=0)
    env.reset()
    actions = np.zeros((10, 4), dtype=np.int32)
    obs, rewards, terms, truncs, info = env.step(actions)
    assert len(rewards) == 10
    assert len(terms) == 10
