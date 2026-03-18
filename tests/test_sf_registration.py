import numpy as np
import pytest


def test_env_registered_on_train_import():
    """Importing train registers 'cs2-dust2' with SF's registry."""
    import train  # noqa: F401 — registration is a side-effect of import
    from sample_factory.algo.utils.context import global_env_registry
    from sample_factory.envs.create_env import create_env

    registry = global_env_registry()
    assert "cs2-dust2" in registry, f"'cs2-dust2' not found in registry keys: {list(registry.keys())}"

    env = create_env("cs2-dust2", cfg={}, env_config=None)
    assert env is not None


def test_registered_env_obs_shape():
    """Wrapped env returns (71,) obs for each agent."""
    import train  # noqa: F401
    from sample_factory.envs.create_env import create_env

    env = create_env("cs2-dust2", cfg={}, env_config=None)
    obs, _ = env.reset()
    assert len(obs) == 10, f"Expected 10 agents, got {len(obs)}"
    for i, ob in enumerate(obs):
        assert ob.shape == (71,), f"agent[{i}] shape {ob.shape} != (71,)"
        assert np.isfinite(ob).all(), f"agent[{i}] has NaN in reset obs"
