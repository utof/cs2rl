import importlib

def test_sample_factory_importable():
    import sample_factory  # noqa: F401
    # sample-factory 2.1.1 ships without a dedicated pettingzoo_envs module;
    # verify the APPO env-registry and training entry point are importable instead.
    from sample_factory.algo.utils.context import global_env_registry  # noqa: F401
    from sample_factory.train import run_rl  # noqa: F401

def test_sb3_absent():
    assert importlib.util.find_spec("stable_baselines3") is None, (
        "stable_baselines3 must be removed from deps"
    )

def test_supersuit_absent():
    assert importlib.util.find_spec("supersuit") is None, (
        "supersuit must be removed from deps"
    )
