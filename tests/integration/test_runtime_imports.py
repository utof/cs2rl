import importlib


def test_pufferlib_importable():
    import pufferlib                   # noqa: F401


def test_sb3_absent():
    assert importlib.util.find_spec("stable_baselines3") is None, (
        "stable_baselines3 must be removed from deps")


def test_supersuit_absent():
    assert importlib.util.find_spec("supersuit") is None, "supersuit must be removed from deps"
