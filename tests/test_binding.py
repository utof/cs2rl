import sys
import ctypes
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "c_env"))
import binding  # noqa


def _make_env(map_data=None):
    from c_env.cs2_env import make_env
    env = make_env(seed=0, map_data=map_data)
    return env._capsule, env


def test_binding_functions_present():
    for name in ("init", "reset", "step", "close", "get_buffers"):
        assert hasattr(binding, name), f"binding.{name} missing"


def test_get_buffers_returns_four_ints(make_map):
    _, env = _make_env(map_data=make_map)
    result = binding.get_buffers(env._capsule)
    assert len(result) == 4
    for ptr in result:
        assert isinstance(ptr, int)
        assert ptr != 0


def test_reset_returns_none(make_map):
    _, env = _make_env(map_data=make_map)
    assert binding.reset(env._capsule) is None


def test_step_returns_none(make_map):
    _, env = _make_env(map_data=make_map)
    binding.reset(env._capsule)
    actions = np.zeros(10, dtype=np.int32)
    assert binding.step(env._capsule, actions) is None


def test_close_idempotent(make_map):
    _, env = _make_env(map_data=make_map)
    assert binding.close(env._capsule) is None
    assert binding.close(env._capsule) is None
