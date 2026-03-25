import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "c_env"))
import binding                         # noqa


def _make_env(map_data=None):
    from c_env.cs2_env import make_env
    env = make_env(seed=0, map_data=map_data)
    return env._capsule, env


def test_binding_functions_present():
    for name in ("init", "reset", "step", "close", "get_buffers", "get_masks"):
        assert hasattr(binding, name), f"binding.{name} missing"


def test_get_masks_returns_nonzero_ptr(make_map):
    _, env = _make_env(map_data=make_map)
    binding.reset(env._capsule)
    ptr = binding.get_masks(env._capsule)
    assert isinstance(ptr, int)
    assert ptr != 0


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


def test_human_controlled_uses_aim_rad_not_bin(make_map):
    """When human_controlled=1, facing must equal aim_rad, not the 16-bin quantized value."""
    _, env = _make_env(map_data=make_map)
    binding.reset(env._capsule)

    agent = env._c_env.game.agents[0]
    agent.human_controlled = 1
    aim = 1.23456                      # arbitrary radians, not on a 16-bin boundary
    agent.aim_rad = aim

    # actions[1] = bin 7 (a different angle) — must be overridden
    actions = np.zeros((10, 7), dtype=np.int32)
    actions[0, 1] = 7
    binding.step(env._capsule, actions.flatten())

    # Read back facing
    facing = env._c_env.game.agents[0].facing
    assert abs(facing - aim) < 1e-5, f"Expected facing≈{aim:.5f}, got {facing:.5f}"
