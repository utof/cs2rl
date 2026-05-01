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
    """Batch 3: binding.step is now 3-arg — capsule, int32 discrete actions,
    float32 continuous_actions. The shape (10,) here is wrong for both, but
    the C side reads N_AGENTS*ACTION_DIM ints/N_AGENTS*AIM_DIM floats. The
    raw int32(10,) buffer happens to be ≥10*7*4 bytes only if reinterpreted —
    use the proper 2D shape now to be safe.
    """
    from _action_spec import ACTION_DIM, AIM_DIM
    _, env = _make_env(map_data=make_map)
    binding.reset(env._capsule)
    actions = np.zeros((10, ACTION_DIM), dtype=np.int32)
    cont = np.zeros((10, AIM_DIM), dtype=np.float32)
    assert binding.step(env._capsule, actions, cont) is None


def test_close_idempotent(make_map):
    _, env = _make_env(map_data=make_map)
    assert binding.close(env._capsule) is None
    assert binding.close(env._capsule) is None


def test_stepstats_has_win_type_flags(make_map):
    """StepStats ctypes struct must expose win_by_detonation and win_by_defuse.
    Accessor: env._c_env.step_stats (ctypes StepStatsC — not a numpy recarray;
    binding.c has no StepStats dtype descriptor, ctypes is the Python-side mirror).
    """
    _, env = _make_env(map_data=make_map)
    ss = env._c_env.step_stats
    assert hasattr(ss, "win_by_detonation"), "StepStatsC missing win_by_detonation"
    assert hasattr(ss, "win_by_defuse"), "StepStatsC missing win_by_defuse"
    # At reset, both must be zero
    env.reset()
    assert int(ss.win_by_detonation) == 0
    assert int(ss.win_by_defuse) == 0


def test_human_controlled_uses_aim_rad_not_bin(make_map):
    """When human_controlled=1, facing must equal aim_rad, ignoring the
    continuous_actions Δyaw buffer.

    Batch 3: pre-Batch-3 this test verified the 16-bin override; now it
    verifies that the human branch in env_step (`if (a->human_controlled)`)
    short-circuits BEFORE reading continuous_actions. We pass a non-zero
    Δyaw to confirm it is ignored — only aim_rad sets facing for human
    agents.
    """
    from _action_spec import ACTION_DIM, AIM_DIM
    _, env = _make_env(map_data=make_map)
    binding.reset(env._capsule)

    agent = env._c_env.game.agents[0]
    agent.human_controlled = 1
    aim = 1.23456                      # arbitrary radians
    agent.aim_rad = aim

    actions = np.zeros((10, ACTION_DIM), dtype=np.int32)
    # Non-zero continuous Δyaw — the human branch must IGNORE this.
    cont = np.zeros((10, AIM_DIM), dtype=np.float32)
    cont[0, 0] = 0.5
    binding.step(env._capsule, actions, cont)

    facing = env._c_env.game.agents[0].facing
    assert abs(facing - aim) < 1e-5, f"Expected facing≈{aim:.5f}, got {facing:.5f}"


# ── Batch 3: continuous-aim plumbing ──


def test_binding_step_accepts_continuous_array(make_map):
    """binding.step now takes 3 args (capsule, int32 actions, float32 cont).

    Batch 3: validates the new signature. Wrong shape on continuous_actions
    raises a Python ValueError (caught Python-side in Cs2Env._prepare_continuous_actions
    before the C call). Correct shape is accepted.
    """
    import pytest

    from _action_spec import ACTION_DIM, AIM_DIM
    _, env = _make_env(map_data=make_map)
    env.reset(seed=0)
    actions = np.zeros((10, ACTION_DIM), dtype=np.int32)
    cont = np.zeros((10, AIM_DIM), dtype=np.float32)
    env.step(actions, cont)            # should not raise
    with pytest.raises(ValueError):
        env.step(actions, np.zeros((10, 2), dtype=np.float32))


def test_binding_default_continuous_actions_zero(make_map):
    """If continuous_actions arg omitted, zero buffer supplied — facing unchanged.

    Batch 3: defensive default keeps legacy callers (smoke-test loops, the
    train.py main path before the policy is wired in T4-T5) working without
    explicit continuous-action arrays. We use the designated bomb carrier
    (an RL agent, not human_controlled) so the continuous branch in env_step
    fires.
    """
    from _action_spec import ACTION_DIM
    _, env = _make_env(map_data=make_map)
    env.reset(seed=0)
    g = env._c_env.game
    i = g.round_designated_carrier_id
    f0 = g.agents[i].facing
    actions = np.zeros((10, ACTION_DIM), dtype=np.int32)
    env.step(actions)                                  # no continuous_actions
    assert g.agents[i].facing == f0, (
        f"facing changed without continuous_actions: f0={f0}, after={g.agents[i].facing}")
