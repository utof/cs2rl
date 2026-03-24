"""Wrapper smoke test — env init, reset, step return correct shapes."""

import numpy as np
from nav import OBS_DIM, ACTION_DIM


def test_import_wrapper():
    from c_env.cs2_env import Cs2Env

    assert Cs2Env is not None


def test_reset_returns_10_obs():
    from c_env.cs2_env import make_env

    env = make_env(seed=0)
    obs, info = env.reset()
    assert obs.shape == (10, OBS_DIM)


def test_step_returns_shapes():
    from c_env.cs2_env import make_env

    env = make_env(seed=0)
    env.reset()
    actions = np.zeros((10, ACTION_DIM), dtype=np.int32)
    obs, rewards, terms, truncs, info = env.step(actions)
    assert len(rewards) == 10
    assert len(terms) == 10


def test_snapshot_state_exposes_agents():
    from c_env.cs2_env import make_env

    env = make_env(seed=0, auto_reset=False)
    env.reset()
    state = env.snapshot_state()
    assert len(state.agents) == 10
    assert state.agents[0].pos.shape == (3,)


def test_make_env_honors_external_buffers():
    import gymnasium
    import pufferlib

    from c_env.cs2_env import make_env

    single_action = gymnasium.spaces.MultiDiscrete([9, 16, 2, 2, 3, 2, 2])
    joint_action = pufferlib.spaces.joint_space(single_action, 10)
    buf = {
        "observations": np.zeros((10, OBS_DIM), dtype=np.float32),
        "rewards": np.zeros(10, dtype=np.float32),
        "terminals": np.zeros(10, dtype=bool),
        "truncations": np.zeros(10, dtype=bool),
        "masks": np.ones(10, dtype=bool),
        "actions": np.zeros(joint_action.shape, dtype=np.int32),
    }

    env = make_env(seed=0, buf=buf)
    obs, _ = env.reset()
    assert obs is buf["observations"]

    actions = np.zeros((10, ACTION_DIM), dtype=np.int32)
    obs, rewards, terms, truncs, _ = env.step(actions)
    assert obs is buf["observations"]
    assert rewards is buf["rewards"]
    assert terms is buf["terminals"]
    assert truncs is buf["truncations"]
