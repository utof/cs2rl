"""Continuous aim crosses the vector boundary for Serial and shared-memory backends."""
import numpy as np
import pytest


def test_serial_forwards_each_env_slice_and_discrete_actions():
    from cs2rl.trainer import HybridAimVecEnv

    class Env:
        num_agents = 2

        def __init__(self):
            self.received = None

        def step(self, actions, continuous_actions=None):
            self.received = (actions.copy(),
                             continuous_actions.copy() if continuous_actions is not None else None)

    class Serial:

        def __init__(self):
            self.envs = [Env(), Env()]
            self.driver_env = self.envs[0]
            self.marker = object()

        def send(self, actions):
            for i, env in enumerate(self.envs):
                env.step(actions[2 * i:2 * i + 2])

    backend = Serial()
    vecenv = HybridAimVecEnv(backend)
    actions = np.arange(28, dtype=np.int32).reshape(4, 7)
    continuous = np.arange(8, dtype=np.float32).reshape(4, 2)
    vecenv.send((actions, continuous))
    assert vecenv.marker is backend.marker
    for i, env in enumerate(backend.envs):
        assert env.received is not None
        np.testing.assert_array_equal(env.received[0], actions[2 * i:2 * i + 2])
        np.testing.assert_array_equal(env.received[1], continuous[2 * i:2 * i + 2])
    vecenv.send(actions)
    for env in backend.envs:
        assert env.received is not None
        assert env.received[1] is None


def test_shared_view_is_float32_zeroed_for_bare_send_and_validates_shape():
    from cs2rl.trainer import HybridAimVecEnv

    class Backend:

        def __init__(self):
            self.sent = None

        def send(self, actions):
            self.sent = actions

    backend = Backend()
    view = np.full((4, 2), 9, dtype=np.float32)
    vecenv = HybridAimVecEnv(backend, cont_action_view_main=view)
    actions = np.zeros((4, 7), dtype=np.int32)
    vecenv.send((actions, np.ones((4, 2), dtype=np.float64)))
    assert backend.sent is actions
    assert np.all(view == 1)
    assert view.dtype == np.float32
    vecenv.send(actions)
    assert np.all(view == 0)
    with pytest.raises(AssertionError, match='cont_action.size'):
        vecenv.send((actions, np.ones((3, 2), dtype=np.float32)))
