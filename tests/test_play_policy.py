# tests/test_play_policy.py
import numpy as np
import pytest
from play_actions import (
    play_fill_actions,
    play_mark_done,
    play_reset_round,
    area_bounds_from_simple_rooms,
    resolve_policy_path,
)


def test_fill_then_human_same_buffers_to_step():
    act_buf = np.zeros((10, 7), np.int32)
    cont_buf = np.zeros((10, 2), np.float32)
    pa = np.zeros((10, 7), np.int32)
    pa[:, 0] = 1
    pc = np.stack([np.arange(10, dtype=np.float32), -np.arange(10, dtype=np.float32)], axis=1)
    seen = {}

    def stub_step(actions, cont):
        seen["act_id"] = id(actions)
        seen["cont_id"] = id(cont)

    play_fill_actions(act_buf, cont_buf, pa, pc)
    act_buf[0] = 7
    stub_step(act_buf, cont_buf)
    assert seen["act_id"] == id(act_buf) and seen["cont_id"] == id(cont_buf)
    assert act_buf[0].tolist() == [7] * 7
    assert np.all(act_buf[1:, 0] == 1)
    assert np.array_equal(cont_buf[1:], pc[1:])


def test_spectate_keeps_all_policy_rows():
    act_buf = np.zeros((10, 7), np.int32)
    cont_buf = np.zeros((10, 2), np.float32)
    pa = np.ones((10, 7), np.int32)
    pc = np.zeros((10, 2), np.float32)
    play_fill_actions(act_buf, cont_buf, pa, pc)
    # human_idx < 0: do not write row 0
    assert np.array_equal(act_buf, pa)


def test_missing_checkpoint_raises():
    with pytest.raises(FileNotFoundError):
        resolve_policy_path("/no/such/cs2rl-policy.pt")


def test_area_bounds_match_simple_rooms():
    b = area_bounds_from_simple_rooms()
    assert b.shape == (17, 4) and b.dtype == np.float32
    assert b[0].tolist() == [0.0, 416.0, 256.0, 672.0]


class _TinyPol:
    hidden_size = 4

    def forward_eval(self, x, state):
        state["lstm_h"] = state["lstm_h"] + 1
        return None


def test_mark_done_is_torch_float_tensor():
    import torch
    from train import init_policy_state
    policy = _TinyPol()
    st = init_policy_state(policy, "cpu")
    terms = np.zeros(10, dtype=np.bool_)
    truncs = np.zeros(10, dtype=np.bool_)
    terms[0] = True
    play_mark_done(st, terms, truncs)
    assert hasattr(st["done"], "float")
    assert st["done"].dtype == torch.float32
    assert float(st["done"][0]) == 1.0


def test_reset_round_zeros_hidden():
    import torch
    from train import init_policy_state
    policy = _TinyPol()
    st = init_policy_state(policy, "cpu")
    st["lstm_h"] += 3
    class _Env:
        def reset(self, seed=None):
            return None, None
    st2 = play_reset_round(_Env(), policy, "cpu")
    assert torch.count_nonzero(st2["lstm_h"]) == 0
