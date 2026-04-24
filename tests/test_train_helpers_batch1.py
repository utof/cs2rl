import numpy as np
import torch

from src.train_helpers_batch1 import (
    split_into_channels,
    symexp,
    symlog,
    target_entropy_schedule,
)


def test_symlog_sign_preserving():
    assert symlog(torch.tensor(0.0)).item() == 0.0
    assert symlog(torch.tensor(1.0)).item() > 0.0
    assert symlog(torch.tensor(-1.0)).item() < 0.0


def test_symlog_monotonic():
    x = torch.linspace(-10, 10, 1001)
    y = symlog(x)
    assert torch.all(y[1:] >= y[:-1])


def test_symlog_inverse():
    x = torch.tensor([-5.0, -0.3, 0.0, 0.3, 5.0, 100.0])
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-5)


def test_symlog_near_zero_identity():
    x = torch.tensor([-1e-4, 0.0, 1e-4])
    y = symlog(x)
    # symlog(x) ≈ x for small x
    assert torch.allclose(y, x, atol=1e-8)


def test_target_entropy_schedule_monotone():
    max_ent = 10.0
    steps = [0, 1_000_000, 5_000_000, 10_000_000, 20_000_000]
    vals = [target_entropy_schedule(s, max_ent, warmup_end=10_000_000) for s in steps]
    # Non-increasing from start to base
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    # Start at 0.7 * max_ent, end at 0.5 * max_ent
    assert abs(vals[0] - 0.7 * max_ent) < 1e-6
    assert abs(vals[-1] - 0.5 * max_ent) < 1e-6


def test_target_entropy_schedule_never_exceeds_max():
    max_ent = 10.0
    for s in range(0, 30_000_000, 500_000):
        v = target_entropy_schedule(s, max_ent, warmup_end=10_000_000)
        assert v <= max_ent


def test_split_into_channels_partitioning():
    """Per-agent channel rewards should sum to the total raw reward (pre-norm)."""
    # Synthetic StepStats-like numpy record
    ss = np.zeros(1,
                  dtype=[
                      ("reward_win", "<f4"),
                      ("reward_kills", "<f4"),
                      ("reward_deaths", "<f4"),
                      ("reward_bomb", "<f4"),
                      ("reward_pbrs", "<f4"),
                      ("reward_shots", "<f4"),
                      ("reward_survival", "<f4"),
                      ("reward_inaction", "<f4"),
                      ("win_by_detonation", "<i1"),
                      ("win_by_defuse", "<i1"),
                  ])
    ss["reward_win"] = 5.0
    ss["reward_kills"] = 2.0
    ss["reward_deaths"] = -1.0
    ss["reward_bomb"] = 0.5
    ss["reward_pbrs"] = 0.1
    ss["reward_shots"] = -0.05
    ss["reward_survival"] = 0.2
    ss["reward_inaction"] = 0.0
    ss["win_by_detonation"] = 1
    ss["win_by_defuse"] = 0
    channels = split_into_channels(ss)
    assert set(channels.keys()) == {"combat", "objective", "positional"}
    total_raw = (ss["reward_win"] + ss["reward_kills"] + ss["reward_deaths"] + ss["reward_bomb"] +
                 ss["reward_pbrs"] + ss["reward_shots"] + ss["reward_survival"] +
                 ss["reward_inaction"]).item()
    total_channel = sum(v.item() if hasattr(v, "item") else float(v) for v in channels.values())
    assert abs(total_raw - total_channel) < 1e-5


def test_split_routes_detonation_win_to_objective():
    ss = np.zeros(1,
                  dtype=[
                      ("reward_win", "<f4"),
                      ("reward_kills", "<f4"),
                      ("reward_deaths", "<f4"),
                      ("reward_bomb", "<f4"),
                      ("reward_pbrs", "<f4"),
                      ("reward_shots", "<f4"),
                      ("reward_survival", "<f4"),
                      ("reward_inaction", "<f4"),
                      ("win_by_detonation", "<i1"),
                      ("win_by_defuse", "<i1"),
                  ])
    ss["reward_win"] = 5.0
    ss["win_by_detonation"] = 1
    c = split_into_channels(ss)
    assert c["objective"] >= 5.0
    assert c["combat"] < 0.01


def test_split_routes_elimination_win_to_combat():
    ss = np.zeros(1,
                  dtype=[
                      ("reward_win", "<f4"),
                      ("reward_kills", "<f4"),
                      ("reward_deaths", "<f4"),
                      ("reward_bomb", "<f4"),
                      ("reward_pbrs", "<f4"),
                      ("reward_shots", "<f4"),
                      ("reward_survival", "<f4"),
                      ("reward_inaction", "<f4"),
                      ("win_by_detonation", "<i1"),
                      ("win_by_defuse", "<i1"),
                  ])
    ss["reward_win"] = 3.0
    ss["win_by_detonation"] = 0
    ss["win_by_defuse"] = 0
    c = split_into_channels(ss)
    assert c["combat"] >= 3.0
    assert c["objective"] < 0.01
