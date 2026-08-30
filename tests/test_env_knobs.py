"""R0-G (Rung 0 spec 2026-08-29): env knobs via CLI.

`--round-time-ticks` / `--laser-range` / `--max-turn-speed` flow
CLI → env_knobs_from_args → make_puffer_env → make_env → binding.init FMT
(positions 23-24 / 30 / 43). None ⇒ nav.py constant, so every caller that
does not pass a knob keeps today's values (fingerprints unchanged at default).

PITFALL (Task 14): the exact-key assertion in test_env_knobs_from_args_and_config
is deliberate — a new knob added to env_knobs_from_args must be added here too.
"""
import numpy as np
import pytest

N_AGENTS, ACTION_DIM, AIM_DIM = 10, 7, 2


def test_round_time_ticks_sets_episode_length(simple_map):
    from c_env.cs2_env import make_env
    env = make_env(map_data=simple_map, round_time=160, seed=1, auto_reset=False)
    try:
        assert env.round_time == 160
        env.reset()
        a = np.zeros((N_AGENTS, ACTION_DIM), np.int32)
        c = np.zeros((N_AGENTS, AIM_DIM), np.float32)
        t = 0
        while True:
            _, _, term, trunc, _ = env.step(a, c)
            t += 1
            if term.any() or trunc.any():
                break
            assert t < 400
        assert t == 160
    finally:
        env.close()


def test_laser_and_turn_speed_reach_static_data(simple_map):
    import binding

    from c_env.cs2_env import make_env
    env = make_env(map_data=simple_map, laser_range=300.0, max_turn_speed=0.5, seed=1)
    try:
        sc = binding.static_data_scalars(env._capsule)
        assert sc["laser_range"] == pytest.approx(300.0)
        # laser_range_sq is not a kwarg: it must be derived from the SAME value
        # (FMT 23-24 disagreeing would make range checks and damage falloff disagree).
        assert sc["laser_range_sq"] == pytest.approx(90000.0)
        assert sc["max_turn_speed"] == pytest.approx(0.5)
        assert sc["round_time"] == env.round_time
    finally:
        env.close()


def test_default_knobs_match_nav_constants(simple_map):
    """None (the default) must resolve to the nav.py constants, not 0 / garbage."""
    import binding

    import nav
    from c_env.cs2_env import make_env
    env = make_env(map_data=simple_map, seed=1)
    try:
        sc = binding.static_data_scalars(env._capsule)
        assert sc["round_time"] == nav.ROUND_TIME == env.round_time
        assert sc["laser_range"] == pytest.approx(nav.LASER_RANGE)
        assert sc["laser_range_sq"] == pytest.approx(nav.LASER_RANGE * nav.LASER_RANGE)
        assert sc["max_turn_speed"] == pytest.approx(nav.MAX_TURN_SPEED_RAD)
    finally:
        env.close()


@pytest.mark.parametrize("bad", [
    dict(round_time=0),
    dict(round_time=-5),
    dict(round_time=2.5),
    dict(laser_range=0.0),
    dict(laser_range=-1.0),
    dict(max_turn_speed=0.0),
    dict(max_turn_speed=-0.1),
])
def test_invalid_knobs_raise_value_error(simple_map, bad):
    """Validation happens in Python BEFORE binding.init: a C-side assert would
    abort a forked Puffer worker with no traceback."""
    from c_env.cs2_env import make_env
    with pytest.raises(ValueError):
        make_env(map_data=simple_map, seed=1, **bad)


def test_env_knobs_from_args_and_config():
    import types

    from train import build_train_config, compute_batch_dims, env_knobs_from_args
    args = types.SimpleNamespace(device="cpu",
                                 seed=1,
                                 timesteps=100_000,
                                 checkpoint_dir="/tmp/x",
                                 round_time_ticks=160,
                                 laser_range=300.0,
                                 max_turn_speed=None,
                                 n_active_per_team=1,
                                 pin_pitch=1,
                                 crouch_enabled=0,
                                 gamma=0.999,
                                 pbrs_gamma=None)
    k = env_knobs_from_args(args)
    assert k == {
        "n_active_per_team": 1,
        "pin_pitch": 1,
        "crouch_enabled": 0,
        "round_time": 160,
        "laser_range": 300.0
    }                                  # None knobs omitted ⇒ env default
    _, bptt, bs = compute_batch_dims(16)
    cfg = build_train_config(args, batch_size=bs, bptt_horizon=bptt)
    for key in ("round_time_ticks", "laser_range", "max_turn_speed"):
        assert key in cfg
    assert cfg["round_time_ticks"] == 160 and cfg["laser_range"] == 300.0
    assert cfg["max_turn_speed"] is None


def test_env_knobs_from_args_legacy_args_object():
    """Harness/dump-config args objects predate the flags: all three omitted."""
    import types

    from train import env_knobs_from_args
    k = env_knobs_from_args(types.SimpleNamespace())
    assert set(k) == {"n_active_per_team", "pin_pitch", "crouch_enabled"}


def test_make_puffer_env_forwards_knobs(simple_map):
    import binding

    from train import make_puffer_env
    env = make_puffer_env(map_data=simple_map,
                          round_time=160,
                          laser_range=300.0,
                          max_turn_speed=0.5)
    try:
        sc = binding.static_data_scalars(env._capsule)
        assert sc["round_time"] == 160
        assert sc["laser_range"] == pytest.approx(300.0)
        assert sc["max_turn_speed"] == pytest.approx(0.5)
    finally:
        env.close()


def test_policy_max_turn_speed_assert(simple_map):
    from train import assert_max_turn_speed_agreement
    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=4, map_data=simple_map)
    try:
        assert_max_turn_speed_agreement(trainer.vecenv, trainer.policy)
        trainer.policy.max_turn_speed.fill_(0.123)
        with pytest.raises(RuntimeError):
            assert_max_turn_speed_agreement(trainer.vecenv, trainer.policy)
    finally:
        cleanup()


def test_cli_flags_declared_default_none():
    """The parser is built inline under ``if __name__ == "__main__"`` (not
    importable), so check the source: each flag is declared, defaults to None
    (⇒ env default) and has the matching dest. The modal arity mirror is
    covered by tests/test_modal_runner.py."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "train.py").read_text()
    for flag, dest, typ in (("--round-time-ticks", "round_time_ticks", "int"),
                            ("--laser-range", "laser_range", "float"), ("--max-turn-speed",
                                                                        "max_turn_speed", "float")):
        m = re.search(rf'add_argument\("{flag}",(.*?)\)\n', src, re.S)
        assert m, flag
        body = m.group(1)
        assert f"type={typ}" in body and "default=None" in body and f'dest="{dest}"' in body, flag
