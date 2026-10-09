"""``--opponent walker`` (Rung 1b): the opponent team plays scripted walkers, not a statue.

The mode is a SCRIPTED opponent like ``noop``, so everything that treats ``noop`` as
"the opponent team does not train" must treat ``walker`` the same: the self-play guard,
the participation vector, the ``--timesteps`` budget and the run-log lines. Each of those
sites has a test here that fails when only that site is reverted to ``== "noop"``. The
rollout override itself (``Cs2PuffeRL._walk_scripted_opponents``) is checked against
``cs2rl.eval.walker`` directly, and end to end in the training tier.
"""
import types
from typing import Any

import numpy as np
import pytest
import torch

from cs2rl.eval.walker import H_MOVE, TRAIN_MIX, TRAIN_WEIGHTS, RandomWalker
from cs2rl.spec.action import ACTION_DIM, AIM_DIM
from cs2rl.train.config import (
    OPPONENT_MODES,
    SCRIPTED_OPPONENT_MODES,
    assert_opponent_self_play_compatible,
    build_participating_rows,
    build_train_config,
    compute_batch_dims,
    resolve_opponent_mode,
)


def test_walker_is_a_scripted_mode_and_the_cli_and_resolver_accept_it():
    assert OPPONENT_MODES == ("self", "noop", "walker")
    assert SCRIPTED_OPPONENT_MODES == ("noop", "walker")
    assert resolve_opponent_mode(types.SimpleNamespace(opponent="walker")) == "walker"


def test_the_self_play_guard_refuses_walker_like_noop():
    assert_opponent_self_play_compatible("walker", False)
    with pytest.raises(ValueError, match="--opponent walker requires --no-self-play"):
        assert_opponent_self_play_compatible("walker", True)


def test_walker_participation_is_the_hero_team_only_like_noop():
    for hero in ("t", "ct"):
        for n_active in (1, 3):
            walker = build_participating_rows(5, n_active, "walker", hero)
            assert np.array_equal(walker, build_participating_rows(5, n_active, "noop", hero))
            assert walker.sum() == 5 * n_active
    assert build_participating_rows(5, 1, "walker", "t").sum() < \
        build_participating_rows(5, 1, "self", "t").sum()


def test_walker_budget_is_the_hero_steps_alone_like_noop():
    """Without the budget site the run ends at half the requested hero steps with exit 0."""

    def cfg(opponent):
        args = types.SimpleNamespace(device="cpu",
                                     seed=1,
                                     timesteps=1_000_000,
                                     checkpoint_dir="/tmp/x",
                                     n_active_per_team=1,
                                     opponent=opponent)
        _, bptt, bs = compute_batch_dims(256)
        return build_train_config(args, batch_size=bs, bptt_horizon=bptt)

    assert cfg("walker")["total_timesteps"] == 10_000_000 == cfg("noop")["total_timesteps"]
    assert cfg("walker")["opponent"] == "walker"
    assert cfg("self")["total_timesteps"] == 5_000_000


@pytest.mark.parametrize("mode,role", [("walker", "a scripted walker mix"),
                                       ("noop", "a stationary statue")])
def test_the_run_log_names_the_scripted_opponent(mode, role, capsys):
    from cs2rl.train.loop import _print_opponent_setup
    from cs2rl.train.selfplay import SelfPlayManager

    trainer = types.SimpleNamespace(_participating_rows_np=np.arange(10) < 1,
                                    _self_play_mgr=SelfPlayManager(opponent_mode=mode))
    plan: Any = types.SimpleNamespace(opponent_mode=mode,
                                      config=dict(total_timesteps=100, participating_timesteps=10))
    _print_opponent_setup(trainer, plan, self_play_enabled=False)
    out = capsys.readouterr().out
    assert f"(--opponent {mode}), not the current policy" in out, out
    assert f"Opponent mode '{mode}': team CT is {role}; 1 of 10 agent rows participate" in out, out


def _fake_trainer(seed, n_agents):
    from cs2rl.train.selfplay import SelfPlayManager
    from cs2rl.train.trainer import Cs2PuffeRL

    t = types.SimpleNamespace(config=dict(seed=seed, device="cpu"),
                              _self_play_mgr=SelfPlayManager(opponent_mode="walker"))
    t._freeze_statue_opponents = types.MethodType(Cs2PuffeRL._freeze_statue_opponents, t)
    t.walk = types.MethodType(Cs2PuffeRL._walk_scripted_opponents, t)
    t.total_agents = n_agents
    t._opponent_walker = types.MethodType(Cs2PuffeRL._build_opponent_walker, t)()
    return t


def _fake_step(n):
    ones = lambda *shape: torch.ones(*shape, dtype=torch.float32)                     # noqa: E731
    return types.SimpleNamespace(action=torch.ones(n, ACTION_DIM, dtype=torch.int32),
                                 cont_action=ones(n, AIM_DIM),
                                 logprob=ones(n),
                                 logprob_d=ones(n),
                                 logprob_c=ones(n),
                                 value=ones(n))


def _roll(seed, ticks=80, n_envs=4, with_done=True):
    """Run the override ``ticks`` times; return the opponent rows' actions and the step."""
    n = 10 * n_envs
    t = _fake_trainer(seed, n)
    opp = np.arange(n) % 10 >= 5
    seen = []
    for k in range(ticks):
        d = torch.zeros(n)
        if with_done and k % 9 == 8:
            d[(torch.arange(n) // 10) % 2 == k % 2] = 1.0
        step = _fake_step(n)
        t.walk(step, d, n)
        seen.append(step.action[torch.as_tensor(opp)].clone())
    return torch.stack(seen), step, opp


def test_trainer_walker_rows_equal_eval_walker_output_for_the_same_seed_and_dones():
    """B-Q1: the trainer drives its opponent rows with exactly eval.walker's moves."""
    n_envs, ticks = 4, 80
    got, step, opp = _roll(seed=5, ticks=ticks, n_envs=n_envs)
    n = 10 * n_envs
    ref = RandomWalker(5 * n_envs, np.random.default_rng(5), mix=TRAIN_MIX, weights=TRAIN_WEIGHTS)
    want = []
    for k in range(ticks):
        if k % 9 == 8:
            done = ((np.arange(n) // 10) % 2 == k % 2)[opp]
            ref.reset(done)
        want.append(ref.step())
    assert np.array_equal(got[:, :, H_MOVE].numpy(), np.array(want))
    # Everything but the move head is the statue's, and the hero rows are untouched.
    other = [h for h in range(ACTION_DIM) if h != H_MOVE]
    assert not got[:, :, other].any()
    assert (step.action[torch.as_tensor(~opp)] == 1).all()
    assert (step.cont_action[torch.as_tensor(~opp)] == 1).all()
    for buf in (step.logprob, step.logprob_d, step.logprob_c, step.value):
        assert not buf[torch.as_tensor(opp)].any() and (buf[torch.as_tensor(~opp)] == 1).all()
    assert not step.cont_action[torch.as_tensor(opp)].any()
    assert (got[:, :, H_MOVE] != 0).any(), "the walkers must walk"


def test_two_seeded_trainers_walk_identically_and_a_done_flag_matters():
    a, _, _ = _roll(seed=3)
    b, _, _ = _roll(seed=3)
    c, _, _ = _roll(seed=4)
    no_done, _, _ = _roll(seed=3, with_done=False)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert not torch.equal(a, no_done), "done flags must start a new episode (re-draw)"


def test_the_walker_cannot_change_its_row_count():
    t = _fake_trainer(0, 20)
    t.walk(_fake_step(20), torch.zeros(20), 20)
    with pytest.raises(AssertionError):
        t.walk(_fake_step(30), torch.zeros(30), 30)


@pytest.mark.training
def test_walker_opponent_rows_walk_and_do_not_participate():
    """B-Q4 in the training tier: a real rollout with ``--opponent walker``."""
    from tests._helpers.trainer_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=16, n_active_per_team=1, opponent="walker")
    try:
        bs = trainer.config["batch_size"]
        assert trainer.config["opponent"] == "walker"
        assert trainer.segments == trainer.total_agents
        trainer.evaluate()
        slot = np.arange(trainer.total_agents) % 10
        opp = torch.as_tensor(slot >= 5)
        hero = torch.as_tensor(slot == 0)
        assert trainer.global_step == bs // 10
        assert not trainer.participating[opp].any() and trainer.participating[hero].all()
        move = trainer.actions[opp][..., H_MOVE]
        assert (move != 0).float().mean() > 0.4, "walker rows must move"
        other = [h for h in range(ACTION_DIM) if h != H_MOVE]
        assert not trainer.actions[opp][..., other].any()
        assert not trainer.cont_actions[opp].any()
        for buf in (trainer.logprobs, trainer.logprobs_d, trainer.logprobs_c, trainer.values):
            assert not buf[opp].any()
        trainer.last_log_time = 0.0
        trainer.train()
        assert trainer.losses[
            "participating_rows"] == trainer.segments * trainer.config["bptt_horizon"] // 10
    finally:
        cleanup()
