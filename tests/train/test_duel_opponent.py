"""``--opponent duel`` (Rung 1c): the opponent team plays #152 L2's duel bot, which shoots back.

The mode is a SCRIPTED opponent like ``noop`` and ``walker``, so the self-play guard, the
participation vector, the ``--timesteps`` budget and the run-log lines treat it the same;
each has a test here that fails when only that site drops ``duel``. Its own sites: the
level mix (``--duel-mix``, recorded in config.json only for duel runs), the env guard (the
bot runs only in the env #152 L2 certified it in), ``DuelOpponent`` (per-episode levels and
per-level outcome counts), the trainer override ``Cs2PuffeRL._duel_scripted_opponents``
(checked against a twin ``DuelOpponent`` here, and against a reference ``DuelBot`` on a
real rollout in the training tier), and the per-level counters in the logged row.
"""
import types
from typing import Any

import numpy as np
import pytest
import torch

from cs2rl.eval.duel_bot import H_SHOOT, PROFILES, DuelBot, DuelBotParams, parse_params
from cs2rl.spec.action import ACTION_DIM, ACTION_MASK_DIM, AIM_DIM
from cs2rl.spec.obs import OBS_DIM
from cs2rl.train.config import (
    DEFAULT_DUEL_MIX,
    OPPONENT_MODES,
    SCRIPTED_OPPONENT_MODES,
    assert_duel_env_supported,
    assert_opponent_self_play_compatible,
    build_participating_rows,
    build_train_config,
    compute_batch_dims,
    parse_duel_mix,
    resolve_opponent_mode,
)
from cs2rl.train.duel_opponent import OUTCOMES, DuelOpponent, duel_rngs

# The env #152 L2 certified the bot in (cs2rl.experiment.oracle_statue's constants).
DUEL_ENV: dict[str, Any] = dict(n_active_per_team=1, crouch_enabled=0, jump_enabled=0)
# Arena-duel's numbers (DuelBot.for_env); any values serve the tests that use them.
GEOMETRY = dict(max_turn_speed=0.7853981852531433,
                half_x=300.0,
                half_y=200.0,
                aim_dz=2.3718490600585938,
                move_speed=220.0)
TWO_LEVELS = "hard,style=adad:0.5+easy,style=run:0.5"


def _cfg(opponent, **kw):
    args = types.SimpleNamespace(device="cpu",
                                 seed=1,
                                 timesteps=1_000_000,
                                 checkpoint_dir="/tmp/x",
                                 opponent=opponent,
                                 **kw)
    _, bptt, bs = compute_batch_dims(256)
    return build_train_config(args, batch_size=bs, bptt_horizon=bptt)


def test_duel_is_a_scripted_mode_and_the_resolver_accepts_it():
    assert OPPONENT_MODES == ("self", "noop", "walker", "duel")
    assert SCRIPTED_OPPONENT_MODES == ("noop", "walker", "duel")
    assert resolve_opponent_mode(types.SimpleNamespace(opponent="duel")) == "duel"


def test_the_self_play_guard_refuses_duel_like_walker():
    assert_opponent_self_play_compatible("duel", False)
    with pytest.raises(ValueError, match="--opponent duel requires --no-self-play"):
        assert_opponent_self_play_compatible("duel", True)


def test_duel_participation_is_the_hero_team_only_like_noop():
    for hero in ("t", "ct"):
        duel = build_participating_rows(5, 1, "duel", hero)
        assert np.array_equal(duel, build_participating_rows(5, 1, "noop", hero))
        assert duel.sum() == 5
    assert build_participating_rows(5, 1, "duel", "t").sum() < \
        build_participating_rows(5, 1, "self", "t").sum()


def test_duel_budget_is_the_hero_steps_alone_and_the_mix_is_in_the_config():
    """Without the budget site the run ends at half the requested hero steps with exit 0."""
    duel = _cfg("duel", **DUEL_ENV)
    assert duel["total_timesteps"] == 10_000_000 == _cfg("noop", **DUEL_ENV)["total_timesteps"]
    assert duel["opponent"] == "duel"
    assert duel["duel_mix"] == parse_duel_mix(DEFAULT_DUEL_MIX)
    assert _cfg("duel", duel_mix="hard,style=stop:1", **DUEL_ENV)["duel_mix"] == \
        [["hard,style=stop", 1.0]]
    for other in ("self", "noop", "walker"):
        assert "duel_mix" not in _cfg(other, **DUEL_ENV), other


@pytest.mark.parametrize("other", ["self", "noop", "walker"])
def test_a_duel_mix_without_opponent_duel_is_refused(other):
    with pytest.raises(ValueError, match=f"--duel-mix needs --opponent duel .*{other}"):
        _cfg(other, duel_mix="easy:1", **DUEL_ENV)


@pytest.mark.parametrize(
    "text,match",
    [
        ("easy", "is not SPEC:WEIGHT"),
        (":1", "is not SPEC:WEIGHT"),
        ("nobody:1", "unknown bot 'nobody'"),
        ("easy,aim=1:1", "unknown dial 'aim'"),
        ("easy,style=zigzag:1", "style must be one of"),
        ("easy:x", "could not convert"),
        ("easy:0.5", "sum to 1"),
        ("easy:0.5+hard:0.5000001", "sum to 1"),                       # numpy's choice would refuse it later
        ("easy:0+hard:1", "positive"),
        ("easy:-0.5+hard:1.5", "positive"),
        ("easy:nan", "positive"),
    ])
def test_parse_duel_mix_refuses_a_bad_mix(text, match):
    with pytest.raises(ValueError, match=match):
        parse_duel_mix(text)


def test_parse_duel_mix_keeps_the_order_and_the_specs():
    assert parse_duel_mix(TWO_LEVELS) == [["hard,style=adad", 0.5], ["easy,style=run", 0.5]]
    assert [w for _, w in parse_duel_mix(DEFAULT_DUEL_MIX)] == [0.4, 0.3, 0.1, 0.2]


def test_parse_params_is_oracle_duels_bot_grammar():
    """The copy in eval.duel_bot (train/ may not import experiment/) must not drift."""
    from cs2rl.experiment.oracle_duel import parse_bot

    specs = [
        "perfect", *PROFILES, "easy,style=run", "hard,reaction_ticks=3,lead=0",
        "elite,style=adad,strafe_ticks=4,focus_initial_deg=1.5", "perfect,lead=1"
    ]
    for spec in specs:
        assert parse_params(spec) == parse_bot(spec), spec
    assert parse_params("hard,lead=0").lead is False and parse_params("perfect") == DuelBotParams()
    for bad in ("walker", "nobody", "easy,aim=1"):
        with pytest.raises(ValueError):
            parse_params(bad)


@pytest.mark.parametrize("knob,value", [("n_active_per_team", 2), ("crouch_enabled", 1),
                                        ("jump_enabled", 1)])
def test_the_duel_env_guard_refuses_each_uncertified_knob(knob, value):
    from cs2rl.env.config import EnvConfig

    assert_duel_env_supported(EnvConfig(**DUEL_ENV))
    with pytest.raises(ValueError, match=f"'{knob}': {value}"):
        assert_duel_env_supported(EnvConfig(**{**DUEL_ENV, knob: value}))
    with pytest.raises(ValueError, match="--opponent duel requires --n-active-per-team 1"):
        _cfg("duel", **{**DUEL_ENV, knob: value})
    _cfg("walker", **{**DUEL_ENV, knob: value})        # the guard is duel's alone


def test_the_duel_env_guard_is_oracle_duels_env():
    """The certified knobs are oracle_statue's, the env every #152 L2 measurement ran."""
    from cs2rl.experiment import oracle_statue

    assert DUEL_ENV == dict(n_active_per_team=oracle_statue.N_ACTIVE_PER_TEAM,
                            crouch_enabled=oracle_statue.CROUCH_ENABLED,
                            jump_enabled=oracle_statue.JUMP_ENABLED)


def test_the_run_log_names_the_duel_bot_and_its_levels(capsys):
    from cs2rl.train.loop import _print_opponent_setup
    from cs2rl.train.selfplay import SelfPlayManager

    trainer = types.SimpleNamespace(_participating_rows_np=np.arange(10) < 1,
                                    _self_play_mgr=SelfPlayManager(opponent_mode="duel"))
    plan: Any = types.SimpleNamespace(opponent_mode="duel",
                                      config=dict(total_timesteps=100,
                                                  participating_timesteps=10,
                                                  duel_mix=parse_duel_mix(TWO_LEVELS)))
    _print_opponent_setup(trainer, plan, self_play_enabled=False)
    out = capsys.readouterr().out
    assert "(--opponent duel), not the current policy" in out, out
    assert "Opponent mode 'duel': team CT is the scripted duel bot mix; 1 of 10 agent rows" in out
    assert ("Duel bot levels (metrics environment/duel/L<i>/<outcome>): L0 hard,style=adad 0.5; "
            "L1 easy,style=run 0.5") in out, out


# ── DuelOpponent ────────────────────────────────────────────────────────────


class _RecordingBot(DuelBot):
    """A DuelBot that records which rows ``configure`` and ``reset`` touched."""

    def __init__(self, *a, **kw):
        self.calls: list = []
        super().__init__(*a, **kw)
        self.calls.clear()             # drop the constructor's own calls

    def configure(self, rows, params):
        self.calls.append(("configure", np.arange(self.n_rows)[rows].tolist(), params))
        super().configure(rows, params)

    def reset(self, rows=None):
        self.calls.append(("reset", None if rows is None else np.asarray(rows).tolist()))
        super().reset(rows)


def _opponent(n_envs=3, mix=TWO_LEVELS, hero_team=0, seed=0, bot_cls=DuelBot):
    bot_rng, level_rng = duel_rngs(seed)
    mix = parse_duel_mix(mix)
    bot = bot_cls(5 * n_envs, bot_rng, parse_params(mix[0][0]), **GEOMETRY)
    return DuelOpponent(bot, mix, rows_per_env=5, hero_team=hero_team, rng=level_rng)


def _tick(n_rows, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, (n_rows, OBS_DIM)), rng.random((n_rows, ACTION_MASK_DIM)) > 0.3


def _done(n_envs, ended):
    done = np.zeros((n_envs, 5), dtype=bool)
    done[list(ended)] = True
    return done.ravel()


def test_rounds_count_against_the_level_of_the_env_that_ended():
    opp = _opponent()
    opp.level[:] = [0, 1, 1]
    obs, mask = _tick(15)
    # Per-tick infos without a winner (the harness's) sit between the terminal ones.
    infos = [{"winner": 0}, {"kills_t": 1}, {"winner": -2}]
    opp.step(obs, mask, _done(3, [0, 2]), infos)
    want = np.zeros((2, len(OUTCOMES)), dtype=np.int64)
    want[0, OUTCOMES.index("win")] = 1                 # env 0, level 0, T (the hero) won
    want[1, OUTCOMES.index("double")] = 1              # env 2, level 1, both died
    assert np.array_equal(opp.pop_counts(), want)
    assert not opp.pop_counts().any(), "pop_counts must reset the counts"


@pytest.mark.parametrize("hero_team,winner,outcome", [(0, 0, "win"), (0, 1, "loss"), (1, 1, "win"),
                                                      (1, 0, "loss"), (1, -1, "timeout"),
                                                      (1, -2, "double")])
def test_the_hero_team_decides_win_and_loss(hero_team, winner, outcome):
    opp = _opponent(n_envs=1, hero_team=hero_team)
    opp.level[:] = 1
    obs, mask = _tick(5)
    opp.step(obs, mask, _done(1, [0]), [{"winner": winner}])
    counts = opp.pop_counts()
    assert counts[1, OUTCOMES.index(outcome)] == 1 and counts.sum() == 1


def test_a_terminal_info_count_that_disagrees_with_the_done_envs_is_refused():
    opp = _opponent()
    obs, mask = _tick(15)
    with pytest.raises(AssertionError, match="2 terminal infos for 1 finished envs"):
        opp.step(obs, mask, _done(3, [1]), [{"winner": 0}, {"winner": 1}])
    with pytest.raises(AssertionError, match="0 terminal infos for 1 finished envs"):
        opp.step(obs, mask, _done(3, [1]), [{"kills_t": 1}])


def test_an_unknown_winner_is_refused():
    opp = _opponent(n_envs=1)
    obs, mask = _tick(5)
    with pytest.raises(ValueError, match="unknown winner 7"):
        opp.step(obs, mask, _done(1, [0]), [{"winner": 7}])


def test_only_the_ended_envs_draw_a_level_and_reset():
    """Risk (b): a row's bot state resets with its env's episode, never with the tick."""
    opp = _opponent(n_envs=4,
                    mix="hard:0.25+easy:0.25+fair:0.25+expert:0.25",
                    seed=3,
                    bot_cls=_RecordingBot)
    bot: Any = opp.bot
    bot.calls.clear()                  # the construction's draw
    obs, mask = _tick(20)
    levels = []
    for _ in range(40):
        before = opp.level.copy()
        opp.step(obs, mask, _done(4, []), [])
        assert not bot.calls and np.array_equal(opp.level, before)
        opp.step(obs, mask, _done(4, [1, 3]), [{"winner": 0}, {"winner": 1}])
        (kind_c, rows_c, params), (kind_r, rows_r) = bot.calls
        assert kind_c == "configure" and kind_r == "reset"
        assert rows_c == rows_r == [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]
        assert params == [opp.params[opp.level[1]]] * 5 + [opp.params[opp.level[3]]] * 5
        assert opp.level[0] == before[0] and opp.level[2] == before[2]
        levels.append(opp.level[[1, 3]].copy())
        bot.calls.clear()
    assert len(np.unique(np.concatenate(levels))) == 4, "every level of the mix must be drawn"


def test_the_row_count_is_fixed():
    opp = _opponent(n_envs=2)
    obs, mask = _tick(15)
    with pytest.raises(ValueError, match="15 opponent rows, the bot has 10"):
        opp.step(obs, mask, np.zeros(15, dtype=bool), [])
    with pytest.raises(ValueError, match="not whole envs of 5"):
        DuelOpponent(DuelBot(7, np.random.default_rng(0), DuelBotParams(), **GEOMETRY),
                     parse_duel_mix("easy:1"),
                     rows_per_env=5,
                     hero_team=0,
                     rng=np.random.default_rng(0))


def test_the_bot_stream_is_the_first_spawned_generators():
    """The bot draws as a DuelBot seeded with duel_rngs(seed)[0] alone would."""
    a, b = duel_rngs(11), duel_rngs(11)
    assert a[0].random() == b[0].random() and a[0].random() != a[1].random()
    assert duel_rngs(12)[0].random() != duel_rngs(11)[0].random()


# ── the trainer override, on a fake trainer ─────────────────────────────────


def _fake_trainer(seed, n_envs, mix=TWO_LEVELS):
    from cs2rl.train.selfplay import SelfPlayManager
    from cs2rl.train.trainer import Cs2PuffeRL

    t = types.SimpleNamespace(config=dict(seed=seed, device="cpu"),
                              _self_play_mgr=SelfPlayManager(opponent_mode="duel"))
    t._freeze_statue_opponents = types.MethodType(Cs2PuffeRL._freeze_statue_opponents, t)
    t.duel = types.MethodType(Cs2PuffeRL._duel_scripted_opponents, t)
    t._opponent_duel = _opponent(n_envs=n_envs, mix=mix, seed=seed)
    return t


def _fake_step(n):
    ones = lambda *shape: torch.ones(*shape, dtype=torch.float32)                     # noqa: E731
    return types.SimpleNamespace(action=torch.ones(n, ACTION_DIM, dtype=torch.int32),
                                 cont_action=ones(n, AIM_DIM),
                                 logprob=ones(n),
                                 logprob_d=ones(n),
                                 logprob_c=ones(n),
                                 value=ones(n))


def test_the_trainer_override_writes_the_duel_opponents_actions_on_the_opponent_rows_only():
    """Risk (a): each opponent row's action is the bot's on THAT row's obs of THIS tick."""
    n_envs, n = 4, 40
    t = _fake_trainer(5, n_envs)
    ref = _opponent(n_envs=n_envs, seed=5)
    opp = np.arange(n) % 10 >= 5
    opp_t = torch.as_tensor(opp)
    rng = np.random.default_rng(9)
    acted = 0
    for k in range(30):
        obs, mask = _tick(n, seed=k)
        done = np.zeros(n, dtype=np.float32)
        infos: list = []
        if k % 7 == 6:
            done[(np.arange(n) // 10) == k % n_envs] = 1.0
            infos = [{"winner": int(rng.integers(-2, 2))}]
        step = _fake_step(n)
        t.duel(step, torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(done),
               torch.as_tensor(mask), infos)
        want_a, want_c = ref.step(obs[opp].astype(np.float32), mask[opp], done[opp] > 0, infos)
        assert np.array_equal(step.action[opp_t].numpy(), want_a), k
        assert np.array_equal(step.cont_action[opp_t].numpy(), want_c), k
        assert (step.action[~opp_t] == 1).all() and (step.cont_action[~opp_t] == 1).all()
        for buf in (step.logprob, step.logprob_d, step.logprob_c, step.value):
            assert not buf[opp_t].any() and (buf[~opp_t] == 1).all()
        acted += int(want_a.any(axis=1).sum())
    assert acted, "the bot must act on some row, or the comparison pins nothing"
    assert np.array_equal(t._opponent_duel.pop_counts(), ref.pop_counts())


def test_the_trainers_bot_geometry_is_oracle_duels():
    """Risk (d): the trainer's bot reads its numbers off its own driver env; they must be
    those of the env every #152 L2 measurement ran (oracle_statue.build_env)."""
    from cs2rl.env.map import make_arena_duel_map
    from cs2rl.experiment.oracle_statue import build_env
    from tests._helpers.trainer_harness import _build_trainer_for_test

    trainer, cleanup = _build_trainer_for_test(num_envs=4,
                                               opponent="duel",
                                               map_data=make_arena_duel_map(),
                                               pin_pitch=1,
                                               **DUEL_ENV)
    ref_env = build_env(0)
    try:
        assert trainer._opponent_duel is not None
        got = trainer._opponent_duel.bot
        want = DuelBot.for_env(ref_env, 1, np.random.default_rng(0), DuelBotParams())
        for name in ("mts", "half", "map_diag", "aim_dz", "vmax", "centre"):
            assert getattr(got, name) == getattr(want, name), name
        assert got.n_rows == 20
        assert trainer._opponent_duel.hero == 0, "the hero is team T while CT is the bot"
        drv, ref = trainer.vecenv.driver_env.config, ref_env.config
        for knob in ("n_active_per_team", "pin_pitch", "crouch_enabled", "jump_enabled", "accuracy",
                     "hitboxes"):
            assert getattr(drv, knob) == getattr(ref, knob), knob
    finally:
        ref_env.close()
        cleanup()


def test_the_modal_runner_passes_the_mix_as_one_token():
    from scripts.modal_runner import request

    assert request.LIVE_TRAIN_OPTION_ARITY["--duel-mix"] == 1
    assert request.LIVE_TRAIN_OPTION_ARITY["--opponent"] == 1
    assert request.validate_train_args(
        ["--timesteps", "1000", "--opponent", "duel", "--duel-mix", TWO_LEVELS]) == 1000


# ── real rollouts (training tier) ───────────────────────────────────────────


def _duel_trainer(num_envs, mix=TWO_LEVELS, seed=0):
    from cs2rl.env.map import make_arena_duel_map
    from tests._helpers.trainer_harness import _build_trainer_for_test

    return _build_trainer_for_test(num_envs=num_envs,
                                   seed=seed,
                                   opponent="duel",
                                   duel_mix=mix,
                                   map_data=make_arena_duel_map(),
                                   pin_pitch=1,
                                   **DUEL_ENV)


@pytest.mark.training
def test_trainer_duel_rows_equal_a_reference_bot_across_rollout_boundaries():
    """I-Q1 (a) and risks (a), (b): over three back-to-back rollouts, every head and the aim
    the env receives for an opponent row equal a reference DuelBot's on the recorded obs and
    masks, the bot resetting a row only on its env's done flag. Recorded at the vector env's
    recv and send, not from DuelOpponent."""
    trainer, cleanup = _duel_trainer(4)
    try:
        vec = trainer.vecenv
        duel = trainer._opponent_duel
        assert duel is not None
        n = trainer.total_agents
        opp = np.flatnonzero(np.arange(n) % 10 >= 5)
        live = np.flatnonzero(np.arange(n) % 10 == 5)
        mix = trainer.config["duel_mix"]
        params = [parse_params(s) for s, _ in mix]
        ref = DuelBot.for_env(vec.driver_env, len(opp), duel_rngs(0)[0], params[0])
        ticks: list = []
        real_recv, real_send = vec.recv, vec.send

        def recv():
            out = real_recv()
            ticks.append(
                dict(obs=np.array(out[0], copy=True),
                     done=np.array(out[2], copy=True),
                     mask=np.array(trainer._action_mask_view_main, copy=True)))
            return out

        def send(pair):
            action, cont = pair
            ticks[-1].update(action=np.array(action, copy=True),
                             cont=cont.cpu().numpy().astype(np.float32),
                             level=duel.level.copy())
            return real_send(pair)

        def configure(envs, levels):
            idx = np.flatnonzero(envs)
            rows = (idx[:, None] * 5 + np.arange(5)).ravel()
            ref.configure(rows, [params[k] for k in np.repeat(levels[idx], 5)])
            ref.reset(rows)

        vec.recv, vec.send = recv, send
        configure(np.ones(len(duel.level), dtype=bool), duel.level.copy())
        firsts = []
        try:
            for _ in range(3):
                firsts.append(len(ticks))
                trainer.evaluate()
        finally:
            vec.recv, vec.send = real_recv, real_send
        for t in ticks:
            ended = t["done"][opp].reshape(-1, 5).any(axis=1)
            if ended.any():
                configure(ended, t["level"])
            act, cont = ref.step_obs(t["obs"][opp], t["mask"][opp])
            assert np.array_equal(act, t["action"][opp])
            assert np.array_equal(cont, t["cont"][opp])
        rounds = sum(int(t["done"][live].astype(bool).sum()) for t in ticks)
        crossing = sum(int((~ticks[i]["done"][live].astype(bool)).sum()) for i in firsts[1:])
        fired = sum(int(t["action"][live, H_SHOOT].sum()) for t in ticks)
        assert rounds >= 1, "no round ended: the per-episode reset is not exercised"
        assert crossing >= 1, "no episode crossed a rollout boundary: risk (b) is not pinned"
        assert fired >= 1, "the bot never fired"
    finally:
        cleanup()


@pytest.mark.training
def test_duel_opponent_rows_play_legal_actions_and_do_not_participate():
    """I-Q4 in the training tier: a real rollout and update with ``--opponent duel``."""
    from cs2rl.policy import _MASK_HEAD_SLICES

    trainer, cleanup = _duel_trainer(16)
    try:
        bs = trainer.config["batch_size"]
        trainer.evaluate()
        slot = np.arange(trainer.total_agents) % 10
        opp = torch.as_tensor(slot >= 5)
        hero = torch.as_tensor(slot == 0)
        live = torch.as_tensor(slot == 5)
        assert trainer.global_step == bs // 10
        assert not trainer.participating[opp].any() and trainer.participating[hero].all()
        for buf in (trainer.logprobs, trainer.logprobs_d, trainer.logprobs_c, trainer.values):
            assert not buf[opp].any()
        for h, (lo, hi) in enumerate(_MASK_HEAD_SLICES):
            a = trainer.actions[..., h].long().unsqueeze(-1)
            legal = torch.gather(trainer.action_masks[..., lo:hi].long(), -1, a).squeeze(-1)
            assert (legal[opp] != 0).all(), f"a duel row stored a mask-illegal bin on head {h}"
        assert trainer.actions[live][..., H_SHOOT].sum() > 0, "the live bot rows must fire"
        assert (trainer.cont_actions[live] != 0).any(), "the live bot rows must aim"
        trainer.last_log_time = 0.0
        logs = trainer.train()
        assert logs is not None
        keys = {f"environment/duel/L{i}/{o}" for i in range(2) for o in OUTCOMES}
        assert keys <= logs.keys(), sorted(k for k in logs if "duel" in k)
        assert sum(logs[k] for k in keys) == logs["environment/episodes"] > 0
    finally:
        cleanup()
