"""R0-I (Task 13): fixed-baseline evaluation — random / oracle opponents.

What is pinned here
-------------------
* The vendored oracle beats the random actor on the arena duel map and is
  NEVER blind (every shot it fires has an enemy in LoS — a blind oracle would
  still beat random on a 100 %-LoS arena, so the win rate alone cannot catch
  a broken vis_prev thread).
* ``episode_outcome`` excludes timeouts (``winner_ct`` counts them).
* ``_episode`` at ``n_active_per_team=2``: team-level credit — a kill by the
  OTHER T agent scores a win for the "policy" side. Per-policy credit is an
  n=1-only property; documented, not hidden.
* ``PolicyActor.from_policy`` drives a live policy and the evaluator splices
  the policy's rows over the opponent's.
* The scheduled eval is NOT skipped by PuffeRL's 0.25 s log throttle: eval
  runs on the eval epoch regardless, and its keys ride on the next logged row.
* ``--eval-interval`` is a CLI flag, a config key, and mirrored in the Modal
  runner's arity table.
"""
import time

import numpy as np
import pytest

from cs2rl.env.c.cs2_env import N_AGENTS, TEAM_SIZE
from tests.conftest import REPO_ROOT


def _arena_env(**kw):
    from cs2rl.env.c.cs2_env import make_env
    from cs2rl.env.config import EnvConfig
    from cs2rl.env.map import make_arena_duel_map
    base = dict(n_active_per_team=1,
                pin_pitch=1,
                crouch_enabled=0,
                round_time=160,
                auto_reset=False,
                seed=11)
    base.update(kw)
    typed = {
        k: base.pop(k)
        for k in ("auto_reset", "seed", "buf", "include_step_stats_in_info", "team_spirit",
                  "config") if k in base
    }
    config = typed["config"] if "config" in typed else EnvConfig(**base)
    return make_env(map_data=make_arena_duel_map(),
                    config=config,
                    auto_reset=typed["auto_reset"],
                    seed=typed["seed"],
                    buf=typed.get("buf", None),
                    include_step_stats_in_info=typed.get("include_step_stats_in_info", False),
                    team_spirit=typed.get("team_spirit", 0.0))


def test_oracle_beats_random():
    from cs2rl.eval.baselines import BaselineEvaluator, OracleActor, RandomActor
    env = _arena_env()
    try:
        ev = BaselineEvaluator(env, episodes=40, seed=0)
        rng = np.random.default_rng(0)
        oracle = OracleActor(rng, ev.max_turn_speed, ev.laser_range, ev.nav)
        res = ev.run_pair(oracle, RandomActor(rng, ev.max_turn_speed))
        assert res["win"] > 0.85 and res["kills_per_episode"] > 0.85, res
    finally:
        env.close()


def test_oracle_never_blind():
    """Binding ruling: shots_with_enemy_in_los == shots_fired over 40 episodes.

    Opponent is IdleActor so every shot in episode_stats is the oracle's
    (StepStats are env-wide, not per-agent). Equality only holds if the
    evaluator threads vis_prev correctly — feeding None every tick makes the
    oracle fire at nothing (it would still walk and, on the arena, win).
    """
    from cs2rl.eval.baselines import BaselineEvaluator, IdleActor, OracleActor
    env = _arena_env()
    try:
        ev = BaselineEvaluator(env, episodes=40, seed=0)
        oracle = OracleActor(np.random.default_rng(0), ev.max_turn_speed, ev.laser_range, ev.nav)
        res = ev.run_pair(oracle, IdleActor())
        assert res["shots_fired"] >= 40, res
        assert res["shots_with_enemy_in_los"] == res["shots_fired"], res
        assert res["win"] == 1.0, res
    finally:
        env.close()


def test_win_definition_excludes_timeouts():
    from cs2rl.eval.baselines import episode_outcome
    # (kills_for, kills_against) → win score
    assert episode_outcome(1, 0) == 1.0 and episode_outcome(0, 1) == 0.0
    assert episode_outcome(0, 0) == 0.0                # timeout is NOT a CT win here
    assert episode_outcome(1, 1) == 0.5                # trade (forward-looking)


class _RowActor:
    """Test helper: `inner` drives only `rows`; every other row is a no-op."""

    def __init__(self, inner, rows):
        self.inner, self.rows = inner, rows

    def reset(self):
        self.inner.reset()

    def act(self, obs, st, vis_prev, env):
        a, c = self.inner.act(obs, st, vis_prev, env)
        keep = np.zeros(N_AGENTS, dtype=bool)
        keep[list(self.rows)] = True
        a[~keep] = 0
        c[~keep] = 0.0
        return a, c


@pytest.mark.parametrize("side", [0, 1])
def test_win_definition_through_episode_at_n_active_2(side):
    """Binding ruling: at n_active_per_team=2 a kill by the OTHER agent of the
    policy's team (row 1 / row 6) still scores a win for the policy side —
    episode_stats.kills_* are TEAM counters. Per-policy credit is n=1-only.
    Timeout (nobody shoots) scores 0 on both sides.
    """
    from cs2rl.eval.baselines import BaselineEvaluator, IdleActor, OracleActor
    env = _arena_env(n_active_per_team=2)
    try:
        ev = BaselineEvaluator(env, episodes=2, seed=0)
        oracle = OracleActor(np.random.default_rng(0), ev.max_turn_speed, ev.laser_range, ev.nav)
        other = 1 if side == 0 else TEAM_SIZE + 1
        shooter = _RowActor(oracle, rows=(other, ))
        win, kills, stats = ev._episode(shooter, IdleActor(), side)
        # >= 1, not == 2: whether the row-masked oracle sweeps BOTH idle enemies
        # inside 160 ticks depends on the spawn draw; the contract under test is
        # the credit, not the sweep. Nobody on the policy's row fired, yet:
        assert win == 1.0 and kills >= 1, (win, kills, stats)
        assert stats["shots_fired"] == stats["shots_with_enemy_in_los"] > 0
        win, kills, _ = ev._episode(IdleActor(), IdleActor(), side)
        assert win == 0.0 and kills == 0, (win, kills)
    finally:
        env.close()


def test_eval_keys_and_selfplay_receives_elimination_only_rate():
    from cs2rl.train.metrics import elimination_only_win_rates
    logs = {"environment/winner_t": 0.2, "environment/winner_ct": 0.7, "environment/timed_out": 0.5}
    wt, wct = elimination_only_win_rates(logs)
    assert wt == 0.2 and wct == pytest.approx(0.2)
    # Missing keys (first epoch before any terminal) → zeros, never KeyError.
    assert elimination_only_win_rates({}) == (0.0, 0.0)


@pytest.mark.training
def test_policy_actor_from_live_policy_fills_all_rows(simple_map):
    from cs2rl.eval.baselines import BaselineEvaluator, PolicyActor
    from tests._helpers.trainer_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=16, map_data=simple_map)
    env = None
    try:
        from cs2rl.env.c.cs2_env import make_env
        env = make_env(map_data=simple_map, seed=1, auto_reset=False)
        ev = BaselineEvaluator(env, episodes=2, seed=0)
        pa = PolicyActor.from_policy(trainer.policy, "cpu")
        opp = PolicyActor.from_policy(trainer.policy, "cpu")           # DISTINCT actor: two LSTM states
        assert pa is not opp and pa.state is None
                                                                       # Splice check: wrap the opponent so its rows are recognisable.
        orig_act = opp.act

        def tagged(obs, st, vis, env):
            a, c = orig_act(obs, st, vis, env)
            c[:] = 0.123
            return a, c

        opp.act = tagged
        seen = {}
        orig_step = env.step

        def spy(act, cont):
            seen["cont"] = cont.copy()
            return orig_step(act, cont)

        env.step = spy
        out = ev.run_pair(pa, opp)
        assert set(out) >= {"win", "win_as_t", "win_as_ct", "kills_per_episode"}
        # last episode was side=1 (policy on CT rows 5-9): T rows carry the opponent tag,
        # CT rows carry live policy output (not the constant tag).
        assert (seen["cont"][:5] == 0.123).all()
        assert not (seen["cont"][5:] == 0.123).all()
        # evaluate() restores train mode on the shared module.
        trainer.policy.eval()
        ev.evaluate(trainer.policy, "cpu")
        assert trainer.policy.training
    finally:
        if env is not None:
            env.close()
        cleanup()


@pytest.mark.training
def test_evaluate_emits_all_eval_keys_and_keeps_training_rng(simple_map):
    """The eval/* key set is the contract analysis reads; and evaluate() must
    not perturb the training torch RNG stream (spec §6 seeding)."""
    import torch

    from cs2rl.eval.baselines import BaselineEvaluator
    from tests._helpers.trainer_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=16, map_data=simple_map)
    env = None
    try:
        from cs2rl.env.c.cs2_env import make_env
        from cs2rl.env.config import EnvConfig
        env = make_env(map_data=simple_map,
                       seed=1,
                       auto_reset=False,
                       config=EnvConfig(round_time=64))
        ev = BaselineEvaluator(env, episodes=2, seed=0)
        torch.manual_seed(123)
        before = torch.rand(3)
        torch.manual_seed(123)
        out = ev.evaluate(trainer.policy, "cpu")
        after = torch.rand(3)
        assert torch.equal(before, after), "evaluate() advanced the training RNG"
        assert set(out) == {
            "eval/win_vs_random", "eval/win_vs_random_as_t", "eval/win_vs_random_as_ct",
            "eval/kills_per_episode_vs_random", "eval/win_vs_oracle", "eval/win_vs_oracle_as_t",
            "eval/win_vs_oracle_as_ct", "eval/kills_per_episode_vs_oracle"
        }
        assert all(isinstance(v, float) for v in out.values())
    finally:
        if env is not None:
            env.close()
        cleanup()


class _StubEvaluator:
    """Counts evaluate() calls; stands in for BaselineEvaluator in the hook test."""

    def __init__(self):
        self.calls = 0
        self.closed = False

        class _Env:

            def close(_self):
                self.closed = True

        self.env = _Env()

    def evaluate(self, policy, device):
        self.calls += 1
        return {"eval/win_vs_random": 0.5, "eval/win_vs_oracle": 0.25}


@pytest.mark.training
def test_scheduled_eval_survives_log_throttle(simple_map):
    """Binding ruling: eval runs on the eval epoch even when PuffeRL's 0.25 s
    log throttle returns logs=None, and its keys land on the NEXT logged row
    (with eval/epoch stamping the epoch they were measured at)."""
    from cs2rl.train.metrics import ScheduledEval
    from tests._helpers.trainer_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=16,
                                               map_data=simple_map,
                                               n_active_per_team=1)
    try:
        # The harness trainer is Cs2PuffeRL (gh#168 W1.5), so its train() is
        # the hybrid-aware body (stock PuffeRL.train cannot unpack the 4-tuple
        # policy output); nothing has to be applied here.
        stub = _StubEvaluator()
        hook = ScheduledEval(stub, interval=1, policy=trainer.policy, device="cpu")
        # Epoch 1: throttled → logs is None; eval must still run and buffer.
        trainer.evaluate()
        trainer.last_log_time = time.time() + 10.0
        logs = trainer.train()
        assert logs is None, "throttle did not engage; test premise broken"
        hook.after_train(trainer, logs)
        assert stub.calls == 1 and hook.pending
        # Epoch 2: logged row → carries the buffered keys plus this epoch's.
        trainer.evaluate()
        trainer.last_log_time = 0.0
        logs = trainer.train()
        assert isinstance(logs, dict)
        hook.after_train(trainer, logs)
        assert stub.calls == 2 and not hook.pending
        assert logs["eval/win_vs_random"] == 0.5 and logs["eval/epoch"] == trainer.epoch
        hook.close()
        assert stub.closed
    finally:
        cleanup()


def test_scheduled_eval_interval_gating():
    """interval=3 → epochs 3, 6 only; interval<=0 is refused (train() never
    builds the hook then — a 0 here would be a modulo-by-zero on the first epoch)."""
    import types

    from cs2rl.train.metrics import ScheduledEval
    stub = _StubEvaluator()
    hook = ScheduledEval(stub, interval=3, policy=None, device="cpu")
    for epoch in range(1, 7):
        hook.after_train(types.SimpleNamespace(epoch=epoch), {})
    assert stub.calls == 2
    with pytest.raises(ValueError):
        ScheduledEval(stub, interval=0, policy=None, device="cpu")


def test_eval_interval_cli_config_and_modal_mirror():
    import re
    import types

    from cs2rl.train.config import build_train_config, compute_batch_dims
    src = (REPO_ROOT / "src" / "cs2rl" / "train" / "__main__.py").read_text()
    m = re.search(r'add_argument\(\s*"--eval-interval",(.*?)\)\n', src, re.S)
    assert m and "type=int" in m.group(1) and "default=0" in m.group(1) \
        and 'dest="eval_interval"' in m.group(1)
    args = types.SimpleNamespace(device="cpu",
                                 seed=1,
                                 timesteps=100_000,
                                 checkpoint_dir="/tmp/x",
                                 gamma=0.999,
                                 pbrs_gamma=None,
                                 eval_interval=7)
    _, bptt, bs = compute_batch_dims(16)
    cfg = build_train_config(args, batch_size=bs, bptt_horizon=bptt)
    assert cfg["eval_interval"] == 7
    args.eval_interval = 0
    assert build_train_config(args, batch_size=bs, bptt_horizon=bptt)["eval_interval"] == 0
    from cs2rl.train.resume import RESUME_CONFIG_ALLOWLIST
    assert "eval_interval" not in RESUME_CONFIG_ALLOWLIST
    from scripts.modal_runner import request
    assert request.LIVE_TRAIN_OPTION_ARITY.get("--eval-interval") == 1


def test_hit_geometry_constants_match_cs2_combat_h():
    """F16 tripwire: eval.baselines vendors the C hit geometry (eye heights,
    torso offsets) as Python literals — the oracle's LoS/aim math
    silently diverges from the sim if either side is edited alone. Regex the
    `static const float NAME = X.Yf;` declarations out of cs2_combat.h and
    compare; a missing name is a failure too (renamed constant = same drift).
    PITFALL: the header is the source of truth; fix eval/baselines.py, not the
    regex, when this trips."""
    import re

    from cs2rl.eval import baselines as eb

    header = (REPO_ROOT / "src" / "cs2rl" / "env" / "c" / "cs2_combat.h").read_text()
    pattern = re.compile(r"static const float\s+(EYE_HEIGHT_STAND|EYE_HEIGHT_CROUCH|"
                         r"TORSO_OFFSET_STAND|TORSO_OFFSET_CROUCH)\s*=\s*([0-9.]+)f")
    found = {name: float(val) for name, val in pattern.findall(header)}
    expected = {
        "EYE_HEIGHT_STAND": eb.EYE_STAND,
        "EYE_HEIGHT_CROUCH": eb.EYE_CROUCH,
        "TORSO_OFFSET_STAND": eb.TORSO_STAND,
        "TORSO_OFFSET_CROUCH": eb.TORSO_CROUCH,
    }
    missing = sorted(set(expected) - set(found))
    assert not missing, f"not found as `static const float` in cs2_combat.h: {missing}"
    for name, py_val in expected.items():
        assert found[name] == py_val, f"{name}: cs2_combat.h={found[name]} eval.baselines={py_val}"
