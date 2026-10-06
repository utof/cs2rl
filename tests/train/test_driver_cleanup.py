"""Drive train(args) on fakes, without GPU or W&B I/O: resource ownership and failures,
and the wiring between train() and the trainer it builds.

The trainer is built by the real `cs2rl.train.compose.build_trainer`; only the expensive
acquisitions it and train() call are replaced.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cs2rl.train import compose, loop


@pytest.fixture
def run_driver(monkeypatch, tmp_path):
    """Keep driver control flow and ScheduledEval real; replace expensive acquisitions.

    Each failure is raised by the boundary that can actually fail. File writes,
    JSON rows, hook scheduling, the wrapper and checkpoint serialization stay real.
    The teardown closes files even on the original leaking implementation.
    """
    import sys

    import pufferlib.vector

    from cs2rl.eval import baselines
    from cs2rl.train import trainer as trainer_module

    events, files, failures = [], [], {}
    # What the fakes were handed and what they returned, keyed by boundary.
    captured: dict = {}
    args = SimpleNamespace(
        device="cpu",
        checkpoint_dir=str(tmp_path),
        seed=7,
        num_envs=1,
        pin_pitch=False,
        map_data=None,
        vec_backend="serial",
        self_play=False,
        timesteps=20,
        save_every_sec=float("inf"),
        eval_interval=1,
        wandb=True,
        run_id="lifetime-test",
    )

    def hit(stage):
        """Raise the exact supplied exception object at one resource boundary."""
        if stage in failures:
            raise failures[stage]

    class Metrics:
        """Track close while retaining a real buffered metrics file."""

        def __init__(self, file):
            self.file = file

        def __getattr__(self, name):
            return getattr(self.file, name)

        def close(self):
            events.append("metrics.close")
            self.file.close()
            hit("metrics.close")

    original_open = Path.open

    def open_metrics(path, *a, **kw):
        """Only wrap the driver's append handle, leaving config/checkpoints alone."""
        if path.name == "metrics.jsonl" and a == ("a", ):
            hit("metrics.open")
            file = original_open(path, *a, **kw)
            files.append(file)
            return Metrics(file)
        return original_open(path, *a, **kw)

    class WandbRun:
        """Capture status without creating an external run."""

        @property
        def url(self):
            hit("wandb.url")
            return "offline-test"

        def finish(self, exit_code=0):
            events.append(("wandb.finish", exit_code))
            hit("wandb.finish")

        def log(self, entry, step):
            hit("wandb.log")

    def wandb_init(**kw):
        """Model only successful acquisition; failed init returns no owned handle."""
        hit("wandb.init")
        return WandbRun()

    class RawVec:
        """The wrapper delegates close to this acquired backend exactly once."""
        driver_env = SimpleNamespace(n_active_per_team=5)

        def close(self):
            events.append("vec.close")
            hit("vec.close")

    def make_vec(*a, **kw):
        captured["vec.make"] = (a, kw)
        hit("vec.make")
        captured["vec"] = RawVec()
        return captured["vec"]

    class Policy:
        """A seeded tensor lets the normal control check checkpoint contents."""

        def __init__(self):
            self.weight = torch.rand(2)
            self.aim_log_std = torch.zeros(2)

        def named_parameters(self):
            return []

        def state_dict(self):
            return {"weight": self.weight}

    def make_policy(*a, **kw):
        captured["build_policy"] = kw
        hit("policy")
        return Policy()

    class Trainer:
        """Run two epochs, including a throttled row, with the real wrapper."""

        def __init__(self, config, vecenv, policy, **kw):
            captured["trainer.init"] = kw
            hit("trainer.init")
            self.vecenv = vecenv
            self.policy = self.uncompiled_policy = policy
            self._self_play_mgr = kw["self_play_mgr"]
            self._participating_rows_np = kw["participating_rows"]
            self.logger = SimpleNamespace(run_id="old")
            self.optimizer = SimpleNamespace(param_groups=[{}])
            self.epoch, self.total_epochs, self.global_step = 0, 2, 0
            self.participating = np.ones(10, dtype=bool)
            self._timing = {}

        def evaluate(self):
            hit("collect")

        def train(self):
            hit("update")
            self.epoch += 1
            self.global_step += 10
            return None if self.epoch == 1 else {"losses/entropy": 2.0}

        def close_resources(self):
            """Setup cleanup has no checkpoint side effect."""
            events.append("trainer.resources.close")
            self.vecenv.close()
            hit("trainer.resources.close")

        def close(self):
            events.append("trainer.close")
            self.vecenv.close()
            # PuffeRL closes its vector before checkpointing; model a save failure.
            hit("trainer.close")

    class EvalEnv:
        round_time = 10

        def close(self):
            events.append("eval.close")
            hit("eval.close")

    def make_eval(*a, **kw):
        captured["eval.make"] = kw
        hit("eval.make")
        captured["eval.env"] = EvalEnv()
        return captured["eval.env"]

    def check_eval(*a):
        captured["eval.check"] = a
        hit("eval.check")

    class Evaluator:

        def __init__(self, env, **kw):
            hit("evaluator.init")
            self.env = env

        def evaluate(self, policy, device):
            events.append("eval.run")
            hit("eval.run")
            return {"eval/test": 0.75}

    def save(state, path):
        events.append("save")
        hit("save")
        torch.save(state, path)

    monkeypatch.setattr(loop, "resolve_pin_pitch", lambda *a, **kw: None)
    monkeypatch.setattr(Path, "open", open_metrics)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=wandb_init))
    monkeypatch.setattr(pufferlib.vector, "make", make_vec)
    monkeypatch.setattr(compose, "check_spawn_counts", lambda *a: hit("spawn.check"))
    monkeypatch.setattr(compose, "build_policy", make_policy)
    monkeypatch.setattr(
        loop, "build_train_config", lambda *a, **kw: {
            "aim_log_std_init": -1.0,
            "aim_log_std_max": 0.0,
            "weight_decay": 1e-4,
            "total_timesteps": 20,
            "participating_timesteps": 20,
        })
    monkeypatch.setattr(trainer_module, "Cs2PuffeRL", Trainer)
    monkeypatch.setattr(compose, "assert_pin_pitch_agreement", lambda *a: hit("pin.check"))
    monkeypatch.setattr(compose, "assert_max_turn_speed_agreement", lambda *a: None)
    monkeypatch.setattr(loop, "build_eval_env", make_eval)
    monkeypatch.setattr(loop, "assert_eval_env_agreement", check_eval)
    monkeypatch.setattr(baselines, "BaselineEvaluator", Evaluator)
    monkeypatch.setattr(loop, "_atomic_save_state_dict", save)
    try:
        yield SimpleNamespace(args=args,
                              events=events,
                              failures=failures,
                              files=files,
                              captured=captured,
                              trainer_module=trainer_module,
                              directory=tmp_path)
    finally:
        for file in files:
            file.close()


def test_success_preserves_order_scheduling_and_outputs(run_driver):
    """A throttled metrics row still evaluates; saves follow environment closes."""
    import json

    run = run_driver
    loop.train(run.args)
    assert run.events == [
        "eval.run", "eval.run", "trainer.close", "vec.close", "eval.close", "save", "metrics.close",
        ("wandb.finish", 0)
    ]
    assert all(file.closed for file in run.files)
    rows = [json.loads(line) for line in (run.directory / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["run_id"] == "lifetime-test"
    assert (rows[0]["epoch"], rows[0]["step"], rows[0]["eval/epoch"]) == (2, 20, 2)
    assert rows[0]["eval/test"] == 0.75
    assert rows[0]["timing/collect_ms"] >= 0 and rows[0]["timing/update_ms"] >= 0
    # F11: --no-self-play keeps the manager (evaluate() needs it) but logs none of its keys.
    assert not [key for key in rows[0] if key.startswith("self_play/")]
    expected = torch.rand(2, generator=torch.Generator().manual_seed(7))
    assert torch.equal(torch.load(run.directory / "dust2_policy.pt")["weight"], expected)


@pytest.mark.parametrize(("stage", "closed"), [
    ("wandb.init", []),
    ("wandb.url", []),
    ("metrics.open", []),
    ("vec.make", ["metrics.close"]),
    ("spawn.check", ["vec.close", "metrics.close"]),
    ("policy", ["vec.close", "metrics.close"]),
    ("trainer.init", ["vec.close", "metrics.close"]),
    ("pin.check", ["trainer.resources.close", "vec.close", "metrics.close"]),
    ("eval.make", ["trainer.resources.close", "vec.close", "metrics.close"]),
    ("eval.check", ["trainer.resources.close", "vec.close", "eval.close", "metrics.close"]),
    ("evaluator.init", ["trainer.resources.close", "vec.close", "eval.close", "metrics.close"]),
    ("collect", ["trainer.close", "vec.close", "eval.close", "metrics.close"]),
    ("update", ["trainer.close", "vec.close", "eval.close", "metrics.close"]),
    ("eval.run", ["trainer.close", "vec.close", "eval.close", "metrics.close"]),
    ("save", ["trainer.close", "vec.close", "eval.close", "metrics.close"]),
])
def test_failure_closes_only_acquired_owners(run_driver, stage, closed):
    """Early setup, eval construction and update exits release every acquired owner."""
    run = run_driver
    error = RuntimeError(stage)
    run.failures[stage] = error
    with pytest.raises(RuntimeError) as caught:
        loop.train(run.args)
    assert caught.value is error
    assert [e for e in run.events if isinstance(e, str) and e.endswith(".close")] == closed
    assert all(file.closed for file in run.files)
    assert [e for e in run.events
            if isinstance(e, tuple)] == ([] if stage == "wandb.init" else [("wandb.finish", 1)])


@pytest.mark.parametrize("owner", ["HybridAimVecEnv", "ScheduledEval"])
def test_failed_ownership_transfer_closes_previous_owner(run_driver, monkeypatch, owner):
    """A failed wrapper/hook constructor never acquires the previous owner's close."""
    run = run_driver
    error = KeyboardInterrupt(owner)

    def fail(*a, **kw):
        raise error

    monkeypatch.setattr(run.trainer_module if owner == "HybridAimVecEnv" else loop, owner, fail)
    with pytest.raises(KeyboardInterrupt) as caught:
        loop.train(run.args)
    assert caught.value is error
    assert run.events == (["vec.close"] if owner == "HybridAimVecEnv" else [
        "trainer.resources.close", "vec.close", "eval.close"
    ]) + ["metrics.close", ("wandb.finish", 1)]


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("stage", ["update", "evaluator.init"])
def test_primary_failure_survives_all_cleanup_failures(run_driver, failure_type, stage):
    """Later cleanup runs and cannot replace the failure that ended training."""
    failure = failure_type(3 if failure_type is SystemExit else stage)
    run = run_driver
    run.failures[stage] = failure
    trainer_close = "trainer.close" if stage == "update" else "trainer.resources.close"
    for close in (trainer_close, "eval.close", "metrics.close", "wandb.finish"):
        run.failures[close] = OSError(close)
    with pytest.raises(type(failure)) as caught:
        loop.train(run.args)
    assert caught.value is failure
    assert run.events == [
        trainer_close, "vec.close", "eval.close", "metrics.close",
        ("wandb.finish", 3 if isinstance(failure, SystemExit) else 1)
    ]
    assert len(failure.__notes__) == 4


@pytest.mark.parametrize("stage", ["trainer.close", "eval.close", "metrics.close", "wandb.finish"])
def test_first_cleanup_failure_is_reported_and_later_cleanup_runs(run_driver, stage):
    """A successful loop must not silently swallow a failing close/save/finish."""
    run = run_driver
    error = OSError(stage)
    run.failures[stage] = error
    with pytest.raises(OSError) as caught:
        loop.train(run.args)
    assert caught.value is error
    assert run.events[-2:] == [
        "metrics.close", ("wandb.finish", 0 if stage == "wandb.finish" else 1)
    ]
    assert run.events.count("eval.close") == run.events.count("trainer.close") == 1


def test_dead_run_keeps_autopsy_and_exit_status(run_driver, monkeypatch):
    """Dead-run SystemExit(3) follows autopsy saving and also closes eval/metrics."""
    run = run_driver
    monkeypatch.setattr(loop.DeadRunDetector, "check", lambda *a: True)
    with pytest.raises(SystemExit) as caught:
        loop.train(run.args)
    assert caught.value.code == 3
    assert (run.directory / "dust2_policy_dead.pt").is_file()
    assert not (run.directory / "dust2_policy.pt").exists()
    assert run.events[-5:] == [
        "trainer.close", "vec.close", "eval.close", "metrics.close", ("wandb.finish", 3)
    ]


def test_optional_resources_can_be_disabled(run_driver):
    """The driver still closes its vector and metrics with neither W&B nor eval."""
    run = run_driver
    run.args.wandb = False
    run.args.eval_interval = 0
    loop.train(run.args)
    assert run.events == ["trainer.close", "vec.close", "save", "metrics.close"]


def test_periodic_save_failure_closes_resources(run_driver):
    """A checkpoint failure during the loop unwinds the same owners as an update."""
    run = run_driver
    run.args.save_every_sec = -1
    error = OSError("periodic checkpoint")
    run.failures["save"] = error
    with pytest.raises(OSError) as caught:
        loop.train(run.args)
    assert caught.value is error
    assert run.events == [
        "eval.run", "save", "trainer.close", "vec.close", "eval.close", "metrics.close",
        ("wandb.finish", 1)
    ]


# ── What train() hands the trainer it builds ───────────────────────────────
# Each test below replaces a source-text pin on train() from before #92 part 2, when
# train() could not be driven without a real run.


def _rows(run):
    import json

    return [json.loads(line) for line in (run.directory / "metrics.jsonl").read_text().splitlines()]


def test_train_envs_get_the_run_config_their_seed_and_their_shm_slot(run_driver, monkeypatch):
    """Env i is built with the run's EnvConfig, `_seed = env_seed_base(--seed) + i` and
    shared-memory slot i.

    The creators and per-env kwargs given to vector.make are called the way pufferlib's
    Serial backend calls them, with the env builder replaced by a recorder. The reward
    override proves the config is the run's and not EnvConfig(): a dropped config trains
    the default weights with nothing failing. vector.make must get no `seed=`: pufferlib
    never forwards it, so the envs would silently ignore --seed.
    """
    from cs2rl.train import envs as train_envs
    from cs2rl.train.config import env_config_from_args

    run = run_driver
    run.args.num_envs = 3
    run.args.reward_ct_survival = 0.0
    run.args.map_data = object()
    loop.train(run.args)

    class Env:
        """Records its construction kwargs and the shared-memory views attached to it."""

        def __init__(self, kw):
            self.kw, self.views = kw, []

        def _attach_cont_action_view(self, shm, idx):
            self.views.append(("cont", shm, idx))

        def _attach_mask_view(self, shm, idx):
            self.views.append(("mask", shm, idx))

    built = []
    monkeypatch.setattr(train_envs, "build_train_env",
                        lambda **kw: built.append(Env(kw)) or built[-1])
    (creators, ), kw = run.captured["vec.make"]
    assert "seed" not in kw
    for i, creator in enumerate(creators):
        creator(*kw["env_args"][i], buf=None, seed=i, **kw["env_kwargs"][i])
    config = env_config_from_args(run.args)
    assert config.rewards.reward_ct_survival == 0.0
    assert len(built) == 3
    for i, env in enumerate(built):
        assert env.kw["_seed"] == train_envs.env_seed_base(7) + i
        assert env.kw["config"] == config
        assert env.kw["map_data"] is run.args.map_data
        assert [(view, idx) for view, _, idx in env.views] == [("cont", i), ("mask", i)]
    trainer_kw = run.captured["trainer.init"]
    assert trainer_kw["cont_action_view_main"].shape[0] == trainer_kw["mask_view_main"].shape[
        0] == 30


def test_eval_env_gets_the_run_config_and_is_checked_against_the_training_env(run_driver):
    """--eval-interval's env is built from the run's map and EnvConfig and checked against
    the training vecenv's driver env (Task 13).

    The check is the only guard against scoring the policy on a different sim than it
    trains on, a failure that only shows as worse numbers.
    """
    from cs2rl.train.config import env_config_from_args

    run = run_driver
    run.args.reward_ct_survival = 0.0
    run.args.map_data = object()
    loop.train(run.args)
    made = run.captured["eval.make"]
    assert made.keys() == {"map_data", "config"}
    assert made["map_data"] is run.args.map_data
    assert made["config"] == env_config_from_args(run.args)
    eval_env, driver_env = run.captured["eval.check"]
    assert eval_env is run.captured["eval.env"]
    assert driver_env is run.captured["vec"].driver_env


def test_noop_opponent_trains_only_the_hero_team_rows(run_driver, capsys):
    """--opponent noop gives the trainer the hero team's rows only, and says so in the log.

    Hardwiring opponent_mode="self" here would train on the statue's rows too and spend
    half the budget on an opponent that never moves, with every metric still finite.
    """
    from cs2rl.train.config import build_participating_rows

    run = run_driver
    run.args.opponent = "noop"
    loop.train(run.args)
    rows = run.captured["trainer.init"]["participating_rows"]
    assert np.array_equal(rows, build_participating_rows(1, 5, opponent_mode="noop", hero_team="t"))
    assert rows.sum() == 5
    assert run.captured["trainer.init"]["self_play_mgr"].opponent_mode == "noop"
    assert "team CT is a stationary statue; 5 of 10 agent rows participate" in capsys.readouterr(
    ).out


def test_selfplay_manager_is_built_from_the_run_flags(run_driver, monkeypatch):
    """--no-self-play, --opponent, the aim-σ cap and the pitch pin each reach their own
    build_selfplay_manager slot.

    The values have distinct types (bool, str, float, int), so two crossed slots give a
    wrong type rather than an equal-looking value. The manager re-applies the cap and
    the pin to every past policy it loads, and a wrong one fails nothing (see
    tests/train/test_selfplay_factory.py).
    """
    from cs2rl.train import selfplay

    recorded, made = [], []

    def spy(**kw):
        recorded.append(kw)
        made.append(selfplay.build_selfplay_manager(**kw))
        return made[-1]

    monkeypatch.setattr(compose, "build_selfplay_manager", spy)
    run = run_driver
    run.args.opponent, run.args.aim_log_std_max, run.args.pin_pitch = "noop", -1.2345, 1
    loop.train(run.args)
    assert [{
        k: (type(v), v)
        for k, v in kw.items()
    } for kw in recorded] == [{
        "self_play_enabled": (bool, False),
        "aim_log_std_max": (float, -1.2345),
        "pin_pitch": (int, 1),
        "opponent_mode": (str, "noop"),
    }]
    assert run.captured["trainer.init"]["self_play_mgr"] is made[0]


def test_selfplay_row_reports_the_pool_and_past_opponent_use(run_driver, monkeypatch):
    """With self-play on, the persisted row carries self_play/pool_size and
    self_play/used_past, the 0/1 flag evaluate() leaves on the trainer, as floats that
    the numeric persist filter keeps.
    """
    run = run_driver
    run.args.self_play = True
    trainer_class = run.trainer_module.Cs2PuffeRL
    evaluate = trainer_class.evaluate

    def evaluate_against_a_past_policy(self):
        evaluate(self)
        self._selfplay_used_past = True

    monkeypatch.setattr(trainer_class, "evaluate", evaluate_against_a_past_policy)
    loop.train(run.args)
    (row, ) = _rows(run)
    assert row["self_play/used_past"] == 1.0
    assert row["self_play/pool_size"] == 0.0


def test_split_flags_reach_build_policy(run_driver):
    """--tct-split-trunk without a checkpoint builds a trunk-split policy.

    With a checkpoint the bits come from cs2rl.train.resume.resolve_resume_split
    (tests/test_tct_split.py drives that half through loop._resume_policy_init).
    """
    run = run_driver
    run.args.tct_split_trunk = True
    loop.train(run.args)
    built = run.captured["build_policy"]
    assert (built["tct_split_heads"], built["tct_split_trunk"]) == (False, True)


def test_collect_and_update_times_reach_the_row(run_driver, monkeypatch):
    """timing/collect_ms is evaluate()'s wall time and timing/update_ms is train()'s.

    A fake perf_counter advances 0.25 s inside evaluate() and 0.5 s inside train(), so
    a timer started or stopped at the wrong call, or a key never written, gives another
    number. The real clock is kept for time.time (checkpoint throttle) and strftime.
    """
    import time

    now = [0.0]
    monkeypatch.setattr(
        loop, "time",
        SimpleNamespace(perf_counter=lambda: now[0], time=time.time, strftime=time.strftime))
    run = run_driver
    trainer_class = run.trainer_module.Cs2PuffeRL
    evaluate, update = trainer_class.evaluate, trainer_class.train

    def slow_evaluate(self):
        now[0] += 0.25
        return evaluate(self)

    def slow_update(self):
        now[0] += 0.5
        return update(self)

    monkeypatch.setattr(trainer_class, "evaluate", slow_evaluate)
    monkeypatch.setattr(trainer_class, "train", slow_update)
    loop.train(run.args)
    (row, ) = _rows(run)
    assert (row["timing/collect_ms"], row["timing/update_ms"]) == (250.0, 500.0)
