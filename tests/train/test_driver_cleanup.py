"""Drive train(args) through resource ownership and failures, without GPU or W&B I/O."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cs2rl.train import loop


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
        hit("vec.make")
        return RawVec()

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
        hit("policy")
        return Policy()

    class Trainer:
        """Run two epochs, including a throttled row, with the real wrapper."""

        def __init__(self, config, vecenv, policy, **kw):
            hit("trainer.init")
            self.vecenv = vecenv
            self.policy = policy
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
        hit("eval.make")
        return EvalEnv()

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
    monkeypatch.setattr(loop, "check_spawn_counts", lambda *a: hit("spawn.check"))
    monkeypatch.setattr(loop, "build_policy", make_policy)
    monkeypatch.setattr(loop, "build_train_config", lambda *a, **kw: {
        "aim_log_std_init": -1.0,
        "aim_log_std_max": 0.0,
    })
    monkeypatch.setattr(trainer_module, "Cs2PuffeRL", Trainer)
    monkeypatch.setattr(loop, "assert_pin_pitch_agreement", lambda *a: hit("pin.check"))
    monkeypatch.setattr(loop, "assert_max_turn_speed_agreement", lambda *a: None)
    monkeypatch.setattr(loop, "build_eval_env", make_eval)
    monkeypatch.setattr(loop, "assert_eval_env_agreement", lambda *a: hit("eval.check"))
    monkeypatch.setattr(baselines, "BaselineEvaluator", Evaluator)
    monkeypatch.setattr(loop, "_atomic_save_state_dict", save)
    try:
        yield SimpleNamespace(args=args,
                              events=events,
                              failures=failures,
                              files=files,
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
    ("pin.check", ["trainer.close", "vec.close", "metrics.close"]),
    ("eval.make", ["trainer.close", "vec.close", "metrics.close"]),
    ("eval.check", ["trainer.close", "vec.close", "eval.close", "metrics.close"]),
    ("evaluator.init", ["trainer.close", "vec.close", "eval.close", "metrics.close"]),
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
        "trainer.close", "vec.close", "eval.close"
    ]) + ["metrics.close", ("wandb.finish", 1)]


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("update"), KeyboardInterrupt("interrupt"),
     SystemExit(3)])
def test_primary_failure_survives_all_cleanup_failures(run_driver, failure):
    """Later cleanup runs and cannot replace the failure that ended training."""
    run = run_driver
    run.failures["update"] = failure
    for stage in ("trainer.close", "eval.close", "metrics.close", "wandb.finish"):
        run.failures[stage] = OSError(stage)
    with pytest.raises(type(failure)) as caught:
        loop.train(run.args)
    assert caught.value is failure
    assert run.events == [
        "trainer.close", "vec.close", "eval.close", "metrics.close",
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
