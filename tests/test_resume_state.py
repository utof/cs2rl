"""R0-C (#134): full-state checkpoint/resume.

Two layers: (1) in-process — save via the overridden save_checkpoint, restore
into a FRESH harness trainer, compare every piece of state the spec lists;
(2) subprocess — a 2-epoch serial CPU run, then `--resume-run` for one more
epoch; asserts the run_id is shared, `resumed_from_step` is stamped, epoch
continues from the checkpoint, and the config guard rejects a knob change.
"""
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "src" / "train.py"


def _make(num_envs=16, seed=3):
    from train import (
        SelfPlayManager,
        _install_full_checkpointing,
        _patch_trainer_with_return_norm,
        _patch_trainer_with_selfplay,
    )
    from train_test_harness import _build_trainer_for_test
    trainer, cleanup = _build_trainer_for_test(num_envs=num_envs, seed=seed)
    _patch_trainer_with_return_norm(trainer)
    mgr = SelfPlayManager(pool_size=15,
                          p_past=0.0,
                          save_every_epochs=25,
                          win_threshold=0.6,
                          phase_length=50)
    _patch_trainer_with_selfplay(trainer, mgr)
    trainer.logger.run_id = "rid-test"
    _install_full_checkpointing(trainer, mgr)
    return trainer, mgr, cleanup


def _snapshot(trainer, mgr):
    from train import _WARMSTART_ATTRS                                             # defined beside collect/restore_train_state
    return {
        "global_step": trainer.global_step,
        "epoch": trainer.epoch,
        "lr": trainer.scheduler.get_last_lr(),
        "ret_mean": trainer._ret_mean.clone(),
        "ret_var": trainer._ret_var.clone(),
        "ret_count": trainer._ret_count.clone(),
        "log_alpha": trainer._log_alpha_tensor.detach().clone(),
        "target_entropy": trainer._batch1_current_target_entropy,
        "ws_phase": trainer._batch1_warmstart_phase,
        "opponent_team": mgr.opponent_team,
        "pool": list(mgr.pool),
                                                                                   # RNG STATES (not fresh draws — a draw-based compare is one stray
                                                                                   # torch.randn away from flaky).
        "py_random": random.getstate(),
        "np_random": np.random.get_state()[1].tobytes(),
        "torch_random": torch.get_rng_state().numpy().tobytes(),
        "warmstart": {
            k: getattr(trainer, k)
            for k in _WARMSTART_ATTRS
        },
        "opt": {
            k: v.clone()
            for k, v in trainer.optimizer.state_dict()["state"].get(0, {}).items()
            if torch.is_tensor(v)
        },
    }


def test_save_writes_three_files_and_never_early_returns():
    trainer, mgr, cleanup = _make()
    try:
        trainer.evaluate()
        trainer.train()
        p1 = trainer.save_checkpoint()
        p2 = trainer.save_checkpoint()                                                             # same epoch — stock PuffeRL would early-return
        assert p1 == p2 and Path(p1).name == f"model_{trainer.epoch:06d}.pt"
        d = Path(trainer.config["data_dir"]) / "rid-test"
        assert (d / "trainer_state.pt").exists() and (d / "train_state.pt").exists()
        ts = torch.load(d / "trainer_state.pt", weights_only=False)
        assert set(ts) == {
            "optimizer_state_dict", "global_step", "agent_step", "update", "model_name", "run_id"
        }
        st = torch.load(d / "train_state.pt", weights_only=False)
        for k in ("log_alpha", "alpha_optimizer", "scheduler", "ret_mean", "ret_var", "ret_count",
                  "warmstart", "self_play", "rng"):
            assert k in st, k
    finally:
        cleanup()


def test_round_trip_restores_everything():
    trainer, mgr, cleanup = _make()
    try:
        for _ in range(2):
            trainer.evaluate()
            trainer.train()
        mgr.opponent_team = "t"
        data_dir = trainer.config["data_dir"]
        # load_state_dict drops pool paths that no longer exist, so the entry
        # that must round-trip has to be a REAL file; the second entry is the
        # missing-path case and must be filtered out on restore.
        sp_path = Path(data_dir) / "sp_000001.pt"
        sp_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trainer.policy.state_dict(), sp_path)
        mgr.pool.append(sp_path)
        mgr.pool.append(Path(data_dir) / "sp_missing.pt")
        trainer.save_checkpoint()
        before = _snapshot(trainer, mgr)
    finally:
        cleanup_a = cleanup
    from train import load_full_resume, resolve_resume_run
    trainer2, mgr2, cleanup2 = _make(seed=99)                                                  # different seed ⇒ different RNG streams
    try:
        paths = resolve_resume_run(Path(data_dir), run_id="rid-test")
        info = load_full_resume(trainer2, mgr2, paths)
        assert info["resumed_from_step"] == before["global_step"]
        after = _snapshot(trainer2, mgr2)
        for k in ("global_step", "epoch", "lr", "target_entropy", "ws_phase", "opponent_team",
                  "py_random", "np_random", "torch_random"):
            assert after[k] == before[k], k
        assert after["pool"] == [sp_path], after["pool"]                                       # missing path dropped
        for k in ("ret_mean", "ret_var", "ret_count", "log_alpha"):
            assert torch.equal(after[k], before[k]), k
        assert after["opt"].keys() == before["opt"].keys()
        for k in before["opt"]:
            assert torch.equal(after["opt"][k], before["opt"][k]), k
        assert after["warmstart"] == before["warmstart"]
                                                                                               # In-place restore contract: the trainer attrs alias the closure's
                                                                                               # tensors, so one more train() must MOVE trainer2._ret_count (a rebind
                                                                                               # would leave the attr frozen while the closure keeps its own copy).
        _rc_obj = trainer2._ret_count
        trainer2.evaluate()
        trainer2.train()
        assert trainer2._ret_count is _rc_obj and float(_rc_obj) > float(before["ret_count"])
                                                                                               # team-spirit is a pure function of global_step (train.py main loop), so
                                                                                               # the global_step equality above is the spec §6 "team-spirit matches" check.
    finally:
        cleanup2()
        cleanup_a()


def test_resumed_schedule_matches_uninterrupted():
    """The restored run's NEXT epoch must land where an uninterrupted run
    lands — not merely "loads without error". Env sampling is not bit-exact
    (xorshift32 state is not checkpointed), so losses are excluded; every
    schedule-derived quantity (LR after the step, step counter, epoch, SAC
    target entropy, team-spirit input) is a pure function of the restored
    state and must be equal."""
    from train import load_full_resume, resolve_resume_run
    t_a, mgr_a, cleanup_a = _make(seed=3)
    try:
        for _ in range(3):
            t_a.evaluate()
            t_a.train()
        want = (t_a.scheduler.get_last_lr(), t_a.optimizer.param_groups[0]["lr"], t_a.global_step,
                t_a.epoch, t_a._batch1_current_target_entropy)
    finally:
        cleanup_a()
    t_b, mgr_b, cleanup_b = _make(seed=3)
    try:
        for _ in range(2):
            t_b.evaluate()
            t_b.train()
        t_b.save_checkpoint()
        data_dir = t_b.config["data_dir"]
        t_c, mgr_c, cleanup_c = _make(seed=99)
        try:
            load_full_resume(t_c, mgr_c, resolve_resume_run(Path(data_dir), run_id="rid-test"))
            t_c.evaluate()
            t_c.train()
            got = (t_c.scheduler.get_last_lr(), t_c.optimizer.param_groups[0]["lr"],
                   t_c.global_step, t_c.epoch, t_c._batch1_current_target_entropy)
            assert got == want, (got, want)
        finally:
            cleanup_c()
    finally:
        cleanup_b()


def test_scheduler_restore_adopts_new_t_max():
    """The sidecar stores CosineAnnealingLR.state_dict(), which includes the
    OLD T_max. A --timesteps extension (allowlisted) must keep last_epoch but
    adopt the NEW trainer's horizon and put the LR on the closed-form cosine at
    that epoch; a same-horizon restore must not touch the optimizer lr."""
    import math

    from train import collect_train_state, restore_train_state
    trainer, mgr, cleanup = _make()
    try:
        for _ in range(2):
            trainer.evaluate()
            trainer.train()
        st = collect_train_state(trainer, mgr)
        lr_same = trainer.optimizer.param_groups[0]["lr"]
        restore_train_state(trainer, mgr, st)          # same horizon: bit-exact
        assert trainer.optimizer.param_groups[0]["lr"] == lr_same
        assert trainer.scheduler.T_max == trainer.total_epochs
        st["scheduler"]["T_max"] = 999                 # pretend the saved run had a longer budget
        restore_train_state(trainer, mgr, st)
        sch = trainer.scheduler
        assert sch.T_max == trainer.total_epochs != 999
        assert sch.last_epoch == st["scheduler"]["last_epoch"] == 2
        base = sch.base_lrs[0]
        want = sch.eta_min + (base - sch.eta_min) * (
            1 + math.cos(math.pi * sch.last_epoch / sch.T_max)) / 2
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(want, rel=1e-9)
        assert sch.get_last_lr()[0] == pytest.approx(want, rel=1e-9)
    finally:
        cleanup()


def test_config_guard_allowlist(tmp_path):
    from train import RESUME_CONFIG_ALLOWLIST, check_resume_config
    old = {"a": 1, "seed": 1, "device": "cpu", "data_dir": "x", "run_id": "r", "gamma": 0.99}
    (tmp_path / "config.json").write_text(json.dumps(old))
    new = dict(old, seed=2, device="cuda", data_dir="y", run_id="q")
    check_resume_config(tmp_path, new)                 # allowlisted diffs OK
    assert {"data_dir", "device", "seed", "run_id"} <= RESUME_CONFIG_ALLOWLIST
                                                       # Task 7 context ruling: n_active_per_team changes the step unit and the
                                                       # participating buffer layout — it must NEVER be allowlisted.
    assert "n_active_per_team" not in RESUME_CONFIG_ALLOWLIST
    with pytest.raises(SystemExit, match="gamma"):
        check_resume_config(tmp_path, dict(old, gamma=0.5))
    with pytest.raises(SystemExit, match="extra_key"):
        check_resume_config(tmp_path, dict(old, extra_key=1))


def test_analyze_tplant_last_row_wins():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from analyze_tplant import dedupe_resume_rows
    rows = [{
        "run_id": "r",
        "step": 100,
        "v": 1
    }, {
        "run_id": "r",
        "step": 200,
        "v": 2
    }, {
        "run_id": "r",
        "step": 200,
        "v": 3
    }, {
        "run_id": "s",
        "step": 200,
        "v": 4
    }]
    out = dedupe_resume_rows(rows)
    assert [(r["run_id"], r["step"], r["v"]) for r in out] == [("r", 100, 1), ("r", 200, 3),
                                                               ("s", 200, 4)]


@pytest.mark.timeout(1800)
def test_subprocess_resume_run(tmp_path):
    ckpt = tmp_path / "run"
    # 16 envs is the FLOOR for the real CLI: batch_size = num_envs*10*64 must be
    # >= the pinned minibatch_size 8192 (build_train_config, pufferl.py:121-124).
    common = [
        "--train", "--device", "cpu", "--vec-backend", "serial", "--num_envs", "16",
        "--no-self-play", "--no-dead-run-abort", "--checkpoint-interval", "1", "--seed", "3",
        "--save_every_sec", "100000"
    ]
    # batch_size(16 envs) = 16*10*64 = 10240 ⇒ timesteps 20480 = 2 epochs
    r = subprocess.run([
        sys.executable,
        str(TRAIN_SCRIPT), *common, "--timesteps", "20480", "--checkpoint-dir",
        str(ckpt), "--run-id", "rid-sub"
    ],
                       cwd=REPO_ROOT,
                       capture_output=True,
                       text=True,
                       timeout=1500)
    assert r.returncode == 0, r.stderr[-3000:]
    models = sorted((ckpt / "rid-sub").glob("model_*.pt"))
    assert len(models) >= 2, models
    rows = [json.loads(line) for line in (ckpt / "metrics.jsonl").read_text().splitlines()]
    assert all(row["run_id"] == "rid-sub" for row in rows)
    last_step = rows[-1]["step"]
    last_lr = rows[-1]["learning_rate"]                # pufferl.py mean_and_log, logged AFTER scheduler.step()

    # Extend the budget 20480 → 30720 (epoch 3). total_timesteps is allowlisted.
    r = subprocess.run([
        sys.executable,
        str(TRAIN_SCRIPT), *common, "--timesteps", "30720", "--resume-run",
        str(ckpt)
    ],
                       cwd=REPO_ROOT,
                       capture_output=True,
                       text=True,
                       timeout=1500)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "resume not bit-exact for env sampling" in r.stdout
    rows2 = [json.loads(line) for line in (ckpt / "metrics.jsonl").read_text().splitlines()]
    new = rows2[len(rows):]
    assert new, "no rows appended after resume"
    assert all(row["run_id"] == "rid-sub" for row in new)
    assert new[0]["resumed_from_step"] == 20480
    assert new[0]["epoch"] == 3 and new[0]["step"] > last_step
    assert len(new) == 1
    # Scheduler horizon adopted the NEW budget (restore_train_state overrides
    # T_max): the first run annealed to lr=0 at its T_max=2; with the OLD T_max
    # re-installed the recursive cosine would bounce the LR back UP at epoch 3
    # (torch lr_scheduler.py CosineAnnealingLR.get_lr). With the new T_max=3 it
    # stays <= the last value.
    assert new[0]["learning_rate"] <= last_lr + 1e-12, (new[0]["learning_rate"], last_lr)

    # Config guard: a non-allowlisted knob change is a hard error. Uses a knob
    # that EXISTS at Task 7 (--num_envs changes batch_size in config.json) and
    # asserts on the guard's own message so an argparse error can never satisfy it.
    mismatch = list(common)
    mismatch[mismatch.index("--num_envs") + 1] = "32"                         # 16 → 32 envs ⇒ batch_size changes
    r = subprocess.run([
        sys.executable,
        str(TRAIN_SCRIPT), *mismatch, "--timesteps", "40960", "--resume-run",
        str(ckpt)
    ],
                       cwd=REPO_ROOT,
                       capture_output=True,
                       text=True,
                       timeout=600)
    assert r.returncode != 0
    assert "config.json mismatch on non-allowlisted keys" in (r.stderr + r.stdout), r.stderr[-2000:]
    assert "batch_size" in (r.stderr + r.stdout)
