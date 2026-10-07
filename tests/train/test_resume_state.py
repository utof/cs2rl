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

from tests.conftest import REPO_ROOT


def _make(num_envs=16, seed=3):
    """A harness trainer whose self-play manager this test owns.

    gh#168 W1.5: the manager is built FIRST and handed to the harness, which
    constructs the production trainer class around it, so `save_checkpoint`
    (full checkpointing is always installed by Cs2PuffeRL.__init__) pickles
    THIS manager's pool state and `restore_train_state` can be checked against
    the same object. Before W1.5 this helper applied the self-play and
    checkpointing patches a second time on top of the harness's own.
    """
    from cs2rl.train.selfplay import SelfPlayManager
    from tests._helpers.trainer_harness import _build_trainer_for_test
    mgr = SelfPlayManager(pool_size=15,
                          p_past=0.0,
                          save_every_epochs=25,
                          win_threshold=0.6,
                          phase_length=50)
    trainer, cleanup = _build_trainer_for_test(num_envs=num_envs, seed=seed, self_play_mgr=mgr)
    assert trainer.logger is not None, "PuffeRL.__init__ replaces a None logger with NoLogger"
    trainer.logger.run_id = "rid-test"
    return trainer, mgr, cleanup


def _snapshot(trainer, mgr):
    # Defined beside collect/restore_train_state.
    from cs2rl.train.resume import _WARMSTART_ATTRS

    return {
        "global_step": trainer.global_step,
        "epoch": trainer.epoch,
        "lr": trainer.scheduler.get_last_lr(),
        "ret_mean": trainer._ret_mean.clone(),
        "ret_var": trainer._ret_var.clone(),
        "ret_count": trainer._ret_count.clone(),
        "log_alpha": trainer._log_alpha_tensor.detach().clone(),
        "target_entropy": trainer._current_target_entropy,
        "ws_phase": trainer._warmstart_phase,
        "opponent_team": mgr.opponent_team,
        "pool": list(mgr.pool),
                                                                                   # RNG STATES (not fresh draws — a draw-based compare is one stray
                                                                                   # torch.randn away from flaky).
        "py_random": random.getstate(),
        "np_random": np.random.get_state(legacy=False)["state"]["key"].tobytes(),
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


@pytest.mark.training
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


@pytest.mark.training
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
    from cs2rl.train.resume import load_full_resume, resolve_resume_run
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


@pytest.mark.training
def test_resumed_schedule_matches_uninterrupted():
    """The restored run's NEXT epoch must land where an uninterrupted run
    lands — not merely "loads without error". Env sampling is not bit-exact
    (xorshift32 state is not checkpointed), so losses are excluded; every
    schedule-derived quantity (LR after the step, step counter, epoch, SAC
    target entropy, team-spirit input) is a pure function of the restored
    state and must be equal."""
    from cs2rl.train.resume import load_full_resume, resolve_resume_run
    t_a, mgr_a, cleanup_a = _make(seed=3)
    try:
        for _ in range(3):
            t_a.evaluate()
            t_a.train()
        want = (t_a.scheduler.get_last_lr(), t_a.optimizer.param_groups[0]["lr"], t_a.global_step,
                t_a.epoch, t_a._current_target_entropy)
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
                   t_c.global_step, t_c.epoch, t_c._current_target_entropy)
            assert got == want, (got, want)
        finally:
            cleanup_c()
    finally:
        cleanup_b()


@pytest.mark.training
def test_scheduler_restore_adopts_new_t_max():
    """The sidecar stores CosineAnnealingLR.state_dict(), which includes the
    OLD T_max. A --timesteps extension (allowlisted) must keep last_epoch but
    adopt the NEW trainer's horizon and put the LR on the closed-form cosine at
    that epoch; a same-horizon restore must not touch the optimizer lr."""
    import math

    from cs2rl.train.resume import collect_train_state, restore_train_state
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


def test_config_guard_allowlist(tmp_path, capsys):
    from cs2rl.train.resume import RESUME_CONFIG_ALLOWLIST, check_resume_config
    old = {"a": 1, "seed": 1, "device": "cpu", "data_dir": "x", "run_id": "r", "gamma": 0.99}
    (tmp_path / "config.json").write_text(json.dumps(old))
    new = dict(old, device="cuda", data_dir="y", run_id="q")
    check_resume_config(tmp_path, new)                 # allowlisted diffs OK
                                                       # data_dir relative→absolute of the SAME dir is not a change (no WARN noise)
    capsys.readouterr()                                # drop the WARN lines from the call above
    (tmp_path / "config.json").write_text(json.dumps(dict(old, data_dir=str(Path("rel/run")))))
    check_resume_config(tmp_path, dict(old, data_dir=str(Path("rel/run").resolve())))
    assert "data_dir" not in capsys.readouterr().out
    (tmp_path / "config.json").write_text(json.dumps(old))
    assert {"data_dir", "device", "run_id"} <= RESUME_CONFIG_ALLOWLIST
                                                       # R0-D (#135): a resumed run's RNGs come from train_state.pt, so a
                                                       # changed --seed must be refused, not silently half-applied.
    assert "seed" not in RESUME_CONFIG_ALLOWLIST
    with pytest.raises(SystemExit, match="seed"):
        check_resume_config(tmp_path, dict(old, seed=2))
                                                       # Task 7 context ruling: n_active_per_team changes the step unit and the
                                                       # participating buffer layout — it must NEVER be allowlisted.
    assert "n_active_per_team" not in RESUME_CONFIG_ALLOWLIST
    with pytest.raises(SystemExit, match="gamma"):
        check_resume_config(tmp_path, dict(old, gamma=0.5))
    with pytest.raises(SystemExit, match="extra_key"):
        check_resume_config(tmp_path, dict(old, extra_key=1))


def _fake_set(d: Path, model_epochs, ts_epoch, st_epoch, *, steps_per_epoch=100):
    """Manufacture a checkpoint set on disk without a trainer: model files for
    `model_epochs`, trainer_state.pt naming model_<ts_epoch> and train_state.pt
    stamped with st_epoch — the shapes a crash between the three atomic writes
    can leave behind."""
    d.mkdir(parents=True, exist_ok=True)
    for e in model_epochs:
        torch.save({}, d / f"model_{e:06d}.pt")
    torch.save(
        {
            "optimizer_state_dict": {},
            "global_step": ts_epoch * steps_per_epoch,
            "agent_step": ts_epoch * steps_per_epoch,
            "update": ts_epoch,
            "model_name": f"model_{ts_epoch:06d}.pt",
            "run_id": "rid",
        }, d / "trainer_state.pt")
    torch.save({"epoch": st_epoch, "global_step": st_epoch * steps_per_epoch}, d / "train_state.pt")


def test_resolve_uses_trainer_state_model_name_not_max(tmp_path):
    """model_000010.pt orphaned by a crash after the model write: the set is
    trainer_state@9 + train_state@9 → resume from 9, and the orphan is
    reported, not silently paired with epoch-9 optimizer state."""
    from cs2rl.train.resume import check_checkpoint_set, resolve_resume_run
    _fake_set(tmp_path / "rid", model_epochs=(9, 10), ts_epoch=9, st_epoch=9)
    paths = resolve_resume_run(tmp_path)               # run_id discovered
    assert paths["run_id"] == "rid" and paths["model_path"].name == "model_000009.pt"
    ts = torch.load(paths["trainer_state_path"], weights_only=False)
    st = torch.load(paths["train_state_path"], weights_only=False)
    check_checkpoint_set(paths["model_path"], ts, st)  # consistent → no raise


def test_mismatched_set_is_refused(tmp_path):
    """trainer_state@10 (names model_000010) + train_state@9: a stale sidecar
    next to a newer model/optimizer (e.g. a torn write, or files copied by
    hand). Must be refused, naming all three epochs."""
    from cs2rl.train.resume import check_checkpoint_set, resolve_resume_run
    _fake_set(tmp_path / "rid", model_epochs=(9, 10), ts_epoch=10, st_epoch=9)
    paths = resolve_resume_run(tmp_path, run_id="rid")
    ts = torch.load(paths["trainer_state_path"], weights_only=False)
    st = torch.load(paths["train_state_path"], weights_only=False)
    with pytest.raises(SystemExit,
                       match=r"model epoch 10.*trainer_state epoch 10.*train_state epoch 9"):
        check_checkpoint_set(paths["model_path"], ts, st)
        # legacy sidecar without the stamp is refused too (epoch -1 in the message)
    with pytest.raises(SystemExit, match="train_state epoch -1"):
        check_checkpoint_set(paths["model_path"], ts, {})


def test_resolve_refuses_incomplete_dirs(tmp_path):
    from cs2rl.train.resume import resolve_resume_run
    d = tmp_path / "rid"
    # model named by trainer_state.pt missing (only the orphan exists)
    _fake_set(d, model_epochs=(10, ), ts_epoch=9, st_epoch=9)
    with pytest.raises(SystemExit, match="model_000009.pt"):
        resolve_resume_run(tmp_path, run_id="rid")
        # stock-PuffeRL dir: no train_state.pt sidecar → [Resume] message, not FileNotFoundError
    _fake_set(d, model_epochs=(9, ), ts_epoch=9, st_epoch=9)
    (d / "train_state.pt").unlink()
    with pytest.raises(SystemExit, match=r"\[Resume\] .*train_state.pt not found"):
        resolve_resume_run(tmp_path, run_id="rid")


def test_metrics_bound_is_checkpoint_interval_wide_both_sides():
    """Rows are throttled to ≥0.25 s (pufferl.py) while checkpoints fire every
    checkpoint_interval epochs unconditionally, so the last row may trail the
    checkpoint by up to checkpoint_interval-1 epochs."""
    from cs2rl.train.resume import check_resume_metrics_bound as bound
    B, ci, last = 1000, 5, 50_000
    assert bound(last, last, ci, B) == (last - ci * B, last + ci * B)  # row exactly at the checkpoint
    bound(last + (ci - 1) * B, last, ci, B)                            # row ci-1 epochs behind: accepted
    bound(last - ci * B, last, ci, B)                                  # checkpoint ci epochs behind the row: accepted
    with pytest.raises(SystemExit, match=r"\[Resume\] restored global_step"):
        bound(last + (ci + 1) * B, last, ci, B)                        # row further behind: refused
    with pytest.raises(SystemExit, match=r"\[Resume\] restored global_step"):
        bound(last - (ci + 1) * B, last, ci, B)                        # row further ahead of the checkpoint: refused


def test_selfplay_pool_paths_persist_absolute(tmp_path, monkeypatch, capsys):
    from cs2rl.train.selfplay import SelfPlayManager
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ckpt").mkdir()
    for n in ("a.pt", "b.pt"):
        torch.save({}, tmp_path / "ckpt" / n)
    mgr = SelfPlayManager(pool_size=15,
                          p_past=0.0,
                          save_every_epochs=25,
                          win_threshold=0.6,
                          phase_length=50)
    mgr._add_to_pool(Path("ckpt/a.pt"))                # relative, as a relative --checkpoint-dir would give
    mgr._add_to_pool(Path("ckpt/b.pt"))
    sd = mgr.state_dict()
    assert all(Path(p).is_absolute() for p in sd["pool"])
    (tmp_path / "ckpt" / "b.pt").unlink()
    monkeypatch.chdir(tmp_path / "ckpt")               # resume from another cwd
    mgr2 = SelfPlayManager(pool_size=15,
                           p_past=0.0,
                           save_every_epochs=25,
                           win_threshold=0.6,
                           phase_length=50)
    mgr2.load_state_dict(sd)
    assert [p.name for p in mgr2.pool] == ["a.pt"]
    assert "dropped 1/2" in capsys.readouterr().out


def test_analyze_tplant_last_row_wins():
    from cs2rl.experiment.analyze_tplant import dedupe_resume_rows
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


# Explicit training integration: this and its flat-map sibling drive the real
# CLI through a fresh run, a resume and the config.json mismatch refusals.
@pytest.mark.training
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
        sys.executable, "-m", "cs2rl.train", *common, "--timesteps", "20480", "--checkpoint-dir",
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
        sys.executable, "-m", "cs2rl.train", *common, "--timesteps", "30720", "--resume-run",
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
    mismatch[mismatch.index("--num_envs") + 1] = "32"                                           # 16 → 32 envs ⇒ batch_size changes
    r = subprocess.run([
        sys.executable, "-m", "cs2rl.train", *mismatch, "--timesteps", "40960", "--resume-run",
        str(ckpt)
    ],
                       cwd=REPO_ROOT,
                       capture_output=True,
                       text=True,
                       timeout=600)
    assert r.returncode != 0
    assert "config.json mismatch on non-allowlisted keys" in (r.stderr + r.stdout), r.stderr[-2000:]
    assert "batch_size" in (r.stderr + r.stdout)

    # R0-J (Task 14): --gamma is a config key, NOT allowlisted — a resumed run
    # with a different discount is a different experiment and must be refused.
    r = subprocess.run([
        sys.executable, "-m", "cs2rl.train", *common, "--timesteps", "40960", "--gamma", "0.9",
        "--resume-run",
        str(ckpt)
    ],
                       cwd=REPO_ROOT,
                       capture_output=True,
                       text=True,
                       timeout=600)
    assert r.returncode != 0
    assert "config.json mismatch on non-allowlisted keys" in (r.stderr + r.stdout), r.stderr[-2000:]
    assert "gamma" in (r.stderr + r.stdout)


@pytest.mark.training
@pytest.mark.slow
def test_subprocess_resume_run_flat_map_without_pin_pitch_flag(tmp_path):
    """Task 12 (R0-H) binding ruling: a flag-less `--resume-run` of a PINNED run
    must be accepted. The CLI default is pin_pitch=None; the run's config.json
    holds the resolved 1 (flat map). If the config guard ran before
    resolve_pin_pitch, env_config_from_args would yield 0 and every retry of a
    Rung 1 arena run (Task 15's loop) would SystemExit "config.json mismatch".
    16 envs: batch_size floor (see test_subprocess_resume_run)."""
    ckpt = tmp_path / "run"
    common = [
        "--train", "--device", "cpu", "--vec-backend", "serial", "--num_envs", "16", "--map",
        "arena-duel", "--n-active-per-team", "1", "--no-self-play", "--no-dead-run-abort",
        "--checkpoint-interval", "1", "--seed", "3", "--save_every_sec", "100000"
    ]
    r = subprocess.run([
        sys.executable, "-m", "cs2rl.train", *common, "--timesteps", "10240", "--checkpoint-dir",
        str(ckpt), "--run-id", "rid-arena"
    ],
                       cwd=REPO_ROOT,
                       capture_output=True,
                       text=True,
                       timeout=1500)
    assert r.returncode == 0, r.stderr[-3000:]
    cfg = json.loads((ckpt / "config.json").read_text())
    assert cfg["pin_pitch"] == 1 and cfg["env"] == "cs2-arena-duel"
    rows = [json.loads(line) for line in (ckpt / "metrics.jsonl").read_text().splitlines()]
    assert rows, "first run logged nothing"
    r = subprocess.run([
        sys.executable, "-m", "cs2rl.train", *common, "--timesteps", "20480", "--resume-run",
        str(ckpt)
    ],
                       cwd=REPO_ROOT,
                       capture_output=True,
                       text=True,
                       timeout=1500)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "config.json mismatch" not in (r.stderr + r.stdout)
    # n_active=1 ⇒ raw rows = 5× participating steps, so several epochs are
    # appended per run; the stamp sits on the FIRST resumed row only.
    rows2 = [json.loads(line) for line in (ckpt / "metrics.jsonl").read_text().splitlines()]
    new = rows2[len(rows):]
    assert new, "no rows appended after resume"
    assert new[0]["resumed_from_step"] == rows[-1]["step"]
    assert all(row["run_id"] == "rid-arena" for row in new)
    # A different map on resume IS refused (env label is not allowlisted).
    other = list(common)
    other[other.index("--map") + 1] = "simple"
    r = subprocess.run([
        sys.executable, "-m", "cs2rl.train", *other, "--timesteps", "30720", "--resume-run",
        str(ckpt)
    ],
                       cwd=REPO_ROOT,
                       capture_output=True,
                       text=True,
                       timeout=600)
    assert r.returncode != 0
    assert "config.json mismatch on non-allowlisted keys" in (r.stderr + r.stdout)


@pytest.mark.training
@pytest.mark.parametrize("failure", ["checkpoint-set", "restore", "metrics-bound", "setup"])
def test_failed_resume_preserves_checkpoints_and_releases_workers(tmp_path, monkeypatch, failure):
    """Rejected/partial resumes must release real resources without publishing state."""
    from types import SimpleNamespace

    from cs2rl.env.map import make_simple_map
    from cs2rl.train import compose, loop
    from cs2rl.train.trainer import Cs2PuffeRL

    args = SimpleNamespace(
        device="cpu",
        checkpoint_dir=str(tmp_path),
        seed=3,
        num_envs=16,
        pin_pitch=None,
        map="simple",
        map_data=make_simple_map(),
        vec_backend="multiprocessing",
        vec_num_workers=2,
        vec_overwork=False,
        self_play=False,
        timesteps=10240,
        checkpoint_interval=1,
        save_every_sec=float("inf"),
        eval_interval=0,
        wandb=False,
        run_id="preserve-resume",
    )
    loop.train(args)
    sidecar = tmp_path / args.run_id / "train_state.pt"
    state = torch.load(sidecar, weights_only=False)
    assert state["epoch"] == 1
    assert state["global_step"] == 10240
    if failure == "checkpoint-set":
        state["epoch"] += 1
        torch.save(state, sidecar)
    elif failure == "restore":
        # Restore copies these tensors AFTER optimizer/counters but BEFORE ret_var.
        # A missing later field therefore exercises a genuinely partial restoration.
        state["log_alpha"].fill_(-2.5)
        state["ret_mean"].fill_(3.25)
        del state["ret_var"]
        torch.save(state, sidecar)
    elif failure == "metrics-bound":
        (tmp_path /
         "metrics.jsonl").write_text(json.dumps({
             "run_id": args.run_id,
             "step": 1000000
         }) + "\n")
    setup_error = RuntimeError("setup agreement failed")
    if failure == "setup":

        def fail(*a):
            raise setup_error

        # The agreement checks run inside build_trainer, after the trainer exists.
        monkeypatch.setattr(compose, "assert_pin_pitch_agreement", fail)

    # Snapshot the ENTIRE checkpoint namespace, including root aliases and every
    # model/optimizer/sidecar byte, after the intentional input corruption.
    before = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file() and p.name not in {"metrics.jsonl", "config.json"}
    }
    acquired = []
    original_init = Cs2PuffeRL.__init__

    def capture_init(self, *a, **kw):
        """Observe real ownership without replacing construction or shutdown."""
        original_init(self, *a, **kw)
        acquired.append(self)
        assert all(worker.is_alive() for worker in self.vecenv.processes)

    monkeypatch.setattr(Cs2PuffeRL, "__init__", capture_init)
    args.resume_run = str(tmp_path)
    args.timesteps = 20480
    expected = {
        "checkpoint-set": (SystemExit, "inconsistent checkpoint set"),
        "restore": (KeyError, "ret_var"),
        "metrics-bound": (SystemExit, "outside"),
        "setup": (RuntimeError, "setup agreement failed"),
    }
    error_type, message = expected[failure]
    try:
        with pytest.raises(error_type, match=message) as caught:
            loop.train(args)
        if failure == "setup":
            assert caught.value is setup_error
        assert len(acquired) == 1
        trainer = acquired[0]
        trainer.utilization.join(timeout=5)
        assert trainer.utilization.stopped and not trainer.utilization.is_alive()
        for worker in trainer.vecenv.processes:
            worker.join(timeout=5)
            assert not worker.is_alive(), "failed resume leaked a vector worker"
        if failure == "restore":
            assert (trainer.epoch, trainer.global_step) == (1, 10240)
            assert torch.all(trainer._log_alpha_tensor == -2.5)
            assert torch.all(trainer._ret_mean == 3.25)
        after = {
            p.relative_to(tmp_path): p.read_bytes()
            for p in tmp_path.rglob("*")
            if p.is_file() and p.name not in {"metrics.jsonl", "config.json"}
        }
        assert after.keys() == before.keys(), "failed resume changed checkpoint filenames"
        changed = [name for name in before if before[name] != after[name]]
        assert not changed, f"failed resume changed checkpoint bytes: {changed}"
    finally:
        # Regression cleanup cannot checkpoint the broken trainer. Keep failed
        # controls from leaking workers or the non-daemon utilization thread.
        for trainer in acquired:
            trainer.vecenv.close()
            trainer.utilization.stop()
            trainer.utilization.join(timeout=5)
            for worker in trainer.vecenv.processes:
                worker.join(timeout=5)
