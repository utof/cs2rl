"""Unit tests for cs2rl/experiment/lib.py, the experiment-runner shared helpers."""

import json

from cs2rl.experiment import lib as exp_lib


def test_resolve_run_id_counts_existing_dirs(tmp_path, monkeypatch):
    """resolve_run_id counts dirs starting with today's DDMMYY- prefix."""
    from datetime import date

    prefix = date.today().strftime("%d%m%y")
    (tmp_path / f"{prefix}-0-foo").mkdir()
    (tmp_path / f"{prefix}-1-bar").mkdir()
    (tmp_path / "legacy-dir").mkdir()  # doesn't start with today's prefix

    run_id = exp_lib.resolve_run_id("baseline", checkpoints_root=tmp_path)
    assert run_id == f"{prefix}-2-baseline"


def test_resolve_run_id_empty_dir(tmp_path):
    from datetime import date
    prefix = date.today().strftime("%d%m%y")
    run_id = exp_lib.resolve_run_id("first", checkpoints_root=tmp_path)
    assert run_id == f"{prefix}-0-first"


def test_ledger_rewrite_appends_new(tmp_path):
    """New run_id appends to end of ledger."""
    ledger = tmp_path / "results.jsonl"
    ledger.write_text('{"run_id":"a","verdict":"keep"}\n{"run_id":"b","verdict":"discard"}\n')
    exp_lib.ledger_upsert(ledger, {"run_id": "c", "verdict": "keep"})
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    assert [r["run_id"] for r in rows] == ["a", "b", "c"]


def test_ledger_rewrite_replaces_existing(tmp_path):
    """Existing run_id replaces in place, preserving order."""
    ledger = tmp_path / "results.jsonl"
    ledger.write_text('{"run_id":"a","verdict":"keep"}\n'
                      '{"run_id":"b","verdict":"discard"}\n'
                      '{"run_id":"c","verdict":"keep"}\n')
    exp_lib.ledger_upsert(ledger, {"run_id": "b", "verdict": "keep", "new_field": 1})
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    assert [r["run_id"] for r in rows] == ["a", "b", "c"]
    assert rows[1]["verdict"] == "keep"
    assert rows[1]["new_field"] == 1


def test_ledger_rewrite_empty_file(tmp_path):
    ledger = tmp_path / "results.jsonl"
    ledger.touch()
    exp_lib.ledger_upsert(ledger, {"run_id": "a", "verdict": "keep"})
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["run_id"] == "a"


def test_ledger_rewrite_atomic(tmp_path, monkeypatch):
    """Writes through .tmp + rename (inspect side-effect indirectly by ensuring
    no partial file exists after rename)."""
    ledger = tmp_path / "results.jsonl"
    ledger.write_text('{"run_id":"a"}\n')
    exp_lib.ledger_upsert(ledger, {"run_id": "b"})
    # .tmp should be gone
    assert not (tmp_path / "results.jsonl.tmp").exists()
    assert ledger.exists()


def test_env_fingerprint_captures_obs_dim(tmp_path):
    """env_fingerprint reads OBS_DIM from a given train.py-like file.

    NOTE: real ACTION_HEAD_SIZES is exported from spec.action (auto-gen
    from cs2_types.h). Batch 3 dropped HEAD_AIM, so the discrete-side
    tuple is (9, 2, 2, 3, 2, 2, 2) — 7 ints. Tests mirror this string
    exactly because env_fingerprint reads it textually.
    """
    fake_train = tmp_path / "train.py"
    fake_train.write_text("OBS_DIM = 105\nACTION_HEAD_SIZES = (9, 2, 2, 3, 2, 2, 2)\n")
    fake_rewards_h = tmp_path / "cs2_rewards.h"
    fake_rewards_h.write_text("#define INACTION_PENALTY -0.0005f\n"
                              "static const float plant_progress_reward = 0.05f;\n"
                              "static const float terminal_win_bonus = 1.0f;\n")
    fake_env_c = tmp_path / "cs2_env.c"
    fake_env_c.write_text("float bombsite_entry_bonus = 0.3f;\n")
    fp = exp_lib.env_fingerprint(
        train_py_path=fake_train,
        rewards_h_path=fake_rewards_h,
        env_c_path=fake_env_c,
    )
    assert fp["obs_dim"] == 105
    assert fp["action_head_sizes"] == [9, 2, 2, 3, 2, 2, 2]
    assert "INACTION_PENALTY" in fp["reward_terms"]
    assert "plant_progress_reward" in fp["reward_terms"]
    assert "terminal_win_bonus" in fp["reward_terms"]
    assert "bombsite_entry_bonus" in fp["reward_terms"]


def test_behavior_hash_stable(tmp_path):
    """behavior_hash is deterministic given the same inputs."""
    fp1 = {
        "obs_dim": 104,
        "action_head_sizes": [9, 16, 2, 2, 3, 2, 2],
        "reward_terms": ["a", "b"],
        "c_env_sha": "abc",
    }
    fp2 = dict(fp1)                    # copy
    cfg_hash = "hhh"
    assert exp_lib.behavior_hash(fp1, cfg_hash) == exp_lib.behavior_hash(fp2, cfg_hash)
                                       # Different config hash → different behavior hash
    assert exp_lib.behavior_hash(fp1, cfg_hash) != exp_lib.behavior_hash(fp1, "different")


def test_status_writes_two_lines(tmp_path):
    """write_status writes status on line 1, reason on line 2."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    exp_lib.write_status(run_dir, "failed", reason="disk full")
    content = (run_dir / "STATUS.txt").read_text()
    assert content.splitlines() == ["failed", "disk full"]


def test_status_no_reason(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    exp_lib.write_status(run_dir, "training")
    assert (run_dir / "STATUS.txt").read_text().rstrip("\n") == "training"
