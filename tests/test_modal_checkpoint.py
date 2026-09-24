"""Behavior tests for scripts.modal_runner.checkpoint: resume and completed-run checks.

One of the per-module runner test files (RUNNER_TEST_FILES in
tests/modal_runner_tables.py). Before you add, move or delete a test here, or
add a helper, read THE PLACEMENT RULE FOR RUNNER TESTS in
tests/test_modal_packaging.py: which file a test belongs in, what the change
costs in the seam manifest, and where helpers go.
"""
import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # scripts/ is a namespace package; tests import scripts.modal_runner
    # the same way the later CLIs will. Do not rely on the editable install.
    sys.path.insert(0, str(ROOT))

import scripts.modal_runner as mrl                                     # noqa: E402, I001
from scripts.modal_runner import checkpoint, core, request             # noqa: E402, I001
from tests.modal_test_helpers import (                                 # noqa: E402
    _live_batch_size, _minimal_completed_tree, _no_torch, _write_metrics)

# ── Resume input: local checkpoint hash + weights-only load ────────────────


def test_validate_local_checkpoint_hashes_and_maps_paths(tmp_path):
    import torch

    ckpt = tmp_path / "warm.pt"
    torch.save({"weight": torch.tensor([1.0, 2.0])}, ckpt)
    provenance = mrl.validate_local_checkpoint(ckpt)
    digest = core.sha256_file(ckpt)
    assert provenance.sha256 == digest
    assert provenance.size == ckpt.stat().st_size
    assert provenance.client_path == mrl.INPUTS_ROOT / "sha256" / f"{digest}.pt"
    assert provenance.mount_path == Path("/artifacts/inputs/sha256") / f"{digest}.pt"


def test_validate_local_checkpoint_rejects_non_checkpoint(tmp_path):
    junk = tmp_path / "nope.txt"
    junk.write_text("not a checkpoint\n")
    with pytest.raises(mrl.ValidationError):
        mrl.validate_local_checkpoint(junk)
    missing = tmp_path / "missing.pt"
    with pytest.raises(mrl.ValidationError):
        mrl.validate_local_checkpoint(missing)


# ── Completion evidence: config normalisation, completed-run checks ────────


def test_normalize_config_strips_only_checkpoint_data_dir():
    raw = {"env": "cs2-dust2", "data_dir": "/artifacts/runs/x/checkpoints", "seed": 2}
    normalized = checkpoint.normalize_config_for_transport(raw)
    assert "data_dir" not in normalized
    assert normalized == {"env": "cs2-dust2", "seed": 2}
    assert checkpoint.normalize_config_for_transport(normalized) == normalized


def test_validate_completed_run_accepts_representative_metrics(tmp_path):
    # compute_batch_dims moved train.py -> train_config.py in the post-rung1a
    # refactor (2026-08-31); the runner's AGENTS_PER_ENV/BPTT_HORIZON mirror is
    # pinned against wherever it actually lives, not against train.py by habit.
    train_src = (ROOT / "src" / "train_config.py").read_text()
    fn = ast.parse(train_src)
    for node in ast.walk(fn):
        if isinstance(node, ast.FunctionDef) and node.name == "compute_batch_dims":
            body = ast.get_source_segment(train_src, node)
            assert body is not None
            assert "agents_per_env = 10" in body
            assert "bptt_horizon = 64" in body
            assert "num_envs * agents_per_env * bptt_horizon" in body
            break
    else:
        raise AssertionError("live compute_batch_dims not found in src/train_config.py")
    assert request.AGENTS_PER_ENV == 10
    assert request.BPTT_HORIZON == 64
    run_root, manifest, effective, ckpt = _minimal_completed_tree(tmp_path)
    evidence = checkpoint.validate_completed_run(run_root, manifest)
    assert evidence.last_step == effective
    assert evidence.checkpoint_sha256 == core.sha256_file(ckpt)
    assert evidence.config_hash == manifest.config_hash
    assert manifest.batch_size == _live_batch_size(256)
    assert manifest.effective_timesteps == (30_000_000 // manifest.batch_size) * manifest.batch_size
    assert manifest.effective_timesteps >= manifest.batch_size
    assert json.loads((run_root / "checkpoints" / "config.json").read_text())["env"] == "cs2-dust2"
    assert manifest.effective_map == "simple"


@pytest.mark.parametrize(
    "defect", ["bad_ckpt", "empty", "malformed", "nonmonotonic", "wrong_hash", "short_step"])
def test_validate_completed_run_rejects_bad_evidence(tmp_path, defect):
    run_root, manifest, effective, ckpt = _minimal_completed_tree(tmp_path)
    if defect == "bad_ckpt":
        ckpt.write_text("nope")
    elif defect == "empty":
        (run_root / "checkpoints" / "metrics.jsonl").write_text("")
    elif defect == "malformed":
        (run_root / "checkpoints" / "metrics.jsonl").write_text("{nope\n")
    elif defect == "nonmonotonic":
        _write_metrics(run_root / "checkpoints" / "metrics.jsonl", [100, 50])
    elif defect == "wrong_hash":
        manifest = mrl.Manifest(**{**manifest.to_dict(), "config_hash": "e" * 64})
    elif defect == "short_step":
        _write_metrics(run_root / "checkpoints" / "metrics.jsonl", [effective - 1])
    with pytest.raises(mrl.ValidationError):
        checkpoint.validate_completed_run(run_root, manifest)


# ── Runner-interpreter checkpoint validation: completed-run check ──────────
#
# The Modal runner process and the training child are DIFFERENT interpreters.
# The runner is the image's standalone /usr/local/bin/python (only uv + modal);
# torch lives exclusively in the PREBUILT_PYTHON venv that runs train.py.
# Verified in a live container on 2026-08-14:
#   runner_executable=/usr/local/bin/python  runner_torch=MISSING
# Every test above runs on a laptop where `import torch` succeeds, so none of
# them can see this. These do: they force the torch-less runner condition.


def test_completed_run_validates_without_runner_torch(tmp_path, monkeypatch):
    """validate_completed_run torch-loads too: without the fallback every clean
    exit is misfiled as failed/invalid_evidence and no run can ever complete."""
    run_root, manifest, effective, ckpt = _minimal_completed_tree(tmp_path)
    _no_torch(monkeypatch, prebuilt=sys.executable)

    evidence = checkpoint.validate_completed_run(run_root, manifest)

    assert evidence.last_step == effective
    assert evidence.checkpoint_sha256 == core.sha256_file(ckpt)
