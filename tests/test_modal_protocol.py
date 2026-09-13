"""Phase 1 bytes protocol: verify_checkpoint triples and ArtifactIndex.read_file."""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.modal_runner_lib as mrl                 # noqa: E402, I001
from tests.test_modal_runner import FakeArtifactIndex  # noqa: E402, I001

CKPT_OK = b"ckpt-bytes"
DIGEST_OK = mrl.sha256_bytes(CKPT_OK)
SIDECAR_OK = {"size": len(CKPT_OK), "sha256": DIGEST_OK}
SIDECAR_OK_BYTES = json.dumps(SIDECAR_OK, separators=(",", ":")).encode()
SIDECAR_OK_REREAD_WS = json.dumps(SIDECAR_OK, indent=2).encode()
SIDECAR_REPLACED_OBJECT = json.dumps({**SIDECAR_OK, "note": "torn"}).encode()


def _load_ok(*args, **kwargs):
    del args, kwargs
    return {}


def _load_raises(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("not weights-only")


def _load_must_not_run(*args, **kwargs):
    del args, kwargs
    raise AssertionError("load must not run before size and digest pass")


PROTOCOL_CASES = [
    pytest.param(
        "missing_sidecar",
        None,
        CKPT_OK,
        None,
        _load_must_not_run,
        id="missing_sidecar",
    ),
    pytest.param(
        "corrupt_sidecar",
        b"{not-json",
        CKPT_OK,
        None,
        _load_must_not_run,
        id="corrupt_sidecar",
    ),
    pytest.param(
        "missing_checkpoint",
        SIDECAR_OK_BYTES,
        None,
        SIDECAR_OK_BYTES,
        _load_must_not_run,
        id="missing_checkpoint",
    ),
    pytest.param(
        "stale_size",
        json.dumps({
            "size": len(CKPT_OK) + 8,
            "sha256": DIGEST_OK
        }).encode(),
        CKPT_OK,
        json.dumps({
            "size": len(CKPT_OK) + 8,
            "sha256": DIGEST_OK
        }).encode(),
        _load_must_not_run,
        id="stale_size",
    ),
    pytest.param(
        "digest_mismatch",
        json.dumps({
            "size": len(CKPT_OK),
            "sha256": "0" * 64
        }).encode(),
        CKPT_OK,
        json.dumps({
            "size": len(CKPT_OK),
            "sha256": "0" * 64
        }).encode(),
        _load_must_not_run,
        id="digest_mismatch",
    ),
    pytest.param(
        "not_loadable",
        SIDECAR_OK_BYTES,
        CKPT_OK,
        SIDECAR_OK_BYTES,
        _load_raises,
        id="not_loadable",
    ),
    pytest.param(
        "replaced",
        SIDECAR_OK_BYTES,
        CKPT_OK,
        SIDECAR_REPLACED_OBJECT,
        _load_ok,
        id="replaced",
    ),
    pytest.param(
        "ok",
        SIDECAR_OK_BYTES,
        CKPT_OK,
        SIDECAR_OK_REREAD_WS,
        _load_ok,
        id="ok",
    ),
]


@pytest.mark.parametrize("case,sidecar,ckpt,reread,load", PROTOCOL_CASES)
def test_verify_checkpoint_unit_table(case, sidecar, ckpt, reread, load):
    verdict = mrl.verify_checkpoint(sidecar, ckpt, reread, load=load)
    assert verdict.ok is (case == "ok")
    if case == "ok":
        assert verdict.reason is None
        assert verdict.checkpoint_bytes == ckpt
        assert verdict.digest == DIGEST_OK
    else:
        assert verdict.reason == case
        assert verdict.checkpoint_bytes is None
        assert verdict.digest is None


def test_verify_checkpoint_replaced_accepts_missing_or_unparsable_reread():
    missing = mrl.verify_checkpoint(SIDECAR_OK_BYTES, CKPT_OK, None, load=_load_ok)
    assert missing.ok is False
    assert missing.reason == "replaced"
    assert missing.checkpoint_bytes is None
    assert missing.digest is None
    unparsable = mrl.verify_checkpoint(SIDECAR_OK_BYTES, CKPT_OK, b"{not-json", load=_load_ok)
    assert unparsable.ok is False
    assert unparsable.reason == "replaced"
    assert unparsable.checkpoint_bytes is None
    assert unparsable.digest is None


def test_path_derive_run_view_corrupt_status_is_validation_error(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / mrl.STATUS_FILENAME).write_bytes(b"{not-json")
    with pytest.raises(mrl.ValidationError, match="corrupt volume json"):
        mrl.derive_run_view(run_root, now=datetime(2026, 8, 13, 12, 0, tzinfo=UTC))


def test_fake_artifact_index_read_file_replace_after_read():
    index = FakeArtifactIndex()
    path = PurePosixPath("runs/ok-id/STATUS.json")
    index.committed[path] = b"first"
    index.replace_after_read[path] = b"second"
    assert index.read_file(path) == b"first"
    assert index.read_file(path) == b"second"
    index.replace_after_read[path] = None
    assert index.read_file(path) == b"second"
    assert index.read_file(path) is None
    assert path not in index.committed
    assert index.read_file(PurePosixPath("runs/missing/STATUS.json")) is None


def test_unused_artifacts_read_file_returns_none():
    unused = mrl._UnusedArtifacts()
    assert unused.read_file(PurePosixPath("runs/ok-id/STATUS.json")) is None
