"""Phase 1 bytes protocol: verify_checkpoint triples and ArtifactIndex.read_file."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import PurePosixPath

import pytest

from tests.conftest import REPO_ROOT

ROOT = REPO_ROOT

import scripts.modal_runner as mrl                                     # noqa: E402, I001
from scripts.modal_runner import checkpoint, training                  # noqa: E402, I001
from tests.modal.test_modal_state import FakeArtifactIndex             # noqa: E402, I001

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)
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


def test_protocol_cases_cover_every_reason_token_and_ok():
    """`PROTOCOL_CASES` has one case per `CHECKPOINT_REASON_TOKENS` entry, plus "ok", and no other.

    gh#197. The table is hand-written, so a case deleted from it would silently
    shrink `test_verify_checkpoint_unit_table`, and a token added to the
    protocol without a row here would go untested. Both are red here. The
    first value of each case is the reason the unit table asserts, so it must
    equal the id: a mislabelled row would test one token under another's name.
    """
    ids = [str(param.id) for param in PROTOCOL_CASES]
    assert len(set(ids)) == len(ids), f"duplicate case id: {ids}"
    expected = {*checkpoint.CHECKPOINT_REASON_TOKENS, "ok"}
    assert set(ids) == expected, (
        f"missing cases {sorted(expected - set(ids))}; unknown cases {sorted(set(ids) - expected)}")
    for param in PROTOCOL_CASES:
        assert param.values[0] == param.id, f"case {param.id} asserts reason {param.values[0]!r}"


def _status_json(**overrides) -> bytes:
    payload = {
        "schema_version": 1,
        "status": "training",
        "attempt_id": "attempt-1",
        "updated_at": NOW.isoformat(),
    }
    payload.update(overrides)
    return json.dumps(payload).encode()


# Every operator-facing message derive_run_view_from_bytes can raise, pinned to
# the input shape that produces it.
#
# WHY a table rather than one case: the two branches wrap ValidationError
# asymmetrically on purpose (see the function's docstring), and that asymmetry
# is invisible from any single row. Both branch-specific messages regressed to
# the bare "corrupt volume json" once already, under a green test.
#
# PITFALL: that regression stayed green because the pin was authored against
# the regressed value. 4cf68b8, the commit that introduced the unified
# protocol, wrote match="corrupt volume json" for a corrupt STATUS file -- the
# message the bug produced, not the one the spec asks for. A pin copied from
# observed behaviour can never fail. So check this table against what the
# pre-unification adapters did and what spec 2.2 asks for, not against what
# the code in front of you currently returns.
#
# Patterns are anchored so none can pass against a longer message that merely
# contains it; do not relax one to a bare substring.
MESSAGE_CASES = [
    pytest.param(b"{not-json", None, r"^corrupt volume status json$", id="corrupt_status_json"),
    pytest.param(
        _status_json(updated_at="not-a-timestamp"),
        None,
        r"^corrupt volume status json$",
        id="status_bad_updated_at",
    ),
    pytest.param(
        json.dumps({
            "status": "training"
        }).encode(),
        None,
        r"^corrupt volume status json$",
        id="status_missing_field",
    ),
    pytest.param(
        None,
        b"{not-json",
        r"^corrupt volume reservation json$",
        id="corrupt_reservation_json",
    ),
    pytest.param(
        None,
        json.dumps({}).encode(),
        r"^corrupt volume reservation json$",
        id="reservation_missing_created_at",
    ),
    pytest.param(
        None,
        json.dumps(["not", "a", "mapping"]).encode(),
        r"^corrupt volume reservation json$",
        id="reservation_is_a_list",
    ),
    pytest.param(
        None,
        json.dumps({
            "created_at": "not-a-timestamp"
        }).encode(),
        r"^corrupt volume timestamp$",
        id="reservation_bad_timestamp",
    ),
    pytest.param(None, None, r"^no STATUS\.json or reservation\.json$", id="both_absent"),
]


@pytest.mark.parametrize(("status_bytes", "reservation_bytes", "pattern"), MESSAGE_CASES)
def test_derive_run_view_from_bytes_messages(status_bytes, reservation_bytes, pattern):
    with pytest.raises(mrl.ValidationError, match=pattern):
        mrl.derive_run_view_from_bytes(status_bytes, reservation_bytes, now=NOW)


def test_bare_corrupt_volume_json_is_the_unlabelled_default():
    # The unlabelled fallback. Three call sites leave it at the default:
    # verify_checkpoint's two (the first sidecar parse and the reread), which
    # catch the ValidationError and discard the message in favour of a
    # corrupt_sidecar / replaced token, and modal_backfill_sidecar's sidecar
    # round-trip check, which lets it out. That last one is therefore the only
    # path on which an operator ever reads this string -- no
    # derive_run_view_from_bytes input reaches it, because both of its branches
    # pass their own label in. Pinned so the default is not "cleaned up" into a
    # branch message, which would mislabel backfill.
    with pytest.raises(mrl.ValidationError, match=r"^corrupt volume json$"):
        mrl.load_volume_json(b"{not-json")


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
    unused = training._UnusedArtifacts()
    assert unused.read_file(PurePosixPath("runs/ok-id/STATUS.json")) is None
