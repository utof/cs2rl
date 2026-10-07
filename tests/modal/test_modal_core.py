"""Behavior tests for scripts.modal_runner.core: mounted paths, statuses, the manifest.

One of the per-module runner test files (RUNNER_TEST_FILES in
tests/modal/modal_runner_tables.py). Before you add, move or delete a test here, or
add a helper, read THE PLACEMENT RULE FOR RUNNER TESTS in
tests/modal/test_modal_packaging.py: which file a test belongs in, what the change
costs in the seam manifest, and where helpers go.
"""
from pathlib import PurePosixPath

import pytest

from tests.conftest import REPO_ROOT

ROOT = REPO_ROOT

import scripts.modal_runner as mrl                                     # noqa: E402, I001
from scripts.modal_runner import core                                  # noqa: E402, I001
from tests.modal.modal_test_helpers import _make_manifest              # noqa: E402

# ── Core names: mounted paths and the Status enum ──────────────────────────


def test_mounted_path_translates_client_roots():
    assert mrl.mounted_path(mrl.SOURCES_ROOT /
                            "abc.tar.gz") == core.VOLUME_MOUNT / "sources" / "abc.tar.gz"
    assert mrl.mounted_path(mrl.INPUTS_ROOT / "sha256" / "d.pt") == (core.VOLUME_MOUNT / "inputs" /
                                                                     "sha256" / "d.pt")
    assert mrl.mounted_path(mrl.RUNS_ROOT / "ok-id" /
                            "STATUS.json") == (core.VOLUME_MOUNT / "runs" / "ok-id" / "STATUS.json")


@pytest.mark.parametrize(
    "relative",
    [
        PurePosixPath("/artifacts/sources/x"),
        PurePosixPath("runs/../secrets"),
        PurePosixPath("runs/ok/../../etc/passwd"),
    ],
)
def test_mounted_path_rejects_absolute_and_dotdot(relative):
    with pytest.raises(mrl.ValidationError):
        mrl.mounted_path(relative)


def test_status_enum_splits_terminal_and_nonterminal():
    assert core.Status.PREPARING.value == "preparing"
    assert core.Status.BUILDING.value == "building"
    assert core.Status.TRAINING.value == "training"
    assert {s.value
            for s in core.Status} >= {
                "preparing",
                "building",
                "training",
                "completed",
                "failed",
                "interrupted",
                "build_failed",
            }


# ── Run manifest: the authoritative map, not the legacy env ────────────────


def test_manifest_records_authoritative_simple_map_not_legacy_env():
    manifest = _make_manifest()
    payload = manifest.to_dict()
    assert payload["schema_version"] == 1
    assert payload["attempt_id"] == "attempt-a"
    assert payload["source_archive_sha256"] == "c" * 64
    assert payload["resume_sha256"] is None
    assert payload["resume_size"] is None
    assert payload["resume_source_path"] is None
    assert payload["modal_version"] == "1.4.3"
    digest = payload["image_digest"]
    assert isinstance(digest, str) and digest.startswith("sha256:")
    assert payload["gpu"] == "T4"
    assert payload["vec_workers"] == 8
    assert payload["effective_map"] == "simple"
    assert payload["cpu_request"] == payload["cpu_soft_limit"] == 8
    assert payload["memory_request_mib"] == payload["memory_hard_limit_mib"] == 16384
    # Live config.json currently lies; the manifest must not copy that field.
    live_config = {"env": "cs2-dust2", "seed": 2, "data_dir": "/artifacts/runs/ok-id/checkpoints"}
    assert live_config["env"] == "cs2-dust2"
    assert "env" not in payload
    assert payload["effective_map"] == "simple"
