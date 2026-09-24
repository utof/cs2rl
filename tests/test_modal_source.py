"""Behavior tests for scripts.modal_runner.source: clean HEAD, safe extraction, bundles.

One of the eight per-module runner test files (RUNNER_TEST_FILES in
tests/modal_runner_tables.py), split by module from the one unsplit runner test
file in W4. A test lives in the file of the module whose behaviour it tests:
the seam manifest (tests/fixtures/modal_test_seam_manifest.json) records that
placement, and the seam gate in tests/test_modal_packaging.py checks it from
below with the reach floor. Tests reach private library names through their
owning submodules; the package facade exposes the production caller surface.
Helpers reached by tests in two or more seam files live in
tests/modal_test_helpers.py, with ownership recomputed by classify_seam.
"""
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # scripts/ is a namespace package; tests import scripts.modal_runner
    # the same way the later CLIs will. Do not rely on the editable install.
    sys.path.insert(0, str(ROOT))

import scripts.modal_runner as mrl                                     # noqa: E402, I001
from scripts.modal_runner import core, source                          # noqa: E402, I001
from tests.modal_test_helpers import _git, _init_source_repo           # noqa: E402

# ── Run preconditions: clean-HEAD validation ───────────────────────────────


def test_validate_clean_head_accepts_matching_clean_commit(tmp_path):
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    assert mrl.validate_clean_head(repo, sha) == sha
    assert mrl.validate_clean_head(repo, sha.upper()) == sha


@pytest.mark.parametrize(
    "bad_sha",
    [
        "not-a-sha",
        "abc",
        "g" * 40,
        "a" * 39,
        "a" * 41,
    ],
)
def test_validate_clean_head_rejects_non_hex_sha(tmp_path, bad_sha):
    repo = _init_source_repo(tmp_path)
    with pytest.raises(mrl.ValidationError):
        mrl.validate_clean_head(repo, bad_sha)


def test_validate_clean_head_rejects_unknown_object(tmp_path):
    repo = _init_source_repo(tmp_path)
    with pytest.raises(mrl.ValidationError):
        mrl.validate_clean_head(repo, "b" * 40)


def test_validate_clean_head_rejects_sha_that_is_not_head(tmp_path):
    repo = _init_source_repo(tmp_path)
    (repo / "readme.txt").write_text("second\n")
    _git(repo, "add", "readme.txt")
    _git(repo, "commit", "-qm", "second")
    parent = _git(repo, "rev-parse", "HEAD^")
    with pytest.raises(mrl.ValidationError):
        mrl.validate_clean_head(repo, parent)


def test_validate_clean_head_rejects_unstaged_tracked_change(tmp_path):
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    (repo / "readme.txt").write_text("dirty\n")
    with pytest.raises(mrl.ValidationError):
        mrl.validate_clean_head(repo, sha)


def test_validate_clean_head_rejects_staged_tracked_change(tmp_path):
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    (repo / "readme.txt").write_text("staged\n")
    _git(repo, "add", "readme.txt")
    with pytest.raises(mrl.ValidationError):
        mrl.validate_clean_head(repo, sha)


# ── Source bundle: safe archive extraction ─────────────────────────────────


def _write_tar(path: Path, info: tarfile.TarInfo, data: bytes = b"") -> None:
    import io

    with tarfile.open(path, "w") as tar:
        payload = io.BytesIO(data) if info.type == tarfile.REGTYPE else None
        if payload is not None:
            info.size = len(data)
        tar.addfile(info, payload)


def test_safe_extract_accepts_git_archive(tmp_path):
    repo = _init_source_repo(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    archive = tmp_path / "src.tar"
    _git(repo, "archive", "--format=tar", f"--output={archive}", sha)
    dest = tmp_path / "out"
    dest.mkdir()
    source.safe_extract_git_archive(archive, dest)
    assert (dest / "readme.txt").read_text() == "hello\n"


def test_safe_extract_rejects_unsafe_members(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    cases: list[tarfile.TarInfo] = []
    link = tarfile.TarInfo("link")
    link.type = tarfile.SYMTYPE
    link.linkname = "readme.txt"
    cases.append(link)
    hard = tarfile.TarInfo("hard")
    hard.type = tarfile.LNKTYPE
    hard.linkname = "readme.txt"
    cases.append(hard)
    fifo = tarfile.TarInfo("fifo")
    fifo.type = tarfile.FIFOTYPE
    cases.append(fifo)
    abs_path = tarfile.TarInfo("/etc/passwd")
    abs_path.type = tarfile.REGTYPE
    cases.append(abs_path)
    traversal = tarfile.TarInfo("foo/../../etc/passwd")
    traversal.type = tarfile.REGTYPE
    cases.append(traversal)
    for index, info in enumerate(cases):
        archive = tmp_path / f"bad-{index}.tar"
        _write_tar(archive, info, data=b"x")
        with pytest.raises(mrl.ValidationError):
            source.safe_extract_git_archive(archive, dest)


# ── Source bundle: deterministic archive + provenance sidecar ──────────────


def _source_repo_with_noise(tmp_path: Path) -> Path:
    repo = _init_source_repo(tmp_path)
    script = repo / "tool.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o755)
    _git(repo, "add", "tool.sh")
    _git(repo, "update-index", "--chmod=+x", "tool.sh")
    _git(repo, "commit", "-qm", "add executable")
    (repo / ".env").write_text("SECRET=1\n")
    (repo / "noise.txt").write_text("untracked\n")
    (repo / "outputs").mkdir()
    (repo / "outputs" / "run.log").write_text("nope\n")
    (repo / ".venv").mkdir()
    (repo / ".venv" / "pyvenv.cfg").write_text("x\n")
    docs_git = repo / "docs" / ".git"
    docs_git.mkdir(parents=True)
    (docs_git / "HEAD").write_text("ref: refs/heads/main\n")
    return repo


def _open_bundle_tar(bundle: Path) -> tarfile.TarFile:
    import gzip

    return tarfile.open(fileobj=gzip.open(bundle, "rb"), mode="r:")


def test_source_bundle_excludes_untracked_and_is_deterministic(tmp_path):
    repo = _source_repo_with_noise(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", f"{sha}^{{tree}}")
    first = tmp_path / "a.tar.gz"
    second = tmp_path / "b.tar.gz"
    prov_a = mrl.create_source_bundle(repo, sha, first)
    prov_b = mrl.create_source_bundle(repo, sha, second)
    assert first.read_bytes() == second.read_bytes()
    assert prov_a.archive_sha256 == prov_b.archive_sha256 == core.sha256_file(first)
    assert prov_a.commit == sha
    assert prov_a.tree == tree
    with _open_bundle_tar(first) as tar:
        names = set(tar.getnames())
    assert "readme.txt" in names
    assert "tool.sh" in names
    assert ".cs2rl-provenance.json" in names
    assert ".env" not in names
    assert "noise.txt" not in names
    assert "outputs/run.log" not in names
    assert ".venv/pyvenv.cfg" not in names
    assert "docs/.git/HEAD" not in names


def test_source_bundle_hash_changes_for_new_commit(tmp_path):
    repo = _source_repo_with_noise(tmp_path)
    sha1 = _git(repo, "rev-parse", "HEAD")
    first = tmp_path / "old.tar.gz"
    mrl.create_source_bundle(repo, sha1, first)
    (repo / "readme.txt").write_text("changed\n")
    _git(repo, "add", "readme.txt")
    _git(repo, "commit", "-qm", "change")
    sha2 = _git(repo, "rev-parse", "HEAD")
    second = tmp_path / "new.tar.gz"
    mrl.create_source_bundle(repo, sha2, second)
    assert core.sha256_file(first) != core.sha256_file(second)


def test_source_bundle_normalizes_modes_and_gzip_header(tmp_path):
    repo = _source_repo_with_noise(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    bundle = tmp_path / "src.tar.gz"
    mrl.create_source_bundle(repo, sha, bundle)
    header = bundle.read_bytes()[:10]
    flags = header[3]
    mtime = int.from_bytes(header[4:8], "little")
    assert flags & 0x08 == 0
    assert mtime == 0
    with _open_bundle_tar(bundle) as tar:
        for member in tar.getmembers():
            mode = member.mode & 0o777
            if member.isdir():
                assert mode == 0o755
            elif member.name.endswith("tool.sh"):
                assert mode == 0o755
            else:
                assert mode == 0o644
        sidecar = tar.extractfile(".cs2rl-provenance.json")
        assert sidecar is not None
        import json

        payload = json.loads(sidecar.read().decode())
    assert payload["commit"] == sha
    assert payload["tree"] == _git(repo, "rev-parse", f"{sha}^{{tree}}")
