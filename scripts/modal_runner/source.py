"""Deterministic source bundles and safe extraction."""
from __future__ import annotations

import gzip
import json
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from . import core
from .core import (
    _COMMIT_SHA_RE,
    _SAFE_TAR_TYPES,
    PROVENANCE_NAME,
    SourceProvenance,
    ValidationError,
)


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git with a list argv. Never a shell string; cwd is the target repo."""
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def validate_clean_head(repo: Path, requested_sha: str) -> str:
    """Require requested_sha to be the clean HEAD commit of repo.

    Checks, in order: 40-hex shape, `cat-file -e <sha>^{commit}`, equality with
    `rev-parse HEAD`, then both unstaged and staged diffs vs that SHA. Untracked
    files are intentionally ignored — they are not shipped (git archive).
    Returns the lowercase canonical SHA.
    """
    if not isinstance(requested_sha, str) or _COMMIT_SHA_RE.fullmatch(requested_sha) is None:
        raise ValidationError(f"git-sha must be 40 hex chars, got {requested_sha!r}")
    canonical = requested_sha.lower()
    probe = _run_git(repo, "cat-file", "-e", f"{canonical}^{{commit}}")
    if probe.returncode != 0:
        raise ValidationError(f"git-sha is not a commit object: {canonical}")
    head = _run_git(repo, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise ValidationError(f"cannot resolve HEAD in {repo}: {head.stderr.strip()}")
    head_sha = head.stdout.strip()
    if head_sha != canonical:
        raise ValidationError(f"git-sha {canonical} is not HEAD ({head_sha})")
    unstaged = _run_git(repo, "diff", "--quiet", canonical, "--")
    if unstaged.returncode != 0:
        raise ValidationError("tracked working tree does not match git-sha")
    staged = _run_git(repo, "diff", "--cached", "--quiet", canonical, "--")
    if staged.returncode != 0:
        raise ValidationError("index does not match git-sha")
    return canonical


def _reject_unsafe_tar_member(member: tarfile.TarInfo) -> None:
    """Fail closed on anything git archive should never emit for our sources."""
    name = member.name
    if member.issym() or member.islnk():
        raise ValidationError(f"archive member is a link: {name}")
    if member.isfifo() or member.ischr() or member.isblk():
        raise ValidationError(f"archive member is a special file: {name}")
    if member.type not in _SAFE_TAR_TYPES:
        raise ValidationError(f"archive member has unsafe type {member.type!r}: {name}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise ValidationError(f"archive member escapes destination: {name}")


def safe_extract_git_archive(archive: Path, destination: Path) -> None:
    """Extract a git tar only after every member has passed the safety scan.

    The producer is trusted git, but this is the last check before bytes land
    on disk. Scan first, then extract — do not extract-and-rollback.
    """
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        for member in members:
            _reject_unsafe_tar_member(member)
        # 3.12 filter='data' is a second belt: no links, no absolute paths.
        tar.extractall(destination, members=members, filter="data")


def _staging_members(staging: Path) -> list[tuple[str, Path]]:
    """Return (posix-relpath, path) pairs including parent dirs, sorted."""
    members: dict[str, Path] = {}
    for path in staging.rglob("*"):
        relative = PurePosixPath(*path.relative_to(staging).parts)
        members[str(relative)] = path
        parent = relative.parent
        while parent.parts:
            key = str(parent)
            members.setdefault(key, staging.joinpath(*parent.parts))
            parent = parent.parent
    return sorted(members.items(), key=lambda item: item[0])


def _repack_deterministic(staging: Path, destination: Path) -> None:
    """Write a gzip tar with frozen metadata so the digest is reproducible.

    uid/gid/mtime/names are zeroed. Dirs and git-executable files are 0755;
    everything else is 0644. gzip mtime=0 and no header filename, so two
    machines packaging the same commit produce the same bytes.
    """
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for name, path in _staging_members(staging):
                    info = tarfile.TarInfo(name=name)
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    if path.is_dir():
                        info.type = tarfile.DIRTYPE
                        info.mode = 0o755
                        tar.addfile(info)
                        continue
                    info.type = tarfile.REGTYPE
                    info.mode = 0o755 if path.stat().st_mode & stat.S_IXUSR else 0o644
                    info.size = path.stat().st_size
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)


def create_source_bundle(repo: Path, sha: str, destination: Path) -> SourceProvenance:
    """Package a clean HEAD commit into a content-addressed gzip tar.

    Staging lives under TemporaryDirectory — never inside the repo — so a crash
    cannot leave a dirty tree or a partial archive next to source.
    """
    canonical = validate_clean_head(repo, sha)
    tree_proc = _run_git(repo, "rev-parse", f"{canonical}^{{tree}}")
    if tree_proc.returncode != 0:
        raise ValidationError(f"cannot resolve tree for {canonical}: {tree_proc.stderr.strip()}")
    tree = tree_proc.stdout.strip()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cs2rl-src-") as tmp:
        tmp_path = Path(tmp)
        raw_tar = tmp_path / "git.tar"
        packed = tmp_path / "packed.tar.gz"
        archive = _run_git(repo, "archive", "--format=tar", f"--output={raw_tar}", canonical)
        if archive.returncode != 0:
            raise ValidationError(f"git archive failed: {archive.stderr.strip()}")
        staging = tmp_path / "tree"
        staging.mkdir()
        safe_extract_git_archive(raw_tar, staging)
        sidecar = {"commit": canonical, "tree": tree}
        (staging / PROVENANCE_NAME).write_text(json.dumps(sidecar, sort_keys=True) + "\n")
        _repack_deterministic(staging, packed)
        destination.write_bytes(packed.read_bytes())
    return SourceProvenance(
        commit=canonical,
        tree=tree,
        archive_sha256=core.sha256_file(destination),
        archive_path=destination,
    )
