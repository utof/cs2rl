"""Demo freshness uses real temporary Git histories, independent of repository age.

The controls break if source comparison hides rename sources, ignores the index
or working tree, strips runtime strings, or accepts an unknown recorded commit.
"""
import subprocess

import numpy as np
import pytest

from cs2rl import train_bc

C_ROOT = train_bc.DEMO_RELEVANT_PATHS[0]


@pytest.fixture
def demo_repo(tmp_path, monkeypatch):
    """A real repository seam; no Git outcomes or production comparisons are mocked."""

    def git(*args):
        return subprocess.run(["git", *args],
                              cwd=tmp_path,
                              check=True,
                              capture_output=True,
                              text=True).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "demo-test@example.invalid")
    git("config", "user.name", "Demo provenance fixture")
    git("config", "core.hooksPath", "/dev/null")
    real_git = train_bc._git
    monkeypatch.setattr(train_bc, "_git", lambda *args, **kwargs: real_git(*args, cwd=tmp_path))
    monkeypatch.setattr(train_bc, "REPO_ROOT", tmp_path)

    def write(path, text):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return target

    return tmp_path, git, write


def _record(git):
    """Commit a fixture snapshot and return its recorded old-format SHA."""
    git("add", ".")
    git("commit", "-qm", "fixture snapshot")
    return git("rev-parse", "HEAD")


@pytest.mark.parametrize("mutation", ["bytes", "comments", "docstrings"])
@pytest.mark.parametrize("state", ["commit", "index", "working"])
def test_relevant_root_move_is_current(demo_repo, mutation, state):
    """Root renames retain provenance, including Python-only documentation edits."""
    root, git, write = demo_repo
    old = "old_env"
    source = '"""Original module documentation."""\n# Original comment\ndef value():\n    """Original function documentation."""\n    return "runtime"\n'
    write(old + "/wrapper.py", source)
    write(old + "/dynamics.c", "int value = 1;\n")
    sha = _record(git)
    (root / "src/cs2rl/env").mkdir(parents=True)
    git("mv", old, "src/cs2rl/env/c")
    if mutation == "comments":
        write(C_ROOT + "/wrapper.py", source.replace("# Original comment", "# Revised comment"))
    elif mutation == "docstrings":
        write(C_ROOT + "/wrapper.py", source.replace("Original", "Revised"))
    if state == "commit":
        _record(git)
    elif state == "index":
        git("add", ".")
    note = train_bc.check_demo_sha(sha)
    if state == "commit":
        assert note is not None and "still reproducible" in note
        assert sha[:9] in note and git("rev-parse", "HEAD")[:9] in note
    else:
        assert note is None


@pytest.mark.parametrize("state", ["commit", "index", "working"])
@pytest.mark.parametrize("replacement", ["VALUE = 2\n", 'VALUE = "changed"\n'])
def test_executable_edits_are_stale_in_every_state(demo_repo, state, replacement):
    """Numeric code and ordinary runtime strings stay significant in every tree."""
    _, git, write = demo_repo
    path = "src/cs2rl/env/map.py"
    write(path, "VALUE = 1\n")
    sha = _record(git)
    write(path, replacement)
    if state == "commit":
        _record(git)
    elif state == "index":
        git("add", path)
    with pytest.raises(ValueError, match="STALE DEMO"):
        train_bc.check_demo_sha(sha)


@pytest.mark.parametrize("mutation", ["python-code", "c-byte", "c-comment", "delete", "add"])
def test_root_move_with_a_source_change_is_stale(demo_repo, mutation):
    """Rename tolerance cannot swallow dynamics edits or added/deleted members."""
    root, git, write = demo_repo
    old = "old_env"
    write(old + "/wrapper.py", "VALUE = 1\n")
    write(old + "/dynamics.c", "int value = 1;\n")
    write(old + "/removed.c", "int removed = 1;\n")
    sha = _record(git)
    (root / "src/cs2rl/env").mkdir(parents=True)
    git("mv", old, "src/cs2rl/env/c")
    if mutation == "python-code":
        write(C_ROOT + "/wrapper.py", "VALUE = 2\n")
    elif mutation == "c-byte":
        write(C_ROOT + "/dynamics.c", "int value = 2;\n")
    elif mutation == "c-comment":
        write(C_ROOT + "/dynamics.c", "int value = 1; // explanation\n")
    elif mutation == "delete":
        git("rm", "-f", C_ROOT + "/removed.c")
    else:
        write(C_ROOT + "/added.c", "int added = 1;\n")
    _record(git)
    with pytest.raises(ValueError, match="STALE DEMO"):
        train_bc.check_demo_sha(sha)


def test_deleted_destination_after_a_root_move_is_stale(demo_repo):
    """An earlier rename must not hide a source whose destination was later removed."""
    root, git, write = demo_repo
    write("old_env/dynamics.c", "int value = 1;\n")
    sha = _record(git)
    (root / "src/cs2rl/env").mkdir(parents=True)
    git("mv", "old_env", "src/cs2rl/env/c")
    _record(git)
    git("rm", C_ROOT + "/dynamics.c")
    _record(git)
    with pytest.raises(ValueError, match="STALE DEMO"):
        train_bc.check_demo_sha(sha)


def test_staged_root_move_then_unstaged_delete_is_stale(demo_repo):
    """Cached rename roots must expose a deleted destination before any commit."""
    root, git, write = demo_repo
    old = "old_env/dynamics.c"
    new = C_ROOT + "/dynamics.c"
    write(old, "int value = 1;\n")
    sha = _record(git)
    (root / "src/cs2rl/env").mkdir(parents=True)
    git("mv", "old_env", "src/cs2rl/env/c")
    (root / new).unlink()
    cached = git("diff", "--cached", "--raw", "-z", "--find-renames=1%", sha, "--")
    working = git("diff", "--raw", "-z", "--find-renames=1%", sha, "--")
    assert cached.split("\0")[0].rsplit(" ", 1)[-1] == "R100", cached
    assert cached.split("\0")[1:3] == [old, new], cached
    assert working.split("\0")[0].rsplit(" ", 1)[-1] == "D", working
    assert working.split("\0")[1:2] == [old], working
    assert git("status", "--porcelain").startswith("RD ")
    with pytest.raises(ValueError, match="STALE DEMO"):
        train_bc.check_demo_sha(sha)


@pytest.mark.parametrize("path,original,indexed", [
    ("src/cs2rl/env/map.py", b"VALUE = 1\n", b"VALUE = 2\n"),
    (C_ROOT + "/dynamics.c", b"int value = 1;\n", b"int value = 2;\n"),
    (C_ROOT + "/dynamics.c", b"int value = 1;\n", b"int value = 1;\n\n"),
    (C_ROOT + "/asset.dat", b"\xff\x01\n", b"\xff\x02\n"),
])
def test_index_edits_are_stale_when_working_bytes_are_restored(demo_repo, path, original, indexed):
    """MM can hide staged Python code or exact non-Python edits in the working diff."""
    root, git, write = demo_repo
    target = write(path, "")
    target.write_bytes(original)
    sha = _record(git)
    target.write_bytes(indexed)
    git("add", path)
    target.write_bytes(original)
    assert git("status", "--porcelain") == "MM " + path
    assert git("diff", "--raw", "-z", sha, "--") == ""
    cached = git("diff", "--cached", "--raw", "-z", sha, "--")
    assert cached.split("\0")[0].rsplit(" ", 1)[-1] == "M", cached
    assert cached.split("\0")[1:2] == [path], cached
    actual_index = subprocess.run(["git", "show", ":" + path],
                                  cwd=root,
                                  check=True,
                                  capture_output=True).stdout
    assert actual_index == indexed and target.read_bytes() == original
    with pytest.raises(ValueError, match="STALE DEMO"):
        train_bc.check_demo_sha(sha)


@pytest.mark.parametrize("mutation", ["docstrings", "python-move", "c-move", "binary-move"])
def test_index_documentation_and_pure_moves_are_current(demo_repo, mutation):
    """Index blob comparisons apply normalization without rejecting staged moves."""
    root, git, write = demo_repo
    source = '"""Old documentation."""\nVALUE = 1\n'
    old = "src/cs2rl/env/map.py" if mutation == "docstrings" else "old_env/member.py"
    if mutation == "c-move":
        old = "old_env/member.c"
        source = "int value = 1;\n"
    elif mutation == "binary-move":
        old = "old_env/member.dat"
    target = write(old, source)
    source_bytes = source.encode()
    if mutation == "binary-move":
        source_bytes = b"\xff\x01\n"
        target.write_bytes(source_bytes)
    sha = _record(git)
    if mutation == "docstrings":
        write(old, source.replace("Old", "New"))
        git("add", old)
        target.write_text(source)
        new = old
        expected_index = source.replace("Old", "New").encode()
        assert git("status", "--porcelain") == "MM " + old
        assert git("diff", "--raw", "-z", sha, "--") == ""
    else:
        (root / "src/cs2rl/env").mkdir(parents=True)
        git("mv", "old_env", "src/cs2rl/env/c")
        new = C_ROOT + "/" + target.name
        expected_index = source_bytes
    cached = git("diff", "--cached", "--raw", "-z", "--find-renames=1%", sha, "--")
    status = cached.split("\0")[0].rsplit(" ", 1)[-1]
    assert status == ("M" if mutation == "docstrings" else "R100"), cached
    assert cached.split("\0")[1:3] == ([old, new] if status == "R100" else [old, ""])
    actual_index = subprocess.run(["git", "show", ":" + new],
                                  cwd=root,
                                  check=True,
                                  capture_output=True).stdout
    assert actual_index == expected_index
    assert (root / new).read_bytes() == source_bytes
    assert train_bc.check_demo_sha(sha) is None


@pytest.mark.parametrize("state", ["commit", "index", "working"])
def test_python_documentation_edits_are_current(demo_repo, state):
    """Only AST docstring positions are ignored; runtime string literals are retained."""
    _, git, write = demo_repo
    path = "src/cs2rl/env/map.py"
    write(
        path,
        '"""Old module docs."""\nclass C:\n    """Old class docs."""\n    def value(self):\n        """Old function docs."""\n        return "runtime"\nasync def other():\n    """Old async docs."""\n    return 1\n'
    )
    sha = _record(git)
    text = (demo_repo[0] / path).read_text().replace("Old", "New")
    write(path, "# Added explanation\n" + text)
    if state == "commit":
        _record(git)
    elif state == "index":
        git("add", path)
    note = train_bc.check_demo_sha(sha)
    assert (note is None) == (state != "commit")


@pytest.mark.parametrize("source", [
    '"""Old"""\nVALUE = __doc__\n',
    'def f():\n    """Old"""\n    return 1\nVALUE = f.__doc__\n',
    'def f():\n    """Old"""\n    return 1\nVALUE = getattr(f, "__doc__")\n',
    'import inspect\ndef f():\n    """Old"""\n    return 1\nVALUE = inspect.getdoc(f)\n',
])
def test_explicit_docstring_reads_remain_significant(demo_repo, source):
    """A locally observable docstring is runtime data, so its edit must be stale."""
    _, git, write = demo_repo
    path = "src/cs2rl/env/map.py"
    write(path, source)
    sha = _record(git)
    write(path, source.replace("Old", "New"))
    with pytest.raises(ValueError, match="STALE DEMO"):
        train_bc.check_demo_sha(sha)


@pytest.mark.parametrize("mutation", ["add", "delete", "syntax"])
def test_head_does_not_hide_working_tree_changes(demo_repo, mutation):
    """HEAD equality cannot bypass additions, deletions, or invalid Python source."""
    _, git, write = demo_repo
    write("src/cs2rl/env/map.py", "VALUE = 1\n")
    sha = _record(git)
    if mutation == "add":
        write(C_ROOT + "/extra.c", "int extra = 1;\n")
    elif mutation == "delete":
        (demo_repo[0] / "src/cs2rl/env/map.py").unlink()
    else:
        write("src/cs2rl/env/map.py", "VALUE = (\n")
    with pytest.raises(ValueError, match="STALE DEMO"):
        train_bc.check_demo_sha(sha)


def test_check_demo_sha_accepts_head_and_rejects_unknown_shas(demo_repo):
    """A real fixture HEAD passes; unavailable history stays closed and short SHAs fail."""
    _, git, write = demo_repo
    write("src/cs2rl/env/map.py", "VALUE = 1\n")
    head = _record(git)
    assert train_bc.check_demo_sha(head) is None
    with pytest.raises(ValueError, match="STALE DEMO"):
        train_bc.check_demo_sha("d" * 40)
    note = train_bc.check_demo_sha("d" * 40, allow_stale=True)
    assert note is not None
    assert "STALE" in note
    with pytest.raises(ValueError, match="not self-identifying"):
        train_bc.check_demo_sha("abc123", allow_stale=True)


def test_check_demo_sha_tolerates_commits_that_cannot_change_a_demo(demo_repo):
    """The unrelated-commit case executes every time, independent of live history."""
    _, git, write = demo_repo
    write("src/cs2rl/env/map.py", "VALUE = 1\n")
    sha = _record(git)
    write("unrelated.txt", "unrelated edit\n")
    head = _record(git)
    note = train_bc.check_demo_sha(sha)
    assert note is not None
    assert "still reproducible" in note and sha[:9] in note and head[:9] in note


def test_without_a_checkout_returns_an_unverifiable_note(demo_repo):
    """Missing Git metadata preserves the existing explicit unverifiable result."""
    root, _, _ = demo_repo
    (root / ".git").rename(root / "saved-git")
    note = train_bc.check_demo_sha("d" * 40)
    assert note is not None
    assert "not a git checkout" in note


def test_allow_stale_retains_the_warning(demo_repo):
    """The escape hatch names the actual stale source and caller-provided demo name."""
    _, git, write = demo_repo
    write("src/cs2rl/env/map.py", "VALUE = 1\n")
    sha = _record(git)
    write("src/cs2rl/env/map.py", "VALUE = 2\n")
    note = train_bc.check_demo_sha(sha, allow_stale=True, name="episode.npz")
    assert note is not None
    assert note.startswith("episode.npz: STALE") and "--allow-stale-demos" in note


def test_old_format_demo_sha_still_loads_after_a_move(demo_repo):
    """The consumer accepts an existing NPZ schema with no new metadata field."""
    root, git, write = demo_repo
    write("old_env/dynamics.c", "int value = 1;\n")
    sha = _record(git)
    (root / "src/cs2rl/env").mkdir(parents=True)
    git("mv", "old_env", "src/cs2rl/env/c")
    _record(git)
    demos = root / "demos"
    demos.mkdir()
    np.savez(demos / "episode.npz",
             obs=np.zeros((1, train_bc.OBS_DIM), dtype=np.float32),
             discrete_actions=np.zeros((1, train_bc.ACTION_DIM), dtype=np.int32),
             continuous_actions=np.zeros((1, train_bc.AIM_DIM), dtype=np.float32),
             dones=np.array([True]),
             tick_count=1,
             seed=0,
             carrier_idx=0,
             spawn_area=0,
             git_sha=sha,
             map=train_bc.EXPECTED_MAP,
             OBS_DIM=train_bc.OBS_DIM,
             ACTION_DIM=train_bc.ACTION_DIM,
             AIM_DIM=train_bc.AIM_DIM)
    assert len(train_bc.load_demos(demos, verbose=False)) == 1


def test_runtime_string_change_is_stale(demo_repo):
    """Ignoring docstrings cannot erase a changed ordinary string literal."""
    _, git, write = demo_repo
    path = "src/cs2rl/env/map.py"
    write(path, 'VALUE = "before"\n')
    sha = _record(git)
    write(path, 'VALUE = "after"\n')
    with pytest.raises(ValueError, match="STALE DEMO"):
        train_bc.check_demo_sha(sha)


@pytest.mark.parametrize("state", ["commit", "index", "working"])
@pytest.mark.parametrize("member,replacement", [
    ("dynamics.c", "int value = 2;\n"),
    ("rules.h", "int value = 1; // explanatory comment\n"),
    ("asset.txt", "changed bytes\n"),
])
def test_same_path_nonpython_edits_are_stale(demo_repo, state, member, replacement):
    """Reject a real M record, even when the changed bytes only add a C comment."""
    _, git, write = demo_repo
    path = C_ROOT + "/" + member
    write(path, "int value = 1;\n")
    sha = _record(git)
    write(path, replacement)
    if state == "commit":
        _record(git)
    elif state == "index":
        git("add", path)
    # Inspect actual Git output independently of the production record decoder:
    # a tiny move+edit fixture previously died in A/D before reaching this branch.
    raw = git("diff", "--raw", "-z", "--find-renames=1%", sha, "--")
    status = raw.split("\0")[0].rsplit(" ", 1)[-1]
    assert status == "M", raw
    print(f"non-Python fixture: {member} {state} status={status}")
    with pytest.raises(ValueError, match="STALE DEMO"):
        train_bc.check_demo_sha(sha)


@pytest.mark.parametrize("state", ["commit", "index", "working"])
@pytest.mark.parametrize("edit", ["code", "comment"])
def test_similar_c_move_plus_edit_is_stale(demo_repo, state, edit):
    """An R score below 100 must still compare C bytes, including C comments."""
    root, git, write = demo_repo
    old = "old_env/dynamics.c"
    retained = "".join(f"int retained_{i} = {i};\n" for i in range(12))
    source = "int value = 1;\n" + retained
    write(old, source)
    sha = _record(git)
    (root / "src/cs2rl/env").mkdir(parents=True)
    git("mv", "old_env", "src/cs2rl/env/c")
    new = C_ROOT + "/dynamics.c"
    changed = source.replace("value = 1",
                             "value = 2") if edit == "code" else source + "// explanation\n"
    write(new, changed)
    if state == "commit":
        _record(git)
    elif state == "index":
        git("add", new)
    raw = git("diff", "--raw", "-z", "--find-renames=1%", sha, "--")
    status = raw.split("\0")[0].rsplit(" ", 1)[-1]
    assert status.startswith("R") and 0 < int(status[1:]) < 100, raw
    assert raw.split("\0")[1:3] == [old, new], raw
    print(f"C move+edit fixture: {edit} {state} status={status}")
    with pytest.raises(ValueError, match="STALE DEMO"):
        train_bc.check_demo_sha(sha)
