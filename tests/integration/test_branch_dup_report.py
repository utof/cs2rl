"""Real Git/CLI controls for branch review data, including discovery outside src/."""
import json
import subprocess
import sys
from typing import Any

import pytest

from tests.conftest import REPO_ROOT


def _git(repo, *args):
    """Keep fixture Git operations independent of the report's implementation."""
    return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()


def _commit(repo, files):
    """Commit fixture blobs; these isolated repos have no project hooks."""
    for name, content in files.items():
        path = repo / name
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode() if isinstance(content, str) else content)
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit",
         "-qm", "fixture")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    """No shared environment or main repository mutations for fixture revisions."""
    _git(tmp_path, "init", "-q")
    return tmp_path


def _cli(repo, base, *extra):
    """Run the actual entry point, so argument and output defects remain visible."""
    return subprocess.run([
        sys.executable,
        str(REPO_ROOT / "scripts/branch_dup_report.py"), "--repo",
        str(repo), "--base", base, *extra
    ],
                          capture_output=True,
                          text=True)


def _report(repo, base, *extra):
    """Findings are data and must leave a successful invocation successful."""
    result = _cli(repo, base, *extra)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _snapshot(repo, revision, project, pairs=()) -> dict[str, Any]:
    """Native tool envelope shape measured on the real project, with literal fixture rows."""
    paths = _git(repo, "ls-tree", "-r", "--name-only", revision).splitlines()
    status = {
        "structuredContent": {
            "project": project,
            "status": "ready",
            "git": {
                "head_sha": revision
            }
        }
    }
    coverage = {
        "structuredContent": {
            "project":
            project,
            "metadata": {
                "generation": "fixture-generation",
                "generation_matches": True,
                "recording_status": "complete",
                "hash_records_complete": True,
                "index_mode": "full"
            },
            "paths": [{
                "path": name,
                "status": "no_recorded_issue",
                "freshness": "metadata_match",
                "coverage": []
            } for name in paths],
            "scopes": [{
                "scope": ".",
                "total": 0,
                "has_more": False,
                "entries": []
            }],
        }
    }
    rows = "".join(f"  {project}.{a} {project}.{b} one.py one.py\n" for a, b in pairs)
    return {
        "version": 1,
        "project": project,
        "requests": {
            "similarity": {
                "project":
                project,
                "max_rows":
                100000,
                "query":
                "MATCH (a)-[:SIMILAR_TO]->(b) RETURN a.qualified_name, b.qualified_name, a.file_path, b.file_path ORDER BY a.qualified_name, b.qualified_name"
            },
            "arity": {
                "project":
                project,
                "max_rows":
                100000,
                "query":
                "MATCH (f) WHERE f.param_count > 12 RETURN f.qualified_name, f.file_path, f.param_count ORDER BY f.qualified_name"
            },
        },
        "status_before": status,
        "status_after": status,
        "coverage_before": [coverage],
        "coverage_after": [coverage],
        "similarity": {
            "content": [{
                "type":
                "text",
                "text":
                f"rows: {len(pairs)}  (cols: a.qualified_name b.qualified_name a.file_path b.file_path)\n{rows}total: {len(pairs)}\n"
            }]
        },
        "arity": {
            "content": [{
                "type":
                "text",
                "text":
                'rows: 0  (cols: f.qualified_name f.file_path f.param_count)\ntotal: 0\n'
            }]
        }
    }


def _exports(repo, base, head):
    """Persist explicit base/head snapshots as a CLI caller would."""
    paths = []
    for label, data in (("base", base), ("head", head)):
        path = repo / f"{label}-export.json"
        path.write_text(json.dumps(data))
        paths.extend([f"--{label}-graph", str(path)])
    return paths


def test_new_bodies_and_signatures_are_discovered_outside_a_file_allowlist(repo):
    """A src-only census, lost namespaces, or incorrect parameter convention must fail."""
    base = _commit(repo, {
        "old.py": "def old():\n    return 23\n",
        "other.py": "def existing():\n    return 23\n"
    })
    head = _commit(
        repo, {
            "deploy/unexpected.py":
            "async def added():\n    return 23\n"
            "class Scope:\n    def method(self):\n        return 23\n"
            "def outer():\n    def nested():\n        return 23\n    return nested\n"
            "def boundary(a,b,c,d,e,f,g,h,i,j,k,l):\n    return 12\n"
            "def high(a,b,c,d,e,f,/,g,h,i,j,*,k,l,m):\n    return 13\n"
            "class Wide:\n    def call(self,a,b,c,d,e,f,g,h,i,j,*args,**kwargs):\n        return 14\n"
        })
    report = _report(repo, base)
    assert report["head"] == head
    pairs = report["identical_bodies"]["head"]
    assert ["old.py:old", "other.py:existing"] in pairs
    assert ["deploy/unexpected.py:Scope.method", "old.py:old"] in pairs
    assert ["deploy/unexpected.py:outer.nested", "old.py:old"] in pairs
    assert ["deploy/unexpected.py:added", "old.py:old"] in report["identical_bodies"]["new"]
    assert ["old.py:old", "other.py:existing"] not in report["identical_bodies"]["new"]
    assert [(row["symbol"], row["parameters"], row["new"])
            for row in report["python_arity"]] == [("deploy/unexpected.py:Wide.call", 13, True),
                                                   ("deploy/unexpected.py:high", 13, True)]
    assert report["graph"]["status"] == "UNMEASURED"
    (repo / "old.py").write_text("broken dirty source")
    assert _report(repo, base) == report


def test_exact_bodies_keep_docstrings_identifiers_and_literals(repo):
    """Lossy normalization would turn these deliberate differences into duplicates."""
    base = _commit(repo, {"one.py": 'def one(x):\n    "one"\n    return x + 1\n'})
    _commit(
        repo, {
            "two.py": 'def two(x):\n    "two"\n    return x + 1\n',
            "three.py": 'def three(y):\n    "one"\n    return y + 1\n',
            "four.py": 'def four(x):\n    "one"\n    return x + 2\n'
        })
    assert _report(repo, base)["identical_bodies"]["head"] == []


def test_growth_boundary_rename_deletion_and_binary_are_explicit(repo):
    """A rename-as-add parser, >=300 threshold, or binary coercion must fail."""
    base = _commit(
        repo, {
            "rename me.txt": "a\n" * 400,
            "deleted.txt": "b\n" * 400,
            "edited.txt": "a\n" * 20,
            "binary.dat": b"\x00one"
        })
    _git(repo, "mv", "rename me.txt", "renamed\tfile.txt")
    _commit(
        repo, {
            "deleted.txt": None,
            "boundary.txt": "x\n" * 300,
            "outside/large.txt": "x\n" * 301,
            "edited.txt": "b\n" * 321,
            "binary.dat": b"\x00two"
        })
    report = _report(repo, base)
    assert [(row["path"], row["net"])
            for row in report["growth_over_300"]] == [("edited.txt", 301),
                                                      ("outside/large.txt", 301)]
    assert report["binary_numstat"] == ["binary.dat"]


def test_graph_delta_normalizes_project_and_direction_and_discloses_partial_coverage(repo):
    """A zero placeholder, directed comparison, or Python graph alias must fail."""
    base = _commit(repo, {"one.py": "x=1\n", "native.h": "void native(void);\n"})
    head = _commit(repo, {"one.py": "x=2\n"})
    old = _snapshot(repo, base, "base-project", [("one.a", "one.b")])
    new = _snapshot(repo, head, "head-project", [("one.b", "one.a"), ("one.a", "one.c")])
    new["arity"]["content"][0]["text"] = (
        'rows: 2  (cols: f.qualified_name f.file_path f.param_count)\n'
        '  head-project.native native.h "13"\n  head-project.alias one.py "14"\ntotal: 2\n')
    for phase in ("before", "after"):
        new[f"coverage_{phase}"][0]["structuredContent"]["scopes"][0].update(total=1,
                                                                             entries=[{
                                                                                 "path":
                                                                                 "ignored",
                                                                                 "kind":
                                                                                 "not_indexed_dir"
                                                                             }])
    report = _report(repo, base, *_exports(repo, old, new))
    assert report["graph"]["status"] == "PARTIAL"
    assert report["graph"]["similarity"]["new"] == [["one.a", "one.c"]]
    assert report["graph"]["non_python_arity"] == [{
        "symbol": "native",
        "path": "native.h",
        "parameters": 13,
        "new": True
    }]
    assert report["python_arity"] == []


@pytest.mark.parametrize("damage", [
    "revision", "generation", "pagination", "rows", "foreign", "stale", "missing_path", "mode",
    "error", "incomplete", "untracked", "row_limit", "query_limit", "no_requests"
])
def test_graph_evidence_is_rejected_when_it_cannot_support_the_revision(repo, damage):
    """Never accept stale, foreign, truncated, incomplete or absent coverage as a clean zero."""
    base = _commit(repo, {"one.py": "x=1\n"})
    snap = _snapshot(repo, base, "project", [("one.a", "one.b")])
    coverage = snap["coverage_after"][0]["structuredContent"]
    if damage == "revision":
        snap["status_after"]["structuredContent"]["git"]["head_sha"] = "f" * 40
    elif damage == "generation":
        coverage["metadata"]["generation_matches"] = False
    elif damage == "pagination":
        coverage["scopes"][0]["has_more"] = True
    elif damage == "rows":
        snap["similarity"]["content"][0]["text"] = "rows: 1\ntotal: 1\n"
    elif damage == "foreign":
        snap["similarity"]["content"][0]["text"] = (
            'rows: 1  (cols: a.qualified_name b.qualified_name a.file_path b.file_path)\n'
            '  foreign.a foreign.b one.py one.py\ntotal: 1\n')
    elif damage == "stale":
        coverage["paths"][0]["freshness"] = "metadata_changed"
    elif damage == "missing_path":
        coverage["paths"] = []
    elif damage == "mode":
        coverage["metadata"]["index_mode"] = "fast"
    elif damage == "error":
        snap["similarity"]["isError"] = True
    elif damage == "incomplete":
        coverage["metadata"]["recording_status"] = "pending"
    elif damage == "untracked":
        snap["similarity"]["content"][0]["text"] = (
            'rows: 1  (cols: a.qualified_name b.qualified_name a.file_path b.file_path)\n'
            '  project.one.a project.one.b one.py untracked.py\ntotal: 1\n')
    elif damage == "row_limit":
        snap["requests"]["similarity"]["max_rows"] = 1
    elif damage == "query_limit":
        snap["requests"]["similarity"]["query"] += " LIMIT 1"
    elif damage == "no_requests":
        del snap["requests"]
    result = _cli(repo, base, *_exports(repo, _snapshot(repo, base, "base"), snap))
    assert result.returncode == 2
    assert "graph" in result.stderr.lower()


def test_parse_failures_and_unsupported_sources_are_disclosed(repo):
    """A skipped invalid blob or unsupported source must not produce a completeness claim."""
    base = _commit(repo, {"ok.py": "x=1\n"})
    _commit(repo, {"bad.py": "def broken(\n", "native.rs": "fn native() {}\n"})
    report = _report(repo, base)
    assert report["python_scan"]["head"]["status"] == "PARTIAL"
    assert report["python_scan"]["head"]["parse_errors"][0]["path"] == "bad.py"
    assert "native.rs" in report["python_scan"]["head"]["non_python_paths"]
    assert _cli(repo, "missing-ref").returncode == 2
