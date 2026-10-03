"""Emit deterministic branch-review JSON; findings do not fail the invocation.

Run: python scripts/branch_dup_report.py --base <commit> [--head HEAD] [--repo .]
Optional --base-graph/--head-graph accept version-1 JSON envelopes of native
codebase-memory-mcp results: project, status_before/status_after (index_status
with verbose=True), coverage_before/coverage_after (arrays of coverage pages
checking ALL tracked paths, at most 128 per page, and the complete '.' scope),
similarity and arity (query_graph results). Capture coverage/status around the
queries without editing the checkout. Use full indexing and these queries:
  MATCH (a)-[:SIMILAR_TO]->(b) RETURN a.qualified_name, b.qualified_name, a.file_path, b.file_path
  MATCH (f) WHERE f.param_count > 12 RETURN f.qualified_name, f.file_path, f.param_count
No MCP transport or similarity algorithm is embedded here. Missing exports are
UNMEASURED. Native schemas are validated, not guessed or silently coerced.

Python bodies are exact canonical ASTs including docstrings, names and literals;
async, nested functions and methods are included. Parameters count every declared
positional-only, positional, keyword-only, *args and **kwargs, including self.
Identity is path plus lexical name; renamed symbols can appear as new findings.
Non-Python arity is graph evidence only where path coverage supports it.
Adapted from the owner's existing private branch-end mechanics.py, without its
machine paths, branch assertions, or documentation-specific census.
"""
import argparse
import ast
import hashlib
import json
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path


def _git(repo, *args):
    """Read Git objects bound to the requested checkout, never the caller's subdirectory."""
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def _scan(repo, revision):
    """Census pinned tracked blobs; preserve failed parses rather than manufacturing clean zeros."""
    paths = _git(repo, "ls-tree", "-r", "--name-only", "-z", revision).decode().split("\0")[:-1]
    groups = defaultdict(list)
    arity, errors, locations = {}, [], {}
    functions = 0
    for path in paths:
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(_git(repo, "show", f"{revision}:{path}"), path)
        except (SyntaxError, ValueError) as exc:
            errors.append({"path": path, "error": str(exc)})
            continue
        parents = {
            id(child): node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            functions += 1
            names = [node.name]
            parent = parents.get(id(node))
            while parent is not None:
                if isinstance(parent, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.append(parent.name)
                parent = parents.get(id(parent))
            symbol = f"{path}:{'.'.join(reversed(names))}"
            # Repeated definitions share a lexical name; ordinal suffixes keep both visible.
            if symbol in locations:
                symbol += f"#{sum(name.split('#')[0] == symbol for name in locations) + 1}"
            locations[symbol] = node.lineno
            body = ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False)
            groups[body].append(symbol)
            args = node.args
            count = len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs) + bool(
                args.vararg) + bool(args.kwarg)
            if count > 12:
                arity[symbol] = count
    pairs = {
        tuple(sorted((a, b)))
        for rows in groups.values()
        for a, b in combinations(rows, 2) if a.rsplit(":", 1)[0] != b.rsplit(":", 1)[0]
    }
    summary = {
        "status": "PARTIAL" if errors else "MEASURED",
        "tracked_files": len(paths),
        "python_files": sum(path.endswith(".py") for path in paths),
        "functions": functions,
        "parse_errors": errors,
        "non_python_paths": [p for p in paths if not p.endswith(".py")],
        "locations": {
            name: line
            for name, line in locations.items()
            if name in arity or any(name in pair for pair in pairs)
        }
    }
    return paths, pairs, arity, summary


def _delta(old, new):
    """Separate existing findings from introduced/removed identities, with stable ordering."""
    return {
        "base": sorted(old),
        "head": sorted(new),
        "new": sorted(new - old),
        "removed": sorted(old - new)
    }


def _query_rows(result, columns):
    """Validate native query row totals; the known hard ceiling cannot establish completeness."""
    if result.get("isError"):
        raise ValueError("graph query returned an error")
    raw = result["content"][0]["text"]
    header = re.search(r"^rows: (\d+)  \(cols: (.+)\)$", raw, re.M)
    total = re.search(r"^total: (\d+)$", raw, re.M)
    rows = [shlex.split(line.strip()) for line in raw.splitlines() if line.startswith("  ")]
    if not header or not total or header[2] != columns or not (len(rows) == int(header[1]) == int(
            total[1]) < 100000):
        raise ValueError("graph query schema or complete row count missing")
    if any(len(row) != len(columns.split()) for row in rows):
        raise ValueError("graph query malformed row")
    return rows


def _graph_snapshot(path, revision, paths, repo):
    """Consume revision/generation-checked native evidence and expose every tracked coverage gap."""
    try:
        data = json.loads(Path(path).read_text())
        if data["version"] != 1:
            raise ValueError("unsupported graph export version")
        project = data["project"]
        for phase in ("before", "after"):
            result = data[f"status_{phase}"]
            status = result["structuredContent"]
            if result.get("isError") or status["project"] != project or status[
                    "status"] != "ready" or status["git"]["head_sha"] != revision:
                raise ValueError("graph project, ready status or revision mismatch")
        generations, coverage, scope_gaps = set(), {}, {}
        for phase in ("before", "after"):
            checked, root_seen = {}, False
            for result in data[f"coverage_{phase}"]:
                page = result["structuredContent"]
                meta = page["metadata"]
                if result.get("isError") or page["project"] != project or not meta[
                        "generation_matches"] or not meta["hash_records_complete"] or meta[
                            "recording_status"] != "complete" or meta["index_mode"] != "full":
                    raise ValueError("graph coverage incomplete, stale or not full-mode")
                generations.add(meta["generation"])
                for scope in page["scopes"]:
                    if scope["has_more"] or len(scope["entries"]) != scope["total"]:
                        raise ValueError("graph coverage scope pagination incomplete")
                    root_seen |= scope["scope"] == "."
                    for gap in scope["entries"]:
                        scope_gaps[(gap["path"], gap["kind"])] = gap
                for row in page["paths"]:
                    if row["freshness"] in ("metadata_changed", "missing"):
                        raise ValueError("graph coverage path is stale or missing")
                    checked[row["path"]] = row
            if not root_seen or set(checked) != set(paths):
                raise ValueError(
                    "graph coverage must check every tracked path and complete root scope")
            coverage.update(checked)
        if len(generations) != 1 or not next(iter(generations)):
            raise ValueError("graph generation changed during export")
        prefix = project + "."
        rows = _query_rows(data["similarity"],
                           "a.qualified_name b.qualified_name a.file_path b.file_path")
        arity_rows = _query_rows(data["arity"], "f.qualified_name f.file_path f.param_count")
        if any(not name.startswith(prefix) for row in rows
               for name in row[:2]) or any(not row[0].startswith(prefix) for row in arity_rows):
            raise ValueError("graph contains foreign project symbols")
        if any(file not in coverage for row in rows
               for file in row[2:]) or any(row[1] not in coverage or int(row[2]) <= 12
                                           for row in arity_rows):
            raise ValueError("graph query includes untracked paths or invalid arity")
        pairs = {tuple(sorted(name[len(prefix):] for name in row[:2])) for row in rows}
        gaps = [
            row for row in coverage.values()
            if row["status"] != "no_recorded_issue" or row["freshness"] != "metadata_match"
        ]
        fallback = []
        for row in sorted(gaps, key=lambda row: row["path"]):
            blob = _git(repo, "show", f"{revision}:{row['path']}")
            ranges = [bounds for gap in row["coverage"] for bounds in gap.get("ranges", [])]
            lines = blob.decode(errors="replace").splitlines() if ranges else []
            fallback.append({
                "path":
                row["path"],
                "bytes":
                len(blob),
                "sha1":
                hashlib.sha1(blob).hexdigest(),
                "ranges": [{
                    **bounds, "source": lines[bounds["start"] - 1:bounds["end"]]
                } for bounds in ranges]
            })
        arity, unmeasured = set(), []
        for name, file, count in arity_rows:
            if file.endswith(".py"):
                continue                                                   # Exact Python AST overrides graph alias/signature attribution.
            row = coverage.get(file)
            if row is None or row in gaps:
                unmeasured.append({
                    "symbol": name[len(prefix):],
                    "path": file,
                    "parameters": int(count)
                })
            else:
                arity.add((name[len(prefix):], file, int(count)))
        details = {
            "project": project,
            "revision": revision,
            "generation": next(iter(generations)),
            "scope_gaps": [scope_gaps[key] for key in sorted(scope_gaps)],
            "raw_similarity_rows": len(rows),
            "coverage_gaps": sorted(gaps, key=lambda row: row["path"]),
            "source_fallback": fallback,
            "unmeasured_non_python_arity": unmeasured
        }
        return pairs, arity, details
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        raise ValueError(f"graph export {path}: {exc}") from exc


def _growth(repo, base, head):
    """Read NUL numstat including rename triples; binary changes have no numeric growth."""
    fields = iter(
        _git(repo, "diff", "--numstat", "-z", "--find-renames", base, head,
             "--").decode().split("\0")[:-1])
    growth, binary = [], []
    for record in fields:
        added, removed, path = record.split("\t", 2)
        if not path:
            next(fields)                               # Old rename path; numstat measures changes at the new path.
            path = next(fields)
        if added == "-":
            binary.append(path)
        elif int(added) - int(removed) > 300:
            growth.append({
                "path": path,
                "added": int(added),
                "removed": int(removed),
                "net": int(added) - int(removed)
            })
    return sorted(growth, key=lambda row: row["path"]), sorted(binary)


def main(argv=None):
    """Pin Git inputs once, then print data; bad invocation or unreadable evidence exits two."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="explicit Git commit/ref")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--base-graph")
    parser.add_argument("--head-graph")
    args = parser.parse_args(argv)
    try:
        base, head = [
            _git(args.repo, "rev-parse", "--verify", "--end-of-options",
                 f"{ref}^{{commit}}").decode().strip() for ref in (args.base, args.head)
        ]
        old_paths, old_pairs, old_arity, old_scan = _scan(args.repo, base)
        paths, pairs, arity, scan = _scan(args.repo, head)
        graph = {
            "status": "UNMEASURED",
            "reason":
            "base/head graph exports not supplied; similarity and non-Python arity unmeasured"
        }
        if bool(args.base_graph) != bool(args.head_graph):
            raise ValueError("both graph exports are required for a graph delta")
        if args.base_graph:
            gp_old, ga_old, gd_old = _graph_snapshot(args.base_graph, base, old_paths, args.repo)
            gp, ga, gd = _graph_snapshot(args.head_graph, head, paths, args.repo)
            graph = {
                "status":
                "PARTIAL" if gd_old["coverage_gaps"] or gd["coverage_gaps"] or gd_old["scope_gaps"]
                or gd["scope_gaps"] else "MEASURED",
                "caveat":
                "best-effort graph signal; source reads do not recover missing similarity or non-Python signatures",
                "base":
                gd_old,
                "head":
                gd,
                "similarity":
                _delta(gp_old, gp),
                "non_python_arity": [{
                    "symbol": name,
                    "path": path,
                    "parameters": count,
                    "new": (name, path, count) not in ga_old
                } for name, path, count in sorted(ga)]
            }
        growth, binary = _growth(args.repo, base, head)
        report = {
            "base":
            base,
            "head":
            head,
            "source":
            "pinned tracked Git revisions; working tree ignored",
            "body_convention":
            "exact canonical AST including docstrings, identifiers and literals",
            "parameter_convention":
            "all declared parameters including self, positional-only, keyword-only, *args and **kwargs",
            "python_scan": {
                "base": old_scan,
                "head": scan
            },
            "graph":
            graph,
            "identical_bodies":
            _delta(old_pairs, pairs),
            "python_arity": [{
                "symbol": name,
                "parameters": count,
                "new": old_arity.get(name) != count
            } for name, count in sorted(arity.items())],
            "growth_over_300":
            growth,
            "binary_numstat":
            binary
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
