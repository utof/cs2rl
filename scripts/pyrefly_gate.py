#!/usr/bin/env python3
"""Freeze this repo's pre-existing pyrefly errors and fail on any change to them.

WHY THIS EXISTS INSTEAD OF `pyrefly check --baseline`
-----------------------------------------------------
pyrefly 1.2.0 ships a `--baseline` flag that looks like exactly this tool. It is
not usable as a gate here. Its suppression key is `(path, kind, start-column)` --
line-insensitive, message-insensitive, and with UNLIMITED MULTIPLICITY. One
baselined entry is a standing licence for that (file, kind, column), not a
budget: five distinct new `missing-attribute` errors planted at src/cs2rl/train.py
(a flat module then; #205 part 3 made it a package) column 14 were all swallowed in a
single run, reporting `0 errors`, exit 0.

Measured leave-one-out over this repo's own 752 errors: 488/752 = 64.9% would be
invisible if they were new. Python's 4-space indents make column collisions
common, which is why the number is so high. The repo-wide counts in this module
(752, 826 and the rates over 752) were measured when the gate was added (0833d5e);
they are not today's counts.

If you are reading this because a hand-rolled comparison beside a built-in flag
looks like wheel-reinvention: it is not. Run
tests/integration/test_pyrefly_gate.py::test_a_new_error_at_a_column_the_baseline_would_blind_is_caught
-- it freezes the same fixture both ways and asserts --baseline goes GREEN while
this gate goes RED.

THE KEY, AND WHAT IT COSTS
--------------------------
    (path, name, concise_description, source_line)

`name` is pyrefly's error-kind field (there is no `kind` key in its JSON).
`source_line` is the stripped text of the offending line, read from the
MATERIALISED tree -- never from the working tree, or the staged-vs-worktree hole
this gate exists to close reopens inside the key itself.

No line number and no column, which buys shift tolerance: prepending 40 blank
lines to a 20-error file produces zero differences. What it costs is a measured
residual of 17.3% -- 130 of 752 frozen occurrences sit across 49 keys of
multiplicity >= 2, so silencing one occurrence while copy-pasting an identical
erroring line into the same file nets to zero. Adding `line` would close that and
cost the shift tolerance, which is the more common event. The trade is priced,
not overlooked. (For comparison: the 3-field key without `source_line` leaves
398/752 = 52.9%, and --baseline leaves 64.9%.)

WHY IT READS THE INDEX AND NOT THE WORKING TREE
-----------------------------------------------
`git checkout-index -a --prefix=` materialises exactly what is about to be
committed, in ~0.06s, moving nothing and stashing nothing -- no `git stash
--keep-index`, which would be reckless in a repo with a permanently dirty tree
containing a private nested documentation repo. A working-tree-reading gate ships broken
staged code green whenever the author has already fixed their copy.

This also makes the gate hermetic: a developer's in-progress error no longer reds
an unrelated test run.

A NOTE ON THE MATERIALISED TREE
-------------------------------
The interpreter's site-package-path includes <repo>/src, so it is not obvious
that the <tmp> copy wins. It does: `search-path` beats site-packages, verified
both for a plant in <tmp>/src/ and for a *caller* resolving the <tmp> copy rather
than the installed one. The whole design rests on this.

`--python-interpreter-path` is passed explicitly and the gate never relies on
VIRTUAL_ENV or on a .venv symlink. Without it the count depends on how the caller
was launched -- `uv run` exports VIRTUAL_ENV and a bare shell does not, which
would make the same test pass or fail on invocation style (752 vs 826).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import NoReturn

SNAPSHOT_NAME = "pyrefly-snapshot.json"
CONFIG_NAME = "pyrefly.toml"

# The four fields that make up a snapshot record. Stored on disk as a JSON object
# per line; the comparison key is PROJECTED from that record by key_of(), and the
# same projection is applied to pyrefly's live output. Keeping the projection
# shared is what makes "drop source_line from the key" a symmetric, single-defect
# change rather than a whole-file shape mismatch.
FIELDS = ("path", "name", "concise_description", "source_line")


def key_of(record: dict) -> tuple:
    """Project a snapshot record (or a live entry) onto the comparison key."""
    return tuple(record[f] for f in FIELDS)


def die(msg: str) -> NoReturn:
    """Abort loudly. Every guard in this file exits through here.

    The NoReturn annotation is load-bearing, not decoration: without it pyrefly
    cannot see that control stops here, and every variable assigned after a
    guard reads as possibly-uninitialized. The gate caught that in its own
    source on its first run.
    """
    print(f"pyrefly-gate: {msg}", file=sys.stderr)
    sys.exit(1)


def git(project: Path, *args: str) -> str:
    """Run git bound to `project`.

    PITFALL: every git call MUST be -C <project>. The gate never chdirs (so that
    `uv run` is never invoked from a foreign cwd), which means a bare `git` call
    is scoped to whatever cwd the caller had. From a subdirectory that silently
    truncates everything: `ls-files -u` reports 0 on an unmerged index,
    checkout-index materialises only the subtree with exit 0 and no warning, and
    the parity check below then compares a truncated tree against an identically
    truncated `ls-files` -- an assertion comparing a thing to itself.
    """
    out = subprocess.run(
        ["git", "-C", str(project), *args],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        die(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def load_snapshot(path: Path) -> Counter:
    """Read the one-object-per-line snapshot into a multiset of keys."""
    counts: Counter = Counter()
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            die(f"{path}:{lineno} is not valid JSON: {exc}")
        missing = [f for f in FIELDS if f not in rec]
        if missing:
            die(f"{path}:{lineno} is missing {missing}")
        counts[key_of(rec)] += 1
    return counts


def dump_snapshot(path: Path, counts: Counter) -> None:
    """Write the snapshot sorted, one JSON object per line.

    One entry per line is not cosmetic: it makes the file merge TEXTUALLY, so two
    branches that each --update produce a readable line-level conflict instead of
    a whole-file one. Do not switch this to json.dump(indent=2).
    """
    lines = []
    for key in sorted(counts):
        # strict=True: FIELDS and the key tuple must stay the same length, or a
        # key-projection change would silently drop a field from the snapshot.
        rec = dict(zip(FIELDS, key, strict=True))
        for _ in range(counts[key]):
            lines.append(json.dumps(rec, sort_keys=True, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n")


def source_line(tree: Path, rel_path: str, line: int, cache: dict) -> str:
    """Stripped text of `rel_path`:`line`, read from the materialised tree."""
    if rel_path not in cache:
        try:
            cache[rel_path] = (tree / rel_path).read_text(errors="replace").splitlines()
        except OSError:
            cache[rel_path] = []
    lines = cache[rel_path]
    return lines[line - 1].strip() if 0 < line <= len(lines) else ""


def collect(project: Path, tmp: Path, interp: Path) -> Counter:
    """Run pyrefly over the materialised tree and return the key multiset."""
    proc = subprocess.run(
        [
            str(project / ".venv" / "bin" / "pyrefly"),
            "check",
            "-c",
            str(tmp / CONFIG_NAME),
            "--python-interpreter-path",
            str(interp),
            "--relative-to",
            str(tmp),
            "--output-format",
            "json",
        ],
        capture_output=True,
        text=True,
                                                                       # NEVER check=True. pyrefly exits 1 whenever errors exist, which is the
                                                                       # normal case while the snapshot is non-empty.
    )

    # Surface WARN lines regardless of exit code. "On a non-clean exit" would be
    # meaningless in a tool whose normal exit IS 1, and the single most important
    # thing pyrefly ever puts on stderr -- "Failed to query interpreter ...
    # falling back" -- comes with exit 1 and a full, plausible-looking result set
    # (826 errors when the gate was added).
    if proc.stderr and "WARN" in proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)

    if not proc.stdout.strip():
        # Measured: 0 bytes on stdout, exit 1, message on stderr only. This is
        # what "no Python files matched" looks like, e.g. after project-includes
        # is narrowed to a directory that does not exist. A gate that catches
        # JSONDecodeError and carries on turns this into a green.
        die("pyrefly produced no output at all (exit "
            f"{proc.returncode}) -- it matched no files. stderr was:\n"
            f"{proc.stderr.rstrip()}")

    try:
        entries = json.loads(proc.stdout)["errors"]
    except (json.JSONDecodeError, KeyError) as exc:
        die(f"could not parse pyrefly output: {exc}\nstderr:\n{proc.stderr.rstrip()}")

    broken = sorted({e["path"] for e in entries if e["name"] in ("parse-error", "invalid-syntax")})
    if broken:
        # A file that does not parse produces a disproportionate wall -- one
        # 6-line file with conflict markers measured 18 entries. Print the cause,
        # not the wall.
        die("a staged file does not parse -- check for unresolved conflict "
            "markers:\n  " + "\n  ".join(broken))

    cache: dict = {}
    counts: Counter = Counter()
    for e in entries:
        # Build a record and project it with key_of(), the SAME function
        # load_snapshot() uses. Hardcoding the tuple here instead would let the
        # two sides drift the moment FIELDS changes -- and the drift would not
        # look like a key change, it would look like every error in the repo
        # being simultaneously added and removed.
        counts[key_of({
            "path": e["path"],
            "name": e["name"],
            "concise_description": e["concise_description"],
            "source_line": source_line(tmp, e["path"], e["line"], cache),
        })] += 1
    return counts


def report(added: Counter, removed: Counter) -> None:
    """Print the differences grouped by path."""
    for label, counts in (("ADDED", added), ("REMOVED", removed)):
        if not counts:
            continue
        print(f"\n{label} {sum(counts.values())}:")
        for key in sorted(counts):
            # Unpack through FIELDS rather than by position: this is the third
            # place that has to agree with the key's shape, and the other two
            # (collect, load_snapshot) already project through key_of.
            rec = dict(zip(FIELDS, key, strict=True))
            for _ in range(counts[key]):
                print(f"  {rec['path']}: {rec['name']}: {rec['concise_description']}")
                if rec.get("source_line"):
                    print(f"      {rec['source_line']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project",
                    type=Path,
                    default=None,
                    help="repo root to check (default: the enclosing repo)")
    ap.add_argument("--snapshot",
                    type=Path,
                    default=None,
                    help=f"snapshot to write on --update (default: <project>/{SNAPSHOT_NAME})")
    ap.add_argument("--update",
                    action="store_true",
                    help="rewrite the snapshot from the current index and print the diff")
    ap.add_argument("--list", action="store_true", help="print the frozen errors and exit")
    args = ap.parse_args()

    # --- step 1: resolve the root, and bind every git call to it -------------
    if args.project is not None:
        project = args.project.resolve()
    else:
        project = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").strip()).resolve()
        if Path.cwd().resolve() != project:
            die(f"refusing to run from {Path.cwd()}; it is not the repo root "
                f"({project}). Pass --project {project} if you meant to.")

    out_snapshot = args.snapshot.resolve() if args.snapshot else project / SNAPSHOT_NAME

    if args.list:
        if not out_snapshot.exists():
            die(f"{out_snapshot} does not exist")
        for key in sorted(load_snapshot(out_snapshot)):
            print(f"{key[0]}: {key[1]}: {key[2]}")
        return 0

    # --- step 2: probe the interpreter, do not merely test that it exists ----
    interp = project / ".venv" / "bin" / "python"
    try:
        probe = subprocess.run([str(interp), "-c", "import sys; print(sys.prefix)"],
                               capture_output=True,
                               text=True)
        rc, err = probe.returncode, probe.stderr.strip()
    except OSError as exc:
        # Wrapping is not optional: a missing file raises FileNotFoundError and a
        # non-executable one raises PermissionError, so an unwrapped probe
        # crashes with a traceback on exactly the two inputs this guard exists to
        # reject -- and a traceback is not an abort message anyone can act on.
        rc, err = 1, str(exc)
    if rc != 0:
        # exists() is not enough. A file that exists but is not executable, and
        # one that is executable with a dead shebang, both return True from
        # exists() and both make pyrefly fall back to the default environment --
        # producing, when the gate was added, 826 errors (ADDED 328 / REMOVED 254,
        # 270 of them bare missing-import) with the only explanation on stderr.
        die(f"{interp} is not a working interpreter: {err}")

    # --- step 3: refuse an unmerged index -----------------------------------
    # MUST precede the parity check: on an unmerged index `git ls-files` prints
    # one line per stage, so parity would compare inflated counts.
    unmerged = sorted(
        {ln.split("\t")[-1]
         for ln in git(project, "ls-files", "-u").splitlines() if ln})
    if unmerged:
        die("the index has unmerged paths; resolve them first:\n  " + "\n  ".join(unmerged))

    tmp = Path(tempfile.mkdtemp(prefix="pyrefly-gate-"))
    # pyrefly silently skips any project-includes pattern whose absolute path has
    # a HIDDEN ANCESTOR. Measured on this repo: the same materialised tree yields
    # 136 covered files under /tmp/x and 26 under /tmp/.x, with only a WARN on
    # stderr. mkdtemp honours TMPDIR, so this is reachable by configuration.
    # Exact-equality polarity would red it as ~700 removals, but blaming the
    # user's code for a tempdir setting is a bad hour; say so instead.
    hidden = [part for part in tmp.parts if part.startswith(".") and part != "."]
    if hidden:
        shutil.rmtree(tmp, ignore_errors=True)
        die(f"refusing to work under a hidden directory ({'/'.join(hidden)} in {tmp}): "
            "pyrefly skips include patterns there and would check only part of the "
            "tree. Set TMPDIR to a path with no dot-component.")
    try:
        # --- step 4: materialise the index ----------------------------------
        git(project, "checkout-index", "-a", f"--prefix={tmp}/")

        # --- step 5: parity, reported on EVERY run --------------------------
        # The counts go to stderr unconditionally because <tmp> is removed in the
        # finally below, so this is the only channel through which a test can
        # observe that the materialised tree was complete.
        n_tracked = len([p for p in git(project, "ls-files", "*.py").splitlines() if p])
        n_materialised = len(list(tmp.rglob("*.py")))
        print(f"pyrefly-gate: materialised {n_materialised} .py, index has {n_tracked}",
              file=sys.stderr)
        if n_materialised != n_tracked:
            # Counts only -- deliberately NOT the paths. Naming them here would
            # make this guard indistinguishable from the unmerged-index guard
            # above and silently disarm that knock-out.
            die(f"materialised tree is incomplete: {n_materialised} != {n_tracked}")

        # --- step 6: the STAGED config ---------------------------------------
        if not (tmp / CONFIG_NAME).is_file():
            die(f"{CONFIG_NAME} is not tracked; the gate cannot check a config it cannot see")

        # --- step 7: the STAGED snapshot -------------------------------------
        # Read the comparison snapshot from the materialised tree, but WRITE to
        # the working tree. That asymmetry is deliberate: it is what makes the
        # `git add` half of the remediation message load-bearing, and it is what
        # stops a commit that stages a gutted snapshot from being compared
        # against the intact working-tree copy and passing green.
        staged_snapshot = tmp / SNAPSHOT_NAME
        if args.update:
            frozen = load_snapshot(staged_snapshot) if staged_snapshot.is_file() else Counter()
        else:
            if not staged_snapshot.is_file():
                die(f"{SNAPSHOT_NAME} is not in the index; run --update && git add it")
            frozen = load_snapshot(staged_snapshot)

        # --- steps 8-10: run, parse, compare ---------------------------------
        current = collect(project, tmp, interp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    added = current - frozen
    removed = frozen - current

    if args.update:
        dump_snapshot(out_snapshot, current)
        report(added, removed)
        print(f"\nwrote {sum(current.values())} entries to {out_snapshot}")
        return 0

    if not added and not removed:
        return 0

    report(added, removed)
    print("\npyrefly-gate: the frozen error set changed.\n"
          "If these changes are intentional, refreeze and stage the result:\n"
          "  uv run python scripts/pyrefly_gate.py --update && git add pyrefly-snapshot.json\n"
          "Both commands are required: --update writes the working tree, and the\n"
          "gate reads the index, so skipping the `git add` reds again identically.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
