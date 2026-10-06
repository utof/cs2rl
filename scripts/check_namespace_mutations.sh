#!/usr/bin/env bash
# Manual, bounded native mutmut sample for the namespace-entry guard.
set -euo pipefail

if [[ ${1:-run} == help || ${1:-run} == --help ]]; then
    cat <<'HELP'
From the checkout root, after installing the normal project/dev dependencies:
  uv venv --python .venv/bin/python .venv-mutmut
  uv pip install --python .venv-mutmut/bin/python -r requirements-mutation.txt
  bash scripts/check_namespace_mutations.sh
  bash scripts/check_namespace_mutations.sh results
  bash scripts/check_namespace_mutations.sh show tests.conftest.x_namespace_entry_problems__mutmut_31

The separate environment runs these stdlib/pytest-only guard consumers against
tracked src/scripts working files, without ignored build outputs or untracked
files and without syncing/rebuilding the editable project or its runtime.
Run refreshes those generated trees, so deleted files cannot remain stale.
Linux/GNU timeout is required. Each command has a 600s deadline and 5s kill grace.
Run selects 14 native mutations of entry naming, directory detection and help
text, with one child. This is a fixed sample for mutmut 3.8.0/current guard;
after changing the guard, inspect native show diffs and update the sample.
Use this entrypoint to keep execution bounded; bare mutmut run selects the file.
Mutmut also generates unselected candidates: 'not checked' is expected for them.
Native results and diffs stay under ignored mutants/; preserve or remove that
cache before a fresh campaign. Original source is never changed by this script.

Review actual diffs and failure output. A native 'killed' label can mean startup
failed before collection; count direct assertion failures separately. Cosmetic
help-text survivors are acceptable. PYTHONPATH and python -m pytest must retain
their case-sensitive spelling. A zero run exit code means the tool completed,
not that every mutant was killed. There is no automatic CI or score threshold.
HELP
    exit 0
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo_root"
tool="$repo_root/.venv-mutmut/bin/mutmut"
interpreter="$repo_root/.venv-mutmut/bin/python"
if [[ ! -x $tool || ! -x $interpreter ]]; then
    echo "Set up the optional environment first: bash scripts/check_namespace_mutations.sh help" >&2
    exit 2
fi
"$interpreter" -c 'from importlib.metadata import version; assert version("mutmut") == "3.8.0" and version("pytest") == "9.0.2", "Install requirements-mutation.txt for the demonstrated tool contract"'
export UV_NO_SYNC=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

case ${1:-run} in
    run)
        [[ $# -le 1 ]] || { echo "run takes no extra arguments" >&2; exit 2; }
        run_started=$SECONDS
        # Copying a clone's .git directory can copy private/unbounded history.
        # A pointer supplies the same index for reads by the unchanged tracked-
        # package safety hook, in ordinary clones and linked worktrees alike.
        [[ ! -L mutants && ! -L mutants/.git ]] || { echo "Refusing a symlinked mutation cache" >&2; exit 2; }
        mkdir -p mutants
        git_dir=$(git rev-parse --absolute-git-dir)
        git_pointer="gitdir: $git_dir"
        if [[ -e mutants/.git ]]; then
            [[ -f mutants/.git && $(cat mutants/.git) == "$git_pointer" ]] || {
                echo "Preserve/remove the existing mutants/ cache: its Git metadata belongs elsewhere" >&2
                exit 2
            }
        else
            printf '%s\n' "$git_pointer" > mutants/.git
        fi
        # Native also_copy recursively includes ignored multi-GB build outputs.
        # Git supplies membership; copy2 preserves current working bytes rather
        # than hiding edits behind git archive's committed/indexed contents.
        # Stage completely before replacing only the generated source trees.
        timeout --kill-after=5s 600s "$interpreter" - <<'PY'
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

cache = Path("mutants")
for name in ("src", "scripts"):
    destination = cache / name
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise SystemExit(f"Refusing a non-directory or symlinked source cache: {destination}")
tracked = subprocess.check_output(["git", "ls-files", "--cached", "--deduplicate", "-z", "--", "src", "scripts"])
with tempfile.TemporaryDirectory(prefix=".source-", dir=cache) as temporary:
    staging = Path(temporary)
    count = size = 0
    for raw in tracked.split(b"\0"):
        if not raw:
            continue
        source = Path(os.fsdecode(raw))
        if any(path.is_symlink() for path in (source, *source.parents)):
            raise SystemExit(f"Refusing a symlinked tracked source path: {source}")
        if not source.exists():
            continue  # An unstaged deletion must not survive in the generated copy.
        if not source.is_file():
            raise SystemExit(f"Tracked source is not a regular file: {source}")
        copied = staging / source
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copied)
        count += 1
        size += copied.stat().st_size
    for name in ("src", "scripts"):
        destination = cache / name
        if destination.exists():
            shutil.rmtree(destination)
        (staging / name).mkdir(exist_ok=True)
        (staging / name).replace(destination)
print(f"Tracked source snapshot: {count} files, {size} bytes")
PY
        # Use native CLI selection, not a custom generator/runner or no-mutate
        # annotations in safety hooks. Explicit names bound execution even if
        # later guard edits increase the generated population.
        selected=()
        for suffix in 11 12 13 14 15 16 19 28 29 30 31 32 33 34; do
            selected+=("tests.conftest.x_namespace_entry_problems__mutmut_$suffix")
        done
        echo "Running ${#selected[@]} selected guard mutations, one child, 600s deadline"
        remaining=$((600 - SECONDS + run_started))
        (( remaining > 0 )) || { echo "Source preparation exhausted the 600s deadline" >&2; exit 124; }
        timeout --kill-after=5s "${remaining}s" "$tool" run --max-children 1 "${selected[@]}"
        ;;
    results)
        [[ $# -eq 1 ]] || { echo "results takes no extra arguments" >&2; exit 2; }
        timeout --kill-after=5s 600s "$tool" results --all true |
            sed -n '/tests\.conftest\.x_namespace_entry_problems__mutmut_/p'
        ;;
    show)
        [[ $# -eq 2 && $2 == tests.conftest.x_namespace_entry_problems__mutmut_* ]] || {
            echo "show requires a namespace-entry mutant name from results" >&2
            exit 2
        }
        timeout --kill-after=5s 600s "$tool" show "$2"
        ;;
    *) echo "Usage: bash scripts/check_namespace_mutations.sh [run|results|show NAME|help]" >&2; exit 2 ;;
esac
