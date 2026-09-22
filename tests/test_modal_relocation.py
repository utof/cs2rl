"""AST gates for the split of scripts/modal_runner_lib.py into scripts/modal_runner/.

Two kinds of clause live here, and a red means something different for each.

RELOCATION-ORACLE clauses prove the W3b split was a pure move. They compare
every top-level segment of the package with its bytes in the frozen monolith
(`scripts/modal_runner_lib.py` at 2bb32ac, loaded and digest-checked by
`_frozen_runner_source`) and pin the per-module segment line sums:
`segment-bytes`, `qualified-count`, `no-oracle-segment`, `manifest-union` and
the recorded sums in `test_relocation_size_contract`. Any later edit to a
relocated symbol turns them red BY DESIGN. They are scheduled for retirement
from the permanent suite in a follow-up commit once the split has landed. Until
then, change relocated code only after that retirement, and never by editing
MANIFEST, REWRITES or the recorded sums to match an edit.

PACKAGE-STRUCTURE clauses stay true after that retirement: the on-disk module
population (`files`, `undeclared-module`), one owner per symbol
(`manifest-duplicates`, `membership`, `duplicates`), module-scope shape
(`header`), the import graph (`runtime-edges`, `annotation-edges`), module names
(`forbidden-name`), the 35% segment budget and import purity. When one of these
reddens on an intended change, update the table its failure message names, in
the same commit as the change.

Every failure message names the clause, the file or symbol, and the edit that
resolves it (`_explain`). Controls plant defects into in-memory copies of the
sources or into a copy of the package under `tmp_path`, never into the live
package.
"""
import ast
from pathlib import Path

import pytest

# The packaging gates' own helpers and constants, imported rather than copied so
# that both files recognise module-level names, module-scope shape, non-stdlib
# imports and the package population by ONE definition; two copies could let the
# gates disagree about the same package. Importing a collected test module does
# not collect its tests twice: none of these names is a test. They are not
# moved to tests/modal_test_helpers.py because that is not a pure move: that
# file admits only names both halves of the runner/client test seam reach, and
# `classify_seam` enforces it. PITFALL: `_module_level_names` also drives
# `classify_seam`, and `_module_level_binding_counts` the seam's placement gate,
# so a change made to either for the seam's sake changes the segment and
# membership gates here as well.
from tests.test_modal_packaging import (
    PACKAGED,
    RUNNER_MODULES,
    _frozen_runner_source,
    _module_level_binding_counts,
    _module_level_names,
    _module_scope_shape_violations,
    _nonstdlib_module_scope_imports,
    _runner_module_population,
    bare_spelling_imports,
)

ROOT = Path(__file__).resolve().parents[1]
# Owner of every top-level symbol, per module (spec section 5.1). Adding,
# removing or moving a symbol means editing its entry here in the same commit.
MANIFEST = {
    'core.py': [
        'ValidationError', 'Action', 'Status', 'mounted_path', 'SourceProvenance', 'sha256_file',
        'FileProvenance', 'RunStatus', 'Manifest', 'RunResult', 'Registry', 'ArtifactIndex',
        'LockLike', 'DerivedStatus', 'CheckpointVerdict', 'sha256_bytes', 'CompletionEvidence',
        'HeartbeatWorker', 'ReloadingVolume', 'PreparedSource', 'TrainingAttemptResult',
        '_UnusedArtifacts', 'PublishOutcome', 'VOLUME_NAME', 'REGISTRY_NAME', 'VOLUME_MOUNT',
        'SOURCES_ROOT', 'INPUTS_ROOT', 'RUNS_ROOT', 'PREBUILT_PYTHON',
        'PREBUILT_LOAD_TIMEOUT_SECONDS', 'ALLOWED_MAPS', 'ALLOWED_GPUS', 'ALLOWED_NUM_ENVS',
        'ALLOWED_CPU_CORES', 'DEFAULT_GPU', 'DEFAULT_NUM_ENVS', 'DEFAULT_CPU_CORES',
        'DEFAULT_MEMORY_MIB', 'DEFAULT_VEC_WORKERS', 'DEFAULT_TIMEOUT_MINUTES',
        'DEFAULT_SAVE_EVERY_SECONDS', 'MIN_MEMORY_MIB', 'MAX_MEMORY_MIB', 'MIN_TIMEOUT_MINUTES',
        'MAX_TIMEOUT_MINUTES', 'MIN_SAVE_EVERY_SECONDS', 'MAX_SAVE_EVERY_SECONDS', 'AGENTS_PER_ENV',
        'BPTT_HORIZON', 'MIN_BATCH_SIZE', '_RUN_ID_RE', '_SECRET_NAME_RE',
        'LIVE_TRAIN_OPTION_ARITY', 'LIVE_TRAIN_OPTIONS', 'RUNNER_OWNED_TRAIN_FLAGS',
        'RUN_ONLY_OPTIONS', 'TERMINAL_STATUSES', 'NONTERMINAL_STATUSES', '_COMMIT_SHA_RE',
        '_SAFE_TAR_TYPES', 'PROVENANCE_NAME', '_PREBUILT_LOAD_SOURCE', 'STATUS_FILENAME',
        'SCHEMA_VERSION', '_ALLOWED_TRANSITIONS', 'HEARTBEAT_INTERVAL', 'STALE_AFTER',
        'RESERVATION_FILENAME', 'MANIFEST_FILENAME', 'FAILURE_UPLOAD', 'ALLOWED_FAILURE_CODES',
        'REDELIVERED', 'UV_BIN', 'TRAIN_SCRIPT', '_PRESERVED_CHILD_ENV_KEYS', '_THREAD_CAP_ENV',
        'CUDA_PROBE_SOURCE', 'TRAIN_LOG_NAME', 'RESULT_FILENAME', 'CHECKPOINT_NAME',
        'CHECKPOINT_SIDECAR_NAME', 'CHECKPOINT_PUBLISH_REASON_NAME', 'DEAD_CHECKPOINT_NAME',
        'CHECKPOINT_SETTLE_SECONDS', 'TERM_GRACE_SECONDS', 'DEAD_RUN_EXIT_CODE',
        'POLL_INTERVAL_SECONDS', 'REASON_SIGNAL', 'REASON_TIMEOUT', 'REASON_DEAD_RUN',
        'REASON_INVALID_EVIDENCE', 'REASON_NONZERO_EXIT', 'REASON_ERROR'
    ],
    'request.py': [
        'ResumeRequest', 'ArtifactClientRequest', 'RunRequest', 'validate_run_id',
        'validate_secret_name', 'parse_train_args', '_split_long_option', '_option_value',
        'validate_train_args', 'build_run_request', 'parse_artifact_client_request'
    ],
    'source.py': [
        '_run_git', 'validate_clean_head', '_reject_unsafe_tar_member', 'safe_extract_git_archive',
        '_staging_members', '_repack_deterministic', 'create_source_bundle'
    ],
    'checkpoint.py': [
        '_import_torch', '_assert_weights_only_loadable', 'validate_local_checkpoint',
        '_load_checkpoint_weights', 'verify_checkpoint', 'normalize_config_for_transport',
        '_iter_metrics_steps', 'validate_completed_run'
    ],
    'state.py': [
        'atomic_write_json', '_read_status', 'transition_status', '_transition_status_unlocked',
        'run_registry_key', 'attempt_registry_key', '_reservation_path', '_manifest_path',
        '_volume_has_run', 'record_run_failure', 'finish_reservation', 'reserve_run',
        'claim_attempt', 'deliver_attempt', 'write_heartbeat', '_load_volume_json',
        '_parse_iso8601', 'derive_status', 'derive_run_view_from_bytes', 'derive_run_view',
        'list_run_artifacts', 'start_heartbeat_worker'
    ],
    'commands.py': [
        '_assemble_train_argv', 'build_train_argv', 'build_dump_config_argv',
        '_is_preserved_child_env_key', 'build_child_env', 'build_install_command',
        'build_train_command', 'build_dump_config_command', 'build_cuda_probe_command'
    ],
    'preflight.py': [
        '_verify_extracted_provenance', '_stop_heartbeat', '_validate_remote_resume',
        '_hash_dumped_config', 'prepare_remote_source'
    ],
    'training.py': [
        '_tee_stream', 'execute_training_attempt', '_checkpoint_generation',
        'publish_stable_checkpoint', '_start_checkpoint_watcher', '_record_publish_reason',
        '_publish_and_note', '_close_log_sink', '_is_dead_run', '_map_child_exit',
        '_metrics_summary', '_optional_checkpoint_sha256', '_write_run_result',
        '_signal_process_group', '_run_training_attempt'
    ]
}
# The only byte differences the relocation allows: {segment: {name: (module,
# exact occurrence count)}}, each a cross-module read rewritten as `module.name`
# so that tests patching the owning module reach it. A relocation-oracle table.
REWRITES = {
    '_assert_weights_only_loadable': {
        'PREBUILT_PYTHON': ('core', 5)
    },
    'validate_local_checkpoint': {
        'sha256_file': ('core', 1)
    },
    'validate_completed_run': {
        'sha256_file': ('core', 1)
    },
    'build_install_command': {
        'PREBUILT_PYTHON': ('core', 1)
    },
    'build_train_command': {
        'PREBUILT_PYTHON': ('core', 1)
    },
    'build_cuda_probe_command': {
        'PREBUILT_PYTHON': ('core', 1)
    },
    '_validate_remote_resume': {
        'validate_local_checkpoint': ('checkpoint', 1)
    },
    'prepare_remote_source': {
        'sha256_file': ('core', 1),
        'transition_status': ('state', 3)
    },
    'create_source_bundle': {
        'sha256_file': ('core', 1)
    },
    'publish_stable_checkpoint': {
        'sha256_file': ('core', 1),
        'validate_local_checkpoint': ('checkpoint', 1)
    },
    '_run_training_attempt': {
        'transition_status': ('state', 2)
    }
}
# Runtime import edges between package modules (spec section 5.1). core stays a
# leaf. A new edge is legal only if the graph stays acyclic; record it here.
DEPENDENCIES = {
    'core': [],
    'request': ['commands', 'core'],
    'source': ['core'],
    'checkpoint': ['core', 'state'],
    'state': ['core', 'request'],
    'commands': ['core'],
    'preflight': ['checkpoint', 'commands', 'core', 'source', 'state'],
    'training': ['checkpoint', 'core', 'preflight', 'state']
}
# Edges that exist only under `if TYPE_CHECKING:` and never execute. commands ->
# request must stay annotation-only: request imports commands at run time, so a
# runtime edge back would be an import cycle.
ANNOTATION_DEPENDENCIES = {'commands': ['request'], 'preflight': ['request']}


def _segments(source):
    """Exact declaration bytes, including decorators and complete class bodies."""
    lines = source.encode().splitlines(keepends=True)
    found = {}
    for name, node in _module_level_names(ast.parse(source)).items():
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        while first > 1 and lines[first - 2].lstrip().startswith(b"#"):
            first -= 1
        found[name] = b"".join(lines[first - 1:node.end_lineno])
    return found


def _package_sources(root):
    """{filename: text} for every package submodule on disk under `root`.

    The population is `_runner_module_population`'s: every declared module,
    read even when absent so that a missing or unreadable member raises an
    OSError naming it, then every undeclared `*.py` in the directory, which the
    gates below report rather than skip. Read as UTF-8 bytes because the sources
    hold non-ASCII prose, and the locale must not decide whether they parse.
    """
    candidates, _ = _runner_module_population(root)
    return {Path(rel).name: (Path(root) / rel).read_bytes().decode("utf-8") for rel in candidates}


def _copy_live_package(tmp_path):
    """Copy the live `scripts/modal_runner/*.py`, facade included, under `tmp_path`.

    Returns the copy's package directory. Controls that need a REAL file on
    disk plant into this copy and read it back through `_package_sources` or
    `_nonstdlib_module_scope_imports(tmp_path)`; nothing ever writes into the
    live package. Only `*.py` is copied, so no `__pycache__` rides along, and
    bytes are copied as-is so no encoding or newline translation happens.
    """
    package = tmp_path / "scripts" / "modal_runner"
    package.mkdir(parents=True)
    for path in (ROOT / "scripts" / "modal_runner").glob("*.py"):
        (package / path.name).write_bytes(path.read_bytes())
    return package


def _manifest_violations(manifest, oracle):
    """Criterion 1's manifest-level clauses: one owner per symbol, and exactly the oracle's names.

    Per-module membership compares each file with its OWN manifest entry, so it
    cannot see a symbol dropped from both the manifest and its module, or one
    declared in, and defined in, two modules. Measured by the W3b structural
    review before these clauses existed: both plants left every gate green.
    `manifest-duplicates` is a package-structure clause; `manifest-union` ties
    the manifest to the relocation oracle's 171 names.
    """
    owners = {}
    for filename, names in manifest.items():
        for name in names:
            owners.setdefault(name, []).append(filename)
    # Annotated because the tuples differ in shape per clause; pyrefly would
    # otherwise infer the first append's shape and reject every other clause.
    violations: list[tuple] = []
    shared = {name: files for name, files in owners.items() if len(files) > 1}
    if shared:
        violations.append(("manifest-duplicates", shared))
    if set(owners) != set(oracle):
        violations.append(("manifest-union", sorted(set(oracle) - set(owners)),
                           sorted(set(owners) - set(oracle))))
    return violations


def _relocation_violations(sources, oracle, manifest=MANIFEST):
    """Combine the manifest, file population, header shape, membership and segment bytes.

    `manifest` defaults to MANIFEST; controls pass an edited copy. Every file in
    `sources` gets the header check, including one no manifest entry declares,
    and every top-level segment is compared with the oracle: a name the oracle
    lacks is a `no-oracle-segment` violation, never a skip.
    """
    violations = _manifest_violations(manifest, oracle)
    if set(sources) != set(manifest):
        violations.append(
            ("files", sorted(set(sources) - set(manifest)), sorted(set(manifest) - set(sources))))
    for filename, source in sources.items():
        declared = manifest.get(filename, [])
        tree = ast.parse(source)
        examined, shape = _module_scope_shape_violations(tree)
        assert examined == len(tree.body), (
            f"the header instrument examined {examined} of {len(tree.body)} module-scope "
            f"statements in {filename}, so its verdict does not cover the file")
        violations.extend(("header", filename, item) for item in shape)
        names = _module_level_names(tree)
        if set(names) != set(declared):
            violations.append(("membership", filename, sorted(set(names) ^ set(declared))))
        duplicates = {
            n: count
            for n, count in _module_level_binding_counts(tree).items() if count != 1
        }
        if duplicates:
            violations.append(("duplicates", filename, duplicates))
        for node in tree.body:
            if isinstance(
                    node,
                (ast.Assign, ast.AnnAssign)) and id(node) not in {id(n)
                                                                  for n in names.values()}:
                violations.append(("membership", filename, "undeclared assignment"))
        for name, segment in _segments(source).items():
            if name not in oracle:
                violations.append(("no-oracle-segment", filename, name))
                continue
            edits = []
            rules = REWRITES.get(name, {})
            counts = dict.fromkeys(rules, 0)
            for node in ast.walk(ast.parse(segment)):
                if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                        and node.attr in rules and node.value.id == rules[node.attr][0]
                        and isinstance(node.ctx, ast.Load)):
                    counts[node.attr] += 1
                    edits.append(
                        (node.lineno - 1, node.col_offset, node.end_col_offset, node.attr.encode()))
            expected = {key: rule[1] for key, rule in rules.items()}
            if counts != expected:
                violations.append(("qualified-count", filename, name, counts, expected))
            lines = segment.splitlines(keepends=True)
            for line, start, end, replacement in sorted(edits, reverse=True):
                lines[line] = lines[line][:start] + replacement + lines[line][end:]
            if b"".join(lines) != oracle[name]:
                violations.append(("segment-bytes", filename, name))
    return violations


def _dependency_edges(sources):
    """({module: runtime edges}, {module: annotation-only edges}) between package modules.

    Only the spelling the package uses resolves to a member: a ONE-dot relative
    import (`from . import core`, `from .core import X`). Every other way to
    reach the package from inside it is recorded as an edge no table allows:
    a `..`-relative import as its dotted spelling (`..`, `..modal_runner`), and
    any absolute spelling as `PACKAGED`. The absolute shapes come from
    `bare_spelling_imports(source, PACKAGED)`, the static guard's walker, so
    this gate and that guard share one definition of an import of the package
    (`import`, from-import, `from scripts import modal_runner`, and literal
    `importlib.import_module` / `__import__`) instead of this file keeping a
    second, weaker copy that missed the package-level and parent spellings.

    PITFALL: that walker is an `ast.walk`, so an absolute spelling is recorded
    as a runtime edge even under `if TYPE_CHECKING:`. That fails closed: no
    table allows `PACKAGED` in either column.
    """
    runtime, annotations = {}, {}
    for filename, source in sources.items():
        module = filename.removesuffix(".py")
        runtime[module], annotations[module] = set(), set()
        if bare_spelling_imports(source, PACKAGED):
            runtime[module].add(PACKAGED)
        stack = [(ast.parse(source), False)]
        while stack:
            node, checking = stack.pop()
            if isinstance(node, ast.If):
                test = node.test
                if ((isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
                        or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")):
                    stack.extend((child, True) for child in node.body)
                    stack.extend((child, checking) for child in node.orelse)
                    continue
            edges = annotations[module] if checking else runtime[module]
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                if node.module:
                    edges.add(node.module.split(".")[0])
                else:
                    edges.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level > 1:
                edges.add("." * node.level + (node.module or ""))
            stack.extend((child, checking) for child in ast.iter_child_nodes(node))
    return runtime, annotations


def _dependency_violations(sources):
    """Criterion 6: resolved edges must equal DEPENDENCIES / ANNOTATION_DEPENDENCIES per module.

    A module in `sources` that DEPENDENCIES does not declare is reported as
    `undeclared-module` rather than skipped.
    """
    runtime, annotations = _dependency_edges(sources)
    failures: list[tuple] = [("undeclared-module", module)
                             for module in sorted(set(runtime) - set(DEPENDENCIES))]
    for module in RUNNER_MODULES:
        expected = set(DEPENDENCIES[module])
        if runtime.get(module) != expected:
            failures.append(("runtime-edges", module, runtime.get(module), expected))
        expected = set(ANNOTATION_DEPENDENCIES.get(module, []))
        if annotations.get(module) != expected:
            failures.append(("annotation-edges", module, annotations.get(module), expected))
    return failures


def _size_violations(sources):
    """Criterion 12: no grab-bag module names, and no module over 35% of the oracle's segment lines."""
    failures: list[tuple] = []
    for filename, source in sources.items():
        if filename.removesuffix(".py") in {"utils", "helpers", "common", "misc"}:
            failures.append(("forbidden-name", filename))
        lines = sum(len(segment.splitlines()) for segment in _segments(source).values())
        if lines > 2483 * .35:
            failures.append(("segment-budget", filename, lines))
    return failures


def _explain(violations):
    """One line per violation: the clause, where it fired, and the edit that resolves it.

    WHY: the gates return tuples, and a red that prints only a tuple list tells
    whoever meets it neither which clause objected nor whether the fix is to
    revert their edit or to update a table in this file. See the module
    docstring for which clauses are relocation-oracle clauses.
    """
    retire = ("This is a relocation-oracle clause: the split must move code without editing it "
              "(no reformatting, re-wrapping or comment edits). Revert the edit, or make it after "
              "the oracle clauses are retired (module docstring).")
    tables = ("RUNNER_MODULES (tests/test_modal_packaging.py) and MANIFEST, DEPENDENCIES and "
              "ANNOTATION_DEPENDENCIES (tests/test_modal_relocation.py)")
    lines = []
    for v in violations:
        kind = v[0]
        if kind == "files":
            text = (f"scripts/modal_runner/ holds undeclared modules {v[1]} and lacks declared "
                    f"modules {v[2]}. A new module needs entries in {tables} in the same commit; "
                    "otherwise delete the stray file.")
        elif kind == "undeclared-module":
            text = (f"scripts/modal_runner/{v[1]}.py is not in DEPENDENCIES. Declare the module in "
                    f"{tables}, or delete it.")
        elif kind == "manifest-duplicates":
            text = (
                f"MANIFEST gives these symbols more than one owner: {v[1]}. Each symbol lives in "
                "exactly one module; delete the extra MANIFEST entries and definitions.")
        elif kind == "manifest-union":
            text = (f"MANIFEST does not declare exactly the oracle's top-level names: missing "
                    f"{v[1]}, not in the oracle {v[2]}. Every relocated symbol must be declared in "
                    f"exactly one module. {retire}")
        elif kind == "header":
            text = (f"{v[1]} line {v[2][1]} ({v[2][0]}): {v[2][2]}. Module scope may hold only the "
                    "docstring, imports, one imports-only `if TYPE_CHECKING:` block without else, "
                    "and declarations; move the statement into a function.")
        elif kind == "membership":
            text = (f"{v[1]}'s top-level names differ from MANIFEST['{v[1]}'] by {v[2]}. If you "
                    f"meant to add, remove or move a symbol, update MANIFEST['{v[1]}'] in "
                    "tests/test_modal_relocation.py in the same commit.")
        elif kind == "duplicates":
            text = (f"{v[1]} binds {v[2]} more than once at module scope; the later binding "
                    "silently shadows the earlier one. Keep one.")
        elif kind == "no-oracle-segment":
            text = (f"{v[1]} defines {v[2]}, which is not one of the symbols moved from "
                    f"scripts/modal_runner_lib.py at 2bb32ac, so its bytes cannot be checked. "
                    f"{retire}")
        elif kind == "qualified-count":
            text = (f"{v[1]}: {v[2]} has qualified references {v[3]} where REWRITES requires "
                    f"{v[4]}. Keep every declared `module.name` read: tests patch these names "
                    f"on the owning module, and an unqualified read never sees the patch. {retire}")
        elif kind == "segment-bytes":
            text = (f"{v[1]}: {v[2]} differs from its bytes in the relocation oracle beyond the "
                    f"qualifiers REWRITES declares. {retire}")
        elif kind == "runtime-edges":
            text = (f"{v[1]}.py imports package modules {v[2]} at run time; DEPENDENCIES['{v[1]}'] "
                    f"allows {v[3]}. Write intra-package imports as one-dot relative imports; "
                    f"absolute (`{PACKAGED}`) and `..` spellings are never allowed. If a new edge "
                    "is intended and keeps the graph acyclic (core stays a leaf), update "
                    "DEPENDENCIES in the same commit.")
        elif kind == "annotation-edges":
            text = (f"{v[1]}.py imports {v[2]} under `if TYPE_CHECKING:`; "
                    f"ANNOTATION_DEPENDENCIES allows {v[3]}. Update ANNOTATION_DEPENDENCIES if the "
                    "annotation-only import is intended.")
        elif kind == "forbidden-name":
            text = (
                f"{v[1]} is a grab-bag module name (utils/helpers/common/misc). Name the module "
                "for the concern it owns.")
        elif kind == "segment-budget":
            text = (f"{v[1]} holds {v[2]} lines of top-level segments (every module-level def, "
                    "class and assignment, with its decorators and the comments directly above "
                    "it), over 35% of the oracle's 2483. Split the concern rather than growing "
                    "one module.")
        else:
            text = f"unrecognised violation {v!r}"
        lines.append(f"[{kind}] {text}")
    return "\n".join(lines) or "(no violations)"


def _assert_no_violations(violations, subject):
    """Fail with `_explain`'s sentences, prefixed by what was being checked."""
    assert violations == [], f"{subject}:\n{_explain(violations)}"


def _assert_rejected(violations, prefix, plant):
    """Fail unless some violation starts with `prefix`; name the plant and what did fire."""
    assert any(v[:len(prefix)] == prefix for v in violations), (
        f"the {plant} plant was not rejected by {prefix}; the gates reported:\n"
        f"{_explain(violations)}")


def _hoist_type_checking_imports(source):
    """Move a module's `if TYPE_CHECKING:` imports to module scope, dedented.

    The plant for "an annotation-only edge became a runtime edge". Asserts the
    block holds relative from-imports only, so the plant cannot silently move
    something else.
    """
    lines = source.splitlines(keepends=True)
    block = next(n for n in ast.parse(source).body if isinstance(n, ast.If))
    assert all(isinstance(n, ast.ImportFrom) and n.level == 1 for n in block.body), (
        f"the TYPE_CHECKING block at line {block.lineno} holds more than relative imports")
    moved = "".join("".join(lines[n.lineno - 1:n.end_lineno]).replace("    ", "", 1)
                    for n in block.body)
    lines[block.lineno - 1:block.end_lineno] = [moved]
    return "".join(lines)


@pytest.fixture
def relocation_subject():
    return _package_sources(ROOT), _segments(_frozen_runner_source())


def test_relocation_combined_contract(relocation_subject):
    """Criteria 1, 2 and 13 on the live package, plus the tables' own consistency."""
    sources, oracle = relocation_subject
    assert len(oracle) == 171, (
        f"the frozen oracle parsed to {len(oracle)} top-level segments, not the recorded 171: "
        "`_segments` or the pinned blob changed. Never edit this number to match.")
    assert sum(len(segment.splitlines()) for segment in oracle.values()) == 2483, (
        "the frozen oracle's segment line total is no longer the recorded 2483, the 35% budget's "
        "denominator; `_segments` changed.")
    assert len(REWRITES) == 11 and sum(
        count for rules in REWRITES.values() for _, count in rules.values()) == 20, (
            "REWRITES must declare the 11 segments and 20 qualified occurrences the relocation "
            "was reviewed with; it is a relocation-oracle table, not one to extend.")
    declared = set(RUNNER_MODULES)
    manifest_modules = {name.removesuffix(".py") for name in MANIFEST}
    keyed = (manifest_modules == declared and set(DEPENDENCIES) == declared
             and set(ANNOTATION_DEPENDENCIES) <= declared)
    assert keyed, (
        f"MANIFEST {sorted(MANIFEST)}, DEPENDENCIES {sorted(DEPENDENCIES)} and "
        f"ANNOTATION_DEPENDENCIES {sorted(ANNOTATION_DEPENDENCIES)} must be keyed by the modules "
        f"RUNNER_MODULES declares ({list(RUNNER_MODULES)}); update them together.")
    _assert_no_violations(_relocation_violations(sources, oracle), "the live package")


@pytest.mark.parametrize(
    "plant",
    ["rhs", "walrus", "default", "decorator", "class-body", "signature", "comment", "unqualified"])
def test_relocation_preserves_exact_segments(relocation_subject, plant):
    sources, oracle = relocation_subject
    filename, name = ("core.py", "VOLUME_NAME")
    if plant in {"default", "decorator", "signature", "comment"}:
        filename, name = "request.py", "validate_run_id"
    elif plant == "class-body":
        filename, name = "request.py", "RunRequest"
    elif plant == "unqualified":
        filename, name = "training.py", "_run_training_attempt"
    segment = _segments(sources[filename])[name].decode()
    if plant == "rhs":
        changed = "VOLUME_NAME = __import__('os').getcwd()\n"
    elif plant == "walrus":
        changed = "VOLUME_NAME = (_hidden := 'cs2rl')\n"
    elif plant == "default":
        changed = segment.replace("value: str", "value: str = __import__('os').getcwd()", 1)
    elif plant == "decorator":
        changed = "@lru_cache()\n" + segment
    elif plant == "class-body":
        lines = segment.splitlines(keepends=True)
        index = next(i for i, line in enumerate(lines) if line.startswith("class RunRequest"))
        lines.insert(index + 1, "    _work = __import__('os').getcwd()\n")
        changed = "".join(lines)
    elif plant == "signature":
        changed = segment.replace("def validate_run_id(", "def validate_run_id(\n    ", 1)
    elif plant == "comment":
        lines = segment.splitlines(keepends=True)
        lines.insert(1, "    # Changed preserved declaration text.\n")
        changed = "".join(lines)
    else:
        assert segment.count("state.transition_status") == 2, (
            "the unqualified plant expects two `state.transition_status` reads in "
            f"{name}; found {segment.count('state.transition_status')}")
        changed = segment.replace("state.transition_status", "transition_status", 1)
    assert changed != segment, f"the {plant} plant did not change {filename}: {name}"
    for _ in range(2):
        _assert_no_violations(_relocation_violations(sources, oracle), "the live package")
        mutated = {**sources, filename: sources[filename].replace(segment, changed, 1)}
        assert set(_module_level_names(ast.parse(mutated[filename]))) == set(MANIFEST[filename]), (
            f"the {plant} plant changed {filename}'s top-level names, so membership rather than "
            "the segment clause could take credit for rejecting it")
        violations = _relocation_violations(mutated, oracle)
        clause = "qualified-count" if plant == "unqualified" else "segment-bytes"
        _assert_rejected(violations, (clause, filename, name), plant)
        if plant == "unqualified":
            count = next(v for v in violations if v[0] == clause)
            reported = (count[3]["transition_status"], count[4]["transition_status"])
            assert reported == (1, 2), (
                f"qualified-count must report 1 of 2 required reads: {_explain([count])}")
            assert not any(v[0] == "segment-bytes" for v in violations), (
                "the un-qualified segment normalises to the oracle bytes, so `qualified-count` "
                f"must be the only objector, not `segment-bytes`:\n{_explain(violations)}")
        _assert_no_violations(_relocation_violations(sources, oracle), "the live package")


def test_relocation_rejects_wrong_owner_with_correct_union(relocation_subject):
    sources, oracle = relocation_subject
    for _ in range(2):
        _assert_no_violations(_relocation_violations(sources, oracle), "the live package")
        segment = _segments(sources["core.py"])["VOLUME_NAME"].decode()
        wrong = dict(sources)
        wrong["core.py"] = wrong["core.py"].replace(segment, "", 1)
        wrong["request.py"] += "\n" + segment

        def union(modules):
            """Ignore module ownership to demonstrate the weaker union-only check."""
            return set().union(*(_module_level_names(ast.parse(s)) for s in modules.values()))

        assert union(wrong) == union(sources), (
            "the wrong-owner plant changed the union of names, so it no longer shows that a "
            "union-only check would pass it")
        failures = _relocation_violations(wrong, oracle)
        objecting = sorted({v[1] for v in failures if v[0] == "membership"})
        assert objecting == [
            "core.py", "request.py"
        ], ("moving VOLUME_NAME from core.py to request.py must fail membership in BOTH files:\n"
            f"{_explain(failures)}")
        _assert_no_violations(_relocation_violations(sources, oracle), "the live package")


@pytest.mark.parametrize(
    "plant", ["unknown", "duplicate", "missing", "else", "later-expression", "later-find-spec"])
def test_relocation_rejects_membership_and_headers(relocation_subject, plant):
    sources, oracle = relocation_subject
    filename = "training.py" if plant.startswith("later-") else "core.py"
    additions = {
        "unknown":
        "\n_ON_CONTAINER = 1\n",
        "duplicate":
        "\nVOLUME_NAME = 'duplicate'\n",
        "else":
        "\nif TYPE_CHECKING:\n    import decimal\nelse:\n    print('work')\n",
        "later-expression":
        "\nprint('work')\n",
        "later-find-spec":
        "\nif importlib.util.find_spec('modal') is not None:\n    _ON_CONTAINER = True\n",
    }
    for _ in range(2):
        _assert_no_violations(_relocation_violations(sources, oracle), "the live package")
        changed = sources[filename] + additions.get(plant, "")
        if plant == "missing":
            changed = changed.replace(_segments(changed)["VOLUME_NAME"].decode(), "", 1)
        failures = _relocation_violations({**sources, filename: changed}, oracle)
        clause = "duplicates" if plant == "duplicate" else "membership" if plant in {
            "missing", "unknown"
        } else "header"
        _assert_rejected(failures, (clause, filename), plant)
        if plant == "else":
            assert any("no else branch" in str(v) for v in failures), (
                f"the else plant must be named by the no-else rule:\n{_explain(failures)}")
        _assert_no_violations(_relocation_violations(sources, oracle), "the live package")


@pytest.mark.parametrize("plant", ["dropped-from-both", "declared-without-oracle", "two-owners"])
def test_relocation_manifest_controls(relocation_subject, plant):
    """Criterion 1's manifest clauses, which per-module membership cannot see.

    Each plant edits MANIFEST and the module TOGETHER, so every file still
    matches its own manifest entry, and the exact violation list shows the
    manifest-level clause is the sole objector:

    * dropped-from-both: `REASON_ERROR` leaves MANIFEST['core.py'] and core.py
      -> `manifest-union`, because the oracle still has it.
    * declared-without-oracle: a `find_spec` probe is declared in
      MANIFEST['core.py'] and appended to core.py -> `no-oracle-segment`,
      because the byte gate used to skip a name the oracle lacks, and
      `manifest-union`.
    * two-owners: `VOLUME_NAME` is declared and defined in both core.py and
      request.py, byte-identical in each -> `manifest-duplicates`.
    """
    sources, oracle = relocation_subject
    manifest = {filename: list(names) for filename, names in MANIFEST.items()}
    changed = dict(sources)
    if plant == "dropped-from-both":
        manifest["core.py"].remove("REASON_ERROR")
        changed["core.py"] = sources["core.py"].replace(
            _segments(sources["core.py"])["REASON_ERROR"].decode(), "", 1)
        expected = [("manifest-union", ["REASON_ERROR"], [])]
    elif plant == "declared-without-oracle":
        manifest["core.py"].append("_ON_CONTAINER")
        changed["core.py"] += ("\n_ON_CONTAINER = "
                               "__import__('importlib').util.find_spec('modal') is not None\n")
        expected = [("manifest-union", [], ["_ON_CONTAINER"]),
                    ("no-oracle-segment", "core.py", "_ON_CONTAINER")]
    else:
        manifest["request.py"].append("VOLUME_NAME")
        changed["request.py"] += "\n" + _segments(sources["core.py"])["VOLUME_NAME"].decode()
        expected = [("manifest-duplicates", {"VOLUME_NAME": ["core.py", "request.py"]})]
    for _ in range(2):
        _assert_no_violations(_relocation_violations(sources, oracle), "the live package")
        violations = _relocation_violations(changed, oracle, manifest)
        assert violations == expected, (
            f"the {plant} plant must be rejected by exactly {[v[0] for v in expected]}; the gates "
            f"reported:\n{_explain(violations)}")
        _assert_no_violations(_relocation_violations(sources, oracle), "the live package")


@pytest.mark.parametrize("spelling", ["TYPE_CHECKING", "typing.TYPE_CHECKING", "os.TYPE_CHECKING"])
def test_relocation_accepts_import_only_type_checking(relocation_subject, spelling):
    sources, oracle = relocation_subject
    sources = dict(sources)
    sources["core.py"] += f"\nif {spelling}:\n    import decimal\n"
    _assert_no_violations(_relocation_violations(sources, oracle),
                          f"a legal imports-only `if {spelling}:` block in core.py")


def test_relocation_dependency_contract(relocation_subject):
    sources, _ = relocation_subject
    _assert_no_violations(_dependency_violations(sources), "the live package's import graph")


@pytest.mark.parametrize("plant", ["runtime-annotation", "core-edge", "missing-edge", "extra-edge"])
def test_relocation_dependency_controls(relocation_subject, plant):
    sources, _ = relocation_subject
    for _ in range(2):
        _assert_no_violations(_dependency_violations(sources), "the live package's import graph")
        changed = dict(sources)
        module = "commands" if plant == "runtime-annotation" else "core" if plant == "core-edge" else "source"
        if plant == "runtime-annotation":
            changed["commands.py"] = _hoist_type_checking_imports(changed["commands.py"])
        elif plant == "missing-edge":
            lines = changed[module + ".py"].splitlines(keepends=True)
            imports = [
                n for n in ast.parse(changed[module + ".py"]).body
                if isinstance(n, ast.ImportFrom) and n.level
            ]
            for node in reversed(imports):
                del lines[node.lineno - 1:node.end_lineno]
            changed[module + ".py"] = "".join(lines)
        else:
            edge = "request" if plant == "runtime-annotation" else "training"
            changed[module + ".py"] += f"\nfrom . import {edge}\n"
        _assert_rejected(_dependency_violations(changed), ("runtime-edges", module), plant)
        _assert_no_violations(_dependency_violations(sources), "the live package's import graph")


@pytest.mark.parametrize("statement, edge", [
    pytest.param("import scripts.modal_runner\n", PACKAGED, id="import-package"),
    pytest.param("import scripts.modal_runner as _pkg\n", PACKAGED, id="import-package-as"),
    pytest.param("from scripts import modal_runner\n", PACKAGED, id="from-parent-import"),
    pytest.param(
        "from scripts.modal_runner import VOLUME_NAME\n", PACKAGED, id="from-package-import-name"),
    pytest.param(
        "from scripts.modal_runner import training\n", PACKAGED, id="from-package-import-module"),
    pytest.param("from scripts.modal_runner.training import _tee_stream\n",
                 PACKAGED,
                 id="from-submodule-import"),
    pytest.param(
        "def _lazy():\n    import importlib\n"
        "    return importlib.import_module('scripts.modal_runner.training')\n",
        PACKAGED,
        id="import-module-literal-in-body"),
    pytest.param("from .. import modal_runner\n", "..", id="parent-relative-package"),
    pytest.param(
        "from ..modal_runner import training\n", "..modal_runner", id="parent-relative-submodule"),
])
def test_relocation_dependency_rejects_every_self_import_spelling_in_core(
        relocation_subject, statement, edge):
    """Criterion 6's leaf clause against every spelling that reaches the package.

    core.py must import no package module. Each row appends one spelling to
    core.py and requires the exact edge it resolves to, with no other clause
    firing: the old walker resolved only one-dot relative imports and
    `scripts.modal_runner[.x]` from-imports and dotted imports, so the
    package-level, parent and `..` spellings and the literal `import_module`
    all left core looking like a leaf.
    """
    sources, _ = relocation_subject
    for _ in range(2):
        _assert_no_violations(_dependency_violations(sources), "the live package's import graph")
        changed = {**sources, "core.py": sources["core.py"] + "\n" + statement}
        failures = _dependency_violations(changed)
        expected = [("runtime-edges", "core", {edge}, set())]
        assert failures == expected, (
            f"core.py importing the package as {statement!r} must be reported as the single "
            f"runtime edge {edge!r}; the gates reported:\n{_explain(failures)}")
        _assert_no_violations(_dependency_violations(sources), "the live package's import graph")


@pytest.mark.parametrize("plant", ["preflight-to-runtime", "extra-annotation"])
def test_relocation_annotation_edge_controls(relocation_subject, plant):
    """The `annotation-edges` clause, which no runtime-edge plant can hold on its own.

    * preflight-to-runtime: preflight's `TYPE_CHECKING` import of request moves
      to module scope, the preflight counterpart of the commands control above.
      Both of preflight's clauses must fire.
    * extra-annotation: source.py gains an annotation-only import of training.
      Its runtime edges are unchanged, so `annotation-edges` is the sole
      objector, and deleting that clause would leave this row green.
    """
    sources, _ = relocation_subject
    if plant == "preflight-to-runtime":
        changed = {**sources, "preflight.py": _hoist_type_checking_imports(sources["preflight.py"])}
        runtime = set(DEPENDENCIES["preflight"])
        expected = [("runtime-edges", "preflight", runtime | {"request"}, runtime),
                    ("annotation-edges", "preflight", set(), {"request"})]
    else:
        changed = {
            **sources, "source.py":
            sources["source.py"] + "\nif TYPE_CHECKING:\n    from .training import _tee_stream\n"
        }
        expected = [("annotation-edges", "source", {"training"}, set())]
    for _ in range(2):
        _assert_no_violations(_dependency_violations(sources), "the live package's import graph")
        failures = _dependency_violations(changed)
        assert failures == expected, (
            f"the {plant} plant must be rejected by exactly {[v[:2] for v in expected]}; the "
            f"gates reported:\n{_explain(failures)}")
        _assert_no_violations(_dependency_violations(sources), "the live package's import graph")


def test_relocation_size_contract(relocation_subject):
    sources, _ = relocation_subject
    sums = [
        sum(len(s.splitlines()) for s in _segments(sources[name + ".py"]).values())
        for name in RUNNER_MODULES
    ]
    recorded = [591, 318, 133, 248, 381, 95, 158, 559]
    assert sums == recorded, (
        f"per-module top-level segment line sums are {sums} for {list(RUNNER_MODULES)}, not the "
        "recorded 591/318/133/248/381/95/158/559. They can change whenever any module-level def, "
        "class or assignment is added, removed or edited, not only a relocated one; "
        "test_relocation_combined_contract names changes to relocated segments. This pin is "
        "a relocation-oracle clause (module docstring).")
    _assert_no_violations(_size_violations(sources), "the live package's module sizes and names")


@pytest.mark.parametrize("plant", ["forbidden-name", "segment-budget"])
def test_relocation_size_controls(tmp_path, relocation_subject, plant):
    """The forbidden-name row plants a REAL `utils.py` into a copy of the package.

    Injecting a dict entry, as this row used to, planted inside the population
    the gate is handed, so it could not show the gate ever sees a file on disk.
    """
    sources, _ = relocation_subject
    package = _copy_live_package(tmp_path)
    for _ in range(2):
        _assert_no_violations(_size_violations(_package_sources(tmp_path)), "the package copy")
        if plant == "forbidden-name":
            (package / "utils.py").write_text("", encoding="utf-8")
            changed = _package_sources(tmp_path)
            (package / "utils.py").unlink()
        else:
            changed = {
                **sources, "core.py": sources["core.py"] + "\ndef _huge():\n" + "    pass\n" * 900
            }
        _assert_rejected(_size_violations(changed), (plant, ), plant)
        _assert_no_violations(_size_violations(_package_sources(tmp_path)), "the restored copy")


@pytest.mark.parametrize("stem", ["utils", "telemetry"])
def test_package_gates_see_an_undeclared_module_on_disk(tmp_path, relocation_subject, stem):
    """A REAL undeclared module, with `import modal` and module-scope work, in a package copy.

    Before the gates enumerated the directory, this file passed every one of
    them (W3b structural review, 2026-09-22). Each gate that should object must:
    the file population (`files`), the header shape, the dependency table
    (`undeclared-module`) and import purity. `utils` is also a forbidden name;
    `telemetry` is not, so its row shows the population clauses fire on any
    undeclared file, not only on a grab-bag name.
    """
    _, oracle = relocation_subject
    package = _copy_live_package(tmp_path)
    rel = f"scripts/modal_runner/{stem}.py"
    declared = [f"scripts/modal_runner/{name}.py" for name in RUNNER_MODULES]

    def gates():
        """Every gate's verdict on the copy, read back from disk."""
        sources = _package_sources(tmp_path)
        return (sources, _relocation_violations(sources, oracle), _dependency_violations(sources),
                _size_violations(sources), _nonstdlib_module_scope_imports(tmp_path))

    for _ in range(2):
        sources, relocation, dependency, size, purity = gates()
        _assert_no_violations(relocation + dependency + size, "the package copy")
        assert purity == (declared, []), f"gate (e) on the package copy reported {purity}"
        (package / f"{stem}.py").write_text("import modal\nprint('work')\n", encoding="utf-8")
        undeclared = _runner_module_population(tmp_path)[1]
        assert undeclared == [rel], f"the population found {undeclared}, not the planted {rel}"
        sources, relocation, dependency, size, purity = gates()
        assert list(sources)[-1] == f"{stem}.py", f"the gates were not handed {rel}"
        _assert_rejected(relocation, ("files", [f"{stem}.py"], []), rel)
        _assert_rejected(relocation, ("header", f"{stem}.py"), rel)
        _assert_rejected(dependency, ("undeclared-module", stem), rel)
        assert (("forbidden-name", f"{stem}.py") in size) == (stem == "utils"), (
            f"the forbidden-name clause must fire for utils.py and only for it:\n{_explain(size)}")
        expected = (declared + [rel], [(rel, 1, "modal")])
        assert purity == expected, (
            f"gate (e) must scan {rel} and report its `import modal`; it reported {purity}")
        (package / f"{stem}.py").unlink()
    assert _package_sources(tmp_path) == _package_sources(ROOT), "the copy was not restored"


def test_package_import_purity_scans_every_module():
    scanned, violations = _nonstdlib_module_scope_imports(ROOT)
    declared = [f"scripts/modal_runner/{name}.py" for name in RUNNER_MODULES]
    assert scanned == declared, (
        f"gate (e) read {scanned}. Any path beyond the eight declared modules is an undeclared "
        "module in scripts/modal_runner/: declare it in RUNNER_MODULES "
        "(tests/test_modal_packaging.py) and this file's tables, or delete it.")
    assert violations == [], (
        f"these (file, line, module) imports execute on import and are not stdlib: {violations}. "
        "Move them into the function that needs them, or under `if TYPE_CHECKING:`.")


@pytest.mark.parametrize("container", ["if", "with", "except", "class", "try"])
def test_package_import_purity_recursive_controls(tmp_path, relocation_subject, container):
    sources, _ = relocation_subject
    package = _copy_live_package(tmp_path)
    plants = {
        "if": "if True:\n    import modal\n",
        "with": "with open('x'):\n    import modal\n",
        "except": "try:\n    pass\nexcept Exception:\n    import modal\n",
        "class": "class Plant:\n    import modal\n",
        "try": "try:\n    import modal\nexcept ImportError:\n    pass\n"
    }
    target = package / "training.py"
    declared = [f"scripts/modal_runner/{name}.py" for name in RUNNER_MODULES]
    for _ in range(2):
        scanned, failures = _nonstdlib_module_scope_imports(tmp_path)
        assert scanned == declared and failures == [], (
            f"gate (e) on the package copy read {scanned} and reported {failures}")
        target.write_text(sources["training.py"] + "\n" + plants[container], encoding="utf-8")
        scanned, failures = _nonstdlib_module_scope_imports(tmp_path)
        assert scanned == declared, f"gate (e) read {scanned}, not the eight declared modules"
        assert len(failures) == 1 and failures[0][
            0] == "scripts/modal_runner/training.py" and failures[0][2] == "modal", (
                f"gate (e) must report the `import modal` inside the {container} container in "
                f"training.py, and only it; it reported {failures}")
        assert _nonstdlib_module_scope_imports(tmp_path, top_level_only=True)[1] == [], (
            f"the tree.body-only instrument was expected to be blind to the {container} plant")
        target.write_text(sources["training.py"], encoding="utf-8")
        assert _nonstdlib_module_scope_imports(tmp_path)[1] == [], "the copy was not restored"


@pytest.mark.parametrize("defect", ["missing", "unreadable"])
def test_package_scans_reject_missing_or_unreadable_member(tmp_path, relocation_subject, defect):
    sources, _ = relocation_subject
    package = _copy_live_package(tmp_path)
    target = package / "training.py"
    for _ in range(2):
        assert len(_package_sources(tmp_path)) == 8, "the package copy lost a module"
        assert len(_nonstdlib_module_scope_imports(tmp_path)[0]) == 8, (
            "gate (e) did not read the copy's eight modules")
        target.unlink()
        if defect == "unreadable":
            target.mkdir()
        for scanner in (_package_sources, _nonstdlib_module_scope_imports):
            with pytest.raises(OSError, match="training.py"):
                scanner(tmp_path)
        if defect == "unreadable":
            target.rmdir()
        target.write_text(sources["training.py"], encoding="utf-8")
        assert _package_sources(tmp_path) == sources, "the copy was not restored"
        assert _nonstdlib_module_scope_imports(tmp_path)[1] == [], "the copy was not restored"
