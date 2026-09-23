"""AST gates for the shape of the scripts/modal_runner/ package.

Every clause here describes the shape the package keeps from now on:

* module population: the submodule files on disk (every `*.py` but
  `__init__.py`) are exactly the modules MANIFEST and DEPENDENCIES declare
  (`files`, `undeclared-module`), and none has a grab-bag name
  (`forbidden-name`);
* ownership: each module's top-level names equal its MANIFEST entry
  (`membership`), no module binds a name twice (`duplicates`) or assigns at
  module scope to a target MANIFEST cannot own (`undeclared-assignment`), and
  MANIFEST gives no symbol two owners (`manifest-duplicates`);
* module-scope shape (`header`): the docstring, imports, one imports-only
  `if TYPE_CHECKING:` block without else, and declarations, nothing else;
* the import graph: intra-package imports equal DEPENDENCIES
  (`runtime-edges`) and ANNOTATION_DEPENDENCIES (`annotation-edges`);
* import purity: no non-stdlib import executes when a module is imported;
* the facade (`facade`): `__init__.py` defines nothing and runs on every
  `import scripts.modal_runner`, so its module scope may hold only its
  docstring, named one-dot relative imports of its submodules and a literal
  `__all__`.

Editing a function body, a docstring or a constant's value trips none of them
unless the edit adds or removes an import. Adding, removing, renaming or moving
a top-level symbol, adding a module, or changing which package modules a module
imports does: update the table the failure message names, in the same commit as
the change. Runtime imports are compared with DEPENDENCIES and imports under
`if TYPE_CHECKING:` with ANNOTATION_DEPENDENCIES, separately, and the
dependency walk reads function bodies too. So a first runtime import of a
module imported only under `TYPE_CHECKING` is a new edge, and so is the reverse;
another import of a module already imported in the same way is not; removing
a module's last runtime import, or its last `TYPE_CHECKING` one, drops that
edge. No module size budget applies.

History: the split commit, which created scripts/modal_runner/, also proved in
this file (then tests/test_modal_relocation.py) that every moved declaration
was byte-identical to `scripts/modal_runner_lib.py` at 2bb32ac apart from 20
declared `module.name` qualifiers; those relocation-oracle checks were retired
after it.

Every failure message names the clause, the file or symbol, and the edit that
resolves it (`_explain`). Controls plant defects into in-memory copies of the
sources or into a copy of the package under `tmp_path`, never into the live
package. A control first re-checks the unplanted package; when that check
fails, its message starts with PRECONDITION and names the contract test whose
message says what to fix (both, for a control that re-checks two), because the
control itself did not break.
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
# so a change made to either for the seam's sake changes the membership and
# duplicates clauses here as well.
from tests.test_modal_packaging import (
    PACKAGED,
    RUNNER_MODULES,
    _module_level_binding_counts,
    _module_level_names,
    _module_scope_shape_violations,
    _nonstdlib_module_scope_imports,
    _runner_module_population,
    bare_spelling_imports,
)

ROOT = Path(__file__).resolve().parents[1]
# Owner of every top-level symbol, per module. Each module's top-level names
# must equal its entry, and no name may appear under two modules. Adding,
# removing, renaming or moving a symbol means editing its entry here in the same
# commit.
MANIFEST = {
    'core.py': [
        'VOLUME_NAME', 'REGISTRY_NAME', 'VOLUME_MOUNT', 'SOURCES_ROOT', 'INPUTS_ROOT', 'RUNS_ROOT',
        'PROVENANCE_NAME', 'STATUS_FILENAME', 'RESERVATION_FILENAME', 'MANIFEST_FILENAME',
        'TRAIN_LOG_NAME', 'RESULT_FILENAME', 'CHECKPOINT_NAME', 'CHECKPOINT_SIDECAR_NAME',
        'CHECKPOINT_PUBLISH_REASON_NAME', 'DEAD_CHECKPOINT_NAME', 'PREBUILT_PYTHON',
        'ValidationError', 'Status', 'mounted_path', 'sha256_file', 'FileProvenance',
        'SCHEMA_VERSION', 'Manifest', 'HEARTBEAT_INTERVAL', 'Registry', 'LockLike', 'sha256_bytes',
        'CompletionEvidence', 'PreparedSource'
    ],
    'request.py': [
        'ALLOWED_MAPS', 'ALLOWED_GPUS', 'ALLOWED_NUM_ENVS', 'ALLOWED_CPU_CORES', 'DEFAULT_GPU',
        'DEFAULT_NUM_ENVS', 'DEFAULT_CPU_CORES', 'DEFAULT_MEMORY_MIB', 'DEFAULT_VEC_WORKERS',
        'DEFAULT_TIMEOUT_MINUTES', 'DEFAULT_SAVE_EVERY_SECONDS', 'MIN_MEMORY_MIB', 'MAX_MEMORY_MIB',
        'MIN_TIMEOUT_MINUTES', 'MAX_TIMEOUT_MINUTES', 'MIN_SAVE_EVERY_SECONDS',
        'MAX_SAVE_EVERY_SECONDS', 'AGENTS_PER_ENV', 'BPTT_HORIZON', 'MIN_BATCH_SIZE', '_RUN_ID_RE',
        '_SECRET_NAME_RE', 'LIVE_TRAIN_OPTION_ARITY', 'LIVE_TRAIN_OPTIONS',
        'RUNNER_OWNED_TRAIN_FLAGS', 'RUN_ONLY_OPTIONS', 'Action', 'ResumeRequest',
        'ArtifactClientRequest', 'RunRequest', 'validate_run_id', 'validate_secret_name',
        'parse_train_args', '_split_long_option', '_option_value', 'validate_train_args',
        'build_run_request', 'parse_artifact_client_request'
    ],
    'source.py': [
        '_COMMIT_SHA_RE', '_SAFE_TAR_TYPES', 'SourceProvenance', '_run_git', 'validate_clean_head',
        '_reject_unsafe_tar_member', 'safe_extract_git_archive', '_staging_members',
        '_repack_deterministic', 'create_source_bundle'
    ],
    'checkpoint.py': [
        'PREBUILT_LOAD_TIMEOUT_SECONDS', '_PREBUILT_LOAD_SOURCE', 'CheckpointVerdict',
        '_import_torch', '_assert_weights_only_loadable', 'validate_local_checkpoint',
        '_load_checkpoint_weights', 'verify_checkpoint', 'normalize_config_for_transport',
        '_iter_metrics_steps', 'validate_completed_run'
    ],
    'state.py': [
        'TERMINAL_STATUSES', '_ALLOWED_TRANSITIONS', 'RunStatus', 'STALE_AFTER', 'ArtifactIndex',
        'FAILURE_UPLOAD', 'ALLOWED_FAILURE_CODES', 'REDELIVERED', 'DerivedStatus',
        'HeartbeatWorker', 'atomic_write_json', '_read_status', 'transition_status',
        '_transition_status_unlocked', 'run_registry_key', 'attempt_registry_key',
        '_reservation_path', '_manifest_path', '_volume_has_run', 'record_run_failure',
        'finish_reservation', 'reserve_run', 'claim_attempt', 'deliver_attempt', 'write_heartbeat',
        '_load_volume_json', '_parse_iso8601', 'derive_status', 'derive_run_view_from_bytes',
        'derive_run_view', 'list_run_artifacts', 'start_heartbeat_worker', '_stop_heartbeat'
    ],
    'commands.py': [
        'UV_BIN', 'TRAIN_SCRIPT', '_PRESERVED_CHILD_ENV_KEYS', '_THREAD_CAP_ENV',
        'CUDA_PROBE_SOURCE', '_assemble_train_argv', 'build_train_argv', 'build_dump_config_argv',
        '_is_preserved_child_env_key', 'build_child_env', 'build_install_command',
        'build_train_command', 'build_dump_config_command', 'build_cuda_probe_command'
    ],
    'preflight.py': [
        'ReloadingVolume', '_verify_extracted_provenance', '_validate_remote_resume',
        '_hash_dumped_config', 'prepare_remote_source'
    ],
    'training.py': [
        'RunResult', 'CHECKPOINT_SETTLE_SECONDS', 'TERM_GRACE_SECONDS', 'DEAD_RUN_EXIT_CODE',
        'POLL_INTERVAL_SECONDS', 'REASON_SIGNAL', 'REASON_TIMEOUT', 'REASON_DEAD_RUN',
        'REASON_INVALID_EVIDENCE', 'REASON_NONZERO_EXIT', 'REASON_ERROR', 'TrainingAttemptResult',
        '_UnusedArtifacts', 'PublishOutcome', '_tee_stream', 'execute_training_attempt',
        '_checkpoint_generation', 'publish_stable_checkpoint', '_start_checkpoint_watcher',
        '_record_publish_reason', '_publish_and_note', '_close_log_sink', '_is_dead_run',
        '_map_child_exit', '_metrics_summary', '_optional_checkpoint_sha256', '_write_run_result',
        '_signal_process_group', '_run_training_attempt'
    ],
}
# Runtime import edges between package modules. core stays a leaf. A new edge
# is legal only if the graph stays acyclic; record it here.
DEPENDENCIES = {
    'core': [],
    'request': ['commands', 'core'],
    'source': ['core'],
    'checkpoint': ['core', 'state'],
    'state': ['core', 'request'],
    'commands': ['core'],
    'preflight': ['checkpoint', 'commands', 'core', 'source', 'state'],
    'training': ['checkpoint', 'core', 'state']
}
# Edges that exist only under `if TYPE_CHECKING:` and never execute. commands ->
# request must stay annotation-only: request imports commands at run time, so a
# runtime edge back would be an import cycle.
ANNOTATION_DEPENDENCIES = {'commands': ['request'], 'preflight': ['request']}
# Module names that say what a module IS rather than what it owns, which is how
# a module becomes a junk drawer; the `forbidden-name` clause and its message
# both read this.
GRAB_BAG_MODULE_NAMES = ("utils", "helpers", "common", "misc")
# Where a package module is declared. Every message about a module on disk that
# the tables do not declare, or the reverse, names these, so the remedy reads the
# same wherever it fires.
_TABLES = ("RUNNER_MODULES (tests/test_modal_packaging.py) and MANIFEST and DEPENDENCIES (and "
           "ANNOTATION_DEPENDENCIES for an import made only under TYPE_CHECKING) in "
           "tests/test_modal_runner_package_shape.py")
_UNDECLARED_MODULE_REMEDY = (
    "A path beyond the modules RUNNER_MODULES declares is a module in scripts/modal_runner/ "
    f"that the tables do not declare: declare it in {_TABLES} in the same commit, or delete it.")
# How many times each control below plants its defect. Each pass starts by
# re-checking the unplanted package (or its tmp_path copy), so the second pass's
# check proves the first pass's plant-and-restore left the sources, or the copied
# files, as it found them. The gates hold no state between calls, so this is only
# about the tree. A control whose restore step broke would otherwise pass once
# and hide it.
_CONTROL_PASSES = 2


def _segments(source):
    """{top-level name: its exact declaration text} for one module's source.

    A segment includes the decorators, the complete class body and the comment
    lines directly above the declaration. Controls use it to cut one
    declaration out of a module, or copy it into another, without hand-writing
    its text.

    PITFALL: the lines are split as bytes, not with `str.splitlines`, which
    also breaks at `\\x0b`, `\\x0c`, `\\x1c`-`\\x1e`, `\\x85` and
    U+2028/U+2029 (measured over every code point). `ast` line numbers count
    only `\\n`, `\\r\\n` and `\\r`, which is all `bytes.splitlines` splits
    on, so a segment cannot drift off its node.
    """
    lines = source.encode().splitlines(keepends=True)
    found = {}
    for name, node in _module_level_names(ast.parse(source)).items():
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        while first > 1 and lines[first - 2].lstrip().startswith(b"#"):
            first -= 1
        found[name] = b"".join(lines[first - 1:node.end_lineno]).decode()
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


def _structure_violations(sources, manifest=MANIFEST):
    """The ownership, module-population and module-scope-shape clauses over `sources`.

    `sources` is a {filename: text} map. Combines the manifest's own clause,
    the file population, grab-bag module names, header shape, membership,
    duplicate bindings and undeclared assignments. `manifest` defaults to
    MANIFEST; controls pass an edited copy. Every file in `sources` gets every
    per-file check, including one no manifest entry declares.
    """
    # Annotated because the gates' tuples differ in shape per clause; pyrefly
    # would otherwise infer the first append's shape and reject the others.
    violations: list[tuple] = []
    # MANIFEST gives every symbol one owner. Per-module membership compares each
    # file with its OWN manifest entry, so it cannot see a symbol declared in,
    # and defined in, two modules; before this clause existed, that plant left
    # every gate green.
    owners = {}
    for filename, names in manifest.items():
        for name in names:
            owners.setdefault(name, []).append(filename)
    shared = {name: files for name, files in owners.items() if len(files) > 1}
    if shared:
        violations.append(("manifest-duplicates", shared))
    if set(sources) != set(manifest):
        violations.append(
            ("files", sorted(set(sources) - set(manifest)), sorted(set(manifest) - set(sources))))
    for filename, source in sources.items():
        if filename.removesuffix(".py") in GRAB_BAG_MODULE_NAMES:
            violations.append(("forbidden-name", filename))
        declared = set(manifest.get(filename, []))
        tree = ast.parse(source)
        examined, shape = _module_scope_shape_violations(tree)
        assert examined == len(tree.body), (
            f"the header instrument examined {examined} of {len(tree.body)} module-scope "
            f"statements in {filename}, so its verdict does not cover the file. This is a "
            "defect in `_module_scope_shape_violations` (tests/test_modal_packaging.py), not in "
            "the package: make it examine every statement in `tree.body` exactly once.")
        violations.extend(("header", filename, item) for item in shape)
        names = _module_level_names(tree)
        if set(names) != declared:
            violations.append(("membership", filename, sorted(set(names) - declared),
                               sorted(declared - set(names))))
        duplicates = {
            n: count
            for n, count in _module_level_binding_counts(tree).items() if count != 1
        }
        if duplicates:
            violations.append(("duplicates", filename, duplicates))
        # `_module_level_names` keeps one node per name: the LAST plain-name
        # binding. Any other module-scope assignment is invisible to membership:
        # an earlier binding of a rebound name, or an annotated attribute target
        # (`VOLUME_MOUNT.name: str = ...`), which the header rule allows because
        # it is an AnnAssign.
        owned = {id(node) for node in names.values()}
        violations.extend(
            ("undeclared-assignment", filename, node.lineno) for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and id(node) not in owned)
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
    """The import-graph clauses: resolved edges must equal DEPENDENCIES / ANNOTATION_DEPENDENCIES.

    Compared per module. A module in `sources` that DEPENDENCIES does not
    declare is reported as `undeclared-module` rather than skipped.
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


def _facade_violations(source):
    """`("facade", line, statement kind)` for each module-scope statement the facade may not hold.

    `source` is the text of scripts/modal_runner/__init__.py. The facade runs
    on every `import scripts.modal_runner`, production's included, yet the
    module-population clauses skip it because it declares no symbol of its own.
    Its module scope may hold only its docstring, one-dot relative imports of
    its submodules (`from .core import X`, `from . import core`; no star
    import, which would re-export whatever the submodule binds) and `__all__`
    bound to a list or tuple of string literals. That one rule covers, for this
    file, what the header rule covers for the modules (a call, a conditional, a
    loop), what the purity rule covers (an absolute or non-stdlib import) and a
    definition, which belongs in the module that owns it.
    """
    violations: list[tuple] = []
    for index, node in enumerate(ast.parse(source).body):
        docstring = (index == 0 and isinstance(node, ast.Expr)
                     and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
        relative = (isinstance(node, ast.ImportFrom) and node.level == 1
                    and all(alias.name != "*" for alias in node.names))
        exported = (isinstance(node, ast.Assign)
                    and [ast.unparse(target) for target in node.targets] == ["__all__"]
                    and isinstance(node.value, (ast.List, ast.Tuple)) and all(
                        isinstance(item, ast.Constant) and isinstance(item.value, str)
                        for item in node.value.elts))
        if not (docstring or relative or exported):
            violations.append(("facade", node.lineno, type(node).__name__))
    return violations


def _precondition(contract, subject="the live package"):
    """The subject of a control's precondition: the unplanted package, or its copy, is clean.

    WHY: every control re-checks the unplanted package, so one real defect in
    the package turns every control red with the same message. The prefix says
    the control is not what broke and names `contract`: the test, or the two
    tests, whose own message says what to fix.
    """
    return f"PRECONDITION (the live package already fails; fix {contract} first): {subject}"


def _explain(violations):
    """One line per violation: the clause, where it fired, and the edit that resolves it.

    WHY: the gates return tuples, and a red that prints only a tuple list tells
    whoever meets it neither which clause objected nor whether the fix is to
    revert their edit or to update a table in this file.
    """
    lines = []
    for v in violations:
        kind = v[0]
        if kind == "files":
            parts = []
            if v[1]:
                parts.append(f"holds modules the tables do not declare, {v[1]}: declare each in "
                             f"{_TABLES} in the same commit, or delete the stray file")
            if v[2]:
                parts.append(f"lacks modules MANIFEST declares, {v[2]}: restore them, or remove "
                             f"their entries from {_TABLES}")
            text = f"scripts/modal_runner/ {'; and it '.join(parts)}."
        elif kind == "undeclared-module":
            text = (f"scripts/modal_runner/{v[1]}.py is not in DEPENDENCIES. Declare the module in "
                    f"{_TABLES}, or delete it.")
        elif kind == "manifest-duplicates":
            text = (
                f"MANIFEST gives these symbols more than one owner: {v[1]}. Each symbol lives in "
                "exactly one module; delete the extra MANIFEST entries and definitions.")
        elif kind == "header":
            text = (f"{v[1]} line {v[2][1]} ({v[2][0]}): {v[2][2]}. Module scope may hold only the "
                    "docstring, imports, one imports-only `if TYPE_CHECKING:` block without else, "
                    "and declarations; move the statement into a function.")
        elif kind == "undeclared-assignment":
            text = (
                f"{v[1]} line {v[2]} is a module-level assignment MANIFEST cannot own: its name "
                "is bound again later in the file, or it annotates a target that is not a "
                "plain name. Bind each module-level name once, to a plain name listed in "
                f"MANIFEST['{v[1]}'], and move any other assignment into a function.")
        elif kind == "membership":
            parts = []
            if v[2]:
                parts.append(f"{v[1]} defines {v[2]}, which MANIFEST['{v[1]}'] lacks")
            if v[3]:
                parts.append(f"MANIFEST['{v[1]}'] lists {v[3]}, which {v[1]} does not define")
            text = (f"{'; '.join(parts)}. If you meant to add, remove or move a symbol, update "
                    f"MANIFEST['{v[1]}'] in tests/test_modal_runner_package_shape.py in the same "
                    "commit.")
        elif kind == "duplicates":
            text = (f"{v[1]} binds {v[2]} more than once at module scope; the later binding "
                    "silently shadows the earlier one. Keep one.")
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
            text = (f"{v[1]} is a grab-bag module name ({'/'.join(GRAB_BAG_MODULE_NAMES)}). Name "
                    "the module for the concern it owns.")
        elif kind == "facade":
            text = (f"scripts/modal_runner/__init__.py line {v[1]} ({v[2]}): the facade runs "
                    "on every `import scripts.modal_runner`, so its module scope may hold only "
                    "its docstring, one-dot relative imports of its submodules (named, never "
                    "`*`) and `__all__` alone bound to a list or tuple of string literals. Move "
                    "the statement into the module that owns it, and re-export a name with "
                    "`from .<module> import <name>`.")
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
def live_sources():
    """{filename: text} for the live package's submodules, read fresh for each test."""
    return _package_sources(ROOT)


def test_package_structure_contract(live_sources):
    """The structure clauses on the live package, plus the tables' consistency."""
    declared = set(RUNNER_MODULES)
    manifest_modules = {name.removesuffix(".py") for name in MANIFEST}
    keyed = (manifest_modules == declared and set(DEPENDENCIES) == declared
             and set(ANNOTATION_DEPENDENCIES) <= declared)
    assert keyed, (
        f"MANIFEST {sorted(MANIFEST)}, DEPENDENCIES {sorted(DEPENDENCIES)} and "
        f"ANNOTATION_DEPENDENCIES {sorted(ANNOTATION_DEPENDENCIES)} must be keyed by the modules "
        f"RUNNER_MODULES declares ({list(RUNNER_MODULES)}); update them together.")
    _assert_no_violations(_structure_violations(live_sources), "the live package")


def test_package_facade_contract():
    """The facade's module-scope rule (`_facade_violations`) on the live `__init__.py`."""
    facade = (ROOT / "scripts" / "modal_runner" / "__init__.py").read_text(encoding="utf-8")
    _assert_no_violations(_facade_violations(facade), "the live facade")


# Each row appends one plant to the live facade, in memory, and pairs it with the
# statement kinds, in order, that the `facade` clause must report for the
# appended lines. The ID ends in the verdict the row asserts.
@pytest.mark.parametrize("plant, kinds", [
    pytest.param(plant, kinds, id=f"{name}-{'rejected' if kinds else 'accepted'}")
    for name, plant, kinds in [
        ("env-mutation", "import os\nos.environ.setdefault('CS2RL_FACADE_PLANT', '1')\n",
         ["Import", "Expr"]),
        ("print", "print('facade side effect')\n", ["Expr"]),
        ("non-stdlib-import", "import numpy\n", ["Import"]),
        ("conditional-import", "if True:\n    from .core import VOLUME_MOUNT\n", ["If"]),
        ("definition", "def _helper():\n    return 1\n", ["FunctionDef"]),
        ("computed-all", "__all__ = sorted(__all__)\n", ["Assign"]),
        ("absolute-import", "from os import environ\n", ["ImportFrom"]),
        ("parent-relative-import", "from .. import run_modal\n", ["ImportFrom"]),
        ("stray-string", "'a string after the docstring'\n", ["Expr"]),
        ("other-assign", "HELPER = ['a']\n", ["Assign"]),
        ("computed-all-element", "__all__ = [print('x')]\n", ["Assign"]),
        ("non-string-all-element", "__all__ = [1]\n", ["Assign"]),
        ("chained-all", "__all__ = HELPER = ['a']\n", ["Assign"]),
        ("set-all", "__all__ = {'a'}\n", ["Assign"]),
        ("star-import", "from .core import *\n", ["ImportFrom"]),
        ("tuple-all", "__all__ = ('a',)\n", []),
        ("re-export", "from .core import VOLUME_MOUNT\n", []),
    ]
])
def test_package_facade_controls(plant, kinds):
    """One plant per row, rejected by the `facade` clause alone, or accepted.

    * env-mutation: `os.environ.setdefault(...)` at the facade's module scope.
      Before this clause existed it, and the bare `print`, turned no gate red:
      the population gates skip `__init__.py`, and the surface gates read names.
    * non-stdlib-import: the purity half. The container-equivalent import in
      tests/test_modal_client.py also objects, but only by running the image.
    * conditional-import, definition, computed-all: the header half, a helper
      that belongs in its owning module, and an `__all__` that runs code.
    * absolute-import, parent-relative-import, stray-string, other-assign,
      computed-all-element, non-string-all-element: one sub-condition of the
      allowed shape each, and weakening it turns that row red. The two import
      rows hold `level == 1` from below (the stdlib `from os import environ`,
      which the purity rule would pass) and from above; stray-string holds
      the docstring to index 0; other-assign holds the `__all__` target;
      computed-all-element holds the element check, non-string-all-element
      its string half.
    * chained-all and set-all: `__all__` must be the only target and a list or
      tuple, so a second target and a set are rejected. tuple-all is ACCEPTED,
      the tuple half of that shape.
    * star-import: re-exports whatever the submodule binds. The client surface
      gate also objects, but only by running the image.
    * re-export: ACCEPTED. A new one-dot relative import is legal shape here;
      whether the name belongs in the facade is the surface gates' question,
      so a clause that rejected every added line would fail this row.
    """
    facade = (ROOT / "scripts" / "modal_runner" / "__init__.py").read_text(encoding="utf-8")
    _assert_no_violations(_facade_violations(facade),
                          _precondition("test_package_facade_contract", "the live facade"))
    # The plant's statements start on the line after the facade's last line. A
    # rejected row names one kind per planted statement, so `strict` also checks
    # that the plant parses into as many statements as the row names.
    lines = [facade.count("\n") + node.lineno for node in ast.parse(plant).body]
    expected = [("facade", line, kind)
                for line, kind in zip(lines, kinds, strict=True)] if kinds else []
    violations = _facade_violations(facade + plant)
    verdict = f"rejected by exactly {expected}" if expected else "accepted"
    assert violations == expected, (
        f"the facade plant {plant!r} must be {verdict}; the gate reported:\n{_explain(violations)}")


def test_package_membership_rejects_wrong_owner_with_correct_union(live_sources):
    precondition = _precondition("test_package_structure_contract")
    for _ in range(_CONTROL_PASSES):
        _assert_no_violations(_structure_violations(live_sources), precondition)
        segment = _segments(live_sources["core.py"])["VOLUME_NAME"]
        wrong = dict(live_sources)
        wrong["core.py"] = wrong["core.py"].replace(segment, "", 1)
        wrong["request.py"] += "\n" + segment

        def union(modules):
            """Ignore module ownership to demonstrate the weaker union-only check."""
            return set().union(*(_module_level_names(ast.parse(s)) for s in modules.values()))

        assert union(wrong) == union(live_sources), (
            "the wrong-owner plant changed the union of names, so it no longer shows that a "
            "union-only check would pass it")
        failures = _structure_violations(wrong)
        objecting = sorted({v[1] for v in failures if v[0] == "membership"})
        assert objecting == [
            "core.py", "request.py"
        ], ("moving VOLUME_NAME from core.py to request.py must fail membership in BOTH files:\n"
            f"{_explain(failures)}")
        _assert_no_violations(_structure_violations(live_sources), precondition)


@pytest.mark.parametrize("plant", [
    "unknown", "duplicate", "missing", "annotated-attribute", "else", "later-expression",
    "later-find-spec"
])
def test_package_ownership_and_header_controls(live_sources, plant):
    """One planted edit per row, rejected by the ownership or header clause it names.

    `annotated-attribute` plants an import-time mutation written as an
    annotated assignment (`VOLUME_MOUNT.name: str = ...`). It must be rejected
    by `undeclared-assignment` ALONE: membership reads plain-name bindings
    only, and the header rule allows an AnnAssign, so deleting that clause
    would leave this row green.
    """
    filename = "training.py" if plant.startswith("later-") else "core.py"
    additions = {
        "unknown":
        "\n_ON_CONTAINER = 1\n",
        "duplicate":
        "\nVOLUME_NAME = 'duplicate'\n",
        "annotated-attribute":
        "\nVOLUME_MOUNT.name: str = 'volume'\n",
        "else":
        "\nif TYPE_CHECKING:\n    import decimal\nelse:\n    print('work')\n",
        "later-expression":
        "\nprint('work')\n",
        "later-find-spec":
        "\nif importlib.util.find_spec('modal') is not None:\n    _ON_CONTAINER = True\n",
    }
    clause = {
        "unknown": "membership",
        "missing": "membership",
        "duplicate": "duplicates",
        "annotated-attribute": "undeclared-assignment",
    }.get(plant, "header")
    precondition = _precondition("test_package_structure_contract")
    for _ in range(_CONTROL_PASSES):
        _assert_no_violations(_structure_violations(live_sources), precondition)
        changed = live_sources[filename] + additions.get(plant, "")
        if plant == "missing":
            changed = changed.replace(_segments(changed)["VOLUME_NAME"], "", 1)
        failures = _structure_violations({**live_sources, filename: changed})
        _assert_rejected(failures, (clause, filename), plant)
        if plant == "annotated-attribute":
            # The plant is the last line of `changed`, which ends in a newline.
            expected = [(clause, filename, changed.count("\n"))]
            assert failures == expected, (
                f"the {plant} plant must be rejected by exactly {expected}; the gates "
                f"reported:\n{_explain(failures)}")
        if plant == "else":
            assert any("no else branch" in str(v) for v in failures), (
                f"the else plant must be named by the no-else rule:\n{_explain(failures)}")
        _assert_no_violations(_structure_violations(live_sources), precondition)


# Each row pairs a plant with the exact violation list it must produce. Its ID is
# the plant plus the verdict DERIVED from that list ("-accepted" when it is
# empty), so a collected ID reads as the claim ("...[dropped-from-both-accepted]")
# and no edit or reordering of the rows can make an ID misstate its assertion.
@pytest.mark.parametrize("plant, expected", [
    pytest.param(plant, expected, id=f"{plant}-{'rejected' if expected else 'accepted'}")
    for plant, expected in [
        ("dropped-from-both", []),
        ("added-to-both", []),
        ("two-owners", [("manifest-duplicates", {
            "VOLUME_NAME": ["core.py", "request.py"]
        })]),
        ("declared-grab-bag", [("forbidden-name", "utils.py")]),
    ]
])
def test_package_manifest_controls(live_sources, plant, expected):
    """MANIFEST edited TOGETHER with the modules: the membership remedy, and the clauses it leaves.

    Every plant keeps each file equal to its own manifest entry, so membership
    has nothing to report, and the exact violation list says which clause, if
    any, still objects (each row's test ID ends in the verdict it asserts):

    * dropped-from-both: `REASON_ERROR` leaves MANIFEST['training.py'] and training.py.
      ACCEPTED: deleting a symbol and its entry is the membership message's
      remedy, and no other clause may hold it red.
    * added-to-both: a new constant is declared in MANIFEST['core.py'] and
      defined in core.py. ACCEPTED, for the same reason.
    * two-owners: `VOLUME_NAME` is declared and defined in both core.py and
      request.py, byte-identical in each -> `manifest-duplicates` alone.
    * declared-grab-bag: an empty `utils.py` is declared in MANIFEST and added
      to the sources -> `forbidden-name` alone. The population clause (`files`)
      is satisfied, so this is the case the name ban exists for.
    """
    manifest = {filename: list(names) for filename, names in MANIFEST.items()}
    changed = dict(live_sources)
    if plant == "dropped-from-both":
        manifest["training.py"].remove("REASON_ERROR")
        changed["training.py"] = live_sources["training.py"].replace(
            _segments(live_sources["training.py"])["REASON_ERROR"], "", 1)
    elif plant == "added-to-both":
        manifest["core.py"].append("POLL_JITTER_SECONDS")
        changed["core.py"] += "\nPOLL_JITTER_SECONDS = 0.5\n"
    elif plant == "two-owners":
        manifest["request.py"].append("VOLUME_NAME")
        changed["request.py"] += "\n" + _segments(live_sources["core.py"])["VOLUME_NAME"]
    elif plant == "declared-grab-bag":
        manifest["utils.py"] = []
        changed["utils.py"] = ""
    else:
        pytest.fail(f"the {plant} row has no plant")
    assert changed != live_sources, f"the {plant} plant did not change the sources"
    precondition = _precondition("test_package_structure_contract")
    for _ in range(_CONTROL_PASSES):
        _assert_no_violations(_structure_violations(live_sources), precondition)
        violations = _structure_violations(changed, manifest)
        verdict = f"rejected by exactly {[v[0] for v in expected]}" if expected else "accepted"
        assert violations == expected, (
            f"the {plant} plant must be {verdict}; the gates reported:\n{_explain(violations)}")
        _assert_no_violations(_structure_violations(live_sources), precondition)


@pytest.mark.parametrize("spelling", ["TYPE_CHECKING", "typing.TYPE_CHECKING", "os.TYPE_CHECKING"])
def test_package_header_accepts_import_only_type_checking(live_sources, spelling):
    _assert_no_violations(_structure_violations(live_sources),
                          _precondition("test_package_structure_contract"))
    sources = dict(live_sources)
    sources["core.py"] += f"\nif {spelling}:\n    import decimal\n"
    _assert_no_violations(_structure_violations(sources),
                          f"a legal imports-only `if {spelling}:` block in core.py")


def test_package_dependency_contract(live_sources):
    _assert_no_violations(_dependency_violations(live_sources), "the live package's import graph")


@pytest.mark.parametrize("plant", ["runtime-annotation", "core-edge", "missing-edge", "extra-edge"])
def test_package_dependency_controls(live_sources, plant):
    precondition = _precondition("test_package_dependency_contract",
                                 "the live package's import graph")
    for _ in range(_CONTROL_PASSES):
        _assert_no_violations(_dependency_violations(live_sources), precondition)
        changed = dict(live_sources)
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
        _assert_no_violations(_dependency_violations(live_sources), precondition)


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
def test_package_dependency_rejects_every_self_import_spelling_in_core(
        live_sources, statement, edge):
    """The leaf clause (core imports no package module) against every spelling of the package.

    core.py must import no package module. Each row appends one spelling to
    core.py and requires the exact edge it resolves to, with no other clause
    firing: the old walker resolved only one-dot relative imports and
    `scripts.modal_runner[.x]` from-imports and dotted imports, so the
    package-level, parent and `..` spellings and the literal `import_module`
    all left core looking like a leaf.
    """
    precondition = _precondition("test_package_dependency_contract",
                                 "the live package's import graph")
    for _ in range(_CONTROL_PASSES):
        _assert_no_violations(_dependency_violations(live_sources), precondition)
        changed = {**live_sources, "core.py": live_sources["core.py"] + "\n" + statement}
        failures = _dependency_violations(changed)
        expected = [("runtime-edges", "core", {edge}, set())]
        assert failures == expected, (
            f"core.py importing the package as {statement!r} must be reported as the single "
            f"runtime edge {edge!r}; the gates reported:\n{_explain(failures)}")
        _assert_no_violations(_dependency_violations(live_sources), precondition)


@pytest.mark.parametrize("plant", ["preflight-to-runtime", "extra-annotation", "typing-attribute"])
def test_package_annotation_edge_controls(live_sources, plant):
    """The `annotation-edges` clause, which no runtime-edge plant can hold on its own.

    * preflight-to-runtime: preflight's `TYPE_CHECKING` import of request moves
      to module scope, the preflight counterpart of the commands control above.
      Both of preflight's clauses must fire.
    * extra-annotation: source.py gains an annotation-only import of training.
      Its runtime edges are unchanged, so `annotation-edges` is the sole
      objector, and deleting that clause would leave this row green.
    * typing-attribute: the same plant under `if typing.TYPE_CHECKING:`.
      `_dependency_edges` recognises the block by its attribute spelling as
      well as by the bare name; drop the attribute half and the import becomes
      a runtime edge, which this row rejects. Nothing else in this file uses
      that spelling on a plant whose edges are compared, so before this row
      that knock-out left every test here green.
    """
    if plant == "preflight-to-runtime":
        changed = {
            **live_sources, "preflight.py":
            _hoist_type_checking_imports(live_sources["preflight.py"])
        }
        runtime = set(DEPENDENCIES["preflight"])
        expected = [("runtime-edges", "preflight", runtime | {"request"}, runtime),
                    ("annotation-edges", "preflight", set(), {"request"})]
    else:
        guard = "typing.TYPE_CHECKING" if plant == "typing-attribute" else "TYPE_CHECKING"
        changed = {
            **live_sources, "source.py":
            live_sources["source.py"] + f"\nif {guard}:\n    from .training import _tee_stream\n"
        }
        expected = [("annotation-edges", "source", {"training"}, set())]
    precondition = _precondition("test_package_dependency_contract",
                                 "the live package's import graph")
    for _ in range(_CONTROL_PASSES):
        _assert_no_violations(_dependency_violations(live_sources), precondition)
        failures = _dependency_violations(changed)
        assert failures == expected, (
            f"the {plant} plant must be rejected by exactly {[v[:2] for v in expected]}; the "
            f"gates reported:\n{_explain(failures)}")
        _assert_no_violations(_dependency_violations(live_sources), precondition)


@pytest.mark.parametrize("stem", ["utils", "telemetry"])
def test_package_gates_see_an_undeclared_module_on_disk(tmp_path, stem):
    """A REAL undeclared module, with `import modal` and module-scope work, in a package copy.

    Before the gates enumerated the directory, this file passed every one of
    them (structural review, 2026-09-22). Each gate that should object must:
    the file population (`files`), the header shape, the dependency table
    (`undeclared-module`) and import purity. `utils` is also a forbidden name;
    `telemetry` is not, so its row shows the population clauses fire on any
    undeclared file, not only on a grab-bag name. The `utils` row is also the
    `forbidden-name` clause's control for a file read from disk; the
    declared-grab-bag-rejected row of `test_package_manifest_controls` shows that
    clause objecting alone.
    """
    package = _copy_live_package(tmp_path)
    rel = f"scripts/modal_runner/{stem}.py"
    declared = [f"scripts/modal_runner/{name}.py" for name in RUNNER_MODULES]
    contracts = "test_package_structure_contract or test_package_dependency_contract"
    purity_contract = "test_package_import_purity_scans_every_module"

    def gates():
        """Every gate's verdict on the copy, read back from disk."""
        sources = _package_sources(tmp_path)
        return (sources, _structure_violations(sources), _dependency_violations(sources),
                _nonstdlib_module_scope_imports(tmp_path))

    for _ in range(_CONTROL_PASSES):
        sources, structure, dependency, purity = gates()
        _assert_no_violations(structure + dependency, _precondition(contracts, "the package copy"))
        assert purity == (declared, []), (
            f"{_precondition(purity_contract, 'gate (e) on the package copy')} reported {purity}. "
            f"{_UNDECLARED_MODULE_REMEDY}")
        (package / f"{stem}.py").write_text("import modal\nprint('work')\n", encoding="utf-8")
        undeclared = _runner_module_population(tmp_path)[1]
        assert undeclared == [rel], f"the population found {undeclared}, not the planted {rel}"
        sources, structure, dependency, purity = gates()
        assert list(sources)[-1] == f"{stem}.py", f"the gates were not handed {rel}"
        _assert_rejected(structure, ("files", [f"{stem}.py"], []), rel)
        _assert_rejected(structure, ("header", f"{stem}.py"), rel)
        _assert_rejected(dependency, ("undeclared-module", stem), rel)
        assert (("forbidden-name", f"{stem}.py") in structure) == (stem == "utils"), (
            "the forbidden-name clause must fire for utils.py and only for it:\n"
            f"{_explain(structure)}")
        expected = (declared + [rel], [(rel, 1, "modal")])
        assert purity == expected, (
            f"gate (e) must scan {rel} and report its `import modal`; it reported {purity}")
        (package / f"{stem}.py").unlink()
    assert _package_sources(tmp_path) == _package_sources(ROOT), "the copy was not restored"


def test_package_import_purity_scans_every_module():
    scanned, violations = _nonstdlib_module_scope_imports(ROOT)
    declared = [f"scripts/modal_runner/{name}.py" for name in RUNNER_MODULES]
    assert scanned == declared, f"gate (e) read {scanned}. {_UNDECLARED_MODULE_REMEDY}"
    assert violations == [], (
        f"these (file, line, module) imports execute on import and are not stdlib: {violations}. "
        "Move them into the function that needs them, or under `if TYPE_CHECKING:`.")


@pytest.mark.parametrize("container", ["if", "with", "except", "class", "try"])
def test_package_import_purity_recursive_controls(tmp_path, live_sources, container):
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
    purity_contract = "test_package_import_purity_scans_every_module"
    for _ in range(_CONTROL_PASSES):
        scanned, failures = _nonstdlib_module_scope_imports(tmp_path)
        assert scanned == declared and failures == [], (
            f"{_precondition(purity_contract, 'gate (e) on the package copy')} read {scanned} and "
            f"reported {failures}. {_UNDECLARED_MODULE_REMEDY}")
        target.write_text(live_sources["training.py"] + "\n" + plants[container], encoding="utf-8")
        scanned, failures = _nonstdlib_module_scope_imports(tmp_path)
        assert scanned == declared, (
            f"gate (e) read {scanned}, not the {len(declared)} modules RUNNER_MODULES declares")
        assert len(failures) == 1 and failures[0][
            0] == "scripts/modal_runner/training.py" and failures[0][2] == "modal", (
                f"gate (e) must report the `import modal` inside the {container} container in "
                f"training.py, and only it; it reported {failures}")
        assert _nonstdlib_module_scope_imports(tmp_path, top_level_only=True)[1] == [], (
            f"the tree.body-only instrument was expected to be blind to the {container} plant")
        target.write_text(live_sources["training.py"], encoding="utf-8")
        assert _nonstdlib_module_scope_imports(tmp_path)[1] == [], "the copy was not restored"


@pytest.mark.parametrize("defect", ["missing", "unreadable"])
def test_package_scans_reject_missing_or_unreadable_member(tmp_path, live_sources, defect):
    package = _copy_live_package(tmp_path)
    target = package / "training.py"
    declared = [f"scripts/modal_runner/{name}.py" for name in RUNNER_MODULES]
    purity_contract = "test_package_import_purity_scans_every_module"
    for _ in range(_CONTROL_PASSES):
        copied = sorted(_package_sources(tmp_path))
        assert copied == sorted(Path(rel).name for rel in declared), (
            f"{_precondition('test_package_structure_contract', 'the package copy')} holds "
            f"{copied}, not the {len(declared)} modules RUNNER_MODULES declares, so this control "
            f"would not start from the live package. {_UNDECLARED_MODULE_REMEDY}")
        scanned = _nonstdlib_module_scope_imports(tmp_path)[0]
        assert scanned == declared, (
            f"{_precondition(purity_contract, 'gate (e)')} read {scanned} from the package copy, "
            f"not the {len(declared)} modules RUNNER_MODULES declares. {_UNDECLARED_MODULE_REMEDY}")
        target.unlink()
        if defect == "unreadable":
            target.mkdir()
        for scanner in (_package_sources, _nonstdlib_module_scope_imports):
            with pytest.raises(OSError, match="training.py"):
                scanner(tmp_path)
        if defect == "unreadable":
            target.rmdir()
        target.write_text(live_sources["training.py"], encoding="utf-8")
        assert _package_sources(tmp_path) == live_sources, "the copy was not restored"
        assert _nonstdlib_module_scope_imports(tmp_path)[1] == [], "the copy was not restored"
