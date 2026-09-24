"""The declared shape of scripts/modal_runner/: the one place a package module is declared.

Every gate that needs the module list reads it from here: the package-shape
gates (tests/test_modal_runner_package_shape.py), the packaging gates
(tests/test_modal_packaging.py), the mount and surface gates
(tests/test_modal_client.py) and the binding census
(tests/test_modal_patch_binding_census.py). Before this file, RUNNER_MODULES, the MANIFEST
keys and the DEPENDENCIES keys stated the list three times across two test
files, and a new module turned most of those gates red with differently worded
messages.

ADDING A MODULE, in this order (the facade docstring points here):
(1) add its MANIFEST and DEPENDENCIES entries below, and add it to the
DEPENDENCIES entry of each module that imports it (ANNOTATION_DEPENDENCIES for
an import made only under TYPE_CHECKING); (2) add a line for it to MODULE MAP
in the docstring of scripts/modal_runner/__init__.py; (3) `git add` the new
file. The mount and package-population gates in tests/test_modal_client.py
read `git ls-files`, so until step 3 they report the module missing although it
is on disk. (4) Create its test file, tests/test_modal_<module>.py (the path
RUNNER_TEST_FILES derives from this list), holding at least one test of the
module with its seam-manifest line, and `git add` it too: the importer census
in tests/test_modal_packaging.py reads `git ls-files` as well. The seam gate,
the reach floor and the binding census read RUNNER_TEST_FILES, so they expect
the file from step (1) on: a missing one fails the seam gate's source reader,
which names it, and one with no test in the seam manifest fails the floor's
scope check. THE PLACEMENT RULE FOR RUNNER TESTS in
tests/test_modal_packaging.py says what a test there must reach and what
adding one costs.

A MODULE THAT SPAWNS OR SIGNALS PROCESSES goes through
`training.ProcessControl`, the kill seam: `test_kill_seam_static_safety` in
tests/test_modal_training.py reads every module of the package (it too
expects the new file from step (1) on) and fails on a real `os.killpg`,
`os.getpgid`, `os.kill`, `signal.signal` or `subprocess.Popen` outside
`ProcessControl.system()`, and on any `killpg`/`getpgid` outside the
process-group guard in `training._signal_process_group`. Its tests hand the
module a ProcessControl whose `spawn`, `getpgid` and `killpg` are fakes (the
tests/conftest.py tripwire makes the real `system()` raise under pytest).
Read that test's clauses before adding either.

Data only: no function and no import, of the package or anything else, so
importing this file from any test module costs nothing and cannot collect a
test.
"""

# Owner of every top-level symbol, per module. Each module's top-level names
# must equal its entry, and no name may appear under two modules. Adding,
# removing, renaming or moving a symbol means editing its entry here in the same
# commit.
MANIFEST = {
    "core.py": [
        "VOLUME_NAME", "REGISTRY_NAME", "VOLUME_MOUNT", "SOURCES_ROOT", "INPUTS_ROOT", "RUNS_ROOT",
        "PROVENANCE_NAME", "STATUS_FILENAME", "RESERVATION_FILENAME", "MANIFEST_FILENAME",
        "TRAIN_LOG_NAME", "RESULT_FILENAME", "CHECKPOINT_NAME", "CHECKPOINT_SIDECAR_NAME",
        "CHECKPOINT_PUBLISH_REASON_NAME", "DEAD_CHECKPOINT_NAME", "PREBUILT_PYTHON",
        "ValidationError", "Status", "mounted_path", "sha256_file", "FileProvenance",
        "SCHEMA_VERSION", "Manifest", "HEARTBEAT_INTERVAL", "Registry", "LockLike", "sha256_bytes",
        "CompletionEvidence", "PreparedSource", "_utc_now", "_event_wait", "Clock",
        "ReloadingVolume", "AttemptContext"
    ],
    "request.py": [
        "ALLOWED_MAPS", "ALLOWED_GPUS", "ALLOWED_NUM_ENVS", "ALLOWED_CPU_CORES", "DEFAULT_GPU",
        "DEFAULT_NUM_ENVS", "DEFAULT_CPU_CORES", "DEFAULT_MEMORY_MIB", "DEFAULT_VEC_WORKERS",
        "DEFAULT_TIMEOUT_MINUTES", "DEFAULT_SAVE_EVERY_SECONDS", "MIN_MEMORY_MIB", "MAX_MEMORY_MIB",
        "MIN_TIMEOUT_MINUTES", "MAX_TIMEOUT_MINUTES", "MIN_SAVE_EVERY_SECONDS",
        "MAX_SAVE_EVERY_SECONDS", "AGENTS_PER_ENV", "BPTT_HORIZON", "MIN_BATCH_SIZE", "_RUN_ID_RE",
        "_SECRET_NAME_RE", "LIVE_TRAIN_OPTION_ARITY", "LIVE_TRAIN_OPTIONS",
        "RUNNER_OWNED_TRAIN_FLAGS", "RUN_ONLY_OPTIONS", "Action", "ResumeRequest",
        "ArtifactClientRequest", "RunRequest", "validate_run_id", "validate_secret_name",
        "parse_train_args", "_split_long_option", "_option_value", "validate_train_args",
        "build_run_request", "parse_artifact_client_request"
    ],
    "source.py": [
        "_COMMIT_SHA_RE", "_SAFE_TAR_TYPES", "SourceProvenance", "_run_git", "validate_clean_head",
        "_reject_unsafe_tar_member", "safe_extract_git_archive", "_staging_members",
        "_repack_deterministic", "create_source_bundle"
    ],
    "checkpoint.py": [
        "PREBUILT_LOAD_TIMEOUT_SECONDS", "_PREBUILT_LOAD_SOURCE", "CheckpointVerdict",
        "_import_torch", "_assert_weights_only_loadable", "validate_local_checkpoint",
        "_load_checkpoint_weights", "verify_checkpoint", "normalize_config_for_transport",
        "iter_metrics_steps", "validate_completed_run"
    ],
    "state.py": [
        "TERMINAL_STATUSES", "_ALLOWED_TRANSITIONS", "RunStatus", "STALE_AFTER", "ArtifactIndex",
        "FAILURE_UPLOAD", "ALLOWED_FAILURE_CODES", "REDELIVERED", "DerivedStatus",
        "HeartbeatWorker", "atomic_write_json", "read_status", "transition_status",
        "_transition_status_unlocked", "run_registry_key", "attempt_registry_key",
        "_reservation_path", "_manifest_path", "_volume_has_run", "record_run_failure",
        "finish_reservation", "reserve_run", "claim_attempt", "deliver_attempt", "write_heartbeat",
        "load_volume_json", "_parse_iso8601", "derive_status", "derive_run_view_from_bytes",
        "derive_run_view", "list_run_artifacts", "start_heartbeat_worker", "stop_heartbeat"
    ],
    "commands.py": [
        "UV_BIN", "TRAIN_SCRIPT", "_PRESERVED_CHILD_ENV_KEYS", "THREAD_CAP_ENV",
        "CUDA_PROBE_SOURCE", "assemble_train_argv", "build_train_argv", "build_dump_config_argv",
        "_is_preserved_child_env_key", "build_child_env", "build_install_command",
        "build_train_command", "build_dump_config_command", "build_cuda_probe_command"
    ],
    "preflight.py": [
        "ExpectedSource", "RemoteResume", "PreflightHost", "_verify_extracted_provenance",
        "_validate_remote_resume", "_hash_dumped_config", "_verify_archive_then_enter_preparing",
        "_extract_verified_source", "_BuiltSource", "_build_in_source", "_fail_preflight",
        "prepare_remote_source"
    ],
    "training.py": [
        "RunResult", "CHECKPOINT_SETTLE_SECONDS", "TERM_GRACE_SECONDS", "DEAD_RUN_EXIT_CODE",
        "POLL_INTERVAL_SECONDS", "REASON_SIGNAL", "REASON_TIMEOUT", "REASON_DEAD_RUN",
        "REASON_INVALID_EVIDENCE", "REASON_NONZERO_EXIT", "REASON_ERROR", "TrainingAttemptResult",
        "_UnusedArtifacts", "PublishOutcome", "_tee_stream", "ProcessControl",
        "execute_training_attempt", "_checkpoint_generation", "publish_stable_checkpoint",
        "_start_checkpoint_watcher", "_record_publish_reason", "_publish_and_note",
        "_close_log_sink", "_is_dead_run", "_map_child_exit", "_metrics_summary",
        "_optional_checkpoint_sha256", "_write_run_result", "_signal_process_group", "_LiveAttempt",
        "_run_training_attempt"
    ],
}
# Runtime import edges between package modules. core stays a leaf. A new edge
# is legal only if the graph stays acyclic; record it here. Values here and in
# ANNOTATION_DEPENDENCIES are declared modules only (the `table-edge` clause):
# an absolute or `..` import, or `from . import <name>`, is never an edge to
# record.
DEPENDENCIES = {
    "core": [],
    "request": ["commands", "core"],
    "source": ["core"],
    "checkpoint": ["core", "state"],
    "state": ["core", "request"],
    "commands": ["core"],
    "preflight": ["checkpoint", "commands", "core", "source", "state"],
    "training": ["checkpoint", "core", "state"]
}
# Edges that exist only under `if TYPE_CHECKING:` and never execute. commands ->
# request must stay annotation-only: request imports commands at run time, so a
# runtime edge back would be an import cycle.
ANNOTATION_DEPENDENCIES = {"commands": ["request"], "preflight": ["request"]}
# The qualified seams, {"owner.name": readers}: the only cross-module names read
# as `owner.name` through the module object, at call time, instead of being
# from-imported. A test that patches the owning module therefore reaches every
# reader (QUALIFIED SEAMS in scripts/modal_runner/__init__.py, whose list must
# match these keys). The `seam` clause in tests/test_modal_runner_package_shape.py
# reports a from-import or an import-time read of a seam, a reader set that
# differs from its entry here, an entry with no reader or whose owner does not
# define its name, and any other `module.name` read across modules through a
# module that a relative import binds.
QUALIFIED_SEAMS = {
    "core.PREBUILT_PYTHON": ("checkpoint", "commands"),
    "core.sha256_file": ("checkpoint", "preflight", "source", "training"),
    "state.transition_status": ("preflight", "training"),
    "checkpoint.validate_local_checkpoint": ("preflight", "training"),
}
# This file's repo-relative path, for failure messages that name it.
TABLES_FILE = "tests/modal_runner_tables.py"
# Where a package module is declared, as every gate's failure message names it,
# so the remedy reads the same wherever it fires.
TABLES = ("MANIFEST and DEPENDENCIES (and ANNOTATION_DEPENDENCIES for an import made only under "
          f"TYPE_CHECKING) in {TABLES_FILE}")
# The package's submodules in MANIFEST order, derived rather than declared, so
# the module list is stated once. `test_package_structure_contract` checks that
# DEPENDENCIES is keyed by exactly these modules.
RUNNER_MODULES = tuple(filename.removesuffix(".py") for filename in MANIFEST)
# Each module's repo-relative path, in the same order: the population the
# module-scope gates must read.
RUNNER_PATHS = tuple(f"scripts/modal_runner/{module}.py" for module in RUNNER_MODULES)
# Each module's test file, in the same order: tests/test_modal_<module>.py. This
# is the one list of runner test files. The seam gate, its reach floor and the
# importer list `_IMPORTERS` (all in tests/test_modal_packaging.py) and the
# binding census (tests/test_modal_patch_binding_census.py) read it under this
# name, so none of them holds a retyped copy, and a module added above brings
# its test file into every one of them (ADDING A MODULE, step 4). A runner test
# in a tests/test_modal_<x>.py that is not derived here is invisible to the
# seam gate, the floor and the binding census (gh#233).
# PITFALL: derive a module from a test file name through this tuple and
# RUNNER_MODULES, never through a `tests/test_modal_*.py` glob, which also
# matches test files that belong to no module.
RUNNER_TEST_FILES = tuple(f"tests/test_modal_{module}.py" for module in RUNNER_MODULES)
