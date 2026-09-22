"""Modal-free request, path, source, checkpoint and run-state library for the optional runner.

WHY this package exists separately from scripts/run_modal.py: status/download
and all validation must work without importing or hydrating a Modal App.
Importing the App would build the CUDA image. Every module in this package
therefore imports only the standard library and its own sibling modules at
module scope. Torch is imported lazily, inside `checkpoint._import_torch` and
`checkpoint._load_checkpoint_weights`, and Modal is never imported here.

THIS FACADE IS DELIBERATELY NARROW. It re-exports exactly the 41 names that
production code reads from the package (scripts/run_modal.py,
scripts/modal_artifacts.py and scripts/modal_backfill_sidecar.py), and nothing
more. `_production_package_surface` in tests/test_modal_client.py derives that
set from those three callers and the surface gates pin `__all__` and the
module's attributes to it. Tests read and patch the OWNING submodule
(`scripts.modal_runner.state`, not `scripts.modal_runner`). The names here are
from-imported copies, so patching one replaces only the facade's binding and
never reaches the submodule code that calls it. Never widen the facade to
satisfy a test; import the owning submodule instead.

QUALIFIED SEAMS. Exactly four names are read across modules through the module
object rather than a from-import: `core.PREBUILT_PYTHON`, `core.sha256_file`,
`state.transition_status` and `checkpoint.validate_local_checkpoint`. Tests
patch them on their owning module, and the attribute read at call time is what
lets that patch reach callers in other modules. Rewriting one as
`from .core import sha256_file` silently detaches it from the patch. Every
other cross-module name is from-imported, so patching it on its owning module
does NOT reach the importing module (training.py and preflight.py hold both
`state` and from-imported `state` names).

PITFALLS:
  * Live train.py argparse accepts prefixes (`--devi` → `--device`). The runner
    must NOT. Only exact long-option names from the mirrored live set are legal.
  * Client Volume APIs take root-relative PurePosixPath (`runs/...`); the
    container sees the same object at `/artifacts/runs/...`. Mixing the two
    namespaces silently talks to the wrong path.
  * config.json's `env` field is only a label: src/train_config.py writes
    `cs2-<map>` from train.py's resolved `--map`, and the historical `cs2-dust2`
    only for callers whose `map` attribute is missing or empty. effective_map is
    the runner's source of truth and is never derived from that field.
"""
from .checkpoint import validate_local_checkpoint, verify_checkpoint
from .core import (
    _THREAD_CAP_ENV,
    ALLOWED_MAPS,
    CHECKPOINT_NAME,
    CHECKPOINT_SIDECAR_NAME,
    DEFAULT_CPU_CORES,
    DEFAULT_GPU,
    DEFAULT_MEMORY_MIB,
    DEFAULT_NUM_ENVS,
    DEFAULT_SAVE_EVERY_SECONDS,
    DEFAULT_TIMEOUT_MINUTES,
    DEFAULT_VEC_WORKERS,
    INPUTS_ROOT,
    REDELIVERED,
    REGISTRY_NAME,
    RESERVATION_FILENAME,
    RUNS_ROOT,
    SCHEMA_VERSION,
    SOURCES_ROOT,
    STATUS_FILENAME,
    TERMINAL_STATUSES,
    VOLUME_NAME,
    Action,
    Manifest,
    ValidationError,
    mounted_path,
    sha256_bytes,
)
from .preflight import prepare_remote_source
from .request import RunRequest, build_run_request, parse_artifact_client_request, validate_run_id
from .source import create_source_bundle, validate_clean_head
from .state import (
    _load_volume_json,
    claim_attempt,
    derive_run_view_from_bytes,
    finish_reservation,
    reserve_run,
)
from .training import execute_training_attempt

__all__ = [
    'ALLOWED_MAPS', 'Action', 'CHECKPOINT_NAME', 'CHECKPOINT_SIDECAR_NAME', 'DEFAULT_CPU_CORES',
    'DEFAULT_GPU', 'DEFAULT_MEMORY_MIB', 'DEFAULT_NUM_ENVS', 'DEFAULT_SAVE_EVERY_SECONDS',
    'DEFAULT_TIMEOUT_MINUTES', 'DEFAULT_VEC_WORKERS', 'INPUTS_ROOT', 'Manifest', 'REDELIVERED',
    'REGISTRY_NAME', 'RESERVATION_FILENAME', 'RUNS_ROOT', 'RunRequest', 'SCHEMA_VERSION',
    'SOURCES_ROOT', 'STATUS_FILENAME', 'TERMINAL_STATUSES', 'VOLUME_NAME', 'ValidationError',
    '_THREAD_CAP_ENV', '_load_volume_json', 'build_run_request', 'claim_attempt',
    'create_source_bundle', 'derive_run_view_from_bytes', 'execute_training_attempt',
    'finish_reservation', 'mounted_path', 'parse_artifact_client_request', 'prepare_remote_source',
    'reserve_run', 'sha256_bytes', 'validate_clean_head', 'validate_local_checkpoint',
    'validate_run_id', 'verify_checkpoint'
]
