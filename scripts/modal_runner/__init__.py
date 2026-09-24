"""Modal-free request, path, source, checkpoint and run-state library for the optional runner.

WHY this package exists separately from scripts/run_modal.py: status/download
and all validation must work without importing or hydrating a Modal App.
Importing the App would build the CUDA image. Every module in this package
therefore imports only the standard library and its own sibling modules at
module scope. Torch is imported lazily, inside `checkpoint._import_torch` and
`checkpoint._load_checkpoint_weights`, and Modal is never imported here.

MODULE MAP (each module's exact symbols are its MANIFEST entry in
tests/modal_runner_tables.py):
  core        the Volume and run-directory layout, the Status enum,
              ValidationError, the path and hash helpers, and the protocols
              and records two or more modules share (see WHERE A NEW NAME GOES)
  request     run and artifact-client requests with their limits and defaults;
              train.py argument validation
  source      clean-HEAD check, deterministic source bundles, safe extraction
  checkpoint  checkpoint loading and validation; completed-run evidence
  state       run status and transitions, reservations, claims, heartbeats,
              derived run views
  commands    install/train/dump-config/CUDA-probe argv and child environments
  preflight   preparing the remote source before a training attempt
  training    one training attempt: child process, checkpoint publication,
              the attempt's result records and reason tokens
  __init__    this facade: its module scope holds only this docstring,
              `from .<module> import ...` lines and `__all__`

WHERE A NEW NAME GOES: put a new name in the one module that reads it; add to
core only a name that two or more modules read. The one exception is the
Volume and run-directory layout (mount, roots and artifact file names), which
core keeps whole even where one module reads a given name, so the layout reads
as one contract. A name this facade re-exports goes in the module that reads
it, like any other: production reads the facade, not the module. A name no
package module reads goes in the module whose concern it is (a layout name in
core). A new cross-module import needs a DEPENDENCIES entry in
tests/modal_runner_tables.py.

ADDING A MODULE: follow the checklist in the docstring of
tests/modal_runner_tables.py, which every gate reads the module list from.

THIS FACADE IS DELIBERATELY NARROW. It re-exports exactly the names that
production code reads from the package, and nothing more.
`_production_package_surface` in tests/test_modal_client.py derives that set
from every tracked scripts/*.py outside this package, and the surface gates pin
`__all__` and the module's attributes to it. Never widen the facade to satisfy
a test; import the owning submodule instead.

PATCHING IN TESTS: patch a name in the namespace the code under test reads it
from at call time.
  * scripts/run_modal.py, scripts/modal_artifacts.py and
    scripts/modal_backfill_sidecar.py import this facade as `mrl` and read
    `mrl.X` at call time, so a test of them patches the facade attribute
    (`scripts.modal_runner.X`). Patching the owning submodule does not change
    what `mrl.X` returns.
  * Code inside the package never reads the facade: its names are
    from-imported copies, so a facade patch never reaches a package caller. A
    caller in the module that defines the name, or one reading a QUALIFIED
    SEAM below, sees a patch on the owning submodule
    (`scripts.modal_runner.state.X`). A caller that from-imports the name from
    a sibling reads its own copy, so patch the importing module instead. For a
    site in BINDING_SITES (tests/modal_patch_binding_campaign.py), install the
    patch through `binding_target(site)`, which names the module to patch.
  `test_patch_target_dichotomy` (tests/test_modal_patch_bindings.py) shows both
  cases on `validate_local_checkpoint`.

QUALIFIED SEAMS. These names, and only these, are read across the submodules
through the module object rather than a from-import: `core.PREBUILT_PYTHON`,
`core.sha256_file`, `state.transition_status` and
`checkpoint.validate_local_checkpoint`. Tests patch them on their owning
module, and the attribute read at call time is what lets that patch reach
callers in other modules. Rewriting one as `from .core import sha256_file`,
reading this facade's copy (`from . import validate_local_checkpoint`), or
binding it at import time (`_HASH = core.sha256_file`, a default argument)
silently detaches it from the patch. Every other cross-module name is
from-imported, so patching it on its owning module does NOT reach the
importing module (training.py and preflight.py hold both `state` and
from-imported `state` names). This facade's own re-export of
`validate_local_checkpoint` is such a copy by design (see PATCHING IN TESTS).
QUALIFIED_SEAMS in tests/modal_runner_tables.py lists them with their readers;
the package-shape `seam` clause enforces it over the submodules, and
`test_package_seam_contract` checks that this list matches it.

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
from .commands import THREAD_CAP_ENV
from .core import (
    CHECKPOINT_NAME,
    CHECKPOINT_SIDECAR_NAME,
    INPUTS_ROOT,
    REGISTRY_NAME,
    RESERVATION_FILENAME,
    RUNS_ROOT,
    SCHEMA_VERSION,
    SOURCES_ROOT,
    STATUS_FILENAME,
    VOLUME_NAME,
    AttemptContext,
    Manifest,
    ValidationError,
    mounted_path,
    sha256_bytes,
)
from .preflight import ExpectedSource, RemoteResume, prepare_remote_source
from .request import (
    ALLOWED_MAPS,
    DEFAULT_CPU_CORES,
    DEFAULT_GPU,
    DEFAULT_MEMORY_MIB,
    DEFAULT_NUM_ENVS,
    DEFAULT_SAVE_EVERY_SECONDS,
    DEFAULT_TIMEOUT_MINUTES,
    DEFAULT_VEC_WORKERS,
    Action,
    RunRequest,
    build_run_request,
    parse_artifact_client_request,
    validate_run_id,
)
from .source import create_source_bundle, validate_clean_head
from .state import (
    REDELIVERED,
    TERMINAL_STATUSES,
    claim_attempt,
    derive_run_view_from_bytes,
    finish_reservation,
    load_volume_json,
    reserve_run,
)
from .training import execute_training_attempt

__all__ = [
    'ALLOWED_MAPS', 'Action', 'AttemptContext', 'CHECKPOINT_NAME', 'CHECKPOINT_SIDECAR_NAME',
    'DEFAULT_CPU_CORES', 'DEFAULT_GPU', 'DEFAULT_MEMORY_MIB', 'DEFAULT_NUM_ENVS',
    'DEFAULT_SAVE_EVERY_SECONDS', 'DEFAULT_TIMEOUT_MINUTES', 'DEFAULT_VEC_WORKERS',
    'ExpectedSource', 'INPUTS_ROOT', 'Manifest', 'REDELIVERED', 'REGISTRY_NAME',
    'RESERVATION_FILENAME', 'RUNS_ROOT', 'RemoteResume', 'RunRequest', 'SCHEMA_VERSION',
    'SOURCES_ROOT', 'STATUS_FILENAME', 'TERMINAL_STATUSES', 'THREAD_CAP_ENV', 'VOLUME_NAME',
    'ValidationError', 'build_run_request', 'claim_attempt', 'create_source_bundle',
    'derive_run_view_from_bytes', 'execute_training_attempt', 'finish_reservation',
    'load_volume_json', 'mounted_path', 'parse_artifact_client_request', 'prepare_remote_source',
    'reserve_run', 'sha256_bytes', 'validate_clean_head', 'validate_local_checkpoint',
    'validate_run_id', 'verify_checkpoint'
]
