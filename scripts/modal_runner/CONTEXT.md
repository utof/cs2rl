# scripts/modal_runner/

The library of the optional Modal cloud-training runner: request validation, source bundles,
checkpoints, run state, and one training attempt inside the container. It imports only the
standard library and its sibling modules at module scope, never Modal, so run status and
downloads work without importing the Modal App. Torch is imported inside two functions of
`checkpoint`.

## The pieces

- `scripts/run_modal.py`, the Modal App that launches a run:
  `modal run --detach scripts/run_modal.py --action run ...`. Importing it constructs the CUDA
  image and registers the App.
- `scripts/modal_artifacts.py` (status and download) and `scripts/modal_backfill_sidecar.py`
  (writes a missing checkpoint sidecar) read the runner without importing that App.
- This package: its `__init__.py` docstring has the MODULE MAP (`core`, `request`, `source`,
  `checkpoint`, `state`, `commands`, `preflight`, `training`) and the rules for new names.
- The Modal SDK is in the non-default `modal` dependency group: `uv sync --group modal`.

The runner stays under `scripts/` by the owner's deliberate layout decision, recorded in
`pyproject.toml`'s banned-api table. It is baked into the runner image and shipped inside the
source archive the container extracts. Training uses the installed `cs2rl` wheel; inherited
`PYTHONPATH` is excluded from the install, probe and training child environment. The archived
`scripts/` remains importable through the extracted source-root working directory. The runner
image keeps its own `PYTHONPATH` for its baked modules.

## Where new code goes

- A new name: the one module that reads it; `core` only for a name two or more modules read,
  plus the Volume and run-directory layout (WHERE A NEW NAME GOES in `__init__.py`).
- A new module: the checklist in `tests/modal/modal_runner_tables.py`'s docstring, which every
  package gate reads the module list from.
- A new train flag the runner must pass: `LIVE_TRAIN_OPTION_ARITY` in `request.py`, a
  hand-kept mirror of `python -m cs2rl.train`'s long options.
- Tests: `tests/modal/test_modal_<module>.py` for the module they test. THE PLACEMENT RULE FOR
  RUNNER TESTS in `tests/modal/test_modal_packaging.py` says what a test there must reach; each
  test also needs its line in `tests/fixtures/modal_test_seam_manifest.json` and
  `GOVERNED_NAME_COUNT` +1 in `tests/modal/test_modal_packaging.py`.

## Traps

- Patch a name where the code under test reads it at call time. The client scripts read the
  facade (`scripts.modal_runner.X`); package modules hold from-imported copies, except the
  QUALIFIED SEAMS listed in `__init__.py`. PATCHING IN TESTS there shows both cases.
- The runner accepts only exact long options (`--device`), never the prefixes argparse accepts
  (`--devi`).
- Client Volume paths are root-relative (`runs/...`); the container sees the same object at
  `/artifacts/runs/...`.
- Launch with `spawn()`, never `.remote()`: a synchronous input is cancelled when the local
  client dies, even under `modal run --detach` (`scripts/run_modal.py`'s PITFALLS).
- A module that spawns or signals a process goes through `training.ProcessControl`;
  `test_kill_seam_static_safety` fails on a raw `os.killpg`, `os.kill` or `subprocess.Popen`
  elsewhere in the package.
- Every run costs money. The client tests run against a fake `modal` module (`fake_modal` in
  `tests/modal/test_modal_client.py`).
