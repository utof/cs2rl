# scripts/

Command-line tools, run by path from the repository root: `uv run python scripts/<name>.py`.
Library code lives in `src/cs2rl/`: no module under `src/` imports a script. #204 moved the
libraries that were here into the package (the experiment gates into `cs2rl.experiment`, the
demo generator into `cs2rl.bc_demos`), and `pyproject.toml`'s banned-api table rejects their
old import names. Tests may import a script as `scripts.<name>`.

The one exception is `scripts/modal_runner/`, the Modal runner's library. It stays here by the
owner's decision, recorded in `pyproject.toml`'s banned-api table: it is baked into the runner
image and shipped inside the source archive the container extracts. It has its own
`CONTEXT.md`.

## What is here

| script | does |
|--------|------|
| `sync_action_spec.py` | regenerates `src/cs2rl/spec/action.py` and `src/cs2rl/spec/obs.py` from `cs2_types.h` |
| `bake_nav.py` | writes `nav_data.h` for the raylib demo, from `make_simple_map()` |
| `sim_fingerprint.py` | hashes a seeded rollout, to prove a sim refactor bit-identical to its parent commit |
| `trainer_equivalence.py` | fingerprints seeded CPU trainer epochs (`run`/`compare`/`seedctl`), to prove a trainer refactor bit-identical to its base |
| `pyrefly_gate.py` | the type gate the pre-commit hook runs; `--update` refreshes `pyrefly-snapshot.json` |
| `run_rung1.sh` | the Rung 1 sweep: seeds, retries with `--resume-run`, the files `cs2rl.experiment.gate` reads |
| `launch_server.sh` | starts a local CS2 server with 2 RL bots per side (`deploy/serversetup.md`) |
| `run_modal.py`, `modal_artifacts.py`, `modal_backfill_sidecar.py` | the Modal launcher, status client and sidecar backfill (`scripts/modal_runner/CONTEXT.md`) |

## Where new code goes

A new command line is a script here; code that a module under `src/` needs goes in
`src/cs2rl/`. Its tests follow `tests/CONTEXT.md`: `tests/modal/` for the Modal scripts,
`tests/integration/` for repo tooling such as the pyrefly gate, otherwise the package the
script drives (`tests/experiment/test_run_rung1_sh.py` tests `run_rung1.sh`).

## Traps

- `scripts/` has no `__init__.py`: it is a namespace package. Tests import `scripts.*` through
  `pythonpath = ["."]` in `pyproject.toml`, and `tests/conftest.py`'s namespace guard stops a
  session that could load another checkout's `scripts/`.
- A script that imports `cs2rl`, run by path from a worktree without
  `env PYTHONPATH=<worktree>/src`, fails: `cs2rl`'s import guard sees the script in one
  checkout and the package in another, and prints both remedies.
- `sync_action_spec.py` overwrites the two spec modules. Run it after a layout edit in
  `cs2_types.h`, and never edit the generated files by hand.
