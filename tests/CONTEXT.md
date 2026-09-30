# tests/

`tests/` mirrors `src/cs2rl/`: a test file lives in the directory that mirrors the `src/cs2rl`
package it is about. `tests/integration/test_tests_layout.py` checks which directories may exist
("The guard" below); which of them a new test goes to is convention.

## Where a new test goes

Find the file's subject, the `src/cs2rl` package it is about, then take the first rule that applies:

1. Named homes. `conftest.py` stays at the root. A test of the Modal runner and its scripts
   (`scripts/modal_*`, `scripts/run_modal.py`) goes to `modal/`. A repo-wide guard, dependency pin
   or tooling test goes to `integration/`.
2. Single. The file uses exactly one `cs2rl` package: that package's directory.
3. Agree. The package it imports from the highest layer of `pyproject.toml`'s `cs2rl layers`
   contract, the package it uses most and the package most of its test functions touch are one
   package: that one.
4. Test functions. Otherwise the package that most of its test functions touch. A tie goes to
   the higher layer, and `spec` never beats another package.
5. Override. If rule 4 names a package the file is not about, decide by hand and say why in the
   review. `experiment/test_run_rung1_sh.py` tests `scripts/run_rung1.sh`, which has no twin, so it
   sits with the Rung 1 gate code in `experiment/`.

Rules 2 to 4 can also name a flat module of `src/cs2rl/` (`policy`, `train_bc`, `bc_demos`,
`profile_step`). Those tests stay at the `tests/` root, and no other `.py` file does but
`conftest.py`.

## Layout

| directory | twin in `src/cs2rl/` | holds |
|-----------|----------------------|-------|
| `tests/` (root) | the flat modules | `conftest.py` and the flat modules' tests |
| `env/` | `env/` | tests of the env package |
| `env/c/` | `env/c/` | tests of the C env and its binding |
| `train/` `eval/` `experiment/` `viz/` `deploy/` | the package of that name | its tests |

`env/c/smoke_test.py` holds a performance test that runs only when the file is named on the
command line.

Every directory under `tests/` that pytest walks into needs its twin, whatever it holds:
`tests/<a>/<b>` needs `src/cs2rl/<a>/<b>/__init__.py`, unless `<a>` is one of the four below. A test
of a package that has no directory yet creates it (`spec/` has none today).

## The four directories with no twin (`NO_TWIN` in the guard)

- `_helpers/`: shared test code, never collected. It holds importable Python
  (`tests._helpers.<module>`); `test_helpers_hold_no_collectable_file` in
  `tests/integration/test_path_constants_exist.py` fails on a test file there. A new non-test
  module goes here; the Modal tests' own helpers sit beside them (`modal/modal_*.py`).
- `fixtures/`: data files the tests read (JSON and text).
- `integration/`: repo-wide guards, pins and tooling tests; their subject is the checkout.
- `modal/`: the Modal runner's tests; the runner lives in `scripts/modal_runner/` (#274).

## The repo root

Use `from tests.conftest import REPO_ROOT`. Never build a root from `Path(__file__).parents[N]`: it
is right only at the depth it was written for, and a move breaks it, sometimes silently (a root
that only feeds a child process's cwd still passes). `tests/_helpers/metrics_census.py` keeps a
root of its own, which `test_path_constants_exist.py` compares with the conftest's.

## The guard

`tests/integration/test_tests_layout.py` walks the real `tests/` on disk and checks four rules:

- (a) every directory it walks into has a twin package, whatever it holds, unless its top
  directory is in `NO_TWIN`;
- (b) every `NO_TWIN` entry is a directory of `tests/`;
- (c) a test file at the root imports a flat module of `cs2rl`;
- (d) no file but the root `conftest.py` and `_helpers/` builds a path from its own `__file__`
  with `.parent` or `.parents`.

Rules 1 to 5 above are convention: the guard checks that a directory has a twin, not that a test
file sits in the right one, apart from (c) at the root. Where a non-test module goes is convention
too. The guard's known limits, each pinned as a case that passes today except the last:

- (c) does not ask where a root test belongs: a package's test that also imports a flat module
  passes at the root.
- No rule places a non-test module: a helper at the root or in a mirror directory passes.
- (d) sees neither `os.path.dirname` nor a root built in two steps
  (`HERE = Path(__file__).resolve()`, then `HERE.parents[2]`).
- The walk skips a directory whose name is not a Python identifier, such as `tests/env-c/`, which
  pytest collects from; its test-file patterns are pytest's defaults, copied by hand.

## Running the tests

Run pytest from the repository root and pass `tests` or files under it (`CONTRIBUTING.md`,
"Tests"). A process that imports pufferlib plants a `resources` symlink in its working
directory; `.gitignore` hides it, so after a run from `tests/` or another subdirectory, delete
the one left there. In a worktree, put its own `src/` first:
`env PYTHONPATH=<worktree>/src UV_NO_SYNC=1 .venv/bin/python -m pytest ...`.

## The session guards (`tests/conftest.py`)

Three guards stop a session that would test the wrong code. `--noconftest` switches all three
off, because they are hooks in `tests/conftest.py`.

- The checkout tripwire (#199), before collection: (a) `cs2rl` resolves to this checkout's
  `src/`; (b) no checkout's `src/` on `sys.path` holds another importable name (a leftover
  module, a stale `.so`, pufferlib's `resources` symlink); (e), added by #205, no directory
  under `src/cs2rl/` imports as a namespace package, and no file there is shadowed by a
  same-named package, extension or source file. Limits: (b) reads only the direct children of
  each `src/`; (e) is a snapshot at session start, and a leftover file with no same-named
  competitor imports silently (import-linter's tests in `tests/integration/test_import_layers.py`
  catch that one).
- One module object per file (#201), at session end: no repo file is loaded under two module
  names. Limits: it reads `sys.modules` at the end, so a copy evicted before then, or one never
  registered, is invisible; imports in a child process are invisible; ruff's banned-api table,
  its static half, runs only on staged files at commit time.
- The namespace guard (#207): (c) before collection, no `sys.path` entry but the root holds a
  `tests` or `scripts` directory or module; (d) at session end, every `tests.*` and `scripts.*`
  module loaded from its own place. Limits: (d) sees only modules still loaded at the end; (c)
  reads `sys.path` once; a child process is invisible to both.

The session-end checks do not run under `-x` after a failure or after a collection error; that
session fails anyway. Under `-n 2` each worker ships its facts to the controller, which judges
them.
