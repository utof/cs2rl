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
