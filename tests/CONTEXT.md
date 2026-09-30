# tests/

`tests/` mirrors `src/cs2rl/`: a test file lives in the directory that mirrors the `src/cs2rl`
package it is about. `tests/integration/test_tests_layout.py` enforces the directory rules below.

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
`profile_step`). Those tests stay at the `tests/` root, and nothing else does but `conftest.py`.

## Layout

| directory | twin in `src/cs2rl/` | holds |
|-----------|----------------------|-------|
| `tests/` (root) | the flat modules | `conftest.py` and the flat modules' tests |
| `env/` | `env/` | tests of the env package |
| `env/c/` | `env/c/` | tests of the C env and its binding; `smoke_test.py` runs only if named |
| `train/` `eval/` `experiment/` `viz/` `deploy/` | the package of that name | its tests |

`tests/<a>` and `tests/<a>/<b>` each need `src/cs2rl/<a>/` or `src/cs2rl/<a>/<b>/` with an
`__init__.py`, unless `<a>` is one of the four below. A test of a package that has no directory yet
creates it (`spec/` has none today).

## The four directories with no twin (`NO_TWIN` in the guard)

- `_helpers/`: shared test code, never collected. It holds importable Python
  (`tests._helpers.<module>`); `test_helpers_hold_no_collectable_file` in
  `tests/integration/test_path_constants_exist.py` fails on a test file there.
- `fixtures/`: data files the tests read (JSON and text).
- `integration/`: repo-wide guards, pins and tooling tests; their subject is the checkout.
- `modal/`: the Modal runner's tests; the runner lives in `scripts/modal_runner/` (#274).

## The repo root

Use `from tests.conftest import REPO_ROOT`. Never build a root from `Path(__file__).parents[N]`: it
is right only at the depth it was written for, and a move breaks it, sometimes silently (a root
that only feeds a child process's cwd still passes). `tests/_helpers/metrics_census.py` keeps a
root of its own, which `test_path_constants_exist.py` compares with the conftest's.

## The guard

`tests/integration/test_tests_layout.py` checks four rules on the real tree:

- (a) a directory of `tests/` that holds a `.py` file has a twin package, unless it is in `NO_TWIN`;
- (b) every `NO_TWIN` entry is a directory of `tests/`;
- (c) a test file at the root imports a flat module of `cs2rl`;
- (d) no file but `conftest.py` and `_helpers/` builds a path from its own `__file__` with
  `.parent` or `.parents`. `os.path.dirname` is not read; the guard pins that limit.
