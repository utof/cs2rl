# Contributing

Work in progress — covers the most common contributor workflows.

## Setup

`README.md` has the path from a fresh clone to a green smoke run and a first test file. After
a C edit, rebuild the extension (`uv sync` does not):

```bash
uv run --with 'ziglang>=0.14,<0.15' python setup.py build_ext --inplace --force
```

In a git worktree, first symlink the main checkout's `.venv` into it (`ln -s <main>/.venv <worktree>/.venv`; `git worktree add` does not create one). Then do not use `uv run`: it syncs the environment first, which re-points the shared editable install at the worktree. Use the venv's interpreter with the worktree's `src/` first instead, e.g. `env UV_NO_SYNC=1 PYTHONPATH=<worktree>/src .venv/bin/python -m pytest tests -q`.

## Adding or changing an action head

The action head layout (names, sizes, ordering) is defined in **one place**: `src/cs2rl/env/c/cs2_types.h`. Everything else is derived from it.

### Where the spec lives

```c
// src/cs2rl/env/c/cs2_types.h
enum ActionHead {
    HEAD_MOVE = 0, HEAD_SHOOT = 1, HEAD_RELOAD = 2, HEAD_WEAPON = 3,
    HEAD_USE = 4, HEAD_CROUCH = 5, HEAD_JUMP = 6,
};
static const int  ACTION_HEAD_SIZES[] = {9, 2, 2, 3, 2, 2, 2};
static const char* ACTION_HEAD_NAMES[] = {
    "move", "shoot", "reload", "weapon", "use", "crouch", "jump",
};
```

Aim is not one of these discrete heads: it is a continuous head of `AIM_DIM` floats (Δyaw, pitch) in a separate buffer.

### Steps to add head N

1. **`cs2_types.h`**: Add enum value `HEAD_NEWHEAD = N`, append size to `ACTION_HEAD_SIZES[]`, append name to `ACTION_HEAD_NAMES[]`, bump `#define ACTION_DIM` and `#define ACTION_MASK_DIM`.
2. **C behaviour code**: Add the head's logic in the appropriate `.h` file. Use `actions[i * ACTION_DIM + HEAD_NEWHEAD]` — never a bare integer.
3. **C masking**: Add mask logic in `cs2_env.h` using `moff[HEAD_NEWHEAD]`.
4. **C stats**: Add `action_newhead[<size>]` to `StepStats` in `cs2_types.h`, sized by the head's number of choices (its `ACTION_HEAD_SIZES` entry, not its index N), and add a `count_action()` call.
5. **Run codegen**: `uv run python scripts/sync_action_spec.py` — this regenerates `src/cs2rl/spec/action.py` (and `src/cs2rl/spec/obs.py`) from the C header.
6. **Python ctypes mirror**: Add the new `StepStats` field to `StepStatsC` in `cs2_env.py`, and add an `action_newhead_{idx}` loop to the hand-listed per-head keys in `_build_terminal_info()` in the same file.
7. **Deploy (if applicable)**: Add the head to `ActionExecutor.cs`, and bump the hard-coded action-cache length `new int[7]` in `deploy/CS2RLBot/CS2RLBot.cs`.
8. **Rebuild + test**: `uv run --with 'ziglang>=0.14,<0.15' python setup.py build_ext --inplace --force`, then the full suite under Tests

You do **not** need to touch the policy (`cs2rl.policy`), `cs2_env.py`'s MultiDiscrete, or `src/cs2rl/spec/action.py` manually — the first two import `ACTION_HEAD_SIZES` from `cs2rl.spec.action`, which is regenerated in step 5.

### What NOT to do

- Never hardcode action indices as bare integers in C — use `HEAD_*` enum values
- Never define `ACTION_HEAD_SIZES` or `ACTION_HEAD_NAMES` in Python — import from `cs2rl.spec.action`
- Never edit `src/cs2rl/spec/action.py` by hand — it's generated

## Code style

- Python: ruff + yapf (enforced by pre-commit hook)
- C: clang-format (enforced by pre-commit hook)
- Pre-commit hook runs automatically if you ran `git config core.hooksPath .githooks`

## Tests

### Design tests around the behavior they protect

Before adding a test or changing its cost, name the production behavior it must
observe and a harmful change that must make it fail. A declaration, mirrored
trace, missing key or constructor shape alone may not prove the consumer uses
the value. Observe the actual effect; check membership before reading a
defaultdict value, and keep an opposite-direction positive control when a
predicate could pass unconditionally.

Use the existing real trainer harness for narrow controller tests. Choose the
smallest measured workload that preserves their intended failure sensitivity.
Record actual rollout, update and minibatch counts: a ratio test that must catch
division by the minibatch count needs more than one executed minibatch. Keep
mutable trainers/environments fresh and run their existing cleanup in finally
blocks. Do not shrink seed, checkpoint continuation, long-update or native
integration experiments merely because another test tolerates a smaller batch.

Before adding an adapter, fixture layer, configuration object or test runner,
the prototyper (or implementer for a small change) checks existing project code,
standard-library APIs and supported installed dependencies. Record the API,
version, official source, an executed success and relevant failure, and the
specific unmet requirement if custom code remains necessary. The implementer
preserves that evidence; the reviewer checks the riskiest reuse or equivalence
claim. API existence alone does not establish equivalent behavior.

Measure the affected selection before and after with the same interpreter,
options and observation method. Keep failing controls and noisy runs. Distinguish
summed testcase durations, external wall time, per-process maximum RSS and
simultaneous process-tree memory. Free-memory headroom is a launch prerequisite,
not measured peak usage. A marker changes scheduling; it does not reduce the
cost or preserve coverage unless the omitted checks run elsewhere.

For file/module/attribute refactors, inventory callers, import aliases,
entrypoints, configuration and persisted state before editing. Graph results
need coverage checks and exact source/AST fallback; a zero is not proof of
absence. Name interfaces for what they own, preserve old-format resume where
state is serialized, and verify the full combined diff. Reuse the measured
prototype on the same branch instead of rebuilding it from prose.

See the [training-test workload postmortem](docs/postmortem-2026-10-04-training-test-workloads.md)
and [architecture-refactor postmortem](docs/postmortem-2026-10-04-architecture-refactors.md).

`tests/CONTEXT.md` says where a new test goes (the directory that mirrors the `src/cs2rl`
package it tests; a file at the `tests/` root is only for a flat module such as `policy`), and
what the session guards in `tests/conftest.py` check.

```bash
uv run python -m pytest -n 2 --dist loadgroup tests -q        # the full suite, one session
uv run python -m pytest tests/env/test_env_config.py -q -n 0  # one test or file: no workers
uv run python -m cs2rl.train --smoke                          # env sanity check
```

The full suite is one session on 2 pytest-xdist workers (`-n 2`); it is the run that HEAD, verifiers and post-merge checks use.
- **Why 2.** Memory. Two workers took 2.9 GB more than a serial session on a 15 GB machine that also runs the owner's work, and four left 74 MB above the 2.5 GB kill line and needed a torch thread cap. `-n auto` is capped to 2 by `tests/conftest.py` when the args name `tests` or a path under it (`PYTEST_XDIST_AUTO_NUM_WORKERS` overrides it), so a mistyped `-n auto` does not start one worker per core. `-n`, `--dist` and `-m` are never in `addopts`: a full run must not depend on undoing them.
- **Why `--dist loadgroup`.** `tests/train/test_reward_loop_equivalence.py` carries `xdist_group("gpu")`, so that file's CUDA work runs in one worker and opens one context there instead of up to two (other files can still open CUDA in either worker). loadgroup appends `@gpu` to those test ids.
- **The guards.** The session-end guards in `tests/conftest.py` (one module object per file, the namespace guard) relay each worker's facts to the controller, so they hold under `-n`. Their `-n 2` tests need pytest-xdist in the environment: run `uv sync --all-groups --inexact` in the main checkout, or they fail with that remedy.
- **No separate sessions.** The four files that used to run each alone (`tests/env/test_arena_duel.py`, `tests/env/c/test_binding.py`, `tests/train/test_pitch_pin.py`, `tests/train/test_train_cli.py`) run inside the `-n 2` session. The recorded reason for "each alone" (cold vis-cache orphans, #251/#254) is closed, and the full session with them passed with MemAvailable never below 6.6 GB from a 10.1 GB start (about 3.5 GB used, no swap growth). Start a full run with at least 7 GB of MemAvailable, which leaves about 1 GB above a 2.5 GB floor. One hazard remains: a cold vis cache. Copy the `vis_cache*.npy` files from the main checkout's `src/cs2rl/` into a fresh worktree before any test run.
- **Fast loop.** `-m "not slow"` (with `-n 2`) is for intermediate per-commit checks only. The `slow` marker (over about 15 s of wall) drops real coverage: the patch-binding campaign, the seed positive control and the fast-math builds. HEAD, verifiers and post-merge run the full suite.

A full run takes several minutes: measured on one development machine on 2026-09-30, about 5.5 min for the full suite at `-n 2` (one run, 1782 tests), against about 8-9 min serial (main session plus the four former each-alone files).
