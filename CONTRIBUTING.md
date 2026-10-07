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

For a trainer change that claims zero behaviour change, compare seeded CPU trainers
before and after with `scripts/trainer_equivalence.py`, from the base checkout and then
from HEAD, with the same thread settings on both sides (in a worktree, use the venv
interpreter form from the setup section instead of `uv run`):

```bash
env CUDA_VISIBLE_DEVICES= uv run python scripts/trainer_equivalence.py run --out base.json
env CUDA_VISIBLE_DEVICES= uv run python scripts/trainer_equivalence.py run --out head.json
env CUDA_VISIBLE_DEVICES= uv run python scripts/trainer_equivalence.py compare base.json head.json
```

`compare` exits 1 on any differing component; `seedctl CASE` is the positive control.
The tool builds through `cs2rl.train.compose.build_trainer` the way the test harness
does, so it does not see what only `train()` adds (weight decay, the aim-σ group, W&B,
metrics rows, the eval hook): compare a short CPU CLI run for that. An intended numeric
change moves the digests; it is a review instrument, not a test.

See the [training-test workload postmortem](docs/postmortem-2026-10-04-training-test-workloads.md)
and [architecture-refactor postmortem](docs/postmortem-2026-10-04-architecture-refactors.md).

`tests/CONTEXT.md` says where a new test goes (the directory that mirrors the `src/cs2rl`
package it tests; a file at the `tests/` root is only for a flat module such as `policy`), and
what the session guards in `tests/conftest.py` check.

```bash
uv run python -m pytest -n 2 --dist loadgroup tests -q        # fast default
uv run python -m pytest -n 2 --dist loadgroup tests -m training -q
uv run python -m pytest -n 2 --dist loadgroup tests -m "slow and not training" -q
uv run python -m pytest -n 2 --dist loadgroup tests -m "" -q  # complete, one session
uv run python -m pytest tests/env/test_env_config.py -q -n 0  # one test or file: no workers
uv run python -m pytest tests/train/test_resume_state.py -m "" -q -n 0
uv run python -m cs2rl.train --smoke                          # env sanity check
```

The default expression in pytest's native configuration is `not slow and not training`.
Ordinary domain, configuration and light numerical tests stay fast, including cheap
trainer construction, return-statistics, seed/RNG and checkpoint-unit checks. Real
trainer rollouts, learning loops and checkpoint continuation carry `training`;
expensive non-training campaigns, native builds and fresh-process checks use `slow`.
Some training tests retain `slow` as well: `-m training` selects them together.

`-m "not slow"` excludes only `slow` tests; it still includes training tests without
that marker. Use the default `not slow and not training` selection for ordinary
validation.

At this combined selection and source-reuse change, collection is 1949 fast,
57 training and 65 extended non-training cases; the complete command collects
2071 (2063 original cases plus seven selection regressions and one parser-freshness
regression). The tiers are disjoint and their union is complete.
The fast and complete `tests` invocations both retain the expected performance-smoke
opt-in skip; naming its file explicitly still opts in. Counts describe this revision
and will change as tests are added. Deselection omits real coverage; it is not a skip
or a cheaper workload for an omitted test.

Use fast validation for ordinary edits and routine HEAD/verifier/post-merge checks.
Add the training tier when trainer, learning or checkpoint-continuation behavior
changes, and the extended tier when the affected native-build, child campaign or
process-isolation behavior changes. Releases and substantial training changes use
the complete command. A single file/node still inherits the default expression:
use `-m "" -n 0` to run every intended case, including training. pytest's last CLI
`-m` overrides the configured expression; nested repo-configured child sessions must
clear it explicitly when their intended consumer belongs to another selection.

Broad selections use 2 pytest-xdist workers (`-n 2`), with the complete suite in one session.
- **Why 2.** Memory. Two workers took 2.9 GB more than a serial session on a 15 GB machine that also runs the owner's work, and four left 74 MB above the 2.5 GB kill line and needed a torch thread cap. `-n auto` is capped to 2 by `tests/conftest.py` when the args name `tests` or a path under it (`PYTEST_XDIST_AUTO_NUM_WORKERS` overrides it), so a mistyped `-n auto` does not start one worker per core. `-n` and `--dist` stay out of `addopts`; the default `-m` selection is cleared by the documented complete command.
- **Why `--dist loadgroup`.** `tests/train/test_reward_loop_equivalence.py` carries `xdist_group("gpu")`, so that file's CUDA work runs in one worker and opens one context there instead of up to two (other files can still open CUDA in either worker). loadgroup appends `@gpu` to those test ids.
- **The guards.** The session-end guards in `tests/conftest.py` (one module object per file, the namespace guard) relay each worker's facts to the controller, so they hold under `-n`. Their `-n 2` tests need pytest-xdist in the environment: run `uv sync --all-groups --inexact` in the main checkout, or they fail with that remedy.
- **One complete session.** The four files that used to run each alone (`tests/env/test_arena_duel.py`, `tests/env/c/test_binding.py`, `tests/train/test_pitch_pin.py`, `tests/train/test_train_cli.py`) run inside the complete `-n 2` session. The recorded reason for "each alone" (cold vis-cache orphans, #251/#254) is closed. The historical 7 GiB available-memory cutoff was a conservative launch estimate, not measured suite usage. Later native accounting measured 3.801693 GiB charged memory peak and a separate 0.684212 GiB swap peak in one full run; these are not a reusable launch requirement or total host/GPU usage. One hazard remains: a cold vis cache. Copy the `vis_cache*.npy` files from the main checkout's `src/cs2rl/` into a fresh worktree before any test run.

The complete validation run for PR #346 reported 2062 passed and one expected
skip in 735.94 s. The unmatched slowdown investigation is owner-deferred in
[#347](https://github.com/utof/cs2rl/issues/347); selection alone establishes no controlled
whole-suite speedup. Preserve workloads and timeouts when choosing a tier.
