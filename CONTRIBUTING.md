# Contributing

Work in progress — covers the most common contributor workflows.

## Setup

```bash
git clone <repo> && cd cs2rl
uv sync
git config core.hooksPath .githooks
uv run --with 'ziglang>=0.14,<0.15' python setup.py build_ext --inplace --force
uv run python -m pytest tests/ -x -q    # whole suite in one session; see Tests for the split run
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
8. **Rebuild + test**: `uv run --with 'ziglang>=0.14,<0.15' python setup.py build_ext --inplace --force && uv run python -m pytest tests/ -x -q` (or the split run under Tests)

You do **not** need to touch `train.py`, `cs2_env.py` MultiDiscrete, or `src/cs2rl/spec/action.py` manually — the first two import from `cs2rl.spec.action`, which is regenerated in step 5.

### What NOT to do

- Never hardcode action indices as bare integers in C — use `HEAD_*` enum values
- Never define `ACTION_HEAD_SIZES` or `ACTION_HEAD_NAMES` in Python — import from `cs2rl.spec.action`
- Never edit `src/cs2rl/spec/action.py` by hand — it's generated

## Code style

- Python: ruff + yapf (enforced by pre-commit hook)
- C: clang-format (enforced by pre-commit hook)
- Pre-commit hook runs automatically if you ran `git config core.hooksPath .githooks`

## Tests

```bash
uv run python -m pytest tests/ -x -q                    # full suite in one session (split run below)
uv run python -m pytest tests/ -x -q -m "not slow"      # skip the multi-minute subprocess and rollout tests
uv run python -m cs2rl.train --smoke                    # env sanity check
```

Four test files are heavy and are best run on their own, each in its own pytest process, rather than in the same session as the rest: `tests/test_arena_duel.py`, `tests/test_binding.py`, `tests/test_pitch_pin.py`, `tests/test_train_cli.py`.

```bash
uv run python -m pytest tests/ -q --ignore=tests/test_arena_duel.py --ignore=tests/test_binding.py \
    --ignore=tests/test_pitch_pin.py --ignore=tests/test_train_cli.py
uv run python -m pytest tests/test_arena_duel.py -q     # then the other three, one at a time
```

A full run takes several minutes: measured on one development machine on 2026-09-29, about 7 min for the main session plus about 1 min for the four heavy files.
