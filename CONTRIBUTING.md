# Contributing

Work in progress — covers the most common contributor workflows.

## Setup

```bash
git clone <repo> && cd cs2rl
uv sync
git config core.hooksPath .githooks
uv run --with 'ziglang>=0.14,<0.15' python setup.py build_ext --inplace --force
uv run python -m pytest tests/ -x -q
```

## Adding or changing an action head

The action head layout (names, sizes, ordering) is defined in **one place**: `src/c_env/cs2_types.h`. Everything else is derived from it.

### Where the spec lives

```c
// src/c_env/cs2_types.h
enum ActionHead {
    HEAD_MOVE = 0, HEAD_AIM = 1, HEAD_SHOOT = 2, HEAD_RELOAD = 3,
    HEAD_WEAPON = 4, HEAD_USE = 5, HEAD_CROUCH = 6, HEAD_JUMP = 7,
};
static const int  ACTION_HEAD_SIZES[] = {9, 16, 2, 2, 3, 2, 2, 2};
static const char* ACTION_HEAD_NAMES[] = {
    "move", "aim", "shoot", "reload", "weapon", "use", "crouch", "jump",
};
```

### Steps to add head N

1. **`cs2_types.h`**: Add enum value `HEAD_NEWHEAD = N`, append size to `ACTION_HEAD_SIZES[]`, append name to `ACTION_HEAD_NAMES[]`, bump `#define ACTION_DIM` and `#define ACTION_MASK_DIM`.
2. **C behaviour code**: Add the head's logic in the appropriate `.h` file. Use `actions[i * ACTION_DIM + HEAD_NEWHEAD]` — never a bare integer.
3. **C masking**: Add mask logic in `cs2_env.h` using `moff[HEAD_NEWHEAD]`.
4. **C stats**: Add `action_newhead[N]` to `StepStats` in `cs2_types.h`, add `count_action()` call.
5. **Run codegen**: `uv run python scripts/sync_action_spec.py` — this regenerates `src/_action_spec.py` from the C header.
6. **Python ctypes mirror**: Add the new `StepStats` field to `StepStatsC` in `cs2_env.py`.
7. **Deploy (if applicable)**: Add the head to `ActionExecutor.cs`.
8. **Rebuild + test**: `uv run --with 'ziglang>=0.14,<0.15' python setup.py build_ext --inplace --force && uv run python -m pytest tests/ -x -q`

You do **not** need to touch `nav.py`, `train.py`, `cs2_env.py` MultiDiscrete, or `_action_spec.py` manually — they all import from `_action_spec` which is regenerated in step 5.

### What NOT to do

- Never hardcode action indices as bare integers in C — use `HEAD_*` enum values
- Never define `ACTION_HEAD_SIZES` or `ACTION_HEAD_NAMES` in Python — import from `_action_spec`
- Never edit `src/_action_spec.py` by hand — it's generated

## Running experiments

See `docs/experiment-framework-guide.md` for the full experiment workflow.

Key rules:
- Never write to `outputs/experiments/baseline.txt` — only the user runs `scripts/promote_baseline.py`
- Never auto-merge `exp/*` branches to `main`
- One experiment at a time (global lock file)

## Code style

- Python: ruff + yapf (enforced by pre-commit hook)
- C: clang-format (enforced by pre-commit hook)
- Pre-commit hook runs automatically if you ran `git config core.hooksPath .githooks`

## Tests

```bash
uv run python -m pytest tests/ -x -q    # full suite (~60 tests, ~2 min)
uv run python src/train.py --smoke       # env sanity check
```
