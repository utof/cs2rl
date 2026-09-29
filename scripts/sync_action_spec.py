#!/usr/bin/env python3
"""Sync Python action + obs specs from cs2_types.h (the single source of truth).

Emits TWO generated modules, both derived from the C header so a dimension
change is a one-file (cs2_types.h) edit that propagates to Python:
  - src/cs2rl/spec/action.py : action head spec (discrete + continuous / AIM_DIM).
  - src/cs2rl/spec/obs.py     : OBS_DIM + OBS_BLOCKS (named per-block start:stop
                           slices) so masking / demo-zeroing code never
                           hardcodes 25 / 53 / 93.

Batch 3: action side emits separate discrete + continuous spec lists.
Backwards-compat aliases (ACTION_HEAD_NAMES, ACTION_HEAD_SIZES, ACTION_DIM,
ACTION_MASK_DIM) stay; they refer to the discrete-only side. New: AIM_DIM,
CONTINUOUS_HEAD_NAMES, DISCRETE_HEAD_SPEC, CONTINUOUS_HEAD_SPEC.

Run after changing ACTION_HEAD_SIZES/ACTION_HEAD_NAMES/AIM_DIM, or any OBS_*
macro (block sizes / OBS_DIM) in cs2_types.h:
    uv run python scripts/sync_action_spec.py

HEADER comes from the C package (cs2rl.c_env.SOURCE_DIR), so this script imports
cs2rl. A restated path went stale silently when the package moved: the generator
would read a missing header, and no test runs it. PITFALL: run by path from a
worktree without `PYTHONPATH=<worktree>/src`, that import resolves to the shared
venv's checkout (main's), and cs2rl's own guard raises its foreign-checkout
ImportError naming both checkouts. That is correct: without the guard, the script
would read main's header and write the worktree's spec modules. Put the worktree's
src/ first: `env PYTHONPATH=<worktree>/src .venv/bin/python scripts/sync_action_spec.py`.
"""
import re
from pathlib import Path

from cs2rl.env.c import SOURCE_DIR

HEADER = SOURCE_DIR / "cs2_types.h"
OUTPUT = Path(__file__).resolve().parent.parent / "src" / "cs2rl" / "spec" / "action.py"
OBS_OUTPUT = Path(__file__).resolve().parent.parent / "src" / "cs2rl" / "spec" / "obs.py"


def main() -> None:
    """Regenerate spec/action.py and spec/obs.py from cs2_types.h.

    Every statement with a side effect (the header read, both writes, the prints)
    lives here, so loading the module binds only HEADER, OUTPUT and OBS_OUTPUT:
    tests/test_path_constants_exist.py reads the two outputs with runpy.run_path
    under a run_name other than __main__, which writes nothing.
    """
    text = HEADER.read_text()

    # Extract ACTION_HEAD_SIZES[] = {9, 2, ...}
    sizes_match = re.search(r"ACTION_HEAD_SIZES\[\]\s*=\s*\{([^}]+)\}", text)
    if not sizes_match:
        raise RuntimeError("Could not find ACTION_HEAD_SIZES[] in cs2_types.h")
    sizes = tuple(int(x.strip()) for x in sizes_match.group(1).split(",") if x.strip())

    # Extract ACTION_HEAD_NAMES[] = {"move", ...}
    names_match = re.search(r'ACTION_HEAD_NAMES\[\]\s*=\s*\{([^}]+)\}', text)
    if not names_match:
        raise RuntimeError("Could not find ACTION_HEAD_NAMES[] in cs2_types.h")
    names = tuple(m.group(1) for m in re.finditer(r'"([^"]+)"', names_match.group(1)))

    # Batch 3: extract AIM_DIM (continuous-aim Δyaw vector dim).
    # Tolerant of clang-format line-wrapping the #define onto multiple lines.
    aim_dim_match = re.search(r"#define\s+AIM_DIM\s*\\?\s*\n?\s*(\d+)", text)
    if not aim_dim_match:
        raise RuntimeError("Could not find #define AIM_DIM in cs2_types.h")
    aim_dim = int(aim_dim_match.group(1))

    if len(sizes) != len(names):
        raise RuntimeError(f"Mismatch: {len(sizes)} sizes vs {len(names)} names")

    action_dim = len(sizes)
    action_mask_dim = sum(sizes)

    # Batch 3: continuous heads. Currently 1 head ("aim"), gaussian, 2D [Δyaw, pitch].
    # Spec format: tuple of (name, distribution_kind, dim) — same shape as
    # DISCRETE_HEAD_SPEC for consumer symmetry. dim is the parameterised
    # dimensionality (Gaussian mean/log_std vector size); the action
    # buffer is dim=AIM_DIM=2 (the sampled Δyaw and absolute pitch).
    continuous_head_names = ("aim", )
    continuous_head_spec = (("aim", "gaussian", aim_dim), )
    discrete_head_spec = tuple(
        (name, "categorical", size) for name, size in zip(names, sizes, strict=True))

    # Pretty-print DISCRETE_HEAD_SPEC across multiple lines so the generated file
    # stays under 100 cols (Ruff E501) — repr() inlines the whole 7-tuple of
    # 3-tuples on one line otherwise.
    discrete_lines = ",\n    ".join(repr(t) for t in discrete_head_spec)

    OUTPUT.write_text(f"# Auto-generated from cs2_types.h — do not edit manually.\n"
                      f"# Regenerate: uv run python scripts/sync_action_spec.py\n"
                      f"\n"
                      f"# Backwards-compat aliases (discrete side only — Batch 3+).\n"
                      f"# Pre-Batch-3 callers reference ACTION_HEAD_SIZES symbolically; keep\n"
                      f"# this list discrete-only so MultiDiscrete spaces still build correctly.\n"
                      f"ACTION_HEAD_NAMES = {names!r}\n"
                      f"ACTION_HEAD_SIZES = {sizes!r}\n"
                      f"ACTION_DIM = {action_dim}\n"
                      f"ACTION_MASK_DIM = {action_mask_dim}\n"
                      f"\n"
                      f"# Batch 3 split: discrete vs continuous heads.\n"
                      f"# DISCRETE_HEAD_SPEC: tuple of (name, 'categorical', n_categories).\n"
                      f"# CONTINUOUS_HEAD_SPEC: tuple of (name, 'gaussian', dim).\n"
                      f"# AIM_DIM is the float-buffer width per agent for the continuous side.\n"
                      f"DISCRETE_HEAD_SPEC = (\n    {discrete_lines},\n)\n"
                      f"CONTINUOUS_HEAD_SPEC = {continuous_head_spec!r}\n"
                      f"CONTINUOUS_HEAD_NAMES = {continuous_head_names!r}\n"
                      f"AIM_DIM = {aim_dim}\n")
    print(f"[sync_action_spec] Wrote {OUTPUT}")
    print(f"  ACTION_DIM={action_dim}  ACTION_MASK_DIM={action_mask_dim}  AIM_DIM={aim_dim}")
    print(f"  NAMES={names}")
    print(f"  SIZES={sizes}")

    # ── Obs block layout → src/cs2rl/spec/obs.py ─────────────────────────────────
    # Parse the PRIMITIVE OBS_* literals from cs2_types.h and recompute the block
    # bases exactly as the C macros derive them (base = prev_base + prev_width).
    # We deliberately do NOT try to eval the C base expressions — we mirror the
    # arithmetic here so the Python table is a self-checking cross-witness: if the
    # recomputed last-block end != the parsed OBS_DIM literal, the two have drifted
    # and we raise (same contract as the env_init tiling assert in cs2_env.h).

    def _obs_int(macro: str) -> int:
        """Read `#define <macro> <int-literal>` from cs2_types.h.

        Only matches bare integer literals — the derived base macros
        (OBS_TEAMMATE_BASE = (…)) are intentionally NOT parsed here; we recompute
        those from the primitive sizes/strides/counts below.
        """
        m = re.search(rf"#define\s+{macro}\s+(-?\d+)\b", text)
        if not m:
            raise RuntimeError(f"Could not find primitive '#define {macro} <int>' in cs2_types.h")
        return int(m.group(1))

    obs_dim = _obs_int("OBS_DIM")
    self_size = _obs_int("OBS_SELF_SIZE")
    tm_stride, tm_count = _obs_int("OBS_TEAMMATE_STRIDE"), _obs_int("OBS_TEAMMATE_COUNT")
    en_stride, en_count = _obs_int("OBS_ENEMY_STRIDE"), _obs_int("OBS_ENEMY_COUNT")
    global_size = _obs_int("OBS_GLOBAL_SIZE")

    # Recompute bases by tiling (mirrors the derived C macros).
    self_base = 0
    tm_base = self_base + self_size
    en_base = tm_base + tm_stride * tm_count
    global_base = en_base + en_stride * en_count
    global_end = global_base + global_size

    if global_end != obs_dim:
        raise RuntimeError(
            f"Obs blocks do not tile OBS_DIM: blocks end at {global_end} but "
            f"OBS_DIM={obs_dim} in cs2_types.h. Fix a block *_SIZE/*_STRIDE or OBS_DIM.")

    # (start, stop) half-open slices — usable directly as obs[start:stop].
    obs_blocks = {
        "self": (self_base, tm_base),
        "teammate": (tm_base, en_base),
        "enemy": (en_base, global_base),
        "global": (global_base, global_end),
    }
    obs_block_lines = ",\n    ".join(f"{name!r}: {rng!r}" for name, rng in obs_blocks.items())

    OBS_OUTPUT.write_text(
        f"# Auto-generated from cs2_types.h — do not edit manually.\n"
        f"# Regenerate: uv run python scripts/sync_action_spec.py\n"
        f"\n"
        f"OBS_DIM = {obs_dim}\n"
        f"\n"
        f"# Per-agent observation block layout, mirrored from the OBS_* macros in\n"
        f"# cs2_types.h. Each value is a (start, stop) half-open slice usable\n"
        f"# directly as obs[start:stop]. Masking / demo-zeroing code MUST index via\n"
        f"# this table (e.g. OBS_BLOCKS['enemy']) — never hardcode 25 / 53 / 93.\n"
        f"OBS_BLOCKS = {{\n    {obs_block_lines},\n}}\n"
        f"\n"
        f"# Per-entity sub-structure of the teammate / enemy blocks (count × stride).\n"
        f"# Lets demo code walk individual teammate/enemy sub-slots if needed.\n"
        f"OBS_TEAMMATE_COUNT = {tm_count}\n"
        f"OBS_TEAMMATE_STRIDE = {tm_stride}\n"
        f"OBS_ENEMY_COUNT = {en_count}\n"
        f"OBS_ENEMY_STRIDE = {en_stride}\n")
    print(f"[sync_action_spec] Wrote {OBS_OUTPUT}")
    print(f"  OBS_DIM={obs_dim}  OBS_BLOCKS={obs_blocks}")


if __name__ == "__main__":
    main()
