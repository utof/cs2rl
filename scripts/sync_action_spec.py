#!/usr/bin/env python3
"""Extract action head spec from cs2_types.h and write src/_action_spec.py.

Batch 3: emits separate discrete + continuous spec lists. Backwards-compat
aliases (ACTION_HEAD_NAMES, ACTION_HEAD_SIZES, ACTION_DIM, ACTION_MASK_DIM)
stay; they refer to the discrete-only side. New: AIM_DIM,
CONTINUOUS_HEAD_NAMES, DISCRETE_HEAD_SPEC, CONTINUOUS_HEAD_SPEC.

Run after changing ACTION_HEAD_SIZES, ACTION_HEAD_NAMES, or AIM_DIM:
    uv run python scripts/sync_action_spec.py
"""
import re
from pathlib import Path

HEADER = Path(__file__).resolve().parent.parent / "src" / "c_env" / "cs2_types.h"
OUTPUT = Path(__file__).resolve().parent.parent / "src" / "_action_spec.py"

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

# Batch 3: continuous heads. Currently 1 head ("aim"), gaussian, 1D Δyaw.
# Spec format: tuple of (name, distribution_kind, dim) — same shape as
# DISCRETE_HEAD_SPEC for consumer symmetry. dim is the parameterised
# dimensionality (Gaussian mean/log_std vector size); for a 1D Gaussian
# the policy emits 2 floats per agent (mean, log_std) but the action
# buffer is dim=AIM_DIM=1 (the sampled Δyaw).
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
