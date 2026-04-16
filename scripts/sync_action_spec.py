#!/usr/bin/env python3
"""Extract action head spec from cs2_types.h and write src/_action_spec.py.

Run after changing ACTION_HEAD_SIZES or ACTION_HEAD_NAMES in cs2_types.h:
    uv run python scripts/sync_action_spec.py
"""
import re
from pathlib import Path

HEADER = Path(__file__).resolve().parent.parent / "src" / "c_env" / "cs2_types.h"
OUTPUT = Path(__file__).resolve().parent.parent / "src" / "_action_spec.py"

text = HEADER.read_text()

# Extract ACTION_HEAD_SIZES[] = {9, 16, 2, ...}
sizes_match = re.search(r"ACTION_HEAD_SIZES\[\]\s*=\s*\{([^}]+)\}", text)
if not sizes_match:
    raise RuntimeError("Could not find ACTION_HEAD_SIZES[] in cs2_types.h")
sizes = tuple(int(x.strip()) for x in sizes_match.group(1).split(",") if x.strip())

# Extract ACTION_HEAD_NAMES[] = {"move", "aim", ...}
names_match = re.search(r'ACTION_HEAD_NAMES\[\]\s*=\s*\{([^}]+)\}', text)
if not names_match:
    raise RuntimeError("Could not find ACTION_HEAD_NAMES[] in cs2_types.h")
names = tuple(m.group(1) for m in re.finditer(r'"([^"]+)"', names_match.group(1)))

if len(sizes) != len(names):
    raise RuntimeError(f"Mismatch: {len(sizes)} sizes vs {len(names)} names")

action_dim = len(sizes)
action_mask_dim = sum(sizes)

OUTPUT.write_text(f"# Auto-generated from cs2_types.h — do not edit manually.\n"
                  f"# Regenerate: uv run python scripts/sync_action_spec.py\n"
                  f"ACTION_HEAD_NAMES = {names!r}\n"
                  f"ACTION_HEAD_SIZES = {sizes!r}\n"
                  f"ACTION_DIM = {action_dim}\n"
                  f"ACTION_MASK_DIM = {action_mask_dim}\n")
print(f"[sync_action_spec] Wrote {OUTPUT}")
print(f"  ACTION_DIM={action_dim}  ACTION_MASK_DIM={action_mask_dim}")
print(f"  NAMES={names}")
print(f"  SIZES={sizes}")
