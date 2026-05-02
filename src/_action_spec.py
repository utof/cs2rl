# Auto-generated from cs2_types.h — do not edit manually.
# Regenerate: uv run python scripts/sync_action_spec.py

# Backwards-compat aliases (discrete side only — Batch 3+).
# Pre-Batch-3 callers reference ACTION_HEAD_SIZES symbolically; keep
# this list discrete-only so MultiDiscrete spaces still build correctly.
ACTION_HEAD_NAMES = ('move', 'shoot', 'reload', 'weapon', 'use', 'crouch', 'jump')
ACTION_HEAD_SIZES = (9, 2, 2, 3, 2, 2, 2)
ACTION_DIM = 7
ACTION_MASK_DIM = 22

# Batch 3 split: discrete vs continuous heads.
# DISCRETE_HEAD_SPEC: tuple of (name, 'categorical', n_categories).
# CONTINUOUS_HEAD_SPEC: tuple of (name, 'gaussian', dim).
# AIM_DIM is the float-buffer width per agent for the continuous side.
DISCRETE_HEAD_SPEC = (
    ('move', 'categorical', 9),
    ('shoot', 'categorical', 2),
    ('reload', 'categorical', 2),
    ('weapon', 'categorical', 3),
    ('use', 'categorical', 2),
    ('crouch', 'categorical', 2),
    ('jump', 'categorical', 2),
)
CONTINUOUS_HEAD_SPEC = (('aim', 'gaussian', 1), )
CONTINUOUS_HEAD_NAMES = ('aim', )
AIM_DIM = 1
