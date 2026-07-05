# Auto-generated from cs2_types.h — do not edit manually.
# Regenerate: uv run python scripts/sync_action_spec.py

OBS_DIM = 107

# Per-agent observation block layout, mirrored from the OBS_* macros in
# cs2_types.h. Each value is a (start, stop) half-open slice usable
# directly as obs[start:stop]. Masking / demo-zeroing code MUST index via
# this table (e.g. OBS_BLOCKS['enemy']) — never hardcode 25 / 53 / 93.
OBS_BLOCKS = {
    'self': (0, 25),
    'teammate': (25, 53),
    'enemy': (53, 93),
    'global': (93, 107),
}

# Per-entity sub-structure of the teammate / enemy blocks (count × stride).
# Lets demo code walk individual teammate/enemy sub-slots if needed.
OBS_TEAMMATE_COUNT = 4
OBS_TEAMMATE_STRIDE = 7
OBS_ENEMY_COUNT = 5
OBS_ENEMY_STRIDE = 8
