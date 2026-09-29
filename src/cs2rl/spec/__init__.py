"""L0, the bottom layer of `cs2rl layers` (pyproject.toml): layouts and paths every layer reads.

Members:
  action -- the action-head layout (ACTION_HEAD_NAMES, ACTION_DIM, AIM_DIM, ...),
            generated from c_env/cs2_types.h by scripts/sync_action_spec.py;
  obs    -- the observation layout (OBS_DIM, OBS_BLOCKS, ...), generated the same way;
  paths  -- the output-directory constants (OUTPUTS_DIR, CHECKPOINTS_DIR, ...).

None of them imports anything first-party, and the layers contract keeps it so.

WHY this file holds a docstring and nothing else: a package without __init__.py is
invisible to grimp (so to every import-linter contract) and is silently dropped from
the wheel by `namespaces = false`; a re-export here would put every member into the
import chain of each one.
"""
