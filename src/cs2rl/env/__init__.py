"""L1 of `cs2rl layers` (pyproject.toml), with c_env and env_factory: the environment's Python side.

Members:
  config -- EnvConfig and RewardWeights, the typed env-knob contract (stdlib only);
  nav    -- NavGraph over the awpy nav mesh, the vis-matrix build, the sim constants;
  map    -- MapData and its builders (make_cs2_map, make_simple_map).

Inside the package, `cs2rl.env layers` puts map and nav above config, so config may
import neither: `c_env.cs2_env` and `train.py --dump-config` rely on config staying
stdlib-only.

WHY this file holds a docstring and nothing else: a re-export here would put nav's
imports (awpy, shapely) into the import chain of every `cs2rl.env.config` importer.
Without the file, grimp would not see the package and the wheel would drop it.
"""
