"""L1 of `cs2rl layers` (pyproject.toml): the environment, its C core and the env factory.

Members:
  config  -- EnvConfig and RewardWeights, the typed env-knob contract (stdlib only);
  nav     -- NavGraph over the awpy nav mesh, the vis-matrix build, the sim constants;
  map     -- MapData and its builders (make_cs2_map, make_simple_map);
  c       -- the C environment: the zig-built `binding` extension, its C sources and
             cs2_env (Cs2Env, make_env); exports SOURCE_DIR and ZIG_OUT;
  factory -- build_env_for, the one place an env is built (the self-play manager's builder
             lives with its class, in cs2rl.train.selfplay).

Inside the package, `cs2rl.env layers` orders factory above c, c above map and nav, and
those above config, so config may import none of them: `c.cs2_env` and `train.py
--dump-config` rely on config staying stdlib-only.

WHY this file holds a docstring and nothing else: a re-export here would put nav's
imports (awpy, shapely) into the import chain of every `cs2rl.env.config` importer.
Without the file, grimp would not see the package and the wheel would drop it.
"""
