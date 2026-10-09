"""L3 of `cs2rl layers` (pyproject.toml), below train: evaluation and the metrics registry.

Members:
  baselines       -- the fixed-baseline evaluator (random, oracle, idle and policy actors);
  metrics_schema  -- REGISTRY, the one registry of every metrics key, and EVAL_KEYS;
  scripted_expert -- the scripted walk-to-bombsite-and-plant expert behind the BC demos;
  walker          -- the random-walker opponent (#152), import-light like metrics_schema.

metrics_schema sits in this layer, below train (#205 part 3): train may import the
registry, and nothing here may import train. The placement is pinned by
tests/integration/test_import_layers.py::test_train_sits_above_eval_and_viz_and_policy_below_them.

WHY this file holds a docstring and nothing else: a re-export here would load
baselines' torch import for every importer of metrics_schema, which must stay light
(tests/train/test_w1_modules.py). Without the file, grimp would not see the package and the
wheel would drop it.
"""
