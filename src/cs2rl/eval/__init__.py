"""L3 of `cs2rl layers` (pyproject.toml), above train: evaluation and the metrics registry.

Members:
  baselines       -- the fixed-baseline evaluator (random, oracle, idle and policy actors);
  metrics_schema  -- REGISTRY, the one registry of every metrics key, and EVAL_KEYS;
  scripted_expert -- the scripted walk-to-bombsite-and-plant expert behind the BC demos.

metrics_schema sits in this layer, above train, on purpose:
tests/test_import_layers.py::test_metrics_schema_sits_above_train pins it.

WHY this file holds a docstring and nothing else: a re-export here would load
baselines' torch import for every importer of metrics_schema, which must stay light
(tests/test_w1_modules.py). Without the file, grimp would not see the package and the
wheel would drop it.
"""
