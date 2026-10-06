# src/cs2rl/train/

`python -m cs2rl.train`: the command line, the run loop, and the PufferLib trainer subclass.
`--smoke`, `--train`, `--record`, `--eval` and `--dump-config` pick the mode; `--help` lists
every flag.

## What is here

`pyproject.toml`'s `cs2rl.train layers` contract orders the modules, top first:

| layer | modules | owns |
|-------|---------|------|
| CLI | `__main__` | the argparse block, torch-free validation, the mode dispatch |
| driver | `loop` | `train(args)`: the run (W&B, metrics.jsonl, seeding, resume, eval hook), the epoch loop, checkpoints, dead-run detection |
| builder | `compose` | `build_trainer`: shared memory, vec env, policy, participation rows, self-play manager and `Cs2PuffeRL`, for `train()` and the test harness |
| modes | `record`, `evaluate`, `trainer` | `--record`, `--eval`, and `Cs2PuffeRL` |
| parts | `selfplay`, `envs`, `update`, `rewards`, `entropy`, `resume`, `metrics` | the self-play pool, env wiring and `--smoke`, the PPO loss, reward normalisation, the entropy schedule, full-state resume, metric rows |
| base | `config` | `build_train_config` (the run's `config.json`), `env_config_from_args` |

The policy network is `cs2rl.policy_net.Dust2Policy`, built by `cs2rl.policy.build_policy`;
both sit below eval, viz and BC. The env is `cs2rl.env`.

## The trainer

`Cs2PuffeRL` (`trainer.py`) subclasses pufferlib's `PuffeRL`. Its `__init__` calls
`_init_return_norm`, `_init_hybrid_aim` and `_init_selfplay`, in that order, then creates
`_timing`. It overrides `train` (the return-normalised PPO update), `evaluate` (the self-play
rollout) and `save_checkpoint` (full-state checkpoints). `compose.build_trainer` wraps the vec env in
`HybridAimVecEnv`, which carries the continuous aim beside the discrete actions, before it
constructs the trainer.

Nothing patches a trainer instance any more: #168 (W1 to W3) moved the four monkeypatches
into the class. The test harness, `tests/_helpers/trainer_harness.py`'s
`_build_trainer_for_test`, builds through `compose.build_trainer` with `env_role="harness"`,
so a harness trainer is built by the same code as a CLI run's and has every behaviour on.

## Where new code goes

- New trainer behaviour: a method or an `_init_*` step on `Cs2PuffeRL`, never a function that
  mutates an instance. State that must survive a resume goes into `collect_train_state` and
  `restore_train_state` (`resume.py`).
- A new flag: the parser in `__main__.py`, `build_train_config` so the run's `config.json`
  records it, and, if Modal must pass it, `LIVE_TRAIN_OPTION_ARITY` in
  `scripts/modal_runner/request.py`, a hand-kept mirror of the long options.
- A new metric: register the key in `cs2rl.eval.metrics_schema`'s `REGISTRY`.
- A new module here: a layer in `cs2rl.train layers`; the contract is exhaustive.
- Its tests: `tests/train/`. `_build_trainer_for_test` builds a small trainer on the Serial
  backend.

## Traps

- `trainer.py` imports torch and pufferlib at module scope. `compose` imports it inside
  `build_trainer`, so that `--dump-config` stays light; `test_cli_module_scope_stays_light`
  (`tests/train/test_w1_modules.py`) fails on a module-level import in `compose` or `loop`.
- The package exports nothing: import a name from the module that owns it. `pyproject.toml`'s
  banned-api table bans, by name, every name the old flat train module bound except its
  standard-library and numpy imports and `__all__`; it does not say where each went.
- `__init__.py` sets `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `OMP_NUM_THREADS` to 1
  unless they are already set.
- In a worktree, put its own `src/` first: `env PYTHONPATH=<worktree>/src python -m cs2rl.train`.
  Without it, `cs2rl`'s import guard refuses to run and prints that command.
