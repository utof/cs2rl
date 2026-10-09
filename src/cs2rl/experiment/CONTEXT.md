# src/cs2rl/experiment/

Experiment tooling: the gates and analyses that read a run's files after it has finished,
and two scripted-bot checks. Each module with a `main()` is a command line, run by module name
from the repository root.

## What is here

| module | reads | prints |
|--------|-------|--------|
| `gate` | the Rung 1 seed directories `scripts/run_rung1.sh` writes (`config.json`, `metrics.jsonl`) | one verdict, exit 0 PASS, 1 FAIL, 2 INVALID |
| `smoke_read` | one Rung 1a run directory | the pre-flight assertions and one verdict, exit 0 PASS, 1 FAIL, 2 SMOKE INVALID |
| `analyze_tplant` | one or more runs' `metrics.jsonl` | the bomb-plant timing drift report |
| `oracle_statue` | nothing: it drives the Rung 1a env with two scripted actors | kill rate and one PASS/FAIL line |
| `oracle_tracker` | nothing: `oracle_statue`'s harness with a random walker as the opponent (#152 L1) | one row per episode, the totals and one PASS/FAIL line |

```bash
uv run python -m cs2rl.experiment.gate outputs/checkpoints/rung1
uv run python -m cs2rl.experiment.smoke_read outputs/checkpoints/rung1a/s0
```

`experiment` is in the top layer of `pyproject.toml`'s `cs2rl layers` contract, beside
`deploy`: it may import any other `cs2rl` package, and none may import it.

## Where new code goes

- A new gate or analysis: a module here with a `main(argv=None)`, launched as
  `python -m cs2rl.experiment.<module>`. If it reads metric columns, add it to `CONSUMERS` in
  `cs2rl.eval.metrics_schema`, name it on each column it reads, and add its (name, path) pair
  to the list in `consumer_key_reads` in `tests/_helpers/metrics_census.py`, where `tests/eval/`
  reads each consumer's keys from; `tests/eval/` compares the registry with the keys each
  consumer reads, in both directions.
- A script that launches runs is not a module here: `scripts/run_rung1.sh` is the pattern.
- Its tests: `tests/experiment/`.

## Traps

- A metrics row can appear twice: a resumed run replays rows, and the later copy wins. `gate`
  drops the replays with `analyze_tplant.dedupe_resume_rows`; `smoke_read` has its own
  `dedupe_by_agent_steps`.
- `game/*` columns are per-episode window means. Weight them by episodes when you aggregate
  across rows, never as a mean of per-row ratios (`gate`'s docstring, "ratios").
- A missing `self_play/used_past` key means self-play was off: read it as 0.0.
- `smoke_read` must run with no third-party package: `tests/experiment/test_smoke_read.py`
  launches it under `python -S`. `__init__.py` holds a docstring and nothing else, because it is
  in the import chain of every module here.
- The gates' rules come from specs and pre-registrations in the owner's local docs folder,
  which is not in the repository. Change the rule there first, then the code.
- In a worktree, put its own `src/` first:
  `env PYTHONPATH=<worktree>/src python -m cs2rl.experiment.gate ...`. From a working directory
  outside any checkout, the import guard cannot tell which checkout was meant, and main's copy
  runs silently (#242).
