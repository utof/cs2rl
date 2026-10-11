# src/cs2rl/experiment/

Experiment tooling: the gates and analyses that read a run's files after it has finished,
three scripted-bot checks, an aim evaluation that plays a checkpoint against scripted
opponents, and a replay saver that records such rounds to watch. Each module with a `main()`
is a command line, run by module name from the repository root.

## What is here

| module | reads | prints |
|--------|-------|--------|
| `gate` | the Rung 1 seed directories `scripts/run_rung1.sh` writes (`config.json`, `metrics.jsonl`) | one verdict, exit 0 PASS, 1 FAIL, 2 INVALID |
| `smoke_read` | one Rung 1a run directory | the pre-flight assertions and one verdict, exit 0 PASS, 1 FAIL, 2 SMOKE INVALID |
| `analyze_tplant` | one or more runs' `metrics.jsonl` | the bomb-plant timing drift report |
| `oracle_statue` | nothing: it drives the Rung 1a env with two scripted actors; the movement-inaccuracy cone follows `EnvConfig.accuracy` (on by default) and the hit model `EnvConfig.hitboxes` (capsule hitboxes by default, #391), so `--accuracy 0 --hitboxes 0` is the exact Rung 1a env | kill rate and one PASS/FAIL line |
| `oracle_tracker` | nothing: `oracle_statue`'s harness with a random walker as the opponent (#152 L1) | one row per episode, the totals and one PASS/FAIL line |
| `oracle_duel` | nothing: two `eval.duel_bot` bots (or a bot and the L1 walker) in `oracle_statue`'s harness, the hero on the T row and then on the CT row (#152 L2); `--accuracy` and `--hitboxes` as in `oracle_statue` | one row per round (seed, spawns, outcome, each side's fired, hit and fast shots), the per-side totals and one PASS/FAIL line, exit 0 PASS, 1 FAIL |
| `aim_eval` | one policy checkpoint and its run's `config.json`, played in `oracle_statue`'s harness against a statue and three walkers | one row per opponent, aim mode and LSTM mode, plus the oracle and obs-oracle reference rows per opponent: kills, TTK, hit/fired, the aim error split along and across the target's motion, the lag in ticks, and on the policy rows the gap to the scripted oracle; no verdict, exit 0 when the run completes |
| `replay` | one policy checkpoint and its run's `config.json`, or `oracle` / `obs-oracle`, played in `oracle_statue`'s harness against one `aim_eval` opponent | one line per round and the `rerun` command that opens the replay; writes the replay (`viz.render.ReplayRecorder`, one Rerun recording per round) and a per-tick text dump beside it, under `outputs/recordings/` by default; exit 0 when the rounds complete |

```bash
uv run python -m cs2rl.experiment.gate outputs/checkpoints/rung1
uv run python -m cs2rl.experiment.smoke_read outputs/checkpoints/rung1a/s0
uv run python -m cs2rl.experiment.replay oracle --vs walker-4-16
```

`experiment` is in the top layer of `pyproject.toml`'s `cs2rl layers` contract, beside
`deploy`: it may import any other `cs2rl` package, and none may import it.

## L2: what `oracle_duel` certifies, and what it does not

On the arena duel (one agent a side, pitch pinned, crouch and jump off) with the cone and the
capsule hitboxes on, seed 0, bots deciding from their own observation rows (measured
2026-10-11, #152): a duel between two copies of one bot is winnable and symmetric (the default
mirror `hard,style=stop` decided 1074 of 1200 rounds with no time-out, T share 0.5019 [0.4720,
0.5317]; before #152 L2 resolved a tick's shots simultaneously, the T, first in agent order,
won 646 of 1000 rounds of an `expert,style=stop` mirror); stopping before a rifle shot beats
running under the movement-inaccuracy cone and not without it (hard stop against hard run, hero
share 0.5867 [0.5553, 0.6174] with the cone, 0.4671 [0.4354, 0.4990] without, 500 rounds a
side); lead matters under the hitboxes (the perfect planted bot hit the L1 walker with 0.7809
of its shots led, 0.2473 unled); and the observation-row bots play as the same bots reading the
C state, round for round, but for a ray that grazes a hitbox within float32 rounding (2 of 10
runs of 300 rounds diverged, each first on such a ray). It does NOT certify recoil or spray
(#392: no shot here has recoil, so M4 is not certified), a cost on firing (#399), crouching
(#393: the bot never crouches and aims at a standing head), a map with inner walls (the bot's
wall check is the map's extent), what CS2's own bots do (the dials are botprofile.db's; how the
focus dials compose is inferred, CS2's code is not public), or anything about a trained policy.
The bot aims its pitch at the hit model's aim point (`hit_geometry`), which a Rung 1 policy
cannot (its pitch head is masked); the arena's `pin_pitch=1` holds every pitch at 0, so that is
no advantage here, but it would be on a map that unpins pitch. Two traps it measured: at 16 Hz
one counter-strafe tick from 220 u/s lands at 72.9 u/s, under the cone's 74.8 u/s start, so a
stop takes one tick; and a perfect stopper loses to a perfect runner (hero share 0.1972): on a
tick the runner starts or reverses, its velocity changes by 75.6 u/s, 4.7 u off the stopper's
one-tick lead and wider than a head's half-width (the median miss of a perfect stopper, cone
off, was off by exactly 4.727 u).

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
