# Validation and its limits

Use [README](../README.md) for setup/smoke commands and
[CONTRIBUTING](../CONTRIBUTING.md#tests) for the full-suite/worktree procedure.
The documented two-worker suite was measured on one development machine;
start with at least 7 GB MemAvailable and copy existing visibility caches into
a worktree. Those measurements are not hardware-independent performance guarantees.

## Existing checks

| Concern | Check |
|---|---|
| Compiled struct/layout boundary | [struct sizes](../tests/env/c/test_struct_sizes.py) and [pitch](../tests/env/c/test_pitch.py) |
| Demo schema and source heuristic | [demo format](../tests/test_demo_format.py), [provenance](../tests/test_demo_provenance.py), [BC loss](../tests/test_bc_loss.py) |
| Checkpoint set and resume state | [resume state](../tests/train/test_resume_state.py), [legacy warm-start keys](../tests/train/test_resume_warmstart_keys.py) |
| Package direction | [import layers](../tests/integration/test_import_layers.py) |
| Orientation paths/names | [doc tokens](../tests/integration/test_doc_tokens_resolve.py) and [source name strings](../tests/integration/test_name_strings_resolve.py) |
| Staged typing changes | [pyrefly gate](../scripts/pyrefly_gate.py) and [its tests](../tests/integration/test_pyrefly_gate.py) |

The doc-token check reads README, CONTRIBUTING and tracked CONTEXT files; it is
not a complete Markdown link/anchor checker. The pyrefly gate reads the index,
so stage intended changes first. Its source-line key has a documented collision
residual; a green typing gate is not a correctness proof. Passing simulator
tests or `--smoke` does not establish learning, policy quality or export parity.

## Rung 1a smoke reader

The [reader](../src/cs2rl/experiment/smoke_read.py) and
[synthetic-row tests](../tests/experiment/test_smoke_read.py) retain the rules
from the 2026-08-31 registration (#152); this is a current-contract summary.
It reads `config.json` and `metrics.jsonl`, deduplicates `agent_steps` last-wins,
then takes at most ten rows at/after 900,000 of the registered 1,000,000
participating steps; fewer than five rows or zero completed episodes invalidates.
Preflights check participating rows, opponent provenance, final budget, statue
inertia and sigma measurability. Move histograms are counts divided by their total.

Episode means are weighted by episode counts; ratios pool numerator and
denominator separately. PASS requires kills/episode ≥0.5 and hit rate ≥0.4.
Sigma-cap breaches void sigma routing but can still PASS; missing sigma is
invalid. Other branches distinguish aim failure, an untrained aim head and
unrouted results. Exit codes are 0 PASS, 1 FAIL, 2 SMOKE INVALID. A verdict is a
readout of this experiment, not authorization to launch the next rung.
