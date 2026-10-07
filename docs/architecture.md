# Architecture

cs2rl trains policies in a C simulation; deployment exports policies for a CS2
server plugin. The package map below describes the current tree. Historical
ADRs explain decisions, while [pyproject.toml](../pyproject.toml) defines the
enforced Python import layers, highest first:

| Layer | Owns |
|---|---|
| `experiment`, `deploy` | Finished-run readers and policy/map export |
| `train` | CLI, run loop and the PufferLib trainer subclass |
| `train_bc`, `bc_demos`, `profile_step`, `viz`, `eval` | Cloning, demos, profiling, viewing, evaluation and metric schema |
| `policy` | Policy factory, checkpoint loaders and hybrid action distribution |
| `policy_net` | The neural network (`Dust2Policy`), imported lazily by `policy` |
| `env` | Simulation wrapper, maps, navigation, configuration and construction |
| `spec` | Generated observation/action layouts and output paths |

Higher layers may import lower layers. The contracts also forbid sibling cycles;
the top `experiment` and `deploy` members may not import one another. Inside
`env`: factory → C wrapper → map/navigation → configuration. Inside `train`:
CLI → loop → compose (the trainer builder) → modes/trainer → training parts →
configuration.

The [env](../src/cs2rl/env/CONTEXT.md), [train](../src/cs2rl/train/CONTEXT.md),
[experiment](../src/cs2rl/experiment/CONTEXT.md) and [scripts](../scripts/CONTEXT.md)
guides describe where new code goes. Tests mirror the owning package;
[tests/CONTEXT.md](../tests/CONTEXT.md) records the exceptions.

## Contracts at seams

- The C header authors obs/action layouts; Python specs are generated from it.
  The ctypes layout hash checks struct declarations, not every positional binding argument.
- Frozen `EnvConfig` and `RewardWeights` own env defaults. Runtime inputs such
  as seed, buffers and map data are supplied separately by the env factory.
- `HybridAimVecEnv` carries continuous aim beside discrete actions; `Cs2PuffeRL`
  owns trainer state and overrides rollout, update and checkpoint methods.
- The metrics registry owns emitted/derived keys. Smoke-reader aggregation defaults missing counters to zero; sigma has separate missing-value checks.
- Full resume checks a consistent checkpoint set and run configuration. It does
  not restore the simulator RNG. Optional Modal execution has its own artifact
  verification protocol; it never starts merely because local training runs.

See [formats](formats.md), [validation](validation.md) and [ADR 0001](adr/0001-seam-explicitness-rule.md).
