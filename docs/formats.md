# Format contracts

## Observation and action

[cs2_types.h](../src/cs2rl/env/c/cs2_types.h) authors the layout;
[obs.py](../src/cs2rl/spec/obs.py) and [action.py](../src/cs2rl/spec/action.py) are
generated consumers. Use their symbols instead of copying dimensions into code.
Currently each observation has 110 floats: self `[0,28)`, teammates `[28,56)`,
enemies `[56,96)`, global `[96,110)`. Teammates have four slots of stride 7;
enemies five of stride 8. Width equality alone does not prove semantic compatibility.

Discrete heads, in order, are move/shoot/reload/weapon/use/crouch/jump with
sizes 9/2/2/3/2/2/2. Seven action entries have 22 mask entries. Continuous aim
has two floats: Δyaw and absolute pitch in radians. Pitch is bounded at ±π/2;
it is not a delta accumulator. [Pitch tests](../tests/env/c/test_pitch.py)
exercise consumption, bounds, observations and combat. The
[contributor recipe](../CONTRIBUTING.md#adding-or-changing-an-action-head)
owns layout changes and regeneration.

## Demonstrations

[bc_demos.py](../src/cs2rl/bc_demos.py) writes one NPZ per successful episode,
recording only the bomb carrier on `simple_v1`. Teammate/enemy blocks are zeroed.

| Member | Generated representation |
|---|---|
| `obs` | float32 `[T, OBS_DIM]` |
| `discrete_actions` | int64 `[T, ACTION_DIM]` |
| `continuous_actions` | float32 `[T, AIM_DIM]` |
| `dones` | bool `[T]`, true on the final planting tick |
| Metadata | `OBS_DIM`, `ACTION_DIM`, `AIM_DIM`, `seed`, `carrier_idx`, `spawn_area`, `bombsite_area`, `tick_count`, `git_sha`, `map` |

[load_demos/check_demo_sha](../src/cs2rl/train_bc.py) validate live dimensions,
map, shapes and provenance; loading converts arrays to training dtypes rather
than proving every stored dtype. Exact duplicate observation/action sequences
can be deduplicated. Provenance compares both Git index and working source:
Python comments/actual docstrings and byte-identical moves may pass, while C
bytes and executable literals remain significant. This is a bounded source
heuristic, not proof of identical rollouts; unknown history fails closed.

## Checkpoints

[Cs2PuffeRL.save_checkpoint](../src/cs2rl/train/trainer.py) writes, in order,
`model_<epoch>.pt`, `train_state.pt`, then `trainer_state.pt`, each atomically.
The last file names the model and stores optimizer/counter state. The extra
sidecar stores alpha/optimizer, scheduler, return statistics, warm-start,
self-play and Python/NumPy/Torch RNG state, plus epoch and step identity.
[Resume](../src/cs2rl/train/resume.py) uses the named model, checks set identity,
and compares configuration before rewriting it. Newer orphan models are ignored.
Legacy warm-start keys map to current keys; current keys win in mixed sidecars.
Simulator xorshift state is not saved, so resumed env sampling is not bit exact.

T/CT heads and trunks are optional policy variants; [policy split tests](../tests/test_tct_split.py)
and resume compatibility checks own their state-dict expectations. A weights-only
restart differs from full-state resume. Modal's
[checkpoint verifier](../scripts/modal_runner/checkpoint.py) additionally checks
artifact sidecar, byte size/hash, loadability and read stability.
