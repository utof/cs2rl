# Glossary

| Term | Meaning here |
|---|---|
| T / CT | Terrorist / counter-terrorist team; a round outcome is a team reward, including dead members |
| Agent row | One player's observation/action/reward slot; parked players still occupy slots |
| Participating row | A row selected to contribute to training; budgets and `global_step` use participating units |
| Env step / tick | One simulator advance; do not equate it with one participating agent step |
| Episode / round | One simulated round; terminal info is reported once per env round |
| Hybrid action | Discrete categorical heads plus continuous yaw/pitch aim |
| Δyaw / pitch | Relative yaw change and absolute pitch target, both in radians |
| PBRS | Potential-based reward shaping; its gamma must match the training discount |
| BC | Behaviour cloning from scripted demonstrations before PPO training |
| BPTT segment | A rollout segment used for recurrent policy updates |
| Self-play pool | Saved policies available to control opponent slots |
| Noop statue | The stationary opponent used by the Rung 1a smoke reader |
| TAG | Team-aligned gradient diagnostic; structural separation can make cross-team cells zero by construction |
| Warm-start entropy | GRACE → RAMP → OFF override after a BC initialization |
| Checkpoint set | Model weights, optimizer/counters and extra training state from one epoch |
| Seam | A boundary with an explicit representation and dependency direction |

[Formats](formats.md) defines stored units; [validation](validation.md) explains
the distinction between a functional check and evidence of learned behaviour.
