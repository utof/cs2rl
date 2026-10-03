# July adversarial verification: reconstructed contract summary

Historical date: 2026-07-06. Reconstructed public edition: 2026-10-03.
The formerly cited root-level report was absent. A same-named private archived
copy was located during migration, but its historical runtime probes were not
re-run. This page is a current-contract reconstruction, not the recovered
original verification report or confirmation of its full finding list.

## Finding 3: round outcome includes dead team members

[Reward tests](../tests/env/c/test_reward.py), especially
`test_loss_penalty_applies_to_fully_dead_team` and
`test_differential_win_magnitudes`, assert terminal reward for all members,
dead or alive. Death must not shield an agent from the team outcome. Aggregated
cross-team win reward can cancel; per-agent rewards and outcome flags carry
different information.

## Finding 4 residual: entropy targets

[target_entropy_schedule](../src/cs2rl/train/entropy.py) falls from 0.5 to 0.35
of max entropy by default. [Helper tests](../tests/train/test_reward_entropy_helpers.py)
pin those defaults and monotonicity. Production supplies config values through
the PPO update; these checks do not establish an optimal exploration schedule.

## Finding 21f: configuration ownership

The dead `TRAINING_CONFIG` removal is recorded in a comment in
[envs.py](../src/cs2rl/train/envs.py). Current ownership is the typed env
configuration and run configuration described in [architecture](architecture.md)
and [ADR 0003](adr/0003-typed-env-config-contract.md). No reconstructed claim
about a historical training run is used as a new gate.
