# 0003. One typed object is the env configuration contract

Status: Accepted
Date: 2026-09-03

## Context and decision

Historical context: constructor signatures, CLI mapping and reward-default
dictionaries repeated the same contract. One typed configuration object is the
single representation of env knobs and reward defaults, and their sole author.
Environment constructors and role builders consume it instead of restating it.
The lower environment/map/navigation/spec layers never import argparse or the
trainer: configuration flows in, not back out.

## Consequences and current implementation

The #165 implementation introduced frozen [EnvConfig/RewardWeights](../../src/cs2rl/env/config.py)
and the [role factory](../../src/cs2rl/env/factory.py); Phase A landed in PR #174.
Runtime seed, buffers and map data are separate inputs. A new knob still needs
its C field/mirror/wiring and CLI route; the decision eliminates repeated
defaults, not every cross-language edit. See the [env guide](../../src/cs2rl/env/CONTEXT.md)
for the current edit recipe and enforcement checks.
