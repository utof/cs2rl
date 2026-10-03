# 0002. Trainer behaviour goes in a PuffeRL subclass, not in monkeypatches

Status: Accepted
Date: 2026-09-03

## Context and decision

Historical context: instance monkeypatches replaced trainer update, rollout and
checkpoint methods, while the harness composed only some production behaviours.
New trainer behaviour goes into a subclass of `PuffeRL` overriding `train` and
`evaluate`, with buffers declared in `__init__`. No new trainer monkeypatch
functions are written. Timing is measured at the call site; vector-env plumbing
belongs in a wrapper around the vecenv.

## Consequences and current implementation

Composition becomes readable as a class and testable through its constructor.
The historical migration required fixed-seed byte-identical checkpoints before
accepting each move; that is a migration condition, not a general resume guarantee.
#168 completed the migration: [Cs2PuffeRL](../../src/cs2rl/train/trainer.py) owns
trainer methods and `HybridAimVecEnv` owns transport. The
[train guide](../../src/cs2rl/train/CONTEXT.md) describes the current constructor.
The decision did not authorize a PufferLib upgrade; its dependency pin was 3.0.0.
