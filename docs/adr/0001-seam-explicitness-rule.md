# 0001. When a seam must be explicit

Status: Accepted
Date: 2026-09-03

## Context and decision

Historical context: env configuration, trainer composition and artifact
protocols carried multiple representations of a contract. A seam must be made
explicit when more than one consumer crosses it, its sides change at different
rates or are edited by different agents, or a mismatch fails silently.
A healthy seam has one direction of dependency and exactly one representation
of the contract. The number of representations is the problem, not consumers.

Generate consumers from one author or census writes against one registry;
matching literals kept in agreement by tests remain duplication with an alarm.
Current examples are the [generated layouts](../formats.md#observation-and-action)
and the [metrics registry](../../src/cs2rl/eval/metrics_schema.py).

## Consequences

Explicit contracts cost up front and move silent disagreements into check
failures. This decision alone does not authorize rewriting existing seams.
[0003](0003-typed-env-config-contract.md), [0002](0002-subclass-pufferl-do-not-mutate.md)
and issues #165/#166/#168 record the separate changes.
