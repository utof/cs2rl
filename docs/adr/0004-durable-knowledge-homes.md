# 0004. Where durable knowledge lives

Status: Accepted; storage and review-history lifecycle clauses superseded by [0005](0005-public-and-private-docs.md)
Date: 2026-09-03

## Historical context and decision

Decisions existed in private memory and review traffic without a later reader.
Each durable kind was assigned one home:

| Kind | Home |
|---|---|
| Decision shaping code | ADRs |
| Analysis of something that went wrong | Postmortems |
| Agent work rules | AGENTS.md, with project workflow in CLAUDE.md |
| System map | Architecture documentation |
| Queued work | GitHub issues linked from tracker #82 |

Agent memory must never be the only copy of durable knowledge. A live progress
pointer is overwritten instead of creating dated session-state files.

## Consequences and supersession

Durable conclusions must be promoted before a branch closes. The original
wording assumed review traffic would not survive and used “promote or discard.”
[0005](0005-public-and-private-docs.md) replaces that lifecycle assumption:
private history is permanent, while reviewed conclusions have public homes.
