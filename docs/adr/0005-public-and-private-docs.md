# 0005. Public conclusions and private working history

Status: Accepted
Date: 2026-10-03
Supersedes: the storage and review-history lifecycle clauses of [0004](0004-durable-knowledge-homes.md)

## Decision

For #202, the owner chose public documentation in the existing cs2rl code
repository: ADRs, architecture, glossary, format contracts, validated procedures
with limits, and selected reviewed postmortems. Setup stays in README and CONTRIBUTING.
Public content excludes raw logs, plans, reports, session records, machine paths
and private file links. Historical summaries identify their provenance and limits.

The original nested documentation repository, including its Git history and
dirty content, is retained locally under `_local/docs`. A root-scoped local
Git exclude hides `_local/`; this is neither encryption nor a backup. No private
remote, history publication or archive publication is part of this decision.

## Consequences

Public conclusions travel with code clones. Private archive and `.superpowers/`
workspaces remain permanent and must never be deleted. Promote useful conclusions
without discarding review traffic. No private pointer is needed to use public
documentation; preservation on one disk leaves independent backup unresolved.
