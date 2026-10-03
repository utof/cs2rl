# Postmortem: why diff-scoped reviews missed duplication

Date of case: 2026-09-03. This public edition distills the original private
postmortem; it does not reproduce its review ledger or claim a fresh branch audit.
The historical cases were issues #163–#166: one runner file owned several
concerns, bomb state overlapped, env defaults were repeated, and artifact
protocol logic existed twice. These are historical examples, not claims that
the current tree still has those exact implementations.

A reviewer confined to a diff cannot compare a new symbol with a pre-existing
equivalent outside that diff. File layout was decided in the plan, but the
review rubric checked task size rather than ownership of the resulting files.
More passes over the same scope did not supply those missing comparisons.

The adopted mechanisms are recorded in [AGENTS.md](../AGENTS.md): implementers
list each new symbol's nearest existing equivalent and explain non-reuse;
task reviewers independently check those rows; plan reviewers assess file
ownership; branch reviewers receive similarity, identical-body, parameter-count
and file-growth measurements before reading the diff.

Those mechanisms expand evidence in bounded ways. They do not prove absence of
duplication or replace correctness review. [ADR 0001](adr/0001-seam-explicitness-rule.md)
states the contract principle, and [0005](adr/0005-public-and-private-docs.md)
preserves the private history behind public conclusions.
