# Postmortem: architecture refactors need more than folder moves

Date: 2026-10-04. This public retrospective draws on the completed
[topology work](https://github.com/utof/cs2rl/issues/184),
[naming cleanup](https://github.com/utof/cs2rl/issues/167) and the later measured
test cleanups. It records bounded lessons, not a fresh audit of every module.
The newer test-cleanup branch still awaits combined full-suite acceptance.

## What caused rework

File placement and names were treated as evidence of ownership. They are useful
signals, but code can still duplicate a contract across differently named files.
A reviewer restricted to a diff cannot discover an existing equivalent outside
it. This is the earlier [duplication-blindness finding](postmortem-2026-09-03-duplication-blindness.md).

Names tied to a development batch or experiment rung also obscured what the
code owned. Changing trainer attribute names exposed another boundary:
attributes saved in checkpoint state are a wire format, even when their names
look private. [The naming migration](https://github.com/utof/cs2rl/pull/337)
preserved old-format loading and covered disk continuation instead of relying
on text replacement.

Structural guards then accumulated their own representations: named homes,
frozen counts, approximate test reach, override routers and AST enumerators.
One placement rule rejected an unrelated integration helper called _git because
a Modal helper had the same name. [The bounded ownership fix](https://github.com/utof/cs2rl/pull/343)
allowed independent private helpers outside that seam while preserving declared
homes and local rebinding checks. It did not remove all placement rules.

The training/preflight tests also translated flat override keys into existing
runner collaborator objects, then maintained duplicated source scanners to
check that translation. Their measured replacement uses the existing
collaborators directly. Unknown-input checks and actual consumer controls
justify removing that extra representation; a constructor alone would not.

## What the successful approach checked

| Boundary | Evidence needed |
|---|---|
| Module move | Tracked-source importer/caller census, alias and entrypoint resolution, plus current graph coverage |
| Private attribute rename | Saved key inventory, old/current/mixed checkpoint loading, precedence and actual disk continuation |
| Replacement interface | Existing owner/API/version, executed success and relevant failure, concrete residual gap |
| Guard simplification | Named policy being retained or retired, positive control, harmful case and restored source |
| Runtime or memory claim | Executed work and comparable observations; moving code does not remove that work |

Graph discovery supplies candidates. Missing edges, wrong scope and stale or
partially parsed files require source fallback. A graph zero cannot certify no
callers or importers. Conversely, a shared bare name is not proof of shared
ownership. Check the actual module, definition and responsibility.

The measured test setup was retained as the implementation's first draft on
the same branch. Rebuilding it in another worktree would discard the direct
link between its code, failure controls and observed results. At the combined
end, verification must assess the whole diff and the relationships across it.

## What remains a decision

The current test-home manifest and reach policy impose maintenance costs, but
ordinary pytest discovery, Ruff and import-linter do not implement all of those
requirements. Removing an internal layout policy may be sensible; call it a
policy change and name the protection being retired. Do not describe an
incomplete substitution as equivalent or add another metadata framework to
preserve a requirement that nobody has justified.

The patch-binding campaign is a separate example. It tests actual imported
binding behavior through fresh child processes. A repeated in-process pytest
experiment leaked an imported value alias and module state after native
monkeypatch teardown. The isolated boundary therefore remains justified in
that experiment. The second identical repetition is still a policy candidate,
not a measured removal, and all existing campaign arms remain.

## Rules carried forward

[CONTRIBUTING](../CONTRIBUTING.md#design-tests-around-the-behavior-they-protect)
assigns upstream discovery and the executed comparison to the prototyper or
implementer; reviewers challenge the riskiest equivalence claim.
[ADR 0001](adr/0001-seam-explicitness-rule.md) keeps a contract with one owner,
and the [architecture map](architecture.md) explains the current module layers.

Name code for its responsibility. Inventory live and persisted consumers before
changing it. Keep behavior tests at the consumer boundary, document the limits
of structural checks, and verify the combined branch. More review rounds over
the same narrow scope do not replace the missing evidence.
