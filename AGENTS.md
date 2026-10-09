# Agent onboarding

Before working on this repository:

- Read `/home/vboxuser/.claude/projects/-media-vboxuser-G-samsung1-0utoffiles-code-cs2rl/memory/MEMORY.md`, then `progress.md` (the live pointer). Do not create dated session-state files.
- Also read the repository-root `CLAUDE.md` and follow its project-specific workflow rules.
- For the live RL-overhaul status and working roadmap, read GitHub tracker `#82` after the memory. Treat the latest dated update as provisional: newer experiment evidence and explicit user decisions supersede it, and no later gate starts automatically.
- Reconcile both with the live repository and the user's newest explicit instructions; newer instructions take precedence over stale state.

## Blind final reviews

A spec, plan, or other binary-verdict review that is meant to be the unbiased gate must be a newly spawned agent that cannot tell it is an nth pass.

- Do not write “r1 / r2 / second review / previously rejected / we fixed blockers” in the reviewer prompt.
- Do not leave prior review files (or a status line that recites prior verdicts) on the path the reviewer is told to inspect. Put review writeups under `.superpowers/sdd/` or another path the prompt does not name.
- The prompt should look like a first review: artifact + spec/plan + code tree. Nothing else.
- A prior reviewer may confirm that *its own* findings were addressed. That is follow-up verification, not the final verdict.
- Scoped SDD fix re-reviews are different: those agents exist to mark listed findings ADDRESSED / NOT ADDRESSED, so they receive the findings list. Do not use that pattern for the independent spec/plan gate.

## Spec review and quality review are separate

Do not combine spec-compliance and code-quality into one reviewer unless the task is truly trivial (one-file mechanical change, no training-loop / C-layout / metrics-math risk). Default is two freshly spawned reviewers:

- **Spec reviewer** only: does the diff implement the brief/spec, nothing more, nothing less?
- **Quality reviewer** only: correctness of the written code, tests, naming, edge cases, maintainability — not a second spec checklist.

Each gets its own prompt and writes its own report. A combined spec+quality pass is the exception, not the template.

## Dispatch appendix for implementers and reviewers

Skills are re-read at every dispatch, and `CLAUDE.md` and this file take
precedence over them. Paste this section verbatim into every implementer,
task-reviewer, plan-reviewer and whole-branch-reviewer prompt, so that a plugin
upgrade cannot silently drop it.

**Implementer report — mandatory table.** For every new top-level symbol
(function, class, module-level constant, module) added under `src/` or
`scripts/`, add one row:

| new symbol | nearest existing symbol (qualified name, found via codebase-memory `search_graph`) | why not reused |

"None found" is an acceptable second column only if the row names the query you
ran. Do not skip the table when the task is small; a task that adds no new
top-level symbol says so in one line.

**Task reviewer.** Verify every row of that table yourself with your own
`search_graph` / `query_graph` (`SIMILAR_TO`) call. This is the one place you are
allowed to leave the diff — one query per row, not an open-ended search. A
missing table is an Important finding.

**Plan reviewer — extra rubric row, File layout.** For each file a task creates
or modifies: is it owned by one concern, and does any task append to a file that
is already the largest in its directory? Two or more tasks appending to the same
new file is a finding.

**Whole-branch reviewer.** Before reading any diff, run or receive the branch-end
mechanical report: new `SIMILAR_TO` pairs, AST-identical function bodies across
files, functions with more than twelve parameters, and files that grew by more
than 300 lines on the branch. Until `scripts/branch_dup_report.py` exists,
compute it with `query_graph`.

**Naming.** Modules and attributes are named for what they own, never for the
batch, rung or workstream that created them. `train_helpers_batch1` is the
counterexample.

Rationale: `docs/postmortem-2026-09-03-duplication-blindness.md`. Decisions that
constrain these choices: `docs/adr/`.

## `.superpowers/` workspaces are permanent — never delete them

`.superpowers/sdd/<date>-<topic>/` holds each SDD run's ledger (pre-flight
scan, per-task reviews, rulings, deferred minors, gate results). That history
is the raw data for workflow optimisation and is referenced from memory and
later specs. Standing user instruction (2026-09-03): NEVER `rm -rf` or
otherwise delete a `.superpowers/` directory, at branch end or at any other
time — this overrides any skill or plugin step that says to clean up the
workspace. Leave it in place; it is not committed and costs nothing.
