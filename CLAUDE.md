## codebase-memory-mcp

- Call `list_projects` or `index_status` and select the graph whose root is the current checkout. Never hardcode a machine-specific graph project name.
- Discover symbols with `search_graph`, then trace callers/callees and read exact snippets. Check coverage for every relied-on path; read reported missed ranges directly. Graph evidence is best effort, not proof of completeness.
- Python/C boundaries can be traced when indexed. Confirm current coverage instead of assuming every C construct was parsed.
- IMPORTS edges can be false on a name collision: `from cs2rl.experiment.gate import main` showed an edge to scripts/analyze_experiment.py (another module defining `main`). Confirm an importer census with an AST scan over `git ls-files`. IMPORTS edges also undercount: for train.py the graph gave 8 importer files, the AST 33, and 0 for a module with 17 (#205 part 3 prototype). Importer counts are AST-only.

- Use the setup in README and CONTRIBUTING. In a shared-venv worktree, use its own `src/` on PYTHONPATH with UV_NO_SYNC=1 and the venv interpreter; never sync the shared environment there.
- Normal validation is `python -m pytest tests -n 2 --dist loadgroup`: pytest defaults to `not slow and not training`. Add `-m training` for real trainer/learning/checkpoint-continuation changes, or `-m "slow and not training"` for affected expensive non-training checks. Releases and substantial training changes use one complete session: `python -m pytest tests -n 2 --dist loadgroup -m ""`. A focused training file/node also needs `-m "" -n 0`. These are explicit deselections, with every original test/workload retained; preserve performance-smoke opt-in and child-session isolation. See CONTRIBUTING for the selection census and commands.
- Read issue bodies AND every comment; comments often re-scope an issue. Use the owner-local issue helper when available, otherwise the GitHub issue page/API with complete comment pagination. In clean clones, private memory and helpers may be unavailable: use committed AGENTS.md, this workflow, public docs and issue history.

## Measure first, then spec/plan

Existing-capability evidence belongs before custom infrastructure: the prototyper, or implementer when no prototype is needed, checks existing project code, standard-library facilities and installed dependencies against current official documentation and supported versions. In the existing report, record the candidate API/version/source, a small executed comparison including the relevant failure case, requirements met, and the specific unmet behavior justifying custom code. Use the existing capability when it fits. The brief carries the evidence; the reviewer independently checks the riskiest claim. Missing evidence blocks acceptance of the custom abstraction. Include this rule explicitly in relevant delegated prompts, including nested delegation; unavailable evidence is unverified, not proof no capability fits. Keep the check bounded and add no research agent or enforcement framework solely for it.

- Before writing a spec or plan, run a measure-only experiment in a worktree. Prototype the change, or a sketch of it, and run the suite and knock-outs against it. Census the real call sites. Try the library, tool or API in a playground. Write the spec/plan FROM those measurements, so it names functions, flags and syntax that were actually run, not guessed.
- Why: prose plans do not converge. Review folds produced the next round's Criticals roughly 1:1 (memory: gate plans must be executed). The #205 part 2a/2b experiments found what no prose draft had:
  - 17 silent stale sites;
  - the EXEMPT-masking class;
  - a leftover namespace directory that silently loads a stale `.so`;
  - a regex false zero, caught only by a known-count check.
- When to prototype (size gate, owner 2026-09-30):
  - Prototype if ANY holds: the change moves or renames a module, file or public symbol; it touches 3+ files or is expected to exceed ~150 changed lines; a changed symbol has callers or importers outside its own file (graph + AST census); it adds or changes a guard, test-session hook, gate or CI step; it relies on library/tool behaviour not yet run here.
  - Skip it if ALL hold: one file plus its test, under ~50 changed lines, no symbol used outside that file changes, no guard or gate touched. Then there is no prototype and no brief review: failing test first, implement, one verifier.
  - In between: a short playground spike of the one uncertain thing, not a full prototype.
  - Re-gate mid-task, upward only: a task that skipped the prototype and crosses a gate line while in progress stops and gets prototyped. Never downgrade mid-task.
- The prototype's code IS the implementation's first draft, never a sketch to retype. Prototype, implementation and PR share ONE worktree and branch (owner 2026-09-30; nothing is pushed before the PR, so a second worktree only adds a handover):
  - the prototyper commits its change on the branch in logical commits and keeps measurement-only changes out of them (knock-outs are restored; a debug knob or timing hack it must keep goes in its own commit). Its diff, scripts and report are still saved under `.superpowers/sdd/` as the audit record;
  - the brief names what to undo (measurement-only commits or hunks, PROTOTYPE labels) and what to change;
  - the implementer is a fresh agent in the SAME worktree and branch. It starts at the prototype's HEAD, undoes only what the brief names, then works to the brief;
  - a prototype line changes only where the brief or a failing test requires it, and the implementer report lists each changed hunk with its reason. No rewrites for style;
  - why: retyping a measured prototype discards both its implementation and the evidence tied to it.
- How:
  - pre-register what you will measure and which result would change the plan. Order the questions by risk: the one most likely to kill the approach runs first. Mark each VALIDATED / PARTIAL / INVALIDATED with one line of nuance. At the first INVALIDATED, stop and report instead of measuring the rest (owner 2026-09-30);
  - the implementer's final diff is reviewed in full either way; execute the applicable validation tiers above and state every tier not executed;
  - include knock-outs: a guard that does not go red when broken is not a guard;
  - check every zero against a known count;
  - pin the base sha, and re-measure if main moves;
  - cite measurements in the spec/plan (report §N + sha1). Reviewers re-derive them; they do not trust them.
  - when the prototype finishes, post a short self-contained findings comment on its GitHub issue: what was measured, the key numbers, the traps, the decision. Raw reports and evidence stay local (they hold machine paths and captured working-tree diffs).
- Measuring replaces most of the review rounds: a brief written from a prototype gets at most ONE review (owner, 2026-09-30). The lead folds its findings; no confirmation round.
- What a review or verifier must flag, each exiting via fix-or-file (owner 2026-09-30): wrong behaviour or a missed requirement; a library/tool function that could replace hand-written code; any FALSE statement in code, comments, docstrings, messages or PR text (false prose is a defect here: postmortem 2026-09-11, #253, #319). Taste (naming preference, true wording that could read better, cleanups with no defect) goes on an OPTIONAL list: fix it only in lines already being edited, else drop it without filing.
  - Likelihood filter (owner 2026-09-30): a defect whose trigger is UNLIKELY (no past incident here and no realistic path to it, e.g. a test directory with a non-identifier name) gets one KNOWN LIMIT line where the code is, not new code or a pin. False prose is always fixed. A defect of a class that has bitten before (guard blind to its own scope, stale path, silent skip) is never "unlikely". Why: Anthropic's "chasing every finding leads to over-engineering"; the #207 part 2 brief review returned 15 findings, 3 of them hypothetical escapes.
- How many implementers (owner 2026-09-30). Every input is read off the brief:
  - start with ONE implementer for the whole brief;
  - start a fresh implementer before step k only if:
    - (a) cascade: step k builds on a function, interface or behaviour that an earlier, not yet checked step creates. Put a review gate there, then hand over;
    - (b) tier change: step k is judgment and the current chunk is mechanical, or the reverse. Mechanical means the brief gives the exact change (exact strings, a script, or the full code); judgment means the brief states a goal the implementer must design;
    - (c) size: the chunk's hand-edited files would exceed ~15, or its hand-written diff ~400 lines. Script-applied edits do not count. These thresholds are UNMEASURED guesses;
  - never parallel writers (conflicting implicit decisions: Cognition 2025, superpowers "never dispatch multiple implementation subagents in parallel"). Read-only agents (review, census, research) may run beside one writer. On this machine even separate-worktree writers run one heavy job at a time;
  - why: superpowers splits one task per implementer because each task gets its own review gate. Here the prototype already ran each commit's green checks and knock-outs, and one verifier reviews the whole branch, so a split pays only where an error would cascade (a), the model tier changes (b), or context fills up (c).

## Evidence is captured by the shell, cited by literal, never typed

Applies to reports, briefs, reviews, ledgers and issue drafts.

- Capture every return code, count, test result, timing and census number directly from the shell. Record the command, checkout, HEAD/status, dirty diff, verbatim output and return code. Use the owner-local evidence helper when available; a clean clone may use an equivalent shell capture. Do not retype output or manufacture an evidence file.
- Evidence is append-only: reruns get a new filename, failures stay. Keep the scripts that compute counts and print their raw input counts. Never delete `.superpowers/` workspaces.
- Cite an exact literal from one captured output/source line, with its file and source revision. Avoid hand-typed line numbers. Before finishing, verify each file/hash/literal (with the owner-local citation checker if available).
- Return the report's path, line count and SHA1, plus the citation-check result. Save a delegated report verbatim, check its hash/citations and spot-check claims before acting on it; never publish a draft before its author finishes.
- No secrets in evidence: no environment dumps or token-printing commands. Judgments stay prose, backed by captured queries.

Public conclusions belong in docs; private archive and review history follow ADR 0005. A clone contains this workflow without needing the owner's local archive.
