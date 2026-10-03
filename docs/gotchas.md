# Operational gotchas

- A shared worktree venv points its editable install at one checkout. Put the
  worktree's `src/` first and avoid environment syncing there; follow
  [CONTRIBUTING](../CONTRIBUTING.md#setup). The import guard catches many wrong-tree
  launches, but a launch outside every checkout cannot infer the intended tree.
- A cold visibility cache can start many memory-heavy workers. Copy existing
  caches before testing in a new worktree; see the [env guide](../src/cs2rl/env/CONTEXT.md).
- C edits require a rebuilt binding and can invalidate BC demonstrations.
  Python documentation edits have different provenance rules; see [formats](formats.md#demonstrations).
- `cs2_render.h` uses CRLF. Preserve those bytes when touching it.
- Resume selects the model named by the checkpoint state, not the largest
  model filename. An interrupted three-file save may leave an inconsistent set.
- A resumed metrics stream can replay steps. Keep the later row and weight
  episode-window means by episodes; a mean of row ratios can mislead.
- Return alone is insufficient to judge behaviour. Pair it with task metrics
  and entropy; a finite smoke run exercises the env without training a policy.
- Historical experiments, ADR contexts and review summaries describe their
  recorded conditions. Current source and explicit owner decisions govern new work.

These entries are supported by the linked guides and checks in [validation](validation.md);
they do not reproduce private run logs or prescribe a new experiment budget.
