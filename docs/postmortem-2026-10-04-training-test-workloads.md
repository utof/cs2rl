# Postmortem: controller tests inherited a larger training workload

Date: 2026-10-04. This retrospective covers seven CPU controller tests, not the
learning experiments or the complete test suite. The measured candidate is
commit e432d1249374ca39a4e69c1ccdfa84cc4ea9582e; final combined-branch full-suite
acceptance remains pending.

## What happened

Six warmstart tests and one entropy-target schedule test used 32 environments
through the real trainer harness. Their assertions concern floor activation,
grace/ramp transitions, the target consumed by alpha loss and optimizer motion.
The inherited environment count made each training call execute seven
minibatches. Those assertions did not require that particular workload.

The existing [trainer harness](../tests/_helpers/trainer_harness.py) already
supports smaller environment counts by clamping its test minibatch size. A new
trainer, shared mutable fixture or special cleanup path was unnecessary. The
prototype changed five test workload literals from 32 to 4 while preserving
the original assertions, tolerances and update sequences.

## Evidence and limits

The selection is [the six warmstart cases](../tests/train/test_warmstart_entropy_trainer.py)
plus test_target_entropy_schedule_applied in
[test_train_env.py](../tests/train/test_train_env.py). Each variant ran in a
fresh serial pytest session, on CPU, with the same interpreter, observation
plugin and options. The second pair reversed both node order and variant order.

| Observed work across the seven cases | Before | After |
|---|---:|---:|
| Passing behavioral cases | 7 | 7 |
| Fresh trainers / evaluate calls / train calls | 7 / 9 / 16 | 7 / 9 / 16 |
| Executed minibatches per train call | 7 | 3 |
| Executed minibatches and policy optimizer steps | 112 | 48 |
| Complete checkpoint saves | 13 | 13 |
| Pair 1 external wall | 84.65 s | 27.12 s |
| Pair 2 external wall | 86.37 s | 24.60 s |
| Pair 1 GNU-time maximum RSS | 1,159,060 KiB | 1,026,408 KiB |
| Pair 2 GNU-time maximum RSS | 1,171,604 KiB | 1,023,760 KiB |

The eight deliberately broken production consumers still failed at the intended
original assertions: floor always true, floor always false, missing consumed
warmstart target with its trace retained, frozen mode-off alpha optimizer,
ignored custom target fraction, divided absolute ratio, alpha updates during
grace, and a floor rearmed during grace. The opposite floor cases also passed
where appropriate. The ratio mutation failed with three actual minibatches;
reducing the workload to one would risk making that check ineffective.

Both pairs show less selected work and lower observed elapsed cost and maximum
RSS. They do not establish a stable percentage improvement or complete-suite
saving. GNU time records an individually accounted process/waited-child
maximum, not simultaneous process-tree memory. The free-memory threshold for a
full run remains a separate safety prerequisite. Scratch directories were
removed and the test processes exited; these observations are not proof of
absence of all allocator, thread or native-handle leaks. A final uninstrumented
selection also passed.

## Why it was easy to miss

A shared harness default became the local workload without a test-specific
reason. Passing behavior assertions showed correctness for that size, but did
not show that its cost was necessary. Static setup inspection also would not
have established the cost: the prototype observed actual loss calls and
optimizer steps, and most of the measured selected work was in train().

A smaller workload changes controller motion and minibatch normalization. A
green run alone would have been insufficient. The harmful controls, preserved
tolerances and real consumers supplied the evidence for this reduction.

## Rules carried forward

The [contributing guide](../CONTRIBUTING.md#design-tests-around-the-behavior-they-protect)
assigns the check to the prototyper or implementer: justify the workload by the
behavior and harmful changes it detects, record actual executed work, and use
the existing implementation. Reviewers challenge loss of that sensitivity.

Keep real integration coverage where its workload is part of the requirement:
seed comparisons, checkpoint continuation, long-update identities and compiler
variants. Sharing mutable trainers, bypassing full cleanup or hiding expensive
checks behind a marker is not this optimization. Historical numerical comments
must be relabeled or remeasured when a workload changes.
