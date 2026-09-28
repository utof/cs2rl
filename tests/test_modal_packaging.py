"""Guards for the Modal runner's packaging, its dependencies and its test seam.

Import spelling is not checked here. ruff's TID251 banned-api table in
pyproject.toml rejects the bare `modal_runner` spelling repo-wide, and the
session guard in tests/conftest.py fails any pytest session that loads one repo
file under two module names.
"""
import ast
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

# BINDING_SITES is a data dict, and the campaign module imports only importlib
# and typing at module scope (it reaches the runner package only inside
# `binding_target`, from a formatted string). So this import pulls in no runner
# module. The reach floor reads it to resolve `binding_target("key")` to the
# key's owning module.
from tests.modal_patch_binding_campaign import BINDING_SITES

# `MANIFEST` is renamed on import because this file's own `MANIFEST` is the seam
# manifest's path; the tables' one is {"<module>.py": [the names it owns]}.
from tests.modal_runner_tables import MANIFEST as RUNNER_OWNERS
from tests.modal_runner_tables import (
    RUNNER_MODULES,
    RUNNER_PATHS,
    RUNNER_TEST_FILES,
)

ROOT = Path(__file__).resolve().parents[1]
# The one legal spelling of the runner package. The reach floor resolves a test
# file's runner imports against it, and tests/test_modal_runner_package_shape.py
# records any absolute import of it from inside the package as an edge no table
# allows. The bare spelling `modal_runner` is banned repo-wide by ruff's TID251
# table in pyproject.toml.
PACKAGED = "scripts.modal_runner"

# ── The modal test seam: concern, recomputed from source ───────────────────
#
# WHY this section exists at all. `tests/test_modal_runner.py` held two suites.
# Splitting it needs an answer to "which half does this name belong to?" that a
# checked-in test can RE-DERIVE, because the alternative -- freeze a manifest,
# then check the files agree with the manifest -- is green on a maximally wrong
# split: review shuffled every name by even/odd index, split the file to match
# its own shuffled manifest, and the consistency check passed. A manifest is a
# record, not evidence. `classify_seam` is the evidence.

MANIFEST = ROOT / "tests" / "fixtures" / "modal_test_seam_manifest.json"

# How many names the seam governs, pinned so that SHRINKING it costs a diff line.
#
# THE HOLE THIS FILLS, demonstrated rather than argued. Every assertion in the
# placement gate iterates the manifest, and the agreement test compares
# `set(manifest)` against `set(computed)`. A name deleted from the tree AND from
# the manifest is therefore examined by nothing: review excised
# `test_allowed_gpus` from both and the seam reported `3 passed`. That is not an
# exotic mutation -- it is exactly what a real deleting commit looks like, because
# deleting the test alone reddens the agreement test and whoever did it fixes
# that before pushing. Regenerating the manifest is the natural fix and it is
# also what hides the loss.
#
# Pinning the size does not stop a deletion; nothing here can, and it should not.
# It makes one visible, which is the same argument the manifest itself rests on:
# a reclassification that moves eleven names moves eleven lines of JSON where a
# reviewer reads them. This moves one number. Change it deliberately, in the
# commit that changes the seam, and say why.
#
# 287 is `len(classify_seam(_seam_sources(), RUNNER_TEST_FILES)[0])`, the count
# after gh#211's signal-test fix. Against gh#163's 285: two runner-side tests
# added in tests/test_modal_training.py
# (`test_publish_note_reaches_the_volume_in_the_right_commit`,
# `test_a_signal_inside_finalize_does_not_finalize_again`), and one helper
# renamed in place (`_record_hash_after_terminal` -> `_record_checkpoint_reads`,
# a key rename that moves no count), nothing removed, no destination moves.
# 285, after gh#163's W5 execute commit, against the W5 types fold's 283: two
# runner-side tests added in tests/test_modal_training.py
# (`test_training_kwargs_routes_every_override`, the training builder's
# self-test, and `test_process_control_tripwire_guards_the_resolution_path`, the
# tripwire's resolution-path control), nothing removed, no destination moves.
# 283, after the fold of the W5 types reviews, against the W5 prepare commit's
# 282: one runner-side helper added in tests/test_modal_training.py
# (`_KillSeamClauses`, the static kill-seam clauses moved out of
# `test_kill_seam_static_safety` so that each checker is measured on its own),
# nothing removed, no destination moves. 282, after the W5 prepare commit,
# against the W5 types commit's 281: one
# runner-side test added in tests/test_modal_preflight.py
# (`test_preflight_kwargs_routes_every_override`), and one renamed in place
# (`test_prepare_records_install_dump_probe_then_launch` ->
# `test_prepare_records_install_dump_probe_in_order`, a key rename that moves
# no count), nothing removed, no destination moves. 281, after the types
# commit, against the W4 per-module tree's 279: two
# runner-side tests added in tests/test_modal_training.py
# (`test_kill_seam_static_safety`, `test_process_control_tripwire_poisons_system`),
# nothing removed, no destination moves. 279 was the count after the
# process-group guard commit (gh#163), when the declared runner set was still
# the one unsplit file (W4's split moved names; a move changes no key).
# Against the W3b split tree's 278: one
# runner-side test added (`test_signal_process_group_refuses_groups_a_live_child_cannot_have`),
# nothing removed, no destination moves. W3b's own move, against 2bb32ac's 273:
# five client-side helper/test names added, one `SEAM_GUARDS` name renamed, no
# destination moves -- the manifest's own diff lists the names. Do not infer
# this value from additions in the packaging file, which the classifier
# excludes.
#
# A GREEN `assert len(manifest) == GOVERNED_NAME_COUNT` IS NOT EVIDENCE THIS
# NUMBER IS RIGHT. It compares two frozen artifacts -- a checked-in manifest
# against a checked-in constant -- so adding names to a governed file changes
# neither and it stays green with a stale count and N ungoverned names on disk.
# Its own message says so: it is a DELETION detector. The addition detector is
# `test_seam_manifest_agrees_with_the_classifier`, which recomputes.
#
# 295, after gh#238 on top of main's 291 (gh#197's census follow-up added
# `_verify_checkpoint_census` and `test_verify_checkpoint_census_rejects_bypass_mutants`
# in tests/test_modal_checkpoint.py to the earlier 289): four names in
# tests/test_modal_training.py, the finalize kill-path pins:
# `test_a_hung_heartbeat_does_not_strand_the_run_in_training` (P3),
# `test_a_signal_while_taking_the_once_gate_returns_at_once` (P2),
# `test_finalize_kills_the_child_before_joining_the_tees` (P1) and its stream
# helper `_BlockingStream`. (The P2 lock double is a class nested in its test,
# so it is not a governed name.) 299, after gh#243: four more names in
# tests/test_modal_training.py, the production-order default for the signal
# tests: `_interrupt_in_production_order`, `_SIGNAL_HOOKS_RELEASE_ALLOWLIST`,
# `_SIGNAL_HOOKS_HANDWRITTEN_ALLOWLIST` and the static census
# `test_signal_tests_fire_handlers_in_production_order`.
GOVERNED_NAME_COUNT = 299

# THE PLACEMENT RULE FOR RUNNER TESTS. The docstring of every runner test file
# points here, so this is the one statement of it: change it here, not there.
#
# A runner test lives in tests/test_modal_<m>.py, the file of the module m (in
# `RUNNER_MODULES`) whose behaviour it tests; `RUNNER_TEST_FILES` in
# tests/modal_runner_tables.py lists those files. Tests reach private library
# names through their owning submodules; the package facade exposes the
# production caller surface. The seam manifest
# (tests/fixtures/modal_test_seam_manifest.json) records each test's file, and
# the reach floor (`reach_floor_violations`) checks it from below: a test must
# reach its file's module, unless `_REACH_EXEMPTIONS` exempts it with a reason
# and the reach it was granted for, and a test in the core file may name only
# `core` in its own body. Both rules read a module only as `mrl.X`, as `<m>.X`
# through a module-level `from scripts.modal_runner import <m>` (or `import
# scripts.modal_runner.<m> as <alias>`), or as `binding_target("key")`; a
# runner name imported by itself (`from scripts.modal_runner import
# build_run_request`) credits nothing, so the split test rejects the import
# (`_runner_imports_the_floor_cannot_resolve`). What a change costs (the gate's
# failure messages spell out the exact edit):
#   * adding a test: one manifest line, and `GOVERNED_NAME_COUNT` +1 in the same
#     commit with the reason in the commit message; deleting one: the reverse;
#     moving one to another runner file: its manifest value, and the count stays;
#   * a helper goes where the tests that reach it are: to their file if they all
#     sit in one seam file, else to tests/modal_test_helpers.py (THE MEMBERSHIP
#     RULE FOR THE SHARED FILE, below). `classify_seam` computes it. A helper is
#     a governed name too (any module-level def, class or constant of a seam
#     file), so adding or deleting one costs the same manifest line and count
#     as a test, and moving a test can move its helpers' destinations with it.
# PITFALLS. The seam reads module-level `test_` functions (`def` or `async def`)
# only: a pytest `class TestX` or a `def testfoo()` is not supported, and the
# gate rejects it. So does a def or class nested under a module-level `if`,
# `try` (its `except`, `else` and `finally` too), `with`, `for`, `while` or
# `match`: pytest collects a test there, but the seam, the floor and the core
# rule read only module-level definitions, so `classify_seam` rejects it
# (`_defs_under_module_level_statements`); skip a conditional test with
# `pytest.mark.skipif` instead. And a runner test never goes in a new
# tests/test_modal_<x>.py: a file outside `RUNNER_TEST_FILES` is invisible to
# the seam, the floor and the binding census (gh#233). A new runner test file
# comes only with a new module in the tables.
# THE KILL SEAM. A test that drives the training attempt, in any file, follows
# the kill-seam rules as well as this one. It hands the attempt a
# `training.ProcessControl` whose `spawn`, `getpgid` and `killpg` are fakes:
# `_training_kwargs` in tests/test_modal_training.py always builds one, and a
# client test's execute wrapper builds its own with all four fields as
# keywords. No test passes the real OS functions, and none but the two
# tripwire tests reads `ProcessControl.system`; a test that leaves `process`
# out meets the autouse tripwire in tests/conftest.py, which makes `system()`
# raise under pytest. `test_kill_seam_static_safety`
# (tests/test_modal_training.py) checks the rules by AST, over tests/, every
# module of the runner package and scripts/run_modal.py, and says why each
# exists.
#
# THE DECLARED RUNNER SET is `RUNNER_TEST_FILES` itself, read under that one
# name by `_seam_sources`, the live `classify_seam` calls, the reach floor and
# the binding census (tests/test_modal_patch_binding_census.py). There is no
# alias for it here any more: with the old one (`RUNNER_FILES`) widened by one
# file and a request test moved into that file, the floor and the census
# stopped reading the test with every gate green, because
# they read `RUNNER_TEST_FILES` and the seam read the alias. A second, retyped
# or widened list is how the gate and the tree drift apart. At the seam's
# reading sites (`_seam_sources`, the `classify_seam` calls) the floor's scope
# check (`_floor_scope_problems`) is red on one: it compares the pairs the floor
# examined with EVERY runner-half test in the manifest, not only those in
# `RUNNER_TEST_FILES`. A widened list in the binding census is caught by the
# census's own scope assertion instead.
CLIENT_FILE = "tests/test_modal_client.py"
PACKAGING_FILE = "tests/test_modal_packaging.py"

# THE MEMBERSHIP RULE FOR THE SHARED FILE. `tests/modal_test_helpers.py` exists
# as of W2's split and carries these words in its own docstring, which is where a
# reader opening that file will look for them. This copy is the one the
# classifier sits next to; they must not drift.
#
# A name earns a place in the shared file by being REACHED BY TESTS IN TWO OR
# MORE SEAM FILES (the declared runner files and the client file). Nothing else
# earns it. A helper that only one file's tests reach belongs in that file --
# however generic the helper looks, and however well its name would read here.
# The rule this replaced, "reached from both halves of the runner/client seam",
# is its special case with one runner file. A consumer outside the seam files
# does not count: the classifier never reads one. `classify_seam` computes
# exactly this rule and will not send a single-file helper to this file, so the
# rule is enforced rather than merely stated.
#
# WHY write down a rule the classifier already computes. Spec §10 criterion 12
# bans a module named `utils` / `helpers` / `common` / `misc`. Its instrument,
# the `forbidden-name` clause in tests/test_modal_runner_package_shape.py, reads
# only the `scripts/modal_runner/` submodules, so a test module is outside its
# scope and there is no conflict here. But the criterion exists because a module
# named for what it IS rather than for what it OWNS becomes a junk drawer, and
# that failure mode does not care which directory it happens in. A one-line
# membership test is what keeps this file a seam artefact instead of a drawer:
# measured at W4's relocation, it holds 13 names -- the 7 reached from both
# halves of the runner/client seam, and the 6 that tests in two or more runner
# test files reach -- and every one of them is reached from two or more seam
# files.
SHARED_FILE = "tests/modal_test_helpers.py"

# The three module-level guards that lived above line 100 of the monolith. They
# are about packaging and dependencies, so they belong to neither half of the
# seam; they are named rather than computed because "is a packaging guard" is a
# judgement about subject matter that no reference graph encodes. Measured: the
# monolith's only three top-level `test_` functions defined above line 100 are
# exactly these, which is the spec's §2.3 "three tests ... belong to neither
# half" recomputed rather than copied.
SEAM_GUARDS = frozenset({
    "test_modal_is_an_explicit_dependency_group",
    "test_local_entrypoints_do_not_import_modal",
    "test_modal_runner_does_not_import_modal_or_torch",
})

# Names the seam does not govern, with the reason.
# Unless a sentence names another tree, the figures in this comment were
# measured at 2bb32ac, by AST over the module-level bindings of every collected
# `tests/test_*.py`. They are that tree's values, not current ones: the file
# counts rise whenever a new test file defines its own `ROOT`, which W3b's
# `tests/test_modal_runner_package_shape.py` does.
#
# `ROOT` is `Path(__file__).resolve().parents[1]` -- module-header boilerplate
# that every destination file defines for itself by construction. At 2bb32ac,
# 6 files define a `ROOT` of their own (`test_modal_argv.py`,
# `test_modal_client.py`, `test_modal_packaging.py`, `test_modal_protocol.py`,
# `test_modal_runner.py`, `test_train_loop_timing.py`). `tests/modal_test_helpers.py`
# defines one too and is correctly absent from that list: it is outside the
# `test_*.py` glob this sentence scopes by.
#
# WAS 5 BEFORE W2, AND THE SIXTH IS THIS SEAM'S OWN CLIENT FILE -- which is why
# the second half of this paragraph had to be rewritten rather than renumbered.
# It used to read "`ROOT` is the ONLY one of the monolith's 261 module-level
# names that any other collected test file also defines". After the split that is
# false by 86: the client half is itself a collected test file and defines 84 of
# those names, and the three relocated `SEAM_GUARDS` are defined here. The claim
# the sentence was making survives once the seam's own four files are excluded
# from "any other", which is what it always meant -- measured that way, `ROOT` is
# still the ONLY collision, and the files it collides with are
# `test_modal_argv.py`, `test_modal_protocol.py` and `test_train_loop_timing.py`,
# 3 of them.
#
# THE 261 IN THIS PARAGRAPH IS THE MONOLITH'S OWN NAME COUNT AND IS HISTORICAL.
# It used to be the live governed count as well, and saying "261 is unchanged"
# here stopped being true on the W3a gates branch, which took the seam to 273
# (then `GOVERNED_NAME_COUNT`). What this paragraph actually measures did NOT
# move: re-measured at 2bb32ac, the collision set is still `ROOT` alone and the
# same 3 files, and the client half still defines 83 of the monolith's governed
# names plus `ROOT` -- which is the 84, and 83 + the 3 relocated `SEAM_GUARDS`
# is the 86. The W3a branch's twelve new client-half names are not monolith
# names, so none of them enters any figure in this paragraph.
#
# So this exemption list is one name long because the collision set is one name
# long, not because the rest were not looked for. Treating `ROOT` as a shared
# helper would make the seam's own scan collide with those unrelated files.
#
# STATED RATHER THAN HIDDEN: an exemption nothing checks is a hole. It is closed
# in both directions as of W2 by `test_the_modal_test_split_matches_concern_
# recomputed_from_source` below, whose last assertion requires `ROOT` to BE
# defined in every destination file, so "not governed" cannot quietly become
# "lost".
SEAM_HEADER_NAMES = frozenset({"ROOT"})

# The seed-count figures below describe 2bb32ac, measured by the own-body signal
# census this comment describes. They move as client tests are added, so
# re-derive them with that census rather than advancing the prose.
#
# A name whose own body reaches the client modules. These two sets are the
# spec's own §2.3 instrument, member for member.
#
# `module` is here because the monolith's client tests bind
# `module = _import_run_modal()` and then talk to `module`: measured, 13 of the
# 63 client tests carry that signal and NO other in their own body. THE
# DENOMINATOR MOVED ON THE W3a GATES BRANCH AND THE NUMERATOR DID NOT -- 54 -> 63
# as the four gates were added, while 13 stayed 13. The mechanism, measured
# rather than assumed: all 9 new client-half gate tests carry the signal set
# `{_import_run_modal, module}`, because GC5c's placement fix gives each one a
# `module = _import_run_modal()` positive control, and this sentence counts
# tests whose ONLY own-body signal is `module`. Bumping the 13 alongside the 54
# would have put a false figure into the file that holds the classifier; it is
# re-derived here, not inferred. Deleting it
# still moves 0 names, because the transitive closure covers the same 13 -- see
# `classify_seam`'s stage-1 note. Every member of both sets has a dedicated case
# in `_SEAM_CLASSIFIER_PROBE`, because before those cases existed, deleting
# `"module"` was measured to leave the whole file green.
_CLIENT_BINDINGS = frozenset({"_import_run_modal", "_run_modal_image_reqs", "module"})
_CLIENT_MODULES = ("scripts.run_modal", "scripts.modal_artifacts", "scripts.modal_backfill_sidecar")

_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _module_level_names(tree):
    """Every module-level binding in `tree`, as {name: defining node}.

    Covers `Assign`/`AnnAssign` targets as well as defs and classes. Dropping
    constants is not a simplification: measured, the monolith has 6 module-level
    assignments, and knocking the `Assign` branch out leaves 5 of those names
    with NO destination at all -- `PINNED_CUDA_CHILD_DIGEST`, `PINNED_CUDA_IMAGE`,
    `PINNED_PUFFERLIB_SDIST` and `PROTOCOL_TOKENS`, all four of which classify to
    the client half, plus `_FORBIDDEN_FLAGS`, which classifies to the runner
    half. (The 6th is `ROOT`, which `SEAM_HEADER_NAMES` excludes on purpose.) A
    census that walks defs alone leaves those five unassigned, and a pinned CUDA
    digest copied into both files can then drift apart with every check in this
    file green.

    It reads `tree.body` only, and so does everything built on it: a def or
    class nested under a module-level `if` or `try` is not here, although
    pytest collects a test defined there. `classify_seam` rejects one
    (`_defs_under_module_level_statements`) rather than this function reading
    it, because a def that may not exist at run time has no one right
    destination.
    """
    found = {}
    for node in tree.body:
        if isinstance(node, _DEFS):
            found[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        found[sub.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found[node.target.id] = node
    return found


def _module_level_binding_counts(tree):
    """{name: how many module-level statements bind it} -- the LIST-shaped census.

    WHY THIS EXISTS NEXT TO `_module_level_names`, which looks like it already
    answers the question: that function returns a DICT, so two module-level
    definitions of one name inside ONE file collapse to a single entry, and
    everything built on it inherits the blindness. The placement gate's
    `duplicated` check counts FILES and therefore cannot see the case at all.
    Measured by review at `84622fc`: appending a second `PINNED_CUDA_IMAGE` to
    tests/test_modal_client.py, shadowing the real digest with
    `...@sha256:deadbeef`, left the whole seam at `3 passed`.

    THE PITFALL, and it is the failure the seam was built to stop rather than a
    new one: two copies of a pinned digest that drift apart. Python does not let
    you have two LIVE definitions of a name across two files -- one import wins
    and the other is dead -- but it lets you have them inside one file, where the
    second silently shadows the first and both are in the source a reader greps.
    So the intra-file shape is the only one the failure can actually take at run
    time, and it was the one shape nothing watched.

    Mirrors `_module_level_names`' branches exactly: defs and classes via
    `_DEFS`, `Assign` targets walked so tuple unpacking counts each name, and
    `AnnAssign`. It counts where that function assigns, so the two cannot come to
    disagree about what a module-level binding is -- which matters, because a
    census that disagreed with the one the gate uses everywhere else would report
    duplicates nobody else believes in.
    """
    counts = {}
    for node in tree.body:
        names = []
        if isinstance(node, _DEFS):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [s.id for t in node.targets for s in ast.walk(t) if isinstance(s, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        for name in names:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _defs_under_module_level_statements(tree):
    """[(name, line, keyword, statement line)] per def or class under a module-level statement.

    The census `classify_seam` rejects on. pytest
    collects a `test` function wherever the module defines it, so a test under
    a module-level `if hasattr(os, "killpg"):` or in the `else` of a `try:
    import X` is collected, run and passed. But `_module_level_names` reads
    `tree.body` only, and the classifier, the manifest, the reach floor, the
    core rule and the floor's scope check are all built on it: the scope check
    compares two populations that lose such a test TOGETHER, so every gate
    stayed green with it in the core file naming a request name.

    Every module-level statement that is not itself a def or class is walked,
    every clause included (`else`, `except`, `finally`, `case`), and a def or
    class found there is reported without descending into its body. That is
    the complement of "module-level definition", not a list of compound kinds:
    a simple statement cannot hold a def, so this is exactly "under a
    module-level compound statement", and a compound kind a later Python adds
    is covered without an edit. A def's own body is never walked: a nested def
    inside a module-level test is ordinary. Assignments and imports under a
    statement are not reported: the header's `if str(ROOT) not in sys.path:`
    and a `try: import X / except ImportError: X = None` fallback are legal.
    A constant bound there is ungoverned, like any non-def binding under a
    compound statement (0 in the ten seam files when this census landed).
    `keyword` is the statement's keyword (`try` for `try ... except*`).
    """
    found = []
    for statement in tree.body:
        if isinstance(statement, _DEFS):
            continue
        keyword = type(statement).__name__.lower().removesuffix("star")
        stack = list(ast.iter_child_nodes(statement))
        while stack:
            sub = stack.pop()
            if isinstance(sub, _DEFS):
                found.append((sub.name, sub.lineno, keyword, statement.lineno))
                continue
            stack.extend(ast.iter_child_nodes(sub))
    return sorted(found, key=lambda row: row[1])


def _referenced_module_names(node, own, universe):
    """Module-level names `node` references, decorators included.

    An `ast.Attribute` is resolved to its ROOT name (`FakeModal.spec` ->
    `FakeModal`), which is what makes attribute-spelled reaches visible. Local
    shadowing is deliberately NOT modelled: over-reporting an edge can only
    merge two names into one destination, while missing one strands a helper --
    the failure that is a NameError at run time.
    """
    out = set()
    for sub in ast.walk(node):
        target = None
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            target = sub.id
        elif isinstance(sub, ast.Attribute):
            root = sub
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                target = root.id
        if target and target != own and target in universe:
            out.add(target)
    return out


def _reaches_client_directly(node, own):
    """Seed test: does this node's OWN body name a client module?"""
    if own in _CLIENT_BINDINGS:
        return True
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in _CLIENT_BINDINGS:
            return True
        if isinstance(sub, ast.Attribute):
            root = sub
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in _CLIENT_BINDINGS:
                return True
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if any(m in sub.value for m in _CLIENT_MODULES):
                return True
    return False


def _is_test_def(name, node):
    """Is the module-level binding `name` -> `node` a pytest test: a `test_` name bound by a def?

    The one definition of "a test" in the seam section. `classify_seam` (which
    names are tests), `_floor_reach` (which pairs the floor examines) and
    `_probe_test_names` all call it. PITFALL: the relocation's floor scope
    assertion equates the floor's examined pairs with the seam manifest's
    `test_` entries, so a second, hand-written copy of this predicate that
    drifts (say, one that forgets `async def`) moves one side of that equation
    and not the other. A `test_` name bound by an assignment or a class is not
    a test here, and neither is a `def testfoo()` without the underscore,
    although pytest collects one (this repo sets no `python_functions`). Both
    then classify as helpers; one that nothing reaches raises in
    `_helper_destination` with a message about pytest tests.
    """
    return name.startswith("test_") and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))


def _closure(start, edges):
    """Every name reachable from the names in `start` along `edges` ({name: names it references}).

    Shared by `classify_seam`'s stage 2 (which tests reach each helper) and the
    reach floor (which helpers a test reaches), so the two cannot come to
    disagree about what "reaches" means. `start` is a name's own out-edges, not
    the name itself: the name is in the result only if a cycle leads back to it.
    """
    seen, stack = set(start), list(start)
    while stack:
        for nxt in edges[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def classify_seam(sources, runner_files):
    """Compute each module-level name's destination file from the code alone.

    `sources` is {relative path: source text}; `runner_files` is the declared
    runner set, the files a runner-half TEST may live in. It is a required
    parameter, not a default, so every caller states the set it classifies
    against: the seam gate, the synthetic probes and the relocation's helper
    map all pass `RUNNER_TEST_FILES` (whose files need not exist for a
    pure-data call), and one probe passes a widened copy to show that the
    rejection keys on the set passed. Returns `(destinations, defined_in)`:
    {name: destination path} and {name: [files that define it]}. Works on the
    unsplit file (one entry) and on the split (several), because it treats the
    union of the files as one namespace -- which is exactly why it can be re-run
    after the split and still be evidence rather than a tautology.

    TWO STAGES, and the split between them is the whole design:

    1. TESTS are classified by transitive closure. A name is CLIENT if its own
       body names a client module, or if it references -- transitively -- a name
       that does. Measured against the seam the spec measured four ways, 0 of
       194 tests land on the wrong side. That 194 is a MONOLITH figure and is
       kept as one deliberately, because the measurement it reports was taken
       against the spec's line-3638 seam, which no longer exists. Do not try to
       re-derive it from this function's inputs, which have grown since: the
       live population is the `test_` names
       `classify_seam(_seam_sources(), RUNNER_TEST_FILES)[0]` sends to a runner file
       or to `CLIENT_FILE` (the `SEAM_GUARDS` sit in a file `_seam_sources()`
       excludes). That population moves with every test either half adds, so
       no live count is written here; derive it when you need it.

       A RUNNER-HALF TEST'S DESTINATION IS THE FILE OF `runner_files` THAT
       DEFINES IT. Which runner file a test belongs in is the placement
       judgement the seam manifest records, and the reach floor
       (`reach_floor_violations`) and the core rule check it from below; this
       function does not re-derive it. So for tests the placement gate's
       `misplaced` check compares on-disk against on-disk. The floor and the
       core rule see SOME tests in the wrong runner file (a request test in the
       training file reaches no training code); only the relocation's one-shot
       declared-placement check sees EVERY one. A runner-half test defined
       OUTSIDE `runner_files` (the client file, the shared file, an undeclared
       file) has no legal destination and raises `ValueError`, naming every
       such test at once.

       WHICH PART OF STAGE 1 EARNS THAT ZERO, because a one-at-a-time census
       gets this backwards. Knocked out singly, on the monolith: the closure
       moves 0 names, `"module"` moves 0, the `_CLIENT_MODULES` string seed
       moves 0, and attribute-root resolution moves 0. Knock out the closure
       AND `"module"` together and 30 names move, 13 of them tests -- the two
       cover the same 13 client tests, so each looks like dead weight until the
       other is gone. `_run_modal_image_reqs` is the one seed that is
       load-bearing alone (5 names). The string seed and attribute-root
       resolution move 0 even jointly with the closure disabled: they are
       over-coverage, kept because an extra edge can only merge two names into
       one destination while a missing one strands a helper.

    2. HELPERS follow the tests that reach them, because a helper has no concern
       of its own -- it has its consumers'. `FakeModal` names nothing
       client-specific; it is client because only client tests reach it. A
       helper whose consumers all sit in ONE seam file goes to that file (the
       client file, or the one runner file its runner consumers share); a helper
       whose consumers span TWO OR MORE seam files goes to `SHARED_FILE`. With
       one runner file that is the old "reached from both halves" rule, and
       measured on that tree, 7 names take the shared answer. A classifier
       forced to pick one file for those would strand them, which is a
       NameError at run time in whichever file lost.

    HONESTY NOTE, because the bar was known before stage 2 was written: stage 1
    alone puts 28 names on the wrong side -- all of them non-test helpers, 0 of
    them tests -- or 21 once the 7 SHARED names are set aside, which stage 1
    structurally cannot produce and which the next sentences are about. Both
    numbers are one measurement under two scopes, so the scope is stated rather
    than left to the reader. Stage 2 was added afterwards, with the target
    already
    known. Tuning a classifier until it matches a number you already have
    certifies it against the answer rather than against the source. The reasons
    to believe stage 2 anyway are that it is a different KIND of rule rather
    than a longer marker list, and that it produced a finding nobody had -- the
    7 shared names -- which a second instrument (a cross-seam reference census
    that uses LINE POSITION, not this closure, as each test's concern)
    reproduces exactly. Bound on that word "second": it shares
    `_module_level_names` and `_referenced_module_names` with this function, so
    it is independent of the seed and the closure but NOT of the reference
    graph. A bug in edge extraction would be invisible to both.

    KNOWN LIMIT, stated rather than hidden: a name reached by no test at all is
    unclassifiable by stage 2 and raises. Measured on the monolith: 0 such
    names. If one appears it is either dead code or a new entry point, and both
    deserve a human, not a default.

    A SOURCE WITH A DEF OR CLASS NESTED UNDER A MODULE-LEVEL STATEMENT RAISES
    `ValueError` before anything is classified, naming every such definition
    by file and line (`_defs_under_module_level_statements` has why). Every
    gate here reads module-level definitions only, and it is this raise that
    makes the agreement and split tests red on one; the reach floor, called
    on the same sources, would pass it.

    SECOND KNOWN LIMIT: this reads the AST, so a reference written inside a
    string of code that a child process runs is invisible to it, exactly as it
    is to every other AST reader. The seam's files do hold such strings
    -- `_container_equivalent_import` in the client half builds one for
    `python -c`, and `_write_probe_stubs` in the runner half writes stub
    modules a child imports -- so the limit is live. Checked 2026-09-22 on the
    W3b tree, by word-matching the string constants of every module-level seam
    function that passes `-c` or writes multi-line code to a file against the
    seam's module-level names: no hits, so the limit changes no destination
    today. A code string that did name one would be a reach this function
    misses, and nothing in this file would object. The monolith's two
    `code = \"\"\"...\"\"\"` blocks are not among them: both moved into THIS file
    with the `SEAM_GUARDS` in W2, and this file is excluded from
    `_seam_sources()`.

    Names mentioned in PROSE -- docstrings that cite a test or helper -- are not
    reaches either: `_referenced_module_names` ignores string constants, and
    `_reaches_client_directly` reads them only for `_CLIENT_MODULES`
    substrings. How many monolith names such prose mentions is a count that
    moves with every docstring that cites one; this docstring used to carry it
    as a figure and it went stale twice, the second time inside the wave that
    rewrote the headers it counted, so it is retired rather than advanced.
    """
    trees = {rel: ast.parse(text) for rel, text in sources.items()}
    nested = [
        f"{rel}:{line} `{name}`, under the module-level `{keyword}` at line {at}"
        for rel, tree in trees.items()
        for name, line, keyword, at in _defs_under_module_level_statements(tree)
    ]
    if nested:
        raise ValueError(
            f"definitions nested under a module-level statement: {nested}. pytest collects a "
            "`test` function or `Test` class there (under an if, try, with, for, while or match, "
            "their else, except and finally included), but the seam, the reach floor, the core "
            "rule and the floor's scope check read only module-level definitions, so such a test "
            "is checked by none of them and such a helper is governed by none. Move each one to "
            "module level: a test that must not always run takes `@pytest.mark.skipif(...)` or "
            "calls `pytest.importorskip(...)` in its body; only the condition's imports and "
            "assignments stay under the statement.")
    defined_in, nodes = {}, {}
    for rel, tree in trees.items():
        for name, node in _module_level_names(tree).items():
            defined_in.setdefault(name, []).append(rel)
            nodes[name] = node

    universe = set(nodes)
    edges = {n: _referenced_module_names(node, n, universe) for n, node in nodes.items()}
    tests = {n for n, node in nodes.items() if _is_test_def(n, node)}

    client = {n for n, node in nodes.items() if _reaches_client_directly(node, n)}
    changed = True
    while changed:                     # stage 1: reach-a-seed, to a fixed point
        changed = False
        for name, targets in edges.items():
            if name not in client and (targets & client):
                client.add(name)
                changed = True

    reached_by = {n: set() for n in nodes}
    for test in tests:                 # stage 2: which tests reach each helper
        for name in _closure(edges[test], edges):
            reached_by[name].add(test)

    governed = [n for n in nodes if n not in SEAM_HEADER_NAMES and n not in SEAM_GUARDS]
    destinations = {name: PACKAGING_FILE for name in SEAM_GUARDS}
    destinations.update(
        _test_destinations([n for n in governed if n in tests], client, defined_in, runner_files))
    for name in governed:
        if name not in tests:
            destinations[name] = _helper_destination(name, reached_by[name], destinations)
    return destinations, defined_in


def _test_destinations(tests, client, defined_in, runner_files):
    """{test: destination} for `classify_seam`'s governed tests; raises on a stray runner test.

    A client test goes to `CLIENT_FILE` wherever it sits (the placement gate
    then compares that with disk). A runner-half test goes to the file of
    `runner_files` that defines it. PITFALL, and the reason this raises rather
    than returning some destination: a runner-half test defined anywhere else
    has no right answer here. Sending it to "the" runner file stopped meaning
    anything once the runner set can hold eight files, and sending it to the
    file it sits in would make an undeclared file a legal home. Every such test
    is collected first and named in one error, so one run lists them all.
    """
    out, stray = {}, {}
    for name in tests:
        if name in client:
            out[name] = CLIENT_FILE
            continue
        outside = [rel for rel in defined_in[name] if rel not in runner_files]
        if outside:
            stray[name] = outside
        else:
            out[name] = defined_in[name][0]
    if stray:
        raise ValueError(
            f"runner-half tests defined outside the declared runner files {sorted(runner_files)}: "
            f"{stray}. A test is runner-half when nothing it reaches, directly or through "
            "helpers, names a client signal (_CLIENT_BINDINGS, _CLIENT_MODULES). If it IS a "
            "runner test, move it into a declared runner file -- the tests/test_modal_<m>.py of "
            "the module it tests (RUNNER_TEST_FILES) -- and set its value in the seam manifest "
            "to that file; a move changes "
            "no key, so GOVERNED_NAME_COUNT stays. If it is meant to be a CLIENT test, it lacks "
            "a client signal: make it reach one (or a helper that does); it then classifies to "
            "the client file, so move it there (the placement check names the manifest edit).")
    return out


def _helper_destination(name, consumers, destinations):
    """Stage 2 for one helper: the single seam file its consumers sit in, else `SHARED_FILE`.

    `consumers` are the tests that reach `name` transitively; each one's file is
    its destination from stage 1 and `_test_destinations`. A consumer whose
    destination is not a seam file (a `SEAM_GUARDS` name, sent to the packaging
    file by fiat) does not count, because the rule is about the seam's own
    files. So a helper reached by a guard and by client tests goes to the
    client file, not to the shared file (the probe's `_guard_and_client_helper`
    pins that). PITFALL: a helper reached by no counting consumer raises
    instead of defaulting, because the two things it can be -- dead code, or a
    new entry point -- want opposite answers. That includes a helper only a
    guard reaches: it has consumers, but none in the seam. A pytest test class
    (`class TestX`) lands here too, since only module-level `test_` defs are
    tests (`_is_test_def`) and nothing reaches the class, and so does a
    `def testfoo()` without the underscore, which pytest also collects. The
    message says so for any name with the prefixes pytest collects (`test`,
    `Test`), because "dead code" points the wrong way for a pytest test.
    """
    files = {destinations[test] for test in consumers} - {PACKAGING_FILE}
    if not files:
        pytest_test = ""
        if name.startswith(("test", "Test")):
            pytest_test = (" If it is a pytest test (pytest collects a function named `test...` "
                           "and a class named `Test...`): the seam supports only module-level "
                           "`test_` functions, so name a function `test_...`, or write each "
                           "method of a class as one, put each in the runner test file of the "
                           "module it tests or in the client file, and add each to the seam "
                           "manifest; the agreement check names the edit.")
        raise ValueError(
            f"{name!r} is reached by no seam test (a SEAM_GUARDS test does not count), so stage 2 "
            "cannot place it; it is dead code, a guard-only helper that belongs beside the guards "
            "in the packaging file, or a new entry point, and needs a human." + pytest_test)
    if len(files) > 1:
        return SHARED_FILE
    return files.pop()


def _seam_sources(root=ROOT, runner_files=RUNNER_TEST_FILES):
    """Return {path: source} for every declared runner file, the client file and the shared file.

    Exclude this classifier's own file: its source mentions client bindings,
    so self-feeding assigns the instruments to the concerns they inspect. The
    three legitimate packaging destinations are supplied by SEAM_GUARDS.

    Derive the current population from
    ``classify_seam(_seam_sources(), RUNNER_TEST_FILES)[0]``. A self-fed histogram
    also depends on which client bindings and client-module paths this file
    happens to mention, docstrings included; it is a diagnostic measurement,
    not an invariant worth another test or prose pin.

    A DECLARED FILE THAT DOES NOT EXIST RAISES `FileNotFoundError`; it is never
    skipped. The error names every missing path and no other declared path. An
    existence filter here is the hole this closes: a declared runner file that
    drops out of the classifier's input silently shrinks every gate built on
    it, and a repo-wide grep for the file's name cannot find a filter that
    holds no file name. `test_the_seam_source_reader_fails_on_a_missing_declared_file`
    pins it. `root` and `runner_files` are parameters (with the live values as
    defaults) so that check can hand this reader a declared set with one bogus
    path, or a root holding copies of every declared file but one, while every
    other declared file really exists; with an empty `root`, every file is
    missing and a reader that raises only when NOTHING is left would pass it.

    Live deletion of a NAME is guarded separately:
    `test_the_modal_test_split_matches_concern_recomputed_from_source` pins
    `GOVERNED_NAME_COUNT` and requires every manifest name to be defined on
    disk, and `test_seam_manifest_agrees_with_the_classifier` compares the
    manifest with the classifier.
    """
    rels = [*runner_files, CLIENT_FILE, SHARED_FILE]
    missing = [rel for rel in rels if not (root / rel).is_file()]
    if missing:
        raise FileNotFoundError(
            f"declared seam file(s) missing: {missing}. The seam never skips a declared file: "
            "restore it; or, if you are ADDING A MODULE (its tests/test_modal_<m>.py is declared "
            "through RUNNER_TEST_FILES before it exists), create the file with its first test; "
            "or remove it from the declaration in the same commit that removes the file.")
    return {rel: (root / rel).read_text(encoding="utf-8") for rel in rels}


def _names_defined_under_tests():
    """{name: [files]} for every module-level name in the files the seam governs
    plus every OTHER collected modal test file.

    The glob is the point. Review moved one test into a brand-new
    `tests/test_modal_stray.py` and the first draft of this gate stayed green,
    because it only ever looked at files the manifest itself names. A
    destination the manifest does not know about has to be reachable, or the
    check is asking the suspect for the list of places to search.

    CALLED BY `test_no_governed_name_is_defined_outside_the_seams_own_files`, and
    the reason that matters is a second review finding: for one round this
    function shipped with no caller in the committed suite at all, so the same
    stray-file plant still left the file green and the only certification was a
    gitignored script. Task 4's placement gate is the other caller. If you are
    about to remove the last caller, delete the function with it.

    Scope is `tests/test_*.py` plus the shared helper module: a stray file that
    pytest never collects is not the threat, and measured, widening past
    `test_*.py` pulls in `tests/capture_dump_config_pre_165.py` and
    `tests/capture_env_config_pre_165b.py`, which each define a `_git` of their
    own and would make a gate built on this red for an unrelated reason.

    CONTRACT for the caller: the return value is a SUPERSET of the seam. It
    covers every `tests/test_*.py` in the repo, not just the four destination
    files, which is exactly what makes a fourth file visible -- and exactly why
    a caller comparing it against the manifest must scope the disk-to-manifest
    direction to names it actually governs rather than flagging every unrelated
    test file's helpers.

    NO EXISTENCE FILTER. The glob's entries exist by construction, so a filter
    here could only ever skip `SHARED_FILE`, a declared seam file, and a
    declared seam file must fail when it is missing, never be skipped. A
    missing shared file fails at `read_text`, naming the path.
    """
    found = {}
    paths = sorted(set((ROOT / "tests").glob("test_*.py")) | {ROOT / SHARED_FILE})
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        for name in _module_level_names(ast.parse(path.read_text(encoding="utf-8"))):
            found.setdefault(name, []).append(rel)
    return found


# ── The reach floor: a test in tests/test_modal_<m>.py reaches module m ─────
#
# `classify_seam` decides runner versus client versus shared. It cannot decide
# WHICH runner file a runner test belongs in: that is a judgement about what
# the test tests, and the seam manifest records it. The reach floor and the
# core rule (`reach_floor_violations`) are a floor under that judgement, not a
# proof of it; the function's docstring states the measured residual.

# Declared exemptions from the reach floor: {(file, test): (granted, reason)}.
# Keyed by FILE as well as test, so a test moved to another file loses its
# exemption there, and its old entry, which now names a file that does not
# define it, fails. An entry must also still be needed: an exempt test that does
# reach its file's module fails (minimality), so this set cannot go stale
# unnoticed.
#
# AN EXEMPTION IS NO WIDER THAN GRANTED. `granted` is the frozenset of runner
# modules the test reached when it was exempted (possibly empty), and the gate
# fails when the test's measured reach differs from it, in either direction.
# Without `granted` nothing checked the reach a reason states, so an edit that
# made the test reach `request` and `state` -- a misplaced request test, the
# shape the floor exists to reject -- passed with every gate green. When the
# reach changes on purpose, re-grant it and re-read the reason in the same
# commit.
#
# THE GRANT IS PER MODULE, NOT PER ROUTE, because reach is a set of runner
# modules. A second route to a granted module passes: the live
# test gaining `mrl.Manifest`, which core also owns, stays green. So a reason
# states the reach the grant checks, as modules, and
# names a route only as history ("at exemption time, through ..."), which a
# later route cannot make false. A reason that says "only through X" is a
# claim nothing checks.
#
# One entry, landed with the eight per-module files in W4's relocation.
# PITFALL: an entry cannot land before its file exists -- an entry whose file
# is not among the seam sources fails `exemption-names-no-such-test` ("not
# among the seam sources") -- and the natural "repair", skipping entries whose
# file is not among the sources, is an existence filter over this set, the
# class of hole `_seam_sources` refuses. Every exemption rule is also pinned,
# both halves, by the `test_reach_floor_*` synthetic probes, which pass their
# own exemptions; for the grant that means wider, narrower, and an empty grant
# against a test that reaches a module (an empty grant is not a wildcard).
_REACH_EXEMPTIONS: dict[tuple[str, str], tuple[frozenset[str], str]] = {
    ("tests/test_modal_preflight.py", "test_recording_volume_reload_restores_committed_run_root"):
    (frozenset({"core"}),
     "it tests the RecordingVolume test double, which lives with its only consumers, the "
     "preflight tests; it reaches only core (at exemption time, through mrl.STATUS_FILENAME)"),
    ("tests/test_modal_training.py", "test_signal_tests_fire_handlers_in_production_order"):
    (frozenset(), "gh#243: it is a static census of the training tests' own signal-seam doubles "
     "(`_signal_hooks`, `_interrupt_in_production_order` and the two allow-lists), read from "
     "its own file by AST; it never imports or reaches a runner module, and it lives here "
     "because the doubles it governs are defined here"),
}

# What to do about each kind of floor violation, keyed like the violations'
# first field. The manifest edit and the `GOVERNED_NAME_COUNT` rule are named
# because they are the part a first-time reader cannot guess.
#
# PITFALL on "reach-floor": the measured reach can be SHORT, and then moving the
# test is the wrong edit. The floor credits a module only as `mrl.X`, `<m>.X` or
# `binding_target("key")` (`_own_module_refs`), through the facade and module
# aliases that runner imports bind unconditionally at module level, in the
# test's file and in the file of each helper it reaches; move one into a
# function or under an `if`/`try` and a test loses reach it really has
# (tests/modal_test_helpers.py's imports carry the core file's manifest test to
# `core`). A runner name imported by itself credits nothing either, but the
# split test rejects that import before it runs the floor
# (`_runner_imports_the_floor_cannot_resolve`), so on the real tree this remedy
# never meets it. The remedy says to check the route first.
_FLOOR_REMEDIES = {
    "reach-floor":
    ("first, if the test does test its file's module, check the route rather than move the "
     "test: the floor credits a module only as `mrl.X`, `<module alias>.X` or "
     "`binding_target(\"key\")`, where the facade and module aliases are runner imports bound "
     "unconditionally at module level, in the test's file and in each reached helper's file, "
     "so restore any such import that moved into a function or under an if/try, and spell a "
     "runner name imported by itself as `mrl.X` or `<module alias>.X`. Otherwise move the test "
     "to the runner test file of the module it tests (RUNNER_TEST_FILES) and set its value in "
     "the seam manifest to that file -- a move changes no key, so GOVERNED_NAME_COUNT stays; "
     "or, if it tests a test double rather than a module, add a (file, test) entry to "
     "_REACH_EXEMPTIONS with the reach it is granted for and its reason"),
    "core-rule":
    ("a test in the core file may name only `core` in its own body: move it to the file of "
     "the module it names and set its seam manifest value to that file (GOVERNED_NAME_COUNT "
     "stays); the core rule has no exemptions"),
    "exemption-not-needed":
    "the test reaches its file's module now: delete its _REACH_EXEMPTIONS entry",
    "exemption-names-no-such-test":
    ("an exemption does not follow its test: re-key the entry to the (file, test) that "
     "exists, or delete it"),
    "exemption-without-reason":
    ("an exemption is legal only with a reason: write it, as a non-empty string, as the "
     "second item of the entry's value"),
    "exemption-malformed":
    ("write the entry's value as (granted, reason): granted is a frozenset of the "
     "RUNNER_MODULES names the test reaches (exactly its measured reach, possibly empty) and "
     "reason a non-empty string saying why the test belongs in its file"),
    "exemption-wider-than-granted":
    ("the test's reach is not the one its exemption was granted for, so the reason may be "
     "false. If the test now tests another module, move it to that module's runner test file, "
     "set its seam manifest value there (GOVERNED_NAME_COUNT stays) and delete the entry; "
     "otherwise re-read the reason against the reach it has now, and re-grant the entry with "
     "that reach and a true reason"),
}


def _bind_runner_alias(dotted, local, facade, subs):
    """Record `local` as a facade alias or a submodule alias if `dotted` names one.

    `dotted` is the full path the import binds `local` to (`_import_bindings`),
    so `import scripts.modal_runner as mrl`, `from scripts import modal_runner
    as mrl`, `from scripts.modal_runner import core` and `import
    scripts.modal_runner.core as core` all come through one rule. A name
    imported FROM a submodule (`from scripts.modal_runner.core import X`) is
    not a module alias and is not recorded. It only records: unbinding `local`
    first is the caller's job.
    """
    prefix = PACKAGED + "."
    if dotted == PACKAGED:
        facade.add(local)
    elif dotted.startswith(prefix) and dotted.removeprefix(prefix) in RUNNER_MODULES:
        subs[local] = dotted.removeprefix(prefix)


def _import_bindings(node):
    """[(local name, dotted path bound to it)] for one `import` or `from ... import`, in order.

    `import a.b as x` binds `x` to `a.b`, but `import a.b` binds `a` to `a`;
    `from a import b as x` binds `x` to `a.b`, a submodule or a plain name
    (`_bind_runner_alias` tells them apart). A relative import keeps its
    leading dots, so it never spells the runner. A star import is skipped:
    what it binds is not in the source (an approximation
    `_resolves_to_module_scope` lists). The binder walk and the module-level
    alias table both take an import's local names from here, so the two
    cannot disagree about what an import binds.
    """
    out = []
    for alias in node.names:
        if alias.name == "*":
            continue
        if isinstance(node, ast.Import):
            dotted = alias.name if alias.asname else alias.name.split(".")[0]
            out.append((alias.asname or dotted, dotted))
        else:
            dotted = "." * node.level + (f"{node.module}." if node.module else "") + alias.name
            out.append((alias.asname or alias.name, dotted))
    return out


def _runner_imports_the_floor_cannot_resolve(sources):
    """[(file, line, import)] per runner import in `sources` that binds no facade or module alias.

    The floor and the core rule read a runner module only through a name
    bound to the facade or to a whole `RUNNER_MODULES` submodule
    (`_bind_runner_alias`). An import that names the runner but binds anything
    else gives them nothing to resolve, in both directions at once: a
    correctly placed request test that calls
    `build_run_request` imported by itself reaches nothing and fails the
    floor, whose remedy then points at a move or an exemption; and a core-file
    test that calls it passes the core rule with every gate green. Rejecting
    the import makes both impossible, and it keeps the three resolution rules
    (`mrl.X`, `<sub>.X` and `binding_target(...)`) the only ones. The split
    test asserts this is empty over the seam files, before it runs the floor.

    Rejected, at any depth (a function-local import included): a name
    imported from the facade or from a submodule (`from scripts.modal_runner
    import build_run_request`, `from scripts.modal_runner.core import X`), a
    star import, an unaliased `import scripts.modal_runner[.<m>]` (it binds
    `scripts`), and an alias of a path that is not the facade or a runner
    module. Legal: `import scripts.modal_runner as mrl`, `from scripts import
    modal_runner [as x]`, `from scripts.modal_runner import <m> [as x]` and
    `import scripts.modal_runner.<m> as x`, wherever they sit, and every
    relative or non-runner import (`scripts.modal_runner_x` is not the
    runner: the prefix check keeps its dot).
    """
    prefix = PACKAGED + "."
    found = []
    for rel, text in sources.items():
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                paths = [(alias.name, alias.asname is not None,
                          f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
                         for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                paths = [(f"{node.module}.{alias.name}", True,
                          f"from {node.module} import {alias.name}") for alias in node.names]
            else:
                continue
            for path, binds_the_path, spelling in paths:
                if path != PACKAGED and not path.startswith(prefix):
                    continue
                if binds_the_path and (path == PACKAGED
                                       or path.removeprefix(prefix) in RUNNER_MODULES):
                    continue
                found.append((rel, node.lineno, spelling))
    return sorted(found)


# ── Scope resolution: does a name, where it is read, mean the module-level alias? ──
#
# A module alias rebound locally is not the module, and the floor must not
# credit reach through it. Every binder Python has can do the rebinding, so the
# resolver below models Python's own scoping rules rather than a list of the
# shapes seen so far. The census the floor was specified against used a lexical
# scope resolver cross-checked against `symtable`; this one is written to the
# same rules, not copied from it, and it reproduces the residual figures in
# `reach_floor_violations`' docstring. The previous rule here, "a store anywhere
# in the node shadows the name everywhere in it", was neither: it missed every
# binder that is not a `Name` store (a nested def's or a lambda's parameter, an
# `except ... as`, a nested import), which credits reach the test does not
# have, and it shadowed names Python does not (a class body's, a
# comprehension's target outside the comprehension).
#
# WHY NOT `symtable` DIRECTLY: under Python 3.12's inlined comprehensions
# (PEP 709) a list, set or dict comprehension gets no child table (a generator
# expression still does), so its target and a same-named read in the enclosing
# function become ONE symbol. Measured on 3.12.3, in a function whose body is
# `ys = [request.x for request in items]` then `return request.y`, `symtable`
# reports that one `request` as global, so the floor would credit
# `request.x`, a read of the comprehension target, as the module. PITFALL for
# whoever re-checks this: asked about `[request.x for request in items]` ALONE,
# `symtable` answers local, which is right; the failure needs the second read
# outside the comprehension. The walk below keeps every comprehension its own
# scope, as the language does on every supported version.
_FUNCTION_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_SCOPE_NODES = (*_FUNCTION_SCOPES, ast.ClassDef, *_COMPREHENSIONS)


def _scope_parts(node):
    """(parts evaluated in the ENCLOSING scope, parts evaluated in `node`'s own scope).

    For a scope-opening node. A def's decorators, argument defaults and
    annotations run where the `def` statement runs, and so do a class's
    decorators, bases and keywords: only the body is inside. (For a GENERIC
    def or class, PEP 695 runs the annotations, bases, keywords and the type
    parameters' bounds in a scope that also sees the type parameters; the
    walk does not model that, and `_resolves_to_module_scope` lists it among
    its approximations.) A comprehension's FIRST iterable is evaluated
    outside it; its targets, conditions, later iterables and element are
    inside. The fields are listed by hand, so a field left out here is one no
    reference rule sees, and a field put in the wrong list is resolved in the
    wrong scope: `test_the_scope_walk_visits_every_expression_ast_walk_visits`
    pins both, that the lists are complete and, by a frame table
    (`_EVERY_FIELD_FRAMES`), which scope each field is evaluated in. PITFALL:
    getting this split wrong mis-resolves exactly the spellings the floor
    most needs, such as a parametrize list naming `request.X` above a test
    that takes pytest's `request` fixture -- the list is evaluated at module
    scope, where `request` is the module.
    """
    if isinstance(node, _COMPREHENSIONS):
        first, *rest = node.generators
        inner = [first.target, *first.ifs]
        for generator in rest:
            inner += [generator.target, generator.iter, *generator.ifs]
        inner += [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
        return [first.iter], inner
    if isinstance(node, ast.ClassDef):
        outer = [*node.decorator_list, *node.bases, *(k.value for k in node.keywords)]
        return outer + list(node.type_params), node.body
    args = node.args
    outer = [*args.defaults, *(d for d in args.kw_defaults if d is not None)]
    if isinstance(node, ast.Lambda):
        return outer, [node.body]
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
    outer += [p.annotation for p in params if p is not None and p.annotation is not None]
    outer += [*node.decorator_list, *([node.returns] if node.returns else []), *node.type_params]
    return outer, node.body


def _comprehension_walrus_targets(comprehension):
    """Names `:=` binds inside `comprehension`, nested comprehensions included.

    PEP 572: a walrus inside a comprehension binds in the nearest enclosing
    scope that is NOT a comprehension, so these are bindings of the function
    (or module) around it, not of the comprehension.
    """
    found, stack = set(), list(_scope_parts(comprehension)[1])
    while stack:
        sub = stack.pop()
        if isinstance(sub, ast.NamedExpr):
            found.add(sub.target.id)
        if isinstance(sub, _SCOPE_NODES) and not isinstance(sub, _COMPREHENSIONS):
            stack.extend(_scope_parts(sub)[0])
        else:
            stack.extend(ast.iter_child_nodes(sub))
    return found


def _scope_binders(parts, *, comprehension=False):
    """(names bound, names declared `global`, names declared `nonlocal`) by one scope's own parts.

    `parts` are the nodes evaluated in the scope (a body, or one module-level
    statement). Every binder counts: assignment, augmented and annotated
    targets, `for`/`with`/`del` targets (all `Name` stores and deletes),
    `except ... as`, `import` and `from ... import` (the local name), a nested
    def's or class's own name, `:=` (including one inside a comprehension,
    which binds HERE unless this scope is itself a comprehension), and `match`
    captures. It does not descend into a nested scope's body, only into the
    parts of it that are evaluated here (`_scope_parts`). Parameters are the
    caller's job, because they are not in `parts`.
    """
    bound, declared_global, declared_nonlocal = set(), set(), set()
    stack = list(parts)
    while stack:
        sub = stack.pop()
        if isinstance(sub, ast.Global):
            declared_global.update(sub.names)
        elif isinstance(sub, ast.Nonlocal):
            declared_nonlocal.update(sub.names)
        elif isinstance(sub, ast.NamedExpr):
            if not comprehension:
                bound.add(sub.target.id)
            stack.append(sub.value)
            continue
        elif isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            bound.add(sub.id)
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            bound.add(sub.name)
        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            bound.update(local for local, _ in _import_bindings(sub))
        elif isinstance(sub, (ast.MatchAs, ast.MatchStar)) and sub.name:
            bound.add(sub.name)
        elif isinstance(sub, ast.MatchMapping) and sub.rest:
            bound.add(sub.rest)
        if isinstance(sub, _SCOPE_NODES):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(sub.name)
            if isinstance(sub, _COMPREHENSIONS) and not comprehension:
                bound |= _comprehension_walrus_targets(sub)
            stack.extend(_scope_parts(sub)[0])
            continue
        stack.extend(ast.iter_child_nodes(sub))
    return bound, declared_global, declared_nonlocal


def _scope_frame(node):
    """(is a class body, names local to the scope `node` opens, names it declares `global`).

    Local means bound in the scope and not declared `global` or `nonlocal`
    there: a `nonlocal` name is found in an enclosing function, and a `global`
    one is the module's.
    """
    is_comprehension = isinstance(node, _COMPREHENSIONS)
    bound, declared_global, declared_nonlocal = _scope_binders(_scope_parts(node)[1],
                                                               comprehension=is_comprehension)
    if isinstance(node, _FUNCTION_SCOPES):
        args = node.args
        params = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
        bound |= {param.arg for param in params if param is not None}
    local = bound - declared_global - declared_nonlocal
    return isinstance(node, ast.ClassDef), local, declared_global


def _scoped_walk(node):
    """Yield `(sub, chain)` for `node` and every node under it.

    `chain` is the tuple of `_scope_frame`s `sub` is evaluated in, innermost
    last; it is empty at module scope, which is where a module-level def's
    decorators and defaults are evaluated. A generic def or class (PEP 695)
    gets one more frame, outside its own, holding its type parameters, which
    are visible in its body and, unlike a class body's names, in its methods.
    Python also lets a generic def's annotations, a generic class's bases and
    keywords, and every type parameter's bound see them; this walk evaluates
    those in the enclosing chain, outside that frame (an approximation
    `_resolves_to_module_scope` lists).
    """
    # Annotated because the seed's empty chain `()` would otherwise be inferred
    # as the only chain type, and pyrefly then rejects every longer one.
    stack: list[tuple[ast.AST, tuple]] = [(node, ())]
    while stack:
        sub, chain = stack.pop()
        yield sub, chain
        if not isinstance(sub, _SCOPE_NODES):
            stack.extend((child, chain) for child in ast.iter_child_nodes(sub))
            continue
        outer, inner = _scope_parts(sub)
        inner_chain = chain
        if getattr(sub, "type_params", None):
            inner_chain += ((False, {param.name for param in sub.type_params}, set()), )
        inner_chain += (_scope_frame(sub), )
        stack.extend((part, chain) for part in outer)
        stack.extend((part, inner_chain) for part in inner)


def _resolves_to_module_scope(name, chain):
    """Does a load of `name`, evaluated in `chain` (innermost last), read the module-level binding?

    Python's rule: the innermost scope that binds the name owns it, except that
    a class body's names are invisible to the scopes nested inside it (a method
    reading `request` skips the class body's `request = ...`), and a `global`
    declaration sends the name straight to the module. `nonlocal` names are not
    local (`_scope_frame`), so the search moves outward to the function that
    binds them.

    THE APPROXIMATIONS, of this function and of the alias table it answers
    for (`_runner_aliases`), each with its direction. STRICTER means the floor
    credits less reach than the test has: a false red, which someone sees.
    LOOSER means it credits reach the test does not have: a false green, which
    nobody sees on the real tree, so each looser one names what else guards it.
    Measured at W4b, none of these shapes occurs in the ten seam files.
      * A class body reads a name it also binds through the class namespace
        first and the module second, so a read that runs BEFORE the class
        body's own binding still sees the module. Treated as local: stricter.
      * Every module-scope read is resolved against the aliases the module
        ENDS with. That is right for bodies, which run after the import has
        finished, but a module-level def's decorators, defaults and
        annotations (and a class's bases) run when the def runs. So a
        parametrize list naming `request.X` that sits after a non-runner
        binding of `request` and before a later runner import is credited
        (looser; ruff E402 rejects the late import), and one after the runner
        import, with `request` rebound further down, is not (stricter).
      * PEP 695 annotation scopes: a generic def's annotations and return
        annotation, a generic class's bases and keywords, every type
        parameter's bound or constraints (`def f[request, T: request.X]`),
        and a `type X[T] = ...` value see the type parameters, but
        `_scoped_walk` evaluates them outside the type-parameter frame. A type
        parameter spelled like an alias and read there is credited as the
        module: looser, and nothing but this note guards it. The frame table
        `_EVERY_FIELD_FRAMES` records where the walk evaluates each of them.
      * A module-level annotation with no value (`request: object`) binds
        nothing at run time, but it is a `Name` store, so it unbinds the
        alias: stricter.
      * A star import (`from x import *`) binds names the source does not
        show. It is skipped, so an alias it may overwrite stays: looser, and
        not knowable from the source; ruff F403 rejects it.
      * An import under a module-level `if`/`try`/`for`/`with` binds no alias
        (`_runner_aliases`), and neither does a function-local runner import,
        which instead hides a same-named module alias from that function.
        Both make the floor stricter and the core rule looser (a core-file
        test that imports `training` in its body and reads `training.X`
        escapes it).
      * A `global` rebinding of the name in any function or class body
        unbinds the alias for the whole module, even for reads that run
        before that function is ever called: stricter.
    """
    for depth, (is_class, local, declared_global) in enumerate(reversed(chain)):
        if is_class and depth:
            continue
        if name in declared_global:
            return True
        if name in local:
            return False
    return True


def _runner_aliases(tree):
    """(facade names, {alias: module}): the runner aliases one file's module scope ends up with.

    Aliases are per file: a helper in the shared file resolves `training.X`
    through the shared file's own imports, not through the importing test's.

    A name is an alias only if the LAST module-level statement binding it is an
    unconditional runner import: a test runs after its module has finished
    importing, so it sees the last binding in statement order. So `from tests
    import helpers as state` after `from scripts.modal_runner import state`
    unbinds `state`, and so does any other module-scope binding of the name
    that comes later: an assignment, a def, or anything under a module-level
    `if`/`try`/`for`/`with`. One import statement can bind a name twice
    (`from scripts.modal_runner import request, build_run_request as
    request`), so an import is read binding by binding, in order, and the
    name keeps its last one. A function or class body that declares the name
    `global` and binds it rebinds it at run time, so that unbinds it too,
    wherever it sits.

    PITFALL: an import inside a function body, or under a module-level
    `if`/`try`, binds no alias here, so a reference through it resolves to
    nothing, which makes the floor stricter and the core rule looser (see
    `_own_module_refs`). A function-local runner import is NOT MODELLED, by
    choice: modelling it exactly needs a per-function alias table that
    follows the last binding in flow order, and no seam file has one
    (measured at W4b: 0). A bare `import scripts.modal_runner` binds
    `scripts`, and `scripts.modal_runner.X` is likewise not resolved; the
    split test rejects that import in a seam file, as it rejects a runner
    name imported by itself (`_runner_imports_the_floor_cannot_resolve`).
    Every approximation of the resolution, with its direction, is listed in
    `_resolves_to_module_scope`.
    """
    facade, subs = set(), {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bindings = _import_bindings(node)
        else:
            bindings = [(local, None) for local in _scope_binders([node])[0]]
        for local, dotted in bindings:
            facade.discard(local)
            subs.pop(local, None)
            if dotted:
                _bind_runner_alias(dotted, local, facade, subs)
    for scope in ast.walk(tree):
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound, declared_global, _ = _scope_binders(_scope_parts(scope)[1])
            for local in bound & declared_global:
                facade.discard(local)
                subs.pop(local, None)
    return facade, subs


def _binding_target_key(call):
    """The literal site key of a `binding_target("key")` call, else None.

    Only the bare-name spelling counts, as in the binding census
    (tests/test_modal_patch_binding_census.py matches `func.id` the same way),
    so the two instruments agree on what a `binding_target` call is.
    """
    if getattr(call.func, "id", None) != "binding_target" or not call.args:
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def _own_module_refs(node, aliases, owners):
    """[(module, spelling)] for each reference in `node`'s OWN body that resolves to a runner module.

    The three resolution rules, and nothing else:
      * `<facade>.X` (`mrl.X`) resolves to X's owner in the tables' MANIFEST;
      * `<sub>.X` resolves to `sub`, where `<sub>` is a module-level alias of
        `scripts.modal_runner.<sub>` in the file that defines `node`;
      * `binding_target("key")` resolves to `BINDING_SITES[key]`'s owner.
    Decorators, argument defaults and annotations count, resolved at the scope
    that evaluates them (`_scope_parts`), so a parametrize list that names
    `mrl.X` counts. An alias counts only where the name READS the module-level
    binding (`_resolves_to_module_scope`): a parameter, an `except ... as`, a
    comprehension or `for`/`with` target, a walrus, a nested import or def, or
    any other local binding of the same name at any depth is not the module.
    The case is live in the runner test files, where a test binds `request =
    mrl.build_run_request(...)` over the `request` submodule alias and reads
    `request.run_id`; a test that takes pytest's `request` fixture shadows it
    the same way.

    PITFALL: these rules are narrower than "anything that touches the module".
    A bare alias passed as a value (`monkeypatch.setattr(training, "x", f)`),
    `mrl.<submodule>.X`, a facade name the tables do not own, and a
    non-literal or unknown site key all resolve to nothing. That makes the
    floor stricter but the core rule LOOSER: a core-file test whose only
    non-core reference takes one of these shapes passes the core rule. A
    runner name imported by itself (`from scripts.modal_runner import
    build_run_request`, then `build_run_request(...)`) would be one more such
    shape, but the split test rejects the import in every seam file
    (`_runner_imports_the_floor_cannot_resolve`) before it runs these rules.
    `binding_target` is matched by its bare name wherever it appears, as the
    binding census matches it, without resolving that name's scope.
    """
    facade, subs = aliases
    refs = []
    for sub, chain in _scoped_walk(node):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
            root = sub.value.id
            module = owners.get(sub.attr) if root in facade else subs.get(root)
            if module and _resolves_to_module_scope(root, chain):
                refs.append((module, f"{root}.{sub.attr}"))
        elif isinstance(sub, ast.Call):
            key = _binding_target_key(sub)
            if key in BINDING_SITES:
                refs.append((BINDING_SITES[key][0], f'binding_target("{key}")'))
    return refs


def _floor_reach(sources, manifest):
    """{(file, test): (own refs, modules reached)} for every test defined in `sources`.

    THE CLOSURE'S NAMESPACE IS THE UNION of every file in `sources` -- the
    same one `classify_seam` builds -- so a helper that sits in the shared file
    still carries its reach to a test in any runner file. A per-file closure
    fails tests whose only route to their module is a shared helper (measured
    on the declared placement: the core test that reaches `core` only through
    `_make_manifest`, once that helper moves to the shared file), and the
    natural "fix" for that, a second exemption, would pass minimality.

    A test's reach is its own references plus the own references of every
    module-level name it reaches transitively along `_referenced_module_names`
    edges. Each name's references resolve through the aliases of the file that
    defines it. `manifest` is the tables' {"<module>.py": [names it owns]},
    passed in so the probes and measurement scripts can supply their own.

    THE EDGES ARE NOT SCOPE-RESOLVED, and that cuts both ways. Own references
    are (`_own_module_refs`), but the helper edges come from
    `_referenced_module_names`, which reads every load of a module-level name
    as that name. So a local variable that shares a helper's name credits the
    helper's reach (a test binding `_make_manifest = 1` and reading it is
    credited with `_make_manifest`'s `core`), which is the looser direction:
    it can pass a misplaced test. The same property is what makes
    pytest fixtures count: a test that takes the module-level fixture
    `fake_modal` as a parameter and uses it reaches the fixture's body, which
    is right, because pytest injects that fixture there. A side-effect fixture,
    one a test takes as a parameter but never reads in its body, gives no edge,
    because a parameter is not a load.
    """
    owners = {name: f.removesuffix(".py") for f, names in manifest.items() for name in names}
    trees = {rel: ast.parse(text) for rel, text in sources.items()}
    aliases = {rel: _runner_aliases(tree) for rel, tree in trees.items()}
    nodes, own, tests = {}, {}, {}
    for rel, tree in trees.items():
        for name, node in _module_level_names(tree).items():
            nodes[name] = node
            own[name] = _own_module_refs(node, aliases[rel], owners)
            if _is_test_def(name, node):
                # Kept per (file, test), not read back from `nodes`/`own`: two
                # files that define one test name (a probe's keying case) must
                # each keep their own node and references.
                tests[(rel, name)] = (node, own[name])
    universe = set(nodes)
    edges = {n: _referenced_module_names(node, n, universe) for n, node in nodes.items()}
    reach = {}
    for (rel, name), (node, refs) in tests.items():
        reached = {module for module, _ in refs}
        for helper in _closure(_referenced_module_names(node, name, universe), edges):
            reached |= {module for module, _ in own[helper]}
        reach[(rel, name)] = (refs, reached)
    return reach


def _exemption_violations(pair, entry, reach, file_module, source_files):
    """The violations of one `_REACH_EXEMPTIONS`-shaped entry, as a list (at most one).

    SHAPE FIRST. `entry` must be a `(granted, reason)` pair. The bare reason
    string of the older shape is refused rather than unpacked, since a
    two-character string would unpack. The reason must be a non-empty string:
    `None` or `0` is not a reason, although `str()` of each is. `granted` must
    be a frozenset of `RUNNER_MODULES` names: a mutable set can change after
    it was granted, and a misspelt module is a grant nothing can
    ever match.

    An entry that names no examined test fails for one of three reasons, and
    the detail says which, because each wants a different edit: the file is not
    a per-module runner test file at all; it is one, but it is not among the
    sources (an entry that landed before its file); or it is among them and does
    not define the test (an entry left behind by a move).

    Then minimality (the test must not reach its file's module) and the grant:
    the test's whole reach, own body plus helpers as the floor measures it, must
    EQUAL `granted`. Wider is the failure the grant exists for (the test now
    tests another module, and the reason is false); narrower fails too, because
    the reason then describes a route the test no longer takes.
    """
    rel, test = pair
    if not isinstance(entry, tuple) or len(entry) != 2:
        return [("exemption-malformed", rel, test,
                 f"its value is {entry!r}, not a (granted, reason) pair")]
    granted, reason = entry
    if not isinstance(reason, str) or not reason.strip():
        return [("exemption-without-reason", rel, test,
                 f"its reason is {reason!r}, not a non-empty string")]
    if not isinstance(granted, frozenset) or not granted <= set(RUNNER_MODULES):
        return [("exemption-malformed", rel, test,
                 f"its granted reach is {granted!r}, not a frozenset of RUNNER_MODULES names")]
    if rel not in file_module:
        return [("exemption-names-no-such-test", rel, test,
                 f"{rel} is not a per-module runner test file (RUNNER_TEST_FILES), so no test "
                 "there is examined or exempt")]
    if rel not in source_files:
        return [("exemption-names-no-such-test", rel, test,
                 f"{rel} is not among the seam sources; an entry lands with its file")]
    if pair not in reach:
        return [("exemption-names-no-such-test", rel, test, f"{rel} does not define {test}")]
    reached = reach[pair][1]
    if file_module[rel] in reached:
        return [("exemption-not-needed", rel, test, f"it reaches {file_module[rel]!r}")]
    if reached != granted:
        return [("exemption-wider-than-granted", rel, test,
                 f"it reaches {sorted(reached)}, granted {sorted(granted)}: beyond the grant "
                 f"{sorted(reached - granted)}, granted but no longer reached "
                 f"{sorted(granted - reached)}")]
    return []


def reach_floor_violations(sources, manifest, exemptions):
    """Check the reach floor and the core rule; return `(violations, examined)`.

    `sources` is {path: source text} -- the union namespace, so pass every seam
    file, not just the runner files. `manifest` is the tables' owner map
    (`RUNNER_OWNERS`); `exemptions` is {(file, test): (granted, reason)}, the
    shape of `_REACH_EXEMPTIONS`. Pure: it reads no file, so the probes can call
    it on synthetic maps whose paths need not exist.

    WHAT IS EXAMINED: every module-level `test_` function in a file of
    `RUNNER_TEST_FILES` (tests/test_modal_<m>.py for m in RUNNER_MODULES). No
    other file is: not the client file, and not the shared file, which name no
    module. `examined` is that set of (file, test) pairs, returned so the gate
    can assert its scope (`_floor_scope_problems`) -- a floor that iterates a
    partial file list otherwise passes every other check. A test nested under
    a module-level `if` or `try` is not examined, and nothing here objects:
    `classify_seam`, which the split test calls on the same sources first,
    rejects it.

    THE RULES. Violations are (rule, file, test, detail) tuples, sorted within
    each pass, and `_FLOOR_REMEDIES` says what to do about each rule:
      * "reach-floor": the test does not reach its file's module through its
        own body or any helper it reaches, and is not exempt;
      * "core-rule": in the core file only, the test's OWN body names a module
        other than `core`. `core` is reached by most runner tests, so the floor
        alone does not police that file. The core rule has no exemptions.
      * per exemption entry (the dict is iterated, not the examined pairs):
        "exemption-malformed" if the value is not a (granted, reason) pair or
        granted is not a frozenset of RUNNER_MODULES names; "exemption-without-
        reason" if the reason is not a non-empty string; "exemption-names-no-
        such-test" if that file is not a runner test file among the sources or
        does not define that test; "exemption-not-needed" (minimality) if the
        test reaches its file's module; "exemption-wider-than-granted" if its
        reach is not exactly the granted set. `_exemption_violations` has why.

    RESIDUAL: A FLOOR, NOT A PLACEMENT CHECK. A test that reaches two modules
    can sit in either file with both rules green; choosing between them is the
    judgement the manifest diff records. A separate AST reach instrument
    measured these figures on the unsplit file's reference graph (138 tests,
    before the process-group guard test was added), and this module's
    `_floor_reach` reproduced every one on that tree and on the tree with the
    guard test (139). RE-MEASURED on the eight per-module files after W4's
    split (2026-09-24, at a06761d), by calling this function on alternative
    placements, each test moved with its file's runner imports: every figure
    below held unchanged, and the guard test still has no other legal file:
      * 48 of the 138 tests have at least one other file where the floor and
        the core rule both stay green;
      * 13 training tests name no non-core module in their own body, so they
        could sit in the core file green (`test_heartbeat_commits_every_60s_
        while_training` is one), and so could the one exempt preflight test;
      * 23 training tests reach `state`, so they could sit in the state file
        green (`test_failed_cleanup_commit_does_not_let_redelivery_write` is
        one);
      * the core rule rejects most, not all, of the maximally wrong split:
        moving every core-reaching test into the core file (95 moves) fails 81
        of them, where the floor alone fails none;
      * the placement the replaced rule produces (each test in the file of the
        non-core module it references most) fails only 3 tests: two CUDA-probe
        tests on the floor and one interrupt test on the core rule;
      * 0 of the 49 request tests reach training, so a request test placed in
        the training file is rejected.
    So after the relocation, a misplaced new test is caught only if it breaks
    the floor or the core rule.
    """
    file_module = dict(zip(RUNNER_TEST_FILES, RUNNER_MODULES, strict=True))
    reach = _floor_reach(sources, manifest)
    examined = {pair for pair in reach if pair[0] in file_module}
    violations = []
    for rel, test in sorted(examined):
        refs, reached = reach[(rel, test)]
        module = file_module[rel]
        if module not in reached and (rel, test) not in exemptions:
            violations.append(
                ("reach-floor", rel, test, f"reaches {sorted(reached)}, not {module!r}"))
        foreign = sorted({spelling for owner, spelling in refs if owner != "core"})
        if module == "core" and foreign:
            violations.append(("core-rule", rel, test, f"its own body names {foreign}"))
    for pair, entry in sorted(exemptions.items()):
        violations.extend(_exemption_violations(pair, entry, reach, file_module, set(sources)))
    return violations, examined


def _floor_scope_problems(examined, manifest):
    """The floor's scope check: its problems, empty when the floor read the manifest's runner half.

    `examined` is `reach_floor_violations`' second value; `manifest` is the seam
    manifest ({name: file}). Every other check on the floor passes on the real
    tree, because every test there already passes it, so a floor that read a
    partial file list would stay green. BOTH SIDES are module-level `test_`
    definitions: a test nested under a module-level `if` or `try` is in
    neither, so this check cannot see one.
    `classify_seam` rejects such a test before the split test gets here.
    Two clauses, each naming its own cause:

    * the examined pairs must EQUAL the manifest's runner-half tests: every
      `test_` entry NOT valued at the client file or the packaging file (the
      `SEAM_GUARDS`). Not "every `test_` entry valued at a file of
      `RUNNER_TEST_FILES`": that population was blind to its own scope. A
      runner set widened at the seam's call sites puts a test in a file that
      the floor never examines and that filter never counts, so both sides
      lost it together and the check stayed green.
    * every file of `RUNNER_TEST_FILES` must hold at least one test in the
      manifest. It is read off the manifest, not off `examined`: once the first
      clause holds the two agree, so this fires only for a file that really
      holds no module-level test, and its message says that rather than
      blaming the floor for skipping the file, which is the first clause's
      diagnosis.
    """
    runner_tests = {(rel, name)
                    for name, rel in manifest.items()
                    if name.startswith("test_") and rel not in (CLIENT_FILE, PACKAGING_FILE)}
    problems = []
    if examined != runner_tests:
        problems.append(
            "the reach floor did not examine exactly the manifest's runner-half tests, so it "
            "passes over a different population than the one it certifies. examined-only="
            f"{sorted(examined - runner_tests)[:10]} manifest-only="
            f"{sorted(runner_tests - examined)[:10]}. A manifest-only test in a file outside "
            "RUNNER_TEST_FILES is read by the seam and not by the floor: runner tests go in "
            "tests/test_modal_<m>.py, one file per module in the tables")
    empty = sorted(set(RUNNER_TEST_FILES) - {rel for rel, _ in runner_tests})
    if empty:
        problems.append(
            f"{empty} hold(s) no test in the seam manifest: every runner test file must hold at "
            "least one test of its module; add one, or remove the module from the tables")
    return problems


def _manifest_edits(*, add=None, delete=None, move=None):
    """The exact seam-manifest edits a gate failure calls for, with the `GOVERNED_NAME_COUNT` rule.

    `add` and `move` are {name: file}; `delete` is an iterable of names. The
    count rule is spelled out because it is the step people miss: adding or
    deleting a key moves `GOVERNED_NAME_COUNT` by the same amount, in the same
    commit, with the reason in the commit message; a move changes no key and
    leaves it alone. The file is written with
    `json.dumps(d, indent=2, sort_keys=True) + "\\n"`.
    """
    add, delete, move = add or {}, sorted(delete or ()), move or {}
    rel = MANIFEST.relative_to(ROOT).as_posix()
    lines = [f'add "{name}": "{dest}"' for name, dest in sorted(add.items())]
    lines += [f'delete "{name}"' for name in delete]
    lines += [f'set "{name}": "{dest}"' for name, dest in sorted(move.items())]
    if not lines:
        return f"{rel} needs no edit and GOVERNED_NAME_COUNT stays."
    delta = len(add) - len(delete)
    count = ("GOVERNED_NAME_COUNT stays: these edits add as many keys as they delete (a move "
             "changes no key)" if delta == 0 else
             f"change GOVERNED_NAME_COUNT by {delta:+d} in the same commit and say why")
    return f"in {rel}: {'; '.join(lines)}. Then {count}."


def _describe_floor_violations(violations):
    """One line per violation, with its rule's remedy from `_FLOOR_REMEDIES`."""
    return "\n".join(f"  [{rule}] {rel}::{test}: {detail}. Remedy: {_FLOOR_REMEDIES[rule]}"
                     for rule, rel, test, detail in violations)


# A synthetic module that exercises all four answers `classify_seam` can give,
# with the answers known by construction rather than measured off the monolith.
# WHY a synthetic probe and not just the real file: the real file certifies the
# classifier against a seam we already knew, which is the weakest kind of
# evidence there is. This certifies the RULE. It is also the only part of this
# section that survives Task 4 unchanged -- the "0 names on the wrong side of
# line 3638" check that certified the classifier against the monolith is a
# one-off by construction, because after the split there is no line 3638 to
# measure against. That check's successor is Task 4's placement gate, which
# swaps position for which-file as the ground truth.
#
# `_import_run_modal` and `_import_backfill` are deliberately NOT defined here:
# a seed does not have to be a module-level binding to seed, and leaving them
# undefined keeps them out of `universe`, so the only signal reaching the
# classifier is the one each probe test is named for.
#
# THERE IS ONE PROBE TEST PER MEMBER of `_CLIENT_BINDINGS` and `_CLIENT_MODULES`,
# and that is not thoroughness for its own sake. A guard whose own allow-set is
# unwatched is this repo's most-repeated defect, and it bit here: the first
# version of this probe named `_import_run_modal` and `scripts.modal_artifacts`
# and nothing else, and deleting `"module"` from `_CLIENT_BINDINGS` was measured
# to leave BOTH tests in this section green.
#
# AND ONE PER MODULE-LEVEL NODE KIND, for the same reason one level down. Review
# measured that this probe parsed to `['FunctionDef']` and nothing else -- no
# `ClassDef`, no `Assign`/`AnnAssign`, no `SEAM_HEADER_NAMES` name -- and ran
# four mutations across three of `classify_seam`'s decision surfaces that the
# probe therefore could not see: dropping `ast.ClassDef` from `_DEFS` (14 classes
# lose a destination), dropping the `Assign` census (5 names lose one), emptying
# `SEAM_HEADER_NAMES`, and adding a governed name to it. Each was caught by the
# MANIFEST AGREEMENT test ALONE -- and that test's own docstring concedes it is
# green on a coordinated edit that regenerates the manifest in the same commit.
# So those surfaces were guarded only by the instrument such an edit rides in on.
# Two more were added here for the same reason review found the first three:
# `AnnAssign` is a separate branch from `Assign`, and `SEAM_GUARDS` had an
# assertion that could not fail.
_SEAM_CLASSIFIER_PROBE = '''
ROOT = "module-header boilerplate: SEAM_HEADER_NAMES, governed by nobody"

_PROBE_PINNED_DIGEST = "sha256:0000"

_PROBE_TIMEOUT_S: int = 5


class _ProbeSharedDouble:
    """A module-level CLASS, reached from both halves."""


def _shared_helper():
    return 1

def _runner_only_helper():
    return _shared_helper(), _PROBE_TIMEOUT_S

def _client_only_helper():
    module = _import_run_modal()
    return module, _PROBE_PINNED_DIGEST

def _guard_and_client_helper():
    return 6

def test_probe_runner_via_helper():
    return _runner_only_helper()

def test_probe_runner_via_shared():
    return _shared_helper(), _ProbeSharedDouble(), ROOT

def test_probe_client_direct():
    return _import_run_modal()

def test_probe_client_by_image_reqs():
    return _run_modal_image_reqs("lock")

def test_probe_client_by_module_binding():
    module = _import_backfill()
    return module.App

def test_probe_client_transitive():
    return _client_only_helper(), _guard_and_client_helper()

def test_probe_client_via_shared():
    return _client_only_helper(), _shared_helper(), _ProbeSharedDouble()

def test_probe_client_by_artifacts_string():
    return "scripts.modal_artifacts"

def test_probe_client_by_run_modal_string():
    return "import scripts.run_modal as rm"

def test_probe_client_by_sidecar_string():
    return "scripts.modal_backfill_sidecar"

def test_modal_is_an_explicit_dependency_group():
    return _import_run_modal(), _guard_and_client_helper()
'''


def test_the_seam_classifier_places_a_planted_name_by_its_reference_graph():
    """Positive control for `classify_seam`, on a module whose answer is known.

    Each of the four destinations is exercised by a name that can only land
    there for the stated reason:

    - six SINGLE-SIGNAL cases -- `_direct`, `_by_image_reqs`,
      `_by_module_binding` and the three `_by_*_string` -- carry exactly one
      client signal each, one per member of `_CLIENT_BINDINGS` and
      `_CLIENT_MODULES`. Deleting any member of either set turns one of them red
      by name; all six deletions were run and each has an objector.
    - `test_probe_client_transitive` and `test_probe_client_via_shared` carry no
      signal at all and are client purely because they call something that is.
      That is the whole reason this is a closure and not a marker list, and the
      closure is what covers `"module"` on the monolith: measured, deleting
      `"module"` from `_CLIENT_BINDINGS` moves 0 names and disabling the closure
      moves 0 names, but doing BOTH moves 30, 13 of them tests. Two mechanisms
      covering the same 13 tests each score dead in a one-at-a-time census.
      Neither is.
    - `_runner_only_helper` carries no marker at all -- no helper does -- and is
      runner because only runner tests reach it.
    - `_shared_helper` is reached from BOTH halves, so it goes to the shared
      module. A classifier forced to pick a half would strand it in whichever
      file lost, which is a NameError at run time and not a collection error, so
      neither `--collect-only` nor a name-set comparison would see it.
    - `_guard_and_client_helper` is reached by a client test and by the planted
      guard. A guard is not a seam consumer (`_helper_destination` drops the
      packaging file), so it goes to the client file; a classifier that counted
      the guard would send it to the shared file. A helper that ONLY a guard
      reaches has no seam consumer at all and raises, with a message that says
      so (the last case below). So do a pytest `class TestX` and a `def
      testfoo()`, whose messages must say that the seam reads only
      module-level `test_` functions, where an orphan helper's must not.

    AND ONE CASE PER MODULE-LEVEL NODE KIND, which is the second half of this
    test and the more easily lost one. Review measured that the probe parsed to
    `['FunctionDef']` and nothing else, so four of `classify_seam`'s decision
    surfaces had no objector here and were caught by the manifest agreement test
    ALONE -- the one instrument whose own docstring concedes it is green on a
    coordinated edit that regenerates the manifest in the same commit. Each of
    the four now turns THIS test red, measured by running the mutation:

      `_DEFS` drops `ast.ClassDef`          -> `_ProbeSharedDouble` unplaced
      `_module_level_names` drops `Assign`  -> `_PROBE_PINNED_DIGEST` unplaced
      ... drops `AnnAssign`                 -> `_PROBE_TIMEOUT_S` unplaced
      `SEAM_HEADER_NAMES` emptied           -> `ROOT` gains a destination
      ... gains a governed name             -> the pinned set differs

    `AnnAssign` gets its own case because it is its own branch: removing only it
    leaves the `Assign` control green, and that mutation is the one where this
    test is the SOLE objector -- the agreement test does not fire, because the
    monolith has no module-level `AnnAssign` for it to lose.

    PITFALL this control exists for: `classify_seam` assigns `SEAM_GUARDS`
    unconditionally, from the constant and before it reads any source. So the
    guard names appear in the result for ANY input, including a source that
    defines none of them. A reader who assumes those were computed will misread
    every other result in this file -- and an earlier revision of this test
    enshrined the confusion, asserting `{destinations[n] for n in SEAM_GUARDS} ==
    {PACKAGING_FILE}`, which iterates the same constant the dict was built from
    and therefore cannot fail. The replacement plants one guard name in the probe
    as a test whose own body would classify it CLIENT, so the OVERRIDE is what is
    measured, and pins the set so that adding or removing a member fires.
    """
    # The probe is keyed as a declared runner file: under `classify_seam`'s
    # rejection rule a runner-half test in an undeclared file raises, so a
    # made-up key would reject the probe's own runner tests.
    probe_file = RUNNER_TEST_FILES[0]
    destinations, defined_in = classify_seam({probe_file: _SEAM_CLASSIFIER_PROBE},
                                             runner_files=RUNNER_TEST_FILES)

    # One per member of _CLIENT_BINDINGS, then one per member of _CLIENT_MODULES.
    assert destinations["test_probe_client_direct"] == CLIENT_FILE
    assert destinations["test_probe_client_by_image_reqs"] == CLIENT_FILE
    assert destinations["test_probe_client_by_module_binding"] == CLIENT_FILE, (
        "a test whose only client signal is binding and dereferencing `module` "
        "was classified as runner. On the monolith 13 client tests carry that "
        "and no other signal in their own body.")
    assert destinations["test_probe_client_by_artifacts_string"] == CLIENT_FILE
    assert destinations["test_probe_client_by_run_modal_string"] == CLIENT_FILE
    assert destinations["test_probe_client_by_sidecar_string"] == CLIENT_FILE

    assert destinations["test_probe_client_transitive"] == CLIENT_FILE, (
        "a test that reaches the client only THROUGH a helper was classified as "
        "runner, so the transitive closure is not running and the classifier has "
        "degenerated into the marker list it was built to replace")
    assert destinations["test_probe_client_via_shared"] == CLIENT_FILE
    assert destinations["_client_only_helper"] == CLIENT_FILE

    assert destinations["test_probe_runner_via_helper"] == probe_file
    assert destinations["test_probe_runner_via_shared"] == probe_file
    assert destinations["_runner_only_helper"] == probe_file, (
        "a helper reached only by runner tests was not sent to the runner half; "
        "stage 2 follows consumers and this one has only runner consumers")

    assert destinations["_shared_helper"] == SHARED_FILE, (
        "a helper reached from BOTH halves was forced into one of them. That is "
        "the stranding this third destination exists to prevent.")
    assert destinations["_guard_and_client_helper"] == CLIENT_FILE, (
        "a helper reached by a client test and by a SEAM_GUARDS test was not sent to the client "
        "file, so the guard was counted as a seam consumer: `_helper_destination` must drop "
        f"the packaging file. got={destinations['_guard_and_client_helper']!r}")

    # ── the module-level node kinds the census must cover ────────────────────
    # `.get` rather than `[...]`: the failure these three exist for is the name
    # being ABSENT from the census, and a KeyError says that far less clearly
    # than the message does.
    assert destinations.get("_ProbeSharedDouble") == SHARED_FILE, (
        "a module-level CLASS reached from both halves was not placed where its "
        "consumers say. If it is missing entirely, `_DEFS` has stopped covering "
        "`ast.ClassDef` and the monolith's 14 classes have no destination at "
        f"all. got={destinations.get('_ProbeSharedDouble')!r}")
    assert destinations.get("_PROBE_PINNED_DIGEST") == CLIENT_FILE, (
        "a module-level ASSIGN reached only by client tests was not placed. If "
        "it is missing entirely, `_module_level_names` has stopped walking "
        "`ast.Assign` -- the branch whose whole point is that a pinned CUDA "
        "digest must not be copied into both files and left to drift. "
        f"got={destinations.get('_PROBE_PINNED_DIGEST')!r}")
    assert destinations.get("_PROBE_TIMEOUT_S") == probe_file, (
        "a module-level ANNASSIGN reached only by runner tests was not placed. "
        "`_module_level_names` handles AnnAssign in a SEPARATE branch from "
        "Assign, so it needs its own control; the Assign control above stays "
        f"green when only this branch is removed. got={destinations.get('_PROBE_TIMEOUT_S')!r}")

    # ── SEAM_HEADER_NAMES: ungoverned must not mean invisible ────────────────
    # The unrelated `ROOT` definers are read off disk inside the message, so it
    # names today's files instead of a count that grows with every new test file.
    seam_files = {*RUNNER_TEST_FILES, CLIENT_FILE, SHARED_FILE, PACKAGING_FILE}
    assert "ROOT" not in destinations, (
        "`ROOT` was given a destination, so `SEAM_HEADER_NAMES` has stopped "
        "exempting it. Governing it makes the seam's scan collide with every "
        "unrelated test file that defines its own `ROOT`; today those are "
        f"{sorted(set(_names_defined_under_tests().get('ROOT', [])) - seam_files)}. "
        "The comment above `SEAM_HEADER_NAMES` has the history.")
    assert defined_in.get("ROOT") == [
        probe_file
    ], ("`ROOT` fell out of `defined_in` as well as out of `destinations`. "
        "Ungoverned must not mean invisible -- Task 4's placement gate reads "
        "this to prove `ROOT` survives into every destination file, so an "
        f"exemption that also hides it is how `ROOT` gets lost. got={defined_in.get('ROOT')!r}")
    assert SEAM_HEADER_NAMES == frozenset(
        {"ROOT"}), ("the exemption set changed. Every name in it is silently ungoverned, so "
                    "an ADDITION here deletes a name from the seam with nothing else "
                    f"objecting -- which is why the set is pinned. got={sorted(SEAM_HEADER_NAMES)}")

    # ── SEAM_GUARDS overrides computed concern, and that IS testable ─────────
    # The previous revision asserted `{destinations[n] for n in SEAM_GUARDS} ==
    # {PACKAGING_FILE}` and called it a check. It cannot fail: `classify_seam`
    # writes that dict from the constant before it reads any source, and the set
    # comprehension iterates the same constant. An assertion that cannot fail is
    # not an assertion. The probe now DEFINES one guard, as a test whose own body
    # would classify it CLIENT, so the override is measured instead of restated.
    planted = "test_modal_is_an_explicit_dependency_group"
    assert planted in SEAM_GUARDS and defined_in.get(planted) == [probe_file], (
        "the planted guard is not both in SEAM_GUARDS and defined by the probe, "
        "so the override assertion below is measuring the wrong thing")
    assert destinations[planted] == PACKAGING_FILE, (
        f"{planted} names a client binding in its own body, so concern alone "
        "would send it to the client half. It is PACKAGING only because "
        "SEAM_GUARDS overrides the computation, and that override is now gone. "
        f"got={destinations[planted]!r}")
    assert SEAM_GUARDS == frozenset({
        "test_modal_is_an_explicit_dependency_group",
        "test_local_entrypoints_do_not_import_modal",
        "test_modal_runner_does_not_import_modal_or_torch",
    }), ("the guard set changed. Removing a member hands that test back to the "
         "computed seam and adding one takes a test out of it; neither shows up "
         f"anywhere else in this test. got={sorted(SEAM_GUARDS)}")
    # The other two are assigned by fiat, from the constant, with nothing in the
    # source defining them. Asserted so nobody reads the override result above as
    # evidence about names the probe never planted.
    unplanted = sorted(SEAM_GUARDS - {planted})
    assert all(name not in defined_in for name in unplanted), (
        f"the probe defines {unplanted}, so their destinations are no longer the "
        "assigned-by-fiat case this asserts")
    assert {destinations[name] for name in unplanted} == {PACKAGING_FILE}

    # A name no test reaches has no consumers to follow, so stage 2 cannot place
    # it. It raises rather than defaulting, because the two things it can be --
    # dead code, or a new entry point -- want opposite answers.
    with pytest.raises(ValueError, match="reached by no seam test") as orphan:
        classify_seam({"orphan.py": "def _reached_by_nothing():\n    return 1\n"},
                      runner_files=RUNNER_TEST_FILES)
    # A pytest test CLASS raises the same way (only module-level `test_` defs
    # are tests, and nothing reaches the class), but "dead code" points the
    # wrong way for the most common pytest idiom, so its message must say what
    # to do with a test class -- and an orphan helper's must not.
    test_class = "class TestRunIdEdges:\n    def test_empty_run_id(self):\n        return 1\n"
    with pytest.raises(ValueError, match="reached by no seam test") as rejected:
        classify_seam({probe_file: test_class}, runner_files=RUNNER_TEST_FILES)
    assert "supports only module-level `test_` functions" in str(rejected.value), (
        f"a pytest test class was rejected without saying the seam reads only module-level "
        f"`test_` functions: {rejected.value}")
    # So does a `def testfoo()`: pytest collects the prefix `test` (this repo
    # sets no `python_functions`), but the seam reads only `test_`.
    with pytest.raises(ValueError, match="reached by no seam test") as rejected:
        classify_seam({probe_file: "def testrunidedges():\n    return 1\n"},
                      runner_files=RUNNER_TEST_FILES)
    assert "supports only module-level `test_` functions" in str(rejected.value), (
        f"a pytest test function without the underscore was rejected as dead code, without "
        f"saying the seam reads only module-level `test_` functions: {rejected.value}")
    orphan_message = str(orphan.value)
    assert "pytest test" not in orphan_message, (
        f"an orphan helper's error talks about pytest tests: {orphan_message}")
    # A helper only a guard reaches has a consumer, but no SEAM consumer, so it
    # raises too, and the message must not claim that nothing reaches it.
    guard_only = ("def _reached_by_a_guard_only():\n    return 1\n\n"
                  f"def {planted}():\n    return _reached_by_a_guard_only()\n")
    with pytest.raises(ValueError, match="reached by no seam test") as rejected:
        classify_seam({probe_file: guard_only}, runner_files=RUNNER_TEST_FILES)
    assert "SEAM_GUARDS test does not count" in str(rejected.value), (
        f"the guard-only helper's error does not say why a guard's reach is ignored: "
        f"{rejected.value}")


def _probe_test_names(sources):
    """The module-level `test_` function names defined in a {path: source} map, in order."""
    return [
        name for text in sources.values()
        for name, node in _module_level_names(ast.parse(text)).items() if _is_test_def(name, node)
    ]


def test_the_seam_classifier_places_runner_tests_by_their_declared_file():
    """Positive control for `classify_seam` over a declared set of SEVERAL runner files.

    The single-file probe above cannot tell "the runner file" from "the runner
    file that defines this test", nor "reached from both halves" from "reached
    from two seam files", because with one runner file they coincide. This
    source set has three runner files and the client file, all
    `RUNNER_TEST_FILES` paths or seam constants (they need not exist on disk):

    - each runner test's destination is the declared file that defines it;
    - a helper reached from TWO RUNNER FILES, and no client test, goes to the
      shared file -- the generalised rule, which the old two-halves rule would
      have sent to "the" runner file;
    - a helper reached from one runner file only goes to that file;
    - a runner-half test in a file outside the declared set -- here the shared
      file, and an undeclared per-module-looking path -- is REJECTED, with every
      such test named in one error, including one that a declared file ALSO
      defines (every defining file is checked, not the first).

    The last pair has its positive half: the same undeclared file, once
    declared, is accepted. So the rejection is about the declared set the
    caller passes, not about the file's name, which is what let the gate pass a
    transitional one-file declared set until W4's relocation made it
    `RUNNER_TEST_FILES`.
    """
    first, second, third = RUNNER_TEST_FILES[:3]
    sources = {
        first: ("def _two_runner_files_helper():\n    return 1\n\n"
                "def _first_file_only_helper():\n    return 2\n\n"
                "def test_in_the_first_file():\n"
                "    return _two_runner_files_helper(), _first_file_only_helper()\n"),
        second: ("def test_in_the_second_file():\n    return _two_runner_files_helper()\n"),
        third: ("def test_in_the_third_file():\n    return 3\n"),
        CLIENT_FILE: ("def test_on_the_client_side():\n    return _import_run_modal()\n"),
    }
    destinations, _ = classify_seam(sources, runner_files=RUNNER_TEST_FILES)
    placed = {name: destinations[name] for name in _probe_test_names(sources)}
    assert placed == {
        "test_in_the_first_file": first,
        "test_in_the_second_file": second,
        "test_in_the_third_file": third,
        "test_on_the_client_side": CLIENT_FILE,
    }, f"a test was not sent to the declared file that defines it: {placed}"
    assert destinations["_two_runner_files_helper"] == SHARED_FILE, (
        "a helper reached from two runner files was not sent to the shared file, so the shared "
        "rule is still 'both halves of the runner/client seam', which after the split strands "
        f"it in one runner file. got={destinations['_two_runner_files_helper']!r}")
    assert destinations["_first_file_only_helper"] == first, (
        "a helper reached from one runner file only did not follow its tests to that file. "
        f"got={destinations['_first_file_only_helper']!r}")

    # `test_defined_in_two_files` is defined in a declared file FIRST and in the
    # undeclared file second: a stray check that read only a test's first
    # defining file would miss it.
    undeclared = "tests/test_modal_undeclared_probe.py"
    twice = "def test_defined_in_two_files():\n    return 6\n"
    stray = {
        **sources,
        first: sources[first] + "\n" + twice,
        SHARED_FILE: "def test_runner_side_in_the_shared_file():\n    return 4\n",
        undeclared: "def test_runner_side_in_an_undeclared_file():\n    return 5\n\n" + twice,
    }
    with pytest.raises(ValueError, match="outside the declared runner files") as rejected:
        classify_seam(stray, runner_files=RUNNER_TEST_FILES)
    for name in ("test_runner_side_in_the_shared_file", "test_runner_side_in_an_undeclared_file",
                 "test_defined_in_two_files"):
        assert name in str(rejected.value), (
            f"the rejection did not name {name}, so one run no longer lists every stray: "
            f"{rejected.value}")
    del stray[SHARED_FILE]
    accepted, _ = classify_seam(stray, runner_files=(*RUNNER_TEST_FILES, undeclared))
    assert accepted["test_runner_side_in_an_undeclared_file"] == undeclared, (
        "the undeclared file's test was not accepted once its file was declared, so the "
        "rejection keys on something other than the declared set")


def test_the_seam_source_reader_fails_on_a_missing_declared_file(tmp_path):
    """`_seam_sources` raises on a missing declared file; it never skips one.

    THE HOLE THIS PINS. An existence filter over the declared seam files drops
    a declared file from the classifier's input without a word, and every gate
    built on that input then passes over a smaller seam. A repo-wide grep for
    the old file's name cannot find such a filter, because the filter holds no
    file name. So this is checked by behaviour.

    TWO SHAPES, and each is needed. (1) The real root plus one bogus declared
    runner path: the shape of a module whose test file is declared before it
    exists. (2) Every declared file removed in turn -- each runner file, the
    client file and the shared file -- from a copy of the declared files: a
    filter over only the client and shared files (the runner files still
    refused) passes shape (1), because shape (1) removes a runner file only.

    WHY EVERY OTHER DECLARED FILE MUST EXIST in both shapes, and not an empty
    `tmp_path` root: with an empty root every declared file is missing, so a
    reader that filters by existence and raises only when NOTHING is left also
    raises, and this check would be green on exactly the reader it exists to
    reject. Here only a reader that refuses the one missing file raises. The
    message must name that path and no other declared path, so a reader that
    raises for some other reason, or blames the wrong file, is red too. The last
    assertion is the positive half: the real declared set is read whole.
    """
    bogus = "tests/test_modal_no_such_declared_file.py"
    declared = [*RUNNER_TEST_FILES, CLIENT_FILE, SHARED_FILE]
    assert not (ROOT / bogus).exists(), f"{bogus} exists, so it cannot stand for a missing file"
    assert all((ROOT / rel).is_file() for rel in declared), (
        "a declared seam file is missing from the tree, so this check cannot isolate the bogus "
        f"path: {[rel for rel in declared if not (ROOT / rel).is_file()]}")
    with pytest.raises(FileNotFoundError) as missing:
        _seam_sources(root=ROOT, runner_files=(*RUNNER_TEST_FILES, bogus))
    message = str(missing.value)
    assert bogus in message, f"the error does not name the missing file: {message}"
    assert [rel for rel in declared if rel in message] == [], (
        f"the error names declared files that exist, so it does not say which one is missing: "
        f"{message}")

    for index, removed in enumerate(declared):
        copy_root = tmp_path / f"without-{index}"
        for rel in declared:
            if rel != removed:
                (copy_root / rel).parent.mkdir(parents=True, exist_ok=True)
                (copy_root / rel).write_bytes((ROOT / rel).read_bytes())
        with pytest.raises(FileNotFoundError) as missing:
            _seam_sources(root=copy_root)
        named = [rel for rel in declared if rel in str(missing.value)]
        assert named == [
            removed
        ], (f"with only {removed} missing, the reader did not raise naming exactly that file: "
            f"{missing.value}")

    assert sorted(_seam_sources()) == sorted(declared), (
        "the source reader did not return every declared seam file for the real tree")


# One source per place a module-level statement can hold a definition, with what
# `_defs_under_module_level_statements` must report, as `name:line:keyword`
# (the statement's keyword), space-separated in line order. Every clause of a
# compound statement has its own case (`else`, `except`, `except*`, `finally`,
# `case`), and so does each definition kind, because a census that walked only
# a statement's `body`, or only its direct children, or looked for one kind,
# passes the others.
_NESTED_DEFINITION_CASES = {
    "a test under an `if`": ("if True:\n    def test_x(): pass\n", "test_x:2:if"),
    "a helper in an `if`'s `else`": ("if False:\n    pass\nelse:\n    def _helper(): pass\n",
                                     "_helper:4:if"),
    "an async test in an `except`":
    ("try:\n    pass\nexcept Exception:\n    async def test_x(): pass\n", "test_x:4:try"),
    "a test in a `try`'s `else`":
    ("try:\n    import tomllib\nexcept ImportError:\n    tomllib = None\nelse:\n"
     "    def test_x(): pass\n", "test_x:6:try"),
    "a class in a `finally`": ("try:\n    pass\nfinally:\n    class TestX: pass\n", "TestX:4:try"),
    "a test in an `except*`, whose statement is a `try` too":
    ("try:\n    pass\nexcept* Exception:\n    def test_x(): pass\n", "test_x:4:try"),
    "a test under a `with`": ("with ctx:\n    def test_x(): pass\n", "test_x:2:with"),
    "a test under a `for`": ("for _ in ():\n    def test_x(): pass\n", "test_x:2:for"),
    "a test under a `while`": ("while True:\n    def test_x(): pass\n    break\n",
                               "test_x:2:while"),
    "a test under a `match` case": ("match 1:\n    case _:\n        def test_x(): pass\n",
                                    "test_x:3:match"),
    "a test two statements deep, named by the outer one":
    ("if True:\n    try:\n        pass\n    finally:\n        def test_x(): pass\n", "test_x:5:if"),
    "two definitions in line order, neither one's body":
    ("if True:\n    def test_a():\n        def inner(): pass\n\n    def test_b(): pass\n",
     "test_a:2:if test_b:5:if"),
}

# Module-level statements that hold no definition, and definitions that are not
# under one: each must report nothing.
_UNNESTED_DEFINITION_CASES = {
    "every seam file's `sys.path` guard":
    "import sys\nif str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\n",
    "an import fallback that binds a name":
    "try:\n    import tomllib\nexcept ImportError:\n    tomllib = None\n",
    "a module-level test holding an `if` with a nested def":
    "def test_x():\n    if True:\n        def inner():\n            pass\n",
    "a module-level class and its methods": "class _Double:\n    def read(self):\n        pass\n",
}


def test_the_seam_rejects_a_definition_nested_under_a_module_level_statement():
    """A def or class under a module-level `if`, `try`, `with`, `for`, `while` or `match` raises.

    THE HOLE THIS PINS. pytest collects a test
    defined under a module-level compound statement, but the classifier, the
    manifest, the reach floor, the core rule and the floor's scope check all
    read `tree.body` (`_module_level_names`), so such a test is checked by
    none of them, and the scope check, comparing two populations that lose it
    together, cannot object. Measured under real pytest before the fix: a
    request-reaching test under `if mrl.ALLOWED_MAPS:` in the core file, and
    the same test in the `else` of a `try: import tomllib` in the training
    file, left this file at 59 passed, each test collected.

    Red half: each of `_NESTED_DEFINITION_CASES` alone reports exactly its
    definitions, and `classify_seam` raises on sources holding them, naming
    every one by file and line, in a runner file and in the shared file.
    Green half: the same test at module level is classified by the existing
    rules (it stays in its file, and in the core file the floor and the core
    rule both object to its request name: the positive control), and
    `_UNNESTED_DEFINITION_CASES` -- every seam file's `sys.path` guard, an
    import fallback, a def nested in a module-level def or class -- report
    nothing.
    """
    wrong: dict[str, object] = {}
    for label, (source, expected) in _NESTED_DEFINITION_CASES.items():
        found = _defs_under_module_level_statements(ast.parse(source))
        got = " ".join(f"{name}:{line}:{keyword}" for name, line, keyword, _ in found)
        if got != expected:
            wrong[f"nested, {label}"] = got
    for label, source in _UNNESTED_DEFINITION_CASES.items():
        got = _defs_under_module_level_statements(ast.parse(source))
        if got:
            wrong[f"not nested, {label}"] = got
    assert not wrong, (
        f"the nested-definition census reported the wrong definitions (expected for a nested "
        f"case, nothing for a not-nested one): {wrong}")

    core_file = _floor_file("core")
    header = ("import sys\nfrom pathlib import Path\n\nROOT = Path(__file__).resolve().parents[1]\n"
              "if str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\n\n"
              "import scripts.modal_runner as mrl\n\n")
    at = header.count("\n") + 1
    request_name = _owned("request")
    nested = header + f"if True:\n    def test_planted():\n        assert mrl.{request_name}\n"
    shared = "if True:\n    def _shared_helper():\n        return 1\n"
    with pytest.raises(ValueError, match="nested under a module-level statement") as rejected:
        classify_seam({core_file: nested, SHARED_FILE: shared}, runner_files=RUNNER_TEST_FILES)
    for fragment in (
            f"{core_file}:{at + 1} `test_planted`, under the module-level `if` at line {at}",
            f"{SHARED_FILE}:2 `_shared_helper`"):
        assert fragment in str(
            rejected.value), (f"the rejection does not name {fragment!r}: {rejected.value}")

    top_level = header + f"def test_planted():\n    assert mrl.{request_name}\n"
    destinations, _ = classify_seam({core_file: top_level}, runner_files=RUNNER_TEST_FILES)
    assert destinations.get("test_planted") == core_file, (
        f"the same test at module level was not classified to its file: {destinations}")
    got = _floor_rules({core_file: top_level})
    expected = [("reach-floor", core_file, "test_planted"),
                ("core-rule", core_file, "test_planted")]
    assert got == expected, (
        f"the same test at module level in the core file was not rejected by the floor and the "
        f"core rule for its request name. got={got}")


def test_seam_manifest_agrees_with_the_classifier():
    """The frozen manifest still says what the code says.

    WHY the manifest is frozen at all, given the classifier can recompute it:
    the manifest is what makes a reclassification VISIBLE IN A DIFF. A change to
    `classify_seam` that quietly moves eleven names is a two-line diff with no
    other trace; the same change with this test in place moves eleven lines of
    JSON as well, in the same commit, where a reviewer reads them. That is also
    the answer to "what stops someone narrowing `_CLIENT_BINDINGS`": nothing
    stops it, but it cannot happen silently.

    This test is green before the split and after it, because `classify_seam`
    reads content and not location. That is deliberate and it is a fix: the
    previous draft's Task 3 test was a POST-split assertion committed PRE-split,
    so this task would have committed a red suite and could not have been
    reviewed independently of the split that follows it.

    WHAT THIS DOES NOT DO, so nobody reads it as more than it is: it compares
    the manifest against the classifier, not against where the names actually
    live on disk. Nothing here would notice a name that the split dropped on the
    floor or duplicated into two files. That is Task 4's placement gate.
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    computed, _ = classify_seam(_seam_sources(), RUNNER_TEST_FILES)
    only_manifest = {n: manifest[n] for n in sorted(set(manifest) - set(computed))}
    only_computed = {n: computed[n] for n in sorted(set(computed) - set(manifest))}
    assert not only_manifest and not only_computed, (
        "the manifest and the classifier disagree about WHICH names exist. "
        f"manifest-only={only_manifest} classifier-only={only_computed}. "
        "If the tree is right: " + _manifest_edits(add=only_computed, delete=only_manifest))
    disagree = {n: (manifest[n], computed[n]) for n in manifest if manifest[n] != computed[n]}
    assert not disagree, (
        f"the manifest is stale: (frozen, recomputed) for each name that moved: {disagree}. "
        "If the recomputed file is right: " + _manifest_edits(move={
            n: new
            for n, (_, new) in disagree.items()
        }))


def test_no_governed_name_is_defined_outside_the_seams_own_files():
    """A name the manifest governs may not be defined under `tests/` elsewhere.

    WHY THIS EXISTS, and it is not the obvious reason. `_names_defined_under_tests`
    scans by GLOB over the test directory rather than by the manifest's own list
    of destination files, because a check that asks the manifest where to look
    cannot see a file the manifest does not know about. Review demonstrated the
    hole on the previous revision: a governed name (`_git`) moved into a
    brand-new `tests/test_modal_stray.py` left this file green, because nothing
    committed here called the scanner at all. It is called now.

    WHAT IT CATCHES: every governed name lives in one of the seam's four legal
    homes, so a governed name appearing in a FIFTH file is either a copy or a
    migration nothing declared. Written pre-split, when the sentence read "every
    governed name lives in `tests/test_modal_runner.py` ... one legal home" and
    "WHAT IT BECOMES after Task 4" was the four-home version; W2 made the second
    half the live rule. Current destination counts are derived from the manifest,
    rather than duplicated in prose:
    `test_seam_manifest_agrees_with_the_classifier` is what verifies that
    manifest against the source.

    WHAT IT DELIBERATELY DOES NOT CATCH: a BRAND-NEW name in a stray file. That
    name is not governed, and an unrelated new test module is legal. Only
    `governed` is in scope -- which is also why the scan's superset (every
    `tests/test_*.py` plus the shared helper module) does not turn this red for
    every helper in those files.

    PITFALL, and it is the one this repo keeps paying for: a scan that reached
    nothing would pass this silently. So the scope controls come FIRST and are
    not decoration. `outside` proves the glob reaches past the seam's own files,
    without which nothing could ever be found straying; `unseen` proves the scan
    actually sees the governed names rather than passing on an empty
    intersection. Both are stated as failures of THIS TEST's instrument, not of
    the tree, because that is what they would mean.
    """
    governed = set(json.loads(MANIFEST.read_text(encoding="utf-8")))
    seam_files = {*RUNNER_TEST_FILES, CLIENT_FILE, SHARED_FILE, PACKAGING_FILE}
    found = _names_defined_under_tests()

    outside = {f for files in found.values() for f in files} - seam_files
    assert outside, ("the scan reached no file outside the seam's own four, so it could not "
                     "report a strayed name even if one existed. `_names_defined_under_tests` "
                     "globs `tests/test_*.py`; if that returned only seam files the glob is "
                     "broken, not the tree.")
    unseen = governed - set(found)
    assert not unseen, ("names the manifest governs are defined nowhere the scan can see, so the "
                        "check below would pass by looking at nothing. Either the scan lost a "
                        "file it should cover, or these names were deleted from the tree without "
                        f"being deleted from the manifest: {sorted(unseen)}")

    strayed = {
        name: sorted(set(files) - seam_files)
        for name, files in found.items() if name in governed and set(files) - seam_files
    }
    assert not strayed, ("these names are governed by the seam manifest but are ALSO defined in a "
                         "file the seam does not own, so a reader has no way to tell which "
                         "definition the suite runs and Task 4's split would silently pick one. "
                         f"{strayed}. Delete (or rename) the copy outside the seam; the governed "
                         "definition stays where its manifest value says, so " + _manifest_edits())


def test_the_modal_test_split_matches_concern_recomputed_from_source():
    """Every governed name lives in the file its RECOMPUTED concern says.

    THE PITFALL THIS EXISTS FOR, measured: a check that compares the split
    against a frozen manifest is green on a maximally wrong split. Review
    reassigned all 250 non-guard, non-shared names runner/client by even/odd
    source index, split the file to match its own shuffled manifest, and got
    `1 passed`. A manifest is a record of a decision; it is not evidence the
    decision was right. So this test re-runs `classify_seam` over the files ON
    DISK and compares the answer to where each name actually sits. The reference
    graph does not change when you shuffle names between two files -- which is
    exactly why the shuffle cannot hide from it.

    Three more holes the first draft had, all demonstrated, all closed here:

    * DELETION, in the shape that leaves the manifest naming the deleted test:
      iterating disk->manifest never examines a manifest name with no definition
      anywhere. Deleting `test_allowed_gpus` outright gave `1 passed`. Hence the
      manifest->disk direction below.
    * A FOURTH FILE was invisible: the scan set was the manifest's own value
      set. Moving a test into a new `tests/test_modal_stray.py` gave
      `1 passed`. Hence `_names_defined_under_tests`'s glob.
    * The HEADER EXCLUSION could hide a loss: `ROOT` is ungoverned, so dropping
      it from a destination would be silent. Hence the last assertion.

    TWO MORE SHAPES THIS GATE SHIPPED GREEN ON, found by review at `84622fc` by
    running them rather than reading them, and closed here. Both are the same
    defect class as each other and as the reason this file exists: a bullet above
    claimed a hole was closed when only ONE SHAPE of that hole was.

    * DELETION THAT ALSO EDITS THE MANIFEST -- which is the shape a real deleting
      commit produces, because leaving the manifest stale reddens
      `test_seam_manifest_agrees_with_the_classifier` and the author fixes that
      before pushing. Excising `test_allowed_gpus` from the file AND removing its
      manifest key gave `3 passed`. Every assertion here iterates `manifest`, and
      the agreement test compares `set(manifest)` against `set(computed)`, so a
      name absent from BOTH is examined by nothing at all. Hence
      `GOVERNED_NAME_COUNT`: the seam's size is pinned, so a deletion can no
      longer be absorbed by regenerating the manifest -- it has to move a number
      a reviewer reads in the diff.
    * AN INTRA-FILE DUPLICATE. `duplicated` counts FILES, and the `on_disk` lists
      come from `_module_level_names`, which is a DICT -- so two module-level
      definitions of one name inside ONE file collapse to a single entry and the
      list length stays 1. Appending a second `PINNED_CUDA_IMAGE` to
      tests/test_modal_client.py gave `3 passed`. That is precisely the failure
      `duplicated`'s own message describes, in the only arrangement Python
      actually permits: you cannot have two live definitions across two files,
      but you can inside one, where the second silently shadows the first. Step
      6's control (d) covered only the cross-file shape. Hence
      `_module_level_binding_counts` and `redefined` below.

    AND THE REACH FLOOR, over the same union namespace (`reach_floor_violations`,
    whose docstring states what it cannot catch, with figures). For a runner
    test, `concern says` is simply the runner file that defines it, so the
    `misplaced` check below compares on-disk with on-disk for tests and cannot
    see a test in the wrong runner file; the floor and the core rule are what
    object to one. Before them, every runner import in the seam files must
    bind the facade or a whole runner module, the only aliases the two rules
    resolve through (`_runner_imports_the_floor_cannot_resolve`). The
    floor's own scope is asserted last: the (file, test)
    pairs it examined must be exactly the manifest's runner-half tests, in
    whatever file the manifest puts them, and every runner test file must hold
    one, or a floor over a partial file list, or a seam reading a runner file
    the floor does not, would pass here unseen (`_floor_scope_problems`).
    """
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = _seam_sources()
    computed, _ = classify_seam(sources, RUNNER_TEST_FILES)
    on_disk = _names_defined_under_tests()
    trees = {
        rel: ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for rel in sorted(set(manifest.values()))
    }

    assert len(manifest) == GOVERNED_NAME_COUNT, (
        f"the seam governs {len(manifest)} names, not {GOVERNED_NAME_COUNT}. If you deleted a test "
        "and regenerated the manifest to match, every other check in this file stays green and the "
        "loss is invisible -- that is the shape this pin exists for. If the change is intended, "
        "move the number and say why in the commit; if it is not, you have lost a test.")

    missing = sorted(n for n in manifest if n not in on_disk)
    assert not missing, (
        "the manifest names definitions that exist nowhere under tests/. Either "
        f"they were deleted or they were moved out of reach of this scan: {missing}. "
        "Restore them, or, if the deletion is intended: " + _manifest_edits(delete=missing))

    strayed = {n: on_disk[n] for n in manifest if n not in computed}
    assert not strayed, (
        "a governed name is defined under tests/ but NOT in any file the seam governs, so no "
        f"concern can be recomputed for it. A new destination is not a place to put split output: "
        f"{strayed}. Move each definition back into the seam file its manifest value names; the "
        "value stays and only the definition moves, so " + _manifest_edits() + " If the name is "
        "instead leaving the seam for good: " + _manifest_edits(delete=strayed))

    duplicated = {n: on_disk[n] for n in manifest if len(on_disk[n]) > 1}
    assert not duplicated, (
        "a governed name is defined in two files. Two copies of a pinned digest "
        f"or a fixture drift apart and every other check here stays green: {duplicated}")

    redefined = {}
    for rel, tree in trees.items():
        for name, times in sorted(_module_level_binding_counts(tree).items()):
            if name in manifest and times > 1:
                redefined[name] = {"file": rel, "module-level definitions": times}
    assert not redefined, (
        "a governed name is defined more than once at module level INSIDE one file, so the second "
        "definition silently shadows the first. `duplicated` above counts files and cannot see "
        "this; it is the same drift, in the only arrangement Python permits. Two copies of a "
        f"pinned digest is the case that motivated the seam: {redefined}")

    misplaced = {
        n: {
            "on disk": on_disk[n][0],
            "concern says": computed[n],
            "manifest says": manifest[n]
        }
        for n in manifest if on_disk[n][0] != computed[n]
    }
    assert not misplaced, (
        f"{len(misplaced)} of {len(manifest)} names are not in the file their recomputed "
        "concern assigns them to. Note that `manifest says` agreeing with `on disk` proves "
        "nothing -- a wrong split and a manifest written to match it agree perfectly, which "
        f"is why this compares against `concern says`. First 10: "
        f"{dict(list(misplaced.items())[:10])}. Move each definition to `concern says`, then " +
        _manifest_edits(
            move={
                n: where["concern says"]
                for n, where in misplaced.items() if where["manifest says"] != where["concern says"]
            }))

    for rel, tree in trees.items():
        defined = set(_module_level_names(tree))
        absent = sorted(SEAM_HEADER_NAMES - defined)
        assert not absent, (
            f"{rel} does not define {absent}, which the seam treats as module-header "
            "boilerplate rather than governing. Ungoverned must not mean lost.")

    # Before the floor, so a test the floor cannot credit because of its
    # file's import is not sent to be moved or exempted by the floor's remedy.
    unresolvable = _runner_imports_the_floor_cannot_resolve(sources)
    assert not unresolvable, (
        "these runner imports bind a name other than the facade or a whole runner module, so the "
        "reach floor and the core rule resolve nothing through them: a test that reads such a "
        "name reaches nothing (a correctly placed test fails the floor) and a core-file test "
        f"escapes the core rule: {unresolvable}. Import the facade (`import scripts.modal_runner "
        "as mrl`) or the module (`from scripts.modal_runner import request`) and spell the name "
        "`mrl.X` or `request.X`; a module-level import, so the floor reads it.")
    violations, examined = reach_floor_violations(sources, RUNNER_OWNERS, _REACH_EXEMPTIONS)
    assert not violations, (f"{len(violations)} reach-floor / core-rule / exemption violation(s):\n"
                            f"{_describe_floor_violations(violations)}")
    # THE FLOOR'S SCOPE ASSERTION: the pairs it examined, exempt or not, are
    # exactly the manifest's runner-half tests, and every runner test file holds
    # one. `_floor_scope_problems` has the why, and its probe pins each clause.
    scope = _floor_scope_problems(examined, manifest)
    assert not scope, "\n".join(scope)


# ── The reach floor's synthetic probes: one per rule, each with both halves ──
#
# WHY THEY EXIST. On the real tree the floor has, by design, no failing case
# except its one exempt test, so a change that makes it more permissive -- every
# reference resolving to every module, a closure that over-reaches, a
# minimality check that does nothing, exemptions keyed by test name alone -- is
# green on the tree and invisible after merge. Each probe below is a pure-data
# call of `reach_floor_violations` on a synthetic {path: source} map, parsed and
# never executed; the paths are `RUNNER_TEST_FILES` entries and need not exist.
# Names that `mrl.X` resolves through are read from the tables rather than
# typed here (`_owned`), so moving a production symbol never breaks a probe.


def _floor_file(module):
    """The per-module test file path for `module`, from `RUNNER_TEST_FILES`."""
    return RUNNER_TEST_FILES[RUNNER_MODULES.index(module)]


def _owned(module):
    """One top-level name the tables say `module` owns, for a probe's `mrl.<name>` reference."""
    return RUNNER_OWNERS[f"{module}.py"][0]


def _floor_rules(sources, exemptions=None):
    """`reach_floor_violations` on a probe map with the real owner table; (rule, file, test) only."""
    violations, _ = reach_floor_violations(sources, RUNNER_OWNERS, exemptions or {})
    return [violation[:3] for violation in violations]


def test_reach_floor_resolves_mrl_names_to_owner():
    """`<facade>.X` resolves to X's owner in the tables' MANIFEST, and only there.

    Positive: a request-file test naming a request-owned name through the
    facade passes, under `mrl` and under a second facade spelling (`from
    scripts import modal_runner as facade`), because the facade alias is read
    from the file's imports, not assumed. Negative: a request-file test naming
    only a core-owned name fails the floor, and so does a core-file test naming
    a facade name the tables do not own (it resolves to nothing, not to a
    default module such as `core`). It also pins what is EXAMINED: an `async
    def` test is, because pytest collects one (`_is_test_def`, whose async arm
    nothing else on the real tree exercises), and the client file's test is
    not, because the client file names no module.

    AND ONLY THROUGH A BOUND FACADE. In a file that binds no facade alias,
    `mrl.<request name>` resolves to nothing, and so does `<name>.<request
    name>` through a module-level name that is not the facade. A rule that
    recognised the literal `mrl`, or that looked any attribute up in the owner
    table whatever its root, passes both.
    """
    req, core_file = _floor_file("request"), _floor_file("core")
    header = "import scripts.modal_runner as mrl\nfrom scripts import modal_runner as facade\n\n"
    tests = (f"def test_names_its_module():\n    return mrl.{_owned('request')}\n\n"
             f"def test_names_it_via_another_spelling():\n    return facade.{_owned('request')}\n\n"
             f"async def test_names_it_asynchronously():\n    return mrl.{_owned('request')}\n\n"
             f"def test_names_another_module():\n    return mrl.{_owned('core')}\n")
    unowned = "no_name_the_tables_give_any_module"
    assert all(unowned not in names for names in RUNNER_OWNERS.values())
    sources = {
        req: header + tests,
        core_file: header + f"def test_names_an_unowned_name():\n    return mrl.{unowned}\n",
        CLIENT_FILE: "def test_on_the_client_side():\n    return 1\n",
    }
    violations, examined = reach_floor_violations(sources, RUNNER_OWNERS, {})
    names = ("test_names_its_module", "test_names_it_via_another_spelling",
             "test_names_it_asynchronously", "test_names_another_module")
    pairs = {(req, name) for name in names} | {(core_file, "test_names_an_unowned_name")}
    assert examined == pairs, f"the floor examined the wrong (file, test) pairs: {sorted(examined)}"
    got = [violation[:3] for violation in violations]
    expected = [("reach-floor", core_file, "test_names_an_unowned_name"),
                ("reach-floor", req, "test_names_another_module")]
    assert got == expected, (
        "`mrl.X` did not resolve to X's owner alone: a test naming its own module's name must "
        "pass, and one naming only another module's name, or a name no module owns, must fail. "
        f"got={violations}")

    unbound = {
        req: ("not_the_facade = object()\n\n"
              f"def test_through_an_unbound_facade():\n    return mrl.{_owned('request')}\n\n"
              f"def test_through_a_non_alias():\n    return not_the_facade.{_owned('request')}\n"),
    }
    got = _floor_rules(unbound)
    expected = [("reach-floor", req, "test_through_a_non_alias"),
                ("reach-floor", req, "test_through_an_unbound_facade")]
    assert got == expected, (
        "a request-owned name read through a root that is not a bound facade alias resolved to "
        f"its owner, so the facade check is gone or assumes the literal `mrl`. got={got}")


def test_reach_floor_resolves_submodule_aliases():
    """`<sub>.X` resolves to `sub` through the file's module-level alias, unless rebound locally.

    Positive: `training.X` (bound by `from scripts.modal_runner import
    training`) passes in the training file; `st.X` (bound by `import
    scripts.modal_runner.state as st`) passes in the state file. Negative: the
    same `training.X` placed in the state file fails, and a training-file test
    that rebinds `training` locally before `training.X` fails too -- the local
    rebinding is not the module (the request test file has live cases of this
    shape: `request = mrl.build_run_request(...)`, then `request.run_id`).
    Every other kind of local binding has its own case in
    `test_reach_floor_sees_every_local_binding_of_a_module_alias`.

    AND ONLY THROUGH A MODULE-LEVEL ALIAS. In a request file that binds no
    `request` at module level, a bare `request.X` resolves to nothing: a name
    is not the module because it is spelled like one. A function-local `from
    scripts.modal_runner import request` is NOT MODELLED as an alias, by
    choice (`_runner_aliases` says why): it credits neither another test in
    the same file that reads `request.X` unbound, which is right, nor its own
    test, which at run time DOES read the module. That first case pins the
    choice, not the language: it is the stricter direction for the floor (and
    the looser one for the core rule), and a change that models local runner
    imports must flip it in the same commit. A rule that read aliases from the
    whole tree, not the module's own statements, passes the second test.
    """
    header = ("from scripts.modal_runner import training\n"
              "import scripts.modal_runner.state as st\n\n")
    training_file, state_file = _floor_file("training"), _floor_file("state")
    sources = {
        training_file:
        header + ("def test_training_alias():\n    return training.anything\n\n"
                  "def test_training_alias_rebound():\n"
                  "    training = object()\n    return training.anything\n"),
        state_file:
        header + ("def test_state_alias():\n    return st.anything\n\n"
                  "def test_training_alias_in_state():\n"
                  "    return training.anything\n"),
    }
    assert _floor_rules(sources) == [
        ("reach-floor", state_file, "test_training_alias_in_state"),
        ("reach-floor", training_file, "test_training_alias_rebound"),
    ], f"submodule aliases resolved wrongly: {_floor_rules(sources)}"

    req = _floor_file("request")
    unbound = {
        req: ("def test_imports_it_in_its_body():\n"
              "    from scripts.modal_runner import request\n"
              "    return request.anything\n\n"
              "def test_reads_it_unbound():\n    return request.anything\n"),
    }
    got = _floor_rules(unbound)
    expected = [("reach-floor", req, "test_imports_it_in_its_body"),
                ("reach-floor", req, "test_reads_it_unbound")]
    assert got == expected, (
        "`request.X` resolved to `request` in a file with no module-level `request` alias: "
        "either a bare name spelled like a module resolves as that module, or a function-local "
        f"import is being read as a module alias. got={got}")


# One request-file source per way Python can make `request` mean something
# other than the module at the point it is read. Each defines `test_shadowed`,
# whose only candidate reach is `request.anything`, so each must fail the floor.
# The header binds `request` to the module first; the case then rebinds it.
# ONE CASE PER ALTERNATIVE: where one condition of the resolver accepts several
# node kinds (`Import` or `ImportFrom`, `MatchAs` or `MatchStar`, a `def`, an
# `async def` or a class), each kind has its own case here, because a case per
# condition dies on whichever kind it happens to use and leaves the others
# unpinned.
_SHADOWING_CASES = {
    "the test's own parameter (pytest's `request` fixture)":
    "def test_shadowed(request):\n    return request.anything\n",
    "an async test's own parameter":
    "async def test_shadowed(request):\n    return request.anything\n",
    "a positional-only parameter":
    "def test_shadowed(request, /):\n    return request.anything\n",
    "a keyword-only parameter":
    "def test_shadowed(*, request=None):\n    return request.anything\n",
    "a *args parameter":
    "def test_shadowed(*request):\n    return request.anything\n",
    "a **kwargs parameter":
    "def test_shadowed(**request):\n    return request.anything\n",
    "a lambda's parameter":
    "def test_shadowed():\n    return lambda request: request.anything\n",
    "a nested def's parameter":
    ("def test_shadowed():\n    def inner(request):\n        return request.anything\n"
     "    return inner\n"),
    "a method's parameter, in a helper class the test reaches":
    ("class _Double:\n    def read(self, request):\n        return request.anything\n\n"
     "def test_shadowed():\n    return _Double\n"),
    "an except-as name":
    ("def test_shadowed():\n    try:\n        pass\n    except Exception as request:\n"
     "        return request.anything\n"),
    "a comprehension target":
    "def test_shadowed(items):\n    return [request.anything for request in items]\n",
    "a for target":
    "def test_shadowed(items):\n    for request in items:\n        return request.anything\n",
    "a with target":
    "def test_shadowed(opened):\n    with opened as request:\n        return request.anything\n",
    "a walrus target":
    "def test_shadowed(value):\n    if (request := value):\n        return request.anything\n",
    "a walrus inside a comprehension (it binds the enclosing function)":
    ("def test_shadowed(items):\n    last = [(request := item) for item in items]\n"
     "    return last, request.anything\n"),
    "a walrus inside a nested comprehension (it binds the enclosing function too)":
    ("def test_shadowed(rows):\n    [[(request := item) for item in row] for row in rows]\n"
     "    return request.anything\n"),
    "a walrus inside another walrus's value":
    ("def test_shadowed(value):\n    if (found := (request := value)):\n"
     "        return found, request.anything\n"),
    "a nested `from ... import ... as`":
    ("def test_shadowed():\n    from tests import modal_test_helpers as request\n"
     "    return request.anything\n"),
    "a nested `import ... as`":
    ("def test_shadowed():\n    import tests.modal_test_helpers as request\n"
     "    return request.anything\n"),
    "a nested `import` of a dotted name, which binds its first component":
    "def test_shadowed():\n    import request.helpers\n    return request.anything\n",
    "a nested def's name": ("def test_shadowed():\n    def request():\n        pass\n\n"
                            "    return request.anything\n"),
    "a nested async def's name": ("def test_shadowed():\n    async def request():\n        pass\n\n"
                                  "    return request.anything\n"),
    "a nested class's name": ("def test_shadowed():\n    class request:\n        pass\n\n"
                              "    return request.anything\n"),
    "a class body's own binding, read later in the same class body":
    ("class _Double:\n    request = None\n    held = request.anything\n\n"
     "def test_shadowed():\n    return _Double\n"),
    "a match capture": ("def test_shadowed(value):\n    match value:\n        case request:\n"
                        "            return request.anything\n"),
    "a match star capture": ("def test_shadowed(value):\n    match value:\n"
                             "        case [*request]:\n            return request.anything\n"),
    "a match mapping's `**rest`":
    ("def test_shadowed(value):\n    match value:\n"
     "        case {**request}:\n            return request.anything\n"),
    "a del":
    "def test_shadowed():\n    request.anything\n    del request\n",
    "a type parameter":
    "def test_shadowed[request]():\n    return request.anything\n",
    "a later module-level `from ... import ... as` of the same name":
    ("from tests import modal_test_helpers as request\n\n"
     "def test_shadowed():\n    return request.anything\n"),
    "a later module-level `import ... as` of the same name":
    ("import tests.modal_test_helpers as request\n\n"
     "def test_shadowed():\n    return request.anything\n"),
    "a later module-level relative import spelled like the runner":
    ("from .scripts.modal_runner import request\n\n"
     "def test_shadowed():\n    return request.anything\n"),
    "one `from ... import` binding the name twice, the last not a runner module":
    ("from scripts.modal_runner import request, build_run_request as request\n\n"
     "def test_shadowed():\n    return request.anything\n"),
    "one `import` binding the name twice, the last not a runner module":
    ("import scripts.modal_runner.request as request, tests.modal_test_helpers as request\n\n"
     "def test_shadowed():\n    return request.anything\n"),
    "a later module-level binding under an if":
    ("if True:\n    request = None\n\n"
     "def test_shadowed():\n    return request.anything\n"),
    "a `global` rebinding in another function":
    ("def _rebind():\n    global request\n    request = None\n\n"
     "def test_shadowed():\n    return request.anything\n"),
    "a `global` rebinding in an async function":
    ("async def _rebind():\n    global request\n    request = None\n\n"
     "def test_shadowed():\n    return request.anything\n"),
    "a `global` rebinding in a class body":
    ("class _Rebind:\n    global request\n    request = None\n\n"
     "def test_shadowed():\n    return request.anything\n"),
}

# The same shapes with the alias NOT rebound where it is read, so each must
# pass. They are what make the resolver exact rather than merely strict: a rule
# that shadowed the name wherever any binder of it appears in the node would
# fail every case above, as it should, and would also fail the class-body,
# comprehension and parametrize cases here, as it should not.
_UNSHADOWED_CASES = {
    "a closure that reads the alias":
    ("def test_unshadowed():\n    def inner():\n        return request.anything\n"
     "    return inner\n"),
    "a method that reads the alias past a class-body binding of the same name":
    ("class _Double:\n    request = None\n\n    def read(self):\n"
     "        return request.anything\n\n"
     "def test_unshadowed():\n    return _Double\n"),
    "a read after a comprehension whose own target is spelled like the alias":
    ("def test_unshadowed(items):\n    firsts = [request for request in items]\n"
     "    return firsts, request.anything\n"),
    "a read after a set comprehension whose own target is spelled like the alias":
    ("def test_unshadowed(items):\n    firsts = {request for request in items}\n"
     "    return firsts, request.anything\n"),
    "a read after a dict comprehension whose own target is spelled like the alias":
    ("def test_unshadowed(items):\n    firsts = {request: 1 for request in items}\n"
     "    return firsts, request.anything\n"),
    "a read after a generator expression whose own target is spelled like the alias":
    ("def test_unshadowed(items):\n    firsts = list(request for request in items)\n"
     "    return firsts, request.anything\n"),
    "a read after a walrus that binds a lambda inside a comprehension, not the test":
    ("def test_unshadowed(items):\n    readers = [lambda: (request := item) for item in items]\n"
     "    return readers, request.anything\n"),
    "a comprehension's first iterable, evaluated outside the comprehension":
    "def test_unshadowed():\n    return [request for request in request.anything]\n",
    "a parametrize list, evaluated at module scope, above a `request` parameter":
    ('@pytest.mark.parametrize("value", [request.anything])\n'
     "def test_unshadowed(request, value):\n    return value\n"),
    "a `global` declaration that only reads":
    "def test_unshadowed():\n    global request\n    return request.anything\n",
    "a `global` declaration in a closure, past the enclosing function's local":
    ("def test_unshadowed():\n    request = None\n\n    def inner():\n        global request\n"
     "        return request.anything\n\n    return request, inner\n"),
    "a module-level rebinding that the runner import then undoes":
    ("request = None\nfrom scripts.modal_runner import request\n\n"
     "def test_unshadowed():\n    return request.anything\n"),
    "one import binding the name twice, the last the runner module":
    ("from scripts.modal_runner import state as request, request\n\n"
     "def test_unshadowed():\n    return request.anything\n"),
}


def test_reach_floor_sees_every_local_binding_of_a_module_alias():
    """A module alias rebound where it is read is not the module: every binder, at every depth.

    THE FAILURE THIS PINS is the looser direction. A floor that credits reach
    through a parameter, a lambda's parameter, an `except ... as`, a
    comprehension target or any other local binding named like a module alias
    passes a test that reaches nothing of that module, and on the real tree
    nothing else objects. Each of `_SHADOWING_CASES` is a request-file test
    whose only candidate reach is `request.anything` through such a binding,
    and each must fail the floor ALONE, so a failure names its binder.

    The positive half is `_UNSHADOWED_CASES`: the neighbouring shapes where the
    name really is the module when it is read -- a closure, a method past a
    class-body name (class scopes are invisible to the scopes inside them), a
    read after each kind of comprehension (whose target is its own) and in its
    first iterable (evaluated outside it), a read after a walrus that binds a
    lambda rather than the test, a parametrize list (evaluated at module
    scope, not inside the test), a `global` that only reads, including one in a
    closure whose enclosing function binds the name, a module-level rebinding
    followed by the runner import, and one import that binds the name twice
    with the runner module last. Each must pass, so a resolver cannot pass
    this test by shadowing everything.

    Every element of the resolver's binder and scope lists whose removal
    changes an answer has a case of its own, on one side or the other; the
    `nonlocal` branch changes none, because a `nonlocal` name is always an
    enclosing function's, never the module. What these cases cannot pin is a
    FIELD the resolver forgets to walk:
    `test_the_scope_walk_visits_every_expression_ast_walk_visits` does that.
    """
    req = _floor_file("request")
    header = "from scripts.modal_runner import request\n\n"
    wrong = {}
    for binder, case in _SHADOWING_CASES.items():
        got = _floor_rules({req: header + case})
        if got != [("reach-floor", req, "test_shadowed")]:
            wrong[f"credited through {binder}"] = got
    for shape, case in _UNSHADOWED_CASES.items():
        got = _floor_rules({req: header + case})
        if got:
            wrong[f"not credited through {shape}"] = got
    assert not wrong, (
        "the floor's scope resolution is wrong for these shapes (a shadowing case must fail the "
        f"floor, an unshadowed one must pass): {wrong}")


# Every field `_scope_parts` hands out by hand, populated at least once: a def's
# and an async def's decorators, type parameters, return annotation, and each
# parameter kind's annotation, default and keyword-only default; a class's
# decorators, type parameters, bases and keywords; a lambda's defaults,
# keyword-only defaults and body; and each comprehension kind's element (or key
# and value), targets, first and later iterables and conditions. Every read is
# a distinct dotted name, so a failure names the field it lost. Parsed, never
# executed.
_EVERY_FIELD_SOURCE = '''
@deco.a
def f[T: bound.b](p: ann.c = dflt.d, /, q: ann.e = dflt.f, *a: star.g, k: kw.h = kwd.i,
                  **kw: kk.j) -> ret.k:
    @inner_deco.l
    class C[U: cbound.m](base.n, metaclass=meta.o):
        x = lambda z=ldef.p, *, y=lkw.q: lbody.r
    xs = [e.s for t in first.t if cond.u for u in second.v if cond2.w]
    d = {key.x: val.y for t in it.z if dcond.aa}
    g = (ge.bb for t in git.cc if gcond.dd)
    st = {se.ee for t in sit.ff if scond.gg}
    return C, xs, d, g, st


@adeco.hh
async def h[V](r: aann.ii = adflt.jj, *, s: akw.kk = akwd.ll) -> aret.mm:
    return r, s
'''

# The node kinds whose every field `_EVERY_FIELD_SOURCE` must populate: each
# kind the resolver opens a scope for, and the three that carry a def's, a
# lambda's and a comprehension's parts. Listed, not read from `_SCOPE_NODES`,
# so that a kind dropped from the resolver is not also dropped from the check.
_EVERY_FIELD_KINDS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.ListComp,
                      ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.arguments, ast.arg,
                      ast.comprehension)

# Where `_scoped_walk` evaluates each read of `_EVERY_FIELD_SOURCE`: the number
# of scope frames around it (the length of its chain; 0 is the module). `f`'s
# body is 2 deep (its type-parameter frame, then its own), a comprehension in
# it 3, `C`'s body 4 and the lambda's body 5. A field moved between
# `_scope_parts`' outer and inner lists moves its read by a frame or more, so
# this pins WHICH scope each field is evaluated in, where the skip check pins
# only that it is visited: moving a comprehension's conditions outside it, or
# a def's defaults inside it, once passed every other check. Checked read by
# read against the language reference: decorators,
# argument defaults and annotations run where the def or class statement
# runs, a class's bases and keywords too, a lambda's defaults where the lambda
# is, a comprehension's first iterable in the enclosing scope and the rest of
# it inside. Thirteen entries record the walk's PEP 695 approximation, one
# frame short of the language's annotation scope (which sees the type
# parameters): every parameter annotation and the return annotation of the
# generic `f` and `h`, the bounds `bound.b` and `cbound.m`, and the generic
# `C`'s bases and keywords (`_resolves_to_module_scope` lists it). Change an
# entry only in the commit that changes the walk, and say why.
_EVERY_FIELD_FRAMES = {
    "deco.a": 0,
    "bound.b": 0,
    "ann.c": 0,
    "dflt.d": 0,
    "ann.e": 0,
    "dflt.f": 0,
    "star.g": 0,
    "kw.h": 0,
    "kwd.i": 0,
    "kk.j": 0,
    "ret.k": 0,
    "inner_deco.l": 2,
    "cbound.m": 2,
    "base.n": 2,
    "meta.o": 2,
    "ldef.p": 4,
    "lkw.q": 4,
    "lbody.r": 5,
    "e.s": 3,
    "first.t": 2,
    "cond.u": 3,
    "second.v": 3,
    "cond2.w": 3,
    "key.x": 3,
    "val.y": 3,
    "it.z": 2,
    "dcond.aa": 3,
    "ge.bb": 3,
    "git.cc": 2,
    "gcond.dd": 3,
    "se.ee": 3,
    "sit.ff": 2,
    "scond.gg": 3,
    "adeco.hh": 0,
    "aann.ii": 0,
    "adflt.jj": 0,
    "akw.kk": 0,
    "akwd.ll": 0,
    "aret.mm": 0,
}


def _expressions_the_scope_walk_skips(source):
    """Every expression under `source`'s statements that `ast.walk` visits and `_scoped_walk` misses.

    Unparsed, so a failure reads as the code it lost. Each module-level
    statement is walked on its own, as `_own_module_refs` walks each
    module-level name's node.
    """
    skipped = []
    for statement in ast.parse(source).body:
        visited = {id(sub) for sub, _ in _scoped_walk(statement)}
        skipped += [
            ast.unparse(sub) for sub in ast.walk(statement)
            if isinstance(sub, ast.expr) and id(sub) not in visited
        ]
    return skipped


def test_the_scope_walk_visits_every_expression_ast_walk_visits():
    """`_scoped_walk` reaches every expression `ast.walk` reaches, each in the scope that evaluates it.

    THE FAILURE THIS PINS is a rule that narrows with everything green.
    `_own_module_refs` reads references off `_scoped_walk`, which reaches a
    def's, class's, lambda's or comprehension's fields only through the lists
    `_scope_parts` writes out by hand; `ast.walk`, which it replaced, visits
    every field by construction. A field dropped from those lists -- argument
    defaults, an annotation, a class base, a later iterable, a condition, a
    dict key, a lambda default -- is a field whose `mrl.X` no rule sees, and
    the core rule, which alone polices the core file, loosens silently. One
    case per field would pin one field each; this invariant pins them all,
    and any field a later Python adds, once a source populates it.

    It is checked first on `_EVERY_FIELD_SOURCE`, so a dropped field is red
    whatever the real tree holds. That source must populate every field of
    `_EVERY_FIELD_KINDS`, and those kinds must cover `_SCOPE_NODES`; both are
    asserted, so trimming the source or adding a scope kind cannot quietly
    shrink what the check covers. It is then checked on the other probe
    sources and on every seam file, for a shape the tree has and the synthetic
    source lacks.

    VISITED IS NOT PLACED. Dropping a field is one failure; putting it in the
    wrong one of `_scope_parts`' two lists is the other, and the visit check
    cannot see it: the field is still walked, only in the wrong scope, so a
    comprehension's condition reads its own target as the module, or a
    default's `training.X` resolves to a parameter named `training` and the
    core rule stops seeing it. Thirteen such
    moves once passed every committed test. So the
    number of frames each read of `_EVERY_FIELD_SOURCE` is walked in must
    equal `_EVERY_FIELD_FRAMES`, a table checked against the language
    reference, compared as a list so a repeated read cannot hide in a dict.
    """
    uncovered = [kind.__name__ for kind in _SCOPE_NODES if kind not in _EVERY_FIELD_KINDS]
    assert not uncovered, (
        f"the resolver opens a scope for {uncovered}, which _EVERY_FIELD_KINDS does not list, so "
        "no field of theirs is checked: list them there and populate them in _EVERY_FIELD_SOURCE")
    # Annotated because pyrefly otherwise infers the keys as the union of the
    # eleven kinds, and then rejects indexing with `type(sub)`, a `type[AST]`.
    populated: dict[type[ast.AST], set[str]] = {kind: set() for kind in _EVERY_FIELD_KINDS}
    for sub in ast.walk(ast.parse(_EVERY_FIELD_SOURCE)):
        if type(sub) in populated:
            filled = {field for field, value in ast.iter_fields(sub) if value not in (None, [])}
            populated[type(sub)] |= filled
    empty = {}
    for kind, filled in populated.items():
        unfilled = sorted(set(kind._fields) - {"type_comment"} - filled)
        if unfilled:
            empty[kind.__name__] = unfilled
    assert not empty, (
        f"_EVERY_FIELD_SOURCE leaves these fields empty everywhere: {empty}, so a scope walk "
        "that dropped them would pass the check below. Populate each with a distinct read.")

    missing = _expressions_the_scope_walk_skips(_EVERY_FIELD_SOURCE)
    assert not missing, (
        f"_scoped_walk never visits these expressions of _EVERY_FIELD_SOURCE: {missing}. Each "
        "is spelled after the field it sits in; `_scope_parts` has dropped that field, so an "
        "`mrl.X` placed there is invisible to the floor and the core rule.")

    walked = sorted((f"{sub.value.id}.{sub.attr}", len(chain))
                    for statement in ast.parse(_EVERY_FIELD_SOURCE).body
                    for sub, chain in _scoped_walk(statement)
                    if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name))
    recorded = sorted(_EVERY_FIELD_FRAMES.items())
    assert walked == recorded, (
        "_scoped_walk evaluates these reads of _EVERY_FIELD_SOURCE in another number of scope "
        f"frames than _EVERY_FIELD_FRAMES records: walked {sorted(set(walked) - set(recorded))}, "
        f"recorded {sorted(set(recorded) - set(walked))}. Each read is spelled after its field, "
        "so `_scope_parts` has moved that field between its outer and inner lists, and a module "
        "alias read there now resolves in the wrong scope.")

    header = "from scripts.modal_runner import request\n\n"
    corpus = {
        "_SEAM_CLASSIFIER_PROBE": _SEAM_CLASSIFIER_PROBE,
        **{
            f"_SHADOWING_CASES[{name!r}]": header + case
            for name, case in _SHADOWING_CASES.items()
        },
        **{
            f"_UNSHADOWED_CASES[{name!r}]": header + case
            for name, case in _UNSHADOWED_CASES.items()
        },
        **_seam_sources(),
    }
    skipped = {}
    for label, source in corpus.items():
        missing = _expressions_the_scope_walk_skips(source)
        if missing:
            skipped[label] = missing[:5]
    assert not skipped, (
        "_scoped_walk never visits these expressions (first five per source), so `_scope_parts` "
        f"leaves out a field that _EVERY_FIELD_SOURCE does not populate: {skipped}")


def test_reach_floor_resolves_binding_target_through_binding_sites():
    """`binding_target("key")` resolves to `BINDING_SITES[key]`'s owner.

    Positive: the call passes in its owner's file. Negative: the same call in
    another (non-core) module's file fails, and an unknown key resolves to
    nothing, so a test whose only reach is a misspelt key fails even in the
    owner's file. The key's owner is read from `BINDING_SITES`, not typed here.
    """
    key = "attempt-watcher"
    owner = BINDING_SITES[key][0]
    other = next(module for module in RUNNER_MODULES if module not in (owner, "core"))
    header = "from tests.modal_patch_binding_campaign import binding_target\n\n"
    sources = {
        _floor_file(owner):
        header + (f'def test_installs_the_site():\n    return binding_target("{key}")\n\n'
                  'def test_installs_an_unknown_site():\n'
                  '    return binding_target("no-such-site")\n'),
        _floor_file(other):
        header + f'def test_installs_it_elsewhere():\n    return binding_target("{key}")\n',
    }
    expected = sorted([("reach-floor", _floor_file(owner), "test_installs_an_unknown_site"),
                       ("reach-floor", _floor_file(other), "test_installs_it_elsewhere")])
    assert _floor_rules(sources) == expected, (
        f"binding_target resolved wrongly: {_floor_rules(sources)}")


def test_reach_floor_follows_module_level_helpers():
    """A test that reaches its module only through helpers passes; without the helper it fails.

    The reach is two hops deep (`test -> _outer -> _inner -> mrl.X`), so the
    closure is transitive, not one level. Removing `_inner` leaves the test's
    call to `_outer` intact and the floor red, which is the negative half.
    """
    req = _floor_file("request")
    inner = f"def _inner():\n    return mrl.{_owned('request')}\n\n"
    rest = ("def _outer():\n    return _inner()\n\n"
            "def test_through_two_helpers():\n    return _outer()\n")
    header = "import scripts.modal_runner as mrl\n\n"
    assert _floor_rules({req: header + inner + rest}) == [], (
        "a test that reaches its module through two helpers failed the floor, so the closure "
        "is not transitive")
    without = _floor_rules({req: header + rest})
    expected = [("reach-floor", req, "test_through_two_helpers")]
    assert without == expected, (
        "a test whose helper chain no longer reaches its module still passed the floor, so the "
        f"floor is crediting reach it cannot see. got={without}")


def test_reach_floor_follows_helpers_in_the_shared_file():
    """The closure's namespace is the union: a shared-file helper carries its reach to a runner file.

    Positive: a state-file test whose only reach is a helper defined in the
    SHARED file passes. The helper's `state.X` resolves through the shared
    file's own import; the test's file imports nothing, so a floor that
    resolved every name through the test's file, or that closed over the
    test's file alone, is red here. Negative: remove the helper from the
    shared file and the same test fails.

    The union is `classify_seam`'s, so it holds the client file's names as well:
    the same helper defined in the CLIENT file carries its reach too. A union
    that dropped the client file is pinned here, not left to the classifier,
    which would call such a helper shared and the placement gate would then
    call it misplaced -- a different failure, reported elsewhere.
    """
    state_file = _floor_file("state")
    test_source = "def test_through_the_shared_file():\n    return _shared_state_helper()\n"
    shared_header = "from scripts.modal_runner import state\n\n"
    helper = "def _shared_state_helper():\n    return state.anything\n"
    for home in (SHARED_FILE, CLIENT_FILE):
        with_helper = _floor_rules({state_file: test_source, home: shared_header + helper})
        assert with_helper == [], (
            f"a test that reaches its module only through a helper in {home} failed the floor, "
            "so the closure is per-file or its namespace is not the union, or the helper's "
            f"reference was resolved through the wrong file's imports. got={with_helper}")
    without = _floor_rules({state_file: test_source, SHARED_FILE: shared_header})
    expected = [("reach-floor", state_file, "test_through_the_shared_file")]
    assert without == expected, f"the test still passed with the shared helper gone. got={without}"


def test_reach_floor_applies_the_core_rule_to_the_core_file():
    """In the core file, a test's OWN body may name only `core`; elsewhere the rule does not apply.

    Negative, one case per resolution rule and per place a reference can sit,
    because the core rule alone polices the core file and nothing on the real
    tree objects once the declared placement passes it: a core-file test whose
    own body names a module other than core through the facade
    (`mrl.<request name>`), a submodule alias (`training.X`) or a binding site
    (`binding_target("interrupt-loader")`, the case the core rule was written
    for: it resolves to `checkpoint`), or whose parametrize list names one, fails the
    core rule, although each also names core and so passes the floor. An
    EXEMPT core-file test fails it too: the core rule has no exemptions.

    Positive: a core-file test naming only core passes; so does one that
    reaches request only through a helper, because the rule is about the own
    body; so does one whose nested function takes a parameter spelled like a
    module (a local, not the module); and a request-file test naming core and
    request in its own body is not subject to it.
    """
    core_file, req = _floor_file("core"), _floor_file("request")
    site = "interrupt-loader"
    assert BINDING_SITES[site][0] != "core", f"{site} no longer names a non-core module"
    header = ("import scripts.modal_runner as mrl\n"
              "from scripts.modal_runner import training\n"
              "from tests.modal_patch_binding_campaign import binding_target\n\n")
    core_name, request_name = _owned("core"), _owned("request")
    sources = {
        core_file:
        header + (f"def _request_helper():\n    return mrl.{request_name}\n\n"
                  f"def test_core_only():\n    return mrl.{core_name}\n\n"
                  f"def test_core_and_request_in_own_body():\n"
                  f"    return mrl.{core_name}, mrl.{request_name}\n\n"
                  f"def test_core_and_a_submodule_alias():\n"
                  f"    return mrl.{core_name}, training.anything\n\n"
                  f"def test_core_and_a_binding_site():\n"
                  f"    return mrl.{core_name}, binding_target(\"{site}\")\n\n"
                  f"@pytest.mark.parametrize(\"value\", [mrl.{request_name}])\n"
                  f"def test_core_with_a_foreign_parametrize_list(value):\n"
                  f"    return mrl.{core_name}, value\n\n"
                  f"def test_core_with_a_local_named_like_a_module():\n"
                  f"    def callback(training):\n        return training.anything\n"
                  f"    return mrl.{core_name}, callback\n\n"
                  f"def test_exempt_but_names_request():\n    return mrl.{request_name}\n\n"
                  f"def test_request_only_through_a_helper():\n"
                  f"    return mrl.{core_name}, _request_helper()\n"),
        req:
        header + (f"def test_request_file_names_core_too():\n"
                  f"    return mrl.{request_name}, mrl.{core_name}\n"),
    }
    exempt = {
        (core_file, "test_exempt_but_names_request"):
        (frozenset({"request"}), "reaches no core, and is exempt")
    }
    got = _floor_rules(sources, exempt)
    expected = sorted(("core-rule", core_file, test) for test in (
        "test_core_and_request_in_own_body",
        "test_core_and_a_submodule_alias",
        "test_core_and_a_binding_site",
        "test_core_with_a_foreign_parametrize_list",
        "test_exempt_but_names_request",
    ))
    assert got == expected, (
        "the core rule fired on the wrong tests: "
        f"missed={sorted(set(expected) - set(got))} extra={sorted(set(got) - set(expected))}")


def test_reach_floor_exemption_must_still_be_needed():
    """Minimality: an exemption for a test that reaches its file's module fails; a needed one holds.

    Positive: a request-file test that reaches only core fails the floor with
    no exemption, and passes -- with no minimality objection -- once exempt
    with a reason. Negative: an exemption for a test that DOES reach request
    fails as not needed, whether it reaches request in its own body or only
    through a helper (minimality reads the whole reach, as the floor does); and
    an exemption whose reason is blank, empty, `None` or not a string fails.
    """
    req = _floor_file("request")
    sources = {
        req: ("import scripts.modal_runner as mrl\n\n"
              f"def _request_helper():\n    return mrl.{_owned('request')}\n\n"
              f"def test_reaches_its_module():\n    return mrl.{_owned('request')}\n\n"
              "def test_reaches_it_through_a_helper():\n    return _request_helper()\n\n"
              f"def test_reaches_only_core():\n    return mrl.{_owned('core')}\n"),
    }
    only_core = frozenset({"core"})
    needed = {(req, "test_reaches_only_core"): (only_core, "tests a test double, not a module")}
    assert _floor_rules(sources) == [("reach-floor", req, "test_reaches_only_core")]
    assert _floor_rules(sources, needed) == [], "a needed, reasoned exemption was not honoured"
    for stale_test in ("test_reaches_its_module", "test_reaches_it_through_a_helper"):
        got = _floor_rules(sources, {**needed, (req, stale_test): (only_core, "no longer true")})
        expected = [("exemption-not-needed", req, stale_test)]
        assert got == expected, (
            f"an exemption for {stale_test}, which reaches its file's module, passed minimality, "
            f"so the allow-set can go stale unnoticed. got={got}")
    for reason in ("  ", "", None, 0):
        got = _floor_rules(sources, {(req, "test_reaches_only_core"): (only_core, reason)})
        expected = [("exemption-without-reason", req, "test_reaches_only_core")]
        assert got == expected, f"an exemption with the reason {reason!r} was accepted. got={got}"


def test_reach_floor_exemptions_are_keyed_by_file():
    """An exemption keyed to another file does not exempt the test, and fails itself.

    Positive: keyed to the (file, test) that defines the failing test, the
    exemption holds. Negative, three shapes: keyed to a runner file that is
    among the sources but does not define the test; keyed to a runner file not
    among the sources at all (the shape a relocated exemption takes before its
    file exists, which must fail rather than be skipped); and keyed to a
    non-runner file that does define a test of that name. Each leaves the
    test's own floor violation standing, and each entry fails, with a detail
    that names its own cause, because each wants a different edit.
    """
    req, state_file = _floor_file("request"), _floor_file("state")
    absent = _floor_file("preflight")
    header = "import scripts.modal_runner as mrl\n\n"
    sources = {
        req: header + f"def test_moved_here():\n    return mrl.{_owned('core')}\n",
        state_file: header + f"def test_stays():\n    return mrl.{_owned('state')}\n",
        CLIENT_FILE: "def test_moved_here():\n    return 1\n",
    }
    entry = (frozenset({"core"}), "tests a test double, not a module")
    assert _floor_rules(sources, {(req, "test_moved_here"): entry}) == []
    causes = {
        state_file: "does not define test_moved_here",
        absent: "not among the seam sources",
        CLIENT_FILE: "not a per-module runner test file",
    }
    for wrong_file, cause in causes.items():
        violations, _ = reach_floor_violations(sources, RUNNER_OWNERS,
                                               {(wrong_file, "test_moved_here"): entry})
        got = [violation[:3] for violation in violations]
        expected = [("reach-floor", req, "test_moved_here"),
                    ("exemption-names-no-such-test", wrong_file, "test_moved_here")]
        assert got == expected, (
            f"an exemption keyed to {wrong_file} was honoured for the test in {req}, or was not "
            f"itself reported: {got}")
        assert cause in violations[-1][3], (
            f"the exemption keyed to {wrong_file} was reported without its cause ({cause!r}): "
            f"{violations[-1][3]!r}")


def test_reach_floor_exemption_is_no_wider_than_granted():
    """An exempt test's whole reach must EQUAL its grant; the entry's shape is checked first.

    The grant rule's positive control, as data. Positive: an
    exempt preflight test that reaches only `core`, granted `{core}` (the live
    entry's shape), passes. Negative: give it `request` and `state` as well --
    in its own body, or only through a helper, because the grant reads the
    whole reach as the floor does -- and it fails as wider than granted, with
    both sets in the detail. Minimality alone passed that edit, since the test
    still reaches no preflight. Granted the reach it now has, it passes again,
    so the rule compares with the grant rather than rejecting any extra module.
    Narrowed, a test granted `{core}` that reaches nothing fails too, and the
    empty grant is legal and holds for it; but a test that reaches `core` on
    an empty grant fails as wider, so an empty grant is not a wildcard.

    SHAPE. The value must be a `(granted, reason)` tuple: the older
    bare-reason shape fails as malformed rather than being unpacked (a
    two-character string would unpack), and so do a list, `None`, a triple, and
    a grant that is a mutable set, a tuple, a string, or a frozenset naming no
    runner module. RULE ORDER: a test that reaches its own file's module fails
    as not needed, not as wider, even when that reach also differs from its
    grant, because deleting the entry is the edit it wants.
    """
    pre = _floor_file("preflight")
    header = "import scripts.modal_runner as mrl\nfrom scripts.modal_runner import state\n\n"
    test = "test_reloads_the_double"
    body = f"def {test}():\n    return mrl.{_owned('core')}"
    granted, reason = frozenset({"core"}), "tests a test double, not a module"

    def rules(source, entry):
        return _floor_rules({pre: header + source}, {(pre, test): entry})

    assert rules(body + "\n", (granted, reason)) == [], "the exact grant was not honoured"
    extra = f"mrl.{_owned('request')}, state.anything"
    wider = {
        "in its own body": f"{body}, {extra}\n",
        "through a helper": f"def _helper():\n    return {extra}\n\n{body}, _helper()\n",
    }
    for where, source in wider.items():
        got = rules(source, (granted, reason))
        expected = [("exemption-wider-than-granted", pre, test)]
        assert got == expected, (
            f"an exempt test that reaches request and state {where}, beyond its grant of "
            f"{{core}}, passed, so its exemption is wider than granted. got={got}")
        regranted = rules(source, (frozenset({"core", "request", "state"}), reason))
        assert regranted == [], f"the grant of the test's exact reach was refused. got={regranted}"
    violations, _ = reach_floor_violations({pre: header + wider["in its own body"]}, RUNNER_OWNERS,
                                           {(pre, test): (granted, reason)})
    detail = violations[0][3]
    assert "reaches ['core', 'request', 'state'], granted ['core']" in detail, (
        f"the wider-than-granted detail does not name the reached and granted sets: {detail!r}")

    narrowed = f"def {test}():\n    return 1\n"
    got = rules(narrowed, (granted, reason))
    expected = [("exemption-wider-than-granted", pre, test)]
    assert got == expected, (
        f"an exempt test that no longer reaches its granted {{core}} passed. got={got}")
    assert rules(narrowed, (frozenset(), reason)) == [], "the empty grant was not honoured"
    # Its negative half: an empty grant is a grant of nothing, not a wildcard. A
    # rule that skipped the check for an empty grant would switch the floor off
    # again for any test so granted.
    got = rules(body + "\n", (frozenset(), reason))
    expected = [("exemption-wider-than-granted", pre, test)]
    assert got == expected, (
        f"an exempt test that reaches core passed on an empty grant, so an empty grant exempts "
        f"it from the grant check. got={got}")

    not_a_pair = [reason, "ok", [granted, reason], None, (reason, ), (granted, reason, "extra")]
    bad_grants = [{"core"}, ("core", ), "core", frozenset({"no_such_module"})]
    expected = [("exemption-malformed", pre, test)]
    for entry in [*not_a_pair, *((grant, reason) for grant in bad_grants)]:
        got = rules(body + "\n", entry)
        assert got == expected, f"the exemption value {entry!r} was not refused. got={got}"

    reaches_its_module = f"def {test}():\n    return mrl.{_owned('preflight')}\n"
    got = rules(reaches_its_module, (granted, reason))
    expected = [("exemption-not-needed", pre, test)]
    assert got == expected, (
        f"an exempt test that reaches its own file's module was not reported as not needed. "
        f"got={got}")


def test_reach_floor_scope_counts_every_runner_half_test_in_the_manifest():
    """`_floor_scope_problems`: each clause fires on its own cause; the real shape passes.

    Positive: a manifest with one test per runner test file, a client test, a
    guard (valued at the packaging file) and two helpers, and a floor that
    examined exactly the eight runner tests: no problem. The client test, the
    guard and the helpers are not runner-half tests, so a check that counted
    every `test_` entry, or every name, is red here.

    Negative, one per shape:
    - the floor skipped a pair: the equality clause, naming it manifest-only;
    - the floor examined a test the manifest lacks: named examined-only;
    - a widened runner set: the manifest values a runner-half
      test at a file outside `RUNNER_TEST_FILES`, which is what a seam reading
      a file the floor does not produces. The floor did not examine it, and the
      equality clause names it. The filter this replaced, "valued at a file of
      `RUNNER_TEST_FILES`", counted it on neither side and passed. Three such
      files: one named like a runner test file, one outside the
      `tests/test_modal_` prefix (a population re-narrowed by that pattern
      passes the first), and the shared file;
    - each runner test file in turn with no test in the manifest, and none
      examined: the empty-file clause alone, saying the file holds no test,
      not that the floor skipped it;
    - the floor examined a core-file test the manifest lacks, and the manifest
      has no other: both clauses, because the empty-file clause reads the
      manifest, not `examined`.
    """
    manifest = {f"test_in_{module}": _floor_file(module) for module in RUNNER_MODULES}
    manifest.update({
        "test_on_the_client_side": CLIENT_FILE,
        "test_modal_is_an_explicit_dependency_group": PACKAGING_FILE,
        "_shared_helper": SHARED_FILE,
        "_request_helper": _floor_file("request"),
    })
    examined = {(_floor_file(module), f"test_in_{module}") for module in RUNNER_MODULES}
    real_shape = _floor_scope_problems(examined, manifest)
    assert real_shape == [], f"the real shape failed: {real_shape}"

    req, core_file = _floor_file("request"), _floor_file("core")
    undeclared = "tests/test_modal_undeclared_probe.py"
    # Outside the `tests/test_modal_` prefix too, so a population re-narrowed
    # by that name pattern counts it on neither side and is red here.
    unprefixed = "tests/test_request_extra.py"
    assert undeclared not in RUNNER_TEST_FILES and unprefixed not in RUNNER_TEST_FILES
    no_core = {name: rel for name, rel in manifest.items() if rel != core_file}
    core_pair = (core_file, "test_in_core")
    core_empty = f"{[core_file]} hold(s) no test in the seam manifest"
    cases = {
        "the floor skipped a pair": (examined - {(req, "test_in_request")}, manifest,
                                     [f"manifest-only={[(req, 'test_in_request')]}"]),
        "the floor examined a test the manifest lacks":
        (examined | {(req, "test_unknown")}, manifest,
         [f"examined-only={[(req, 'test_unknown')]} manifest-only=[]"]),
        "an examined core test the manifest lacks":
        (examined, no_core, [f"examined-only={[core_pair]}", core_empty]),
    }
    for home in (undeclared, unprefixed, SHARED_FILE):
        valued_elsewhere = {**manifest, "test_valued_elsewhere": home}
        fragment = f"manifest-only={[(home, 'test_valued_elsewhere')]}"
        cases[f"a runner-half test valued at {home}"] = (examined, valued_elsewhere, [fragment])
    # One per runner test file, so an empty-file clause that skips any of them is red.
    for module in RUNNER_MODULES:
        rel = _floor_file(module)
        without = {name: at for name, at in manifest.items() if at != rel}
        fragment = f"{[rel]} hold(s) no test in the seam manifest"
        cases[f"{rel} with no test"] = (examined - {(rel, f"test_in_{module}")}, without,
                                        [fragment])
    for label, (examined_case, manifest_case, expected) in cases.items():
        got = _floor_scope_problems(examined_case, manifest_case)
        assert len(got) == len(expected) and all(
            fragment in problem for fragment, problem in zip(expected, got, strict=True)), (
                f"{label}: expected one problem per fragment {expected}, got {got}")


# One source per import shape, with the imports
# `_runner_imports_the_floor_cannot_resolve` must report, as it spells them.
# One case per alternative on each side: every way an import can name the
# runner and bind no facade or module alias, and every legal spelling next to
# it, so a check that dropped any branch or allowed any of them is red.
_RUNNER_IMPORT_CASES = {
    "a facade name imported by itself": ("from scripts.modal_runner import build_run_request\n",
                                         ["from scripts.modal_runner import build_run_request"]),
    "a facade name beside a module in one import":
    ("from scripts.modal_runner import build_run_request, core\n",
     ["from scripts.modal_runner import build_run_request"]),
    "a name imported from a submodule": ("from scripts.modal_runner.core import VOLUME_MOUNT\n",
                                         ["from scripts.modal_runner.core import VOLUME_MOUNT"]),
    "a star import": ("from scripts.modal_runner import *\n",
                      ["from scripts.modal_runner import *"]),
    "an unaliased import of the facade, which binds `scripts`": ("import scripts.modal_runner\n",
                                                                 ["import scripts.modal_runner"]),
    "an unaliased import of a module, which binds `scripts`":
    ("import scripts.modal_runner.core\n", ["import scripts.modal_runner.core"]),
    "an alias of a runner path that is no runner module":
    ("import scripts.modal_runner.no_such_module as nosuch\n",
     ["import scripts.modal_runner.no_such_module as nosuch"]),
    "a name imported by itself inside a test":
    ("def test_x():\n    from scripts.modal_runner import build_run_request\n"
     "    return build_run_request\n", ["from scripts.modal_runner import build_run_request"]),
    "the facade alias": ("import scripts.modal_runner as mrl\n", []),
    "the facade imported from its parent":
    ("from scripts import modal_runner\nfrom scripts import modal_runner as facade\n", []),
    "modules imported from the facade": ("from scripts.modal_runner import request, core as c\n",
                                         []),
    "a module alias": ("import scripts.modal_runner.request as req\n", []),
    "a module imported inside a test":
    ("def test_x():\n    from scripts.modal_runner import request\n    return request\n", []),
    "a relative import spelled like the runner":
    ("from .scripts.modal_runner import build_run_request\n", []),
    "a package whose name extends the runner's":
    ("import scripts.modal_runner_extra\nfrom scripts.modal_runner_extra import thing\n", []),
    "imports of other packages": ("import os\nfrom scripts import run_modal\n"
                                  "from tests.modal_patch_binding_campaign import binding_target\n",
                                  []),
}


def test_a_runner_import_must_bind_the_facade_or_a_whole_module():
    """A runner import in a seam file that binds anything but the facade or a module is reported.

    THE HOLE THIS PINS. The floor and the core
    rule resolve `mrl.X`, `<module alias>.X` and `binding_target("key")`, and
    nothing else. A core-file test that imports `build_run_request` by itself
    and calls it passes the core rule; a request-file test doing the same
    fails the floor, whose remedy then points at a move or an exemption. The
    split test asserts this census is empty before it runs the floor.

    Each of `_RUNNER_IMPORT_CASES` alone reports exactly its imports; across
    several files every one is named with its file and line. The last
    assertion pins WHY the import is rejected rather than read: the core rule
    does not see a call through it. A change that resolves such names must
    flip that assertion in the same commit, and may then relax the rejection.
    """
    wrong = {}
    for label, (source, expected) in _RUNNER_IMPORT_CASES.items():
        got = [
            spelling
            for _, _, spelling in _runner_imports_the_floor_cannot_resolve({"f.py": source})
        ]
        if got != expected:
            wrong[label] = got
    assert not wrong, f"runner imports reported wrongly (expected per case): {wrong}"

    core_file = _floor_file("core")
    by_name = "from scripts.modal_runner import build_run_request, core\n"
    sources = {
        core_file: "import sys\n" + by_name,
        SHARED_FILE: "import scripts.modal_runner\n",
        CLIENT_FILE: "import scripts.modal_runner as mrl\n",
    }
    got = _runner_imports_the_floor_cannot_resolve(sources)
    expected = sorted([(core_file, 2, "from scripts.modal_runner import build_run_request"),
                       (SHARED_FILE, 1, "import scripts.modal_runner")])
    assert got == expected, f"the census did not name each import by file and line: {got}"

    core_test = by_name + "\ndef test_c():\n    return core.anything, build_run_request()\n"
    assert _floor_rules({core_file: core_test}) == [], (
        "the core rule now sees a call through a runner name imported by itself: update this "
        "probe, and consider relaxing `_runner_imports_the_floor_cannot_resolve`")


def test_modal_is_an_explicit_dependency_group():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["dependency-groups"]["modal"] == ["modal>=1.4.3,<2"]


def test_local_entrypoints_do_not_import_modal():
    # sys.path.insert("src"): train.py resolves its generated siblings with bare
    # imports (`from _action_spec import ...`), so the src dir itself must be on
    # the child's path. Relying on the editable install's .pth instead would make
    # this test pass/fail on ambient venv state (any `uv sync --no-install-project`
    # removes it) and could silently import siblings from a DIFFERENT checkout.
    code = """
import sys
sys.path.insert(0, "src")
import src.train
import scripts.exp_lib
import scripts.run_experiment
assert 'modal' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_modal_runner_does_not_import_modal_or_torch():
    # Fresh subprocess: the parent may already have torch (the resume-input
    # checkpoint tests) or modal (later runner tests) in sys.modules.
    code = """
import sys
import scripts.modal_runner
assert 'modal' not in sys.modules
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def _runner_module_population(repo_root):
    """The package's submodule files as found ON DISK, checked against RUNNER_MODULES.

    Returns `(candidates, undeclared)` as repo-relative posix paths.
    `candidates` is every declared module in RUNNER_MODULES order, followed by
    every undeclared `*.py` in `scripts/modal_runner/`, sorted; `undeclared` is
    that tail alone.

    WHY the directory and not the constant: a scan whose population is its own
    declared list cannot see a file nobody declared. Measured by the W3b
    structural review (2026-09-22): a real `scripts/modal_runner/utils.py`
    holding `import modal` and a module-scope call passed every structural gate
    while the population was RUNNER_MODULES. The glob is the one
    `scripts/run_modal.py`'s mount loop uses, so every file the image ships
    other than the facade is a file these gates read.

    PITFALLS: declared modules stay in `candidates` even when absent, so a
    reader raises with the missing member's path instead of silently scanning
    seven. `__init__.py` is excluded: it is the facade and declares nothing.
    Its module scope has a stricter rule of its own (docstring, one-dot
    relative imports, a literal `__all__`), `_facade_violations` in
    tests/test_modal_runner_package_shape.py; the surface gates in
    tests/test_modal_client.py pin the names it exports.
    pathlib's `*` also matches dotfiles, and `*.py` matches a directory so
    named. The mount loop globs both too: a regular dotfile ships, while a
    dangling one (an Emacs `.#core.py` lock is a dangling symlink) and the
    directory fail the image build (FileNotFoundError / IsADirectoryError;
    Modal does not check the path when the image is defined). So both are in
    the population here, and a gate reading either raises.
    """
    root = Path(repo_root)
    declared = list(RUNNER_PATHS)
    found = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "scripts" / "modal_runner").glob("*.py") if path.name != "__init__.py")
    undeclared = [rel for rel in found if rel not in declared]
    return declared + undeclared, undeclared


def _is_type_checking_test(test):
    """True iff an `if` statement's test is `TYPE_CHECKING` or `<anything>.TYPE_CHECKING`.

    The one definition of a `TYPE_CHECKING` block for every gate that reads
    one: gate (e) exempts its body from the purity check, gate (g) allows one
    imports-only block at module scope, and the package-shape dependency walk
    (tests/test_modal_runner_package_shape.py) files its imports under
    ANNOTATION_DEPENDENCIES. They used to hold three copies of this test, and a
    copy that drifted would make one gate exempt a block another calls illegal.

    Matching by name has an accepted limit (Ruling 29): `if os.TYPE_CHECKING:`
    is recognised too. It fails closed the other way: `if TYPE_CHECKING and X:`
    is a `BoolOp` and is not recognised. A bare `TYPE_CHECKING` is only
    typing's while nothing rebinds it (`from os import environ as
    TYPE_CHECKING` makes the block run); inside scripts/modal_runner/ the
    package-shape `trusted-binding` clause rejects any such binding.
    """
    return ((isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
            or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"))


# The synthetic module text the gate (e) and gate (g) knock-outs plant into.
_MONOLITH_STUB = ROOT / "tests" / "fixtures" / "modal_runner_monolith_stub.txt"


def _monolith_stub_source():
    """The knock-outs' fixture text, after checking each property they rely on.

    WHY A CHECKED-IN STUB: five knock-outs of gates (e) and (g) were written
    against the pre-split monolith and read it with
    `git show 2bb32ac:scripts/modal_runner_lib.py`, so they failed in a
    shallow clone or a source tarball (gh#223). None of the live package
    modules can stand in: they change with every edit, all but core.py hold
    relative imports, and core.py's `__future__` import is not its line 19.

    The knock-outs splice each plant in after line 19 (`_splice_into_stub`),
    assert violation line numbers counted from there, and argue that every
    row is inert on the unplanted text. Each property those arguments rest on
    is asserted here, so an edit to the stub that breaks one fails naming it.
    """
    text = _MONOLITH_STUB.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    nodes = list(ast.walk(tree))
    body_kinds = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.FunctionDef,
                  ast.ClassDef)
    plain = [
        alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names
    ]
    froms = [node.module for node in tree.body if isinstance(node, ast.ImportFrom)]
    lazy = [
        alias.name for function in nodes if isinstance(function, ast.FunctionDef)
        for node in ast.walk(function) if isinstance(node, ast.Import) for alias in node.names
    ]
    claims = [
        ("line 19 is `from __future__ import annotations`", len(lines) > 18
         and lines[18] == "from __future__ import annotations\n"),
        ("module scope holds only the docstring, imports and declarations (so no `if`, "
         "`try` or `TYPE_CHECKING` block)", ast.get_docstring(tree) is not None
         and all(isinstance(node, body_kinds) for node in tree.body[1:])),
        ("every module-scope import is stdlib, and no plain import is dotted",
         all("." not in name and name in sys.stdlib_module_names for name in plain)
         and all(module and module.split(".")[0] in sys.stdlib_module_names for module in froms)),
        ("no relative import", not any(isinstance(n, ast.ImportFrom) and n.level for n in nodes)),
        ("no async def", not any(isinstance(n, ast.AsyncFunctionDef) for n in nodes)),
        ("no import in a class body", not any(
            isinstance(child, (ast.Import, ast.ImportFrom))
            for n in nodes if isinstance(n, ast.ClassDef) for child in n.body)),
        ("torch is imported in exactly two function bodies", lazy == ["torch", "torch"]),
    ]
    broken = [claim for claim, holds in claims if not holds]
    assert not broken, (
        f"{_MONOLITH_STUB.relative_to(ROOT)} no longer satisfies: {broken}. The gate (e) and "
        "gate (g) knock-outs' line numbers and inertness arguments rest on each; restore it.")
    return text


def _splice_into_stub(text):
    """The stub with `text` inserted after its line 19, the `__future__` import.

    A plant above that line would raise `SyntaxError: from __future__ imports
    must occur at the beginning of the file`, a red for the wrong reason.
    Nothing here is executed, only parsed, so a plant may name `os` or `modal`
    without either being importable.
    """
    lines = _monolith_stub_source().splitlines(keepends=True)
    return "".join(lines[:19]) + text + "".join(lines[19:])


def _nonstdlib_module_scope_imports(repo_root, top_level_only=False, *, paths=None):
    """Criterion 5: non-stdlib imports that execute when a package module is imported.

    Returns `(scanned, violations)`: the repo-relative paths actually read, and
    sorted `(path, lineno, root module)` tuples. The default population is
    `_runner_module_population`'s, so an undeclared module on disk is scanned
    too and shows up in `scanned`; `paths` replaces it for controls that plant
    into a single file.

    Function bodies and recognized TYPE_CHECKING bodies are exempt; class bodies,
    other containers, and TYPE_CHECKING else branches execute and are inspected.
    Missing/unreadable files raise with their path rather than pretending to scan.
    """
    root = Path(repo_root)
    candidates = [
        root / rel for rel in (paths if paths is not None else _runner_module_population(root)[0])
    ]
    scanned = []
    violations = []
    for path in candidates:
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        scanned.append(rel)
        # Annotated `ast.AST` rather than inferred `ast.stmt`, because
        # `ast.iter_child_nodes` yields `AST` (expressions and `alias` nodes
        # included) and the pre-commit pyrefly gate reds on the unannotated
        # `stack.extend`. Nothing below reads an attribute that only `stmt` has:
        # `node.lineno` is reached solely inside the `Import` / `ImportFrom`
        # narrowing.
        nodes: list[ast.AST]
        if top_level_only:
            nodes = list(tree.body)
        else:
            nodes = []
            stack: list[ast.AST] = list(tree.body)
            while stack:
                node = stack.pop()
                nodes.append(node)
                # A function body does not run on import. A CLASS body does, so
                # it is walked; the methods inside it are `FunctionDef`s and get
                # skipped here on the next iteration, which is the right answer
                # for the same reason.
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                # `if TYPE_CHECKING:` never runs, so its body costs nothing at
                # import and is exempt. `node.orelse` IS still walked -- that is
                # precisely the branch that does run -- and skipping the whole
                # `If` node would be a real blind spot rather than an exemption.
                if isinstance(node, ast.If) and _is_type_checking_test(node.test):
                    stack.extend(node.orelse)
                    continue
                stack.extend(ast.iter_child_nodes(node))
        for node in nodes:
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = [node.module.split(".")[0]]
            else:
                continue
            for name in roots:
                if name not in sys.stdlib_module_names:
                    violations.append((rel, node.lineno, name))
    return scanned, sorted(violations)


def test_gate_e_criterion_5_reddens_on_plants_the_tree_body_instrument_misses(tmp_path):
    """KNOCK-OUT. Plants live in a `tmp_path` file named
    `scripts/modal_runner/core.py` whose text is the stub from
    `_monolith_stub_source`, not the live package; nothing live is edited (GC2).

    THE DELIVERABLE IS THE SECOND ASSERTION of the first block. A version of this
    gate built on `tree.body` PASSES the decoy -- that is the exact blindness the
    decoy exists to prevent -- so the blind instrument's green is ASSERTED here,
    not described in prose. Prose cannot fail.

    Every plant goes AFTER LINE 19 (GC2). Line 19 is
    `from __future__ import annotations`, the first statement after the
    docstring, and a `__future__` import must precede all other code: a plant
    above it raises `SyntaxError: from __future__ imports must occur at the
    beginning of the file`, which is a red for the wrong reason.
    `_monolith_stub_source` asserts the plant site, so an edit that moves that
    line is loud.

    THE TABLE IS ONE ROW PER INDEPENDENT CLAUSE of the gate, because a gate whose
    headline clause has a knock-out and whose secondary clauses have none is the
    defect this branch has shipped three times. Nothing here is a variation on
    the decoy for its own sake -- each row is the only observation that kills one
    mutant:

      * `import modal as ...` is the ONLY row that objects to resolving names
        through `alias.asname`; that mutant is silent on the stub AND on
        the decoy, because a plain `import modal` has no `asname`.
      * `from modal.functions import ...` objects to dropping `ImportFrom`
        handling entirely (silent on the stub, whose from-imports are all
        stdlib) and to keeping the full dotted path instead of the root.
      * `import xml.etree.ElementTree` is the same root-resolution clause for
        `Import`, and it was MISSING until fault seeding found it: the stub has
        no dotted plain `import` at module scope, and neither did any other row
        here, so `alias.name` without `.split(".")[0]` survived everything. It is
        a FALSE-RED mutant rather than a false-green one -- the gate would have
        reported `os.path` as non-stdlib -- and a gate that reddens on legal code
        gets switched off, which is the failure mode this file is built around.
      * `import numpy` objects to a DENY-LIST implementation. A gate that looks
        for `modal` and `torch` by name passes the decoy and every other row
        here; `sys.stdlib_module_names` is an ALLOW-LIST, and the polarity is the
        reason `ruff` TID253 was measured GREEN on the decoy and rejected.
      * the two-violation row is the only place either "report just the first
        offender" or "return the walk's own order" is visible; every other row
        has at most one violation, where both mutants are identity functions.
      * the missing-target block at the end objects to skipping an unreadable
        target, and it is what makes the `scanned` assertions mean anything: a
        file the gate could not read must raise, never drop out of `scanned`
        and leave an empty violation list behind.
      * the `if`-guarded row objects to adding `If` to the skip set, and it is
        the shape gate (g) plants (`if importlib.util.find_spec(...)`), so the
        two gates' readings of the same construct stay pinned together. Its test
        is a `Call`, so it also stops the `TYPE_CHECKING` exemption from widening
        into "any `If`" -- but a `Call` is neither a `Name` nor an `Attribute`,
        which leaves the widenings that keep the node-class test and drop the
        NAME test invisible to it. THE EXEMPTION HAS TWO CLAUSES AND NEEDS THREE
        ROWS; this one holds only the outermost. The next two hold the rest, and
        they exist because "the headline clause has a knock-out, the secondary
        clause has none" is this branch's named defect class in its B2 form --
        the guard's own allow-set going unwatched.
      * the `nametestguard` row (`_DEBUG = False` + `if _DEBUG:`) objects to
        matching any bare `ast.Name` test. Drop `test.id == "TYPE_CHECKING"` and
        every flag-guarded module-scope import in the file becomes exempt: that
        is a SILENT GREEN ON A REAL IMPORT, not a missed edge case. No other row
        here has a bare-`Name` `If` test THAT IS NOT `TYPE_CHECKING` -- the
        `typechecking` and `typecheckingelse` rows have one each, and both stay
        green under that mutant because the exemption is exactly what they
        exercise -- so nothing else can see it.
      * the `attrtestguard` row (`import os` + `if os.name:`) is the same clause
        on the attribute spelling. `typing.TYPE_CHECKING` is exempt because of
        `test.attr`, NOT because it is an `ast.Attribute`; without this row,
        `isinstance(test, ast.Attribute)` on its own passes every other
        observation in the file.
      * the CLASS-BODY row objects to putting `ClassDef` back in the skip set.
        A class body executes at import time, so this is a real violation; the
        stub's class holds no import, so nothing else can see it.
      * the class-METHOD row is the other half of that change. Once the walk
        enters class bodies it reaches the methods inside them, and a method is
        a `FunctionDef` whose body does not run on import -- this row is what
        says the walk stops there instead of flagging every lazy import in a
        class.
      * the NESTED-CONTAINER row (`class` > `try` > `class` > `import modal`)
        is the only row whose offending import sits inside a NESTED container
        -- three containers deep, not one. IN CONTAINERS, NOT DEPTHS, and the
        distinction is the claim: under this file's own unit (nodes below
        `Module`, top-level = 1) six other rows put their offending import
        below depth 1 as well, all of them at depth 2. What makes this row the
        only one is that every other container in this table -- the decoy's
        `try`, the `if`-guarded header, the class body -- is TOP-LEVEL, and a
        top-level node enters the walk from `tree.body` rather than by being
        descended into, so a mutant that expands depth 1 and then stops passes
        every one of them while going blind to everything nested inside. The
        nesting runs `ClassDef` > `Try` > `ClassDef` > `Import` ON PURPOSE. It
        is one row, but it objects to dropping EITHER container type from the
        descent as well as to dropping both -- MEASURED: the obvious two-deep
        shapes each hold only half of that, a `try:` inside a class body is
        blind to "stop at a nested `ClassDef`" and a class inside a `try:` is
        blind to "stop at a nested `Try`", while this one kills all three
        mutants. The instrument is a stack, not a single expansion, and this
        is the only observation in the file that says so.
      * the two `TYPE_CHECKING` rows are POSITIVE CONTROLS FOR AN EXEMPTION,
        which is the shape that has gone wrong here before: the subject has 0
        such blocks, so an unexercised exemption is indistinguishable from a
        gate that never meets one. Both spellings are covered (bare
        `TYPE_CHECKING`, `typing.TYPE_CHECKING`), and each imports something
        NON-stdlib on purpose -- `import decimal` inside the block would go green
        whether or not the exemption existed and would certify nothing.
      * the `TYPE_CHECKING`-with-`else` row objects to exempting the whole `If`
        node rather than just its body. The `else:` branch is exactly the branch
        that DOES execute, so skipping it would be a blind spot wearing an
        exemption's clothes.
      * the function-body `import torch` row is the NEGATIVE control that
        separates this instrument from bare `ast.walk`. It is the shape the stub
        uses twice, so tests/test_modal_runner_package_shape.py's
        `test_package_import_purity_scans_every_module` already objects to
        descending -- this row says so where a reader of the
        instrument is standing.
      * the `async def` row is the only thing that objects to dropping
        `AsyncFunctionDef` from the skip set. The stub has no async def, so
        every other observation here is blind to that entry.
      * the relative-import row is the only thing that objects to dropping
        `node.level == 0`. Silent on the stub, which has no relative
        import; the package modules all have them, and without this row the
        gate would redden over legal intra-package imports.

    EVERY ROW BELOW IS INERT ON THE UNMODIFIED SUBJECT, by construction, and
    `_monolith_stub_source` asserts each property: the stub has no class-body
    import, no async def, no `TYPE_CHECKING` block, no
    relative import, no dotted plain `import`, no conditional module-scope import
    of any spelling, and nothing imported inside a container that RUNS at import
    time. Stated that way rather than as "nothing below depth 1", which is false:
    the stub has two imports below depth 1, both `torch`, in `_import_torch`
    and `_load_checkpoint_weights` -- function bodies, which is the `lazy` row's
    subject and does not execute on import. That is the point. Each one is the
    ONLY observation in the suite that holds its clause, which is why they are
    rows in a knock-out rather than sentences in a docstring.

    INTERACTION WITH GATE (g), criterion 13 -- stated here and in the green test
    because the two `TYPE_CHECKING` rows and the `if`-guarded row look, from the
    outside, like the same construct getting two different verdicts. They are not
    the same construct:

      * gate (g)'s `find_spec` header is green under gate (e) THROUGH THE WALK,
        not through the exemption. It is an ordinary `If`, and gate (e) does read
        the imports inside it -- which is exactly what the `ifguard` row asserts.
      * the two gates disagree about the `TYPE_CHECKING` BODY on purpose.
        Criterion 13 governs its SHAPE (at most one block, imports only);
        criterion 5 exempts it from the purity check because it never executes.
        Complementary, not contradictory, and neither subsumes the other.
      * so if a future edit moves an `import` INSIDE the `find_spec` header,
        gate (e) reddens -- CORRECTLY. That import runs.
    """

    def plant(text, slug):
        """Write a tmp_path repo whose `scripts/modal_runner/core.py` is the
        stub with `text` spliced in after line 19; return its root."""
        target = tmp_path / slug / "scripts" / "modal_runner" / "core.py"
        target.parent.mkdir(parents=True)
        target.write_text(_splice_into_stub(text), encoding="utf-8")
        return target.parents[2]

    decoy = plant("try:\n    import modal\nexcept ImportError:\n    modal = None\n", "decoy")
    scanned, violations = _nonstdlib_module_scope_imports(decoy,
                                                          paths=["scripts/modal_runner/core.py"])
    assert scanned == [
        "scripts/modal_runner/core.py"
    ], (f"the knock-out did not reach the planted copy, so its red below would be "
        f"unrelated to the plant. Scanned: {scanned}")
    assert violations == [
        ("scripts/modal_runner/core.py", 21, "modal")
    ], ("gate (e) missed a try-wrapped module-scope `import modal` -- the exact shape an "
        "optional dependency gets written in, and the one that makes importing the runner "
        f"library cost a Modal import. Reported: {violations}")

    _, blind = _nonstdlib_module_scope_imports(decoy,
                                               top_level_only=True,
                                               paths=["scripts/modal_runner/core.py"])
    assert blind == [], (
        "`tree.body` was supposed to be BLIND to this plant, and it is the blindness this "
        "gate exists to rule out. If it now sees the plant, the two instruments no longer "
        "differ anywhere measurable and this knock-out has stopped certifying that gate "
        f"(e) walks past depth 1. tree.body reported: {blind}")

    for slug, text, expected, clause in [
        ("asname", "import modal as _modal_shim\n", [("scripts/modal_runner/core.py", 20, "modal")],
         "resolve `Import` through `alias.name`, never `alias.asname`"),
        ("fromimport", "from modal.functions import FunctionCall\n", [
            ("scripts/modal_runner/core.py", 20, "modal")
        ], "`ImportFrom` counts, and its module resolves to the ROOT, not the dotted path"),
        ("denylist", "import numpy\n", [("scripts/modal_runner/core.py", 20, "numpy")],
         "the check is an allow-list over `sys.stdlib_module_names`, not a deny-list"),
        ("two", "import numpy\nimport modal\n", [("scripts/modal_runner/core.py", 20, "numpy"),
                                                 ("scripts/modal_runner/core.py", 21, "modal")],
         "EVERY offending import is reported, sorted. The walk pops its stack from the "
         "end, so an unsorted return hands these back in reverse source order and a "
         "report-only-the-first implementation drops the second -- and this is the only "
         "row with two violations in one file, so nothing else can see either mutant"),
        ("ifguard", 'if os.environ.get("CS2RL_MODAL"):\n    import modal\n', [
            ("scripts/modal_runner/core.py", 21, "modal")
        ], "the walk descends through `If` -- gate (g) plants exactly this header"),
        ("classbody", "class _C:\n    import modal\n", [
            ("scripts/modal_runner/core.py", 21, "modal")
        ], "a CLASS BODY EXECUTES at import time, so it is walked, not skipped"),
        ("classmethod", "class _C:\n    def m(self):\n        import torch\n        return torch\n",
         [], "walking class bodies must not reach METHOD bodies -- a method is a `FunctionDef` "
         "and does not run on import"),
        ("nestedcontainers", "class _C3:\n    try:\n        class _C4:\n            import modal\n"
         "    except ImportError:\n        pass\n", [("scripts/modal_runner/core.py", 23, "modal")],
         "descent is a STACK and not a single expansion -- an import THREE containers "
         "deep still executes at import time. Every other container in this table is "
         "top-level, so a walk that expands depth 1 and then stops passes all of them"),
        ("typechecking", "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import modal\n",
         [], "`if TYPE_CHECKING:` NEVER runs, so it costs nothing at import and is exempt. The "
         "import inside is deliberately NON-stdlib: `import decimal` would pass for the wrong "
         "reason and certify nothing"),
        ("typecheckingattr", "import typing\nif typing.TYPE_CHECKING:\n    import modal\n", [],
         "the `typing.TYPE_CHECKING` spelling is the same exemption; matching only the bare "
         "`Name` leaves half the idiom reddening"),
        ("typecheckingelse",
         "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import modal\n"
         "else:\n    import numpy\n", [
             ("scripts/modal_runner/core.py", 24, "numpy")
         ], "the `else:` branch of a `TYPE_CHECKING` block is exactly the branch that DOES run, so "
         "exempting the whole `If` node instead of just its body is a blind spot, not an "
         "exemption"),
        ("dotted", "import xml.etree.ElementTree\n", [],
         "a DOTTED `Import` resolves to its ROOT -- `sys.stdlib_module_names` holds "
         "top-level names only, so keeping the dotted path reddens the gate over stdlib"),
        ("lazy", "def _lazy():\n    import torch\n    return torch\n", [],
         "a function-body import is LEGAL; descending into it is what makes bare "
         "`ast.walk` red on the correct module"),
        ("asynclazy", "async def _lazy_async():\n    import torch\n    return torch\n", [],
         "`AsyncFunctionDef` is in the skip set for the same reason `FunctionDef` is; "
         "the module has 0 async defs today, so nothing else objects to dropping it"),
        ("relative", "from .paths import RUN_ROOT\n", [],
         "a relative import is intra-package, not a third-party dependency -- "
         "`node.level == 0`"),
        ("nametestguard", "_DEBUG = False\nif _DEBUG:\n    import modal\n", [
            ("scripts/modal_runner/core.py", 22, "modal")
        ], "the `TYPE_CHECKING` exemption matches the NAME, not merely the node class -- a "
         "bare `Name` test that is not `TYPE_CHECKING` still runs, so the import under it "
         "still costs what it costs"),
        ("attrtestguard", "import os\nif os.name:\n    import modal\n", [
            ("scripts/modal_runner/core.py", 22, "modal")
        ], "same clause on the attribute spelling: `x.TYPE_CHECKING` is exempt, any other "
         "attribute test is not -- the exemption reads `test.attr`, not `isinstance(test, "
         "ast.Attribute)`"),
    ]:
        scanned, violations = _nonstdlib_module_scope_imports(
            plant(text, slug), paths=["scripts/modal_runner/core.py"])
        assert scanned == [
            "scripts/modal_runner/core.py"
        ], (f"the {slug} plant was not read at all, so its verdict is vacuous: {scanned}")
        assert violations == expected, f"gate (e) is wrong about: {clause}. Got {violations}"

    # A root with no target at all. The helper must raise, naming the file,
    # because an unread target that silently dropped out of `scanned` would
    # leave an empty violation list -- a vacuous green. Filtering candidates
    # through `is_file()` (the W3a behaviour) is the mutant this pins.
    empty = tmp_path / "noscripts"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="core.py"):
        _nonstdlib_module_scope_imports(empty, paths=["scripts/modal_runner/core.py"])


def _module_scope_shape_violations(tree):
    """Classify every module-scope statement; what a declaration contains is not checked.

    A block is recognised by `_is_type_checking_test`, the predicate gate (e)
    uses. A single imports-only block is legal, without else.
    """
    examined = 0
    violations = []
    type_checking_blocks = 0
    for index, node in enumerate(tree.body):
        examined += 1
        kind = type(node).__name__
        if isinstance(node, ast.Expr):
            # The docstring, and only the docstring. `index == 0` is the whole
            # rule: a string literal anywhere else at module scope is either
            # dead or a comment somebody wrote wrong, and a non-string `Expr`
            # is a bare CALL, which is the case the kind census hides.
            if index == 0 and isinstance(node.value, ast.Constant) and isinstance(
                    node.value.value, str):
                continue
            violations.append(
                (kind, node.lineno, "only the module docstring may be a bare expression at module "
                 "scope"))
        elif isinstance(node, ast.If):
            if not _is_type_checking_test(node.test):
                violations.append(
                    (kind, node.lineno, "the only conditional allowed at module scope is "
                     "`if TYPE_CHECKING:`"))
                continue
            type_checking_blocks += 1
            if node.orelse:
                violations.append(
                    ("If", node.lineno, "TYPE_CHECKING blocks must have no else branch"))
            if type_checking_blocks > 1:
                violations.append(
                    (kind, node.lineno, "a module may hold at most one `if TYPE_CHECKING:` block"))
            # The offending BODY statement is what gets named, not the `If`.
            # Whoever meets this red has to be told which line to delete, and
            # "the block is wrong" does not say that.
            for inner in node.body:
                if not isinstance(inner, (ast.Import, ast.ImportFrom)):
                    violations.append((type(inner).__name__, inner.lineno,
                                       "an `if TYPE_CHECKING:` body may hold imports only"))
        # Everything else at module scope is an import or a declaration, or it
        # is a violation. `Expr` and `If` are deliberately absent: both are legal in exactly one shape, so
        # they are decided by the branches above. Putting either here is the
        # blind kind-only allow-list this whole gate exists to rule out.
        elif not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                   ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            violations.append(
                (kind, node.lineno, "only imports and declarations may appear at module scope"))
    return examined, violations


def test_criterion_13_reddens_on_the_conditional_modal_probe(tmp_path):
    """KNOCK-OUT 1, and the criterion-13 half of a two-file pair.

    THE PLANT IS THE HEADER EVERY OTHER GATE IN THIS PLAN IS GREEN ON:

        import importlib.util
        if importlib.util.find_spec('modal') is not None:
            _ON_CONTAINER = True
        else:
            _ON_CONTAINER = False

    It never imports Modal, so a container-equivalent import of the package
    succeeds and criterion 3 stays green. Its body assigns rather than imports,
    so criterion 5 stays green -- and note WHY, because it is not the obvious
    reason: this is an ORDINARY `If`, so gate (e) DESCENDS THROUGH IT and does
    read the imports inside. It is green because there are none, not because of
    the `TYPE_CHECKING` exemption (Ruling 30). Move an `import` inside this
    header and gate (e) reddens CORRECTLY -- that import runs -- and the red is a
    plant-design error here, not a gate (e) defect.

    BOTH HALVES OF THE CONTRAST ARE ASSERTED, NOT DESCRIBED. Criterion 5's half
    is the second block below. Criterion 3's half is a shipped test in the other
    file:

        tests/test_modal_client.py::test_gate_a_criterion_3_stays_green_on_a_conditional_modal_probe

    which names this test by node id in return. It lives there rather than here
    because `tests/test_modal_packaging.py` imports no module under tests/ but
    two it reads data from, tests/modal_runner_tables.py and the
    `BINDING_SITES` dict of tests/modal_patch_binding_campaign.py, and that
    exclusion is load-bearing -- `_seam_sources()` omits this file because
    self-feeding the classifier returns a poisoned destination count -- so
    duplicating that gate's subprocess helper into this file would put two copies
    of one gate in the tree, which is the defect class this branch exists to
    close. Neither half is optional: together they establish that criterion 13 is
    the SOLE objector to this header.

    AND THE THIRD BLOCK HOLDS THE TWO GATES TO ONE READING OF `TYPE_CHECKING`.
    Both call `_is_type_checking_test`, so they agree by construction; the
    block reads each shape's verdict off both gates, so a gate that stops
    calling the shared predicate, or a predicate that narrows or widens, turns
    a row red. Before the predicate was shared, gate (e) and gate (g) held two
    copies of it, and this block was the only thing that kept them in step: a
    narrowing seeded into either copy made the `os.TYPE_CHECKING` row, and no
    other test, go red.

    CRITERION 4 IS DELIBERATELY NOT IN THE CONTRAST. Gate (d) reads
    `runner_image.local_files` and the git index and never reads module source at
    all, so "criterion 4 is green on this plant" would be green for a reason
    unrelated to the plant. A control that cannot see the subject is not evidence
    about it -- that is precisely the class this branch closes, and asserting it
    would dress a vacuous green as a contrast.
    """
    planted = _splice_into_stub("import importlib.util\n"
                                "if importlib.util.find_spec('modal') is not None:\n"
                                "    _ON_CONTAINER = True\n"
                                "else:\n"
                                "    _ON_CONTAINER = False\n")

    tree = ast.parse(planted)
    examined, violations = _module_scope_shape_violations(tree)
    assert examined == len(tree.body), (
        f"gate (g) stopped short of the whole module scope: {examined} of {len(tree.body)}")
    assert violations == [
        ("If", 21, "the only conditional allowed at module scope is `if TYPE_CHECKING:`")
    ], ("criterion 13 missed the `find_spec` header -- the one module-scope construct that is "
        "green on criteria 3, 4 and 5, and which W3b's eight brand-new module headers have "
        f"nothing else stopping them from acquiring. Reported: {violations}")

    # The criterion-5 half of the contrast, on the SAME planted source. Gate (e)
    # takes a repo root, so the plant is materialised rather than parsed.
    target = tmp_path / "scripts" / "modal_runner" / "core.py"
    target.parent.mkdir(parents=True)
    target.write_text(planted, encoding="utf-8")
    scanned, impure = _nonstdlib_module_scope_imports(tmp_path,
                                                      paths=["scripts/modal_runner/core.py"])
    assert scanned == [
        "scripts/modal_runner/core.py"
    ], (f"gate (e) never read the planted copy, so its green below is vacuous: {scanned}")
    assert impure == [], (
        "gate (e) was expected to stay GREEN on this plant -- that is the whole contrast. A red "
        "here means the plant acquired an import that executes, which would be a correct gate "
        f"(e) verdict and a plant-design error on this side (Ruling 30). Reported: {impure}")

    # The two gates must read `TYPE_CHECKING` the same way. Each row plants `if <shape>:` holding a NON-stdlib import, then asks
    # both helpers the same question: gate (e) EXEMPTS the body iff it recognises
    # the block, gate (g) reports no conditional violation iff it recognises the
    # block. The verdicts are read off the helpers, never hardcoded, so the row
    # objects no matter WHICH gate is the one that moved. `recognised` is asserted
    # as well as the agreement, because two gates that both stopped recognising
    # anything would agree vacuously.
    for shape, recognised, why in [
        ("TYPE_CHECKING", True, "the bare `Name` spelling"),
        ("typing.TYPE_CHECKING", True, "the `typing.` attribute spelling"),
        ("os.TYPE_CHECKING", True, "ANY attribute whose `attr` is `TYPE_CHECKING` -- Ruling 29's "
         "accepted limit"),
        ("_probe()", False, "a `Call` test is not recognised by either gate"),
        ("TYPE_CHECKING is True", False, "a `Compare` test is not recognised by either gate"),
    ]:
        slug = "agree_" + shape.replace(".", "_").replace("(", "").replace(")", "").replace(
            " ", "_")
        block = _splice_into_stub(f"if {shape}:\n    import modal\n")
        agree_root = tmp_path / slug / "scripts"
        (agree_root / "modal_runner").mkdir(parents=True)
        (agree_root / "modal_runner" / "core.py").write_text(block, encoding="utf-8")
        agree_scanned, agree_impure = _nonstdlib_module_scope_imports(
            agree_root.parent, paths=["scripts/modal_runner/core.py"])
        assert agree_scanned == [
            "scripts/modal_runner/core.py"
        ], (f"the {slug} plant was not read, so gate (e)'s verdict on it is vacuous: "
            f"{agree_scanned}")
        _, agree_shape = _module_scope_shape_violations(ast.parse(block))
        e_exempts = agree_impure == []
        g_recognises = agree_shape == []
        assert e_exempts == recognised and g_recognises == recognised, (
            f"the two gates no longer read `if {shape}:` the same way, or no longer read it as "
            f"{recognised}. This is {why}. Both must call `_is_type_checking_test`: a block "
            "one gate exempts and the other calls illegal leaves whoever meets the red with no "
            f"way to attribute it. gate (e) exempted: {e_exempts} ({agree_impure}); gate (g) "
            f"recognised: {g_recognises} ({agree_shape})")


def test_gate_g_criterion_13_reddens_on_module_scope_work_the_kind_census_hides():
    """KNOCK-OUT 2. The two shapes a kind-only allow-list is green on.

    ROW 1 -- A BARE MODULE-SCOPE CALL. `logging.basicConfig(level=logging.INFO)`
    adds NO `If` node; measured on the stub, the census goes from
    `Import 2 | Expr 1` to `Import 3 | Expr 2` and every other kind is
    unchanged. A gate that allows the kind `Expr` because
    `body[0]` is one passes this, passes the stub, and passes every other
    check in this plan. Of the four defect shapes this task covers, it is the
    likeliest to appear in a real W3b header -- somebody configures logging, or
    calls `os.environ.setdefault`, at the top of a new module -- and it is the
    one nothing else in the branch can see.

    ROW 2 -- A MODULE-SCOPE `try:`. This is the only observation holding the
    helper's final `elif` -- its declaration tuple -- at all: drop that branch, or
    add `ast.Try` to the tuple, and nothing else in this file objects. `try: import
    X / except ImportError:` is the shape an optional dependency actually gets
    written in, which is why this kind and not `With` / `For` / `While` gets the
    row. The plant's body ASSIGNS rather than imports on purpose -- an `import
    modal` in there would make it gate (e)'s decoy, and then a red could be
    either gate's and the row would certify neither.

    Note both rows land at a DIFFERENT line (21 and 20) because row 1 needs a
    preceding `import logging`. Asserting the line is what makes these knock-outs
    say "this statement", and a plant whose line number was not re-derived is the
    common way that stops being true.
    """
    for text, expected, clause in [
        ("import logging\nlogging.basicConfig(level=logging.INFO)\n", [
            ("Expr", 21, "only the module docstring may be a bare expression at module scope")
        ], "a bare CALL at module scope adds no `If` and one more `Expr`, so a gate that "
         "allows the kind `Expr` because the docstring is one is green on it"),
        ("try:\n    _X = 1\nexcept Exception:\n    _X = 2\n", [
            ("Try", 20, "only imports and declarations may appear at module scope")
        ], "a module-scope `try:` is neither an import nor a declaration, and it is the "
         "only row holding the helper's final `elif` and its declaration tuple"),
    ]:
        tree = ast.parse(_splice_into_stub(text))
        examined, violations = _module_scope_shape_violations(tree)
        assert examined == len(tree.body), (
            f"gate (g) stopped short of the whole module scope: {examined} of {len(tree.body)}")
        assert violations == expected, f"criterion 13 is wrong about: {clause}. Got {violations}"


def test_gate_g_criterion_13_reddens_on_a_type_checking_body_that_is_not_imports_only():
    """KNOCK-OUT 3. A `TYPE_CHECKING` block is exempt from criterion 5, not from
    criterion 13 -- and the kind census cannot tell a legal one from this.

    MEASURED, and it is the sharpest evidence in this task that a kind-level
    instrument is the wrong one:

        if TYPE_CHECKING:            ->  ImportFrom 4 | If 1 | Import 2 | ...
            import decimal

        if TYPE_CHECKING:            ->  ImportFrom 4 | If 1 | Import 2 | ...
            import decimal
            _X = 1

    Byte-identical censuses. One is legal and one is not, and the difference is
    one statement INSIDE the block, which no count of module-scope kinds reaches.
    The knock-out's counterpart -- the legal block going green -- is
    `test_gate_g_criterion_13_allows_exactly_one_type_checking_block_recognised_by_name`;
    without that pair a gate could reject the construct outright and pass this.

    THE VIOLATION NAMES THE INNER STATEMENT (`Assign`, line 23), not the `If` at
    line 21. Whoever meets this red has to be told which line to move out of the
    block; "the block is malformed" does not say that.

    WHY THIS SHAPE IS WORTH A GATE: `if TYPE_CHECKING:` bodies never execute, so
    a non-import in one is dead code that a reader nonetheless takes for a live
    declaration -- and gate (e) will never object, because its whole rule is that
    the block costs nothing at import. Criterion 13 polices the construct;
    criterion 5 declines to look inside it. Complementary, and neither subsumes
    the other (Ruling 33).
    """
    tree = ast.parse(
        _splice_into_stub("from typing import TYPE_CHECKING\n"
                          "if TYPE_CHECKING:\n"
                          "    import decimal\n"
                          "    _X = 1\n"))

    examined, violations = _module_scope_shape_violations(tree)
    assert examined == len(tree.body), (
        f"gate (g) stopped short of the whole module scope: {examined} of {len(tree.body)}")
    assert violations == [
        ("Assign", 23, "an `if TYPE_CHECKING:` body may hold imports only")
    ], ("criterion 13 accepted a `TYPE_CHECKING` block carrying a statement that is not an "
        "import. It must name the OFFENDING BODY STATEMENT -- `Assign` at line 23 -- not the "
        f"`If` at line 21, or the red does not say which line to move. Reported: {violations}")


def test_gate_g_criterion_13_allows_exactly_one_type_checking_block_recognised_by_name():
    """KNOCK-OUT 4 plus the positive control the other three depend on.

    ROW 1 IS THE GREEN HALF THE WHOLE TASK RESTS ON. The stub these plants
    splice into has no module-scope `If`, so on this substrate a
    gate that simply rejects every one of them is observationally identical to
    a correct gate on all three red knock-outs. Row 1 says the construct is
    ALLOWED. On the live package, commands.py and preflight.py each hold one
    such block, so tests/test_modal_runner_package_shape.py's
    `test_package_structure_contract` and
    `test_package_header_accepts_import_only_type_checking` say so as well.

    ROWS 2 AND 3 HOLD THE `at most one` CLAUSE AND THE NAME CLAUSE, which are
    secondary clauses of the same branch -- and "the headline clause has a
    knock-out, the secondary clause has none" is this branch's named defect class
    in its B2 form: the guard's own allow-set going unwatched. Ruling 31 measured
    it on gate (e), where three mutants that kept the node-class test and dropped
    the NAME test survived the entire file. The same mutants live here, and
    knock-out 1's plant cannot see them -- its `If` test is a `Compare`
    (`find_spec(...) is not None`), which is neither a `Name` nor an `Attribute`,
    so `isinstance(test, (ast.Name, ast.Attribute))` passes it. Rows 3 and 4 are
    the bare-`Name` and plain-`Attribute` cases that do object.

    ROW 5 IS THE OTHER HALF OF THE NAME CLAUSE, and it is a GREEN one:
    `if typing.TYPE_CHECKING:` is the same exemption gate (e) grants, matched on
    `test.attr` rather than on the node being an `Attribute`. Both gates must
    agree about which construct this IS -- if criterion 13 recognised a narrower
    set than criterion 5 exempts, the same block would be "exempt" to one and
    "not a `TYPE_CHECKING` block" to the other, and whoever met the red could not
    attribute it. Ruling 29 records the accepted limit that comes with matching
    by name: `if some_module.TYPE_CHECKING:` is recognised too. It fails CLOSED
    in the other direction -- `if TYPE_CHECKING and X:` is a `BoolOp` and
    reddens here, which is row 6.

    The helper's no-else clause has no row here. Its control is
    tests/test_modal_runner_package_shape.py's
    `test_package_ownership_and_header_controls[else]`, which plants into an
    in-memory copy of the live package's sources and runs the combined
    instrument over it; criterion 5
    independently checks the imports in that branch.
    """
    for text, expected, clause in [
        ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import decimal\n", [],
         "a single imports-only `if TYPE_CHECKING:` block is LEGAL -- a gate that rejects "
         "every module-scope `If` passes every other observation in this task and "
         "false-reddens on the headers W3b is about to write"),
        ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import decimal\n"
         "if TYPE_CHECKING:\n    import fractions\n", [
             ("If", 23, "a module may hold at most one `if TYPE_CHECKING:` block")
         ], "at most ONE such block per module; the second is what this row holds, and "
         "nothing else in the file has two"),
        ("_DEBUG = False\nif _DEBUG:\n    import decimal\n", [
            ("If", 21, "the only conditional allowed at module scope is `if TYPE_CHECKING:`")
        ], "the recognition matches the NAME, not merely the node class -- drop "
         "`test.id == \"TYPE_CHECKING\"` and every flag-guarded module-scope block becomes "
         "legal shape"),
        ("import os\nif os.name:\n    import decimal\n", [
            ("If", 21, "the only conditional allowed at module scope is `if TYPE_CHECKING:`")
        ], "same clause on the attribute spelling: the rule reads `test.attr`, not "
         "`isinstance(test, ast.Attribute)`"),
        ("import typing\nif typing.TYPE_CHECKING:\n    import decimal\n", [],
         "`typing.TYPE_CHECKING` is the same construct and the same exemption gate (e) "
         "grants; recognising only the bare `Name` leaves half the real idiom reddening"),
        ("from typing import TYPE_CHECKING\nif TYPE_CHECKING and os.name:\n"
         "    import decimal\n", [
             ("If", 21, "the only conditional allowed at module scope is `if TYPE_CHECKING:`")
         ], "`if TYPE_CHECKING and X:` is a `BoolOp`, not a recognised `TYPE_CHECKING` test. "
         "The name-based rule fails CLOSED in this direction, which is the direction it "
         "should fail in"),
    ]:
        tree = ast.parse(_splice_into_stub(text))
        examined, violations = _module_scope_shape_violations(tree)
        assert examined == len(tree.body), (
            f"gate (g) stopped short of the whole module scope: {examined} of {len(tree.body)}")
        assert violations == expected, f"criterion 13 is wrong about: {clause}. Got {violations}"
