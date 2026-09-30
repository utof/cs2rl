"""Behavior tests for scripts.modal_runner.checkpoint: resume and completed-run checks.

One of the per-module runner test files (RUNNER_TEST_FILES in
tests/modal/modal_runner_tables.py). Before you add, move or delete a test here, or
add a helper, read THE PLACEMENT RULE FOR RUNNER TESTS in
tests/modal/test_modal_packaging.py: which file a test belongs in, what the change
costs in the seam manifest, and where helpers go.
"""
import ast
import inspect
import json
import sys
import textwrap
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT

ROOT = REPO_ROOT

import scripts.modal_runner as mrl                                     # noqa: E402, I001
from scripts.modal_runner import checkpoint, core, request             # noqa: E402, I001
from tests.modal.modal_test_helpers import (                           # noqa: E402
    _live_batch_size, _minimal_completed_tree, _no_torch, _write_metrics)

# ── Resume input: local checkpoint hash + weights-only load ────────────────


def test_validate_local_checkpoint_hashes_and_maps_paths(tmp_path):
    import torch

    ckpt = tmp_path / "warm.pt"
    torch.save({"weight": torch.tensor([1.0, 2.0])}, ckpt)
    provenance = mrl.validate_local_checkpoint(ckpt)
    digest = core.sha256_file(ckpt)
    assert provenance.sha256 == digest
    assert provenance.size == ckpt.stat().st_size
    assert provenance.client_path == mrl.INPUTS_ROOT / "sha256" / f"{digest}.pt"
    assert provenance.mount_path == Path("/artifacts/inputs/sha256") / f"{digest}.pt"


def test_validate_local_checkpoint_rejects_non_checkpoint(tmp_path):
    junk = tmp_path / "nope.txt"
    junk.write_text("not a checkpoint\n")
    with pytest.raises(mrl.ValidationError):
        mrl.validate_local_checkpoint(junk)
    missing = tmp_path / "missing.pt"
    with pytest.raises(mrl.ValidationError):
        mrl.validate_local_checkpoint(missing)


# ── Completion evidence: config normalisation, completed-run checks ────────


def test_normalize_config_strips_only_checkpoint_data_dir():
    raw = {"env": "cs2-dust2", "data_dir": "/artifacts/runs/x/checkpoints", "seed": 2}
    normalized = checkpoint.normalize_config_for_transport(raw)
    assert "data_dir" not in normalized
    assert normalized == {"env": "cs2-dust2", "seed": 2}
    assert checkpoint.normalize_config_for_transport(normalized) == normalized


def test_validate_completed_run_accepts_representative_metrics(tmp_path):
    # compute_batch_dims moved train.py -> train_config.py in the post-rung1a
    # refactor (2026-08-31); the runner's AGENTS_PER_ENV/BPTT_HORIZON mirror is
    # pinned against wherever it actually lives, not against train.py by habit.
    train_src = (ROOT / "src" / "cs2rl" / "train" / "config.py").read_text()
    fn = ast.parse(train_src)
    for node in ast.walk(fn):
        if isinstance(node, ast.FunctionDef) and node.name == "compute_batch_dims":
            body = ast.get_source_segment(train_src, node)
            assert body is not None
            assert "agents_per_env = 10" in body
            assert "bptt_horizon = 64" in body
            assert "num_envs * agents_per_env * bptt_horizon" in body
            break
    else:
        raise AssertionError("live compute_batch_dims not found in src/cs2rl/train/config.py")
    assert request.AGENTS_PER_ENV == 10
    assert request.BPTT_HORIZON == 64
    run_root, manifest, effective, ckpt = _minimal_completed_tree(tmp_path)
    evidence = checkpoint.validate_completed_run(run_root, manifest)
    assert evidence.last_step == effective
    assert evidence.checkpoint_sha256 == core.sha256_file(ckpt)
    assert evidence.config_hash == manifest.config_hash
    assert manifest.batch_size == _live_batch_size(256)
    assert manifest.effective_timesteps == (30_000_000 // manifest.batch_size) * manifest.batch_size
    assert manifest.effective_timesteps >= manifest.batch_size
    assert json.loads((run_root / "checkpoints" / "config.json").read_text())["env"] == "cs2-dust2"
    assert manifest.effective_map == "simple"


@pytest.mark.parametrize(
    "defect", ["bad_ckpt", "empty", "malformed", "nonmonotonic", "wrong_hash", "short_step"])
def test_validate_completed_run_rejects_bad_evidence(tmp_path, defect):
    run_root, manifest, effective, ckpt = _minimal_completed_tree(tmp_path)
    if defect == "bad_ckpt":
        ckpt.write_text("nope")
    elif defect == "empty":
        (run_root / "checkpoints" / "metrics.jsonl").write_text("")
    elif defect == "malformed":
        (run_root / "checkpoints" / "metrics.jsonl").write_text("{nope\n")
    elif defect == "nonmonotonic":
        _write_metrics(run_root / "checkpoints" / "metrics.jsonl", [100, 50])
    elif defect == "wrong_hash":
        manifest = mrl.Manifest(**{**manifest.to_dict(), "config_hash": "e" * 64})
    elif defect == "short_step":
        _write_metrics(run_root / "checkpoints" / "metrics.jsonl", [effective - 1])
    with pytest.raises(mrl.ValidationError):
        checkpoint.validate_completed_run(run_root, manifest)


# ── Runner-interpreter checkpoint validation: completed-run check ──────────
#
# The Modal runner process and the training child are DIFFERENT interpreters.
# The runner is the image's standalone /usr/local/bin/python (only uv + modal);
# torch lives exclusively in the PREBUILT_PYTHON venv that runs train.py.
# Verified in a live container on 2026-08-14:
#   runner_executable=/usr/local/bin/python  runner_torch=MISSING
# Every test above runs on a laptop where `import torch` succeeds, so none of
# them can see this. These do: they force the torch-less runner condition.


def test_completed_run_validates_without_runner_torch(tmp_path, monkeypatch):
    """validate_completed_run torch-loads too: without the fallback every clean
    exit is misfiled as failed/invalid_evidence and no run can ever complete."""
    run_root, manifest, effective, ckpt = _minimal_completed_tree(tmp_path)
    _no_torch(monkeypatch, prebuilt=sys.executable)

    evidence = checkpoint.validate_completed_run(run_root, manifest)

    assert evidence.last_step == effective
    assert evidence.checkpoint_sha256 == core.sha256_file(ckpt)


# ── Reason tokens: the protocol's vocabulary has one source ────────────────


def _verify_checkpoint_census(source: str) -> tuple[list[str], list[str]]:
    """Read `verify_checkpoint`'s source; return (its `fail` literals, rule violations).

    What: the AST side of the gh#197 census, split out of the test so the
    positive control below can run the same rules over a mutant source text.
    A token reaches a caller only as the `reason` of a failure verdict, so the
    census has to see every way this function can build one, not just the
    `fail("...")` spelling. The rules, each naming what it closes:
      * every `fail(...)` has exactly one string-literal argument (a variable
        hides the token from the census);
      * (R1) every `fail` or `CheckpointVerdict` read is the callee of a call
        (type annotations excepted): `f = fail; f("x")` would otherwise be
        invisible;
      * (R2) every `CheckpointVerdict(...)` outside `fail`'s own body carries
        a literal `ok=True` keyword: a direct `CheckpointVerdict(ok=False,
        reason="x", ...)` would otherwise ship a token past the census, the
        launch-map totality test and the protocol table at once.
    PITFALL: a violation is returned, not asserted, so the caller decides; the
    real-source test asserts none, the positive control asserts the named one.
    Anything not built from `CheckpointVerdict` or `fail` (say
    `dataclasses.replace`) is still outside these rules.
    """
    tree = ast.parse(textwrap.dedent(source))
    fail_defs = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "fail"
    ]
    inside_fail = {id(inner) for fdef in fail_defs for inner in ast.walk(fdef)}
    callees = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    # `-> CheckpointVerdict` and `x: CheckpointVerdict` name the type without
    # building one; R1 must not read an annotation as an alias.
    annotations = [
        node.returns
        for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.returns is not None
    ] + [
        node.annotation for node in ast.walk(tree)
        if isinstance(node, (ast.arg, ast.AnnAssign)) and node.annotation is not None
    ]
    in_annotation = {id(inner) for ann in annotations for inner in ast.walk(ann)}
    literals: list[str] = []
    problems: list[str] = []
    if len(fail_defs) != 1:
        problems.append(f"expected one nested `def fail`, found {len(fail_defs)}")
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                and node.id in ("fail", "CheckpointVerdict") and id(node) not in callees
                and id(node) not in in_annotation):
            problems.append(f"R1 {node.id} read without being called (alias?) line {node.lineno}")
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(
            func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else None)
        if name == "fail":
            if not (len(node.args) == 1 and not node.keywords):
                problems.append(f"fail() call is not fail(<literal>): {ast.dump(node)}")
                continue
            (arg, ) = node.args
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                problems.append(
                    f"fail() argument is not a string literal, the census cannot see it: "
                    f"{ast.dump(arg)}")
                continue
            literals.append(arg.value)
        elif name == "CheckpointVerdict" and id(node) not in inside_fail:
            ok = [kw.value for kw in node.keywords if kw.arg == "ok"]
            if not (len(ok) == 1 and isinstance(ok[0], ast.Constant) and ok[0].value is True
                    and not node.args):
                problems.append(
                    f"R2 CheckpointVerdict built outside fail() without a literal ok=True, "
                    f"line {node.lineno}; route every failure through fail(\"<token>\")")
    return literals, problems


def test_verify_checkpoint_fail_literals_are_exactly_the_reason_tokens():
    """Every failure `verify_checkpoint` can return names a token of `CHECKPOINT_REASON_TOKENS`,
    and every token is used.

    gh#197. The tuple is the single source that the launch error map and the
    protocol parametrizes derive from. Nothing at run time ties the function
    to it (`fail` is a closure over a literal), so this census reads the
    function's AST: an eighth token added to the function alone, or a token
    deleted from the tuple alone, makes the two sets differ. The rules in
    `_verify_checkpoint_census` also refuse the routes that would bypass the
    literal set (an aliased `fail`, a direct `CheckpointVerdict(ok=False,
    ...)`); `test_verify_checkpoint_census_rejects_bypass_mutants` proves they
    fire, so this test's green is not the census being blind.
    """
    literals, problems = _verify_checkpoint_census(inspect.getsource(checkpoint.verify_checkpoint))
    assert not problems, problems
    assert literals, "no fail(...) call found; the census is reading the wrong function"
    tokens = checkpoint.CHECKPOINT_REASON_TOKENS
    assert len(set(tokens)) == len(tokens), f"duplicate token in the tuple: {tokens}"
    assert set(literals) == set(tokens), (
        f"verify_checkpoint fails with {sorted(set(literals) - set(tokens))} not in "
        f"CHECKPOINT_REASON_TOKENS; the tuple lists {sorted(set(tokens) - set(literals))} the "
        "function never returns. Change both together, then add the launch sentence in "
        "scripts/run_modal.py (_LAUNCH_CHECKPOINT_ERRORS) and the cases in tests.")


@pytest.mark.parametrize(
    ("inserted", "rule"),
    [
        pytest.param(
            '    if sidecar_bytes == b"eighth":\n'
            '        return CheckpointVerdict(ok=False, reason="eighth_token",\n'
            '                                 checkpoint_bytes=None, digest=None)\n',
            "R2",
            id="direct_ok_false_verdict",
        ),
        pytest.param(
            '    f = fail\n'
            '    if sidecar_bytes == b"eighth":\n'
            '        return f("eighth_token")\n',
            "R1",
            id="aliased_fail",
        ),
        pytest.param(
            '    tok = "eighth_token"\n'
            '    if sidecar_bytes == b"eighth":\n'
            '        return fail(tok)\n',
            "not a string literal",
            id="non_literal_fail_argument",
        ),
    ],
)
def test_verify_checkpoint_census_rejects_bypass_mutants(inserted, rule):
    """Positive control for the census: each bypass route, spliced into the real source, is named.

    The guard's own scope is what goes unwatched (the review of #245 found the
    first census green on both an aliased `fail` and a direct
    `CheckpointVerdict(ok=False, ...)`). Each case splices one mutation into
    `verify_checkpoint`'s real source just before its first check and asserts
    `_verify_checkpoint_census` reports the rule meant to catch it, and that
    the mutant's token did not leak into the literal set unnoticed.
    PITFALL: the splice anchors on the first check's text; if that line is
    reworded the anchor assert below fails first, and the fix is a new anchor,
    not deleting the case.
    """
    source = textwrap.dedent(inspect.getsource(checkpoint.verify_checkpoint))
    anchor = "    if sidecar_bytes is None:\n"
    assert source.count(anchor) == 1, "splice anchor moved; pick the first check's new text"
    _, clean = _verify_checkpoint_census(source)
    assert not clean, clean
    mutant = source.replace(anchor, inserted + anchor)
    _, problems = _verify_checkpoint_census(mutant)
    assert any(
        rule in problem
        for problem in problems), (f"census missed the {rule} mutant; problems were {problems}")
