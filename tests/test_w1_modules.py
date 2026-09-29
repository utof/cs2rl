"""Structural guards for the modules split out of train.py (W1, spec 2026-08-31).

WHAT: every module carved out of src/cs2rl/train.py gets four properties checked in a
FRESH interpreter, one subprocess per case:

  1. it imports at all, standalone (no "works only because train.py imported it
     first" ordering luck);
  2. it does not pull `cs2rl.train` back in — the dependency graph stays acyclic, so
     the leaf really is a leaf;
  3. its module scope stays free of torch / nav / c_env.cs2_env / rerun;
  4. the only sibling edge any of them has is into a LEAF, and the leaves
     import no sibling except each other in the one allowed direction
     (train_shared -> env_config). The shape spec §2 W1 fixes: leaves at the
     bottom, everything else a spoke off them, never spoke-to-spoke.

WHY a subprocess and not a plain import: pytest's session has already imported
half the repo by the time any test body runs, so `"torch" not in sys.modules`
in-process measures the session, not the module. Every assert here has to start
from an empty sys.modules or it silently passes forever.

WHY property 3 is load-bearing (measured, not stylistic): `from cs2rl import train`
today pulls neither torch nor nav nor c_env, because all ~35 torch imports in train.py
are function-local ON PURPOSE. That is what makes `train.py --dump-config`
cost ~1 s instead of ~30 s, which in turn is what makes it usable as the
Modal/run_rung1 fingerprint step (tests/test_train_cli.py's "--dump-config means
zero side-effects"). train.py imports each of these modules at ITS module level,
so a single module-scope `import torch` added to any of them silently destroys
that guarantee for every caller — and nothing else in the suite would notice.

EXTENDING THIS FILE: the split proceeds in several tasks. Add each new module's
name to W1_MODULES as it lands — the spec makes adding it part of the SAME task
that creates the module, precisely so a module cannot slip in unguarded.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO_ROOT / "src" / "cs2rl" / "train.py"

# Modules split out of train.py (W1, spec 2026-08-31), plus env_config, which
# owns the env contract that used to live partly in train_shared (spec
# 2026-09-03 §2.1). Every module here must import standalone and stay light.
#
# Every name in this file is the FULL dotted name (`cs2rl.X`), because each one
# is compared against sys.modules: a bare `X` is never a key there, so a bare
# entry would make every "not in sys.modules" check below pass forever.
#
# `env_factory` (W3) is here for a reason beyond bookkeeping: it is the module
# whose module scope is MOST tempting to make heavy, since its whole job is
# constructing envs. Two function-local imports carry the two reasons: `from
# cs2rl.c_env.cs2_env import make_env` in `build_env_for` stays function-local so
# `from cs2rl import train` stays free of torch/nav/c_env, and `from cs2rl.train import
# SelfPlayManager` in `build_selfplay_manager` stays function-local to break
# the cycle (train.py imports env_factory at module level).
#
# `metrics_schema` (W4) is here for the mirror-image reason: it is a registry of
# STRINGS whose whole value is being cheap to import, and it took ownership of
# `EVAL_KEYS` from `eval_baselines` — a module with torch and c_env.cs2_env at
# its scope. If that ownership ever flipped back, or someone imported a policy
# class to spell a type hint, eight string constants would start costing a torch
# import, and only this test would say so.
#
# `trainer` (gh#168 W1) is deliberately NOT here: it subclasses PuffeRL, so it
# imports pufferlib (and through it torch) at module scope and is heavy by
# construction. It cannot pass property 3, and train.py / train_test_harness.py
# import it function-locally for exactly that reason (knock-out W1-K3: a
# module-level `from cs2rl.trainer import Cs2PuffeRL` in train.py turns
# test_import_train_stays_light_and_really_imports_the_shims red naming torch).
W1_MODULES = ("cs2rl.train_shared", "cs2rl.resume_state", "cs2rl.train_config",
              "cs2rl.train_metrics", "cs2rl.train_update", "cs2rl.env_factory",
              "cs2rl.eval.metrics_schema", "cs2rl.env.config")

# W1 modules train.py deliberately does NOT import at its module level, and why.
#
# Every module CARVED OUT of train.py must be a module-level import there, because
# train.py's own body still reads names that moved into it (see
# test_import_train_stays_light_and_really_imports_the_shims). This is the list of
# the exceptions — modules the guard covers that train.py has no reason to name.
#
# WHY a carve-out list and not a hand-written list of the modules that ARE
# imported: a hand list only has to shrink for the "the shims are real imports,
# not lazy function-local ones" check to quietly stop covering a module, which is
# the exact regression this test exists to catch. The imported set is DERIVED from
# train.py's own AST below and asserted equal to `W1_MODULES - NOT_IMPORTED_BY_TRAIN`,
# so both directions bite: a shim that goes lazy fails, and a carve-out that starts
# being imported fails too. Adding a name here is then a deliberate, reasoned edit
# rather than a deletion nobody notices.
#
# An entry can be a prohibition, not only a description. metrics_schema sits in the
# layer above train in pyproject.toml's `cs2rl layers` contract, so train.py importing
# it, at any scope, is an upward edge lint-imports rejects (tests/test_import_layers.py).
# That entry leaves only if the layering changes, never because an import appeared;
# tests/test_import_layers.py::test_metrics_schema_sits_above_train pins the layering.
NOT_IMPORTED_BY_TRAIN = {
    "cs2rl.eval.metrics_schema":
    "took EVAL_KEYS from eval.baselines, not from train.py, so train.py's body holds no "
    "reference to it; and it is in the layer above train (pyproject.toml's `cs2rl layers` "
    "contract), so a train.py import of it is an upward edge lint-imports rejects",
}


def _train_module_level_imports():
    """W1 modules `src/cs2rl/train.py` imports at its MODULE level, parsed from source.

    Only `tree.body` is scanned — a top-level `import`/`from ... import`, never one
    nested in a function, a `try:` or the `if __name__ == "__main__":` block. That
    is the whole point: the property under test is that the shims are imported
    unconditionally when `train` is imported, so a conditionally-imported module
    correctly reads as absent here rather than as a satisfied shim.

    Both spellings count: `from cs2rl.X import ...` names `cs2rl.X` as its module,
    and `from cs2rl import X` names `cs2rl`, with X among the imported names.
    """
    import ast
    tree = ast.parse(TRAIN_PY.read_text())
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    return tuple(m for m in W1_MODULES if m in imported)


TRAIN_MODULE_LEVEL_IMPORTS = _train_module_level_imports()

# TWO leaves. train_shared owns the names moved out of train.py; env_config owns
# the env contract. train_config -> env_config is the load-bearing edge between
# the leaves and the spokes (env_config_from_args builds an EnvConfig); train.py
# imports both. The reverse edge would make "leaf" meaningless — pyproject.toml's
# `cs2rl layers` contract pins it (env.config is in the env layer, train_shared one
# above; tests/test_import_layers.py runs it), and so does
# tests/test_env_config.py::test_module_is_stdlib_only.
LEAVES = frozenset({"cs2rl.train_shared", "cs2rl.env.config"})

# Imports whose presence in sys.modules means the import-lightness invariant is
# gone. `cs2rl.c_env.cs2_env` rather than `cs2rl.c_env` on purpose: the package
# itself is cheap, the ctypes/binding module underneath it is not.
#
# `rerun` stands in for `cs2rl.viz`, which imports it at module scope and which train.py
# reaches only inside record_episode (`--record`). The layers contract ignores the
# train -> viz pair and the scope pin in tests/test_import_layers.py keeps its import
# statements inside a def, but neither sees a function-local import that is CALLED at
# module scope; test_import_train_stays_light_and_really_imports_the_shims, through
# this entry, does.
HEAVY = ("torch", "cs2rl.env.nav", "cs2rl.c_env.cs2_env", "rerun")


def _run_child(body: str) -> subprocess.CompletedProcess:
    """Run `body` in a fresh interpreter that inherits this session's environment.

    Uses sys.executable (the venv python pytest itself runs under), so the child
    sees the same installed packages without paying for a `uv run` resolve. It
    inherits PYTHONPATH too, so `cs2rl` resolves to the same checkout as here,
    which tests/conftest.py's tripwire has already required to be this one.
    """
    code = f"import sys\n{body}"
    return subprocess.run([sys.executable, "-c", code],
                          cwd=REPO_ROOT,
                          capture_output=True,
                          text=True,
                          timeout=120)


@pytest.mark.parametrize("mod", W1_MODULES)
def test_new_module_imports_standalone_and_stays_light(mod):
    """Each split-out module imports alone, pulls no `cs2rl.train`, pulls nothing heavy."""
    r = _run_child(f"""
import importlib
importlib.import_module({mod!r})
heavy = [m for m in {HEAVY!r} if m in sys.modules]
assert not heavy, f"{mod} module scope imported {{heavy}} — import-lightness invariant broken"
assert "cs2rl.train" not in sys.modules, (
    "{mod} imported `cs2rl.train` — that is an import CYCLE: train.py imports {mod} at its "
    "module level, so this only appears to work while some other module got there first")
""")
    assert r.returncode == 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"


@pytest.mark.parametrize("mod", W1_MODULES)
def test_only_sibling_edge_is_to_the_leaf(mod):
    """No spoke-to-spoke import: every sibling edge points at train_shared.

    Property 2 (no cycle back to `train`) does not cover this. A NON-cyclic,
    import-light sibling edge — say train_update importing train_metrics for one
    helper — passes every other check in this file while quietly recreating the
    tangle the split exists to remove: the next extraction then has to move two
    modules to move one, and `from cs2rl import train_config` starts paying for
    resume_state. Asserting the shape here makes that a test failure at the
    moment it is written rather than an architecture review two branches later.
    """
    r = _run_child(f"""
import importlib
importlib.import_module({mod!r})
allowed = {{{mod!r}}} | {set(LEAVES)!r}
siblings = [m for m in {W1_MODULES!r} if m in sys.modules and m not in allowed]
assert not siblings, (
    f"{mod} imported {{siblings}} at module scope; the only sibling edge any "
    "split-out module may have is -> one of {sorted(LEAVES)} (spec 2026-08-31 §2 W1)")
""")
    assert r.returncode == 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"


def test_import_train_stays_light_and_really_imports_the_shims():
    """`from cs2rl import train` still pulls nothing heavy — and the shims are real imports.

    The second half matters: if the split-out modules were imported lazily
    (inside functions) instead of at train.py's module level, the ten
    module-level reads of moved names in train.py's own body — the argparse
    defaults under `if __name__ == "__main__"` among them — would NameError at
    run time while the whole test suite stayed green, because the suite never
    executes that block.
    """
    expected = set(W1_MODULES) - set(NOT_IMPORTED_BY_TRAIN)
    assert set(TRAIN_MODULE_LEVEL_IMPORTS) == expected, (
        f"train.py's module-level W1 imports are {sorted(TRAIN_MODULE_LEVEL_IMPORTS)}, expected "
        f"{sorted(expected)}. A module that dropped out went LAZY (function-local) — the ten "
        "module-level reads of moved names in train.py's body would then NameError at run time "
        "while this suite stayed green. A module that appeared is listed in "
        "NOT_IMPORTED_BY_TRAIN, whose entry says why train.py must not import it (for "
        "metrics_schema: an upward edge pyproject.toml's `cs2rl layers` contract rejects); "
        "remove the import, not the entry.")
    r = _run_child(f"""
from cs2rl import train
heavy = [m for m in {HEAVY!r} if m in sys.modules]
assert not heavy, f"`from cs2rl import train` pulled {{heavy}} — --dump-config is no longer cheap"
missing = [m for m in {TRAIN_MODULE_LEVEL_IMPORTS!r} if m not in sys.modules]
assert not missing, f"train.py does not import {{missing}} at module level"
""")
    assert r.returncode == 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"


def test_import_train_test_harness_stays_light():
    """`from cs2rl import train_test_harness` pulls nothing heavy either.

    The harness is not a W1 module (it is test-only and imports train function-locally),
    so the probes above never import it. Its lightness is what lets a test import it
    without paying for torch, and it rests on `_build_trainer_for_test` importing
    `cs2rl.trainer` function-locally: trainer subclasses PuffeRL and imports torch at
    module scope, so one module-scope trainer import here loads torch, cs2rl.train and
    cs2rl.trainer, and neither import-linter contract nor the scope pin objects to that
    same-layer (L2) edge, which forms no cycle. This probe is the check that does.
    """
    r = _run_child(f"""
from cs2rl import train_test_harness
heavy = [m for m in {HEAVY!r} if m in sys.modules]
assert not heavy, f"`from cs2rl import train_test_harness` pulled {{heavy}}"
""")
    assert r.returncode == 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"


def _main_block_body():
    """Top-level statements of train.py's `if __name__ == "__main__":` block.

    Parsed from source, never executed: the block builds the argparse parser and
    then trains, so importing it is not an option.
    """
    import ast
    tree = ast.parse(TRAIN_PY.read_text())
    for node in tree.body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"):
            return node.body
    raise AssertionError('no `if __name__ == "__main__":` block in src/cs2rl/train.py')


def test_train_aliases_itself_into_sys_modules_first():
    """The self-alias is the FIRST statement of train.py's __main__ block.

    WHAT is pinned: `sys.modules.setdefault("cs2rl.train", sys.modules["__main__"])`,
    and its POSITION. Position is half the contract — anything above it that
    triggers a `from cs2rl.train import ...` (directly or through a helper) executes
    train.py's body a second time before the alias can prevent it, and the alias
    then quietly protects nothing.

    WHY the whole thing exists: `python -m cs2rl.train` binds this file to "__main__",
    so a runtime `from cs2rl.train import ...` — which eval_baselines does function-locally
    inside PolicyActor — imports a SECOND copy of the module. Two copies means
    two sets of module constants and cross-copy `isinstance` returning False.

    This test is a source pin and proves only that the statement is written.
    That the statement WORKS is test_script_run_has_exactly_one_train_module
    below; the two are a pair and neither is sufficient alone.
    """
    import ast
    first = _main_block_body()[0]
    src = ast.unparse(first)
    assert src == 'sys.modules.setdefault(\'cs2rl.train\', sys.modules[\'__main__\'])', (
        f"first statement of train.py's __main__ block is {src!r}, not the sys.modules "
        "self-alias — see the comment block at that line for why order matters")


def test_script_run_has_exactly_one_train_module():
    """BEHAVIOURAL half: run `python -m cs2rl.train` and prove the alias works.

    WHY a separate test from the AST pin: the pin is satisfied by the statement
    merely existing. This one runs the real file in a real child interpreter and
    checks the two things the alias is FOR, and it is the only check in the suite
    that can see them — the §3 determinism gate runs `--eval-interval 0
    --no-self-play`, exactly the flag set on which no runtime `from cs2rl.train import`
    occurs, so the gate is structurally blind here.

    Channel (both observations are about the CHILD's interpreter state, so they
    have to be made inside it):

      1. a sitecustomize.py on the child's PYTHONPATH registers an atexit hook
         that performs the same `from cs2rl.train import ...` eval_baselines performs
         and reports whether sys.modules["cs2rl.train"] IS sys.modules["__main__"]. atexit runs
         before module teardown, so sys.modules is intact; it also runs after
         `--dump-config`'s sys.exit(0), which is why that cheap path suffices
         instead of a full training run.
      2. `-X importtime` prints one line per module body EXECUTED. With the
         alias the hook's import is a sys.modules hit and prints nothing;
         without it, the body runs again and stderr carries a
         `... | cs2rl.train` line (importtime prints the qualified name). Absence is the proof — so the test also asserts
         importtime produced output at all, or "no train line" would pass
         vacuously the day the flag stops working.

    PITFALL: do NOT rewrite this as `runpy.run_path(..., run_name="__main__")`
    plus a post-hoc identity assert. run_path restores the real
    sys.modules["__main__"] when it returns, so the assert compares the alias
    against the restored module and fails for a reason that has nothing to do
    with train.py (measured: `is` -> False after the call, True inside it).
    """
    import os
    import re
    import tempfile

    # The hook's import is the exact one eval_baselines makes.
    probe_src = ("import atexit, sys\n"
                 "def _probe():\n"
                 "    from cs2rl.train import load_policy_from_checkpoint\n"
                 "    train = sys.modules['cs2rl.train']\n"
                 "    print('ALIASPROBE identity=%s' % (train is sys.modules['__main__']),\n"
                 "          file=sys.stderr, flush=True)\n"
                 "atexit.register(_probe)\n")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "sitecustomize.py").write_text(probe_src)
        argv = [
            sys.executable, "-X", "importtime", "-m", "cs2rl.train", "--dump-config",
            "--checkpoint-dir",
            str(td)
        ]
        # PREPEND the probe dir: replacing PYTHONPATH would drop whatever put this
        # checkout's src/ first, and the child would run another checkout's cs2rl.
        pythonpath = os.pathsep.join(filter(None, [str(td), os.environ.get("PYTHONPATH")]))
        r = subprocess.run(argv,
                           cwd=REPO_ROOT,
                           env=dict(os.environ, PYTHONPATH=pythonpath),
                           capture_output=True,
                           text=True,
                           timeout=300)

    assert r.returncode == 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"

    # Anti-vacuity: the probe must have run, and importtime must have produced
    # output. Either one silently missing turns both assertions below into
    # unfalsifiable green.
    importtime_lines = re.findall(r"^import time:", r.stderr, re.M)
    assert "ALIASPROBE" in r.stderr, (
        "the atexit probe never printed — the observation channel is broken, so this "
        f"test proves nothing. STDERR:\n{r.stderr[-2000:]}")
    assert importtime_lines, ("-X importtime produced no output; the 'no train import' "
                              "assertion below would pass vacuously")

    assert "ALIASPROBE identity=True" in r.stderr, (
        "`from cs2rl.train import` in the child did NOT resolve to sys.modules['__main__'] — "
        "the run is carrying two copies of train.py")
    reimports = re.findall(r"^import time:.*\|\s*cs2rl\.train$", r.stderr, re.M)
    assert not reimports, (
        f"train.py's module body executed a second time under the name 'cs2rl.train': "
        f"{reimports} — "
        "the self-alias is missing or is no longer the first statement of the __main__ block")


def test_mask_head_slices_is_complete_in_a_leaf_only_interpreter():
    """`train_shared._MASK_HEAD_SLICES` holds one slice per action head.

    WHY this needs its own subprocess, and why that subprocess must NEVER have
    imported `cs2rl.train`: _MASK_HEAD_SLICES is not a single assignment but a
    three-statement construct (empty list, a `for` loop appending slices, a
    `del`). If the loop were left behind in train.py while the list moved to the
    leaf, `from cs2rl import train_shared` would still succeed and hand out an EMPTY
    list, while `from cs2rl import train` would run the leftover loop and fill THE SAME list
    object — so any interpreter that has imported `cs2rl.train` sees a correct length
    and this assert becomes vacuous. Importing train_shared alone is the only
    arrangement that can observe the half-move.

    Consequence if it ever regresses: `_apply_action_masks` zips with
    strict=True, so a short list raises at the first forward pass rather than
    silently unmasking heads. The blast radius is loud; the GUARD is what would
    have been silent, which is the point of testing it here.
    """
    r = _run_child("""
assert "cs2rl.train" not in sys.modules
from cs2rl import train_shared
from cs2rl.spec.action import ACTION_HEAD_SIZES
assert "cs2rl.train" not in sys.modules, "this check is vacuous once `cs2rl.train` is imported"
assert len(train_shared._MASK_HEAD_SLICES) == len(ACTION_HEAD_SIZES), (
    f"_MASK_HEAD_SLICES has {len(train_shared._MASK_HEAD_SLICES)} entries for "
    f"{len(ACTION_HEAD_SIZES)} action heads — the construct at the top of "
    "train_shared.py was moved only partially")
# The slices must also tile [0, sum(sizes)) contiguously, which is what makes
# them a valid decomposition of the flat action-mask row rather than merely a
# list of the right length.
assert train_shared._MASK_HEAD_SLICES[0][0] == 0
for (lo, hi), size in zip(train_shared._MASK_HEAD_SLICES, ACTION_HEAD_SIZES, strict=True):
    assert hi - lo == size, (lo, hi, size)
for (_, hi), (lo, _) in zip(train_shared._MASK_HEAD_SLICES,
                            train_shared._MASK_HEAD_SLICES[1:], strict=False):
    assert hi == lo, "mask head slices are not contiguous"
""")
    assert r.returncode == 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
