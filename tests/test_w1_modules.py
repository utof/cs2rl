"""Structural guards for the modules split out of train.py (W1, spec 2026-08-31).

WHAT: every module carved out of src/train.py gets four properties checked in a
FRESH interpreter, one subprocess per case:

  1. it imports at all, standalone (no "works only because train.py imported it
     first" ordering luck);
  2. it does not pull `train` back in — the dependency graph stays acyclic, so
     the leaf really is a leaf;
  3. its module scope stays free of torch / nav / c_env.cs2_env;
  4. the only sibling edge any of them has is `-> train_shared`, and
     train_shared itself has none (the shape spec §2 W1 fixes: one leaf,
     everything else a spoke off it, never spoke-to-spoke).

WHY a subprocess and not a plain import: pytest's session has already imported
half the repo by the time any test body runs, so `"torch" not in sys.modules`
in-process measures the session, not the module. Every assert here has to start
from an empty sys.modules or it silently passes forever.

WHY property 3 is load-bearing (measured, not stylistic): `import train` today
pulls neither torch nor nav nor c_env, because all ~35 torch imports in train.py
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
SRC = REPO_ROOT / "src"

# Modules split out of train.py. Grows per task (spec §2 W1 / W3 / W4).
#
# `env_factory` (W3) is here for a reason beyond bookkeeping: it is the module
# whose module scope is MOST tempting to make heavy, since its whole job is
# constructing envs. Its `from train import make_puffer_env` has to stay
# function-local both to break the cycle (train.py imports it at module level)
# and to keep `import train` free of torch/nav/c_env.
W1_MODULES = ("train_shared", "resume_state", "train_config", "train_metrics", "train_update",
              "env_factory")

# The one module every other split-out module is allowed to depend on. Spec §2 W1:
# "The leaf imports nothing from train.py or the other new modules; every other new
# module may import train_shared."
LEAF = "train_shared"

# Imports whose presence in sys.modules means the import-lightness invariant is
# gone. `c_env.cs2_env` rather than bare `c_env` on purpose: the package itself
# is cheap, the ctypes/binding module underneath it is not.
HEAVY = ("torch", "nav", "c_env.cs2_env")


def _run_child(body: str) -> subprocess.CompletedProcess:
    """Run `body` in a fresh interpreter with src/ on sys.path.

    Uses sys.executable (the venv python pytest itself runs under), so the child
    sees the same installed packages without paying for a `uv run` resolve.
    """
    code = f"import sys\nsys.path.insert(0, {str(SRC)!r})\n{body}"
    return subprocess.run([sys.executable, "-c", code],
                          cwd=REPO_ROOT,
                          capture_output=True,
                          text=True,
                          timeout=120)


@pytest.mark.parametrize("mod", W1_MODULES)
def test_new_module_imports_standalone_and_stays_light(mod):
    """Each split-out module imports alone, pulls no `train`, pulls nothing heavy."""
    r = _run_child(f"""
import importlib
importlib.import_module({mod!r})
heavy = [m for m in {HEAVY!r} if m in sys.modules]
assert not heavy, f"{mod} module scope imported {{heavy}} — import-lightness invariant broken"
assert "train" not in sys.modules, (
    "{mod} imported `train` — that is an import CYCLE: train.py imports {mod} at its "
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
    modules to move one, and `import train_config` starts paying for
    resume_state. Asserting the shape here makes that a test failure at the
    moment it is written rather than an architecture review two branches later.
    """
    r = _run_child(f"""
import importlib
importlib.import_module({mod!r})
allowed = {{{mod!r}, {LEAF!r}}} if {mod!r} != {LEAF!r} else {{{LEAF!r}}}
siblings = [m for m in {W1_MODULES!r} if m in sys.modules and m not in allowed]
assert not siblings, (
    f"{mod} imported {{siblings}} at module scope; the only sibling edge any "
    "split-out module may have is -> {LEAF} (spec 2026-08-31 §2 W1)")
""")
    assert r.returncode == 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"


def test_import_train_stays_light_and_really_imports_the_shims():
    """`import train` still pulls nothing heavy — and the shims are real imports.

    The second half matters: if the split-out modules were imported lazily
    (inside functions) instead of at train.py's module level, the ten
    module-level reads of moved names in train.py's own body — the argparse
    defaults under `if __name__ == "__main__"` among them — would NameError at
    run time while the whole test suite stayed green, because the suite never
    executes that block.
    """
    r = _run_child(f"""
import train
heavy = [m for m in {HEAVY!r} if m in sys.modules]
assert not heavy, f"`import train` pulled {{heavy}} — --dump-config is no longer cheap"
missing = [m for m in {W1_MODULES!r} if m not in sys.modules]
assert not missing, f"train.py does not import {{missing}} at module level"
""")
    assert r.returncode == 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"


def _main_block_body():
    """Top-level statements of train.py's `if __name__ == "__main__":` block.

    Parsed from source, never executed: the block builds the argparse parser and
    then trains, so importing it is not an option.
    """
    import ast
    tree = ast.parse((SRC / "train.py").read_text())
    for node in tree.body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"):
            return node.body
    raise AssertionError('no `if __name__ == "__main__":` block in src/train.py')


def test_train_aliases_itself_into_sys_modules_first():
    """The self-alias is the FIRST statement of train.py's __main__ block.

    WHAT is pinned: `sys.modules.setdefault("train", sys.modules["__main__"])`,
    and its POSITION. Position is half the contract — anything above it that
    triggers a `from train import ...` (directly or through a helper) executes
    train.py's body a second time before the alias can prevent it, and the alias
    then quietly protects nothing.

    WHY the whole thing exists: a script run binds this file to "__main__", so a
    runtime `from train import ...` — which eval_baselines does function-locally
    inside PolicyActor — imports a SECOND copy of the module. Two copies means
    two sets of module constants and cross-copy `isinstance` returning False.

    This test is a source pin and proves only that the statement is written.
    That the statement WORKS is test_script_run_has_exactly_one_train_module
    below; the two are a pair and neither is sufficient alone.
    """
    import ast
    first = _main_block_body()[0]
    src = ast.unparse(first)
    assert src == 'sys.modules.setdefault(\'train\', sys.modules[\'__main__\'])', (
        f"first statement of train.py's __main__ block is {src!r}, not the sys.modules "
        "self-alias — see the comment block at that line for why order matters")


def test_script_run_has_exactly_one_train_module():
    """BEHAVIOURAL half: run train.py AS A SCRIPT and prove the alias works.

    WHY a separate test from the AST pin: the pin is satisfied by the statement
    merely existing. This one runs the real file in a real child interpreter and
    checks the two things the alias is FOR, and it is the only check in the suite
    that can see them — the §3 determinism gate runs `--eval-interval 0
    --no-self-play`, exactly the flag set on which no runtime `from train import`
    occurs, so the gate is structurally blind here.

    Channel (both observations are about the CHILD's interpreter state, so they
    have to be made inside it):

      1. a sitecustomize.py on the child's PYTHONPATH registers an atexit hook
         that performs the same `import train` eval_baselines performs and
         reports whether the result IS sys.modules["__main__"]. atexit runs
         before module teardown, so sys.modules is intact; it also runs after
         `--dump-config`'s sys.exit(0), which is why that cheap path suffices
         instead of a full training run.
      2. `-X importtime` prints one line per module body EXECUTED. With the
         alias the hook's `import train` is a sys.modules hit and prints
         nothing; without it, the body runs again and stderr carries a
         `... | train` line. Absence is the proof — so the test also asserts
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

    # `import train` inside the hook is the exact import eval_baselines makes.
    probe_src = ("import atexit, sys\n"
                 "def _probe():\n"
                 "    import train\n"
                 "    print('ALIASPROBE identity=%s' % (train is sys.modules['__main__']),\n"
                 "          file=sys.stderr, flush=True)\n"
                 "atexit.register(_probe)\n")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "sitecustomize.py").write_text(probe_src)
        argv = [
            sys.executable, "-X", "importtime",
            str(SRC / "train.py"), "--dump-config", "--checkpoint-dir",
            str(td)
        ]
        r = subprocess.run(argv,
                           cwd=REPO_ROOT,
                           env=dict(os.environ, PYTHONPATH=str(td)),
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
        "`import train` in the child did NOT return sys.modules['__main__'] — the "
        "script run is carrying two copies of train.py")
    reimports = re.findall(r"^import time:.*\|\s*train$", r.stderr, re.M)
    assert not reimports, (
        f"train.py's module body executed a second time under the name 'train': {reimports} — "
        "the self-alias is missing or is no longer the first statement of the __main__ block")


def test_mask_head_slices_is_complete_in_a_leaf_only_interpreter():
    """`train_shared._MASK_HEAD_SLICES` holds one slice per action head.

    WHY this needs its own subprocess, and why that subprocess must NEVER have
    imported `train`: _MASK_HEAD_SLICES is not a single assignment but a
    three-statement construct (empty list, a `for` loop appending slices, a
    `del`). If the loop were left behind in train.py while the list moved to the
    leaf, `import train_shared` would still succeed and hand out an EMPTY list,
    while `import train` would run the leftover loop and fill THE SAME list
    object — so any interpreter that has imported `train` sees a correct length
    and this assert becomes vacuous. Importing train_shared alone is the only
    arrangement that can observe the half-move.

    Consequence if it ever regresses: `_apply_action_masks` zips with
    strict=True, so a short list raises at the first forward pass rather than
    silently unmasking heads. The blast radius is loud; the GUARD is what would
    have been silent, which is the point of testing it here.
    """
    r = _run_child("""
assert "train" not in sys.modules
import train_shared
from _action_spec import ACTION_HEAD_SIZES
assert "train" not in sys.modules, "this check is vacuous once `train` is imported"
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
