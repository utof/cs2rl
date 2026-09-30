"""Structural guards for the modules split out of train.py (W1, spec 2026-08-31).

WHAT: every module carved out of src/cs2rl/train.py gets four properties checked in a
FRESH interpreter, one subprocess per case:

  1. it imports at all, standalone (no "works only because train.py imported it
     first" ordering luck);
  2. it does not pull the run driver or the CLI (`cs2rl.train.loop`,
     `cs2rl.train.__main__`) back in — the dependency graph stays acyclic, so the leaf
     really is a leaf;
  3. its module scope stays free of torch / nav / env.c.cs2_env / rerun;
  4. the only sibling edge any of them has is into a LEAF, and the leaves import no
     sibling except each other in the one allowed direction (policy -> env.factory ->
     env.config). The shape spec §2 W1 fixes: leaves at the bottom, everything else a
     spoke off them, never spoke-to-spoke.

WHY a subprocess and not a plain import: pytest's session has already imported
half the repo by the time any test body runs, so `"torch" not in sys.modules`
in-process measures the session, not the module. Every assert here has to start
from an empty sys.modules or it silently passes forever.

WHY property 3 is load-bearing (measured, not stylistic): `import cs2rl.train.__main__`
today pulls neither torch nor nav nor env.c, because the torch imports in the modules it
reaches are function-local ON PURPOSE (about 35 of them in the flat train.py, before
#205 part 3). That is what makes `python -m cs2rl.train --dump-config` cost ~1 s instead
of ~30 s, which in turn is what makes it usable as the Modal/run_rung1 fingerprint step
(tests/test_train_cli.py's "--dump-config means zero side-effects"). The CLI module
imports each of these modules, or one that does, at ITS module level, so a single
module-scope `import torch` added to any of them silently destroys that guarantee for
every caller — and nothing else in the suite would notice.

EXTENDING THIS FILE: the split proceeds in several tasks. Add each new module's
name to W1_MODULES as it lands — the spec makes adding it part of the SAME task
that creates the module, precisely so a module cannot slip in unguarded.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Modules split out of train.py (W1, spec 2026-08-31), plus env.config, which
# owns the env contract that used to live partly in train_shared (spec
# 2026-09-03 §2.1). Every module here must import standalone and stay light.
#
# Every name in this file is the FULL dotted name (`cs2rl.X`), because each one
# is compared against sys.modules: a bare `X` is never a key there, so a bare
# entry would make every "not in sys.modules" check below pass forever.
#
# `env.factory` (W3) is here for a reason beyond bookkeeping: it is the module
# whose module scope is MOST tempting to make heavy, since its whole job is
# constructing envs. Its one function-local import, `from cs2rl.env.c.cs2_env import
# make_env` in `build_env_for`, stays function-local so `import cs2rl.train.__main__`
# stays free of torch/nav/env.c. (It had a second, `from cs2rl.train import
# SelfPlayManager` in `build_selfplay_manager`, which broke a cycle; that builder moved
# to cs2rl.train.selfplay in #205 part 3 and the cycle went with it.)
#
# `metrics_schema` (W4) is here for the mirror-image reason: it is a registry of
# STRINGS whose whole value is being cheap to import, and it took ownership of
# `EVAL_KEYS` from `eval.baselines` — a module with torch and env.c.cs2_env at
# its scope. If that ownership ever flipped back, or someone imported a policy
# class to spell a type hint, eight string constants would start costing a torch
# import, and only this test would say so.
#
# `cs2rl.train.trainer` (gh#168 W1) is deliberately NOT here: it subclasses PuffeRL, so
# it imports pufferlib (and through it torch) at module scope and is heavy by
# construction. It cannot pass property 3, and cs2rl.train.loop and
# tests/_helpers/trainer_harness.py import it function-locally for exactly that reason
# (knock-out W1-K3: a module-level `from cs2rl.train.trainer import Cs2PuffeRL` in
# cs2rl/train/loop.py turns test_cli_module_scope_stays_light red naming torch).
W1_MODULES = ("cs2rl.policy", "cs2rl.train.resume", "cs2rl.train.config", "cs2rl.train.metrics",
              "cs2rl.train.update", "cs2rl.env.factory", "cs2rl.eval.metrics_schema",
              "cs2rl.env.config")

# THREE leaves. cs2rl.policy owns the network and the names train_shared used to hold
# (log-std constants, action masks); env.config owns the env contract; env.factory
# (L1, importing only env.config at module scope) builds envs from it, and the policy
# imports it for load_policy_from_checkpoint's env. Every cs2rl.train.* spoke may
# import any of the three; train.config -> env.config is the load-bearing edge between
# a leaf and a spoke (env_config_from_args builds an EnvConfig). The reverse edges would
# make "leaf" meaningless — pyproject.toml's `cs2rl layers` contract pins them (the
# policy and env sit below the whole train package; tests/test_import_layers.py runs it),
# and so does tests/test_env_config.py::test_module_is_stdlib_only for env.config.
LEAVES = frozenset({"cs2rl.policy", "cs2rl.env.config", "cs2rl.env.factory"})

# Imports whose presence in sys.modules means the import-lightness invariant is
# gone. `cs2rl.env.c.cs2_env` rather than `cs2rl.env.c` on purpose: the package
# itself is cheap, the ctypes/binding module underneath it is not.
#
# `rerun` stands in for `cs2rl.viz.render`, which imports it at module scope and which
# cs2rl.train.record reaches only inside record_episode (`--record`). The layers
# contract lets train import viz at ANY scope (train sits above viz, and no
# ignore_imports entry is left to pin), so it cannot see a function-local import that is
# CALLED at module scope; test_cli_module_scope_stays_light, through this entry, does.
HEAVY = ("torch", "cs2rl.env.nav", "cs2rl.env.c.cs2_env", "rerun")


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
drivers = [m for m in ("cs2rl.train.loop", "cs2rl.train.__main__") if m in sys.modules]
assert not drivers, (
    f"{mod} imported {{drivers}} — that is an import CYCLE: the run driver and the CLI import "
    "{mod} at their module level, so this only appears to work while some other module got "
    "there first")
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


def test_cli_module_scope_stays_light():
    """The CLI's module scope pulls nothing heavy.

    `import cs2rl.train.__main__` runs `__main__.py`'s module scope under its own name, so
    its `if __name__ == "__main__":` block is skipped: what loads is exactly what every
    launch loads before argparse. One module-scope `import torch` in any module it imports
    (loop, envs, record, evaluate, config, policy) would make `--dump-config` pay for torch.

    NOT checked here: that every name the main block reads is bound at module scope. The
    old train.py's shim test guarded that, because the block read names its shims
    imported. It is ruff F821's job now (`select` in pyproject.toml includes "F"; the
    pre-commit hook runs it on every commit): an unbound name in the block is an
    undefined-name error, and a copy of that scope analysis here would be a second one
    to keep right.
    """
    r = _run_child(f"""
import cs2rl.train.__main__
heavy = [m for m in {HEAVY!r} if m in sys.modules]
assert not heavy, f"the CLI's module scope pulled {{heavy}} — --dump-config is no longer cheap"
""")
    assert r.returncode == 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"


def test_import_train_test_harness_stays_light():
    """`from tests._helpers import trainer_harness` pulls nothing heavy either.

    The harness is not a W1 module (it is test-only and imports the train modules
    function-locally), so the probes above never import it. Its lightness is what lets a
    test import it without paying for torch, and it rests on `_harness_parts` and
    `_build_trainer_for_test` importing `cs2rl.train.trainer` function-locally: the
    trainer subclasses PuffeRL and imports torch at module scope, so one module-scope
    trainer import here loads torch. No import-linter contract sees the harness (it
    lives under tests/, outside `cs2rl`), so this probe is the check that does.
    """
    r = _run_child(f"""
from tests._helpers import trainer_harness
heavy = [m for m in {HEAVY!r} if m in sys.modules]
assert not heavy, f"`from tests._helpers import trainer_harness` pulled {{heavy}}"
""")
    assert r.returncode == 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"


def test_env_c_package_import_stays_light():
    """`import cs2rl.env.c` binds SOURCE_DIR and ZIG_OUT and loads no submodule and no numpy.

    WHY: viz/play.py, scripts/bake_nav.py, scripts/sync_action_spec.py and several test
    modules import the package for its two path constants (#205 part 2b), some at
    module scope or at collection. One `from . import binding` in its __init__ would
    load the built .so (and cs2_env, numpy) for every one of them.
    PITFALL: the probe runs through _run_child (cwd = REPO_ROOT, this session's
    PYTHONPATH), so cs2rl's guard refuses another checkout's install. A child started
    from a tmp cwd without PYTHONPATH would import main's package, whose __init__ may
    be anything, and pass or fail for that tree. The location check names src/, not
    REPO_ROOT: a worktree nests under main's root, so containment in the root would
    accept a worktree's file from main.
    """
    r = _run_child(f"""
import cs2rl.env.c as package
from pathlib import Path
loaded = sorted(m for m in sys.modules if m.startswith("cs2rl.env.c.") or m == "numpy")
assert not loaded, f"`import cs2rl.env.c` loaded {{loaded}}: keep its __init__ to pathlib"
here = Path(package.__file__).resolve()
assert here.is_relative_to({str(REPO_ROOT / "src")!r}), f"imported another checkout's {{here}}"
assert package.SOURCE_DIR == here.parent, package.SOURCE_DIR
assert package.ZIG_OUT == here.parent / "zig-out", package.ZIG_OUT
""")
    assert r.returncode == 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"


def test_script_run_has_exactly_one_cli_module():
    """Run `python -m cs2rl.train` and prove nothing re-imports the CLI under its own name.

    WHY (#205 part 3): train.py was both the `-m` script and an importable module, so a
    runtime `from cs2rl.train import ...` (eval.baselines' PolicyActor did one) executed
    it a second time as `cs2rl.train`, and train.py carried a `sys.modules` self-alias
    against that. As a package, `-m cs2rl.train` runs `cs2rl/train/__main__.py` under the
    name `__main__`, and the names production imports live in other modules. The alias is
    gone; this checks the property it protected: at exit, `cs2rl.train.__main__` was never
    imported, and no module body executed twice.

    Channel: a sitecustomize.py on the child's PYTHONPATH registers an atexit hook that
    makes the import eval.baselines makes and reports `cs2rl.train.__main__ in
    sys.modules`. `-X importtime` prints one line per module body executed.
    PITFALL: do NOT rewrite this with `runpy.run_path(..., run_name="__main__")` plus a
    post-hoc assert: run_path restores sys.modules["__main__"] when it returns.
    """
    import os
    import re
    import tempfile

    probe_src = (
        "import atexit, sys\n"
        "def _probe():\n"
        "    from cs2rl.policy import load_policy_from_checkpoint\n"
        "    print('ALIASPROBE cli_imported=%s' % ('cs2rl.train.__main__' in sys.modules),\n"
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
    names = re.findall(r"^import time:.*\|\s*(\S+)$", r.stderr, re.M)
    assert "ALIASPROBE" in r.stderr, (
        "the atexit probe never printed — the observation channel is broken, so this "
        f"test proves nothing. STDERR:\n{r.stderr[-2000:]}")
    assert "cs2rl.train.config" in names, ("-X importtime did not list cs2rl.train.config; "
                                           "the checks below would pass vacuously")
    assert "ALIASPROBE cli_imported=False" in r.stderr, (
        "something imported `cs2rl.train.__main__` under its own name: the run carries two "
        "copies of the CLI module")
    # cs2rl only: importtime also lists third-party names more than once (a failed
    # optional import, a platform probe such as `nt`), which says nothing about ours.
    ours = [n for n in names if n.split(".")[0] == "cs2rl"]
    twice = sorted({n for n in ours if ours.count(n) > 1})
    assert not twice, f"cs2rl module bodies executed twice: {twice}"


def test_mask_head_slices_is_complete_in_a_leaf_only_interpreter():
    """`policy._MASK_HEAD_SLICES` holds one slice per action head.

    WHY this needs its own subprocess, and why that subprocess must NEVER have
    imported `cs2rl.train`: _MASK_HEAD_SLICES is not a single assignment but a
    three-statement construct (empty list, a `for` loop appending slices, a
    `del`). If the loop were ever left behind in a module that imports the policy
    (cs2rl.train.loop, say) while the list lives in policy.py, `import cs2rl.policy`
    alone would still succeed and hand out an EMPTY list, while an interpreter that had
    imported that module would run the leftover loop and fill THE SAME list object — so
    any interpreter that has imported that module sees a correct length and this assert
    becomes vacuous. Importing policy alone, with no `cs2rl.train.*` module loaded
    (importing any of them loads the package `cs2rl.train` first, which the child asserts
    absent), is the only arrangement that can observe the half-move.

    Consequence if it ever regresses: `_apply_action_masks` zips with
    strict=True, so a short list raises at the first forward pass rather than
    silently unmasking heads. The blast radius is loud; the GUARD is what would
    have been silent, which is the point of testing it here.
    """
    r = _run_child("""
assert "cs2rl.train" not in sys.modules
from cs2rl import policy
from cs2rl.spec.action import ACTION_HEAD_SIZES
assert "cs2rl.train" not in sys.modules, "this check is vacuous once `cs2rl.train` is imported"
assert len(policy._MASK_HEAD_SLICES) == len(ACTION_HEAD_SIZES), (
    f"_MASK_HEAD_SLICES has {len(policy._MASK_HEAD_SLICES)} entries for "
    f"{len(ACTION_HEAD_SIZES)} action heads — the construct at the top of "
    "train_shared.py was moved only partially")
# The slices must also tile [0, sum(sizes)) contiguously, which is what makes
# them a valid decomposition of the flat action-mask row rather than merely a
# list of the right length.
assert policy._MASK_HEAD_SLICES[0][0] == 0
for (lo, hi), size in zip(policy._MASK_HEAD_SLICES, ACTION_HEAD_SIZES, strict=True):
    assert hi - lo == size, (lo, hi, size)
for (_, hi), (lo, _) in zip(policy._MASK_HEAD_SLICES,
                            policy._MASK_HEAD_SLICES[1:], strict=False):
    assert hi == lo, "mask head slices are not contiguous"
""")
    assert r.returncode == 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
