"""Structural guards for the modules split out of train.py (W1, spec 2026-08-31).

WHAT: every module carved out of src/train.py gets three properties checked in a
FRESH interpreter, one subprocess per case:

  1. it imports at all, standalone (no "works only because train.py imported it
     first" ordering luck);
  2. it does not pull `train` back in — the dependency graph stays acyclic, so
     the leaf really is a leaf;
  3. its module scope stays free of torch / nav / c_env.cs2_env.

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
W1_MODULES = ("train_shared", "resume_state", "train_config")

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
