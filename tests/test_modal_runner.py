"""tests/test_modal_runner.py — import boundary for the optional Modal runner (plan task 14.1).

The Modal training runner (scripts/run_modal.py + helpers, later plan tasks) is an
OPTIONAL transport for one training run. Two invariants keep it from leaking into
the default local/scientific environment:

  * `modal` is a NON-DEFAULT dependency group pinned to `modal>=1.4.3,<2`. A plain
    `uv sync` must not pull it (non-default groups are excluded unless explicitly
    requested via `--group modal`), so the locked scientific stack stays
    byte-identical with or without the group present in pyproject.toml.
  * The local entrypoints (`src.train`, `scripts.exp_lib`, `scripts.run_experiment`)
    must be importable WITHOUT modal installed — i.e. nothing in the local stack
    imports modal at module scope. The runner must import modal lazily, inside the
    code paths that actually talk to Modal.

Pitfalls this file is careful about:
  * The import check runs in a FRESH subprocess: the pytest process itself may
    legitimately have `modal` in sys.modules once later runner tests exist, so
    asserting on the parent's sys.modules would give false failures.
  * `cwd=ROOT` makes the `src` / `scripts` namespace packages resolvable in the
    child; the child also prepends `src/` to sys.path because train.py imports its
    generated siblings bare (`from _action_spec import ...`). This keeps the test
    hermetic: it must not depend on the editable install's .pth, which any
    `uv sync --no-install-project` removes from the venv.
"""
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
