"""The maintained NVML distribution must supply Torch's pynvml import."""

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_MODULE = Path(str(
    importlib.metadata.distribution("nvidia-ml-py").locate_file("pynvml.py")))


def test_pytest_resolves_nvml_from_maintained_distribution():
    """Catch a local module shadowing the maintained distribution during pytest."""
    import pynvml

    assert Path(pynvml.__file__).resolve() == OFFICIAL_MODULE.resolve()


def test_src_first_torch_import_uses_maintained_nvml_without_deprecation_warning():
    """A fresh src-first process catches shadowing and the deprecated import hook."""
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "src")
with warnings.catch_warnings(record=True) as observed:
    warnings.simplefilter("always")
    import torch
    import pynvml
print(json.dumps({
    "module": str(Path(pynvml.__file__).resolve()),
    "warnings": [str(item.message) for item in observed],
}))
""",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    loaded = json.loads(child.stdout)
    assert Path(loaded["module"]) == OFFICIAL_MODULE.resolve()
    assert not any("The pynvml package is deprecated" in msg for msg in loaded["warnings"])
