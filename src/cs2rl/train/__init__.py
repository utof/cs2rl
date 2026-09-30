"""The training package: `python -m cs2rl.train` (see __main__.py).

It exports nothing: import the module that owns a name.
The three thread-count defaults below are set before any submodule imports
numpy, as train.py's module scope did.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
