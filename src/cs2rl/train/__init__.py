"""The training package: `python -m cs2rl.train` (see __main__.py).

It exports nothing: import the module that owns a name.
The three thread-count defaults below are set before any submodule imports
numpy, as train.py's module scope did.
"""

import os

# DELIBERATE DELTA from the flat train.py (#205 part 3): cs2rl.viz.play and
# cs2rl.viz.play_actions used to get these defaults for free, because they imported
# cs2rl.train (measured: OPENBLAS_NUM_THREADS was "1" in both at base, and is unset
# now). They import nothing under cs2rl.train any more, so they run with the
# platform's default BLAS thread counts. Accepted: they are interactive viewers, and
# these defaults exist for the vectorised training workers (one BLAS thread per env
# process, so N workers do not oversubscribe the cores). train_bc sets its own.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
