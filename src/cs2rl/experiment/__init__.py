"""Experiment tooling: the gate readers and offline analyses.

Each module with a `main()` is a CLI, launched by module name from the repo root:

    python -m cs2rl.experiment.gate outputs/checkpoints/rung1

From a worktree, put its own src/ first (`env PYTHONPATH=<checkout>/src python -m
...`). Otherwise the #199 guard in `cs2rl/__init__.py` refuses the import, because
the shared .venv's editable install names main's copy. From a working directory
outside any checkout the guard cannot tell, and main's copy runs silently (#242).

WHY this file holds a docstring and nothing else (#204): it is in the import chain
of every module below it, and `smoke_read` must run with no third-party import
available (tests/test_rung1a_smoke_read.py launches it under `python -S`). A
re-export here would put the re-exported module's imports into that chain, and
would make runpy warn that the module was "found in sys.modules" before it ran as
`__main__` under `-m`.
"""
