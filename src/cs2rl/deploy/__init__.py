"""Deploy exporters: the ONNX policy export and the map-data sidecar for deploy/CS2RLBot.

DEPLOY SUSPENDED 2026-05-03 (see each module's docstring). Each module is a CLI,
launched by module name from the repo root:

    python -m cs2rl.deploy.export_mapdata --map de_dust2

deploy/verify_onnx.py stays a script under deploy/ and imports
`cs2rl.deploy.export_policy`.

WHY this file holds a docstring and nothing else (#204): a re-export would make
runpy warn that the module was "found in sys.modules" before it ran as `__main__`
under `-m`, and would load the re-exported module's imports (torch and onnx, for
export_policy) for every importer of the package.
"""
