"""R0-F (#136): rewards stay finite with AND without -ffast-math.

WHAT: builds a scratch `binding` via `zig build -Dfast_math={true,false}`
into a tmp prefix (NEVER through setup.py — that overwrites the production
.so in src/cs2rl/env/c) and steps 500 ticks on two maps whose bombsite_dist
contains inf / a 4×max sentinel: bombsites=[] and a simple_map with an
unreachable room. Every reward must be finite in both builds.

WHY: under -ffast-math `isfinite()` folds to true, so a guard on it is dead
code and inf leaks into the PBRS potential. The C code now guards on
`bombsite_dist_scale > 0` and `dist < 1e29f` (plain float compares survive
fast-math). The strict (-Dfast_math=false) build proves the guard is not
merely "works because the optimiser happened to be kind".

PITFALLS:
  - setup.py has no __main__ guard: importing it runs setuptools on pytest's
    argv → SystemExit. The zig resolver is inlined here instead.
  - zig resolution mirrors setup.py: PY_ZIG → `python -m ziglang` → PATH.
    The ziglang wheel does NOT put `zig` on PATH, so a which()-gate would
    skip this test everywhere the production .so is actually built.
  - An "unknown option" from zig means build.zig lost -Dfast_math — that is
    a FAILURE, not a skip. Only dependency-fetch problems skip.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig

import pytest

from cs2rl.env.c import SOURCE_DIR

# The zig build's working directory, from the package itself: a restated path here went
# stale silently, because the test skips wherever zig is missing.
C_DIR = SOURCE_DIR

CHECK = r"""
import importlib.util, sys, numpy as np
from pathlib import Path
# ORDER IS LOAD-BEARING: cs2_env.py does `from cs2rl.env.c import binding`, which
# takes whatever sys.modules["cs2rl.env.c.binding"] already holds and otherwise
# loads the PRODUCTION .so next to it. Pre-seeding that key with the scratch
# build BEFORE importing cs2_env is what makes this script test the variant; drop
# the pre-seed and it silently tests the production .so (the check below is what
# catches it).
# PITFALL: binding is single-phase init (PyModule_Create), and for such a module
# module_from_spec ITSELF registers the key (measured, #199), so deleting only the
# explicit assignment below still tests the variant; the pre-seed is the whole load.
import cs2rl.env.c
spec = importlib.util.spec_from_file_location("cs2rl.env.c.binding",
                                              next(Path(sys.argv[1]).glob("binding*")))
scratch = importlib.util.module_from_spec(spec)
sys.modules["cs2rl.env.c.binding"] = scratch
spec.loader.exec_module(scratch)
cs2rl.env.c.binding = scratch
from cs2rl.env.c import cs2_env
if not cs2_env.binding.__file__.startswith(sys.argv[1]):
    raise RuntimeError("wrong binding loaded: " + cs2_env.binding.__file__)
from cs2rl.env.c.cs2_env import make_env
from cs2rl.env.map import SIMPLE_ROOMS, make_simple_map
from cs2rl.spec.action import ACTION_HEAD_SIZES, AIM_DIM
from cs2rl.env.nav import N_AGENTS
maps = {"nobomb": make_simple_map(bombsites=[]),
        "unreachable": make_simple_map(rooms=list(SIMPLE_ROOMS) +
                        [(len(SIMPLE_ROOMS), 5000.0, 5000.0, 5100.0, 5100.0, 0.0, False)])}
for name, md in maps.items():
    env = make_env(map_data=md, seed=1)
    rng = np.random.default_rng(0)
    env.reset()
    for _ in range(500):
        act = np.stack([rng.integers(0, n, size=N_AGENTS) for n in ACTION_HEAD_SIZES], 1).astype(np.int32)
        cont = rng.uniform(-0.5, 0.5, (N_AGENTS, AIM_DIM)).astype(np.float32)
        _, rew, *_ = env.step(act, cont)
        if not np.isfinite(rew).all():
            raise RuntimeError(f"non-finite reward on map {name}: {rew}")
    env.close()
print("OK")
"""


def _find_zig():
    """Return argv prefix for zig, mirroring setup.py._find_zig (PY_ZIG →
    `python -m ziglang` → `zig` on PATH). Inlined because setup.py cannot be
    imported without running setuptools. Raises RuntimeError if no route works."""
    if pz := os.environ.get("PY_ZIG"):
        return [pz]
    try:
        subprocess.check_call([sys.executable, "-m", "ziglang", "version"],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        return [sys.executable, "-m", "ziglang"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    if shutil.which("zig"):
        return ["zig"]
    raise RuntimeError("no zig via PY_ZIG / ziglang / PATH")


def _prod_so():
    return next(C_DIR.glob("binding*.so"))


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.slow                      # two zig variant builds (~20 s)
@pytest.mark.parametrize("fast_math", ["true", "false"])
def test_rewards_finite_under_both_fast_math_settings(tmp_path, fast_math):
    try:
        zig = _find_zig()
    except RuntimeError as e:
        pytest.skip(str(e))
    import numpy

    # The option must exist in build.zig — an absent option is a regression, not a skip.
    help_out = subprocess.run([*zig, "build", "--help"], cwd=C_DIR, capture_output=True, text=True)
    assert "-Dfast_math" in help_out.stdout, f"build.zig lost the fast_math option:\n{help_out.stdout[-1500:]}"

    prod = _prod_so()
    before = _sha(prod)
    prefix = tmp_path / "out"
    cmd = [
        *zig, "build", f"-Dfast_math={fast_math}", "-Dlink_python=false", "--prefix",
        str(prefix), "--cache-dir",
        str(tmp_path / "zig-cache"), f"-Dpython_include={sysconfig.get_path('include')}",
        f"-Dnumpy_include={numpy.get_include()}"
    ]
    r = subprocess.run(cmd, cwd=C_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        err = r.stderr
        if "unknown option" in err.lower() or "invalid option" in err.lower():
            pytest.fail(f"build.zig rejected -Dfast_math:\n{err[-2000:]}")
        # Skip ONLY for dependency-fetch problems (raylib from build.zig.zon);
        # a C compile error in binding.c must FAIL.
        if "dependency" in err or "fetch" in err or "build.zig.zon" in err:
            pytest.skip(f"variant build failed to configure: {err[-500:]}")
        pytest.fail(f"variant build failed to COMPILE:\n{err[-3000:]}")
    built = next((prefix / "lib").glob("*binding*"))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    shutil.copy2(built, scratch / f"binding{sysconfig.get_config_var('EXT_SUFFIX')}")
    # The child inherits this session's PYTHONPATH, so `cs2rl` resolves to the checkout
    # tests/conftest.py's tripwire already required; only the binding is swapped.
    r = subprocess.run([sys.executable, "-c", CHECK, str(scratch)],
                       capture_output=True,
                       text=True,
                       timeout=600)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-2000:]
    assert _sha(prod) == before, "production .so was touched by the variant build"
