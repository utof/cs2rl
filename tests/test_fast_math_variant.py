"""R0-F (#136): rewards stay finite with AND without -ffast-math.

WHAT: builds a scratch `binding` via `zig build -Dfast_math={true,false}`
into a tmp prefix (NEVER through setup.py — that overwrites the production
.so in src/cs2rl/env/c) and steps 500 ticks on two maps whose bombsite_dist
contains inf / a 4×max sentinel: bombsites=[] and a simple_map with an
unreachable room. Every reward must be finite in both builds.

WHY: under -ffast-math `isfinite()` folds to true, so a guard on it is dead
code and inf leaks into the PBRS potential. The C code now guards on
`bombsite_dist_scale > 0` and `dist < 1e29f` (plain float compares survive
fast-math). What each half catches, measured by knock-outs (#288):
  - The pair at runtime catches REMOVED guards: with no guard at all,
    `[false]` goes red. It does NOT catch the #136 shape itself: with the
    pre-#136 `isfinite(dist)` guards restored, both builds stay green (the
    fast-math build's reward clamp scrubs the NaN, and the strict build
    honours isfinite).
  - The #136 shape (`isfinite`/`INFINITY` under fast-math) is caught at build
    time instead: build.zig's fast-math flags carry
    -Werror=nan-infinity-disabled, so `[true]`'s variant build fails to
    compile with "use of infinity is undefined behavior".
  - The `[true]` half also pins the flag's PRESENCE: it builds with
    --verbose-cc and asserts the binding.c compile line carries
    -Werror=nan-infinity-disabled. Without that, deleting the flag from
    build.zig keeps the suite green (measured by the #288 verifier's mutant c),
    and the one line that catches the #136 shape would go unnoticed.

PITFALLS:
  - zig comes from setup.py's own resolver (PY_ZIG → `python -m ziglang` →
    PATH), so the test builds with the zig the production build uses. That
    import works only because setup.py guards its setup() call with
    `if __name__ == "__main__"`; drop the guard and importing this module
    runs setuptools on pytest's argv and errors. The import also pulls
    setuptools into every session that collects this file.
  - No zig is a FAILURE, never a skip: a skip here hid this guard for a whole
    refactor wave (#288). The ziglang wheel does NOT put `zig` on PATH; the
    dev group installs it into .venv, where `python -m ziglang` finds it.
  - An "unknown option" from zig means build.zig lost -Dfast_math — that is
    a FAILURE, not a skip. Only dependency-fetch problems skip, and only when
    stderr holds no C compiler diagnostic (stderr carries absolute checkout
    paths, which may themselves contain "dependency" or "fetch").
"""
import hashlib
import re
import shutil
import subprocess
import sys
import sysconfig

import pytest

from cs2rl.env.c import SOURCE_DIR
from setup import _find_zig

# The zig build's working directory, from the package itself: a restated path here went
# stale silently, because the test skipped wherever zig was missing (#288).
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


def _prod_so():
    return next(C_DIR.glob("binding*.so"))


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.slow                      # two zig variant builds (~20 s)
@pytest.mark.parametrize("fast_math", ["true", "false"])
def test_rewards_finite_under_both_fast_math_settings(tmp_path, fast_math):
    zig = _find_zig()
    import numpy

    # The option must exist in build.zig — an absent option is a regression, not a skip.
    # setup's resolver never raises: with no PY_ZIG and no ziglang it falls back to a
    # bare `zig`, so "no zig anywhere" surfaces here, as this first subprocess's
    # FileNotFoundError. pytest.fail runs OUTSIDE the except block, so the remedy is the
    # visible line instead of a chained "During handling of the above exception" traceback.
    try:
        help_out = subprocess.run([*zig, "build", "--help"],
                                  cwd=C_DIR,
                                  capture_output=True,
                                  text=True)
    except FileNotFoundError:
        help_out = None
    if help_out is None:
        pytest.fail(
            "no zig: in the MAIN checkout run `uv sync --all-groups "
            "--inexact` (never in a worktree: it re-points the shared editable install, "
            "#220), or set PY_ZIG=<zig binary>",
            pytrace=False)
    assert "-Dfast_math" in help_out.stdout, f"build.zig lost the fast_math option:\n{help_out.stdout[-1500:]}"

    prod = _prod_so()
    before = _sha(prod)
    prefix = tmp_path / "out"
    cmd = [
        *zig, "build", f"-Dfast_math={fast_math}", "-Dlink_python=false", "--prefix",
        str(prefix), "--cache-dir",
        str(tmp_path / "zig-cache"), f"-Dpython_include={sysconfig.get_path('include')}",
        f"-Dnumpy_include={numpy.get_include()}", "--verbose-cc"
    ]
    r = subprocess.run(cmd, cwd=C_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        err = r.stderr
        if "unknown option" in err.lower() or "invalid option" in err.lower():
            pytest.fail(f"build.zig rejected -Dfast_math:\n{err[-2000:]}")
        # Skip ONLY for dependency-fetch problems (raylib from build.zig.zon);
        # a C compile error in binding.c must FAIL. The substring test alone is not
        # enough: stderr carries absolute checkout paths, so a checkout path containing
        # "dependency" or "fetch" would turn a compile error into a skip (#288). A C
        # diagnostic (`<file>.c:12:3: error:`) therefore always fails; a .zon fetch
        # error (`build.zig.zon:12:20: error:`) does not match and may still skip.
        compile_error = re.search(r"\.[ch]:\d+:\d+: error:", err)
        if not compile_error and ("dependency" in err or "fetch" in err or "build.zig.zon" in err):
            pytest.skip(f"variant build failed to configure: {err[-500:]}")
        # Every `: error:` line first (a tail slice lost some), then the log tail.
        errors = "\n".join(line for line in err.splitlines() if ": error:" in line)
        pytest.fail(f"variant build failed to COMPILE:\n{errors}\n--- log tail ---\n{err[-3000:]}")
    if fast_math == "true":
        # Pin the flag itself: the build only fails on the #136 shape while
        # -Werror=nan-infinity-disabled is in c_flags_fast, and deleting it leaves every
        # other assert here green (#288, verifier mutant c). --verbose-cc prints the
        # compile command on stderr, but only when zig actually compiles: the fresh
        # --cache-dir above is what guarantees that, so keep the two together.
        # PITFALL: no binding.c line at all means the instrument changed (a zig upgrade
        # that words or routes the log differently), not that the flag is present, so that
        # case fails with its own message instead of passing.
        compile_lines = [line for line in r.stderr.splitlines() if "binding.c" in line]
        assert compile_lines, "zig --verbose-cc printed no binding.c compile line; cannot check the flag"
        assert any("-Werror=nan-infinity-disabled" in line for line in compile_lines), (
            "the fast-math build no longer carries -Werror=nan-infinity-disabled "
            "(src/cs2rl/env/c/build.zig c_flags_fast), the only thing that catches the #136 "
            "isfinite/INFINITY shape at build time (#288)")
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
