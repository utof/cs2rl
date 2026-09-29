"""The C environment: the zig-built `binding` extension, its C sources, and cs2_env.py.

Members:
  binding -- the extension module, built from binding.c by setup.py's ZigBuild (or
             `zig build`) into this directory. Untracked: a fresh clone has none;
  cs2_env -- Cs2Env and make_env, the Python side over binding.

Exports two paths and nothing else:
  SOURCE_DIR -- this directory: the C sources, build.zig, the baked nav_data.h and the
                built extension;
  ZIG_OUT    -- SOURCE_DIR / "zig-out", where `zig build` installs libcs2_play.so,
                cs2_demo and cs2_demo's resources.

WHY derive and not restate: every consumer that named this directory by a literal
path (play.py's library lookup, bake_nav's output, sync_action_spec's header, the
tests that look for a built cs2_demo) tolerated a missing path silently, so a move
of the package left them stale with every test green (#205 part 2b). Taken from
here, the location moves with the package.

PITFALL: importing this package must stay light, pathlib only. Never import binding
(it loads the .so) or cs2_env (numpy, the ctypes layer) here: play.py, the scripts and
test collection import these constants. tests/test_w1_modules.py pins that
`import cs2rl.c_env` loads no submodule and no numpy.
"""
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent
ZIG_OUT = SOURCE_DIR / "zig-out"
