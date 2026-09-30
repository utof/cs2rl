# setup.py — thin build shim that invokes `zig build` to compile binding.c.
#
# Why not pure CMake? Zig bundles its own C compiler + libc for all platforms,
# so users need only `ziglang` (auto-installed by uv as a build dep) instead of
# CMake + MSVC/GCC + vcpkg.
#
# Responsibility split:
#   setup.py  — discovers Python/NumPy headers, finds zig binary, places the output
#   build.zig — compiles binding.c with the correct flags
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def _find_zig():
    """Return argv prefix to invoke zig.

    Priority:
      1. PY_ZIG env var — explicit override (CI, custom installs)
      2. sys.executable -m ziglang — auto-installed ziglang PyPI package
         (ziglang does NOT put `zig` on PATH; it is invoked via -m)
      3. `zig` on system PATH — fallback for developers with Zig installed manually
    """
    if pz := os.environ.get("PY_ZIG"):
        return [pz]
    try:
        subprocess.check_call(
            [sys.executable, "-m", "ziglang", "version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return [sys.executable, "-m", "ziglang"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ["zig"]


class ZigExtension(Extension):

    def __init__(self, name, source_dir=""):
        super().__init__(name, sources=[])
        self.source_dir = str(Path(source_dir).resolve())


class ZigBuild(build_ext):

    def build_extension(self, ext):
        # Imported here so numpy is only required at build time.
        import numpy

        src = Path(ext.source_dir)     # src/cs2rl/env/c/ (absolute)
        zig = _find_zig()
        python_include = sysconfig.get_path("include")
        numpy_include = numpy.get_include()
        link_python = "true" if sys.platform == "win32" else "false"

        subprocess.check_call(
            zig + [
                "build",
                f"-Dpython_include={python_include}",
                f"-Dnumpy_include={numpy_include}",
                f"-Dlink_python={link_python}",
            ],
            cwd=src,
        )

        # Zig outputs libbinding.so in zig-out/lib (Linux/macOS) or
        # binding.dll in zig-out/bin (Windows — DLLs land in bin, not lib).
        zig_out_lib = src / "zig-out" / "lib"
        zig_out_bin = src / "zig-out" / "bin"
        candidates = list(zig_out_lib.glob("*binding*")) + list(zig_out_bin.glob("*binding*"))
        if not candidates:
            raise RuntimeError(
                f"zig build produced no binding artifact in {zig_out_lib} or {zig_out_bin}. "
                "Check zig build output above for errors.")

        # ONE write, to the path setuptools owns for this extension:
        # build_lib/cs2rl/env/c/binding<EXT_SUFFIX>. Placing it anywhere else is
        # setuptools' job, not ours: a wheel packs build_lib, and `--inplace` /
        # an editable install copy it into src/cs2rl/env/c/ afterwards
        # (build_ext.copy_extensions_to_source). PITFALL: get_ext_fullpath does
        # not create the directory, and `setup.py build_ext --inplace` runs no
        # build_py that would, so mkdir it here.
        dest = Path(self.get_ext_fullpath(ext.name))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], dest)


# Guarded so tests/env/c/test_fast_math_variant.py can import _find_zig (#146, #288); an
# unguarded import runs setuptools on pytest's argv and exits. Every build path still
# reaches setup(): `python setup.py ...` runs as __main__, and so does setuptools'
# PEP 517 backend (build_meta execs this file with __name__ == "__main__"), which is
# also what uv's build and Modal's `pip install --no-build-isolation` go through.
if __name__ == "__main__":
    setup(
        ext_modules=[ZigExtension("cs2rl.env.c.binding", source_dir="src/cs2rl/env/c")],
        cmdclass={"build_ext": ZigBuild},
    )
