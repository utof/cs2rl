# setup.py — thin build shim that invokes `zig build` to compile binding.c.
#
# Why not pure CMake? Zig bundles its own C compiler + libc for all platforms,
# so users need only `ziglang` (auto-installed by uv as a build dep) instead of
# CMake + MSVC/GCC + vcpkg.
#
# Responsibility split:
#   setup.py  — discovers Python/NumPy headers, finds zig binary, renames output
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
        import numpy  # imported here so it's only required at build time

        src     = Path(ext.source_dir)           # src/c_env/ (absolute)
        out_dir = src                             # .so lands next to C sources
        lib_dir = Path(self.build_lib).resolve()  # setuptools staging dir
        lib_dir.mkdir(parents=True, exist_ok=True)

        zig            = _find_zig()
        python_include = sysconfig.get_path("include")
        numpy_include  = numpy.get_include()
        # EXT_SUFFIX is the full suffix including SOABI + extension:
        #   Linux:   .cpython-312-x86_64-linux-gnu.so
        #   macOS:   .cpython-312-darwin.so
        #   Windows: .cp312-win_amd64.pyd
        ext_suffix  = sysconfig.get_config_var("EXT_SUFFIX")
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
                "Check zig build output above for errors."
            )
        built     = candidates[0]
        dest_name = f"binding{ext_suffix}"

        shutil.copy2(built, out_dir / dest_name)
        shutil.copy2(built, lib_dir / dest_name)


setup(
    name="cs2rl-env",
    ext_modules=[ZigExtension("binding", source_dir="src/c_env")],
    cmdclass={"build_ext": ZigBuild},
)
