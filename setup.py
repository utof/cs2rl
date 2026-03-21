import subprocess
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, source_dir=""):
        super().__init__(name, sources=[])
        self.source_dir = str(Path(source_dir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        build_dir = Path(self.build_temp) / ext.name
        build_dir.mkdir(parents=True, exist_ok=True)
        out_dir = Path(ext.source_dir).resolve()
        lib_dir = Path(self.build_lib).resolve()
        lib_dir.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                "cmake",
                ext.source_dir,
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={out_dir}",
            ],
            cwd=build_dir,
        )
        subprocess.check_call(["cmake", "--build", "."], cwd=build_dir)
        # Also copy to build/lib so setuptools --inplace copy step succeeds
        import shutil
        for so in out_dir.glob("binding*.so"):
            shutil.copy2(so, lib_dir / so.name)


setup(
    name="cs2rl-env",
    ext_modules=[CMakeExtension("binding", source_dir="src/c_env")],
    cmdclass={"build_ext": CMakeBuild},
)
