"""setup.py"""

from pathlib import Path
import subprocess
import sys

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        Extension.__init__(self, name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def run(self) -> None:
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake not installed")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: "CMakeBuild") -> None:
        extdir = str(Path(self.get_ext_fullpath(ext.name)).parent.resolve())
        cfg = "Debug" if self.debug else "Release"

        cmake_command = [
            "cmake",
            ext.sourcedir,
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        build_command = ["cmake", "--build", ".", "--config", cfg, "--", "-j2"]

        Path(self.build_temp).mkdir(exist_ok=True, parents=True)
        for cmd in [cmake_command, build_command]:
            subprocess.check_call(cmd, cwd=self.build_temp)


with open("requirements.txt") as req:
    requirements = req.readlines()

setup(
    name="ml_utils",
    version="1.1.0",
    author="justin chiu",
    description="machine learning utilities",
    url="https://github.com/jfc4050/ml_utils",
    ext_modules=[CMakeExtension("ml_utils/ml_utils")],
    cmdclass={"build_ext": CMakeBuild},
    packages=find_packages(exclude=("tests",)),
    install_requires=requirements,
    zip_safe=False,
)
