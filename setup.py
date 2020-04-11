import os
import subprocess

import setuptools
from setuptools.command.build_ext import build_ext


class CMakeExtension(setuptools.Extension):
    """An extension with no sources.

    We do not want distutils to handle any of the compilation (instead we rely
    on CMake), so we always pass an empty list to the constructor.
    """

    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class BuildExt(build_ext):
    """Our custom build_ext command.

    Uses CMake to build extensions instead of a bare compiler (e.g. gcc, clang).
    """

    def run(self):
        try:
            subprocess.check_call(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extension_dir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DPython_TARGET_VERSION=3.7",
            "-DCMAKE_CXX_COMPILER=clang++",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extension_dir,
        ]
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        env = os.environ.copy()
        subprocess.check_call(["make", "-j" + str(os.cpu_count())],
                              cwd=self.build_temp,
                              env=env)


setuptools.setup(
    name="pynim",
    version="0.0.1",
    author="ZHOU Zikang",
    author_email="zhouzikang666@gmail.com",
    description="A Framework for Reinforcement Learning in Game of Nim",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ZikangZhou/NimRL",
    ext_modules=[CMakeExtension("pynim", sourcedir="nim_rl")],
    cmdclass=dict(build_ext=BuildExt),
    zip_safe=False,
)
