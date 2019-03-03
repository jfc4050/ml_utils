"""setup.py"""

from pathlib import Path
import subprocess
import sys

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CppExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class MakeFileBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        output_dir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()

        subprocess.check_call([
            'make',
            '-e',
            f'OSX={int(sys.platform == "darwin")}',
            f'PYTHON_VERSION={".".join(str(x) for x in sys.version_info[:2])}',
            f'PKG={str(output_dir)}'
        ])


setup(
    name='ml_utils',
    version='0.2',
    author='justin chiu',
    description='machine learning utilities',
    url='https://github.com/jfc4050/ml_utils',
    ext_modules=[CppExtension('ml_utils/ml_utils')],
    cmdclass={'build_ext': MakeFileBuild},
    packages=find_packages(),
    zip_safe=False
)
